# Integer Representation and Two's Complement

> The way computers represent negative numbers -- two's complement -- is an ingenious mechanism that allows a single adder circuit to handle both addition and subtraction.

## Learning Objectives

- [ ] Explain the difference between unsigned and signed integers
- [ ] Manually verify the two's complement mechanism through hand calculation
- [ ] Explain the causes of and solutions for overflow
- [ ] Understand the differences in endianness (byte order)
- [ ] Understand the mechanism of fixed-point numbers and their application in financial calculations
- [ ] Be aware of the characteristics and limitations of integer types in each language

## Prerequisites


---

## 1. Unsigned Integers

### 1.1 Basics

```
Unsigned integers: all bits are used for value representation

  Range representable with N bits: 0 to 2^N - 1

  8-bit (uint8):   0 to 255
  16-bit (uint16):  0 to 65,535
  32-bit (uint32):  0 to 4,294,967,295 (approx. 4.3 billion)
  64-bit (uint64):  0 to 18,446,744,073,709,551,615 (approx. 1.8 x 10^19)

  Example: 8-bit representation
  0000 0000 =   0
  0000 0001 =   1
  0111 1111 = 127
  1000 0000 = 128
  1111 1111 = 255
```

### 1.2 Unsigned Integer Arithmetic

```
Unsigned integer addition (8-bit):

  Basic addition:
    0000 0011 (3)
  + 0000 0101 (5)
  ──────────────
    0000 1000 (8)

  Addition with carry:
    0110 1100 (108)
  + 0011 0101 (53)
  ──────────────
    1010 0001 (161)

  Wraparound (overflow):
    1111 1111 (255)
  + 0000 0001 (1)
  ──────────────
  1 0000 0000 -> truncated to 8 bits -> 0000 0000 (0)
  Carry flag = 1 (carry out)

  Unsigned integer subtraction:
    Actually performed as "add the two's complement"
    5 - 3 -> 5 + (-3) -> 5 + (256 - 3) -> 5 + 253 = 258 -> 8-bit: 2

    0000 0101 (5)
  + 1111 1101 (253 = two's complement representation of -3)
  ──────────────
  1 0000 0010 -> discard carry -> 0000 0010 (2)


Unsigned integer multiplication:

  8-bit x 8-bit -> max 255 x 255 = 65,025 -> 16 bits needed
  -> Multiplication result requires up to twice the original bit width

  Practical notes:
  - C language: unsigned char multiplication is promoted to int before execution
  - Storing the result back to the original type may cause overflow
  - Safer to perform intermediate calculations in a wider type
```

### 1.3 Unsigned Integers in Each Language

```python
# Python: integers have no upper limit (arbitrary-precision integers)
x = 2**64  # 18446744073709551616 -- handled without issue
x = 2**1000  # Also no problem

# However, ctypes and struct have fixed-width limitations
import struct
struct.pack('B', 255)   # uint8: OK
# struct.pack('B', 256)   # struct.error: ubyte format requires 0 <= number <= 255

# struct format characters
# 'B' = uint8,  'b' = int8
# 'H' = uint16, 'h' = int16
# 'I' = uint32, 'i' = int32
# 'Q' = uint64, 'q' = int64

# Fixed-width integers with ctypes
import ctypes
val = ctypes.c_uint8(255)
print(val.value)  # 255
val = ctypes.c_uint8(256)
print(val.value)  # 0 (wraparound)

# Fixed-width integers with numpy
import numpy as np
a = np.uint8(255)
print(a + np.uint8(1))  # 0 (wraparound, with warning)
```

```rust
// Rust: explicit type annotation is required
let a: u8 = 255;    // OK
// let b: u8 = 256;    // Compile error!
let c: u32 = 4_294_967_295;  // OK (underscores for readability)
let d: u64 = 18_446_744_073_709_551_615;  // OK

// Type conversion
let small: u8 = 200;
let large: u32 = small as u32;   // Safe widening (remains 200)
let back: u8 = large as u8;     // Truncation (200 is preserved)

// u16 -> u8 truncation
let big: u16 = 300;
let truncated: u8 = big as u8;  // 300 - 256 = 44

// usize: platform-dependent size (32-bit OS = 32bit, 64-bit OS = 64bit)
let index: usize = 42;  // Used for array indexing
```

```go
// Go: clear type system
var a uint8 = 255
var b uint16 = 65535
var c uint32 = 4294967295
var d uint64 = 18446744073709551615

// uint: platform-dependent (32 or 64 bits)
var e uint = 42

// byte is an alias for uint8
var f byte = 0xFF

// Type conversion is explicit
var g uint32 = uint32(a)  // uint8 -> uint32
var h uint8 = uint8(b)    // uint16 -> uint8 (truncation)

// No overflow checking (wraparound)
var i uint8 = 255
i++  // i = 0 (wraparound, no error)

// Constants from the math package
import "math"
fmt.Println(math.MaxUint8)   // 255
fmt.Println(math.MaxUint16)  // 65535
fmt.Println(math.MaxUint32)  // 4294967295
```

```javascript
// JavaScript: Number type is 64-bit floating point
// -> The range of safely representable integers is limited
Number.MAX_SAFE_INTEGER  // 9007199254740991 (2^53 - 1)
Number.MIN_SAFE_INTEGER  // -9007199254740991

// Beyond the safe range, precision is lost
console.log(9007199254740992 === 9007199254740993);  // true! (indistinguishable)

// BigInt for arbitrary precision
const big = 18446744073709551615n;  // OK
const sum = big + 1n;  // 18446744073709551616n

// TypedArray for fixed-width unsigned integers
const u8 = new Uint8Array([255]);
const u16 = new Uint16Array([65535]);
const u32 = new Uint32Array([4294967295]);

// DataView for reading/writing binary data
const buffer = new ArrayBuffer(4);
const view = new DataView(buffer);
view.setUint32(0, 0xDEADBEEF, true);  // true = little-endian
console.log(view.getUint8(0).toString(16));  // 'ef'
```

```c
// C: fixed-width integer types (stdint.h recommended)
#include <stdint.h>
#include <limits.h>

uint8_t  a = 255;           // 0 to 255
uint16_t b = 65535;          // 0 to 65535
uint32_t c = 4294967295U;   // 0 to 4,294,967,295
uint64_t d = 18446744073709551615ULL;  // 0 to 2^64-1

// size_t: for representing memory sizes (always unsigned)
size_t len = sizeof(int);  // 4 or 8

// Traditional types (size is platform-dependent, not recommended)
unsigned char      uc;    // At least 8 bits
unsigned short     us;    // At least 16 bits
unsigned int       ui;    // At least 16 bits (typically 32 bits)
unsigned long      ul;    // At least 32 bits
unsigned long long ull;   // At least 64 bits

// Literal suffixes
uint32_t x = 42U;      // unsigned
uint64_t y = 42ULL;    // unsigned long long
```

```java
// Java: no unsigned integer types (partial support since Java 8)

// All integer types in Java are signed
byte  b = 127;     // -128 to 127
short s = 32767;   // -32768 to 32767
int   i = 2147483647;  // -2^31 to 2^31-1
long  l = 9223372036854775807L;  // -2^63 to 2^63-1

// Java 8+: unsigned operation methods on Integer/Long
int unsigned = Integer.parseUnsignedInt("4294967295");  // 0xFFFFFFFF
String str = Integer.toUnsignedString(unsigned);  // "4294967295"
int result = Integer.divideUnsigned(unsigned, 2);  // 2147483647
int cmp = Integer.compareUnsigned(-1, 1);  // positive value (0xFFFFFFFF > 1)

// Treating byte as unsigned
byte byteVal = (byte) 0xFF;  // Stored as -1
int unsignedByte = byteVal & 0xFF;  // Retrieved as 255
```

---

## 2. Signed Integers -- Two's Complement

### 2.1 Comparison of Methods for Representing Negative Numbers

```
Three methods for representing negative numbers (8-bit example):

  Method 1: Sign-Magnitude
  ─────────────────────────────
    Most significant bit = sign (0: positive, 1: negative)
    Remaining 7 bits = absolute value

    +5 = 0_0000101
    -5 = 1_0000101

    Problems:
    - Two zeros exist: +0 (0000 0000) and -0 (1000 0000)
    - Addition requires special circuitry
    - Range: -127 to +127

  Method 2: One's Complement
  ─────────────────────────────
    Negative number = bit-invert all bits

    +5 = 0000 0101
    -5 = 1111 1010

    Problems:
    - Two zeros: +0 (0000 0000) and -0 (1111 1111)
    - Carry handling required (end-around carry)
    - Range: -127 to +127

  Method 3: Two's Complement (the modern standard)
  ─────────────────────────────
    Negative number = bit-invert all bits + 1

    +5 = 0000 0101
    -5 = 1111 1011  (0000 0101 -> invert -> 1111 1010 -> +1 -> 1111 1011)

    Advantages:
    - Only one zero (0000 0000)
    - A single adder handles both addition and subtraction!
    - Range: -128 to +127 (asymmetric but rational)
```

### 2.2 Mathematical Understanding of Two's Complement

```
The essence of two's complement:
  -x = 2^N - x  (N = number of bits)

  For 8-bit: -x = 256 - x

  Example: -5 = 256 - 5 = 251 = 1111 1011

  Why this unifies addition:
  5 + (-5) = 5 + 251 = 256 = 1_0000_0000 (9 bits)
  -> Discard the MSB (carry) that doesn't fit in 8 bits -> 0000 0000 = 0

  3 + (-5) = 3 + 251 = 254 = 1111 1110
  -> Interpreted as two's complement: -2

  -> Hardware doesn't need to be aware of the sign; it simply adds!


Mathematical background (modular arithmetic / congruence):

  Two's complement is equivalent to arithmetic mod 2^N

  Example (mod 256):
  -5 = 251 (mod 256)   <- 256 - 5 = 251
  -1 = 255 (mod 256)   <- 256 - 1 = 255

  Addition: 3 + (-5) = 3 + 251 = 254 = -2 (mod 256)

  -> Two's complement is precisely the residue group Z/2^N Z
  -> The group operation for addition holds naturally, making it hardware-efficient


Important properties of two's complement:

  1. Sign determination: if MSB is 1, the number is negative
     0xxx xxxx -> positive (0 to 127)
     1xxx xxxx -> negative (-128 to -1)

  2. Sign extension: when widening bit width, replicate the MSB
     int8 -> int16: -5 (1111 1011) -> (1111 1111 1111 1011) = -5
     int8 -> int16:  5 (0000 0101) -> (0000 0000 0000 0101) = 5

  3. Negation: ~x + 1 = -x
     ~0000 0101 = 1111 1010
     1111 1010 + 1 = 1111 1011 = -5

  4. Absolute value: |x| = x if positive, ~x + 1 if negative
```

### 2.3 Complete Table of Two's Complement (8-bit)

```
8-bit two's complement table (key values):

  Binary        Decimal   Hex      Description
  ──────────────────────────────────────
  0111 1111    +127     0x7F    INT8_MAX
  0111 1110    +126     0x7E
  ...
  0000 0010      +2     0x02
  0000 0001      +1     0x01
  0000 0000       0     0x00    Zero
  1111 1111      -1     0xFF
  1111 1110      -2     0xFE
  1111 1101      -3     0xFD
  ...
  1000 0010    -126     0x82
  1000 0001    -127     0x81
  1000 0000    -128     0x80    INT8_MIN

  Pattern observations:
  - Positive numbers: 0x00-0x7F (0-127)
  - Negative numbers: 0x80-0xFF (-128 to -1)
  - -1 = all bits 1 (0xFF)
  - INT8_MIN = only MSB is 1 (0x80)
  - Around zero: ... -3, -2, -1, 0, +1, +2, +3 ...
  - As bit patterns, they are contiguous: ... FD, FE, FF, 00, 01, 02, 03 ...
```

### 2.4 Two's Complement Conversion Steps

```
Positive -> Negative conversion:

  Method 1: Invert all bits + 1
    +42 = 0010 1010
    Invert -> 1101 0101
    +1     -> 1101 0110 = -42

  Method 2: 2^N - x
    -42 = 256 - 42 = 214 = 1101 0110

  Method 3: Find the rightmost 1, invert all bits to its left
    +42 = 0010 1010
              ^ rightmost 1
    Invert -> 1101 0110 = -42

  Reverse conversion (negative -> positive): apply the same operation again
    -42 = 1101 0110
    Invert -> 0010 1001
    +1     -> 0010 1010 = +42


Several concrete examples:

  +1 -> -1:
    0000 0001 -> invert -> 1111 1110 -> +1 -> 1111 1111 = 0xFF = -1

  +100 -> -100:
    0110 0100 -> invert -> 1001 1011 -> +1 -> 1001 1100 = 0x9C = -100

  +127 -> -127:
    0111 1111 -> invert -> 1000 0000 -> +1 -> 1000 0001 = 0x81 = -127

  -128 -> ???:
    1000 0000 -> invert -> 0111 1111 -> +1 -> 1000 0000 = -128 (itself!)
    -> -128 is a special value with no symmetric positive counterpart in 8-bit
```

### 2.5 Two's Complement Addition and Subtraction

```
Worked examples of addition/subtraction in two's complement:

  Example 1: 50 + 30 = 80
    0011 0010 (50)
  + 0001 1110 (30)
  ──────────────
    0101 0000 (80)   Sign bit = 0, no overflow

  Example 2: 50 + (-30) = 20
    0011 0010 (50)
  + 1110 0010 (-30)
  ──────────────
  1 0001 0100 -> discard carry -> 0001 0100 (20)

  Example 3: -50 + (-30) = -80
    1100 1110 (-50)
  + 1110 0010 (-30)
  ──────────────
  1 1011 0000 -> discard carry -> 1011 0000
  1011 0000 = -(~1011 0000 + 1) = -(0100 1111 + 1) = -(0101 0000) = -80

  Example 4: 100 + 50 = 150 -> Overflow!
    0110 0100 (100)
  + 0011 0010 (50)
  ──────────────
    1001 0110 -> as two's complement: -106 (incorrect!)
  positive + positive = negative -> Overflow! (8-bit signed max is 127)

  Example 5: -100 + (-50) = -150 -> Overflow!
    1001 1100 (-100)
  + 1100 1110 (-50)
  ──────────────
  1 0110 1010 -> discard carry -> 0110 1010 = 106 (incorrect!)
  negative + negative = positive -> Overflow! (8-bit signed min is -128)

  Subtraction is converted to "adding the two's complement":
  A - B = A + (-B) = A + (~B + 1)

  Example: 30 - 50 = -20
    0001 1110 (30)
  + 1100 1110 (-50)  <- two's complement of 50
  ──────────────
    1110 1100 -> as two's complement: -20
```

### 2.6 Range of Signed Integers

```
Range of N-bit two's complement: -2^(N-1) to 2^(N-1) - 1

  Type     Bits   Minimum                        Maximum
  ──────────────────────────────────────────────────────
  int8     8        -128                       127
  int16    16       -32,768                    32,767
  int32    32       -2,147,483,648             2,147,483,647 (approx. +/- 2.1 billion)
  int64    64       -9,223,372,036,854,775,808  9,223,372,036,854,775,807

  Why does the negative side have one more value?
  ────────────────────
  For 8-bit:
  Positive max: 0111 1111 = +127
  Negative min: 1000 0000 = -128

  Inverting -128 and adding 1:
  1000 0000 -> 0111 1111 -> 1000 0000 = -128 (maps back to itself!)
  -> -128 has no corresponding positive value via the inversion operation

  Symmetry issue:
  - abs(INT_MIN) overflows!
  - abs(-128) is 128, but 128 cannot be represented in int8
  - C: abs(INT_MIN) is undefined behavior
  - Java: Math.abs(Integer.MIN_VALUE) returns Integer.MIN_VALUE

  Safe absolute value calculation:
  long safe_abs(int x) {
      return (long)x < 0 ? -(long)x : (long)x;  // Convert to a wider type
  }
```

### 2.7 Sign Extension and Zero Extension

```
Sign Extension: widening a signed integer's bit width

  int8 -> int16:
  +5:  0000 0101 -> 0000 0000 0000 0101  (extend MSB 0 to the left)
  -5:  1111 1011 -> 1111 1111 1111 1011  (extend MSB 1 to the left)

  int16 -> int32:
  -100: 1111 1111 1001 1100
      -> 1111 1111 1111 1111 1111 1111 1001 1100

  Rule: copy the MSB (sign bit) to the new bits
  -> The value is preserved


Zero Extension: widening an unsigned integer's bit width

  uint8 -> uint16:
  200: 1100 1000 -> 0000 0000 1100 1000  (pad with zeros on the left)
  255: 1111 1111 -> 0000 0000 1111 1111

  Rule: always pad with zeros on the left
  -> The value is preserved


Caution in C:
  int8_t x = -5;     // 1111 1011
  uint16_t y = x;    // Sign extension -> 1111 1111 1111 1011 -> as uint16: 65531!
  // Intention: -5 -> becomes 65531
  // Correct: int16_t y = x; for sign extension

  uint8_t a = 200;   // 1100 1000
  int16_t b = a;     // Zero extension -> 0000 0000 1100 1000 -> 200
  // OK: value is preserved when converting unsigned -> signed (if within range)
```

```python
# Simulating sign extension in Python

def sign_extend(value, from_bits, to_bits):
    """Sign extend: widen a signed integer from from_bits to to_bits"""
    # Check the sign bit
    if value & (1 << (from_bits - 1)):
        # Negative: fill upper bits with 1s
        mask = ((1 << to_bits) - 1) ^ ((1 << from_bits) - 1)
        return value | mask
    return value

# int8 -> int32
print(sign_extend(0xFB, 8, 32))   # 0xFFFFFFFB = -5 (32-bit)
print(sign_extend(0x05, 8, 32))   # 0x00000005 = +5 (32-bit)

# 8-bit signed -> Python integer
def int8_to_python(byte_val):
    """Interpret a uint8 value as a signed int8"""
    if byte_val & 0x80:
        return byte_val - 256
    return byte_val

print(int8_to_python(0xFB))  # -5
print(int8_to_python(0x05))  # 5
print(int8_to_python(0x80))  # -128
```

---

## 3. Overflow

### 3.1 What Is Overflow

```
Overflow: when the result of an operation exceeds the representable range

  Unsigned 8-bit:
  255 + 1 = 256 -> 0 (wraparound)

  1111 1111
+ 0000 0001
──────────
1 0000 0000 -> truncated to 8 bits -> 0000 0000 = 0

  Signed 8-bit (two's complement):
  127 + 1 = 128? -> -128 (overflow!)

  0111 1111  (+127)
+ 0000 0001  (+1)
──────────
  1000 0000  (-128)  <- positive + positive = negative is clearly wrong

  Overflow detection:
  - positive + positive = negative -> overflow
  - negative + negative = positive -> overflow
  - positive + negative never overflows
```

### 3.2 Unsigned vs Signed Overflow

```
Unsigned overflow (wraparound):
  In C, this is "well-defined behavior"
  Result is mod 2^N

  uint8: 255 + 1 = 0
  uint8: 0 - 1 = 255
  uint16: 65535 + 1 = 0
  uint32: 4294967295 + 1 = 0

  Use cases:
  - Hash computation (exploiting wraparound)
  - Counters (wrapping back to 0 by design)
  - CRC calculation


Signed overflow:
  In C/C++, this is "undefined behavior"!
  -> The compiler may assume it "never happens" and optimize accordingly

  Example:
  int x = INT_MAX;
  if (x + 1 > x) {  // The compiler may assume this is always true
      // ...          // The overflow check could be eliminated
  }

  GCC -O2 optimization example:
  // Original code
  int check_overflow(int x) {
      return x + 1 > x;
  }
  // After optimization: always returns 1 (assumes overflow doesn't happen)

  -> -fwrapv option guarantees signed wraparound behavior
  -> -ftrapv option traps on overflow
```

### 3.3 Real-World Bugs and Incidents

```python
# Famous overflow incidents

# 1. Ariane 5 Rocket Explosion (1996)
#    64-bit floating point -> 16-bit signed integer conversion
#    Horizontal velocity exceeded 32,767, causing overflow -> loss of control -> explosion
#    Damage: $500 million

# 2. Pac-Man Level 256 Bug (1980)
#    Level counter managed as an 8-bit unsigned integer
#    Clearing level 255 -> level 256 = 0x100 -> 0x00 in 8 bits
#    The right half of the screen became the garbled "kill screen"

# 3. Boeing 787 Power Loss (2015)
#    32-bit counter reached 2^31 after 248 days
#    int32 overflow -> power control system shutdown
#    Mitigation: restart within 248 days (!)

# 4. Year 2038 Problem (Y2K38)
#    Unix time: seconds since January 1, 1970 (int32)
#    2^31 - 1 = 2,147,483,647 seconds = January 19, 2038 03:14:07 UTC
#    -> Migration to int64 is required
import time
# Timestamp of the 2038 problem
print(2**31 - 1)  # 2147483647
# Most modern systems have already migrated to 64-bit

# 5. Civilization's "Nuclear Gandhi" (famous, though somewhat legendary)
#    Gandhi's aggression parameter (uint8) was 1; adopting democracy subtracted 2
#    1 - 2 = -1 -> as uint8: 255 (maximum) -> hyper-aggressive
#    Note: it has been suggested this may have been by design rather than a bug

# 6. Heap Buffer Overflow (security)
#    Integer overflow can be an entry point for buffer overflow attacks
#    Example: size_t size = user_input_width * user_input_height * 4;
#    Large width and height cause the multiplication to overflow
#    -> A small buffer is allocated
#    -> Heap corruption on write -> arbitrary code execution
```

### 3.4 The Binary Search Overflow Bug

```python
# Famous binary search bug (discovered in JDK 6)

# Bad: classic midpoint calculation (potential overflow)
def binary_search_buggy(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2  # <- low + high can overflow!
        # e.g., low=2^30, high=2^30+100 -> low+high > INT_MAX
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Good: safe midpoint calculation
def binary_search_safe(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = low + (high - low) // 2  # Overflow-safe
        # Or: mid = (low + high) >>> 1  (unsigned right shift in Java/C#)
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Overflow-safe average via bit manipulation
def safe_average(a, b):
    """Average without overflow"""
    return (a & b) + ((a ^ b) >> 1)
    # Common bits + half of the differing bits
```

### 3.5 Overflow Countermeasures in Each Language

```python
# Python: arbitrary-precision integers -> no overflow!
x = 2**100 + 1  # No problem
# Python is the only language where you don't need to worry about integer overflow

# However, ctypes/struct/numpy have fixed widths, so overflow exists
import numpy as np
a = np.int8(127)
print(a + np.int8(1))  # -128 (wraparound)

# Safe fixed-width arithmetic
def safe_add_int32(a, b):
    """Simulate int32 overflow in Python"""
    result = a + b
    if result > 2**31 - 1 or result < -(2**31):
        raise OverflowError(f"int32 overflow: {a} + {b} = {result}")
    return result
```

```rust
// Rust: compile-time and runtime detection
let x: u8 = 255;

// Debug build: panic (program halts)
// let y = x + 1;  // thread 'main' panicked at 'attempt to add with overflow'

// Release build: wraparound (default)
// Explicit overflow control methods:
let a = x.checked_add(1);    // Option<u8> -> None
let b = x.saturating_add(1); // 255 (saturates at the upper limit)
let c = x.wrapping_add(1);   // 0 (explicit wraparound)
let d = x.overflowing_add(1); // (0, true) -- value and overflow flag

// Practical usage guidelines:
// checked_add: when correctness is paramount (finance, scientific computing)
// saturating_add: when clamping at limits is desired (volume, brightness)
// wrapping_add: when wraparound is by design (hash, counters)
// overflowing_add: when you want to know if overflow occurred (low-level CPU emulation)

// Practical example of saturating
fn adjust_volume(current: u8, delta: i8) -> u8 {
    if delta >= 0 {
        current.saturating_add(delta as u8)
    } else {
        current.saturating_sub((-delta) as u8)
    }
}
// adjust_volume(250, 10) -> 255 (saturates at 255)
// adjust_volume(5, -10) -> 0 (saturates at 0)
```

```java
// Java: silent wraparound (dangerous!)
int x = Integer.MAX_VALUE;  // 2147483647
int y = x + 1;              // -2147483648 (no warning!)

// Java 8+: Math.addExact()
try {
    int z = Math.addExact(x, 1);  // ArithmeticException
} catch (ArithmeticException e) {
    System.out.println("Overflow detected!");
}

// Safe arithmetic methods in the Math class
Math.addExact(a, b);       // Addition (throws on overflow)
Math.subtractExact(a, b);  // Subtraction
Math.multiplyExact(a, b);  // Multiplication
Math.negateExact(a);       // Negation
Math.incrementExact(a);    // +1
Math.decrementExact(a);    // -1
Math.toIntExact(longVal);  // long -> int (throws if out of range)
```

```c
// C: signed overflow is undefined behavior (most dangerous)
int x = INT_MAX;
int y = x + 1;  // Undefined behavior! The compiler can do anything it wants.
// GCC optimizations may eliminate overflow checks

// Safe addition check (signed):
#include <limits.h>
#include <stdbool.h>

bool safe_add_int(int a, int b, int *result) {
    if (b > 0 && a > INT_MAX - b) return false;  // Overflow
    if (b < 0 && a < INT_MIN - b) return false;  // Underflow
    *result = a + b;
    return true;
}

// Safe multiplication check (signed):
bool safe_mul_int(int a, int b, int *result) {
    if (a == 0 || b == 0) {
        *result = 0;
        return true;
    }
    if (a > 0 && b > 0 && a > INT_MAX / b) return false;
    if (a > 0 && b < 0 && b < INT_MIN / a) return false;
    if (a < 0 && b > 0 && a < INT_MIN / b) return false;
    if (a < 0 && b < 0 && a < INT_MAX / b) return false;
    *result = a * b;
    return true;
}

// GCC/Clang builtins:
int result;
if (__builtin_add_overflow(a, b, &result)) {
    // Overflow occurred
}
if (__builtin_mul_overflow(a, b, &result)) {
    // Overflow occurred
}
```

```go
// Go: silent wraparound (similar to Java)
package main

import (
    "fmt"
    "math"
)

func main() {
    var x int32 = math.MaxInt32  // 2147483647
    x++  // -2147483648 (wraparound, no error)
    fmt.Println(x)

    // Safe addition
    a, b := int32(2000000000), int32(1000000000)
    if safeAddInt32(a, b) {
        fmt.Println("OK:", a+b)
    } else {
        fmt.Println("Overflow!")
    }
}

func safeAddInt32(a, b int32) bool {
    if b > 0 && a > math.MaxInt32-b {
        return false
    }
    if b < 0 && a < math.MinInt32-b {
        return false
    }
    return true
}
```

---

## 4. Endianness (Byte Order)

### 4.1 Big-Endian and Little-Endian

```
Endianness: the byte order when storing multi-byte values in memory

  Value: 0x12345678 (32-bit integer)

  Big-Endian:
  Address:  0x00  0x01  0x02  0x03
  Value:    0x12  0x34  0x56  0x78
  -> Most significant byte (MSB) stored at the lowest address
  -> Same order as human reading
  -> Standard for network communication (network byte order)

  Little-Endian:
  Address:  0x00  0x01  0x02  0x03
  Value:    0x78  0x56  0x34  0x12
  -> Least significant byte (LSB) stored at the lowest address
  -> Intel/AMD x86/x64, ARM (default)
  -> Lower bytes are processed first during addition, simplifying circuitry

  Bi-Endian:
  -> Switchable. ARM, MIPS, PowerPC
  -> ARM is practically used in little-endian mode in most cases
```

### 4.2 Practical Impact of Endianness

```python
import struct

value = 0x12345678

# Pack as big-endian
big = struct.pack('>I', value)
print(big.hex())  # '12345678'

# Pack as little-endian
little = struct.pack('<I', value)
print(little.hex())  # '78563412'

# Note for network communication:
# Network = big-endian
# x86 PC = little-endian
# -> Byte order conversion is needed during send/receive

import socket
# Host byte order -> network byte order
port = 8080
network_port = socket.htons(port)  # host to network short

ip = 0xC0A80001  # 192.168.0.1
network_ip = socket.htonl(ip)  # host to network long

# Byte swap implementation using bit operations
def bswap32(n):
    return ((n & 0xFF000000) >> 24) | \
           ((n & 0x00FF0000) >> 8) | \
           ((n & 0x0000FF00) << 8) | \
           ((n & 0x000000FF) << 24)

def bswap16(n):
    return ((n & 0xFF00) >> 8) | ((n & 0x00FF) << 8)

print(hex(bswap32(0x12345678)))  # 0x78563412
print(hex(bswap16(0x1234)))      # 0x3412
```

### 4.3 Detecting Endianness

```python
import sys
print(sys.byteorder)  # 'little' (x86/ARM) or 'big'

# Determining endianness by examining the beginning of binary files:
# BMP image: starts with 'BM' (0x42 0x4D) -> little-endian
# JPEG: starts with 0xFF 0xD8 -> endianness-independent
# ELF: offset 5 contains 1 (LE) or 2 (BE)
# UTF-16 BOM: 0xFE 0xFF (BE) or 0xFF 0xFE (LE)

# Practical code for endianness detection
def detect_endianness():
    """Determine the endianness of the execution environment"""
    import struct
    if struct.pack('@I', 1) == struct.pack('<I', 1):
        return 'little'
    else:
        return 'big'
```

### 4.4 Binary Protocols and Endianness

```python
# Practical: designing and implementing binary protocols

import struct

# Packet header example (network byte order = big-endian)
class PacketHeader:
    FORMAT = '>HHI'  # Big-endian: uint16 type, uint16 length, uint32 sequence
    SIZE = struct.calcsize(FORMAT)

    def __init__(self, msg_type, length, sequence):
        self.msg_type = msg_type
        self.length = length
        self.sequence = sequence

    def pack(self):
        return struct.pack(self.FORMAT, self.msg_type, self.length, self.sequence)

    @classmethod
    def unpack(cls, data):
        msg_type, length, sequence = struct.unpack(cls.FORMAT, data[:cls.SIZE])
        return cls(msg_type, length, sequence)

# Usage example
header = PacketHeader(msg_type=1, length=100, sequence=42)
packed = header.pack()
print(packed.hex())  # '0001006400000002a'

# Receiving side
received = PacketHeader.unpack(packed)
print(f"Type: {received.msg_type}, Len: {received.length}, Seq: {received.sequence}")


# Binary file format example
class BMPHeader:
    """BMP image header (little-endian)"""
    FORMAT = '<2sIHHI'  # Little-endian: signature, filesize, reserved1, reserved2, data_offset

    @classmethod
    def read(cls, filepath):
        with open(filepath, 'rb') as f:
            data = f.read(struct.calcsize(cls.FORMAT))
            sig, size, r1, r2, offset = struct.unpack(cls.FORMAT, data)
            return {
                'signature': sig,  # b'BM'
                'filesize': size,
                'data_offset': offset
            }
```

---

## 5. Fixed-Point Numbers

### 5.1 How Fixed-Point Works

```
Fixed-point numbers: handle decimals via integer arithmetic with a fixed decimal point position

  Q8.8 format (16 bits: 8-bit integer part + 8-bit fractional part):

  Bits: IIIIIIII.FFFFFFFF

  Example: representing 3.75 in Q8.8
  Integer part: 3 = 0000 0011
  Fractional part: 0.75 = 0.5 + 0.25 = 2^(-1) + 2^(-2) = 1100 0000
  Result: 0000 0011.1100 0000 = 0x03C0

  Stored value = real value x 2^(fractional bits)
  3.75 x 256 = 960 = 0x03C0

  Reverse conversion: real value = stored value / 2^(fractional bits)
  960 / 256 = 3.75


Fixed-point arithmetic:

  Addition/subtraction: perform integer addition directly (when decimal point positions match)
    3.75 + 1.25 -> 960 + 320 = 1280 -> 1280/256 = 5.0

  Multiplication: right-shift the result (by the number of fractional bits)
    3.75 x 2.0 -> 960 x 512 = 491520 -> 491520 >> 8 = 1920 -> 1920/256 = 7.5

  Division: left-shift the dividend, then divide
    3.75 / 2.0 -> (960 << 8) / 512 = 245760 / 512 = 480 -> 480/256 = 1.875


Commonly used fixed-point formats:

  Q1.15 (16-bit): Signal processing (-1.0 to +0.999969)
  Q8.8  (16-bit): General purpose (-128.0 to +127.996)
  Q16.16 (32-bit): Games/graphics
  Q1.31 (32-bit): High-precision signal processing
  Q32.32 (64-bit): High-precision computation


Use cases:
  - Financial calculations (currency is fixed to 2 decimal places)
  - Embedded systems (microcontrollers without an FPU)
  - Games (3D graphics in the DSP era)
  - Audio processing (DSP)
  - GPS coordinates (integers in micro-degree units)
```

### 5.2 Fixed-Point Implementation

```python
# Fixed-point number library implementation

class FixedPoint:
    """Q16.16 fixed-point number"""
    FRAC_BITS = 16
    SCALE = 1 << FRAC_BITS  # 65536
    MASK = (1 << 32) - 1     # 32-bit mask

    def __init__(self, value=0):
        if isinstance(value, float):
            self._raw = int(value * self.SCALE)
        elif isinstance(value, int):
            self._raw = value * self.SCALE
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    @classmethod
    def from_raw(cls, raw):
        """Create directly from internal value"""
        obj = cls.__new__(cls)
        obj._raw = raw
        return obj

    def to_float(self):
        return self._raw / self.SCALE

    def __add__(self, other):
        return FixedPoint.from_raw(self._raw + other._raw)

    def __sub__(self, other):
        return FixedPoint.from_raw(self._raw - other._raw)

    def __mul__(self, other):
        # Right-shift by fractional bits after multiplication
        return FixedPoint.from_raw((self._raw * other._raw) >> self.FRAC_BITS)

    def __truediv__(self, other):
        # Left-shift dividend before dividing
        return FixedPoint.from_raw((self._raw << self.FRAC_BITS) // other._raw)

    def __repr__(self):
        return f"FixedPoint({self.to_float():.6f})"

    def __eq__(self, other):
        return self._raw == other._raw

# Usage example
a = FixedPoint(3.75)
b = FixedPoint(2.0)
print(a + b)      # FixedPoint(5.750000)
print(a - b)      # FixedPoint(1.750000)
print(a * b)      # FixedPoint(7.500000)
print(a / b)      # FixedPoint(1.875000)
```

### 5.3 Using Integers for Financial Calculations

```python
# Bad: financial calculations with floating point (dangerous!)
price = 0.1 + 0.2
print(price)  # 0.30000000000000004
print(price == 0.3)  # False!

# Good: financial calculations with integers (cent units)
price_cents = 10 + 20  # 30 cents
print(price_cents / 100)  # 0.3

# Good: use Decimal type
from decimal import Decimal, ROUND_HALF_UP
price = Decimal('0.1') + Decimal('0.2')
print(price)  # 0.3
print(price == Decimal('0.3'))  # True

# Best practices for currency calculations
class Money:
    """Integer-based monetary representation"""

    def __init__(self, amount_cents):
        self._cents = int(amount_cents)

    @classmethod
    def from_string(cls, s):
        """'1234.56' -> Money(123456)"""
        d = Decimal(s) * 100
        return cls(int(d))

    @classmethod
    def from_float(cls, f):
        """From float (not recommended, but sometimes necessary)"""
        return cls(round(f * 100))

    def __add__(self, other):
        return Money(self._cents + other._cents)

    def __sub__(self, other):
        return Money(self._cents - other._cents)

    def __mul__(self, factor):
        """Amount x quantity"""
        result = Decimal(self._cents) * Decimal(str(factor))
        return Money(int(result.quantize(Decimal('1'), rounding=ROUND_HALF_UP)))

    def __repr__(self):
        sign = '-' if self._cents < 0 else ''
        abs_cents = abs(self._cents)
        return f"${sign}{abs_cents // 100}.{abs_cents % 100:02d}"

# Usage example
item = Money.from_string('1980')     # $1980.00
tax = item * Decimal('0.1')          # $198.00
total = item + tax                    # $2178.00
print(total)                          # $2178.00


# Best practices in practice
# Database: DECIMAL(10, 2) -- 10 integer digits, 2 decimal digits
# JavaScript: handle all amounts as cents (integers), convert only for display
# Java: use BigDecimal
# Python: use decimal.Decimal
```

---

## 6. Integer Type Selection Guide

### 6.1 Type Selection Guidelines

```
Integer type selection flowchart:

  1. Do you need negative numbers?
     YES -> Signed integer
     NO  -> Unsigned integer

  2. What range is needed?
     +-------------------------------------+
     | Range                   Recommended  |
     +-------------------------------------+
     | 0-255                   uint8/byte  |
     | 0-65535                 uint16      |
     | 0-~4.3 billion          uint32      |
     | Larger                  uint64      |
     | -128 to 127             int8        |
     | -32768 to 32767         int16       |
     | -~2.1 billion to ~2.1B  int32       |
     | Larger                  int64       |
     | Arbitrarily large       BigInteger  |
     +-------------------------------------+

  3. Special use cases:
     - Array index: size_t / usize (C/Rust)
     - Timestamp: int64 (to avoid the 2038 problem)
     - ID/hash: uint64
     - Currency: Decimal / BigDecimal
     - Flags: uint8 / uint16 / uint32
     - Loop counter: int (language default integer type)

  4. Performance considerations:
     - The CPU's native width (32/64-bit) is the fastest
     - uint8/uint16 may incur extension/truncation costs
     - However, smaller types can be advantageous if memory bandwidth is the bottleneck
     - SIMD: smaller types -> more elements processed simultaneously
```

### 6.2 Default Integer Types by Language

```
Default integer types and recommendations by language:

  Python:  int (arbitrary precision, no overflow)
           -> No need to worry about type selection

  Go:      int (platform-dependent: 32 or 64 bits)
           -> Use int32, int64 when a specific size is needed
           -> Use int for array indices

  Rust:    i32 (default inferred type)
           -> Should always specify types explicitly
           -> Use usize for array indices

  Java:    int (32-bit signed)
           -> long is often needed (timestamps, etc.)
           -> unsigned via Integer.toUnsignedXxx() methods

  C/C++:   int (at least 16 bits, typically 32 bits)
           -> Should use fixed-width types from stdint.h

  JavaScript: Number (64-bit floating point)
           -> Integer precision is 53 bits (MAX_SAFE_INTEGER)
           -> BigInt for arbitrary precision

  Swift:   Int (platform-dependent: 32 or 64 bits)
           -> Typically use Int
           -> Int8, UInt32, etc. for special cases

  C#:      int (32-bit signed)
           -> long, uint, ulong also available
           -> BigInteger (System.Numerics) for arbitrary precision
```

### 6.3 Pitfalls of Implicit Type Conversion

```c
// Implicit type conversion in C (integer promotion)

// 1. Integer promotion: types smaller than int are converted to int
uint8_t a = 200;
uint8_t b = 100;
uint8_t c = a + b;  // 200 + 100 = 300 -> int(300) -> uint8(44)
// a and b are promoted to int for addition, result truncated to uint8

// 2. Mixed signed/unsigned arithmetic
int x = -1;
unsigned int y = 1;
if (x < y) {
    printf("x < y\n");  // Expected output
} else {
    printf("x >= y\n");  // Actually prints this!
}
// -1 is converted to unsigned int -> 0xFFFFFFFF = 4294967295 > 1

// 3. Implicit conversion during comparisons
int len = -1;
if (len < sizeof(int)) {
    // sizeof returns size_t (unsigned)
    // len(-1) is converted to size_t -> huge positive value
    // -> This condition is false!
}

// Safe pattern
if (len >= 0 && (size_t)len < sizeof(int)) {
    // Check for negative value first
}
```

```python
# Type conversion pitfalls in Python

# Python 3's // operator (floor division)
print(7 // 2)     # 3 (truncates toward zero for positive numbers)
print(-7 // 2)    # -4 (truncates toward negative infinity)
# Different from C's -7 / 2 = -3 (truncates toward zero)!

# Python 3's % operator (modulo)
print(7 % 2)      # 1
print(-7 % 2)     # 1 (Python's modulo always has the same sign as the divisor)
# Different from C's -7 % 2 = -1!

# int to bool
bool(0)    # False
bool(1)    # True
bool(-1)   # True (anything non-zero is True)

# bool to int
int(True)  # 1
int(False) # 0
True + True  # 2 (bool is a subclass of int)
```

---

## 7. Practical Exercises

### Exercise 1: Two's Complement (Basic)
Perform the following calculations by hand in 8-bit two's complement:
1. The bit representation of -42
2. The addition 50 + (-30)
3. The addition -100 + (-50) (does overflow occur?)

### Exercise 2: Overflow Detection (Intermediate)
In your language of choice, implement a function that determines whether the addition of two 32-bit signed integers will overflow. Do so without using widening to 64-bit integers.

### Exercise 3: Endianness Conversion (Advanced)
Implement a program that reads a 4-byte integer from a binary file and displays the value interpreted in both little-endian and big-endian.

### Exercise 4: Fixed-Point Arithmetic (Intermediate)
Calculate the following using Q8.8 fixed-point numbers and compare the results to float:
1. 3.14 + 2.71
2. 3.14 x 2.71
3. 10.0 / 3.0

### Exercise 5: Type Conversion Pitfalls (Practical)
Predict the output of the following C code and explain why it produces that result:
```c
unsigned int a = 1;
int b = -1;
printf("%d\n", a > b);  // ???
printf("%u\n", b);       // ???
```

### Exercise Solutions

```python
# Exercise 1 Solutions

# 1. Bit representation of -42 (8-bit two's complement)
# +42 = 0010 1010
# Invert = 1101 0101
# +1    = 1101 0110 = 0xD6 = -42
print(f"-42 = {(-42) & 0xFF:08b} = 0x{(-42) & 0xFF:02X}")
# -42 = 11010110 = 0xD6

# 2. 50 + (-30) addition
#   0011 0010 (50)
# + 1110 0010 (-30)
# = 1 0001 0100 -> discard carry -> 0001 0100 = 20
print(f"50 + (-30) = {(50 + (-30)) & 0xFF}")
# 50 + (-30) = 20

# 3. -100 + (-50) addition
# -100 = 1001 1100 (0x9C)
# -50  = 1100 1110 (0xCE)
#   1001 1100
# + 1100 1110
# = 1 0110 1010 -> discard carry -> 0110 1010 = 106
# negative + negative = positive -> Overflow! (-150 is outside the 8-bit signed range)
result = ((-100) & 0xFF) + ((-50) & 0xFF)
print(f"-100 + (-50) = {result & 0xFF} (unsigned), interpreted as {result & 0xFF if result & 0xFF < 128 else (result & 0xFF) - 256}")
# -> 106 (positive value) = Overflow (correct result would be -150)


# Exercise 2 Solution
def will_overflow_int32(a, b):
    """Determine whether 32-bit signed integer addition will overflow"""
    INT32_MAX = 2**31 - 1   # 2147483647
    INT32_MIN = -(2**31)    # -2147483648

    if b > 0 and a > INT32_MAX - b:
        return True  # Positive overflow
    if b < 0 and a < INT32_MIN - b:
        return True  # Negative overflow
    return False

# Test
print(will_overflow_int32(2**31 - 1, 1))      # True
print(will_overflow_int32(2**31 - 1, 0))      # False
print(will_overflow_int32(-(2**31), -1))       # True
print(will_overflow_int32(100, -50))           # False


# Exercise 3 Solution
import struct

def read_as_both_endian(data):
    """Interpret 4 bytes in both endiannesses"""
    le_value = struct.unpack('<I', data)[0]
    be_value = struct.unpack('>I', data)[0]
    le_signed = struct.unpack('<i', data)[0]
    be_signed = struct.unpack('>i', data)[0]

    print(f"Bytes: {data.hex()}")
    print(f"Little-Endian unsigned: {le_value} (0x{le_value:08X})")
    print(f"Big-Endian unsigned:    {be_value} (0x{be_value:08X})")
    print(f"Little-Endian signed:   {le_signed}")
    print(f"Big-Endian signed:      {be_signed}")

# Test
read_as_both_endian(b'\x12\x34\x56\x78')
# Bytes: 12345678
# Little-Endian unsigned: 2018915346 (0x78563412)
# Big-Endian unsigned:    305419896 (0x12345678)
# Little-Endian signed:   2018915346
# Big-Endian signed:      305419896


# Exercise 4 Solution
class Q8_8:
    """Q8.8 fixed-point number"""
    FRAC = 8
    SCALE = 256

    def __init__(self, value):
        if isinstance(value, float):
            self._raw = int(value * self.SCALE)
        else:
            self._raw = value * self.SCALE

    @classmethod
    def _from_raw(cls, raw):
        obj = cls.__new__(cls)
        obj._raw = raw
        return obj

    def to_float(self):
        return self._raw / self.SCALE

    def __add__(self, other):
        return Q8_8._from_raw(self._raw + other._raw)

    def __mul__(self, other):
        return Q8_8._from_raw((self._raw * other._raw) >> self.FRAC)

    def __truediv__(self, other):
        return Q8_8._from_raw((self._raw << self.FRAC) // other._raw)

    def __repr__(self):
        return f"Q8.8({self.to_float():.4f})"

a = Q8_8(3.14)
b = Q8_8(2.71)
print(f"3.14 + 2.71 = {(a + b)} (float: {3.14 + 2.71})")
print(f"3.14 * 2.71 = {(a * b)} (float: {3.14 * 2.71})")
c = Q8_8(10.0)
d = Q8_8(3.0)
print(f"10.0 / 3.0 = {(c / d)} (float: {10.0 / 3.0})")
# Fixed-point has limited precision, so small differences from float will occur


# Exercise 5 Solution
# unsigned int a = 1;
# int b = -1;
# printf("%d\n", a > b);
# -> Output: 0 (false)
# -> b is converted to unsigned int, -1 becomes 4294967295
# -> 1 > 4294967295 is false

# printf("%u\n", b);
# -> Output: 4294967295
# -> The bit pattern of -1 (0xFFFFFFFF) interpreted as unsigned
```


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify configuration file path and format |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Steps

1. **Check error messages**: Read the stack trace and identify the location of the error
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use log output or a debugger to validate hypotheses
5. **Fix and regression test**: After fixing, run tests on related areas as well

```python
# Debugging utilities
import logging
import traceback
from functools import wraps

# Logger setup
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

Steps for diagnosing performance issues:

1. **Identify the bottleneck**: Measure using profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Examine disk and network I/O conditions
4. **Check concurrent connections**: Inspect connection pool status

| Problem Type | Diagnostic Tool | Solution |
|-------------|----------------|----------|
| CPU load | cProfile, py-spy | Algorithm improvements, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Asynchronous I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes the criteria for technology selection decisions.

| Criterion | Prioritize when | Can compromise when |
|-----------|----------------|-------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+---------------------------------------------------+
|          Architecture Selection Flow               |
+---------------------------------------------------+
|                                                    |
|  (1) Team size?                                    |
|    +-- Small (1-5 people) -> Monolith              |
|    +-- Large (10+ people) -> Go to (2)             |
|                                                    |
|  (2) Deployment frequency?                         |
|    +-- Once a week or less -> Monolith + modules   |
|    +-- Daily/multiple times -> Go to (3)           |
|                                                    |
|  (3) Team independence?                            |
|    +-- High -> Microservices                       |
|    +-- Moderate -> Modular Monolith                |
|                                                    |
+---------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Costs**
- A fast short-term approach can become technical debt in the long run
- Conversely, over-engineering incurs high short-term costs and may delay the project

**2. Consistency vs Flexibility**
- A unified technology stack has lower learning costs
- Adopting diverse technologies enables best-fit solutions but increases operational costs

**3. Level of Abstraction**
- High abstraction offers greater reusability but can make debugging difficult
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
        """Describe the background and problem"""
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

## Practical Application Scenarios

### Scenario 1: MVP Development at a Startup

**Situation:** Need to release a product quickly with limited resources

**Approach:**
- Choose a simple architecture
- Focus on the minimum viable set of features
- Automated tests only for the critical path
- Introduce monitoring early on

**Lessons Learned:**
- Don't pursue perfection (YAGNI principle)
- Get user feedback early
- Manage technical debt consciously

### Scenario 2: Modernizing a Legacy System

**Situation:** Incrementally renewing a system that has been in operation for over 10 years

**Approach:**
- Use the Strangler Fig pattern for gradual migration
- Create Characterization Tests first if existing tests are absent
- Use an API gateway to allow old and new systems to coexist
- Perform data migration in stages

| Phase | Work | Estimated Duration | Risk |
|-------|------|-------------------|------|
| 1. Investigation | Current state analysis, dependency mapping | 2-4 weeks | Low |
| 2. Foundation | CI/CD setup, test environment | 4-6 weeks | Low |
| 3. Migration start | Sequential migration from peripheral features | 3-6 months | Medium |
| 4. Core migration | Migration of core features | 6-12 months | High |
| 5. Completion | Decommission the old system | 2-4 weeks | Medium |

### Scenario 3: Development with a Large Team

**Situation:** 50+ engineers working on the same product

**Approach:**
- Clarify boundaries with Domain-Driven Design
- Assign ownership per team
- Manage shared libraries via Inner Source
- Design API-first to minimize inter-team dependencies

```python
# API contract definition between teams
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

**Situation:** A system that requires millisecond-level response times

**Optimization Points:**
1. Cache strategy (L1: in-memory, L2: Redis, L3: CDN)
2. Leverage asynchronous processing
3. Connection pooling
4. Query optimization and index design

| Optimization Method | Impact | Implementation Cost | Applicable Scenario |
|--------------------|--------|-------------------|-------------------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Asynchronous processing | Medium | Medium | I/O-heavy processing |
| DB optimization | High | High | Slow queries |
| Code optimization | Low-Medium | High | CPU-bound cases |
---

## FAQ

### Q1: Why was two's complement adopted?
**A**: Because a single adder can handle both addition and subtraction, dramatically reducing hardware costs. Sign-magnitude requires separate circuits for addition and subtraction, plus complex handling of the two zeros (+0/-0). Two's complement is also mathematically elegant (the ring mod 2^N) and efficient to implement.

### Q2: Will the Year 2038 problem actually happen?
**A**: It can occur on systems that continue to use 32-bit time_t. Most desktop OSes and servers have already migrated to 64-bit. The real concern is embedded systems (IoT devices, industrial control equipment), many of which remain in operation with firmware that is difficult to update.

### Q3: Why do Python integers have no upper limit?
**A**: Python internally uses a variable-length integer representation (ob_digit array). It dynamically allocates memory as needed, allowing integers of arbitrary size as long as memory permits. The trade-off is that arithmetic is slower compared to fixed-width integers.

### Q4: Why is signed integer overflow undefined behavior in C?
**A**: Because the standard also accommodated platforms that used representations other than two's complement (sign-magnitude, one's complement). Additionally, making it undefined allows the compiler to assume "overflow never happens" and perform optimizations (loop unrolling, induction variable optimization, etc.). C23 now mandates two's complement.

### Q5: Why doesn't Java have unsigned integer types?
**A**: Designer James Gosling judged that "unsigned types are a source of confusion." This was because bugs from mixing signed/unsigned arithmetic in C were rampant. Since Java 8, unsigned operation methods (compareUnsigned, divideUnsigned, etc.) have been added to the Integer class.

### Q6: Why does the rounding direction of integer division differ between languages?
**A**: Because there are multiple mathematically rational definitions. C99/Java/Go use "truncation toward zero" (-7/2=-3), while Python uses "floor toward negative infinity" (-7//2=-4). Both have pros and cons, and the sign of the remainder changes accordingly. Python's approach is more mathematically consistent, but C's matches the hardware division instruction.

---

## Summary

| Concept | Key Points |
|---------|-----------|
| Unsigned integers | All bits used for value. Range: 0 to 2^N-1 |
| Two's complement | Negative = bit inversion + 1. Unified processing with a single adder |
| Sign extension | Replicate MSB when widening bit width. Zero-fill for unsigned |
| Overflow | Operation exceeds representable range. Behavior varies by language |
| C undefined behavior | Signed overflow is undefined. Beware of compiler optimization pitfalls |
| Endianness | Byte storage order. Network = BE, x86 = LE |
| Fixed-point | Fixed decimal point position. Used in finance and embedded systems |
| Type conversion pitfalls | Watch out for signed/unsigned mixing and integer promotion |

---

## Recommended Next Guides

---

## References
1. Bryant, R. E. & O'Hallaron, D. R. "Computer Systems: A Programmer's Perspective." Chapter 2.
2. Warren, H. S. "Hacker's Delight." 2nd Edition, Chapters 2-4.
3. Goldberg, D. "What Every Computer Scientist Should Know About Floating-Point Arithmetic." 1991.
4. IEEE. "IEEE 754-2019 Standard for Floating-Point Arithmetic."
5. ISO/IEC 9899:2024 (C23). "Programming Languages -- C."
6. Seacord, R. C. "Secure Coding in C and C++." 2nd Edition, Addison-Wesley, 2013.
7. Bloch, J. "Nearly All Binary Searches and Mergesorts are Broken." Google Research Blog, 2006.
