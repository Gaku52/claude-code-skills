# Floating-Point Numbers — IEEE 754 Complete Guide

> The fact that 0.1 + 0.2 !== 0.3 will continue to haunt programmers forever unless they understand the internal representation of floating-point numbers. This chapter systematically covers the structure of the IEEE 754 standard, the nature of precision problems, rounding error accumulation mechanisms, common pitfalls in numerical computing, and practical countermeasures.

---

## Learning Objectives

- [ ] Explain the structure of IEEE 754 (sign, exponent, mantissa) at the bit level
- [ ] Understand the differences and uses of normalized numbers, denormalized numbers, and special values
- [ ] Logically explain why 0.1 + 0.2 !== 0.3 using binary expansion
- [ ] Grasp the operating principles of rounding modes (round to nearest even, etc.)
- [ ] Identify and implement workarounds for catastrophic cancellation, absorption, and rounding error accumulation
- [ ] Apply practical precision countermeasures such as Kahan compensated summation and Decimal types
- [ ] Explain the design intent of low-precision formats for AI/GPU (BF16, FP8, etc.)
- [ ] Write safe floating-point comparison patterns in multiple languages

## Prerequisites


---

## 1. Why Floating-Point Is Necessary

### 1.1 Limitations of Fixed-Point

The most straightforward way to represent decimals in a computer is "fixed-point." The decimal point position is fixed within a bit string, with upper bits allocated to the integer part and lower bits to the fractional part.

```
Fixed-point representation (32-bit, 16.16 format):

  +-------------------+-------------------+
  | Integer: 16 bits  | Fraction: 16 bits |
  +-------------------+-------------------+

  Representable range:
    Integer part: 0 to 65535 (unsigned)
    Fractional part: precision of 1/65536 ~ 0.0000153

  Problem: Range of values in scientific computing
  -----------------------------------------
    Electron mass:        9.109 x 10^(-31) kg
    Avogadro's number:    6.022 x 10^(23) mol^(-1)
    Solar mass:           1.989 x 10^(30) kg
    Planck constant:      6.626 x 10^(-34) J*s

    Ratio of maximum to minimum: approximately 10^64

  With fixed-point 16.16 format:
    Representable range: approximately 0.0000153 to 65535.9999
    -> Neither electron mass nor solar mass can be represented
    -> Even increasing bit width is inefficient (200+ bits needed)

  Conclusion: Exponential notation is essential
    -> "Floating" point = dynamically shifting the decimal point position
    -> Covers a vast range with few bits
```

Fixed-point is still used in DSP (Digital Signal Processing) and parts of game engines, but floating-point is essential for general-purpose scientific computing.

### 1.2 Correspondence with Scientific Notation

The idea behind floating-point is scientific notation itself.

```
Scientific notation (decimal example):

  6.022 x 10^23
  ^        ^  ^
  significand base exponent

  Rules:
  - The significand is >= 1 and < 10 (normalized)
  - The base is 10
  - The exponent is an integer

IEEE 754 (binary version of scientific notation):

  (-1)^S x 1.M x 2^(E - bias)
    ^       ^   ^       ^
   sign  mantissa base  exponent (bias-corrected)

  Rules:
  - The base is 2 (fixed)
  - Normalization: the integer part of the mantissa is always 1 (implicit leading 1 bit)
  - The exponent field uses biased representation (unsigned integer - bias value)

Correspondence:
  +----------------+------------------+--------------------+
  | Scientific     | IEEE 754         | Role               |
  | Notation       |                  |                    |
  +----------------+------------------+--------------------+
  | +/- sign       | S (1 bit)        | Positive/negative  |
  | significand    | 1.M (implicit    | Significant digits |
  |  (1.xxx)       |  1 + M)          |                    |
  | x10^n          | x2^(E-bias)      | Scale (digit       |
  |                |                  |  position)         |
  +----------------+------------------+--------------------+
```

### 1.3 Historical Background of Floating-Point

Before IEEE 754 was established, each computer manufacturer used proprietary floating-point formats. IBM System/360 used a hexadecimal floating-point format (base 16), DEC VAX had its own F/D/G/H formats, and Cray used a proprietary 64-bit format. This lack of compatibility caused serious problems, with programs that worked correctly on one machine routinely producing different results on another.

In 1985, a group led by William Kahan established the IEEE 754 standard. The standard had the following design goals:

1. **Deterministic behavior**: Always return the same result for the same input
2. **Gradual underflow**: Instead of suddenly becoming zero near zero, precision is gradually lost
3. **Systematic handling of special values**: Represent infinity and undefined as values rather than exceptions
4. **Explicit control of rounding modes**: Specify multiple rounding methods

The standard was revised as IEEE 754-2008 in 2008 and IEEE 754-2019 in 2019, adding half precision (binary16), quadruple precision (binary128), and decimal floating-point (decimal32/64/128).

---

## 2. Structure of IEEE 754

### 2.1 Bit Layout

In IEEE 754, a floating-point number is stored by dividing it into three fields: "Sign," "Exponent," and "Mantissa / Significand."

```
IEEE 754 binary32 (single precision, 32 bits):

  Bit position: 31   30        23  22                    0
              +---+-----------+---------------------------+
              | S | Exponent E|       Mantissa M          |
              |1b |  8 bits   |       23 bits             |
              +---+-----------+---------------------------+

  S: Sign bit (0 = positive, 1 = negative)
  E: Exponent (biased unsigned integer)
  M: Mantissa (fractional part excluding the implicit leading 1)

IEEE 754 binary64 (double precision, 64 bits):

  Bit position: 63   62              52  51                                   0
              +---+-----------------+------------------------------------------+
              | S |   Exponent E    |              Mantissa M                  |
              |1b |   11 bits       |              52 bits                     |
              +---+-----------------+------------------------------------------+

IEEE 754 binary16 (half precision, 16 bits):

  Bit position: 15  14     10  9          0
              +---+--------+--------------+
              | S | Exp E  |  Mantissa M  |
              |1b | 5 bits |  10 bits     |
              +---+--------+--------------+
```

### 2.2 Comparison of Precision Formats

```
+------------+-------+-------+-------+--------------------------+---------------+
| Name       | Width | Exp   | Mant  | Representable range      | Decimal digits|
|            |       |       |       | (absolute value)         |               |
+------------+-------+-------+-------+--------------------------+---------------+
| binary16   | 16bit |  5bit | 10bit | +/-6.55 x 10^4          | ~3.3 digits   |
| binary32   | 32bit |  8bit | 23bit | +/-3.4 x 10^38          | ~7.2 digits   |
| binary64   | 64bit | 11bit | 52bit | +/-1.8 x 10^308         | ~15.9 digits  |
| binary128  |128bit | 15bit |112bit | +/-1.2 x 10^4932        | ~34.0 digits  |
+------------+-------+-------+-------+--------------------------+---------------+
| bfloat16   | 16bit |  8bit |  7bit | +/-3.4 x 10^38          | ~2.4 digits   |
| TF32       | 19bit |  8bit | 10bit | +/-3.4 x 10^38          | ~3.3 digits   |
| FP8(E4M3)  |  8bit |  4bit |  3bit | +/-448                  | ~1.2 digits   |
| FP8(E5M2)  |  8bit |  5bit |  2bit | +/-57344                | ~0.9 digits   |
+------------+-------+-------+-------+--------------------------+---------------+
```

### 2.3 Value Calculation Formula

The value of a floating-point number falls into three categories depending on the exponent field value.

```
* Normalized Numbers
  Condition: 0 < E < E_max (exponent is neither all 0s nor all 1s)

  Value = (-1)^S x (1 + M x 2^(-p)) x 2^(E - bias)

    Where:
      S     = sign bit (0 or 1)
      M     = integer value of the mantissa (0 to 2^p - 1)
      p     = number of mantissa bits (binary32: 23, binary64: 52)
      E     = integer value of the exponent field
      bias  = 2^(k-1) - 1 (k = number of exponent bits)

  Bias values:
    binary16: bias = 15    (E: 1-30  -> exponent: -14 to +15)
    binary32: bias = 127   (E: 1-254 -> exponent: -126 to +127)
    binary64: bias = 1023  (E: 1-2046 -> exponent: -1022 to +1023)
    binary128: bias = 16383

  Implicit Leading 1 Bit:
    The mantissa of normalized numbers always has the form 1.xxxxx...
    The leading 1 is not stored, gaining 1 bit of precision
    -> binary32 stores 23 bits but achieves 24-bit precision

* Denormalized Numbers (Subnormal Numbers)
  Condition: E = 0, M != 0

  Value = (-1)^S x (0 + M x 2^(-p)) x 2^(1 - bias)

  -> The implicit leading bit becomes 0
  -> The exponent is fixed at 1 - bias (note: not 0 - bias)

* Special Values
  E = 0,     M = 0  -> +/-0 (signed zero)
  E = E_max, M = 0  -> +/-Infinity
  E = E_max, M != 0 -> NaN (Not a Number)
```

### 2.4 Concrete Conversion Examples: Decimal to IEEE 754

#### Example 1: Convert 6.5 to binary32

```
Step 1: Determine the sign
  6.5 > 0, so S = 0

Step 2: Convert the absolute value to binary
  Integer part: 6 = 110 (binary)
  Fractional part: 0.5 = 0.1 (binary)  <- 0.5 x 2 = 1.0 -> 1
  6.5 = 110.1 (binary)

Step 3: Normalize (convert to 1.xxx x 2^n form)
  110.1 = 1.101 x 2^2

Step 4: Determine each field
  S = 0
  E = 2 + 127 = 129 = 10000001 (binary)
  M = 10100000000000000000000 (the part after "1.", 23 bits)

Step 5: Assemble the bit string
  0 10000001 10100000000000000000000
  ^  ^         ^
  S  E(8bit)   M(23bit)

  Hexadecimal: 0x40D00000

Verification (Python):
  >>> import struct
  >>> struct.pack('>f', 6.5).hex()
  '40d00000'  # Match
```

#### Example 2: Convert -12.375 to binary32

```
Step 1: S = 1 (negative number)

Step 2: Convert |-12.375| to binary
  Integer part: 12 = 1100 (binary)
  Fractional part:
    0.375 x 2 = 0.75  -> 0
    0.75  x 2 = 1.5   -> 1
    0.5   x 2 = 1.0   -> 1
    -> 0.375 = 0.011 (binary)
  12.375 = 1100.011 (binary)

Step 3: Normalize
  1100.011 = 1.100011 x 2^3

Step 4: Determine each field
  S = 1
  E = 3 + 127 = 130 = 10000010 (binary)
  M = 10001100000000000000000

Step 5: Bit string
  1 10000010 10001100000000000000000
  Hexadecimal: 0xC1460000
```

#### Example 3: Convert 0.1 to binary64 (infinite repeating example)

```
Converting 0.1 to binary:

  0.1 x 2 = 0.2  -> 0
  0.2 x 2 = 0.4  -> 0
  0.4 x 2 = 0.8  -> 0
  0.8 x 2 = 1.6  -> 1
  0.6 x 2 = 1.2  -> 1
  0.2 x 2 = 0.4  -> 0   <- "0011" repetition begins
  0.4 x 2 = 0.8  -> 0
  0.8 x 2 = 1.6  -> 1
  0.6 x 2 = 1.2  -> 1
  ...

  0.1 (decimal) = 0.0 0011 0011 0011 0011 0011 ... (binary, infinite repeating)

  Normalized: 1.1001100110011001100110011... x 2^(-4)

  In binary64, the mantissa is 52 bits, so rounding occurs at the 53rd bit:

  Stored mantissa (52 bits):
  1001100110011001100110011001100110011001100110011010
                                                  ^
                                        Rounding (round to nearest even)

  Value actually stored:
  0.1000000000000000055511151231257827021181583404541015625

  Difference from the true value 0.1: approximately 5.55 x 10^(-18)
  -> Very small, but problematic when accumulated
```

### 2.5 Reverse Conversion from IEEE 754: Decoding Bit Strings

```python
# Decoding bit strings in Python

import struct

def decode_float32(hex_str):
    """Decode IEEE 754 components from a 32-bit hexadecimal string"""
    n = int(hex_str, 16)
    sign = (n >> 31) & 1
    exponent = (n >> 23) & 0xFF
    mantissa = n & 0x7FFFFF

    print(f"Hexadecimal: {hex_str}")
    print(f"Binary:      {n:032b}")
    print(f"Sign (S):    {sign} ({'negative' if sign else 'positive'})")
    print(f"Exponent(E): {exponent} (= {exponent} - 127 = {exponent - 127})")
    print(f"Mantissa(M): {mantissa:023b}")

    if exponent == 0 and mantissa == 0:
        value = 0.0 * (-1 if sign else 1)
        print(f"Category: Zero ({'+' if not sign else '-'}0)")
    elif exponent == 0:
        value = (-1)**sign * (mantissa / 2**23) * 2**(-126)
        print(f"Category: Denormalized number")
    elif exponent == 255 and mantissa == 0:
        value = float('inf') * (-1 if sign else 1)
        print(f"Category: {'Negative' if sign else 'Positive'} infinity")
    elif exponent == 255:
        value = float('nan')
        print(f"Category: NaN")
    else:
        value = (-1)**sign * (1 + mantissa / 2**23) * 2**(exponent - 127)
        print(f"Category: Normalized number")

    print(f"Value:       {value}")
    return value

# Usage examples
decode_float32("40D00000")  # -> 6.5
decode_float32("C1460000")  # -> -12.375
decode_float32("3DCCCCCD")  # -> 0.10000000149011612 (approximation of 0.1)
```

```c
/* Decoding bit strings in C */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

void decode_float32(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));  /* Type punning (safe method) */

    uint32_t sign     = (bits >> 31) & 1;
    uint32_t exponent = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;

    printf("Value:    %.*g\n", 9, f);
    printf("Bits:     ");
    for (int i = 31; i >= 0; i--) {
        printf("%d", (bits >> i) & 1);
        if (i == 31 || i == 23) printf(" ");
    }
    printf("\n");
    printf("Sign:     %u (%s)\n", sign, sign ? "negative" : "positive");
    printf("Exponent: %u (actual exponent = %d)\n", exponent, (int)exponent - 127);
    printf("Mantissa: 0x%06X\n", mantissa);

    if (exponent == 0 && mantissa == 0)
        printf("Category: %szero\n", sign ? "negative " : "positive ");
    else if (exponent == 0)
        printf("Category: denormalized number\n");
    else if (exponent == 255 && mantissa == 0)
        printf("Category: %sinfinity\n", sign ? "negative " : "positive ");
    else if (exponent == 255)
        printf("Category: NaN\n");
    else
        printf("Category: normalized number\n");
}

int main(void) {
    decode_float32(6.5f);
    decode_float32(-12.375f);
    decode_float32(0.1f);
    return 0;
}
```

---

## 3. Complete Guide to Special Values

### 3.1 Signed Zero

IEEE 754 has two kinds of zero: +0 and -0.

```
+0: S=0, E=00000000, M=00000000000000000000000  (0x00000000)
-0: S=1, E=00000000, M=00000000000000000000000  (0x80000000)

Behavior in comparisons:
  +0.0 == -0.0   -> True (not distinguished in equality comparison)
  +0.0 is -0.0   -> Implementation-dependent

Cases where the sign matters:
  1.0 / (+0.0)  -> +Inf
  1.0 / (-0.0)  -> -Inf  <- Signs differ!

  copysign(1.0, +0.0) -> +1.0
  copysign(1.0, -0.0) -> -1.0

  atan2(+0.0, -1.0) -> +pi
  atan2(-0.0, -1.0) -> -pi  <- Math functions give different results

Rationale:
  - Preserves sign information during underflow
  - Retains the direction of limit values
  - Correctly handles branch cuts in complex arithmetic
```

```python
# Checking signed zero in Python
import math

pos_zero = +0.0
neg_zero = -0.0

print(pos_zero == neg_zero)         # True
print(math.copysign(1, pos_zero))   # 1.0
print(math.copysign(1, neg_zero))   # -1.0

# Detecting negative zero
def is_negative_zero(x):
    return x == 0.0 and math.copysign(1, x) < 0

print(is_negative_zero(-0.0))  # True
print(is_negative_zero(+0.0))  # False
```

### 3.2 Infinity

```
+Inf: S=0, E=11111111, M=00000000000000000000000  (0x7F800000)
-Inf: S=1, E=11111111, M=00000000000000000000000  (0xFF800000)

Operations that produce infinity:
  1.0 / 0.0     -> +Inf
  -1.0 / 0.0    -> -Inf
  1e308 * 10    -> +Inf (overflow, for binary64)
  log(0.0)      -> -Inf
  exp(1000)     -> +Inf

Arithmetic rules involving infinity:
  +------------------+----------+
  | Operation        | Result   |
  +------------------+----------+
  | Inf + Inf        | +Inf     |
  | Inf + finite     | +Inf     |
  | Inf * pos finite | +Inf     |
  | Inf * neg finite | -Inf     |
  | Inf * 0          | NaN      |
  | Inf - Inf        | NaN      |
  | Inf / Inf        | NaN      |
  | finite / Inf     | +/-0     |
  | Inf > any finite | True     |
  +------------------+----------+
```

```python
import math

inf = float('inf')

# Generating and computing with infinity
print(1.0 / 0.0)       # inf (Python raises ZeroDivisionError by default)
# Note: In Python, 1.0/0.0 raises ZeroDivisionError
# Use float('inf') to generate directly

print(inf + inf)        # inf
print(inf + 1e308)      # inf
print(inf * -1)         # -inf
print(inf * 0)          # nan
print(inf - inf)        # nan
print(inf / inf)        # nan
print(1.0 / inf)        # 0.0
print(inf > 1e308)      # True

# Checking for infinity
print(math.isinf(inf))          # True
print(math.isinf(-inf))         # True
print(math.isinf(1e308))        # False
print(math.isinf(1e308 * 10))   # True (overflow)
```

### 3.3 NaN (Not a Number)

NaN is a special value representing "undefined results" and is one of the biggest pitfalls in floating-point arithmetic.

```
Bit representation of NaN (binary32):
  Exponent = 11111111 (all bits 1)
  Mantissa != 0 (any non-zero value)

  Two types of NaN:
    Signaling NaN (sNaN): Most significant bit of mantissa = 0, rest != 0
      -> Raises an exception when used
      -> Can be used to detect uninitialized variables
    Quiet NaN (qNaN):     Most significant bit of mantissa = 1
      -> Propagates without raising exceptions
      -> The NaN returned as the result of most operations

  qNaN example: 0 11111111 10000000000000000000000 (0x7FC00000)
  sNaN example: 0 11111111 00000000000000000000001 (0x7F800001)

Operations that produce NaN:
  0.0 / 0.0    -> NaN
  Inf - Inf    -> NaN
  Inf * 0      -> NaN
  Inf / Inf    -> NaN
  sqrt(-1.0)   -> NaN (in real number arithmetic)
  NaN op any   -> NaN (NaN propagates regardless of operation type)
```

```python
import math
import numpy as np

x = float('nan')

# Fundamental property of NaN: not equal to itself
print(x == x)     # False  <- Behavior specified by IEEE 754
print(x != x)     # True
print(x > 0)      # False
print(x < 0)      # False
print(x >= 0)     # False
print(x <= 0)     # False
# -> All comparisons with NaN return False except !=

# Methods for detecting NaN
print(math.isnan(x))            # True  <- Recommended
print(x != x)                    # True  <- Traditional idiom (not recommended)
print(np.isnan(x))              # True  <- NumPy

# NaN propagation (spreads like "poison")
print(x + 1)          # nan
print(x * 0)          # nan
print(x ** 0)         # 1.0  <- Exception! Specified by IEEE 754
print(0 * float('inf'))  # nan
print(max(x, 5))      # nan (Python standard)
print(min(x, 5))      # nan

# Aggregating lists containing NaN
values = [1.0, 2.0, float('nan'), 4.0]
print(sum(values))     # nan (result is NaN if even one NaN exists)
print(max(values))     # nan
print(min(values))     # nan

# NaN-safe aggregation (NumPy)
arr = np.array(values)
print(np.nansum(arr))    # 7.0 (sum ignoring NaN)
print(np.nanmean(arr))   # 2.333... (mean excluding NaN)
print(np.nanmax(arr))    # 4.0
```

### 3.4 NaN Behavior Differences Across Languages

```
NaN handling in various languages:

+--------------+-----------------+--------------------------------+
| Language     | NaN generation  | Notes                          |
+--------------+-----------------+--------------------------------+
| Python       | float('nan')    | Use math.isnan() to check      |
| C/C++        | NAN, nan()      | isnan() macro (<math.h>)       |
| Java         | Double.NaN      | Use Double.isNaN() to check    |
| JavaScript   | NaN             | Use Number.isNaN()             |
|              |                 | typeof NaN === 'number' !      |
|              |                 | isNaN("hello") -> true (trap!) |
| Rust         | f64::NAN        | f64::is_nan(), safe due to     |
|              |                 | non-comparable                 |
| Go           | math.NaN()      | Use math.IsNaN() to check      |
| SQL          | NULL != NaN     | Use IS NULL (different concept |
|              |                 | from NaN)                      |
+--------------+-----------------+--------------------------------+

JavaScript NaN traps:
  typeof NaN === 'number'    // true! It's a Number type yet "Not a Number"
  NaN === NaN                // false
  NaN !== NaN                // true
  isNaN("hello")             // true <- global isNaN performs type coercion
  Number.isNaN("hello")      // false <- correct check
  [NaN].includes(NaN)        // true <- includes uses SameValueZero
  [NaN].indexOf(NaN)         // -1   <- indexOf uses ===
  new Set([NaN, NaN]).size   // 1    <- Set uses SameValueZero
```

### 3.5 Denormalized Numbers (Subnormal Numbers)

```
The problem near the minimum normalized number:

  Normalized:        (-1)^S x 1.M x 2^(E-bias)
  Minimum normalized: 1.000...0 x 2^(-126) ~ 1.18 x 10^(-38) [binary32]

  If denormalized numbers did not exist:
  +----------------------------------------------+
  | ... --- min normalized -- large gap -- 0      |
  |                           ^                   |
  |             No representable numbers in gap    |
  |             a != b yet a - b = 0 is possible   |
  +----------------------------------------------+

  With denormalized numbers:
  +----------------------------------------------+
  | ... --- min normalized -- denorms -- 0        |
  |                           ^^^^^^              |
  |             Precision degrades gradually       |
  |             toward zero                        |
  |             a - b = 0  <=>  a = b guaranteed   |
  +----------------------------------------------+

Formula for denormalized numbers (binary32):
  Value = (-1)^S x 0.M x 2^(-126)

  The implicit leading bit becomes 0 instead of 1
  The exponent is fixed at -126 (not -127)

  Smallest positive denormalized number:
    0.000...001 x 2^(-126) = 2^(-23) x 2^(-126) = 2^(-149)
    ~ 1.4 x 10^(-45)

  For binary64:
    Minimum normalized:    2^(-1022) ~ 2.22 x 10^(-308)
    Minimum denormalized:  2^(-1074) ~ 4.94 x 10^(-324)
```

Performance note on denormalized numbers: On many CPUs, arithmetic with denormalized numbers is significantly slower than with normalized numbers (10x to 100x). This is because denormalized numbers are processed by microcode rather than the hardware fast path. GPUs and game engines may enable "Flush to Zero (FTZ)" mode, rounding denormalized numbers to zero to maintain performance.

---

## 4. The Nature of Precision Problems

### 4.1 Why 0.1 + 0.2 !== 0.3

This is the most famous floating-point pitfall, caused by the fact that "finite decimal fractions in base 10 become infinite repeating fractions in base 2."

```
Correspondence between decimal and binary repeating fractions:

  Fractions exactly representable in decimal: denominators with only factors of 2 and 5
    0.5 = 1/2       -> finite decimal
    0.25 = 1/4      -> finite decimal
    0.125 = 1/8     -> finite decimal
    0.1 = 1/10      -> finite decimal
    0.2 = 1/5       -> finite decimal
    1/3              -> 0.333... (infinite repeating)

  Fractions exactly representable in binary: denominators that are powers of 2 only
    1/2 = 0.1       -> finite
    1/4 = 0.01      -> finite
    1/8 = 0.001     -> finite
    1/10 = 0.0(0011) -> infinite repeating!
    1/5 = 0.0(0110)  -> infinite repeating!
    1/3 = 0.(01)     -> infinite repeating

  Conclusion: 0.1, 0.2, and 0.3 are all infinite repeating fractions in binary
  -> In IEEE 754, they are rounded to a finite number of bits
  -> Rounding error occurs
```

```
Exact stored values of 0.1, 0.2, and 0.3 in binary64:

  Value stored for 0.1:
    0.1000000000000000055511151231257827021181583404541015625
    Error: +5.55 x 10^(-18)

  Value stored for 0.2:
    0.200000000000000011102230246251565404236316680908203125
    Error: +1.11 x 10^(-17)

  Value stored for the result of 0.1 + 0.2:
    0.3000000000000000444089209850062616169452667236328125
    (additional rounding occurs during addition)

  Value stored for 0.3:
    0.299999999999999988897769753748434595763683319091796875
    Error: -1.11 x 10^(-17)

  Therefore:
    (0.1 + 0.2) - 0.3
    = 0.300000000000000044... - 0.29999999999999998...
    = 5.55 x 10^(-17)
    != 0

  -> 0.1 + 0.2 > 0.3!
```

```python
# Detailed examination of the 0.1 + 0.2 problem

from decimal import Decimal

# Check the exact values of floats using Decimal
print(f"Stored value of 0.1: {Decimal(0.1)}")
print(f"Stored value of 0.2: {Decimal(0.2)}")
print(f"Stored value of 0.3: {Decimal(0.3)}")
print(f"Value of 0.1+0.2:    {Decimal(0.1) + Decimal(0.2)}")
print()
print(f"Difference: {(Decimal(0.1) + Decimal(0.2)) - Decimal(0.3)}")
print(f"   = {float(0.1) + float(0.2) - float(0.3)}")

# Output:
# Stored value of 0.1: 0.1000000000000000055511151231257827021181583404541015625
# Stored value of 0.2: 0.200000000000000011102230246251565404236316680908203125
# Stored value of 0.3: 0.299999999999999988897769753748434595763683319091796875
# Value of 0.1+0.2:    0.3000000000000000166533453693773481063544750213623046875
# Difference: 1.77635683940025046E-17

# Results in major languages
# Python:     0.1 + 0.2 == 0.30000000000000004
# JavaScript: 0.1 + 0.2 === 0.30000000000000004
# C/C++:      0.1 + 0.2 == 0.30000000000000004
# Java:       0.1 + 0.2 == 0.30000000000000004
# Ruby:       0.1 + 0.2 == 0.30000000000000004
# -> Same result in all languages (because they all follow IEEE 754)
```

### 4.2 Rounding Modes

IEEE 754 specifies 5 rounding modes.

```
Five rounding modes:

  1. Round to Nearest, Ties to Even (default)
     - Round to the nearest representable value
     - When exactly halfway, round to the value with an even least significant bit
     - Also called "Banker's Rounding"
     - Minimizes statistical bias

  2. Round to Nearest, Ties Away from Zero
     - When exactly halfway, round away from zero
     - Equivalent to the "round half up" taught in school
     - Added in IEEE 754-2008

  3. Round toward +Infinity (Ceiling)
     - Always round toward positive direction

  4. Round toward -Infinity (Floor)
     - Always round toward negative direction

  5. Round toward Zero (Truncation)
     - Always round toward zero (truncate)

Examples of Round to Nearest Even (explained in decimal):
  +------+------------------+--------------+
  | Value| Round half up    | Round to even|
  +------+------------------+--------------+
  | 0.5  | 1 (round up)     | 0 (even)     |
  | 1.5  | 2 (round up)     | 2 (even)     |
  | 2.5  | 3 (round up)     | 2 (even)     |
  | 3.5  | 4 (round up)     | 4 (even)     |
  | 4.5  | 5 (round up)     | 4 (even)     |
  | 0.4  | 0                | 0            |
  | 0.6  | 1                | 1            |
  +------+------------------+--------------+

  Round half up: 0+2+3+4+5 = 14 (biased: .5 always rounds up)
  Round to even: 0+2+2+4+4 = 12 (no bias: half round up, half round down)

  -> Prevents statistical bias in large numbers of rounding operations
  -> Important in financial calculations and simulations
```

```python
# Checking rounding modes in Python
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

# Python's built-in round() uses round-to-even
print(round(0.5))   # 0  <- Round to even
print(round(1.5))   # 2  <- Round to even
print(round(2.5))   # 2  <- Round to even
print(round(3.5))   # 4  <- Round to even

# Explicitly specifying rounding mode with Decimal
d = Decimal('2.5')
print(d.quantize(Decimal('1'), rounding=ROUND_HALF_UP))    # 3 (round half up)
print(d.quantize(Decimal('1'), rounding=ROUND_HALF_EVEN))  # 2 (round to even)
```

### 4.3 Catastrophic Cancellation

Catastrophic cancellation is a phenomenon where the subtraction of two nearly equal values causes a dramatic loss of significant digits. It is one of the most serious precision problems in numerical computing.

```
Principle of catastrophic cancellation:

  Consider a decimal floating-point with 7 significant digits:

  a = 1.234567 x 10^5  (= 123456.7)
  b = 1.234566 x 10^5  (= 123456.6)

  a - b = 0.000001 x 10^5 = 0.1000000 x 10^0

  The original values had 7 digits of precision,
  but the subtraction result has only 1 effective digit!
  The remaining 6 digits are "fabricated" zeros.

  Binary example (binary64):
  a = 1.000000000000001 x 2^50
  b = 1.000000000000000 x 2^50
  a - b = 0.000000000000001 x 2^50 = 1.0 x 2^(-2)

  -> Of the 52-bit mantissa, only the lowest few bits are significant
  -> Most of the precision is lost
```

```python
# Classic example of catastrophic cancellation: quadratic formula

import math

def quadratic_naive(a, b, c):
    """Naive quadratic formula (prone to catastrophic cancellation)"""
    discriminant = b*b - 4*a*c
    sqrt_d = math.sqrt(discriminant)
    x1 = (-b + sqrt_d) / (2 * a)
    x2 = (-b - sqrt_d) / (2 * a)
    return x1, x2

def quadratic_stable(a, b, c):
    """Stable version that avoids catastrophic cancellation"""
    discriminant = b*b - 4*a*c
    sqrt_d = math.sqrt(discriminant)

    # Compute the root that avoids cancellation first based on the sign of b
    if b >= 0:
        q = -0.5 * (b + sqrt_d)
    else:
        q = -0.5 * (b - sqrt_d)

    x1 = q / a
    x2 = c / q  # Using Vieta's formula: x1 * x2 = c/a
    return x1, x2

# Test: a=1, b=10^8, c=1 -> true roots are x ~ -10^(-8), x ~ -10^8
a, b, c = 1, 1e8, 1

naive = quadratic_naive(a, b, c)
stable = quadratic_stable(a, b, c)

print(f"Naive solution:  x1 = {naive[0]:.15e}, x2 = {naive[1]:.15e}")
print(f"Stable solution: x1 = {stable[0]:.15e}, x2 = {stable[1]:.15e}")
print(f"Theoretical:     x1 ~ -1e-08,             x2 = -1e+08")

# Naive:  x1 = -7.450580596923828e-09 <- large error
# Stable: x1 = -1.000000000000000e-08 <- accurate
```

### 4.4 Loss of Significance by Addition (Absorption)

```
Principle of absorption:

  When adding a large value and a small value, the information
  in the small value is lost.

  Example (decimal floating-point with 7 significant digits):
  a = 1.234567 x 10^10
  b = 1.234567 x 10^0

  Aligning exponents for addition:
  a = 1.234567  x 10^10
  b = 0.0000000001234567 x 10^10
      ^
      Parts beyond 7 digits cannot be stored -> truncated
  b' = 0.0000000 x 10^10

  a + b' = 1.234567 x 10^10 = a (b's information is completely lost)

  Concrete example in binary64:
  1e16 + 1.0 - 1e16 = ?
    1e16 = 10000000000000000.0
    1e16 + 1 -> 10000000000000000.0 (1.0 is lost)
    result - 1e16 -> 0.0

  However:
  -1e16 + 1e16 + 1.0 = ?
    -1e16 + 1e16 -> 0.0
    0.0 + 1.0 -> 1.0

  -> The order of operations changes the result!
  -> Floating-point addition does not satisfy the associative law
```

```python
# Confirming absorption

# Difference in results due to operation order
a = 1e16
b = 1.0

print(f"(a + b) - a = {(a + b) - a}")   # 0.0 (b's information is lost)
print(f"(a - a) + b = {(a - a) + b}")   # 1.0 (correct result)

# More serious example: adding many small values to a large value
big = 1e15
result_forward = big
for i in range(1000000):
    result_forward += 1.0

result_reverse = 0.0
for i in range(1000000):
    result_reverse += 1.0
result_reverse += big

print(f"Forward sum:  {result_forward}")      # Low precision
print(f"Reverse sum:  {result_reverse}")      # Somewhat more accurate
print(f"Theoretical:  {big + 1000000.0}")
```

### 4.5 Accumulation of Rounding Errors

```python
# Rounding error accumulation: adding 0.1 one thousand times

# Naive summation
total_naive = 0.0
for i in range(1000):
    total_naive += 0.1
print(f"Naive sum:      {total_naive}")         # 99.99999999999986
print(f"Error:          {total_naive - 100.0}")  # -1.4e-13

# math.fsum (uses extended precision internally)
import math
total_fsum = math.fsum([0.1] * 1000)
print(f"math.fsum:      {total_fsum}")           # 100.00000000000007
print(f"Error:          {total_fsum - 100.0}")    # 7.1e-14

# Kahan compensated summation algorithm
def kahan_sum(values):
    """Kahan's compensated summation: tracks rounding error via a correction term"""
    total = 0.0
    compensation = 0.0  # Correction term to track accumulated error
    for value in values:
        y = value - compensation      # Apply correction
        t = total + y                 # Add (rounding error occurs here)
        compensation = (t - total) - y  # Capture rounding error
        total = t
    return total

total_kahan = kahan_sum([0.1] * 1000)
print(f"Kahan sum:      {total_kahan}")          # 100.00000000000007
print(f"Error:          {total_kahan - 100.0}")   # very small

# Neumaier's improved version (more robust than Kahan)
def neumaier_sum(values):
    """Neumaier's compensated summation: correct even when |total| < |value|"""
    total = 0.0
    compensation = 0.0
    for value in values:
        t = total + value
        if abs(total) >= abs(value):
            compensation += (total - t) + value
        else:
            compensation += (value - t) + total
        total = t
    return total + compensation

total_neumaier = neumaier_sum([0.1] * 1000)
print(f"Neumaier sum:   {total_neumaier}")
```

```
Operating principle of Kahan compensated summation (illustrated):

  At each step, the rounding error that would be lost is
  accumulated in the compensation variable, and applied
  as a correction in the next step.

  Step n:
    y = value[n] - compensation   <- Correct for previous error
    t = total + y                 <- Rounding occurs (error e is produced)
    compensation = (t - total) - y  <- Capture error e
                 = ((total + y + e) - total) - y
                 = y + e - y
                 = e                <- The rounding error itself
    total = t

  Normal summation:  error = O(n x eps)     <- Error grows proportionally to n
  Kahan summation:   error = O(eps)         <- Independent of n!

  Where eps = machine epsilon (binary64: approximately 2.22 x 10^(-16))
```

---

## 5. Density Distribution and ULP of Floating-Point Numbers

### 5.1 Non-Uniform Distribution on the Number Line

Floating-point numbers are not uniformly distributed on the number line. Representable values are densely packed near zero and the spacing between adjacent values widens toward larger magnitudes.

```
Density of floating-point numbers on the number line:

  0                                                     +Inf
  |========|||||----+----+----+----+--------+--------+---->
  ^                                                     ^
  Denormals        Normalized                           Inf
  (most dense)  (spacing doubles with each exponent increase)

  The number of representable values between powers of 2 is always constant:
  [1, 2): 2^23 values (binary32) = 8,388,608 values
  [2, 4): 2^23 values -> spacing is 2x that of [1,2)
  [4, 8): 2^23 values -> spacing is 4x that of [1,2)
  ...
  [2^n, 2^(n+1)): 2^23 values -> spacing is 2^(n-0) x 2^(-23)
```

### 5.2 ULP (Unit in the Last Place)

ULP is "the weight of the least significant bit" and represents the minimum gap between a floating-point number and its adjacent value.

```
ULP variation in binary32:

  +------------------+------------------+------------------+
  | Value range      | ULP (gap to next)| Meaning          |
  +------------------+------------------+------------------+
  | [0.5, 1.0)       | 5.96 x 10^(-8)  | ~0.00006% prec.  |
  | [1.0, 2.0)       | 1.19 x 10^(-7)  |                  |
  | [1000, 2000)     | 6.10 x 10^(-5)  | ~0.006% prec.    |
  | [10^6, 2x10^6)   | 6.25 x 10^(-2)  | 2nd decimal limit|
  | [2^23, 2^24)     | 1.0              | Integer limit!   |
  | [2^24, 2^25)     | 2.0              | Odd nums lost    |
  | [10^30, ...)     | ~10^23           | Almost no prec.  |
  +------------------+------------------+------------------+

  Important thresholds:
    binary32: loses integer precision above 2^24 = 16,777,216
    binary64: loses integer precision above 2^53 = 9,007,199,254,740,992

  -> JavaScript's Number.MAX_SAFE_INTEGER = 2^53 - 1 = 9007199254740991
```

```python
# Checking ULP and integer precision limits

import numpy as np

# Integer precision limit of binary32
f32 = np.float32

print(f"2^23     = {f32(2**23)}")              # 8388608.0
print(f"2^23 + 1 = {f32(2**23 + 1)}")          # 8388609.0 (exact)
print(f"2^24     = {f32(2**24)}")              # 16777216.0
print(f"2^24 + 1 = {f32(2**24 + 1)}")          # 16777216.0 <- same value!
print(f"2^24 + 2 = {f32(2**24 + 2)}")          # 16777218.0
print(f"2^24 + 3 = {f32(2**24 + 3)}")          # 16777220.0 <- jumped by +4!

print()

# Integer precision limit of binary64
print(f"2^53     = {float(2**53)}")             # 9007199254740992.0
print(f"2^53 + 1 = {float(2**53 + 1)}")         # 9007199254740992.0 <- same!

# Impact on JavaScript
# JSON.parse('{"id": 9007199254740993}')
# -> {"id": 9007199254740992}  <- The ID changes!
# -> This is why Twitter returns snowflake IDs as strings

# Computing ULP
def ulp(x):
    """Compute the ULP for a given value"""
    return np.spacing(x)

for val in [0.5, 1.0, 1000.0, 1e6, 1e15]:
    print(f"ULP({val:>10}) = {ulp(val):.6e}")
```

---

## 6. Pitfalls and Anti-patterns in Numerical Computing

### 6.1 Anti-pattern 1: Equality Comparison of Floating-Point Numbers

```python
# --- Anti-pattern: using == for floating-point comparison ---

# Dangerous code
total = 0.0
for _ in range(10):
    total += 0.1

if total == 1.0:       # <- May never be True!
    print("Total is 1.0")
else:
    print(f"Total is {total}")  # Total is 0.9999999999999999

# Danger in loop termination conditions
x = 0.0
while x != 1.0:       # <- Risk of infinite loop!
    x += 0.1
    if x > 2.0:       # Without a safety valve, this truly loops forever
        break

# --- Correct patterns ---

import math

# Pattern 1: math.isclose() (Python 3.5+)
print(math.isclose(0.1 + 0.2, 0.3))  # True
# Default: rel_tol=1e-9, abs_tol=0.0

# Pattern 2: Comparison using relative error
def nearly_equal(a, b, rel_tol=1e-9, abs_tol=1e-12):
    """Comparison considering both relative and absolute error"""
    if a == b:  # Correctly handles Inf == Inf, +0 == -0
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    diff = abs(a - b)
    return diff <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# Pattern 3: Use < or > in loops
x = 0.0
while x < 1.0:  # Use < instead of !=
    x += 0.1
```

```c
/* Floating-point comparison in C */
#include <math.h>
#include <float.h>
#include <stdbool.h>

/* Anti-pattern */
bool bad_compare(double a, double b) {
    return a == b;  /* Unreliable for floating-point */
}

/* Correct pattern: combination of relative and absolute error */
bool nearly_equal(double a, double b, double rel_tol, double abs_tol) {
    if (a == b) return true;  /* Handle Inf, 0 */
    if (isnan(a) || isnan(b)) return false;

    double diff = fabs(a - b);
    double larger = fmax(fabs(a), fabs(b));

    return diff <= fmax(rel_tol * larger, abs_tol);
}

/* Usage example */
int main(void) {
    double x = 0.1 + 0.2;
    double y = 0.3;

    /* NG */
    if (x == y) { /* Never reached */ }

    /* OK */
    if (nearly_equal(x, y, 1e-9, 1e-12)) {
        /* Correctly reached */
    }

    return 0;
}
```

### 6.2 Anti-pattern 2: Using Floating-Point for Financial Calculations

```python
# --- Anti-pattern: using float for financial calculations ---

# Dangerous code
price = 19.99
tax_rate = 0.08
tax = price * tax_rate          # 1.5992000000000002
total = price + tax             # 21.5892
print(f"Tax included: ${total:.2f}")    # $21.59 (appears correct on display, but...)

# Error accumulates with large transaction volumes
daily_amounts = [0.01] * 1000000  # 1 million 1-cent transactions
total = sum(daily_amounts)
print(f"Total: ${total:.2f}")  # May not equal $10000.00

# --- Correct pattern 1: Decimal type ---
from decimal import Decimal, ROUND_HALF_UP, getcontext

# Set precision
getcontext().prec = 28

price = Decimal('19.99')       # Generate from string (don't go through float!)
tax_rate = Decimal('0.08')
tax = (price * tax_rate).quantize(
    Decimal('0.01'),
    rounding=ROUND_HALF_UP
)
total = price + tax
print(f"Tax included: ${total}")       # $21.59 (exact)

# Exact even with large transaction volumes
daily_amounts = [Decimal('0.01')] * 1000000
total = sum(daily_amounts)
print(f"Total: ${total}")       # $10000.00 (exact)

# --- Correct pattern 2: Integer arithmetic (in cents) ---
price_cents = 1999             # $19.99 = 1999 cents
tax_rate_bps = 800             # 8% = 800 basis points
tax_cents = (price_cents * tax_rate_bps + 5000) // 10000  # Rounding
total_cents = price_cents + tax_cents

print(f"Tax included: ${total_cents / 100:.2f}")  # $21.59 (exact)

# Note: Decimal('0.1') and Decimal(0.1) are different!
print(Decimal('0.1'))   # 0.1 (exact)
print(Decimal(0.1))     # 0.1000000000000000055511151231257827... (error via float)
```

### 6.3 Other Common Pitfalls

```python
# Pitfall 1: Order dependence of operations (failure of associativity)
a, b, c = 1e20, -1e20, 1.0
print(f"(a + b) + c = {(a + b) + c}")   # 1.0
print(f"a + (b + c) = {a + (b + c)}")   # 0.0  <- Different result!

# Pitfall 2: Failure of distributive law
a, b, c = 1e15, 1.0, -1e15
print(f"a x (b + c) = {a * (b + c)}")     # Expected: complex
# In general, a*(b+c) != a*b + a*c

# Pitfall 3: Non-transitivity of comparison
# a < b and b < c does not always imply a < c (when NaN is involved)
a, b, c = 1.0, float('nan'), 2.0
print(f"a < b: {a < b}")  # False
print(f"b < c: {b < c}")  # False
print(f"a < c: {a < c}")  # True  <- NaN breaks comparisons

# Pitfall 4: Sort instability
import random
values = [1.0, float('nan'), 2.0, float('nan'), 0.5]
# sorted(values) -> NaN position is indeterminate, sort may break

# Pitfall 5: Hash consistency
# In Python, hash(0) == hash(0.0) == hash(Decimal('0'))
# However, hash(float('nan')) is consistent per call, but
# since NaN == NaN is False, using NaN as a dict key is problematic

# Pitfall 6: Type conversion trap (JavaScript)
# JSON.parse('{"value": 9007199254740993}') -> 9007199254740992
# -> Large integer IDs can change during JSON parsing
```

---

## 7. Precision Countermeasures in Practice

### 7.1 Recommended Approaches by Use Case

```
+--------------------+---------------------------------------------+
| Use case           | Recommended approach                        |
+--------------------+---------------------------------------------+
| Finance/Accounting | Decimal type or integers (in cents)          |
| Scientific comp.   | double + error analysis + compensated sum    |
| Games/Graphics     | float32 is sufficient (performance priority) |
| ML/Inference       | float16 / bfloat16 / INT8 quantization      |
| Cryptography       | Never use floating-point (integers/fixed     |
|                    | point only)                                 |
| DB monetary values | DECIMAL/NUMERIC type (arbitrary precision)   |
| Web API IDs        | Strings (integers > 2^53 break in JSON)     |
| Coordinates/Geo    | double (float32 has ~1m error on Earth)      |
+--------------------+---------------------------------------------+
```

### 7.2 Epsilon Comparison Implementation Patterns

```python
import math
from typing import Optional

def robust_float_equal(
    a: float,
    b: float,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12
) -> bool:
    """
    Robust floating-point comparison function.

    Correctly handles edge cases:
    - NaN: Returns False even for NaN vs NaN (IEEE 754 compliant)
    - Inf: True for same-sign Inf
    - -0 and +0: True (IEEE 754 compliant)
    - Very small values: Determined by abs_tol
    - Normal values: Determined by rel_tol

    Parameters:
        a, b: Floating-point numbers to compare
        rel_tol: Relative tolerance (default 1e-9)
        abs_tol: Absolute tolerance (default 1e-12)

    Returns:
        Whether a and b are sufficiently close
    """
    # NaN handling (NaN is not equal to anything)
    if math.isnan(a) or math.isnan(b):
        return False

    # Exact match (including Inf == Inf, +0 == -0)
    if a == b:
        return True

    # Compute difference
    diff = abs(a - b)

    # Infinity case (Inf with different signs)
    if math.isinf(a) or math.isinf(b):
        return False

    # Determine by relative error or absolute error
    larger = max(abs(a), abs(b))
    return diff <= max(rel_tol * larger, abs_tol)


# Tests
assert robust_float_equal(0.1 + 0.2, 0.3)          # True
assert not robust_float_equal(1.0, 2.0)             # False
assert robust_float_equal(float('inf'), float('inf'))  # True
assert not robust_float_equal(float('inf'), float('-inf'))  # False
assert not robust_float_equal(float('nan'), float('nan'))   # False
assert robust_float_equal(0.0, -0.0)                # True
assert robust_float_equal(1e-15, 1.1e-15, rel_tol=0.1)  # True
```

### 7.3 Error Management in Scientific Computing

```python
# Predicting errors using the condition number

import numpy as np

# Ill-conditioned system of linear equations
A_bad = np.array([
    [1.0, 1.0],
    [1.0, 1.0001]
])
b = np.array([2.0, 2.0001])

cond = np.linalg.cond(A_bad)
print(f"Condition number: {cond:.0f}")  # Approximately 40000

# Large condition number -> small input changes cause large output variations
x = np.linalg.solve(A_bad, b)
print(f"Solution: {x}")  # [1.0, 1.0]

# Slightly perturb b
b_perturbed = b + np.array([0.0001, 0.0])
x_perturbed = np.linalg.solve(A_bad, b_perturbed)
print(f"Perturbed solution: {x_perturbed}")  # May differ significantly

# Error upper bound: ||dx||/||x|| <= cond(A) x ||db||/||b||
# -> If condition number is 10^4, a 10^(-12) input error can become
#    a 10^(-8) result error
```

---

## 8. AI/GPU and Low-Precision Floating-Point

### 8.1 Why AI Works with Low Precision

In neural network training and inference, high numerical precision is not always necessary. The reasons are as follows.

```
Why AI works with low precision:

  1. Noise tolerance
     - SGD (Stochastic Gradient Descent) inherently contains noise
     - Mini-batch sampling noise > quantization noise
     - Moderate noise can even have a regularization effect

  2. Gradient direction matters; magnitude is secondary
     - Can be adjusted by learning rate
     - Converges if direction is approximately correct

  3. Memory bandwidth is the bottleneck
     - GPU compute far exceeds memory transfer speed
     - Smaller data -> faster memory transfer -> faster overall
     - FP16: half the memory of FP32 -> 2x batch size or 2x speed

  4. Dedicated hardware exists
     - Tensor Cores: execute FP16/BF16/FP8 matrix products at ultra-high speed
     - A100: 312 TFLOPS at FP16, 19.5 TFLOPS at FP32 (16x difference)
     - H100: 3958 TFLOPS at FP8
```

### 8.2 Detailed Comparison of Each Format

```
BF16 (bfloat16) vs FP16 (IEEE half):

  FP16: +-+-----+----------+
        |S|E(5b)| M(10bit) |  Range: +/-65504, Precision: ~3.3 digits
        +-+-----+----------+

  BF16: +-+--------+-------+
        |S| E(8b)  |M(7bit)|  Range: +/-3.4x10^38, Precision: ~2.4 digits
        +-+--------+-------+

  FP32: +-+--------+-----------------------+
        |S| E(8b)  |      M(23bit)         |  Reference
        +-+--------+-----------------------+

  BF16 design philosophy:
  - Same 8-bit exponent as FP32 -> same value range
  - Mantissa reduced from 23 to 7 bits -> lower precision
  - Simple conversion to/from FP32 (just truncate the upper 16 bits)
  - Overflow/underflow occurs at the same thresholds as FP32
  - -> More stable training than FP16 (wider range)

  +--------+---------+----------------+----------------------+
  | Format | Range   | Precision      | Primary use          |
  +--------+---------+----------------+----------------------+
  | FP32   | 10^38   | 7.2 digits     | Reference, master    |
  |        |         |                | weights              |
  | TF32   | 10^38   | 3.3 digits     | A100 Tensor Core     |
  | BF16   | 10^38   | 2.4 digits     | Training             |
  |        |         |                | (Google/Meta)        |
  | FP16   | 65504   | 3.3 digits     | Inference, mobile    |
  | FP8    | 448     | 1.2 digits     | Forward pass         |
  | E4M3   |         |                | (H100 onwards)       |
  | FP8    | 57344   | 0.9 digits     | Backward pass        |
  | E5M2   |         |                | (H100 onwards)       |
  | INT8   |-128~127 | Integer        | Quantized inference  |
  | INT4   | -8~7    | Integer        | Extreme quantization |
  |        |         |                | (LLM)               |
  +--------+---------+----------------+----------------------+
```

### 8.3 Mixed Precision Training

```
Mixed precision training flow:

  +-----------------------------------------------------+
  |                     Training Loop                     |
  |                                                       |
  |  +----------+    FP32->FP16     +--------------+     |
  |  |Master wts|------------------>|FP16 wt copy  |     |
  |  | (FP32)   |                   +------+-------+     |
  |  +-----+----+                         |              |
  |        ^                              v              |
  |   Update in FP32             Forward pass in FP16    |
  |        ^                              |              |
  |        |                              v              |
  |   +----+----+              +--------------+          |
  |   |FP32 grad|<------------|  FP16 loss   |          |
  |   |(convert)|   FP16->FP32 | x loss_scale|          |
  |   +---------+              +------+-------+          |
  |                                    |                 |
  |                           Backward pass in FP16      |
  |                                    |                 |
  |                             +------+-------+         |
  |                             | FP16 grads   |         |
  |                             | / loss_scale |         |
  |                             +--------------+         |
  +-----------------------------------------------------+

  Why Loss Scaling is needed:
  - FP16 minimum normalized: approximately 6 x 10^(-8)
  - Gradients become very small in later training stages (< 10^(-7))
  - In FP16, gradients underflow to zero
  - -> Scale up the loss -> gradients are also scaled up -> within FP16 range
  - -> Scale down before parameter update to restore original values
  - Dynamic Loss Scaling: automatically adjusts scale based on overflow frequency
```

```python
# Mixed precision training example in PyTorch
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()  # Automatically manages Loss Scaling

for data, target in dataloader:
    optimizer.zero_grad()

    # autocast: automatically selects optimal precision for each operation
    with autocast():
        output = model(data)        # Forward pass in FP16
        loss = loss_fn(output, target)

    # Loss Scaling + backward pass in FP16
    scaler.scale(loss).backward()

    # Parameter update in FP32
    scaler.step(optimizer)
    scaler.update()
```

---

## 9. Language and Platform-Specific Considerations

### 9.1 Floating-Point Support by Language

```
+--------------+----------------+-----------------------------------+
| Language     | Default type   | Notes                             |
+--------------+----------------+-----------------------------------+
| C/C++        | double (64bit) | float, long double available.     |
|              |                | long double is 80/128bit (env     |
|              |                | dependent). #include <cfloat>     |
|              |                | for precision constants.          |
| Java         | double (64bit) | strictfp for strict IEEE 754.     |
|              |                | BigDecimal for arbitrary prec.    |
| Python       | float (64bit)  | decimal.Decimal for arbitrary     |
|              |                | prec. fractions.Fraction for      |
|              |                | rationals.                        |
| JavaScript   | Number (64bit) | The only numeric type (BigInt     |
|              |                | added in ES2020). TypedArray      |
|              |                | for float32.                      |
| Rust         | f64 (64bit)    | f32 also available. NaN           |
|              |                | comparisons cause compile error   |
|              |                | (safe).                           |
| Go           | float64 (64bit)| math/big for arbitrary precision. |
| C#           | double (64bit) | decimal (128bit, base-10) avail.  |
| SQL          | FLOAT/DOUBLE   | DECIMAL/NUMERIC is base-10 fixed  |
|              |                | point.                            |
+--------------+----------------+-----------------------------------+
```

### 9.2 Compiler Optimizations and Floating-Point

```
Compiler optimization levels and floating-point precision:

  GCC/Clang options:
  -O0: No optimization -> Strictly IEEE 754 compliant
  -O2: Standard optimizations -> Usually preserves precision
  -O3: Aggressive optimizations -> Some transformations may apply
  -Ofast: Most aggressive -> Includes -ffast-math (dangerous!)

  Effects of -ffast-math (GCC/Clang):
  +----------------------------+-------------------------------+
  | Permitted transformations  | Consequence                   |
  +----------------------------+-------------------------------+
  | Assumes no NaN or Inf      | isnan() always returns false  |
  | Assumes associativity      | (a+b)+c transformed to a+(b+c)|
  | Assumes distributivity     | a*b+a*c transformed to a*(b+c)|
  | Ignores signed zero        | -0.0 treated as +0.0         |
  | Pre-computes reciprocals   | a/b/c transformed to a/(b*c) |
  +----------------------------+-------------------------------+

  -> -ffast-math can destroy numerical correctness
  -> Never use for scientific or financial calculations
  -> May be acceptable for games/graphics

  Safe optimization flags:
    gcc -O2 -fno-fast-math  # -O2 is safe
    gcc -O2 -ffp-contract=off  # Disable FMA (fused multiply-add)
```

---

## 10. Floating-Point Debugging Techniques

### 10.1 Visualizing Bit Representations

When debugging floating-point issues, the most important step is to visualize "what bit pattern the value is stored as."

```python
import struct
import math

def visualize_float64(value):
    """Visualize the bit representation of a float64 in detail"""
    # float -> bytes -> int
    raw_bytes = struct.pack('>d', value)
    bits = int.from_bytes(raw_bytes, 'big')

    # Field extraction
    sign = (bits >> 63) & 1
    exponent = (bits >> 52) & 0x7FF
    mantissa = bits & 0xFFFFFFFFFFFFF

    # Bit string
    bit_str = f"{bits:064b}"
    formatted = f"{bit_str[0]} {bit_str[1:12]} {bit_str[12:]}"

    print(f"Value:       {value}")
    print(f"Hexadecimal: {raw_bytes.hex()}")
    print(f"Bit string:  {formatted}")
    print(f"Sign (S):    {sign} ({'negative' if sign else 'positive'})")
    print(f"Exponent(E): {exponent} (unbiased: {exponent - 1023})")
    print(f"Mantissa(M): {mantissa:052b}")

    # Classification
    if exponent == 0 and mantissa == 0:
        print(f"Category:    {'Negative' if sign else 'Positive'} zero")
    elif exponent == 0:
        actual_exp = 1 - 1023
        value_calc = (-1)**sign * (mantissa / 2**52) * 2**actual_exp
        print(f"Category:    Denormalized number (effective exponent: {actual_exp})")
    elif exponent == 2047 and mantissa == 0:
        print(f"Category:    {'Negative' if sign else 'Positive'} infinity")
    elif exponent == 2047:
        snan = (mantissa >> 51) & 1 == 0
        print(f"Category:    {'Signaling' if snan else 'Quiet'} NaN")
    else:
        actual_exp = exponent - 1023
        print(f"Category:    Normalized number (effective exponent: {actual_exp})")
        print(f"Exact value: 1.{mantissa:052b} x 2^{actual_exp}")

    # Adjacent values
    if not (math.isnan(value) or math.isinf(value)):
        next_val = struct.unpack('>d', (bits + 1).to_bytes(8, 'big'))[0]
        prev_val = struct.unpack('>d', (bits - 1).to_bytes(8, 'big'))[0]
        print(f"Next value:  {next_val} (gap: {next_val - value})")
        print(f"Prev value:  {prev_val} (gap: {value - prev_val})")

    print()

# Debug examples
visualize_float64(0.1)
visualize_float64(0.2)
visualize_float64(0.1 + 0.2)
visualize_float64(0.3)
```

### 10.2 Error Tracking and Interval Arithmetic

To evaluate the reliability of numerical computation, there are methods that track not only the result but also an "error upper bound." Interval arithmetic tracks the interval containing the true value at each computation step.

```python
class Interval:
    """Simple interval arithmetic class (for debugging)"""

    def __init__(self, lo, hi=None):
        if hi is None:
            # Generate interval from a floating-point number (accounting for rounding error)
            import sys
            eps = sys.float_info.epsilon
            if lo == 0.0:
                self.lo, self.hi = -5e-324, 5e-324
            else:
                ulp = abs(lo) * eps
                self.lo = lo - ulp
                self.hi = lo + ulp
        else:
            self.lo = lo
            self.hi = hi

    def __add__(self, other):
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other):
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other):
        products = [
            self.lo * other.lo, self.lo * other.hi,
            self.hi * other.lo, self.hi * other.hi
        ]
        return Interval(min(products), max(products))

    def __repr__(self):
        mid = (self.lo + self.hi) / 2
        radius = (self.hi - self.lo) / 2
        return f"[{self.lo:.17g}, {self.hi:.17g}] (width: {self.hi - self.lo:.3e})"

    def contains(self, value):
        return self.lo <= value <= self.hi

# Interval arithmetic for 0.1 + 0.2
a = Interval(0.1)
b = Interval(0.2)
c = a + b
print(f"Interval of 0.1: {a}")
print(f"Interval of 0.2: {b}")
print(f"0.1+0.2:         {c}")
print(f"Contains 0.3:    {c.contains(0.3)}")
```

### 10.3 Detecting Numerical Instability

```python
import math
import warnings

def check_numerical_stability(func, x, delta=1e-8):
    """Simple numerical stability check for a function"""
    y = func(x)
    y_plus = func(x + delta)
    y_minus = func(x - delta)

    # Approximate condition number: |x * f'(x) / f(x)|
    if abs(y) > 0:
        deriv_approx = (y_plus - y_minus) / (2 * delta)
        cond_approx = abs(x * deriv_approx / y)
    else:
        cond_approx = float('inf')

    # Symmetry check
    forward_diff = y_plus - y
    backward_diff = y - y_minus
    if abs(forward_diff) > 0:
        symmetry = abs(forward_diff - backward_diff) / abs(forward_diff)
    else:
        symmetry = 0.0

    print(f"f({x}) = {y}")
    print(f"Estimated condition number: {cond_approx:.2e}")
    if cond_approx > 1e10:
        warnings.warn(f"Very large condition number ({cond_approx:.2e}): possible numerical instability")
    print(f"Difference symmetry: {symmetry:.2e}")
    if symmetry > 0.01:
        warnings.warn("Low difference symmetry: rounding error may have significant impact")

# Test: function that produces catastrophic cancellation
def unstable_func(x):
    """Catastrophic cancellation occurs when x is near 0"""
    return (1 - math.cos(x)) / (x * x)  # Theoretical value converges to 0.5 as x->0

def stable_func(x):
    """Mathematically equivalent but numerically stable"""
    return 2 * (math.sin(x/2) / x) ** 2  # Using the half-angle formula

print("=== Unstable implementation ===")
check_numerical_stability(unstable_func, 1e-8)
print()
print("=== Stable implementation ===")
check_numerical_stability(stable_func, 1e-8)
```

### 10.4 Trapping Floating-Point Exceptions

```python
# Detecting floating-point exceptions in Python

import numpy as np

# NumPy floating-point exception settings
# By default, only warnings. 'raise' makes them raise exceptions
old_settings = np.seterr(all='raise')  # Raise exception on all FP errors

try:
    result = np.float64(1e308) * np.float64(10)  # Overflow
except FloatingPointError as e:
    print(f"Caught: {e}")

try:
    result = np.float64(0.0) / np.float64(0.0)  # Invalid operation
except FloatingPointError as e:
    print(f"Caught: {e}")

np.seterr(**old_settings)  # Restore settings

# Detection with warnings module
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

try:
    result = np.float64(1.0) / np.float64(0.0)
except RuntimeWarning as e:
    print(f"Caught warning: {e}")

warnings.resetwarnings()
```

```c
/* Trapping floating-point exceptions in C */
#define _GNU_SOURCE
#include <stdio.h>
#include <fenv.h>
#include <math.h>
#include <signal.h>

/* Floating-point exception handler */
void fpe_handler(int sig) {
    printf("Floating-point exception occurred!\n");

    /* Check exception flags */
    if (fetestexcept(FE_DIVBYZERO))
        printf("  - Division by zero\n");
    if (fetestexcept(FE_OVERFLOW))
        printf("  - Overflow\n");
    if (fetestexcept(FE_UNDERFLOW))
        printf("  - Underflow\n");
    if (fetestexcept(FE_INVALID))
        printf("  - Invalid operation\n");
    if (fetestexcept(FE_INEXACT))
        printf("  - Inexact (rounding occurred)\n");

    feclearexcept(FE_ALL_EXCEPT);
}

int main(void) {
    /* Post-hoc check using exception flags (recommended) */
    feclearexcept(FE_ALL_EXCEPT);

    volatile double a = 1.0;
    volatile double b = 0.0;
    volatile double result = a / b;

    if (fetestexcept(FE_DIVBYZERO)) {
        printf("Division by zero occurred: result = %f\n", result);
    }

    feclearexcept(FE_ALL_EXCEPT);
    result = sqrt(-1.0);

    if (fetestexcept(FE_INVALID)) {
        printf("Invalid operation occurred: result = %f\n", result);
    }

    return 0;
}
```

---

## 11. Practice Exercises

### Exercise 1: Manual IEEE 754 Conversion (Fundamentals)

Compute the binary32 (single precision) bit representation of the following values by hand. Show each step explicitly (sign determination, binary conversion, normalization, field determination).

1. **-0.75**
2. **100.0**
3. **0.1** (show rounding at the 52nd bit)

```
Solution example (-0.75):

  Step 1: Sign
    -0.75 < 0 -> S = 1

  Step 2: Binary conversion
    0.75 x 2 = 1.5 -> 1
    0.5  x 2 = 1.0 -> 1
    -> 0.75 = 0.11 (binary)

  Step 3: Normalize
    0.11 = 1.1 x 2^(-1)

  Step 4: Field determination
    S = 1
    E = -1 + 127 = 126 = 01111110 (binary)
    M = 10000000000000000000000 (23 bits)

  Result: 1 01111110 10000000000000000000000
  Hexadecimal: 0xBF400000

  Verification: struct.pack('>f', -0.75).hex() -> 'bf400000'
```

### Exercise 2: Experiencing Precision Limits (Applied)

Verify the following in Python and explain each result based on IEEE 754 operating principles.

1. `float(2**53) == float(2**53 + 1)` -> Result and reason
2. Difference between `1e20 + 1 - 1e20` and `1e20 - 1e20 + 1`
3. Difference between `sum([0.1]*10) == 1.0` and `math.fsum([0.1]*10) == 1.0`
4. `(0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3)` -> Verify associativity
5. Predict the output of the following code and explain:

```python
x = 1e16
print(x + 1 == x)
print(x + 2 == x)
```

### Exercise 3: Implementing a Safe Floating-Point Library (Advanced)

Implement a floating-point utility module in Python that meets the following requirements.

**Requirements:**
1. `safe_equal(a, b, rel_tol, abs_tol)`: Comparison function that correctly handles all edge cases (NaN, Inf, -0, denormalized numbers)
2. `safe_sum(values)`: High-precision sum using Kahan compensated summation
3. `safe_mean(values)`: Mean that ignores NaN and returns NaN for empty lists
4. `float_info(x)`: Function that returns IEEE 754 decomposition info as a dictionary

**Test cases:**
```python
# safe_equal
assert safe_equal(0.1 + 0.2, 0.3)
assert safe_equal(float('inf'), float('inf'))
assert not safe_equal(float('nan'), float('nan'))
assert safe_equal(+0.0, -0.0)

# safe_sum
assert abs(safe_sum([0.1] * 10) - 1.0) < 1e-15

# safe_mean
assert safe_mean([1.0, float('nan'), 3.0]) == 2.0
assert math.isnan(safe_mean([]))

# float_info
info = float_info(6.5)
assert info['sign'] == 0
assert info['exponent'] == 2
assert info['category'] == 'normal'
```

---

## 12. Historical Accidents and Failures Caused by Floating-Point

Floating-point precision problems are not merely of theoretical interest -- they have caused serious real-world accidents and economic losses. The following are representative cases.

### 12.1 Patriot Missile Interception Failure (1991)

During the 1991 Gulf War, a Patriot missile system deployed in Dhahran, Saudi Arabia, failed to intercept a Scud missile, resulting in the deaths of 28 US soldiers.

```
Cause: Accumulation of rounding errors in time calculation

  Internal time management:
  - Clock in 0.1-second increments managed in 24-bit fixed-point
  - 0.1 (decimal) = 0.0001100110011001100... (binary, infinite repeating)
  - Truncated to 24 bits: 0.00011001100110011001100
  - Error per tick: approximately 9.5 x 10^(-8) seconds

  Accumulated error after 100 hours of continuous operation:
  - 100 hours = 360,000 seconds = 3,600,000 x 0.1-second ticks
  - Accumulated error = 3,600,000 x 9.5 x 10^(-8) ~ 0.34 seconds

  Scud missile speed: approximately 1,676 m/s
  0.34-second tracking error = approximately 570 m offset

  -> Target fell outside the radar tracking gate, interception failed

Lessons:
  - Even small rounding errors can become catastrophic over long accumulation
  - Special care is needed when using floating-point in real-time systems
  - Periodic error reset or correction is essential
```

### 12.2 Ariane 5 Rocket Explosion (1996)

ESA's (European Space Agency) Ariane 5 rocket exploded shortly after launch. Development costs were approximately $7 billion, and 4 satellites (approximately $500 million) were also lost.

```
Cause: Overflow in conversion from 64-bit floating-point to 16-bit integer

  Ariane 4 code was reused for Ariane 5
  - Process converting horizontal velocity from float64 to int16
  - In Ariane 4, velocity stayed within int16 range
  - Ariane 5 was more powerful -> higher horizontal velocity -> int16 overflow
  - Ada Operand Error exception raised
  - Inertial navigation system shut down
  - Backup used identical code -> simultaneously shut down
  - Loss of control -> self-destruction

  The problematic conversion code (Ada, conceptual reproduction):
    horizontal_bias := INTEGER(horizontal_velocity);
    -- Constraint_Error when horizontal_velocity exceeds 32768

Lessons:
  - Range checking is always required when converting floating-point to integer
  - Prerequisite verification is essential when reusing code
  - Redundant systems must not share identical bugs
```

### 12.3 Vancouver Stock Exchange Index Error (1982)

```
Cause: Accumulation of truncation errors in index calculation

  Vancouver Stock Exchange (VSE) stock index:
  - Started at 1000.000 in 1982
  - Recalculated the index after each trade, truncating at 3 decimal places (floor)
  - Approximately 3000 trades per day -> maximum 0.0005 truncation error per trade
  - After 22 months: index was 524.811 (should have been approximately 1098)

  Detailed cause:
    Truncation (floor) always has a negative directional bias
    Used truncation instead of round-to-even
    3000/day x 22 months ~ 2,000,000 truncations
    Accumulated error: approximately 52% of the index was lost

  Fix: Changed from truncation to rounding and recalculated the index

Lessons:
  - The choice of rounding mode makes a dramatic difference across many operations
  - Rounding rule selection is particularly important in financial systems
  - Truncation has systematic bias
```

### 12.4 Excel Date Bug and Floating-Point

```
Famous floating-point related issues in Excel:

  1. February 29, 1900 problem
     Excel treats 1900 as a leap year (it actually is not)
     Bug intentionally kept for Lotus 1-2-3 compatibility

  2. Precision issues
     Excel internally uses IEEE 754 binary64
     But display precision is limited to 15 digits
     =1/3*3 displays as 1.0 (internal value is 0.999...99)
     -> Display rounding can mask precision problems

  3. Large number subtraction
     =1E15+1-1E15 -> 0 (correct answer is 1)
     -> Loss of significance due to absorption
     -> Scientific computing in spreadsheets requires caution
```

---

## 13. Floating-Point and Formal Verification

### 13.1 Mathematical Properties of Floating-Point Arithmetic

Floating-point arithmetic has algebraic properties that differ from real number arithmetic. Understanding these differences is essential for writing correct programs.

```
Mathematical laws that do NOT hold for floating-point arithmetic:

  +-----------------+--------------+-----------------------------+
  | Law             | Real numbers | Floating-point              |
  +-----------------+--------------+-----------------------------+
  | Associativity   | (a+b)+c      | (1e20+1)-1e20 = 0          |
  | (a+b)+c=a+(b+c)| = a+(b+c)    | 1e20+(1-1e20) = -1e20+1 = 1|
  +-----------------+--------------+-----------------------------+
  | Distributivity  | a x (b+c)    | Generally does not hold     |
  | a(b+c)=ab+ac   | = a x b+a x c| Mismatch due to rounding    |
  +-----------------+--------------+-----------------------------+
  | Inverse element | a+(-a) = 0   | Holds (exactly 0)           |
  |                 | a x (1/a) = 1| Generally does not hold      |
  +-----------------+--------------+-----------------------------+
  | Transitivity    | a<b, b<c     | Fails due to NaN            |
  |                 | -> a<c       | NaN<1: False, NaN<2: False  |
  +-----------------+--------------+-----------------------------+
  | Reflexivity     | a = a        | Fails: NaN != NaN           |
  +-----------------+--------------+-----------------------------+

  Laws that DO hold for floating-point:
  - Commutativity: a + b = b + a, a x b = b x a (always holds)
  - Monotonicity: a <= b => a + c <= b + c (holds except for NaN)
  - Sterbenz theorem: if a/2 <= b <= 2a, then b - a is computed exactly
```

### 13.2 Exact Operations

In IEEE 754, under specific conditions, operation results are guaranteed to be exact.

```python
# Examples of operations computed exactly

# 1. Subtraction of same-sign values (Sterbenz theorem)
a = 1.5
b = 1.0
# a/2 <= b <= 2a is satisfied, so a - b is exact
print(a - b)  # 0.5 (exact)

# 2. Multiplication/division by powers of 2
x = 3.14159
print(x * 2.0)     # 6.28318 (exact: only exponent adjustment)
print(x * 0.5)     # 1.570795 (exact: only exponent adjustment)
print(x * 4.0)     # 12.56636 (exact)

# 3. FMA (Fused Multiply-Add)
import math
# math.fma(a, b, c) = a*b + c computed with a single rounding (Python 3.13+)
# -> The intermediate result of a*b is not rounded, so it is more accurate
#    than the regular a*b + c

# 4. Double-double arithmetic
def two_sum(a, b):
    """Compute a + b with high precision. s + e = a + b holds exactly."""
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return s, e  # s is the rounded sum, e is the error

s, e = two_sum(1e16, 1.0)
print(f"Sum: {s}, Error: {e}")  # Sum: 1e16, Error: 1.0
# -> The lost information is preserved in e
```

---

## 14. FAQ (Frequently Asked Questions)

### Q1: Should I use float or double?

**A**: As a general rule, **use double (64-bit)**. On most modern CPUs, there is no significant performance difference between float and double operations. The cases where float should be chosen are limited:

- GPU/AI: float16/bfloat16/float32 used due to VRAM capacity constraints
- Large arrays: Memory usage is halved (NumPy's dtype='float32')
- SIMD optimization: float32 can process 2x the elements of float64 simultaneously
- Games/Graphics: float32 precision is often sufficient

In C/C++, `double` is the default literal type, and `printf`'s `%f` also accepts double. Python's `float` is internally a C double (64-bit).

### Q2: Why is Banker's Rounding the default?

**A**: To minimize statistical bias.

With standard rounding (half up), 0.5 is always rounded up, causing results to be biased positively when large numbers of rounding operations are performed. For example, rounding 0.5, 1.5, 2.5, 3.5, 4.5 gives 1+2+3+4+5=15, but round-to-even gives 0+2+2+4+4=12, which is closer to the true sum of 12.5.

This bias is particularly problematic in financial calculations. When millions of transaction amounts are each rounded, standard rounding produces systematic profits/losses. Round-to-even statistically eliminates this problem. IEEE 754 made this the default because it minimizes rounding error accumulation in general-purpose computing as well.

### Q3: Why doesn't JavaScript have an integer type?

**A**: The result of Brendan Eich prioritizing simplicity during the 1995 design. All numbers are treated as IEEE 754 binary64 (double).

Integers up to `2^53 = 9007199254740992` can be represented exactly, but precision is lost beyond that. `Number.MAX_SAFE_INTEGER = 9007199254740991` is the largest integer that can be safely handled.

`BigInt` was added in ES2020, enabling arbitrary-precision integers. However, BigInt and Number cannot be mixed in operations (`1n + 1` is an error).

Since the JSON specification does not include BigInt, large integer IDs (such as Twitter's snowflake IDs) are sent and received as strings as a practical standard.

### Q4: Are there cases where -ffast-math is acceptable?

**A**: In game engines and some signal processing contexts where "a sufficiently close result obtained quickly is acceptable," its use may be tolerated. However, use it only after understanding the following risks:

- `isnan()`, `isinf()` always return false
- NaN/Inf propagation is not guaranteed
- Operation order may be changed (associativity assumed)
- `-0.0` and `+0.0` are not distinguished

Never use for scientific computing, financial calculations, or cryptographic processing.

### Q5: What is "machine epsilon"?

**A**: Machine epsilon is the smallest floating-point number epsilon such that `1.0 + epsilon > 1.0`. In other words, it is the smallest value that, when added to 1.0, produces a result distinguishable from 1.0.

- binary32: epsilon = 2^(-23) ~ 1.19 x 10^(-7)
- binary64: epsilon = 2^(-52) ~ 2.22 x 10^(-16)

Machine epsilon represents the "upper bound on relative rounding error." When rounding any real number x to its nearest floating-point representation fl(x), `|fl(x) - x| / |x| <= epsilon/2` holds.

In Python, it can be obtained via `sys.float_info.epsilon`; in C, via `DBL_EPSILON` (`<float.h>`).

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

### Key Concept Overview

| Concept | Key Point |
|---------|-----------|
| IEEE 754 structure | Sign(1) + Exponent(8/11) + Mantissa(23/52). Implicit leading 1 bit |
| Normalized numbers | `(-1)^S x 1.M x 2^(E-bias)`. Standard floating-point numbers |
| Denormalized numbers | `(-1)^S x 0.M x 2^(1-bias)`. Gradual underflow |
| Special values | +/-0, +/-Inf, NaN. NaN != NaN is the key caution |
| Precision issues | 0.1 is an infinite repeating fraction in binary. Use epsilon for comparison |
| Catastrophic cancellation | Subtraction of close values drastically reduces significant digits. Avoid through formula transformation |
| Absorption | Adding values of different magnitudes causes the smaller value to vanish. Addition order matters |
| Rounding modes | Default is round to nearest even. Minimizes statistical bias |
| Density distribution | Dense near zero, sparse for large values. ULP is proportional to exponent |
| Integer precision limit | float32: 2^24, float64: 2^53 -- integer precision lost beyond these |
| AI low precision | BF16/FP16/FP8. Memory bandwidth is the bottleneck; speed over precision |
| Mixed precision training | Weights in FP32, computation in FP16/BF16. Loss Scaling is essential |

### Countermeasure Checklist

```
[ ] Avoid == comparison for floating-point; use math.isclose() or epsilon comparison
[ ] Use Decimal type or integer arithmetic for financial calculations
[ ] Use Kahan compensated summation or math.fsum for large-scale addition
[ ] Consider formula transformations to avoid catastrophic cancellation
[ ] Pay attention to addition order (add smallest values first)
[ ] Use dedicated isnan() functions for NaN detection
[ ] Handle large integer IDs in JSON as strings
[ ] Do not casually use -ffast-math compiler flag
[ ] Check condition numbers in numerical computing; beware of ill-conditioned problems
[ ] Select appropriate precision format for AI/ML use cases
```

---

## Recommended Next Reading


---

## References

1. Goldberg, D. "What Every Computer Scientist Should Know About Floating-Point Arithmetic." *ACM Computing Surveys*, Vol. 23, No. 1, pp. 5-48, 1991. -- The classic treatise on floating-point. Required reading for every programmer.
2. IEEE. "IEEE 754-2019: Standard for Floating-Point Arithmetic." IEEE, 2019. -- The current IEEE 754 standard. Includes binary16, binary128, and decimal formats.
3. Kahan, W. "How Java's Floating-Point Hurts Everyone Everywhere." *Lecture Notes*, UC Berkeley, 1998. -- Critique and recommendations regarding Java's floating-point implementation by the lead designer of IEEE 754.
4. Muller, J.-M. et al. "Handbook of Floating-Point Arithmetic." 2nd Edition, Birkhauser, 2018. -- Comprehensive reference on floating-point arithmetic. Covers algorithms and error analysis.
5. Micikevicius, P. et al. "Mixed Precision Training." *ICLR 2018*. -- NVIDIA's proposal for mixed precision training. Theory and practice of Loss Scaling.
6. Higham, N. J. "Accuracy and Stability of Numerical Algorithms." 2nd Edition, SIAM, 2002. -- The definitive text on accuracy and stability of numerical algorithms.
7. Patterson, D. A. and Hennessy, J. L. "Computer Organization and Design." 6th Edition, Morgan Kaufmann, 2020. -- Computer architecture textbook. Includes explanation of floating-point hardware.
