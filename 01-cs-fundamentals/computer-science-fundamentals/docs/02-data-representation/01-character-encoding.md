# Character Encoding and Unicode

> The cause of garbled text is always "encoding mismatch" -- understanding how UTF-8 works allows you to prevent it at the root.

## Learning Objectives

- [ ] Explain the evolution from ASCII to Unicode to UTF-8
- [ ] Manually verify the UTF-8 byte structure through hand calculation
- [ ] Explain the causes of and solutions for garbled text (mojibake)
- [ ] Explain the differences between UTF-8, UTF-16, and UTF-32
- [ ] Understand the concept and implementation of Unicode normalization (NFC/NFD)
- [ ] Be aware of string handling considerations in each programming language

## Prerequisites


---

## 1. History of Character Encoding

### 1.1 ASCII (1963)

```
ASCII: 7 bits = 128 characters

  0x00-0x1F: Control characters (LF, HT, NULL, etc.)
  0x20:      Space
  0x30-0x39: Digits '0'-'9'
  0x41-0x5A: Uppercase 'A'-'Z'
  0x61-0x7A: Lowercase 'a'-'z'
  0x7F:      DEL

  Characteristics:
  - English only. Cannot represent Japanese, Chinese, etc.
  - Difference between uppercase and lowercase is 0x20 (bit 5)
    'A' = 0x41 = 0100 0001
    'a' = 0x61 = 0110 0001
    →  diff = 0010 0000 = 0x20 = 32
  - Digits '0'-'9' are 0x30-0x39 (lower 4 bits are the numeric value itself)
```

### 1.2 Complete ASCII Table

```
ASCII Code Table (all 128 characters):

  Control Characters (0x00-0x1F):
  Dec  Hex  Char  Description
  ───  ───  ────  ────────────────
    0  0x00  NUL  Null (string terminator)
    1  0x01  SOH  Start of Heading
    2  0x02  STX  Start of Text
    3  0x03  ETX  End of Text (Ctrl+C)
    4  0x04  EOT  End of Transmission (Ctrl+D)
    7  0x07  BEL  Bell (terminal bell sound)
    8  0x08  BS   Backspace
    9  0x09  HT   Horizontal Tab
   10  0x0A  LF   Line Feed (Unix newline)
   11  0x0B  VT   Vertical Tab
   12  0x0C  FF   Form Feed (page break)
   13  0x0D  CR   Carriage Return (old Mac newline)
   27  0x1B  ESC  Escape (start of ANSI escape sequences)
   31  0x1F  US   Unit Separator

  Newline conventions:
  Unix/Linux/macOS: LF (0x0A) = \n
  Windows: CR+LF (0x0D 0x0A) = \r\n
  Classic Mac (OS 9 and earlier): CR (0x0D) = \r

  Printable Characters (0x20-0x7E):
  Dec  Hex  Char │ Dec  Hex  Char │ Dec  Hex  Char
  ───  ───  ──── │ ───  ───  ──── │ ───  ───  ────
   32  0x20  SP  │  64  0x40  @   │  96  0x60  `
   33  0x21  !   │  65  0x41  A   │  97  0x61  a
   34  0x22  "   │  66  0x42  B   │  98  0x62  b
   35  0x23  #   │  67  0x43  C   │  99  0x63  c
   36  0x24  $   │  68  0x44  D   │ 100  0x64  d
   37  0x25  %   │  69  0x45  E   │ 101  0x65  e
   38  0x26  &   │  70  0x46  F   │ 102  0x66  f
   39  0x27  '   │  71  0x47  G   │ 103  0x67  g
   40  0x28  (   │  72  0x48  H   │ 104  0x68  h
   41  0x29  )   │  73  0x49  I   │ 105  0x69  i
   42  0x2A  *   │  74  0x4A  J   │ 106  0x6A  j
   43  0x2B  +   │  75  0x4B  K   │ 107  0x6B  k
   44  0x2C  ,   │  76  0x4C  L   │ 108  0x6C  l
   45  0x2D  -   │  77  0x4D  M   │ 109  0x6D  m
   46  0x2E  .   │  78  0x4E  N   │ 110  0x6E  n
   47  0x2F  /   │  79  0x4F  O   │ 111  0x6F  o
   48  0x30  0   │  80  0x50  P   │ 112  0x70  p
   49  0x31  1   │  81  0x51  Q   │ 113  0x71  q
   50  0x32  2   │  82  0x52  R   │ 114  0x72  r
   51  0x33  3   │  83  0x53  S   │ 115  0x73  s
   52  0x34  4   │  84  0x54  T   │ 116  0x74  t
   53  0x35  5   │  85  0x55  U   │ 117  0x75  u
   54  0x36  6   │  86  0x56  V   │ 118  0x76  v
   55  0x37  7   │  87  0x57  W   │ 119  0x77  w
   56  0x38  8   │  88  0x58  X   │ 120  0x78  x
   57  0x39  9   │  89  0x59  Y   │ 121  0x79  y
   58  0x3A  :   │  90  0x5A  Z   │ 122  0x7A  z
   59  0x3B  ;   │  91  0x5B  [   │ 123  0x7B  {
   60  0x3C  <   │  92  0x5C  \   │ 124  0x7C  |
   61  0x3D  =   │  93  0x5D  ]   │ 125  0x7D  }
   62  0x3E  >   │  94  0x5E  ^   │ 126  0x7E  ~
   63  0x3F  ?   │  95  0x5F  _   │ 127  0x7F  DEL

  Practically important ASCII properties:
  - 'A'-'Z': 0x41-0x5A (bit 5 = 0)
  - 'a'-'z': 0x61-0x7A (bit 5 = 1)
  - Case toggle: c ^ 0x20
  - '0'-'9': 0x30-0x39 (c - 0x30 yields the numeric value)
  - Printable characters: range 0x20-0x7E
```

### 1.3 Design Philosophy of ASCII

```
Why ASCII uses 7 bits:

  Teletype communications in the 1960s:
  - 5 bits (Baudot code): only 32 characters representable
  - 6 bits: 64 characters (BCDIC, etc.)
  - 7 bits: 128 characters -> all English characters + control chars + symbols
  - 8 bits: 256 characters -> considered "extravagant" at the time

  The remaining 1 bit (8th bit):
  - Used for parity checking (communication error detection)
  - Later repurposed for extended character sets such as ISO 8859

  Why the uppercase/lowercase design is clever:
  - Toggled by switching bit 5 ON/OFF
  - Easily converted in hardware
  - Alphabetical order approximately equals ascending code order

  Design of control characters:
  - Ctrl+key = ASCII code of key - 0x40
    Ctrl+A = 0x41 - 0x40 = 0x01 (SOH)
    Ctrl+C = 0x43 - 0x40 = 0x03 (ETX -> interrupt signal)
    Ctrl+D = 0x44 - 0x40 = 0x04 (EOT -> EOF)
    Ctrl+G = 0x47 - 0x40 = 0x07 (BEL -> bell sound)
    Ctrl+H = 0x48 - 0x40 = 0x08 (BS -> backspace)
    Ctrl+I = 0x49 - 0x40 = 0x09 (HT -> tab)
    Ctrl+J = 0x4A - 0x40 = 0x0A (LF -> line feed)
    Ctrl+M = 0x4D - 0x40 = 0x0D (CR -> carriage return)
```

### 1.4 The Era of Character Encoding Chaos

| Encoding | Era | Target | Problem |
|----------|-----|--------|---------|
| ASCII | 1963 | English | Only 128 characters |
| Latin-1 (ISO-8859-1) | 1987 | Western European | 256 characters, no Japanese support |
| Shift_JIS | 1982 | Japanese | Windows standard. Variable-length. Difficult to mix with other languages |
| EUC-JP | 1985 | Japanese | UNIX standard. Incompatible with Shift_JIS |
| ISO-2022-JP | 1994 | Japanese | Email standard. Switches via escape sequences |
| GB2312/GBK | 1980 | Chinese | Incompatible with Japanese |
| Big5 | 1984 | Traditional Chinese | Incompatible with Simplified Chinese |
| KS X 1001 | 1986 | Korean | Used as EUC-KR |

-> **Each language/region had its own character encoding** -> **an interoperability nightmare**

### 1.5 Japanese Character Encoding in Detail

```
[JIS X 0208 (JIS Level 1 and Level 2)]
  - 6,879 characters (4,888 kanji + 1,991 non-kanji)
  - Managed in a 94x94 matrix
  - Kuten code: row number (1-94) x point number (1-94)

[Shift_JIS (MS Kanji Code)]
  Structure:
  - Single-byte characters: 0x00-0x7F (ASCII) + 0xA1-0xDF (half-width katakana)
  - Double-byte characters: 1st byte 0x81-0x9F, 0xE0-0xEF
                             2nd byte 0x40-0x7E, 0x80-0xFC

  Problems:
  - The kanji for "table" (0x955C) has 0x5C as its 2nd byte = '\' -> path separator collision
  - Similar issues with characters like "so" and "nou" (the "dame-moji" problem)
  - Half-width katakana conflicts with upper byte range of ASCII

  Example: byte sequence for the kanji compound "hyouji" (display)
  hyou = 0x95 0x5C  (2nd byte is 0x5C = backslash)
  ji   = 0x8E 0xA6
  -> In C string literals, the backslash acts as an escape character

[EUC-JP (Extended Unix Code)]
  Structure:
  - Single-byte characters: 0x00-0x7F (ASCII)
  - Double-byte characters: each byte 0xA1-0xFE
  - Triple-byte characters: 0x8F + 2 bytes (JIS Level 3)

  Advantages:
  - No collision with ASCII (uses only upper byte range)
  - Long-standing standard on Unix-based systems

[ISO-2022-JP (JIS Code)]
  Structure:
  - Switches between ASCII/Japanese via escape sequences
  - Start ASCII: ESC ( B
  - Start JIS X 0208: ESC $ B
  - Uses only the 7-bit range (suitable for email transmission)

  Example: encoding "ABCaiu" (ABC + Japanese hiragana)
  ESC ( B -> ASCII mode -> 41 42 43
  ESC $ B -> JIS mode -> 24 22 24 24 24 26
  ESC ( B -> Return to ASCII mode
```

### 1.6 The Birth of Unicode (1991)

The philosophy of Unicode: **Assign a unique code point to every character in every language**

```
Unicode code points:

  U+0041  = 'A' (Latin letter)
  U+3042  = 'a' (Hiragana, Japanese)
  U+4E16  = 'shi' (CJK Unified Ideograph, meaning "world")
  U+1F600 = 'grinning face' (Emoji)
  U+1F4A9 = 'pile of poo' (Emoji)

  Range: U+0000 to U+10FFFF (approximately 1.1 million code points)
  Assigned: approximately 150,000 characters (as of 2024)
  -> Capable of accommodating all characters used by humanity
```

### 1.7 Unicode Plane Structure

```
Unicode's 17 Planes (Plane 0-16):

  Plane 0: BMP (Basic Multilingual Plane) U+0000 - U+FFFF
    - Contains the vast majority of commonly used characters
    - Latin, Hiragana, Katakana, main CJK Ideographs
    - 65,536 code points

  Plane 1: SMP (Supplementary Multilingual Plane) U+10000 - U+1FFFF
    - Emoji (U+1F600-U+1F64F, etc.)
    - Cuneiform, Egyptian Hieroglyphs
    - Ancient scripts, musical symbols, mathematical symbols

  Plane 2: SIP (Supplementary Ideographic Plane) U+20000 - U+2FFFF
    - CJK Unified Ideographs Extension B and beyond (rare kanji)
    - Approximately 42,711 characters

  Plane 3: TIP (Tertiary Ideographic Plane) U+30000 - U+3FFFF
    - CJK Unified Ideographs Extension G and beyond

  Plane 14: SSP (Supplementary Special-purpose Plane) U+E0000 - U+EFFFF
    - Tag characters, variation selectors

  Planes 15-16: PUA (Private Use Areas)
    - Private use areas (freely definable by organizations or individuals)

  Plane visualization:
  +-------------------------------------------+
  | Plane 0 (BMP): Nearly all modern scripts  |
  | U+0000-U+FFFF                             |
  | ASCII, Latin, Greek, Cyrillic,            |
  | Hiragana, Katakana, most CJK Ideographs   |
  +-------------------------------------------+
  | Plane 1 (SMP): Emoji, ancient scripts     |
  | U+10000-U+1FFFF                           |
  +-------------------------------------------+
  | Plane 2 (SIP): Rare CJK Ideographs       |
  | U+20000-U+2FFFF                           |
  +-------------------------------------------+
  | Planes 3-13: Mostly unassigned            |
  +-------------------------------------------+
  | Plane 14 (SSP): Special purpose           |
  +-------------------------------------------+
  | Planes 15-16: Private Use Areas           |
  +-------------------------------------------+
```

---

## 2. UTF-8 -- The Modern Standard

### 2.1 UTF-8 Byte Structure

```
UTF-8 Encoding Rules:

  Code Point Range            Bytes  Byte Pattern
  ──────────────────────────────────────────────────
  U+0000  - U+007F         1 byte   0xxxxxxx
  U+0080  - U+07FF         2 bytes  110xxxxx 10xxxxxx
  U+0800  - U+FFFF         3 bytes  1110xxxx 10xxxxxx 10xxxxxx
  U+10000 - U+10FFFF       4 bytes  11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

  The number of leading 1s in the first byte = number of bytes
  Continuation bytes always start with 10
  -> Character boundaries can be identified from any byte position (self-synchronization)

  Byte classification (leading bit patterns):
  0xxxxxxx -> First byte of a 1-byte character (ASCII-compatible)
  10xxxxxx -> Continuation byte (not a leading byte)
  110xxxxx -> First byte of a 2-byte character
  1110xxxx -> First byte of a 3-byte character
  11110xxx -> First byte of a 4-byte character

  Significance of self-synchronization:
  - Any byte immediately reveals whether it is a character start or a continuation
  - The next character boundary can be found even from the middle of a stream
  - A single corrupted byte affects at most 1 character (UTF-16 can affect up to 2)
```

### 2.2 Concrete Examples (Encoding Procedure)

```
'A' = U+0041:
  Binary: 100 0001 (7 bits -> fits in 1 byte)
  UTF-8: 0_1000001 = 0x41 (1 byte, fully compatible with ASCII)

'yen sign' = U+00A5:
  Binary: 10 100101 (8 bits -> 2 bytes required)
  Template: 110xxxxx 10xxxxxx
  Split: 00010 100101
  Fill:  110_00010 10_100101
  Result: 0xC2 0xA5 (2 bytes)

  Verification: 00010_100101 = 0x0A5 = 165 = U+00A5

'a' (Hiragana) = U+3042:
  Binary: 0011 0000 0100 0010 (16 bits -> 3 bytes required)
  Template: 1110xxxx 10xxxxxx 10xxxxxx
  Split: 0011 000001 000010
  Fill:  1110_0011 10_000001 10_000010
  Result: 0xE3 0x81 0x82 (3 bytes)

  Encoding steps:
  U+3042 = 0011 000001 000010
  Template: 1110xxxx 10xxxxxx 10xxxxxx
  Fill:     1110_0011 10_000001 10_000010
  Result:   E3 81 82

'kan' (kanji for "Chinese character") = U+6F22:
  Binary: 0110 1111 0010 0010
  Split: 0110 111100 100010
  Fill:  1110_0110 10_111100 10_100010
  Result: 0xE6 0xBC 0xA2 (3 bytes)

'grinning face' = U+1F600:
  Binary: 0001 1111 0110 0000 0000 (21 bits -> 4 bytes required)
  Template: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
  Split: 000 011111 011000 000000
  Fill:  11110_000 10_011111 10_011000 10_000000
  Result: 0xF0 0x9F 0x98 0x80 (4 bytes)

'tsuchiyoshi' = U+20BB7 (JIS Level 3 kanji):
  Binary: 0010 0000 1011 1011 0111
  Split: 000 100000 101110 110111
  Fill:  11110_000 10_100000 10_101110 10_110111
  Result: 0xF0 0xA0 0xAE 0xB7 (4 bytes)
```

### 2.3 Advantages of UTF-8

| Property | Description |
|----------|-------------|
| ASCII-compatible | ASCII text is valid UTF-8 as-is |
| Variable-length | 1-4 bytes. Compact for English, supports all languages |
| Self-synchronizing | Character boundaries recoverable from any byte position |
| Sortable | Lexicographic byte order approximately equals ascending code point order |
| No BOM required | No endianness issues (unlike UTF-16) |
| NUL-safe | 0x00 byte never appears except for U+0000 (safe for C strings) |
| Widely adopted | Over 98% of the Web uses UTF-8 |

### 2.4 UTF-8 Decoding Algorithm

```python
def utf8_decode_manual(byte_sequence):
    """Manually decode a UTF-8 byte sequence (educational implementation)"""
    result = []
    i = 0
    while i < len(byte_sequence):
        b = byte_sequence[i]

        if b < 0x80:
            # 1-byte character (0xxxxxxx)
            codepoint = b
            i += 1
        elif b < 0xC0:
            # Continuation byte (10xxxxxx) at the start is an error
            raise ValueError(f"Invalid continuation byte: 0x{b:02X} at position {i}")
        elif b < 0xE0:
            # 2-byte character (110xxxxx 10xxxxxx)
            if i + 1 >= len(byte_sequence):
                raise ValueError("Incomplete 2-byte character")
            codepoint = ((b & 0x1F) << 6) | (byte_sequence[i+1] & 0x3F)
            # Overlong detection: values below U+0080 should use 1 byte
            if codepoint < 0x80:
                raise ValueError(f"Overlong encoding: U+{codepoint:04X}")
            i += 2
        elif b < 0xF0:
            # 3-byte character (1110xxxx 10xxxxxx 10xxxxxx)
            if i + 2 >= len(byte_sequence):
                raise ValueError("Incomplete 3-byte character")
            codepoint = ((b & 0x0F) << 12) | \
                       ((byte_sequence[i+1] & 0x3F) << 6) | \
                       (byte_sequence[i+2] & 0x3F)
            if codepoint < 0x800:
                raise ValueError(f"Overlong encoding: U+{codepoint:04X}")
            # Surrogate pair range is invalid
            if 0xD800 <= codepoint <= 0xDFFF:
                raise ValueError(f"Surrogate: U+{codepoint:04X}")
            i += 3
        elif b < 0xF8:
            # 4-byte character (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if i + 3 >= len(byte_sequence):
                raise ValueError("Incomplete 4-byte character")
            codepoint = ((b & 0x07) << 18) | \
                       ((byte_sequence[i+1] & 0x3F) << 12) | \
                       ((byte_sequence[i+2] & 0x3F) << 6) | \
                       (byte_sequence[i+3] & 0x3F)
            if codepoint < 0x10000:
                raise ValueError(f"Overlong encoding: U+{codepoint:04X}")
            if codepoint > 0x10FFFF:
                raise ValueError(f"Out of range: U+{codepoint:04X}")
            i += 4
        else:
            raise ValueError(f"Invalid leading byte: 0x{b:02X}")

        result.append(chr(codepoint))

    return ''.join(result)


# Test
test_bytes = bytes([0xE3, 0x81, 0x82])  # Hiragana 'a'
print(utf8_decode_manual(test_bytes))  # a (hiragana)

test_bytes2 = bytes([0xF0, 0x9F, 0x98, 0x80])  # grinning face emoji
print(utf8_decode_manual(test_bytes2))  # grinning face emoji


def utf8_encode_manual(text):
    """Manually encode a string to UTF-8 byte sequence (educational implementation)"""
    result = bytearray()
    for char in text:
        cp = ord(char)
        if cp < 0x80:
            result.append(cp)
        elif cp < 0x800:
            result.append(0xC0 | (cp >> 6))
            result.append(0x80 | (cp & 0x3F))
        elif cp < 0x10000:
            result.append(0xE0 | (cp >> 12))
            result.append(0x80 | ((cp >> 6) & 0x3F))
            result.append(0x80 | (cp & 0x3F))
        else:
            result.append(0xF0 | (cp >> 18))
            result.append(0x80 | ((cp >> 12) & 0x3F))
            result.append(0x80 | ((cp >> 6) & 0x3F))
            result.append(0x80 | (cp & 0x3F))
    return bytes(result)

# Test
print(utf8_encode_manual("a").hex())  # e38182 (hiragana 'a')
print(utf8_encode_manual("grinning face").hex())  # f09f9880
```

---

## 3. UTF-16 and UTF-32

### 3.1 UTF-16

```
UTF-16 Encoding:

  BMP (U+0000-U+FFFF): Directly represented as 2 bytes
  SMP and above (U+10000-U+10FFFF): Surrogate pairs (4 bytes)

  Surrogate pair calculation:
  1. Subtract 0x10000 from the code point (yields a 20-bit value)
  2. Upper 10 bits + 0xD800 -> High surrogate (0xD800-0xDBFF)
  3. Lower 10 bits + 0xDC00 -> Low surrogate (0xDC00-0xDFFF)

  Example: grinning face U+1F600
  1. 0x1F600 - 0x10000 = 0x0F600
  2. Upper 10 bits: 0x0F600 >> 10 = 0x003D -> + 0xD800 = 0xD83D
  3. Lower 10 bits: 0x0F600 & 0x3FF = 0x0200 -> + 0xDC00 = 0xDE00
  4. UTF-16: 0xD83D 0xDE00

  Endianness issue:
  UTF-16BE: 0xD8 0x3D 0xDE 0x00 (big-endian)
  UTF-16LE: 0x3D 0xD8 0x00 0xDE (little-endian)
  -> Determined by BOM (U+FEFF):
    FF FE -> little-endian
    FE FF -> big-endian

  Environments that use UTF-16:
  - Windows API (WCHAR, wchar_t)
  - Java (char type, String)
  - JavaScript (String)
  - .NET (System.String)
  - macOS/iOS (internal representation of NSString / CFString)
```

```python
# Surrogate pair calculation

def codepoint_to_utf16(cp):
    """Calculate UTF-16 encoding from a code point"""
    if cp < 0x10000:
        return [cp]
    else:
        cp -= 0x10000
        high = 0xD800 + (cp >> 10)
        low = 0xDC00 + (cp & 0x3FF)
        return [high, low]

def utf16_to_codepoint(units):
    """Recover the code point from UTF-16 code units"""
    if len(units) == 1:
        return units[0]
    else:
        high, low = units
        return ((high - 0xD800) << 10) + (low - 0xDC00) + 0x10000

# Test
print([hex(u) for u in codepoint_to_utf16(0x1F600)])  # ['0xd83d', '0xde00']
print(hex(utf16_to_codepoint([0xD83D, 0xDE00])))      # 0x1f600
```

### 3.2 UTF-32

```
UTF-32 Encoding:

  Represents all code points as fixed 4-byte values
  -> Simplest, but worst memory efficiency

  Examples:
  'A'          = 0x00000041
  'a' (hira.)  = 0x00003042
  grinning face = 0x0001F600

  Advantages:
  - Fixed-length, so index access is O(1)
  - Extremely simple implementation
  - Direct comparison of code points is easy

  Disadvantages:
  - Even ASCII characters use 4 bytes (4x compared to UTF-8)
  - High memory usage
  - Endianness issues exist (UTF-32BE/UTF-32LE)
  - Rarely used in practice

  Use cases:
  - Python 3 internal representation (depends on code points: Latin-1/UCS-2/UCS-4 switching)
  - Parts of ICU (International Components for Unicode)
  - Internal use in text processing libraries
```

### 3.3 Encoding Comparison

```
Comparison table of each UTF:

  Character  UTF-8      UTF-16LE      UTF-32LE
  ─────── ────────── ──────────── ────────────
  'A'       41         41 00         41 00 00 00
  'yen'     C2 A5      A5 00         A5 00 00 00
  'a'(hira) E3 81 82   42 30         42 30 00 00
  'kan'     E6 BC A2   22 6F         22 6F 00 00
  grin face F0 9F 98   3D D8 00 DE   00 F6 01 00
            80

  Size comparison ("Hello, sekai!" = 8 ASCII chars + 2 CJK chars):
  UTF-8:  8x1 + 2x3 = 14 bytes
  UTF-16: 8x2 + 2x2 = 20 bytes
  UTF-32: 10x4       = 40 bytes

  Size comparison (Japanese text "nihongo no tesuto" = 7 characters):
  UTF-8:  7x3 = 21 bytes
  UTF-16: 7x2 = 14 bytes
  UTF-32: 7x4 = 28 bytes

  -> For Japanese text, UTF-16 is the most compact
  -> For English text, UTF-8 is the most compact
  -> For mixed text, UTF-8 usually wins
  -> UTF-32 is always the largest
```

---

## 4. Causes and Solutions for Garbled Text (Mojibake)

### 4.1 Common Garbled Text Patterns

```python
# Reproducing and analyzing garbled text

text = "konnichiha"  # Japanese greeting in hiragana

# Byte sequence encoded as UTF-8
utf8_bytes = text.encode('utf-8')
# b'\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf'

# Pattern 1: Decoded as Latin-1 -> garbled text (most common)
wrong = utf8_bytes.decode('latin-1')
# Typical pattern of interpreting UTF-8 bytes as Latin-1

# Pattern 2: Decoded as Shift_JIS -> different garbled text
wrong2 = utf8_bytes.decode('shift_jis', errors='replace')
# Produces meaningless characters

# Pattern 3: Double encoding (UTF-8 encoded again as UTF-8)
double_encoded = text.encode('utf-8').decode('latin-1').encode('utf-8')
# Requires the reverse operation to recover
recovered = double_encoded.decode('utf-8').encode('latin-1').decode('utf-8')
print(recovered)  # 'konnichiha'

# Correct: decode with the proper encoding
correct = utf8_bytes.decode('utf-8')
# 'konnichiha'
```

### 4.2 Diagnosing Garbled Text

```python
# Automatic encoding diagnosis for garbled text

def diagnose_encoding(broken_text, expected_text=None):
    """Diagnose the encoding of garbled text"""
    # Try common garbled text patterns
    patterns = [
        # (encode_as, decode_as, description)
        ('latin-1', 'utf-8', 'UTF-8 opened as Latin-1'),
        ('cp1252', 'utf-8', 'UTF-8 opened as Windows-1252'),
        ('shift_jis', 'utf-8', 'UTF-8 opened as Shift_JIS'),
        ('utf-8', 'shift_jis', 'Shift_JIS opened as UTF-8'),
        ('utf-8', 'euc-jp', 'EUC-JP opened as UTF-8'),
        ('euc-jp', 'utf-8', 'UTF-8 opened as EUC-JP'),
    ]

    results = []
    for enc, dec, desc in patterns:
        try:
            recovered = broken_text.encode(enc).decode(dec)
            if expected_text and recovered == expected_text:
                results.append((desc, recovered, "MATCH"))
            elif recovered.isprintable() and not any(c == '\ufffd' for c in recovered):
                results.append((desc, recovered, "? Possible"))
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

    return results


# Automatic detection using the chardet library
# pip install chardet
import chardet

def detect_encoding(byte_data):
    """Estimate the encoding of a byte sequence"""
    result = chardet.detect(byte_data)
    return result
    # {'encoding': 'utf-8', 'confidence': 0.99, 'language': ''}

# Usage example
with open('unknown_file.txt', 'rb') as f:
    raw_data = f.read()
    detection = chardet.detect(raw_data)
    print(f"Estimated: {detection['encoding']} (confidence: {detection['confidence']:.2%})")
    text = raw_data.decode(detection['encoding'])
```

### 4.3 Garbled Text Prevention Checklist

```
Unify UTF-8 across all layers:

  +------------------------------+
  | File save: UTF-8 (no BOM)    |
  +------------------------------+
  | HTTP: Content-Type: text/html;|
  |       charset=utf-8          |
  +------------------------------+
  | HTML: <meta charset="utf-8"> |
  +------------------------------+
  | DB: CHARACTER SET utf8mb4    |
  |    (MySQL's utf8 only        |
  |     supports up to 3 bytes!  |
  |     utf8mb4 is true UTF-8)   |
  +------------------------------+
  | Python: open(f, encoding='utf-8') |
  +------------------------------+
  | JSON: UTF-8 by default       |
  +------------------------------+

  Note: MySQL's "utf8" supports only up to 3 bytes (no emoji)
  -> Always use "utf8mb4"!
```

### 4.4 Character Encoding Settings in Various Environments

```python
# === Python ===

# File I/O (always specify encoding explicitly)
with open('file.txt', 'r', encoding='utf-8') as f:
    text = f.read()

with open('file.txt', 'w', encoding='utf-8') as f:
    f.write(text)

# Python 3.15+: default will change to UTF-8
# Python 3.7+: UTF-8 mode
# PYTHONUTF8=1 environment variable or python3 -X utf8

# CSV files (BOM-prefixed UTF-8 required for Excel compatibility)
with open('data.csv', 'w', encoding='utf-8-sig') as f:
    # utf-8-sig = BOM (EF BB BF) + UTF-8
    f.write('Name,Age\n')

# Conversion between byte strings and text strings
text = "Japanese text"
encoded = text.encode('utf-8')     # str -> bytes
decoded = encoded.decode('utf-8')  # bytes -> str

# Error handling
text = b'\xff\xfe'.decode('utf-8', errors='replace')   # '??' (replacement)
text = b'\xff\xfe'.decode('utf-8', errors='ignore')    # '' (ignore)
text = b'\xff\xfe'.decode('utf-8', errors='backslashreplace')  # '\\xff\\xfe'
```

```sql
-- === MySQL ===

-- When creating a database
CREATE DATABASE mydb
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- When creating a table
CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(100) CHARACTER SET utf8mb4
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- At connection time
SET NAMES utf8mb4;
-- Or via connection parameter: charset=utf8mb4

-- Altering an existing table
ALTER TABLE users CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Verification
SHOW VARIABLES LIKE 'character_set%';
SHOW VARIABLES LIKE 'collation%';

-- Difference between utf8 and utf8mb4:
-- utf8:    Max 3 bytes (BMP only, no emoji, U+0000-U+FFFF)
-- utf8mb4: Max 4 bytes (full Unicode support, emoji OK, U+0000-U+10FFFF)
-- -> Always use utf8mb4
```

```javascript
// === JavaScript / Node.js ===

// File I/O
const fs = require('fs');
const text = fs.readFileSync('file.txt', 'utf-8');
fs.writeFileSync('file.txt', text, 'utf-8');

// Buffer operations
const buf = Buffer.from('konnichiha', 'utf-8');
console.log(buf);        // <Buffer e3 81 93 e3 82 93 ...>
console.log(buf.length); // 15 (byte count)
const str = buf.toString('utf-8');

// TextEncoder / TextDecoder (Web API & Node.js)
const encoder = new TextEncoder();  // Default UTF-8
const decoder = new TextDecoder('utf-8');

const encoded = encoder.encode('Hello, World');
const decoded = decoder.decode(encoded);

// Decoding Shift_JIS (Node.js: iconv-lite)
// const iconv = require('iconv-lite');
// const text = iconv.decode(buffer, 'Shift_JIS');

// Character encoding in fetch API
// Response.text() defaults to UTF-8
// Refers to the charset in the Content-Type header
```

```go
// === Go ===

package main

import (
    "fmt"
    "strings"
    "unicode/utf8"

    "golang.org/x/text/encoding/japanese"
    "golang.org/x/text/transform"
)

func main() {
    // Go strings are UTF-8 by default
    s := "konnichiha"
    fmt.Println(len(s))                    // 15 (byte count)
    fmt.Println(utf8.RuneCountInString(s)) // 5 (character count)

    // Iterating by rune (Unicode code point)
    for i, r := range s {
        fmt.Printf("byte[%d]: U+%04X '%c'\n", i, r, r)
    }

    // UTF-8 validity check
    fmt.Println(utf8.ValidString(s))  // true

    // Converting from Shift_JIS to UTF-8
    sjisReader := transform.NewReader(
        strings.NewReader(sjisData),
        japanese.ShiftJIS.NewDecoder(),
    )
    // utf8Data, _ := io.ReadAll(sjisReader)
    _ = sjisReader
}
```

---

## 5. Unicode Pitfalls

### 5.1 Combining Characters and Normalization

```python
# Two ways to represent the Japanese character "ga"

# 1. Precomposed character (NFC): 1 character
ga_nfc = '\u304C'  # 'ga' (1 code point)
len(ga_nfc)  # 1

# 2. Combining character (NFD): base character + combining dakuten
ga_nfd = '\u304B\u3099'  # 'ka' + combining dakuten mark (2 code points)
len(ga_nfd)  # 2

# They look identical as 'ga', but == comparison returns False!
ga_nfc == ga_nfd  # False!

# Solution: normalize with unicodedata.normalize
import unicodedata
unicodedata.normalize('NFC', ga_nfd) == ga_nfc  # True
```

### 5.2 Four Normalization Forms

```python
import unicodedata

# NFC (Canonical Decomposition, followed by Canonical Composition)
# -> Most common. Precomposed form. Recommended for the Web
nfc = unicodedata.normalize('NFC', text)

# NFD (Canonical Decomposition)
# -> Used by macOS file systems (HFS+)
nfd = unicodedata.normalize('NFD', text)

# NFKC (Compatibility Decomposition, followed by Canonical Composition)
# -> Best for search/comparison. Unifies compatibility characters
nfkc = unicodedata.normalize('NFKC', text)

# NFKD (Compatibility Decomposition)
# -> Most decomposed form
nfkd = unicodedata.normalize('NFKD', text)

# NFC vs NFD example:
text = "ga"  # U+304C
nfc = unicodedata.normalize('NFC', text)
nfd = unicodedata.normalize('NFD', text)
print(len(nfc), [hex(ord(c)) for c in nfc])  # 1 ['0x304c']
print(len(nfd), [hex(ord(c)) for c in nfd])  # 2 ['0x304b', '0x3099']

# NFKC vs NFC example:
# Full-width alphanumerics -> half-width alphanumerics
text2 = "\uff21\uff11"  # Full-width 'A' and '1'
nfc2 = unicodedata.normalize('NFC', text2)    # Full-width (unchanged)
nfkc2 = unicodedata.normalize('NFKC', text2)  # "A1" (converted to half-width)
print(nfkc2)  # A1

# Compatibility decomposition examples:
# circled 1 -> 1  (U+2460 -> U+0031)
# kilo sign -> ki + ro  (U+3314 -> U+30AD U+30ED)
# fi ligature -> fi  (U+FB01 -> U+0066 U+0069)
# superscript 2 -> 2   (U+00B2 -> U+0032)

# Practical usage guidelines:
# - Storage/display: NFC (most widely supported)
# - Search/comparison: NFKC (unifies compatibility characters)
# - macOS file names: NFD (OS converts automatically)
```

### 5.3 Surrogate Pairs (A UTF-16 Issue)

```
Code point representation in UTF-16:

  U+0000 - U+FFFF:   Directly as 2 bytes (BMP: Basic Multilingual Plane)
  U+10000 - U+10FFFF: 4 bytes (surrogate pair required)

  Example: grinning face U+1F600
  1. U+1F600 - 0x10000 = 0x0F600
  2. Upper 10 bits: 0x003D -> + 0xD800 = 0xD83D (high surrogate)
  3. Lower 10 bits: 0x0200 -> + 0xDC00 = 0xDE00 (low surrogate)
  4. UTF-16: 0xD83D 0xDE00

  -> This is why JavaScript's string.length returns 2 for emoji
  'grinning face'.length === 2  // true! (UTF-16 internal representation)
  [...'grinning face'].length === 1  // true (iterator works per code point)
```

### 5.4 Grapheme Clusters

```python
# Cases where a single "visible character" consists of multiple code points

# Example 1: Family emoji
family = "family emoji"  # man + woman + girl + boy joined by ZWJ
print(len(family))  # 11 (code point count!)
# Composition: man U+1F468 + ZWJ + woman U+1F469 + ZWJ + girl U+1F467 + ZWJ + boy U+1F466
# ZWJ = Zero Width Joiner (U+200D)

# Example 2: Flag emoji
flag_jp = "JP flag"
print(len(flag_jp))  # 2
# Composition: U+1F1EF (Regional Indicator J) + U+1F1F5 (Regional Indicator P)

# Example 3: Skin tone modifier
wave = "waving hand medium skin tone"
print(len(wave))  # 2
# Composition: U+1F44B (waving hand) + U+1F3FD (skin tone modifier: Medium)

# Example 4: Combining characters (accented letters)
e_acute = "e with acute"  # U+0065 + U+0301 (NFD) or U+00E9 (NFC)

# To correctly count "visible characters":
# Python: regex library (third-party, not the standard re)
import regex  # pip install regex
text = "family emoji + konnichiha"
graphemes = regex.findall(r'\X', text)
print(len(graphemes))  # 6 (1 family + 5 hiragana)

# JavaScript:
# Intl.Segmenter API (ES2022+)
# const segmenter = new Intl.Segmenter('ja', { granularity: 'grapheme' });
# const segments = [...segmenter.segment(text)];
# segments.length; // correct grapheme cluster count

# Go:
# golang.org/x/text/unicode/norm package
# rivo/uniseg package
```

### 5.5 String Length and Indexing

```python
# What "string length" means differs across languages

text = "Hello, sekai! grinning face"

# Python 3: code point count
print(len(text))                    # 11
print(len(text.encode('utf-8')))    # 18 (byte count)

# Correct slicing in Python
# text[7:9] is based on code point units

# Comparison across languages:
# +-------------+----------------------+--------+
# | Language    | Meaning of length    | Value  |
# +-------------+----------------------+--------+
# | Python 3    | Code point count     | 11     |
# | JavaScript  | UTF-16 code units    | 12*    |
# | Java        | UTF-16 code units    | 12*    |
# | Rust (str)  | Byte count           | 18     |
# | Go          | Byte count           | 18     |
# | C (strlen)  | Byte count (to NUL)  | 18     |
# | Swift       | Grapheme clusters    | 11**   |
# +-------------+----------------------+--------+
# * Emoji counts as 2 (surrogate pair)
# ** Swift is the most intuitive but at O(n) cost
```

```javascript
// String length pitfalls in JavaScript

const text = "Hello, sekai! grinning face";

// .length is the UTF-16 code unit count
console.log(text.length);  // 12 (emoji is 2 due to surrogate pair)

// Code point count
console.log([...text].length);  // 11

// Correct string operations
// Bad: text[10] is only the high surrogate of the emoji
console.log(text[10]);  // '\uD83D' (broken character)

// Safe: Array.from() or spread syntax
const chars = [...text];
console.log(chars[10]);  // grinning face emoji

// Safe: codePointAt / String.fromCodePoint
for (const cp of text) {
    console.log(cp.codePointAt(0).toString(16));
}

// Grapheme cluster segmentation (ES2022+)
const segmenter = new Intl.Segmenter('ja', { granularity: 'grapheme' });
const segments = [...segmenter.segment(text)];
console.log(segments.length);  // 11
```

```rust
// String operations in Rust

fn main() {
    let text = "Hello, sekai! grinning face";

    // .len() is the byte count
    println!("{}", text.len());  // 18

    // .chars().count() is the code point count
    println!("{}", text.chars().count());  // 11

    // Iterating by bytes
    for b in text.bytes() {
        print!("{:02X} ", b);
    }
    println!();

    // Iterating by code points
    for c in text.chars() {
        println!("U+{:04X} '{}'", c as u32, c);
    }

    // String slicing panics if not at a character boundary
    // let s = &text[0..7]; // OK (ASCII portion)
    // let s = &text[0..8]; // Panics! Cuts in the middle of UTF-8

    // Safe slicing
    if text.is_char_boundary(7) {
        let s = &text[0..7];
        println!("{}", s);  // "Hello, "
    }

    // char_indices to get byte positions and chars
    for (i, c) in text.char_indices() {
        println!("byte[{}]: U+{:04X} '{}'", i, c as u32, c);
    }
}
```

---

## 6. Special Characters and Control Characters

### 6.1 Invisible Characters

```
Invisible/confusable Unicode characters to watch out for:

  [Zero-width characters]
  U+200B  Zero Width Space
  U+200C  Zero Width Non-Joiner (ZWNJ)
  U+200D  Zero Width Joiner (ZWJ)
  U+FEFF  BOM / Zero Width No-Break Space

  [Directional control characters]
  U+200E  Left-to-Right Mark (LRM)
  U+200F  Right-to-Left Mark (RLM)
  U+202A  Left-to-Right Embedding
  U+202B  Right-to-Left Embedding
  U+202C  Pop Directional Formatting
  U+2066  Left-to-Right Isolate
  U+2067  Right-to-Left Isolate
  U+2069  Pop Directional Isolate

  [Security concerns]
  - Bidi Override (U+202E) reverses text direction
    -> File name spoofing: "document.pdf" might actually appear as "document.exe"
  - Zero-width characters injected into passwords or usernames
    -> Look identical but are different strings
  - Homoglyph attacks: Cyrillic 'a' (U+0430) vs Latin 'a' (U+0061)
    -> "apple.com" (with Cyrillic 'a') is indistinguishable from "apple.com"
```

```python
# Detecting and removing invisible characters

import unicodedata

def detect_invisible_chars(text):
    """Detect and report invisible characters"""
    invisible = []
    for i, char in enumerate(text):
        cat = unicodedata.category(char)
        if cat.startswith('C') and char not in '\n\r\t':
            # C = Control, Cf = Format, Co = Private Use
            invisible.append((i, hex(ord(char)), unicodedata.name(char, '???'), cat))
    return invisible

def remove_invisible_chars(text):
    """Remove invisible characters (preserving newlines and tabs)"""
    return ''.join(
        c for c in text
        if not unicodedata.category(c).startswith('C')
        or c in '\n\r\t'
    )

# Homoglyph detection
def detect_homoglyphs(text):
    """Detect characters that look like Latin letters but belong to different scripts"""
    suspicious = []
    for i, char in enumerate(text):
        if char.isalpha():
            script = unicodedata.name(char, '').split()[0] if unicodedata.name(char, '') else ''
            # Characters that look Latin but are Cyrillic or Greek
            if script in ('CYRILLIC', 'GREEK') and char.lower() in 'abcdefghijklmnopqrstuvwxyz':
                suspicious.append((i, char, hex(ord(char)), script))
    return suspicious


# Unicode category overview
categories = """
L  = Letter
  Lu = Uppercase Letter
  Ll = Lowercase Letter
  Lt = Titlecase Letter
  Lm = Modifier Letter
  Lo = Other Letter (kanji, hiragana, etc.)

M  = Mark (combining characters)
  Mn = Nonspacing Mark (dakuten, etc.)
  Mc = Spacing Combining Mark
  Me = Enclosing Mark

N  = Number
  Nd = Decimal Digit Number (0-9, etc.)
  Nl = Letter Number (Roman numerals, etc.)
  No = Other Number (circled numbers, etc.)

P  = Punctuation
S  = Symbol
Z  = Separator (whitespace)
  Zs = Space Separator
  Zl = Line Separator
  Zp = Paragraph Separator

C  = Other (control characters, etc.)
  Cc = Control
  Cf = Format (ZWJ, BOM, etc.)
  Co = Private Use
  Cs = Surrogate
"""
```

### 6.2 Variation Selectors

```
Variation Selectors:

  The same kanji can have different glyph forms:
  Different stroke counts for the "shinnyou" radical in certain kanji
  Old vs new character forms

  IVS (Ideographic Variation Sequence):
  Base character + variation selector (U+E0100-U+E01EF) specifies the glyph form

  Example:
  U+8FBB + U+E0100 -> "tsuji" (1-dot shinnyou variant)
  U+8FBB + U+E0101 -> "tsuji" (2-dot shinnyou variant)

  Notes:
  - The font must support IVS
  - Most environments display only the default glyph
  - Important for official documents such as family registers and resident cards
```

---

## 7. Character Encoding in Practice

### 7.1 Web Applications

```html
<!-- Character encoding specification in HTML -->
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Must appear within the first 1024 bytes -->
    <meta charset="utf-8">
    <!-- Also specified via the Content-Type header -->
    <!-- Content-Type: text/html; charset=utf-8 -->
    <title>Character Encoding Test</title>
</head>
<body>
    <!-- Default form encoding -->
    <form accept-charset="utf-8" method="post">
        <input type="text" name="name" value="">
        <button type="submit">Submit</button>
    </form>
</body>
</html>
```

```python
# Character encoding handling in Flask

from flask import Flask, request, Response
import json

app = Flask(__name__)

@app.route('/api/data', methods=['POST'])
def handle_data():
    # Decode the request
    # Flask auto-detects charset from Content-Type
    text = request.data.decode('utf-8')

    # JSON response (ensure_ascii=False to output non-ASCII characters directly)
    data = {"message": "Hello", "status": "ok"}
    response = Response(
        json.dumps(data, ensure_ascii=False),
        content_type='application/json; charset=utf-8'
    )
    return response

# For Django:
# settings.py: DEFAULT_CHARSET = 'utf-8' (default)
# FILE_CHARSET = 'utf-8'
```

### 7.2 Database Operations

```python
# Emoji support in MySQL

import mysql.connector

# Specify utf8mb4 at connection time
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='password',
    database='mydb',
    charset='utf8mb4',
    collation='utf8mb4_unicode_ci'
)

cursor = conn.cursor()

# Inserting data containing emoji
cursor.execute(
    "INSERT INTO messages (content) VALUES (%s)",
    ("Hello grinning face party emoji",)
)
conn.commit()

# For PostgreSQL:
# Full UTF-8 support by default
# CREATE DATABASE mydb ENCODING 'UTF8' LC_COLLATE 'en_US.UTF-8';
```

### 7.3 File I/O

```python
# Character encoding handling for CSV files

import csv

# Reading a UTF-8 CSV
with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# Excel-compatible CSV output (BOM-prefixed UTF-8)
with open('output.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age', 'Address'])
    writer.writerow(['Taro Tanaka', '30', 'Tokyo'])

# Reading a Shift_JIS CSV (from a legacy system)
with open('legacy.csv', 'r', encoding='shift_jis') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# Bulk conversion from Shift_JIS to UTF-8
def convert_encoding(input_path, output_path, from_enc='shift_jis', to_enc='utf-8'):
    with open(input_path, 'r', encoding=from_enc) as f_in:
        content = f_in.read()
    with open(output_path, 'w', encoding=to_enc) as f_out:
        f_out.write(content)


# JSON files (always UTF-8)
import json

data = {"name": "Tanaka", "hobbies": ["reading", "grinning face"]}

# ensure_ascii=False is important!
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Result: {"name": "Tanaka", "hobbies": ["reading", "grinning face"]}
# With ensure_ascii=True: {"\u540d\u524d": "\u7530\u4e2d", ...}
```

### 7.4 Character Encoding on the Command Line

```bash
# Determine the character encoding of a file
file -i document.txt
# document.txt: text/plain; charset=utf-8

# Convert using nkf (Network Kanji Filter)
nkf -w input_sjis.txt > output_utf8.txt     # Shift_JIS -> UTF-8
nkf -s input_utf8.txt > output_sjis.txt     # UTF-8 -> Shift_JIS
nkf --guess input.txt                        # Estimate encoding

# Convert using iconv
iconv -f SHIFT_JIS -t UTF-8 input.txt > output.txt

# Python one-liner
python3 -c "
import sys
sys.stdout.buffer.write(
    sys.stdin.buffer.read().decode('shift_jis').encode('utf-8')
)" < input_sjis.txt > output_utf8.txt

# Check byte sequence with hexdump
echo -n "a" | xxd
# 00000000: e381 82                                  ...
# -> E3 81 82 = UTF-8 for hiragana 'a'

# Check locale
locale
# LANG=en_US.UTF-8

# Set character encoding via environment variables
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

---

## 8. Practical Exercises

### Exercise 1: UTF-8 Encoding (Basic)
Calculate the UTF-8 byte sequence for the following characters by hand:
1. 'Z' (U+005A)
2. Yen sign (U+00A5)
3. Kanji 'kan' (U+6F22)
4. Kanji 'tsuchiyoshi' (U+20BB7)

### Exercise 2: Garbled Text Analysis (Intermediate)
Decode the byte sequence `E6 97 A5 E6 9C AC E8 AA 9E` as UTF-8 and determine the original string.

### Exercise 3: Unicode Normalization Implementation (Advanced)
Create a Python function that determines whether two strings are "visually identical" (i.e., match after NFC/NFD normalization).

### Exercise 4: Surrogate Pair Calculation
Convert the following code points to UTF-16 surrogate pairs:
1. U+1F4A9
2. U+1F1EF (Regional Indicator J)
3. U+20000

### Exercise 5: Character Encoding Conversion Tool
Implement a CLI tool in Python with the following features:
- Automatic detection of input file encoding
- Conversion to a specified encoding
- Conversion report output

### Exercise Solutions

```python
# Exercise 1 Solutions

# 'Z' = U+005A
# Binary: 101 1010 (7 bits -> 1 byte)
# UTF-8: 0_1011010 = 0x5A
print('Z'.encode('utf-8').hex())  # 5a

# Yen sign = U+00A5
# Binary: 10 100101 (8 bits -> 2 bytes)
# Split: 00010 100101
# UTF-8: 110_00010 10_100101 = 0xC2 0xA5
print('\u00a5'.encode('utf-8').hex())  # c2a5

# 'kan' = U+6F22
# Binary: 0110 1111 0010 0010 (16 bits -> 3 bytes)
# Split: 0110 111100 100010
# UTF-8: 1110_0110 10_111100 10_100010 = 0xE6 0xBC 0xA2
print('\u6f22'.encode('utf-8').hex())  # e6bca2

# 'tsuchiyoshi' = U+20BB7
# Binary: 0010 0000 1011 1011 0111 (21 bits -> 4 bytes)
# Split: 000 100000 101110 110111
# UTF-8: 11110_000 10_100000 10_101110 10_110111 = 0xF0 0xA0 0xAE 0xB7
print('\U00020bb7'.encode('utf-8').hex())  # f0a0aeb7


# Exercise 2 Solution
bytes_data = bytes([0xE6, 0x97, 0xA5, 0xE6, 0x9C, 0xAC, 0xE8, 0xAA, 0x9E])
result = bytes_data.decode('utf-8')
print(result)  # 'nihongo' (Japanese word meaning "Japanese language")

# Manual decoding:
# E6 97 A5:
#   1110_0110 10_010111 10_100101
#   -> U+65E5 = 'nichi' (sun/day)


# Exercise 3 Solution
import unicodedata

def visual_equal(s1, s2):
    """Determine whether two strings are visually identical"""
    # Compare after NFC normalization
    nfc1 = unicodedata.normalize('NFC', s1)
    nfc2 = unicodedata.normalize('NFC', s2)
    if nfc1 == nfc2:
        return True

    # Also compare after NFKC normalization (absorbs compatibility character differences)
    nfkc1 = unicodedata.normalize('NFKC', s1)
    nfkc2 = unicodedata.normalize('NFKC', s2)
    return nfkc1 == nfkc2

# Test
print(visual_equal('\u304C', '\u304B\u3099'))  # True (ga = ka + dakuten)
print(visual_equal('\uff21', 'A'))              # True (full-width A = half-width A, NFKC)
print(visual_equal('\u3042', '\u30A2'))         # False (hiragana 'a' != katakana 'a')
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

## FAQ

### Q1: Is a BOM (Byte Order Mark) necessary?
**A**: Not needed for UTF-8 (and can even be harmful). It is needed for UTF-16/UTF-32 to determine endianness. A BOM (U+FEFF) at the beginning of a UTF-8 file can cause errors in shell scripts or CSV parsing. BOM-prefixed UTF-8 (`utf-8-sig`) may be required to correctly open CSVs in Excel.

### Q2: How are emoji implemented?
**A**: Standardized in Unicode. Assigned starting from U+1F600. Skin tones are modified via modifiers (U+1F3FB-U+1F3FF). Family emoji are composed by joining multiple emoji with ZWJ (Zero Width Joiner). National flags are represented as pairs of Regional Indicator symbols (e.g., JP = U+1F1EF U+1F1F5).

### Q3: Are non-ASCII file names safe?
**A**: OS-dependent:
- macOS: Forces NFD normalization (file names may take unintended forms)
- Linux: Stored as byte sequences (UTF-8 recommended but not enforced)
- Windows: Stored in UTF-16 (internally an extension of UCS-2)
For maximum safety, use only alphanumeric characters, hyphens, and underscores.

### Q4: Why should you use utf8mb4 instead of utf8 in MySQL?
**A**: MySQL's `utf8` is a proprietary implementation that supports only up to 3 bytes (BMP only, U+0000-U+FFFF). Emoji (U+1F600, etc.) and CJK Extended ideographs (U+20000 and beyond) require 4 bytes, making `utf8mb4` essential. `utf8mb4` conforms to the actual UTF-8 specification.

### Q5: What are CJK Unified Ideographs?
**A**: A Unicode region that unifies kanji characters shared across Chinese, Japanese, and Korean (Han Unification). For example, the character for "sea" has slightly different glyph forms in Chinese, Japanese, and Korean, but is assigned the same code point (U+6D77). This is one of Unicode's most controversial design decisions, and proper font selection is important.

### Q6: What is the correct way to compare strings?
**A**: Depends on the use case:
- Exact match: Normalize to NFC first, then compare byte sequences
- Case-insensitive: Use casefold() (in Python) then compare (more accurate than str.lower())
- Search: Normalize to NFKC to absorb compatibility character differences
- Locale-dependent sorting: Use ICU's Collation
- Security: Also check for homoglyphs and invisible characters

---

## Summary

| Concept | Key Points |
|---------|-----------|
| ASCII | 7-bit, 128 characters. English only. Foundation of everything |
| Unicode | Unifies all languages. Code points U+0000 to U+10FFFF |
| UTF-8 | Variable-length (1-4 bytes). ASCII-compatible. Used by 98%+ of the Web |
| UTF-16 | Variable-length (2 or 4 bytes). Internal representation of Windows/Java/JS |
| UTF-32 | Fixed-length (4 bytes). Simple but memory-inefficient |
| Garbled text | Caused by encoding mismatch. Unify UTF-8 across all layers |
| Normalization | Note the differences between NFC/NFD/NFKC/NFKD. Normalize before comparison |
| Grapheme clusters | 1 "character" does not always equal 1 code point. Prominent with emoji |
| Surrogate pairs | Mechanism for representing code points outside the BMP in UTF-16 |
| Security | Watch out for homoglyphs, invisible characters, and Bidi overrides |

---

## Recommended Next Guides

---

## References
1. Unicode Consortium. "The Unicode Standard." https://unicode.org/
2. Pike, R. & Thompson, K. "Hello World, or Kalimera kosme, or konnichiha sekai." UTF-8 Design, 1992.
3. Spolsky, J. "The Absolute Minimum Every Software Developer Absolutely, Positively Must Know About Unicode and Character Sets." 2003.
4. W3C. "Character encodings for beginners." https://www.w3.org/International/
5. RFC 3629. "UTF-8, a transformation format of ISO 10646." IETF, 2003.
6. Davis, M. & Suignard, M. "Unicode Security Considerations." Unicode Technical Report #36.
7. Unicode Consortium. "Unicode Normalization Forms." Unicode Standard Annex #15.
8. Unicode Consortium. "Unicode Text Segmentation." Unicode Standard Annex #29.
