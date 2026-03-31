# Compression Algorithms

> The essence of data compression is "redundancy elimination," and Shannon's entropy limit defines the theoretical lower bound of compressibility. Huffman coding removes statistical redundancy, LZ-family algorithms remove dictionary-based redundancy, and DEFLATE eliminates both simultaneously. Understanding these principles and selecting the appropriate compression method for each use case is an essential competency for software engineers.

---

## Learning Objectives

- [ ] Explain the theoretical limits of compression based on information entropy and the Shannon limit
- [ ] Understand the mechanism and applicable scenarios of Run-Length Encoding (RLE)
- [ ] Implement Huffman coding from tree construction through encoding and decoding
- [ ] Explain and implement a basic version of LZ77's sliding window approach
- [ ] Understand LZ78/LZW dictionary construction and explain the history of the GIF patent issue
- [ ] Explain how DEFLATE performs two-stage compression (LZ77 + Huffman)
- [ ] Immediately distinguish between and explain the appropriate use of lossless vs. lossy compression
- [ ] Compare characteristics of modern compression algorithms (zstd, Brotli, LZ4)
- [ ] Outline the principles of lossy compression in JPEG / MP3 / H.264
- [ ] Design practical compression strategies for HTTP compression, database compression, etc.

## Prerequisites

- Basic data structures (binary trees, priority queues)

---

## 1. Foundations of Information Theory

### 1.1 Shannon's Entropy

In 1948, Claude Shannon defined the mathematical concept of "information content" in data in his paper "A Mathematical Theory of Communication." This concept is the theoretical foundation of data compression and serves as the benchmark for evaluating the performance of all compression algorithms.

#### Definition of Entropy

When an information source X outputs n symbols x_1, x_2, ..., x_n with probabilities p(x_1), p(x_2), ..., p(x_n), the information entropy H(X) is defined by the following formula:

```
  Information Entropy H(X):

    H(X) = - Σ p(x_i) × log₂(p(x_i))     [unit: bits/symbol]
            i=1..n

  Intuitive meaning:
    - H(X) represents "how much information is needed to predict the next symbol"
    - Low entropy → easy to predict → high redundancy → highly compressible
    - High entropy → hard to predict → close to random → hard to compress
```

#### Concrete Calculation Examples

```
  Example 1: Fair coin toss (heads 50%, tails 50%)

    H = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5))
    H = -(0.5 × (-1) + 0.5 × (-1))
    H = -(-0.5 + (-0.5))
    H = 1.0 bits/symbol

    → At least 1 bit is needed to represent the result (this matches intuition)

  Example 2: Biased coin (heads 90%, tails 10%)

    H = -(0.9 × log₂(0.9) + 0.1 × log₂(0.1))
    H = -(0.9 × (-0.152) + 0.1 × (-3.322))
    H ≈ -((-0.137) + (-0.332))
    H ≈ 0.469 bits/symbol

    → Can be represented in less than 1 bit! Bias = redundancy = compressible

  Example 3: 4-symbol source (A:50%, B:25%, C:12.5%, D:12.5%)

    H = -(0.5 × log₂(0.5) + 0.25 × log₂(0.25)
         + 0.125 × log₂(0.125) + 0.125 × log₂(0.125))
    H = -(0.5 × (-1) + 0.25 × (-2) + 0.125 × (-3) + 0.125 × (-3))
    H = -(-0.5 + (-0.5) + (-0.375) + (-0.375))
    H = 1.75 bits/symbol

    → Fixed-length coding requires 2 bits, but entropy is 1.75 bits
    → With Huffman coding A=0, B=10, C=110, D=111:
       Average code length = 0.5×1 + 0.25×2 + 0.125×3 + 0.125×3 = 1.75 bits
       → Exactly matches the entropy! (Huffman coding is optimal in this case)
```

#### Shannon Limit (Source Coding Theorem)

```
  Shannon's Source Coding Theorem (First Theorem):

    For any distortion-free (lossless) encoding,
    the average code length L per symbol must be at least the entropy H.

      L ≥ H(X)

    Moreover, there exists an encoding that achieves an average code length
    arbitrarily close to H(X).

      H(X) ≤ L < H(X) + 1

    Meaning:
    - No algorithm, no matter how sophisticated, can losslessly compress below the entropy
    - Algorithms that achieve compression rates close to the entropy do exist
    - This is the "theoretical limit of compression"

  Practical implications:
    ┌──────────────────────────────────────────────────────────┐
    │  Data Type              │ Entropy      │ Compressibility  │
    ├──────────────────────────────────────────────────────────┤
    │  Natural language text  │ ~1-2 bit/char│ High (2-5x)      │
    │  Source code            │ ~2-3 bit/char│ Moderate (2-4x)  │
    │  Encrypted data         │ ≈8 bit/byte  │ Nearly impossible │
    │  Already compressed     │ ≈8 bit/byte  │ Nearly impossible │
    │  Random byte sequence   │ 8 bit/byte   │ Impossible        │
    └──────────────────────────────────────────────────────────┘
```

#### Entropy Calculation in Python

```python
"""
Information Entropy Calculation

Calculates the information entropy of given data based on Shannon's
entropy formula. This enables us to determine the theoretical compression limit.
"""
import math
from collections import Counter
from typing import Union


def calculate_entropy(data: Union[str, bytes]) -> float:
    """Calculate Shannon's entropy.

    Args:
        data: Data to analyze (string or bytes)

    Returns:
        Entropy value in bits/symbol

    Raises:
        ValueError: If empty data is provided

    >>> calculate_entropy("AAAA")
    0.0
    >>> round(calculate_entropy("AB" * 50), 2)
    1.0
    >>> round(calculate_entropy("ABRACADABRA"), 2)
    2.04
    """
    if not data:
        raise ValueError("Data is empty")

    total = len(data)
    freq = Counter(data)

    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def analyze_compressibility(data: Union[str, bytes]) -> dict:
    """Analyze the compressibility of data.

    Computes the theoretical minimum size and upper bound of the
    compression ratio based on entropy.

    Args:
        data: Data to analyze

    Returns:
        Dictionary containing analysis results
    """
    entropy = calculate_entropy(data)
    total = len(data)

    # Theoretical minimum number of bits
    min_bits = entropy * total
    min_bytes = math.ceil(min_bits / 8)

    # Bits per symbol (for fixed-length encoding)
    unique_symbols = len(set(data))
    fixed_bits_per_symbol = math.ceil(math.log2(unique_symbols)) if unique_symbols > 1 else 1
    fixed_total_bits = fixed_bits_per_symbol * total

    return {
        "length": total,
        "unique_symbols": unique_symbols,
        "entropy_per_symbol": round(entropy, 4),
        "fixed_bits_per_symbol": fixed_bits_per_symbol,
        "theoretical_min_bits": round(min_bits, 2),
        "theoretical_min_bytes": min_bytes,
        "fixed_length_bits": fixed_total_bits,
        "max_compression_ratio": round(fixed_total_bits / min_bits, 2) if min_bits > 0 else float('inf'),
    }


# Usage examples
if __name__ == "__main__":
    samples = [
        ("Uniform distribution", "ABCD" * 25),
        ("Highly biased", "A" * 90 + "B" * 10),
        ("Natural language style", "the quick brown fox jumps over the lazy dog"),
        ("Maximum redundancy", "AAAAAAAAAA"),
    ]

    for label, data in samples:
        result = analyze_compressibility(data)
        print(f"\n--- {label}: '{data[:30]}...' ---")
        print(f"  Length: {result['length']} symbols")
        print(f"  Unique: {result['unique_symbols']} types")
        print(f"  Entropy: {result['entropy_per_symbol']} bit/symbol")
        print(f"  Theoretical min: {result['theoretical_min_bits']} bits "
              f"({result['theoretical_min_bytes']} bytes)")
        print(f"  Fixed length: {result['fixed_length_bits']} bits")
        print(f"  Max compression ratio: {result['max_compression_ratio']}x")
```

### 1.2 Self-Information and Conditional Entropy

To deepen the understanding of entropy, here are some related concepts.

```
  Self-Information:

    When an event x occurs, its information content is defined as:

      I(x) = -log₂(p(x))   [unit: bits]

    Intuition: rarer events carry more information
      - p(x) = 1 (certain to occur)  → I(x) = 0 bits (no information)
      - p(x) = 0.5                   → I(x) = 1 bit
      - p(x) = 0.01 (rare)           → I(x) ≈ 6.64 bits (high information content)

    Entropy is the expected value of self-information:
      H(X) = E[I(X)] = Σ p(x) × I(x)

  Conditional Entropy H(Y|X):

    The uncertainty of Y given knowledge of X.
    In data compression, the more "context" is exploited,
    the lower the conditional entropy becomes, enabling
    more efficient compression.

    Example: In English text
      - H(next character) ≈ 4.0 bit/char (no context)
      - H(next character | previous 1 character) ≈ 3.3 bit/char
      - H(next character | previous 2 characters) ≈ 3.0 bit/char
      ...
      - Shannon's estimate: true entropy of English ≈ 1.0-1.5 bit/char

    → High-compression algorithms like PPM and LZMA exploit long contexts
      to approach the conditional entropy
```

### 1.3 Lossless vs. Lossy Compression

Data compression falls into two broad categories. This distinction is fundamental to the nature of compression and serves as the starting point for understanding all compression techniques.

```
  ┌────────────────────────────────────────────────────────────────┐
  │                 Classification of Data Compression              │
  ├─────────────────────────┬──────────────────────────────────────┤
  │ Lossless Compression    │ Lossy Compression                    │
  ├─────────────────────────┼──────────────────────────────────────┤
  │ Original data is fully  │ An approximation of the original     │
  │ recoverable             │ data is recovered                    │
  │ Not a single bit is lost│ Information imperceptible to humans  │
  │                         │ is removed                           │
  │                         │                                      │
  │ Theoretical limit:      │ Limit:                               │
  │  Shannon limit          │  Depends on acceptable quality       │
  │                         │  (Rate-Distortion Theory)            │
  │                         │                                      │
  │ Key algorithms:         │ Key algorithms:                      │
  │  - Huffman coding       │  - DCT (JPEG, MPEG)                 │
  │  - Arithmetic coding    │  - Wavelet (JPEG 2000)               │
  │  - LZ77 / LZ78 / LZW   │  - Psychoacoustic model (MP3, AAC)   │
  │  - DEFLATE              │  - Motion compensation (H.264,       │
  │  - BWT + MTF            │    H.265, AV1)                       │
  │                         │  - Quantization                      │
  │                         │                                      │
  │ Use cases:              │ Use cases:                           │
  │  - Text/source code     │  - Photos/images (JPEG, WebP, AVIF) │
  │  - Executable files     │  - Music (MP3, AAC, Opus)            │
  │  - Database backups     │  - Video (H.264, H.265, AV1)        │
  │  - Archives (ZIP, etc.) │  - Voice calls (Speex, Opus)         │
  │  - Network transfer     │                                      │
  │                         │                                      │
  │ Compression ratio:      │ Compression ratio:                   │
  │  ~2-5x                  │  ~10-1000x or more                   │
  │ (depends on data)       │ (depends on acceptable quality)      │
  └─────────────────────────┴──────────────────────────────────────┘
```

#### When Lossless Compression Is Essential

For data such as text files and program binaries, where even a single bit change alters the meaning, lossless compression must be used. When distributing a program via a ZIP archive, if the extracted file differs by even one byte from the original, it will not function correctly.

#### When Lossy Compression Is Acceptable

For images, audio, and video, where human perceptual characteristics can be exploited to remove "imperceptible information" without practical issues, lossy compression achieves extremely high compression ratios. When comparing a JPEG quality 75 image with an uncompressed image, it is difficult to visually distinguish the difference in most cases.

---

## 2. Lossless Compression Algorithms

### 2.1 Run-Length Encoding (RLE)

RLE is the simplest data compression algorithm, representing consecutive runs of the same value as a pair of "value + repeat count."

#### Basic Principle

```
  How RLE works:

  Input:  AAAAAABBCCCCDDDDDDDD
         ~~~~~~^^~~~~^^^^^^^^
         A×6   B×2 C×4  D×8

  Output: A6B2C4D8

  Compression ratio: 20 characters → 8 characters = 60% reduction

  ─────────────────────────────────────
  A more realistic example (binary data):

  Input bytes: FF FF FF FF FF 00 00 00 00 00 00 00 FF FF
                ~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~  ~~~~~
                   FF × 5          00 × 7            FF × 2

  RLE output: (FF, 5) (00, 7) (FF, 2)
              6 bytes (3 pairs × 2 bytes)

  Original: 14 bytes → RLE: 6 bytes = 57% reduction
```

#### Weakness of RLE

```
  Cases where RLE backfires:

  Input: ABCDEFGHIJ
  Output: A1B1C1D1E1F1G1H1I1J1

  Original: 10 characters → RLE: 20 characters = 100% expansion!

  Countermeasures:
  1. Introduce a literal mode
     - Output non-repeating portions as-is (distinguished by mode flag)
     - Example: [LIT:10]ABCDEFGHIJ  (10 literal characters)
               [RUN:6]A             (6-count run)

  2. Selective application
     - Use only for data with frequent long runs of identical values
     - FAX data, images with large single-color regions, blank spreadsheet cells, etc.
```

#### RLE Implementation in Python

```python
"""
Run-Length Encoding (RLE) Implementation

Compresses consecutive identical values into (value, count) pairs.
The simplest compression algorithm, used for educational purposes
and specific applications (FAX, BMP).
"""
from typing import Union


def rle_encode(data: Union[str, bytes]) -> list[tuple]:
    """Encode data using RLE.

    Args:
        data: Input data (string or bytes)

    Returns:
        List of (value, consecutive count) tuples

    >>> rle_encode("AAABBBCCCCDD")
    [('A', 3), ('B', 3), ('C', 4), ('D', 2)]
    >>> rle_encode("ABCD")
    [('A', 1), ('B', 1), ('C', 1), ('D', 1)]
    >>> rle_encode("")
    []
    """
    if not data:
        return []

    result = []
    current = data[0]
    count = 1

    for i in range(1, len(data)):
        if data[i] == current:
            count += 1
        else:
            result.append((current, count))
            current = data[i]
            count = 1

    result.append((current, count))
    return result


def rle_decode(encoded: list[tuple]) -> str:
    """Decode RLE-encoded data.

    Args:
        encoded: List of (value, consecutive count) tuples

    Returns:
        Decoded string

    >>> rle_decode([('A', 3), ('B', 2), ('C', 4)])
    'AAABBCCCC'
    """
    return "".join(char * count for char, count in encoded)


def rle_compression_ratio(data: str) -> dict:
    """Analyze the efficiency of RLE compression.

    Args:
        data: Input string

    Returns:
        Analysis results of compression efficiency
    """
    encoded = rle_encode(data)
    # Estimate each pair as "character + digit count of the repeat number"
    encoded_size = sum(1 + len(str(count)) for _, count in encoded)
    original_size = len(data)

    return {
        "original_size": original_size,
        "encoded_pairs": len(encoded),
        "encoded_size_estimate": encoded_size,
        "compression_ratio": round(original_size / encoded_size, 2) if encoded_size > 0 else 0,
        "effective": encoded_size < original_size,
    }


# Usage examples
if __name__ == "__main__":
    test_cases = [
        "AAAAAABBCCCCDDDDDDDD",        # Good for RLE
        "ABCDEFGHIJKLMNOP",             # Bad for RLE
        "WWWWWWWWWWWWBWWWWWWWWWWWWBBB",  # Mixed
        "A" * 1000,                     # Extreme repetition
    ]

    for data in test_cases:
        result = rle_compression_ratio(data)
        status = "Effective" if result["effective"] else "Counterproductive"
        print(f"Input: '{data[:40]}...' ({result['original_size']} chars)")
        print(f"  → {result['encoded_pairs']} pairs, "
              f"Estimated size: {result['encoded_size_estimate']} chars, "
              f"Ratio: {result['compression_ratio']}x [{status}]")
```

### 2.2 Huffman Coding

Published in 1952 by MIT graduate student David Huffman, this algorithm constructs the optimal prefix-free code among variable-length codes. By assigning shorter codes to more frequent symbols and longer codes to less frequent ones, it minimizes the overall code length.

#### Properties of Huffman Coding

```
  Importance of Prefix-Free Codes:

    No codeword is a prefix of any other codeword.
    → Uniquely decodable without delimiters (instantaneously decodable code)

    Example: A=0, B=10, C=110, D=111

      Coded sequence: 01011001111100
      Decoding:       0|10|110|0|111|110|0
                      A  B  C   A  D   C  A  ← uniquely determined

    Counterexample (not prefix-free):
      A=0, B=01, C=1  ← A's code "0" is a prefix of B's code "01"!
      Coded sequence: 01 → is it A,C or B? Ambiguous
```

#### Huffman Tree Construction Procedure

```
  Input: "ABRACADABRA" (11 characters)

  Step 1: Count character frequencies
  ┌──────┬──────┐
  │ Char │ Freq │
  ├──────┼──────┤
  │  A   │  5   │
  │  B   │  2   │
  │  R   │  2   │
  │  C   │  1   │
  │  D   │  1   │
  └──────┴──────┘

  Step 2: Merge from lowest frequency upward (using a priority queue)

    Initial state (sorted by frequency): C:1  D:1  B:2  R:2  A:5

    Iteration 1: Merge C(1) and D(1) → CD
                 Remaining: B:2  [CD]:2  R:2  A:5

    Iteration 2: Merge B(2) and CD → B,CD
                 Remaining: R:2  [B,CD]:4  A:5

    Iteration 3: Merge R(2) and B,CD → R,B,CD
                 Remaining: A:5  [R,B,CD]:6

    Iteration 4: Merge A(5) and R,B,CD → A,R,B,CD
                 Root node complete

  Step 3: Completed Huffman tree

              [11]
             /    \
           A(5)   [6]
                 /    \
              R(2)   [4]
                    /    \
                 B(2)   [2]
                       /    \
                    C(1)   D(1)

  Step 4: Code assignment (left branch = 0, right branch = 1)

    A: 0        (1 bit)   ← Most frequent → shortest code
    R: 10       (2 bits)
    B: 110      (3 bits)
    C: 1110     (4 bits)
    D: 1111     (4 bits)  ← Least frequent → longest code

  Step 5: Encoding

    A  B   R  A  C    A  D    A  B   R  A
    0  110 10 0  1110 0  1111 0  110 10 0

    = 0 110 10 0 1110 0 1111 0 110 10 0
    = 011010011100111101101 00
    = 23 bits

  Result:
    Original:       11 characters × 8 bits = 88 bits
    Huffman coded:  23 bits (+ tree information must be stored as a header)
    Code-only compression ratio: 88 / 23 ≈ 3.83x

  Average code length calculation:
    L = 5/11 × 1 + 2/11 × 2 + 2/11 × 3 + 1/11 × 4 + 1/11 × 4
      = 5/11 + 4/11 + 6/11 + 4/11 + 4/11
      = 23/11
      ≈ 2.09 bits/symbol

  Comparison with entropy:
    H = -(5/11×log₂(5/11) + 2/11×log₂(2/11) + 2/11×log₂(2/11)
         + 1/11×log₂(1/11) + 1/11×log₂(1/11))
    H ≈ 2.04 bits/symbol

    → L = 2.09 ≥ H = 2.04 (satisfies the Shannon limit)
    → The difference is only 0.05 bits/symbol (very efficient)
```

#### Complete Huffman Coding Implementation in Python

```python
"""
Complete Huffman Coding Implementation

Constructs a Huffman tree, encodes, and decodes data.
Uses a priority queue (heapq) to efficiently build the optimal prefix-free code.
"""
import heapq
from collections import Counter
from typing import Optional


class HuffmanNode:
    """A node in the Huffman tree."""

    def __init__(self, char: Optional[str], freq: int,
                 left: Optional['HuffmanNode'] = None,
                 right: Optional['HuffmanNode'] = None):
        self.char = char    # Only leaf nodes hold a character
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: 'HuffmanNode') -> bool:
        """For comparison in the priority queue."""
        return self.freq < other.freq

    def is_leaf(self) -> bool:
        return self.char is not None


class HuffmanCoding:
    """Huffman encoding/decoding class."""

    def __init__(self):
        self.root: Optional[HuffmanNode] = None
        self.codes: dict[str, str] = {}
        self.reverse_codes: dict[str, str] = {}

    def build_tree(self, text: str) -> HuffmanNode:
        """Build a Huffman tree from text.

        Args:
            text: Input text

        Returns:
            Root node of the Huffman tree

        Raises:
            ValueError: If empty text is provided
        """
        if not text:
            raise ValueError("Text is empty")

        freq = Counter(text)

        # Special handling for single-character input
        if len(freq) == 1:
            char = list(freq.keys())[0]
            self.root = HuffmanNode(char=None, freq=freq[char],
                                    left=HuffmanNode(char, freq[char]))
            self._generate_codes(self.root, "")
            return self.root

        # Add leaf nodes to the priority queue
        heap: list[HuffmanNode] = []
        for char, count in freq.items():
            heapq.heappush(heap, HuffmanNode(char, count))

        # Merge nodes from lowest frequency to build the Huffman tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(
                char=None,
                freq=left.freq + right.freq,
                left=left,
                right=right,
            )
            heapq.heappush(heap, merged)

        self.root = heap[0]
        self._generate_codes(self.root, "")
        return self.root

    def _generate_codes(self, node: Optional[HuffmanNode], code: str) -> None:
        """Traverse the Huffman tree to generate the code table."""
        if node is None:
            return

        if node.is_leaf():
            # Fallback when the tree has only one node
            self.codes[node.char] = code if code else "0"
            self.reverse_codes[code if code else "0"] = node.char
            return

        self._generate_codes(node.left, code + "0")
        self._generate_codes(node.right, code + "1")

    def encode(self, text: str) -> str:
        """Encode text using Huffman coding.

        Args:
            text: Input text

        Returns:
            A string of '0's and '1's representing the bit sequence

        >>> hc = HuffmanCoding()
        >>> hc.build_tree("ABRACADABRA")  # doctest: +ELLIPSIS
        <...>
        >>> encoded = hc.encode("ABRACADABRA")
        >>> len(encoded)  # 23 bits
        23
        """
        if not self.codes:
            self.build_tree(text)
        return "".join(self.codes[char] for char in text)

    def decode(self, encoded: str) -> str:
        """Decode a Huffman-coded bit sequence.

        Args:
            encoded: Bit string consisting of '0's and '1's

        Returns:
            Decoded text
        """
        if self.root is None:
            raise ValueError("Huffman tree has not been built")

        result = []
        node = self.root

        for bit in encoded:
            node = node.left if bit == "0" else node.right

            if node.is_leaf():
                result.append(node.char)
                node = self.root

        return "".join(result)

    def get_code_table(self) -> dict[str, dict]:
        """Return the code table with statistics."""
        return {
            char: {
                "code": code,
                "length": len(code),
            }
            for char, code in sorted(self.codes.items(), key=lambda x: len(x[1]))
        }

    def print_tree(self, node: Optional[HuffmanNode] = None,
                   prefix: str = "", is_left: bool = True) -> None:
        """Visually display the Huffman tree."""
        if node is None:
            node = self.root
        if node is None:
            return

        connector = "├── " if is_left else "└── "
        label = f"'{node.char}' ({node.freq})" if node.is_leaf() else f"[{node.freq}]"

        if prefix == "":
            print(label)
        else:
            print(prefix + connector + label)

        new_prefix = prefix + ("│   " if is_left else "    ")

        if not node.is_leaf():
            self.print_tree(node.left, new_prefix, True)
            self.print_tree(node.right, new_prefix, False)


# Usage example and verification
if __name__ == "__main__":
    text = "ABRACADABRA"
    print(f"Input text: '{text}' ({len(text)} chars, {len(text)*8} bits)")
    print()

    hc = HuffmanCoding()
    hc.build_tree(text)

    print("=== Huffman Tree ===")
    hc.print_tree()
    print()

    print("=== Code Table ===")
    for char, info in hc.get_code_table().items():
        print(f"  '{char}': {info['code']} ({info['length']} bits)")
    print()

    encoded = hc.encode(text)
    print(f"=== Encoding Result ===")
    print(f"  Bit sequence: {encoded}")
    print(f"  Length: {len(encoded)} bits ({len(encoded)/8:.1f} bytes)")
    print(f"  Compression ratio: {len(text)*8/len(encoded):.2f}x")
    print()

    decoded = hc.decode(encoded)
    print(f"=== Decoding Result ===")
    print(f"  Decoded text: '{decoded}'")
    print(f"  Matches original: {decoded == text}")
```

### 2.3 Arithmetic Coding

Huffman coding assigns integer-bit-length codewords to each symbol, so when symbol probabilities are not powers of 2, there can be up to 1 bit/symbol difference from the entropy. Arithmetic coding overcomes this limitation and achieves compression rates closer to the entropy.

```
  Basic idea of arithmetic coding:

    Encode the entire message as a single real number within the interval [0, 1).

    Example: Symbol probabilities A=0.6, B=0.2, C=0.2  Message: "BAC"

    Step 1: Initial interval [0.0, 1.0)
      A → [0.0, 0.6)
      B → [0.6, 0.8)
      C → [0.8, 1.0)

    Step 2: First symbol B → select interval [0.6, 0.8)
      Subdivide this interval:
      A → [0.6, 0.72)     (0.6 + 0.6×0.2 = 0.72)
      B → [0.72, 0.76)
      C → [0.76, 0.8)

    Step 3: Next symbol A → select interval [0.6, 0.72)
      Subdivide this interval:
      A → [0.6, 0.672)
      B → [0.672, 0.696)
      C → [0.696, 0.72)

    Step 4: Next symbol C → select interval [0.696, 0.72)

    Result: "BAC" can be represented by any value in [0.696, 0.72)
            e.g., 0.7 → binary 0.1011... → minimizes required bits

  Comparison with Huffman coding:
    ┌──────────────────┬────────────────────┬──────────────────────┐
    │ Property         │ Huffman Coding     │ Arithmetic Coding    │
    ├──────────────────┼────────────────────┼──────────────────────┤
    │ Coding unit      │ Per symbol         │ Entire message       │
    │ Min code length  │ 1 bit/symbol       │ No theoretical limit │
    │ Gap from entropy │ Up to 1 bit/symbol │ Within 2 bits (total)│
    │ Adaptive support │ Requires tree      │ Only probability     │
    │                  │ reconstruction     │ updates needed       │
    │ Computational    │ Low                │ Slightly higher      │
    │ cost             │                    │                      │
    │ Patent issues    │ None               │ Existed (now expired)│
    │ Usage            │ DEFLATE, JPEG      │ JPEG2000, H.264/265 │
    └──────────────────┴────────────────────┴──────────────────────┘

  ANS (Asymmetric Numeral Systems):
    A new entropy coding method published by Jarek Duda in 2009.
    Achieves compression rates equivalent to arithmetic coding at
    speeds comparable to Huffman coding.
    Adopted in Zstandard, LZFSE (Apple), CRAM (bioinformatics).
```

### 2.4 LZ77 (Lempel-Ziv 1977)

Published in 1977 by Abraham Lempel and Jacob Ziv, LZ77 is a pioneer of dictionary-based compression and serves as the foundation for many modern compression algorithms (DEFLATE, zstd, LZ4, etc.).

#### Sliding Window Mechanism

```
  Basic idea of LZ77:
    Replace previously seen string patterns with (distance, length) references.
    Uses a sliding window (search buffer + lookahead buffer).

  ┌─────────────────────────────────────────┐
  │          Sliding Window                 │
  │  ┌──────────────────┬──────────────┐    │
  │  │  Search Buffer    │ Lookahead    │    │
  │  │  (already         │ Buffer       │    │
  │  │   processed)      │ (to process) │    │
  │  │  ← 32KB etc. →   │ ← 258B etc.→ │    │
  │  └──────────────────┴──────────────┘    │
  │         ↑ Find longest match ↑ Current  │
  └─────────────────────────────────────────┘

  Output format:
    Match found → (distance, length)  Distance and length reference
    No match    → literal(byte)       Literal byte

  ─────────────────────────────────────────
  Example: Input "AABCAABCAABC"

  Processing flow:
  Position 0: 'A' → no match in search buffer → literal 'A'
  Position 1: 'A' → match at distance 1, length 1 → (1,1)
              Note: short matches may be less efficient than literals
  Position 2: 'B' → no match → literal 'B'
  Position 3: 'C' → no match → literal 'C'
  Position 4: 'AABCAABC' → matches "AABC" starting at position 0
              → (4, 4) ← distance 4, length 4
  Position 8: 'AABC' → matches "AABC" starting at position 4
              → (4, 4)

  Output: lit(A), lit(A), lit(B), lit(C), (4,4), (4,4)

  ─────────────────────────────────────────
  Self-reference example (an important property of LZ77):

  Input: "ABABABABABAB"

  Position 0: lit(A)
  Position 1: lit(B)
  Position 2: "ABABABABAB" → matches "AB" at position 0, but
              the lookahead buffer can extend the match beyond the search buffer!
              → (2, 10) ← distance 2, length 10

  During expansion: expanding (2,10) at position 2
    → Reference position 0 → copy A (write to position 2)
    → Reference position 1 → copy B (write to position 3)
    → Reference position 2 → copy A (write to position 4) ← the A just written!
    → ... repeat

  This way "ABABABABABAB" can be represented with just 3 tokens:
  lit(A), lit(B), (2,10).
```

#### LZ77 Implementation in Python

```python
"""
LZ77 Compression Algorithm Implementation

Compresses data by replacing previously seen patterns with
(distance, length) references using a sliding window approach.
The foundational algorithm behind DEFLATE, zstd, and others.
"""
from dataclasses import dataclass
from typing import Union


@dataclass
class LZ77Token:
    """An LZ77 output token."""
    is_literal: bool
    value: Union[str, None] = None      # Character for literal tokens
    distance: int = 0                    # Distance for reference tokens
    length: int = 0                      # Length for reference tokens

    def __repr__(self) -> str:
        if self.is_literal:
            return f"lit('{self.value}')"
        return f"({self.distance},{self.length})"


class LZ77:
    """LZ77 encoder/decoder.

    Args:
        window_size: Search buffer size (default: 4096)
        min_match: Minimum match length to output as a reference (default: 3)
    """

    def __init__(self, window_size: int = 4096, min_match: int = 3):
        self.window_size = window_size
        self.min_match = min_match

    def encode(self, data: str) -> list[LZ77Token]:
        """Encode data using LZ77.

        Args:
            data: Input string

        Returns:
            List of LZ77Tokens
        """
        tokens = []
        pos = 0

        while pos < len(data):
            best_distance = 0
            best_length = 0

            # Start position of search buffer
            search_start = max(0, pos - self.window_size)

            # Find the longest match in the search buffer
            for i in range(search_start, pos):
                match_length = 0
                # Allow self-references by comparing one character at a time
                while (pos + match_length < len(data) and
                       match_length < 258):  # Maximum match length limit
                    # Self-reference: cyclic reference within distance
                    ref_pos = i + match_length
                    if ref_pos >= pos:
                        ref_pos = i + (match_length % (pos - i))
                    if data[ref_pos] == data[pos + match_length]:
                        match_length += 1
                    else:
                        break

                if match_length > best_length:
                    best_length = match_length
                    best_distance = pos - i

            if best_length >= self.min_match:
                tokens.append(LZ77Token(
                    is_literal=False,
                    distance=best_distance,
                    length=best_length,
                ))
                pos += best_length
            else:
                tokens.append(LZ77Token(
                    is_literal=True,
                    value=data[pos],
                ))
                pos += 1

        return tokens

    def decode(self, tokens: list[LZ77Token]) -> str:
        """Decode an LZ77 token sequence.

        Args:
            tokens: List of LZ77Tokens

        Returns:
            Restored string
        """
        result = []

        for token in tokens:
            if token.is_literal:
                result.append(token.value)
            else:
                # Expand reference (one character at a time for self-reference support)
                start = len(result) - token.distance
                for i in range(token.length):
                    result.append(result[start + i])

        return "".join(result)


# Usage examples
if __name__ == "__main__":
    encoder = LZ77(window_size=256, min_match=3)

    test_cases = [
        "ABCABCABCABC",
        "AABCAABCAABC",
        "the cat sat on the mat",
        "ABABABABABABABABABAB",
    ]

    for data in test_cases:
        tokens = encoder.encode(data)
        decoded = encoder.decode(tokens)

        print(f"Input: '{data}'")
        print(f"  Tokens: {tokens}")
        print(f"  Token count: {len(tokens)} (original: {len(data)} chars)")
        print(f"  Decoded: '{decoded}'")
        print(f"  Match: {decoded == data}")
        print()
```

### 2.5 LZ78 / LZW

While LZ77 uses a sliding window, LZ78 (1978) takes an approach that builds an explicit dictionary. LZW (Lempel-Ziv-Welch, 1984) is an improved version widely used in the GIF image format.

```
  LZ78 dictionary construction:

    Input: "ABABABCABC"

    Dictionary (initial state: empty):

    Step | Input | In dict?     | Output      | Add to dict
    ─────┼───────┼──────────────┼─────────────┼──────────
    1    | A     | No           | (0, 'A')    | 1: "A"
    2    | B     | No           | (0, 'B')    | 2: "B"
    3    | AB    | Yes→No(AB)   | (1, 'B')    | 3: "AB"
    4    | A     | Yes          |             |
    4    | AB    | Yes          |             |
    4    | ABC   | No           | (3, 'C')    | 4: "ABC"
    5    | A     | Yes          |             |
    5    | AB    | Yes          |             |
    5    | ABC   | Yes→(done)   | (4, '')     | 5: "ABCA"

    Output: (0,A) (0,B) (1,B) (3,C) (4,EOF)

  LZW improvements:
    - Initialize the dictionary with all symbols (0-255)
    - Output consists only of dictionary indices (no characters included)
    - Enables simpler and faster implementation

  ─────────────────────────────────────────
  LZW GIF patent issue:

    1985: Unisys (formerly Sperry) obtains the LZW patent
    1994: It becomes known that CompuServe's GIF uses LZW
          → Unisys demands licensing fees
    1995: PNG format development begins (using DEFLATE instead of LZW)
    2003: US patent expires
    2004: Worldwide patent expiration

    Lesson: A prime example of software patents impacting standard technologies
            → Importance of open standards and royalty-free licensing

  LZ77 vs LZ78 comparison:
    ┌────────────┬──────────────────┬──────────────────┐
    │ Property   │ LZ77             │ LZ78             │
    ├────────────┼──────────────────┼──────────────────┤
    │ Dict type  │ Implicit (window)│ Explicit (table) │
    │ Memory use │ Proportional to  │ Proportional to  │
    │            │ window size      │ dictionary size  │
    │ Compression│ Generally good   │ Roughly equal    │
    │ ratio      │                  │ to LZ77          │
    │ Decode     │ Fast             │ Fast             │
    │ speed      │                  │                  │
    │ Typical    │ DEFLATE, zstd    │ GIF (LZW)        │
    │ application│                  │                  │
    │ Patent     │ None             │ LZW had past     │
    │ issues     │                  │ issues           │
    └────────────┴──────────────────┴──────────────────┘
```

### 2.6 DEFLATE Algorithm

DEFLATE is an algorithm published by Phil Katz in 1993 that performs two-stage compression combining LZ77 and Huffman coding. It is used extremely widely in ZIP, gzip, zlib, PNG, HTTP compression, and more, making it one of the most important compression algorithms of the internet era.

```
  DEFLATE processing flow:

  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
  │ Input    │ → │ LZ77         │ → │ Huffman     │ → │ Compressed│
  │ Data     │    │ Compression  │    │ Encoding    │    │ Data     │
  └──────────┘    │              │    │             │    └──────────┘
                  │ Convert to   │    │ Further     │
                  │ literals and │    │ compress    │
                  │ (dist,len)   │    │ literals/   │
                  │ tokens       │    │ lengths/    │
                  │              │    │ distances   │
                  │              │    │ with Huffman│
                  └──────────────┘    └─────────────┘

  Details:

  Stage 1 - LZ77:
    - Search buffer: up to 32KB
    - Lookahead buffer: up to 258 bytes
    - Match length: 3-258 bytes
    - Output: literal bytes (0-255) and match references (distance, length)

  Stage 2 - Huffman coding:
    DEFLATE uses two independent Huffman trees:
    1. Literal/length tree (symbols 0-285)
       - 0-255: literal bytes
       - 256: end-of-block marker
       - 257-285: match lengths (corresponding to ranges 3-258)
    2. Distance tree (symbols 0-29)
       - Each symbol corresponds to a distance range

  Block structure:
    DEFLATE data consists of multiple blocks.
    Each block is one of three types:

    Type 0: Uncompressed block (stores data as-is)
    Type 1: Fixed Huffman codes (uses predefined Huffman trees)
    Type 2: Dynamic Huffman codes (generates block-specific Huffman trees)

    Type 2 is most efficient for typical data, but due to
    the header overhead of the Huffman tree itself,
    Type 1 may be more efficient for small blocks.

  ─────────────────────────────────────────
  Formats that use DEFLATE:

    ┌──────────┬───────────────────────────────────────────┐
    │ Format   │ Description                               │
    ├──────────┼───────────────────────────────────────────┤
    │ ZIP      │ Archive + DEFLATE (multi-file support)    │
    │ gzip     │ DEFLATE + CRC32 + header (single stream) │
    │ zlib     │ DEFLATE + Adler-32 (library format)      │
    │ PNG      │ Filtering + zlib (for images)             │
    │ HTTP     │ Content-Encoding: gzip / deflate          │
    │ PDF      │ FlateDecode filter                        │
    │ JAR/WAR  │ Java archives (ZIP format based)          │
    │ DOCX/XLSX│ Office Open XML (ZIP format based)        │
    └──────────┴───────────────────────────────────────────┘
```

### 2.7 Burrows-Wheeler Transform (BWT)

BWT is not a compression algorithm itself, but a reversible data rearrangement transform that dramatically improves the efficiency of subsequent compression (MTF + entropy coding). It is the core technology behind bzip2.

```
  How BWT works:

  Input: "BANANA$" ($ is the terminator)

  Step 1: Generate all cyclic rotations
    BANANA$
    ANANA$B
    NANA$BA
    ANA$BAN
    NA$BANA
    A$BANAN
    $BANANA

  Step 2: Sort lexicographically
    $BANANA     → last character: A
    A$BANAN     → last character: N
    ANA$BAN     → last character: N
    ANANA$B     → last character: B
    BANANA$     → last character: $ ← position of original string (row 4)
    NA$BANA     → last character: A
    NANA$BA     → last character: A

  Step 3: Extract the last column
    BWT output: "ANNB$AA" (+ original position: 4)

  Effect: Same characters tend to cluster together!
    → "ANNB$AA" has 'A' concentrated nearby
    → Extremely good compatibility with Move-to-Front (MTF) transform
    → Subsequent entropy coding becomes highly efficient

  bzip2 compression pipeline:
    Input → BWT → MTF → RLE → Huffman coding → Output
```

### 2.8 Comparison of Modern Compression Algorithms

```
  Evolution and characteristics of algorithms:

  ┌────────────┬──────┬────────┬────────┬────────┬──────────────────┐
  │ Name       │ Year │ Ratio  │Compress│Decomp  │ Primary use      │
  │            │      │(high=  │ speed  │ speed  │                  │
  │            │      │ good)  │(MB/s)  │(MB/s)  │                  │
  ├────────────┼──────┼────────┼────────┼────────┼──────────────────┤
  │ gzip       │ 1992 │ ★★★   │ 36     │ 410    │ HTTP, general    │
  │ bzip2      │ 1996 │ ★★★★  │ 14     │ 56     │ Archives         │
  │ LZ4        │ 2011 │ ★★    │ 780    │ 4,560  │ DB, real-time    │
  │ Snappy     │ 2011 │ ★★    │ 565    │ 1,800  │ BigTable, Kafka  │
  │ Brotli     │ 2015 │ ★★★★★ │ 5      │ 430    │ HTTP (Google)    │
  │ zstd       │ 2016 │ ★★★★  │ 510    │ 1,380  │ General (Meta)   │
  │ LZMA2      │ 2001 │ ★★★★★ │ 3      │ 120    │ 7z, xz           │
  └────────────┴──────┴────────┴────────┴────────┴──────────────────┘

  Speed vs. compression ratio tradeoff diagram:

  Compression ratio (high)
    ↑
    │  ×LZMA2
    │  ×Brotli
    │       ×bzip2
    │         ×zstd
    │    ×gzip
    │
    │                    ×Snappy
    │                     ×LZ4
    │
    └──────────────────────────────→ Compression speed (fast)

  Zstandard (zstd) features:
    - Developed by Yann Collet at Meta (formerly Facebook)
    - Flexibly adjustable speed/ratio via compression levels 1-22
    - Level 1: gzip-beating ratio at LZ4-like speed
    - Level 19: Approaches LZMA compression ratio
    - Dictionary compression: Can apply pre-built dictionaries to small data (JSON, etc.)
    - Standardized in RFC 8478, RFC 8878

  Brotli features:
    - Developed by Google for web delivery
    - Built-in static dictionary (common HTML/CSS/JS patterns)
    - 20-26% smaller output than gzip for web content
    - Standardized in RFC 7932, supported by all major browsers
    - Ideal for CDN pre-compression (slow to compress but fast to decompress)

  Selection guidelines:
  ┌──────────────────────┬──────────────────────────────┐
  │ Use case             │ Recommended algorithm        │
  ├──────────────────────┼──────────────────────────────┤
  │ DB internals/message │ LZ4 or Snappy (ultra-low     │
  │ queues               │ latency)                     │
  │ File systems         │ zstd (best balance)          │
  │ Web static delivery  │ Brotli (highest ratio + CDN) │
  │ Web dynamic compress │ gzip or zstd (speed +        │
  │                      │ compatibility)               │
  │ Archival storage     │ LZMA2/7z (highest ratio)     │
  │ Log compression      │ zstd (dictionary compression │
  │                      │ is effective)                │
  │ Legacy compatibility │ gzip (most widely supported) │
  └──────────────────────┴──────────────────────────────┘
```

---

## 3. Lossy Compression

Lossy compression exploits human perceptual characteristics to remove imperceptible information, achieving compression ratios far exceeding lossless compression.

### 3.1 Image Compression

#### JPEG Compression in Detail

```
  JPEG compression pipeline:

  ┌──────┐  ┌────────┐  ┌─────┐  ┌──────┐  ┌──────────┐  ┌──────┐
  │ RGB  │→│ YCbCr  │→│Down- │→│ DCT  │→│ Quanti-  │→│Entro-│
  │Image │  │Convert │  │samp- │  │Trans-│  │ zation   │  │ py   │
  │      │  │        │  │ling  │  │(8×8) │  │(info loss)│  │Coding│
  └──────┘  └────────┘  └─────┘  └──────┘  └──────────┘  └──────┘

  Details of each step:

  1. Color space conversion (RGB → YCbCr):
     Y  = Luminance (brightness)      → Human eyes are most sensitive to this
     Cb = Blue chrominance (color diff) → Lower sensitivity
     Cr = Red chrominance (color diff)  → Lower sensitivity

     Formula:
       Y  =  0.299R + 0.587G + 0.114B
       Cb = -0.169R - 0.331G + 0.500B + 128
       Cr =  0.500R - 0.419G - 0.081B + 128

  2. Chroma subsampling (downsampling):
     4:4:4 → No chrominance thinning (highest quality)
     4:2:2 → Halve chrominance horizontally (broadcast quality)
     4:2:0 → Halve chrominance both horizontally and vertically (JPEG standard)
             → Data volume is half of 4:4:4

  3. DCT (Discrete Cosine Transform):
     Transform each 8×8 pixel block to the frequency domain
     → Top-left: DC component (block average luminance)
     → Moving toward bottom-right: higher frequency components (finer patterns)

     Before DCT (spatial domain)    After DCT (frequency domain)
     ┌─┬─┬─┬─┬─┬─┬─┬─┐      ┌──┬──┬──┬──┬──┬──┬──┬──┐
     │p│i│x│e│l│ │l│u│      │DC│lo│lo│ :│ :│ :│ :│hi│
     ├─┼─┼─┼─┼─┼─┼─┼─┤      ├──┼──┼──┼──┼──┼──┼──┼──┤
     │m│i│n│a│n│c│e│ │      │lo│ :│ :│ :│ :│ :│ :│ :│
     │v│a│l│u│e│s│ │a│      │ :│ :│ :│ :│ :│ :│ :│ :│
     │r│r│a│n│g│e│d│ │      │ :│ :│ :│ :│ :│ :│ :│ :│
     │s│p│a│t│i│a│l│y│      │hi│ :│ :│ :│ :│ :│ :│hi│
     └─┴─┴─┴─┴─┴─┴─┴─┘      └──┴──┴──┴──┴──┴──┴──┴──┘

  4. Quantization (where information is lost):
     Divide DCT coefficients by a quantization table and round.
     Lower quality settings → divide by larger values → high frequencies become 0 → more info loss

     Before quantization:     Quantization table (Q50):  After quantization:
     [120  -5  3  1]       [16  11  10  16]        [8  0  0  0]
     [ -8   2 -1  0]  ÷    [12  12  14  19]   =    [0  0  0  0]
     [  3  -1  0  0]       [14  13  16  24]        [0  0  0  0]
     [  0   0  0  0]       [14  17  22  29]        [0  0  0  0]
     (simplified 4×4 example)

     → Produces many zeros → zigzag scan → efficiently compressed with RLE

  5. Entropy coding:
     - Zigzag scan (scan in order from low to high frequency)
     - DC component: Huffman code the difference from the previous block
     - AC components: Huffman code zero-runs + non-zero coefficients

  JPEG quality vs. compression ratio (1920×1080 photo):
    ┌────────┬──────────┬──────────┬──────────────────────┐
    │Quality │ Size     │ Ratio    │ Visual quality        │
    ├────────┼──────────┼──────────┼──────────────────────┤
    │ 100    │ ~2.0 MB  │ 3:1      │ Nearly lossless       │
    │ 95     │ ~800 KB  │ 7.5:1    │ Difference barely     │
    │        │          │          │ perceptible           │
    │ 85     │ ~500 KB  │ 12:1     │ Sufficient for web    │
    │ 75     │ ~300 KB  │ 20:1     │ Common default        │
    │ 50     │ ~150 KB  │ 40:1     │ Slight degradation    │
    │        │          │          │ visible               │
    │ 25     │ ~80 KB   │ 75:1     │ Block noise prominent │
    │ 10     │ ~50 KB   │ 120:1    │ Significant           │
    │        │          │          │ degradation           │
    └────────┴──────────┴──────────┴──────────────────────┘
```

#### Comprehensive Image Format Comparison

```
  ┌────────┬────────┬───────┬────────┬──────────┬────────────────────┐
  │ Format │ Compr. │ Alpha │ Anim.  │ Color    │ Primary use        │
  ├────────┼────────┼───────┼────────┼──────────┼────────────────────┤
  │ JPEG   │ Lossy  │ ×     │ ×      │ 24bit    │ Photos in general  │
  │ PNG    │ Lossless│ α    │ APNG   │ 48bit+α  │ UI, screenshots    │
  │ GIF    │ Lossless│ 1bit │ ○      │ 8bit     │ Simple animations  │
  │        │        │       │        │          │ (256 colors)       │
  │ WebP   │ Both   │ α     │ ○      │ 24bit+α  │ Web images (Google)│
  │ AVIF   │ Lossy  │ α     │ ○      │ 36bit+α  │ Next-gen web images│
  │ JPEG XL│ Both   │ α     │ ○      │ 32bit+α  │ JPEG successor     │
  │ HEIF   │ Lossy  │ α     │ ○      │ 30bit    │ Apple Photos       │
  │ SVG    │ -      │ ○     │ CSS    │ -        │ Vector graphics    │
  │ TIFF   │ Both   │ α     │ ×      │ 64bit    │ Print, medical     │
  │ BMP    │None/RLE│ ×     │ ×      │ 32bit    │ Windows (not       │
  │        │        │       │        │          │ recommended)       │
  └────────┴────────┴───────┴────────┴──────────┴────────────────────┘

  Next-gen format compression efficiency (vs. JPEG):
    WebP:    ~25-35% smaller (equivalent quality)
    AVIF:    ~50% smaller (equivalent quality)
    JPEG XL: ~60% smaller + lossless transcoding from JPEG is possible
```

### 3.2 Audio Compression

```
  Audio data basics:

    Uncompressed CD quality (PCM):
      Sampling rate: 44,100 Hz
      Quantization depth: 16 bit
      Channels: 2 (stereo)
      Bitrate: 44,100 × 16 × 2 = 1,411,200 bps ≈ 1.4 Mbps
      1 minute: ~10 MB

  Psychoacoustic Model:

    Exploits human hearing characteristics to remove inaudible sounds.

    1. Absolute threshold of hearing:
       Minimum audible sound pressure varies by frequency.
       1-5 kHz is most sensitive; below 20 Hz and above 20 kHz are inaudible.

    2. Frequency masking:
       Weak sounds near a strong sound's frequency band are not perceived.
       Example: With a 60 dB sound at 1 kHz,
               a 40 dB sound at 1.1 kHz is inaudible → can be removed

    3. Temporal masking:
       Quiet sounds immediately before/after a loud sound are not perceived.
       Forward masking: ~50-200 ms
       Backward masking: ~5-20 ms

  Audio codec comparison:
    ┌──────────┬──────┬────────────┬────────────┬────────────────┐
    │ Codec    │ Year │ Rec. rate  │ Quality    │ Primary use    │
    ├──────────┼──────┼────────────┼────────────┼────────────────┤
    │ MP3      │ 1993 │128-320kbps │ Standard-  │ Music (legacy) │
    │          │      │            │ High       │                │
    │ AAC      │ 1997 │ 96-256kbps │ High       │ Apple, YouTube │
    │ Vorbis   │ 2000 │ 96-320kbps │ High       │ Games, Web     │
    │ Opus     │ 2012 │ 64-256kbps │ Highest    │ VoIP, Web,     │
    │          │      │            │            │ all-purpose    │
    │ FLAC     │ 2001 │800-1200kbps│ Lossless   │ Audio archival │
    │ ALAC     │ 2004 │800-1200kbps│ Lossless   │ Apple Music    │
    └──────────┴──────┴────────────┴────────────┴────────────────┘

  Notable Opus characteristics:
    - Supports 6 kbps (voice calls) to 510 kbps (high-quality music)
    - Ultra-low latency: minimum 2.5 ms (ideal for real-time communication)
    - Hybrid of SILK (voice-oriented) and CELT (music-oriented)
    - IETF standard (RFC 6716), completely royalty-free
    - Adopted by Discord, WebRTC, Signal, etc.
```

### 3.3 Video Compression

```
  Basic concepts of video compression:

    Uncompressed 1080p 30fps:
      1920 × 1080 × 3 bytes × 30 fps = 186,624,000 B/s
      ≈ 186 MB/s ≈ 1.49 Gbps
      → 1 minute = ~11.2 GB ← practically unusable

  Inter-frame prediction:

    ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
    │  I   │   │  P   │   │  B   │   │  B   │   │  P   │
    │Frame │   │Frame │   │Frame │   │Frame │   │Frame │
    │      │──→│      │   │      │   │      │──→│      │
    │(full)│   │(diff)│   │(bidir)│  │(bidir)│  │(diff)│
    └──────┘   └──────┘   └──────┘   └──────┘   └──────┘
         ↑──────────────────↑──↑──────────↑
               Reference relationships

    I frame (Intra):
      - Self-contained image (compression similar to JPEG)
      - Random access entry point
      - Largest size (typically 3-10x that of P frames)

    P frame (Predicted):
      - Stores only the "difference" from the previous I/P frame
      - Motion compensation: "This 16×16 block moved by (dx, dy)"
      - Residual: Compress post-motion-compensation differences with DCT + quantization

    B frame (Bi-predictive):
      - Predicted from both previous and next frames (bidirectional)
      - Smallest size (50-70% of P frames)
      - Requires preceding and following frames for decoding → introduces latency

  Video codec evolution:
    ┌──────────┬──────┬────────────┬──────────────┬──────────────┐
    │ Codec    │ Year │ Block size │ vs. MPEG-2   │ Notes        │
    ├──────────┼──────┼────────────┼──────────────┼──────────────┤
    │ MPEG-2   │ 1995 │ 16×16      │ (baseline)   │ DVD, TV      │
    │ H.264    │ 2003 │ 4×4-16×16  │ 50% reduction│ Blu-ray, Web │
    │ VP9      │ 2013 │ 4×4-64×64  │ 30% less than│ YouTube      │
    │          │      │            │ H.264        │              │
    │ H.265    │ 2013 │ 4×4-64×64  │ 75% reduction│ 4K/8K, Apple │
    │ AV1      │ 2018 │4×4-128×128 │ 75-80%       │ YouTube,     │
    │          │      │            │ reduction    │ Netflix      │
    │ VVC      │ 2020 │4×4-128×128 │ 85% reduction│ 8K, next-gen │
    │          │      │            │              │ broadcast    │
    └──────────┴──────┴────────────┴──────────────┴──────────────┘

  Significance of AV1:
    - Developed by the Alliance for Open Media (AOMedia)
      → Google, Netflix, Amazon, Apple, Meta, Microsoft, etc. participate
    - Compression ratio equal to or better than H.265 (HEVC)
    - Completely royalty-free (resolves HEVC's expensive licensing issues)
    - Hardware decoder adoption is progressing
    - Gradually expanding adoption in YouTube, Netflix, Twitch, etc.
```

---

## 4. Compression in Practice

### 4.1 HTTP Compression

Reducing transfer size is one of the most effective optimizations for web performance, and HTTP compression is at its core.

```
  HTTP compression negotiation:

  Browser                                   Server
  │                                          │
  │  GET /index.html HTTP/1.1                │
  │  Accept-Encoding: gzip, deflate, br      │
  │ ────────────────────────────────────────→ │
  │                                          │
  │  HTTP/1.1 200 OK                         │
  │  Content-Encoding: br                    │
  │  Content-Type: text/html                 │
  │  Vary: Accept-Encoding                  │
  │  [Brotli-compressed response body]       │
  │ ←──────────────────────────────────────── │
  │                                          │

  Compression effectiveness by Content-Type:
    ┌──────────────────────┬──────────┬──────────────────────────┐
    │ Content-Type         │ Reduction│ Reason                   │
    ├──────────────────────┼──────────┼──────────────────────────┤
    │ text/html            │ 70-85%   │ Many repeated tags       │
    │ text/css             │ 75-85%   │ Regular structure        │
    │ application/javascript│ 70-80%  │ Repeated variable names  │
    │                      │          │ and syntax               │
    │ application/json     │ 75-90%   │ Many repeated key names  │
    │ image/jpeg           │ 0-2%     │ Already lossy compressed │
    │ image/png            │ 0-5%     │ Already DEFLATE          │
    │                      │          │ compressed               │
    │ font/woff2           │ 0-1%     │ Already Brotli compressed│
    │ image/svg+xml        │ 60-75%   │ XML text, so effective   │
    └──────────────────────┴──────────┴──────────────────────────┘

  Nginx compression configuration example:
    # gzip settings
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;           # 1-9, 6 is a good balance
    gzip_min_length 256;         # Don't compress below 256 bytes
    gzip_types
        text/plain
        text/css
        text/xml
        application/json
        application/javascript
        application/xml
        image/svg+xml;

    # Images, videos, and fonts are excluded (already compressed)

  Brotli vs gzip usage guidelines:
    - Static files (CDN delivery): Brotli level 11 (pre-compressed)
    - Dynamic responses (API): gzip level 6 or zstd (speed-focused)
    - Legacy browser support: Use gzip as a fallback alongside Brotli
```

### 4.2 Database Compression

```
  Compression features of major databases:

  PostgreSQL - TOAST (The Oversized-Attribute Storage Technique):
    - Automatically compresses column data exceeding 2KB using LZ-family
    - LZ4 also available from PostgreSQL 14+
    - Operates transparently without user awareness
    - ALTER TABLE t ALTER COLUMN c SET COMPRESSION lz4;

  MySQL InnoDB:
    - Page compression (ROW_FORMAT=COMPRESSED)
    - Transparent page compression (innodb_compression_algorithm)
    - Punch-hole compression (filesystem-level)

  Column-oriented database compression:
    Same-type data aligned in columns yields very high compression efficiency.

    ┌─────────────────────────────────────────────────┐
    │ Row-oriented (MySQL, PostgreSQL):                │
    │ [Name:Alice, Age:30, City:Tokyo]                │
    │ [Name:Bob,   Age:25, City:Osaka]                │
    │ [Name:Carol, Age:30, City:Tokyo]                │
    │ → Mixed types → hard to compress                │
    │                                                  │
    │ Column-oriented (ClickHouse, Parquet):           │
    │ Name: [Alice, Bob, Carol, ...]   → Dictionary   │
    │ Age:  [30, 25, 30, ...]          → Delta + RLE  │
    │ City: [Tokyo, Osaka, Tokyo, ...] → Dictionary   │
    │ → Same-type data is contiguous → very effective  │
    └─────────────────────────────────────────────────┘

    Key column-oriented compression techniques:
    - Dictionary Encoding:
      Effective for low-cardinality columns (gender, country names, etc.)
      ["Tokyo","Osaka","Tokyo","Tokyo"] → dict{0:"Tokyo",1:"Osaka"} + [0,1,0,0]

    - Delta Encoding:
      Effective for sequential values like timestamps
      [1000, 1001, 1003, 1005] → [1000, +1, +2, +2]

    - Bit Packing:
      When the value range is narrow, store with the minimum number of bits
      [1, 3, 2, 0, 3] → each value in 2 bits (not 8 bits)

    - RLE (Run-Length Encoding):
      Effective when a sorted column has consecutive identical values
      [Tokyo, Tokyo, Tokyo, Osaka, Osaka] → [(Tokyo,3), (Osaka,2)]
```

### 4.3 Leveraging Python Compression Libraries

```python
"""
Comparison of Python Standard Library and Third-Party Compression

Usage and performance comparison of commonly used compression libraries
in practice.
"""
import gzip
import zlib
import bz2
import lzma
import time
import os
from typing import Callable


def compress_benchmark(data: bytes,
                       compressors: dict[str, Callable]) -> list[dict]:
    """Run benchmarks for multiple compression algorithms.

    Args:
        data: Byte sequence to compress
        compressors: {name: compression function} dictionary

    Returns:
        List of results for each algorithm
    """
    results = []
    original_size = len(data)

    for name, compress_func in compressors.items():
        # Compress
        start = time.perf_counter()
        compressed = compress_func(data)
        compress_time = time.perf_counter() - start

        compressed_size = len(compressed)
        ratio = original_size / compressed_size if compressed_size > 0 else 0

        results.append({
            "name": name,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": round(ratio, 2),
            "reduction_pct": round((1 - compressed_size / original_size) * 100, 1),
            "compress_time_ms": round(compress_time * 1000, 2),
        })

    return sorted(results, key=lambda x: x["compressed_size"])


# Define compression functions
compressors = {
    "zlib (level 6)":   lambda d: zlib.compress(d, 6),
    "zlib (level 9)":   lambda d: zlib.compress(d, 9),
    "gzip (level 6)":   lambda d: gzip.compress(d, compresslevel=6),
    "gzip (level 9)":   lambda d: gzip.compress(d, compresslevel=9),
    "bz2 (level 9)":    lambda d: bz2.compress(d, compresslevel=9),
    "lzma (preset 6)":  lambda d: lzma.compress(d, preset=6),
}

# Generate test data
test_data = b"Hello, World! " * 10000  # Highly repetitive data

results = compress_benchmark(test_data, compressors)
print(f"Original data: {len(test_data):,} bytes")
print(f"{'Algorithm':<20} {'Compressed':>10} {'Ratio':>8} {'Reduced':>8} {'Time':>10}")
print("-" * 60)
for r in results:
    print(f"{r['name']:<20} {r['compressed_size']:>10,} {r['ratio']:>7.1f}x "
          f"{r['reduction_pct']:>6.1f}% {r['compress_time_ms']:>8.2f}ms")
```

---

## 5. Anti-Patterns and Design Considerations

### Anti-Pattern 1: The Double Compression Trap

```
  Problem:
    Attempting to further compress already-compressed data.
    This yields virtually no improvement in compression ratio
    and only wastes CPU time.

  Typical mistakes:
    ┌──────────────────────────────────────────────────────┐
    │ × ZIP-compressing JPEG images then gzip-transferring │
    │   → JPEG is already compressed. ZIP/gzip has no effect│
    │                                                      │
    │ × Applying Content-Encoding: gzip to woff2 fonts     │
    │   → woff2 is internally Brotli-compressed            │
    │                                                      │
    │ × Compressing encrypted data                         │
    │   → Encrypted data is near-random and incompressible │
    │   → Moreover, compression ratio variations can leak  │
    │     cryptographic information (CRIME/BREACH attacks)  │
    └──────────────────────────────────────────────────────┘

  Correct approach:
    ┌──────────────────────────────────────────────────────┐
    │ ○ Compress then encrypt (order matters!)             │
    │   → Raw data → compress → encrypt → transfer        │
    │                                                      │
    │ ○ Control compression application/exemption by       │
    │   Content-Type                                       │
    │   → Text types: compress                             │
    │   → Images/video/fonts: do not compress              │
    │                                                      │
    │ ○ Combining different principles IS effective        │
    │   → LZ77 + Huffman = DEFLATE (dictionary +          │
    │     statistical fusion)                              │
    │   → BWT + MTF + Huffman = bzip2                     │
    └──────────────────────────────────────────────────────┘
```

### Anti-Pattern 2: Maximum Compression Level Obsession

```
  Problem:
    Always selecting the highest compression level without
    considering processing speed. Ignoring the speed/ratio tradeoff.

  Specific examples:
    ┌──────────────────────────────────────────────────────┐
    │ × Applying Brotli level 11 to API responses          │
    │   → Compression takes hundreds of milliseconds,      │
    │     degrading latency                                │
    │   → Level 4-6 provides sufficient compression        │
    │                                                      │
    │ × Using bzip2 for a real-time database               │
    │   → Decompression is too slow, degrading read        │
    │     latency                                          │
    │   → LZ4 or Snappy is appropriate                     │
    │                                                      │
    │ × Applying high-compression algorithms to small data │
    │   (< 1KB)                                            │
    │   → Header overhead actually increases size          │
    │   → Set a minimum like gzip_min_length 256;          │
    └──────────────────────────────────────────────────────┘

  Correct approach:
    ┌──────────────────────────────────────────────────────┐
    │ Select compression level based on use case:          │
    │                                                      │
    │ Real-time (DB, message queues):                      │
    │   → LZ4 / Snappy / zstd level 1-3                   │
    │   → Target < 1ms for both compress and decompress    │
    │                                                      │
    │ Online (API, web delivery):                          │
    │   → gzip level 4-6 / Brotli level 4-6               │
    │   → Target compression time < 10ms                   │
    │                                                      │
    │ Offline (archives, backups):                          │
    │   → LZMA / Brotli level 11 / zstd level 19+         │
    │   → Compression time is acceptable; pursue maximum   │
    │     compression ratio                                │
    └──────────────────────────────────────────────────────┘
```

### Anti-Pattern 3: Ignoring CRIME / BREACH Attacks

```
  Problem:
    Keeping compression enabled over TLS while returning responses
    that include user input.

  Attack principle:
    1. Attacker injects a guessed string into the request
    2. Server includes a secret (CSRF token, etc.) in the response
    3. Response is compressed, encrypted, and returned
    4. If the guess is correct → matches the secret → compression ratio increases → size decreases
    5. Attacker identifies the secret one character at a time from size changes

  Countermeasures:
    - Disable TLS-level compression (currently disabled by default)
    - Use HTTP compression in conjunction with SameSite cookies + CSRF tokens
    - Consider disabling compression for responses containing security tokens
```

---

## 6. Practical Exercises

### Exercise 1 (Basic): Manual Huffman Coding

Manually construct a Huffman code for the following string.

```
  Task: For "MISSISSIPPI" (11 characters):

  (a) Count the frequency of each character
  (b) Build the Huffman tree (show the construction process in a diagram)
  (c) Determine the codeword for each character
  (d) Calculate the total number of bits after encoding "MISSISSIPPI"
  (e) Compare with the entropy and evaluate the efficiency of the Huffman code
  (f) Compare with fixed-length coding (assigning equal bits to each character)

  Hint:
    Character frequencies: I=4, S=4, P=2, M=1
    Number of unique characters: 4 → fixed-length requires 2 bits/char
```

### Exercise 2 (Applied): Implementing and Comparing Compression Algorithms

```
  Task: Implement the following programs in Python.

  (a) Implement an RLE encoder/decoder and verify with these test cases:
      - "AAABBBCCCDDD" → effective
      - "ABCDEFGH"     → counterproductive
      - Binary data (bytes type support)

  (b) Implement a Huffman coding encoder/decoder:
      - Display the code table built from "ABRACADABRA"
      - Verify correctness of encoding and decoding
      - Calculate the difference from entropy

  (c) Compress the following files using Python standard library compression
      functions and compare:
      - A text file (10KB or more)
      - Artificially generated repetitive data
      - Random byte sequence (os.urandom)
      Summarize compression ratios and processing times in a table.
```

### Exercise 3 (Advanced): Understanding LZ77 and DEFLATE

```
  Task:

  (a) Implement an LZ77 encoder/decoder.
      - Sliding window size should be configurable (default 4096)
      - Minimum match length is 3
      - Correctly handle self-references (distance < length)
      Test: "ABCABCABCABC" → token sequence → decode → matches original

  (b) Apply Huffman coding to the above LZ77 output to create a program
      that simulates DEFLATE's basic two-stage compression.

  (c) For the following data, measure compression ratios with LZ77 window
      sizes of 64, 256, 1024, and 4096, and plot the relationship between
      window size and compression ratio:
      - Text data (highly repetitive)
      - Text data (low repetition)
      - Random data

  Advanced:
  (d) Implement BWT (Burrows-Wheeler Transform) and recreate bzip2's
      basic structure with the pipeline:
      BWT → MTF → RLE → Huffman
```

---

## 7. FAQ

### Q1: Can already-compressed data be compressed further?

**A**: In principle, this is nearly impossible. Compressed data is in a state where redundancy has been removed, and its entropy is close to the limit (near-random). Compressing a JPEG image with ZIP produces virtually no change in file size, or may even slightly increase it.

However, **applying different-principle compressions in stages** can be effective. Just as DEFLATE combines LZ77 (dictionary-based) and Huffman coding (statistics-based), designs that remove different types of redundancy in stages are valid.

### Q2: What is the difference between ZIP, gzip, and tar.gz?

**A**: Each has a different design philosophy.

| Format | Archive | Compression | Individual file access | Culture |
|--------|---------|-------------|----------------------|---------|
| ZIP | Yes (multi-file) | Yes (DEFLATE) | Possible (each file compressed independently) | Windows |
| gzip | No (single stream) | Yes (DEFLATE) | N/A | Unix |
| tar | Yes (multi-file) | No | Sequential read within tar | Unix |
| tar.gz | Yes (via tar) | Yes (entire archive compressed at once) | No (must decompress first) | Unix |
| tar.zst | Yes (via tar) | Yes (zstd) | Possible with seekable zstd | Modern |

ZIP compresses each file independently, enabling fast individual access but unable to exploit inter-file redundancy. tar.gz concatenates all files first and compresses them as a whole, achieving higher compression ratios but requiring full decompression to access files in the middle.

### Q3: Why doesn't ZIP compression reduce video file sizes?

**A**: Video files (MP4, MKV, etc.) already have advanced lossy compression applied (H.264 / H.265 / AV1, etc.). They are typically compressed 100-1000x from the original uncompressed video data, leaving extremely little residual redundancy. It is fundamentally difficult for lossless compression algorithms like ZIP or gzip to find additional redundancy.

For the same reason, double compression is ineffective on already-compressed data such as JPEG images, MP3 audio, and woff2 fonts. The correct design for HTTP server compression settings is to exclude these Content-Types from compression targets.

### Q4: What characteristics make data particularly amenable to compression?

**A**: Compression efficiency directly correlates with data redundancy. Data with the following characteristics can expect high compression ratios:

1. **Abundant repetitive patterns**: Log files (many lines in the same format), HTML/XML (tag repetition), source code (syntax pattern repetition)
2. **Biased symbol frequency distribution**: Natural language text (high frequency of 'e', 'the', etc.)
3. **Strong local correlation**: Image data (similarity of adjacent pixels), time-series data (gradual value changes)

Conversely, encrypted data, already-compressed data, and random byte sequences are inherently resistant to compression.

### Q5: How should compression be designed for web services?

**A**: The following multi-layered approach is recommended:

1. **Static assets**: Pre-compress with Brotli (level 11) at build time and deliver via CDN. Also generate gzip for fallback.
2. **Dynamic API responses**: Online compress with gzip level 4-6 or zstd. Brotli is not suitable for dynamic use due to slow compression.
3. **Images**: Use WebP / AVIF as appropriate. Implement format fallback with the `<picture>` element.
4. **Database**: Leverage TOAST (PostgreSQL) or InnoDB compression. Use column compression in column-oriented DBs.
5. **Message queues / cache**: Low-latency compression with LZ4 or Snappy.

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is most important. Understanding deepens not just through theory but by actually writing code and verifying behavior.

### Q2: What common mistakes do beginners make?

Skipping fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before proceeding to the next step.

### Q3: How is this knowledge applied in practice?

This topic's knowledge is frequently applied in daily development work. It is particularly important during code reviews and architecture design.

---

## 8. Summary

### Knowledge Organization Table

| Concept | Core Point |
|---------|-----------|
| Entropy | A measure of information content in data. Defines the theoretical lower bound of lossless compression |
| Shannon limit | No lossless compression, no matter how sophisticated, can compress below the entropy |
| RLE | Represents consecutive identical values as (value, count). Simplest but limited in applicability |
| Huffman coding | Optimal prefix-free code based on frequency. Used internally in DEFLATE, JPEG, etc. |
| Arithmetic coding | Represents the entire message as a single number. More efficient than Huffman |
| LZ77 | References past patterns via a sliding window. Foundation of DEFLATE |
| LZ78 / LZW | Builds an explicit dictionary. Used in GIF (history of patent issues) |
| DEFLATE | LZ77 + Huffman. Core of ZIP, gzip, PNG, HTTP |
| BWT | Reversible data rearrangement transform. Core technology of bzip2 |
| zstd | Best balance of speed and compression ratio among modern algorithms |
| Brotli | Highest compression ratio for web delivery. Built-in static dictionary |
| LZ4 / Snappy | Ultra-fast compression/decompression. For real-time and DB use |
| JPEG | DCT + quantization + Huffman. Standard photo format |
| H.264 / AV1 | Inter-frame prediction + motion compensation. Standard video codecs |
| HTTP compression | 70-85% reduction for text types. Images, etc. are excluded |

### Selection Flowchart

```
  Compression algorithm selection:

  What type of data?
  ├── Text/binary (lossless required)
  │   ├── Real-time requirement?
  │   │   ├── Yes → LZ4 or Snappy
  │   │   └── No
  │   │       ├── Web delivery?
  │   │       │   ├── Static → Brotli (level 11)
  │   │       │   └── Dynamic → gzip (level 6) or zstd (level 3)
  │   │       ├── Archival storage?
  │   │       │   └── LZMA2 (7z) or zstd (level 19)
  │   │       └── General purpose?
  │   │           └── zstd (level 3-6)
  │   │
  ├── Image
  │   ├── Lossless required? → PNG or WebP (lossless)
  │   ├── Photo? → JPEG (quality 75-85) or WebP or AVIF
  │   └── UI/logo? → SVG (vector) or PNG
  │
  ├── Audio
  │   ├── Lossless required? → FLAC
  │   ├── Music distribution? → AAC or Opus (128-256 kbps)
  │   └── Calls/VoIP? → Opus (32-64 kbps)
  │
  └── Video
      ├── Maximum compatibility? → H.264
      ├── High efficiency? → H.265 or AV1
      └── Royalty-free? → AV1 or VP9
```

---

## 9. Recommended Next Reading


---

## 10. References

### Foundational Theory

1. Shannon, C. E. "A Mathematical Theory of Communication." *Bell System Technical Journal*, Vol. 27, pp. 379-423, 623-656, 1948. -- The original paper on information theory. Defined the concept of entropy and the source coding theorem.
2. Huffman, D. A. "A Method for the Construction of Minimum-Redundancy Codes." *Proceedings of the IRE*, Vol. 40, No. 9, pp. 1098-1101, 1952. -- The original Huffman coding paper. Born from a student assignment at MIT.
3. Cover, T. M. & Thomas, J. A. *Elements of Information Theory*. 2nd Edition, Wiley-Interscience, 2006. -- The standard textbook on information theory. Comprehensively covers entropy, coding theorems, and Rate-Distortion theory.

### Dictionary-Based Compression

4. Ziv, J. & Lempel, A. "A Universal Algorithm for Sequential Data Compression." *IEEE Transactions on Information Theory*, Vol. 23, No. 3, pp. 337-343, 1977. -- The original LZ77 paper.
5. Ziv, J. & Lempel, A. "Compression of Individual Sequences via Variable-Rate Coding." *IEEE Transactions on Information Theory*, Vol. 24, No. 5, pp. 530-536, 1978. -- The original LZ78 paper.
6. Welch, T. A. "A Technique for High-Performance Data Compression." *Computer*, Vol. 17, No. 6, pp. 8-19, 1984. -- The original LZW paper.

### Modern Compression Technologies

7. Collet, Y. "Zstandard Compression and the application/zstd Media Type." RFC 8478, IETF, 2018. -- The RFC specification for Zstandard.
8. Alakuijala, J. & Szabadka, Z. "Brotli Compressed Data Format." RFC 7932, IETF, 2016. -- The RFC specification for Brotli.
9. Deutsch, P. "DEFLATE Compressed Data Format Specification version 1.3." RFC 1951, IETF, 1996. -- The official DEFLATE specification.

### Image and Video Compression

10. Wallace, G. K. "The JPEG Still Picture Compression Standard." *Communications of the ACM*, Vol. 34, No. 4, pp. 30-44, 1991. -- An explanatory paper on the JPEG standard.
11. Wiegand, T. et al. "Overview of the H.264/AVC Video Coding Standard." *IEEE Transactions on Circuits and Systems for Video Technology*, Vol. 13, No. 7, pp. 560-576, 2003. -- A comprehensive overview of H.264.
12. Chen, Y. et al. "An Overview of Core Coding Tools in the AV1 Video Codec." *2018 Picture Coding Symposium (PCS)*, IEEE, 2018. -- Technical overview of AV1.

### Supplementary Materials

13. Duda, J. "Asymmetric Numeral Systems: Entropy Coding Combining Speed of Huffman Coding with Compression Rate of Arithmetic Coding." *arXiv:1311.2540*, 2013. -- The original ANS paper. Foundation technology of zstd.
14. Burrows, M. & Wheeler, D. J. "A Block-sorting Lossless Data Compression Algorithm." *Technical Report 124*, Digital Equipment Corporation, 1994. -- The original BWT paper. Foundation technology of bzip2.

---

## Recommended Next Reading

- [Storage Capacity and Units](./05-storage-capacity.md) - Proceed to the next topic

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://ja.wikipedia.org/) - Overview of technical concepts
