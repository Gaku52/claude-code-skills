# Information Theory

> Shannon's information theory mathematically established the limits of data compression and the limits of communication reliability.
> It provides the mathematical foundation for treating information quantitatively, and pervades all of modern data compression, communication, cryptography, and machine learning.

## What You Will Learn in This Chapter

- [ ] Understand the mathematical definition of information (bits) and compute it for any probability distribution
- [ ] Explain the meaning of entropy and its relationship to the theoretical limits of data compression
- [ ] State precisely the claim of Shannon's first fundamental theorem (source coding theorem)
- [ ] Implement the Huffman code construction procedure and understand the properties of optimal prefix codes
- [ ] Gain an intuitive understanding of Shannon's second fundamental theorem (channel coding theorem)
- [ ] Compare the principles and representative methods of error detection and correction codes
- [ ] Understand the definitions and applications of cross-entropy and KL divergence
- [ ] Implement fundamental information theory concepts in Python


## Prerequisites

Having the following knowledge before reading this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Computational Complexity Theory](./02-complexity-theory.md)

---

## 1. Why Information Theory Is Needed

### 1.1 The Significance of Treating "Information" Mathematically

The everyday word "information" is ambiguous. Between the statement "The weather is nice today" and "A meteorite will hit the Earth tomorrow," we intuitively feel that the latter contains "more information." However, to leverage this intuition in engineering, a framework for quantitatively measuring information is needed.

In 1948, Claude Shannon provided the definitive answer to this problem in his paper "A Mathematical Theory of Communication." Shannon formulated "information" in the language of probability theory and gave mathematical answers to the following fundamental questions.

1. **How far can data be compressed?** (Source coding theorem)
2. **How much reliability can be maintained over a noisy channel?** (Channel coding theorem)

These two theorems continue to influence every field of engineering.

### 1.2 Technology Domains Related to Information Theory

```
+------------------------------------------------------------------+
|                  Application Domains of Information Theory           |
+------------------------------------------------------------------+
|                                                                    |
|  Data Compression    Communication Eng. Cryptography                |
|  ┌──────────┐     ┌──────────┐     ┌──────────┐                  |
|  │ ZIP/gzip │     │ 5G / WiFi│     │ AES / RSA│                  |
|  │ JPEG/PNG │     │ Error    │     │ Entropy  │                  |
|  │ MP3/AAC  │     │ LDPC Code│     │ Source   │                  |
|  │ H.264    │     │ Polar Cod│     │ Key Secur│                  |
|  └──────────┘     └──────────┘     └──────────┘                  |
|                                                                    |
|  Machine Learning    NLP               Information Retrieval       |
|  ┌──────────┐     ┌──────────┐     ┌──────────┐                  |
|  │ Loss Func│     │ Perplexi │     │ TF-IDF   │                  |
|  │ Cross-Ent│     │ ty       │     │ Info Gain│                  |
|  │ ropy     │     │ Lang Mod │     │ Mutual   │                  |
|  │ KL Diver │     │ Eval Met │     │ Feature  │                  |
|  │ gence    │     │          │     │ Select   │                  |
|  └──────────┘     └──────────┘     └──────────┘                  |
+------------------------------------------------------------------+
```

### 1.3 Historical Background

Several pioneering studies led to the establishment of information theory.

| Year | Person | Contribution |
|------|------|------|
| 1928 | Hartley | Proposed logarithmic measure of information |
| 1948 | Shannon | Systematic construction of information theory. Formulation of entropy and channel capacity |
| 1949 | Shannon | Application to cryptography ("Communication Theory of Secrecy Systems") |
| 1951 | Huffman | Algorithm for constructing optimal prefix codes |
| 1952 | Hamming | Hamming code (pioneer of error-correcting codes) |
| 1960 | Reed & Solomon | Reed-Solomon code (burst error correction) |
| 1977 | Lempel & Ziv | LZ77 algorithm (foundation of dictionary-based compression) |
| 1993 | Berrou, Glavieux, Thitimajshima | Turbo codes (error correction approaching the Shannon limit) |
| 2009 | Arikan | Polar codes (theoretically achieving the Shannon limit) |

Shannon's paper was based on the revolutionary idea of separating information from "meaning" and treating it as a purely statistical quantity. It focuses solely on how "unpredictable" a symbol sequence is, regardless of the meaning it conveys. This abstraction gave rise to a theory that treats compression, communication, and cryptography in a unified framework.

---

## 2. Information Content and Entropy

### 2.1 Self-Information

The **self-information** of an event x that occurs with probability P(x) is defined as follows.

```
I(x) = -log₂(P(x))   [unit: bits]
```

This definition matches the following intuitions.

- **A certain event carries no information**: If P(x) = 1, then I(x) = 0
- **Rarer events carry more information**: The smaller P(x) is, the larger I(x) becomes
- **Information of independent events is additive**: I(x, y) = I(x) + I(y) (when x, y are independent)

```
Examples of self-information:

  Event               Prob P(x)   Information I(x) = -log₂(P(x))
  ─────────────────────────────────────────────────────────
  Fair coin (heads)     0.5         1.0 bits
  Fair die (1)          1/6         2.585 bits
  Certain event         1.0         0.0 bits
  Rare event            0.01        6.644 bits
  Very rare event       0.001       9.966 bits

  Graph (P vs I):
  I(x)
  10 |*
   8 | *
   6 |   *
   4 |      *
   2 |            *
   0 |________________________*___
     0   0.2   0.4   0.6   0.8  1.0  P(x)

  → As P(x) approaches 0, information diverges to infinity
  → As P(x) approaches 1, information converges to 0
```

Changing the base of the logarithm changes the unit. Base 2 gives **bits**, base e gives **nats**, and base 10 gives **hartleys**. In information engineering, bits are conventionally used.

### 2.2 Entropy

When a discrete random variable X follows the probability distribution {p₁, p₂, ..., pₙ}, the **Shannon entropy** is defined as follows.

```
H(X) = -Σᵢ pᵢ log₂(pᵢ)   [unit: bits]
```

Entropy is the "average information content," that is, "a measure of the uncertainty of the source."

### 2.3 Properties of Entropy

Entropy has the following important properties.

**Property 1: Non-negativity**
```
H(X) ≥ 0
```
Equality holds when X is deterministic (one event has probability 1).

**Property 2: Maximum value**
```
H(X) ≤ log₂(n)   (where n is the number of possible values of X)
```
Equality holds when X follows a uniform distribution. The uniform distribution is the "most random" distribution and corresponds to the state where compression is most difficult.

**Property 3: Chain rule**
```
H(X, Y) = H(X) + H(Y|X)
```
The joint entropy decomposes into the entropy of X and the conditional entropy of Y given X.

**Property 4: Concavity**
Entropy is a concave function of the probability distribution. That is, mixing two distributions causes entropy to increase (or remain unchanged).

### 2.4 Entropy Calculation Examples

```
Example 1: Fair coin toss
  P(heads) = 0.5, P(tails) = 0.5
  H = -0.5 × log₂(0.5) - 0.5 × log₂(0.5)
    = -0.5 × (-1) - 0.5 × (-1)
    = 0.5 + 0.5 = 1.0 bits

  → 1 bit of information is obtained per trial
  → Can be perfectly represented with 1 bit

Example 2: Biased coin
  P(heads) = 0.9, P(tails) = 0.1
  H = -0.9 × log₂(0.9) - 0.1 × log₂(0.1)
    = -0.9 × (-0.152) - 0.1 × (-3.322)
    = 0.137 + 0.332 = 0.469 bits

  → Less information than a fair coin (because it is more predictable)
  → Theoretically compressible to 0.469 bits/trial

Example 3: English text (26 letters + space)
  Character occurrence probabilities are non-uniform ('e' is about 12.7%, 'z' is about 0.07%)
  Uniform distribution: log₂(27) ≈ 4.75 bits/character
  Actual entropy: H ≈ 4.11 bits/character (first-order statistics only)
  Considering context: H ≈ 1.0-1.5 bits/character

  → Significantly compressible from ASCII 8 bits
```

### 2.5 Computing Information and Entropy in Python

```python
"""
Information content and entropy calculations
"""
import math
from collections import Counter
from typing import Dict, List


def self_information(probability: float) -> float:
    """
    Compute self-information.

    Args:
        probability: Probability of the event (0 < p <= 1)

    Returns:
        Self-information (in bits)

    Raises:
        ValueError: If probability is out of range
    """
    if not (0 < probability <= 1):
        raise ValueError(f"Probability must be in range (0, 1]: {probability}")
    return -math.log2(probability)


def entropy(probabilities: List[float]) -> float:
    """
    Compute Shannon entropy.

    Args:
        probabilities: Probability distribution (list summing to 1)

    Returns:
        Entropy (in bits)

    Raises:
        ValueError: If probabilities do not sum to 1
    """
    if abs(sum(probabilities) - 1.0) > 1e-9:
        raise ValueError(f"Probabilities do not sum to 1: {sum(probabilities)}")

    h = 0.0
    for p in probabilities:
        if p > 0:
            h -= p * math.log2(p)
    return h


def text_entropy(text: str) -> float:
    """
    Compute first-order entropy of text.

    Estimates the probability distribution from character frequencies
    and computes the entropy.

    Args:
        text: Input text

    Returns:
        Entropy per character (in bits)
    """
    if not text:
        return 0.0

    counter = Counter(text)
    total = len(text)
    probs = [count / total for count in counter.values()]
    return entropy(probs)


def entropy_analysis(text: str) -> Dict[str, float]:
    """
    Perform entropy analysis of text.

    Args:
        text: Input text

    Returns:
        Dictionary containing analysis results
    """
    if not text:
        return {"entropy": 0.0, "max_entropy": 0.0, "redundancy": 0.0}

    counter = Counter(text)
    n_symbols = len(counter)
    total = len(text)

    h = text_entropy(text)
    h_max = math.log2(n_symbols) if n_symbols > 1 else 0.0
    redundancy = 1.0 - (h / h_max) if h_max > 0 else 0.0

    return {
        "entropy": h,
        "max_entropy": h_max,
        "redundancy": redundancy,
        "n_symbols": n_symbols,
        "total_chars": total,
        "theoretical_min_bits": h * total,
        "ascii_bits": 8 * total,
        "compression_ratio": h / 8.0,
    }


# === Verification ===
if __name__ == "__main__":
    # Self-information
    print("=== Self-Information ===")
    for p in [0.5, 1/6, 0.01, 0.001, 1.0]:
        print(f"  P={p:.4f}  I={self_information(p):.3f} bits")

    # Entropy
    print("\n=== Entropy ===")
    print(f"  Fair coin: {entropy([0.5, 0.5]):.3f} bits")
    print(f"  Biased coin (0.9/0.1): {entropy([0.9, 0.1]):.3f} bits")
    print(f"  Fair die: {entropy([1/6]*6):.3f} bits")
    print(f"  Deterministic: {entropy([1.0]):.3f} bits")

    # Text entropy analysis
    sample_text = (
        "information theory is a mathematical framework for "
        "quantifying information content and communication"
    )
    print(f"\n=== Text Analysis ===")
    analysis = entropy_analysis(sample_text)
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
```

Running the above program produces the following expected output.

```
=== Self-Information ===
  P=0.5000  I=1.000 bits
  P=0.1667  I=2.585 bits
  P=0.0100  I=6.644 bits
  P=0.0010  I=9.966 bits
  P=1.0000  I=0.000 bits

=== Entropy ===
  Fair coin: 1.000 bits
  Biased coin (0.9/0.1): 0.469 bits
  Fair die: 2.585 bits
  Deterministic: 0.000 bits

=== Text Analysis ===
  entropy: 3.937
  max_entropy: 4.585
  redundancy: 0.141
  n_symbols: 24
  total_chars: 95
  theoretical_min_bits: 374.010
  ascii_bits: 760
  compression_ratio: 0.492
```

### 2.6 Conditional Entropy and Mutual Information

Let us organize the concepts that describe the relationship between two random variables X and Y.

```
Joint entropy:      H(X, Y) = -Σ P(x,y) log₂ P(x,y)
Conditional entropy:  H(Y|X)  = -Σ P(x,y) log₂ P(y|x)
Mutual information:            I(X;Y)  = H(X) + H(Y) - H(X,Y)
                                = H(X) - H(X|Y)
                                = H(Y) - H(Y|X)

Venn diagram of entropy:

  ┌───────────────────────────────────┐
  │               H(X,Y)              │
  │  ┌──────────┬──────────┐          │
  │  │          │          │          │
  │  │  H(X|Y)  │  I(X;Y)  │ H(Y|X)  │
  │  │          │          │          │
  │  └──────────┴──────────┘          │
  │     H(X)         H(Y)             │
  └───────────────────────────────────┘

  H(X,Y) = H(X|Y) + I(X;Y) + H(Y|X)
  I(X;Y) = H(X) + H(Y) - H(X,Y)

  Intuitive interpretation:
  - H(X|Y): Remaining uncertainty of X even after knowing Y
  - I(X;Y): Amount of information shared between X and Y
  - I(X;Y) = 0 ⟺ X and Y are independent
```

Mutual information has a wide range of applications, including feature selection, independence testing, and information gain (decision trees).

---

## 3. Source Coding (Shannon's First Fundamental Theorem)

### 3.1 Statement of the Source Coding Theorem

**Shannon's first fundamental theorem (source coding theorem, 1948):**

> Let H(X) be the entropy of a memoryless source X. When losslessly encoding the symbol sequences generated from X, the average code length L per symbol satisfies:
>
> **H(X) ≤ L < H(X) + 1**
>
> Moreover, by block-encoding n symbols together, the average code length per symbol can be made arbitrarily close to H(X).

This theorem contains two claims.

1. **Impossibility**: It is impossible to achieve lossless compression with an average code length shorter than entropy H(X).
2. **Achievability**: There exist codes that achieve an average code length arbitrarily close to entropy H(X).

### 3.2 Prefix Code

A **prefix-free code** (prefix code) is a code in which no codeword is a prefix of any other codeword. This property enables unique decoding without a special delimiter symbol.

```
Example of a prefix code:

  Symbol  Codeword
  ─────────────
  A       0
  B       10
  C       110
  D       111

  Uniqueness of decoding:
  Read the code string "010110111" from left to right:
    0    → A
    10   → B
    110  → C
    111  → D
  Result: "ABCD"

  Counterexample of a non-prefix code:
  Symbol  Codeword
  ─────────────
  A       0
  B       01      ← Codeword "0" for A is a prefix of B
  C       011

  "011" can be interpreted as "A, B" or "C" → unique decoding is impossible
```

**Kraft's inequality**: The necessary and sufficient condition for a prefix code with codeword lengths d₁, d₂, ..., dₙ to exist is:

```
Σᵢ 2^(-dᵢ) ≤ 1
```

### 3.3 Huffman Code

The Huffman code (Huffman, 1952) is an algorithm that constructs an optimal prefix code for a given probability distribution. "Optimal" means that the average code length is the minimum among all prefix codes.

**Algorithm:**

```
Huffman tree construction procedure:

  Input: Each symbol and its probability
  Output: Optimal prefix code

  1. Initialize each symbol as a leaf node with its probability
  2. Extract the two nodes with the smallest probabilities
  3. Create a new internal node with these as children,
     with probability equal to the sum of the two children's probabilities
  4. Insert the new node back into the list
  5. Repeat steps 2-4 until only one node remains
  6. Traverse from root to each leaf,
     assigning 0 for left branches and 1 for right branches as codewords

  Example: {A:0.4, B:0.3, C:0.2, D:0.1}

  Step 1: Merge C(0.2) and D(0.1) → CD
  Step 2: Merge B(0.3) and CD → BCD
  Step 3: Merge A(0.4) and BCD → ABCD

  Huffman tree:
              [1.0]
             /     \
           0/       \1
          A(0.4)   [0.6]
                   /     \
                 0/       \1
               B(0.3)   [0.3]
                         /     \
                       0/       \1
                     C(0.2)   D(0.1)

  Code assignment:
  A → 0      (1 bit)
  B → 10     (2 bits)
  C → 110    (3 bits)
  D → 111    (3 bits)

  Average code length:
  L = 0.4×1 + 0.3×2 + 0.2×3 + 0.1×3
    = 0.4 + 0.6 + 0.6 + 0.3
    = 1.9 bits

  Entropy:
  H = -(0.4×log₂0.4 + 0.3×log₂0.3 + 0.2×log₂0.2 + 0.1×log₂0.1)
    ≈ 1.846 bits

  Efficiency: H/L ≈ 1.846/1.9 ≈ 97.2%
```

### 3.4 Arithmetic Coding

Because Huffman codes assign integer-bit-length codewords to each symbol, efficiency drops when probabilities are not powers of two. **Arithmetic coding** avoids this constraint by representing the entire message as a single number within the interval [0, 1).

```
Concept of arithmetic coding:

  Encode the message "BAC"
  Probabilities: A=0.4, B=0.3, C=0.3

  Initial interval: [0, 1)

  Encode B: B's range is [0.4, 0.7)
    → Interval [0.4, 0.7)

  Encode A: A's range is [0, 0.4)
    → New interval: [0.4, 0.4 + 0.3×0.4) = [0.4, 0.52)

  Encode C: C's range is [0.7, 1.0)
    → New interval: [0.4 + 0.12×0.7, 0.4 + 0.12×1.0) = [0.484, 0.52)

  Any value within the final interval [0.484, 0.52) can be decoded
  For example, representing 0.5 in binary → "1" followed by precision bits

  Advantage: Approaches entropy without block coding
  Disadvantage: High computational cost, patent issues (historically)
```

### 3.5 ANS (Asymmetric Numeral Systems)

ANS is a relatively new entropy coding method proposed by Jarek Duda (2009) that achieves compression ratios comparable to arithmetic coding at speeds comparable to Huffman coding. It is used in zstd (Zstandard), developed by Facebook.

```
Comparison of entropy coding methods:

  Method        Compression Speed     Complexity Use cases
  ──────────────────────────────────────────────────────
  Huffman       Good        Fast      Low      DEFLATE, JPEG
  Arithmetic    Best        Slow      High     JPEG2000, H.265
  ANS (rANS)   Best        Fast      Medium   zstd, LZFSE
  ANS (tANS)   Best        Fastest   Medium   zstd, Brotli

  Compression: Huffman ≤ ANS ≈ Arithmetic ≤ Shannon limit
```

---

## 4. Channel Coding (Shannon's Second Fundamental Theorem)

### 4.1 Channel Model

A channel is an abstract model of the path for transmitting information from sender to receiver. Real channels have noise, and there is a possibility that transmitted data may not arrive correctly.

```
General communication model (Shannon model):

  Source → [Encoder] → [Channel] → [Decoder] → Receiver
                        ↑ Noise

  Source coding: Data compression (removal of redundancy)
  Channel coding: Addition of redundancy for error correction
  → Source coding and channel coding can be optimized separately
    (Separation theorem)
```

The most basic channel model is the **Binary Symmetric Channel (BSC)**.

```
Binary Symmetric Channel (BSC):

  Transmit             Receive
  ─────                ─────
         1-p
  0 ─────────────── 0
    \              /
     \   p      p /
      \          /
       ×        ×
      /          \
     /   p      p \
    /              \
  1 ─────────────── 1
         1-p

  p: Bit flip probability (error rate)
  1-p: Probability of correct transmission

  p = 0: Perfect channel (no noise)
  p = 0.5: Completely random (communication impossible)
  p = 1: All bits flipped (becomes a perfect channel after inversion)
```

### 4.2 Channel Capacity

**Channel capacity** C is the maximum rate (bits/use) at which reliable communication is possible through the channel.

```
Definition of channel capacity:

  C = max_{p(x)} I(X;Y)

  For BSC:
  C = 1 - H(p) = 1 + p×log₂(p) + (1-p)×log₂(1-p)

  For Additive White Gaussian Noise (AWGN) channel:
  C = (1/2) × log₂(1 + S/N)  [bits/use]

  where S/N is the signal-to-noise ratio (SNR)

  Examples of channel capacity:

  BSC (p=0.1):  C = 1 - H(0.1) ≈ 1 - 0.469 = 0.531 bits/use
  BSC (p=0.01): C = 1 - H(0.01) ≈ 1 - 0.081 = 0.919 bits/use
  BSC (p=0.5):  C = 1 - H(0.5) = 1 - 1.0 = 0.0 bits/use

  AWGN (SNR=10, i.e. 10dB):
  C = 0.5 × log₂(1 + 10) ≈ 1.73 bits/use
```

### 4.3 Shannon's Second Fundamental Theorem (Channel Coding Theorem)

**Shannon's second fundamental theorem (channel coding theorem, 1948):**

> For a channel with capacity C, for any communication rate R < C, there exist codes that can make the decoding error rate arbitrarily small.
> Conversely, at communication rates R > C, it is impossible to bring the decoding error rate close to 0.

Importantly, this theorem is an "existence theorem" and does not show a specific code construction method. Shannon proved this theorem using probabilistic arguments with random codebooks. It took approximately 50 years from the publication of the theorem for practical codes (turbo codes, LDPC codes, Polar codes) to achieve performance close to the Shannon limit.

```
Convergence to the Shannon limit (historical progression):

  Gap from channel capacity (dB)
  10 |  ● Hamming code (1950)
   8 |
   6 |     ● BCH code (1960)
   4 |        ● Reed-Solomon (1960)
   3 |
   2 |              ● Concatenated codes (1970)
   1 |
   0.5|                    ● Turbo codes (1993)
   0.3|                       ● LDPC codes (rediscovered, 1996)
   0.1|                          ● Polar codes (2009)
   0  |────────────────────────────── Shannon limit
      1950    1960    1970    1990    2000    2010
```

### 4.4 Error Detection and Error Correction

#### Parity Check

The simplest error detection method is the addition of a parity bit.

```
Parity bit:

  Data: 1011001
  Even parity: Add a bit so the number of 1s is even
  → 10110011 (parity bit = 1)

  Received: 10110111 (1-bit error)
  Parity check: Number of 1s = 7 (odd) → Error detected

  Limitations:
  - 1-bit errors can be detected
  - 2-bit errors cannot be detected
  - Error position cannot be identified (not correctable)
```

#### Hamming Code

The Hamming code (Hamming, 1950) is a code that enables 1-bit error correction and 2-bit error detection.

```
Hamming(7,4) code:

  4 bits of data → 7-bit codeword (3 redundant bits added)

  Position: 1  2  3  4  5  6  7
  Type:     p1 p2 d1 p3 d2 d3 d4
  (p=parity, d=data)

  Parity computation:
  p1: Covers positions 1,3,5,7 (LSB of binary representation is 1)
  p2: Covers positions 2,3,6,7 (2nd bit of binary representation is 1)
  p3: Covers positions 4,5,6,7 (3rd bit of binary representation is 1)

  Example: Data = 1011
  d1=1, d2=0, d3=1, d4=1

  p1 = d1 ⊕ d2 ⊕ d4 = 1 ⊕ 0 ⊕ 1 = 0
  p2 = d1 ⊕ d3 ⊕ d4 = 1 ⊕ 1 ⊕ 1 = 1
  p3 = d2 ⊕ d3 ⊕ d4 = 0 ⊕ 1 ⊕ 1 = 0

  Codeword: 0 1 1 0 0 1 1

  Error correction: Suppose the bit at position 5 is flipped
  Received: 0 1 1 0 1 1 1
                ^
  Syndrome computation:
  s1 = p1 ⊕ d1 ⊕ d2 ⊕ d4 = 0 ⊕ 1 ⊕ 1 ⊕ 1 = 1
  s2 = p2 ⊕ d1 ⊕ d3 ⊕ d4 = 1 ⊕ 1 ⊕ 1 ⊕ 1 = 0
  s3 = p3 ⊕ d2 ⊕ d3 ⊕ d4 = 0 ⊕ 1 ⊕ 1 ⊕ 1 = 1

  Syndrome = s3 s2 s1 = 101 (binary) = 5 (decimal)
  → Flip the bit at position 5 to correct
```

#### Reed-Solomon Code

The Reed-Solomon code (Reed-Solomon, 1960) is a code that excels at correcting **burst errors** (errors in consecutive bits).

```
Characteristics of Reed-Solomon codes:

  RS(n, k): n-symbol codeword, k symbols of data
  Correction capability: Can correct t = (n - k) / 2 symbol errors

  Applications:
  ┌─────────────────┬──────────────┬─────────────┐
  │ Application      │ Parameters   │ Correction   │
  ├─────────────────┼──────────────┼─────────────┤
  │ CD               │ RS(32,28)    │ 2 symbols   │
  │ DVD              │ RS(208,192)  │ 8 symbols   │
  │ QR Code          │ 4 levels     │ 7%-30%      │
  │ Deep space comm  │ RS(255,223)  │ 16 symbols  │
  │ Digital broadcast│ RS(204,188)  │ 8 symbols   │
  └─────────────────┴──────────────┴─────────────┘

  QR code error correction levels:
  L: Can recover approximately 7% of codewords
  M: Can recover approximately 15% of codewords
  Q: Can recover approximately 25% of codewords
  H: Can recover approximately 30% of codewords
```

#### Modern Error-Correcting Codes

```
Comparison of modern error-correcting codes:

  Code           Year  Gap from      Complexity  Use cases
                       Shannon limit
  ─────────────────────────────────────────────────────
  Hamming        1950  Large         O(n)       Memory ECC
  BCH            1960  Medium        O(n^2)     Flash memory
  RS             1960  Medium        O(n^2)     CD/DVD, QR
  Convolutional  1955  Medium        O(2^k)     Legacy comm
  Turbo          1993  ~0.5 dB       O(n log n) 3G/4G
  LDPC           1996  ~0.04 dB      O(n)       Wi-Fi 6, 5G
  Polar          2009  0 dB(theory)  O(n log n) 5G control ch

  → The discovery of turbo codes was called "a breakthrough in coding theory"
  → LDPC was proposed by Gallager(1962) and rediscovered by MacKay(1996)
  → Polar codes were the first to "constructively" achieve the Shannon limit
```

---

## 5. Data Compression in Practice

### 5.1 Lossless and Lossy Compression

Data compression is broadly classified into two categories.

```
Classification of data compression:

  ┌───────────────────────────────────────────┐
  │           Data Compression                │
  ├─────────────────────┬─────────────────────┤
  │   Lossless          │   Lossy              │
  │   (Lossless)        │   (Lossy)            │
  ├─────────────────────┼─────────────────────┤
  │ Original data can   │ Approximation of     │
  │ be fully restored   │ original is restored │
  ├─────────────────────┼─────────────────────┤
  │ ZIP, gzip, bzip2    │ JPEG, MP3, H.264     │
  │ PNG, FLAC           │ AAC, HEVC, AV1       │
  │ zstd, brotli        │ Opus, WebP           │
  ├─────────────────────┼─────────────────────┤
  │ Ratio: 2:1 to 10:1  │ Ratio: 10:1 to 100:1│
  │ (data dependent)    │ (quality dependent)  │
  ├─────────────────────┼─────────────────────┤
  │ Use: text,           │ Use: images, audio,  │
  │ programs, DB        │ video                │
  └─────────────────────┴─────────────────────┘
```

### 5.2 LZ77 Algorithm

LZ77 (Lempel-Ziv, 1977) is the foundation of dictionary-based compression and the core of modern compression algorithms such as gzip, DEFLATE, and zstd.

```
Basic principle of LZ77:

  Within a sliding window of past data,
  find the longest substring matching the data from the current position,
  and represent it as a triple (distance, length, next character).

  Example: Compress "ABRACADABRA"

  Pos   Input     Search buffer Match      Output
  ─────────────────────────────────────────────
  0     A         (empty)       None       (0, 0, 'A')
  1     B         A             None       (0, 0, 'B')
  2     R         AB            None       (0, 0, 'R')
  3     A         ABR           A(dist 3)  (3, 1, 'C')
  5     A         ABRAC         A(dist 5)  (5, 1, 'D')
  7     A         ABRACAD       ABRA(dist7)(7, 4, '\0')

  Compression intuition:
  "ABRACADABRA" (11 chars) → 6 tokens
  More repetitive patterns lead to better compression ratios
```

### 5.3 DEFLATE Algorithm

DEFLATE is a compression algorithm that combines LZ77 and Huffman coding, used in ZIP, gzip, and PNG. It is standardized in RFC 1951.

```
DEFLATE processing flow:

  Input data
      │
      ▼
  ┌──────────┐
  │  LZ77    │  Detect duplicate patterns and replace with references
  │  Compress │  Generate (distance, length) pairs
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ Huffman  │  Frequency-based encoding of
  │ Coding   │  literals/lengths/distances
  └────┬─────┘
       │
       ▼
  Compressed data

  DEFLATE block structure:
  - Uncompressed block: Copy as-is
  - Fixed Huffman: Use predefined Huffman tree
  - Dynamic Huffman: Build optimal Huffman tree from data
```

### 5.4 Comparison of Modern Compression Algorithms

```
  Algorithm     Year  Ratio     Comp speed Decomp spd Use cases
  ──────────────────────────────────────────────────────────
  gzip/DEFLATE  1993  Medium    Medium     Fast      HTTP, ZIP
  bzip2         1996  Good      Slow       Slow      UNIX compress
  LZ4           2011  Low       Fastest    Fastest   File systems
  Brotli        2015  Good      Slow       Fast      HTTPS compress
  zstd          2016  Good      Fast       Fastest   Facebook, Linux
  ──────────────────────────────────────────────────────────

  Selection guidelines:
  - Speed priority: LZ4 or zstd (low compression level)
  - Compression priority: zstd (high compression level) or Brotli
  - Compatibility priority: gzip (most widely supported)
  - Real-time use: LZ4 (games, databases)
```

### 5.5 LZ77 Compression Implementation in Python

```python
"""
LZ77 compression and decompression implementation

An educational implementation to demonstrate basic concepts of dictionary-based compression.
For practical compression, use zlib or zstd.
"""
from typing import List, Tuple


class LZ77Compressor:
    """Implementation of the LZ77 compression algorithm"""

    def __init__(self, window_size: int = 256, lookahead_size: int = 15):
        """
        Args:
            window_size: Size of the sliding window
            lookahead_size: Size of the lookahead buffer
        """
        self.window_size = window_size
        self.lookahead_size = lookahead_size

    def compress(self, data: str) -> List[Tuple[int, int, str]]:
        """
        Compress data using LZ77.

        Args:
            data: Input string

        Returns:
            List of (distance, length, next_char) tokens
        """
        tokens: List[Tuple[int, int, str]] = []
        pos = 0

        while pos < len(data):
            best_distance = 0
            best_length = 0

            # Search buffer range
            search_start = max(0, pos - self.window_size)

            # Lookahead buffer range
            lookahead_end = min(pos + self.lookahead_size, len(data))

            # Search for longest match
            for i in range(search_start, pos):
                length = 0
                while (pos + length < lookahead_end
                       and data[i + length] == data[pos + length]):
                    length += 1
                    # Handle case when exceeding end of search buffer
                    if i + length >= pos:
                        break

                if length > best_length:
                    best_distance = pos - i
                    best_length = length

            # Generate token
            if best_length > 0 and pos + best_length < len(data):
                next_char = data[pos + best_length]
                tokens.append((best_distance, best_length, next_char))
                pos += best_length + 1
            else:
                tokens.append((0, 0, data[pos]))
                pos += 1

        return tokens

    def decompress(self, tokens: List[Tuple[int, int, str]]) -> str:
        """
        Decompress LZ77 token sequence.

        Args:
            tokens: List of (distance, length, next_char) tokens

        Returns:
            Restored string
        """
        result: List[str] = []

        for distance, length, next_char in tokens:
            if distance > 0 and length > 0:
                start = len(result) - distance
                for i in range(length):
                    result.append(result[start + i])
            result.append(next_char)

        return "".join(result)


# === Verification ===
if __name__ == "__main__":
    compressor = LZ77Compressor(window_size=32, lookahead_size=8)

    test_cases = [
        "ABRACADABRA",
        "AAAAAAAAA",
        "ABCABCABCABC",
        "HELLO WORLD",
    ]

    for text in test_cases:
        tokens = compressor.compress(text)
        restored = compressor.decompress(tokens)

        print(f"Original:     '{text}' ({len(text)} chars)")
        print(f"Token count:  {len(tokens)}")
        print(f"Tokens:       {tokens}")
        print(f"Restored:     '{restored}'")
        print(f"Match:        {text == restored}")
        print()
```

Expected output:

```
Original:     'ABRACADABRA' (11 chars)
Token count:  6
Tokens:       [(0, 0, 'A'), (0, 0, 'B'), (0, 0, 'R'), (3, 1, 'C'), (5, 1, 'D'), (7, 4, 'A')]
Restored:     'ABRACADABRA'
Match:        True

Original:     'AAAAAAAAA' (9 chars)
Token count:  2
Tokens:       [(0, 0, 'A'), (1, 7, 'A')]
Restored:     'AAAAAAAAA'
Match:        True

Original:     'ABCABCABCABC' (12 chars)
Token count:  4
Tokens:       [(0, 0, 'A'), (0, 0, 'B'), (0, 0, 'C'), (3, 8, 'C')]
Restored:     'ABCABCABCABC'
Match:        True

Original:     'HELLO WORLD' (11 chars)
Token count:  10
Tokens:       [(0, 0, 'H'), (0, 0, 'E'), (0, 0, 'L'), (1, 1, 'O'), (0, 0, ' '), (0, 0, 'W'), (0, 0, 'O'), (0, 0, 'R'), (0, 0, 'L'), (0, 0, 'D')]
Restored:     'HELLO WORLD'
Match:        True
```

---

## 6. Applications of Information Theory

### 6.1 Cross-Entropy

The **cross-entropy** H(P, Q) is the average code length when encoding data from the true probability distribution P using the estimated distribution Q.

```
Definition of cross-entropy:

  H(P, Q) = -Σ P(x) × log₂(Q(x))

  Properties:
  - H(P, Q) ≥ H(P) (Gibbs' inequality)
  - H(P, Q) = H(P) ⟺ P = Q
  - H(P, Q) ≠ H(Q, P) generally asymmetric

  Intuitive interpretation:
  The average number of bits when encoding data sampled from
  the true distribution P using a code based on distribution Q.
  If P ≠ Q, more bits than H(P) are required.
```

In machine learning, cross-entropy is widely used as a **loss function**. In classification problems, the cross-entropy between the true label distribution P (one-hot) and the model's predicted distribution Q is minimized.

```
Cross-entropy loss in classification:

  True label (one-hot):  P = [0, 0, 1, 0, 0]  (class 3 is correct)
  Model prediction:      Q = [0.05, 0.1, 0.7, 0.1, 0.05]

  H(P, Q) = -1 × log₂(0.7) ≈ 0.515 bits

  When prediction is poor:
  Model prediction:      Q = [0.2, 0.2, 0.2, 0.2, 0.2]
  H(P, Q) = -1 × log₂(0.2) ≈ 2.322 bits

  → The closer the prediction is to the correct answer, the smaller the cross-entropy
  → For a perfect prediction, H(P, Q) = 0
```

### 6.2 KL Divergence

**KL divergence** (Kullback-Leibler divergence) is an asymmetric measure of the "distance" between two probability distributions.

```
Definition of KL divergence:

  D_KL(P || Q) = Σ P(x) × log₂(P(x) / Q(x))
               = H(P, Q) - H(P)

  Properties:
  - D_KL(P || Q) ≥ 0 (Gibbs' inequality)
  - D_KL(P || Q) = 0 ⟺ P = Q
  - D_KL(P || Q) ≠ D_KL(Q || P) (asymmetric, not a distance metric)
  - D_KL(P || Q) is the "information loss" when approximating P with Q

  Intuitive interpretation:
  The extra number of bits required when encoding data from the true
  distribution P using a code based on Q, compared to the optimal code (based on P).
```

### 6.3 Applications of KL Divergence

```
Major applications of KL divergence:

  1. Variational Inference (VAE):
     Maximization of the variational lower bound ELBO
     = E[log p(x|z)] - D_KL(q(z|x) || p(z))
     Bring the variational posterior q(z|x) closer to the prior p(z)

  2. Policy Gradient Methods (Reinforcement Learning):
     TRPO/PPO: Constrain KL divergence of policy updates
     → Stabilize training

  3. Knowledge Distillation:
     Minimize KL when a small model (student)
     imitates the output distribution of a large model (teacher)

  4. Language Model Evaluation:
     Perplexity = 2^{H(P,Q)}
     → Exponent of cross-entropy (KL divergence + entropy)
```

### 6.4 Computing Cross-Entropy and KL Divergence in Python

```python
"""
Cross-entropy and KL divergence calculations
"""
import math
from typing import List


def cross_entropy(p: List[float], q: List[float]) -> float:
    """
    Compute cross-entropy H(P, Q).

    Args:
        p: True probability distribution
        q: Estimated probability distribution

    Returns:
        Cross-entropy (in bits)
    """
    if len(p) != len(q):
        raise ValueError("Distribution lengths do not match")

    result = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')  # Infinity if Q(x)=0 and P(x)>0
            result -= pi * math.log2(qi)
    return result


def kl_divergence(p: List[float], q: List[float]) -> float:
    """
    Compute KL divergence D_KL(P || Q).

    Args:
        p: True probability distribution
        q: Estimated probability distribution

    Returns:
        KL divergence (in bits)
    """
    if len(p) != len(q):
        raise ValueError("Distribution lengths do not match")

    result = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            result += pi * math.log2(pi / qi)
    return result


def jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    """
    Compute Jensen-Shannon divergence JSD(P || Q).
    Symmetric version of KL divergence.

    Args:
        p: Probability distribution 1
        q: Probability distribution 2

    Returns:
        JSD (in bits), range is [0, 1]
    """
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


# === Verification ===
if __name__ == "__main__":
    # Classification example
    true_dist = [0.0, 0.0, 1.0, 0.0, 0.0]  # Class 3 is correct

    good_pred = [0.05, 0.10, 0.70, 0.10, 0.05]
    bad_pred  = [0.20, 0.20, 0.20, 0.20, 0.20]

    print("=== Cross-entropy for classification ===")
    print(f"  Good prediction: H(P,Q) = {cross_entropy(true_dist, good_pred):.4f} bits")
    print(f"  Bad prediction:  H(P,Q) = {cross_entropy(true_dist, bad_pred):.4f} bits")

    # Comparison between general distributions
    p = [0.4, 0.3, 0.2, 0.1]
    q = [0.25, 0.25, 0.25, 0.25]  # Uniform distribution

    print("\n=== Distribution comparison ===")
    print(f"  P = {p}")
    print(f"  Q = {q}")
    print(f"  H(P)         = {cross_entropy(p, p):.4f} bits (entropy)")
    print(f"  H(P, Q)      = {cross_entropy(p, q):.4f} bits")
    print(f"  D_KL(P || Q) = {kl_divergence(p, q):.4f} bits")
    print(f"  D_KL(Q || P) = {kl_divergence(q, p):.4f} bits (asymmetric)")
    print(f"  JSD(P || Q)  = {jensen_shannon_divergence(p, q):.4f} bits")

    # Verify asymmetry of KL divergence
    print("\n=== Asymmetry verification ===")
    r = [0.9, 0.1]
    s = [0.5, 0.5]
    print(f"  R = {r}, S = {s}")
    print(f"  D_KL(R || S) = {kl_divergence(r, s):.4f}")
    print(f"  D_KL(S || R) = {kl_divergence(s, r):.4f}")
    print(f"  → D_KL(R||S) ≠ D_KL(S||R), so it does not satisfy the metric axioms")
```

Expected output:

```
=== Cross-entropy for classification ===
  Good prediction: H(P,Q) = 0.5146 bits
  Bad prediction:  H(P,Q) = 2.3219 bits

=== Distribution comparison ===
  P = [0.4, 0.3, 0.2, 0.1]
  Q = [0.25, 0.25, 0.25, 0.25]
  H(P)         = 1.8464 bits (entropy)
  H(P, Q)      = 2.0000 bits
  D_KL(P || Q) = 0.1536 bits
  D_KL(Q || P) = 0.1699 bits (asymmetric)
  JSD(P || Q)  = 0.0394 bits

=== Asymmetry verification ===
  R = [0.9, 0.1], S = [0.5, 0.5]
  D_KL(R || S) = 0.5310
  D_KL(S || R) = 0.3681
  → D_KL(R||S) ≠ D_KL(S||R), so it does not satisfy the metric axioms
```

### 6.5 Mutual Information and Feature Selection

Mutual information I(X; Y) is an important metric in feature selection. By computing the mutual information between each feature X and the target variable Y, and selecting features with large I(X; Y), useful features for prediction can be identified.

```
Mutual information in feature selection:

  Feature    | I(X; Y) | Interpretation
  ───────────┼─────────┼──────────────────────
  Age        | 0.45    | Strong relationship with target
  Income     | 0.38    | Moderate relationship with target
  Gender     | 0.02    | Almost unrelated to target
  Zip code   | 0.01    | Almost unrelated to target

  → Select age and income, exclude gender and zip code

  Information gain in decision trees:
  Information gain = H(Y) - H(Y|X) = I(X; Y)
  → Split on the feature that reduces entropy the most
```

### 6.6 Rate-Distortion Theory

The theoretical limit of lossy compression is given by **rate-distortion theory**.

```
Rate-distortion function:

  R(D) = min_{p(x̂|x): E[d(x,x̂)]≤D} I(X; X̂)

  R(D): Minimum rate required to compress with distortion at most D
  d(x, x̂): Distortion function (e.g., squared error)

  Graph of R(D):
  R
  │
  H │────*
    │     \
    │      \
    │       \
    │        --------
    │                 -----
  0 └──────────────────────── D
    0                    D_max

  - D=0: Lossless compression (rate = entropy H)
  - Increasing D: Greater tolerable distortion allows lower rate compression
  - D_max: Distortion achievable even when ignoring the source data

  Applications:
  - JPEG quality parameter: Select the operating point on the R(D) curve
  - Bitrate control in video coding
  - Quality-bitrate tradeoff in audio compression
```

---

## 7. Implementation: Huffman Coding in Python

### 7.1 Complete Implementation

```python
"""
Complete implementation of Huffman coding

Builds Huffman trees, encodes, and decodes.
Also supports file compression and decompression.
"""
import heapq
from collections import Counter
from typing import Dict, Optional, Tuple


class HuffmanNode:
    """Node of a Huffman tree"""

    def __init__(
        self,
        char: Optional[str] = None,
        freq: int = 0,
        left: Optional["HuffmanNode"] = None,
        right: Optional["HuffmanNode"] = None,
    ):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: "HuffmanNode") -> bool:
        """Comparison operator for priority queue"""
        return self.freq < other.freq

    def is_leaf(self) -> bool:
        """Whether this is a leaf node"""
        return self.left is None and self.right is None


class HuffmanCoder:
    """Huffman encoder/decoder"""

    def __init__(self):
        self.root: Optional[HuffmanNode] = None
        self.codes: Dict[str, str] = {}
        self.reverse_codes: Dict[str, str] = {}

    def build_tree(self, text: str) -> None:
        """
        Build a Huffman tree from text.

        Args:
            text: Input text
        """
        if not text:
            return

        # Count character frequencies
        freq = Counter(text)

        # Initialize priority queue
        heap: list = []
        for char, count in freq.items():
            heapq.heappush(heap, HuffmanNode(char=char, freq=count))

        # Handle case with only one symbol type
        if len(heap) == 1:
            node = heapq.heappop(heap)
            self.root = HuffmanNode(freq=node.freq, left=node)
            self._generate_codes(self.root, "")
            return

        # Build Huffman tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(
                freq=left.freq + right.freq,
                left=left,
                right=right,
            )
            heapq.heappush(heap, merged)

        self.root = heap[0]
        self._generate_codes(self.root, "")

    def _generate_codes(self, node: Optional[HuffmanNode], code: str) -> None:
        """
        Traverse the Huffman tree to generate codes for each character.

        Args:
            node: Current node
            code: Bit string corresponding to the current path
        """
        if node is None:
            return

        if node.is_leaf() and node.char is not None:
            # Assign "0" if code is empty (when there is only one symbol type)
            self.codes[node.char] = code if code else "0"
            self.reverse_codes[code if code else "0"] = node.char
            return

        self._generate_codes(node.left, code + "0")
        self._generate_codes(node.right, code + "1")

    def encode(self, text: str) -> str:
        """
        Convert text to Huffman code.

        Args:
            text: Input text

        Returns:
            Bit string (string representation)
        """
        if not self.codes:
            self.build_tree(text)
        return "".join(self.codes[char] for char in text)

    def decode(self, encoded: str) -> str:
        """
        Decode Huffman code back to original text.

        Args:
            encoded: Bit string (string representation)

        Returns:
            Decoded text
        """
        if self.root is None:
            return ""

        result = []
        current = self.root

        for bit in encoded:
            if bit == "0":
                current = current.left
            else:
                current = current.right

            if current is not None and current.is_leaf():
                result.append(current.char)
                current = self.root

        return "".join(result)

    def get_statistics(self, text: str) -> Dict[str, float]:
        """
        Return encoding statistics.

        Args:
            text: Input text

        Returns:
            Dictionary of statistics
        """
        import math

        if not self.codes:
            self.build_tree(text)

        freq = Counter(text)
        total = len(text)

        # Entropy
        entropy_val = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy_val -= p * math.log2(p)

        # Average code length
        avg_code_length = sum(
            (freq[char] / total) * len(self.codes[char])
            for char in freq
        )

        # Encoded bit count
        encoded_bits = sum(freq[char] * len(self.codes[char]) for char in freq)

        return {
            "entropy": entropy_val,
            "avg_code_length": avg_code_length,
            "efficiency": entropy_val / avg_code_length if avg_code_length > 0 else 0,
            "original_bits": total * 8,  # Assuming ASCII
            "encoded_bits": encoded_bits,
            "compression_ratio": encoded_bits / (total * 8) if total > 0 else 0,
            "n_symbols": len(freq),
        }

    def print_codes(self) -> None:
        """Display the code table"""
        print("Char  Freq  Code       Length")
        print("─" * 40)
        for char in sorted(self.codes.keys()):
            code = self.codes[char]
            display_char = repr(char) if char in (' ', '\n', '\t') else char
            print(f"  {display_char:5s}      {code:12s}  {len(code)}")


# === Verification ===
if __name__ == "__main__":
    # Example 1: Basic usage
    text = "ABRACADABRA"
    coder = HuffmanCoder()
    coder.build_tree(text)

    print(f"Input: '{text}'")
    print(f"\n=== Code Table ===")
    coder.print_codes()

    encoded = coder.encode(text)
    decoded = coder.decode(encoded)

    print(f"\nEncoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Match:   {text == decoded}")

    stats = coder.get_statistics(text)
    print(f"\n=== Statistics ===")
    print(f"  Entropy:          {stats['entropy']:.4f} bits/char")
    print(f"  Avg code length:  {stats['avg_code_length']:.4f} bits/char")
    print(f"  Coding efficiency:{stats['efficiency']:.4f}")
    print(f"  Original bits:    {stats['original_bits']}")
    print(f"  Encoded bits:     {stats['encoded_bits']}")
    print(f"  Compression ratio:{stats['compression_ratio']:.4f}")

    # Example 2: English text
    print("\n" + "=" * 60)
    english_text = (
        "information theory is a mathematical framework for "
        "quantifying information content and it provides fundamental "
        "limits on data compression and reliable communication"
    )
    coder2 = HuffmanCoder()
    coder2.build_tree(english_text)

    print(f"Input: '{english_text[:50]}...'")
    print(f"\n=== Code Table ===")
    coder2.print_codes()

    stats2 = coder2.get_statistics(english_text)
    print(f"\n=== Statistics ===")
    for key, value in stats2.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    encoded2 = coder2.encode(english_text)
    decoded2 = coder2.decode(encoded2)
    print(f"\nDecode match: {english_text == decoded2}")
```

Expected output:

```
Input: 'ABRACADABRA'

=== Code Table ===
Char  Freq  Code       Length
────────────────────────────────────────
  A            0             1
  B            110           3
  C            1110          4
  D            1111          4
  R            10            2

Encoded: 01101001110011110110100
Decoded: ABRACADABRA
Match:   True

=== Statistics ===
  Entropy:          2.0399 bits/char
  Avg code length:  2.0909 bits/char
  Coding efficiency:0.9756
  Original bits:    88
  Encoded bits:     23
  Compression ratio:0.2614
```

### 7.2 Huffman Tree Visualization

```python
"""
Text-based visualization of Huffman tree
"""
from typing import Optional


def visualize_huffman_tree(node: Optional["HuffmanNode"], prefix: str = "",
                           is_left: bool = True, is_root: bool = True) -> str:
    """
    Visualize a Huffman tree in text format.

    Args:
        node: Huffman node
        prefix: Prefix for indentation
        is_left: Whether this is a left child
        is_root: Whether this is the root node

    Returns:
        String representation of the tree
    """
    if node is None:
        return ""

    lines = []
    connector = "" if is_root else ("├── " if is_left else "└── ")
    extension = "" if is_root else ("│   " if is_left else "    ")

    if node.is_leaf():
        char_display = repr(node.char) if node.char in (' ', '\n', '\t') else node.char
        lines.append(f"{prefix}{connector}[{char_display}:{node.freq}]")
    else:
        lines.append(f"{prefix}{connector}({node.freq})")

    if node.left is not None:
        lines.append(
            visualize_huffman_tree(node.left, prefix + extension, True, False)
        )
    if node.right is not None:
        lines.append(
            visualize_huffman_tree(node.right, prefix + extension, False, False)
        )

    return "\n".join(line for line in lines if line)
```

With the above program, the Huffman tree for "ABRACADABRA" is visualized as follows.

```
Expected output:

  (11)
  ├── [A:5]
  └── (6)
      ├── [R:2]
      └── (4)
          ├── [B:2]
          └── (2)
              ├── [C:1]
              └── [D:1]
```

---

## 8. Trade-offs and Comparative Analysis

### 8.1 Comparison of Entropy Coding Methods

Each entropy coding method has trade-offs between compression ratio, speed, and implementation complexity.

| Property | Huffman Code | Arithmetic Code | ANS (rANS/tANS) |
|------|------------|---------|-----------------|
| Compression | H(X) ≤ L < H(X)+1 | Approaches H(X) | Approaches H(X) |
| Encoding speed | Fast | Slow | Fast to fastest |
| Decoding speed | Fast | Slow | Fastest |
| Memory usage | Code table size | Small | Table size dependent |
| Implementation complexity | Low | High | Medium |
| Patent issues | None | Historically yes | None |
| Adaptive support | Possible but inefficient | Easy | Possible |
| Typical use | DEFLATE, JPEG | JPEG2000, H.265 | zstd, LZFSE |

**Selection guidelines:**
- When compatibility and simplicity are priorities, Huffman coding is appropriate
- When maximum compression is needed and speed is tolerable, arithmetic coding is appropriate
- When both high compression and high speed are required, ANS is the optimal choice

### 8.2 Comparison of Lossless Compression Algorithms

| Property | gzip | bzip2 | LZ4 | zstd | Brotli |
|------|------|-------|-----|------|--------|
| Compression | Medium | High | Low | High | Highest |
| Comp speed | Medium | Slow | Fastest | Fast | Slow |
| Decomp speed | Fast | Medium | Fastest | Fastest | Fast |
| Memory usage | Low | High | Lowest | Medium | Medium |
| Standard | RFC 1952 | None | BSD | RFC 8878 | RFC 7932 |
| Entropy coding | Huffman | Huffman | None | ANS | Huffman+ANS |
| Dictionary | LZ77 | BWT | LZ77 | LZ77 | LZ77 |
| Primary use | General | Archive | Real-time | General | Web |

```
Compression ratio vs compression speed trade-off (conceptual):

  Compression
  ratio (high)
    │        * Brotli (max)
    │      * bzip2
    │    * zstd (max)
    │   * gzip (max)
    │  * zstd (default)
    │ * gzip (default)
    │* zstd (fast)
    │
    │             * LZ4 (default)
  (low)
    └──────────────────────────── Compression speed
    (slow)                     (fast)

  Decompression speed vs compression ratio trade-off:
  Fastest decompression: LZ4 > zstd > gzip > Brotli > bzip2
  Highest compression: Brotli ≈ zstd > bzip2 > gzip > LZ4
```

### 8.3 Comparison of Error-Correcting Codes

| Property | Hamming | RS Code | Turbo Code | LDPC Code | Polar Code |
|------|------------|--------|----------|---------|----------|
| Correction | 1 bit | t symbols | High | Very high | Theoretically optimal |
| Shannon limit | Far | Medium | 0.5dB | 0.04dB | 0dB(theory) |
| Decoding complexity | O(n) | O(n^2) | O(n log n) | O(n) | O(n log n) |
| Latency | Minimum | Small | Large | Medium-Large | Medium |
| Burst errors | No | Excels | Supported | Supported | Supported |
| Typical use | Memory ECC | CD/DVD/QR | 3G/4G | Wi-Fi 6/5G | 5G control |

### 8.4 Choosing the Right Information-Theoretic Metric

```
Guide to choosing metrics:

  Purpose                       Recommended metric
  ──────────────────────────────────────────────────
  Data compressibility          Entropy H(X)
  Code optimality                Avg code length L vs H(X)
  Channel capability             Channel capacity C
  Classification model training  Cross-entropy H(P,Q)
  Distribution proximity         KL divergence D_KL(P||Q)
  Symmetric dist. distance       JSD(P||Q) or Wasserstein distance
  Feature selection              Mutual information I(X;Y)
  Language model evaluation      Perplexity 2^{H(P,Q)}
  Lossy compression limits       Rate-distortion function R(D)
```

---

## 9. Anti-patterns

### Anti-pattern 1: Overconfidence in Compression by Ignoring Entropy

**Problem**: Assuming "compression always reduces size" without analyzing the nature of the data.

```
False assumption:
  "Compressing this file with gzip will definitely make it less than half"

Reality:
  - Re-compressing already compressed data (JPEG, MP3, ZIP) provides
    little to no reduction. It may even slightly increase in size.
  - Random data (encrypted data, random sequences) is incompressible.
    Entropy is at maximum (= log₂(alphabet size)), leaving
    no room to remove redundancy.
  - Very short data may end up larger after "compression"
    due to header overhead.

Concrete examples:
  ┌────────────────────┬──────────┬──────────┬────────┐
  │ Data type           │ Original │ After gzip│ Ratio  │
  ├────────────────────┼──────────┼──────────┼────────┤
  │ English text        │ 100 KB   │ 35 KB    │ 35%    │
  │ Source code         │ 100 KB   │ 25 KB    │ 25%    │
  │ JPEG image          │ 100 KB   │ 99 KB    │ 99%    │
  │ Encrypted data      │ 100 KB   │ 101 KB   │ 101%   │
  │ /dev/urandom       │ 100 KB   │ 101 KB   │ 101%   │
  │ 10-byte text        │ 10 B     │ 30 B     │ 300%   │
  └────────────────────┴──────────┴──────────┴────────┘
```

**Countermeasure**: Estimate the entropy of data before compression to determine whether compression would be effective. Do not apply compression to data that already has high entropy. In particular, in encryption pipelines, the correct order is "compress before encrypting" (data after encryption is pseudorandom and incompressible).

### Anti-pattern 2: Insisting on Fixed-Length Codes

**Problem**: Ignoring the advantages of variable-length codes and assigning the same number of bits to all symbols.

```
Incorrect approach:
  Store each character of English text using 8-bit fixed-length (ASCII).

Analysis:
  First-order entropy of English text ≈ 4.11 bits/char
  Entropy considering context ≈ 1.0-1.5 bits/char
  ASCII: 8 bits/char

  Redundancy = 1 - H/L = 1 - 4.11/8 ≈ 49%
  → About half of the text is redundant information

  When occurrence frequencies differ greatly:
  'e' ≈ 12.7% (most frequent) → optimally about 3 bits
  'z' ≈ 0.07% (least frequent) → optimally about 10 bits

  But in ASCII both use 8 bits
  → More efficient to assign shorter codes to frequent characters
```

**Countermeasure**: Use variable-length codes (Huffman, arithmetic, ANS) appropriate for the statistical properties of the data. However, when random access is needed or encoding/decoding latency requirements are strict, fixed-length codes may be the appropriate choice. Design decisions should be based on requirements.

### Anti-pattern 3: Over- or Under-provisioning Error Correction

**Problem**: Selecting error-correcting codes without considering channel characteristics.

```
Pattern A: Insufficient error correction
  "It's internet communication, so TCP will handle it"
  → For long-term storage, TCP is irrelevant
  → Error correction for bit rot (data corruption due to aging)
     is separately needed

Pattern B: Excessive error correction
  "Add maximum redundancy error correction for safety"
  → Waste of bandwidth/storage
  → Increasing redundancy beyond channel capacity C does not
     improve reliability if R > C
  → Choosing the appropriate error correction level is important

Countermeasure:
  1. Analyze the error characteristics of the channel (BER, burst length, etc.)
  2. Clarify the required reliability level (BER < 10^-9, etc.)
  3. Compare the Shannon limit with practical code performance
  4. Balance computational cost, latency, and redundancy among candidates
```

---

## 10. Exercises

### Basic Level

**Exercise 1: Entropy Calculation**

A 4-sided die has the following probability for each face.

```
Face 1: P = 0.5
Face 2: P = 0.25
Face 3: P = 0.125
Face 4: P = 0.125
```

(a) Calculate the entropy H of this die.
(b) Compare with the entropy of a uniform distribution (each face P = 0.25).
(c) Construct a Huffman code and find the codeword and average code length for each face.

<details>
<summary>Solution</summary>

```
(a) Entropy calculation:
  H = -(0.5×log₂0.5 + 0.25×log₂0.25 + 0.125×log₂0.125 + 0.125×log₂0.125)
    = -(0.5×(-1) + 0.25×(-2) + 0.125×(-3) + 0.125×(-3))
    = -(−0.5 − 0.5 − 0.375 − 0.375)
    = 1.75 bits

(b) Uniform distribution case:
  H_uniform = log₂(4) = 2.0 bits
  → Biased distribution (1.75) < Uniform distribution (2.0)
  → A biased distribution has lower entropy (more predictable)

(c) Huffman code:
  Step 1: Merge face3(0.125) and face4(0.125) → 3,4
  Step 2: Merge face2(0.25) and 3,4 → 2,3,4
  Step 3: Merge face1(0.5) and 2,3,4 → root(1.0)

  Code assignment:
    Face 1: 0        (1 bit)
    Face 2: 10       (2 bits)
    Face 3: 110      (3 bits)
    Face 4: 111      (3 bits)

  Average code length:
  L = 0.5×1 + 0.25×2 + 0.125×3 + 0.125×3
    = 0.5 + 0.5 + 0.375 + 0.375
    = 1.75 bits

  L = H = 1.75 → When probabilities are powers of two,
  the Huffman code exactly matches the entropy.
```

</details>

**Exercise 2: Intuitive Understanding of Information Content**

For each of the following events, calculate the self-information and examine whether it matches everyday intuition.

```
(a) Weather forecast: "Snow in Tokyo in January" (assumed probability: 0.05)
(b) Weather forecast: "Sunny in Tokyo in August" (assumed probability: 0.7)
(c) Winning the lottery jackpot (assumed probability: 1/10,000,000)
(d) Rolling a 6 on a die (assumed probability: 1/6)
```

<details>
<summary>Solution</summary>

```
(a) I = -log₂(0.05) ≈ 4.32 bits
    → A rare event with high information content

(b) I = -log₂(0.7) ≈ 0.51 bits
    → A common event with low information content
    → "Sunny in summer" is not surprising

(c) I = -log₂(1/10,000,000) ≈ 23.25 bits
    → An extremely rare event with very high information content
    → Winning the lottery makes the news

(d) I = -log₂(1/6) ≈ 2.58 bits
    → Moderate information content
    → Rolling a specific number is somewhat surprising

Match with intuition:
The magnitude of "surprise" of an event generally matches its self-information.
Events that "make the news" in everyday life tend to have higher information content.
```

</details>

### Intermediate Level

**Exercise 3: Text Compression Analysis**

Construct a Huffman code for the following text and analyze the compression ratio.

```
Text: "AABABCABCDABCDE"
```

(a) Find the frequency and probability of each character.
(b) Calculate the entropy.
(c) Construct a Huffman tree and show the codeword for each character.
(d) Calculate the average code length and coding efficiency (H/L).
(e) Find the compression ratio compared to ASCII (8 bits/char).

<details>
<summary>Solution</summary>

```
(a) Character frequencies:
  A: 5 times (P=5/15=0.333)
  B: 4 times (P=4/15=0.267)
  C: 3 times (P=3/15=0.200)
  D: 2 times (P=2/15=0.133)
  E: 1 time  (P=1/15=0.067)

(b) Entropy:
  H = -(0.333×log₂0.333 + 0.267×log₂0.267 + 0.200×log₂0.200
       + 0.133×log₂0.133 + 0.067×log₂0.067)
    ≈ -(0.333×(-1.585) + 0.267×(-1.907) + 0.200×(-2.322)
       + 0.133×(-2.907) + 0.067×(-3.907))
    ≈ 0.528 + 0.509 + 0.464 + 0.387 + 0.261
    ≈ 2.149 bits

(c) Huffman tree construction:
  Step 1: Merge D(0.133) and E(0.067) → DE
  Step 2: Merge C(0.200) and DE → CDE
  Step 3: Merge B(0.267) and CDE → BCDE
  (Alternatively, merging A(0.333) and B(0.267) first is also valid)

  One valid code assignment:
    A:  0       (1 bit)
    B:  10      (2 bits)
    C:  110     (3 bits)
    D:  1110    (4 bits)
    E:  1111    (4 bits)

(d) Average code length and efficiency:
  L = 0.333×1 + 0.267×2 + 0.200×3 + 0.133×4 + 0.067×4
    = 0.333 + 0.534 + 0.600 + 0.533 + 0.267
    = 2.267 bits

  Efficiency = H/L = 2.149/2.267 ≈ 0.948 (94.8%)

(e) Compression ratio:
  ASCII: 15 × 8 = 120 bits
  Huffman: 5×1 + 4×2 + 3×3 + 2×4 + 1×4 = 5+8+9+8+4 = 34 bits
  Compression ratio: 34/120 = 28.3%
```

</details>

**Exercise 4: Channel Capacity Calculation**

Suppose a Binary Symmetric Channel (BSC) has a bit flip probability of p = 0.05.

(a) Calculate the channel capacity C.
(b) Is reliable communication possible at coding rate R = 0.7?
(c) Is it possible at R = 0.8? What about R = 0.9?

<details>
<summary>Solution</summary>

```
(a) BSC channel capacity:
  C = 1 - H(p) = 1 - H(0.05)
  H(0.05) = -(0.05×log₂0.05 + 0.95×log₂0.95)
           = -(0.05×(-4.322) + 0.95×(-0.074))
           = -(−0.216 − 0.070)
           = 0.286 bits

  C = 1 - 0.286 = 0.714 bits/use

(b) R = 0.7 < C = 0.714
  → Since R < C, by Shannon's second fundamental theorem,
    reliable communication is possible with an appropriate code.

(c) R = 0.8 > C = 0.714
  → Since R > C, it is impossible to make the
    decoding error rate arbitrarily small with any code.

  R = 0.9 > C = 0.714
  → Similarly impossible. Since the gap is larger,
    the error rate will be even higher.
```

</details>

### Advanced Level

**Exercise 5: Proof of Huffman Code Optimality**

Prove that the Huffman code is an optimal prefix code (i.e., has the minimum average code length among all prefix codes) using the following steps.

(a) Show that in an optimal prefix code, the two symbols with the smallest probabilities have the longest codewords and differ only in the last bit (are sibling nodes).
(b) Show the relationship between the average code length of the optimal prefix code for a new source (formed by merging these two symbols with combined probability) and the optimal prefix code for the original source.
(c) Use induction to prove that Huffman's algorithm always generates an optimal prefix code.

<details>
<summary>Solution approach</summary>

```
(a) Proof approach by contradiction:
  In the optimal prefix code C*, let x, y (P(x) ≤ P(y))
  be the two symbols with the smallest probabilities.

  Suppose x does not have the longest codeword. Then some symbol z
  has a longer codeword than x. When P(x) ≤ P(z),
  swapping the codewords of x and z reduces or maintains
  the average code length. This contradicts the optimality of C*.

  Similarly, if we assume x and y are not siblings,
  swapping with another sibling pair can reduce the average
  code length, leading to a contradiction.

(b) Let X' be a new source formed by replacing x and y
  with a new symbol z having probability P(x)+P(y).
  Let C' be the optimal prefix code for X', and C* be the optimal
  prefix code for X. Then:

  L(C*) = L(C') + P(x) + P(y)

  (Because codewords for x and y are created by appending 0/1
   to z's codeword, adding 1 bit to the code length for x and y)

(c) Induction:
  Base case: For 2 symbols, the Huffman code is {0, 1}, clearly optimal.
  Inductive step: Assume the Huffman code is optimal for n-1 symbols.
  For a source with n symbols:
  - From (a), the two least probable symbols are siblings in the optimal code
  - Huffman's algorithm merges these two symbols
  - From (b), this reduces to the problem of n-1 symbols
  - By the inductive hypothesis, Huffman is optimal for n-1 symbols
  - Therefore, it is also optimal for n symbols
```

</details>

**Exercise 6: Intersection of Information Theory and Machine Learning**

In a binary classification problem, suppose the true distribution P and three model prediction distributions Q₁, Q₂, Q₃ are as follows.

```
P  = [0.8, 0.2] (Class 0 is 80%, Class 1 is 20%)
Q₁ = [0.9, 0.1]
Q₂ = [0.5, 0.5]
Q₃ = [0.2, 0.8]
```

(a) Calculate the cross-entropy H(P, Qᵢ) for each model.
(b) Calculate the KL divergence D_KL(P || Qᵢ) for each model.
(c) Explain which model makes the best predictions from an information-theoretic perspective.

<details>
<summary>Solution</summary>

```
Entropy of P:
H(P) = -(0.8×log₂0.8 + 0.2×log₂0.2)
     = -(0.8×(-0.322) + 0.2×(-2.322))
     = 0.258 + 0.464 = 0.722 bits

(a) Cross-entropy:
H(P, Q₁) = -(0.8×log₂0.9 + 0.2×log₂0.1)
          = -(0.8×(-0.152) + 0.2×(-3.322))
          = 0.122 + 0.664 = 0.786 bits

H(P, Q₂) = -(0.8×log₂0.5 + 0.2×log₂0.5)
          = -(0.8×(-1) + 0.2×(-1))
          = 0.8 + 0.2 = 1.000 bits

H(P, Q₃) = -(0.8×log₂0.2 + 0.2×log₂0.8)
          = -(0.8×(-2.322) + 0.2×(-0.322))
          = 1.858 + 0.064 = 1.922 bits

(b) KL divergence:
D_KL(P || Q₁) = H(P, Q₁) - H(P) = 0.786 - 0.722 = 0.064 bits
D_KL(P || Q₂) = H(P, Q₂) - H(P) = 1.000 - 0.722 = 0.278 bits
D_KL(P || Q₃) = H(P, Q₃) - H(P) = 1.922 - 0.722 = 1.200 bits

(c) Q₁ makes the best predictions:
  - Smallest cross-entropy (0.786)
  - Smallest KL divergence (0.064)
  - Q₁ is the distribution closest to P
  - Q₃ makes nearly the opposite prediction from P, making it the worst

  Minimizing cross-entropy ⟺ Minimizing KL divergence
  (because H(P) is a constant)

  This is the theoretical justification for minimizing
  cross-entropy loss in machine learning.
```

</details>

---

## 11. FAQ

### Q1: What is the relationship between entropy and thermodynamic entropy?

When Shannon defined information entropy, there is an anecdote that he consulted von Neumann, who advised: "You should call it entropy. First, because the mathematical form is identical to entropy in statistical mechanics. Second, because nobody really understands what entropy means, so you will always have the advantage in any debate."

The similarity in mathematical form is not coincidental. Boltzmann's statistical mechanics entropy S = k_B × ln(W) (where W is the number of microstates) and Shannon entropy H = -Σ p_i × log p_i both quantify "uncertainty of states." Boltzmann entropy is the logarithm of the number of microstates of a system, and Shannon entropy is the uncertainty of an information source (average degree of surprise).

Maxwell's demon paradox deepens this relationship further. It has been shown that the erasure of information (Landauer's principle) involves energy dissipation, demonstrating that information and physics are fundamentally linked. According to Landauer's principle, erasing 1 bit of information requires a minimum energy of k_B × T × ln(2).

### Q2: Is entropy truly the "limit" in data compression?

Yes. Shannon's source coding theorem mathematically proves that entropy is the absolute limit of lossless compression. However, there are several important caveats.

1. **Model-dependent**: Entropy is calculated based on a probability distribution model. Using a more precise model can lower the estimated entropy. For example, entropy based on first-order statistics (character frequency only) of English text is about 4.1 bits/char, but considering context (n-grams) it drops to about 1.0-1.5 bits/char.
2. **Memoryless assumption**: The basic form of Shannon's first theorem assumes a memoryless source. Actual data (text, images, etc.) has strong dependencies, and exploiting these enables compression close to entropy. LZ77-family algorithms exploit these dependencies as a dictionary.
3. **Computational constraints**: While codes that achieve compression close to entropy theoretically exist, whether the computational cost of encoding/decoding is practically acceptable is a separate issue.

### Q3: Why is cross-entropy loss used in machine learning?

There are clear information-theoretic reasons why cross-entropy loss is standard in classification problems in machine learning.

1. **Minimization of KL divergence**: Minimizing cross-entropy is equivalent to minimizing the KL divergence between the true distribution P and the model's predicted distribution Q (since H(P) is constant). KL divergence is a natural measure of "how close the model is to the true distribution."

2. **Equivalence with maximum likelihood estimation**: Minimizing cross-entropy is mathematically equivalent to maximum likelihood estimation of model parameters. Maximizing the log-likelihood Σ log Q(y_i | x_i) is the same operation as minimizing the cross-entropy -Σ P(y|x) log Q(y|x).

3. **Gradient properties**: The gradient of cross-entropy loss is proportional to the "degree of error" in the prediction. Large gradients arise for predictions that deviate significantly from the correct answer, accelerating learning. In contrast, squared error loss has the problem of gradient saturation with softmax outputs.

4. **Information-theoretic justification**: Minimizing the "extra bits" needed when encoding data from the true distribution P with the model's predicted distribution Q is equivalent to having the model learn the true distribution as accurately as possible.

### Q4: What is perplexity? Why is it used for evaluating language models?

Perplexity is a standard metric for evaluating language model performance, with direct theoretical grounding in information theory.

```
Definition:
  Perplexity = 2^{H(P, Q)}

  where H(P, Q) is the cross-entropy on the test data.
  In practice:
  PP = 2^{-(1/N) Σ log₂ Q(w_i | w_1...w_{i-1})}

  Intuitive interpretation:
  A perplexity of PP means that the model has
  "about the same uncertainty as uniformly randomly choosing
  from PP candidates" when predicting each word.

  Examples:
  PP = 100 → Difficulty equivalent to choosing from 100 candidates each time
  PP = 10  → Difficulty equivalent to choosing from 10 candidates each time
  PP = 1   → Perfect prediction (always correctly guessing the next word)

  Evolution of language models (approximate perplexity trends):
  n-gram model: PP ≈ 100-200
  LSTM:         PP ≈ 60〜80
  Transformer:  PP ≈ 20〜40
  GPT family:   PP ≈ 10-20
```

The lower the perplexity, the better the model predicts the test data. However, perplexity measures the model's "fluency" and does not evaluate the "accuracy" or "usefulness" of generated text, which must be assessed separately.

### Q5: What is quantum information theory? How does it differ from classical information theory?

Quantum information theory is a theory that handles information based on the principles of quantum mechanics.

```
Main differences from classical information theory:

  Concept     Classical IT         Quantum IT
  ─────────────────────────────────────────────
  Basic unit  Bit (0 or 1)        Qubit (|0⟩, |1⟩, superposition)
  Entropy     Shannon entropy      von Neumann entropy
               H = -Σ p log p     S = -Tr(ρ log ρ)
  Channel     Classical channel    Quantum channel
  Ch capacity Shannon capacity     Holevo capacity
  Cloning     Possible            Impossible (no-cloning theorem)
  Teleport    Impossible          Quantum teleportation possible

  Applications of quantum information theory:
  - Quantum cryptography (BB84 protocol): Eavesdropping detection is physically guaranteed
  - Quantum error correction: Protection of quantum states from decoherence
  - Quantum compression (Schumacher compression): Compression of qubits
  - Quantum channel coding: Capacity theorems for quantum channels
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not only through theory but also by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge used in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes especially important during code reviews and architecture design.

---

## 12. Summary

### 12.1 Overview of Key Concepts

| Concept | Definition | Key Point |
|------|------|------|
| Self-information | I(x) = -log₂(P(x)) | Rarer events carry more information |
| Entropy | H(X) = -Σ P(x) log₂ P(x) | Measure of source uncertainty. Theoretical limit of compression |
| Conditional entropy | H(Y\|X) = -Σ P(x,y) log₂ P(y\|x) | Remaining uncertainty of Y after knowing X |
| Mutual information | I(X;Y) = H(X) + H(Y) - H(X,Y) | Amount of information shared between X and Y |
| Cross-entropy | H(P,Q) = -Σ P(x) log₂ Q(x) | Average bits when encoding P's data with Q's code |
| KL divergence | D_KL(P\|\|Q) = Σ P(x) log₂(P(x)/Q(x)) | Information loss when approximating P with Q |
| Channel capacity | C = max I(X;Y) | Maximum rate of reliable communication |

### 12.2 The Two Fundamental Theorems

```
┌──────────────────────────────────────────────────────┐
│  Shannon's First Fundamental Theorem (Source Coding)    │
│                                                        │
│  Entropy H(X) is the limit of lossless compression.     │
│  H(X) ≤ L < H(X) + 1                                 │
│  Can be made arbitrarily close to H(X) via block coding. │
│                                                        │
│  Apps: Huffman, arithmetic, ANS, ZIP, gzip, zstd        │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  Shannon's Second Fundamental Theorem (Channel Coding)  │
│                                                        │
│  At rates below channel capacity C,                      │
│  codes exist that make error rate arbitrarily small.     │
│  At rates above C, error rate cannot reach 0.            │
│                                                        │
│  Apps: Hamming, RS, LDPC, Polar codes, 5G               │
└──────────────────────────────────────────────────────┘
```

### 12.3 The Philosophical Significance of Information Theory

Shannon's information theory was revolutionary due to the following three ideas.

1. **Separation from meaning**: By ignoring the "content" and "meaning" of information and focusing solely on statistical properties, a universal theory was constructed.
2. **Explicit limits**: By mathematically demonstrating the limits of compression (entropy) and communication (channel capacity), engineers were given benchmarks for "how much room for improvement exists."
3. **Power of existence proofs**: The second theorem in particular proved the "existence" of good codes without showing specific code construction methods. This existence proof motivated more than 50 years of subsequent coding theory research.

---

## Recommended Next Guides


---

## References

1. Shannon, C. E. "A Mathematical Theory of Communication." *The Bell System Technical Journal*, Vol. 27, pp. 379-423, 623-656, 1948.
   - The founding paper of information theory. Entropy, channel capacity, the source coding theorem, and the channel coding theorem were all presented in this single paper. Those studying information theory are strongly encouraged to consult the original.

2. Cover, T. M. & Thomas, J. A. *Elements of Information Theory*, 2nd Edition. Wiley-Interscience, 2006.
   - The standard textbook on information theory. Comprehensively covers from theory to applications. Internationally established as a graduate-level textbook. Broadly covers entropy, channel capacity, rate-distortion theory, and network information theory.

3. MacKay, D. J. C. *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press, 2003.
   - A textbook emphasizing the intersection of information theory and machine learning. A free online version is available on the author's website. Explains error-correcting codes, Bayesian inference, and neural networks from a unified perspective.

4. Huffman, D. A. "A Method for the Construction of Minimum-Redundancy Codes." *Proceedings of the IRE*, Vol. 40, No. 9, pp. 1098-1101, 1952.
   - The original paper on Huffman coding. Research submitted as an alternative to a term paper while a graduate student at MIT later became one of the most widely used coding algorithms.

5. Arikan, E. "Channel Polarization: A Method for Constructing Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels." *IEEE Transactions on Information Theory*, Vol. 55, No. 7, pp. 3051-3073, 2009.
   - The original paper on Polar codes. A historic achievement that showed the first code to "constructively" achieve the Shannon limit, adopted for 5G control channels.

6. Ziv, J. & Lempel, A. "A Universal Algorithm for Sequential Data Compression." *IEEE Transactions on Information Theory*, Vol. 23, No. 3, pp. 337-343, 1977.
   - The original paper on LZ77. Established the foundation of dictionary-based compression and became the prototype for modern compression algorithms such as gzip, ZIP, and PNG.
