# Fundamentals of Cryptography

> Modern cryptography is grounded in the computational complexity theory of computer science, and the existence of "hard-to-solve problems" guarantees its security. Cryptographic technology is the infrastructure underpinning the information-based society, enabling confidentiality of communications, data integrity, and identity verification.

## Learning Objectives

- [ ] Explain the historical development of cryptography and the transition from classical to modern cryptography
- [ ] Explain the differences between symmetric-key and public-key cryptography along with their mathematical foundations
- [ ] Understand the properties and applications of hash functions
- [ ] Explain the mechanism of TLS/HTTPS at an overview level
- [ ] Understand the mechanism and trust model of digital signatures
- [ ] Evaluate cryptographic security from a computational complexity perspective
- [ ] Explain the necessity and major candidates for post-quantum cryptography
- [ ] Avoid anti-patterns in cryptographic implementations


## Prerequisites

Having the following knowledge before reading this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Information Theory](./03-information-theory.md)

---

## 1. History and Development of Cryptography

Cryptography derives from the Greek words kryptos (hidden) and graphein (to write). Cryptographic technology has a history spanning thousands of years, evolving as a means of protecting information from military and diplomatic communications to modern internet security.

### 1.1 Classical Cryptography

Classical cryptography is a collective term for cryptographic methods used before the existence of computers. Most are based on "character substitution" or "character transposition."

#### Caesar Cipher (1st Century BC)

The simplest substitution cipher, said to have been used by Gaius Julius Caesar for military communications. It encrypts by shifting the alphabet by a fixed number of positions.

```
How the Caesar Cipher works:

  With shift = 3:

  Plaintext:  A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
  Ciphertext: D E F G H I J K L M N O P Q R S T U V W X Y Z A B C

  Encryption: HELLO -> KHOOR
  Decryption: KHOOR -> HELLO (shift 3 positions in reverse)

  Weakness: Key space is only 25 possibilities -> instantly breakable by brute force
```

#### Vigenere Cipher (16th Century)

A polyalphabetic cipher that uses multiple shift values as a key to overcome the weakness of the Caesar cipher. By repeatedly applying the key string, the same plaintext character is converted to different ciphertext characters.

```
Vigenere Cipher:

  Key: KEY
  Plaintext: HELLOWORLD

  H + K(10) = R
  E + E(4)  = I
  L + Y(24) = J
  L + K(10) = V
  O + E(4)  = S
  W + Y(24) = U
  O + K(10) = Y
  R + E(4)  = V
  L + Y(24) = J
  D + K(10) = N

  Ciphertext: RIJVSUYV JN

  Weakness: Once the key length is known, it can be broken by frequency analysis (Kasiski method)
```

#### Enigma Cipher Machine (20th Century)

An electromechanical cipher device used by the German military during World War II. By combining multiple rotors, with each rotor rotating after encrypting one character, it effectively achieved a polyalphabetic cipher. The key space reached approximately 1.59 x 10^20, making it extremely strong for its time.

However, Alan Turing's codebreaking team, building on prior Polish research, developed a decryption device called Bombe and exploited structural weaknesses to successfully break Enigma. This success is said to have significantly contributed to the Allied victory.

```
Structure of Enigma (simplified):

  Input -> Plugboard -> Rotor 1 -> Rotor 2 -> Rotor 3
                                                    |
                                                Reflector (UKW)
                                                    |
       <- Plugboard <- Rotor 1 <- Rotor 2 <- Rotor 3
  Output

  Features:
  - Rotor 1 advances one step per character (odometer-style)
  - A character is never encrypted to itself (structural weakness)
  - Daily key settings (rotor selection, initial position, plug wiring)
```

### 1.2 Transition to Modern Cryptography

The turning points from classical to modern cryptography were Claude Shannon's "Communication Theory of Secrecy Systems" in 1949 and Diffie and Hellman's "New Directions in Cryptography" in 1976.

**Shannon's Information-Theoretic Security (1949)**

Shannon proved that the one-time pad (where the key is a random string of the same length as the plaintext, used only once) is information-theoretically secure. This means that if the key is sufficiently long, no amount of computational power can extract plaintext information from the ciphertext. However, it is impractical because the key must be as long as the plaintext.

**Kerckhoffs's Principle (proposed in 1883, still applicable today)**

The security of a cryptographic system should depend only on the secrecy of the key, not on the secrecy of the algorithm. All modern cryptographic algorithms (AES, RSA, etc.) are publicly available, and their security is based on key secrecy.

```
Timeline of Cryptographic Development:

  Ancient       Caesar Cipher (substitution)
     |
  16th Century  Vigenere Cipher (polyalphabetic)
     |
  1883          Kerckhoffs's Principle
     |
  1918          One-Time Pad (Vernam Cipher)
     |
  1940s         Enigma decryption (Turing)
     |
  1949          Shannon's secrecy communication theory
     |
  1976          Diffie-Hellman Key Exchange
     |
  1977          DES (symmetric cipher standard) / RSA (public-key cryptography)
     |
  1991          PGP (Pretty Good Privacy)
     |
  2000          AES (standardized as DES successor)
     |
  2008          Bitcoin (application of cryptographic technology)
     |
  2018          TLS 1.3 (modern protocol)
     |
  2024          NIST Post-Quantum Cryptography Standards announced
```

### 1.3 Basic Concepts and Terminology of Cryptography

Essential basic terminology for studying cryptography is organized below.

| Term | Definition | Example |
|------|-----------|---------|
| Plaintext | Original data before encryption | "Hello, World!" |
| Ciphertext | Data after encryption | "7f83b1657ff1fc..." |
| Key | Secret parameter used for encryption/decryption | 256-bit random byte sequence |
| Encryption | Operation that converts plaintext to ciphertext | Encrypt with AES-256-GCM |
| Decryption | Operation that restores ciphertext to plaintext | Decrypt with the correct key |
| Cipher algorithm | Procedure for encryption/decryption | AES, RSA, ChaCha20 |
| Key space | Total number of possible keys | AES-256: 2^256 possibilities |
| Cipher suite | Combination of multiple cryptographic algorithms | TLS_AES_256_GCM_SHA384 |

```
Basic Model of a Cryptographic System:

  Sender (Alice)                             Receiver (Bob)
  +------------+                          +------------+
  | Plaintext M |                          | Plaintext M |
  |      |      |                          |      ^      |
  | Encryption  |      +-----------+      | Decryption  |
  | E(K, M)     |----->| Ciphertext C|----->| D(K, C)     |
  |      |      |      +-----------+      |      ^      |
  |  Key K      |                          |  Key K      |
  +------------+                          +------------+

  Attacker (Eve) can intercept ciphertext C,
  but cannot recover plaintext M without knowing key K

  Correctness condition: D(K, E(K, M)) = M
```

---

## 2. Symmetric-Key Cryptography

Symmetric-key cryptography uses the same key for both encryption and decryption. Compared to public-key cryptography, it has lower computational cost and can encrypt large amounts of data at high speed, making it the primary method still used today for data encryption.

### 2.1 Block Ciphers and Stream Ciphers

Symmetric-key cryptography is broadly classified into "block ciphers" and "stream ciphers."

```
Classification:

  Symmetric-Key Cryptography
  +-- Block Cipher: Processes in fixed-length blocks
  |   +-- DES (56-bit key, deprecated)
  |   +-- 3DES (equivalent to 112-bit, deprecated)
  |   +-- AES (128/192/256-bit key, current standard)
  |
  +-- Stream Cipher: Processes in 1-byte or 1-bit units
      +-- RC4 (deprecated, prohibited in TLS)
      +-- ChaCha20 (currently recommended, adopted in TLS 1.3)
```

**Block ciphers** divide the plaintext into fixed-size blocks (e.g., 128 bits for AES) and encrypt each block. A "mode of operation" is needed to handle data longer than the block length.

**Stream ciphers** generate a pseudorandom sequence (keystream) from the key and encrypt by XORing it with the plaintext. They are suitable for real-time communication.

### 2.2 AES (Advanced Encryption Standard)

AES is a block cipher standardized by NIST in 2001, based on the Rijndael algorithm designed by Belgian cryptographers Joan Daemen and Vincent Rijmen. It is currently the most widely used symmetric-key cipher algorithm in the world.

```
AES Basic Parameters:

  +----------------+----------+----------+----------+
  |                | AES-128  | AES-192  | AES-256  |
  +----------------+----------+----------+----------+
  | Key length     | 128 bit  | 192 bit  | 256 bit  |
  | Block length   | 128 bit  | 128 bit  | 128 bit  |
  | Number of rounds| 10      | 12       | 14       |
  | Key space      | 2^128    | 2^192    | 2^256    |
  +----------------+----------+----------+----------+

  Processing per round (4 steps):
  1. SubBytes   -- Non-linear byte substitution via S-Box
  2. ShiftRows  -- Cyclic shift by row
  3. MixColumns -- Galois field arithmetic by column (except final round)
  4. AddRoundKey -- XOR with the round key
```

#### AES Round Processing

```
AES Round Processing Flow:

  +------------------+
  |   Input Block    |  4x4 byte matrix (State)
  |  (128 bits)      |
  +--------+---------+
           |
  +------------------+
  |    SubBytes      |  Non-linear transformation of each byte via S-Box
  |  (Non-linear     |  -> Resistance to linear cryptanalysis
  |   substitution)  |
  +--------+---------+
           |
  +------------------+
  |    ShiftRows     |  Row 0: No shift
  |  (Row shift)     |  Row 1: Cyclic left shift by 1 byte
  |                  |  Row 2: Cyclic left shift by 2 bytes
  |                  |  Row 3: Cyclic left shift by 3 bytes
  +--------+---------+
           |
  +------------------+
  |   MixColumns     |  Transform each column via
  |  (Column mixing) |  matrix multiplication over GF(2^8)
  |                  |  -> Achieves diffusion
  +--------+---------+
           |
  +------------------+
  |   AddRoundKey    |  XOR with the round key
  |  (Key addition)  |
  +--------+---------+
           |
  +------------------+
  |   Output Block   |
  +------------------+
```

### 2.3 Modes of Operation

To actually use a block cipher, a "mode of operation" that defines how to process data exceeding the block length is essential. The choice of mode directly affects security, so correct understanding is required.

#### ECB Mode (Electronic Codebook) --- Do NOT Use

ECB is the simplest mode that encrypts each block independently, but since identical plaintext blocks always produce identical ciphertext blocks, patterns are leaked. This is a critical vulnerability, and ECB must never be used for practical encryption.

```
ECB Mode Problem (Penguin Problem):

  Original image (bitmap)     After ECB encryption      After CBC encryption
  +---------------+          +---------------+        +---------------+
  |   #######     |          |   #######     |        | @$#!@$#!@$#   |
  |  ##!!!!###    |          |  ##!!!!###    |        | !$@#$!@#!$@   |
  | ##!!!!!!##    |          | ##!!!!!!##    |        | @#!$@#!$@#!   |
  |  ##!!!!##     |          |  ##!!!!##     |        | $!@#!$@#!$@   |
  |   ######      |          |   ######      |        | #!$@#!$@#!$   |
  +---------------+          +---------------+        +---------------+
  Penguin shape visible       Shape remains intact!     Completely randomized

  -> In ECB, blocks of the same color produce the same ciphertext,
    so the image outline is preserved even after encryption
```

#### CBC Mode (Cipher Block Chaining)

Before encrypting each block, it is XORed with the ciphertext of the previous block. An initialization vector (IV) is used for the first block. The same plaintext produces different ciphertext with different IVs.

```
CBC Mode:

  IV -+
      |
  P1 -XOR-> E(K) --> C1 -+
                           |
              P2 -----XOR-> E(K) --> C2 -+
                                          |
                              P3 -----XOR-> E(K) --> C3

  Encryption: Ci = E(K, Pi XOR C_{i-1}),  C0 = IV
  Decryption: Pi = D(K, Ci) XOR C_{i-1}

  Note: May be vulnerable to padding oracle attacks
```

#### GCM Mode (Galois/Counter Mode) --- Currently Recommended

GCM is an authenticated encryption (AEAD: Authenticated Encryption with Associated Data) that combines CTR (Counter) mode with GHASH (Galois Hash). It can detect data tampering simultaneously with encryption and is used as standard in TLS 1.3.

```
GCM Mode (AEAD):

  Key K, IV (96-bit recommended), Plaintext P, Associated Data A

  +---------------------------------------------+
  |  Encrypt with CTR mode:                      |
  |    Counter = IV || 0...01                    |
  |    Ci = Pi XOR E(K, Counter + i)             |
  |                                              |
  |  Generate authentication tag with GHASH:     |
  |    T = GHASH(H, A, C) XOR E(K, IV||0..0)    |
  |    H = E(K, 0^128)                           |
  +---------------------------------------------+

  Output: (Ciphertext C, Authentication Tag T)

  During decryption: Verify authentication tag before decrypting
  -> Does not decrypt tampered ciphertext (secure)
```

### 2.4 ChaCha20-Poly1305

ChaCha20 is a stream cipher designed by Daniel J. Bernstein, combined with Poly1305 to achieve AEAD. It provides security equivalent to AES-GCM while operating at high speed even in environments without AES hardware acceleration (AES-NI), such as mobile devices. It is adopted alongside AES-256-GCM in TLS 1.3.

```
ChaCha20 Features:

  - 256-bit key + 96-bit nonce + 32-bit counter
  - 20 rounds of Quarter Round operations
  - Composed only of addition, XOR, and rotation (ARX structure)
    -> Constant timing against side-channel attacks
  - Fast even in software implementation

  AES-GCM vs ChaCha20-Poly1305:

  +------------------+-------------+--------------------+
  |                  | AES-256-GCM | ChaCha20-Poly1305  |
  +------------------+-------------+--------------------+
  | Key length       | 256 bit     | 256 bit            |
  | Nonce length     | 96 bit      | 96 bit             |
  | With HW accel.   | Very fast   | Fast               |
  | Without HW accel.| Slow        | Fast (no difference)|
  | TLS 1.3          | Standard    | Standard           |
  | Timing safety    | Impl. dep.  | Structurally safe  |
  +------------------+-------------+--------------------+
```

### 2.5 Learning Symmetric-Key Cryptography with Python

The following code is a complete implementation example of AES-256-GCM encryption/decryption using Python's `cryptography` library.

```python
"""
AES-256-GCM encryption/decryption demo
Dependency: pip install cryptography
"""
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt_aes_gcm(plaintext: bytes, key: bytes, aad: bytes = b"") -> tuple[bytes, bytes]:
    """
    Encrypt plaintext with AES-256-GCM.

    Args:
        plaintext: Plaintext data to encrypt
        key: 256-bit (32-byte) key
        aad: Associated data (not encrypted but included in authentication)

    Returns:
        (nonce, ciphertext): Tuple of nonce and ciphertext+authentication tag
    """
    # Nonce must be unique per encryption (96-bit recommended)
    nonce = os.urandom(12)  # 96 bits = 12 bytes
    aesgcm = AESGCM(key)
    # Authentication tag (16 bytes) is automatically appended to ciphertext
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return nonce, ciphertext


def decrypt_aes_gcm(nonce: bytes, ciphertext: bytes, key: bytes, aad: bytes = b"") -> bytes:
    """
    Decrypt ciphertext with AES-256-GCM.

    Args:
        nonce: Nonce used during encryption
        ciphertext: Ciphertext + authentication tag
        key: 256-bit (32-byte) key
        aad: Same associated data used during encryption

    Returns:
        Decrypted plaintext

    Raises:
        cryptography.exceptions.InvalidTag: If the authentication tag is invalid
    """
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, aad)


def main():
    # Securely generate a 256-bit key
    key = AESGCM.generate_key(bit_length=256)
    print(f"Key (hex): {key.hex()}")
    print(f"Key length: {len(key) * 8} bits")

    # Plaintext and associated data
    plaintext = "Cryptography is the foundation of information security.".encode("utf-8")
    aad = b"metadata:version=1"

    # Encryption
    nonce, ciphertext = encrypt_aes_gcm(plaintext, key, aad)
    print(f"\nNonce (hex): {nonce.hex()}")
    print(f"Ciphertext (hex): {ciphertext.hex()}")
    print(f"Ciphertext length: {len(ciphertext)} bytes (plaintext {len(plaintext)} + tag 16)")

    # Decryption
    decrypted = decrypt_aes_gcm(nonce, ciphertext, key, aad)
    print(f"\nDecrypted: {decrypted.decode('utf-8')}")
    assert decrypted == plaintext, "Decrypted result does not match original plaintext"

    # Tampering detection demo: modify 1 byte of ciphertext
    tampered = bytearray(ciphertext)
    tampered[0] ^= 0xFF  # Flip the first byte
    try:
        decrypt_aes_gcm(nonce, bytes(tampered), key, aad)
        print("Error: Tampering was not detected")
    except Exception as e:
        print(f"\nTampering detected: {type(e).__name__}")
        print("-> GCM's authentication tag detected ciphertext tampering")


if __name__ == "__main__":
    main()
```

```
Example output:

  Key (hex): a1b2c3d4e5f6...(64 hex characters)
  Key length: 256 bits

  Nonce (hex): 1a2b3c4d5e6f7a8b9c0d1e2f
  Ciphertext (hex): 8f3a2b1c4d5e...(plaintext + 16-byte tag)
  Ciphertext length: 70 bytes (plaintext 54 + tag 16)

  Decrypted: Cryptography is the foundation of information security.

  Tampering detected: InvalidTag
  -> GCM's authentication tag detected ciphertext tampering
```

---

## 3. Public-Key Cryptography (Asymmetric Cryptography)

Public-key cryptography (asymmetric cryptography) is a revolutionary concept published by Whitfield Diffie and Martin Hellman in 1976. It uses different keys for encryption and decryption, solving the greatest challenge of symmetric-key cryptography: the "key distribution problem."

### 3.1 The Key Distribution Problem and the Birth of Public-Key Cryptography

In symmetric-key cryptography, both communicating parties need to securely share the same key. However, they want to use encryption precisely because there is no secure communication channel -- a chicken-and-egg problem.

```
Key Distribution Problem:

  Number of keys needed for n parties to communicate with each other:

  Symmetric-key: n(n-1)/2 keys needed
    2 people ->   1 key
    10 people ->  45 keys
    100 people -> 4,950 keys
    1000 people -> 499,500 keys

  Public-key: Each person only needs 1 key pair (public key + private key)
    2 people ->   2 key pairs
    10 people ->  10 key pairs
    100 people -> 100 key pairs
    1000 people -> 1,000 key pairs

  -> Key management complexity improved from O(n^2) to O(n)
```

### 3.2 RSA Cryptography

RSA was the first practical public-key cryptographic algorithm, published in 1977 by Ron Rivest, Adi Shamir, and Leonard Adleman. Its security is based on the difficulty of factoring large integers.

```
RSA Key Generation and Encryption (Overview):

  Key Generation:
  1. Choose large primes p, q (at least 1024 bits each)
  2. Compute n = p x q
  3. Compute phi(n) = (p-1)(q-1) (Euler's totient function)
  4. Choose e satisfying gcd(e, phi(n)) = 1 (typically e = 65537)
  5. Find d satisfying e x d = 1 (mod phi(n))

  Public key: (n, e)
  Private key: (n, d)

  Encryption: c = m^e mod n
  Decryption: m = c^d mod n

  Security basis:
  - Finding p, q from n (integer factorization) is computationally hard
  - RSA-2048: n is approximately a 617-digit integer
  - Cannot be factored with current classical computers
```

#### RSA Numerical Example (small values for educational purposes)

```
RSA Numerical Example:

  1. Prime selection: p = 61, q = 53
  2. n = 61 x 53 = 3233
  3. phi(n) = (61-1)(53-1) = 60 x 52 = 3120
  4. e = 17  (verify gcd(17, 3120) = 1)
  5. d = 2753  (17 x 2753 = 46801 = 15 x 3120 + 1)

  Public key: (3233, 17)
  Private key: (3233, 2753)

  Encryption (m = 65 = 'A'):
    c = 65^17 mod 3233 = 2790

  Decryption:
    m = 2790^2753 mod 3233 = 65

  * In practice, RSA uses n of 2048 bits (617+ digits) or more
```

### 3.3 Elliptic Curve Cryptography (ECC)

Elliptic Curve Cryptography is a cryptographic scheme based on the difficulty of the discrete logarithm problem on elliptic curves. It achieves equivalent security to RSA with much shorter key lengths and better computational efficiency, making it widely adopted in modern systems.

```
Elliptic Curve Equation:

  y^2 = x^3 + ax + b  (mod p)

  Point addition on the curve (P + Q = R):
  - The line through points P and Q intersects the curve at a third point,
    which is reflected across the x-axis

  Scalar multiplication:
  - nP = P + P + ... + P (n additions)

  Elliptic Curve Discrete Logarithm Problem (ECDLP):
  - Given Q = nP, finding n is computationally hard
  - This is the security basis

  Key Length Comparison:
  +-------------+----------+----------+----------------+
  | Security    | RSA key  | ECC key  | Ratio          |
  | Level       | length   | length   | (RSA/ECC)      |
  +-------------+----------+----------+----------------+
  |  80 bit     | 1024 bit | 160 bit  | 6.4x           |
  | 112 bit     | 2048 bit | 224 bit  | 9.1x           |
  | 128 bit     | 3072 bit | 256 bit  | 12x            |
  | 192 bit     | 7680 bit | 384 bit  | 20x            |
  | 256 bit     |15360 bit | 521 bit  | 29.5x          |
  +-------------+----------+----------+----------------+

  Representative Curves:
  - P-256 (secp256r1): NIST recommended, widely used in TLS
  - Curve25519: Designed by Daniel Bernstein, used in SSH/Signal
  - secp256k1: Used in Bitcoin
```

### 3.4 Diffie-Hellman Key Exchange

Diffie-Hellman (DH) key exchange is a protocol for two parties to share a common secret key over an insecure communication channel. It does not perform encryption itself, but rather securely generates key material for use with symmetric-key cryptography.

```
Diffie-Hellman Key Exchange:

  Public parameters: Prime p, Generator g

  Alice                                   Bob
  +-------------------+                 +-------------------+
  | Choose secret a    |                 | Choose secret b    |
  | A = g^a mod p      |                 | B = g^b mod p      |
  |                    |                 |                    |
  |         --- Send A ---------->       |
  |         <-- Send B ----------        |
  |                    |                 |                    |
  | s = B^a mod p      |                 | s = A^b mod p      |
  |   = (g^b)^a mod p  |                 |   = (g^a)^b mod p  |
  |   = g^(ab) mod p   |                 |   = g^(ab) mod p   |
  +-------------------+                 +-------------------+

  Shared secret: s = g^(ab) mod p (they match)

  The attacker knows A = g^a mod p and B = g^b mod p,
  but cannot efficiently determine a or b (discrete logarithm problem)

  Modern implementation: ECDH (Elliptic Curve Diffie-Hellman)
  -> TLS 1.3 key exchange uses ECDH (X25519 or P-256)
```

### 3.5 Learning Public-Key Cryptography with Python

The following code demonstrates Elliptic Curve Diffie-Hellman (ECDH) key exchange.

```python
"""
ECDH key exchange and key derivation with HKDF demo
Dependency: pip install cryptography
"""
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def ecdh_key_exchange():
    """
    Demo of ECDH key exchange using X25519.
    Alice and Bob generate a shared key over an insecure communication channel.
    """
    # Generate Alice's key pair
    alice_private = X25519PrivateKey.generate()
    alice_public = alice_private.public_key()

    # Generate Bob's key pair
    bob_private = X25519PrivateKey.generate()
    bob_public = bob_private.public_key()

    # Key exchange: Derive shared secret from partner's public key and own private key
    alice_shared = alice_private.exchange(bob_public)
    bob_shared = bob_private.exchange(alice_public)

    # Verify that both shared secrets match
    assert alice_shared == bob_shared, "Shared secrets do not match"
    print(f"Shared secret (hex): {alice_shared.hex()}")
    print(f"Shared secret length: {len(alice_shared)} bytes (256 bits)")

    # Derive encryption key from shared secret using HKDF
    # Best practice: Use a KDF rather than using the shared secret directly as a key
    derived_key_alice = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"handshake data",
    ).derive(alice_shared)

    derived_key_bob = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"handshake data",
    ).derive(bob_shared)

    assert derived_key_alice == derived_key_bob
    print(f"\nDerived key (hex): {derived_key_alice.hex()}")
    print(f"Derived key length: {len(derived_key_alice)} bytes")
    print("-> Use this key with symmetric-key ciphers such as AES-256-GCM")


if __name__ == "__main__":
    ecdh_key_exchange()
```

```
Example output:

  Shared secret (hex): 3a1b4c2d5e6f...(64 hex characters)
  Shared secret length: 32 bytes (256 bits)

  Derived key (hex): 7f8e9d0a1b2c...(64 hex characters)
  Derived key length: 32 bytes
  -> Use this key with symmetric-key ciphers such as AES-256-GCM
```

---

## 4. Hash Functions

A cryptographic hash function is a one-way function that transforms arbitrary-length input data into a fixed-length hash value (digest). Unlike encryption, it is computationally infeasible to recover the original data from a hash value, and it is widely used as the foundation for data integrity verification, password storage, and digital signatures.

### 4.1 Required Properties of Hash Functions

There are three main properties that a cryptographic hash function must satisfy.

```
Three Security Requirements of Hash Functions:

  1. Preimage resistance
     Given a hash value h, it is computationally hard
     to find m such that H(m) = h

     Cannot find m from h (one-wayness)

  2. Second preimage resistance
     Given a message m1, it is computationally hard
     to find m2 != m1 such that H(m1) = H(m2)

     Even knowing m1, cannot find m2 with the same hash

  3. Collision resistance
     It is computationally hard to find any pair
     (m1, m2) such that H(m1) = H(m2) and m1 != m2

     * Birthday attack: Can find collisions for n-bit hash in O(2^(n/2))
       -> Collision resistance of SHA-256 is equivalent to 128 bits

  Additional desirable properties:
  - Avalanche effect: A 1-bit change in input changes approximately 50% of the output
  - Efficiency: Hash computation should be fast
```

### 4.2 Major Hash Algorithms

```
Hash Algorithm Comparison:

  +------------+------------+----------+--------------------------+
  | Algorithm  | Output len | Status   | Primary use              |
  +------------+------------+----------+--------------------------+
  | MD5        | 128 bit    | Deprecated| Legacy systems          |
  | SHA-1      | 160 bit    | Deprecated| Git (remains for compat.)|
  | SHA-256    | 256 bit    | Recommended| TLS, Bitcoin, general  |
  | SHA-384    | 384 bit    | Recommended| High-security req.     |
  | SHA-512    | 512 bit    | Recommended| High-security req.     |
  | SHA3-256   | 256 bit    | Recommended| SHA-2 alternative      |
  | BLAKE3     | 256 bit    | Recommended| High-speed use cases   |
  +------------+------------+----------+--------------------------+

  Reasons for deprecation:
  - MD5: Collision attacks demonstrated in 2004. Can generate collision pairs in seconds
  - SHA-1: Google demonstrated collision with SHAttered attack in 2017
```

#### SHA-256 Structure (Merkle-Damgard Construction)

```
SHA-256 Processing Flow:

  Input Message
       |
  Padding (adjust message length to a multiple of 512)
       |
  Split into 512-bit blocks: M1, M2, ..., Mn
       |
  +----------------------------------------------+
  |                                              |
  |  IV --> [Compression] --> [Compression] --> ... --> Hash value
  |              ^                ^              |
  |             M1               M2              |
  |                                              |
  |  Merkle-Damgard Construction:                |
  |  H0 = IV                                    |
  |  Hi = f(H_{i-1}, Mi)  (compression function)|
  |  Hash value = Hn                             |
  +----------------------------------------------+

  SHA-256 Compression Function:
  - 8 32-bit working variables (a, b, c, d, e, f, g, h)
  - 64 rounds of operations
  - Each round uses addition, rotation, and logical operations
```

### 4.3 Password Hashing

For password storage, dedicated password hash functions must be used instead of general-purpose hash functions (such as SHA-256). General-purpose hash functions are designed to be "fast," but for password hashing, being "intentionally slow" is important. This is to delay brute-force attacks by attackers.

```
Password Hashing Requirements:

  NG: SHA-256(password)
  -> GPU can compute billions of hashes per second
  -> Instantly cracked with rainbow tables (precomputed hash dictionaries)

  NG: SHA-256(password + salt)
  -> Salt prevents rainbow tables, but still too fast

  OK: Dedicated password hash functions
  - bcrypt:  Blowfish-based, speed adjustable via cost parameter
  - scrypt:  Requires memory consumption (GPU/ASIC resistant)
  - Argon2:  2015 PHC winner, latest recommendation
    - Argon2id: Side-channel resistant + GPU resistant (recommended)

  +------------+--------------+--------------+--------------+
  |            | bcrypt       | scrypt       | Argon2id     |
  +------------+--------------+--------------+--------------+
  | Year       | 1999         | 2009         | 2015         |
  | CPU resist.| Good         | Good         | Excellent    |
  | GPU resist.| Fair         | Good         | Excellent    |
  | Memory req.| 4KB fixed    | Variable     | Variable     |
  | Recommend. | Legacy compat| Sufficiently | Most         |
  |            |              | secure       | recommended  |
  +------------+--------------+--------------+--------------+
```

### 4.4 HMAC (Hash-based Message Authentication Code)

HMAC is a method that combines a hash function with a secret key to generate a message authentication code (MAC). It can simultaneously verify data integrity and authenticity, and is widely used in API authentication (e.g., AWS Signature V4).

```
HMAC Structure:

  HMAC(K, m) = H((K' XOR opad) || H((K' XOR ipad) || m))

  K' = Key (adjusted to block size)
  opad = Block of repeated 0x5c
  ipad = Block of repeated 0x36

  +-----------------------------+
  | Inner hash:                 |
  |   inner = H((K' XOR ipad) || m) |
  |                             |
  | Outer hash:                 |
  |   HMAC = H((K' XOR opad) || inner) |
  +-----------------------------+

  Features:
  - More secure than simple H(K || m) or H(m || K)
  - Prevents length extension attacks
  - Widely used in TLS, IPsec, JWT (HS256), etc.
```

### 4.5 Learning Hash Functions with Python

```python
"""
Hash function and HMAC demo
Dependencies: Standard library only (hashlib, hmac)
"""
import hashlib
import hmac
import os


def hash_demo():
    """Demo of SHA-256 hash computation and avalanche effect."""
    # Basic hash computation
    message = "Hello, Cryptography!"
    hash_value = hashlib.sha256(message.encode()).hexdigest()
    print(f"Input:   '{message}'")
    print(f"SHA-256: {hash_value}")
    print(f"Output length: {len(hash_value)} chars = {len(hash_value) * 4} bits")

    # Avalanche effect: change just 1 character
    message_modified = "Hello, Cryptography?"  # '!' -> '?'
    hash_modified = hashlib.sha256(message_modified.encode()).hexdigest()
    print(f"\nInput:   '{message_modified}'")
    print(f"SHA-256: {hash_modified}")

    # Calculate number of differing bits
    original_bits = bin(int(hash_value, 16))[2:].zfill(256)
    modified_bits = bin(int(hash_modified, 16))[2:].zfill(256)
    diff_bits = sum(a != b for a, b in zip(original_bits, modified_bits))
    print(f"\nDiffering bits: {diff_bits} / 256 ({diff_bits/256*100:.1f}%)")
    print("-> A single character change alters approximately half the bits (avalanche effect)")


def hmac_demo():
    """Demo of HMAC-SHA256 computation and verification."""
    # Generate HMAC key
    key = os.urandom(32)
    message = b"Transfer $100 to Bob"

    # HMAC computation
    mac = hmac.new(key, message, hashlib.sha256).hexdigest()
    print(f"\nMessage: {message.decode()}")
    print(f"HMAC-SHA256: {mac}")

    # Verification: correct message
    mac_verify = hmac.new(key, message, hashlib.sha256).hexdigest()
    is_valid = hmac.compare_digest(mac, mac_verify)
    print(f"\nVerification (correct message): {'OK' if is_valid else 'NG'}")

    # Verification: tampered message
    tampered_message = b"Transfer $900 to Bob"
    mac_tampered = hmac.new(key, tampered_message, hashlib.sha256).hexdigest()
    is_valid_tampered = hmac.compare_digest(mac, mac_tampered)
    print(f"Verification (tampered message): {'OK' if is_valid_tampered else 'NG (tampering detected)'}")

    # compare_digest performs timing-safe comparison
    # String == comparison is vulnerable to timing side-channel attacks
    print("\nNote: hmac.compare_digest() uses timing-safe comparison")


def multiple_hash_algorithms():
    """Comparison of multiple hash algorithms."""
    data = b"The quick brown fox jumps over the lazy dog"
    algorithms = ["md5", "sha1", "sha256", "sha384", "sha512", "sha3_256"]

    print(f"\nInput: {data.decode()}\n")
    print(f"{'Algorithm':<12} {'Output':>6}  Hash value (first 32 chars)")
    print("-" * 70)

    for algo_name in algorithms:
        h = hashlib.new(algo_name, data)
        digest = h.hexdigest()
        bits = h.digest_size * 8
        print(f"{algo_name:<12} {bits:>4}bit  {digest[:32]}...")


if __name__ == "__main__":
    hash_demo()
    hmac_demo()
    multiple_hash_algorithms()
```

```
Example output:

  Input:   'Hello, Cryptography!'
  SHA-256: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e7...
  Output length: 64 chars = 256 bits

  Input:   'Hello, Cryptography?'
  SHA-256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd...

  Differing bits: 131 / 256 (51.2%)
  -> A single character change alters approximately half the bits (avalanche effect)

  Message: Transfer $100 to Bob
  HMAC-SHA256: 5d41402abc4b2a76b9719d911017c592ae2...

  Verification (correct message): OK
  Verification (tampered message): NG (tampering detected)

  Note: hmac.compare_digest() uses timing-safe comparison

  Input: The quick brown fox jumps over the lazy dog

  Algorithm    Output  Hash value (first 32 chars)
  ----------------------------------------------------------------------
  md5           128bit  9e107d9d372bb6826bd81d3542a419d6...
  sha1          160bit  2fd4e1c67a2d28fced849ee1bb76e739...
  sha256        256bit  d7a8fbb307d7809469ca9abcb0082e4f...
  sha384        384bit  ca737f1014a48f4c0b6dd43cb177b0af...
  sha512        512bit  07e547d9586f6a73f73fbac0435ed769...
  sha3_256      256bit  69070dda01975c8c120c3aada1b28239...
```

---

## 5. How TLS/HTTPS Works

TLS (Transport Layer Security) is a protocol that encrypts communication over the Internet, preventing eavesdropping, tampering, and impersonation. Nearly all modern Internet communications depend on TLS, including web browsing (HTTPS), email (SMTPS/IMAPS), and VPNs.

### 5.1 Purpose and Guarantees of TLS

TLS achieves the following three security objectives.

```
Three Guarantees Provided by TLS:

  1. Confidentiality
     -> Third parties cannot read the communication content
     -> Achieved with AES-256-GCM / ChaCha20-Poly1305

  2. Integrity
     -> Guarantees that communication content has not been tampered with
     -> Achieved with HMAC / AEAD authentication tags

  3. Authenticity
     -> Guarantees that the communication partner is genuine
     -> Achieved with X.509 certificates + digital signatures

  Position in the Protocol Stack:

  +---------------------+
  |  HTTP / SMTP / ...  |  Application Layer
  +---------------------+
  |       TLS 1.3       |  <- Encryption happens here
  +---------------------+
  |        TCP          |  Transport Layer
  +---------------------+
  |        IP           |  Network Layer
  +---------------------+
```

### 5.2 TLS 1.3 Handshake

TLS 1.3, standardized as RFC 8446 in 2018, is the latest TLS version, significantly simplified and accelerated compared to TLS 1.2.

```
TLS 1.3 Full Handshake (1-RTT):

  Client                                   Server
  |                                        |
  |  ClientHello                           |
  |  +- Supported cipher suites           |
  |  +- Supported groups (X25519, P-256)   |
  |  +- key_share (ECDH public key)        |
  |  +- supported_versions (TLS 1.3)       |
  |------------------------------------>   |
  |                                        |
  |                         ServerHello    |
  |                  +- Selected cipher suite|
  |                  +- key_share (ECDH key)|
  |  <------------------------------------|
  |                                        |
  |  [Derive shared secret here]           |
  |  shared_secret = ECDH(my_key, peer_key)|
  |  -> Derive handshake_keys              |
  |                                        |
  |  {EncryptedExtensions}                 |
  |  {Certificate}        (server cert)    |
  |  {CertificateVerify}  (signature)      |
  |  {Finished}           (MAC)            |
  |  <===================================  |
  |  (hereafter, {} content is encrypted)  |
  |                                        |
  |  {Finished}                            |
  |  ===================================> |
  |                                        |
  |  <=== Encrypted application data ===>  |
  |                                        |

  Removed in TLS 1.3 (security improvements):
  - RSA key exchange (no forward secrecy)
  - Static DH (no forward secrecy)
  - CBC mode ciphers (padding oracle attack risk)
  - RC4, DES, 3DES, MD5, SHA-1
  - Compression (CRIME attack)
  - Renegotiation

  TLS 1.3 Cipher Suites (only 5):
  - TLS_AES_256_GCM_SHA384
  - TLS_AES_128_GCM_SHA256
  - TLS_CHACHA20_POLY1305_SHA256
  - TLS_AES_128_CCM_SHA256
  - TLS_AES_128_CCM_8_SHA256
```

### 5.3 Forward Secrecy

Forward secrecy (Perfect Forward Secrecy, PFS) is the property that guarantees past communications cannot be decrypted even if the long-term secret key is compromised in the future.

```
How Forward Secrecy Works:

  RSA Key Exchange (TLS 1.2 and earlier, no forward secrecy):
  +----------------------------------------+
  | Client encrypts pre-master secret      |
  | with server's RSA public key and sends |
  |                                        |
  | Problem: If server's private key leaks,|
  | all previously recorded communications |
  | can be decrypted                       |
  +----------------------------------------+

  ECDHE Key Exchange (TLS 1.3, with forward secrecy):
  +----------------------------------------+
  | Generate new ephemeral ECDH key pair   |
  | for each connection and exchange keys  |
  |                                        |
  | Advantage: Even if long-term key leaks,|
  | session-specific ephemeral keys cannot |
  | be recovered                           |
  | -> Past communications remain secure   |
  +----------------------------------------+

  "E" = Ephemeral
  The "E" in ECDHE provides forward secrecy
```

### 5.4 X.509 Certificates and Chain of Trust

TLS uses X.509 certificates to verify the server's identity. Certificates are signed by Certificate Authorities (CAs), forming a Chain of Trust.

```
Certificate Chain (Chain of Trust):

  +-------------------------------+
  |  Root CA Certificate          | <- Pre-installed in OS/Browser
  |  (Self-signed)                |    ~100-150 trusted roots
  |  e.g.: DigiCert Global Root G2|
  +-----------+-------------------+
              | Signs
              v
  +-------------------------------+
  |  Intermediate CA Certificate  | <- Signed by Root CA
  |  e.g.: DigiCert SHA2 Secure  |
  |       Server CA               |
  +-----------+-------------------+
              | Signs
              v
  +-------------------------------+
  |  Server Certificate (leaf)    | <- Signed by Intermediate CA
  |  e.g.: www.example.com       |
  |  Contains:                    |
  |  - Domain name (SAN)         |
  |  - Public key                |
  |  - Validity period           |
  |  - Issuer (Intermediate CA)  |
  |    signature                  |
  +-------------------------------+

  Verification Process:
  1. Verify server certificate signature with intermediate CA's public key
  2. Verify intermediate CA certificate signature with root CA's public key
  3. Confirm root CA exists in the trust store
  4. Confirm certificate is within its validity period
  5. Confirm certificate has not been revoked (CRL/OCSP)
  6. Confirm domain name matches the request
```

### 5.5 0-RTT Reconnection

TLS 1.3 includes a feature that enables sending encrypted data with 0-RTT (Zero Round Trip Time) when reconnecting to a previously connected server.

```
0-RTT Reconnection:

  Receive PSK (Pre-Shared Key) from server after initial connection

  On reconnection:
  Client                        Server
  |  ClientHello               |
  |  + early_data (0-RTT data) |
  |----------------------------> | <- Can send app data
  |                              |    with the first message
  |         ServerHello          |
  |  <------------------------  |
  |                              |
  |  <=== Encrypted comm. ===>  |

  Advantage: Reduced latency (data sent with first message)
  Risk: Possibility of replay attack
  -> Should only be used for idempotent operations (GET requests, etc.)
```

---

## 6. Digital Signatures

Digital signatures are a technology that mathematically guarantees the authenticity (identity of the sender) and integrity (no tampering) of a message. Unlike handwritten signatures, they depend on the message content, making forgery extremely difficult and enabling non-repudiation.

### 6.1 How Digital Signatures Work

```
Digital Signature Processing Flow:

  Signing (Sender):
  +----------------------------------------------+
  |                                              |
  |  Message M                                   |
  |       |                                      |
  |  Compute hash: h = H(M)                      |
  |       |                                      |
  |  Sign with private key: s = Sign(privkey, h)  |
  |       |                                      |
  |  Send (M, s)                                  |
  |                                              |
  +----------------------------------------------+

  Verification (Receiver):
  +----------------------------------------------+
  |                                              |
  |  Receive (M, s)                               |
  |       |                                      |
  |  Compute hash: h = H(M)                      |
  |       |                                      |
  |  Verify with public key: Verify(pubkey, h, s) -> true/false |
  |       |                                      |
  |  true: Signature is valid (sender is genuine & not tampered) |
  |  false: Signature is invalid (forged or tampered)            |
  |                                              |
  +----------------------------------------------+

  Difference from encryption:
  - Encryption: Encrypt with public key -> Decrypt with private key
  - Digital signature: Sign with private key -> Verify with public key
  -> Key usage direction is reversed
```

### 6.2 Major Signature Algorithms

```
Signature Algorithm Comparison:

  +------------+--------------+-----------+------------------+
  | Algorithm  | Security     | Key       | Use              |
  |            | Basis        | Length    |                  |
  +------------+--------------+-----------+------------------+
  | RSA-PSS    | Factoring    | 2048+ bit | TLS, code signing|
  | ECDSA      | ECDLP        | 256 bit   | TLS, Bitcoin     |
  | Ed25519    | ECDLP        | 256 bit   | SSH, Signal      |
  | Ed448      | ECDLP        | 448 bit   | High security    |
  +------------+--------------+-----------+------------------+

  Ed25519 Features:
  - Uses Edwards curve on Curve25519
  - Deterministic signatures (same message+key always produces same signature)
    -> Does not depend on random number quality (avoids ECDSA vulnerability)
  - Fast: Signing ~50us, Verification ~100us (typical CPU)
  - Compact: Signature 64 bytes, Public key 32 bytes
```

### 6.3 Learning Digital Signatures with Python

```python
"""
Ed25519 digital signature demo
Dependency: pip install cryptography
"""
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature


def digital_signature_demo():
    """Demo of signing and verification with Ed25519."""
    # Generate key pair
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    # Byte representation of public key (32 bytes)
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat,
    )
    pub_bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    print(f"Public key (hex): {pub_bytes.hex()}")
    print(f"Public key length: {len(pub_bytes)} bytes")

    # Sign a message
    message = "This document has not been tampered with.".encode("utf-8")
    signature = private_key.sign(message)
    print(f"\nMessage: {message.decode()}")
    print(f"Signature (hex): {signature.hex()}")
    print(f"Signature length: {len(signature)} bytes")

    # Signature verification (normal case)
    try:
        public_key.verify(signature, message)
        print("\nSignature verification: OK (valid signature)")
    except InvalidSignature:
        print("\nSignature verification: NG")

    # Signature verification (tampering detection)
    tampered_message = "This document has been tampered with.".encode("utf-8")
    try:
        public_key.verify(signature, tampered_message)
        print("Signature verification: OK (tampering not detected -- error)")
    except InvalidSignature:
        print("Signature verification: NG (tampering detected -- correct behavior)")

    # Verification with a different private key (impersonation detection)
    fake_private_key = Ed25519PrivateKey.generate()
    fake_signature = fake_private_key.sign(message)
    try:
        public_key.verify(fake_signature, message)
        print("Signature verification: OK (impersonation not detected -- error)")
    except InvalidSignature:
        print("Signature verification: NG (impersonation detected -- correct behavior)")


if __name__ == "__main__":
    digital_signature_demo()
```

```
Example output:

  Public key (hex): 7d4a3b2c1e0f...(64 hex characters)
  Public key length: 32 bytes

  Message: This document has not been tampered with.
  Signature (hex): 8f3a2b1c4d5e6f...(128 hex characters)
  Signature length: 64 bytes

  Signature verification: OK (valid signature)
  Signature verification: NG (tampering detected -- correct behavior)
  Signature verification: NG (impersonation detected -- correct behavior)
```

### 6.4 Applications of Digital Signatures

Digital signatures go beyond simple message authentication and permeate the entire modern software ecosystem.

```
Major Applications of Digital Signatures:

  1. Code Signing
     - OS verifies the origin of applications
     - Apple: Code signing required, Notarization
     - Windows: Authenticode signing
     - Android: APK signing

  2. Package Managers
     - apt/yum: Verify repository authenticity with GPG signatures
     - npm/PyPI: Sigstore-based signing is growing
     - Docker: Container image signing (Cosign/Notation)

  3. Git Commit Signing
     - Sign commits with GPG or SSH keys
     - GitHub: Displays "Verified" badge
     - git commit -S -m "signed commit"

  4. Electronic Contracts / Digital Signatures
     - Digital signatures on PDF documents
     - Legal validity based on e-signature laws in various countries

  5. Blockchain
     - Transaction signing (ECDSA / Ed25519)
     - Bitcoin: ECDSA on secp256k1 curve
     - Ethereum: Same + EIP-712 typed data signing

  6. JWT (JSON Web Token)
     - RS256: RSA-PSS + SHA-256
     - ES256: ECDSA P-256 + SHA-256
     - EdDSA: Ed25519 (RFC 8037)
```

---

## 7. Cryptography and Computational Complexity

The security of cryptography relies on the existence of "hard problems" in computational complexity theory. This section delves into the relationship between cryptography and computational complexity theory, understanding why certain algorithms are considered secure.

### 7.1 Computational Security vs Information-Theoretic Security

```
Two Definitions of Security:

  Information-Theoretic Security (Unconditional Security):
  +--------------------------------------------+
  | Cannot be broken even with infinite         |
  | computational power                         |
  | Example: One-Time Pad                       |
  | Requirements: Key length >= plaintext,      |
  | completely random key, no key reuse         |
  | -> Key distribution is impractical          |
  +--------------------------------------------+

  Computational Security:
  +--------------------------------------------+
  | Cannot be broken by polynomial-time         |
  | attackers                                   |
  | Examples: AES-256, RSA-2048, ECDSA P-256    |
  | Basis: Assumption that certain mathematical |
  | problems are "hard" (often unproven)        |
  | -> Standard security model for modern       |
  |    cryptography                             |
  +--------------------------------------------+
```

### 7.2 Hard Problems Underlying Cryptography

```
Important Hard Problems in Cryptography:

  1. Integer Factorization Problem
     +----------------------------------------------+
     | Given a composite number N = p * q,           |
     | efficiently find the prime factors p, q       |
     |                                              |
     | Best classical algorithm:                     |
     |   General Number Field Sieve                  |
     |   -> Sub-exponential time (neither polynomial |
     |      nor exponential)                         |
     |                                              |
     | Dependent cryptography: RSA                   |
     +----------------------------------------------+

  2. Discrete Logarithm Problem (DLP)
     +----------------------------------------------+
     | Given prime p and generator g,                |
     | find x satisfying g^x = h (mod p)            |
     |                                              |
     | Best classical algorithm:                     |
     |   Number Field Sieve (equivalent to           |
     |   factorization)                              |
     |                                              |
     | Dependent cryptography: Diffie-Hellman, DSA   |
     +----------------------------------------------+

  3. Elliptic Curve Discrete Logarithm Problem (ECDLP)
     +----------------------------------------------+
     | Given point P and Q = nP on elliptic curve E, |
     | find scalar n                                 |
     |                                              |
     | Best classical algorithm:                     |
     |   Pollard rho method -- O(sqrt(n))            |
     |   -> No more efficient attack known than DLP  |
     |   -> Achieves high security with short keys   |
     |                                              |
     | Dependent cryptography: ECDH, ECDSA, Ed25519  |
     +----------------------------------------------+

  4. Lattice Problems
     +----------------------------------------------+
     | Learning With Errors (LWE):                   |
     |   As + e = b (mod q), find s                  |
     |   A: public matrix, e: small error vector     |
     |                                              |
     | Best classical/quantum algorithm:             |
     |   Exponential time (hard even for quantum     |
     |   computers)                                  |
     |                                              |
     | Dependent cryptography: ML-KEM, ML-DSA        |
     | (post-quantum)                                |
     +----------------------------------------------+
```

### 7.3 Security Levels and Key Lengths

The security level expresses the computational effort needed to break a cipher in bits. n-bit security means that the best attack requires 2^n operations.

```
Security Level Correspondence Table:

  +--------------+----------+----------+----------+----------+
  | Security     | Symmetric| RSA      | ECC      | Hash     |
  | Level        | (AES)    |          |          |(collision)|
  +--------------+----------+----------+----------+----------+
  |  128 bit     | AES-128  | RSA-3072 | P-256    | SHA-256  |
  |  192 bit     | AES-192  | RSA-7680 | P-384    | SHA-384  |
  |  256 bit     | AES-256  | RSA-15360| P-521    | SHA-512  |
  +--------------+----------+----------+----------+----------+

  Estimated time to break (128-bit security):
  - 2^128 corresponds to approximately 3.4 * 10^38 operations
  - Even deploying all the world's supercomputers (~10^18 ops/s)
    would require approximately 10^13 years (~700x the age of the universe)
```

---

## 8. Post-Quantum Cryptography (PQC)

### 8.1 The Quantum Computing Threat

Quantum computers leverage principles of quantum mechanics (superposition, entanglement) to deliver computational power that overwhelmingly exceeds classical computers for certain problems. The particular threats to cryptography are Shor's algorithm and Grover's algorithm.

```
Impact of Quantum Algorithms on Cryptography:

  Shor's Algorithm (1994):
  +----------------------------------------------+
  | Solves factoring and discrete logarithm      |
  | in polynomial time                           |
  |                                              |
  | Affected cryptography:                       |
  | - RSA -> Completely broken                   |
  | - DH/DSA -> Completely broken                |
  | - ECDH/ECDSA -> Completely broken            |
  |                                              |
  | Required qubits (logical):                   |
  | - RSA-2048: ~4,000 logical qubits            |
  | - ECC-256: ~2,500 logical qubits             |
  | (Physical qubits need 100-1000x for error    |
  |  correction)                                 |
  +----------------------------------------------+

  Grover's Algorithm (1996):
  +----------------------------------------------+
  | Speeds up search problems to O(sqrt(N))      |
  |                                              |
  | Impact:                                      |
  | - Halves symmetric cipher security           |
  |   AES-128 -> 64-bit equivalent (insufficient)|
  |   AES-256 -> 128-bit equivalent (sufficiently|
  |   secure)                                    |
  | - Reduces hash function preimage resistance  |
  |   SHA-256 -> 128-bit equivalent (sufficiently|
  |   secure)                                    |
  |                                              |
  | Countermeasure: Doubling key length suffices  |
  +----------------------------------------------+

  Quantum Computer Timeline (estimated):
  +------+------------------------------+
  | 2024 | ~1,000 physical qubits (noisy)|
  | 2030 | ~10,000 physical qubits (est.)|
  | 2035+| Cryptographic threat level (est.)|
  +------+------------------------------+

  "Harvest Now, Decrypt Later" Attack:
  - Record and store current encrypted communications
  - Decrypt in the future with quantum computers
  - Data requiring long-term secrecy needs countermeasures now
```

### 8.2 NIST Post-Quantum Cryptography Standards

NIST has been advancing the post-quantum cryptography standardization process since 2016 and announced the first standards in 2024.

```
NIST PQC Standards (announced 2024):

  Key Encapsulation Mechanism (KEM):
  +------------+----------+----------+----------+------------+
  | Name       | Basis    | Pub. key | Cipher.  | Security   |
  +------------+----------+----------+----------+------------+
  | ML-KEM-512 | MLWE     | 800 B    | 768 B    | 128 bit    |
  | ML-KEM-768 | MLWE     | 1184 B   | 1088 B   | 192 bit    |
  | ML-KEM-1024| MLWE     | 1568 B   | 1568 B   | 256 bit    |
  +------------+----------+----------+----------+------------+
  ML-KEM = Module Lattice KEM (formerly CRYSTALS-Kyber)

  Digital Signatures:
  +----------------+----------+----------+----------+----------+
  | Name           | Basis    | Pub. key | Sig. len | Security |
  +----------------+----------+----------+----------+----------+
  | ML-DSA-44      | MLWE     | 1312 B   | 2420 B   | 128 bit  |
  | ML-DSA-65      | MLWE     | 1952 B   | 3293 B   | 192 bit  |
  | ML-DSA-87      | MLWE     | 2592 B   | 4595 B   | 256 bit  |
  | SLH-DSA-128s   | Hash     | 32 B     | 7856 B   | 128 bit  |
  | SLH-DSA-128f   | Hash     | 32 B     | 17088 B  | 128 bit  |
  +----------------+----------+----------+----------+----------+
  ML-DSA = Module Lattice DSA (formerly CRYSTALS-Dilithium)
  SLH-DSA = Stateless Hash-based DSA (formerly SPHINCS+)

  Key Size Comparison with Traditional Methods:
  +--------------------+------------+------------------+
  |                    | Pub. key   | Security basis   |
  +--------------------+------------+------------------+
  | RSA-2048           | 256 B      | Factoring        |
  | ECDSA P-256        | 64 B       | ECDLP            |
  | Ed25519            | 32 B       | ECDLP            |
  | ML-KEM-768 (PQC)   | 1184 B     | Lattice (MLWE)   |
  | ML-DSA-65 (PQC)    | 1952 B     | Lattice (MLWE)   |
  +--------------------+------------+------------------+

  -> PQC has larger key sizes but gains quantum resistance
```

### 8.3 Intuitive Understanding of Lattice Cryptography

```
Basic Idea of Lattice Cryptography (Learning With Errors: LWE):

  Intuition using 2-dimensional lattice points:

  y
  |    .     .     .     .     .
  |
  |       .     .     .     .     .
  |
  |    .     .     .     .     .
  |
  |       .     .     .     .     .
  |
  +-------------------------------------> x

  Lattice points (set of points arranged at exactly equal intervals)

  LWE Problem:
  1. You are given points with "small noise" added to lattice points
  2. Recover the original lattice structure (secret key) from these

  y
  |    x   x     x     x   x
  |
  |      x    x   x   x     x
  |
  |   x    x     x   x     x
  |
  |      x     x   x    x    x
  |
  +-------------------------------------> x

  x: Points with added noise (positions slightly shifted)

  -> When noise is sufficiently large, estimating the original lattice
     is extremely difficult
  -> Believed to be hard even for quantum computers
```

### 8.4 Hybrid Approach and Migration Strategy

The current migration to post-quantum cryptography recommends a "hybrid approach." This combines traditional cryptography with PQC, maintaining security even if one of them is broken.

```
Hybrid Key Exchange (Example in TLS):

  Client -> Server:
    X25519 public key + ML-KEM-768 public key

  Server -> Client:
    X25519 public key + ML-KEM-768 encapsulation

  Shared secret = KDF(X25519 shared secret || ML-KEM shared secret)

  -> Even if X25519 is broken by quantum computers, ML-KEM provides protection
  -> Even if ML-KEM has unknown vulnerabilities, X25519 provides protection

  Deployment examples:
  - Chrome/Firefox: X25519Kyber768 enabled by default
  - AWS: ML-KEM hybrid support in s2n-tls
  - Signal: PQXDH protocol (X25519 + ML-KEM)
```

---

## 9. Cryptographic Implementation and Best Practices

The mathematical security of a cryptographic algorithm and the security of its implementation are separate matters. Even theoretically secure algorithms can harbor critical vulnerabilities due to implementation errors.

### 9.1 Secure Random Number Generation

The quality of randomness in cryptography is fundamental to security. Using predictable random numbers allows keys and nonces to be predicted, destroying the entire cryptographic system.

```
Random Number Generation Hierarchy:

  +---------------------------------------------+
  | CSPRNG (Cryptographically Secure PRNG)       | <- For cryptographic use
  | os.urandom(), /dev/urandom                   |
  | secrets module (Python)                      |
  +---------------------------------------------+
  | PRNG (Pseudorandom Number Generator)         | <- For simulations, etc.
  | random module (Python)                       |
  | Math.random() (JavaScript)                   |
  | * NEVER use for cryptography                 |
  +---------------------------------------------+
  | TRNG (True Random Number Generator)          | <- Hardware
  | Intel RDRAND/RDSEED                          |
  | Physical phenomena (radioactive decay,       |
  | thermal noise, etc.)                         |
  +---------------------------------------------+
```

```python
"""
Comparison of secure and insecure random number generation
Dependencies: Standard library only
"""
import secrets
import random
import os


def secure_random_demo():
    """Demo of cryptographically secure random number generation."""
    # --- Secure methods ---
    # secrets module (Python 3.6+, designed for cryptographic use)
    secure_token = secrets.token_hex(32)  # 256-bit random token
    print(f"secrets.token_hex(32): {secure_token}")

    # os.urandom (obtained from OS entropy pool)
    secure_bytes = os.urandom(32)
    print(f"os.urandom(32):        {secure_bytes.hex()}")

    # Secure random integer
    secure_int = secrets.randbelow(2**256)
    print(f"secrets.randbelow():   {secure_int}")

    # --- Insecure methods (NEVER use for cryptography) ---
    # random module uses Mersenne Twister
    # Internal state can be fully recovered from 624 outputs
    random.seed(42)  # Predictable seed -> predictable output
    insecure = random.getrandbits(256)
    print(f"\nrandom.getrandbits(): {insecure}")
    print("  The above MUST NOT be used for cryptography (Mersenne Twister is not cryptographically secure)")
    print("  Internal state can be fully recovered by observing just 624 outputs")


if __name__ == "__main__":
    secure_random_demo()
```

### 9.2 Key Management Best Practices

```
Key Management Principles:

  1. Key Generation
     - Always use CSPRNG
     - Choose sufficient key length (AES-256, Ed25519, etc.)
     - When deriving keys from passwords, use a KDF
       (PBKDF2, scrypt, Argon2id)

  2. Key Storage
     - Zero-clear keys in memory after use
     - Store keys on disk in encrypted form
     - Utilize HSMs (Hardware Security Modules)
     - Never embed keys directly in environment variables or source code
     - Cloud: AWS KMS, GCP Cloud KMS, Azure Key Vault

  3. Key Rotation
     - Update keys periodically (e.g., annually)
     - Immediate rotation if a key is compromised
     - Plan for re-encryption of old data

  4. Key Destruction
     - Securely destroy keys that are no longer needed
     - Crypto-erase: Destroy the key to make data unreadable
```

### 9.3 Side-Channel Attacks

The security of cryptography depends not only on the mathematical properties of the algorithm but also on the physical behavior of its implementation. Side-channel attacks exploit physical information leaked during cryptographic computation to infer the secret key.

```
Major Side-Channel Attacks:

  1. Timing Attack
     +--------------------------------------+
     | Infer secret information from        |
     | processing time differences          |
     |                                      |
     | Example: Password comparison that    |
     | checks from the beginning and        |
     | immediately returns false on mismatch|
     | -> Processing time reveals the number|
     |    of matching characters            |
     |                                      |
     | Countermeasure: Use constant-time    |
     | comparison                           |
     | Python: hmac.compare_digest()        |
     | Node.js: crypto.timingSafeEqual()    |
     +--------------------------------------+

  2. Power Analysis Attack
     +--------------------------------------+
     | Infer secret key from power          |
     | consumption patterns during          |
     | cryptographic processing             |
     |                                      |
     | Countermeasure: Masking (add random  |
     | values to hide intermediate values)  |
     +--------------------------------------+

  3. Cache Timing Attack
     +--------------------------------------+
     | Infer memory access patterns from    |
     | CPU cache hit/miss                   |
     |                                      |
     | Example: Attack on AES T-Table impl. |
     | Countermeasure: Use AES-NI (hardware |
     | instructions) or bitsliced impl.     |
     +--------------------------------------+

  4. Spectre/Meltdown Family Attacks
     +--------------------------------------+
     | Leak data via side effects of        |
     | speculative execution                |
     | Countermeasure: OS/hardware-level    |
     | mitigations                          |
     +--------------------------------------+
```

---

## 10. Comparative Analysis of Cryptographic Methods

### 10.1 Symmetric-Key vs Public-Key Cryptography

```
Comprehensive Comparison Table:

  +------------------+-----------------------+-----------------------+
  |                  | Symmetric-Key         | Public-Key            |
  |                  | (Symmetric)           | (Asymmetric)          |
  +------------------+-----------------------+-----------------------+
  | Number of keys   | 1 (shared key)        | 2 (public + private)  |
  | Speed            | Fast (several GB/s)   | Slow (100-1000x slower)|
  | Key length       | 128 bit               | RSA: 3072 bit         |
  | (128-bit equiv.) |                       | ECC: 256 bit          |
  | Key distribution | Pre-sharing required  | Public key is public  |
  | Scalability      | O(n^2) key pairs      | O(n) key pairs        |
  | Use              | Data encryption       | Key exchange, signing,|
  |                  |                       | authentication        |
  | Examples         | AES, ChaCha20         | RSA, ECDH, Ed25519   |
  | Quantum resist.  | Double key length     | Completely broken     |
  |                  |                       | by Shor               |
  +------------------+-----------------------+-----------------------+

  Practical Operation (Hybrid Approach):
  1. Public-key crypto (ECDH) securely exchanges shared key
  2. Symmetric-key crypto (AES-256-GCM) encrypts data
  -> Combines the strengths of both
```

### 10.2 Cipher Suite Selection Guidelines

```
Recommended Cryptographic Configurations (as of 2025):

  +---------------+--------------------------------------+
  | Use           | Recommended Configuration             |
  +---------------+--------------------------------------+
  | Data encrypt. | AES-256-GCM or ChaCha20-Poly1305     |
  | Key exchange  | X25519 (ECDH) + ML-KEM-768 (PQC)     |
  | Digital sig.  | Ed25519 or ECDSA P-256               |
  | Hash          | SHA-256 / SHA-384 / SHA3-256          |
  | Password store| Argon2id                              |
  | MAC           | HMAC-SHA256 / Poly1305                |
  | KDF           | HKDF-SHA256 / Argon2id                |
  | TLS           | TLS 1.3 only (1.2 and earlier depr.)  |
  +---------------+--------------------------------------+

  Configurations to avoid:
  - DES / 3DES / RC4 (weak)
  - RSA key exchange (no forward secrecy)
  - MD5 / SHA-1 (collision resistance broken)
  - CBC mode alone (padding oracle attack risk)
  - ECB mode (pattern leakage)
  - RSA/DH with 1024 bits or less
```

---

## 11. Anti-Patterns

Many patterns exist in cryptographic implementations that appear correct but create critical vulnerabilities. Below are representative anti-patterns.

### 11.1 Anti-Pattern 1: Using ECB Mode

**Problem:** ECB (Electronic Codebook) mode always converts identical plaintext blocks to identical ciphertext blocks. This preserves structural patterns of the plaintext in the ciphertext, greatly compromising confidentiality.

```python
"""
Anti-pattern: Demo of ECB mode danger
Dependency: pip install cryptography
"""
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os


def demonstrate_ecb_weakness():
    """Show that identical blocks produce identical ciphertext in ECB mode."""
    key = os.urandom(32)  # AES-256

    # Plaintext with repeated identical 16-byte blocks
    block = b"AAAAAAAAAAAAAAAA"  # 16 bytes = 1 AES block
    plaintext = block * 4  # Same block repeated 4 times

    # --- ECB mode (DANGEROUS) ---
    cipher_ecb = Cipher(algorithms.AES(key), modes.ECB())
    encryptor_ecb = cipher_ecb.encryptor()
    ciphertext_ecb = encryptor_ecb.update(plaintext) + encryptor_ecb.finalize()

    # ECB: identical blocks produce identical ciphertext
    blocks_ecb = [ciphertext_ecb[i:i+16] for i in range(0, len(ciphertext_ecb), 16)]
    print("ECB mode (INSECURE):")
    for i, b in enumerate(blocks_ecb):
        print(f"  Block {i+1}: {b.hex()}")
    print(f"  All blocks identical: {len(set(blocks_ecb)) == 1}")

    # --- GCM mode (SECURE) ---
    nonce = os.urandom(12)
    cipher_gcm = Cipher(algorithms.AES(key), modes.GCM(nonce))
    encryptor_gcm = cipher_gcm.encryptor()
    ciphertext_gcm = encryptor_gcm.update(plaintext) + encryptor_gcm.finalize()
    tag = encryptor_gcm.tag

    blocks_gcm = [ciphertext_gcm[i:i+16] for i in range(0, len(ciphertext_gcm), 16)]
    print("\nGCM mode (SECURE):")
    for i, b in enumerate(blocks_gcm):
        print(f"  Block {i+1}: {b.hex()}")
    print(f"  All blocks identical: {len(set(blocks_gcm)) == 1}")


if __name__ == "__main__":
    demonstrate_ecb_weakness()
```

```
Example output:

  ECB mode (INSECURE):
    Block 1: 8a2e1f3b4c5d6e7f...
    Block 2: 8a2e1f3b4c5d6e7f...  <- Identical
    Block 3: 8a2e1f3b4c5d6e7f...  <- Identical
    Block 4: 8a2e1f3b4c5d6e7f...  <- Identical
    All blocks identical: True

  GCM mode (SECURE):
    Block 1: 3f7a9c2e1d4b8f6a...
    Block 2: 5e8b1d3f7a9c2e4b...  <- Different
    Block 3: 2d4b8f6a3f7a9c1e...  <- Different
    Block 4: 9c2e4b5e8b1d3f7a...  <- Different
    All blocks identical: False
```

**Countermeasure:** Always use authenticated encryption modes (AES-GCM or ChaCha20-Poly1305). There is no legitimate reason to use ECB mode.

### 11.2 Anti-Pattern 2: Using Custom Cryptographic Algorithms

**Problem:** Creating custom cryptographic algorithms is extremely risky even for professional cryptographers. The security of cryptography becomes trustworthy only after years of public cryptanalysis, so unverified custom algorithms should be considered "insecure" by default.

```
Why Custom Cryptography is Dangerous:

  1. Schneier's Law:
     "Anyone can create a cipher that they themselves cannot break,
      but that doesn't mean it is secure"

  2. Being able to "use" encryption and it being "secure"
     are completely different things. Being able to encrypt and
     decrypt does not imply security.

  3. Looking back at the AES standardization process:
     - 15 algorithms submitted as candidates
     - Over 3 years of public analysis
     - Cryptographers worldwide attempted attacks
     - Rijndael was ultimately selected

  4. Alternatives:
     - Use existing standardized algorithms
     - Use trusted libraries
       Python: cryptography, PyCryptodome
       Go: crypto/*, golang.org/x/crypto
       Rust: ring, RustCrypto
       C: OpenSSL, libsodium (NaCl)
     - libsodium/NaCl high-level API is the safest

  NG example:
  +----------------------------------------------+
  | def my_encrypt(data, key):                   |
  |     return bytes(d ^ k for d, k              |
  |                  in zip(data, cycle(key)))    |
  |                                              |
  | -> Simple XOR cipher. Key is instantly        |
  |   recovered via known-plaintext attack.       |
  |   Key repetition makes it as weak as the     |
  |   Vigenere cipher.                           |
  +----------------------------------------------+

  OK example:
  +----------------------------------------------+
  | from cryptography.hazmat.primitives          |
  |     .ciphers.aead import AESGCM              |
  |                                              |
  | key = AESGCM.generate_key(bit_length=256)    |
  | nonce = os.urandom(12)                       |
  | ct = AESGCM(key).encrypt(nonce, data, ad)    |
  +----------------------------------------------+
```

### 11.3 Anti-Pattern 3: Nonce/IV Reuse

**Problem:** Reusing a nonce (Number used ONCE) with the same key in GCM mode or ChaCha20-Poly1305 allows the XOR of plaintexts to be obtained from the XOR of ciphertexts, and also enables authentication tag forgery. This is a catastrophic vulnerability.

```
Destructive Power of Nonce Reuse:

  Encrypting two plaintexts with the same key K and same nonce N:

  C1 = P1 XOR E(K, N)
  C2 = P2 XOR E(K, N)

  Attacker computes C1 XOR C2:
  C1 XOR C2 = (P1 XOR E(K,N)) XOR (P2 XOR E(K,N))
            = P1 XOR P2

  -> XOR of the two plaintexts is revealed
  -> If one plaintext is known or guessable, the other is also revealed

  For GCM it's even worse:
  - Authentication key H is leaked
  - Authentication tags for arbitrary messages can be forged

  Countermeasures:
  - Manage 96-bit nonces with a counter (prevent reuse)
  - When using random nonces, limit encryptions to 2^32 or fewer
    (to keep collision probability negligible via birthday paradox)
  - XChaCha20-Poly1305: 192-bit nonce makes random generation safe
```

### 11.4 Anti-Pattern 4: Encryption Without Authentication

**Problem:** Performing only encryption without message authentication makes it impossible to detect ciphertext tampering. Attackers can manipulate ciphertext to intentionally alter decryption results through bit-flipping attacks and padding oracle attacks.

```
Danger of Encrypt-only:

  CTR mode (no authentication):
  C = P XOR E(K, Counter)

  Attacker flips specific bits of C:
  C' = C XOR Delta

  Decryption yields:
  P' = C' XOR E(K, Counter)
     = (C XOR Delta) XOR E(K, Counter)
     = P XOR Delta

  -> Specific bits of plaintext are predictably flipped

  Example: Bit manipulation like "$100" -> "$900" is possible

  Countermeasures:
  - Always use AEAD (AES-GCM, ChaCha20-Poly1305)
  - Use the Encrypt-then-MAC pattern
  - Always verify authentication tags before decryption
```

---

## 12. Exercises

### 12.1 Beginner Exercises

**Exercise 1: Verify the Avalanche Effect of Hashes**

Use SHA-256 to verify the following.

1. Compute the SHA-256 hash of the string "Hello"
2. Compute the SHA-256 hash of the string "hello" (changed first letter to lowercase)
3. Compare the two hash values bit by bit and calculate the percentage of differing bits
4. Verify that the result is approximately 50% and explain why this is called the avalanche effect

```python
# Hint
import hashlib

h1 = hashlib.sha256(b"Hello").hexdigest()
h2 = hashlib.sha256(b"hello").hexdigest()
# For bit comparison, convert to binary using int(h, 16) and compare
```

**Exercise 2: Implement and Break the Caesar Cipher**

1. Implement Caesar cipher encryption and decryption functions in Python
2. Decrypt the ciphertext "KHOOR ZRUOG" with all 26 shifts and find the meaningful plaintext (brute-force attack)
3. Explain why this attack is possible (small key space)

### 12.2 Intermediate Exercises

**Exercise 3: Manual Calculation of Diffie-Hellman Key Exchange**

Perform a DH key exchange by hand with the following small parameters.

```
Public parameters: p = 23, g = 5

Alice: secret value a = 6
  A = g^a mod p = 5^6 mod 23 = ?

Bob: secret value b = 15
  B = g^b mod p = 5^15 mod 23 = ?

Shared secret:
  Alice: s = B^a mod p = ?
  Bob:   s = A^b mod p = ?

Verify: Do Alice and Bob's shared secrets match?
```

**Exercise 4: Secure File Encryption with AES-GCM**

Implement a file encryption program that meets the following requirements.

1. Derive a key from a password using Argon2id
2. Encrypt the file with AES-256-GCM
3. Save salt + nonce + ciphertext + tag to the output file
4. During decryption, re-derive the key from the password, then decrypt and authenticate

### 12.3 Advanced Exercises

**Exercise 5: Observing the TLS 1.3 Handshake**

1. Run `openssl s_client -connect example.com:443 -tls1_3` and observe the handshake log
2. Identify the cipher suite used, key exchange algorithm, and certificate chain
3. Capture a TLS 1.3 handshake with Wireshark and analyze the ClientHello/ServerHello structure
4. Write a comparison report on differences from TLS 1.2 (number of RTTs, timing of encryption start)

**Exercise 6: Analysis of Post-Quantum Cryptography Key Size Impact**

1. Calculate the size difference between ML-KEM-768 public key (1184 bytes) and X25519 public key (32 bytes)
2. Estimate the data volume increase for hybrid key exchange (X25519 + ML-KEM-768) in a TLS handshake
3. Consider the impact on mobile networks (bandwidth-constrained environments)

---

## 13. Frequently Asked Questions (FAQ)

### Q1: Should I use AES-128 or AES-256?

**A:** AES-256 is generally recommended. The reasons are as follows.

- AES-128 theoretically provides 128-bit security and is sufficiently secure against classical computers
- However, Grover's algorithm reduces it to the equivalent of 64 bits on quantum computers
- AES-256 maintains 128-bit equivalent security even in a quantum environment
- The speed reduction of AES-256 is about 40% (14 rounds vs 10 rounds), but this is negligible in environments with AES-NI hardware acceleration
- AES-256 is essential for long-term data storage (10+ years)

### Q2: Is communication completely secure if it uses HTTPS?

**A:** HTTPS (TLS) provides communication channel encryption, but cannot be said to be "completely secure." The following points require attention.

- **The communication channel is protected:** Eavesdroppers can only see ciphertext
- **Endpoints are not protected:** Encryption for server-side data storage is a separate matter
- **Server trustworthiness is a separate issue:** Data is available in plaintext to malicious servers
- **Limitations of the certificate trust model:** There have been cases of CAs issuing fraudulent certificates (DigiNotar incident, 2011)
- **TLS version and configuration matter:** TLS 1.0/1.1 are deprecated. TLS 1.3 is recommended
- **0-RTT replay risk:** 0-RTT data can be subject to replay attacks

### Q3: Should I not use SHA-256 for password hashing?

**A:** SHA-256 must not be used alone for password storage. The reasons are as follows.

- SHA-256 is a general-purpose hash designed to be fast
- GPUs can compute billions of SHA-256 hashes per second
- This means attackers can execute brute-force and dictionary attacks in extremely short time
- Use dedicated slow functions for password hashing (Argon2id, bcrypt, scrypt)
- These allow tuning computational cost (CPU time, memory usage) via parameters
- Additionally, adding a salt (unique random value per password) prevents precomputation attacks (rainbow tables)

### Q4: Should RSA no longer be used?

**A:** RSA is still widely used, but elliptic curve cryptography (ECC) should be preferred for new designs.

- RSA-2048 remains secure against classical computers
- However, it will be broken by Shor's algorithm on quantum computers
- Key size is much larger compared to ECC (RSA-3072 vs P-256: ~12x)
- ECC is also faster computationally
- RSA key exchange has been removed in TLS 1.3 (only remaining for signature use)
- For new development, Ed25519 (signing) + X25519 (key exchange) is recommended

### Q5: How should encrypted data be backed up?

**A:** Key management is the biggest challenge in backing up encrypted data.

- Encrypted data and keys must be stored in separate locations
- Losing a key makes data permanently unrecoverable (this is the intended behavior)
- Key backup strategies:
  - Key splitting (Shamir's Secret Sharing): Key can be recovered when k of n people gather
  - Key storage in HSM (Hardware Security Module)
  - Backup of key files encrypted with a master key
- Plan for handling data encrypted with old keys during key rotation

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just from theory alone, but by actually writing code and confirming how it works.

### Q2: What are common mistakes beginners make?

Skipping fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently utilized in daily development work. It becomes especially important during code reviews and architecture design.

---

## 14. Summary

| Concept | Key Point |
|---------|----------|
| Classical cryptography | Caesar cipher, Vigenere cipher, Enigma. Broken due to small key spaces or structural weaknesses |
| Symmetric-key crypto | Fast. AES-256-GCM is the current standard. Key sharing is the challenge (solved by public-key crypto) |
| Public-key crypto | Solves the key distribution problem. RSA is based on factoring, ECC on elliptic curve discrete logarithm |
| Hash functions | One-way. SHA-256 is standard. Use Argon2id for password storage |
| HMAC | Hash + secret key for message authentication. Widely used in API authentication |
| TLS 1.3 | 1-RTT handshake, ECDHE + AEAD, forward secrecy required |
| Digital signatures | Sign with private key, verify with public key. Ed25519 is currently recommended |
| Computational complexity | Cryptographic security is based on assumptions of "hard problems." 128-bit security is standard |
| Post-quantum crypto | Lattice cryptography (ML-KEM, ML-DSA) is the NIST standard. Migrating via hybrid approach |
| Anti-patterns | ECB prohibited, custom crypto prohibited, nonce reuse prohibited, encryption-only (no auth) prohibited |
| Key management | Generate with CSPRNG, store in HSM/KMS, rotate periodically, destroy securely |
| Side channels | Timing attacks, power analysis. Countermeasures include constant-time implementations and hardware acceleration |

### Next Steps for Learning

1. **Get hands-on**: Implement the exercises in Section 12 as actual code
2. **Observe TLS**: Capture your browser's TLS communications with Wireshark and analyze them
3. **Try post-quantum crypto**: Experience PQC algorithms with liboqs (Open Quantum Safe)
4. **Challenge CTFs**: Practice cryptanalysis at CryptoHack (cryptohack.org)
5. **Read papers**: Read through the NIST PQC standardization final report

---

## Recommended Next Guides

---

## References

1. Ferguson, N. & Schneier, B. *Cryptography Engineering: Design Principles and Practical Applications*. Wiley, 2010. -- A comprehensive textbook on cryptographic engineering covering design principles to secure implementation methods.
2. Katz, J. & Lindell, Y. *Introduction to Modern Cryptography*. 3rd ed. CRC Press, 2020. -- A standard textbook rigorously explaining the theoretical foundations of modern cryptography, systematically covering definitions of computational security through concrete constructions.
3. NIST. "Post-Quantum Cryptography Standardization." 2024. https://csrc.nist.gov/projects/post-quantum-cryptography -- NIST's post-quantum cryptography standardization project, publishing final standard documents for ML-KEM, ML-DSA, and SLH-DSA.
4. Bernstein, D. J. & Lange, T. "Post-quantum cryptography." *Nature*, 549(7671), 188-194, 2017. -- A Nature paper providing an accessible overview of post-quantum cryptography, comparing lattice, code-based, multivariate polynomial, and hash-based signature approaches.
5. Rescorla, E. "The Transport Layer Security (TLS) Protocol Version 1.3." RFC 8446, 2018. https://datatracker.ietf.org/doc/html/rfc8446 -- The official TLS 1.3 specification, defining the handshake protocol, record protocol, and cipher suite details.
6. Boneh, D. & Shoup, V. *A Graduate Course in Applied Cryptography*. 2023. https://toc.cryptobook.us/ -- A textbook based on Stanford University's cryptography course, freely available online, deeply covering both theory and practice.
7. Aumasson, J.-P. *Serious Cryptography: A Practical Introduction to Modern Encryption*. 2nd ed. No Starch Press, 2024. -- A practical introduction to cryptography, explaining the internal structure of each algorithm at the code level with extensive coverage of implementation considerations.

