# Blockchain Fundamentals

> Blockchain is a distributed ledger technology that solves the "trust problem" through technology. Without requiring a central authority, all participants guarantee the legitimacy of transactions through consensus. This chapter systematically explains hash chain structures, consensus mechanisms (consensus algorithms), and smart contract mechanisms from the ground up.

---

## Learning Objectives

- [ ] Understand hash chain structure and the principles of tamper resistance
- [ ] Explain the internal structure of a block (header, Merkle tree, nonce)
- [ ] Compare the differences among major consensus algorithms (PoW, PoS, BFT-based)
- [ ] Understand the role of cryptographic techniques (hash functions, public key cryptography, digital signatures)
- [ ] Grasp the concept and operating principles of smart contracts
- [ ] Explain the fundamental concepts of DeFi, token standards, and Layer 2
- [ ] Organize the blockchain trilemma and real-world challenges
- [ ] Implement basic blockchain structures in Python


## Prerequisites

Having the following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Machine Learning Basics](./01-machine-learning-basics.md)

---

## 1. What Is Blockchain?

### 1.1 Comparison with Centralized Systems

The vast majority of modern information systems are built on centralized architectures. Bank account balances are recorded in bank databases, and SNS messages are stored on platform company servers. While this model has clear advantages, it inherently contains fundamental vulnerabilities.

```
Traditional Centralized System:

  ┌────────────────────────────────────────────┐
  │           Central Management Server         │
  │  ┌────────────────────────────────────┐    │
  │  │        Central Database            │    │
  │  │  ┌─────┐ ┌─────┐ ┌─────┐         │    │
  │  │  │Tx 1 │ │Tx 2 │ │Tx 3 │  ...    │    │
  │  │  └─────┘ └─────┘ └─────┘         │    │
  │  └────────────────────────────────────┘    │
  │       Administrator holds full authority    │
  └──────────────┬─────────────────────────────┘
         ┌───────┼───────┐
         │       │       │
       ┌─┴─┐  ┌─┴─┐  ┌─┴─┐
       │ A │  │ B │  │ C │   ← Users trust the administrator
       └───┘  └───┘  └───┘

  Advantages:
  - High processing speed (read/write to a single DB)
  - Clear authority management
  - Established disaster recovery procedures

  Problems:
  - Single Point of Failure
    → Entire service goes down if the central DB fails
  - Risk of tampering/censorship by the administrator
    → Data integrity depends on the administrator's honesty
  - Centralized management of privacy
    → Massive amounts of personal data concentrated in one place
```

Blockchain provides a fundamental approach to these problems.

```
Blockchain (Distributed Ledger):

  ┌───┐      ┌───┐      ┌───┐      ┌───┐
  │ A │──────│ B │──────│ C │──────│ D │
  └─┬─┘      └─┬─┘      └─┬─┘      └─┬─┘
    │          │          │          │
  ┌─┴────┐  ┌─┴────┐  ┌─┴────┐  ┌─┴────┐
  │Ledger│  │Ledger│  │Ledger│  │Ledger│
  │ Copy │  │ Copy │  │ Copy │  │ Copy │
  └──────┘  └──────┘  └──────┘  └──────┘
  All participants hold and verify identical copies of the ledger

  Core Properties:
  (1) Decentralization: All participants hold complete copies of the data
  (2) Tamper Resistance: Modification of past data is cryptographically infeasible
  (3) Transparency:      All transactions are public and verifiable by anyone
  (4) Non-centralization: The system operates without a specific administrator
  (5) Fault Tolerance:   The entire network continues operating even if some nodes go down
```

### 1.2 Historical Background of Blockchain

Blockchain technology did not emerge suddenly; it is the culmination of decades of research in cryptography and distributed systems.

| Year | Event | Significance |
|------|-------|-------------|
| 1976 | Diffie-Hellman key exchange published | Foundational theory of public key cryptography |
| 1977 | RSA encryption invented | Practical realization of public key cryptography |
| 1979 | Merkle's hash tree patent | Efficient data integrity verification |
| 1982 | Lamport's Byzantine Generals Problem | Theoretical framework for distributed consensus |
| 1991 | Haber & Stornetta's timestamp chain | Tamper detection via hash chains |
| 1997 | Hashcash (Adam Back) | Prototype of Proof of Work |
| 2004 | Reusable Proof of Work (Hal Finney) | Reuse of PoW tokens |
| 2008 | Bitcoin whitepaper (Satoshi Nakamoto) | First practical blockchain |
| 2009 | Bitcoin network launch | Genesis Block creation |
| 2014 | Ethereum whitepaper (Vitalik Buterin) | Smart contract platform |
| 2015 | Ethereum mainnet launch | Programmable blockchain |
| 2020 | Ethereum 2.0 Beacon Chain launch | Beginning of PoS transition |
| 2022 | The Merge (Ethereum's complete PoW→PoS transition) | 99.95% energy consumption reduction |

### 1.3 Clarification of Definitions

The term "blockchain" can carry different meanings depending on context. This chapter proceeds based on the following definitions.

**Blockchain (narrow sense)**: A chronological data structure of blocks linked by cryptographic hash functions. Each block contains the hash value of the previous block, enabling tamper detection.

**Blockchain (broad sense)**: Refers to the entire distributed ledger system that combines hash chain structure with P2P networks, consensus algorithms, and incentive mechanisms.

---

## 2. Hash Chains and Block Structure

### 2.1 Fundamentals of Cryptographic Hash Functions

The foundation of blockchain's tamper resistance is the cryptographic hash function. A hash function transforms arbitrary-length input data into a fixed-length hash value (digest) and satisfies the following properties.

**Five Properties a Cryptographic Hash Function Must Satisfy:**

1. **Deterministic**: Always returns the same output for the same input
2. **Fast Computation**: Can efficiently compute the hash value for any input
3. **Pre-image Resistance**: Computationally infeasible to find m satisfying h = H(m) given a hash value h
4. **Second Pre-image Resistance**: Given input m1, computationally infeasible to find m2 (m2 ≠ m1) satisfying H(m1) = H(m2)
5. **Collision Resistance**: Computationally infeasible to find distinct m1, m2 satisfying H(m1) = H(m2)

```python
"""
Code Example 1: Verify the fundamental properties of SHA-256 hash function
"""
import hashlib


def sha256(data: str) -> str:
    """Hash a string with SHA-256 and return it as a hexadecimal string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# --- Verify Determinism ---
msg = "Hello, Blockchain!"
hash1 = sha256(msg)
hash2 = sha256(msg)
assert hash1 == hash2, "The same input always returns the same hash value"
print(f"Input: '{msg}'")
print(f"SHA-256: {hash1}")
# Example output: SHA-256: 7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284adfa93...

# --- Verify Avalanche Effect ---
msg_a = "Hello, Blockchain!"
msg_b = "Hello, Blockchain?"  # Changed the last character by 1
hash_a = sha256(msg_a)
hash_b = sha256(msg_b)

# Calculate bitwise difference
bits_a = bin(int(hash_a, 16))[2:].zfill(256)
bits_b = bin(int(hash_b, 16))[2:].zfill(256)
diff_bits = sum(a != b for a, b in zip(bits_a, bits_b))

print(f"\n--- Avalanche Effect ---")
print(f"Input A: '{msg_a}' -> {hash_a[:16]}...")
print(f"Input B: '{msg_b}' -> {hash_b[:16]}...")
print(f"Different bits: {diff_bits}/256 ({diff_bits/256*100:.1f}%)")
# Ideally about 50% of bits change

# --- Verify Fixed-Length Output ---
short_input = "A"
long_input = "A" * 10000
print(f"\n--- Fixed-Length Output ---")
print(f"1-char input:     {sha256(short_input)} (length: {len(sha256(short_input))})")
print(f"10000-char input: {sha256(long_input)} (length: {len(sha256(long_input))})")
# Both are 64-character (256-bit) hexadecimal strings
```

### 2.2 Internal Structure of a Block

Each block in a blockchain consists of two regions: a Header and a Body. The header stores metadata, and the body stores a group of transactions.

```
Detailed Block Structure (Bitcoin Model):

  Block N-1              Block N                Block N+1
  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
  │ Block Header   │    │ Block Header   │    │ Block Header   │
  │ ┌────────────┐ │    │ ┌────────────┐ │    │ ┌────────────┐ │
  │ │ version    │ │    │ │ version    │ │    │ │ version    │ │
  │ │ prevHash ──┼─┼────│ │ prevHash ──┼─┼────│ │ prevHash   │ │
  │ │ merkleRoot │ │    │ │ merkleRoot │ │    │ │ merkleRoot │ │
  │ │ timestamp  │ │    │ │ timestamp  │ │    │ │ timestamp  │ │
  │ │ difficulty │ │    │ │ difficulty │ │    │ │ difficulty │ │
  │ │ nonce      │ │    │ │ nonce      │ │    │ │ nonce      │ │
  │ └────────────┘ │    │ └────────────┘ │    │ └────────────┘ │
  ├────────────────┤    ├────────────────┤    ├────────────────┤
  │ Block Body     │    │ Block Body     │    │ Block Body     │
  │ ┌────────────┐ │    │ ┌────────────┐ │    │ ┌────────────┐ │
  │ │ Tx Count   │ │    │ │ Tx Count   │ │    │ │ Tx Count   │ │
  │ │ Tx 1       │ │    │ │ Tx 1       │ │    │ │ Tx 1       │ │
  │ │ Tx 2       │ │    │ │ Tx 2       │ │    │ │ Tx 2       │ │
  │ │ Tx 3       │ │    │ │ Tx 3       │ │    │ │ Tx 3       │ │
  │ │ ...        │ │    │ │ ...        │ │    │ │ ...        │ │
  │ └────────────┘ │    │ └────────────┘ │    │ └────────────┘ │
  └────────────────┘    └────────────────┘    └────────────────┘

  Header Field Details:
  ─────────────────────────────────────────────────────────
  version     : Protocol version (4 bytes)
  prevHash    : Header hash of the immediately preceding block (32 bytes)
                → This is the backbone of the chain structure
  merkleRoot  : Merkle tree root hash of all transactions (32 bytes)
  timestamp   : Unix timestamp at block creation (4 bytes)
  difficulty  : Mining difficulty target (4 bytes)
  nonce       : Value searched for in PoW (4 bytes)
  ─────────────────────────────────────────────────────────
  Bitcoin header size: 80 bytes total (fixed length)
```

**How the prevHash Field Achieves Tamper Resistance:**

```
Cascading Impact of Tampering:

  If you want to tamper with Tx1 in Block 100:

  Step 1: Modify Tx1's data
          → merkleRoot changes

  Step 2: Block 100's header hash changes
          → Mismatch with Block 101's prevHash

  Step 3: Correct Block 101's prevHash
          → Block 101's header hash changes
          → Mismatch with Block 102's prevHash

  Step 4: All blocks from Block 102 onward must be recalculated

  Step 5: Furthermore, PoW must be recalculated for each block
          (= enormous computational cost)

  Step 6: Additionally, one must generate blocks faster than the
          legitimate chain with more than half the network's
          computational power

  Conclusion: Tampering with past blocks is computationally infeasible

  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
  │Blk 98│→│Blk 99│→│Blk100│→│Blk101│→│Blk102│
  └──────┘  └──────┘  └──┬───┘  └──────┘  └──────┘
                         │
                    Tamper point
                         │
                         ▼
                    All blocks from this point
                    onward must be recalculated
```

### 2.3 Merkle Tree

A Merkle tree is a hash binary tree structure for efficiently verifying the integrity of transactions. It is a data structure patented by Ralph Merkle in 1979 and serves as the foundation for transaction verification in blockchains.

```
Merkle Tree Structure (with 4 transactions):

                 ┌──────────────────┐
                 │   Merkle Root    │
                 │  H(H_AB + H_CD)  │
                 └────────┬─────────┘
                    ┌─────┴─────┐
              ┌─────┴─────┐  ┌──┴──────────┐
              │   H_AB    │  │    H_CD     │
              │ H(H_A+H_B)│  │ H(H_C+H_D) │
              └─────┬─────┘  └──┬──────────┘
                ┌───┴───┐    ┌──┴───┐
           ┌────┴──┐ ┌──┴───┐ ┌┴────┐ ┌─────┐
           │  H_A  │ │ H_B  │ │ H_C │ │ H_D │
           │H(Tx_A)│ │H(Tx_B)│ │H(Tx_C)│ │H(Tx_D)│
           └───┬───┘ └──┬───┘ └──┬──┘ └──┬──┘
               │        │        │       │
             Tx_A     Tx_B     Tx_C    Tx_D


  Verification Efficiency (SPV: Simplified Payment Verification):
  ─────────────────────────────────────────────────────
  To verify that Tx_B is included in a block:

  Required information: only H_A, H_CD, and the Merkle Root

  Verification steps:
  1. H_B = H(Tx_B)                   ... computed locally
  2. H_AB = H(H_A + H_B)             ... H_A is provided
  3. Root = H(H_AB + H_CD)           ... H_CD is provided
  4. Compare the computed Root with the block header's merkleRoot

  Computational complexity: O(log n)  ← where n is the number of transactions
  Example: 1000 Tx → verification complete with about 10 hash computations
  Example: 1 million Tx → verification complete with about 20 hash computations
```

```python
"""
Code Example 2: Implement Merkle tree construction and verification
"""
import hashlib
from typing import List, Optional, Tuple


def hash_data(data: str) -> str:
    """Hash data with SHA-256."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def hash_pair(left: str, right: str) -> str:
    """Concatenate and hash two hash values."""
    return hashlib.sha256((left + right).encode("utf-8")).hexdigest()


class MerkleTree:
    """Class for building and verifying a Merkle tree."""

    def __init__(self, transactions: List[str]):
        if not transactions:
            raise ValueError("Transaction list cannot be empty")
        self.transactions = transactions
        self.leaves = [hash_data(tx) for tx in transactions]
        self.tree: List[List[str]] = []  # Holds hashes at each level
        self.root = self._build_tree()

    def _build_tree(self) -> str:
        """Build the Merkle tree and return the root hash."""
        current_level = self.leaves[:]
        self.tree.append(current_level[:])

        while len(current_level) > 1:
            next_level = []
            # If the number of nodes is odd, duplicate the last node
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])
            for i in range(0, len(current_level), 2):
                parent = hash_pair(current_level[i], current_level[i + 1])
                next_level.append(parent)
            self.tree.append(next_level[:])
            current_level = next_level

        return current_level[0]

    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        Return the inclusion proof (Merkle Proof) for the transaction at the given index.
        Return value: list of [(hash_value, position)]. Position is "left" or "right"
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Index {index} is out of range")

        proof = []
        idx = index

        for level in self.tree[:-1]:  # Exclude the root level
            # If odd number of elements, duplicate the last one
            level_copy = level[:]
            if len(level_copy) % 2 == 1:
                level_copy.append(level_copy[-1])

            if idx % 2 == 0:
                # Self is on the left → use the right sibling as proof
                sibling_idx = idx + 1
                if sibling_idx < len(level_copy):
                    proof.append((level_copy[sibling_idx], "right"))
            else:
                # Self is on the right → use the left sibling as proof
                proof.append((level_copy[idx - 1], "left"))
            idx //= 2

        return proof

    @staticmethod
    def verify_proof(
        tx_hash: str,
        proof: List[Tuple[str, str]],
        root: str,
    ) -> bool:
        """Verify a Merkle proof."""
        current = tx_hash
        for sibling_hash, position in proof:
            if position == "left":
                current = hash_pair(sibling_hash, current)
            else:
                current = hash_pair(current, sibling_hash)
        return current == root


# --- Usage Example ---
transactions = [
    "Alice -> Bob: 10 BTC",
    "Bob -> Charlie: 5 BTC",
    "Charlie -> Dave: 3 BTC",
    "Dave -> Eve: 1 BTC",
]

tree = MerkleTree(transactions)
print(f"Merkle Root: {tree.root[:16]}...")

# Get inclusion proof for Tx 1 (Bob -> Charlie)
proof = tree.get_proof(1)
tx_hash = hash_data(transactions[1])
print(f"\nTx: '{transactions[1]}'")
print(f"Tx Hash: {tx_hash[:16]}...")
print(f"Proof steps: {len(proof)}")

# Verification
is_valid = MerkleTree.verify_proof(tx_hash, proof, tree.root)
print(f"Verification result: {'Valid' if is_valid else 'Invalid'}")

# Verify a tampered transaction
tampered_hash = hash_data("Bob -> Charlie: 50 BTC")
is_valid_tampered = MerkleTree.verify_proof(tampered_hash, proof, tree.root)
print(f"Tampered Tx verification result: {'Valid' if is_valid_tampered else 'Invalid (tampering detected)'}")
```

### 2.4 Genesis Block

The first block of a blockchain is called the Genesis Block, a special block with no prevHash. Bitcoin's Genesis Block was generated on January 3, 2009, by Satoshi Nakamoto. The coinbase transaction of this block contains the headline from The Times newspaper of that day: "Chancellor on brink of second bailout for banks," widely known as a critical message about the existing financial system.

### 2.5 P2P Network and Block Propagation

Blockchain networks adopt a P2P (Peer-to-Peer) structure, where nodes communicate directly without going through a central server. New transactions and blocks are propagated throughout the entire network via the "gossip protocol."

```
Block Propagation in a P2P Network:

  Step 1: Miner M discovers a block
  ┌───┐
  │ M │  ← Generates a new block
  └─┬─┘
    │ Broadcast
    ├───────────┬───────────┐
    ▼           ▼           ▼
  ┌───┐     ┌───┐       ┌───┐
  │ A │     │ B │       │ C │  ← Directly connected peers
  └─┬─┘     └─┬─┘       └─┬─┘
    │         │            │
    ▼         ▼            ▼
  ┌───┐     ┌───┐       ┌───┐
  │ D │     │ E │       │ F │  ← Further propagation
  └───┘     └───┘       └───┘

  Processing steps at each node:
  1. Receive the block
  2. Verify the block header hash
  3. Check if previous_hash matches the end of its own chain
  4. Verify signatures of all transactions
  5. Check if PoW condition is satisfied
  6. Verification success → Add to own chain + forward to other peers
  7. Verification failure → Discard the block

  Node Types (Bitcoin):
  ─────────────────────────────────────────
  Full Node:
    Holds all blockchain data (approx. 500 GB+)
    Independently verifies all transactions
    Has complete autonomy

  Light Node (SPV Node):
    Holds only block headers (approx. 50 MB)
    Verifies specific Tx inclusion via Merkle proofs
    Requires queries to full nodes

  Mining Node:
    Full node + mining capability
    Attempts to generate new blocks
    Typically via mining pools
```

### 2.6 Forks (Chain Splits)

A fork is a phenomenon where a blockchain diverges, broadly classified into temporary forks and protocol-change forks.

```
Temporary Fork (Stale Block):
──────────────────────────────────────────────

  When two miners discover a block almost simultaneously:

       ┌──────┐
   ┌──→│Blk N'│  ← Discovered by Miner A
   │   │(ver A)│
   │   └──────┘
  ┌──────┐
  │Blk N-1│
  └──┬───┘
   │   ┌──────┐
   └──→│Blk N │  ← Discovered by Miner B
       │(ver B)│
       └──────┘

  Resolution: "Longest Chain Rule"
  Determined by which fork the next block is built on

       ┌──────┐   ┌──────┐
   ┌──→│Blk N'│──→│Blk N+1│  ← This one is longer → canonical chain
   │   └──────┘   └───────┘
  ┌──────┐
  │Blk N-1│
  └──┬───┘
   │   ┌──────┐
   └──→│Blk N │  ← Discarded as an orphan block
       └──────┘

  → Reason why "6 confirmations" is recommended in Bitcoin:
    The probability of a fork being reversed after 6 blocks is less than 0.0002%


Protocol Forks:
──────────────────────
  Hard Fork:
    A non-backward-compatible protocol change
    Old nodes judge new blocks as invalid → chain permanently diverges
    Examples: Bitcoin → Bitcoin Cash (2017, block size debate)
              Ethereum → Ethereum Classic (2016, rollback after The DAO incident)

  Soft Fork:
    A backward-compatible protocol change
    Old nodes still judge new blocks as valid
    Example: Bitcoin's SegWit (2017, separation of signature data)
```

---

## 3. Consensus Mechanisms

### 3.1 The Need for Distributed Consensus and the Byzantine Generals Problem

In a distributed system without a central authority, some consensus formation protocol (consensus mechanism) is needed for all nodes to agree on the same ledger state. The theoretical foundation for this problem is the Byzantine Generals Problem.

```
Byzantine Generals Problem:
─────────────────────────────────────────────

  Setup:
  - 4 generals are besieging an enemy castle
  - If all attack simultaneously they win; if discoordinated they lose
  - Communication is only possible through messengers
  - However, there may be traitors among the generals

           General A (honest)
          /           \
    "Attack"       "Attack"
        /               \
  General B (honest)    General C (traitor)
        \               /
    "Attack"       "Retreat"  ← False message
        \               /
           General D (honest)
           → B says "Attack", C says "Retreat"... which to believe?

  Problem:
  In a situation where traitors exist, how can the
  honest generals reach correct consensus?

  Theoretical Result (Lamport, Shostak, Pease 1982):
  - If the number of traitors is f, at least 3f + 1 nodes are required
  - Consensus is possible if fewer than 1/3 of participants are traitors

  Blockchain consensus mechanisms provide
  practical solutions to this problem
```

### 3.2 Proof of Work (PoW)

PoW is the consensus mechanism first put into practical use with Bitcoin, where the node (miner) that first solves a computational puzzle earns the right to add a block.

**Operating Principle of PoW:**

```
PoW Puzzle:
  SHA-256(block_header) < target

  The target is a threshold that varies with difficulty
  → Find a nonce that produces N leading zeros in the hash value

  Example: difficulty = 4 (first 4 digits are zero)
  ┌────────────────────────────────────────────────┐
  │ nonce=0:      a3f2b1c8e9d...  → Does not meet condition │
  │ nonce=1:      7c4e9d2f0a1...  → Does not meet condition │
  │ nonce=2:      f1a8b3c7d2e...  → Does not meet condition │
  │ ...                                             │
  │ nonce=74839:  00009ab3f21...  → Does not meet condition │
  │ nonce=74840:  00003d8a1f2...  → Does not meet condition │
  │ ...                                             │
  │ nonce=198247: 0000038f7c1...  → Meets condition!        │
  └────────────────────────────────────────────────┘

  Asymmetry:
  - Finding the solution: Difficult (brute-force search, billions of calculations on average)
  - Verifying the solution: Easy (just one hash calculation)
  → This asymmetry is the essence of PoW

  Difficulty Adjustment:
  Bitcoin: Adjusted every 2016 blocks (approximately 2 weeks)
  Target: Average 10 minutes/block
  - Blocks generated too fast → difficulty increases
  - Blocks generated too slow → difficulty decreases
```

**Analysis of the 51% Attack:**

An attacker who controls more than 51% of the network's total computational power can theoretically generate a longer chain than the legitimate chain, enabling double spending. In Bitcoin this requires an astronomical cost and is considered practically impossible, but actual incidents have occurred on smaller chains with low hash rates (51% attacks on Ethereum Classic, 2019-2020).

```python
"""
Code Example 3: Proof of Work Mining Simulation
"""
import hashlib
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class BlockHeader:
    """Data class representing a block header."""
    version: int
    prev_hash: str
    merkle_root: str
    timestamp: float
    difficulty: int
    nonce: int = 0

    def to_string(self) -> str:
        """Convert header information to a string."""
        return (
            f"{self.version}{self.prev_hash}{self.merkle_root}"
            f"{self.timestamp}{self.difficulty}{self.nonce}"
        )

    def compute_hash(self) -> str:
        """Compute the SHA-256 hash of the header."""
        return hashlib.sha256(
            self.to_string().encode("utf-8")
        ).hexdigest()


def mine_block(header: BlockHeader) -> tuple[int, str, float]:
    """
    Execute PoW mining.
    Return value: (nonce, hash_value, elapsed_time)
    """
    target = "0" * header.difficulty
    start_time = time.time()
    attempts = 0

    while True:
        hash_result = header.compute_hash()
        attempts += 1

        if hash_result[:header.difficulty] == target:
            elapsed = time.time() - start_time
            print(f"  Block found!")
            print(f"  Nonce: {header.nonce}")
            print(f"  Hash:  {hash_result}")
            print(f"  Attempts: {attempts:,}")
            print(f"  Elapsed time: {elapsed:.3f} seconds")
            return header.nonce, hash_result, elapsed

        header.nonce += 1

        # Progress display (every 1 million attempts)
        if attempts % 1_000_000 == 0:
            print(f"  ... {attempts:,} attempts in progress ...")


# --- Compare mining time by difficulty ---
print("=== PoW Mining Simulation ===\n")

results = []
for difficulty in range(1, 6):
    print(f"--- Difficulty: {difficulty} (first {difficulty} digits are zero) ---")
    header = BlockHeader(
        version=1,
        prev_hash="0" * 64,
        merkle_root="abcdef1234567890" * 4,
        timestamp=time.time(),
        difficulty=difficulty,
    )
    nonce, block_hash, elapsed = mine_block(header)
    results.append((difficulty, nonce, elapsed))
    print()

print("=== Results Summary ===")
print(f"{'Difficulty':>12} {'Nonce':>12} {'Time(sec)':>12}")
print("-" * 40)
for d, n, t in results:
    print(f"{d:>12} {n:>12,} {t:>12.3f}")
# Each increase of 1 in difficulty requires approximately 16x more computation on average
```

### 3.3 Proof of Stake (PoS)

PoS is a consensus mechanism that assigns block generation rights based on held tokens (stake) instead of computational resources. It avoids the enormous energy consumption of PoW while securing network safety through economic incentives.

**Comparison Table: PoW vs PoS**

| Item | Proof of Work (PoW) | Proof of Stake (PoS) |
|------|--------------------|--------------------|
| Basis for block generation rights | Computational power (hash rate) | Stake amount (held tokens) |
| Required resources | High-performance GPU/ASIC, large amounts of electricity | Tokens for staking, standard servers |
| Energy consumption | Extremely large (100-150 TWh/year class) | Extremely small (less than 0.05% of PoW) |
| Attack cost | 51% of network's computational power | 33% of network's stake |
| Penalty for misconduct | Wasted electricity (indirect loss) | Stake confiscation (Slashing, direct loss) |
| Entry barrier | Procurement of specialized hardware | Purchase of tokens |
| Centralization risk | Mining pool oligopoly | Power concentration among large holders |
| Finality | Probabilistic (practically confirmed after 6 confirmations) | Deterministic (confirmed per Epoch) |
| Representative implementations | Bitcoin, Litecoin, Dogecoin | Ethereum 2.0, Cardano, Polkadot |

**Ethereum's PoS (The Merge, September 2022):**

- 32 ETH stake required to become a validator
- Validator committees are randomly selected every Epoch (32 slots, approximately 6.4 minutes)
- Selected validators propose blocks, and other validators vote (Attestation)
- Stake is confiscated (Slashing) for misconduct (double voting, contradictory voting)
- The Merge reduced energy consumption by 99.95%

**Nothing-at-Stake Problem and Countermeasures:**

In PoS, when a fork (chain split) occurs, validators can vote on both forks at no computational cost. In PoW, mining on both forks splits computational power and thus serves as a natural deterrent, but PoS lacks this natural disincentive. Ethereum's Casper FFG protocol addresses this problem by establishing Slashing conditions that automatically confiscate the stake of validators who cast contradictory votes.

### 3.4 Other Consensus Mechanisms

**Delegated Proof of Stake (DPoS):**

Token holders vote for representatives (delegates), and a small number of elected delegates produce blocks. This mechanism resembles representative democracy and enables high-speed processing, but carries centralization risk. EOS (21 delegates) and Tron are representative implementations.

**BFT-based (PBFT, Tendermint):**

Consensus is formed through explicit voting processes among nodes. The advantage is that immediate finality is achieved with agreement from more than 2/3 of participants, but communication volume grows at O(n^2) as the number of participating nodes increases, making it suitable for scales of tens to hundreds of nodes. Cosmos (Tendermint BFT) and Hyperledger Fabric are representative examples.

**Proof of Authority (PoA):**

Only pre-approved validators produce blocks. This trust model is based on complete identity verification, and while the decentralization of public chains is lost, processing speed is extremely fast, making it suitable for private/consortium chains. It is adopted in VeChain and testnets (Goerli, etc.).

**Proof of History (PoH):**

A mechanism adopted by Solana, used in combination with PoS. It proves the passage of time through a cryptographic hash chain, reducing block generation overhead. It achieves functionality close to a "Verifiable Delay Function (VDF)" through sequential SHA-256 computations.

**Comparison Table: Comprehensive Consensus Mechanism Comparison**

| Mechanism | Speed (TPS) | Decentralization | Energy Efficiency | Finality | Representative Examples |
|-----------|-----------|--------|--------------|--------------|--------|
| PoW | 3-7 | High | Very low | Probabilistic (~60 min) | Bitcoin, Litecoin |
| PoS | 15-100 | High | Very high | Deterministic (~13 min) | Ethereum 2.0, Cardano |
| DPoS | 1000-4000 | Moderate | High | Deterministic (seconds) | EOS, Tron |
| BFT-based | 1000-10000 | Low (node count limited) | High | Immediate | Cosmos, Hyperledger |
| PoA | Thousands+ | Low | High | Immediate | VeChain |
| PoH+PoS | Thousands-tens of thousands | Moderate | High | Deterministic (seconds) | Solana |

---

## 4. Cryptographic Foundations

### 4.1 Public Key Cryptography and Elliptic Curve Cryptography

Digital identity and transaction authentication in blockchain are based on public key cryptography. In particular, Elliptic Curve Cryptography (ECC) is widely used. Both Bitcoin and Ethereum use ECDSA (Elliptic Curve Digital Signature Algorithm) with the secp256k1 curve.

```
Key Derivation Process:

  ┌───────────────┐
  │  Private Key   │    A random 256-bit integer
  │                │    Example: 0x1a2b3c... (32 bytes)
  └───────┬───────┘
          │
          │ Elliptic curve multiplication (one-way function)
          │ Public Key = Private Key × G (generator point)
          │ * Reverse computation is computationally infeasible (discrete logarithm problem)
          ▼
  ┌───────────────┐
  │  Public Key    │    A point (x, y) on the elliptic curve
  │                │    Uncompressed format: 65 bytes
  └───────┬───────┘    Compressed format: 33 bytes
          │
          │ Apply hash functions
          │ Bitcoin: RIPEMD-160(SHA-256(public key))
          │ Ethereum: Last 20 bytes of Keccak-256(public key)
          ▼
  ┌───────────────┐
  │   Address      │    Used as transaction destination
  │                │    Bitcoin: 1A1zP1... (Base58Check)
  └───────────────┘    Ethereum: 0xAb5801... (hexadecimal)

  Important property:
  Private Key → Public Key → Address (one-way only)
  Address → Public Key → Private Key (reverse computation infeasible)
```

**How Digital Signatures Work:**

The sender signs a transaction using a private key, and the receiver (and all nodes) verify the signature using the sender's public key. This guarantees that (1) the transaction was created by the holder of the private key (authentication), and (2) the content of the transaction has not been tampered with (integrity).

### 4.2 Wallet Types and Key Management

Private key management is the most critical security element in blockchain usage. If a private key is lost, the assets are permanently lost; if a private key is leaked, the assets are stolen.

**Wallet Classification:**

| Category | Type | Features | Risks | Examples |
|----------|------|----------|-------|---------|
| Hot Wallet | Browser Extension | High convenience, easy DApp integration | Malware, phishing | MetaMask, Phantom |
| Hot Wallet | Mobile App | Portable | Device loss, malware | Trust Wallet, Rainbow |
| Cold Wallet | Hardware | Offline storage, high security | Physical loss, malfunction | Ledger, Trezor |
| Cold Wallet | Paper Wallet | Completely offline | Physical damage, loss | Printed on paper |

**HD Wallet (Hierarchical Deterministic Wallet):**

A standard defined in BIP-32/39/44 that can deterministically derive an infinite number of key pairs from a single seed (typically a 12 or 24 word mnemonic phrase). Backup is completed simply by storing the seed phrase.

### 4.3 Transaction Structure

Bitcoin transactions use the UTXO (Unspent Transaction Output) model, while Ethereum uses the account model.

```
UTXO Model (Bitcoin):
─────────────────────
  A model that manages "unspent change"

  Example: Alice sends 3 BTC to Bob
  (Alice previously received 5 BTC)

  Input:                    Output:
  ┌──────────────────┐     ┌──────────────────┐
  │ UTXO: 5 BTC      │     │ Bob: 3 BTC       │
  │ (past output      │ ──→ │ (new UTXO)       │
  │  addressed to     │     ├──────────────────┤
  │  Alice)           │     │ Alice: 1.999 BTC │
  │ + Alice's         │     │ (change UTXO)    │
  │   signature       │     ├──────────────────┤
  └──────────────────┘     │ Fee: 0.001 BTC   │
                           │ (to miner)        │
                           └──────────────────┘

  Sum of Inputs = Sum of Outputs + Fee
  5 BTC = 3 + 1.999 + 0.001

Account Model (Ethereum):
─────────────────────────
  A model that directly manages account balances

  ┌─────────────────────────────┐
  │ Alice's Account              │
  │ Balance: 10 ETH → 6.999 ETH│
  │ Nonce: 5 → 6               │
  └─────────────────────────────┘
           │ 3 ETH + Gas
           ▼
  ┌─────────────────────────────┐
  │ Bob's Account                │
  │ Balance: 2 ETH → 5 ETH     │
  └─────────────────────────────┘
```

---

## 5. Smart Contracts

### 5.1 What Is a Smart Contract?

A smart contract is a program deployed on a blockchain that automatically executes when predefined conditions are met. Nick Szabo proposed the concept in 1996, and the first general-purpose implementation was realized with the launch of the Ethereum mainnet in 2015.

The logic "if condition X is met, execute process Y" is automatically enforced on the blockchain without third-party intervention.

**Core Properties of Smart Contracts:**

1. **Immutability**: Code modification is principally impossible after deployment (Upgradeable Proxy patterns are an exception)
2. **Transparency**: Code is public and verifiable by anyone
3. **Automatic Execution**: Executes without human intervention when conditions are met
4. **Deterministic Execution**: All nodes produce the same result for the same input
5. **Gas Cost**: Computational fees (Gas) are required for execution

### 5.2 Ethereum Virtual Machine (EVM)

The EVM is the smart contract execution environment of Ethereum, a Turing-complete stack-based virtual machine. All nodes execute code on the same EVM and verify that results match.

```
EVM Architecture:

  ┌─────────────────────────────────────────┐
  │          Ethereum Virtual Machine        │
  │  ┌────────────┐  ┌───────────────────┐  │
  │  │   Stack    │  │    Memory         │  │
  │  │ (1024 deep)│  │ (volatile, per    │  │
  │  │            │  │  function call)   │  │
  │  └────────────┘  └───────────────────┘  │
  │  ┌────────────────────────────────────┐  │
  │  │         Storage                    │  │
  │  │  (persistent, Key-Value store)     │  │
  │  │  256-bit Key → 256-bit Value      │  │
  │  └────────────────────────────────────┘  │
  │  ┌────────────────────────────────────┐  │
  │  │         Bytecode                   │  │
  │  │  Solidity → Compile → Bytecode    │  │
  │  │  PUSH, POP, ADD, SSTORE, CALL ... │  │
  │  └────────────────────────────────────┘  │
  └─────────────────────────────────────────┘

  Account Types:
  ┌──────────────────────────────┐
  │ EOA (Externally Owned Account)│
  │ = Operated by a human with   │
  │   a private key              │
  │ - Address, Balance, Nonce    │
  │ - Can issue transactions     │
  ├──────────────────────────────┤
  │ Contract Account             │
  │ = Smart contract             │
  │ - Address, Balance, Code     │
  │ - Storage (persistent data)  │
  │ - Can call other contracts   │
  └──────────────────────────────┘

  Gas Mechanism:
  ────────────────────────────────
  Each opcode has a defined Gas cost

  Operation                 Gas Cost
  ──────────────────────────────────
  ADD (addition)            3
  MUL (multiplication)      5
  SLOAD (Storage read)      2100
  SSTORE (Storage write)    20000 (new) / 5000 (update)
  CREATE (contract creation) 32000
  ──────────────────────────────────

  Transaction Fee = Gas Used × Gas Price
  Example: Uniswap swap
      Gas Used: approx. 150,000
      Gas Price: 30 Gwei (= 0.00000003 ETH)
      Fee: 150,000 × 0.00000003 = 0.0045 ETH
```

### 5.3 Smart Contract Implementation Example in Solidity

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title DecentralizedCrowdfunding
 * @notice A decentralized crowdfunding contract
 *
 * Features:
 * - Project owner sets a funding goal and deadline
 * - Contributors fund with ETH
 * - Goal reached before deadline → Owner can withdraw funds
 * - Deadline expired without reaching goal → Contributors are refunded
 */
contract DecentralizedCrowdfunding {
    // --- State Variables ---
    address public owner;
    uint256 public goal;
    uint256 public deadline;
    uint256 public totalFunded;
    bool public claimed;

    mapping(address => uint256) public contributions;

    // --- Events ---
    event Funded(address indexed contributor, uint256 amount);
    event Claimed(address indexed owner, uint256 amount);
    event Refunded(address indexed contributor, uint256 amount);

    // --- Modifiers ---
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier beforeDeadline() {
        require(block.timestamp < deadline, "Deadline passed");
        _;
    }

    modifier afterDeadline() {
        require(block.timestamp >= deadline, "Deadline not reached");
        _;
    }

    // --- Constructor ---
    constructor(uint256 _goal, uint256 _durationSeconds) {
        require(_goal > 0, "Goal must be positive");
        require(_durationSeconds > 0, "Duration must be positive");
        owner = msg.sender;
        goal = _goal;
        deadline = block.timestamp + _durationSeconds;
    }

    // --- Funding ---
    function fund() external payable beforeDeadline {
        require(msg.value > 0, "Must send ETH");
        contributions[msg.sender] += msg.value;
        totalFunded += msg.value;
        emit Funded(msg.sender, msg.value);
    }

    // --- Withdraw funds upon reaching goal ---
    function claim() external onlyOwner afterDeadline {
        require(totalFunded >= goal, "Goal not reached");
        require(!claimed, "Already claimed");
        claimed = true;
        uint256 amount = address(this).balance;
        // Checks-Effects-Interactions pattern to prevent reentrancy attacks
        (bool success, ) = payable(owner).call{value: amount}("");
        require(success, "Transfer failed");
        emit Claimed(owner, amount);
    }

    // --- Refund if goal not reached ---
    function refund() external afterDeadline {
        require(totalFunded < goal, "Goal was reached");
        uint256 amount = contributions[msg.sender];
        require(amount > 0, "No contribution");
        // Checks-Effects-Interactions pattern
        contributions[msg.sender] = 0;
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Transfer failed");
        emit Refunded(msg.sender, amount);
    }
}
```

### 5.4 Mimicking Smart Contract Logic in Python

To understand the concept of smart contracts, the following implementation mimics Solidity logic in Python.

```python
"""
Code Example 4: Mimic smart contract concepts in Python
Reproducing the behavior of a decentralized escrow (third-party deposit)
"""
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional


class EscrowState(Enum):
    """State transitions of the escrow."""
    AWAITING_PAYMENT = auto()
    AWAITING_DELIVERY = auto()
    COMPLETE = auto()
    REFUNDED = auto()


@dataclass
class EscrowContract:
    """
    A class mimicking decentralized escrow logic.

    In an actual smart contract:
    - State is persisted on the blockchain
    - Automatic caller authentication via msg.sender
    - ETH sending/receiving is supported at the language level
    - All operations are recorded as transactions
    """
    buyer: str
    seller: str
    arbiter: str  # Dispute resolver
    amount: float
    state: EscrowState = EscrowState.AWAITING_PAYMENT
    balance: float = 0.0
    created_at: float = field(default_factory=time.time)
    events: list = field(default_factory=list)

    def _emit(self, event_name: str, data: dict) -> None:
        """Record an event (equivalent to blockchain event logs)."""
        entry = {
            "event": event_name,
            "data": data,
            "timestamp": time.time(),
        }
        self.events.append(entry)
        print(f"  [Event] {event_name}: {data}")

    def deposit(self, sender: str, value: float) -> None:
        """
        Buyer deposits into the escrow.
        Equivalent to a Solidity payable function.
        """
        if sender != self.buyer:
            raise PermissionError("Only the buyer can deposit")
        if self.state != EscrowState.AWAITING_PAYMENT:
            raise RuntimeError(f"Invalid state: {self.state.name}")
        if value != self.amount:
            raise ValueError(f"Must deposit the exact amount: {self.amount}")

        self.balance += value
        self.state = EscrowState.AWAITING_DELIVERY
        self._emit("Deposited", {"buyer": sender, "amount": value})

    def confirm_delivery(self, sender: str) -> None:
        """
        Buyer confirms receipt of goods and transfers funds to seller.
        Applies the Checks-Effects-Interactions pattern.
        """
        # Checks: Verify preconditions
        if sender != self.buyer:
            raise PermissionError("Only the buyer can confirm")
        if self.state != EscrowState.AWAITING_DELIVERY:
            raise RuntimeError(f"Invalid state: {self.state.name}")

        # Effects: Update state (change state before transfer)
        payout = self.balance
        self.balance = 0.0
        self.state = EscrowState.COMPLETE

        # Interactions: External call (transfer)
        self._emit("DeliveryConfirmed", {
            "buyer": sender,
            "seller": self.seller,
            "amount": payout,
        })
        print(f"  → Transferred {payout} to {self.seller}")

    def refund(self, sender: str) -> None:
        """Arbiter executes a refund."""
        if sender != self.arbiter:
            raise PermissionError("Only the arbiter can issue refunds")
        if self.state != EscrowState.AWAITING_DELIVERY:
            raise RuntimeError(f"Invalid state: {self.state.name}")

        refund_amount = self.balance
        self.balance = 0.0
        self.state = EscrowState.REFUNDED

        self._emit("Refunded", {
            "arbiter": sender,
            "buyer": self.buyer,
            "amount": refund_amount,
        })
        print(f"  → Refunded {refund_amount} to {self.buyer}")


# --- Usage Example ---
print("=== Decentralized Escrow Simulation ===\n")

escrow = EscrowContract(
    buyer="Alice",
    seller="Bob",
    arbiter="Charlie",
    amount=1.5,
)
print(f"Contract created: {escrow.buyer} → {escrow.seller}, Amount: {escrow.amount} ETH")
print(f"State: {escrow.state.name}\n")

# Scenario 1: Normal transaction
print("--- Scenario: Normal Transaction ---")
escrow.deposit("Alice", 1.5)
print(f"State: {escrow.state.name}, Balance: {escrow.balance}")
escrow.confirm_delivery("Alice")
print(f"State: {escrow.state.name}, Balance: {escrow.balance}\n")

# Scenario 2: Detection of unauthorized operations
print("--- Scenario: Detection of Unauthorized Operations ---")
escrow2 = EscrowContract(buyer="Alice", seller="Bob", arbiter="Charlie", amount=2.0)
escrow2.deposit("Alice", 2.0)
try:
    escrow2.confirm_delivery("Bob")  # Seller tries to confirm on their own
except PermissionError as e:
    print(f"  [Rejected] {e}")
try:
    escrow2.deposit("Alice", 2.0)  # Double deposit
except RuntimeError as e:
    print(f"  [Rejected] {e}")
```

### 5.5 Layer 2 Scaling Solutions

Since the throughput of the Ethereum mainnet (Layer 1) at approximately 15 TPS cannot sustain large-scale usage, Layer 2 solutions have been developed.

```
Layer 2 Scaling Overview:

  ┌─────────────────────────────────────────────────┐
  │                    Layer 2                       │
  │  ┌──────────────────┐  ┌──────────────────────┐ │
  │  │ Optimistic Rollup│  │    ZK Rollup         │ │
  │  │                  │  │                      │ │
  │  │ Execute Tx off-  │  │ Execute Tx off-chain │ │
  │  │ chain, and if    │  │ and mathematically   │ │
  │  │ fraud is found,  │  │ prove correctness    │ │
  │  │ challenges can be│  │ with zero-knowledge  │ │
  │  │ raised within    │  │ proofs before        │ │
  │  │ 7 days           │  │ submitting to L1     │ │
  │  │                  │  │                      │ │
  │  │ Examples:        │  │ Examples:            │ │
  │  │   Arbitrum       │  │   zkSync             │ │
  │  │   Optimism       │  │   StarkNet           │ │
  │  │   Base           │  │   Polygon zkEVM      │ │
  │  └────────┬─────────┘  └──────────┬───────────┘ │
  └───────────┼───────────────────────┼─────────────┘
              │ Compressed data +     │
              │ proofs                │
              ▼                       ▼
  ┌─────────────────────────────────────────────────┐
  │          Layer 1 (Ethereum Mainnet)              │
  │    Security foundation + Data availability +     │
  │    Final settlement                              │
  └─────────────────────────────────────────────────┘

  Comparison: Optimistic Rollup vs ZK Rollup
  ┌──────────────┬────────────────────┬──────────────────┐
  │ Item          │ Optimistic Rollup  │ ZK Rollup        │
  ├──────────────┼────────────────────┼──────────────────┤
  │ Verification  │ Fraud Proof        │ Validity Proof   │
  │ method        │                    │                  │
  │ Withdrawal    │ Approx. 7 days     │ Minutes to hours │
  │ time          │                    │                  │
  │ Computational │ Low                │ High (proof      │
  │ cost          │                    │ generation)      │
  │ EVM           │ High               │ Improving        │
  │ compatibility │                    │                  │
  │ Maturity      │ High               │ Rapidly evolving │
  └──────────────┴────────────────────┴──────────────────┘
```

---

## 6. DeFi and Token Economics

### 6.1 Overview of DeFi (Decentralized Finance)

DeFi is a collective term for systems that provide financial services through smart contracts, eliminating financial intermediaries such as banks and securities firms. It has grown rapidly since the "DeFi Summer" of 2020, with TVL (Total Value Locked) reaching over $180 billion at its peak.

**Major DeFi Protocol Categories:**

| Category | Function | Representative Protocols | Mechanism |
|----------|----------|--------------------------|-----------|
| DEX (Decentralized Exchange) | Token exchange | Uniswap, SushiSwap, Curve | AMM (Automated Market Maker) |
| Lending | Borrowing and lending | Aave, Compound | Collateralized lending, variable interest rates |
| Stablecoins | Value-stable currency | MakerDAO (DAI), USDC | Crypto/fiat-backed |
| Derivatives | Futures/options | dYdX, GMX | On-chain derivatives |
| Insurance | Risk hedging | Nexus Mutual | Decentralized insurance pool |
| Yield Aggregator | Yield optimization | Yearn Finance | Automatic strategy switching |

**AMM (Automated Market Maker) Principle:**

The core of Uniswap is the constant product formula `x * y = k`. By keeping the product of the quantities of two tokens in a liquidity pool constant at all times, token exchange is realized without an order book.

### 6.2 Token Standards

Tokens on Ethereum are standardized by ERCs (Ethereum Requests for Comments).

| Standard | Type | Features | Use Cases |
|----------|------|----------|-----------|
| ERC-20 | Fungible Token (FT) | Each token has equal value | Currency (USDC), Governance (UNI) |
| ERC-721 | Non-Fungible Token (NFT) | Each token has a unique identifier | Digital art, game items |
| ERC-1155 | Multi-Token | Manages FTs and NFTs in a single contract | Game items in general |
| ERC-4626 | Tokenized Vault | Standard for yield-bearing tokens | Lending pools, staking |

---

## 7. Blockchain Limitations and Challenges

### 7.1 Scalability Trilemma

The scalability trilemma proposed by Vitalik Buterin is the proposition that it is extremely difficult to simultaneously maximize the three properties of "decentralization," "security," and "scalability" in a distributed system.

```
Scalability Trilemma:

          Decentralization
             /\
            /  \
           /    \
          / Ideal \
         / (impossible)\
        /    zone      \
       /──────────────\
      /                \
  Security ──────── Scalability

  Position of Each Chain:
  ─────────────────────────────────────────
  Bitcoin:
    Decentralization ★★★★★  Security ★★★★★  Scalability ★☆☆☆☆
    → Approx. 7 TPS, fully decentralized, highest security

  Ethereum (L1):
    Decentralization ★★★★☆  Security ★★★★★  Scalability ★★☆☆☆
    → Approx. 15 TPS, high decentralization, security maintained after PoS transition

  Solana:
    Decentralization ★★☆☆☆  Security ★★★☆☆  Scalability ★★★★★
    → Thousands of TPS, decentralization constrained by high validator requirements

  Ethereum + L2:
    Decentralization ★★★★☆  Security ★★★★☆  Scalability ★★★★☆
    → Scales on L2 while inheriting L1 security

  Reference (Centralized):
  Visa: 65,000 TPS (peak)
  → Completely centralized but with overwhelming scalability
```

### 7.2 Security Challenges

**Smart Contract Vulnerabilities:**

Since smart contracts are difficult to modify after deployment, vulnerabilities can lead to significant asset losses.

| Vulnerability | Description | Major Incident |
|---------------|-------------|----------------|
| Reentrancy Attack | Same function is recursively called during external call | The DAO incident (2016, approx. $60 million) |
| Integer Overflow | Arithmetic overflow | Beauty Chain incident (2018) |
| Flash Loan Attack | Uncollateralized borrowing → price manipulation → repayment in 1 Tx | bZx attack (2020) |
| Oracle Manipulation | Manipulating prices from external data sources | Mango Markets (2022, approx. $110 million) |
| Access Control Flaws | Lack of permission checks | Parity Wallet freeze (2017, approx. $150 million) |

### 7.3 Social and Regulatory Challenges

- **Environmental Issues**: Enormous energy consumption of PoW (mitigating trend with PoS transition)
- **Regulatory Uncertainty**: Regulations differ by country and are changing rapidly
- **Speculation and Volatility**: Extreme price fluctuations of crypto assets
- **Money Laundering**: Risk of exploiting anonymity for illicit fund transfers
- **True Decentralization**: Power concentration in mining pools and large stakers
- **Private Key Management**: User self-custody responsibility and UX challenges

---

## 8. Blockchain Implementation in Python

### 8.1 Minimal Blockchain Implementation

The following is an educational blockchain implementation demonstrating the basic mechanisms of hash chain structure, block generation, and tamper detection.

```python
"""
Code Example 5: Complete educational blockchain implementation
Includes hash chain, PoW mining, and tamper detection
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Transaction:
    """Data class representing a transaction (trade record)."""
    sender: str
    receiver: str
    amount: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "timestamp": self.timestamp,
        }

    def compute_hash(self) -> str:
        tx_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(tx_string.encode("utf-8")).hexdigest()


@dataclass
class Block:
    """Data class representing a block."""
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    nonce: int = 0
    hash: str = ""

    def compute_hash(self) -> str:
        """Compute the SHA-256 hash of the block."""
        block_data = {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode("utf-8")).hexdigest()


class Blockchain:
    """
    Educational blockchain implementation.
    Supports PoW mining, chain validation, and tamper detection.
    """

    def __init__(self, difficulty: int = 2):
        """
        Args:
            difficulty: PoW difficulty (number of leading zeros in hash)
        """
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        """Generate the Genesis Block."""
        genesis = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0" * 64,
        )
        genesis.hash = self._proof_of_work(genesis)
        self.chain.append(genesis)

    def _proof_of_work(self, block: Block) -> str:
        """Execute PoW mining to find a valid hash."""
        target = "0" * self.difficulty
        block.nonce = 0
        computed_hash = block.compute_hash()

        while not computed_hash.startswith(target):
            block.nonce += 1
            computed_hash = block.compute_hash()

        return computed_hash

    def add_transaction(self, sender: str, receiver: str, amount: float) -> None:
        """Add a new transaction to the pending list."""
        if amount <= 0:
            raise ValueError("Transfer amount must be positive")
        tx = Transaction(sender=sender, receiver=receiver, amount=amount)
        self.pending_transactions.append(tx)

    def mine_pending_transactions(self, miner_address: str) -> Block:
        """
        Mine a new block containing pending transactions.
        """
        # Add mining reward transaction
        reward_tx = Transaction(
            sender="NETWORK",
            receiver=miner_address,
            amount=6.25,  # Block reward
        )
        transactions = self.pending_transactions + [reward_tx]

        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=transactions,
            previous_hash=self.chain[-1].hash,
        )

        # PoW mining
        new_block.hash = self._proof_of_work(new_block)
        self.chain.append(new_block)

        # Clear pending list
        self.pending_transactions = []
        return new_block

    def is_chain_valid(self) -> bool:
        """Verify the integrity of the entire blockchain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Recompute hash and compare
            if current.hash != current.compute_hash():
                print(f"  [Fraud] Block {i}: Hash mismatch")
                return False

            # Verify link to previous block
            if current.previous_hash != previous.hash:
                print(f"  [Fraud] Block {i}: previous_hash mismatch")
                return False

            # Verify PoW condition
            if not current.hash.startswith("0" * self.difficulty):
                print(f"  [Fraud] Block {i}: PoW condition not met")
                return False

        return True

    def get_balance(self, address: str) -> float:
        """Calculate the balance for a given address."""
        balance = 0.0
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == address:
                    balance -= tx.amount
                if tx.receiver == address:
                    balance += tx.amount
        return balance

    def print_chain(self) -> None:
        """Display the contents of the blockchain."""
        for block in self.chain:
            print(f"\n--- Block {block.index} ---")
            print(f"  Timestamp:     {time.ctime(block.timestamp)}")
            print(f"  Previous Hash: {block.previous_hash[:16]}...")
            print(f"  Hash:          {block.hash[:16]}...")
            print(f"  Nonce:         {block.nonce}")
            print(f"  Transactions:  {len(block.transactions)}")
            for tx in block.transactions:
                print(f"    {tx.sender} -> {tx.receiver}: {tx.amount}")


# --- Usage Example ---
print("=== Educational Blockchain ===\n")

bc = Blockchain(difficulty=2)

# Add transactions
bc.add_transaction("Alice", "Bob", 10.0)
bc.add_transaction("Bob", "Charlie", 5.0)
print("Mining Block 1...")
block1 = bc.mine_pending_transactions("Miner1")
print(f"Block 1 mining complete (nonce: {block1.nonce})")

bc.add_transaction("Charlie", "Alice", 3.0)
bc.add_transaction("Alice", "Dave", 2.0)
print("Mining Block 2...")
block2 = bc.mine_pending_transactions("Miner1")
print(f"Block 2 mining complete (nonce: {block2.nonce})")

# Display entire chain
bc.print_chain()

# Check balances
print(f"\n--- Balances ---")
for addr in ["Alice", "Bob", "Charlie", "Dave", "Miner1"]:
    print(f"  {addr}: {bc.get_balance(addr):.2f}")

# Integrity verification
print(f"\nIs chain valid: {bc.is_chain_valid()}")

# --- Tampering Demo ---
print("\n=== Tampering Demo ===")
print("Tampering with Block 1's transaction...")
bc.chain[1].transactions[0] = Transaction("Alice", "Bob", 1000.0)
print(f"Is chain valid after tampering: {bc.is_chain_valid()}")
```

---

## 9. Anti-Patterns and Design Pitfalls

### 9.1 Anti-Pattern 1: Unprotected Design Against Reentrancy Attacks

**Problem**: The pattern of updating state after transferring funds to an external contract is a fatal vulnerability that allows reentrancy attacks. In The DAO incident of 2016, approximately $60 million worth of ETH was drained due to this vulnerability.

```
Reentrancy Attack Mechanism:

  ┌─────────────────┐        ┌──────────────────┐
  │ Attacker Contract│        │ Vulnerable Contract│
  │                  │        │                   │
  │ Call withdraw()  │───────→│ 1. Check balance: OK│
  │                  │        │ 2. Send ETH        │
  │  ┌──────────────┐│←───────│     ↓              │
  │  │ receive()    ││        │                    │
  │  │  → Call      ││───────→│ 1. Check balance: OK!│
  │  │  withdraw()  ││        │  (not yet updated) │
  │  │  again       ││        │ 2. Send ETH        │
  │  └──────────────┘│←───────│     ↓              │
  │  ... (repeats)   │        │ 3. Update balance=0│
  │                  │        │  (updated only now) │
  └─────────────────┘        └──────────────────┘

  Vulnerable code (pseudocode):
  ─────────────────────────────
  function withdraw(amount):
      require(balances[msg.sender] >= amount)
      msg.sender.call{value: amount}("")  # ← Attacker's receive() is called here
      balances[msg.sender] -= amount       # ← State update comes after the transfer!

  Fixed version (Checks-Effects-Interactions pattern):
  ─────────────────────────────
  function withdraw(amount):
      require(balances[msg.sender] >= amount)   # Checks: Verify conditions
      balances[msg.sender] -= amount             # Effects: Update state (before transfer!)
      msg.sender.call{value: amount}("")         # Interactions: External call

  Additional safety measures:
  - Use ReentrancyGuard modifier (OpenZeppelin implementation)
  - Pull Payment pattern (delegate transfers to the recipient)
```

### 9.2 Anti-Pattern 2: "Put Everything on the Blockchain" Design

**Problem**: Attempting to store large amounts of data or high-frequency updates directly on the blockchain leads to exploding gas costs and reduced practicality.

```
Incorrect Design:
─────────────────────────
  X Store image data directly on the blockchain
     → 1 MB image ≈ thousands of dollars in gas costs
     → All nodes permanently retain the image data

  X Record every second of sensor data on-chain
     → Astronomical gas costs
     → Waste of network bandwidth

  X Replace a single organization's internal database with blockchain
     → Distributed consensus overhead is wasteful
     → A traditional RDBMS is superior in every way

Correct Design:
─────────────────────────
  O Off-chain storage + on-chain hash
     → Store data body on IPFS/Arweave
     → Record only the hash on the blockchain
     → Separate tamper detection from data availability

  O Off-chain computation + on-chain verification
     → Execute computation on Layer 2 or off-chain
     → Submit only the proof of results to Layer 1

  O Determine whether blockchain is truly necessary
     → Does a trust problem exist among multiple stakeholders?
     → Is there a situation where a central administrator cannot be trusted?
     → Is tamper resistance indispensable?
     → If none of the above apply, a conventional DB is sufficient
```

### 9.3 Anti-Pattern 3: "Silver Bullet" Thinking About Blockchain

The tendency to apply blockchain as a universal solution is a typical error in technology selection. The following decision criteria should be referenced.

**Blockchain Adoption Decision Flowchart:**

```
  Q1: Do multiple organizations/participants need to share data?
  │
  ├─ No → Use a conventional database
  │
  └─ Yes
      │
      Q2: Is there a single trusted administrator that all participants trust?
      │
      ├─ Yes → Use a conventional database
      │
      └─ No
          │
          Q3: Is tamper resistance/transparency of data important?
          │
          ├─ No → A distributed database (CockroachDB, etc.) is sufficient
          │
          └─ Yes
              │
              Q4: Are the participants unspecified/general public?
              │
              ├─ Yes → Public chain (Ethereum, Solana, etc.)
              │
              └─ No → Consortium chain (Hyperledger, etc.)
```

### 9.4 Security Best Practices

The following summarizes guidelines for ensuring security in smart contract development.

```
Smart Contract Development Security Checklist:
─────────────────────────────────────────────────

  Design Phase:
  □ Strictly follow the Checks-Effects-Interactions pattern
  □ Apply ReentrancyGuard (OpenZeppelin) to functions with external calls
  □ Principle of least privilege: Grant only minimum necessary permissions to each function
  □ Access control: Appropriately use modifiers such as onlyOwner, onlyRole
  □ Integer arithmetic: Leverage Solidity 0.8+ built-in overflow detection
  □ Pull Payment pattern: Design transfers so users withdraw themselves

  Testing Phase:
  □ Unit tests: Cover normal and abnormal cases for all functions
  □ Fuzz testing: Use Foundry's forge-std/Test
  □ Invariant testing: Verify contract invariants with assertions
  □ Fork testing: Fork mainnet state for testing close to production environment

  Pre-Deployment:
  □ External audit: Review by specialized security audit firms
  □ Bug bounty: Establish a reward program for vulnerability reports
  □ Formal verification: Mathematically prove correctness with tools like Certora, Halmos
  □ Long-term testing on testnet

  Post-Deployment:
  □ Monitoring system: Detect abnormal transaction patterns
  □ Circuit Breaker: Temporarily pause contract when issues arise
  □ Timelock: Set delays for critical parameter changes
  □ Multisig: Manage admin privileges with a multisig wallet
```

```python
"""
Code Example 6: Demonstration of the Checks-Effects-Interactions pattern
Comparison of code vulnerable to reentrancy attacks and secure code
"""
from typing import Dict


class VulnerableVault:
    """
    Mimics a vulnerable vault contract.
    Vulnerable to reentrancy attacks because state is updated after external calls.
    """

    def __init__(self):
        self.balances: Dict[str, float] = {}

    def deposit(self, user: str, amount: float) -> None:
        self.balances[user] = self.balances.get(user, 0) + amount

    def withdraw(self, user: str, amount: float, callback=None) -> None:
        """Vulnerable withdrawal: updates balance after transfer (callback)."""
        balance = self.balances.get(user, 0)
        if balance < amount:
            raise ValueError("Insufficient balance")

        # Interaction before Effects → vulnerable to reentrancy
        if callback:
            callback(amount)  # ← Attacker can call withdraw again

        self.balances[user] = balance - amount  # ← Still referencing the old balance


class SecureVault:
    """
    Mimics a secure vault contract.
    Applies the Checks-Effects-Interactions pattern.
    """

    def __init__(self):
        self.balances: Dict[str, float] = {}
        self._locked = False  # ReentrancyGuard

    def deposit(self, user: str, amount: float) -> None:
        self.balances[user] = self.balances.get(user, 0) + amount

    def withdraw(self, user: str, amount: float, callback=None) -> None:
        """Secure withdrawal: performs external call after state update."""
        # ReentrancyGuard
        if self._locked:
            raise RuntimeError("Reentrancy attack detected: function is locked")
        self._locked = True

        try:
            # Checks: Verify conditions
            balance = self.balances.get(user, 0)
            if balance < amount:
                raise ValueError("Insufficient balance")

            # Effects: Update state (executed before external call)
            self.balances[user] = balance - amount

            # Interactions: External call
            if callback:
                callback(amount)
        finally:
            self._locked = False


# --- Reentrancy Attack Simulation ---
print("=== Reentrancy Attack Simulation ===\n")

# Vulnerable contract
print("--- VulnerableVault (Vulnerable) ---")
vault = VulnerableVault()
vault.deposit("attacker", 10.0)
stolen = [0.0]
attack_count = [0]


def malicious_callback(amount: float) -> None:
    """Mimics the attacker's receive function."""
    attack_count[0] += 1
    stolen[0] += amount
    if attack_count[0] < 3:
        try:
            vault.withdraw("attacker", amount, malicious_callback)
        except (ValueError, RecursionError):
            pass


vault.withdraw("attacker", 10.0, malicious_callback)
print(f"  Amount stolen: {stolen[0]}  Reentrancy count: {attack_count[0]}")

# Secure contract
print("\n--- SecureVault (Secure) ---")
secure = SecureVault()
secure.deposit("attacker", 10.0)


def malicious_callback_secure(amount: float) -> None:
    try:
        secure.withdraw("attacker", amount, malicious_callback_secure)
    except RuntimeError as e:
        print(f"  [Defense successful] {e}")


secure.withdraw("attacker", 10.0, malicious_callback_secure)
print(f"  Attacker's balance: {secure.balances.get('attacker', 0)}")
```

---

## 10. Real-World Applications of Blockchain

### 10.1 Major Use Cases

Blockchain technology is being applied in diverse areas beyond finance.

| Domain | Use Case | Specific Example | Blockchain Advantage |
|--------|----------|------------------|---------------------|
| Supply Chain | Product tracking | IBM Food Trust | Tamper-resistant history management |
| Healthcare | Electronic medical record sharing | MedRec (MIT) | Patient-driven data management |
| Real Estate | Registry management | Swedish Land Registry | Reduction of intermediary costs |
| Intellectual Property | Copyright management | Ascribe | Proof of priority via timestamps |
| Voting | Electronic voting | Voatz | Balancing transparency and tamper resistance |
| Energy | P2P electricity trading | Power Ledger | Electricity trading without intermediaries |
| Gaming | Digital asset ownership | Axie Infinity | True digital ownership |
| Identity | Decentralized ID (DID) | Microsoft ION | Self-sovereign identity |

### 10.2 CBDC (Central Bank Digital Currency)

Central banks in various countries are researching and developing digital versions of legal tender using distributed ledger technology. China's digital yuan (e-CNY) has undergone large-scale pilot experiments, and the European Central Bank's digital euro is also under development. CBDCs are inherently centralized as they are issued and managed by central banks, but they leverage some advantages of distributed ledger technology (programmability, traceability).

### 10.3 Web3 and the Vision of a Decentralized Internet

Web3 is a new paradigm for the internet based on blockchain technology, aiming for users to reclaim ownership of their data.

```
Evolution of the Web:

  Web 1.0 (1990s~): Read-only
    Static HTML pages, one-way information distribution
    Examples: Personal homepages, Yahoo! Directory

  Web 2.0 (2000s~): Read-write
    User-generated content, social media, cloud
    Examples: Facebook, YouTube, Twitter
    Challenge: Platforms monopolize data

  Web 3.0 (2020s~): Read-write + Own
    Data sovereignty through blockchain
    Examples: DeFi, NFT, DAO, Decentralized social media
    Challenges: UX, scalability, regulation

  ┌────────────┬──────────┬──────────┬──────────┐
  │            │ Web 1.0  │ Web 2.0  │ Web 3.0  │
  ├────────────┼──────────┼──────────┼──────────┤
  │ Data Mgmt  │Distributed│Centralized│Distributed│
  │ Auth       │ None     │ OAuth etc│ Wallet   │
  │ Payment    │Credit card│ E-payment│ Crypto   │
  │ Governance │ None     │ Corporate│ DAO      │
  └────────────┴──────────┴──────────┴──────────┘
```

---

## 11. Practical Exercises

### Exercise 1: [Basic] Manual Blockchain Construction and Hash Chain Verification

```
Follow the steps below to manually construct a chain of 3 blocks.
Use Python's hashlib.

Step 1: Create the Genesis Block
  - index: 0
  - previous_hash: "0000000000000000000000000000000000000000000000000000000000000000"
  - data: "Genesis Block"
  - timestamp: Any fixed value (e.g., "2024-01-01T00:00:00")
  - hash = SHA-256(index + previous_hash + data + timestamp)

Step 2: Create Block 1
  - index: 1
  - previous_hash: Hash of the Genesis Block
  - data: "Alice sends 10 BTC to Bob"
  - timestamp: "2024-01-01T00:10:00"

Step 3: Create Block 2
  - index: 2
  - previous_hash: Hash of Block 1
  - data: "Bob sends 5 BTC to Charlie"
  - timestamp: "2024-01-01T00:20:00"

Step 4: Verification
  (a) Display the hash of each block and confirm the chain linkage
  (b) If Block 1's data is tampered to "Alice sends 100 BTC to Bob",
      confirm from which block the hash changes
  (c) Implement a validation function that detects tampering

Expected Learning Outcomes:
  - Gain hands-on understanding of hash chain mechanics
  - Confirm that tampering at one point cascades through the entire chain
```

### Exercise 2: [Applied] Analysis of the Relationship Between PoW Difficulty and Computational Cost

```python
# Extend the following code to analyze the relationship between
# PoW difficulty and computational cost.
#
# Tasks:
# (1) Vary difficulty from 1 to 5 and measure mining time at each level
# (2) Calculate the multiplication factor of average computation count per difficulty increase
# (3) Output results in tabular format and confirm exponential growth
# (4) (Advanced) Take the average of 5 trials and report statistical variance

import hashlib
import time
import statistics


def mine_with_stats(data: str, prev_hash: str, difficulty: int) -> dict:
    """
    Execute PoW mining and return statistics.

    Return value:
        {
            "difficulty": int,
            "nonce": int,
            "hash": str,
            "attempts": int,
            "elapsed_seconds": float,
        }
    """
    target = "0" * difficulty
    nonce = 0
    start = time.time()

    while True:
        text = f"{prev_hash}{data}{nonce}"
        h = hashlib.sha256(text.encode()).hexdigest()
        nonce += 1
        if h[:difficulty] == target:
            return {
                "difficulty": difficulty,
                "nonce": nonce - 1,
                "hash": h,
                "attempts": nonce,
                "elapsed_seconds": time.time() - start,
            }

# Extend freely from here
```

### Exercise 3: [Advanced] Multi-Node Consensus Simulation

```
Design a program that simulates consensus on a distributed network.

Requirements:
(1) Create 5 instances of a Node class
(2) Each node holds its own copy of the blockchain
(3) Implement a mechanism to broadcast transactions
(4) Each node independently attempts mining, and the first node to succeed
    propagates the block to other nodes
(5) Other nodes verify the block's validity before adding it to their chain
(6) Implement the "Longest Chain Rule":
    When chain lengths differ, adopt the longest valid chain as canonical

Advanced Tasks:
(7) Designate one node as a "malicious node" and have it generate
    blocks containing fraudulent transactions, then observe the behavior
(8) Create a situation where a fork (chain split) occurs and observe
    the resolution process

Design Hints:
  - Node class: Has blockchain, pending_transactions, peers as attributes
  - Network class: A simulation class that mediates inter-node communication
  - Each node's validate_block() method detects fraudulent blocks
```

---

## 12. FAQ

### Q1: What problems is blockchain suitable for? What situations is it not suitable for?

**Suitable situations:**
- **Data sharing among multiple stakeholders**: When multiple organizations that cannot trust each other manage common data (e.g., international supply chain tracking, trade finance)
- **Records requiring tamper resistance**: Real estate registration, academic credentials, medical records, and other situations where record legitimacy is critical
- **Eliminating intermediaries**: International remittances (processes that conventionally take days through multiple banks can be sent directly via blockchain)
- **Processes requiring transparency**: Voting systems, donation tracking, public procurement

**Unsuitable situations:**
- **Data management within a single organization**: Within an organization, administrators can be trusted, so conventional databases are faster and less costly
- **Situations requiring high-speed processing**: Real-time systems requiring thousands of TPS or more (though the threshold is rising with Layer 2 development)
- **When privacy is paramount**: Public chains expose all data (privacy technologies such as zero-knowledge proofs are still developing)
- **Storing large amounts of data**: Blockchain storage costs are thousands of times higher than conventional cloud storage

### Q2: What is Bitcoin's "halving"? Why is it important?

The halving is a mechanism where mining rewards are cut in half approximately every 4 years (210,000 blocks).

| Period | Block Reward | Cumulative Issuance Rate (approx.) |
|--------|-------------|-----------------------------------|
| 2009 (launch) | 50 BTC | - |
| 2012 (1st halving) | 25 BTC | ~50% |
| 2016 (2nd halving) | 12.5 BTC | ~75% |
| 2020 (3rd halving) | 6.25 BTC | ~87.5% |
| 2024 (4th halving) | 3.125 BTC | ~93.75% |
| Around 2140 | 0 BTC | 100% (21 million BTC cap) |

The halving is important because Bitcoin's supply schedule is completely predictable, realizing a deflationary monetary design. While central banks determine the issuance volume of fiat currency at their discretion, Bitcoin's supply is determined by mathematical rules.

### Q3: What are the differences between private and public chains? Which should you choose?

| Property | Public Chain | Consortium Chain | Private Chain |
|----------|-------------|-----------------|---------------|
| Participation | Anyone can join freely | Approved organizations only | Within a single organization |
| Transparency | All transactions fully public | Shared among participants | Internal only |
| Consensus | PoW, PoS, etc. | Often BFT-based | PoA, Raft, etc. |
| Speed | Slow (7-15 TPS) | Fast (1000+ TPS) | Very fast |
| Decentralization | High | Moderate | Low |
| Representative examples | Bitcoin, Ethereum | Hyperledger Fabric, R3 Corda | Enterprise internal systems |
| Suitable uses | DeFi, NFT, public goods | Supply chain, trade finance | Internal audit trails |

Consortium or private chains are often adopted for enterprise use. However, the criticism that "how is a private chain different from a distributed database?" remains strong, along with the argument that the essential value of blockchain (decentralization of trust) is lost.

### Q4: Can bugs in smart contracts be fixed?

In principle, the code of a deployed smart contract is immutable. However, the following strategies exist:

1. **Proxy Pattern**: Design that delegates the logic portion to a separate contract, allowing the Proxy contract's reference target to be changed. OpenZeppelin's UUPS and Transparent Proxy are representative implementations.
2. **Migration**: Deploy a new contract and encourage users to migrate. The old contract is disabled using a pause function.
3. **Governance**: A DAO (Decentralized Autonomous Organization) vote decides whether to approve the upgrade.

In all cases, thorough testing and auditing before deployment is the most important measure.

### Q5: What are Zero-Knowledge Proofs (ZKP)? How do they relate to blockchain?

A zero-knowledge proof is a cryptographic protocol that "proves a proposition is true without revealing any information about the proposition's content." In blockchain, it is attracting attention primarily for the following two uses:

1. **Scalability (ZK Rollup)**: Compresses thousands of transactions into a single zero-knowledge proof and submits it to Layer 1, dramatically improving throughput.
2. **Privacy**: Conceals transaction amounts and sender/receiver identities while proving the legitimacy of transactions (such as Zcash's zk-SNARKs).

---

## 13. Glossary

| Term | English | Definition |
|------|---------|-----------|
| Hash Function | Hash Function | A function that transforms arbitrary-length input into fixed-length output |
| Merkle Tree | Merkle Tree | A hash binary tree that efficiently verifies data integrity |
| Nonce | Nonce | Number Used Once; a value searched for in PoW that is used only once |
| Consensus | Consensus | A mechanism for distributed nodes to agree on the same state |
| Finality | Finality | The state where a transaction is confirmed and cannot be reversed |
| Staking | Staking | Depositing tokens to become a validator in PoS |
| Slashing | Slashing | A penalty that confiscates the stake of validators who commit misconduct |
| Gas | Gas | The unit of computational cost required for smart contract execution on the EVM |
| DApp | Decentralized Application | A decentralized application |
| TVL | Total Value Locked | The total value of assets deposited in DeFi protocols |
| MEV | Maximal Extractable Value | Profit obtained by manipulating transaction order within a block |
| DAO | Decentralized Autonomous Organization | A decentralized autonomous organization governed by smart contracts |

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying how things work.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to applications. We recommend firmly understanding the foundational concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently utilized in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Hash Chain | Each block contains the hash of the previous block, forming a tamper detection chain |
| Merkle Tree | A hash binary tree that can verify transaction inclusion in O(log n) |
| PoW | Consensus using the asymmetry of computational puzzles (hard to solve, easy to verify) |
| PoS | Block generation rights assigned based on stake amount. 99.95% energy savings vs PoW |
| BFT-based | Immediate finality through explicit voting. Constrained by node count |
| Public Key Cryptography | One-way derivation of Private Key → Public Key → Address manages identity |
| Smart Contracts | Programs that execute automatically on the blockchain. Deterministically executed on the EVM |
| DeFi | Provides financial services through smart contracts, eliminating financial intermediaries |
| Layer 2 | Scaling technology that improves throughput while inheriting L1 security |
| Trilemma | Simultaneously maximizing decentralization, security, and scalability is extremely difficult |

---

## Recommended Next Guides


---

## References

1. Nakamoto, S. "Bitcoin: A Peer-to-Peer Electronic Cash System." 2008. https://bitcoin.org/bitcoin.pdf
2. Buterin, V. "Ethereum Whitepaper: A Next-Generation Smart Contract and Decentralized Application Platform." 2014. https://ethereum.org/whitepaper
3. Antonopoulos, A. M. *Mastering Bitcoin: Programming the Open Blockchain.* 2nd ed., O'Reilly Media, 2017. ISBN: 978-1491954386
4. Antonopoulos, A. M. & Wood, G. *Mastering Ethereum: Building Smart Contracts and DApps.* O'Reilly Media, 2018. ISBN: 978-1491971949
5. Lamport, L., Shostak, R., & Pease, M. "The Byzantine Generals Problem." *ACM Transactions on Programming Languages and Systems*, Vol. 4, No. 3, 1982, pp. 382-401.
6. Szabo, N. "Smart Contracts: Building Blocks for Digital Markets." 1996. https://www.fon.hum.uva.nl/rob/Courses/InformationInSpeech/CDROM/Literature/LOTwinterschool2006/szabo.best.vwh.net/smart_contracts_2.html
7. Wood, G. "Ethereum: A Secure Decentralised Generalised Transaction Ledger (Yellow Paper)." 2014. https://ethereum.github.io/yellowpaper/paper.pdf
8. Ethereum Foundation. "The Merge." https://ethereum.org/en/roadmap/merge/
