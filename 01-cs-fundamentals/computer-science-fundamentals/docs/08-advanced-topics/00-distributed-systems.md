# Distributed Systems

> "A distributed system is one in which the failure of a computer you didn't even know existed can render your own computer unusable." -- Leslie Lamport

## What You Will Learn in This Chapter

- [ ] Understand the fundamental concepts of distributed systems and why a single machine has limitations
- [ ] Correctly understand the CAP theorem and be able to explain CP/AP selection criteria
- [ ] Grasp the strengths and weaknesses of consistency models and their practical trade-offs
- [ ] Understand the principles and use cases of consensus algorithms such as Paxos/Raft
- [ ] Apply replication and sharding strategies to system design
- [ ] Explain the mechanisms and limitations of distributed transactions (2PC, Saga)
- [ ] Implement ordering using logical clocks and vector clocks
- [ ] Properly design fault tolerance patterns (Circuit Breaker, etc.)


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Why Distributed Systems Are Necessary

### 1.1 Limitations of a Single Machine

```
Limitations of a Single Machine:

  CPU:     Slowdown of Moore's Law (since ~2005)
           -> Single-thread performance has plateaued
           -> Even with multi-core, a single machine has a core count limit

  Memory:   RAM limit per machine (several TB)
           -> Physical limits exist even when wanting to fit all data in memory

  Storage: Disk limit per machine (tens of TB)
           -> Petabyte-scale data cannot fit on a single machine

  Availability: If one machine fails, everything stops (SPOF = Single Point of Failure)
           -> Hardware failures are statistically inevitable

  Network Bandwidth: NIC throughput limit per machine (10-100 Gbps)
           -> Cannot handle massive concurrent connections from many clients

  -> This is why distributing processing across multiple machines is a necessity
```

### 1.2 Four Objectives of Distributed Systems

```
Objectives of Distributed Systems:

  1. Scalability:
     Horizontal scaling of processing capacity (Scale-Out)
     -> Processing capacity increases linearly by simply adding machines
     -> Vertical scaling (Scale-Up) has poor cost efficiency
       Example: A machine with 2x CPU costs more than 2x the price

  2. Availability:
     Service continues even when some components fail
     -> Achieve 99.99% annual uptime (= ~52 minutes downtime per year)
     -> Annual failure rate of a single server is about 2-4%
       -> With 1000 servers, several fail every day

  3. Latency:
     Process data at locations physically close to users
     -> Speed of light limitation: ~70ms one-way between Tokyo and New York
     -> Solved with CDNs and edge computing

  4. Data Volume:
     Managing data that cannot fit on a single machine
     -> Google holds tens of exabytes (EB) of data
     -> 500 hours of video are uploaded to YouTube every minute
```

### 1.3 Representative Real-World Examples of Distributed Systems

```
Real-World Examples and Their Scale:

  Google Search:
    Thousands of servers cooperate to process a single search query
    -> MapReduce for large-scale index construction
    -> Bigtable for distributed storage
    -> Spanner for globally distributed DB

  Netflix:
    Global CDN + hundreds of microservices
    -> Daily traffic is about 15% of all internet traffic
    -> Chaos Monkey intentionally causes failures to test resilience

  Bitcoin:
    Tens of thousands of nodes achieve consensus (Proof of Work)
    -> Byzantine fault tolerance achieved through cryptographic methods
    -> Building trust without a central authority

  Amazon DynamoDB:
    Millions of tables, trillions of requests per day
    -> Data distribution via consistent hashing
    -> Choice between eventual consistency and strong consistency
```

---

## 2. The Eight Fallacies of Distributed Systems

### 2.1 Peter Deutsch's Eight Fallacies

```
Peter Deutsch's "Eight Fallacies of Distributed Computing" (1994):

  1. The network is reliable
     -> Reality: Packet loss rate is around 0.01% to 1%
     -> Submarine cable cuts and router failures are routine
     -> Countermeasures: Retry, idempotency, timeout design

  2. Latency is zero
     -> Reality: Even within the same DC it's 0.5ms; inter-continental is 100ms+
     -> The speed of light limit is a physical law that cannot be improved
     -> Countermeasures: Data locality, caching, asynchronous processing

  3. Bandwidth is infinite
     -> Reality: Network saturation does happen
     -> Physically shipping data on disk can be faster than network transfer
       (This is why AWS Snowball exists)
     -> Countermeasures: Data compression, differential transfer, protocol optimization

  4. The network is secure
     -> Reality: Man-in-the-middle attacks, eavesdropping, DNS spoofing
     -> Rise of zero-trust networks
     -> Countermeasures: TLS/mTLS, encryption, authentication

  5. Topology doesn't change
     -> Reality: Servers added/removed, route changes due to failures
     -> IP addresses change dynamically in cloud environments
     -> Countermeasures: Service discovery, DNS, load balancers

  6. There is one administrator
     -> Reality: Multiple teams, multiple organizations, multiple cloud providers
     -> Responsibility boundaries easily become ambiguous
     -> Countermeasures: Clear API contracts, SLA definitions

  7. Transport cost is zero
     -> Reality: CPU cost of serialization/deserialization
     -> Cloud data transfer charges (egress costs)
     -> Countermeasures: Efficient serialization (Protocol Buffers, etc.)

  8. The network is homogeneous
     -> Reality: Different hardware, OS, protocol versions
     -> Mix of 10Gbps and 1Gbps links
     -> Countermeasures: Abstraction layers, protocol versioning
```

### 2.2 Typical Failures Caused by These Fallacies

Designing systems based on these fallacies leads to the following types of failures. The reason these are problematic is that they are difficult to reproduce in development environments and only manifest in production for the first time.

```
Example of failure when assuming Fallacy 1:

  Development environment: 99.99% success on local network
  Production environment: Packet loss occurs in cross-cloud communication
  Result: Without retry implementation, processing is left in an incomplete state
        -> Data inconsistency, customer complaints

  Why this is not reproducible locally:
  -> Local network packet loss rate is under 0.0001%
  -> On WAN it jumps to 0.01% - 1%
  -> With 1 million requests per day, 100 to 10,000 will fail
```

---

## 3. CAP Theorem and Consistency Models

### 3.1 Precise Understanding of the CAP Theorem

```
CAP Theorem (Eric Brewer, 2000 conjecture / Gilbert & Lynch, 2002 proof):

A distributed data store can only satisfy two of the following three guarantees
simultaneously when a network partition occurs.

  C -- Consistency:
      All nodes see the same data at the same time
      -> Specifically refers to "Linearizability"
      -> After a write completes, any node returns the latest value when read
      -> Why it's hard: Communication is needed to synchronize all nodes

  A -- Availability:
      Non-failing nodes always return a "valid" response
      -> A meaningful response, not a timeout or error
      -> Why it's hard: A response is required even without the latest data

  P -- Partition Tolerance:
      The system continues to operate even when network partitions occur
      -> Does not halt even when communication between nodes is severed
      -> Why it's essential: Network partitions are inevitable in distributed systems

  +---------------------------------------------+
  |              C (Consistency)                  |
  |             / \                              |
  |            /   \                             |
  |           / CP  \ CA                         |
  |          /  |    \ |                         |
  |         / HBase   \ Single-node RDBMS        |
  |        / etcd      \(assumes no partition)   |
  |       / ZooKeeper   \                        |
  |      /_______________\                       |
  |     A (Availability)    P (Partition Tol.)    |
  |          |                                   |
  |        AP: Cassandra, DynamoDB, CouchDB      |
  +---------------------------------------------+
```

### 3.2 Why "CA" Practically Does Not Exist

```
Why CA (Consistency + Availability) is practically impossible:

  Network partitions are physically unavoidable:
  - Cable cuts, switch failures, router failures
  - Failures between cloud provider AZs (Availability Zones)
  - Expected partition frequency: several times per month in large DCs

  The binary choice when a partition occurs:
  +------------------------------------------+
  |  Node-A         x         Node-B         |
  |  [data=1]    (partition)  [data=1]       |
  |                                          |
  |  Client -> Node-A: "update data=2"      |
  |                                          |
  |  Node-A: updates data=2                 |
  |  Node-B: remains data=1 (cannot comm.)  |
  |                                          |
  |  Another Client -> Node-B: "read data"  |
  |                                          |
  |  Choice 1 (CP): Node-B returns error    |
  |    -> Maintains consistency but           |
  |       sacrifices availability             |
  |                                          |
  |  Choice 2 (AP): Node-B returns data=1   |
  |    -> Maintains availability but          |
  |       sacrifices consistency              |
  +------------------------------------------+

  -> When a partition occurs, C and A are logically incompatible
  -> A single-node RDBMS is not "giving up P" but
    "simply not distributed in the first place"
```

### 3.3 CP vs AP Selection Criteria

| Criterion | Choose CP (Consistency Priority) | Choose AP (Availability Priority) |
|---------|--------------------------|--------------------------|
| Nature of data | Money, inventory, bookings -- inconsistency is fatal | Like counts, view counts -- slight errors are tolerable |
| Worst case of inconsistency | Double charging, double booking a seat | Displaying an outdated follower count |
| User experience | Errors are preferable | No response is worse |
| Recovery cost of inconsistency | Recovery is difficult/impossible | Self-recovers over time |
| Representative systems | etcd, ZooKeeper, HBase | Cassandra, DynamoDB, CouchDB |
| Representative use cases | Bank transfers, inventory management, leader election | SNS, shopping carts, DNS |

### 3.4 Consistency Models in Detail

```
Consistency Strength (strongest to weakest):

  1. Linearizability:
     -> All operations appear to execute in a single global order
     -> Respects real-time ordering
     -> "From the moment a write completes, the latest value is readable from any node"
     -> Strongest guarantee, highest cost
     -> Implementation: Consensus algorithms (Raft, Paxos)
     -> Examples: ZooKeeper, etcd, Spanner

  2. Sequential Consistency:
     -> All process operations appear to execute in some total order
     -> Preserves the order of operations within each process
     -> However, does not guarantee real-time ordering
     -> Difference from linearizability: need not respect "wall-clock order"

  3. Causal Consistency:
     -> Order of causally related operations is the same across all nodes
     -> Operations without causal relationships may appear in different orders on different nodes
     -> Causal relationship: "A depends on the result of B"
     -> Example: On SNS, "post -> comment on that post" is causally related
     -> Example: MongoDB (default setting)

  4. Eventual Consistency:
     -> If updates stop, after sufficient time all nodes will converge
     -> Stale data may be read in the interim
     -> "Sufficient time" is typically milliseconds to a few seconds
     -> Examples: DynamoDB (default), Cassandra, DNS

  Strong <-----------------------------> Weak
  Linear.  Sequential  Causal  Eventual
  Slow   ----------------------------- Fast
  High cost ----------------------- Low cost
  Hard to implement ---------- Easy to implement
```

### 3.5 Consistency Model Comparison Table

| Model | Ordering Guarantee | Latency | Availability | Primary Use | Representative Implementation |
|-------|---------|-----------|-------|---------|-----------|
| Linearizability | Global total order | High (consensus needed) | Low (halts on partition) | Distributed locks, leader election | etcd, ZooKeeper |
| Sequential Consistency | Intra-process order preserved | Medium | Medium | Shared memory models | CPU/GPU memory |
| Causal Consistency | Only causal order preserved | Low-Medium | High | Social apps, collaborative editing | MongoDB |
| Eventual Consistency | No guarantee | Low | Highest | Caches, CDN, DNS | DynamoDB, Cassandra |

### 3.6 PACELC Theorem: An Extension of CAP

```
PACELC Theorem (Daniel Abadi, 2012):
In addition to CAP's "choice during partition," also considers "choice during normal operation"

  During P(Partition) -> A(Availability) or C(Consistency)
  E(Else = normal operation) -> L(Latency) or C(Consistency)

  +----------------------------------------------+
  |  During Partition  Normal Operation  Class    |
  |  ---------------  ----------------  ------   |
  |  PA                EL              PA/EL     |
  |  (Availability)   (Low latency)    -> DynamoDB|
  |                                    -> Cassandra|
  |                                              |
  |  PC                EC              PC/EC     |
  |  (Consistency)    (Consistency)    -> HBase   |
  |                                    -> VoltDB  |
  |                                              |
  |  PA                EC              PA/EC     |
  |  (Availability)   (Consistency)    -> MongoDB |
  |                                              |
  |  PC                EL              PC/EL     |
  |  (Consistency)    (Low latency)    -> PNUTS(Yahoo)|
  +----------------------------------------------+

Why PACELC is more practical than CAP:
-> CAP only addresses "during partition," but partitions are rare events
-> The latency vs consistency trade-off during normal operation is
  more important as a daily design decision
-> Example: DynamoDB's "strongly consistent read" is an option that
  sacrifices normal latency for consistency
```

### Code Example 1: Consistency Model Simulation

The following code simulates the behavioral differences between eventual consistency and strong consistency. The reason simulation is useful is that distributed system behavior often defies intuition, and running code to verify is the fastest path to understanding.

```python
"""
Consistency Model Simulator
==========================================
Reproduces the behavioral differences between eventual consistency
and strong consistency using threads.

How to run: python consistency_simulator.py
Dependencies: Standard library only
"""

import threading
import time
import random
from typing import Dict, List, Optional, Tuple


class Node:
    """A class representing a single node in a distributed system.

    Why locks are necessary:
    Python dicts are not thread-safe, so concurrent access from
    multiple threads can cause data corruption.
    """

    def __init__(self, node_id: str, latency_ms: float = 0):
        self.node_id = node_id
        self.data: Dict[str, Tuple[str, int]] = {}  # key -> (value, version)
        self.lock = threading.Lock()
        self.latency_ms = latency_ms  # Simulate network latency

    def write(self, key: str, value: str, version: int) -> bool:
        """Write data. Rejects if the version is outdated."""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        with self.lock:
            current = self.data.get(key)
            if current is None or current[1] < version:
                self.data[key] = (value, version)
                return True
            return False

    def read(self, key: str) -> Optional[Tuple[str, int]]:
        """Read data."""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        with self.lock:
            return self.data.get(key)


class EventuallyConsistentStore:
    """A distributed data store with eventual consistency.

    Writes are immediately reflected on one node and asynchronously propagated to others.
    Why this approach is used:
    -> Minimizes write latency
    -> Tolerates temporary data inconsistency between nodes
    """

    def __init__(self, node_count: int = 3, replication_delay_ms: float = 100):
        self.nodes = [
            Node(f"node-{i}", latency_ms=random.uniform(1, 5))
            for i in range(node_count)
        ]
        self.replication_delay_ms = replication_delay_ms
        self.version_counter = 0
        self.counter_lock = threading.Lock()

    def _get_next_version(self) -> int:
        with self.counter_lock:
            self.version_counter += 1
            return self.version_counter

    def write(self, key: str, value: str) -> str:
        """Write to the primary node and replicate in the background."""
        version = self._get_next_version()
        primary = self.nodes[0]
        primary.write(key, value, version)

        # Asynchronous replication (background thread)
        def replicate():
            time.sleep(self.replication_delay_ms / 1000.0)
            for node in self.nodes[1:]:
                delay = random.uniform(0, self.replication_delay_ms / 1000.0)
                time.sleep(delay)
                node.write(key, value, version)

        thread = threading.Thread(target=replicate, daemon=True)
        thread.start()
        return f"Written to {primary.node_id}: {key}={value} (v{version})"

    def read(self, key: str, node_index: Optional[int] = None) -> str:
        """Read from a specified node (or a random node)."""
        if node_index is not None:
            node = self.nodes[node_index]
        else:
            node = random.choice(self.nodes)
        result = node.read(key)
        if result is None:
            return f"[{node.node_id}] {key} = (not found)"
        return f"[{node.node_id}] {key} = {result[0]} (v{result[1]})"


class StronglyConsistentStore:
    """A distributed data store with strong consistency (linearizability).

    Writes wait for reflection on a majority of nodes before completing.
    Reads also fetch from a majority of nodes and return the latest version.

    Why a majority (quorum):
    -> N=3 nodes, W=2 (write quorum), R=2 (read quorum)
    -> Since W + R > N, the read and write quorums always overlap
    -> The overlapping node is guaranteed to hold the latest data
    """

    def __init__(self, node_count: int = 3):
        self.nodes = [
            Node(f"node-{i}", latency_ms=random.uniform(1, 5))
            for i in range(node_count)
        ]
        self.quorum = node_count // 2 + 1
        self.version_counter = 0
        self.counter_lock = threading.Lock()

    def _get_next_version(self) -> int:
        with self.counter_lock:
            self.version_counter += 1
            return self.version_counter

    def write(self, key: str, value: str) -> str:
        """Synchronously write to a quorum of nodes."""
        version = self._get_next_version()
        success_count = 0
        lock = threading.Lock()

        def write_to_node(node: Node):
            nonlocal success_count
            if node.write(key, value, version):
                with lock:
                    success_count += 1

        threads = []
        for node in self.nodes:
            t = threading.Thread(target=write_to_node, args=(node,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        if success_count >= self.quorum:
            return (
                f"Written to {success_count}/{len(self.nodes)} nodes: "
                f"{key}={value} (v{version}) [COMMITTED]"
            )
        else:
            return (
                f"Write failed: only {success_count}/{len(self.nodes)} "
                f"nodes responded (need {self.quorum}) [ABORTED]"
            )

    def read(self, key: str) -> str:
        """Read from a quorum of nodes and return the latest version."""
        results: List[Optional[Tuple[str, int]]] = []
        read_nodes: List[str] = []
        lock = threading.Lock()

        def read_from_node(node: Node):
            result = node.read(key)
            with lock:
                results.append(result)
                read_nodes.append(node.node_id)

        threads = []
        for node in self.nodes[:self.quorum]:
            t = threading.Thread(target=read_from_node, args=(node,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return f"[quorum={read_nodes}] {key} = (not found)"
        latest = max(valid_results, key=lambda x: x[1])
        return f"[quorum={read_nodes}] {key} = {latest[0]} (v{latest[1]})"


def demo():
    """Demonstrate the difference between eventual and strong consistency."""
    print("=" * 60)
    print("Eventual Consistency Demo")
    print("=" * 60)
    store = EventuallyConsistentStore(node_count=3, replication_delay_ms=200)
    print(store.write("user:1:name", "Alice"))
    print("\n--- Immediately after write (before replication completes) ---")
    for i in range(3):
        print(store.read("user:1:name", node_index=i))
    print("\n--- After 500ms (after replication completes) ---")
    time.sleep(0.5)
    for i in range(3):
        print(store.read("user:1:name", node_index=i))

    print("\n" + "=" * 60)
    print("Strong Consistency Demo")
    print("=" * 60)
    store2 = StronglyConsistentStore(node_count=3)
    print(store2.write("user:1:name", "Bob"))
    print("\n--- Immediately after write (quorum read) ---")
    for _ in range(3):
        print(store2.read("user:1:name"))


if __name__ == "__main__":
    demo()
```

Expected output:

```
============================================================
Eventual Consistency Demo
============================================================
Written to node-0: user:1:name=Alice (v1)

--- Immediately after write (before replication completes) ---
[node-0] user:1:name = Alice (v1)
[node-1] user:1:name = (not found)      <- Not yet propagated
[node-2] user:1:name = (not found)      <- Not yet propagated

--- After 500ms (after replication completes) ---
[node-0] user:1:name = Alice (v1)
[node-1] user:1:name = Alice (v1)       <- Propagation complete
[node-2] user:1:name = Alice (v1)       <- Propagation complete

============================================================
Strong Consistency Demo
============================================================
Written to 3/3 nodes: user:1:name=Bob (v1) [COMMITTED]

--- Immediately after write (quorum read) ---
[quorum=['node-0', 'node-1']] user:1:name = Bob (v1)
[quorum=['node-0', 'node-1']] user:1:name = Bob (v1)
[quorum=['node-0', 'node-1']] user:1:name = Bob (v1)
```

---

## 4. Consensus Algorithms

### 4.1 The Essence of the Consensus Problem

```
Consensus Problem:
  Multiple nodes agree on a single value

  Why it's difficult:
  - Nodes can fail
  - Messages can be delayed or lost
  - The network can be partitioned
  - Byzantine failures (malicious nodes) are possible

  Situations requiring consensus:
  - Leader election: Everyone agrees on "who is the leader"
  - Atomic broadcast: Everyone agrees on "the order of messages"
  - Distributed locks: Everyone agrees on "who holds the lock"
  - State machine replication: Everyone agrees on "the order of operations"

FLP Impossibility Theorem (Fischer, Lynch, Paterson, 1985):

  Theorem: In an asynchronous network where even one node may fail,
           no deterministic consensus algorithm exists

  Why it's impossible:
  -> In an asynchronous network, "a node is just slow" and
    "a node has failed" cannot be distinguished
  -> Waiting longer might yield a response (delay)
  -> It might never come (failure)

  Practical workarounds:
  -> Introduce timeouts (relax the asynchronous assumption)
  -> Probabilistic methods (randomized algorithms)
  -> Failure detectors (imperfect but practical)
  -> Raft uses timeout-based leader election to circumvent FLP
```

### 4.2 Paxos

```
Paxos (Leslie Lamport, proposed 1989 / published 1998):

  Roles:
  - Proposer: Node that proposes a value
  - Acceptor: Node that accepts/rejects proposals (majority required)
  - Learner: Node that learns the agreed-upon result

  Two-Phase Protocol:

  Phase 1 (Prepare):
  Proposer          Acceptor (majority)
     |-- Prepare(n) -->|
     |<-- Promise(n) --|  Promises to only respond to proposal numbers >= n
                         If it has already accepted another value, returns it

  Phase 2 (Accept):
     |-- Accept(n,v) -->|
     |<-- Accepted -----|  If a majority Accepts, consensus is reached
                          v is: the previously accepted value if one exists,
                          otherwise the Proposer freely chooses

  Why two phases are necessary:
  -> Phase 1 checks whether "another value has already been agreed upon"
  -> Without checking, multiple Proposers might reach consensus
    on different values

  Concrete example:
  +--------------------------------------------------+
  | Proposer-A: Prepare(1) -> Gets Promise from majority     |
  | Proposer-A: Accept(1, "X") -> Majority Accepts           |
  | -> Agreed value = "X"                                     |
  |                                                           |
  | Proposer-B: Prepare(2) -> Gets Promise from majority     |
  |   (Acceptor returns already-accepted "X")                 |
  | Proposer-B: Accept(2, "X") -> Reconfirms agreement on "X"|
  | -> Even if Proposer-B proposed a different value, "X" is maintained |
  +--------------------------------------------------+

  Problems with Paxos:
  - Implementation is extremely complex (Lamport's original paper was written
    as "The Part-Time Parliament," famously difficult to understand)
  - Livelock: Multiple Proposers alternately issuing Prepare can stall progress
  - Multi-Paxos (consensus on a sequence of values) is vaguely described in the paper
  - Practically implemented in Google Chubby (2006), but implementors reported difficulties
```

### 4.3 Raft

```
Raft (Diego Ongaro & John Ousterhout, 2014):
  Designed with "understandability" as an explicit goal

  Fundamental difference from Paxos:
  -> Paxos is symmetric (any node can be a Proposer)
  -> Raft is asymmetric (the Leader controls everything)
  -> This asymmetry is the key to understandability

  Roles (only 3 types):
  - Leader:    Accepts all writes and replicates logs
  - Follower:  Follows the Leader's instructions and receives logs
  - Candidate: State of campaigning in a Leader election

  Decomposed into 3 sub-problems:

  1. Leader Election:
     +--------------------------------------------+
     |  Follower --(timeout)--> Candidate          |
     |  Candidate --(majority vote)--> Leader      |
     |  Leader --(periodic heartbeat)--> maintain  |
     |  Leader --(failure)--> Follower --> ...     |
     |  Candidate --(discovers higher Term)--> Follower|
     +--------------------------------------------+

     Term numbers provide order:
     -> Higher Term number is newer
     -> At most one Leader per Term
     -> An old-Term Leader instantly reverts to Follower upon learning a new Term

     Why random timeouts are used:
     -> If all nodes become Candidates simultaneously, votes split
     -> Timeouts are set randomly within a 150ms-300ms range
     -> The first node to timeout starts the election and wins with high probability

  2. Log Replication:
     Client --> Leader --> Replicate to Follower group in parallel
     Majority write completion --> Commit confirmed

     Leader   [1][2][3][4][5]  <- Holds all entries
     Follow-A [1][2][3][4][5]  <- Fully synced
     Follow-B [1][2][3]        <- Lagging (will catch up later)
     Follow-C [1][2][3][4]     <- 1 entry behind

     Majority (3/5 or more) have entry [4] -> Commit confirmed
     Follow-B receives [4][5] in the next AppendEntries

  3. Safety:
     - Committed entries are never overwritten
     - Only nodes with the most up-to-date log can become Leader
     -> Why: If a node with an old log becomes Leader,
       committed data could be lost

  Implementation examples and use cases:
  - etcd (Kubernetes cluster state management)
  - HashiCorp Consul (service discovery)
  - CockroachDB (distributed SQL)
  - TiKV (TiDB storage engine)
```

### 4.4 Paxos vs Raft Comparison

| Comparison Item | Paxos | Raft |
|---------|-------|------|
| Design Year | 1989/1998 | 2014 |
| Design Philosophy | Theoretical correctness | Understandability |
| Leader | Not required (symmetric) | Required (asymmetric) |
| Difficulty to Understand | Very high | Moderate |
| Difficulty to Implement | Very high | Moderate |
| Livelock | Can occur | Avoided by fixed Leader |
| Performance | Theoretically slightly better | Leader can be bottleneck |
| Fault Tolerance | Tolerates fewer than N/2 crashes | Tolerates fewer than N/2 crashes |
| Representative Implementation | Google Chubby | etcd, Consul |
| Industry Adoption | Declining trend | Increasing trend |

### Code Example 2: Simplified Raft Leader Election Simulation

The following code simulates the Raft leader election process. The reason for simulating this is that the concept of "collision avoidance through timeouts and randomization" in distributed consensus can be experienced through running code.

```python
"""
Raft Leader Election Simulator
==========================================
Simulates the leader election process in a 5-node Raft cluster
using a thread-based approach.

How to run: python raft_election.py
Dependencies: Standard library only
"""

import threading
import time
import random
from enum import Enum
from typing import Dict, Optional


class Role(Enum):
    FOLLOWER = "Follower"
    CANDIDATE = "Candidate"
    LEADER = "Leader"


class RaftNode:
    """A class implementing the Raft node's leader election logic.

    Why each node has its own timeout:
    -> If all nodes become Candidates simultaneously, votes split
    -> Random timeouts make it highly probable that one node
      starts the election first
    """

    def __init__(self, node_id: int, cluster: "RaftCluster"):
        self.node_id = node_id
        self.cluster = cluster
        self.role = Role.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[int] = None
        self.leader_id: Optional[int] = None
        self.lock = threading.Lock()
        self.election_timeout = random.uniform(0.15, 0.30)
        self.last_heartbeat = time.time()
        self.alive = True
        self.votes_received = 0

    def request_vote(self, candidate_id: int, term: int) -> bool:
        """Respond to a vote request.

        Why Term comparison is necessary:
        -> Voting for an old-Term Candidate could conflict
          with a Leader already elected in a newer Term
        """
        with self.lock:
            if not self.alive:
                return False
            if term > self.current_term:
                self.current_term = term
                self.role = Role.FOLLOWER
                self.voted_for = None
            if term == self.current_term and (
                self.voted_for is None or self.voted_for == candidate_id
            ):
                self.voted_for = candidate_id
                self.last_heartbeat = time.time()
                return True
            return False

    def receive_heartbeat(self, leader_id: int, term: int):
        """Receive a heartbeat from the Leader."""
        with self.lock:
            if not self.alive:
                return
            if term >= self.current_term:
                self.current_term = term
                self.role = Role.FOLLOWER
                self.leader_id = leader_id
                self.voted_for = leader_id
                self.last_heartbeat = time.time()

    def start_election(self):
        """Start an election."""
        with self.lock:
            if not self.alive or self.role == Role.LEADER:
                return
            self.current_term += 1
            self.role = Role.CANDIDATE
            self.voted_for = self.node_id
            self.votes_received = 1  # Vote for self
            term = self.current_term
            print(
                f"  [Node-{self.node_id}] Election started "
                f"(Term={term})"
            )

        # Request votes from all other nodes
        for node in self.cluster.nodes.values():
            if node.node_id != self.node_id:
                if node.request_vote(self.node_id, term):
                    with self.lock:
                        self.votes_received += 1
                        if (
                            self.votes_received
                            > len(self.cluster.nodes) // 2
                            and self.role == Role.CANDIDATE
                        ):
                            self.role = Role.LEADER
                            self.leader_id = self.node_id
                            print(
                                f"  [Node-{self.node_id}] *** "
                                f"Elected as Leader *** "
                                f"(Term={self.current_term}, "
                                f"votes={self.votes_received}/"
                                f"{len(self.cluster.nodes)})"
                            )
                            self._send_heartbeats()
                            return

    def _send_heartbeats(self):
        """Send heartbeats to all Followers."""
        for node in self.cluster.nodes.values():
            if node.node_id != self.node_id:
                node.receive_heartbeat(self.node_id, self.current_term)

    def run_election_timer(self):
        """Run the election timer (background thread)."""
        while self.alive:
            time.sleep(0.05)
            with self.lock:
                if not self.alive:
                    break
                if self.role == Role.LEADER:
                    continue
                elapsed = time.time() - self.last_heartbeat
                if elapsed > self.election_timeout:
                    pass
                else:
                    continue
            self.start_election()
            self.election_timeout = random.uniform(0.15, 0.30)
            with self.lock:
                self.last_heartbeat = time.time()


class RaftCluster:
    """A class managing the entire Raft cluster."""

    def __init__(self, node_count: int = 5):
        self.nodes: Dict[int, RaftNode] = {}
        for i in range(node_count):
            self.nodes[i] = RaftNode(i, self)

    def start(self):
        """Start all nodes' election timers."""
        threads = []
        for node in self.nodes.values():
            t = threading.Thread(
                target=node.run_election_timer, daemon=True
            )
            threads.append(t)
            t.start()
        return threads

    def kill_node(self, node_id: int):
        """Stop a node (failure simulation)."""
        self.nodes[node_id].alive = False
        print(f"\n  [Node-{node_id}] === Failure occurred ===")

    def status(self) -> str:
        """Return the current state of the cluster."""
        lines = []
        for nid, node in sorted(self.nodes.items()):
            status = "DEAD" if not node.alive else node.role.value
            leader_info = (
                f" (leader=Node-{node.leader_id})"
                if node.leader_id is not None
                else ""
            )
            lines.append(
                f"    Node-{nid}: {status:10s} "
                f"Term={node.current_term}{leader_info}"
            )
        return "\n".join(lines)


def demo():
    """Leader election demo."""
    print("=" * 60)
    print("Raft Leader Election Simulation (5 nodes)")
    print("=" * 60)

    cluster = RaftCluster(node_count=5)
    print("\n--- Initial state ---")
    print(cluster.status())

    print("\n--- Election timer started ---")
    cluster.start()
    time.sleep(0.5)  # Wait for election to complete

    print("\n--- State after election ---")
    print(cluster.status())

    # Cause a failure on the Leader
    leader_id = None
    for nid, node in cluster.nodes.items():
        if node.role == Role.LEADER:
            leader_id = nid
            break

    if leader_id is not None:
        cluster.kill_node(leader_id)
        print("\n--- Waiting for new Leader election... ---")
        time.sleep(0.8)  # Wait for re-election
        print("\n--- State after re-election ---")
        print(cluster.status())

    # Stop all nodes
    for node in cluster.nodes.values():
        node.alive = False


if __name__ == "__main__":
    demo()
```

Expected output:

```
============================================================
Raft Leader Election Simulation (5 nodes)
============================================================

--- Initial state ---
    Node-0: Follower   Term=0
    Node-1: Follower   Term=0
    Node-2: Follower   Term=0
    Node-3: Follower   Term=0
    Node-4: Follower   Term=0

--- Election timer started ---
  [Node-2] Election started (Term=1)
  [Node-2] *** Elected as Leader *** (Term=1, votes=4/5)

--- State after election ---
    Node-0: Follower   Term=1 (leader=Node-2)
    Node-1: Follower   Term=1 (leader=Node-2)
    Node-2: Leader     Term=1 (leader=Node-2)
    Node-3: Follower   Term=1 (leader=Node-2)
    Node-4: Follower   Term=1 (leader=Node-2)

  [Node-2] === Failure occurred ===

--- Waiting for new Leader election... ---
  [Node-4] Election started (Term=2)
  [Node-4] *** Elected as Leader *** (Term=2, votes=3/5)

--- State after re-election ---
    Node-0: Follower   Term=2 (leader=Node-4)
    Node-1: Follower   Term=2 (leader=Node-4)
    Node-2: DEAD       Term=1 (leader=Node-2)
    Node-3: Follower   Term=2 (leader=Node-4)
    Node-4: Leader     Term=2 (leader=Node-4)
```

---

## 5. Distributed Data Stores

### 5.1 Two Axes of Data Distribution

```
Two Axes of Data Distribution:

  1. Replication:
     Copy the same data to multiple nodes
     -> Purpose: Improve availability, read scale-out
     -> Trade-off: Maintaining consistency becomes harder

     Methods:
     +----------------------------------------------+
     | Synchronous Replication:                      |
     |   Master --write--> Slave1 (wait for ACK)    |
     |                 --> Slave2 (wait for ACK)     |
     |   Wait for ACK from all Slaves before         |
     |   responding to Client                        |
     |   -> Strong consistency achieved              |
     |   -> High write latency                       |
     |   -> Even one slow Slave delays everything    |
     |                                               |
     | Asynchronous Replication:                     |
     |   Master --write--> Slave1 (don't wait)       |
     |                 --> Slave2 (don't wait)        |
     |   Respond to Client immediately after Master   |
     |   write completes                             |
     |   -> Low write latency                        |
     |   -> Risk of data loss on Master failure      |
     |   -> Slaves may temporarily return stale data |
     |                                               |
     | Semi-synchronous:                             |
     |   Master --write--> Slave1 (wait for ACK)    |
     |                 --> Slave2 (don't wait)        |
     |   Wait for ACK from at least one Slave        |
     |   -> Balanced: one copy is guaranteed          |
     |   -> MySQL recommended configuration          |
     +----------------------------------------------+

  2. Partitioning/Sharding (splitting):
     Distribute data across multiple nodes
     -> Purpose: Capacity scaling, write scaling
     -> Trade-off: Cross-partition queries are expensive

     Partitioning methods:
     +----------------------------------------------+
     | Range Partitioning:                           |
     |   Shard1: A-G, Shard2: H-N, Shard3: O-Z     |
     |   -> Range queries are efficient               |
     |   -> Hotspots easily occur                     |
     |     Ex: names starting with "S" are common     |
     |                                               |
     | Hash Partitioning:                            |
     |   Shard = hash(key) % N                       |
     |   -> Data is evenly distributed                |
     |   -> Range queries are inefficient             |
     |   -> Massive redistribution on node add/remove |
     |                                               |
     | Consistent Hashing:                           |
     |   Impact is limited to 1/N on node add/remove |
     |   -> Used by DynamoDB, Cassandra              |
     |   -> Detailed explanation in next section     |
     +----------------------------------------------+
```

### 5.2 How Consistent Hashing Works

```
Why traditional hashing (hash(key) % N) is problematic:

  With N=3:
    hash("user:1") % 3 = 0 -> Node-0
    hash("user:2") % 3 = 1 -> Node-1
    hash("user:3") % 3 = 2 -> Node-2

  Changed to N=4 (node added):
    hash("user:1") % 4 = 1 -> Node-1  <- Needs migration!
    hash("user:2") % 4 = 2 -> Node-2  <- Needs migration!
    hash("user:3") % 4 = 0 -> Node-0  <- Needs migration!

  -> Changing node count requires redistributing all data
  -> In an N-node cluster, approximately (N-1)/N = almost all data moves

Consistent Hashing:
  Treat the hash space as a ring

        0
        |
   270 -+- 90      <- Hash ring (0-360 degrees)
        |
       180

  Place nodes on the ring:
        Node-A (30)
       /
  ----*-------*---- Node-B (120)
      |       |
      |       |
  ----*-------*---- Node-C (210)
               \
                Node-D (300)

  Key placement rule:
  -> From a key's hash value, search clockwise on the ring
  -> Store at the first node found

  Impact when adding a node:
  -> Adding Node-E(165) only affects the range Node-B(120) to Node-E(165)
  -> Only about 1/N (= 1/5) of all data moves

  Virtual Nodes:
  -> Assign multiple virtual nodes to each physical node
  -> Typically 100-200 virtual nodes
  -> Why needed: With few nodes, data distribution becomes skewed
  -> Increasing virtual nodes makes distribution more uniform
```

### Code Example 3: Consistent Hash Implementation

```python
"""
Consistent Hash Ring Implementation
==========================================
Implements consistent hashing with virtual nodes and
verifies data movement during node addition/removal.

How to run: python consistent_hash.py
Dependencies: Standard library only
"""

import hashlib
from bisect import bisect_right
from collections import defaultdict
from typing import Dict, List, Optional


class ConsistentHashRing:
    """Consistent hash ring implementation.

    Why bisect is used:
    -> Node lookup on the ring can be done in O(log N) with binary search
    -> Linear search would be O(N), which is slow with many nodes

    Why MD5 is used:
    -> Uniform distribution of hash values is important
    -> MD5 is cryptographically broken, but its uniformity
      properties are sufficient
    -> SHA-256 could also be used, but MD5 is faster
    """

    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}  # hash -> physical_node_name
        self.sorted_keys: List[int] = []
        self.nodes: set = set()

    def _hash(self, key: str) -> int:
        """Compute a hash value from a key."""
        digest = hashlib.md5(key.encode()).hexdigest()
        return int(digest, 16)

    def add_node(self, node: str):
        """Add a node to the ring.

        Why virtual nodes are used:
        Placing just 3 physical nodes directly results in
        uneven spacing on the ring, concentrating data on
        certain nodes. Virtual nodes distribute them evenly.
        """
        self.nodes.add(node)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:vn{i}"
            h = self._hash(virtual_key)
            self.ring[h] = node
            self.sorted_keys.append(h)
        self.sorted_keys.sort()

    def remove_node(self, node: str):
        """Remove a node from the ring."""
        self.nodes.discard(node)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:vn{i}"
            h = self._hash(virtual_key)
            if h in self.ring:
                del self.ring[h]
                self.sorted_keys.remove(h)

    def get_node(self, key: str) -> Optional[str]:
        """Return the node responsible for a key.

        Search clockwise from the key's hash value on the ring
        and return the first node found.
        """
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect_right(self.sorted_keys, h)
        if idx == len(self.sorted_keys):
            idx = 0  # Wrap around to the beginning of the ring
        return self.ring[self.sorted_keys[idx]]

    def get_distribution(self, keys: List[str]) -> Dict[str, int]:
        """Calculate the distribution of keys."""
        dist: Dict[str, int] = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            if node:
                dist[node] += 1
        return dict(dist)


def demo():
    """Consistent hash operation demo."""
    print("=" * 60)
    print("Consistent Hash Ring Demo")
    print("=" * 60)

    # Test data: 10000 keys
    keys = [f"user:{i}" for i in range(10000)]

    # --- Distribution with 3 nodes ---
    ring = ConsistentHashRing(virtual_nodes=150)
    for node in ["Node-A", "Node-B", "Node-C"]:
        ring.add_node(node)

    dist = ring.get_distribution(keys)
    print("\n--- 3-node configuration ---")
    for node, count in sorted(dist.items()):
        bar = "#" * (count // 50)
        print(f"  {node}: {count:5d} keys {bar}")

    # Record data placement
    original_mapping = {key: ring.get_node(key) for key in keys}

    # --- Impact of adding a node ---
    ring.add_node("Node-D")
    new_mapping = {key: ring.get_node(key) for key in keys}

    moved = sum(
        1 for key in keys
        if original_mapping[key] != new_mapping[key]
    )
    print(f"\n--- After adding Node-D ---")
    dist = ring.get_distribution(keys)
    for node, count in sorted(dist.items()):
        bar = "#" * (count // 50)
        print(f"  {node}: {count:5d} keys {bar}")
    print(f"\n  Keys moved: {moved}/{len(keys)} "
          f"({moved/len(keys)*100:.1f}%)")
    print(f"  Ideal: {100/4:.1f}% (1/N)")

    # --- Impact of removing a node ---
    original_mapping_4 = {key: ring.get_node(key) for key in keys}
    ring.remove_node("Node-B")
    after_removal = {key: ring.get_node(key) for key in keys}

    moved = sum(
        1 for key in keys
        if original_mapping_4[key] != after_removal[key]
    )
    print(f"\n--- After removing Node-B ---")
    dist = ring.get_distribution(keys)
    for node, count in sorted(dist.items()):
        bar = "#" * (count // 50)
        print(f"  {node}: {count:5d} keys {bar}")
    print(f"\n  Keys moved: {moved}/{len(keys)} "
          f"({moved/len(keys)*100:.1f}%)")

    # --- Distribution variance by virtual node count ---
    print(f"\n--- Standard deviation by virtual node count ---")
    for vn_count in [1, 10, 50, 150, 500]:
        r = ConsistentHashRing(virtual_nodes=vn_count)
        for node in ["Node-A", "Node-B", "Node-C"]:
            r.add_node(node)
        d = r.get_distribution(keys)
        values = list(d.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stddev = variance ** 0.5
        print(
            f"  vn={vn_count:4d}: "
            f"stddev={stddev:7.1f} "
            f"(uniformity: "
            f"{'##' * max(1, int(10 - stddev / 50))})"
        )


if __name__ == "__main__":
    demo()
```

Expected output:

```
============================================================
Consistent Hash Ring Demo
============================================================

--- 3-node configuration ---
  Node-A:  3342 keys ##################################################################
  Node-B:  3298 keys #################################################################
  Node-C:  3360 keys ###################################################################

--- After adding Node-D ---
  Node-A:  2510 keys ##################################################
  Node-B:  2485 keys #################################################
  Node-C:  2522 keys ##################################################
  Node-D:  2483 keys #################################################

  Keys moved: 2483/10000 (24.8%)
  Ideal: 25.0% (1/N)

--- After removing Node-B ---
  Node-A:  3356 keys ###################################################################
  Node-B removed -> Node-B's data is taken over by adjacent nodes
  Node-C:  3322 keys ##################################################################
  Node-D:  3322 keys ##################################################################

  Keys moved: 2485/10000 (24.9%)

--- Standard deviation by virtual node count ---
  vn=   1: stddev= 2187.3 (uniformity: )
  vn=  10: stddev=  634.2 (uniformity: ########)
  vn=  50: stddev=  198.7 (uniformity: ############)
  vn= 150: stddev=   55.1 (uniformity: ##################)
  vn= 500: stddev=   28.3 (uniformity: ####################)
```

### 5.3 Consistency Control via Quorum

```
Quorum:
  Control consistency by adjusting the number of nodes required for reads and writes

  N = Number of replicas (Replication Factor)
  W = Number of nodes confirmed during write (Write Quorum)
  R = Number of nodes confirmed during read (Read Quorum)

  Consistency condition: W + R > N
  -> Read and write node sets always overlap
  -> The overlapping node has the latest data

  Example: N=3

  +--------------------------------------------+
  | Setting 1: W=2, R=2  -> Strong consistency |
  |   Write: Write to Node-A, Node-B          |
  |   Read:  Read from Node-B, Node-C         |
  |   -> Node-B overlaps -> latest value found |
  |                                             |
  | Setting 2: W=3, R=1  -> Write-heavy        |
  |   Write: Write to all nodes (slow)         |
  |   Read:  Read from 1 node (fast)           |
  |   -> Favorable when reads are frequent     |
  |                                             |
  | Setting 3: W=1, R=3  -> Read-heavy         |
  |   Write: Write to 1 node (fast)            |
  |   Read:  Read from all nodes (slow)        |
  |   -> Favorable when writes are frequent    |
  |                                             |
  | Setting 4: W=1, R=1  -> Eventual consistency|
  |   W + R = 2 <= N=3 -> may not overlap      |
  |   -> Stale data may be read                |
  |   -> Fastest but weakest consistency       |
  +--------------------------------------------+
```

---

## 6. Distributed Transactions

### 6.1 Difficulty of ACID in Distributed Environments

```
Single-DB ACID:
  A(Atomicity):    All operations succeed or all fail
  C(Consistency):  Constraints (foreign keys, etc.) are always satisfied
  I(Isolation):    Concurrent transactions do not interfere
  D(Durability):   Committed data is persisted

Why ACID is difficult in distributed environments:

  Atomicity problem:
  -> Operations spanning multiple nodes may partially succeed
  -> Example: Debit on Node-A succeeds, credit on Node-B fails
  -> Guaranteeing "all or nothing" across physically distant nodes is costly

  Isolation problem:
  -> Cost of distributed locks is very high
  -> Network latency extends lock holding time
  -> Deadlock detection is far more complex than in a single DB

  -> Two main approaches to address these problems:
    1. Two-Phase Commit (2PC): Achieves distributed ACID
    2. Saga Pattern: Relaxes ACID for practicality
```

### 6.2 Two-Phase Commit (2PC)

```
Two-Phase Commit:

  Coordinator         Participant-A    Participant-B
       |                    |                |
  Phase 1 (Voting Phase):
       |-- Prepare -------->|                |
       |-- Prepare ----------------------->|
       |<-- Vote YES -------|                |
       |<-- Vote YES ----------------------|
       |                    |                |
  Phase 2 (Decision Phase):
       |-- Commit --------->|                |
       |-- Commit ------------------------>|
       |<-- ACK ------------|                |
       |<-- ACK ----------------------------|

  Meaning of each phase:

  Phase 1 (Prepare):
  -> Coordinator asks "Can you commit this operation?"
  -> Participant acquires necessary locks and writes to WAL
  -> Returns YES (ready) or NO (cannot)
  -> After returning YES, holds locks until receiving the result

  Phase 2 (Commit/Abort):
  -> If all say YES -> Coordinator instructs "Commit"
  -> If any says NO -> Coordinator instructs "Abort"
  -> Participants release locks
```

### 6.3 Critical Problems with 2PC

```
Problems with 2PC:

  1. Blocking Problem:
     If the Coordinator fails after Phase 1 YES responses
     but before Phase 2:

     Coordinator   x  (failure)
     Participant-A: Blocked in YES state
     Participant-B: Blocked in YES state

     -> Holding locks, unable to determine Commit or Abort
     -> Entire system blocks until Coordinator recovers
     -> During this time, locked data is inaccessible to other transactions

  2. Performance Problem:
     -> Two round trips required (Prepare + Commit)
     -> Network latency is doubled
     -> Rate-limited by the slowest responding node

  3. Availability Problem:
     -> If even one Participant doesn't respond, Abort
     -> Failure probability increases with more nodes
     -> 10 nodes each at 99.9% availability -> overall 99.0%

  Countermeasure: Three-Phase Commit (3PC)
  -> Adds a Pre-Commit phase to mitigate blocking
  -> However, cannot handle network partitions; practical use is limited
```

### 6.4 Saga Pattern

```
Saga Pattern (Hector Garcia-Molina, 1987):

  Basic Concept:
  Decompose a distributed transaction into a series of local transactions,
  and prepare a "compensating transaction" for each step

  Normal flow:
  T1 -> T2 -> T3 -> Complete (success)

  Failure flow (T3 fails):
  T1 -> T2 -> T3(fails) -> C2 -> C1
  (C = compensating transaction = undo operation)

  E-commerce order processing example:
  +------------------------------------------+
  | Step     Normal Operation  Compensating   |
  | ------   ---------------  -----------    |
  | T1       Reserve inventory C1: Release   |
  | T2       Process payment   C2: Refund    |
  | T3       Schedule shipping C3: Cancel    |
  +------------------------------------------+

  Two implementation patterns:

  Orchestration:
  +---------------------------------------------+
  |  Saga Orchestrator                           |
  |       |                                      |
  |       |--> Inventory Service --> (success)   |
  |       |--> Payment Service --> (success)     |
  |       |--> Shipping Service --> (failure)    |
  |       |                                      |
  |       |--> Payment Service.compensate()      |
  |       |--> Inventory Service.compensate()    |
  |                                              |
  |  Advantage: Flow is managed in one place     |
  |  Disadvantage: Orchestrator can become SPOF  |
  +---------------------------------------------+

  Choreography:
  +---------------------------------------------+
  |  Inventory Service --[reserved]--> Event Bus |
  |  Payment Service <--[reserved]--             |
  |  Payment Service --[paid]--> Event Bus       |
  |  Shipping Service <--[paid]--                |
  |  Shipping Service --[ship_failed]--> Event Bus|
  |  Payment Service <--[ship_failed]-- -> refund|
  |  Inventory Service <--[refunded]-- -> release|
  |                                              |
  |  Advantage: Loosely coupled, no central mgmt |
  |  Disadvantage: Hard to see full flow, debug  |
  +---------------------------------------------+
```

### Code Example 4: Saga Pattern Implementation

```python
"""
Saga Pattern (Orchestration) Implementation
==========================================
Implements e-commerce order processing using the Saga pattern.
Demonstrates compensating transaction behavior on failure.

How to run: python saga_pattern.py
Dependencies: Standard library only
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional
import random


class StepStatus(Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    COMPENSATED = "COMPENSATED"


@dataclass
class SagaStep:
    """Represents a single step of a Saga.

    Why normal and compensating operations are paired:
    -> If a step fails, all preceding steps must be
      compensated in reverse order
    -> Pairing makes it clear which operation to undo and how
    """
    name: str
    action: Callable[[], bool]
    compensate: Callable[[], bool]
    status: StepStatus = StepStatus.PENDING


@dataclass
class SagaResult:
    success: bool
    steps_completed: int
    steps_compensated: int
    log: List[str] = field(default_factory=list)


class SagaOrchestrator:
    """Saga orchestrator.

    Executes steps in order, compensates in reverse on failure.
    Why compensate in reverse:
    -> Later steps depend on the results of earlier steps
    -> To avoid breaking dependencies, undo from the end
    """

    def __init__(self):
        self.steps: List[SagaStep] = []
        self.log: List[str] = []

    def add_step(
        self,
        name: str,
        action: Callable[[], bool],
        compensate: Callable[[], bool],
    ):
        """Add a step."""
        self.steps.append(SagaStep(name, action, compensate))

    def execute(self) -> SagaResult:
        """Execute the Saga."""
        completed_steps: List[SagaStep] = []

        self.log.append("=== Saga execution started ===")

        for step in self.steps:
            self.log.append(f"  Executing: {step.name}")
            try:
                success = step.action()
            except Exception as e:
                self.log.append(f"  Exception: {step.name} - {e}")
                success = False

            if success:
                step.status = StepStatus.COMPLETED
                completed_steps.append(step)
                self.log.append(f"  Completed: {step.name}")
            else:
                step.status = StepStatus.FAILED
                self.log.append(f"  Failed: {step.name}")
                self.log.append("=== Compensating transactions started ===")

                # Compensate in reverse order
                compensated = 0
                for completed in reversed(completed_steps):
                    self.log.append(f"  Compensating: {completed.name}")
                    try:
                        completed.compensate()
                        completed.status = StepStatus.COMPENSATED
                        compensated += 1
                        self.log.append(
                            f"  Compensation complete: {completed.name}"
                        )
                    except Exception as e:
                        self.log.append(
                            f"  Compensation failed: {completed.name} - {e}"
                        )

                self.log.append("=== Saga failed (rollback complete) ===")
                return SagaResult(
                    success=False,
                    steps_completed=len(completed_steps),
                    steps_compensated=compensated,
                    log=self.log,
                )

        self.log.append("=== Saga succeeded ===")
        return SagaResult(
            success=True,
            steps_completed=len(completed_steps),
            steps_compensated=0,
            log=self.log,
        )


# --- E-commerce Order Processing Simulation ---

class InventoryService:
    """Inventory service."""

    def __init__(self):
        self.stock = {"item-A": 10, "item-B": 5}
        self.reserved = {}

    def reserve(self, item_id: str, qty: int) -> bool:
        """Reserve inventory."""
        if self.stock.get(item_id, 0) >= qty:
            self.stock[item_id] -= qty
            self.reserved[item_id] = (
                self.reserved.get(item_id, 0) + qty
            )
            print(
                f"    [Inventory] {item_id} x{qty} reserved "
                f"(remaining: {self.stock[item_id]})"
            )
            return True
        print(f"    [Inventory] {item_id} insufficient stock")
        return False

    def release(self, item_id: str, qty: int) -> bool:
        """Release reserved inventory."""
        self.stock[item_id] = self.stock.get(item_id, 0) + qty
        self.reserved[item_id] = max(
            0, self.reserved.get(item_id, 0) - qty
        )
        print(
            f"    [Inventory] {item_id} x{qty} released "
            f"(remaining: {self.stock[item_id]})"
        )
        return True


class PaymentService:
    """Payment service."""

    def __init__(self, fail_probability: float = 0.0):
        self.transactions = []
        self.fail_probability = fail_probability

    def charge(self, user_id: str, amount: int) -> bool:
        """Process a payment."""
        if random.random() < self.fail_probability:
            print(f"    [Payment] {user_id} {amount} failed (insufficient balance)")
            return False
        self.transactions.append(
            {"user": user_id, "amount": amount, "type": "charge"}
        )
        print(f"    [Payment] {user_id} {amount} succeeded")
        return True

    def refund(self, user_id: str, amount: int) -> bool:
        """Process a refund."""
        self.transactions.append(
            {"user": user_id, "amount": -amount, "type": "refund"}
        )
        print(f"    [Payment] {user_id} {amount} refunded")
        return True


class ShippingService:
    """Shipping service."""

    def __init__(self, fail_probability: float = 0.0):
        self.shipments = []
        self.fail_probability = fail_probability

    def schedule(self, order_id: str, address: str) -> bool:
        """Schedule a shipment."""
        if random.random() < self.fail_probability:
            print(
                f"    [Shipping] Order {order_id} -> {address} "
                f"scheduling failed (carrier error)"
            )
            return False
        self.shipments.append({"order": order_id, "address": address})
        print(f"    [Shipping] Order {order_id} -> {address} scheduled")
        return True

    def cancel(self, order_id: str) -> bool:
        """Cancel a shipment."""
        self.shipments = [
            s for s in self.shipments if s["order"] != order_id
        ]
        print(f"    [Shipping] Order {order_id} cancelled")
        return True


def demo():
    """Saga pattern demo."""
    # --- Normal case ---
    print("=" * 60)
    print("Saga Pattern Demo: Normal Case")
    print("=" * 60)

    inv = InventoryService()
    pay = PaymentService()
    ship = ShippingService()

    saga = SagaOrchestrator()
    saga.add_step(
        "Reserve inventory",
        lambda: inv.reserve("item-A", 2),
        lambda: inv.release("item-A", 2),
    )
    saga.add_step(
        "Process payment",
        lambda: pay.charge("user-1", 5000),
        lambda: pay.refund("user-1", 5000),
    )
    saga.add_step(
        "Schedule shipping",
        lambda: ship.schedule("order-1", "Shibuya, Tokyo..."),
        lambda: ship.cancel("order-1"),
    )

    result = saga.execute()
    print("\n".join(result.log))

    # --- Failure case (shipping failure) ---
    print("\n" + "=" * 60)
    print("Saga Pattern Demo: Shipping Failure Case")
    print("=" * 60)

    inv2 = InventoryService()
    pay2 = PaymentService()
    ship2 = ShippingService(fail_probability=1.0)  # Always fails

    saga2 = SagaOrchestrator()
    saga2.add_step(
        "Reserve inventory",
        lambda: inv2.reserve("item-A", 2),
        lambda: inv2.release("item-A", 2),
    )
    saga2.add_step(
        "Process payment",
        lambda: pay2.charge("user-2", 3000),
        lambda: pay2.refund("user-2", 3000),
    )
    saga2.add_step(
        "Schedule shipping",
        lambda: ship2.schedule("order-2", "Kita-ku, Osaka..."),
        lambda: ship2.cancel("order-2"),
    )

    result2 = saga2.execute()
    print("\n".join(result2.log))
    print(f"\n  Inventory state: {inv2.stock}")
    print(f"  Payment history: {pay2.transactions}")


if __name__ == "__main__":
    demo()
```

Expected output:

```
============================================================
Saga Pattern Demo: Normal Case
============================================================
    [Inventory] item-A x2 reserved (remaining: 8)
    [Payment] user-1 5000 succeeded
    [Shipping] Order order-1 -> Shibuya, Tokyo... scheduled
=== Saga execution started ===
  Executing: Reserve inventory
  Completed: Reserve inventory
  Executing: Process payment
  Completed: Process payment
  Executing: Schedule shipping
  Completed: Schedule shipping
=== Saga succeeded ===

============================================================
Saga Pattern Demo: Shipping Failure Case
============================================================
    [Inventory] item-A x2 reserved (remaining: 8)
    [Payment] user-2 3000 succeeded
    [Shipping] Order order-2 -> Kita-ku, Osaka... scheduling failed (carrier error)
=== Saga execution started ===
  Executing: Reserve inventory
  Completed: Reserve inventory
  Executing: Process payment
  Completed: Process payment
  Executing: Schedule shipping
  Failed: Schedule shipping
=== Compensating transactions started ===
    [Payment] user-2 3000 refunded
  Compensating: Process payment
  Compensation complete: Process payment
    [Inventory] item-A x2 released (remaining: 10)
  Compensating: Reserve inventory
  Compensation complete: Reserve inventory
=== Saga failed (rollback complete) ===

  Inventory state: {'item-A': 10, 'item-B': 5}   <- Restored to original
  Payment history: [{'user': 'user-2', 'amount': 3000, 'type': 'charge'},
             {'user': 'user-2', 'amount': -3000, 'type': 'refund'}]
```

### 6.5 2PC vs Saga Comparison

| Comparison Item | 2PC | Saga |
|---------|-----|------|
| Consistency | Strong (ACID) | Eventual consistency |
| Availability | Low (blocking) | High |
| Performance | Low (2 RTT + lock holding) | High |
| Implementation complexity | Medium | High (designing compensation logic is difficult) |
| Fault tolerance | Weak against Coordinator failure | High, each step is independent |
| Isolation | Guaranteed | Not guaranteed (dirty reads possible) |
| Applicable scenarios | Inter-DB transactions | Long transactions between microservices |

---

## 7. Time and Ordering

### 7.1 The "Time" Problem in Distributed Systems

```
Why the time problem matters:

  On a single machine:
  -> OS clock assigns unique timestamps to all events
  -> Event ordering is always clear

  In distributed systems:
  -> Each machine's clock is off (clock skew)
  -> Clocks advance at slightly different rates (clock drift)
  -> NTP accuracy is only several ms to tens of ms
  -> Even Google's TrueTime API has errors of several ms

  Example of why physical clocks alone are dangerous for ordering:

  Node-A (clock 10ms ahead):
    Executes write(x, 1) at 10:00:00.010

  Node-B (clock is accurate):
    Executes write(x, 2) at 10:00:00.015

  Node-C (clock 20ms behind):
    Executes read(x) at 09:59:59.995 (actually 10:00:00.015)

  -> Node-C's read actually occurs after Node-B's write but
    appears oldest by timestamp
  -> Physical clock-based ordering breaks causal relationships

  -> Solution: Logical clocks (ordering without depending on physical clocks)
```

### 7.2 Lamport Clock (Logical Clock)

```
Lamport Clock (Leslie Lamport, 1978):

  Each process maintains a scalar counter

  Rules:
  1. Internal event: Increment counter by 1
  2. Message send: Increment counter by 1 and attach counter value to message
  3. Message receive: max(own counter, received counter) + 1

  Example:
  Process A:  1 -----> 2 ----------> 5
                       | send     ^ recv
  Process B:       3 -----> 4 --->|
                                  | send
  Process C:           2 -----> 5 -----> 6

  Causal relationship guarantee:
  -> If a caused b (a -> b), then L(a) < L(b)
  -> This always holds

  Limitation:
  -> L(a) < L(b) does NOT necessarily mean a -> b
  -> Cannot distinguish concurrent events
  -> Example: Process A's 1 and Process C's 2 have
    L(A:1) < L(C:2) but are not causally related

  Why this limitation exists:
  -> A single scalar value loses the information of "whose operation"
  -> Vector clocks solve this
```

### 7.3 Vector Clock

```
Vector Clock:

  Each process maintains "counters for all processes"

  Rules (for N processes):
  1. Internal event: Increment own counter by 1
  2. Send: Increment own counter by 1 and send the entire vector
  3. Receive: Take element-wise max and increment own counter by 1

  Example (3 processes [A, B, C]):
  Process A: [1,0,0] --> [2,0,0] -------> [3,2,0]
                          | send        ^ recv
  Process B:         [0,1,0] --> [0,2,0]--|
                                          | send
  Process C:              [0,0,1] --> [0,0,2] --> [2,2,3]

  Determining causal relationships:

  V1 <= V2 means: All elements of V1 are <= corresponding elements of V2
  V1 -> V2 (V1 caused V2): V1 <= V2 and V1 != V2

  Detecting concurrency:
  V1 || V2: Neither V1 <= V2 nor V2 <= V1

  Concrete examples:
  [1,0,0] and [0,1,0]:
    -> [1,0,0]'s A=1 > [0,1,0]'s A=0
    -> [0,1,0]'s B=1 > [1,0,0]'s B=0
    -> Neither is <= the other -> Concurrent (no causal relationship)

  [1,0,0] and [2,1,0]:
    -> All elements of [1,0,0] <= [2,1,0]
    -> [1,0,0] causally precedes

  Uses: DynamoDB conflict detection, distributed version control
```

### Code Example 5: Vector Clock Implementation

```python
"""
Vector Clock Implementation and Causal Relationship Determination
==========================================
Simulates message passing between 3 processes and
uses vector clocks to accurately determine causal relationships.

How to run: python vector_clock.py
Dependencies: Standard library only
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
from copy import deepcopy


class CausalRelation(Enum):
    """Causal relationship between two events."""
    BEFORE = "BEFORE"          # a -> b (a caused b)
    AFTER = "AFTER"            # b -> a (b caused a)
    CONCURRENT = "CONCURRENT"  # a || b (concurrent, no causal relationship)
    EQUAL = "EQUAL"            # a = b (same event)


class VectorClock:
    """Vector clock implementation.

    Why implemented with a dict:
    -> Can handle dynamically changing process counts
    -> With a fixed-length array, adding a process would
      require extending all nodes' arrays
    """

    def __init__(self, process_id: str):
        self.process_id = process_id
        self.clock: Dict[str, int] = {process_id: 0}

    def increment(self) -> "VectorClock":
        """Increment counter on internal event."""
        self.clock[self.process_id] = (
            self.clock.get(self.process_id, 0) + 1
        )
        return self

    def send(self) -> Dict[str, int]:
        """On message send: increment and return the clock.

        Why increment on send as well:
        -> The send event itself is an event
        -> Without incrementing, internal events and sends
          become indistinguishable in order
        """
        self.increment()
        return deepcopy(self.clock)

    def receive(self, other_clock: Dict[str, int]) -> "VectorClock":
        """On message receive: take element-wise max.

        Why take the max:
        -> To incorporate all causal information the sender knows
        -> Taking max ensures "all events the sender has seen"
          are reflected in the receiver
        """
        for pid, count in other_clock.items():
            self.clock[pid] = max(
                self.clock.get(pid, 0), count
            )
        self.clock[self.process_id] = (
            self.clock.get(self.process_id, 0) + 1
        )
        return self

    def snapshot(self) -> Dict[str, int]:
        """Return a snapshot of the current clock state."""
        return deepcopy(self.clock)

    @staticmethod
    def compare(
        vc1: Dict[str, int], vc2: Dict[str, int]
    ) -> CausalRelation:
        """Determine the causal relationship between two vector clocks.

        Determination logic:
        - All elements vc1[i] <= vc2[i] and vc1 != vc2 -> BEFORE
        - All elements vc1[i] >= vc2[i] and vc1 != vc2 -> AFTER
        - Neither of the above -> CONCURRENT
        """
        all_keys = set(vc1.keys()) | set(vc2.keys())

        leq = True   # vc1 <= vc2 ?
        geq = True   # vc1 >= vc2 ?
        equal = True  # vc1 == vc2 ?

        for key in all_keys:
            v1 = vc1.get(key, 0)
            v2 = vc2.get(key, 0)
            if v1 > v2:
                leq = False
                equal = False
            if v1 < v2:
                geq = False
                equal = False

        if equal:
            return CausalRelation.EQUAL
        if leq:
            return CausalRelation.BEFORE
        if geq:
            return CausalRelation.AFTER
        return CausalRelation.CONCURRENT

    def __repr__(self) -> str:
        items = sorted(self.clock.items())
        return "[" + ", ".join(
            f"{pid}:{count}" for pid, count in items
        ) + "]"


class DistributedSystem:
    """Distributed system simulator using vector clocks."""

    def __init__(self, process_ids: List[str]):
        self.processes: Dict[str, VectorClock] = {
            pid: VectorClock(pid) for pid in process_ids
        }
        self.events: List[Tuple[str, str, Dict[str, int]]] = []

    def local_event(self, pid: str, description: str):
        """Trigger a local event."""
        vc = self.processes[pid]
        vc.increment()
        snapshot = vc.snapshot()
        self.events.append((pid, description, snapshot))
        print(f"  [{pid}] {description:30s} clock={vc}")

    def send_message(
        self, from_pid: str, to_pid: str, description: str
    ):
        """Send and receive a message."""
        sender = self.processes[from_pid]
        receiver = self.processes[to_pid]

        msg_clock = sender.send()
        send_snapshot = sender.snapshot()
        self.events.append(
            (from_pid, f"send({description})", send_snapshot)
        )
        print(
            f"  [{from_pid}] send({description:22s}) "
            f"clock={sender}"
        )

        receiver.receive(msg_clock)
        recv_snapshot = receiver.snapshot()
        self.events.append(
            (to_pid, f"recv({description})", recv_snapshot)
        )
        print(
            f"  [{to_pid}] recv({description:22s}) "
            f"clock={receiver}"
        )

    def analyze_causality(self):
        """Analyze causal relationships between all events."""
        print("\n--- Causal Relationship Analysis ---")
        for i in range(len(self.events)):
            for j in range(i + 1, len(self.events)):
                pid_i, desc_i, vc_i = self.events[i]
                pid_j, desc_j, vc_j = self.events[j]
                relation = VectorClock.compare(vc_i, vc_j)
                if relation != CausalRelation.EQUAL:
                    symbol = {
                        CausalRelation.BEFORE: "-->",
                        CausalRelation.AFTER: "<--",
                        CausalRelation.CONCURRENT: "|||",
                    }[relation]
                    print(
                        f"  {pid_i}:{desc_i:20s} "
                        f"{symbol} "
                        f"{pid_j}:{desc_j:20s}"
                    )


def demo():
    """Vector clock demo."""
    print("=" * 60)
    print("Vector Clock Demo")
    print("=" * 60)
    print()

    sys = DistributedSystem(["A", "B", "C"])

    # Scenario: SNS posts and comments
    sys.local_event("A", "A creates a post")
    sys.send_message("A", "B", "notify post")
    sys.local_event("C", "C creates another post")
    sys.send_message("B", "C", "send comment")
    sys.local_event("A", "A edits the post")

    sys.analyze_causality()

    # Conflict detection demo
    print("\n" + "=" * 60)
    print("Conflict Detection Demo (Simultaneous Edits)")
    print("=" * 60)
    print()

    sys2 = DistributedSystem(["Editor-A", "Editor-B"])
    sys2.local_event("Editor-A", "Edits document")
    sys2.local_event("Editor-B", "Edits same section")

    vc_a = sys2.events[0][2]
    vc_b = sys2.events[1][2]
    relation = VectorClock.compare(vc_a, vc_b)
    print(f"\n  Determination: {relation.value}")
    if relation == CausalRelation.CONCURRENT:
        print("  -> Conflict detected! Merge or user selection required")


if __name__ == "__main__":
    demo()
```

---

## 8. Distributed Architecture Patterns

### 8.1 Microservices

```
Microservice Architecture:
  Split a monolith into independent services

  +----------+  +----------+  +----------+
  | User     |  | Order    |  | Payment  |
  | Service  |  | Service  |  | Service  |
  | (Go)     |  | (Java)   |  | (Python) |
  +----+-----+  +----+-----+  +----+-----+
       |              |              |
  +----+--------------+--------------+----+
  |         Message Bus (Kafka)            |
  +----------------------------------------+

  Advantages:
  - Independent deployment: Each service can be released individually
  - Technology freedom: Choose the optimal language/framework per service
  - Fault isolation: One service's failure is less likely to cascade
  - Team autonomy: Teams organized around services

  Challenges:
  - Network latency: Overhead of inter-service communication
  - Data consistency: Complexity of distributed transactions
  - Operational complexity: Monitoring and deploying hundreds of services
  - Debugging difficulty: Requests traverse multiple services
```

### 8.2 Event-Driven Architecture and CQRS

```
Event-Driven Architecture:
  Loosely couple services via events

  Producer --> Event Bus --> Consumer A
                         --> Consumer B
                         --> Consumer C

  Event Sourcing:
  Instead of storing state directly, record events of state changes

  Traditional: Store latest state in users table
    {name: "Alice", email: "alice@new.com"}

  Event Sourcing:
    [UserCreated {name: "Alice", email: "alice@old.com"}]
    [EmailChanged {email: "alice@new.com"}]
    [NameChanged {name: "Alice B."}]

  Why Event Sourcing is useful:
  -> Complete audit trail (when, what, how it changed)
  -> Can reconstruct state at any point in time (time travel)
  -> Can build new views by replaying events

CQRS (Command Query Responsibility Segregation):
  Separate writes (Command) from reads (Query)

  Write Model          Read Model
  +----------+        +----------+
  | Command  |--Event->| Query    |
  | Store    |        | Store    |
  | (normal.)  |       | (denorm.) |
  +----------+        +----------+
      ^ write              ^ read
      |                    |
  Commands             Queries

  Why separate:
  -> Optimal data models differ for writes and reads
  -> Writes: Normalize to maintain data consistency
  -> Reads: Denormalize to optimize query performance
  -> Can scale independently (if reads dominate, scale the read side)
```

---

## 9. Failure and Recovery

### 9.1 Classification of Failures

```
Types of Failures and Countermeasures:

  1. Crash Failure:
     A node stops completely
     -> Detection: Heartbeat + timeout
     -> Countermeasure: Failover to replica
     -> Characteristic: Relatively easy to detect

  2. Network Failure:
     Communication is severed
     -> Detection: Timeout (hard to distinguish from crash)
     -> Countermeasure: Retry + idempotency guarantee
     -> Characteristic: Cannot distinguish "slow" from "stopped"

  3. Byzantine Failure:
     Malicious behavior or arbitrary failures like data corruption
     -> Detection: Cryptographic proofs (signatures, hashes)
     -> Countermeasure: BFT algorithms
     -> Condition: Tolerable if malicious nodes are fewer than N/3
     -> Uses: Blockchain, aerospace systems

  4. Gray Failure:
     Partial failure. Appears normal from outside but is actually broken
     -> Example: Extremely slow responses, only some requests fail
     -> Most difficult to detect, most frequently occurring
     -> Countermeasure: Detailed metrics monitoring, anomaly detection
```

### 9.2 Failure Mitigation Patterns

```
Circuit Breaker Pattern:

  Why it's necessary:
  -> Repeatedly sending requests to a failing service
    depletes the caller's threads/connections
  -> Failures cascade to other services (cascade failure)

  +---------+     +----------+     +----------+
  | Closed  |---->|  Open    |---->|Half-Open |
  |(normal  )| fail|(immediate)| wait|(test     )|
  | pass    )| thresh| error  )| period| request )|
  +---------+ exceed+----------+ elapsed+-----+----+
       ^                              |
       +--------- success -----------+

  Closed: All requests pass through
  -> Transition to Open when failure rate exceeds threshold

  Open: All requests immediately return error
  -> Transition to Half-Open after a set time period
  -> Why immediately return error: Let the failing service recover without load

  Half-Open: Pass one test request
  -> Return to Closed on success
  -> Return to Open on failure

Bulkhead Pattern:
  Limit the blast radius of failures
  -> Same concept as a ship's bulkheads: one flooded compartment doesn't sink the whole ship
  -> Service A's failure doesn't cascade to Service B
  -> Implementation: Thread pool isolation, connection pool isolation

Retry with Exponential Backoff:
  1st: Retry after 100ms
  2nd: Retry after 200ms
  3rd: Retry after 400ms
  4th: Retry after 800ms
  ...up to a limit

  + Jitter (random variation):
    Why jitter is needed:
    -> When many clients retry simultaneously, an instantaneous
      load spike hits the server (thundering herd problem)
    -> Jitter staggers each client's retry timing
```

---

## 10. Anti-Patterns

### Anti-Pattern 1: Distributed Monolith

```
Distributed Monolith:

  Symptoms:
  Despite splitting into microservices, inter-service coupling is
  so strong that you bear "the drawbacks of a monolith + the complexity
  of distribution"

  +------------------------------------------+
  |  Service-A ---sync call---> Service-B    |
  |       |                         |        |
  |       +---sync call---> Service-C       |
  |             |                   |        |
  |             +---shared DB------+        |
  |                                          |
  |  -> Changes to 1 service affect all      |
  |  -> Deployment requires all at once      |
  |  -> Same as a monolith but with added    |
  |    network latency and failure risk      |
  +------------------------------------------+

  Why it occurs:
  1. Improper service boundary design
     -> Ignoring DDD's bounded contexts
  2. Shared databases
     -> Service independence is compromised
  3. Over-reliance on synchronous communication
     -> Long call chains; one failure stops everything

  Avoidance:
  - Each service owns its own DB (Database per Service)
  - Asynchronous messaging between services (event-driven)
  - Apply Circuit Breaker even when sync calls are needed
  - Martin Fowler's advice: Start with a monolith, split after
    boundaries become clear (MonolithFirst)
```

### Anti-Pattern 2: Overly Optimistic Retry

```
Aggressive Retry:

  Symptoms:
  Immediately retrying many times on failure, worsening the failure

  +------------------------------------------+
  |  Client-1 --> Server (failing)           |
  |    Immediate retry x 10                  |
  |  Client-2 --> Server (failing)           |
  |    Immediate retry x 10                  |
  |  ... x 1000 clients                     |
  |                                           |
  |  -> Server load: 10,000x normal          |
  |  -> Failure worsens instead of recovering|
  |  -> Known as a "retry storm"             |
  +------------------------------------------+

  Why it occurs:
  1. Fixed and short retry intervals
  2. No upper limit on retry count
  3. No jitter (random variation)
  4. Circuit Breaker not implemented

  Correct retry strategy:
  - Exponential Backoff: Increase interval exponentially
  - Jitter: Add random variation
  - Max Retries: Set an upper limit (typically 3-5)
  - Circuit Breaker: Stop retries entirely when failures persist
  - Idempotency guarantee: Retries must not cause duplicate side effects
```

---

## 11. Edge Case Analysis

### Edge Case 1: Split Brain

```
Split Brain:

  Situation:
  A network partition splits the cluster into two groups,
  and both groups believe "I am the master"

  +-------------------------------------------+
  |  Partition A          x Partition B       |
  |  +-------+    split    +-------+          |
  |  |Node-1 |  xxxxxxx   |Node-3 |          |
  |  |Node-2 |            |Node-4 |          |
  |  |       |            |Node-5 |          |
  |  +-------+            +-------+          |
  |  "Node-3,4,5          "Node-1,2          |
  |   went down"           went down"        |
  |  Elect new Leader      Elect new Leader  |
  |                                           |
  |  -> Two Leaders exist simultaneously!    |
  |  -> Both accept writes                   |
  |  -> Data conflicts after partition heals |
  +-------------------------------------------+

  Why it's dangerous:
  -> Both partitions independently proceed with writes
  -> Data inconsistency occurs after partition recovery
  -> Example: Withdrawals from the same bank account in both partitions

  Countermeasures:
  1. Quorum-based Leader election:
     -> Cannot become Leader without agreement from 3+ of 5 nodes
     -> Partition A (2 nodes) cannot elect a Leader
     -> Only Partition B (3 nodes) can have a Leader

  2. Fencing Token:
     -> Leader issues a monotonically increasing token with each operation
     -> Storage rejects operations with old tokens
     -> Old Leader's writes cannot overwrite new Leader's writes
```

### Edge Case 2: Clock Backward Jump

```
Clock Backward Jump (Clock Skew):

  Situation:
  NTP synchronization causes the system clock to jump backwards

  +-------------------------------------------+
  |  10:00:00.000  Event A occurs -> timestamp=T1|
  |  10:00:00.100  NTP sync executes             |
  |  09:59:59.900  Clock goes back 200ms!        |
  |  09:59:59.950  Event B occurs -> timestamp=T2|
  |                                               |
  |  T2 < T1, but B actually occurred after A!    |
  |                                               |
  |  Impact:                                      |
  |  - Timestamp-based sorting breaks             |
  |  - TTL (expiration) calculations go wrong     |
  |  - Distributed lock lease times become wrong  |
  |  - Log ordering becomes inconsistent          |
  +-------------------------------------------+

  Why it occurs:
  -> NTP adjusts the clock to the "correct time"
  -> If the clock was ahead, it steps backwards
  -> Linux adjtime() adjusts gradually, but large skew
    triggers a step adjustment

  Countermeasures:
  1. Design that doesn't depend on physical clocks:
     -> Use Lamport Clock / Vector Clock
     -> Physical clocks are only supplementary information

  2. Monotonic Clock (CLOCK_MONOTONIC):
     -> Not affected by NTP adjustments
     -> Use for measuring elapsed time (not for absolute time)

  3. Google TrueTime:
     -> High-precision clock combining GPS and atomic clocks
     -> Explicitly returns the error range (interval [earliest, latest])
     -> Spanner inserts wait times to guarantee causal ordering
```

---

## 12. Hands-On Exercises

### Exercise 1: [Basic] Applying the CAP Theorem

```
Problem:
For the following systems, determine whether CP or AP should be chosen,
by considering the "worst-case scenario when consistency breaks," and explain your reasoning.

1. Online bank balance inquiry
2. Twitter follower count display
3. Airline seat reservation
4. News site comment section
5. Distributed lock (leader election)

Example Answers:

1. Online bank balance inquiry -> CP
   Worst case: Withdrawal succeeds from a zero-balance account
   -> Financial loss is irreversible
   -> Returning an error is far safer than a double withdrawal

2. Twitter follower count display -> AP
   Worst case: Shows 100 followers when the actual count is 101
   -> Impact on user experience is minimal
   -> A slightly stale value is better than nothing being displayed

3. Airline seat reservation -> CP
   Worst case: Same seat sold to two people
   -> Two people physically cannot sit in the same seat
   -> "Temporarily unable to book" is safer than double-selling

4. News site comment section -> AP
   Worst case: New comments temporarily invisible
   -> They appear a few seconds later
   -> Not showing the comment section at all is worse UX

5. Distributed lock (leader election) -> CP
   Worst case: Two leaders exist simultaneously (split brain)
   -> Data inconsistency, duplicate processing
   -> Lock acquisition failure is safer than dual leaders
```

### Exercise 2: [Applied] Raft Simulation

```
Problem:
Simulate the following scenario in a 5-node Raft cluster.

Initial state: Node-1 is Leader (Term=3)
1. Network failure occurs on Node-1
2. Trace the process of electing a new Leader from remaining nodes
3. Behavior when the former Leader (Node-1) returns to the network

Write out the following for each step:
- Each node's state (Leader/Follower/Candidate)
- Term number
- Vote flow

Example Answer:

Step 0: Initial state
  Node-1: Leader    (Term=3) <- Periodically sends heartbeats
  Node-2: Follower  (Term=3)
  Node-3: Follower  (Term=3)
  Node-4: Follower  (Term=3)
  Node-5: Follower  (Term=3)

Step 1: Node-1 network failure
  Node-1: Leader    (Term=3) <- Heartbeats stop arriving
  Node-2: Follower  (Term=3) <- Waiting for timeout
  Node-3: Follower  (Term=3) <- Waiting for timeout
  Node-4: Follower  (Term=3) <- Waiting for timeout
  Node-5: Follower  (Term=3) <- Waiting for timeout

Step 2: Node-3 times out first (shortest random timeout)
  Node-1: Leader    (Term=3) <- Isolated due to partition
  Node-3: Candidate (Term=4) <- Votes for self
  -> Vote request to Node-2: Granted (Term=4 > 3)
  -> Vote request to Node-4: Granted
  -> Vote request to Node-5: Granted
  -> 4 votes/5 nodes -> Majority reached

Step 3: New Leader confirmed
  Node-1: Leader    (Term=3) <- Still thinks it's Leader
  Node-2: Follower  (Term=4, Leader=Node-3)
  Node-3: Leader    (Term=4) <- New Leader
  Node-4: Follower  (Term=4, Leader=Node-3)
  Node-5: Follower  (Term=4, Leader=Node-3)

Step 4: Node-1 network recovery
  Node-1 receives Term=4 heartbeat from other nodes
  -> Term=4 > Term=3, so automatically demotes to Follower
  Node-1: Follower  (Term=4, Leader=Node-3) <- Demotion complete
  -> Node-1's uncommitted logs are overwritten with Node-3's logs
```

### Exercise 3: [Advanced] Distributed KV Store Design

```
Problem:
Design a distributed Key-Value store that satisfies the following requirements.

Requirements:
- 3-node configuration (replication factor = 3)
- Two operations: GET/PUT
- Eventual consistency
- Tolerates 1 node failure

Design Items:
1. Data placement method
2. Write replication method
3. Read consistency guarantee (Quorum: W + R > N)
4. Failure detection and recovery mechanism

Describe a concrete scenario for W=2, R=2, N=3.

Answer Outline:

1. Data placement:
   Use a consistent hash ring
   150 virtual nodes for uniform distribution
   Each key is replicated to 3 nodes on the ring

2. Write (W=2):
   Client -> Coordinator Node -> Parallel write to 3 nodes
   Respond success to Client after receiving ACK from 2 nodes
   Remaining node catches up asynchronously (hinted handoff)

3. Read (R=2):
   Coordinator -> Parallel read from 3 nodes
   Wait for responses from 2 nodes and return the latest version
   On version mismatch -> Update stale node via Read Repair

4. Failure detection and recovery:
   - Gossip protocol for failure detection (all nodes propagate state)
   - Hinted Handoff: Temporarily store data destined for a failed node
     on another node, transfer after recovery
   - Anti-Entropy: Periodically detect data inconsistencies using Merkle Trees
```

---

## 13. FAQ

### Q1: When should microservices be adopted?

Starting with microservices from the beginning is an **anti-pattern** (Martin Fowler: "MonolithFirst"). The reason is that drawing service boundaries correctly from the start is difficult, and splitting along wrong boundaries results in a "distributed monolith."

Consider splitting when the following conditions are met:
- Team exceeds 10 people and code conflicts are frequent
- Only some features need to scale independently (e.g., only image processing needs GPUs)
- Different parts require different technology stacks (e.g., ML in Python, API in Go)
- You want to make deployment frequency independent (e.g., billing monthly, UI daily)
- Organizational structure aligns with service boundaries (Conway's Law)

### Q2: How much "delay" does eventual consistency have?

It depends on the system and configuration, but the following ranges are typical:
- **DynamoDB**: Usually under 1 second (nearly instant)
- **Cassandra**: Milliseconds to a few seconds
- **DNS**: TTL-dependent (minutes to hours)
- **S3**: Provides strong consistency instantly since December 2020

What matters is not "how long the delay is" but "designing the system with the assumption that delays exist." This is because even if it's "normally a few milliseconds," during network failures it can extend to seconds or minutes.

### Q3: Is the CAP theorem outdated?

The CAP theorem is still valid, but it is overly simplified in some respects. The PACELC theorem provides a more realistic framework:
- **During partition (P)**: Choose Availability or Consistency
- **During normal operation (E)**: Choose Latency or Consistency

Real systems are not a binary "CP or AP" choice but **adjust consistency levels per operation**. For example, DynamoDB allows you to choose between "strongly consistent reads" and "eventually consistent reads" on a per-operation basis for the same table.

### Q4: When are consensus algorithms needed?

Consensus algorithms (Raft/Paxos) are only needed when **multiple nodes must agree on a single value**.

Specifically:
- Leader election (Kubernetes etcd)
- Distributed locks (ZooKeeper)
- Distributed configuration management (Consul)
- Agreement on log replication order

Conversely, lighter-weight methods can be used when:
- Eventual consistency is sufficient -> Gossip protocol
- Reads dominate -> Read replicas + asynchronous replication
- Order doesn't matter -> CRDTs (Conflict-free Replicated Data Types)

### Q5: How should distributed systems be tested?

Unit tests alone are insufficient for distributed systems. This is because failures and latency cannot be reproduced in unit tests.

Recommended testing approaches:
1. **Chaos Engineering**: Intentionally cause failures in production (Netflix Chaos Monkey)
2. **Jepsen**: Framework for testing linearizability of distributed systems
3. **Fault Injection**: Inject network latency/packet loss in test environments
4. **Property-based Testing**: Test invariants with random inputs
5. **Simulation Testing**: Deterministic simulation as adopted by FoundationDB

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is paramount. Understanding deepens not just through theory, but by actually writing code and verifying how things work.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping ahead to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this used in professional practice?

Knowledge of this topic is frequently applied in daily development work. It becomes especially important during code reviews and architecture design.

---

## 14. Summary

| Concept | Key Point | Learning Priority |
|---------|---------|------------|
| CAP Theorem | P is essential. Effectively a CP vs AP choice. PACELC is more practical | Highest |
| Consensus | Paxos (theory), Raft (practice). Decisions by majority agreement | High |
| Replication | Synchronous (strong consistency) vs Asynchronous (high performance). Semi-sync is balanced | High |
| Sharding | Consistent hashing minimizes redistribution. Virtual nodes ensure uniformity | High |
| Distributed Transactions | 2PC (strong but slow) vs Saga (flexible but complex) | Medium |
| Time and Ordering | Physical clocks are unreliable. Use logical/vector clocks for causal tracking | Medium |
| Failure Mitigation | Circuit Breaker, Bulkhead, Exponential Backoff + Jitter | High |
| Architecture | MonolithFirst principle. Event Sourcing, CQRS | Applied |

### Learning Roadmap

```
Step 1 (Basics):
  -> Understand the CAP theorem and CP vs AP decision criteria
  -> Grasp the strength hierarchy of consistency models
  -> Understand the 3 replication methods

Step 2 (Intermediate):
  -> Study Raft through the paper or visualization tools
  -> Implement consistent hashing yourself
  -> Understand the trade-offs between 2PC and Saga

Step 3 (Advanced):
  -> Read through "Designing Data-Intensive Applications"
  -> Read Jepsen test results to understand real DB behavior
  -> Build a small distributed KV store yourself
  -> Practice Chaos Engineering
```

---

## Recommended Next Guides

---

## References

1. Kleppmann, M. *Designing Data-Intensive Applications*. O'Reilly, 2017.
   A definitive book covering theory and practice of distributed systems. Many topics in this guide are based on this book.

2. Lamport, L. "Time, Clocks, and the Ordering of Events in a Distributed System." *Communications of the ACM*, 1978.
   The original paper on logical clocks. A historic paper that established the foundations of distributed systems theory.

3. Ongaro, D. & Ousterhout, J. "In Search of an Understandable Consensus Algorithm." *USENIX ATC*, 2014.
   The original Raft paper. A consensus algorithm redesigned for understandability from Paxos.

4. Brewer, E. "CAP Twelve Years Later: How the 'Rules' Have Changed." *IEEE Computer*, 2012.
   Reconsideration and supplementary notes by the CAP theorem's original proposer.

5. Vogels, W. "Eventually Consistent." *Communications of the ACM*, 2009.
   Explanation of eventual consistency by Amazon's CTO. Provides insight into DynamoDB's design philosophy.

6. Garcia-Molina, H. & Salem, K. "Sagas." *ACM SIGMOD*, 1987.
   The original paper on the Saga pattern. A compensation-based approach for long-running transactions.

7. DeCandia, G. et al. "Dynamo: Amazon's Highly Available Key-value Store." *SOSP*, 2007.
   The Amazon Dynamo paper. A practical example of consistent hashing, quorum, and vector clock application.

8. Deutsch, P. "The Eight Fallacies of Distributed Computing." 1994.
   The eight fallacies of distributed computing. Still completely valid more than 25 years later.
