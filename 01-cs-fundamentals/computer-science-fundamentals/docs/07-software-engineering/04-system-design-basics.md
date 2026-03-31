# Introduction to System Design

> System design is a "no single right answer" problem — it is the art of making the best choices among trade-offs.

## What You Will Learn in This Chapter

- [ ] Understand the basic concepts of scalability
- [ ] Be able to explain the CAP theorem
- [ ] Know the major system design patterns
- [ ] Understand the mechanisms and types of load balancing
- [ ] Be able to appropriately choose caching strategies
- [ ] Learn database design patterns
- [ ] Understand how to use message queues
- [ ] Grasp the advantages and disadvantages of microservices architecture
- [ ] Learn API design principles
- [ ] Acquire design techniques for availability and reliability
- [ ] Master practical estimation techniques for system design


## Prerequisites

Understanding the following topics beforehand will help you get more out of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Clean Code](./03-clean-code.md)

---

## 1. Scalability

### 1.1 Directions of Scaling

```
Two directions of scaling:

  Vertical Scaling (Scale-Up):
  -> Enhance the machine (add CPU, RAM)
  -> Has limits, causes downtime
  -> Simple but expensive

  Horizontal Scaling (Scale-Out):
  -> Add more machines
  -> Theoretically infinite scaling
  -> Complex but cost-effective

  Typical Web Architecture:
  +--------+   +--------------+   +----------+
  | Client |-->| Load Balancer|-->| Web x N  |
  +--------+   +--------------+   +----+-----+
                                       |
                              +--------+--------+
                              v        v        v
                          +------+ +------+ +------+
                          |Cache | | DB   | |Queue |
                          |Redis | |Master| |SQS   |
                          +------+ |Slave | +------+
                                   +------+
```

### 1.2 Vertical Scaling in Practice

```
Characteristics of Vertical Scaling (Scale-Up):

  Advantages:
  - Easiest to implement (no application changes required)
  - Avoids the complexity of distributed systems
  - Data consistency is naturally maintained on a single node
  - Easy to operate (fewer monitoring targets)

  Limitations:
  - Hardware has physical limits
    - CPU: up to 128-256 cores
    - RAM: up to 6-12 TB
    - Storage: IOPS ceiling
  - Cost increases exponentially
    - 2x performance != 2x cost; typically 3-5x
  - Becomes a Single Point of Failure (SPOF)
  - Downtime occurs during scale-up

  When to use:
  - When traffic is relatively small (QPS < ~1,000)
  - Early in a project when scale is unknown
  - Temporary database performance improvements
  - Short-term traffic spikes

  Examples:
  - AWS EC2: t3.micro -> m5.24xlarge (96 vCPU, 384 GB RAM)
  - RDS: db.t3.micro -> db.r5.24xlarge
  - Azure: Standard_B1s -> Standard_M128ms (128 vCPU, 3.8 TB RAM)
```

### 1.3 Horizontal Scaling in Practice

```
Characteristics of Horizontal Scaling (Scale-Out):

  Advantages:
  - Theoretically infinite scalability
  - Cost-effective (uses commodity hardware)
  - High fault tolerance (other nodes continue if one goes down)
  - Scale changes possible without downtime

  Challenges:
  - Distributed system complexity
    - Ensuring data consistency
    - Distributed transactions
    - Handling network failures
  - State management
    - Session management (Sticky Session vs shared store)
    - Cache consistency
  - Increased deployment and operational complexity

  Considerations for scaling out:
  +--------------------------------------------------+
  | Layer          | Strategy                         |
  +--------------------------------------------------+
  | Web/API       | Make stateless + LB               |
  | Session       | Externalize to Redis/Memcached    |
  | Database      | Read Replica + Sharding           |
  | Files         | Object storage like S3/GCS        |
  | Task Processing| Message Queue + Workers          |
  | Search        | Elasticsearch/Solr                |
  +--------------------------------------------------+
```

### 1.4 Stateless Architecture

```python
# --- Stateful vs Stateless Servers ---

# Bad: Stateful (difficult to scale out)
class StatefulServer:
    def __init__(self):
        self.sessions = {}  # Stores sessions in server memory

    def login(self, user_id, password):
        session_id = generate_session_id()
        self.sessions[session_id] = {
            "user_id": user_id,
            "login_time": datetime.now()
        }
        return session_id

    def get_user(self, session_id):
        # Session info only exists on THIS server!
        session = self.sessions.get(session_id)
        if not session:
            raise AuthenticationError("Invalid session")
        return session["user_id"]

# Good: Stateless (can freely scale out)
class StatelessServer:
    def __init__(self, session_store):
        # Sessions stored in an external store (e.g., Redis)
        self.session_store = session_store

    def login(self, user_id, password):
        session_id = generate_session_id()
        self.session_store.set(session_id, {
            "user_id": user_id,
            "login_time": datetime.now().isoformat()
        }, ttl=3600)  # Expires in 1 hour
        return session_id

    def get_user(self, session_id):
        # Session can be accessed from any server
        session = self.session_store.get(session_id)
        if not session:
            raise AuthenticationError("Invalid session")
        return session["user_id"]
```

```
Patterns for achieving statelessness:

  1. Session Externalization
     - Store sessions in Redis/Memcached
     - Use JWT tokens to eliminate server-side sessions

  2. File Storage Externalization
     - Store user uploads in S3/GCS
     - Do not depend on local disk

  3. Configuration Externalization
     - Environment variables
     - Configuration services like Consul, etcd
     - AWS Parameter Store / Secrets Manager

  4. Cache Externalization
     - Use Redis/Memcached as a cache layer
     - Use local cache only for volatile data

  +---------------------------------------------+
  |              Load Balancer                   |
  |   +---------+---------+---------+           |
  |   | Server1 | Server2 | Server3 | <- Can be |
  |   |(no state)|(no state)|(no state)|  added   |
  |   +----+----+----+----+----+----+ freely    |
  |        +---------+---------+                 |
  |                  v                           |
  |         +--------------+                     |
  |         | Redis/Shared | <- State is         |
  |         |     DB       |    centralized here |
  |         +--------------+                     |
  +---------------------------------------------+
```

---

## 2. CAP Theorem

### 2.1 CAP Theorem Basics

```
CAP Theorem: A distributed system can only guarantee two out of three simultaneously

  C -- Consistency: All nodes see the same data at the same time
  A -- Availability: Every request receives a response
  P -- Partition Tolerance: System operates even during network partitions

  Network partitions are unavoidable -> Effectively a choice between CP or AP

  CP (Consistency Priority): Returns an error during partitions
  -> Bank transfers, inventory management
  -> PostgreSQL, MongoDB (default), ZooKeeper

  AP (Availability Priority): May return stale data during partitions
  -> Social media timelines, shopping carts
  -> Cassandra, DynamoDB, CouchDB

  PACELC Theorem (CAP Extension):
  During Partition (P): Choose A or C
  During normal operation (E): Choose Latency (L) or Consistency (C)
```

### 2.2 Consistency Models in Detail

```
Types of consistency models (from strongest to weakest):

  1. Linearizability
     - Strongest consistency guarantee
     - All operations appear to execute in a single global order
     - Readable from any node immediately after write
     - Example: Zookeeper, etcd
     - Cost: Highest latency

  2. Sequential Consistency
     - Operation order within each process is maintained
     - Order between processes is not guaranteed
     - Example: Distributed queues

  3. Causal Consistency
     - Causally related operations are guaranteed to be ordered
     - Causally unrelated operations may appear in any order
     - Example: Messaging apps (replies always appear after the original message)

  4. Eventual Consistency
     - Weakest guarantee
     - If writes stop, all nodes will eventually converge to the same value
     - Stale values may be returned during reads
     - Example: DNS, S3, DynamoDB (default setting)
     - Cost: Lowest latency

  Practical selection guidelines:
  +--------------------------------------------------+
  | Use Case                 | Recommended Model      |
  +--------------------------------------------------+
  | Bank transfers           | Linearizability        |
  | Inventory management     | Linearizability or     |
  |                          | Causal Consistency     |
  | User profiles            | Eventual Consistency   |
  | Social media likes count | Eventual Consistency   |
  | Message delivery         | Causal Consistency     |
  | E-commerce order processing | Linearizability     |
  | Search index updates     | Eventual Consistency   |
  | Leader election          | Linearizability        |
  +--------------------------------------------------+
```

### 2.3 Data Replication

```
Replication strategies:

  1. Single-Leader
     +----------+    +-----------+
     |  Leader   |--->| Follower1 |  Writes go to Leader only
     | (Master)  |--->| Follower2 |  Reads from anywhere
     +----------+    | Follower3 |
                     +-----------+
     - Advantage: Easy to maintain consistency
     - Disadvantage: Leader is a single point of failure, writes don't scale
     - Example: MySQL, PostgreSQL, MongoDB

  2. Multi-Leader
     +----------+    +----------+
     |  Leader1  |<-->|  Leader2  |  Writes can go to any leader
     | (Tokyo)   |    | (US-East) |
     +----------+    +----------+
     - Advantage: Improved write availability, low latency
     - Disadvantage: Conflict resolution required
     - Example: CouchDB, Galera Cluster

  3. Leaderless
     +------+  +------+  +------+
     |Node1 |  |Node2 |  |Node3 |  All nodes are equal
     +------+  +------+  +------+
     - Quorum: W + R > N ensures consistency
       - W=write node count, R=read node count, N=total node count
       - Example: N=3, W=2, R=2 -> Write completes when 2 succeed, read from 2
     - Advantage: High availability, no single point of failure
     - Disadvantage: Complex implementation, conflict resolution needed
     - Example: Cassandra, DynamoDB, Riak

  Conflict resolution strategies:
  1. Last Write Wins (LWW)
     - The write with the latest timestamp wins
     - Simple but risk of data loss
  2. Merge
     - Retain and combine both changes
     - CRDTs (Conflict-free Replicated Data Types) are effective
  3. Application-level resolution
     - Present conflicts to the user for selection
     - Example: Google Docs collaborative editing
```

---

## 3. Load Balancing

### 3.1 Load Balancing Basics

```
Load balancer placement:

  Client -> [LB1] -> Web Server -> [LB2] -> App Server -> [LB3] -> DB

  L4 (Transport Layer) Load Balancer:
  - Distributes at TCP/UDP level
  - Does not inspect packet contents
  - Fast, low overhead
  - Examples: AWS NLB, HAProxy (L4 mode), Linux IPVS

  L7 (Application Layer) Load Balancer:
  - Distributes at HTTP/HTTPS level
  - Can route based on URL path, headers, cookies, etc.
  - SSL termination, compression, caching features
  - Examples: AWS ALB, Nginx, HAProxy (L7 mode), Envoy

  Distribution Algorithms:
  +--------------------------------------------------+
  | Algorithm              | Description              |
  +--------------------------------------------------+
  | Round Robin            | Distributes in sequence  |
  | Weighted Round Robin   | Distributes by weight    |
  | Least Connections      | Routes to server with    |
  |                        | fewest connections        |
  | Shortest Response Time | Routes to fastest server |
  | IP Hash                | Fixed by client IP       |
  | Consistent Hashing     | Minimizes impact of      |
  |                        | node addition/removal     |
  +--------------------------------------------------+
```

### 3.2 Consistent Hashing

```python
import hashlib
from bisect import bisect_right

class ConsistentHash:
    """Example implementation of consistent hashing"""

    def __init__(self, nodes=None, replicas=150):
        """
        Args:
            nodes: List of initial nodes
            replicas: Number of virtual nodes per node (more = more uniform distribution)
        """
        self.replicas = replicas
        self.ring = {}       # hash value -> node name
        self.sorted_keys = []  # sorted list of hash values

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        """Compute hash value for a key"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str):
        """Add a node"""
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
            self.sorted_keys.append(hash_value)
        self.sorted_keys.sort()

    def remove_node(self, node: str):
        """Remove a node"""
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            del self.ring[hash_value]
            self.sorted_keys.remove(hash_value)

    def get_node(self, key: str) -> str:
        """Get the node responsible for a key"""
        if not self.ring:
            raise ValueError("No nodes available")

        hash_value = self._hash(key)
        idx = bisect_right(self.sorted_keys, hash_value)
        if idx == len(self.sorted_keys):
            idx = 0  # Wrap around the ring
        return self.ring[self.sorted_keys[idx]]

# Usage example
ch = ConsistentHash(["server-1", "server-2", "server-3"])

# Data assignment
for key in ["user:1001", "user:1002", "user:1003", "order:5001"]:
    node = ch.get_node(key)
    print(f"{key} -> {node}")

# Adding a node -> only ~1/N of keys are affected
ch.add_node("server-4")

# Removing a node -> only ~1/N of keys are affected
ch.remove_node("server-2")
```

### 3.3 Health Checks and Failover

```
Types of health checks:

  1. Passive Health Check
     - Monitors the results of actual requests
     - Excludes a node when error rate exceeds threshold
     - No additional traffic required

  2. Active Health Check
     - Periodically sends requests to a dedicated endpoint
     - /health or /ready endpoints
     - Can detect failures faster
```

```python
# Example implementation of a health check endpoint
from flask import Flask, jsonify
import psycopg2
import redis

app = Flask(__name__)

@app.route("/health")
def health_check():
    """Basic health check (is the server running?)"""
    return jsonify({"status": "ok"}), 200

@app.route("/health/detailed")
def detailed_health_check():
    """Detailed health check (also checks dependent services)"""
    checks = {}

    # Database connectivity check
    try:
        conn = psycopg2.connect("postgresql://localhost/myapp")
        conn.execute("SELECT 1")
        conn.close()
        checks["database"] = {"status": "healthy"}
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "error": str(e)}

    # Redis connectivity check
    try:
        r = redis.Redis()
        r.ping()
        checks["cache"] = {"status": "healthy"}
    except Exception as e:
        checks["cache"] = {"status": "unhealthy", "error": str(e)}

    # Disk space check
    import shutil
    usage = shutil.disk_usage("/")
    free_percent = usage.free / usage.total * 100
    if free_percent > 10:
        checks["disk"] = {"status": "healthy", "free_percent": round(free_percent, 1)}
    else:
        checks["disk"] = {"status": "warning", "free_percent": round(free_percent, 1)}

    # Overall status determination
    overall = "healthy"
    for check in checks.values():
        if check["status"] == "unhealthy":
            overall = "unhealthy"
            break
        if check["status"] == "warning":
            overall = "degraded"

    status_code = 200 if overall == "healthy" else 503
    return jsonify({"status": overall, "checks": checks}), status_code
```

```
Failover patterns:

  1. Active-Passive
     +----------+    +----------+
     |  Active   |    | Passive  |  Monitored via Heartbeat
     | (running) |--->| (standby)|  If Active goes down,
     +----------+    +----------+  Passive takes over
     - Pros: Simple, high data consistency
     - Cons: Passive resources are idle

  2. Active-Active
     +----------+    +----------+
     | Active-1  |<-->| Active-2  |  Both handle requests
     | (running) |    | (running) |  Continues if one goes down
     +----------+    +----------+
     - Pros: Resource efficient, high availability
     - Cons: Complex data synchronization
```

---

## 4. Caching

### 4.1 Cache Hierarchy

```
Cache hierarchy (closer to client at the top):

  +-----------------------------------------------------+
  |  Client Cache                                        |
  |  - Browser cache (Cache-Control, ETag)               |
  |  - Mobile app local storage                          |
  |  - Latency: 0ms (no network needed)                  |
  +-----------------------------------------------------+
  |  CDN Cache                                           |
  |  - CloudFront, Cloudflare, Fastly                    |
  |  - Geographically distributed edge servers           |
  |  - Latency: 1-10ms                                   |
  +-----------------------------------------------------+
  |  Application Cache                                   |
  |  - Redis, Memcached                                  |
  |  - In-memory sub-millisecond access                  |
  |  - Latency: 1-5ms                                    |
  +-----------------------------------------------------+
  |  Database Cache                                      |
  |  - Query cache                                       |
  |  - Buffer pool (InnoDB Buffer Pool, etc.)            |
  |  - Latency: 1-10ms                                   |
  +-----------------------------------------------------+
  |  Disk Cache                                          |
  |  - OS page cache                                     |
  |  - SSD cache                                         |
  |  - Latency: 0.1-1ms                                  |
  +-----------------------------------------------------+
```

### 4.2 Caching Strategies

```python
import redis
import json
from datetime import timedelta

r = redis.Redis()

# --- 1. Cache-Aside (Lazy Loading) ---
# Application controls cache reads and writes
# Most common pattern

def get_user_cache_aside(user_id: str) -> dict:
    """Cache-Aside: Check cache first, if miss then fetch from DB and cache"""
    cache_key = f"user:{user_id}"

    # 1. Check cache
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)  # Cache hit

    # 2. Cache miss -> Fetch from DB
    user = db.find_user(user_id)
    if user is None:
        return None

    # 3. Store in cache (with TTL)
    r.setex(cache_key, timedelta(hours=1), json.dumps(user.to_dict()))

    return user.to_dict()

def update_user_cache_aside(user_id: str, data: dict):
    """Invalidate cache on update"""
    db.update_user(user_id, data)
    r.delete(f"user:{user_id}")  # Delete cache


# --- 2. Write-Through ---
# Update cache and DB simultaneously on write

def save_user_write_through(user_id: str, data: dict):
    """Write-Through: Update DB and cache simultaneously"""
    # Write to DB
    db.save_user(user_id, data)

    # Also write to cache
    cache_key = f"user:{user_id}"
    r.setex(cache_key, timedelta(hours=1), json.dumps(data))


# --- 3. Write-Behind (Write-Back) ---
# Write to cache first, then asynchronously persist to DB

class WriteBehindCache:
    def __init__(self):
        self.write_queue = []

    def save(self, user_id: str, data: dict):
        """Write to cache first"""
        cache_key = f"user:{user_id}"
        r.setex(cache_key, timedelta(hours=1), json.dumps(data))

        # Add to write queue (asynchronously persisted to DB)
        self.write_queue.append(("user", user_id, data))

    def flush(self):
        """Batch write queue contents to DB"""
        while self.write_queue:
            entity_type, entity_id, data = self.write_queue.pop(0)
            db.save(entity_type, entity_id, data)


# --- 4. Read-Through ---
# Cache itself manages loading from DB

class ReadThroughCache:
    def __init__(self, loader):
        self.loader = loader  # Data retrieval function

    def get(self, key: str) -> dict:
        cached = r.get(key)
        if cached:
            return json.loads(cached)

        # Cache automatically fetches from DB
        data = self.loader(key)
        if data:
            r.setex(key, timedelta(hours=1), json.dumps(data))
        return data

# Usage example
user_cache = ReadThroughCache(loader=lambda key: db.find_user(key.split(":")[1]))
user = user_cache.get("user:1001")
```

```
Comparison of caching strategies:

  +--------------------------------------------------------+
  | Strategy      | Advantages        | Disadvantages       |
  +--------------------------------------------------------+
  | Cache-Aside   | Simple, versatile | Increased latency   |
  |               |                   | on cache miss        |
  +--------------------------------------------------------+
  | Write-Through | High consistency  | Slow writes          |
  |               |                   | (writes to 2 places) |
  +--------------------------------------------------------+
  | Write-Behind  | Fast writes       | Risk of data loss    |
  |               | Batch optimization| Complex              |
  |               | possible          | implementation       |
  +--------------------------------------------------------+
  | Read-Through  | Simplifies        | May have limited     |
  |               | application code  | customizability      |
  +--------------------------------------------------------+
```

### 4.3 Cache Challenges and Countermeasures

```
Common cache challenges:

  1. Cache Stampede (Thundering Herd)
     - A popular key's cache expires
     - Massive requests simultaneously hit the DB -> DB overload
     Countermeasures:
     - Locking: Only one request accesses DB, others wait
     - Probabilistic early renewal: Randomly refresh before TTL expires
     - Double caching: Backup cache behind the primary cache

  2. Cache Penetration
     - Massive requests for non-existent keys
     - Always cache miss -> DB hit every time
     Countermeasures:
     - Negative caching: Cache non-existent keys too (with short TTL)
     - Bloom filter: Fast existence check

  3. Cache Avalanche
     - Many keys expire at the same time
     - Simultaneous DB access surge
     Countermeasures:
     - Add random jitter to TTL
     - Gradual cache warm-up

  4. Data Inconsistency
     - DB is updated but cache is stale
     Countermeasures:
     - Ensure consistency with Write-Through/Write-Behind
     - Cache invalidation patterns
     - Set short TTLs
```

```python
# Implementation example for cache stampede protection
import time
import threading

class StampedeProtectedCache:
    """Cache with stampede protection"""

    def __init__(self):
        self.locks = {}  # Lock per key
        self.lock_manager = threading.Lock()

    def get_or_compute(self, key: str, compute_fn, ttl_seconds: int = 3600):
        """Get from cache. On miss, acquire lock so only 1 request queries DB"""
        # Check cache first
        cached = r.get(key)
        if cached:
            return json.loads(cached)

        # Acquire lock (only 1 request per key)
        lock = self._get_lock(key)
        acquired = lock.acquire(timeout=5)

        if not acquired:
            # Lock acquisition failed -> wait briefly and retry
            time.sleep(0.1)
            cached = r.get(key)
            return json.loads(cached) if cached else None

        try:
            # Double-check (another thread may have cached during lock wait)
            cached = r.get(key)
            if cached:
                return json.loads(cached)

            # Fetch from DB and cache
            value = compute_fn()
            if value is not None:
                # Add jitter to TTL (avalanche prevention)
                import random
                jitter = random.randint(0, ttl_seconds // 10)
                r.setex(key, ttl_seconds + jitter, json.dumps(value))
            return value
        finally:
            lock.release()

    def _get_lock(self, key):
        with self.lock_manager:
            if key not in self.locks:
                self.locks[key] = threading.Lock()
            return self.locks[key]
```

---

## 5. Database Design

### 5.1 RDB vs NoSQL Selection

```
Database selection criteria:

  When to choose RDB (Relational DB):
  - Relationships between data are important (JOINs needed)
  - Transactions (ACID) are required
  - Schema is stable
  - Complex queries are needed
  - Data consistency is critical
  Examples: User management, order management, accounting systems

  When to choose NoSQL:
  - Very large data volumes
  - High write throughput needed
  - Schema changes frequently
  - Geographically distributed data
  - Flexible data model needed
  Examples: Log storage, IoT data, content management

  Types of NoSQL:
  +----------------------------------------------------+
  | Type              | Characteristics   | Examples     |
  +----------------------------------------------------+
  | Key-Value         | Fast, simple      | Redis,       |
  |                   |                   | DynamoDB     |
  +----------------------------------------------------+
  | Document          | Flexible schema   | MongoDB,     |
  |                   | JSON-like         | CouchDB      |
  +----------------------------------------------------+
  | Column Family     | For large-scale   | Cassandra,   |
  |                   | analytics,        | HBase        |
  |                   | fast writes       |              |
  +----------------------------------------------------+
  | Graph             | Relationship      | Neo4j,       |
  |                   | traversal,        | Neptune      |
  |                   | for recommendations, SNS |       |
  +----------------------------------------------------+
  | Time Series       | Specialized for   | InfluxDB,    |
  |                   | time-series data, | TimescaleDB  |
  |                   | IoT, metrics      |              |
  +----------------------------------------------------+
```

### 5.2 Sharding

```
Sharding (Horizontal Partitioning):
  - Distributes data across multiple databases
  - Each shard is an independent database instance

  Sharding strategies:

  1. Range Sharding
     - Split by key range
     Example: user_id 1-1000 -> Shard1, 1001-2000 -> Shard2
     - Advantage: Range queries are efficient
     - Disadvantage: Hotspots are likely

  2. Hash Sharding
     - Split by hash value of key
     Example: hash(user_id) % 4 -> Shard number
     - Advantage: Uniform distribution
     - Disadvantage: Range queries are inefficient

  3. Directory-Based Sharding
     - A lookup service determines the shard
     - Advantage: Flexible assignment
     - Disadvantage: Lookup service is a single point of failure

  Sharding challenges:
  +--------------------------------------------------+
  | Challenge          | Countermeasure               |
  +--------------------------------------------------+
  | Difficulty with    | Application-side JOIN        |
  | JOINs              | Data denormalization         |
  +--------------------------------------------------+
  | Transaction        | Two-phase commit             |
  | difficulty         | Saga pattern                 |
  +--------------------------------------------------+
  | Resharding         | Consistent hashing           |
  | (repartitioning)   | Sharding middleware like     |
  |                    | Vitess                       |
  +--------------------------------------------------+
  | Hotspots           | Careful shard key selection  |
  |                    | Salting (adding salt)        |
  +--------------------------------------------------+
```

### 5.3 Database Index Design

```sql
-- Basic principles of indexing

-- 1. Primary key automatically creates an index
CREATE TABLE users (
    id BIGINT PRIMARY KEY,         -- Auto-indexed
    email VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP NOT NULL
);

-- 2. Create indexes on columns used in search conditions
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_status ON users (status);

-- 3. Composite index (column order matters)
-- Supports WHERE status = 'active' AND created_at > '2024-01-01'
CREATE INDEX idx_users_status_created ON users (status, created_at);
-- Used from the leftmost column (Leftmost Prefix Rule)
-- OK:  WHERE status = 'active' -> used
-- OK:  WHERE status = 'active' AND created_at > ... -> used
-- NG:  WHERE created_at > ... -> NOT used (status is the leading column)

-- 4. Covering index (all needed columns included in the index)
-- SELECT email, name FROM users WHERE status = 'active';
CREATE INDEX idx_users_covering ON users (status, email, name);
-- No need to access table data -> very fast

-- 5. Unique index (uniqueness constraint)
CREATE UNIQUE INDEX idx_users_email_unique ON users (email);

-- Notes on not over-indexing:
-- - INSERT/UPDATE/DELETE become slower (indexes need updating too)
-- - Consumes storage
-- - Guideline: up to 5-10 indexes per table
```

---

## 6. Message Queues

### 6.1 Message Queue Basics

```
Use cases for message queues:

  1. Asynchronous Processing
     - Sending emails, image processing, report generation
     - Reduces user wait time

  2. Peak Load Leveling
     - Absorbs sudden request spikes with the queue
     - Workers process at a constant pace

  3. Loose Coupling Between Services
     - Service A sends a message to the queue
     - Service B receives the message from the queue
     - A and B do not communicate directly

  4. Event Notification
     - Order completed -> Inventory update, email, loyalty points
     - Each service independently processes events

  Architecture:
  +----------+    +----------+    +----------+
  | Producer |-->|  Queue   |-->| Consumer |
  | (sender) |    | (queue)  |    | (receiver)|
  +----------+    +----------+    +----------+

  Major messaging patterns:
  +---------------------------------------------+
  | Pattern          | Description               |
  +---------------------------------------------+
  | Point-to-Point   | 1 message -> 1 consumer   |
  | (Queue)          | Distributed task processing|
  +---------------------------------------------+
  | Pub/Sub          | 1 message -> multiple      |
  | (Topic)          | subscribers                |
  +---------------------------------------------+
  | Request-Reply    | Send a request and         |
  |                  | wait for a reply           |
  +---------------------------------------------+
```

### 6.2 Message Queue Implementation Example

```python
# --- SQS + Python example ---
import boto3
import json

sqs = boto3.client("sqs", region_name="ap-northeast-1")
QUEUE_URL = "https://sqs.ap-northeast-1.amazonaws.com/123456789/my-queue"

# Producer (send messages)
def send_email_task(to: str, subject: str, body: str):
    """Submit an email task to the queue"""
    message = {
        "task": "send_email",
        "payload": {
            "to": to,
            "subject": subject,
            "body": body
        },
        "created_at": datetime.now().isoformat()
    }
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(message),
        MessageGroupId="email-tasks"  # For FIFO queues
    )

# Consumer (process messages)
def process_messages():
    """Retrieve and process messages from the queue"""
    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20,  # Long polling
            VisibilityTimeout=60  # Hidden from other workers during processing
        )

        messages = response.get("Messages", [])
        for msg in messages:
            try:
                task = json.loads(msg["Body"])
                handle_task(task)

                # Processing complete -> Delete message
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"]
                )
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                # Not deleted -> Will be reprocessed after VisibilityTimeout

def handle_task(task: dict):
    """Process based on task type"""
    if task["task"] == "send_email":
        payload = task["payload"]
        email_service.send(payload["to"], payload["subject"], payload["body"])
    elif task["task"] == "generate_report":
        report_service.generate(task["payload"]["report_id"])
    else:
        logger.warning(f"Unknown task type: {task['task']}")
```

### 6.3 Messaging Platform Comparison

```
Comparison of major messaging platforms:

  +---------------------------------------------------------+
  | Product      | Features              | Use Cases         |
  +---------------------------------------------------------+
  | Amazon SQS   | Fully managed         | Simple            |
  |              | Infinite scale        | task queues       |
  |              | FIFO queue support    |                   |
  +---------------------------------------------------------+
  | RabbitMQ     | Rich routing          | Complex routing   |
  |              | Protocol standard     | Enterprise        |
  |              | (AMQP)               | integration       |
  +---------------------------------------------------------+
  | Apache Kafka | Ultra-high throughput | Event streams     |
  |              | Log-based             | Real-time         |
  |              | Replayable            | analytics,        |
  |              |                       | microservices     |
  +---------------------------------------------------------+
  | Redis Streams| Low latency           | Real-time         |
  |              | Simple                | messaging,        |
  |              | Built into Redis      | chat              |
  +---------------------------------------------------------+
  | Google       | Fully managed         | GCP ecosystem     |
  | Pub/Sub      | Globally distributed  | Event-driven      |
  +---------------------------------------------------------+
```

---

## 7. Microservices Architecture

### 7.1 Monolith vs Microservices

```
Monolith Architecture:
  +---------------------------------+
  |           Monolith               |
  |  +-----+------+------+-----+   |
  |  | UI  | Auth | Order| Pay  |   |
  |  +-----+------+------+-----+   |
  |        Single deployment unit    |
  +---------------------------------+
  Advantages:
  - Easy to develop and test
  - Simple deployment
  - Fast in-process communication
  - Easy transactions

  Disadvantages:
  - Difficult to partially scale
  - Technology stack is fixed
  - Coordination between teams required
  - Codebase becomes bloated

Microservices Architecture:
  +------+  +------+  +------+  +------+
  | Auth |  | Order|  | Pay  |  |Notify|
  | Svc  |  | Svc  |  | Svc  |  | Svc  |
  |      |  |      |  |      |  |      |
  | DB   |  | DB   |  | DB   |  | DB   |
  +------+  +------+  +------+  +------+
  Advantages:
  - Independent deployment
  - Independent scaling
  - Technology choice freedom
  - Team autonomy

  Disadvantages:
  - Operational complexity (distributed systems)
  - Network communication overhead
  - Difficult to ensure data consistency
  - Difficult to debug

Decision criteria:
  Startup -> Start with a monolith
  Team under 10 -> Monolith is sufficient
  Clear boundaries -> Consider microservices
  Independent scaling needed -> Consider microservices
  "Monolith First" approach is recommended
```

### 7.2 Inter-Service Communication

```
Inter-service communication patterns:

  1. Synchronous Communication (REST/gRPC)
     +---------+   HTTP/gRPC   +---------+
     | Service |-------------->| Service |
     |    A    |<--------------+    B    |
     +---------+   Response    +---------+
     - Advantage: Intuitive, immediate response
     - Disadvantage: High coupling, risk of cascading failures

  2. Asynchronous Communication (Message Queue)
     +---------+    +-------+    +---------+
     | Service |-->| Queue |-->| Service |
     |    A    |   +-------+   |    B    |
     +---------+               +---------+
     - Advantage: Loose coupling, high fault tolerance
     - Disadvantage: Cannot get results immediately, difficult to debug

  3. Event-Driven
     +---------+    +------------+    +---------+
     | Service |-->| Event Bus  |-->| Service |
     |    A    |   | (Kafka,etc)|-->| Service |
     +---------+   +------------+-->| Service |
                                     +---------+
     - Advantage: Fully decoupled, easy to add new services
     - Disadvantage: Event ordering guarantees, debugging difficulty
```

```python
# --- Inter-service communication implementation examples ---

# 1. REST API call (with circuit breaker)
import requests
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
def call_payment_service(order_id: str, amount: float) -> dict:
    """Call payment service (with circuit breaker)"""
    try:
        response = requests.post(
            "http://payment-service/api/v1/charge",
            json={"order_id": order_id, "amount": amount},
            timeout=5  # Timeout setting is essential
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        raise PaymentServiceTimeout("Payment service timed out")
    except requests.ConnectionError:
        raise PaymentServiceUnavailable("Payment service is unavailable")

# 2. Saga Pattern (distributed transactions)
class OrderSaga:
    """Order processing saga (compensation transactions for each step)"""

    def execute(self, order):
        try:
            # Step 1: Reserve inventory
            reservation = inventory_service.reserve(order.items)

            try:
                # Step 2: Process payment
                payment = payment_service.charge(order.amount)

                try:
                    # Step 3: Arrange shipping
                    shipping = shipping_service.schedule(order)
                except Exception:
                    # Step 3 failed -> Compensate Step 2 (refund)
                    payment_service.refund(payment.transaction_id)
                    raise
            except Exception:
                # Step 2 failed -> Compensate Step 1 (release inventory)
                inventory_service.cancel_reservation(reservation.id)
                raise
        except Exception as e:
            order.status = "failed"
            order.failure_reason = str(e)
            raise OrderFailedError(str(e)) from e

        order.status = "completed"
        return order
```

### 7.3 API Gateway Pattern

```
Roles of an API Gateway:

  +--------+
  | Client |
  +---+----+
      |
  +---v------------------------------+
  |        API Gateway                |
  |  - Routing                        |
  |  - Authentication & Authorization |
  |  - Rate Limiting                  |
  |  - Request/Response Transformation|
  |  - Logging & Monitoring           |
  |  - Caching                        |
  |  - Circuit Breaker                |
  +---+------+------+--------------+
      |      |      |
  +---v-++--v--++--v--+
  |Auth |||Order|||Product|
  | Svc |||Svc  ||| Svc   |
  +-----++-----++-------+

  Common implementations:
  - AWS API Gateway
  - Kong
  - Envoy + Istio
  - Nginx
  - Netflix Zuul / Spring Cloud Gateway
```

---

## 8. API Design

### 8.1 RESTful API Design

```
REST API Design Principles:

  1. Resource-Oriented
     - URLs represent resources (nouns)
     - HTTP methods represent operations

  +----------------------------------------------------+
  | Operation    | Method | URL Example                  |
  +----------------------------------------------------+
  | List         | GET    | /api/v1/users               |
  | Detail       | GET    | /api/v1/users/123           |
  | Create       | POST   | /api/v1/users               |
  | Full Update  | PUT    | /api/v1/users/123           |
  | Partial Update| PATCH | /api/v1/users/123           |
  | Delete       | DELETE | /api/v1/users/123           |
  | Sub-resource | GET    | /api/v1/users/123/orders    |
  +----------------------------------------------------+

  2. Status Codes
  +----------------------------------------------------+
  | Code   | Meaning          | Usage                   |
  +----------------------------------------------------+
  | 200    | OK               | Normal response          |
  | 201    | Created          | Resource created         |
  | 204    | No Content       | Success (no response body)|
  | 400    | Bad Request      | Client error             |
  | 401    | Unauthorized     | Authentication failed    |
  | 403    | Forbidden        | Authorization failed     |
  | 404    | Not Found        | Resource doesn't exist   |
  | 409    | Conflict         | Conflict (duplicate, etc.)|
  | 422    | Unprocessable    | Validation error         |
  | 429    | Too Many Requests| Rate limit exceeded      |
  | 500    | Internal Error   | Server error             |
  | 503    | Service Unavail. | Service temporarily down |
  +----------------------------------------------------+

  3. Pagination
  - Cursor-based (recommended): /api/v1/users?cursor=abc123&limit=20
  - Offset-based: /api/v1/users?page=3&per_page=20

  4. Filtering & Sorting
  - /api/v1/users?status=active&sort=created_at&order=desc
  - /api/v1/products?min_price=1000&max_price=5000&category=electronics

  5. Versioning
  - URL path: /api/v1/users (most common)
  - Header: Accept: application/vnd.myapi.v1+json
  - Query parameter: /api/users?version=1
```

### 8.2 API Response Design

```python
# --- Unified response format ---

# Success response
{
    "status": "success",
    "data": {
        "id": "user-001",
        "name": "Alice",
        "email": "alice@example.com",
        "created_at": "2024-01-15T10:30:00Z"
    }
}

# List response (with pagination)
{
    "status": "success",
    "data": [
        {"id": "user-001", "name": "Alice"},
        {"id": "user-002", "name": "Bob"}
    ],
    "pagination": {
        "total": 150,
        "page": 1,
        "per_page": 20,
        "total_pages": 8,
        "next_cursor": "eyJpZCI6InVzZXItMDIwIn0="
    }
}

# Error response
{
    "status": "error",
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "There are issues with the input values",
        "details": [
            {
                "field": "email",
                "message": "Email address format is invalid"
            },
            {
                "field": "password",
                "message": "Password must be at least 8 characters"
            }
        ]
    }
}
```

### 8.3 Rate Limiting

```python
# --- Rate limiting implementation patterns ---

# 1. Fixed Window Counter
import time

class FixedWindowRateLimiter:
    """Fixed window rate limiter"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, client_id: str) -> bool:
        current_window = int(time.time() / self.window_seconds)
        key = f"rate_limit:{client_id}:{current_window}"

        count = r.incr(key)
        if count == 1:
            r.expire(key, self.window_seconds)

        return count <= self.max_requests

# 2. Sliding Window Log
class SlidingWindowLogRateLimiter:
    """Sliding window log rate limiter"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        key = f"rate_limit:{client_id}"

        # Remove old entries
        r.zremrangebyscore(key, 0, window_start)

        # Count requests within current window
        count = r.zcard(key)

        if count < self.max_requests:
            r.zadd(key, {str(now): now})
            r.expire(key, self.window_seconds)
            return True

        return False

# 3. Token Bucket
class TokenBucketRateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum number of tokens in the bucket
            refill_rate: Tokens replenished per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate

    def is_allowed(self, client_id: str, tokens_needed: int = 1) -> bool:
        key = f"token_bucket:{client_id}"
        now = time.time()

        # Get current token count and last update time
        data = r.hgetall(key)
        if not data:
            # First time: Fill bucket and consume 1 token
            r.hset(key, mapping={
                "tokens": str(self.capacity - tokens_needed),
                "last_refill": str(now)
            })
            r.expire(key, 3600)
            return True

        current_tokens = float(data[b"tokens"])
        last_refill = float(data[b"last_refill"])

        # Replenish tokens based on elapsed time
        elapsed = now - last_refill
        new_tokens = min(
            self.capacity,
            current_tokens + elapsed * self.refill_rate
        )

        if new_tokens >= tokens_needed:
            r.hset(key, mapping={
                "tokens": str(new_tokens - tokens_needed),
                "last_refill": str(now)
            })
            return True

        return False

# Usage
limiter = TokenBucketRateLimiter(capacity=100, refill_rate=10)  # 100 requests/burst, 10 requests/sec
if limiter.is_allowed("user:123"):
    process_request()
else:
    return {"error": "Rate limit exceeded"}, 429
```

---

## 9. Availability and Reliability

### 9.1 Calculating Availability

```
Availability:
  Targets are defined by SLA (Service Level Agreement)

  +------------------------------------------+
  | Availability  | Downtime/Year | Common Name |
  +------------------------------------------+
  | 99%           | 3.65 days     | Two nines   |
  | 99.9%         | 8.76 hours    | Three nines |
  | 99.95%        | 4.38 hours    |             |
  | 99.99%        | 52.6 minutes  | Four nines  |
  | 99.999%       | 5.26 minutes  | Five nines  |
  +------------------------------------------+

  Serial configuration availability:
  A -> B -> C
  Overall availability = A's availability x B's x C's
  Example: 99.9% x 99.9% x 99.9% = 99.7% (8.76h -> 26.3h downtime)

  Parallel configuration availability:
  A --> +
       +--> Output
  B --> +
  Overall availability = 1 - (1 - A's availability) x (1 - B's availability)
  Example: 1 - (0.001 x 0.001) = 99.9999% (31.5 seconds downtime)
```

### 9.2 Reliability Patterns

```
Patterns to improve reliability:

  1. Retry
     - Recovery from transient failures
     - Increase retry intervals with exponential backoff
     - Set a maximum retry count

  2. Circuit Breaker
     +----------+ Failures  +----------+ Timeout   +----------+
     |  Closed  | increase  |   Open   | expires   |Half-Open |
     | (normal) | --------> | (blocked)| --------->| (testing)|
     +----------+           +----------+           +-----+----+
          ^                                              |
          +---------- Success -------------------------+

  3. Bulkhead
     - Isolate resources to prevent failure propagation
     - Examples: Thread pool isolation, connection pool isolation

  4. Timeout
     - Set timeouts for all network calls
     - Do not rely on default timeouts

  5. Fallback
     - Alternative when the primary process fails
     - Serve data from cache
     - Return default values
     - Graceful degradation
```

```python
# --- Retry with Exponential Backoff ---
import time
import random

def retry_with_exponential_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
):
    """Retry with exponential backoff"""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except (ConnectionError, TimeoutError) as e:
            if attempt == max_retries:
                raise  # Last retry also failed -> raise exception

            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())  # 0.5x-1.5x jitter

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

# Usage example
result = retry_with_exponential_backoff(
    lambda: external_api.fetch_data("user-001"),
    max_retries=3,
    base_delay=1.0
)


# --- Circuit Breaker Implementation ---
import time
import threading
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"        # Normal: pass requests through
    OPEN = "open"            # Blocked: immediately reject requests
    HALF_OPEN = "half_open"  # Testing: allow only 1 request through

class CircuitBreaker:
    """Circuit Breaker implementation"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_try_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker is OPEN. Request rejected."
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            else:
                self.failure_count = 0

    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

    def _should_try_reset(self):
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
```

---

## 10. Back-of-the-Envelope Estimation

### 10.1 Latency Fundamentals

```
Approximate latency for various operations:

  +--------------------------------------------------+
  | Operation                      | Latency          |
  +--------------------------------------------------+
  | L1 cache reference             | 0.5 ns           |
  | L2 cache reference             | 7 ns             |
  | Main memory reference          | 100 ns           |
  | SSD random read                | 150 us           |
  | HDD random read                | 10 ms            |
  | Same-DC round trip             | 0.5 ms           |
  | Redis GET                      | 0.1-1 ms         |
  | MySQL query (indexed)          | 1-10 ms          |
  | Tokyo-Osaka round trip         | 5-10 ms          |
  | Tokyo-US round trip            | 100-200 ms       |
  | TCP connection establishment   | 50-150 ms        |
  | TLS handshake                  | 100-300 ms       |
  +--------------------------------------------------+

  1 second = 1,000 ms = 1,000,000 us = 1,000,000,000 ns
```

### 10.2 Practical Estimation

```
System design estimation:

  SNS with 1 million DAU:
  - Peak QPS: 1M / 86400 x 3 ~ 35 QPS (writes)
  - Read QPS: 35 x 100 = 3,500 QPS
  - 1 web server: 1,000-10,000 QPS -> 1-4 servers suffice
  - DB can be handled with read replicas + caching

  100 million DAU:
  - Write QPS: 3,500
  - Read QPS: 350,000
  - Web servers: dozens
  - DB sharding required
  - CDN + Redis required

Storage estimation example:
  Chat app with 1 million DAU:
  - 50 messages per user per day
  - Average 200 bytes per message
  - Daily messages: 1M x 50 = 50 million
  - Daily data: 50M x 200B = 10 GB/day
  - Annual data: 10 GB x 365 = 3.65 TB/year
  - 5-year retention: ~18 TB (54 TB with replication)

Bandwidth estimation:
  Video streaming service:
  - 5 million DAU
  - Average viewing: 1 hour/day
  - Video bitrate: 5 Mbps
  - Peak concurrent viewers: DAU x 10% = 500,000
  - Peak bandwidth: 500K x 5 Mbps = 2.5 Tbps
  - Per CDN site after distribution: 2.5 Tbps / 10 = 250 Gbps

QPS calculation template:
  DAU x actions/day / 86400 = average QPS
  Average QPS x peak multiplier (2-5) = peak QPS
  Peak QPS x safety factor (1.5-2) = required capacity
```

### 10.3 Practical Estimation Example

```
Twitter-like service design estimation:

  Assumptions:
  - MAU: 300 million
  - DAU: 150 million (50% of MAU)
  - Posts per user: 2 tweets/day
  - Views per user: 200 tweets/day
  - 1 tweet: average 300 bytes (text)
  - Image attachment rate: 20%, average image size: 500KB

  QPS:
  - Write: 150M x 2 / 86400 ~ 3,500 QPS
  - Read: 150M x 200 / 86400 ~ 350,000 QPS
  - Peak read: 350,000 x 3 ~ 1,000,000 QPS (1 million QPS!)

  Storage:
  - Text: 150M x 2 x 300B = 90 GB/day -> 33 TB/year
  - Images: 150M x 2 x 0.2 x 500KB = 30 TB/day -> 10.8 PB/year
  - 5-year retention: Text 165TB + Images 54PB

  Design considerations:
  - Reads vastly outnumber writes -> Caching is most important
  - Timeline pre-computation (Fan-out on write) is effective
  - Images via CDN + object storage
  - Hot users (millions of followers) need special handling
```

---

## 11. Practical System Design Examples

### 11.1 URL Shortener Design

```
Requirements:
  - Convert long URLs to short URLs
  - Accessing a short URL redirects to the original URL
  - DAU: 10 million
  - Short URL generation: 100 million per day
  - Read/write ratio: 100:1

Design:

  API:
  - POST /api/v1/shorten {"url": "https://very-long-url.com/..."}
    -> {"short_url": "https://tny.io/a1b2c3"}
  - GET /a1b2c3 -> 301 Redirect

  Data Model:
  +-----------------------------------+
  | short_urls table                   |
  +-----------------------------------+
  | id (PK)          | BIGINT         |
  | short_key         | VARCHAR(7)     |
  | original_url      | TEXT           |
  | created_at        | TIMESTAMP      |
  | expires_at        | TIMESTAMP      |
  | click_count       | BIGINT         |
  +-----------------------------------+

  Short key generation methods:
  - Base62 encoding: [a-zA-Z0-9] = 62 characters
  - 7 characters: 62^7 = 3.5 trillion combinations (sufficient key space)
  - Method 1: Counter-based (convert ID to Base62)
  - Method 2: Hash-based (first 7 chars of MD5/SHA256)
  - Method 3: Pre-generated Key Generation Service

  Architecture:
  +--------+   +------+   +------+   +------+
  | Client |-->| LB   |-->| API  |-->| Cache|
  +--------+   +------+   |Server|   |Redis |
                           +--+---+   +--+---+
                              |          |
                           +--v----------v---+
                           |    Database      |
                           |   (Sharded)      |
                           +-----------------+

  Cache strategy:
  - Reads are 100x more frequent -> Cache-Aside with Redis
  - Popular URLs: TTL = 24 hours
  - Target overall cache hit rate: 80%+
```

### 11.2 Notification System Design

```
Requirements:
  - 3 channels: push notifications, SMS, email
  - 100 million notifications per day
  - Per-user notification settings (opt-in/out)
  - Delivery guarantee (at least once delivery)

Architecture:
  +----------+   +------------+
  | Service A |-->|            |
  | Service B |-->| Notification|--> +--------+ --> APNs
  | Service C |-->|  Service    |    | Queue  | --> FCM
  | Scheduler |-->|            |--> | (Kafka)| --> SMS Gateway
  +----------+   +------------+    +--------+ --> Email(SES)

  Processing flow:
  1. Service sends notification request
  2. Notification service receives it
     - Check user's notification settings
     - Apply rate limiting
     - Apply template
  3. Enqueue to channel-specific queue
  4. Workers send to each provider
  5. Log delivery results

  Considerations:
  - Idempotency: Deduplication to avoid sending the same notification twice
  - Priority: Urgent notifications processed via priority queue
  - Batch processing: Mass sends optimized via batching
  - Fallback: Push notification failure -> Fallback to email
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how things work.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in professional practice?

The knowledge from this topic is frequently applied in daily development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|-----------|
| Scaling | Vertical (enhance) vs Horizontal (add). Horizontal is mainstream |
| CAP Theorem | CP (consistency) vs AP (availability). Choose based on use case |
| Load Balancing | L4/L7, Consistent Hashing |
| Cache | Cache-Aside is most common. Stampede protection is essential |
| Database | RDB vs NoSQL selection. Sharding |
| Message Queue | Asynchronous processing. Loose coupling between services |
| Microservices | Start with Monolith First |
| API Design | RESTful principles, unified responses, rate limiting |
| Availability | Circuit Breaker, Retry, Fallback |
| Estimation | DAU -> QPS -> Number of servers estimation |

---

## Recommended Next Guides

---

## References
1. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly, 2017.
2. Alex Xu. "System Design Interview." 2020.
3. Alex Xu. "System Design Interview Vol. 2." 2022.
4. Newman, S. "Building Microservices." 2nd Edition, O'Reilly, 2021.
5. Richardson, C. "Microservices Patterns." Manning, 2018.
6. Nygard, M. "Release It!" 2nd Edition, Pragmatic Bookshelf, 2018.
7. Burns, B. "Designing Distributed Systems." O'Reilly, 2018.
8. Fowler, M. "Patterns of Enterprise Application Architecture." Addison-Wesley, 2002.
