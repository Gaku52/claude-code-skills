# @claude-code-skills/crdt

Conflict-free Replicated Data Types (CRDTs) with mathematical proofs of strong eventual consistency.

## Features

- **G-Counter**: Grow-only counter (increment only)
- **PN-Counter**: Positive-negative counter (increment/decrement)
- **LWW-Element-Set**: Last-write-wins set with timestamps
- **OR-Set**: Observed-remove set with add-wins semantics
- **Strong Eventual Consistency**: Mathematical guarantees of convergence
- **Zero Dependencies**: Pure TypeScript implementation

## Installation

```bash
npm install @claude-code-skills/crdt
```

## Quick Start

### G-Counter (Grow-only Counter)

```typescript
import { GCounter } from '@claude-code-skills/crdt';

const counter1 = new GCounter();
const counter2 = new GCounter();

// Replica 1 increments twice
counter1.increment('replica-1');
counter1.increment('replica-1');

// Replica 2 increments once
counter2.increment('replica-2');

// Merge replicas
const merged = counter1.merge(counter2);
console.log(merged.value());  // 3
```

### PN-Counter (Increment/Decrement)

```typescript
import { PNCounter } from '@claude-code-skills/crdt';

const counter = new PNCounter();

counter.increment('r1');  // +1
counter.increment('r1');  // +2
counter.decrement('r1');  // +1

console.log(counter.value());  // 1
```

### OR-Set (Add-Wins Set)

```typescript
import { ORSet } from '@claude-code-skills/crdt';

const set1 = new ORSet<string>();
const set2 = new ORSet<string>();

// Concurrent adds
set1.add('apple');
set2.add('banana');

// Merge
const merged = set1.merge(set2);
console.log(merged.values());  // ['apple', 'banana']
```

### LWW-Element-Set (Timestamp-based Set)

```typescript
import { LWWElementSet } from '@claude-code-skills/crdt';

const set = new LWWElementSet<string>();

set.add('apple', { time: 100, replicaId: 'r1' });
set.remove('apple', { time: 200, replicaId: 'r1' });

console.log(set.contains('apple'));  // false (removed)

// Helper for current time
const timestamp = LWWElementSet.now('replica-1');
set.add('banana', timestamp);
```

## Mathematical Properties

All CRDTs in this library are based on **join-semilattices**:

- **Associative**: `merge(merge(a,b),c) = merge(a,merge(b,c))`
- **Commutative**: `merge(a,b) = merge(b,a)`
- **Idempotent**: `merge(a,a) = a`

These properties guarantee **strong eventual consistency**: all replicas that have received the same updates will converge to the same state.

## When to Use

### G-Counter
- Distributed metrics (view counts, downloads)
- Monotonically increasing values
- No need for decrements

### PN-Counter
- Counters that need both increment and decrement
- Inventory tracking (limited, see caveats)
- Vote tallying

### OR-Set
- Collaborative editing (shopping lists, todo lists)
- No clock synchronization available
- Add-wins conflict resolution desired

### LWW-Element-Set
- When timestamps are available
- Last-write-wins semantics acceptable
- Simpler than OR-Set

## Caveats

### All CRDTs
- Metadata grows over time (tombstones)
- Not suitable for strong consistency requirements
- Network partition tolerance comes with eventual consistency

### PN-Counter
- Cannot enforce minimum value (could go negative)
- Not suitable for inventory with strict non-negative constraint

### LWW-Element-Set
- Requires synchronized clocks
- Add-remove anomaly possible with clock skew

### OR-Set
- Larger metadata (one UUID per add)
- Unbounded tombstone growth

## Performance

| Operation | G-Counter | PN-Counter | LWW-Set | OR-Set |
|-----------|-----------|------------|---------|--------|
| Add/Increment | O(1) | O(1) | O(1) | O(1) |
| Remove | N/A | N/A | O(1) | O(n) |
| Contains | O(n) | O(n) | O(1) | O(n) |
| Value/Size | O(n) | O(n) | O(n) | O(n) |
| Merge | O(n+m) | O(n+m) | O(n+m) | O(n+m) |

where n and m are the number of elements/replicas.

## References

- Shapiro, M., et al. (2011). "Conflict-free Replicated Data Types". Technical Report, INRIA.

## License

MIT
