/**
 * Conflict-free Replicated Data Types (CRDTs)
 *
 * A comprehensive library of state-based CRDTs with mathematical proofs of
 * strong eventual consistency. Designed for distributed systems that require
 * high availability and partition tolerance (AP in CAP theorem).
 *
 * @packageDocumentation
 *
 * @remarks
 * ## What are CRDTs?
 *
 * CRDTs (Conflict-free Replicated Data Types) are data structures that can be
 * replicated across multiple nodes and updated concurrently without coordination.
 * They guarantee **strong eventual consistency**: all replicas converge to the
 * same state after receiving all updates.
 *
 * ## Mathematical Foundation
 *
 * All CRDTs in this library are based on join-semilattices:
 * - **Partial order**: ⊑ (less than or equal)
 * - **Merge operation**: ⊔ (least upper bound / join)
 * - **Properties**:
 *   - Associative: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)
 *   - Commutative: a ⊔ b = b ⊔ a
 *   - Idempotent: a ⊔ a = a
 *
 * ## Included CRDTs
 *
 * ### Counters
 * - **G-Counter**: Grow-only counter (increment only)
 * - **PN-Counter**: Positive-negative counter (increment and decrement)
 *
 * ### Sets
 * - **LWW-Element-Set**: Last-write-wins set (requires timestamps)
 * - **OR-Set**: Observed-remove set (add-wins, no timestamps needed)
 *
 * ## Usage Example
 *
 * ```typescript
 * import { GCounter, ORSet } from '@claude-code-skills/crdt';
 *
 * // Distributed counter
 * const counter1 = new GCounter();
 * const counter2 = new GCounter();
 *
 * counter1.increment('replica-1');
 * counter2.increment('replica-2');
 *
 * const merged = counter1.merge(counter2);
 * console.log(merged.value());  // 2
 *
 * // Distributed set with add-wins semantics
 * const set1 = new ORSet<string>();
 * const set2 = new ORSet<string>();
 *
 * set1.add('apple');
 * set2.add('banana');
 *
 * const mergedSet = set1.merge(set2);
 * console.log(mergedSet.values());  // ['apple', 'banana']
 * ```
 *
 * ## Convergence Guarantees
 *
 * All CRDTs in this library guarantee:
 * 1. **Eventual delivery**: All updates eventually reach all replicas
 * 2. **Convergence**: Replicas that have received the same updates have the same state
 * 3. **Termination**: Merge operations always terminate
 * 4. **Safety**: No updates are lost
 *
 * ## References
 *
 * - Shapiro, M., Preguiça, N., Baquero, C., & Zawirski, M. (2011).
 *   "Conflict-free Replicated Data Types"
 *   Technical Report, INRIA.
 *   https://hal.inria.fr/inria-00555588
 *
 * @see {@link https://github.com/Gaku52/claude-code-skills | GitHub Repository}
 */

// Type exports
export type { CRDT, Timestamp, UniqueElement } from './types.js';

// Counter CRDTs
export { GCounter } from './g-counter.js';
export { PNCounter } from './pn-counter.js';

// Set CRDTs
export { LWWElementSet } from './lww-set.js';
export { ORSet } from './or-set.js';
