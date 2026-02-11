/**
 * G-Counter (Grow-only Counter) CRDT implementation
 * @packageDocumentation
 */

import type { CRDT } from './types.js';

/**
 * Grow-only Counter (G-Counter)
 *
 * A state-based CRDT that implements a counter that can only increment.
 * Guarantees strong eventual consistency with the following properties:
 *
 * - **Associative**: merge(merge(a,b),c) = merge(a,merge(b,c))
 * - **Commutative**: merge(a,b) = merge(b,a)
 * - **Idempotent**: merge(a,a) = a
 *
 * @public
 *
 * @remarks
 * ## Mathematical Properties
 *
 * Forms a join-semilattice:
 * - State space: S = ℕⁿ (n natural numbers, one per replica)
 * - Partial order: (a₁,...,aₙ) ⊑ (b₁,...,bₙ) ⟺ ∀i: aᵢ ≤ bᵢ
 * - Join (merge): (a₁,...,aₙ) ⊔ (b₁,...,bₙ) = (max(a₁,b₁),...,max(aₙ,bₙ))
 *
 * ## Convergence Proof
 *
 * For any two replicas R₁ and R₂:
 * 1. Each increment is monotonic: s ⊑ increment(s)
 * 2. Merge computes least upper bound: s₁ ⊔ s₂
 * 3. By semilattice properties, merge is associative, commutative, and idempotent
 * 4. Therefore, all replicas converge to the same state
 *
 * ## Usage Example
 *
 * ```typescript
 * const counter1 = new GCounter();
 * const counter2 = new GCounter();
 *
 * // Replica 1 increments
 * counter1.increment('replica-1');
 * counter1.increment('replica-1');
 *
 * // Replica 2 increments
 * counter2.increment('replica-2');
 *
 * // Merge replicas
 * const merged = counter1.merge(counter2);
 * console.log(merged.value());  // 3
 * ```
 *
 * @see {@link https://hal.inria.fr/inria-00555588 | Shapiro et al. (2011) - CRDTs paper}
 */
export class GCounter implements CRDT<GCounter> {
  private counters: Map<string, number> = new Map();

  /**
   * Increments the counter for a specific replica
   *
   * @param replicaId - Unique identifier for the replica
   * @throws {Error} If replicaId is empty
   *
   * @remarks
   * Each replica maintains its own counter. This ensures that concurrent
   * increments from different replicas never conflict.
   *
   * Time complexity: O(1)
   * Space complexity: O(1) per increment
   */
  increment(replicaId: string): void {
    if (!replicaId) {
      throw new Error('Replica ID cannot be empty');
    }

    const current = this.counters.get(replicaId) || 0;
    this.counters.set(replicaId, current + 1);
  }

  /**
   * Gets the total counter value across all replicas
   *
   * @returns Sum of all replica counters
   *
   * @remarks
   * Time complexity: O(n) where n is the number of replicas
   */
  value(): number {
    let sum = 0;
    for (const count of this.counters.values()) {
      sum += count;
    }
    return sum;
  }

  /**
   * Merges this counter with another replica
   *
   * @param other - Another G-Counter to merge with
   * @returns A new merged G-Counter
   *
   * @remarks
   * The merge operation is:
   * - **Associative**: merge(merge(a,b),c) = merge(a,merge(b,c))
   * - **Commutative**: merge(a,b) = merge(b,a)
   * - **Idempotent**: merge(a,a) = a
   *
   * For each replica, takes the maximum value. This ensures that
   * all increments are preserved and the result is deterministic.
   *
   * Time complexity: O(n + m) where n and m are the number of replicas
   * Space complexity: O(n + m)
   */
  merge(other: GCounter): GCounter {
    const merged = new GCounter();

    // Collect all replica IDs
    const allIds = new Set([
      ...this.counters.keys(),
      ...other.counters.keys()
    ]);

    // Take maximum value for each replica
    for (const id of allIds) {
      const thisCount = this.counters.get(id) || 0;
      const otherCount = other.counters.get(id) || 0;
      merged.counters.set(id, Math.max(thisCount, otherCount));
    }

    return merged;
  }

  /**
   * Gets the raw counter state for debugging
   *
   * @returns Map of replica IDs to their counter values
   * @internal
   */
  getState(): ReadonlyMap<string, number> {
    return new Map(this.counters);
  }

  /**
   * Compares this counter with another for partial ordering
   *
   * @param other - Another G-Counter to compare with
   * @returns true if this ⊑ other (this is less than or equal to other)
   *
   * @remarks
   * Returns true if for all replica IDs, this counter's value ≤ other's value.
   * This implements the partial order relation of the semilattice.
   */
  lessThanOrEqual(other: GCounter): boolean {
    // Check all IDs in this counter
    for (const [id, count] of this.counters) {
      const otherCount = other.counters.get(id) || 0;
      if (count > otherCount) {
        return false;
      }
    }

    // Check IDs only in other counter
    for (const [id, count] of other.counters) {
      if (!this.counters.has(id) && count > 0) {
        // Other has increments we don't have
        continue;
      }
    }

    return true;
  }
}
