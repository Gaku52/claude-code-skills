/**
 * PN-Counter (Positive-Negative Counter) CRDT implementation
 * @packageDocumentation
 */

import { GCounter } from './g-counter.js';
import type { CRDT } from './types.js';

/**
 * Positive-Negative Counter (PN-Counter)
 *
 * A state-based CRDT that implements a counter supporting both increment and decrement.
 * Built on top of two G-Counters: one for increments, one for decrements.
 *
 * @public
 *
 * @remarks
 * ## Mathematical Properties
 *
 * Inherits semilattice properties from G-Counter:
 * - State: S = ℕⁿ × ℕⁿ (two vectors of natural numbers)
 * - Value: value(P, N) = sum(P) - sum(N)
 * - Merge: (P₁, N₁) ⊔ (P₂, N₂) = (P₁ ⊔ P₂, N₁ ⊔ N₂)
 *
 * ## Convergence Proof
 *
 * Since both positive and negative counters are G-Counters:
 * 1. Each G-Counter independently converges
 * 2. Merge is componentwise: merge both P and N
 * 3. Value is deterministic function of (P, N)
 * 4. Therefore, all replicas converge to the same value
 *
 * ## Usage Example
 *
 * ```typescript
 * const counter1 = new PNCounter();
 * const counter2 = new PNCounter();
 *
 * // Replica 1: +2, -1
 * counter1.increment('replica-1');
 * counter1.increment('replica-1');
 * counter1.decrement('replica-1');
 *
 * // Replica 2: +1
 * counter2.increment('replica-2');
 *
 * // Merge
 * const merged = counter1.merge(counter2);
 * console.log(merged.value());  // (2+1) - 1 = 2
 * ```
 *
 * @see {@link https://hal.inria.fr/inria-00555588 | Shapiro et al. (2011) - CRDTs paper}
 */
export class PNCounter implements CRDT<PNCounter> {
  private positive: GCounter = new GCounter();
  private negative: GCounter = new GCounter();

  /**
   * Increments the counter for a specific replica
   *
   * @param replicaId - Unique identifier for the replica
   * @throws {Error} If replicaId is empty
   *
   * @remarks
   * Internally calls increment on the positive G-Counter.
   *
   * Time complexity: O(1)
   */
  increment(replicaId: string): void {
    this.positive.increment(replicaId);
  }

  /**
   * Decrements the counter for a specific replica
   *
   * @param replicaId - Unique identifier for the replica
   * @throws {Error} If replicaId is empty
   *
   * @remarks
   * Internally calls increment on the negative G-Counter.
   * The decrement is represented as an increment to the negative counter.
   *
   * Time complexity: O(1)
   */
  decrement(replicaId: string): void {
    this.negative.increment(replicaId);
  }

  /**
   * Gets the counter value (positive - negative)
   *
   * @returns Net counter value
   *
   * @remarks
   * Time complexity: O(n) where n is the total number of replicas
   */
  value(): number {
    return this.positive.value() - this.negative.value();
  }

  /**
   * Merges this counter with another replica
   *
   * @param other - Another PN-Counter to merge with
   * @returns A new merged PN-Counter
   *
   * @remarks
   * Merges both the positive and negative G-Counters independently.
   * Inherits all semilattice properties from G-Counter.
   *
   * Time complexity: O(n + m) where n and m are the number of replicas
   */
  merge(other: PNCounter): PNCounter {
    const merged = new PNCounter();
    merged.positive = this.positive.merge(other.positive);
    merged.negative = this.negative.merge(other.negative);
    return merged;
  }

  /**
   * Gets the raw state for debugging
   *
   * @returns Object with positive and negative counter states
   * @internal
   */
  getState(): {
    positive: ReadonlyMap<string, number>;
    negative: ReadonlyMap<string, number>;
  } {
    return {
      positive: this.positive.getState(),
      negative: this.negative.getState()
    };
  }
}
