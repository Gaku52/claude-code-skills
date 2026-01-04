/**
 * LWW-Element-Set (Last-Write-Wins Element Set) CRDT implementation
 * @packageDocumentation
 */

import type { CRDT, Timestamp } from './types.js';

/**
 * Last-Write-Wins Element Set (LWW-Element-Set)
 *
 * A state-based CRDT that implements a set with add and remove operations.
 * Conflicts are resolved using Last-Write-Wins strategy based on timestamps.
 *
 * @public
 *
 * @typeParam T - Type of elements in the set
 *
 * @remarks
 * ## Conflict Resolution
 *
 * Uses timestamps to resolve conflicts:
 * - Each add/remove has a timestamp (time, replicaId)
 * - For tie-breaking: if time₁ = time₂, use lexicographic order of replicaId
 * - Element is in set if: most recent operation is "add"
 *
 * ## Convergence Properties
 *
 * 1. **Deterministic timestamps**: (time, replicaId) pairs are totally ordered
 * 2. **Merge semantics**: For each element, keep the most recent timestamp for both add and remove
 * 3. **Membership rule**: element ∈ set ⟺ timestamp(add) > timestamp(remove)
 * 4. **Convergence**: All replicas converge to same set membership
 *
 * ## Limitations
 *
 * - Requires synchronized clocks (or logical clocks like Lamport timestamps)
 * - Removed elements' metadata grows unbounded (unless GC is implemented)
 * - Add-remove anomaly: if clocks are skewed, an older remove can override a newer add
 *
 * ## Usage Example
 *
 * ```typescript
 * const set1 = new LWWElementSet<string>();
 * const set2 = new LWWElementSet<string>();
 *
 * // Replica 1 adds "apple"
 * set1.add('apple', { time: 100, replicaId: 'r1' });
 *
 * // Replica 2 adds "banana", removes "apple"
 * set2.add('banana', { time: 200, replicaId: 'r2' });
 * set2.remove('apple', { time: 300, replicaId: 'r2' });
 *
 * // Merge
 * const merged = set1.merge(set2);
 * console.log(merged.contains('apple'));   // false (removed at t=300)
 * console.log(merged.contains('banana'));  // true
 * ```
 *
 * @see {@link https://hal.inria.fr/inria-00555588 | Shapiro et al. (2011) - CRDTs paper}
 */
export class LWWElementSet<T> implements CRDT<LWWElementSet<T>> {
  protected added: Map<T, Timestamp> = new Map();
  protected removed: Map<T, Timestamp> = new Map();

  /**
   * Adds an element to the set with a timestamp
   *
   * @param element - Element to add
   * @param timestamp - Timestamp for this add operation
   * @throws {Error} If timestamp is invalid
   *
   * @remarks
   * Only updates if the new timestamp is newer than the existing one.
   *
   * Time complexity: O(1)
   */
  add(element: T, timestamp: Timestamp): void {
    this.validateTimestamp(timestamp);

    const currentAdded = this.added.get(element);
    if (!currentAdded || this.isNewer(timestamp, currentAdded)) {
      this.added.set(element, timestamp);
    }
  }

  /**
   * Removes an element from the set with a timestamp
   *
   * @param element - Element to remove
   * @param timestamp - Timestamp for this remove operation
   * @throws {Error} If timestamp is invalid
   *
   * @remarks
   * Only updates if the new timestamp is newer than the existing one.
   *
   * Time complexity: O(1)
   */
  remove(element: T, timestamp: Timestamp): void {
    this.validateTimestamp(timestamp);

    const currentRemoved = this.removed.get(element);
    if (!currentRemoved || this.isNewer(timestamp, currentRemoved)) {
      this.removed.set(element, timestamp);
    }
  }

  /**
   * Checks if an element is in the set
   *
   * @param element - Element to check
   * @returns true if the element's most recent operation is "add"
   *
   * @remarks
   * An element is in the set if:
   * 1. It has been added, AND
   * 2. Either it hasn't been removed, OR the add is newer than the remove
   *
   * Time complexity: O(1)
   */
  contains(element: T): boolean {
    const addedTime = this.added.get(element);
    const removedTime = this.removed.get(element);

    if (!addedTime) return false;
    if (!removedTime) return true;

    // Element is in set if add is newer than remove
    return this.isNewer(addedTime, removedTime);
  }

  /**
   * Gets all elements currently in the set
   *
   * @returns Array of elements where add > remove
   *
   * @remarks
   * Time complexity: O(n) where n is the total number of elements ever added
   */
  values(): T[] {
    const result: T[] = [];
    for (const element of this.added.keys()) {
      if (this.contains(element)) {
        result.push(element);
      }
    }
    return result;
  }

  /**
   * Gets the number of elements in the set
   *
   * @returns Count of elements where add > remove
   *
   * @remarks
   * Time complexity: O(n) where n is the total number of elements ever added
   */
  size(): number {
    return this.values().length;
  }

  /**
   * Compares two timestamps
   *
   * @param t1 - First timestamp
   * @param t2 - Second timestamp
   * @returns true if t1 is newer than t2
   * @internal
   *
   * @remarks
   * Comparison order:
   * 1. Compare time values
   * 2. If equal, use lexicographic order of replicaId for tie-breaking
   */
  protected isNewer(t1: Timestamp, t2: Timestamp): boolean {
    if (t1.time > t2.time) return true;
    if (t1.time === t2.time) return t1.replicaId > t2.replicaId;
    return false;
  }

  /**
   * Validates timestamp structure
   *
   * @param timestamp - Timestamp to validate
   * @throws {Error} If timestamp is invalid
   * @internal
   */
  protected validateTimestamp(timestamp: Timestamp): void {
    if (typeof timestamp.time !== 'number' || timestamp.time < 0) {
      throw new Error('Timestamp time must be a non-negative number');
    }
    if (!timestamp.replicaId) {
      throw new Error('Timestamp replicaId cannot be empty');
    }
  }

  /**
   * Merges this set with another replica
   *
   * @param other - Another LWW-Element-Set to merge with
   * @returns A new merged LWW-Element-Set
   *
   * @remarks
   * For each element:
   * - Take the newer add timestamp
   * - Take the newer remove timestamp
   *
   * This ensures that all replicas converge to the same set membership.
   *
   * Time complexity: O(n + m) where n and m are the number of elements
   */
  merge(other: LWWElementSet<T>): LWWElementSet<T> {
    const merged = new LWWElementSet<T>();

    // Collect all elements
    const allElements = new Set([
      ...this.added.keys(),
      ...this.removed.keys(),
      ...other.added.keys(),
      ...other.removed.keys()
    ]);

    for (const element of allElements) {
      // Merge add timestamps
      const thisAdded = this.added.get(element);
      const otherAdded = other.added.get(element);
      if (thisAdded && otherAdded) {
        merged.added.set(
          element,
          this.isNewer(thisAdded, otherAdded) ? thisAdded : otherAdded
        );
      } else if (thisAdded) {
        merged.added.set(element, thisAdded);
      } else if (otherAdded) {
        merged.added.set(element, otherAdded);
      }

      // Merge remove timestamps
      const thisRemoved = this.removed.get(element);
      const otherRemoved = other.removed.get(element);
      if (thisRemoved && otherRemoved) {
        merged.removed.set(
          element,
          this.isNewer(thisRemoved, otherRemoved) ? thisRemoved : otherRemoved
        );
      } else if (thisRemoved) {
        merged.removed.set(element, thisRemoved);
      } else if (otherRemoved) {
        merged.removed.set(element, otherRemoved);
      }
    }

    return merged;
  }

  /**
   * Creates a timestamp for the current time
   *
   * @param replicaId - Unique identifier for this replica
   * @returns Timestamp with current time
   * @public
   *
   * @remarks
   * Uses Date.now() for wall-clock time. For production, consider using
   * logical clocks (Lamport or Vector clocks) for better causal ordering.
   */
  static now(replicaId: string): Timestamp {
    return { time: Date.now(), replicaId };
  }
}
