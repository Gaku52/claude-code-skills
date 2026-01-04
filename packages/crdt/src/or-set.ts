/**
 * OR-Set (Observed-Remove Set) CRDT implementation
 * @packageDocumentation
 */

import type { CRDT, UniqueElement } from './types.js';

/**
 * Observed-Remove Set (OR-Set)
 *
 * A state-based CRDT that implements a set with add and remove operations.
 * Resolves add-remove conflicts by keeping the add (observed-remove semantics).
 *
 * @public
 *
 * @typeParam T - Type of elements in the set
 *
 * @remarks
 * ## Conflict Resolution
 *
 * Unlike LWW-Set which uses timestamps, OR-Set uses unique tags:
 * - Each add operation creates a unique (value, UUID) pair
 * - Remove only removes elements it has "observed" (has seen)
 * - Concurrent add-remove: add wins (add-wins bias)
 *
 * ## Mathematical Properties
 *
 * - **Add-wins semantics**: If add and remove are concurrent, element stays in set
 * - **Observed-remove**: Can only remove elements that were previously added
 * - **Tombstone set**: Tracks removed UUIDs to prevent resurrection
 *
 * ## Convergence Proof
 *
 * For any two replicas R₁ and R₂:
 * 1. Elements: set of (value, UUID) pairs
 * 2. Tombstones: set of removed UUIDs
 * 3. Merge: union of elements (excluding tombstones) + union of tombstones
 * 4. Set union is associative, commutative, and idempotent
 * 5. Therefore, all replicas converge to the same set
 *
 * ## Advantages over LWW-Set
 *
 * - No clock synchronization required
 * - Add-wins bias is often more intuitive than LWW
 * - Concurrent adds always preserved
 *
 * ## Disadvantages
 *
 * - Larger metadata (one UUID per add operation)
 * - Tombstones grow unbounded (unless GC is implemented)
 *
 * ## Usage Example
 *
 * ```typescript
 * const set1 = new ORSet<string>();
 * const set2 = new ORSet<string>();
 *
 * // Replica 1 adds "apple"
 * const uuid1 = set1.add('apple');
 *
 * // Replica 2 independently adds "apple"
 * const uuid2 = set2.add('apple');
 *
 * // Replica 1 removes its "apple" (only knows about uuid1)
 * set1.remove('apple');
 *
 * // Merge: Replica 2's "apple" (uuid2) survives
 * const merged = set1.merge(set2);
 * console.log(merged.contains('apple'));  // true (uuid2 still exists)
 * ```
 *
 * @see {@link https://hal.inria.fr/inria-00555588 | Shapiro et al. (2011) - CRDTs paper}
 */
export class ORSet<T> implements CRDT<ORSet<T>> {
  protected elements: Map<string, UniqueElement<T>> = new Map();
  protected tombstones: Set<string> = new Set();

  /**
   * Adds an element to the set with a unique identifier
   *
   * @param value - Value to add
   * @returns UUID for this add operation
   *
   * @remarks
   * Each add creates a unique (value, UUID) pair. This allows the same
   * value to be added multiple times with different UUIDs, enabling
   * concurrent adds to be distinguished.
   *
   * Time complexity: O(1)
   */
  add(value: T): string {
    const uuid = this.generateUUID();
    const element: UniqueElement<T> = { value, uuid };
    this.elements.set(uuid, element);
    return uuid;
  }

  /**
   * Removes an element from the set
   *
   * @param value - Value to remove
   *
   * @remarks
   * Removes ALL instances of the value that this replica has observed.
   * Adds their UUIDs to the tombstone set to prevent resurrection.
   *
   * Concurrent adds from other replicas will survive because they have
   * UUIDs that this replica hasn't observed yet.
   *
   * Time complexity: O(n) where n is the number of elements
   */
  remove(value: T): void {
    // Remove all elements with this value
    for (const [uuid, element] of this.elements) {
      if (element.value === value) {
        this.tombstones.add(uuid);
        this.elements.delete(uuid);
      }
    }
  }

  /**
   * Checks if a value is in the set
   *
   * @param value - Value to check
   * @returns true if at least one non-tombstoned instance exists
   *
   * @remarks
   * Time complexity: O(n) where n is the number of elements
   */
  contains(value: T): boolean {
    for (const element of this.elements.values()) {
      if (element.value === value) {
        return true;
      }
    }
    return false;
  }

  /**
   * Gets all values in the set
   *
   * @returns Array of values (may contain duplicates if same value added multiple times)
   *
   * @remarks
   * Time complexity: O(n) where n is the number of elements
   */
  values(): T[] {
    return Array.from(this.elements.values()).map(e => e.value);
  }

  /**
   * Gets unique values in the set
   *
   * @returns Array of unique values
   *
   * @remarks
   * Time complexity: O(n) where n is the number of elements
   */
  uniqueValues(): T[] {
    const unique = new Set<T>();
    for (const element of this.elements.values()) {
      unique.add(element.value);
    }
    return Array.from(unique);
  }

  /**
   * Gets the number of elements in the set (counting duplicates)
   *
   * @returns Count of all (value, UUID) pairs
   *
   * @remarks
   * Time complexity: O(1)
   */
  size(): number {
    return this.elements.size;
  }

  /**
   * Gets the number of unique values in the set
   *
   * @returns Count of distinct values
   *
   * @remarks
   * Time complexity: O(n)
   */
  uniqueSize(): number {
    return this.uniqueValues().length;
  }

  /**
   * Merges this set with another replica
   *
   * @param other - Another OR-Set to merge with
   * @returns A new merged OR-Set
   *
   * @remarks
   * Merge algorithm:
   * 1. Union of all elements (value, UUID pairs)
   * 2. Exclude elements whose UUIDs are in either tombstone set
   * 3. Union of tombstone sets
   *
   * This ensures:
   * - All concurrent adds are preserved
   * - Removed elements stay removed
   * - Associative, commutative, and idempotent
   *
   * Time complexity: O(n + m) where n and m are the number of elements
   */
  merge(other: ORSet<T>): ORSet<T> {
    const merged = new ORSet<T>();

    // Add all elements not in other's tombstones
    for (const [uuid, element] of this.elements) {
      if (!other.tombstones.has(uuid)) {
        merged.elements.set(uuid, element);
      }
    }

    // Add all elements from other not in this tombstones
    for (const [uuid, element] of other.elements) {
      if (!this.tombstones.has(uuid)) {
        merged.elements.set(uuid, element);
      }
    }

    // Union of tombstones
    merged.tombstones = new Set([
      ...this.tombstones,
      ...other.tombstones
    ]);

    return merged;
  }

  /**
   * Generates a globally unique identifier
   *
   * @returns UUID string
   * @internal
   *
   * @remarks
   * Uses a combination of timestamp, random number, and counter for uniqueness.
   * For production, consider using crypto.randomUUID() or a proper UUID library.
   */
  protected generateUUID(): string {
    // Simple UUID generation (for production, use crypto.randomUUID() or uuid library)
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 15);
    const counter = (this.elements.size).toString(36);
    return `${timestamp}-${random}-${counter}`;
  }

  /**
   * Gets internal state for debugging
   *
   * @returns Object with elements and tombstones
   * @internal
   */
  getState(): {
    elements: ReadonlyMap<string, UniqueElement<T>>;
    tombstones: ReadonlySet<string>;
  } {
    return {
      elements: new Map(this.elements),
      tombstones: new Set(this.tombstones)
    };
  }
}
