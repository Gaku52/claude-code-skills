/**
 * CRDT type definitions
 * @packageDocumentation
 */

/**
 * Timestamp for Last-Write-Wins conflict resolution
 * @public
 */
export interface Timestamp {
  /** Unix timestamp in milliseconds */
  time: number;
  /** Unique replica identifier for tie-breaking */
  replicaId: string;
}

/**
 * Unique element with UUID for OR-Set
 * @public
 */
export interface UniqueElement<T> {
  /** The actual value */
  value: T;
  /** Globally unique identifier */
  uuid: string;
}

/**
 * Base interface for all CRDTs
 * @public
 */
export interface CRDT<T> {
  /**
   * Merges this CRDT with another replica
   * @param other - Another CRDT replica to merge with
   * @returns A new merged CRDT
   */
  merge(other: T): T;
}
