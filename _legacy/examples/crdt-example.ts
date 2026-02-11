/**
 * CRDT Example
 *
 * Demonstrates conflict-free replicated data types with
 * strong eventual consistency guarantees.
 */

import {
  GCounter,
  PNCounter,
  LWWElementSet,
  ORSet
} from '../packages/crdt/src/index.js';

console.log('='.repeat(60));
console.log('CRDT (Conflict-free Replicated Data Types) Example');
console.log('='.repeat(60));

// Example 1: G-Counter (Grow-only Counter)
console.log('\n## Example 1: G-Counter (Distributed Page Views)\n');

const pageViews1 = new GCounter();
const pageViews2 = new GCounter();

// Replica 1: US datacenter
pageViews1.increment('us-east-1');
pageViews1.increment('us-east-1');
pageViews1.increment('us-east-1');

// Replica 2: EU datacenter
pageViews2.increment('eu-west-1');
pageViews2.increment('eu-west-1');

console.log('Before merge:');
console.log(`  US datacenter:  ${pageViews1.value()} views`);
console.log(`  EU datacenter:  ${pageViews2.value()} views`);

// Merge replicas
const mergedViews = pageViews1.merge(pageViews2);
console.log(`After merge:    ${mergedViews.value()} views`);
console.log('✅ Strong eventual consistency: all replicas converge');

// Example 2: PN-Counter (Votes)
console.log('\n## Example 2: PN-Counter (Upvotes/Downvotes)\n');

const votes1 = new PNCounter();
const votes2 = new PNCounter();

// Replica 1: Some users vote
votes1.increment('user-1');  // upvote
votes1.increment('user-2');  // upvote
votes1.decrement('user-3');  // downvote

// Replica 2: Other users vote
votes2.increment('user-4');  // upvote
votes2.decrement('user-5');  // downvote
votes2.decrement('user-6');  // downvote

console.log('Before merge:');
console.log(`  Replica 1: ${votes1.value()} net votes`);
console.log(`  Replica 2: ${votes2.value()} net votes`);

const mergedVotes = votes1.merge(votes2);
console.log(`After merge:  ${mergedVotes.value()} net votes`);
console.log('✅ Convergence guaranteed by semilattice properties');

// Example 3: OR-Set (Shopping List)
console.log('\n## Example 3: OR-Set (Collaborative Shopping List)\n');

const aliceList = new ORSet<string>();
const bobList = new ORSet<string>();

// Alice adds items
console.log('Alice adds: milk, eggs, bread');
aliceList.add('milk');
aliceList.add('eggs');
aliceList.add('bread');

// Bob adds items (concurrent)
console.log('Bob adds: butter, eggs (concurrent)');
bobList.add('butter');
bobList.add('eggs');

// Alice removes bread
console.log('Alice removes: bread');
aliceList.remove('bread');

console.log('\nBefore merge:');
console.log(`  Alice's list: [${aliceList.values().join(', ')}]`);
console.log(`  Bob's list:   [${bobList.values().join(', ')}]`);

// Merge
const mergedList = aliceList.merge(bobList);
console.log(`After merge:    [${mergedList.values().join(', ')}]`);
console.log('✅ Add-wins semantics: concurrent adds are preserved');

// Example 4: LWW-Element-Set (User Presence)
console.log('\n## Example 4: LWW-Element-Set (User Online Status)\n');

const presence1 = new LWWElementSet<string>();
const presence2 = new LWWElementSet<string>();

// User joins at t=100
presence1.add('alice', { time: 100, replicaId: 'server-1' });
console.log('Alice joins at t=100');

// User leaves at t=200 (on different replica)
presence2.remove('alice', { time: 200, replicaId: 'server-2' });
console.log('Alice leaves at t=200');

// User rejoins at t=300
presence1.add('alice', { time: 300, replicaId: 'server-1' });
console.log('Alice rejoins at t=300');

console.log('\nBefore merge:');
console.log(`  Server 1: alice online = ${presence1.contains('alice')}`);
console.log(`  Server 2: alice online = ${presence2.contains('alice')}`);

const mergedPresence = presence1.merge(presence2);
console.log(`After merge:  alice online = ${mergedPresence.contains('alice')}`);
console.log('✅ Last-write-wins: most recent timestamp (t=300 add) prevails');

// Example 5: Demonstrating Convergence
console.log('\n## Example 5: Convergence Property\n');

const counter1 = new GCounter();
const counter2 = new GCounter();
const counter3 = new GCounter();

// All replicas increment independently
counter1.increment('r1');
counter2.increment('r2');
counter3.increment('r3');

// Different merge orders
const merge12_3 = counter1.merge(counter2).merge(counter3);
const merge13_2 = counter1.merge(counter3).merge(counter2);
const merge23_1 = counter2.merge(counter3).merge(counter1);

console.log('Three replicas, each increments once');
console.log(`Merge order (1,2),3: ${merge12_3.value()}`);
console.log(`Merge order (1,3),2: ${merge13_2.value()}`);
console.log(`Merge order (2,3),1: ${merge23_1.value()}`);
console.log('✅ Commutative & Associative: merge order doesn\'t matter');

console.log('\n' + '='.repeat(60));
console.log('Mathematical Guarantees:');
console.log('  • Associative: merge(merge(a,b),c) = merge(a,merge(b,c))');
console.log('  • Commutative: merge(a,b) = merge(b,a)');
console.log('  • Idempotent:  merge(a,a) = a');
console.log('  ⇒ Strong Eventual Consistency');
console.log('='.repeat(60));
