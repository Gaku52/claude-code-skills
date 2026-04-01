# Reactive Streams

> Reactive Streams is a pattern for declaratively processing "asynchronous data streams." Understand the foundations of RxJS Observables, backpressure, and event-driven architecture.

## What You Will Learn

- [ ] Understand how the Observable pattern works
- [ ] Grasp RxJS operators and pipelines
- [ ] Learn the concept of backpressure
- [ ] Understand the types and use cases of Subjects
- [ ] Master reactive patterns in Angular and React
- [ ] Acquire testing and debugging techniques


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [async/await](./02-async-await.md)

---

## 1. Promise vs Observable

```
Promise:
  -> Single value (resolves only once)
  -> Execution starts upon creation (eager)
  -> Cannot be cancelled

Observable:
  -> Multiple values (a stream that flows over time)
  -> Execution starts upon subscription (lazy)
  -> Can be cancelled (unsubscribe)

  Promise:    ──────●              (single value)
  Observable: ──●──●──●──●──│     (multiple values, with completion)
              ──●──●──✗              (terminated by error)

Use cases:
  Promise: API calls, DB queries (single result)
  Observable: WebSocket, user input, timers (continuous events)
```

### 1.1 Observable Lifecycle

```
Observable Lifecycle:

  Creation
    │
    ▼
  Subscription  ← subscribe() call
    │
    ▼
  Emission
    │  next(value)  ──→ Observer's next callback
    │  next(value)  ──→ Observer's next callback
    │  ...
    │
    ├── complete()  ──→ Observer's complete callback ──→ End
    │
    └── error(err)  ──→ Observer's error callback ──→ End

  Unsubscription  ← unsubscribe() call
    │
    ▼
  Teardown  ← Execution of teardown logic
```

### 1.2 Hot Observable vs Cold Observable

```
Cold Observable:
  -> A new data stream is created each time it is subscribed to
  -> Each subscriber receives an independent stream
  -> Examples: HTTP requests, file reads

  Subscriber A: ──1──2──3──4──│
  Subscriber B:    ──1──2──3──4──│  (independent stream)

Hot Observable:
  -> The data source continues emitting values regardless of subscriptions
  -> Subscribing mid-stream means past values are not received
  -> Examples: WebSocket, mouse events, stock price feeds

  Source:       ──1──2──3──4──5──6──│
  Subscriber A: ──1──2──3──4──5──6──│  (subscribed from the start)
  Subscriber B:       ──3──4──5──6──│  (subscribed mid-stream)
```

```typescript
// Cold Observable example
const cold$ = new Observable(subscriber => {
  // New random value each time it is subscribed to
  subscriber.next(Math.random());
  subscriber.complete();
});

cold$.subscribe(v => console.log('A:', v)); // A: 0.123...
cold$.subscribe(v => console.log('B:', v)); // B: 0.456... (different value)

// Hot Observable example (using Subject)
const hot$ = new Subject<number>();

hot$.subscribe(v => console.log('A:', v));
hot$.next(1); // A: 1
hot$.next(2); // A: 2

hot$.subscribe(v => console.log('B:', v));
hot$.next(3); // A: 3, B: 3 (B only receives values from 3 onward)
```

---

## 2. RxJS Basics

```typescript
import { Observable, of, from, interval, fromEvent } from 'rxjs';
import { map, filter, take, debounceTime, switchMap } from 'rxjs/operators';

// Creating Observables
const numbers$ = of(1, 2, 3, 4, 5);
const array$ = from([10, 20, 30]);
const timer$ = interval(1000); // 0, 1, 2, ... every second

// Pipeline (operator chain)
numbers$.pipe(
  filter(n => n % 2 === 0),  // Even numbers only
  map(n => n * 10),          // Multiply by 10
).subscribe(value => console.log(value)); // 20, 40

// Practical search box example
const searchInput = document.getElementById('search');
fromEvent(searchInput, 'input').pipe(
  debounceTime(300),                           // Wait for 300ms of inactivity
  map(event => (event.target as HTMLInputElement).value),
  filter(query => query.length >= 2),          // At least 2 characters
  switchMap(query => fetch(`/api/search?q=${query}`).then(r => r.json())),
  // switchMap: cancels the previous request when a new value arrives
).subscribe(results => {
  renderSearchResults(results);
});
```

### 2.1 Ways to Create Observables

```typescript
import {
  Observable, of, from, interval, timer, fromEvent,
  defer, range, EMPTY, NEVER, throwError,
  generate, iif
} from 'rxjs';
import { ajax } from 'rxjs/ajax';

// 1. Custom Observable
const custom$ = new Observable<number>(subscriber => {
  subscriber.next(1);
  subscriber.next(2);
  subscriber.next(3);
  setTimeout(() => {
    subscriber.next(4);
    subscriber.complete();
  }, 1000);

  // Teardown logic (executed on unsubscription)
  return () => {
    console.log('Cleanup processing');
  };
});

// 2. Static creation functions
const values$ = of('a', 'b', 'c');                    // Synchronously emits 3 values
const arr$ = from([1, 2, 3]);                          // Observable from array
const promise$ = from(fetch('/api/data'));              // Observable from Promise
const iter$ = from(new Map([['a', 1], ['b', 2]]));    // Observable from Iterable

// 3. Timer-based
const interval$ = interval(1000);                       // 0, 1, 2, ... (1-second intervals)
const timerOnce$ = timer(3000);                         // Emits 0 after 3 seconds
const timerRepeat$ = timer(0, 1000);                    // Starts immediately, 1-second intervals

// 4. Event-based
const clicks$ = fromEvent(document, 'click');
const resize$ = fromEvent(window, 'resize');
const keydown$ = fromEvent<KeyboardEvent>(document, 'keydown');

// 5. AJAX
const data$ = ajax.getJSON('/api/users');
const post$ = ajax({
  url: '/api/users',
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: { name: 'Tanaka', email: 'tanaka@example.com' },
});

// 6. Conditional branching
const source$ = iif(
  () => Math.random() > 0.5,
  of('heads'),
  of('tails'),
);

// 7. Deferred creation (Observable is created only at subscription time)
const deferred$ = defer(() => {
  const timestamp = Date.now();
  return of(timestamp);
});

// 8. Generator-style
const fib$ = generate(
  [0, 1],                           // Initial value
  ([a, b]) => a < 100,              // Condition
  ([a, b]) => [b, a + b] as [number, number],  // Update
  ([a, b]) => a,                     // Result selector
);
// -> 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
```

### 2.2 Observer Pattern Details

```typescript
// Observer interface
interface Observer<T> {
  next: (value: T) => void;
  error: (err: any) => void;
  complete: () => void;
}

// Passing a complete Observer
const subscription = numbers$.subscribe({
  next: value => console.log('Value:', value),
  error: err => console.error('Error:', err),
  complete: () => console.log('Complete'),
});

// Partial Observer (optional callbacks)
numbers$.subscribe(
  value => console.log(value),          // next only
);

numbers$.subscribe({
  next: value => console.log(value),
  error: err => console.error(err),     // complete omitted
});

// Subscription management
const sub = interval(1000).subscribe(v => console.log(v));

// Unsubscribe after 5 seconds
setTimeout(() => {
  sub.unsubscribe(); // Release resources
  console.log('Unsubscribed');
}, 5000);

// Managing multiple Subscriptions together
import { Subscription } from 'rxjs';

const parentSub = new Subscription();

parentSub.add(interval(1000).subscribe(v => console.log('A:', v)));
parentSub.add(interval(2000).subscribe(v => console.log('B:', v)));
parentSub.add(interval(3000).subscribe(v => console.log('C:', v)));

// Unsubscribe all at once
setTimeout(() => parentSub.unsubscribe(), 10000);
```

---

## 3. Key Operators

```
Transformation:
  map       — Transform values
  switchMap — Switch to a new Observable (cancels previous)
  mergeMap  — Execute Observables concurrently
  concatMap — Execute Observables sequentially
  exhaustMap — Ignore new Observables until the current one completes
  scan      — Calculate accumulated values (streaming version of reduce)
  pluck     — Extract a specific property from objects
  pairwise  — Pair the previous value with the current value

Filtering:
  filter       — Pass only values that match a condition
  take         — Take only the first N items
  takeUntil    — Take until another Observable emits
  takeWhile    — Take while the condition is true
  skip         — Skip the first N items
  debounceTime — Pass through after a period of inactivity
  throttleTime — Pass only one value per time interval
  distinctUntilChanged — Pass only when the value changes
  first        — First value (with optional condition)
  last         — Last value (with optional condition)
  elementAt    — Nth value

Combination:
  merge       — Merge multiple Observables
  combineLatest — Combine the latest values from each Observable
  zip         — Combine values from each Observable one-to-one
  forkJoin    — Get the last value from all Observables (similar to Promise.all)
  concat      — Concatenate sequentially (next starts after previous completes)
  race        — Adopt the Observable that emits first
  withLatestFrom — Attach the latest value from other streams when the main stream emits

Error:
  catchError  — Handle and recover from errors
  retry       — Retry on error
  retryWhen   — Conditional retry (deprecated in RxJS 7, merged into retry)

Utility:
  tap         — Side effects (debugging, logging)
  delay       — Delay value emission
  timeout     — Error if no value arrives within a specified time
  finalize    — Cleanup on completion/error/unsubscription
  share       — Convert Cold to Hot (multicast)
  shareReplay — Replay the latest N values
```

### 3.1 Transformation Operator Details

```typescript
import {
  of, from, interval, fromEvent, timer
} from 'rxjs';
import {
  map, switchMap, mergeMap, concatMap, exhaustMap,
  scan, pairwise, bufferTime, groupBy, toArray
} from 'rxjs/operators';

// === switchMap: Latest request only (search, autocomplete) ===
const search$ = fromEvent<Event>(searchInput, 'input').pipe(
  debounceTime(300),
  map(e => (e.target as HTMLInputElement).value),
  switchMap(query =>
    // Cancels the previous request when new input arrives
    fetch(`/api/search?q=${query}`).then(r => r.json())
  ),
);

// === mergeMap: Concurrent execution (bulk email sending, parallel file downloads) ===
const sendEmails$ = from(emailList).pipe(
  mergeMap(
    email => sendEmail(email),
    5,  // Limit max concurrency to 5
  ),
);

// === concatMap: Sequential execution (when order guarantee is required) ===
const uploadFiles$ = from(files).pipe(
  concatMap(file =>
    // Upload one at a time in order (next starts after previous completes)
    uploadFile(file)
  ),
);

// === exhaustMap: Preventing double submission (form submit button) ===
const submitForm$ = fromEvent(submitBtn, 'click').pipe(
  exhaustMap(() =>
    // Ignore new clicks while a request is in progress
    fetch('/api/submit', { method: 'POST', body: formData })
      .then(r => r.json())
  ),
);

// === scan: Accumulated computation (state management) ===
const actions$ = new Subject<{ type: string; payload: any }>();
const state$ = actions$.pipe(
  scan((state, action) => {
    switch (action.type) {
      case 'INCREMENT':
        return { ...state, count: state.count + 1 };
      case 'DECREMENT':
        return { ...state, count: state.count - 1 };
      case 'SET_NAME':
        return { ...state, name: action.payload };
      default:
        return state;
    }
  }, { count: 0, name: '' }),
);

// === pairwise: Compare with the previous value ===
const scrollPosition$ = fromEvent(window, 'scroll').pipe(
  map(() => window.scrollY),
  pairwise(),
  map(([prev, curr]) => ({
    direction: curr > prev ? 'down' : 'up',
    delta: Math.abs(curr - prev),
  })),
);

// === bufferTime: Batch processing at regular intervals ===
const events$ = fromEvent(document, 'mousemove').pipe(
  bufferTime(1000),  // Emit as an array of events every second
  filter(events => events.length > 0),
  map(events => ({
    count: events.length,
    avgX: events.reduce((sum, e: any) => sum + e.clientX, 0) / events.length,
  })),
);

// === groupBy: Group a stream ===
interface LogEntry {
  level: 'info' | 'warn' | 'error';
  message: string;
}

const logs$ = from<LogEntry[]>([
  { level: 'info', message: 'Started' },
  { level: 'error', message: 'Failed' },
  { level: 'info', message: 'Processing' },
  { level: 'warn', message: 'Slow query' },
  { level: 'error', message: 'Timeout' },
]);

logs$.pipe(
  groupBy(log => log.level),
  mergeMap(group$ =>
    group$.pipe(
      toArray(),
      map(entries => ({ level: group$.key, entries })),
    )
  ),
).subscribe(group => {
  console.log(`${group.level}: ${group.entries.length} entries`);
});
```

### 3.2 Flattening Operator Comparison

```
switchMap vs mergeMap vs concatMap vs exhaustMap:

Input:    ──A─────B─────C──│

switchMap (latest only):
  A: ──a1──a2──(cancelled)
  B:       ──b1──b2──(cancelled)
  C:             ──c1──c2──c3──│
  Output: ──a1──a2──b1──b2──c1──c2──c3──│

mergeMap (concurrent):
  A: ──a1──a2──a3──│
  B:       ──b1──b2──b3──│
  C:             ──c1──c2──c3──│
  Output: ──a1──a2──b1──a3──b2──c1──b3──c2──c3──│

concatMap (sequential):
  A: ──a1──a2──a3──│
  B:               ──b1──b2──b3──│
  C:                             ──c1──c2──c3──│
  Output: ──a1──a2──a3──b1──b2──b3──c1──c2──c3──│

exhaustMap (ignore while in progress):
  A: ──a1──a2──a3──│
  B:  (ignored)
  C:             ──c1──c2──c3──│
  Output: ──a1──a2──a3──c1──c2──c3──│

When to use:
  switchMap  -> Search, autocomplete (only the latest is needed)
  mergeMap   -> Parallel downloads (all results needed, order doesn't matter)
  concatMap  -> Sequential file processing (order guarantee required)
  exhaustMap -> Form submission (prevent double submission)
```

### 3.3 Combination Operator Details

```typescript
import {
  merge, combineLatest, zip, forkJoin, concat, race, withLatestFrom
} from 'rxjs';

// === merge: Merging multiple streams ===
const keyboard$ = fromEvent(document, 'keydown');
const mouse$ = fromEvent(document, 'click');
const touch$ = fromEvent(document, 'touchstart');

const userActivity$ = merge(keyboard$, mouse$, touch$).pipe(
  throttleTime(1000),
  tap(() => resetIdleTimer()),
);

// === combineLatest: Combine the latest values from each stream ===
const selectedCategory$ = new BehaviorSubject<string>('all');
const searchQuery$ = new BehaviorSubject<string>('');
const sortOrder$ = new BehaviorSubject<string>('newest');

const filteredProducts$ = combineLatest([
  selectedCategory$,
  searchQuery$,
  sortOrder$,
]).pipe(
  debounceTime(100),
  switchMap(([category, query, sort]) =>
    fetch(`/api/products?category=${category}&q=${query}&sort=${sort}`)
      .then(r => r.json())
  ),
);

// === zip: Combine one-to-one ===
const names$ = of('Alice', 'Bob', 'Charlie');
const ages$ = of(25, 30, 35);
const cities$ = of('Tokyo', 'Osaka', 'Kyoto');

zip(names$, ages$, cities$).pipe(
  map(([name, age, city]) => ({ name, age, city })),
).subscribe(person => console.log(person));
// { name: 'Alice', age: 25, city: 'Tokyo' }
// { name: 'Bob', age: 30, city: 'Osaka' }
// { name: 'Charlie', age: 35, city: 'Kyoto' }

// === forkJoin: Get the last value after all complete (equivalent to Promise.all) ===
const dashboardData$ = forkJoin({
  users: ajax.getJSON('/api/users'),
  orders: ajax.getJSON('/api/orders'),
  stats: ajax.getJSON('/api/stats'),
  notifications: ajax.getJSON('/api/notifications'),
}).pipe(
  catchError(err => {
    console.error('Failed to fetch dashboard data:', err);
    return of({ users: [], orders: [], stats: null, notifications: [] });
  }),
);

// === race: Adopt the fastest stream ===
const primary$ = ajax.getJSON('https://primary-api.com/data');
const fallback$ = ajax.getJSON('https://fallback-api.com/data');

const data$ = race(primary$, fallback$); // Use whichever responds first

// === withLatestFrom: Attach the latest value from other streams when the main stream emits ===
const saveButton$ = fromEvent(saveBtn, 'click');
const formValue$ = new BehaviorSubject(getFormValues());

saveButton$.pipe(
  withLatestFrom(formValue$),
  switchMap(([_, formData]) =>
    fetch('/api/save', { method: 'POST', body: JSON.stringify(formData) })
  ),
).subscribe();
```

---

## 4. Types of Subjects

```
Types and Characteristics of Subjects:

Subject (basic):
  Cannot receive values emitted before subscription
  A subscribes -> next(1) -> next(2) -> B subscribes -> next(3)
  A: 1, 2, 3
  B:       3

BehaviorSubject (retains the latest value):
  Immediately receives the latest value upon subscription. Requires an initial value
  A subscribes(initial value 0) -> next(1) -> next(2) -> B subscribes -> next(3)
  A: 0, 1, 2, 3
  B:          2, 3

ReplaySubject (replays N past values):
  Buffers a specified number of past values and sends them to new subscribers
  A subscribes -> next(1) -> next(2) -> next(3) -> B subscribes(replay=2)
  A: 1, 2, 3
  B:          2, 3  (latest 2 values are replayed)

AsyncSubject (last value only):
  Emits only the last value at complete()
  next(1) -> next(2) -> next(3) -> complete()
  A: 3
  B: 3 (receives the last value even if subscribed after complete)
```

```typescript
import { Subject, BehaviorSubject, ReplaySubject, AsyncSubject } from 'rxjs';

// === BehaviorSubject: Ideal for current state management ===
interface AppState {
  user: User | null;
  theme: 'light' | 'dark';
  language: string;
}

class StateService {
  private state$ = new BehaviorSubject<AppState>({
    user: null,
    theme: 'light',
    language: 'ja',
  });

  // Get the current state (synchronous)
  get currentState(): AppState {
    return this.state$.getValue();
  }

  // Get a stream of state
  select<K extends keyof AppState>(key: K): Observable<AppState[K]> {
    return this.state$.pipe(
      map(state => state[key]),
      distinctUntilChanged(),
    );
  }

  // Update state
  update(partial: Partial<AppState>): void {
    this.state$.next({
      ...this.currentState,
      ...partial,
    });
  }
}

const stateService = new StateService();

// Watch for theme changes
stateService.select('theme').subscribe(theme => {
  document.body.className = `theme-${theme}`;
});

// Watch for user information changes
stateService.select('user').subscribe(user => {
  if (user) {
    console.log(`Welcome, ${user.name}`);
  }
});

// === ReplaySubject: Retaining event history ===
class EventBus {
  private events$ = new ReplaySubject<AppEvent>(10); // Retain the latest 10 events

  emit(event: AppEvent): void {
    this.events$.next(event);
  }

  on(type: string): Observable<AppEvent> {
    return this.events$.pipe(
      filter(event => event.type === type),
    );
  }

  // Get events including past ones
  history(): Observable<AppEvent> {
    return this.events$.asObservable();
  }
}

// === AsyncSubject: Final result upon completion ===
class ConfigLoader {
  private config$ = new AsyncSubject<Config>();

  async load(): Promise<void> {
    try {
      const config = await fetch('/api/config').then(r => r.json());
      this.config$.next(config);
      this.config$.complete(); // The last value is emitted upon complete()
    } catch (err) {
      this.config$.error(err);
    }
  }

  getConfig(): Observable<Config> {
    return this.config$.asObservable();
  }
}
```

---

## 5. Practical Example: Real-time Dashboard

```typescript
import { combineLatest, timer, Subject, BehaviorSubject } from 'rxjs';
import {
  switchMap, catchError, retry, map, share,
  distinctUntilChanged, tap, takeUntil, startWith, scan
} from 'rxjs/operators';

// === Dashboard Service ===
class DashboardService {
  private destroy$ = new Subject<void>();
  private refreshTrigger$ = new BehaviorSubject<void>(undefined);

  // Stats (every 5 seconds + manual refresh)
  readonly stats$ = this.refreshTrigger$.pipe(
    switchMap(() => timer(0, 5000)),
    switchMap(() =>
      fetch('/api/stats')
        .then(r => r.json())
        .catch(() => ({ error: true }))
    ),
    retry({ count: 3, delay: 2000 }),
    catchError(() => of({ error: true, data: null })),
    share(), // Multicast (shared among multiple subscribers)
    takeUntil(this.destroy$),
  );

  // Alerts (every 10 seconds)
  readonly alerts$ = timer(0, 10000).pipe(
    switchMap(() =>
      fetch('/api/alerts')
        .then(r => r.json())
        .catch(() => [])
    ),
    catchError(() => of([])),
    share(),
    takeUntil(this.destroy$),
  );

  // Real-time updates via WebSocket
  readonly liveEvents$ = new Observable<ServerEvent>(subscriber => {
    const ws = new WebSocket('wss://api.example.com/events');

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        subscriber.next(data);
      } catch (err) {
        console.error('WebSocket parse error:', err);
      }
    };

    ws.onerror = (err) => subscriber.error(err);
    ws.onclose = () => subscriber.complete();

    return () => ws.close();
  }).pipe(
    retry({ count: 5, delay: (error, retryCount) => timer(Math.min(1000 * Math.pow(2, retryCount), 30000)) }),
    share(),
    takeUntil(this.destroy$),
  );

  // Integrated dashboard state
  readonly dashboardState$ = combineLatest([
    this.stats$.pipe(startWith(null)),
    this.alerts$.pipe(startWith([])),
    this.liveEvents$.pipe(
      scan((events: ServerEvent[], event) => [...events.slice(-50), event], []),
      startWith([]),
    ),
  ]).pipe(
    map(([stats, alerts, events]) => ({
      stats,
      alerts,
      events,
      lastUpdated: new Date(),
    })),
    distinctUntilChanged((prev, curr) =>
      JSON.stringify(prev) === JSON.stringify(curr)
    ),
  );

  refresh(): void {
    this.refreshTrigger$.next();
  }

  destroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }
}
```

---

## 6. Backpressure

```
Backpressure:
  -> Control when the Producer is faster than the Consumer

  Producer: ──●●●●●●●●●●──→ Fast
  Consumer: ──●───●───●───→ Slow
                         -> Memory overflow / latency accumulation

Strategies:
  1. Buffering: Temporarily queue values (limited by memory)
  2. Dropping: Discard old values (keep only the latest)
  3. Sampling: Get the latest value at regular intervals
  4. Throttling: Pass only one value per time interval
  5. Windowing: Group by time or count

Backpressure in RxJS:
  -> bufferTime: Batch processing at time intervals
  -> bufferCount: Batch processing by count
  -> throttleTime: Pass values at regular intervals
  -> sampleTime: Get the latest value at regular intervals
  -> auditTime: Pass the latest value after a specified time from when a value arrives
  -> debounceTime: Pass the latest value after waiting a specified time from when a value arrives
  -> window/windowTime: Group by time windows
```

### 6.1 Practical Backpressure Examples

```typescript
import {
  fromEvent, interval, Subject
} from 'rxjs';
import {
  bufferTime, bufferCount, throttleTime, sampleTime,
  auditTime, debounceTime, windowTime, mergeAll,
  tap, filter, map, scan
} from 'rxjs/operators';

// === Throttling mouse movement ===
const mouseMove$ = fromEvent<MouseEvent>(document, 'mousemove');

// throttleTime: Pass the first value, ignore for the specified duration
mouseMove$.pipe(
  throttleTime(16), // ~60fps
  map(e => ({ x: e.clientX, y: e.clientY })),
).subscribe(pos => updateCursor(pos));

// sampleTime: Get the latest value at regular intervals
mouseMove$.pipe(
  sampleTime(100), // Latest position every 100ms
  map(e => ({ x: e.clientX, y: e.clientY })),
).subscribe(pos => sendAnalytics(pos));

// auditTime: Pass the latest value after a specified time from when a value arrives
mouseMove$.pipe(
  auditTime(200),
).subscribe(e => updateTooltip(e));

// === Batch log sending ===
const logStream$ = new Subject<LogEntry>();

// Batch send every 100 entries or every 5 seconds
logStream$.pipe(
  bufferTime(5000, undefined, 100), // 5 seconds or 100 entries
  filter(batch => batch.length > 0),
).subscribe(async batch => {
  await fetch('/api/logs', {
    method: 'POST',
    body: JSON.stringify(batch),
  });
});

// === Metrics aggregation with windows ===
const requestStream$ = new Subject<{ endpoint: string; duration: number }>();

requestStream$.pipe(
  windowTime(60000), // 1-minute window
  mergeAll(),
  scan((acc, req) => ({
    count: acc.count + 1,
    totalDuration: acc.totalDuration + req.duration,
    maxDuration: Math.max(acc.maxDuration, req.duration),
  }), { count: 0, totalDuration: 0, maxDuration: 0 }),
).subscribe(metrics => {
  console.log(`Requests/min: ${metrics.count}`);
  console.log(`Avg duration: ${metrics.totalDuration / metrics.count}ms`);
  console.log(`Max duration: ${metrics.maxDuration}ms`);
});

// === Difference between debounceTime and throttleTime ===
//
// debounceTime(300):
//   Input:  ──a─b─c───────d─e──────│
//   Output: ──────────c─────────e──│
//   -> Emits after input settles down (ideal for search)
//
// throttleTime(300):
//   Input:  ──a─b─c───────d─e──────│
//   Output: ──a───────────d────────│
//   -> Passes the first value immediately, then waits (ideal for scroll)
//
// auditTime(300):
//   Input:  ──a─b─c───────d─e──────│
//   Output: ──────c───────────e────│
//   -> Passes the latest value after a specified time from when a value arrives
//
// sampleTime(300):
//   Input:  ──a─b─c───────d─e──────│
//   Output: ──b─────c──────e───────│
//   -> Samples the latest value at regular intervals
```

---

## 7. Error Handling

```typescript
import { of, throwError, timer, EMPTY, Observable } from 'rxjs';
import {
  catchError, retry, retryWhen, delay, take,
  tap, finalize, timeout, switchMap
} from 'rxjs/operators';

// === Basic error handling ===
const data$ = ajax.getJSON('/api/data').pipe(
  catchError(err => {
    console.error('API error:', err);
    return of({ fallback: true, data: [] }); // Fallback value
  }),
);

// === Retry strategies ===
// Simple retry
const withRetry$ = ajax.getJSON('/api/unstable').pipe(
  retry(3), // Retry 3 times (4 total attempts)
  catchError(err => {
    console.error('All retries failed:', err);
    return EMPTY;
  }),
);

// Retry with exponential backoff (RxJS 7+)
const withBackoff$ = ajax.getJSON('/api/unstable').pipe(
  retry({
    count: 5,
    delay: (error, retryCount) => {
      const delayMs = Math.min(1000 * Math.pow(2, retryCount - 1), 30000);
      console.log(`Retry ${retryCount}: after ${delayMs}ms`);
      return timer(delayMs);
    },
    resetOnSuccess: true,
  }),
  catchError(err => {
    notifyUser('Unable to connect to the service');
    return EMPTY;
  }),
);

// === Conditional retry ===
const smartRetry$ = ajax('/api/data').pipe(
  retry({
    count: 3,
    delay: (error, retryCount) => {
      // Do not retry 4xx errors
      if (error.status >= 400 && error.status < 500) {
        return throwError(() => error);
      }
      // Retry only 5xx errors
      return timer(1000 * retryCount);
    },
  }),
);

// === Timeout ===
const withTimeout$ = ajax.getJSON('/api/slow-endpoint').pipe(
  timeout({
    each: 5000, // Error if the interval between emitted values exceeds 5 seconds
    with: () => throwError(() => new Error('Request timeout')),
  }),
  catchError(err => {
    if (err.message === 'Request timeout') {
      return of({ timeout: true });
    }
    return throwError(() => err);
  }),
);

// === finalize: Cleanup ===
function loadData(): Observable<Data> {
  showLoadingSpinner();

  return ajax.getJSON<Data>('/api/data').pipe(
    retry(2),
    catchError(err => {
      showErrorNotification(err.message);
      return EMPTY;
    }),
    finalize(() => {
      hideLoadingSpinner(); // Executed regardless of success/failure/unsubscription
    }),
  );
}

// === Error classification and handling ===
class ApiService {
  request<T>(url: string): Observable<T> {
    return ajax.getJSON<T>(url).pipe(
      catchError(err => {
        switch (err.status) {
          case 401:
            this.authService.logout();
            return throwError(() => new UnauthorizedError());
          case 403:
            return throwError(() => new ForbiddenError());
          case 404:
            return throwError(() => new NotFoundError(url));
          case 429:
            // Rate limit: respect the Retry-After header
            const retryAfter = parseInt(err.response?.headers?.get('Retry-After') || '5');
            return timer(retryAfter * 1000).pipe(
              switchMap(() => this.request<T>(url)),
            );
          default:
            return throwError(() => new ApiError(err.message, err.status));
        }
      }),
    );
  }
}
```

---

## 8. Reactive Patterns in Angular

```typescript
// === Usage in Angular components ===
@Component({
  selector: 'app-user-list',
  template: `
    <input [formControl]="searchControl" placeholder="Search...">

    <div *ngIf="loading$ | async" class="spinner">Loading...</div>

    <ul>
      <li *ngFor="let user of users$ | async; trackBy: trackById">
        {{ user.name }} - {{ user.email }}
      </li>
    </ul>

    <div *ngIf="error$ | async as error" class="error">
      {{ error.message }}
    </div>
  `,
})
export class UserListComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  searchControl = new FormControl('');
  loading$ = new BehaviorSubject<boolean>(false);
  error$ = new BehaviorSubject<Error | null>(null);

  users$: Observable<User[]>;

  constructor(private userService: UserService) {}

  ngOnInit(): void {
    this.users$ = this.searchControl.valueChanges.pipe(
      startWith(''),
      debounceTime(300),
      distinctUntilChanged(),
      tap(() => {
        this.loading$.next(true);
        this.error$.next(null);
      }),
      switchMap(query =>
        this.userService.searchUsers(query).pipe(
          catchError(err => {
            this.error$.next(err);
            return of([]);
          }),
          finalize(() => this.loading$.next(false)),
        )
      ),
      takeUntil(this.destroy$),
    );
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  trackById(index: number, user: User): number {
    return user.id;
  }
}

// === Caching in Angular services ===
@Injectable({ providedIn: 'root' })
export class UserService {
  private cache$ = new Map<string, Observable<User>>();

  constructor(private http: HttpClient) {}

  getUser(id: string): Observable<User> {
    if (!this.cache$.has(id)) {
      this.cache$.set(id,
        this.http.get<User>(`/api/users/${id}`).pipe(
          shareReplay({ bufferSize: 1, refCount: true }),
          // refCount: true -> Discard the cache when subscriber count reaches 0
        )
      );
    }
    return this.cache$.get(id)!;
  }

  searchUsers(query: string): Observable<User[]> {
    return this.http.get<User[]>(`/api/users`, {
      params: { q: query },
    });
  }

  // Reactive CRUD
  private refresh$ = new Subject<void>();

  users$ = this.refresh$.pipe(
    startWith(undefined),
    switchMap(() => this.http.get<User[]>('/api/users')),
    shareReplay(1),
  );

  createUser(user: CreateUserDto): Observable<User> {
    return this.http.post<User>('/api/users', user).pipe(
      tap(() => this.refresh$.next()), // Refresh after creation
    );
  }
}
```

---

## 9. Reactive Patterns in React

```typescript
import { useEffect, useState, useRef, useMemo } from 'react';
import { Subject, BehaviorSubject, Observable, Subscription } from 'rxjs';
import { debounceTime, distinctUntilChanged, switchMap, catchError } from 'rxjs/operators';

// === Custom hook: useObservable ===
function useObservable<T>(observable$: Observable<T>, initialValue: T): T {
  const [value, setValue] = useState<T>(initialValue);

  useEffect(() => {
    const subscription = observable$.subscribe({
      next: setValue,
      error: err => console.error('Observable error:', err),
    });
    return () => subscription.unsubscribe();
  }, [observable$]);

  return value;
}

// === Custom hook: useSubject ===
function useSubject<T>(): [Subject<T>, (value: T) => void] {
  const subjectRef = useRef<Subject<T>>();
  if (!subjectRef.current) {
    subjectRef.current = new Subject<T>();
  }

  const emit = useMemo(
    () => (value: T) => subjectRef.current!.next(value),
    [],
  );

  useEffect(() => {
    return () => subjectRef.current!.complete();
  }, []);

  return [subjectRef.current, emit];
}

// === Search component ===
function SearchComponent() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const searchSubject = useRef(new Subject<string>());

  useEffect(() => {
    const subscription = searchSubject.current.pipe(
      debounceTime(300),
      distinctUntilChanged(),
      tap(() => {
        setLoading(true);
        setError(null);
      }),
      switchMap(query =>
        from(fetch(`/api/search?q=${query}`).then(r => r.json())).pipe(
          catchError(err => {
            setError(err.message);
            return of([]);
          }),
        )
      ),
      tap(() => setLoading(false)),
    ).subscribe(setResults);

    return () => subscription.unsubscribe();
  }, []);

  return (
    <div>
      <input
        type="text"
        onChange={e => searchSubject.current.next(e.target.value)}
        placeholder="Search..."
      />
      {loading && <div className="spinner">Searching...</div>}
      {error && <div className="error">{error}</div>}
      <ul>
        {results.map(r => (
          <li key={r.id}>{r.title}</li>
        ))}
      </ul>
    </div>
  );
}

// === WebSocket hook ===
function useWebSocket<T>(url: string): {
  messages$: Observable<T>;
  send: (data: any) => void;
  status: 'connecting' | 'connected' | 'disconnected';
} {
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const wsRef = useRef<WebSocket | null>(null);
  const messages$ = useMemo(() => new Subject<T>(), []);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setStatus('connected');
    ws.onclose = () => setStatus('disconnected');
    ws.onmessage = (event) => {
      try {
        messages$.next(JSON.parse(event.data));
      } catch (err) {
        console.error('Parse error:', err);
      }
    };

    return () => {
      ws.close();
      messages$.complete();
    };
  }, [url]);

  const send = (data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  };

  return { messages$: messages$.asObservable(), send, status };
}
```

---

## 10. Testing and Debugging

```typescript
import { TestScheduler } from 'rxjs/testing';
import { map, filter, delay, debounceTime, switchMap } from 'rxjs/operators';

// === Marble Testing ===
describe('RxJS operator tests', () => {
  let scheduler: TestScheduler;

  beforeEach(() => {
    scheduler = new TestScheduler((actual, expected) => {
      expect(actual).toEqual(expected);
    });
  });

  // Testing the map operator
  it('multiplies values by 10', () => {
    scheduler.run(({ cold, expectObservable }) => {
      const source$ = cold(' -a-b-c-|', { a: 1, b: 2, c: 3 });
      const expected = '     -a-b-c-|';
      const result$ = source$.pipe(map(x => x * 10));

      expectObservable(result$).toBe(expected, { a: 10, b: 20, c: 30 });
    });
  });

  // Testing the filter operator
  it('passes only even numbers', () => {
    scheduler.run(({ cold, expectObservable }) => {
      const source$ = cold(' -a-b-c-d-|', { a: 1, b: 2, c: 3, d: 4 });
      const expected = '     ---b---d-|';
      const result$ = source$.pipe(filter(x => x % 2 === 0));

      expectObservable(result$).toBe(expected, { b: 2, d: 4 });
    });
  });

  // Testing debounceTime
  it('300ms debounce works correctly', () => {
    scheduler.run(({ cold, expectObservable }) => {
      const source$ = cold(' -a--b-----c--|');
      const expected = '     ---- 300ms b 196ms c--|';
      // b is emitted 300ms after a, c is emitted after 300ms+ since b

      const result$ = source$.pipe(debounceTime(300));
      expectObservable(result$).toBe(expected);
    });
  });

  // Testing errors
  it('catches errors and returns a fallback value', () => {
    scheduler.run(({ cold, expectObservable }) => {
      const source$ = cold(' -a-b-#', { a: 1, b: 2 }, new Error('fail'));
      const expected = '     -a-b-(c|)';

      const result$ = source$.pipe(
        catchError(() => of(0)),
      );

      expectObservable(result$).toBe(expected, { a: 1, b: 2, c: 0 });
    });
  });
});

// === Marble Syntax Reference ===
// -  : 1 frame (10ms in virtual time)
// a  : Value emission (values are defined in object literals)
// |  : complete
// #  : error
// ^  : subscribe point (used in hot observables)
// !  : unsubscribe point
// () : Synchronous group (emitting multiple values in the same frame)

// === Debugging with tap ===
const debuggedStream$ = source$.pipe(
  tap({
    next: val => console.log('[DEBUG] next:', val),
    error: err => console.error('[DEBUG] error:', err),
    complete: () => console.log('[DEBUG] complete'),
    subscribe: () => console.log('[DEBUG] subscribe'),
    unsubscribe: () => console.log('[DEBUG] unsubscribe'),
    finalize: () => console.log('[DEBUG] finalize'),
  }),
  map(transform),
  tap(val => console.log('[DEBUG] after map:', val)),
);
```

---

## 11. Reactive Streams Specification (Java/Kotlin)

```
Reactive Streams Specification (JVM):

  Publisher<T>
    └── subscribe(Subscriber<T>)

  Subscriber<T>
    ├── onSubscribe(Subscription)
    ├── onNext(T)
    ├── onError(Throwable)
    └── onComplete()

  Subscription
    ├── request(long n)    ← The core of backpressure
    └── cancel()

  Processor<T, R>
    └── Publisher<R> + Subscriber<T>

Implementation libraries:
  -> Project Reactor (Spring WebFlux)
  -> RxJava 3
  -> Akka Streams
  -> Kotlin Coroutines Flow
```

```kotlin
// Kotlin Flow example (lightweight version of Reactive Streams)
import kotlinx.coroutines.flow.*

// Creating a Flow
fun fibonacci(): Flow<Long> = flow {
    var a = 0L
    var b = 1L
    while (true) {
        emit(a)
        val temp = a + b
        a = b
        b = temp
    }
}

// Usage
suspend fun main() {
    fibonacci()
        .take(10)
        .filter { it % 2 == 0L }
        .map { it * it }
        .collect { println(it) }
}

// StateFlow (equivalent to BehaviorSubject)
class UserViewModel : ViewModel() {
    private val _state = MutableStateFlow(UserState())
    val state: StateFlow<UserState> = _state.asStateFlow()

    fun loadUser(id: String) {
        viewModelScope.launch {
            _state.update { it.copy(loading = true) }
            try {
                val user = userRepository.getUser(id)
                _state.update { it.copy(user = user, loading = false) }
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message, loading = false) }
            }
        }
    }
}

// SharedFlow (equivalent to Subject)
class EventBus {
    private val _events = MutableSharedFlow<AppEvent>(
        replay = 0,
        extraBufferCapacity = 64,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val events: SharedFlow<AppEvent> = _events.asSharedFlow()

    suspend fun emit(event: AppEvent) {
        _events.emit(event)
    }
}
```

---

## 12. Performance Optimization

```typescript
// === Leverage multicasting with share / shareReplay ===

// BAD: Each subscribe triggers an independent HTTP request
const user$ = ajax.getJSON('/api/user/1');
user$.subscribe(u => updateHeader(u));   // Request 1
user$.subscribe(u => updateSidebar(u));  // Request 2 (wasteful)

// GOOD: Share results with shareReplay
const user$ = ajax.getJSON('/api/user/1').pipe(
  shareReplay({ bufferSize: 1, refCount: true }),
);
user$.subscribe(u => updateHeader(u));   // Request 1
user$.subscribe(u => updateSidebar(u));  // Retrieved from cache (no request)

// === Preventing memory leaks ===

// BAD: Subscribing without takeUntil
class MyComponent {
  ngOnInit() {
    interval(1000).subscribe(v => this.update(v));
    // Memory leak after component is destroyed
  }
}

// GOOD: Auto-unsubscribe with takeUntil
class MyComponent implements OnDestroy {
  private destroy$ = new Subject<void>();

  ngOnInit() {
    interval(1000).pipe(
      takeUntil(this.destroy$),
    ).subscribe(v => this.update(v));
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }
}

// === Avoid unnecessary recomputation ===
const expensive$ = source$.pipe(
  distinctUntilChanged(), // Only when the value changes
  map(computeExpensiveResult),
  shareReplay(1),         // Share results
);

// === Scheduling with observeOn / subscribeOn ===
import { asyncScheduler, asapScheduler, animationFrameScheduler } from 'rxjs';
import { observeOn, subscribeOn } from 'rxjs/operators';

// Render on animation frame
source$.pipe(
  observeOn(animationFrameScheduler), // Synchronize with requestAnimationFrame
).subscribe(value => {
  updateUI(value); // Smooth rendering at 60fps
});
```

---

## 13. When to Use

```
Observable is appropriate:
  ✓ WebSocket message streams
  ✓ User input (search, scroll, resize)
  ✓ Real-time data (stock prices, chat)
  ✓ Combining multiple event sources
  ✓ Angular HttpClient / Forms
  ✓ Complex async orchestration
  ✓ Scenarios requiring backpressure control

Promise/async-await is appropriate:
  ✓ One-off API calls
  ✓ DB queries
  ✓ File operations
  ✓ Simple async processing
  ✓ Node.js server-side processing

Signals (Angular/Solid/Preact) are appropriate:
  ✓ UI state management
  ✓ Derived value computation
  ✓ Synchronous reactivity

Guidelines:
  -> One-off -> Promise
  -> Stream -> Observable
  -> UI state -> Signals (if available)
  -> When in doubt -> Promise (simplicity first)
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Perform input data validation
- Implement proper error handling
- Also create test code

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main data processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Get processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation and add the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Delete by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient version: {slow_time:.4f}s")
    print(f"Efficient version:   {fast_time:.6f}s")
    print(f"Speedup factor: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be mindful of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks
---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping straight to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Promise | Observable |
|---------|---------|-----------|
| Number of values | 1 | 0 to infinite |
| Execution | Eager | Lazy |
| Cancellation | Not possible | Possible |
| Operators | Limited | Rich |
| Use case | One-off I/O | Streams |
| Backpressure | None | Supported |
| Multicast | N/A | share / Subject |

| Subject | Characteristic | Initial Value | Replay |
|---------|---------------|---------------|--------|
| Subject | Basic | None | None |
| BehaviorSubject | Retains latest value | Required | 1 |
| ReplaySubject | Replays N values | None | N |
| AsyncSubject | Last value only | None | On complete |

| Operator | Use Case | Cancellation | Order Guarantee |
|----------|----------|-------------|-----------------|
| switchMap | Search | Yes | Latest only |
| mergeMap | Concurrent processing | No | Unspecified |
| concatMap | Sequential processing | No | Guaranteed |
| exhaustMap | Prevent double submission | Ignored | First only |

---

## Recommended Next Guides

---

## References
1. RxJS Documentation. rxjs.dev.
2. Reactive Streams Specification. reactive-streams.org.
3. Ben Lesh. "RxJS: Observable, Observer, and Subscription." rxjs.dev.
4. Angular Documentation. "Observables in Angular." angular.dev.
5. Kotlin Documentation. "Asynchronous Flow." kotlinlang.org.
6. Erik Meijer. "Your Mouse is a Database." ACM Queue, 2012.
