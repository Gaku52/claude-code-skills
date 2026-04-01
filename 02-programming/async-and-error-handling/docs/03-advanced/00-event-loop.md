# The Event Loop

> The event loop is the heart of asynchronous processing in Node.js and browsers. By understanding microtasks, macrotasks, and execution order, you can accurately predict the behavior of asynchronous code.

## What You Will Learn in This Chapter

- [ ] Understand the mechanics and phases of the event loop
- [ ] Grasp the execution order of microtasks and macrotasks
- [ ] Learn best practices for avoiding event loop blocking
- [ ] Understand the differences between Node.js and browser event loops
- [ ] Master the use of Worker Threads / Web Workers
- [ ] Acquire performance measurement and debugging techniques


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Overview of the Event Loop

```
Node.js Event Loop (libuv-based):

  ┌──────────────────────────────────────┐
  │           Event Loop                 │
  │                                      │
  │   ┌─────────────────────┐            │
  │   │ timers              │ ← setTimeout, setInterval │
  │   └──────────┬──────────┘            │
  │   ┌──────────▼──────────┐            │
  │   │ pending callbacks   │ ← I/O callbacks │
  │   └──────────┬──────────┘            │
  │   ┌──────────▼──────────┐            │
  │   │ idle, prepare       │ ← Internal use │
  │   └──────────┬──────────┘            │
  │   ┌──────────▼──────────┐            │
  │   │ poll                │ ← Retrieve I/O events │
  │   └──────────┬──────────┘            │
  │   ┌──────────▼──────────┐            │
  │   │ check               │ ← setImmediate │
  │   └──────────┬──────────┘            │
  │   ┌──────────▼──────────┐            │
  │   │ close callbacks     │ ← close events │
  │   └──────────┬──────────┘            │
  │              └──→ Next loop          │
  └──────────────────────────────────────┘

  Between each phase:
    → Process the process.nextTick() queue
    → Process the Promise microtask queue
```

### 1.1 Details of Each Phase

```
timers phase:
  → Executes callbacks for setTimeout() and setInterval()
  → Minimum delay is 1ms (even if 0 is specified, it is rounded up to 1ms)
  → Timers execute "at least N ms later," not exactly N ms later
  → A large number of timers will consume time in this phase

pending callbacks phase:
  → Executes I/O callbacks deferred from the previous iteration
  → Callbacks for system operations such as TCP connection errors
  → Example: callbacks for ECONNREFUSED errors

idle, prepare phase:
  → Used internally by Node.js only
  → Not directly accessible from user code

poll phase (the most important):
  → Retrieves new I/O events and executes I/O callbacks
  → Processes fs.readFile, HTTP request responses, DB query results, etc.
  → May block in this phase (if there are no other tasks)
  → Block duration is capped at the nearest timer in the next timers phase

check phase:
  → Executes setImmediate() callbacks
  → Guaranteed to execute immediately after the poll phase
  → Inside I/O callbacks, executes before setTimeout(fn, 0)

close callbacks phase:
  → Handles close events such as socket.on('close', ...)
  → Used for cleanup processing
```

### 1.2 Overall Execution Flow

```
Node.js Process Startup
    │
    ▼
┌──────────────────────────────────────┐
│ 1. Module loading and compilation    │
│    → Resolve require() / import      │
│    → Synchronous execution of        │
│      top-level code                  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 2. Process the process.nextTick      │
│    queue                             │
│    → Process the microtask queue     │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 3. Event Loop Starts                 │
│    ┌─→ timers                        │
│    │   → nextTick + microtasks       │
│    │   pending callbacks             │
│    │   → nextTick + microtasks       │
│    │   idle, prepare                 │
│    │   → nextTick + microtasks       │
│    │   poll (I/O wait)               │
│    │   → nextTick + microtasks       │
│    │   check (setImmediate)          │
│    │   → nextTick + microtasks       │
│    │   close callbacks               │
│    │   → nextTick + microtasks       │
│    └─← Next iteration               │
└──────────────────────────────────────┘
                   │
                   ▼ (When there are no more tasks to process)
┌──────────────────────────────────────┐
│ 4. Process Exit                      │
│    → Emit 'exit' event              │
│    → process.exit()                  │
└──────────────────────────────────────┘
```

---

## 2. Microtasks vs Macrotasks

```
Microtasks (Priority: High):
  → Promise.then/catch/finally
  → queueMicrotask()
  → process.nextTick() (Node.js, highest priority)
  → MutationObserver (Browser)

Macrotasks (Priority: Low):
  → setTimeout / setInterval
  → setImmediate (Node.js)
  → I/O callbacks
  → UI rendering (Browser)
  → requestAnimationFrame (Browser, before rendering)
  → MessageChannel

Execution Order:
  1. The call stack becomes empty
  2. Process all microtask queues
  3. Process one macrotask
  4. → Return to step 2

Priority Order in Node.js:
  process.nextTick > Promise microtask > setImmediate > setTimeout
```

### 2.1 Basic Execution Order

```javascript
// Execution order quiz
console.log("1: Synchronous");

setTimeout(() => console.log("2: setTimeout"), 0);

Promise.resolve().then(() => console.log("3: Promise"));

queueMicrotask(() => console.log("4: queueMicrotask"));

console.log("5: Synchronous");

// Output:
// 1: Synchronous
// 5: Synchronous
// 3: Promise        ← Microtask
// 4: queueMicrotask ← Microtask
// 2: setTimeout     ← Macrotask
```

### 2.2 Nested Asynchronous Processing

```javascript
// A slightly more complex example
console.log("start");

setTimeout(() => {
  console.log("timeout 1");
  Promise.resolve().then(() => console.log("promise in timeout"));
}, 0);

Promise.resolve().then(() => {
  console.log("promise 1");
  setTimeout(() => console.log("timeout in promise"), 0);
});

setTimeout(() => console.log("timeout 2"), 0);

console.log("end");

// Output:
// start
// end
// promise 1          ← Microtask
// timeout 1          ← Macrotask 1
// promise in timeout ← Microtask within timeout 1
// timeout 2          ← Macrotask 2
// timeout in promise ← Macrotask within promise 1
```

### 2.3 process.nextTick vs Promise vs queueMicrotask

```javascript
// Priority order in Node.js
console.log("1: Synchronous");

process.nextTick(() => {
  console.log("2: nextTick");
});

Promise.resolve().then(() => {
  console.log("3: Promise");
});

queueMicrotask(() => {
  console.log("4: queueMicrotask");
});

setImmediate(() => {
  console.log("5: setImmediate");
});

setTimeout(() => {
  console.log("6: setTimeout");
}, 0);

console.log("7: Synchronous");

// Output:
// 1: Synchronous
// 7: Synchronous
// 2: nextTick           ← nextTick queue (highest priority)
// 3: Promise            ← Microtask queue
// 4: queueMicrotask     ← Microtask queue
// 5: setImmediate       ← check phase
// 6: setTimeout         ← timers phase
// Note: The order of setImmediate and setTimeout(,0) may vary depending on timing
```

### 2.4 Danger of Recursive nextTick Calls

```javascript
// Bad: nextTick starvation problem
// When nextTick is called recursively, the event loop cannot progress
function recursiveNextTick() {
  process.nextTick(() => {
    console.log("nextTick");
    recursiveNextTick(); // nextTick executes forever
  });
}
recursiveNextTick();
// setTimeout callbacks will never execute!

// Good: Use setImmediate (allows one event loop iteration)
function recursiveImmediate() {
  setImmediate(() => {
    console.log("immediate");
    recursiveImmediate(); // Other tasks have a chance to execute
  });
}
```

### 2.5 Advanced Execution Order Puzzles

```javascript
// Execution order with async/await
async function asyncA() {
  console.log("A1");
  await Promise.resolve();
  console.log("A2");
}

async function asyncB() {
  console.log("B1");
  await asyncA();
  console.log("B2");
}

console.log("start");

asyncB();

Promise.resolve().then(() => console.log("P1"));

console.log("end");

// Output:
// start
// B1
// A1        ← Synchronous part of asyncA
// end
// A2        ← After await (microtask)
// P1        ← Promise.then (microtask)
// B2        ← After await asyncA() (microtask)

// Key points:
// - The portion of an async function before await executes synchronously
// - await is internally converted to .then()
// - The continuation after each await is queued as a microtask
```

```javascript
// Promise chain execution order
Promise.resolve()
  .then(() => console.log("then 1"))
  .then(() => console.log("then 2"))
  .then(() => console.log("then 3"));

Promise.resolve()
  .then(() => console.log("then A"))
  .then(() => console.log("then B"))
  .then(() => console.log("then C"));

// Output:
// then 1  ← First stage of the first Promise chain
// then A  ← First stage of the second Promise chain
// then 2  ← Second stage of the first Promise chain
// then B  ← Second stage of the second Promise chain
// then 3  ← Third stage of the first Promise chain
// then C  ← Third stage of the second Promise chain

// Key point: .then() is added to the microtask queue one stage at a time.
// When the first .then() executes, the next .then() is added to the queue.
// This results in round-robin-like alternating execution.
```

---

## 3. Blocking the Event Loop

```
Operations that block the event loop:
  → Synchronous file I/O (fs.readFileSync)
  → Heavy computation (encryption, image processing)
  → Parsing large JSON (JSON.parse)
  → Exponential backtracking in regular expressions
  → Infinite loops / long-running loops
  → Synchronous HTTP requests
  → Sorting large arrays

Impact of blocking:
  → All asynchronous processing stops
  → HTTP requests become unresponsive
  → WebSocket messages are delayed
  → Timers become inaccurate
  → Health checks time out
  → Clients receive timeout errors

Countermeasures:
  1. Do not use synchronous APIs (use fs.readFile, not fs.readFileSync)
  2. Delegate CPU-intensive processing to Worker threads
  3. Split large loops (yield with setImmediate)
  4. Use streaming to split large data
  5. Validate regex safety (ReDoS prevention)
```

### 3.1 Detecting and Avoiding Blocking

```javascript
// Bad: Blocking
function processLargeArray(items) {
  for (const item of items) { // 1 million items
    heavyComputation(item);   // Event loop stops
  }
}

// Good: Batched execution (yield control to event loop with setImmediate)
async function processLargeArrayAsync(items, batchSize = 1000) {
  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    for (const item of batch) {
      heavyComputation(item);
    }
    // Yield control to the event loop between batches
    await new Promise(resolve => setImmediate(resolve));
  }
}

// Good: Parallel execution with Worker Threads
const { Worker } = require('worker_threads');
function runInWorker(data) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('./heavy-task.js', { workerData: data });
    worker.on('message', resolve);
    worker.on('error', reject);
  });
}
```

### 3.2 Streaming Large JSON

```javascript
const { createReadStream } = require('fs');
const { pipeline } = require('stream/promises');
const JSONStream = require('jsonstream2');

// Bad: Loading large JSON all at once (memory + blocking issues)
async function processLargeJsonBad(filePath) {
  const data = JSON.parse(await fs.readFile(filePath, 'utf8')); // 500MB → blocks
  for (const item of data) {
    await processItem(item);
  }
}

// Good: Streaming sequential processing
async function processLargeJsonGood(filePath) {
  const stream = createReadStream(filePath)
    .pipe(JSONStream.parse('*')); // Emit each array element one at a time

  for await (const item of stream) {
    await processItem(item);
  }
}

// Good: Streaming NDJSON (Newline Delimited JSON)
const readline = require('readline');

async function processNDJSON(filePath) {
  const rl = readline.createInterface({
    input: createReadStream(filePath),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    if (line.trim()) {
      const item = JSON.parse(line);
      await processItem(item);
    }
  }
}
```

### 3.3 Regex Backtracking Prevention (ReDoS)

```javascript
// Bad: Dangerous regex (exponential backtracking)
const dangerousRegex = /^(a+)+$/;
// Takes exponential time against "aaaaaaaaaaaaaaaaab"

// Bad: This is also a ReDoS vulnerability
const emailRegex = /^([a-zA-Z0-9]+\.)*[a-zA-Z0-9]+@[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*$/;

// Good: Writing safe regular expressions
// 1. Use specific character classes that avoid backtracking
const safeEmailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

// 2. Use the re2 library (a regex engine that does not backtrack)
const RE2 = require('re2');
const safeRegex = new RE2('^[a-z]+$');

// 3. Execute regex with a timeout
function safeRegexTest(regex, input, timeoutMs = 100) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(`
      const { parentPort, workerData } = require('worker_threads');
      const result = new RegExp(workerData.pattern).test(workerData.input);
      parentPort.postMessage(result);
    `, {
      eval: true,
      workerData: { pattern: regex.source, input },
    });

    const timeout = setTimeout(() => {
      worker.terminate();
      reject(new Error('Regex execution timed out'));
    }, timeoutMs);

    worker.on('message', result => {
      clearTimeout(timeout);
      resolve(result);
    });
  });
}
```

### 3.4 Monitoring the Event Loop

```javascript
// Measure event loop delay
function monitorEventLoop(thresholdMs = 100) {
  let lastTime = process.hrtime.bigint();

  setInterval(() => {
    const now = process.hrtime.bigint();
    const delta = Number(now - lastTime) / 1_000_000; // ns → ms
    const lag = delta - 1000; // Difference from the expected 1000ms

    if (lag > thresholdMs) {
      console.warn(`Event loop lag: ${lag.toFixed(1)}ms`);
    }

    lastTime = now;
  }, 1000);
}

// Precise measurement using perf_hooks
const { monitorEventLoopDelay } = require('perf_hooks');

const histogram = monitorEventLoopDelay({ resolution: 20 });
histogram.enable();

// Periodically output statistics
setInterval(() => {
  console.log({
    min: histogram.min / 1e6,      // ns → ms
    max: histogram.max / 1e6,
    mean: histogram.mean / 1e6,
    p50: histogram.percentile(50) / 1e6,
    p95: histogram.percentile(95) / 1e6,
    p99: histogram.percentile(99) / 1e6,
  });
  histogram.reset();
}, 10000);

// Expose as Prometheus metrics
const { collectDefaultMetrics, register, Histogram } = require('prom-client');

collectDefaultMetrics(); // Default metrics include event loop delay

const eventLoopLag = new Histogram({
  name: 'nodejs_eventloop_lag_seconds',
  help: 'Lag of event loop in seconds',
  buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
});

// Health check endpoint
app.get('/health', (req, res) => {
  const h = monitorEventLoopDelay({ resolution: 20 });
  h.enable();
  setTimeout(() => {
    h.disable();
    const p99 = h.percentile(99) / 1e6;
    if (p99 > 500) {
      res.status(503).json({ status: 'unhealthy', eventLoopLag: p99 });
    } else {
      res.status(200).json({ status: 'healthy', eventLoopLag: p99 });
    }
  }, 1000);
});
```

---

## 4. Worker Threads (Node.js)

```javascript
// === Main Thread ===
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

// Worker Pool implementation
class WorkerPool {
  constructor(workerPath, numWorkers) {
    this.workerPath = workerPath;
    this.workers = [];
    this.freeWorkers = [];
    this.taskQueue = [];

    for (let i = 0; i < numWorkers; i++) {
      this.addWorker();
    }
  }

  addWorker() {
    const worker = new Worker(this.workerPath);
    worker.on('message', (result) => {
      // Resolve the task's Promise
      worker.currentResolve(result);
      worker.currentResolve = null;

      // If there are tasks in the queue, execute the next one
      if (this.taskQueue.length > 0) {
        const { data, resolve, reject } = this.taskQueue.shift();
        this.runTask(worker, data, resolve, reject);
      } else {
        this.freeWorkers.push(worker);
      }
    });

    worker.on('error', (err) => {
      if (worker.currentReject) {
        worker.currentReject(err);
      }
    });

    this.workers.push(worker);
    this.freeWorkers.push(worker);
  }

  runTask(worker, data, resolve, reject) {
    worker.currentResolve = resolve;
    worker.currentReject = reject;
    worker.postMessage(data);
  }

  execute(data) {
    return new Promise((resolve, reject) => {
      if (this.freeWorkers.length > 0) {
        const worker = this.freeWorkers.pop();
        this.runTask(worker, data, resolve, reject);
      } else {
        this.taskQueue.push({ data, resolve, reject });
      }
    });
  }

  async shutdown() {
    for (const worker of this.workers) {
      await worker.terminate();
    }
  }
}

// Usage example
const pool = new WorkerPool('./crypto-worker.js', 4); // 4 workers

// Compute hashes concurrently
async function hashPasswords(passwords) {
  const results = await Promise.all(
    passwords.map(pw => pool.execute({ password: pw }))
  );
  return results;
}

// === Worker Thread (crypto-worker.js) ===
const { parentPort } = require('worker_threads');
const crypto = require('crypto');

parentPort.on('message', ({ password }) => {
  // Execute CPU-intensive processing in the worker
  const hash = crypto.pbkdf2Sync(password, 'salt', 100000, 64, 'sha512');
  parentPort.postMessage(hash.toString('hex'));
});
```

### 4.1 Shared Memory with SharedArrayBuffer

```javascript
// Main thread
const { Worker } = require('worker_threads');

// Shared memory buffer (accessible from all workers)
const sharedBuffer = new SharedArrayBuffer(1024 * Int32Array.BYTES_PER_ELEMENT);
const sharedArray = new Int32Array(sharedBuffer);

// Multiple workers write to shared memory
const workers = [];
for (let i = 0; i < 4; i++) {
  const worker = new Worker('./shared-worker.js', {
    workerData: { buffer: sharedBuffer, workerId: i },
  });
  workers.push(worker);
}

// === shared-worker.js ===
const { parentPort, workerData } = require('worker_threads');
const { buffer, workerId } = workerData;
const sharedArray = new Int32Array(buffer);

// Thread-safe operations with Atomics
Atomics.add(sharedArray, 0, 1); // Atomic addition

// Inter-thread synchronization with Atomics.wait / Atomics.notify
Atomics.wait(sharedArray, 1, 0); // Wait while sharedArray[1] is 0
// ... Another thread wakes it with Atomics.notify(sharedArray, 1)

parentPort.postMessage({ done: true, workerId });
```

---

## 5. The Browser Event Loop

```
Browser Event Loop:

  ┌──────────────────────────────────┐
  │ 1. Execute one macrotask         │
  │ 2. Execute all microtasks        │
  │ 3. Render (if necessary)         │
  │    → requestAnimationFrame       │
  │    → Style calculation           │
  │    → Layout                      │
  │    → Paint                       │
  │ 4. → Return to step 1           │
  └──────────────────────────────────┘

  Important: If there are a large number of microtasks
  → Rendering is delayed
  → The UI appears to freeze

requestAnimationFrame:
  → Executes before the next render
  → Optimal for animations (60fps = 16.6ms interval)
  → A separate queue, neither microtask nor macrotask
```

### 5.1 requestAnimationFrame in Detail

```javascript
// requestAnimationFrame executes before rendering
console.log("1: Synchronous");

requestAnimationFrame(() => console.log("2: rAF"));

setTimeout(() => console.log("3: setTimeout"), 0);

Promise.resolve().then(() => console.log("4: Promise"));

console.log("5: Synchronous");

// Output:
// 1: Synchronous
// 5: Synchronous
// 4: Promise         ← Microtask
// 2: rAF             ← Before rendering (usually before setTimeout)
// 3: setTimeout      ← Macrotask
// Note: The order of rAF and setTimeout may differ depending on browser implementation

// === Smooth Animation ===
function animate(element, targetX, duration) {
  const startX = element.offsetLeft;
  const startTime = performance.now();

  function frame(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);

    // Easing function
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic

    element.style.left = startX + (targetX - startX) * eased + 'px';

    if (progress < 1) {
      requestAnimationFrame(frame);
    }
  }

  requestAnimationFrame(frame);
}

// === requestIdleCallback (low-priority tasks) ===
// Executes when the browser is in an idle state
function processNonUrgentWork(tasks) {
  function doWork(deadline) {
    // Check remaining time in the frame with deadline.timeRemaining()
    while (tasks.length > 0 && deadline.timeRemaining() > 1) {
      const task = tasks.shift();
      task();
    }

    if (tasks.length > 0) {
      requestIdleCallback(doWork);
    }
  }

  requestIdleCallback(doWork, { timeout: 5000 }); // Wait up to 5 seconds
}

// Usage example: Sending analytics
processNonUrgentWork([
  () => sendAnalytics('page_view', { path: location.pathname }),
  () => preloadImages(nextPageImages),
  () => prefetchData('/api/next-page'),
]);
```

### 5.2 Web Workers

```javascript
// === Main Thread ===
const worker = new Worker('worker.js');

// Sending and receiving messages
worker.postMessage({ type: 'process', data: largeDataset });

worker.onmessage = (event) => {
  const { result, stats } = event.data;
  updateUI(result);
  console.log('Processing stats:', stats);
};

worker.onerror = (error) => {
  console.error('Worker error:', error.message);
};

// Transferable Objects (ownership transfer, not copying)
const buffer = new ArrayBuffer(1024 * 1024); // 1MB
worker.postMessage({ buffer }, [buffer]); // Transfer (no copy)
// At this point, buffer is no longer usable

// === worker.js ===
self.onmessage = (event) => {
  const { type, data } = event.data;

  switch (type) {
    case 'process': {
      const startTime = performance.now();

      // CPU-intensive processing (does not block the main thread)
      const result = data.map(item => {
        return heavyComputation(item);
      });

      const duration = performance.now() - startTime;

      self.postMessage({
        result,
        stats: {
          itemCount: data.length,
          duration: `${duration.toFixed(2)}ms`,
        },
      });
      break;
    }
  }
};

// === Using Workers like RPC with the Comlink library ===
// Main thread
import * as Comlink from 'comlink';

const api = Comlink.wrap(new Worker('api-worker.js'));

// Call Worker methods as if calling them directly
const result = await api.processData(largeDataset);
const hash = await api.hashPassword('secret');

// api-worker.js
import * as Comlink from 'comlink';

const api = {
  processData(data) {
    return data.map(item => heavyComputation(item));
  },
  hashPassword(password) {
    // CPU-intensive hash computation
    return computeHash(password);
  },
};

Comlink.expose(api);
```

---

## 6. Node.js vs Browser Differences

```
┌──────────────────────────────────────────────────┐
│              Node.js vs Browser                    │
├─────────────────┬────────────────────────────────┤
│     Node.js     │         Browser                 │
├─────────────────┼────────────────────────────────┤
│ libuv-based     │ Browser engine proprietary impl │
│ 6 phases        │ Task queue + rendering          │
│ setImmediate ○  │ setImmediate △ (IE only)        │
│ nextTick ○      │ nextTick ✗                      │
│ Worker Threads  │ Web Workers                     │
│ No rendering    │ Rendering is interleaved        │
│ Multiple task   │ Single task queue (basic)       │
│   queues        │                                 │
│ fs, net, etc    │ DOM, fetch, etc                 │
│ Server-side     │ Client-side                     │
└─────────────────┴────────────────────────────────┘

setImmediate vs setTimeout(fn, 0):
  Node.js:
    → Inside I/O callbacks: setImmediate runs first
    → Top level: order is non-deterministic
  Browser:
    → Only setTimeout(fn, 0) (minimum delay 4ms)
    → setImmediate is non-standard
```

```javascript
// Node.js: Order inside I/O callbacks
const fs = require('fs');

fs.readFile('file.txt', () => {
  setTimeout(() => console.log('timeout'), 0);
  setImmediate(() => console.log('immediate'));
});

// Output (always in this order):
// immediate    ← I/O callback → check phase runs first
// timeout

// Node.js: Order at top level (non-deterministic)
setTimeout(() => console.log('timeout'), 0);
setImmediate(() => console.log('immediate'));

// Output (may vary between runs):
// timeout   or  immediate
// immediate     timeout
// → Depends on process startup timing
```

---

## 7. Practical Patterns

### 7.1 Async Iterators and the Event Loop

```javascript
// for-await-of and the event loop
const { once } = require('events');
const { createReadStream } = require('fs');

async function processFile(filePath) {
  const stream = createReadStream(filePath, { encoding: 'utf8' });
  let lineCount = 0;

  for await (const chunk of stream) {
    const lines = chunk.split('\n');
    for (const line of lines) {
      lineCount++;
      await processLine(line);

      // Yield control to the event loop every 1000 lines
      if (lineCount % 1000 === 0) {
        await new Promise(resolve => setImmediate(resolve));
      }
    }
  }

  return lineCount;
}
```

### 7.2 Promise.all and the Event Loop

```javascript
// Promise.all starts all Promises simultaneously
// → Running a large number of Promises concurrently can exhaust resources

// Bad: 10,000 HTTP requests simultaneously
const urls = Array(10000).fill('https://api.example.com/data');
const results = await Promise.all(urls.map(url => fetch(url)));
// → Socket exhaustion, memory pressure

// Good: Limit concurrency
async function promisePool(tasks, concurrency = 10) {
  const results = [];
  const executing = new Set();

  for (const [index, task] of tasks.entries()) {
    const promise = task().then(result => {
      executing.delete(promise);
      return result;
    });
    executing.add(promise);
    results[index] = promise;

    if (executing.size >= concurrency) {
      await Promise.race(executing);
    }
  }

  return Promise.all(results);
}

// Usage
const results = await promisePool(
  urls.map(url => () => fetch(url).then(r => r.json())),
  10, // Maximum 10 concurrent
);
```

### 7.3 Graceful Shutdown

```javascript
const http = require('http');

const server = http.createServer(handler);

// Track new requests
const connections = new Set();
server.on('connection', (conn) => {
  connections.add(conn);
  conn.on('close', () => connections.delete(conn));
});

// Signal handling
let isShuttingDown = false;

async function gracefulShutdown(signal) {
  if (isShuttingDown) return;
  isShuttingDown = true;

  console.log(`${signal} received. Starting graceful shutdown...`);

  // 1. Stop accepting new requests
  server.close(() => {
    console.log('Server closed');
  });

  // 2. Mark health check as unhealthy (detach from load balancer)
  // → Check isShuttingDown in the /health endpoint

  // 3. Wait for in-progress requests to complete (max 30 seconds)
  const forceTimeout = setTimeout(() => {
    console.log('Force shutdown: destroying remaining connections');
    connections.forEach(conn => conn.destroy());
  }, 30000);

  // 4. Clean up resources
  try {
    await Promise.allSettled([
      db.end(),
      redis.quit(),
      messageQueue.close(),
    ]);
    console.log('Resources cleaned up');
  } catch (err) {
    console.error('Cleanup error:', err);
  }

  clearTimeout(forceTimeout);
  process.exit(0);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Health check
app.get('/health', (req, res) => {
  if (isShuttingDown) {
    res.status(503).json({ status: 'shutting-down' });
  } else {
    res.status(200).json({ status: 'healthy' });
  }
});
```

### 7.4 Timer Precision Issues

```javascript
// setTimeout(fn, 0) is not actually 0ms
// Node.js: minimum 1ms
// Browser: minimum 4ms (when nested 5 or more times)

// When high-precision timing is needed
function preciseTimeout(callback, ms) {
  const start = performance.now();

  function check() {
    const elapsed = performance.now() - start;
    if (elapsed >= ms) {
      callback();
    } else if (ms - elapsed > 10) {
      setTimeout(check, 0); // Wait roughly
    } else {
      // Busy-wait for the last milliseconds (for precision)
      setImmediate(check);
    }
  }

  if (ms <= 0) {
    setImmediate(callback);
  } else {
    setTimeout(check, Math.max(0, ms - 10));
  }
}

// setInterval "drift" problem
// Bad: Want to execute every 1 second, but it gradually drifts
let count = 0;
const start = Date.now();
setInterval(() => {
  count++;
  const expected = count * 1000;
  const actual = Date.now() - start;
  console.log(`Drift: ${actual - expected}ms`);
}, 1000);

// Good: Self-correcting timer
function preciseInterval(callback, intervalMs) {
  let expected = Date.now() + intervalMs;

  function step() {
    const drift = Date.now() - expected;
    callback();
    expected += intervalMs;
    setTimeout(step, Math.max(0, intervalMs - drift));
  }

  setTimeout(step, intervalMs);
}
```

---

## 8. Debugging and Troubleshooting

### 8.1 Common Problem Patterns

```javascript
// Problem 1: Callbacks executing in unintended order
function fetchAndProcess() {
  let result = null;

  fetch('/api/data')
    .then(r => r.json())
    .then(data => { result = data; });

  console.log(result); // null! (async processing has not completed)
}

// Problem 2: Unhandled Promise Rejection
// In Node.js 15+, this crashes the process
async function riskyOperation() {
  const data = await fetch('/api/data'); // Error is not caught
  return data.json();
}

riskyOperation(); // No .catch() or try-catch → UnhandledPromiseRejection

// Countermeasure: Global handlers
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection:', reason);
  // Send logs and shut down gracefully
  gracefulShutdown('unhandledRejection');
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  // Shut down immediately (state may be inconsistent)
  process.exit(1);
});

// Problem 3: Memory leak (forgetting to remove event listeners)
const EventEmitter = require('events');
const emitter = new EventEmitter();

// Bad: Listeners accumulate
function handleRequest(req) {
  emitter.on('data', (data) => {
    // A new listener is added for each request
    // → Memory leak
  });
}

// Good: Use once, or manually remove
function handleRequestFixed(req) {
  const handler = (data) => {
    // Processing
  };
  emitter.on('data', handler);

  // Remove when the request ends
  req.on('close', () => {
    emitter.removeListener('data', handler);
  });
}

// Detecting MaxListenersExceededWarning
emitter.setMaxListeners(20); // Default is 10
// If this warning appears, suspect a listener leak
```

### 8.2 Node.js Diagnostic Tools

```javascript
// Connect to Chrome DevTools with the --inspect flag
// node --inspect server.js
// Open chrome://inspect in Chrome

// CPU Profiling
const { writeHeapSnapshot } = require('v8');
const { Session } = require('inspector');

// Taking a heap snapshot
app.get('/debug/heap', (req, res) => {
  const filename = writeHeapSnapshot();
  res.json({ file: filename });
});

// Taking a CPU profile
app.get('/debug/profile', async (req, res) => {
  const session = new Session();
  session.connect();

  session.post('Profiler.enable');
  session.post('Profiler.start');

  // Profile for 10 seconds
  await new Promise(resolve => setTimeout(resolve, 10000));

  session.post('Profiler.stop', (err, { profile }) => {
    session.disconnect();
    // Save profile as a .cpuprofile file
    fs.writeFileSync('profile.cpuprofile', JSON.stringify(profile));
    res.json({ message: 'Profile saved' });
  });
});

// Tracking the event loop with async_hooks
const async_hooks = require('async_hooks');

const resources = new Map();

const hook = async_hooks.createHook({
  init(asyncId, type, triggerAsyncId) {
    resources.set(asyncId, { type, triggerAsyncId, created: Date.now() });
  },
  destroy(asyncId) {
    resources.delete(asyncId);
  },
});

// Enable (has performance overhead, use only for debugging)
hook.enable();

// Display active async resources
setInterval(() => {
  console.log(`Active async resources: ${resources.size}`);
  const types = {};
  for (const [, { type }] of resources) {
    types[type] = (types[type] || 0) + 1;
  }
  console.log(types);
}, 10000);
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Create test code as well

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate the input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main logic for data processing"""
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

Extend the basic implementation to add the following features.

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
    """Efficient search using a hash map"""
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

**Key Points:**
- Be aware of algorithmic time complexity
- Choose appropriate data structures
- Measure results with benchmarks

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes the criteria for making technical choices.

| Criterion | When to Prioritize | When Compromise is Acceptable |
|---------|------------|-------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development Speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
┌─────────────────────────────────────────────────┐
│          Architecture Selection Flow             │
├─────────────────────────────────────────────────┤
│                                                 │
│  (1) Team size?                                 │
│    ├─ Small (1-5 people) → Monolith             │
│    └─ Large (10+ people) → Go to (2)            │
│                                                 │
│  (2) Deployment frequency?                      │
│    ├─ Once a week or less → Monolith +          │
│    │    module separation                       │
│    └─ Daily / multiple times → Go to (3)        │
│                                                 │
│  (3) Team independence?                         │
│    ├─ High → Microservices                      │
│    └─ Moderate → Modular monolith               │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Cost**
- A method that is faster in the short term may become technical debt in the long term
- Conversely, over-engineering incurs high short-term costs and can delay the project

**2. Consistency vs Flexibility**
- A unified technology stack has lower learning costs
- Adopting diverse technologies enables best-fit choices but increases operational costs

**3. Level of Abstraction**
- Higher abstraction improves reusability but can make debugging difficult
- Lower abstraction is more intuitive but tends to produce code duplication

```python
# Architecture Decision Record template
class ArchitectureDecisionRecord:
    """Create an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe the background and challenges"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decision"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """Add a consequence"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """Add a rejected alternative"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Output in Markdown format"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Background\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## Real-World Application Scenarios

### Scenario 1: MVP Development at a Startup

**Situation:** Need to quickly release a product with limited resources

**Approach:**
- Choose a simple architecture
- Focus on the minimum viable feature set
- Automated tests only for the critical path
- Introduce monitoring from the start

**Lessons Learned:**
- Do not aim for perfection (YAGNI principle)
- Obtain user feedback early
- Manage technical debt consciously

### Scenario 2: Modernizing a Legacy System

**Situation:** Incrementally renovating a system that has been in operation for over 10 years

**Approach:**
- Use the Strangler Fig pattern for gradual migration
- If existing tests are missing, create Characterization Tests first
- Use an API gateway to coexist old and new systems
- Perform data migration in stages

| Phase | Work | Estimated Duration | Risk |
|---------|---------|---------|--------|
| 1. Investigation | Current state analysis, dependency mapping | 2-4 weeks | Low |
| 2. Foundation | CI/CD setup, test environment | 4-6 weeks | Low |
| 3. Migration Start | Migrate peripheral features first | 3-6 months | Medium |
| 4. Core Migration | Migrate core features | 6-12 months | High |
| 5. Completion | Decommission old system | 2-4 weeks | Medium |

### Scenario 3: Development with a Large Team

**Situation:** 50+ engineers developing the same product

**Approach:**
- Clarify boundaries with Domain-Driven Design
- Set ownership per team
- Manage shared libraries using Inner Source
- Design API-first to minimize inter-team dependencies

```python
# API contract definition between teams
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """API contract between teams"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # Response time SLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """Check SLA compliance"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """Output in OpenAPI format"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# Usage example
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### Scenario 4: Performance-Critical System

**Situation:** A system that requires millisecond-level response times

**Optimization Points:**
1. Caching strategy (L1: In-memory, L2: Redis, L3: CDN)
2. Leverage asynchronous processing
3. Connection pooling
4. Query optimization and index design

| Optimization Method | Effect | Implementation Cost | Use Case |
|-----------|------|-----------|---------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Async processing | Medium | Medium | I/O-heavy processing |
| DB optimization | High | High | When queries are slow |
| Code optimization | Low-Medium | High | CPU-bound cases |

---

## Leveraging in Team Development

### Code Review Checklist

Points to check in code reviews related to this topic:

- [ ] Are naming conventions consistent?
- [ ] Is error handling appropriate?
- [ ] Is test coverage sufficient?
- [ ] Is there any performance impact?
- [ ] Are there any security concerns?
- [ ] Is documentation updated?

### Best Practices for Knowledge Sharing

| Method | Frequency | Audience | Effect |
|------|------|------|------|
| Pair programming | As needed | Complex tasks | Immediate feedback |
| Tech talks | Weekly | Entire team | Horizontal knowledge spread |
| ADR (Decision records) | Per decision | Future members | Decision transparency |
| Retrospectives | Biweekly | Entire team | Continuous improvement |
| Mob programming | Monthly | Important designs | Consensus building |

### Managing Technical Debt

```
Priority Matrix:

        High Impact
          │
    ┌─────┼─────┐
    │ Plan │ Act  │
    │ for  │ imme-│
    │ later│ dia- │
    │      │ tely │
    ├─────┼─────┤
    │ Record│ Next │
    │ only │Sprint│
    │      │      │
    └─────┼─────┘
          │
        Low Impact
    Low Frequency  High Frequency
```
---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not only through theory but by actually writing code and observing its behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in daily development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|------|---------|
| Event Loop | Scheduler for async processing, composed of 6 phases (Node.js) |
| Microtasks | Promise.then, all processed after each macrotask |
| Macrotasks | setTimeout, processed one at a time |
| process.nextTick | Higher priority than microtasks, beware of starvation |
| Avoiding Blocking | No synchronous I/O, use Workers, split execution |
| Browser | Rendering occurs between macrotasks, rAF runs before rendering |
| Worker Threads | Delegate CPU-intensive processing, shared memory via SharedArrayBuffer |
| Monitoring | perf_hooks, async_hooks, heap snapshots |
| Graceful Shutdown | Signal handling, resource cleanup, timeouts |

---

## 9. FAQ

### Q1: Is setTimeout(fn, 0) really 0ms?

The minimum delay in Node.js is 1ms. In browsers, it is typically 4ms (when nesting is 5 levels or deeper). This is defined by the specification. When precise timing is needed, use `performance.now()` for self-correction, or use `setImmediate` (Node.js) or `requestAnimationFrame` (Browser).

### Q2: How does async/await affect the event loop?

`async/await` is syntactic sugar that internally uses Promises. The code immediately after `await` is queued as a microtask. Therefore, `await` does not block the event loop. However, if the awaited target performs synchronously heavy computation, that computation itself will block the event loop.

### Q3: Should I use process.nextTick() or queueMicrotask()?

For new code, `queueMicrotask()` is recommended. `process.nextTick()` is Node.js-specific and has higher priority than microtasks, which can cause starvation problems. `queueMicrotask()` is a web standard and works in browsers as well. However, `process.nextTick()` is appropriate when you need to ensure execution before I/O callbacks.

### Q4: Does the process exit when the event loop becomes empty?

Yes. Node.js automatically exits when all event loop queues are empty and there are no pending I/O operations or timers. Active handles such as `setInterval` or `server.listen()` prevent the process from exiting. Calling `unref()` excludes a handle from the event loop count, allowing the process to exit if there are no other active handles.

### Q5: Are the event loops in Deno/Bun different from Node.js?

Deno is based on Tokio (Rust's async runtime), which differs from Node.js's libuv, but the concepts of microtasks/macrotasks are the same. Bun has its own event loop implementation (JavaScriptCore + liburing on Linux) that maintains high compatibility with Node.js while improving performance. The fundamental execution order rules are common across all environments.

---

## Recommended Next Reading

---

## References
1. Node.js Documentation. "The Node.js Event Loop."
2. Jake Archibald. "In The Loop." JSConf.Asia, 2018.
3. Node.js Documentation. "Worker Threads."
4. MDN Web Docs. "The event loop." developer.mozilla.org.
5. libuv Documentation. "Design overview." docs.libuv.org.
6. Erin Zimmer. "Further Adventures of the Event Loop." JSConf EU, 2018.
