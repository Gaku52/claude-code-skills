# Callbacks

> Callbacks are the most primitive pattern for asynchronous processing. Understand Node.js error-first callbacks, the callback hell problem, and the evolution toward Promises.

## What You Will Learn in This Chapter

- [ ] Understand how callbacks work and how to use them
- [ ] Identify the problems and causes of callback hell
- [ ] Learn the meaning of the error-first pattern
- [ ] Compare callback implementations across different languages
- [ ] Master the migration pattern from callbacks to Promises


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. Callback Basics

### 1.1 What Is a Callback?

```
Callback = A function passed with the message "call me when processing is complete"

  Synchronous:
    const result = readFile("data.txt");
    console.log(result);

  Asynchronous (callback):
    readFile("data.txt", (error, result) => {
      console.log(result);
    });
    // readFile returns immediately. The result arrives later via the callback

Types of callbacks:
  1. Synchronous callbacks: map, filter, sort, etc. (executed immediately)
  2. Asynchronous callbacks: Called after I/O completion (setTimeout, fs.readFile, etc.)
```

### 1.2 Synchronous Callbacks vs Asynchronous Callbacks

```javascript
// === Synchronous Callbacks ===
// Passed as function arguments and executed immediately on the spot

// Array.map: Apply a callback to each element
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map((n) => n * 2);  // [2, 4, 6, 8, 10]

// Array.filter: Keep only elements for which the callback returns true
const evens = numbers.filter((n) => n % 2 === 0);  // [2, 4]

// Array.reduce: Calculate an accumulated value
const sum = numbers.reduce((acc, n) => acc + n, 0);  // 15

// Array.sort: Inject comparison logic via a callback
const users = [
  { name: "Tanaka", age: 30 },
  { name: "Yamada", age: 25 },
  { name: "Suzuki", age: 35 },
];
users.sort((a, b) => a.age - b.age);
// [{ name: "Yamada", age: 25 }, { name: "Tanaka", age: 30 }, { name: "Suzuki", age: 35 }]

// Array.forEach: Execute side effects for each element
numbers.forEach((n) => {
  console.log(n);
});

// Array.find: Return the first element matching the condition
const firstEven = numbers.find((n) => n % 2 === 0);  // 2

// All of these are "synchronous callbacks"
// -> All processing is complete by the time the function returns
```

```javascript
// === Asynchronous Callbacks ===
// Passed to a function and executed after I/O completion or after a set time

const fs = require('fs');

// Node.js file reading
console.log('1. Starting read');

fs.readFile('/path/to/file', 'utf8', (err, data) => {
  // This function is called "later" (when file reading is complete)
  console.log('3. File read complete:', data);
});

console.log('2. After issuing read command (not yet complete)');

// Output order:
// 1. Starting read
// 2. After issuing read command (not yet complete)
// 3. File read complete: (file contents)
```

### 1.3 Callbacks as Event Listeners

```javascript
// Browser: Event listeners
document.getElementById('btn').addEventListener('click', (event) => {
  console.log('Clicked!', event.target);
});

// Registering multiple events
const button = document.getElementById('submit');

button.addEventListener('click', handleClick);
button.addEventListener('mouseenter', handleHover);
button.addEventListener('mouseleave', handleLeave);

function handleClick(event) {
  event.preventDefault();
  console.log('Button clicked');
}

function handleHover(event) {
  event.target.style.backgroundColor = '#f0f0f0';
}

function handleLeave(event) {
  event.target.style.backgroundColor = '';
}

// Removing an event listener
button.removeEventListener('click', handleClick);
```

### 1.4 Timer Callbacks

```javascript
// setTimeout: Execute once after a specified time
setTimeout(() => {
  console.log('Executed after 3 seconds');
}, 3000);

// setInterval: Execute repeatedly at a specified interval
const intervalId = setInterval(() => {
  console.log('Executed every 1 second');
}, 1000);

// Stop
setTimeout(() => {
  clearInterval(intervalId);
  console.log('Timer stopped');
}, 5000);

// requestAnimationFrame: Execute on each paint frame (browser)
function animate(timestamp) {
  // Animation processing
  updatePosition(timestamp);
  render();

  // Request the next frame
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
```

---

## 2. Node.js Error-First Callbacks

### 2.1 Basic Pattern

```
Node.js convention (error-first callback):
  callback(error, result)

  -> 1st argument: error (null on success)
  -> 2nd argument: result (undefined on error)

  Advantages:
  - Uniform error checking
  - Hard to ignore errors (habit of checking the 1st argument)

  Problems:
  - Requires if (err) check every time
  - No type safety (any)
  - Nesting tends to get deep
```

```javascript
const fs = require('fs');

// Basic error-first callback
fs.readFile('/path/to/file', 'utf8', (err, data) => {
  if (err) {
    console.error('Error:', err.message);
    return;
  }
  console.log('Data:', data);
});

// Writing
fs.writeFile('/path/to/output', 'Hello, World!', 'utf8', (err) => {
  if (err) {
    console.error('Write failed:', err.message);
    return;
  }
  console.log('File written successfully');
});

// Reading a directory
fs.readdir('/path/to/dir', (err, files) => {
  if (err) {
    console.error('Failed to read directory:', err.message);
    return;
  }
  console.log('Files:', files);
});
```

### 2.2 Implementing the Error-First Pattern

```javascript
// Custom error-first function
function readJsonFile(path, callback) {
  fs.readFile(path, 'utf8', (err, data) => {
    if (err) {
      callback(err, null);
      return;
    }
    try {
      const parsed = JSON.parse(data);
      callback(null, parsed);
    } catch (parseError) {
      callback(parseError, null);
    }
  });
}

// Usage
readJsonFile('config.json', (err, config) => {
  if (err) {
    console.error('Failed to read config:', err.message);
    return;
  }
  console.log('Config loaded:', config);
});
```

```javascript
// Asynchronous database operations (callback style)
function getUser(userId, callback) {
  db.query('SELECT * FROM users WHERE id = ?', [userId], (err, rows) => {
    if (err) {
      callback(err, null);
      return;
    }
    if (rows.length === 0) {
      callback(new Error(`User ${userId} not found`), null);
      return;
    }
    callback(null, rows[0]);
  });
}

function getUserOrders(userId, callback) {
  db.query('SELECT * FROM orders WHERE user_id = ?', [userId], (err, rows) => {
    if (err) {
      callback(err, null);
      return;
    }
    callback(null, rows);
  });
}

// HTTP request (callback style)
const http = require('http');

function fetchJSON(url, callback) {
  http.get(url, (res) => {
    let data = '';

    res.on('data', (chunk) => {
      data += chunk;
    });

    res.on('error', (err) => {
      callback(err, null);
    });

    res.on('end', () => {
      try {
        const parsed = JSON.parse(data);
        callback(null, parsed);
      } catch (parseErr) {
        callback(parseErr, null);
      }
    });
  }).on('error', (err) => {
    callback(err, null);
  });
}

// Usage
fetchJSON('http://api.example.com/users/1', (err, user) => {
  if (err) {
    console.error('Failed to fetch user:', err.message);
    return;
  }
  console.log('User:', user);
});
```

### 2.3 Callback Design Patterns

```javascript
// Pattern 1: Configuration object and callback
function connectToDatabase(options, callback) {
  const { host, port, database, user, password } = options;

  const connection = new DatabaseConnection({
    host, port, database, user, password
  });

  connection.connect((err) => {
    if (err) {
      callback(err, null);
      return;
    }

    // Connection successful: Check migrations
    connection.checkMigrations((err, needsMigration) => {
      if (err) {
        connection.close();
        callback(err, null);
        return;
      }

      if (needsMigration) {
        connection.runMigrations((err) => {
          if (err) {
            connection.close();
            callback(err, null);
            return;
          }
          callback(null, connection);
        });
      } else {
        callback(null, connection);
      }
    });
  });
}

// Pattern 2: EventEmitter style
const EventEmitter = require('events');

class FileProcessor extends EventEmitter {
  process(filePath) {
    fs.readFile(filePath, 'utf8', (err, data) => {
      if (err) {
        this.emit('error', err);
        return;
      }

      this.emit('data', data);

      const lines = data.split('\n');
      this.emit('line-count', lines.length);

      for (const line of lines) {
        this.emit('line', line);
      }

      this.emit('complete', { totalLines: lines.length });
    });
  }
}

// Usage
const processor = new FileProcessor();

processor.on('data', (data) => {
  console.log(`Loaded ${data.length} bytes`);
});

processor.on('line', (line) => {
  // Process each line
});

processor.on('complete', ({ totalLines }) => {
  console.log(`Processed ${totalLines} lines`);
});

processor.on('error', (err) => {
  console.error('Error:', err.message);
});

processor.process('large-file.txt');
```

---

## 3. Callback Hell

### 3.1 The Core Problem

```javascript
// Bad: Callback hell: Nesting gets deep and readability collapses
getUser(userId, (err, user) => {
  if (err) { handleError(err); return; }
  getOrders(user.id, (err, orders) => {
    if (err) { handleError(err); return; }
    getOrderDetails(orders[0].id, (err, details) => {
      if (err) { handleError(err); return; }
      getShippingInfo(details.shippingId, (err, shipping) => {
        if (err) { handleError(err); return; }
        getTrackingInfo(shipping.trackingId, (err, tracking) => {
          if (err) { handleError(err); return; }
          // 5 levels of nesting at this point
          console.log(tracking);
        });
      });
    });
  });
});

// Problems:
// 1. "Pyramid-shaped" code expanding horizontally
// 2. Duplicated error handling
// 3. Difficult variable scope management
// 4. Hard to follow the flow of execution
// 5. Difficult to write tests
// 6. Complex control flow implementation (conditionals, loops)
```

### 3.2 Typical Callback Hell Encountered in Practice

```javascript
// E-commerce order processing (callback hell version)
function processOrder(userId, cartId, paymentInfo, callback) {
  // 1. User authentication
  authenticateUser(userId, (err, user) => {
    if (err) { callback(err); return; }

    // 2. Get cart
    getCart(cartId, (err, cart) => {
      if (err) { callback(err); return; }

      // 3. Check inventory
      checkInventory(cart.items, (err, availability) => {
        if (err) { callback(err); return; }

        if (!availability.allAvailable) {
          callback(new Error('Some items are out of stock'));
          return;
        }

        // 4. Calculate total
        calculateTotal(cart, user, (err, total) => {
          if (err) { callback(err); return; }

          // 5. Process payment
          processPayment(paymentInfo, total, (err, paymentResult) => {
            if (err) {
              // Rollback inventory on payment failure
              releaseInventory(cart.items, (rollbackErr) => {
                if (rollbackErr) {
                  console.error('Rollback failed:', rollbackErr);
                }
                callback(err);
              });
              return;
            }

            // 6. Create order
            createOrder(user, cart, paymentResult, (err, order) => {
              if (err) {
                // Refund payment on order creation failure
                refundPayment(paymentResult.id, (refundErr) => {
                  if (refundErr) {
                    console.error('Refund failed:', refundErr);
                  }
                  callback(err);
                });
                return;
              }

              // 7. Send notification
              sendOrderConfirmation(user.email, order, (err) => {
                if (err) {
                  console.error('Email failed:', err);
                  // Ignore email failure and treat as success
                }
                callback(null, order);
              });
            });
          });
        });
      });
    });
  });
}
```

### 3.3 Improvement Technique 1: Separate with Named Functions

```javascript
// Somewhat improved: Separate with named functions
function handleTracking(err, tracking) {
  if (err) { handleError(err); return; }
  console.log(tracking);
}

function handleShipping(err, shipping) {
  if (err) { handleError(err); return; }
  getTrackingInfo(shipping.trackingId, handleTracking);
}

function handleDetails(err, details) {
  if (err) { handleError(err); return; }
  getShippingInfo(details.shippingId, handleShipping);
}

function handleOrders(err, orders) {
  if (err) { handleError(err); return; }
  getOrderDetails(orders[0].id, handleDetails);
}

function handleUser(err, user) {
  if (err) { handleError(err); return; }
  getOrders(user.id, handleOrders);
}

// Entry point
getUser(userId, handleUser);

// Improvement: Shallow nesting
// Remaining problem: Functions are defined in reverse order, making the flow hard to follow
```

### 3.4 Improvement Technique 2: async Library

```javascript
// Flow control using the async.js library
const async = require('async');

// async.waterfall: Serial execution (passing previous result to next)
async.waterfall([
  // Step 1: Get user
  (cb) => getUser(userId, cb),

  // Step 2: Get orders (user is the result from the previous step)
  (user, cb) => getOrders(user.id, (err, orders) => {
    cb(err, user, orders);
  }),

  // Step 3: Get order details
  (user, orders, cb) => getOrderDetails(orders[0].id, (err, details) => {
    cb(err, user, orders, details);
  }),

  // Step 4: Get shipping info
  (user, orders, details, cb) => {
    getShippingInfo(details.shippingId, cb);
  },
], (err, shippingInfo) => {
  if (err) {
    handleError(err);
    return;
  }
  console.log('Shipping:', shippingInfo);
});

// async.parallel: Parallel execution
async.parallel({
  users: (cb) => fetchUsers(cb),
  orders: (cb) => fetchOrders(cb),
  products: (cb) => fetchProducts(cb),
}, (err, results) => {
  if (err) {
    handleError(err);
    return;
  }
  console.log(results.users, results.orders, results.products);
});

// async.series: Serial execution (results kept separate)
async.series([
  (cb) => createBackup(cb),
  (cb) => runMigrations(cb),
  (cb) => verifyData(cb),
], (err, results) => {
  if (err) {
    console.error('Pipeline failed:', err);
    return;
  }
  console.log('All steps completed');
});

// async.eachLimit: Iteration with concurrency limit
const urls = ['url1', 'url2', 'url3', /* ... */];

async.eachLimit(urls, 5, (url, cb) => {
  fetchAndProcess(url, cb);
}, (err) => {
  if (err) {
    console.error('Processing failed:', err);
    return;
  }
  console.log('All URLs processed');
});
```

### 3.5 Improvement Technique 3: Control Flow Abstraction

```javascript
// Custom flow control function
function waterfall(tasks, finalCallback) {
  let index = 0;

  function next(err, ...args) {
    if (err) {
      finalCallback(err);
      return;
    }

    if (index >= tasks.length) {
      finalCallback(null, ...args);
      return;
    }

    const task = tasks[index++];
    try {
      task(...args, next);
    } catch (e) {
      finalCallback(e);
    }
  }

  next(null);
}

// Usage
waterfall([
  (cb) => getUser(userId, cb),
  (user, cb) => getOrders(user.id, cb),
  (orders, cb) => getOrderDetails(orders[0].id, cb),
], (err, details) => {
  if (err) {
    console.error(err);
    return;
  }
  console.log('Details:', details);
});

// Parallel execution function
function parallel(tasks, finalCallback) {
  const results = {};
  let completed = 0;
  let hasError = false;
  const keys = Object.keys(tasks);

  keys.forEach((key) => {
    taskskey => {
      if (hasError) return;

      if (err) {
        hasError = true;
        finalCallback(err);
        return;
      }

      results[key] = result;
      completed++;

      if (completed === keys.length) {
        finalCallback(null, results);
      }
    });
  });
}
```

---

## 4. Callback Patterns Across Languages

### 4.1 Python Callbacks

```python
import threading
import time
from typing import Callable, Optional, Any

# Callback patterns in Python

# Basic callback
def fetch_data(url: str, on_success: Callable, on_error: Callable) -> None:
    """Fetch data asynchronously (thread-based)"""
    def worker():
        try:
            import urllib.request
            response = urllib.request.urlopen(url)
            data = response.read().decode('utf-8')
            on_success(data)
        except Exception as e:
            on_error(e)

    thread = threading.Thread(target=worker)
    thread.start()

# Usage
def handle_success(data):
    print(f"Received: {data[:100]}...")

def handle_error(error):
    print(f"Error: {error}")

fetch_data("https://api.example.com/data", handle_success, handle_error)

# Decorators as callbacks
def retry(max_retries: int = 3, delay: float = 1.0):
    """Retry decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (2 ** attempt))
        return wrapper
    return decorator

@retry(max_retries=3, delay=0.5)
def unreliable_api_call():
    """Unreliable API call"""
    import random
    if random.random() < 0.5:
        raise ConnectionError("Connection failed")
    return {"status": "ok"}

# Context manager + callback
class TimedOperation:
    """Measure operation time and report via callback"""
    def __init__(self, name: str, on_complete: Callable[[str, float], None]):
        self.name = name
        self.on_complete = on_complete
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.on_complete(self.name, elapsed)
        return False

# Usage
def log_timing(name: str, elapsed: float):
    print(f"[TIMING] {name}: {elapsed:.3f}s")

with TimedOperation("data_processing", log_timing):
    time.sleep(0.5)  # Some processing
# Output: [TIMING] data_processing: 0.501s
```

### 4.2 Rust Callbacks

```rust
use std::thread;
use std::sync::mpsc;

// Rust: Using closures as callbacks
// Subject to ownership and lifetime constraints

// Basic callback
fn process_async<F>(data: Vec<i32>, callback: F)
where
    F: FnOnce(Vec<i32>) + Send + 'static,
{
    thread::spawn(move || {
        let result: Vec<i32> = data.iter().map(|x| x * 2).collect();
        callback(result);
    });
}

// Usage
fn main() {
    process_async(vec![1, 2, 3, 4, 5], |result| {
        println!("Result: {:?}", result);
    });

    thread::sleep(std::time::Duration::from_secs(1));
}

// Error handling with Result type
fn fetch_data<F>(url: &str, callback: F)
where
    F: FnOnce(Result<String, Box<dyn std::error::Error>>) + Send + 'static,
{
    let url = url.to_string();
    thread::spawn(move || {
        let result = reqwest::blocking::get(&url)
            .and_then(|resp| resp.text());
        match result {
            Ok(body) => callback(Ok(body)),
            Err(e) => callback(Err(Box::new(e))),
        }
    });
}

// Callbacks using trait objects
trait EventHandler: Send {
    fn on_data(&self, data: &[u8]);
    fn on_error(&self, error: &str);
    fn on_complete(&self);
}

struct DataProcessor {
    handler: Box<dyn EventHandler>,
}

impl DataProcessor {
    fn new(handler: Box<dyn EventHandler>) -> Self {
        DataProcessor { handler }
    }

    fn process(&self, data: &[u8]) {
        if data.is_empty() {
            self.handler.on_error("Empty data");
            return;
        }
        self.handler.on_data(data);
        self.handler.on_complete();
    }
}
```

### 4.3 Go Callbacks

```go
package main

import (
    "fmt"
    "io/ioutil"
    "net/http"
    "time"
)

// Go: Using functions as first-class citizens
// However, goroutine + channel is more idiomatic in Go

// Callback type definitions
type ResultCallback func(data []byte, err error)
type ProgressCallback func(current, total int)

// HTTP request with callback
func fetchWithCallback(url string, callback ResultCallback) {
    go func() {
        resp, err := http.Get(url)
        if err != nil {
            callback(nil, err)
            return
        }
        defer resp.Body.Close()

        body, err := ioutil.ReadAll(resp.Body)
        if err != nil {
            callback(nil, err)
            return
        }
        callback(body, nil)
    }()
}

// Download with progress
func downloadWithProgress(url string, progress ProgressCallback, done ResultCallback) {
    go func() {
        resp, err := http.Get(url)
        if err != nil {
            done(nil, err)
            return
        }
        defer resp.Body.Close()

        contentLength := int(resp.ContentLength)
        data := make([]byte, 0, contentLength)
        buf := make([]byte, 4096)
        received := 0

        for {
            n, err := resp.Body.Read(buf)
            if n > 0 {
                data = append(data, buf[:n]...)
                received += n
                progress(received, contentLength)
            }
            if err != nil {
                break
            }
        }

        done(data, nil)
    }()
}

func main() {
    // Usage example
    fetchWithCallback("https://api.example.com/data", func(data []byte, err error) {
        if err != nil {
            fmt.Println("Error:", err)
            return
        }
        fmt.Println("Received:", len(data), "bytes")
    })

    // Go idiom: channels are preferred
    ch := make(chan []byte, 1)
    errCh := make(chan error, 1)

    go func() {
        resp, err := http.Get("https://api.example.com/data")
        if err != nil {
            errCh <- err
            return
        }
        defer resp.Body.Close()
        body, _ := ioutil.ReadAll(resp.Body)
        ch <- body
    }()

    select {
    case data := <-ch:
        fmt.Println("Received:", len(data), "bytes")
    case err := <-errCh:
        fmt.Println("Error:", err)
    case <-time.After(5 * time.Second):
        fmt.Println("Timeout")
    }
}
```

### 4.4 C# Callbacks

```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;

// C#: Callbacks via delegates and events

// Delegate definitions
public delegate void DataCallback(string data);
public delegate void ErrorCallback(Exception error);

public class AsyncFetcher
{
    // Event-based callbacks
    public event EventHandler<DataEventArgs> DataReceived;
    public event EventHandler<ErrorEventArgs> ErrorOccurred;
    public event EventHandler Completed;

    // Method accepting callbacks
    public void FetchData(string url, Action<string> onSuccess, Action<Exception> onError)
    {
        Task.Run(async () =>
        {
            try
            {
                using var client = new HttpClient();
                var data = await client.GetStringAsync(url);
                onSuccess(data);
            }
            catch (Exception ex)
            {
                onError(ex);
            }
        });
    }

    // Method that fires events
    public async void FetchDataEvent(string url)
    {
        try
        {
            using var client = new HttpClient();
            var data = await client.GetStringAsync(url);
            DataReceived?.Invoke(this, new DataEventArgs(data));
            Completed?.Invoke(this, EventArgs.Empty);
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(this, new ErrorEventArgs(ex));
        }
    }
}

// Usage
var fetcher = new AsyncFetcher();

// Callback with lambda expressions
fetcher.FetchData(
    "https://api.example.com/data",
    data => Console.WriteLine($"Success: {data.Length} chars"),
    error => Console.WriteLine($"Error: {error.Message}")
);

// Callback with events
fetcher.DataReceived += (sender, args) =>
{
    Console.WriteLine($"Data received: {args.Data.Length} chars");
};
fetcher.ErrorOccurred += (sender, args) =>
{
    Console.WriteLine($"Error: {args.Error.Message}");
};
fetcher.FetchDataEvent("https://api.example.com/data");
```

---

## 5. Callbacks as Higher-Order Functions

### 5.1 Function Composition and Callbacks

```javascript
// Callbacks are a type of "higher-order function"
// Passing "what to do" as an argument

// Strategy pattern: Injecting algorithms via callbacks
function sortUsers(users, comparator) {
  return [...users].sort(comparator);
}

const users = [
  { name: "Tanaka", age: 30, score: 85 },
  { name: "Yamada", age: 25, score: 92 },
  { name: "Suzuki", age: 35, score: 78 },
];

// Sort by age
const byAge = sortUsers(users, (a, b) => a.age - b.age);

// Sort by score (descending)
const byScore = sortUsers(users, (a, b) => b.score - a.score);

// Sort by name
const byName = sortUsers(users, (a, b) => a.name.localeCompare(b.name, 'ja'));
```

```typescript
// Middleware pattern (Express style)
type Middleware = (req: Request, res: Response, next: () => void) => void;

class Router {
  private middlewares: Middleware[] = [];

  use(middleware: Middleware): void {
    this.middlewares.push(middleware);
  }

  handle(req: Request, res: Response): void {
    let index = 0;

    const next = () => {
      if (index < this.middlewares.length) {
        const middleware = this.middlewares[index++];
        middleware(req, res, next);
      }
    };

    next();
  }
}

// Usage
const router = new Router();

// Logging middleware
router.use((req, res, next) => {
  console.log(`${req.method} ${req.url}`);
  next();
});

// Authentication middleware
router.use((req, res, next) => {
  if (!req.headers.authorization) {
    res.status(401).send('Unauthorized');
    return;
  }
  next();
});

// Handler
router.use((req, res, next) => {
  res.json({ message: 'Hello, World!' });
});
```

### 5.2 Currying Callbacks

```typescript
// Curried callbacks
function createLogger(prefix: string) {
  return function(message: string) {
    console.log(`[${prefix}] ${new Date().toISOString()} ${message}`);
  };
}

const infoLog = createLogger('INFO');
const errorLog = createLogger('ERROR');
const debugLog = createLogger('DEBUG');

infoLog('Server started');    // [INFO] 2024-01-01T00:00:00.000Z Server started
errorLog('Connection lost');  // [ERROR] 2024-01-01T00:00:00.000Z Connection lost

// Callback factory
function createRetryCallback<T>(
  fn: (callback: (err: Error | null, result?: T) => void) => void,
  maxRetries: number,
  delay: number,
): Promise<T> {
  return new Promise((resolve, reject) => {
    let attempts = 0;

    function attempt() {
      fn((err, result) => {
        if (!err) {
          resolve(result!);
          return;
        }

        attempts++;
        if (attempts >= maxRetries) {
          reject(err);
          return;
        }

        setTimeout(attempt, delay * Math.pow(2, attempts));
      });
    }

    attempt();
  });
}

// Usage
const result = await createRetryCallback(
  (cb) => fetchData('https://api.example.com/data', cb),
  3,
  1000,
);
```

---

## 6. Callback Pitfalls

### 6.1 The Zalgo Problem (Mixing Sync and Async)

```javascript
// Bad: Zalgo: Synchronous or asynchronous behavior depends on conditions
function getData(cache, key, callback) {
  if (cache[key]) {
    // Warning: Calling callback synchronously
    callback(null, cache[key]);
  } else {
    // Calling callback asynchronously
    db.query(key, (err, data) => {
      if (!err) cache[key] = data;
      callback(err, data);
    });
  }
}

// Problem: Execution order of calling code becomes unpredictable
let result;
getData(cache, 'key', (err, data) => {
  result = data;
});
// Whether result is set depends on the cache state
// -> Very bug-prone

// Good: Always make it asynchronous
function getDataFixed(cache, key, callback) {
  if (cache[key]) {
    // Make it asynchronous with process.nextTick
    process.nextTick(() => callback(null, cache[key]));
  } else {
    db.query(key, (err, data) => {
      if (!err) cache[key] = data;
      callback(err, data);
    });
  }
}

// Better: queueMicrotask (works in both browser and Node.js)
function getDataBetter(cache, key, callback) {
  if (cache[key]) {
    queueMicrotask(() => callback(null, cache[key]));
  } else {
    db.query(key, (err, data) => {
      if (!err) cache[key] = data;
      callback(err, data);
    });
  }
}
```

### 6.2 Double Callback Invocation

```javascript
// Bad: Callback may be called twice
function processFile(path, callback) {
  fs.readFile(path, 'utf8', (err, data) => {
    if (err) {
      callback(err);
      // Warning: Missing return!
    }
    // This executes even on error
    const processed = transform(data); // data is undefined -> error
    callback(null, processed);
  });
}

// Good: Early return
function processFileFixed(path, callback) {
  fs.readFile(path, 'utf8', (err, data) => {
    if (err) {
      callback(err);
      return; // <- Important
    }
    try {
      const processed = transform(data);
      callback(null, processed);
    } catch (transformErr) {
      callback(transformErr);
    }
  });
}

// Better: once wrapper
function once(fn) {
  let called = false;
  return function(...args) {
    if (called) {
      console.warn('Callback called more than once');
      return;
    }
    called = true;
    fn(...args);
  };
}

function processFileSafe(path, callback) {
  const safeCallback = once(callback);

  fs.readFile(path, 'utf8', (err, data) => {
    if (err) {
      safeCallback(err);
      return;
    }
    try {
      const processed = transform(data);
      safeCallback(null, processed);
    } catch (transformErr) {
      safeCallback(transformErr);
    }
  });
}
```

### 6.3 Swallowed Errors

```javascript
// Bad: Errors inside callbacks do not propagate outward
try {
  getUser(userId, (err, user) => {
    if (err) throw err; // <- This will NOT be caught!
    // The callback executes in a separate call stack
    // so try-catch does not work
  });
} catch (err) {
  // This is never reached
  console.error(err);
}

// Good: Handle errors inside the callback
getUser(userId, (err, user) => {
  if (err) {
    console.error('Error:', err.message);
    // Error recovery or alerting
    return;
  }
  // Normal processing
});

// Acceptable: Catch errors with domain (deprecated, shown for reference)
const domain = require('domain');
const d = domain.create();

d.on('error', (err) => {
  console.error('Domain caught:', err);
});

d.run(() => {
  getUser(userId, (err, user) => {
    if (err) throw err; // Domain catches this
  });
});
```

### 6.4 Memory Leaks

```javascript
// Bad: Memory leak via closures
function createConnection(config) {
  const connection = new DatabaseConnection(config);
  const largeBuffer = Buffer.alloc(100 * 1024 * 1024); // 100MB

  return {
    query(sql, callback) {
      // largeBuffer is retained by the closure (even if unused)
      connection.execute(sql, (err, rows) => {
        callback(err, rows);
      });
    },
    close() {
      connection.close();
    }
  };
}

// Good: Do not hold unnecessary references
function createConnectionFixed(config) {
  const connection = new DatabaseConnection(config);

  // largeBuffer is outside the function scope
  function processLargeData() {
    const largeBuffer = Buffer.alloc(100 * 1024 * 1024);
    // Reference disappears after use
    return transform(largeBuffer);
  }

  return {
    query(sql, callback) {
      connection.execute(sql, callback);
    },
    close() {
      connection.close();
    }
  };
}

// Bad: Accumulating event listeners
function setupHandler(element) {
  // A listener is added every time this is called
  element.addEventListener('click', () => {
    doSomething();
  });
}

// Good: Remove existing listeners
function setupHandlerFixed(element) {
  // Keep a reference with a named function
  if (element._clickHandler) {
    element.removeEventListener('click', element._clickHandler);
  }

  element._clickHandler = () => {
    doSomething();
  };
  element.addEventListener('click', element._clickHandler);
}

// Better: Use AbortController
function setupHandlerModern(element) {
  const controller = new AbortController();

  element.addEventListener('click', () => {
    doSomething();
  }, { signal: controller.signal });

  // Cleanup
  return () => controller.abort();
}
```

---

## 7. Migrating from Callbacks to Promises

### 7.1 Manual Promisification

```javascript
// Manually promisify
function readFilePromise(path) {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err, data) => {
      if (err) reject(err);
      else resolve(data);
    });
  });
}

// Usage
readFilePromise('file.txt')
  .then(data => console.log(data))
  .catch(err => console.error(err));

// async/await version
async function main() {
  try {
    const data = await readFilePromise('file.txt');
    console.log(data);
  } catch (err) {
    console.error(err);
  }
}
```

### 7.2 util.promisify

```javascript
// Node.js: Convert callbacks to Promises with util.promisify
const { promisify } = require('util');
const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);
const readdir = promisify(fs.readdir);

// Callback version
fs.readFile('file.txt', 'utf8', (err, data) => {
  if (err) throw err;
  console.log(data);
});

// Promise version
readFile('file.txt', 'utf8')
  .then(data => console.log(data))
  .catch(err => console.error(err));

// async/await version
async function main() {
  try {
    const data = await readFile('file.txt', 'utf8');
    console.log(data);
  } catch (err) {
    console.error(err);
  }
}

// fs/promises (Node.js 14+)
const fsPromises = require('fs/promises');

async function modernFileOps() {
  const data = await fsPromises.readFile('file.txt', 'utf8');
  await fsPromises.writeFile('output.txt', data.toUpperCase());
  const files = await fsPromises.readdir('.');
  console.log(files);
}
```

### 7.3 Generic promisify Function

```typescript
// Generic promisify implementation
function promisify<T>(
  fn: (...args: [...any[], (err: Error | null, result: T) => void]) => void
): (...args: any[]) => Promise<T> {
  return function (...args: any[]): Promise<T> {
    return new Promise((resolve, reject) => {
      fn(...args, (err: Error | null, result: T) => {
        if (err) {
          reject(err);
        } else {
          resolve(result);
        }
      });
    });
  };
}

// promisify for callbacks with multiple return values
function promisifyMultiResult(fn) {
  return function (...args) {
    return new Promise((resolve, reject) => {
      fn(...args, (err, ...results) => {
        if (err) {
          reject(err);
        } else {
          resolve(results);
        }
      });
    });
  };
}

// Convert EventEmitter to Promise
function waitForEvent(emitter, eventName, timeout = 5000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Timeout waiting for event: ${eventName}`));
    }, timeout);

    emitter.once(eventName, (...args) => {
      clearTimeout(timer);
      resolve(args.length === 1 ? args[0] : args);
    });

    emitter.once('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

// Usage
const server = createServer();
const connection = await waitForEvent(server, 'connection', 10000);
```

### 7.4 Wrapper Class for Callback APIs

```typescript
// Wrap a legacy callback API in a modern interface
class DatabaseWrapper {
  private db: LegacyDatabase;

  constructor(connectionString: string) {
    this.db = new LegacyDatabase(connectionString);
  }

  // Wrap the callback API with a Promise
  query<T>(sql: string, params?: any[]): Promise<T[]> {
    return new Promise((resolve, reject) => {
      this.db.query(sql, params || [], (err: Error | null, rows: T[]) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  // Transaction
  async transaction<T>(fn: (tx: TransactionContext) => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.db.beginTransaction((err: Error | null, tx: any) => {
        if (err) {
          reject(err);
          return;
        }

        const context = new TransactionContext(tx);

        fn(context)
          .then((result) => {
            tx.commit((err: Error | null) => {
              if (err) reject(err);
              else resolve(result);
            });
          })
          .catch((error) => {
            tx.rollback((rollbackErr: Error | null) => {
              if (rollbackErr) {
                console.error('Rollback failed:', rollbackErr);
              }
              reject(error);
            });
          });
      });
    });
  }

  // Connection pool
  getConnection(): Promise<Connection> {
    return new Promise((resolve, reject) => {
      this.db.getConnection((err: Error | null, conn: any) => {
        if (err) reject(err);
        else resolve(new Connection(conn));
      });
    });
  }
}

// Usage (clean async/await)
const db = new DatabaseWrapper('postgres://localhost/mydb');

async function getUserOrders(userId: string) {
  const [user] = await db.query<User>('SELECT * FROM users WHERE id = $1', [userId]);
  if (!user) throw new Error('User not found');

  const orders = await db.query<Order>(
    'SELECT * FROM orders WHERE user_id = $1',
    [userId]
  );

  return { user, orders };
}
```

---

## 8. Best Practices in Production

### 8.1 Callback Design Rules

```
1. Always use the error-first pattern
   Follow the callback(err, result) format

2. Always call callbacks asynchronously
   Use process.nextTick / queueMicrotask to avoid the Zalgo problem

3. Call callbacks only once
   Prevent double invocation with a once() wrapper

4. Always pass errors through the callback
   Use callback(err) instead of throw

5. Keep nesting to 3 levels or fewer
   Separate into named functions, or use the async library

6. Migrate to Promise / async-await if possible
   Avoid callbacks in new code
```

### 8.2 Migration Strategy

```typescript
// Gradual migration strategy

// Step 1: Wrap existing callback APIs
const readFileAsync = promisify(fs.readFile);

// Step 2: Write new functions with async/await
async function loadConfig(): Promise<Config> {
  const data = await readFileAsync('config.json', 'utf8');
  return JSON.parse(data);
}

// Step 3: Create a dual interface for functions that accept callbacks
function getData(
  key: string,
  callback?: (err: Error | null, data?: Data) => void,
): Promise<Data> | void {
  const promise = getDataInternal(key);

  if (callback) {
    promise
      .then(data => callback(null, data))
      .catch(err => callback(err));
    return;
  }

  return promise;
}

// Use with callback style
getData('key', (err, data) => {
  if (err) handleError(err);
  else console.log(data);
});

// Use with Promise style
const data = await getData('key');
```


---

## Hands-On Exercises

### Exercise 1: Basic Implementation

Implement code that meets the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Write test code as well

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

Extend the basic implementation by adding the following features.

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
    print(f"Speedup:             {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be aware of algorithmic complexity
- Choose appropriate data structures
- Measure results with benchmarks
---


## FAQ

### Q1: What is the most important point to focus on when learning this topic?

Gaining hands-on experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts covered in this guide before moving on to the next step.

### Q3: How is this applied in real-world development?

Knowledge of this topic is frequently used in everyday development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Callback | A function called upon completion |
| Error-first | The (err, result) convention |
| Callback hell | Deep nesting -> Solved with Promises |
| Synchronous callbacks | map, filter, sort, reduce |
| Asynchronous callbacks | I/O, timers, events |
| Zalgo problem | Avoid mixing sync and async |
| Double invocation | Prevent with a once() wrapper |
| Memory leaks | Watch out for closure references |

### The Evolution of Callbacks

```
Callbacks (1990s~)
  | Problem: Callback hell
Promise (ES2015 / 2015~)
  | Improvement: Chainable, error propagation
async/await (ES2017 / 2017~)
  | Improvement: Synchronous-style writing
Reactive Streams (RxJS, etc.)
  | Extension: Stream processing
AsyncIterator / for-await-of (ES2018)
  -> Asynchronous iteration
```

---

## Recommended Next Guides

---

## References
1. Node.js Documentation. "Asynchronous Programming."
2. Ogden, M. "Callback Hell." callbackhell.com.
3. Havoc Pennington. "Don't Release Zalgo!" blog.izs.me.
4. Casciaro, M. & Mammoliti, L. "Node.js Design Patterns." Packt Publishing, 2020.
5. Mozilla Developer Network. "Callback function." MDN Web Docs.
6. Caolan McMahon. "async.js." github.com/caolan/async.
7. Node.js API. "util.promisify." nodejs.org.
8. Rust Documentation. "Closures." doc.rust-lang.org.
