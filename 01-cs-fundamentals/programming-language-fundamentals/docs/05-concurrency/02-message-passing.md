# Message Passing

> "Communicate by sending and receiving messages, not by sharing memory." A design principle for concurrent processing that fundamentally eliminates data races.

## Learning Objectives

- [ ] Understand the concepts, history, and theoretical foundations of message passing
- [ ] Implement channel-based communication (Go, Rust)
- [ ] Understand the design philosophy and implementation of the actor model (Erlang/Elixir, Akka)
- [ ] Know the formal background of CSP (Communicating Sequential Processes)
- [ ] Accurately judge when to use shared memory vs. message passing
- [ ] Implement key patterns such as pipeline, fan-out/fan-in
- [ ] Avoid failure patterns such as deadlocks and livelocks
- [ ] Understand the application of message passing in distributed systems


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [async/await (Asynchronous Programming)](./01-async-await.md)

---

## 1. Fundamental Concepts of Message Passing

### 1.1 Definition and Historical Background

Message Passing is a communication method where concurrently operating computational entities (processes, threads, actors, etc.) communicate solely by sending and receiving messages without sharing memory. This concept evolved from two independent research efforts in the 1970s: Carl Hewitt's Actor Model (1973) and Tony Hoare's CSP (Communicating Sequential Processes, 1978).

In the shared memory model, multiple threads read and write to the same memory space, requiring synchronization mechanisms such as mutexes and semaphores. In contrast, with message passing, each computational entity has its own memory space, and data exchange is performed through message copying or ownership transfer. This structurally eliminates data races.

```
Shared Memory Model:
  +-----------+     +----------------+     +-----------+
  | Thread A  |---->|  Shared Data   |<----| Thread B  |
  +-----------+  ^  +----------------+  ^  +-----------+
                 |                      |
           Mutex/Lock              Mutex/Lock

  Issues:
  - Deadlock: Lock ordering inconsistencies
  - Data Races: Forgotten locks, insufficient synchronization
  - Priority Inversion: Low-priority thread holds lock
  - Convoy Effect: Slow thread throttles everything

Message Passing Model:
  +-----------+  message   +-----------+  message   +-----------+
  | Process A |----------->| Process B |----------->| Process C |
  | [state a] |  channel   | [state b] |  channel   | [state c] |
  +-----------+            +-----------+            +-----------+

  Each process owns its own state. Communication is via messages only.
  Data races are structurally impossible.
```

### 1.2 Go's Proverb and Design Philosophy

The Go language's official documentation states the following proverb:

> "Do not communicate by sharing memory; share memory by communicating."

This means "instead of directly manipulating data from multiple goroutines, transfer data ownership through channels." By following the convention that the sender no longer touches the data after sending it through a channel, concurrent access problems are avoided.

### 1.3 Classification: Synchronous and Asynchronous

Message passing is broadly classified into two types based on send/receive timing.

```
Synchronous (Rendezvous):
  +--------+              +--------+
  | Sender |--- send ---->|Receiver|
  |        |   (block)    |        |
  | wait   |<-- ack  -----|  recv  |
  +--------+              +--------+
  The sender blocks until the receiver accepts.
  Corresponds to Go's unbuffered channels, Ada's rendezvous.

Asynchronous (Buffered):
  +--------+              +--------------+     +--------+
  | Sender |--- send ---->|  Message     |---->|Receiver|
  |        |  (non-block) |  Queue       |     |        |
  +--------+              +--------------+     +--------+
  The sender returns immediately. Messages accumulate in the queue.
  Corresponds to Erlang mailboxes, Go's buffered channels.
```

| Property | Synchronous | Asynchronous |
|----------|------------|--------------|
| Blocking on send | Waits until received | Returns immediately (except when buffer is full) |
| Latency | High (waiting cost) | Low (sender continues immediately) |
| Deadlock risk | High (potential for mutual waiting) | Low (but watch for buffer exhaustion) |
| Memory usage | Low | Requires buffer memory |
| Debugging ease | High (causality is clear) | Low (timing-dependent bugs) |
| Representative examples | Go unbuffered chan, Ada rendezvous | Erlang mailbox, Go buffered chan |
| Back-pressure | Occurs naturally | Requires explicit design |

---

## 2. Theoretical Foundations: CSP and the Actor Model

### 2.1 CSP (Communicating Sequential Processes)

CSP is a formal process algebra published by Tony Hoare in 1978. In CSP, a process is an independent entity performing sequential computation that communicates only through named channels.

The key characteristics of CSP are:

1. **Channels are first-class citizens**: Channels have names, are typed, and can be passed as variables
2. **Synchronous communication is the default**: Send and receive occur simultaneously (rendezvous)
3. **Selective communication**: `select`/`alt` constructs wait on multiple channels simultaneously
4. **Sequential composition**: Execution within a process is sequential

```
CSP Process Composition Operators:

  P ; Q        Sequential composition: Execute P then Q
  P || Q       Parallel composition: Execute P and Q concurrently
  P [] Q       External choice: Environment chooses P or Q
  P |~| Q      Internal choice: Process non-deterministically chooses P or Q

  Channel communication:
  c!v          Send value v on channel c
  c?x          Receive a value from channel c and bind to x

  Example: Buffer (size 1) in CSP notation
  BUFFER = left?x -> right!x -> BUFFER
```

Go's goroutines and channels are strongly influenced by CSP. Go's `select` statement corresponds to CSP's external choice.

### 2.2 The Actor Model

The Actor Model was proposed by Carl Hewitt in 1973 and systematized by Gul Agha in 1986. In the Actor Model, an actor is the basic unit of computation, and each actor has the following capabilities:

1. **Receive messages**: Take messages from an asynchronous mailbox
2. **Modify internal state**: Update its own state (inaccessible from outside)
3. **Send messages**: Send messages to other actors
4. **Create new actors**: Spawn child actors

```
Actor Model Structure:

  +-------------------------------------+
  |            Actor System             |
  |                                     |
  |  +----------+     +----------+     |
  |  | Actor A  | msg | Actor B  |     |
  |  |+--------+|---->|+--------+|     |
  |  ||Mailbox ||     ||Mailbox ||     |
  |  |+--------+|     |+--------+|     |
  |  ||Behavior||     ||Behavior||     |
  |  |+--------+|     |+--------+|     |
  |  || State  ||     || State  ||     |
  |  |+--------+|     |+--------+|     |
  |  +----------+     +----+-----+     |
  |        ^               | spawn     |
  |        | msg           v           |
  |  +-----+----+     +----------+     |
  |  | Actor D  |     | Actor C  |     |
  |  |+--------+|     |+--------+|     |
  |  ||Mailbox ||<----||Mailbox ||     |
  |  |+--------+| msg |+--------+|     |
  |  ||Behavior||     ||Behavior||     |
  |  |+--------+|     |+--------+|     |
  |  || State  ||     || State  ||     |
  |  |+--------+|     |+--------+|     |
  |  +----------+     +----------+     |
  |                                     |
  +-------------------------------------+

  Each actor is an independent computational unit with:
  (1) A mailbox (receive queue)
  (2) A behavior (message processing logic)
  (3) Internal state (not publicly accessible)
```

### 2.3 Comparison of CSP and the Actor Model

| Property | CSP | Actor Model |
|----------|-----|-------------|
| Communication method | Named channels (synchronous) | Direct addressing (asynchronous) |
| Identity target | Channels | Actors (addresses) |
| Buffering | None (synchronous by default) | Mailbox (unbounded queue) |
| Message ordering | FIFO per channel | Not guaranteed (FIFO between same sender) |
| Composability | Algebraic composition possible | Easy dynamic topology changes |
| Fault handling | Outside the model (handled separately) | Systematized via supervision trees (Erlang) |
| Representative implementations | Go, Clojure core.async | Erlang/OTP, Akka, Orleans |
| Theoretical strength | Formal verification (FDR etc.) | Location transparency, natural for distribution |
| Application domain | Structured concurrent processing | Large-scale distributed, fault-tolerant systems |

---

## 3. Go Channels: CSP in Practice

### 3.1 Basic Channel Operations

Go channels are a type-safe message passing mechanism. By restricting channel direction, you can enforce send-only or receive-only at the type level.

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    // === Unbuffered Channel (Synchronous) ===
    unbuffered := make(chan string)

    go func() {
        // Blocks until the receiver is ready
        unbuffered <- "hello"
        fmt.Println("Send complete") // Printed after reception
    }()

    time.Sleep(100 * time.Millisecond) // Confirm sender is blocked
    msg := <-unbuffered
    fmt.Println("Received:", msg)

    // === Buffered Channel (Asynchronous) ===
    buffered := make(chan int, 3) // Buffer up to 3 elements

    // Does not block as long as there is buffer space
    buffered <- 10
    buffered <- 20
    buffered <- 30
    // buffered <- 40 // Would block on full buffer (deadlock)

    fmt.Println(<-buffered) // 10 (FIFO)
    fmt.Println(<-buffered) // 20
    fmt.Println(<-buffered) // 30

    // === Channel Direction Constraints ===
    // chan<- int : send-only
    // <-chan int : receive-only
    producer := func(out chan<- int) {
        for i := 0; i < 5; i++ {
            out <- i
        }
        close(out) // Notify send completion
    }

    consumer := func(in <-chan int) {
        for v := range in { // Iterate until closed
            fmt.Printf("Received: %d\n", v)
        }
    }

    ch := make(chan int, 5)
    go producer(ch)
    consumer(ch) // Receive on the main goroutine
}
```

### 3.2 Multiplexing with the select Statement

Go's `select` statement waits on multiple channel operations simultaneously and non-deterministically selects one that is ready. This corresponds to CSP's external choice operator.

```go
package main

import (
    "context"
    "fmt"
    "math/rand"
    "time"
)

// Processing with timeout
func fetchWithTimeout(url string, timeout time.Duration) (string, error) {
    result := make(chan string, 1)
    errCh := make(chan error, 1)

    go func() {
        // Simulation: random delay
        delay := time.Duration(rand.Intn(500)) * time.Millisecond
        time.Sleep(delay)
        result <- fmt.Sprintf("Response from %s (took %v)", url, delay)
    }()

    select {
    case res := <-result:
        return res, nil
    case err := <-errCh:
        return "", err
    case <-time.After(timeout):
        return "", fmt.Errorf("timeout after %v", timeout)
    }
}

// First Response Wins
func queryMultipleServers(servers []string) string {
    result := make(chan string, len(servers))

    for _, server := range servers {
        go func(s string) {
            // Query each server
            delay := time.Duration(rand.Intn(300)) * time.Millisecond
            time.Sleep(delay)
            result <- fmt.Sprintf("Response from %s", s)
        }(server)
    }

    return <-result // Return the first response
}

// Context-based cancellation propagation
func longRunningTask(ctx context.Context) error {
    for i := 0; ; i++ {
        select {
        case <-ctx.Done():
            fmt.Printf("Task interrupted: %v\n", ctx.Err())
            return ctx.Err()
        default:
            fmt.Printf("Processing: step %d\n", i)
            time.Sleep(100 * time.Millisecond)
        }
    }
}

func main() {
    // Timeout example
    res, err := fetchWithTimeout("https://example.com", 200*time.Millisecond)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println(res)
    }

    // First response example
    servers := []string{"server-a", "server-b", "server-c"}
    fmt.Println(queryMultipleServers(servers))

    // Context cancellation example
    ctx, cancel := context.WithTimeout(context.Background(), 350*time.Millisecond)
    defer cancel()
    longRunningTask(ctx)
}
```

### 3.3 Worker Pool Pattern

The worker pool, a representative concurrency pattern, consists of a fixed number of worker goroutines that receive tasks from a job channel and process them.

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Job represents a task for workers to process
type Job struct {
    ID      int
    Payload string
}

// Result represents the processing result of a job
type Result struct {
    JobID    int
    Output   string
    Duration time.Duration
}

// worker receives work from the jobs channel and writes to the results channel
func worker(id int, jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
    defer wg.Done()
    for job := range jobs {
        start := time.Now()
        // Simulate processing
        time.Sleep(50 * time.Millisecond)
        output := fmt.Sprintf("Worker %d processed job %d: %s",
            id, job.ID, job.Payload)
        results <- Result{
            JobID:    job.ID,
            Output:   output,
            Duration: time.Since(start),
        }
    }
}

func main() {
    const numWorkers = 4
    const numJobs = 20

    jobs := make(chan Job, numJobs)
    results := make(chan Result, numJobs)

    // Start workers
    var wg sync.WaitGroup
    for w := 1; w <= numWorkers; w++ {
        wg.Add(1)
        go worker(w, jobs, results, &wg)
    }

    // Submit jobs
    for j := 1; j <= numJobs; j++ {
        jobs <- Job{ID: j, Payload: fmt.Sprintf("task-%d", j)}
    }
    close(jobs) // Notify that there are no more jobs

    // Wait for all workers to complete and close the results channel
    go func() {
        wg.Wait()
        close(results)
    }()

    // Collect results
    for result := range results {
        fmt.Printf("Job %2d: %s (%v)\n",
            result.JobID, result.Output, result.Duration)
    }
}
```

```
Worker Pool Data Flow:

  +-----------+     +----------------------------+     +------------+
  |           |     |      jobs channel          |     |            |
  | Producer  |---->| [job1][job2][job3]...      |     | Collector  |
  |           |     +------+---+---+-------------+     |            |
  +-----------+            |   |   |                   +----^-------+
                           v   v   v                        |
                      +----++----++----+                    |
                      | W1 || W2 || W3 |   Workers          |
                      +--+-++--+-++--+-+                    |
                         |     |     |                      |
                         v     v     v                      |
                      +----------------------------+        |
                      |    results channel         |--------+
                      | [res1][res2][res3]...      |
                      +----------------------------+

  Characteristics:
  - Fixed worker count controls resource consumption
  - Closing the jobs channel terminates all workers
  - WaitGroup detects when all workers are done
  - Back-pressure occurs naturally through channel buffers
```

### 3.4 Pipeline Pattern

A pipeline decomposes processing into stages, with each stage connected by channels. It has a structure similar to Unix pipes (`cmd1 | cmd2 | cmd3`).

```go
package main

import (
    "fmt"
    "math"
    "sync"
)

// generate sends slice elements to a channel (source stage)
func generate(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums {
            out <- n
        }
        close(out)
    }()
    return out
}

// square squares each input value (transformation stage)
func square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- n * n
        }
        close(out)
    }()
    return out
}

// filter passes only values meeting a condition (filter stage)
func filter(in <-chan int, pred func(int) bool) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            if pred(n) {
                out <- n
            }
        }
        close(out)
    }()
    return out
}

// fanOut distributes one input channel to n workers
func fanOut(in <-chan int, n int, process func(int) int) []<-chan int {
    outs := make([]<-chan int, n)
    for i := 0; i < n; i++ {
        out := make(chan int)
        outs[i] = out
        go func() {
            for v := range in {
                out <- process(v)
            }
            close(out)
        }()
    }
    return outs
}

// fanIn merges multiple channels into one
func fanIn(channels ...<-chan int) <-chan int {
    out := make(chan int)
    var wg sync.WaitGroup
    for _, ch := range channels {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for v := range c {
                out <- v
            }
        }(ch)
    }
    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

func main() {
    // Basic pipeline: generate -> square -> filter -> output
    isPrime := func(n int) bool {
        if n < 2 {
            return false
        }
        for i := 2; i <= int(math.Sqrt(float64(n))); i++ {
            if n%i == 0 {
                return false
            }
        }
        return true
    }

    pipeline := filter(
        square(generate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
        isPrime,
    )

    fmt.Println("Squares that are prime:")
    for v := range pipeline {
        fmt.Println(v) // Outputs 4, 9, 25, 49
    }
}
```

---

## 4. Rust Channels: Safety Through Ownership

### 4.1 Standard Library mpsc Channel

Rust's `std::sync::mpsc` (Multi-Producer, Single-Consumer) channel works with the ownership system to prevent data races at compile time.

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    // === Basic usage ===
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let data = String::from("Hello from thread!");
        tx.send(data).unwrap();
        // data has been moved, so it cannot be used here
        // println!("{}", data); // Compile error!
    });

    let received = rx.recv().unwrap();
    println!("Received: {}", received);

    // === Multiple producers ===
    let (tx, rx) = mpsc::channel();

    for i in 0..5 {
        let tx_clone = tx.clone();
        thread::spawn(move || {
            let msg = format!("Message from thread {}", i);
            tx_clone.send(msg).unwrap();
            thread::sleep(Duration::from_millis(100));
        });
    }
    drop(tx); // Drop the original tx (rx terminates when all senders are dropped)

    // Receive as an iterator (until all senders are dropped)
    for msg in rx {
        println!("{}", msg);
    }

    // === Synchronous channel (buffered) ===
    let (tx, rx) = mpsc::sync_channel(2); // Buffer size 2

    thread::spawn(move || {
        tx.send(1).unwrap(); // Returns immediately
        tx.send(2).unwrap(); // Returns immediately
        println!("Before buffer full");
        tx.send(3).unwrap(); // Buffer full, blocks until received
        println!("Third send complete");
    });

    thread::sleep(Duration::from_millis(500));
    println!("Starting reception");
    for val in rx {
        println!("Value: {}", val);
    }
}
```

### 4.2 crossbeam Channel: A Feature-Rich Alternative

The `crossbeam-channel` crate provides MPMC (Multi-Producer, Multi-Consumer) channels and a `select!` macro.

```rust
use crossbeam_channel::{bounded, unbounded, select, Receiver, Sender};
use std::thread;
use std::time::Duration;

// Request processing with timeout
fn request_with_timeout(timeout: Duration) -> Result<String, String> {
    let (tx, rx) = bounded(1);

    thread::spawn(move || {
        // Simulate processing
        thread::sleep(Duration::from_millis(200));
        let _ = tx.send("Processing complete".to_string());
    });

    select! {
        recv(rx) -> msg => msg.map_err(|e| e.to_string()),
        default(timeout) => Err("Timeout".to_string()),
    }
}

// Fan-in: Merge multiple sources into one
fn fan_in(receivers: Vec<Receiver<String>>) -> Receiver<String> {
    let (tx, rx) = unbounded();

    for r in receivers {
        let tx = tx.clone();
        thread::spawn(move || {
            for msg in r {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });
    }

    rx
}

fn main() {
    // Timeout example
    match request_with_timeout(Duration::from_millis(300)) {
        Ok(msg) => println!("Success: {}", msg),
        Err(e) => println!("Failure: {}", e),
    }

    // Select on multiple channels
    let (s1, r1) = bounded::<String>(0);
    let (s2, r2) = bounded::<String>(0);

    thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        s1.send("Channel 1".to_string()).unwrap();
    });

    thread::spawn(move || {
        thread::sleep(Duration::from_millis(200));
        s2.send("Channel 2".to_string()).unwrap();
    });

    // Wait on both channels
    for _ in 0..2 {
        select! {
            recv(r1) -> msg => println!("r1: {:?}", msg),
            recv(r2) -> msg => println!("r2: {:?}", msg),
        }
    }
}
```

---

## 5. Actor Model Implementations

### 5.1 Erlang/Elixir: Actors via OTP

Erlang/OTP is the most mature implementation of the actor model. Erlang processes are not OS threads but lightweight processes managed by the VM (approximately 300 bytes each), enabling millions of concurrent processes.

```elixir
# === Actor using GenServer (Elixir) ===
defmodule BankAccount do
  use GenServer

  # --- Client API ---
  def start_link(initial_balance) do
    GenServer.start_link(__MODULE__, initial_balance)
  end

  def deposit(account, amount) when amount > 0 do
    GenServer.call(account, {:deposit, amount})
  end

  def withdraw(account, amount) when amount > 0 do
    GenServer.call(account, {:withdraw, amount})
  end

  def balance(account) do
    GenServer.call(account, :balance)
  end

  # Asynchronous notification (no reply needed)
  def notify(account, message) do
    GenServer.cast(account, {:notify, message})
  end

  # --- Server Callbacks ---
  @impl true
  def init(initial_balance) do
    {:ok, %{balance: initial_balance, history: []}}
  end

  @impl true
  def handle_call({:deposit, amount}, _from, state) do
    new_balance = state.balance + amount
    new_state = %{
      balance: new_balance,
      history: [{:deposit, amount, new_balance} | state.history]
    }
    {:reply, {:ok, new_balance}, new_state}
  end

  @impl true
  def handle_call({:withdraw, amount}, _from, state) do
    if state.balance >= amount do
      new_balance = state.balance - amount
      new_state = %{
        balance: new_balance,
        history: [{:withdraw, amount, new_balance} | state.history]
      }
      {:reply, {:ok, new_balance}, new_state}
    else
      {:reply, {:error, :insufficient_funds}, state}
    end
  end

  @impl true
  def handle_call(:balance, _from, state) do
    {:reply, state.balance, state}
  end

  @impl true
  def handle_cast({:notify, message}, state) do
    IO.puts("Notification: #{message}")
    {:noreply, state}
  end
end

# Usage
{:ok, account} = BankAccount.start_link(1000)
{:ok, balance} = BankAccount.deposit(account, 500)
IO.puts("Balance: #{balance}")  # => Balance: 1500

{:ok, balance} = BankAccount.withdraw(account, 200)
IO.puts("Balance: #{balance}")  # => Balance: 1300

{:error, :insufficient_funds} = BankAccount.withdraw(account, 5000)
IO.puts("Balance: #{BankAccount.balance(account)}")  # => Balance: 1300
```

### 5.2 Erlang/OTP Supervision Trees

One of the greatest strengths of the actor model is the Supervision Tree, which can isolate faults and automatically recover.

```
Erlang/OTP Supervision Tree Structure:

        +--------------+
        |  Application |
        |  Supervisor  |
        +------+-------+
               |
       +-------+-------+
       v       v       v
  +--------++------++----------+
  | Worker || Sub  || Worker   |
  | A      || Sup  || C        |
  +--------++--+---++----------+
               |
          +----+----+
          v    v    v
      +----++----++----+
      | W1 || W2 || W3 |
      +----++----++----+

  Restart Strategies:
  - :one_for_one   : Restart only the crashed child
  - :one_for_all   : Restart all children when one crashes
  - :rest_for_one  : Restart the crashed child and all children after it

  "Let it crash" Philosophy:
  Instead of writing error handling inside the actor,
  let the actor crash and have the supervisor restart it.
  Simpler and more robust than defensive programming.
```

```elixir
# Supervision tree definition
defmodule MyApp.Supervisor do
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, :ok, opts)
  end

  @impl true
  def init(:ok) do
    children = [
      # Child process specifications
      {BankAccount, 0},
      {MyApp.Cache, []},
      {MyApp.EventBus, []}
    ]

    # one_for_one: Restart individually
    # max_restarts: Allow up to 3 restarts within 5 seconds
    Supervisor.init(children,
      strategy: :one_for_one,
      max_restarts: 3,
      max_seconds: 5
    )
  end
end
```

---

## 6. Key Message Passing Patterns

### 6.1 Pattern Overview

Message passing has recurring design patterns. Below is a summary of major patterns and their uses.

| Pattern | Description | Use Case | Language Example |
|---------|------------|----------|-----------------|
| Request-Reply | Send a message, then wait for a response | RPC-like synchronous processing | Go (chan pair), Erlang (call) |
| Fire-and-Forget | Send only, don't wait for a response | Logging, notifications | Erlang (cast), Akka (tell) |
| Pipeline | Connect stages in series | Data transformation flows | Go (chan chain), Unix pipe |
| Fan-Out/Fan-In | Distribute processing and aggregate results | Parallel batch processing | Go (multiple goroutines) |
| Publish-Subscribe | Broadcast to all subscribers | Event distribution | Erlang (pg), Redis Pub/Sub |
| Scatter-Gather | Query multiple sources, aggregate all results | Distributed search | Go (select + WaitGroup) |
| Router | Route messages based on conditions | Load balancing | Akka (Router), Go (select) |
| Dead Letter | Handle undeliverable messages | Error handling | Akka (DeadLetter), RabbitMQ |

### 6.2 Fan-Out / Fan-In Pattern in Detail

Fan-Out distributes work from a single source to multiple workers, and Fan-In merges the results from multiple workers into a single channel.

```
Fan-Out / Fan-In Pattern:

                Fan-Out                    Fan-In

  +--------+    +---------+              +---------+    +--------+
  |        |--->| Worker1 |--------------|         |    |        |
  |        |    +---------+              |         |    |        |
  | Source |    +---------+              | Merger  |--->| Sink   |
  |        |--->| Worker2 |--------------|         |    |        |
  |        |    +---------+              |         |    |        |
  |        |    +---------+              |         |    |        |
  |        |--->| Worker3 |--------------|         |    |        |
  +--------+    +---------+              +---------+    +--------+

  Data Flow:
  1. Source submits tasks to the jobs channel
  2. Multiple Workers take from the jobs channel (Fan-Out)
  3. Each Worker writes results to the results channel
  4. Merger aggregates all results (Fan-In)
  5. Sink consumes the final results

  Benefits:
  - Parallelize CPU-bound processing
  - Control throughput by adjusting worker count
  - Each worker is independent, isolating faults
```

### 6.3 Publish-Subscribe Pattern

In the Publish-Subscribe (Pub/Sub) pattern, a publisher sends messages to a topic, and all subscribers subscribed to that topic receive the messages.

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// EventBus implements topic-based Pub/Sub
type EventBus struct {
    mu          sync.RWMutex
    subscribers map[string][]chan interface{}
}

func NewEventBus() *EventBus {
    return &EventBus{
        subscribers: make(map[string][]chan interface{}),
    }
}

// Subscribe subscribes to a topic and returns a channel for receiving messages
func (eb *EventBus) Subscribe(topic string, bufferSize int) <-chan interface{} {
    eb.mu.Lock()
    defer eb.mu.Unlock()

    ch := make(chan interface{}, bufferSize)
    eb.subscribers[topic] = append(eb.subscribers[topic], ch)
    return ch
}

// Publish sends a message to all subscribers of a topic
func (eb *EventBus) Publish(topic string, msg interface{}) {
    eb.mu.RLock()
    defer eb.mu.RUnlock()

    for _, ch := range eb.subscribers[topic] {
        // Non-blocking send (discard if buffer is full)
        select {
        case ch <- msg:
        default:
            // Subscriber cannot keep up, discard message
            fmt.Printf("Warning: subscriber for topic %s cannot process messages\n", topic)
        }
    }
}

// Close closes all channels for a topic
func (eb *EventBus) Close(topic string) {
    eb.mu.Lock()
    defer eb.mu.Unlock()

    for _, ch := range eb.subscribers[topic] {
        close(ch)
    }
    delete(eb.subscribers, topic)
}

func main() {
    bus := NewEventBus()

    // Subscriber A: "orders" topic
    ordersA := bus.Subscribe("orders", 10)
    go func() {
        for msg := range ordersA {
            fmt.Printf("[Subscriber A] Order received: %v\n", msg)
        }
    }()

    // Subscriber B: "orders" topic
    ordersB := bus.Subscribe("orders", 10)
    go func() {
        for msg := range ordersB {
            fmt.Printf("[Subscriber B] Order received: %v\n", msg)
        }
    }()

    // Subscriber C: "logs" topic
    logs := bus.Subscribe("logs", 100)
    go func() {
        for msg := range logs {
            fmt.Printf("[Log] %v\n", msg)
        }
    }()

    // Publish
    bus.Publish("orders", map[string]interface{}{
        "id": 1, "item": "Book", "price": 2980,
    })
    bus.Publish("orders", map[string]interface{}{
        "id": 2, "item": "Pen", "price": 150,
    })
    bus.Publish("logs", "Order processing started")

    time.Sleep(100 * time.Millisecond)
    bus.Close("orders")
    bus.Close("logs")
}
```

### 6.4 Request-Reply Pattern (Go Implementation)

Request-Reply is a pattern where a reply channel is embedded in the request, and the caller waits for the response.

```go
package main

import (
    "fmt"
    "time"
)

// Request is a request containing a reply channel
type Request struct {
    Query    string
    ReplyCh  chan Response  // Channel to receive the response
}

// Response is the response to a request
type Response struct {
    Answer string
    Err    error
}

// server is a goroutine that receives and processes requests
func server(requests <-chan Request) {
    for req := range requests {
        // Process the request
        time.Sleep(50 * time.Millisecond) // Processing time
        req.ReplyCh <- Response{
            Answer: fmt.Sprintf("Answer to: %s", req.Query),
            Err:    nil,
        }
    }
}

// ask sends a request to the server and waits for the response
func ask(requests chan<- Request, query string, timeout time.Duration) (Response, error) {
    replyCh := make(chan Response, 1)
    requests <- Request{Query: query, ReplyCh: replyCh}

    select {
    case resp := <-replyCh:
        return resp, resp.Err
    case <-time.After(timeout):
        return Response{}, fmt.Errorf("request timed out")
    }
}

func main() {
    requests := make(chan Request, 10)
    go server(requests)

    resp, err := ask(requests, "What is message passing?", time.Second)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Response:", resp.Answer)
    }

    close(requests)
}
```

---

## 7. Choosing Between Shared Memory and Message Passing

### 7.1 Detailed Decision Criteria

Which model to use depends on the nature of the problem. Refer to the following decision flowchart.

```
Decision Flowchart:

  Are the processing units independent?
  |
  +-- Yes --> Is the cost of copying data acceptable?
  |           |
  |           +-- Yes --> Message passing recommended
  |           |           (channels or actors)
  |           |
  |           +-- No ---> Need to share large data structures
  |                       +-- Read-only -------> Share immutable data
  |                       +-- Read-write ------> Shared memory + RWMutex
  |
  +-- No ---> Is there a need to share tightly-coupled state?
              |
              +-- Yes --> Shared memory (Mutex/Atomic) recommended
              |           * Be very careful about race conditions
              |
              +-- No ---> Reconsider task decomposition
                          Can it be redesigned into independent units?
```

### 7.2 Detailed Comparison Table

| Evaluation Axis | Message Passing | Shared Memory |
|----------------|----------------|--------------|
| Data race safety | Structurally safe | Programmer's responsibility |
| Performance (low latency) | Message copy overhead | Direct access is fast |
| Performance (throughput) | High throughput with pipelines | May degrade under lock contention |
| Scalability | Easy horizontal scaling | Limited to a single machine |
| Debugging ease | Traceable via message tracing | Non-deterministic hard-to-reproduce bugs |
| Code complexity | Low when following patterns | Lock design tends to become complex |
| Memory efficiency | Increased consumption from message copies | Efficient through direct sharing |
| Fault isolation | Isolated at process boundaries | One thread's anomaly affects everything |
| Distribution support | Naturally extends over the network | Distributed shared memory is extremely difficult |
| Type safety | Static checking via channel types (Go, Rust) | Limited type safety for lock targets |
| Testability | Easy to inject and inspect messages | Hard to mock, timing-dependent |

### 7.3 Specific Scenarios Where Each Model Excels

**Scenarios where message passing is appropriate:**

- Inter-stage communication in ETL pipelines (Extract-Transform-Load)
- Inter-service communication in microservices (gRPC, message queues)
- Player session management in game servers (each player as an actor)
- Data collection and processing from IoT devices
- Request handling in web servers (one goroutine per request)
- Fault-tolerant telecom systems (Erlang/OTP's design target)

**Scenarios where shared memory is appropriate:**

- In-memory caches (`sync.Map` or `RwLock<HashMap>`)
- Atomic counters (metrics collection, rate limiting)
- Read-heavy config sharing (`RwLock` to enable concurrent reads)
- High-frequency trading order books (nanosecond latency requirements)
- Matrix operations in scientific computing (copying large data is impractical)

---

## 8. Distributed Message Passing

### 8.1 Message Passing Over the Network

A major advantage of message passing is that the transition from inter-process communication to network communication is natural. In the actor model, "Location Transparency" allows messages to be sent and received without awareness of whether actors are on the same machine or different machines.

```
Distributed Message Passing Architecture:

  Node A (Tokyo)                    Node B (Osaka)
  +-------------------+              +-------------------+
  |  +-----+          |    TCP/UDP   |          +-----+  |
  |  |Act-1|----------------------------->     |Act-3|  |
  |  +-----+          |   serialize  |          +-----+  |
  |  +-----+          |   + send     |          +-----+  |
  |  |Act-2|<-----------------------------     |Act-4|  |
  |  +-----+          |  deserialize |          +-----+  |
  |                    |   + deliver  |                   |
  +-------------------+              +-------------------+

  Network Considerations:
  1. Serialization (Protocol Buffers, JSON, MessagePack)
  2. Message delivery guarantees (at-most-once, at-least-once, exactly-once)
  3. Ordering guarantees (causal ordering, total ordering)
  4. Failure detection (heartbeats, timeouts)
  5. Network partitions (relationship with the CAP theorem)
```

### 8.2 Message Delivery Guarantees

There are three guarantee levels for message delivery in distributed systems:

| Guarantee Level | Description | Trade-off | Implementation Examples |
|----------------|------------|-----------|------------------------|
| At-most-once | Delivered at most once. No duplicates | Possible message loss | UDP, Erlang default |
| At-least-once | Delivered at least once. No loss | Possible duplicates. Idempotency required | TCP + ACK, Kafka |
| Exactly-once | Delivered exactly once | High cost. Complete realization is difficult | Kafka Transactions |

### 8.3 Relationship with Message Queues

Message queues (MQ) are the network-level extension of inter-process channels and actor mailboxes.

```
Positioning of Message Queues:

  Intra-process         Inter-process            Inter-machine
  +-----------+       +---------------+      +----------------+
  | Channel   |       | Unix Domain   |      | Message Queue  |
  | (Go)      |       | Socket        |      | (RabbitMQ,     |
  |           |       | Named Pipe    |      |  Kafka, NATS)  |
  +-----------+       +---------------+      +----------------+
  <- Low latency                        High reliability / persistence ->
  <- Simple                             Rich features ->

  Value-add of Message Queues:
  - Persistence: Save messages to disk
  - Retry: Automatic resend on delivery failure
  - Routing: Topics, pattern matching
  - Monitoring: Metrics for message volume, latency
  - Dead Letter: Divert unprocessable messages
```

---

## 9. Anti-Patterns and Pitfalls

### 9.1 Anti-Pattern 1: Overuse of Channels

A common anti-pattern in Go is trying to use channels even when a simple mutex would suffice.

```go
// === Anti-pattern: Channel-based counter ===
// Unnecessarily complex and poor performance
type ChannelCounter struct {
    incrementCh chan struct{}
    getCh       chan chan int
}

func NewChannelCounter() *ChannelCounter {
    c := &ChannelCounter{
        incrementCh: make(chan struct{}),
        getCh:       make(chan chan int),
    }
    go func() {
        count := 0
        for {
            select {
            case <-c.incrementCh:
                count++
            case replyCh := <-c.getCh:
                replyCh <- count
            }
        }
    }()
    return c
}

func (c *ChannelCounter) Increment() { c.incrementCh <- struct{}{} }
func (c *ChannelCounter) Get() int {
    replyCh := make(chan int, 1)
    c.getCh <- replyCh
    return <-replyCh
}

// === Recommended: Simple counter with sync/atomic ===
// Simple and fast
import "sync/atomic"

type AtomicCounter struct {
    count int64
}

func (c *AtomicCounter) Increment() { atomic.AddInt64(&c.count, 1) }
func (c *AtomicCounter) Get() int64 { return atomic.LoadInt64(&c.count) }

// Decision criteria:
// - Simple state protection -> sync.Mutex / sync/atomic
// - Data ownership transfer -> channels
// - Waiting on multiple async events -> channels + select
// - Pipeline processing -> channels
```

### 9.2 Anti-Pattern 2: Goroutine Leaks

When the receiving side disappears, the sending goroutine blocks forever, causing a memory leak (goroutine leak).

```go
// === Anti-pattern: Goroutine leak ===
func leakySearch(query string) string {
    results := make(chan string)

    // Query 3 backends
    go func() { results <- searchBackendA(query) }()
    go func() { results <- searchBackendB(query) }()
    go func() { results <- searchBackendC(query) }()

    // Return only the first result
    return <-results
    // Problem: The remaining 2 goroutines are blocked forever!
    // They try to send to results, but there is no receiver
}

// === Fixed version: Cancellation via Context and buffer ===
func safeSearch(ctx context.Context, query string) string {
    // Set buffer size equal to number of senders
    results := make(chan string, 3)

    ctx, cancel := context.WithCancel(ctx)
    defer cancel() // Cancel others once we get the first result

    search := func(backend func(context.Context, string) string) {
        select {
        case results <- backend(ctx, query):
        case <-ctx.Done():
            return // Exit on cancellation
        }
    }

    go search(searchBackendA)
    go search(searchBackendB)
    go search(searchBackendC)

    return <-results
}

// Example of a backend that respects cancellation
func searchBackendA(ctx context.Context, query string) string {
    select {
    case <-time.After(200 * time.Millisecond):
        return "Result from A"
    case <-ctx.Done():
        return "" // Return immediately on cancellation
    }
}
```

### 9.3 Anti-Pattern 3: Deadlock (Circular Wait)

Even with message passing, deadlock can occur when two processes wait for each other's messages.

```go
// === Anti-pattern: Deadlock between channels ===
func deadlock() {
    chA := make(chan int)
    chB := make(chan int)

    // Goroutine 1: Send to chA then receive from chB
    go func() {
        chA <- 1      // Blocks until chB receives
        val := <-chB  // Never reached
        fmt.Println(val)
    }()

    // Goroutine 2: Send to chB then receive from chA
    go func() {
        chB <- 2      // Blocks until chA receives
        val := <-chA  // Never reached
        fmt.Println(val)
    }()

    // Both goroutines are waiting for each other - deadlock!
}

// === Fixed version: Non-deterministic selection with select ===
func noDeadlock() {
    chA := make(chan int, 1)
    chB := make(chan int, 1)

    go func() {
        for i := 0; i < 5; i++ {
            select {
            case chA <- i:
            case val := <-chB:
                fmt.Println("G1 received:", val)
            }
        }
    }()

    go func() {
        for i := 0; i < 5; i++ {
            select {
            case chB <- i + 100:
            case val := <-chA:
                fmt.Println("G2 received:", val)
            }
        }
    }()

    time.Sleep(time.Second)
}
```

### 9.4 Anti-Pattern 4: Memory Exhaustion from Unbounded Buffers

In asynchronous message passing, if the buffer size is unbounded and the consumer cannot keep up, memory will be exhausted.

```go
// === Anti-pattern: Unbounded buffer (risk of memory exhaustion) ===
func unboundedBuffer() {
    // Go channels have fixed buffers, but
    // implementing an unbounded queue with slices risks memory exhaustion
    type UnboundedChan struct {
        in   chan int
        out  chan int
        buf  []int
    }
    // If production rate > consumption rate, buf grows indefinitely

    // === Recommended: Design back-pressure ===
    // Natural back-pressure with buffered channels
    ch := make(chan int, 100) // Producer blocks at 100 elements

    // Or explicit rate limiting
    // rate.NewLimiter(rate.Every(time.Millisecond), 100)
}
```

---

## 10. Performance Characteristics and Design Guidelines

### 10.1 Channel / Mailbox Costs

The cost of message passing primarily consists of the following elements:

```
Cost Structure of Message Passing:

  +--------------------------------------------+
  |       Cost of Sending a Message             |
  |                                             |
  |  1. Memory Allocation                       |
  |     - Creating the message object           |
  |     - Adding envelope (metadata)            |
  |                                             |
  |  2. Data Copy or Ownership Transfer         |
  |     - Pass by value: Deep copy of data      |
  |     - Ownership transfer: Pointer move      |
  |       (low cost)                            |
  |                                             |
  |  3. Synchronization Overhead                |
  |     - Acquiring the channel's internal lock |
  |     - Waking the goroutine (context switch) |
  |                                             |
  |  4. Scheduling                              |
  |     - Adding receiver goroutine to run queue|
  |     - M:N scheduler cost                    |
  +--------------------------------------------+

  Go Channel Internals:
  +---------------------------------+
  |          hchan struct           |
  |  +---------------------------+  |
  |  | buf: Ring buffer          |  |
  |  | qcount: Current count     |  |
  |  | dataqsiz: Buffer size     |  |
  |  | elemsize: Element size    |  |
  |  | sendx: Send index         |  |
  |  | recvx: Receive index      |  |
  |  | recvq: Receive wait queue |  |
  |  | sendq: Send wait queue    |  |
  |  | lock: mutex               |  |
  |  +---------------------------+  |
  +---------------------------------+
```

### 10.2 Summary of Design Guidelines

| Guideline | Description | Concrete Example |
|-----------|------------|-----------------|
| Clarify ownership | Sender should not touch data after sending | Go: Avoid slice operations after sending |
| Right-size buffers | Choose buffers that are neither too large nor too small | 2-3x the worker count is a good guideline |
| Set timeouts | Set timeouts on all channel operations | `select` + `time.After` / Context |
| Graceful shutdown | Propagate termination by closing channels | Sender closes, receiver detects with `range` |
| Design error propagation | Send errors as messages too | Send `Result[T]` types (value + error) |
| Monitoring and observability | Measure message counts and latency | Embed Prometheus metrics |

---

## 11. Comprehensive Comparison of Message Passing by Language

| Language / Runtime | Model | Channel Type | select Equivalent | Buffer | Ownership | Distribution Support |
|-------------------|-------|-------------|-------------------|--------|-----------|---------------------|
| Go | CSP | `chan T` | `select` | Any size | Convention-based | None standard (gRPC etc.) |
| Rust (std) | CSP | `mpsc::Sender/Receiver` | None (recv only) | Unbounded/Sync | Enforced by type system | None standard |
| Rust (crossbeam) | CSP | `Sender/Receiver` | `select!` | bounded/unbounded | Enforced by type system | None standard |
| Rust (tokio) | CSP | `mpsc`, `broadcast`, `watch` | `tokio::select!` | Any size | Enforced by type system | None standard |
| Erlang/OTP | Actor | Mailbox | `receive` (pattern match) | Unbounded | Copy (immutable values) | Native (Distributed Erlang) |
| Elixir | Actor | Mailbox | `receive` | Unbounded | Copy (immutable values) | Native (OTP) |
| Scala (Akka) | Actor | `ActorRef` | `receive` (pattern match) | Mailbox config | JVM GC | Akka Cluster |
| Kotlin | CSP | `Channel<T>` | `select` | Rendezvous/Buffered | Coroutine scope | None standard |
| Clojure | CSP | `core.async/chan` | `alts!` / `alts!!` | Any size | Immutable data structures | None standard |
| Swift | Actor | `actor` type | `async let` + `TaskGroup` | None (direct call) | Value types + Sendable | None standard |

---

## 12. Exercises

### 12.1 Basic Exercise: Temperature Conversion Pipeline

**Task**: Implement a program in Go that converts a slice of Fahrenheit temperatures to Celsius using a channel-based pipeline, outputting only those below freezing.

```
Requirements:
1. generate stage: Send Fahrenheit temperatures from a slice to a channel
2. convert stage: Convert Fahrenheit to Celsius (C = (F - 32) * 5/9)
3. filter stage: Pass only values below 0 degrees
4. Each stage must run as an independent goroutine
```

<details>
<summary>Solution (click to expand)</summary>

```go
package main

import "fmt"

func generateF(temps ...float64) <-chan float64 {
    out := make(chan float64)
    go func() {
        for _, t := range temps {
            out <- t
        }
        close(out)
    }()
    return out
}

func toCelsius(in <-chan float64) <-chan float64 {
    out := make(chan float64)
    go func() {
        for f := range in {
            out <- (f - 32.0) * 5.0 / 9.0
        }
        close(out)
    }()
    return out
}

func belowFreezing(in <-chan float64) <-chan float64 {
    out := make(chan float64)
    go func() {
        for c := range in {
            if c < 0 {
                out <- c
            }
        }
        close(out)
    }()
    return out
}

func main() {
    fahrenheits := []float64{32, 212, 0, -40, 50, 14, 20, 100}
    for c := range belowFreezing(toCelsius(generateF(fahrenheits...))) {
        fmt.Printf("%.1f C (below freezing)\n", c)
    }
}
```
</details>

### 12.2 Intermediate Exercise: Concurrent Web Crawler with Timeout

**Task**: Design a program that concurrently fetches multiple URLs with timeout control and aggregates the results.

```
Requirements:
1. Receive a list of URLs and fetch each URL in a separate goroutine
2. Set an individual timeout (3 seconds) for each fetch
3. Also set an overall timeout (10 seconds)
4. Return results as a slice of structs containing success/failure info
5. Implement cancellation propagation via Context
6. Limit maximum concurrent connections (semaphore pattern)
```

<details>
<summary>Solution skeleton (click to expand)</summary>

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type FetchResult struct {
    URL      string
    Body     string
    Duration time.Duration
    Err      error
}

func crawl(ctx context.Context, urls []string, maxConcurrency int,
    perURLTimeout, totalTimeout time.Duration) []FetchResult {

    ctx, cancel := context.WithTimeout(ctx, totalTimeout)
    defer cancel()

    results := make([]FetchResult, 0, len(urls))
    var mu sync.Mutex
    var wg sync.WaitGroup

    // Semaphore: Limit concurrency
    semaphore := make(chan struct{}, maxConcurrency)

    for _, url := range urls {
        wg.Add(1)
        go func(u string) {
            defer wg.Done()

            semaphore <- struct{}{} // Acquire semaphore
            defer func() { <-semaphore }()

            start := time.Now()
            urlCtx, urlCancel := context.WithTimeout(ctx, perURLTimeout)
            defer urlCancel()

            body, err := fetch(urlCtx, u) // fetch is a hypothetical function
            result := FetchResult{
                URL:      u,
                Body:     body,
                Duration: time.Since(start),
                Err:      err,
            }

            mu.Lock()
            results = append(results, result)
            mu.Unlock()
        }(url)
    }

    wg.Wait()
    return results
}

func fetch(ctx context.Context, url string) (string, error) {
    // Hypothetical implementation: fetch with HTTP client
    select {
    case <-time.After(100 * time.Millisecond):
        return fmt.Sprintf("Content of %s", url), nil
    case <-ctx.Done():
        return "", ctx.Err()
    }
}

func main() {
    urls := []string{
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
        "https://example.com/d",
    }
    results := crawl(context.Background(), urls, 2,
        3*time.Second, 10*time.Second)
    for _, r := range results {
        if r.Err != nil {
            fmt.Printf("[FAIL] %s: %v\n", r.URL, r.Err)
        } else {
            fmt.Printf("[OK]   %s (%v)\n", r.URL, r.Duration)
        }
    }
}
```
</details>

### 12.3 Advanced Exercise: Erlang/Elixir-Style Supervision Tree (Go Implementation)

**Task**: Implement a simplified Supervisor pattern in Go.

```
Requirements:
1. Define a Worker interface (Start, Stop methods)
2. Supervisor manages multiple Workers
3. When a Worker panics, the Supervisor automatically restarts it
4. Restart strategy: one-for-one (restart only the crashed Worker)
5. Set a restart limit (up to 3 times within 5 seconds)
6. When the limit is exceeded, the Supervisor itself reports an error
```

<details>
<summary>Design hints (click to expand)</summary>

```go
// Worker interface
type Worker interface {
    Name() string
    Run(ctx context.Context) error
}

// Supervisor structure
type Supervisor struct {
    workers      []Worker
    maxRestarts  int
    window       time.Duration
    restartLog   []time.Time // Restart timestamp history
}

// Implementation points:
// 1. Launch each Worker in a separate goroutine
// 2. Detect failures via errgroup or custom error channel
// 3. Catch panics with recover()
// 4. Restart decision: Check if restart count within window <= maxRestarts
// 5. Graceful shutdown via Context cancellation
```
</details>

---

## 13. Frequently Asked Questions (FAQ)

### Q1: Should I use channels or mutexes?

**A**: Based on Go's official Wiki ("Share Memory By Communicating"), the decision criteria are as follows:

- **Use channels**: When transferring data ownership, coordinating multiple async events, pipeline processing
- **Use mutexes**: When the sole purpose is protecting data, such as simple counters, caches, or configuration values

In general, use channels when "data flows" and mutexes when "protecting data." Go's standard library itself heavily uses the shared memory model with `sync.Map`, `sync.Pool`, etc., and is not exclusively channel-based.

### Q2: Can Erlang's actor model really handle millions of processes?

**A**: Erlang VM (BEAM) processes are not OS threads but lightweight entities managed by the VM. Each process starts with approximately 300 bytes of initial memory, and since heaps are independent, garbage collection is performed per-process. WhatsApp processed 2 million concurrent connections on a single server using Erlang as of 2012. However, as the number of processes increases, scheduler overhead grows, so actual scalability depends on the workload pattern.

### Q3: How does message passing relate to inter-service communication in microservices?

**A**: The concept of message passing applies at all scales: intra-process (channels), inter-process (IPC), and inter-machine (network). Asynchronous messaging in microservices (Kafka, RabbitMQ, NATS, etc.) can be seen as extending the actor model concept to the system architecture level. Each service corresponds to an actor, and the message queue corresponds to a mailbox. Synchronous gRPC/REST calls correspond to the Request-Reply pattern, and event-driven architecture corresponds to the Pub/Sub pattern.

### Q4: How are Go channels implemented internally?

**A**: Go channels are implemented as the `runtime.hchan` struct. Internally, they contain a ring buffer (for buffered channels), a send wait queue (list of sudogs), a receive wait queue, and a mutex. In other words, channels internally use locks. However, the key point is that programmers don't need to handle locks directly and can communicate safely through the high-level API of channel send/receive. For unbuffered channels, an optimization called "direct send" copies the value directly to the receiver's stack.

### Q5: Can deadlocks not occur with message passing?

**A**: Deadlocks can occur even with message passing. For example, when two processes try to send to each other's synchronous channels (circular wait), deadlock occurs. The Go runtime detects the state where all goroutines are blocked and reports `fatal error: all goroutines are asleep - deadlock!`, but it cannot detect when only some goroutines are deadlocked. Preventive measures include: (1) use buffered channels instead of synchronous ones, (2) use `select` + `default` for non-blocking sends, and (3) set timeouts.

### Q6: What is the practical difference between the actor model and CSP?

**A**: The biggest difference is the addressing method of communication. In CSP, you communicate by naming channels, so the sender doesn't need to know the receiver (it only needs to know the channel). In the actor model, you need to know the other actor's address (PID). The practical impact is that CSP is well-suited for building static data flows (pipelines, etc.), while the actor model is suited for systems with dynamically changing topologies (chat rooms, game player management, etc.). Additionally, Erlang's actor model has the significant advantage of built-in fault recovery through supervision trees.

---

## 14. Advanced Topics

### 14.1 Structured Concurrency

A concept gaining attention recently is "structured concurrency." Traditional goroutines and threads follow a "fire-and-forget" approach, where they can escape parental control. In structured concurrency, concurrent tasks are guaranteed to complete within their parent's scope.

```
Unstructured Concurrency (Traditional):
  main() {
      go task1()    // Running somewhere (unknown when it finishes)
      go task2()    // Potential for leaks
      // If main exits first, tasks become orphaned
  }

Structured Concurrency:
  main() {
      scope {
          task1()   // Guaranteed to complete within scope
          task2()   // Exceptions propagate to the scope
      }             // Waits for all tasks to complete
      // By the time we reach here, all tasks are done
  }
```

In Go, the `errgroup` package provides a form of structured concurrency. Kotlin's `coroutineScope`, Swift's `TaskGroup`, and Java's Project Loom `StructuredTaskScope` implement similar concepts.

### 14.2 Session Types

Session types are a theoretical technique for expressing communication protocols on channels at the type level. For example, expressing a protocol like "send an integer -> receive a string -> end" as a type and detecting protocol violations at compile time. While still a research-stage technology, it is experimentally available through Rust's `session-types` crate.

### 14.3 Reactive Streams

Reactive Streams is a specification for stream processing with asynchronous back-pressure. Java's `Flow API`, Reactor (Spring WebFlux), RxJava, and Akka Streams implement this specification. It applies message passing concepts to stream processing, providing a mechanism for producers to adjust their send rate to match the consumer's pace.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just from theory alone, but from actually writing code and observing its behavior.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge applied in practice?

The knowledge of this topic is frequently used in everyday development work. It is particularly important during code reviews and architecture design.

---

## Summary

| Model | Communication Method | Safety | Scalability | Fault Isolation | Representative Languages |
|-------|---------------------|--------|-------------|-----------------|------------------------|
| Channel (CSP) | Typed sync/async messages | High | Within single process | Limited | Go, Rust, Kotlin |
| Actor | Async mailbox | Highest | Distribution-ready | Supervision tree | Erlang, Elixir, Akka |
| Shared Memory | Mutex + shared variables | Programmer-dependent | Within single machine | None | C, C++, Java |
| Reactive Streams | Streams with back-pressure | High | Specialized for stream processing | Limited | Java, Scala, Kotlin |

The core of message passing is "clarifying data ownership and coordinating through communication." Go channels, Rust's ownership-based mpsc, and Erlang's actor model each realize this principle through different approaches. Which model to choose should be determined based on the application's characteristics (latency requirements, fault tolerance, scalability, team expertise).

---

## Recommended Next Reading


---

## References

1. Hoare, C. A. R. "Communicating Sequential Processes." Prentice Hall, 1985. (The definitive text on CSP. The PDF is freely available on the author's website)
2. Hewitt, C., Bishop, P., & Steiger, R. "A Universal Modular ACTOR Formalism for Artificial Intelligence." IJCAI, 1973. (The original paper on the Actor Model)
3. Armstrong, J. "Making Reliable Distributed Systems in the Presence of Software Errors." PhD Thesis, Royal Institute of Technology, Stockholm, 2003. (Joe Armstrong's doctoral thesis systematically explaining Erlang/OTP design philosophy)
4. Donovan, A. & Kernighan, B. "The Go Programming Language." Addison-Wesley, Ch.8-9, 2015. (Official reference for Go's goroutines and channels)
5. Klabnik, S. & Nichols, C. "The Rust Programming Language." No Starch Press, Ch.16, 2019. (Official guide to Rust's concurrency and ownership)
6. Agha, G. "Actors: A Model of Concurrent Computation in Distributed Systems." MIT Press, 1986. (Theoretical systematization of the Actor Model)
7. Go Blog. "Share Memory By Communicating." https://go.dev/blog/codelab-share (Official Go blog concurrency guide)
8. Erlang Documentation. "OTP Design Principles." https://www.erlang.org/doc/design_principles/ (Official Erlang/OTP design principles documentation)
