# CPU Scheduling

> Scheduling is the algorithm that determines "which process gets the CPU next."

## What You Will Learn in This Chapter

- [ ] Compare the major scheduling algorithms
- [ ] Explain the difference between preemptive and non-preemptive scheduling
- [ ] Understand how the Linux scheduler works
- [ ] Understand scheduling strategies in multicore environments
- [ ] Explain the requirements and implementation of real-time scheduling
- [ ] Calculate scheduling performance metrics


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the content in [Threads and Concurrency](./01-threads.md)

---

## 1. Scheduling Fundamentals

### 1.1 Why Scheduling Is Necessary

```
The Essence of Multiprogramming:

  A CPU can execute only one process at a time per core
  -> When multiple processes are simultaneously in the Ready state,
     we need to determine "fairly" and "efficiently" who runs first

  Modern systems have hundreds to thousands of processes/threads
  existing simultaneously
  -> The quality of scheduling determines the overall system
     responsiveness and throughput

CPU Bursts and I/O Bursts:

  Process execution pattern:
  [CPU execution] -> [I/O wait] -> [CPU execution] -> [I/O wait] -> ...

  CPU-bound process: Long CPU bursts (scientific computing, video encoding)
  I/O-bound process: Short CPU bursts (web servers, editors)

  +---------------------------------------------------+
  | CPU-bound:                                         |
  | ████████████░░████████████░░██████████████░░████   |
  |                                                    |
  | I/O-bound:                                         |
  | ██░░░░░░██░░░░░░██░░░░░░██░░░░░░██░░░░░░██░░░░   |
  |                                                    |
  | █ = CPU execution  ░ = I/O wait                    |
  +---------------------------------------------------+

  The scheduler should prioritize I/O-bound processes:
  -> I/O-bound processes quickly release the CPU and enter I/O wait
  -> While waiting, other processes can use the CPU
  -> Overall throughput improves
```

### 1.2 Scheduler Goals

```
Five Evaluation Metrics for Schedulers:

  1. CPU Utilization:
     Keep the CPU as busy as possible
     Target: 40%-90% (depending on system type)
     100% is overload, 0% is idle

  2. Throughput:
     Maximize the number of completed processes per unit time
     Particularly important in batch processing systems

  3. Turnaround Time:
     Total time from submission to completion
     Turnaround = Waiting time + Execution time + I/O time
     Important for batch processing

  4. Waiting Time:
     Total time spent waiting in the Ready state
     The only metric directly affected by the scheduling algorithm

  5. Response Time:
     Time from request submission to first response
     Most important for interactive systems (GUI, web servers)

  Trade-off Relationships:
  +-------------------------------------------+
  | CPU utilization ↑   <-> Response time ↑   |
  | Throughput ↑        <-> Individual         |
  |                         response time ↑    |
  | Fairness ↑          <-> Throughput ↓       |
  |                                            |
  | -> No universal algorithm exists           |
  | -> The optimal solution differs depending  |
  |    on system purpose                       |
  +-------------------------------------------+

  Priority Metrics by System Type:
  +--------------------+---------------------------+
  | Batch system       | Throughput, CPU util.     |
  | Interactive system | Response time, fairness   |
  | Real-time          | Deadline compliance       |
  | Server             | Throughput, response time |
  | Desktop            | Response time, fairness   |
  +--------------------+---------------------------+
```

### 1.3 Preemptive vs Non-Preemptive

```
Preemptive vs Non-Preemptive:

  Non-Preemptive (Cooperative):
    Waits until the process voluntarily releases the CPU
    Release timing:
    - Process termination
    - I/O request (transition to Blocked state)
    - yield() (voluntary surrender)

    Advantages: Fewer context switches, simpler implementation
    Disadvantages: One process can monopolize the CPU, poor responsiveness
    Examples: Windows 3.1, early Mac OS

  Preemptive:
    The OS forcibly switches the CPU via timer interrupts
    Switch timing:
    - Time quantum (time slice) expiration
    - A higher-priority process enters the Ready state
    - Interrupt occurrence

    Advantages: Prevents monopolization by one process, good responsiveness
    Disadvantages: Context switch overhead, increased synchronization complexity
    Examples: All modern OSes (Linux, Windows, macOS)

  Context Switch Cost:
  +-------------------------------------------+
  | 1. Save CPU state (registers, PC, etc.)   |
  | 2. Update PCB (Process Control Block)     |
  | 3. Switch memory map                      |
  | 4. Flush TLB                              |
  | 5. Cache cold start                       |
  |                                           |
  | Typical cost: 1-10 microseconds           |
  | Effective cost including cache misses:    |
  | tens of microseconds                      |
  +-------------------------------------------+
```

### 1.4 When Scheduling Occurs

```
Four Timings When the Dispatcher Operates:

  1. Running -> Waiting:
     When a process issues an I/O request or calls wait()
     -> Non-preemptive (voluntary surrender)

  2. Running -> Ready:
     When a timer interrupt occurs
     -> Preemptive (forced switch)

  3. Waiting -> Ready:
     When an I/O completion interrupt occurs
     -> Preemptive (switch depending on priority)

  4. Process Termination:
     When exit() is called
     -> Non-preemptive (inevitable)

  Dispatcher Processing:
  1. Save the context of the current process
  2. Scheduler selects the next process
  3. Restore the context of the selected process
  4. Switch to user mode
  5. Jump to the appropriate location in the selected process

  Dispatch Latency:
  The time it takes for the dispatcher to stop one process
  and start another
  -> Should be as short as possible (ideally below a few microseconds)
```

---

## 2. Major Scheduling Algorithms

### 2.1 FCFS (First-Come, First-Served)

```
FCFS: Execute in arrival order. FIFO queue. Simplest algorithm.

  Arrivals: P1(24ms) P2(3ms) P3(3ms)
  +------------------------+---+---+
  |        P1 (24ms)       |P2 |P3 |
  +------------------------+---+---+
  0                        24  27  30

  Waiting time: P1=0, P2=24, P3=27
  Average waiting time: (0 + 24 + 27) / 3 = 17ms

  Turnaround time: P1=24, P2=27, P3=30
  Average turnaround: (24 + 27 + 30) / 3 = 27ms

  If the arrival order were P2, P3, P1:
  +---+---+------------------------+
  |P2 |P3 |        P1 (24ms)       |
  +---+---+------------------------+
  0   3   6                        30

  Waiting time: P2=0, P3=3, P1=6
  Average waiting time: (0 + 3 + 6) / 3 = 3ms
  -> Performance varies greatly depending on arrival order!

Problem - Convoy Effect:
  When short processes queue behind a long process,
  the short processes are unnecessarily delayed

  CPU-bound process (long) occupies the CPU
  -> I/O-bound processes (short) all wait in the Ready queue
  -> I/O devices become idle
  -> CPU-bound process initiates I/O
  -> I/O-bound processes quickly process CPU bursts and enter I/O wait
  -> CPU becomes idle
  -> CPU-bound process returns after I/O completion
  -> Repeat...

  Result: Both CPU utilization and I/O device utilization decrease

Characteristics:
  - Non-preemptive
  - No starvation (all processes eventually execute)
  - Simplest implementation
  - Generally long average waiting time
  - Not suitable for non-batch systems
```

### 2.2 SJF (Shortest Job First)

```
SJF: Execute the process with the shortest execution time first

  Arrivals: P1(6ms) P2(8ms) P3(7ms) P4(3ms) (all arrive at time 0)
  +---+------+-------+--------+
  |P4 |  P1  |  P3   |   P2   |
  +---+------+-------+--------+
  0   3      9      16       24

  Waiting time: P4=0, P1=3, P3=9, P2=16
  Average waiting time: (0 + 3 + 9 + 16) / 4 = 7ms

  *With FCFS: (0 + 6 + 14 + 21) / 4 = 10.25ms

Proof of SJF Optimality (Intuitive Understanding):
  Given n jobs t1 <= t2 <= ... <= tn,
  Average waiting time when executed in SJF order:
  = (0 + t1 + (t1+t2) + ... + (t1+t2+...+t(n-1))) / n
  = Sum((n-i)*ti) / n  (i=1..n)

  Shorter jobs have larger coefficients (n-i) -> Total is minimized
  -> SJF is the optimal algorithm that minimizes average waiting time

Problems:
  1. Difficult to predict the next execution time (CPU burst length)
     -> Estimate using exponential averaging of past burst lengths
     tau(n+1) = alpha * t(n) + (1-alpha) * tau(n)
     alpha = 0: Use only past estimates
     alpha = 1: Use only the most recent measured value
     Typically alpha = 0.5

  2. Starvation:
     If short processes keep arriving,
     long processes may never execute

SRTF (Shortest Remaining Time First):
  Preemptive version of SJF
  -> When a new process arrives,
     switch if its remaining time is shorter than the currently running one

  Arrivals: P1(0, 8ms) P2(1, 4ms) P3(2, 9ms) P4(3, 5ms)
  +--+----+-----+----------+-------+
  |P1| P2 | P4  |   P1     |  P3   |
  +--+----+-----+----------+-------+
  0  1    5    10       17      26

  t=0: P1 starts (remaining 8)
  t=1: P2 arrives (remaining 4 < P1's remaining 7) -> Switch to P2
  t=2: P3 arrives (remaining 9 > P2's remaining 3) -> P2 continues
  t=3: P4 arrives (remaining 5 > P2's remaining 2) -> P2 continues
  t=5: P2 completes, P4(rem 5) vs P1(rem 7) vs P3(rem 9) -> P4
  t=10: P4 completes -> P1(remaining 7)
  t=17: P1 completes -> P3(remaining 9)
  t=26: P3 completes

  Average waiting time: (9 + 0 + 15 + 2) / 4 = 6.5ms
```

### 2.3 Round Robin

```
Round Robin: Execute alternately using a time quantum (time slice)

  Processes: P1(24ms) P2(3ms) P3(3ms), Quantum=4ms

  +----+---+---+----+----+----+----+----+----+
  | P1 |P2 |P3 | P1 | P1 | P1 | P1 | P1 |    |
  +----+---+---+----+----+----+----+----+----+
  0    4   7  10   14   18   22   26   30

  P1: 0-4(rem 20), 10-14(rem 16), 14-18(rem 12), 18-22(rem 8), 22-26(rem 4), 26-30(done)
  P2: 4-7(done, 3ms < 4ms quantum)
  P3: 7-10(done)

  Waiting time: P1 = (10-4) = 6ms (just the first wait is 6ms)
               * More precisely, P1's total waiting time = 30-24 = 6ms
               P2 = 4ms, P3 = 7ms
  Average waiting time: (6 + 4 + 7) / 3 = 5.67ms

  Response time: P1=0, P2=4, P3=7
  Average response time: (0 + 4 + 7) / 3 = 3.67ms
  -> Significantly improved over FCFS (response: 0, 24, 27 = 17ms)

Impact of Time Quantum:

  When the quantum is too short (e.g., 1ms):
  -> Context switches occur frequently
  -> Throughput significantly decreases due to overhead
  -> Effective execution time decreases

  When the quantum is too long (e.g., 100ms):
  -> Degenerates to FCFS
  -> Response time worsens

  Optimal quantum:
  +----------------------------------------------+
  | Rules of thumb:                               |
  | - Set so that 80% of CPU bursts complete      |
  |   within the quantum                          |
  | - Typically 10-100ms                          |
  | - At least 100x the context switch time       |
  |                                               |
  | If context switch = 10us                      |
  | -> Minimum quantum 1ms (100x = 1ms)           |
  | -> Practically around 10ms                    |
  +----------------------------------------------+

  Quantum vs Performance:
  +------------+----------+----------+-------------+
  | Quantum    | Response | Through- | Context     |
  |            | Time     | put      | Switches    |
  +------------+----------+----------+-------------+
  | 1ms        | Best     | Worst    | Very many   |
  | 10ms       | Good     | Good     | Moderate    |
  | 100ms      | Poor     | Good     | Few         |
  | Infinity   | FCFS     | FCFS     | Fewest      |
  +------------+----------+----------+-------------+

Characteristics:
  - Preemptive
  - Fair (all processes get equal CPU time)
  - No starvation
  - Worse average waiting time than SJF but better response time
  - Ideal for interactive systems
```

### 2.4 Priority Scheduling

```
Priority Scheduling: Execute the highest-priority process first

  Process  Burst  Priority (lower = higher)
  P1       10ms   3
  P2        1ms   1
  P3        2ms   4
  P4        1ms   5
  P5        5ms   2

  Execution order:
  +--+-----+----------+--+-+
  |P2| P5  |   P1     |P3|P4|
  +--+-----+----------+--+-+
  0  1     6         16 18 19

  Waiting time: P1=6, P2=0, P3=16, P4=18, P5=1
  Average waiting time: (6 + 0 + 16 + 18 + 1) / 5 = 8.2ms

Methods for Determining Priority:

  Static Priority:
  - Determined at process creation, does not change
  - Examples: nice value, user privilege level

  Dynamic Priority:
  - Varies based on execution behavior
  - Example: Priority increases upon return from I/O wait
  - Example: Priority decreases with CPU time consumption

  Priority Based on Internal Factors:
  - Memory requirements
  - Number of open files
  - CPU/I/O burst ratio

  Priority Based on External Factors:
  - Process importance
  - Payment amount (cloud)
  - Policy considerations

Starvation Problem:
  Low-priority processes may never execute

  Solution - Aging:
  Gradually increase priority based on waiting time

  Example: Increase priority by 1 level every 15 seconds
  +----------+----------+----------+
  | Time     | Initial  | Effective|
  |          | Priority | Priority |
  +----------+----------+----------+
  |  0 sec   |   20     |   20     |
  | 15 sec   |   20     |   19     |
  | 30 sec   |   20     |   18     |
  | ...      |   ...    |   ...    |
  |300 sec   |   20     |    0     | <- Highest priority
  +----------+----------+----------+

  -> Every process eventually executes (starvation prevention)

Priority Inversion:
  A phenomenon where a high-priority process waits for
  a low-priority process to release a lock

  H(high) -> Waiting for lock
  M(medium) -> Running (preempted L)
  L(low) -> Holding lock but preempted

  -> H effectively has lower priority than M!

  Solutions:
  1. Priority Inheritance:
     Lock holder inherits the waiter's priority
     L -> Temporarily gets H's priority -> Releases lock -> Original priority

  2. Priority Ceiling:
     Assign a ceiling priority to the lock
     Immediately elevate to that priority upon lock acquisition

  Real-world example: Mars Pathfinder (1997)
  -> System resets occurred frequently due to priority inversion
  -> Resolved by enabling priority inheritance in VxWorks
```

### 2.5 Multilevel Queue

```
Multilevel Queue:
  Multiple priority queues with priority ordering between queues

  +---------------------------------+
  | [Q0] Real-time processes        | <- Highest priority
  +---------------------------------+
  | [Q1] System processes           |
  +---------------------------------+
  | [Q2] Interactive processes      |
  +---------------------------------+
  | [Q3] Batch processes            | <- Lowest priority
  +---------------------------------+

  Each queue has its own scheduling algorithm:
  Q0: FCFS (real-time)
  Q1: Priority scheduling
  Q2: Round robin (short quantum)
  Q3: FCFS

  Inter-queue scheduling:
  - Fixed priority: Lower queues don't execute until upper queues are empty
    -> Risk of starvation for lower queues
  - Time slice allocation: Allocate CPU time proportionally to each queue
    Q0: 10%, Q1: 20%, Q2: 50%, Q3: 20%

  Problem: Processes remain in fixed queues
  -> Even if behavior changes, processes don't move to appropriate queues
```

### 2.6 Multilevel Feedback Queue (MLFQ)

```
Multilevel Feedback Queue:
  Multiple priority queues with dynamic process migration

  Rules:
  1. Execute processes in higher-priority queues first
  2. Within the same queue, use round robin
  3. Demote if the time quantum is fully consumed
  4. Promote (or maintain) upon return from I/O wait
  5. Periodically boost all processes to the highest priority (boost)

  High priority [Q0] --- Quantum 8ms
                ↓ Timeout (CPU-bound)
  Mid priority  [Q1] --- Quantum 16ms
                ↓ Timeout
  Low priority  [Q2] --- FCFS
                ↑ Promoted on I/O completion

  Operational Example:
  1. New processes enter Q0
  2. Complete or enter I/O wait within 8ms -> Stay in Q0
  3. Use up 8ms -> Demoted to Q1
  4. Use up 16ms in Q1 -> Demoted to Q2
  5. Periodic boost moves everyone back to Q0 (starvation prevention)

  Parameters to Configure:
  +----------------------------------------------+
  | Parameters to set:                            |
  | 1. Number of queues                           |
  | 2. Scheduling algorithm for each queue        |
  | 3. Time quantum for each queue                |
  | 4. Promotion conditions                       |
  | 5. Demotion conditions                        |
  | 6. Initial queue for new processes            |
  | 7. Boost period                               |
  |                                               |
  | -> Many parameters = most complex but most    |
  |    flexible                                   |
  | -> Adopted by many general-purpose OSes       |
  +----------------------------------------------+

  Gaming Prevention:
  Malicious processes intentionally issue I/O to
  maintain high priority

  Countermeasure: Accounting
  -> Track cumulative CPU usage time within the queue
  -> Demote when cumulative time exceeds threshold
  -> Count voluntary surrenders just before I/O as well
```

### 2.7 Algorithm Comparison

```
+--------------+-----------+-----------+-----------+----------+----------+
| Algorithm    | Preemptive| Starvation| Avg Wait  | Response | Impl.    |
|              |           |           | Time      | Time     | Complexity|
+--------------+-----------+-----------+-----------+----------+----------+
| FCFS         | No        | None      | Long      | Poor     | O Simple |
| SJF          | No        | Yes       | Shortest  | Medium   | D Predict|
| SRTF         | Yes       | Yes       | Shortest  | Good     | D Predict|
| RR           | Yes       | None      | Medium    | Good     | O Simple |
| Priority     | Both      | Yes       | Medium    | Good     | O        |
| MLFQ         | Yes       | None(*)   | Short     | Good     | X Complex|
+--------------+-----------+-----------+-----------+----------+----------+
* Starvation prevented by boosting
```

---

## 3. Linux Schedulers

### 3.1 History of Linux Schedulers

```
Evolution of Linux Schedulers:

  Linux 2.4 and earlier: O(n) Scheduler
  -> Scans all processes in the Run queue to select the next
  -> Slows down proportionally with process count
  -> Insufficient SMP support

  Linux 2.6.0-2.6.22: O(1) Scheduler
  -> Separate queues per priority level
  -> Alternates between Active and Expired arrays
  -> Selects next process in O(1)
  -> Uses heuristics to identify interactive processes
  -> Heuristics were inaccurate, causing fairness issues

  Linux 2.6.23-6.5: CFS (Completely Fair Scheduler)
  -> Inspired by Con Kolivas's SD/RSDL scheduler
  -> Developed by Ingo Molnar
  -> Fair scheduling using a red-black tree
  -> Stable scheduler used for over 10 years

  Linux 6.6+: EEVDF (Earliest Eligible Virtual Deadline First)
  -> Developed by Peter Zijlstra
  -> Improved version of CFS, based on virtual deadlines
  -> Better latency and workload isolation

  +----------+--------+----------------------+
  | Version  | Era    | Scheduler            |
  +----------+--------+----------------------+
  | <=2.4    | <=2003 | O(n) Scheduler       |
  | 2.6.0    | 2003   | O(1) Scheduler       |
  | 2.6.23   | 2007   | CFS                  |
  | 6.6      | 2023   | EEVDF                |
  +----------+--------+----------------------+
```

### 3.2 CFS (Completely Fair Scheduler)

```
CFS (Completely Fair Scheduler, 2007-2023):

  Principle: Distribute CPU time "fairly" to all processes
  -> Ideal: n processes each get 1/n of CPU time
  -> Reality: Since only one process runs at a time, perfect fairness
     is impossible
  -> "Run the process in the most unfair state" next

  Virtual Runtime (vruntime):
  -> vruntime increases as a process executes
  -> The process with the smallest vruntime is in the "most unfair state"
  -> Select that process next

  vruntime Calculation:
  vruntime += execution_time * (NICE_0_LOAD / weight)

  nice value -> weight -> vruntime increase rate
  +----------+----------+-------------------+
  | nice     | weight   | vruntime speed    |
  +----------+----------+-------------------+
  | -20      | 88761    | Slowest (most CPU time)   |
  |  -5      | 3121     | Slightly slow     |
  |   0      | 1024     | Baseline (1x)     |
  |   5      |  335     | Slightly fast     |
  |  19      |   15     | Fastest (least CPU time)  |
  +----------+----------+-------------------+

  A difference of 1 in nice value results in ~1.25x CPU time difference
  -> nice 0 vs nice 1: 55% vs 45% (~10% difference)
  -> nice 0 vs nice 5: 75% vs 25% (~3x difference)

  Data Structure: Red-Black Tree
  -> Retrieve the process with minimum vruntime in O(log n)
  -> Actually cached, so effectively O(1)

  +-------------------------------------+
  |          Red-Black Tree              |
  |         +---+                        |
  |         | 5 |                        |
  |        / \                           |
  |    +---+ +---+                       |
  |    | 3 | | 8 |                       |
  |   / \     \                          |
  | +---++---++---+                      |
  | | 1 || 4 || 9 |                      |
  | +---++---++---+                      |
  |  ^ Next to execute                   |
  | (minimum vruntime)                   |
  +-------------------------------------+

  The leftmost node (min vruntime) is cached in O(1)
  -> Effectively O(1) selection time

  Scheduling Granularity:
  -> sched_latency: Target time for all processes to run once
     (default 6ms x process count, max 24ms)
  -> sched_min_granularity: Minimum execution time (default 0.75ms)
  -> When nr_running > sched_latency / sched_min_granularity,
     each process is guaranteed sched_min_granularity

  Group Scheduling:
  -> CPU bandwidth control via cgroup v2
  -> User A and User B each get fair CPU usage
  -> Even User A (100 processes) vs User B (1 process) gets 50:50

  Example (CPU control with cgroups):
  +----------------------------------------------+
  | /sys/fs/cgroup/                               |
  | +-- group_a/                                  |
  | |   +-- cpu.max = "50000 100000"              |
  | |   |   -> Can use 50ms out of 100ms (50%)    |
  | |   +-- cgroup.procs = 1234, 1235...          |
  | +-- group_b/                                  |
  | |   +-- cpu.max = "25000 100000"              |
  | |   |   -> Can use 25ms out of 100ms (25%)    |
  | |   +-- cgroup.procs = 2345, 2346...          |
  | +-- cpu.max = "max 100000" (root)             |
  +----------------------------------------------+
```

### 3.3 EEVDF (Earliest Eligible Virtual Deadline First)

```
EEVDF (Earliest Eligible Virtual Deadline First, Linux 6.6+):

  Successor to CFS. Addresses CFS's shortcomings:

  CFS Issues:
  1. Insufficient handling of latency-sensitive processes
  2. Complex bonus handling for sleeping processes
  3. Non-intuitive relationship between nice values and latency
  4. Relies on many heuristics

  How EEVDF Works:
  -> Assigns a virtual deadline to each process
  -> Selects the eligible process with the earliest deadline

  Virtual deadline = Virtual start time + Time slice / weight

  Two Conditions:
  1. Eligible: vruntime is sufficiently small
     -> Not "over-borrowing" CPU time
  2. Earliest Deadline: Has the earliest virtual deadline
     -> Prioritizes processes that should complete sooner

  +------------------------------------------------+
  | CFS:                                            |
  | Selects "the process with the least CPU time"   |
  | -> No latency guarantee                         |
  |                                                 |
  | EEVDF:                                          |
  | Selects "the eligible process with the earliest |
  | deadline"                                       |
  | -> Latency is naturally controlled              |
  | -> No heuristics needed                         |
  +------------------------------------------------+

  Advantages:
  - Improved latency-sensitive workloads (audio, video)
  - Eliminates heuristics like sleeper bonus
  - Simpler and more predictable behavior
  - Time slice concept directly tied to nice values

  sched_ext (Linux 6.11+):
  -> Custom schedulers via BPF programs
  -> Define scheduling policies from user space
  -> Switch schedulers without rebooting
  -> Developed and used by Meta (Facebook)
  -> Optimized for specific use cases like gaming, servers, etc.
```

### 3.4 Real-Time Scheduling

```
Linux Real-Time Scheduling Classes:

  In priority order:
  1. SCHED_DEADLINE: Deadline-based (highest priority)
  2. SCHED_FIFO: Real-time FIFO
  3. SCHED_RR: Real-time round robin
  4. SCHED_OTHER (CFS/EEVDF): Normal processes (lowest priority)
  5. SCHED_BATCH: For batch processing (CPU-bound)
  6. SCHED_IDLE: Lowest priority (runs only when idle)

  Real-time processes always have priority over normal processes:
  +-----------------------------------------------+
  | Priority 0-99:   Real-time (FIFO/RR/DEADLINE) |
  | Priority 100-139: Normal processes (CFS/EEVDF)|
  |                   nice -20 -> priority 100     |
  |                   nice   0 -> priority 120     |
  |                   nice +19 -> priority 139     |
  +-----------------------------------------------+

  SCHED_FIFO:
  -> FIFO (first-come, first-served) within the same priority
  -> Does not release CPU until a higher priority arrives
  -> Runs until yield() or termination
  -> Danger: A runaway process can hang the system

  SCHED_RR:
  -> SCHED_FIFO with added time quantum
  -> Round robin within the same priority
  -> Immediately preempted by higher priority

  SCHED_DEADLINE:
  -> Based on EDF (Earliest Deadline First)
  -> Three parameters: Runtime, Deadline, Period
  -> Runtime: Guaranteed CPU time within each Period
  -> Deadline: The time by which Runtime must complete
  -> Period: Task repetition cycle
  -> Reserves bandwidth via CBS (Constant Bandwidth Server)

  Example: Audio processing (5ms/20ms period)
  +-------------------------------------------+
  | Period = 20ms                              |
  | Runtime = 5ms                              |
  | Deadline = 20ms                            |
  |                                            |
  | -> 5ms of CPU time guaranteed every 20ms   |
  | -> CPU bandwidth = 5/20 = 25%              |
  |                                            |
  | +----------+--------------------+          |
  | |Run(5ms)  |   Wait(15ms)      |-> Repeat |
  | +----------+--------------------+          |
  | 0          5                    20         |
  +-------------------------------------------+
```

```bash
# Real-time scheduling configuration examples

# Check current scheduling policy
chrt -p $$

# Run with SCHED_FIFO (priority 50)
sudo chrt -f 50 ./realtime_app

# Run with SCHED_RR (priority 30)
sudo chrt -r 30 ./realtime_app

# Run with SCHED_DEADLINE
# Runtime=5ms, Deadline=10ms, Period=20ms
sudo chrt -d --sched-runtime 5000000 \
             --sched-deadline 10000000 \
             --sched-period 20000000 \
             0 ./realtime_app

# Real-time bandwidth limit (runaway prevention)
# Default: Up to 950ms of real-time usage per 1 second
cat /proc/sys/kernel/sched_rt_runtime_us   # 950000
cat /proc/sys/kernel/sched_rt_period_us    # 1000000

# Disable real-time bandwidth limit (dangerous)
echo -1 > /proc/sys/kernel/sched_rt_runtime_us
```

### 3.5 Multicore Scheduling

```
Multicore Scheduling Challenges:

  1. Load Balancing:
     Distribute processes evenly across cores

     Push migration: Move processes from overloaded cores
     Pull migration: Idle cores pull processes

     +---------+   +---------+   +---------+
     | CPU 0   |   | CPU 1   |   | CPU 2   |
     | [P1,P2, |   | [P5]    |   | []      |
     |  P3,P4] |   |         |   | idle    |
     +----+----+   +---------+   +----+----+
          | push                      | pull
          +-----------> P3 <----------+

  2. Cache Affinity:
     Keep processes running on the same core as much as possible
     -> Data remains in L1/L2 cache
     -> Moving to another core causes cache misses

     Cache warm-up cost:
     L1 cache: ~1ns (a few cycles)
     L2 cache: ~5ns
     L3 cache: ~20ns
     Main memory: ~100ns

     -> Weigh the cost of migration vs the benefit of balance

  3. NUMA (Non-Uniform Memory Access) Support:
     Memory access speed depends on CPU location

     +------------------+  +------------------+
     | NUMA Node 0      |  | NUMA Node 1      |
     | +------+ +-----+ |  | +-----+ +------+ |
     | |CPU 0-3| |Mem A| |<->| |Mem B| |CPU 4-7| |
     | +------+ +-----+ |  | +-----+ +------+ |
     +------------------+  +------------------+
       Local access          Remote access
       ~100ns                ~300ns (3x slower)

     -> Processes should run on the NUMA node closest to their memory
     -> Linux can be controlled with numactl, libnuma

  4. SMT (Simultaneous Multi-Threading / Hyper-Threading):
     Two or more logical cores per physical core
     -> Shares execution units, so not 100% performance gain
     -> Typically 20-30% performance improvement
     -> Scheduler considers how to use SMT siblings

  Linux Scheduling Domains:
  +--------------------------------------+
  | SD_LEVEL_0: SMT (Hyper-Threading)    |
  | SD_LEVEL_1: MC (Multicore)           |
  | SD_LEVEL_2: PKG (Package/Socket)     |
  | SD_LEVEL_3: NUMA (NUMA Node)         |
  |                                      |
  | Each level has different load        |
  | balancing frequency and thresholds:  |
  | - SMT: Balance frequently            |
  | - NUMA: Balance infrequently (costly)|
  +--------------------------------------+
```

```bash
# Multicore scheduling in practice

# Set CPU affinity
taskset -p 0x3 $$              # Restrict to CPU 0,1 (bitmask)
taskset -c 0-3 ./my_program   # Run on CPU 0-3

# Check NUMA
numactl --hardware              # Display NUMA topology
numactl --membind=0 ./program  # Restrict to Node 0 memory
numactl --cpunodebind=0 --membind=0 ./program  # CPU + memory on Node 0

# Check scheduling statistics
cat /proc/schedstat             # Global statistics
cat /proc/$$/sched              # Per-process statistics
cat /proc/$$/status | grep -i cpu  # Execution core

# Per-CPU run queue length
cat /proc/schedstat | awk '{if(NR%3==1) print "CPU"(NR-1)/3": rq_len="$2}'

# Real-time CPU utilization monitoring
mpstat -P ALL 1                 # Per core
pidstat -u 1                    # Per process
htop                            # Interactive (CPU bar graph display)
```

---

## 4. Schedulers in Other OSes

### 4.1 Windows Scheduler

```
Windows Scheduler:

  Priority-based preemptive scheduling
  32 priority levels:
  - 0: Zero page thread (dedicated to memory zeroing)
  - 1-15: Normal processes (dynamic priority)
  - 16-31: Real-time processes (fixed priority)

  Priority class x Relative priority = Effective priority:
  +--------------+------+------+------+------+
  | Priority     | Low  |Normal| High | RT   |
  | Class        |      |      |      |      |
  +--------------+------+------+------+------+
  | Idle         |  1   |  1   |  1   | 16   |
  | Below Normal |  5   |  7   |  9   | 21   |
  | Normal       |  6   |  8   | 10   | 24   |
  | Above Normal |  7   |  9   | 11   | 25   |
  | High         | 11   | 13   | 15   | 26   |
  | Realtime     | 16   | 24   | 31   | 31   |
  +--------------+------+------+------+------+

  Dynamic Priority Boost:
  - Foreground window thread: +2
  - I/O completion: +1 to +8 (depending on device type)
  - Return from GUI input wait: +2
  -> After boost, gradually returns to base priority by 1

  Multimedia Class Scheduler Service (MMCSS):
  -> Automatically grants high priority to audio/video playback threads
  -> Periodically guarantees CPU time
```

### 4.2 macOS/iOS Scheduler

```
macOS/iOS (XNU Kernel) Scheduler:

  Multilevel feedback queue + decay scheduling

  Scheduling Bands:
  1. RT (Real-time): Fixed priority
  2. System High: Kernel threads
  3. System: Kernel threads
  4. User Initiated: User operations
  5. Default: General
  6. Utility: Long-running tasks
  7. Background: Background tasks
  8. Maintenance: Maintenance tasks

  Quality of Service (QoS):
  -> Apps specify a QoS class to communicate their intent
  -> The system automatically adjusts CPU, I/O, and timer priorities

  GCD (Grand Central Dispatch) Integration:
  -> dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{ ... });
  -> Scheduler and GCD cooperate to optimally allocate resources

  Apple Silicon (M1+) Efficiency/Performance Cores:
  -> P-core (Performance): High-priority tasks
  -> E-core (Efficiency): Low-priority / background tasks
  -> Scheduler automatically assigns based on QoS

  +--------------------------------------------+
  | QoS: User Interactive -> P-core priority    |
  | QoS: Background       -> E-core only       |
  | QoS: Default          -> Dynamic based on   |
  |                           load              |
  +--------------------------------------------+
```

---

## 5. Scheduling Applications

### 5.1 Scheduling in Container Environments

```
Containers and CPU Scheduling:

  Docker Containers:
  -> Control CPU bandwidth via cgroups v2
  -> Scheduler guarantees fairness at the cgroup level as well

  CPU Limit Configuration:
  # Limit to 50% of CPU
  docker run --cpus="0.5" nginx

  # Use only CPU 0 and 1
  docker run --cpuset-cpus="0,1" nginx

  # Set relative CPU weight
  docker run --cpu-shares=512 app_low    # Half of default
  docker run --cpu-shares=2048 app_high  # Double the default

  Kubernetes CPU Resources:
  +--------------------------------------+
  | resources:                           |
  |   requests:                          |
  |     cpu: "250m"    # 0.25 CPU guar.  |
  |   limits:                            |
  |     cpu: "500m"    # 0.5 CPU limit   |
  |                                      |
  | requests -> cpu.shares (CFS weight)  |
  | limits -> cpu.max (bandwidth limit)  |
  |                                      |
  | 1000m = 1 CPU = 100ms/100ms          |
  |  500m = 0.5 CPU = 50ms/100ms         |
  |  250m = 0.25 CPU = 25ms/100ms        |
  +--------------------------------------+

  CPU Throttling Issue:
  -> When limits are set, bursty CPU usage triggers throttling
  -> Heavy short-term CPU consumption forces waiting until the period ends
  -> Causes latency spikes
  -> Mitigation: Don't set limits, or set them with ample headroom
```

### 5.2 Database CPU Scheduling

```
Scheduling Considerations for Databases:

  PostgreSQL:
  -> Each connection is an independent process (fork)
  -> Relies on the OS scheduler
  -> hugepages recommended (reduce TLB misses)
  -> NUMA-local memory placement is important

  MySQL (InnoDB):
  -> Thread pool + worker threads
  -> Control thread count with innodb_thread_concurrency
  -> Relies on OS scheduler but also has internal lightweight scheduler

  Recommended Settings (Linux):
  +----------------------------------------------+
  | # CPU Affinity                                |
  | taskset -c 0-15 postgres                      |
  |                                               |
  | # NUMA Local Memory                           |
  | numactl --interleave=all postgres             |
  |                                               |
  | # Real-time priority (use with caution)       |
  | chrt -r 5 postgres                            |
  |                                               |
  | # CPU Governor (disable power saving)         |
  | cpupower frequency-set -g performance         |
  |                                               |
  | # Disable CPU C-States (reduce latency)       |
  | echo 0 > /dev/cpu_dma_latency                 |
  +----------------------------------------------+
```

### 5.3 Scheduling in Virtualized Environments

```
Virtual Machine Scheduling:

  Two Levels of Scheduling:
  1. Hypervisor schedules virtual CPUs
  2. Guest OS schedules processes on virtual CPUs

  +---------------------------------------------+
  | Guest OS A          Guest OS B               |
  | +-------------+    +-------------+           |
  | | P1 P2 P3    |    | P4 P5       |           |
  | | Scheduler   |    | Scheduler   |           |
  | | vCPU0 vCPU1 |    | vCPU0 vCPU1 |           |
  | +------+------+    +------+------+           |
  |        |                  |                   |
  | +------+------------------+------+            |
  | |    Hypervisor Scheduler        |            |
  | |    pCPU 0   pCPU 1   pCPU 2   |            |
  | +--------------------------------+            |
  +---------------------------------------------+

  Problem: Lock Holder Preemption
  -> A vCPU holding a lock in the guest OS gets
     preempted by the hypervisor
  -> Other vCPUs spin waiting for the lock
  -> Wastes CPU time

  Mitigations:
  - Pause loop exiting (Intel VT-x)
  - Halting (detect spinning and notify hypervisor)
  - Avoid CPU overcommit (vCPU <= pCPU)

  VMware vSphere:
  -> Proportional Share Based Scheduling
  -> Allocate resources via CPU shares, reservations, limits
  -> DRS (Distributed Resource Scheduler) for cross-cluster balancing
```

---

## 6. Practical Exercises

### Exercise 1: [Basic] - Manual Scheduling Calculation

```
Schedule the following processes using FCFS, SJF, and RR (q=2),
and compare the average waiting time and average turnaround time.

| Process | Arrival Time | Burst Time |
|---------|-------------|------------|
| P1      | 0           | 6          |
| P2      | 1           | 3          |
| P3      | 2           | 1          |
| P4      | 3           | 4          |

Solution (FCFS):
+----------+---+-+----+
|  P1(6)   |P2 |P3| P4 |
+----------+---+-+----+
0          6   9 10   14

Waiting time: P1=0, P2=6-1=5, P3=9-2=7, P4=10-3=7
Average waiting time: (0+5+7+7)/4 = 4.75ms

Turnaround: P1=6, P2=9-1=8, P3=10-2=8, P4=14-3=11
Average turnaround: (6+8+8+11)/4 = 8.25ms

Solution (SJF Non-Preemptive):
t=0: Only P1 -> Start P1
t=6: P1 completes. Ready: P2(3), P3(1), P4(4) -> P3 (shortest)
t=7: P3 completes. Ready: P2(3), P4(4) -> P2
t=10: P2 completes. -> P4
t=14: P4 completes

+----------+-+---+----+
|  P1(6)   |P3|P2 | P4 |
+----------+-+---+----+
0          6 7   10   14

Waiting time: P1=0, P3=7-2-1=4, P2=7-1=6, P4=10-3=7
-> Corrected: P1=0, P3=6-2=4, P2=7-1=6, P4=10-3=7
Average waiting time: (0+6+4+7)/4 = 4.25ms

Solution (RR, q=2):
t=0-2:   P1(rem 4)  -> Ready: P2, P1
t=2-4:   P2(rem 1)  -> Ready: P3, P4, P1, P2
t=4-5:   P3(done)   -> Ready: P4, P1, P2
t=5-7:   P4(rem 2)  -> Ready: P1, P2, P4
t=7-9:   P1(rem 2)  -> Ready: P2, P4, P1
t=9-10:  P2(done)   -> Ready: P4, P1
t=10-12: P4(done)   -> Ready: P1
t=12-14: P1(done)

Waiting time: P1=14-6=8, P2=10-1-3=6, P3=5-2-1=2, P4=12-3-4=5
Average waiting time: (8+6+2+5)/4 = 5.25ms
```

### Exercise 2: [Intermediate] - Observing the Linux Scheduler

```bash
# Check process scheduling information
chrt -p $$                    # Current scheduling policy
cat /proc/$$/sched            # Scheduling statistics (Linux)

# Change nice value
nice -n 10 sleep 100 &        # Start with nice value 10
renice -n 5 -p <PID>          # Change nice value of running process

# Monitor CPU utilization
top                           # Real-time monitoring
mpstat -P ALL 1               # Per-core utilization
pidstat -u 1                  # Per-process CPU utilization

# Measure scheduling latency
perf sched record sleep 10     # Record scheduling events for 10 seconds
perf sched latency             # Scheduling latency statistics

# Scheduler debug information
cat /proc/sched_debug          # Scheduler internal state
cat /sys/kernel/debug/sched/debug  # Debug info (debugfs)

# Check/modify CFS parameters
cat /proc/sys/kernel/sched_latency_ns          # Target latency
cat /proc/sys/kernel/sched_min_granularity_ns  # Minimum granularity
cat /proc/sys/kernel/sched_wakeup_granularity_ns  # Wakeup granularity
```

### Exercise 3: [Intermediate] - Experiencing CPU-Bound vs I/O-Bound Differences

```bash
#!/bin/bash
# Generate CPU-bound processes
cpu_bound() {
    echo "CPU-bound (PID: $$) starting with nice $1"
    nice -n $1 dd if=/dev/urandom of=/dev/null bs=1M count=1000 2>&1
}

# Generate I/O-bound processes
io_bound() {
    echo "I/O-bound (PID: $$) starting"
    for i in $(seq 1 1000); do
        dd if=/dev/zero of=/tmp/test_$$ bs=4K count=1 2>/dev/null
        rm -f /tmp/test_$$
    done
}

# Run simultaneously with different nice values and compare
time nice -n  0 python3 -c "sum(range(10**8))" &
time nice -n 19 python3 -c "sum(range(10**8))" &
wait

# Result: The process with nice 0 finishes first
```

### Exercise 4: [Advanced] - Container CPU Limits

```bash
# Test CPU limits with Docker
# Limit to 0.5 cores
docker run --rm --cpus="0.5" ubuntu:22.04 \
    bash -c "time dd if=/dev/urandom of=/dev/null bs=1M count=500"

# No CPU limit
docker run --rm ubuntu:22.04 \
    bash -c "time dd if=/dev/urandom of=/dev/null bs=1M count=500"

# Check CPU throttling
docker run -d --name test --cpus="0.2" ubuntu:22.04 sleep 3600
# Container cgroup statistics
cat /sys/fs/cgroup/system.slice/docker-$(docker inspect test --format '{{.Id}}').scope/cpu.stat
# nr_throttled: Number of times throttled
# throttled_time: Total time throttled

# Cleanup
docker rm -f test
```


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify config file path and format |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify execution user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check error messages**: Read the stack trace and identify the point of occurrence
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use logging output or debuggers to test hypotheses
5. **Fix and regression test**: After fixing, also run tests on related areas

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function input/output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return value: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception occurred: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Diagnostic procedure when performance issues occur:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Examine disk and network I/O status
4. **Check concurrent connections**: Examine connection pool status

| Issue Type | Diagnostic Tool | Mitigation |
|-----------|----------------|------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexes, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

Summary of decision criteria for making technology choices.

| Criterion | When Prioritized | When Compromisable |
|-----------|-----------------|-------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal information, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+---------------------------------------------------+
|          Architecture Selection Flow               |
+---------------------------------------------------+
|                                                    |
|  (1) Team size?                                    |
|    +- Small (1-5 people) -> Monolith               |
|    +- Large (10+ people) -> Go to (2)              |
|                                                    |
|  (2) Deployment frequency?                         |
|    +- Weekly or less -> Monolith + module split     |
|    +- Daily/multiple times -> Go to (3)            |
|                                                    |
|  (3) Team independence?                            |
|    +- High -> Microservices                        |
|    +- Medium -> Modular monolith                   |
|                                                    |
+---------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Cost**
- A quick short-term approach may become technical debt in the long run
- Conversely, over-engineering has high short-term costs and can delay the project

**2. Consistency vs Flexibility**
- A unified tech stack has lower learning costs
- Diverse technologies enable best-fit choices but increase operational costs

**3. Level of Abstraction**
- High abstraction offers better reusability but may make debugging harder
- Low abstraction is more intuitive but tends to produce code duplication

```python
# Design decision recording template
class ArchitectureDecisionRecord:
    """Create an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe the background and problem"""
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
        md += f"## Context\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "+" if c['type'] == 'positive' else "!"
            md += f"- {icon} {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## Practical Application Scenarios

### Scenario 1: MVP Development at a Startup

**Situation:** Need to quickly release a product with limited resources

**Approach:**
- Choose a simple architecture
- Focus on the minimum necessary features
- Automated tests only for critical paths
- Introduce monitoring from the start

**Lessons Learned:**
- Don't aim for perfection (YAGNI principle)
- Get user feedback early
- Manage technical debt consciously

### Scenario 2: Legacy System Modernization

**Situation:** Gradually modernize a system that has been running for 10+ years

**Approach:**
- Use the Strangler Fig pattern for gradual migration
- Create Characterization Tests first if existing tests are absent
- Use an API gateway to coexist old and new systems
- Perform data migration incrementally

| Phase | Work Content | Estimated Duration | Risk |
|-------|-------------|-------------------|------|
| 1. Investigation | Current state analysis, dependency mapping | 2-4 weeks | Low |
| 2. Foundation | CI/CD setup, test environment | 4-6 weeks | Low |
| 3. Migration Start | Migrate peripheral features first | 3-6 months | Medium |
| 4. Core Migration | Migrate core features | 6-12 months | High |
| 5. Completion | Decommission legacy system | 2-4 weeks | Medium |

### Scenario 3: Development with a Large Team

**Situation:** 50+ engineers developing the same product

**Approach:**
- Define boundaries clearly with Domain-Driven Design
- Set ownership per team
- Manage shared libraries using Inner Source approach
- Design API-first to minimize inter-team dependencies

```python
# Inter-team API contract definition
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
    """Inter-team API contract"""
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

**Situation:** A system requiring millisecond-level responses

**Optimization Points:**
1. Cache strategy (L1: In-memory, L2: Redis, L3: CDN)
2. Leverage async processing
3. Connection pooling
4. Query optimization and index design

| Optimization Method | Effect | Impl. Cost | Application |
|--------------------|--------|-----------|-------------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Async processing | Medium | Medium | I/O-heavy processing |
| DB optimization | High | High | When queries are slow |
| Code optimization | Low-Med | High | When CPU-bound |
---

## 7. FAQ

### Q1: What is the optimal time quantum value?

Generally 10-100ms. Too short increases context switch overhead; too long degrades responsiveness. Linux's default is about 6ms (derived from CFS's target latency, sched_latency_ns / number of processes). However, in EEVDF, the explicit time slice concept is naturally derived from the nice value's weight.

### Q2: What is the difference between a real-time OS and a general-purpose OS?

A Real-Time OS (RTOS) provides guarantees that "processing must be completed by the deadline." Hard real-time (aircraft control, ABS) means violations are fatal. Soft real-time (video playback, VoIP) means violations only degrade quality. Linux can achieve "near real-time" with the CONFIG_PREEMPT_RT patch (fully merged into mainline in Linux 6.12). Notable RTOSes include VxWorks, FreeRTOS, QNX, and Zephyr.

### Q3: What is the difference between nice values and priority?

The nice value is a relative "niceness" level set by the user (-20 to +19). A process with a higher nice value is "nicer to other processes" = yields its CPU time. The OS's internal priority is calculated from the nice value, but other factors (return from I/O wait, real-time priority, etc.) are also considered. In Linux, internal priorities 100-139 correspond to nice -20 to +19. Regular users can only increase nice values (root privileges are required to lower them).

### Q4: Why does Linux use CFS/EEVDF instead of MLFQ?

MLFQ requires many parameters (number of queues, quantum for each queue, promotion/demotion conditions, etc.) to be properly configured, and these depend on the workload. CFS is based on the single principle of "fairness," has fewer parameters, and performs well across a wide range of workloads. EEVDF further improves latency guarantees and eliminates heuristics. However, with sched_ext (Linux 6.11+), it's now possible to implement a custom MLFQ scheduler based on BPF.

### Q5: How much does scheduling affect performance?

For typical workloads, the impact of scheduler choice is a few percent to ~10%. However, the impact is significant in the following cases:
- Latency-sensitive workloads (games, audio, HFT): Differences of a few ms to tens of ms directly affect quality
- Large numbers of threads (hundreds to thousands): Scheduler overhead becomes noticeable
- NUMA environments: Improper scheduling can make memory access 3x slower
- Container environments: CPU throttling can increase 99th percentile latency by 10x or more

### Q6: How can you reduce context switches?

1. Match thread count to the number of CPU cores (avoid excessive thread creation)
2. Reduce thread count with I/O multiplexing (epoll/kqueue/io_uring)
3. Use user-space scheduling (goroutines, Tokio, green threads)
4. Set CPU affinity to prevent cross-core migration
5. Use real-time priority to prevent preemption (with caution)
6. Use appropriate lock granularity to reduce switches caused by lock contention

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is most important. Understanding deepens not just through theory but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 8. Summary

| Algorithm | Characteristics | Use Case |
|-----------|----------------|----------|
| FCFS | Simple, Convoy Effect | Batch processing |
| SJF/SRTF | Optimal but hard to predict | Theoretically important |
| Round Robin | Fair, good responsiveness | Interactive systems |
| Priority | Prioritizes important processes | Systems with real-time elements |
| MLFQ | Dynamic priority, most flexible | General-purpose OSes |
| CFS | Fair via virtual runtime | Linux 2.6.23-6.5 |
| EEVDF | Improved via virtual deadlines | Linux 6.6+ |

---

## Recommended Next Guides

---

## References
1. Silberschatz, A. et al. "Operating System Concepts." 10th Ed, Ch.5, 2018.
2. Love, R. "Linux Kernel Development." 3rd Ed, Ch.4, 2010.
3. Arpaci-Dusseau, R. H. & Arpaci-Dusseau, A. C. "Operating Systems: Three Easy Pieces." Ch.7-10, 2018.
4. Molnar, I. "CFS Scheduler." Linux Kernel Documentation, 2007.
5. Zijlstra, P. "EEVDF Scheduler." Linux Kernel Mailing List, 2023.
6. Corbet, J. "An EEVDF CPU scheduler for Linux." LWN.net, 2023.
7. Corbet, J. "Extensible scheduling with sched_ext." LWN.net, 2024.
8. Tanenbaum, A. S. & Bos, H. "Modern Operating Systems." 4th Ed, Ch.2, 2014.
