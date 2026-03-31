# Process Concepts and Management

> A process is a "running program" and is the most fundamental unit of execution managed by the OS.
> This chapter systematically covers the internal structure of processes, state transitions, the creation model (fork/exec), inter-process communication (IPC),
> and scheduling, providing a thorough understanding of the core mechanisms of the OS.

## What You Will Learn in This Chapter

- [ ] Explain the definition and components of a process (PCB, memory layout)
- [ ] Understand the 5-state process model and context switching
- [ ] Explain the design intent of the fork/exec model and the Copy-on-Write mechanism
- [ ] Compare and contrast CPU scheduling algorithms
- [ ] Implement inter-process communication (pipes, shared memory, signals)
- [ ] Understand the causes and remedies for zombie and orphan processes


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. What Is a Process?

### 1.1 The Fundamental Difference Between Programs and Processes

"Program" and "process" are distinctly different concepts. A program is an instruction sequence (binary file) stored on disk and is inherently a static entity. A process, on the other hand, refers to the dynamic entity in which that program has been loaded into memory and is operating with a CPU execution context (register values, program counter, etc.).

This relationship is similar to that between a recipe (program) and a dish being cooked (process). Just as you can simultaneously cook multiple dishes from the same recipe, multiple processes can be created simultaneously from a single program.

```
Program vs Process:

  Program: An executable file on disk (static)
    - /usr/bin/python3, /usr/bin/bash, etc.
    - An ELF binary or script with execute permissions
    - Does not consume CPU time by itself

  Process: A program loaded into memory and currently executing (dynamic)
    - Uniquely identified by a PID (Process ID)
    - Has its own memory space (virtual address space)
    - Has state such as current CPU register values, open files, etc.

  One program -> can become multiple processes
  (Example: each Chrome tab is an independent process)
  (Example: multiple users running bash simultaneously)

  The reverse does not hold:
  One process always corresponds to one program
  (Although exec() can transform it into another program mid-execution,
   at any given moment it is executing a single program)
```

### 1.2 Process Memory Layout

Each process has an independent virtual address space. This virtual address space is mapped to physical memory by the kernel's Memory Management Unit (MMU). The virtual address space exists to provide memory protection between processes and to enable the use of an address space larger than physical memory.

```
Virtual address space of a process (Linux x86-64):

  +----------------------------------------+ 0xFFFFFFFFFFFFFFFF
  |         Kernel space                   | <- Not directly accessible
  |     (shared across all processes)      |   from user processes
  |                                        |   (requires privilege level 0)
  +----------------------------------------+ 0x00007FFFFFFFFFFF
  |                                        |   (canonical boundary)
  |         Stack     | growth direction   | <- Local variables,
  |    +--------------------+              |   function arguments,
  |    | main() frame       |              |   return addresses
  |    +--------------------+              |
  |    | func_a() frame     |              |   Stack size limit:
  |    +--------------------+              |   typically 8MB (ulimit -s)
  |    | func_b() frame     |              |
  |    +--------------------+              |
  |              |                         |
  |                                        |
  |         (unused region)                | <- Guard zone between
  |                                        |   stack and heap
  |              ^                         |
  |    +--------------------+              |
  |    | Dynamically allocated |            |
  |    | memory via malloc()   |            |
  |    +--------------------+              |
  |         Heap       ^ growth direction  | <- Managed by malloc/free,
  |                                        |   new/delete
  +----------------------------------------+
  |         BSS segment                    | <- Uninitialized global variables
  |    (zero-initialized)                  |   int count; // BSS
  +----------------------------------------+
  |         Data segment                   | <- Initialized global variables
  |                                        |   int max = 100; // Data
  +----------------------------------------+
  |         Text segment                   | <- Machine instructions of the program
  |    (read-only, executable)             |   Cannot be modified (SEGV prevention)
  +----------------------------------------+ 0x0000000000400000
                                              (typical start address)

  Why separate BSS and Data?
  -> BSS only needs to hold the information "fill with zeros,"
    so the size of the executable can be reduced.
    Example: int arr[1000000]; goes into BSS, and the executable
    only records the instruction "zero-initialize 4MB."
```

### 1.3 PCB (Process Control Block)

The PCB (Process Control Block) is the core data structure used by the OS to manage processes. During a context switch, the CPU state of the current process is saved to its PCB, and the CPU state is restored from the next process's PCB. Without the PCB, the OS could not track process states, and multitasking would be impossible.

```
PCB (Process Control Block) structure:

  +-----------------------------------------------------+
  | PCB (in Linux: the task_struct structure)             |
  +-----------------------------------------------------+
  |                                                     |
  |  [Identification Information]                        |
  |  +-- PID (Process ID): unique integer value          |
  |  +-- PPID (Parent Process ID)                        |
  |  +-- UID/GID (Owner/Group)                           |
  |  +-- Session ID, Process Group ID                    |
  |                                                     |
  |  [CPU Context]                                       |
  |  +-- Program Counter (PC/RIP)                        |
  |  +-- Stack Pointer (SP/RSP)                          |
  |  +-- General-purpose Registers (RAX, RBX, RCX, ...)  |
  |  +-- Flags Register (RFLAGS)                         |
  |  +-- FPU/SSE/AVX Registers                           |
  |                                                     |
  |  [Scheduling Information]                            |
  |  +-- Process State (Running/Ready/Blocked/...)       |
  |  +-- Priority (nice value: -20 to +19)               |
  |  +-- Scheduling Policy (CFS/RT/FIFO)                 |
  |  +-- CPU Usage Time Statistics                       |
  |                                                     |
  |  [Memory Management Information]                     |
  |  +-- Page Table Base Address (CR3)                   |
  |  +-- List of Virtual Memory Areas (vm_area_struct)   |
  |  +-- Code/Data/Heap/Stack Boundaries                 |
  |  +-- Shared Library Mapping Information              |
  |                                                     |
  |  [I/O and File Information]                          |
  |  +-- File Descriptor Table                           |
  |  |   fd 0 -> stdin                                  |
  |  |   fd 1 -> stdout                                 |
  |  |   fd 2 -> stderr                                 |
  |  |   fd 3 -> /tmp/data.txt (example)                |
  |  +-- Current Directory                               |
  |  +-- umask                                           |
  |                                                     |
  |  [Signal Information]                                |
  |  +-- Pending Signal Mask                             |
  |  +-- Signal Handler Table                            |
  |  +-- Blocked Signal Mask                             |
  |                                                     |
  +-----------------------------------------------------+

  Size of task_struct in Linux:
  The expected size is several KB to around a dozen KB.
  The kernel efficiently manages this using the slab allocator.
```

### 1.4 Code Example 1: Retrieving Process Information (C)

The following C program retrieves and displays basic information about its own process. `getpid()`, `getppid()`, and `getuid()` are POSIX system calls that return information stored in the PCB.

```c
/* process_info.c
 * Compile: gcc -Wall -o process_info process_info.c
 * Run: ./process_info
 *
 * Purpose: Retrieve the basic attributes of a process (PID, PPID, UID, etc.)
 *          and confirm a subset of the information a process holds.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <limits.h>

int global_initialized = 42;  /* Placed in the Data segment */
int global_uninitialized;     /* Placed in the BSS segment */

int main(void)
{
    int stack_variable = 100;               /* Placed on the stack */
    int *heap_variable = malloc(sizeof(int)); /* Placed on the heap */
    if (heap_variable == NULL) {
        perror("malloc");
        return 1;
    }
    *heap_variable = 200;

    printf("=== Basic Process Information ===\n");
    printf("PID  (Process ID):        %d\n", getpid());
    printf("PPID (Parent Process ID): %d\n", getppid());
    printf("UID  (User ID):           %d\n", getuid());
    printf("GID  (Group ID):          %d\n", getgid());
    printf("SID  (Session ID):        %d\n", getsid(0));

    printf("\n=== Memory Address Verification ===\n");
    printf("Text (main function):        %p\n", (void *)main);
    printf("Data (initialized):          %p\n", (void *)&global_initialized);
    printf("BSS  (uninitialized):        %p\n", (void *)&global_uninitialized);
    printf("Heap (malloc):               %p\n", (void *)heap_variable);
    printf("Stack (local variable):      %p\n", (void *)&stack_variable);

    /* Verify the size ordering of addresses:
     * Text < Data < BSS < Heap < ... < Stack
     * is the typical layout we can confirm */
    printf("\n=== Address Order Verification ===\n");
    if ((void *)main < (void *)&global_initialized)
        printf("Text < Data: Normal layout\n");
    if ((void *)&global_uninitialized < (void *)heap_variable)
        printf("BSS < Heap: Normal layout\n");
    if ((void *)heap_variable < (void *)&stack_variable)
        printf("Heap < Stack: Normal layout\n");

    /* Check resource limits */
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
        printf("\n=== Resource Limits ===\n");
        printf("Max open files: soft=%lu, hard=%lu\n",
               (unsigned long)rl.rlim_cur, (unsigned long)rl.rlim_max);
    }
    if (getrlimit(RLIMIT_STACK, &rl) == 0) {
        printf("Stack size:     soft=%lu bytes (%lu MB)\n",
               (unsigned long)rl.rlim_cur,
               (unsigned long)rl.rlim_cur / (1024 * 1024));
    }

    free(heap_variable);
    return 0;
}
```

Expected output:
```
=== Basic Process Information ===
PID  (Process ID):        12345
PPID (Parent Process ID): 12300
UID  (User ID):           1000
GID  (Group ID):          1000
SID  (Session ID):        12200

=== Memory Address Verification ===
Text (main function):        0x401196
Data (initialized):          0x404030
BSS  (uninitialized):        0x404038
Heap (malloc):               0x1a3b2a0
Stack (local variable):      0x7ffd5a3c1e4c

=== Address Order Verification ===
Text < Data: Normal layout
BSS < Heap: Normal layout
Heap < Stack: Normal layout

=== Resource Limits ===
Max open files: soft=1024, hard=1048576
Stack size:     soft=8388608 bytes (8 MB)
```

---

## 2. Process State Transitions

### 2.1 The 5-State Model

A process transitions through multiple states from creation to termination. The most basic model is the following 5-state model. To understand why five states are necessary, consider the purpose of each state.

- **New (Created)**: The process is being created. The OS is allocating a PCB and preparing the memory space. This state exists to handle cases where process creation does not complete immediately (e.g., resource shortage).
- **Ready (Runnable)**: The process is waiting for CPU allocation. It has all the resources needed for execution, but is waiting because another process is currently using the CPU.
- **Running (Executing)**: The process is executing instructions on the CPU. In a single-core system, at most one process can be in the Running state at any given moment. In a multi-core system, as many processes as the number of cores can be in the Running state simultaneously.
- **Blocked (Waiting)**: The process is waiting for I/O completion or an event to occur. For example, a process that has requested a disk read is moved to this state because it does not need the CPU until the data is ready. Without the Blocked state, processes waiting for I/O would needlessly occupy the CPU.
- **Terminated (Exited)**: Execution has completed. The OS is reclaiming resources. Even after exit() is called, the process remains in this state until the parent calls wait().

```
5-State Model transition diagram:

  +-------+    admit     +-------+
  |  New  |------------->| Ready |<-------------------+
  +-------+              +---+---+                    |
                      dispatch |                      | preempt
                   (scheduler  |                      | (time slice
                     selects)  v                      |  expired/priority)
                        +----------+                  |
                        | Running  |------------------+
                        +----+--+--+
                   I/O req   |  |  exit()/
                   /event    |  |  signal
                   wait      |  |
                            v  v
                  +---------+  +------------+
                  | Blocked |  | Terminated |
                  +----+----+  +------------+
                I/O    |
                done   |
                /event |
                occurs |
                       v
                  +---------+
                  |  Ready  |
                  +---------+

  Important: There is no direct transition from Blocked -> Running.
  A process whose I/O has completed first moves to the Ready state
  and enters the Running state only when selected by the scheduler.
  This design allows the scheduler to centrally manage CPU allocation.
```

### 2.2 The 7-State Model (With Swapping Support)

When physical memory is insufficient, the OS evacuates parts of processes to disk (swap space). To represent this mechanism, the 7-state model adds two states to the 5-state model.

```
7-State Model (with swapping support):

  +-------+         +-------+
  |  New  |-------->| Ready |<------------------------+
  +-------+         +--+--+-+                         |
                       |  |                            | preempt
                       |  |  swap out                  |
                       |  v                            |
                       |  +----------------+           |
                       |  | Ready/Suspend  |           |
                       |  |  (on disk)     |           |
                       |  +-------+--------+           |
                       |     swap in |                  |
                       |          v                     |
                  dispatch|  +-------+                 |
                       v  |  | Ready |                 |
                    +----------+                       |
                    | Running  |------------- ---------+
                    +--+---+--+
                       |   |
              I/O req  |   | exit()
                      v   v
              +---------+ +------------+
              | Blocked | | Terminated |
              +--+--+---+ +------------+
          I/O    |  | swap out
          done   |  |
                 v  v
          +-------+ +------------------+
          | Ready | | Blocked/Suspend  |
          +-------+ |  (on disk)       |
                    +------+-----------+
                    I/O    |
                    done   |
                           v
                    +----------------+
                    | Ready/Suspend  |
                    +----------------+

  Ready/Suspend:    Runnable but swapped out from memory
  Blocked/Suspend:  Waiting for I/O and swapped out

  Why is Blocked/Suspend necessary?
  -> When memory is insufficient, Blocked processes will not execute soon,
    making them optimal candidates for swapping out. After I/O completion,
    they move to Ready/Suspend and are swapped in as needed.
```

### 2.3 Context Switch Details

A context switch is the operation of switching the process currently executing on the CPU. Without this operation, multitasking cannot be achieved, but since it is pure overhead, it needs to be executed quickly.

```
Context switch procedure:

  Time ->

  Process A (Running)          Kernel              Process B (Ready)
  -------------------          ------              -----------------
       |                         |                     |
  [1] Interrupt/                 |                     |
      system call occurs         |                     |
       |                         |                     |
       |  ---- trap -------->    |                     |
       |                    [2] Save Process A's       |
       |                        CPU context            |
       |                        to PCB-A               |
       |                        - General registers    |
       |                        - Program counter      |
       |                        - Stack pointer        |
       |                        - RFLAGS               |
       |                        - FPU/SSE registers    |
       |                         |                     |
       |                    [3] Scheduler selects      |
       |                        the next process       |
       |                        -> selects Process B   |
       |                         |                     |
       |                    [4] Switch address space   |
       |                        - Change CR3 register  |
       |                         to Process B's        |
       |                         page table            |
       |                        - TLB flush            |
       |                         |                     |
       |                    [5] Restore CPU context    |
       |                        from PCB-B             |
       |                         |                     |
       |                         |  ---- return --->   |
       |                         |                [6] Process B
       |                         |                    resumes execution
  (Ready state)                  |                (Running state)
```

**Cost factors of context switching:**

| Cost Factor | Description | Impact |
|------------|-------------|--------|
| Register save/restore | Saving and restoring general-purpose registers, FPU, etc. | Low |
| TLB flush | Invalidation of the page table cache | High |
| Cache pollution | L1/L2/L3 cache contents replaced with the new process's data | High |
| Pipeline flush | Clearing the CPU instruction pipeline | Medium |
| Kernel data structure updates | Updating run queues, statistics, etc. | Low |

The expected time for a context switch is on the order of a few microseconds on modern hardware. However, including cache warm-up (recovery from a cold cache), the indirect impact can extend to tens of microseconds.

### 2.4 Code Example 2: Observing Process States (Python)

The following Python script pseudo-observes process state transitions. It creates a child process and reads the state at each stage from the `/proc` filesystem (requires a Linux environment).

```python
#!/usr/bin/env python3
"""process_states.py
A demo program for observing process state transitions.
Run on a Linux environment (uses the /proc filesystem).

Run: python3 process_states.py
"""
import os
import sys
import time
import signal

def get_process_state(pid):
    """Retrieve the process state of the specified PID from /proc.

    The /proc/[pid]/status file records the process state.
    The values in the State line are as follows:
      R: Running (executing/runnable)
      S: Sleeping (interruptible sleep = Blocked)
      D: Disk sleep (uninterruptible sleep)
      T: Stopped
      Z: Zombie
    """
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("State:"):
                    return line.strip()
    except FileNotFoundError:
        return "Process does not exist (already terminated)"
    except PermissionError:
        return "Insufficient permissions to read"
    return "State unknown"

def main():
    print(f"Parent process PID: {os.getpid()}")
    print(f"Parent state: {get_process_state(os.getpid())}")
    print()

    # Create a child process
    pid = os.fork()

    if pid == 0:
        # --- Child process ---
        print(f"[Child] PID={os.getpid()} created (Ready -> Running)")

        # Demo transitioning to Blocked state via I/O operation
        print(f"[Child] Entering Blocked state via I/O operation (sleep)...")
        time.sleep(2)  # sleep puts the process in Sleeping (S) state

        print(f"[Child] I/O complete -> Ready -> Running")
        print(f"[Child] Normal exit -> Terminated")
        sys.exit(0)
    else:
        # --- Parent process ---
        time.sleep(0.5)  # Wait to ensure the child enters sleep state

        # Observe the child process state
        state = get_process_state(pid)
        print(f"[Parent] Child process (PID={pid}) state: {state}")
        print(f"         -> Should be S (Sleeping) since it's in sleep")
        print()

        # Wait for the child to terminate, but don't call wait() immediately
        time.sleep(3)  # The child should have already exited

        # Check for zombie state before calling wait()
        state = get_process_state(pid)
        print(f"[Parent] Child process state (before wait): {state}")
        print(f"         -> May be Z (Zombie) since wait() hasn't been called")

        # Reap the zombie with wait()
        result_pid, status = os.waitpid(pid, 0)
        print(f"\n[Parent] waitpid complete: child PID={result_pid}, exit status={os.WEXITSTATUS(status)}")

        # After wait(), the process has been destroyed
        state = get_process_state(pid)
        print(f"[Parent] Child process state (after wait): {state}")

if __name__ == "__main__":
    if sys.platform != "linux":
        print("Note: This program is Linux-only (uses /proc)")
        print("On macOS, /proc does not exist, so the state retrieval portions will not work")
        print("You can check process states with the ps command instead")
        print()
    main()
```

---

## 3. Process Creation: The fork/exec Model

### 3.1 Design Philosophy: Why Separate fork and exec?

The Unix/Linux process creation model consists of a combination of two system calls: `fork()` and `exec()`. Compared to a method like Windows's CreateProcess() that launches a new process with a single API, the fork/exec model has the following advantages:

1. **Flexible redirection**: By manipulating file descriptors after fork() but before exec(), you can freely redirect the child process's standard I/O. Shell pipes (`ls | grep foo`) are implemented using this mechanism.

2. **Pre-configuration of the environment**: You can configure the child process's environment variables, working directory, signal mask, resource limits, etc. before exec().

3. **Security**: You can close unnecessary file descriptors or restrict privileges before exec().

```
Overview of the fork/exec model:

  Parent process (PID=1000, /bin/bash)
       |
  [1]  | fork() system call
       |
       +----------------------------------------+
       |                                        |
  Parent process (PID=1000)              Child process (PID=1001)
  fork() returns the child's PID         fork() returns 0
  (= 1001)                                    |
       |                               [2]    | Preparation before exec()
       |                                      | - Redirect fd 1 to output file
       |                                      | - Close unnecessary fds
       |                                      | - Set environment variables
       |                                      |
       |                               [3]    | execvp("ls", ["ls", "-la"])
       |                                      |
       |                               [4] Process memory space is
       |                                   overwritten with the "ls" binary
       |                                   (PID does not change)
       |                                      |
       |                                      | ls executes
       |                                      |
       |                               [5]    | exit(0)
       |                                      |
  [6] waitpid(1001, &status, 0)               |
       | <- Receives child's exit status      v Destroyed
       |
  (Proceeds to process the next command)

  Important properties:
  - The child process after fork() is an almost complete copy of the parent
    (memory contents, file descriptors, signal handlers, etc.)
  - exec() replaces the process's memory space with a new program
  - After a successful exec(), control never returns to the original program's code
  - PID changes with fork(), but does not change with exec()
```

### 3.2 Copy-on-Write (CoW)

fork() needs to copy the parent process's memory space to the child, but copying all memory immediately is extremely inefficient. Modern OSes solve this problem using Copy-on-Write (CoW) technology.

```
How Copy-on-Write works:

  [Immediately after fork()]

  Parent's page table          Physical memory      Child's page table
  +----------------+          +----------+         +----------------+
  | vaddr A -> pf 1 | ------->| page f 1 |<--------| vaddr A -> pf 1 |
  | (read-only)    |          | "Hello"  |         | (read-only)    |
  +----------------+          +----------+         +----------------+
  | vaddr B -> pf 2 | ------->| page f 2 |<--------| vaddr B -> pf 2 |
  | (read-only)    |          | data...  |         | (read-only)    |
  +----------------+          +----------+         +----------------+

  -> Both processes share the same physical pages
  -> All pages are marked as read-only

  [Child process writes to vaddr A]

  Parent's page table          Physical memory      Child's page table
  +----------------+          +----------+         +----------------+
  | vaddr A -> pf 1 | ------->| page f 1 |         | vaddr A -> pf 3 |
  | (read-write)   |          | "Hello"  |         | (read-write)   |
  +----------------+          +----------+         +----------------+
  | vaddr B -> pf 2 | ------->| page f 2 |<--------| vaddr B -> pf 2 |
  | (read-only)    |          | data...  |         | (read-only)    |
  +----------------+          +----------+         +----------------+
                              | page f 3 |
                              | "World"  | <- New copy
                              +----------+

  -> Only the page that was written to is copied
  -> Pages that are not written to remain shared

  Benefits of CoW:
  1. fork() is fast (only page table copy, tens of microseconds)
  2. In the fork()+exec() pattern, exec() replaces all pages,
     so pages copied at fork() time are entirely wasted. CoW avoids this waste.
  3. Reduced memory usage (read-only pages remain shared)
```

### 3.3 Code Example 3: Command Execution via fork/exec (C)

The following program reproduces the basic fork/exec operations performed internally by a shell. It also includes the implementation of redirection.

```c
/* fork_exec_demo.c
 * Compile: gcc -Wall -o fork_exec_demo fork_exec_demo.c
 * Run: ./fork_exec_demo
 *
 * Purpose: Execute commands using the fork/exec model and
 *          understand how redirection works.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>

/* Execute the specified command in a child process.
 * If output_file is non-NULL, redirect stdout to that file.
 *
 * Why set up the redirect after fork but before exec?
 * -> exec() replaces the current process's program, but
 *   the file descriptor table is inherited as-is (unless
 *   FD_CLOEXEC is set). Therefore, if we reassign fd 1 (stdout)
 *   to a file before exec, the new program's standard output
 *   will automatically go to that file.
 */
int run_command(char *const argv[], const char *output_file)
{
    pid_t pid = fork();

    if (pid < 0) {
        /* fork failed. Possible causes:
         * - Process count limit (RLIMIT_NPROC) reached
         * - Out of memory
         * - PID space exhaustion (extremely rare) */
        perror("fork");
        return -1;
    }

    if (pid == 0) {
        /* --- Child process --- */

        /* If redirection is specified */
        if (output_file != NULL) {
            /* Open the output file
             * O_WRONLY: write-only
             * O_CREAT:  create if it doesn't exist
             * O_TRUNC:  delete existing contents
             * 0644:     rw-r--r-- permissions */
            int fd = open(output_file, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) {
                perror("open");
                _exit(1);  /* Use _exit() in child processes.
                            * Using exit() might execute atexit handlers
                            * registered in the parent process. */
            }

            /* Reassign fd 1 (stdout) to the file */
            if (dup2(fd, STDOUT_FILENO) < 0) {
                perror("dup2");
                _exit(1);
            }

            /* The original fd is no longer needed, so close it
             * (fd 1 already points to the same file) */
            close(fd);
        }

        /* Execute the program (the current process contents are replaced) */
        execvp(argv[0], argv);

        /* If execvp returns, an error has occurred */
        fprintf(stderr, "execvp failed: %s: %s\n", argv[0], strerror(errno));
        _exit(127);  /* Convention: return 127 when command is not found */
    }

    /* --- Parent process --- */
    int status;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return -1;
    }

    if (WIFEXITED(status)) {
        int exit_code = WEXITSTATUS(status);
        printf("[Parent] Child process (PID=%d) exited with code %d\n",
               pid, exit_code);
        return exit_code;
    } else if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        printf("[Parent] Child process (PID=%d) terminated by signal %d\n",
               pid, sig);
        return -1;
    }

    return -1;
}

int main(void)
{
    /* Command 1: Execute ls -la (to standard output) */
    printf("=== Running ls -la ===\n");
    char *cmd1[] = {"ls", "-la", "/tmp", NULL};
    run_command(cmd1, NULL);

    /* Command 2: Redirect ls -la output to a file */
    printf("\n=== ls -la > /tmp/ls_output.txt ===\n");
    char *cmd2[] = {"ls", "-la", "/tmp", NULL};
    run_command(cmd2, "/tmp/ls_output.txt");
    printf("Output written to /tmp/ls_output.txt\n");

    /* Command 3: Execute a nonexistent command (error handling demo) */
    printf("\n=== Running a nonexistent command ===\n");
    char *cmd3[] = {"nonexistent_command", NULL};
    run_command(cmd3, NULL);

    return 0;
}
```

### 3.4 Process Trees and Process Hierarchy

In Unix/Linux, all processes form a tree structure connected by parent-child relationships. The root of this tree is the init process with PID 1 (systemd in modern Linux).

```
Typical Linux system process tree:

  systemd (PID=1)  <- Ancestor of all user-space processes
    |
    +-- systemd-journald (PID=200)    <- Log management daemon
    |
    +-- systemd-udevd (PID=210)       <- Device management
    |
    +-- sshd (PID=500)                <- SSH daemon
    |   |
    |   +-- sshd (PID=1500)           <- SSH session (privilege separation)
    |       |
    |       +-- bash (PID=1501)       <- Login shell
    |           |
    |           +-- vim (PID=1600)    <- Editor launched by user
    |           |
    |           +-- python3 (PID=1700) <- Script launched by user
    |               |
    |               +-- python3 (PID=1701) <- Child created by fork()
    |
    +-- nginx (PID=600)               <- Web server (master process)
    |   +-- nginx (PID=601)           <- Worker process
    |   +-- nginx (PID=602)           <- Worker process
    |   +-- nginx (PID=603)           <- Worker process
    |   +-- nginx (PID=604)           <- Worker process
    |
    +-- cron (PID=700)                <- Periodic execution daemon
    |
    +-- docker (PID=800)              <- Container runtime
        +-- containerd-shim (PID=900)
            +-- sleep (PID=901)       <- Process inside container

  Process groups and sessions:
  +-----------------------------------------------------+
  | Session (SID=1501)                                   |
  |  +-----------------------------------------------+  |
  |  | Foreground process group (PGID=1600)           |  |
  |  |  vim (PID=1600)                                |  |
  |  +-----------------------------------------------+  |
  |  +-----------------------------------------------+  |
  |  | Background process group (PGID=1700)           |  |
  |  |  python3 (PID=1700)                            |  |
  |  |  python3 (PID=1701)                            |  |
  |  +-----------------------------------------------+  |
  |                                                     |
  |  Session leader: bash (PID=1501)                     |
  |  Controlling terminal: /dev/pts/0                    |
  +-----------------------------------------------------+

  Ctrl+C sends SIGINT to the entire foreground process group.
  Background process groups are not affected.
```

---

## 4. Special Process States

### 4.1 Zombie Processes

A zombie process occurs when a child process has terminated but the parent process has not yet collected its exit status via wait(). Zombie processes consume little CPU time or memory, but they continue to hold a PCB (including the process ID), so if a large number are created, they can exhaust the PID space.

**Why zombies exist**: This is the mechanism for conveying the child's exit status to the parent. If the OS completely deleted the PCB at the moment the child terminates, the parent would be unable to obtain the child's exit code when calling wait().

### 4.2 Orphan Processes

When a parent process terminates before its child process, the child becomes an "orphan process." Orphan processes are adopted by PID 1 (init/systemd), which calls wait() on their behalf to reclaim resources.

### 4.3 Daemon Processes

A daemon is a process that runs in the background without a controlling terminal. System services such as web servers (nginx), databases (MySQL), and log managers (syslogd) operate as daemons.

```
Classic steps for daemonization:

  [1] fork() -> exit() the parent
      -> The child becomes an orphan and is adopted by init
      -> The shell detects the parent's termination and returns to the prompt

  [2] setsid()
      -> Creates a new session and becomes the session leader
      -> Detaches from the controlling terminal

  [3] Second fork() -> exit() the first child
      -> No longer a session leader, so it cannot
        re-acquire a controlling terminal (safety measure)

  [4] chdir("/")
      -> Change the current directory to root
      -> Prevents the daemon from locking a specific directory

  [5] umask(0)
      -> Clear the file creation mask
      -> Allows explicit control of permissions for files created by the daemon

  [6] Redirect stdin/stdout/stderr to /dev/null
      -> Since there is no controlling terminal, these fds are unusable
      -> Logging goes to syslog or a dedicated log file

  Modern method:
  With systemd, registering a service as Type=simple
  eliminates the need for the complex steps above. systemd handles it.
```

### 4.4 Code Example 4: Creating and Reaping Zombie Processes (C)

```c
/* zombie_demo.c
 * Compile: gcc -Wall -o zombie_demo zombie_demo.c
 * Run: ./zombie_demo
 *
 * Purpose: Understand the mechanism by which zombie processes occur
 *          and how to reap them.
 *          During execution, check with "ps aux | grep zombie"
 *          in another terminal to see the Z-state process.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

/* === Method 1: Explicitly reap with wait() === */
void demo_explicit_wait(void)
{
    printf("\n=== Method 1: Explicit wait() ===\n");
    pid_t pid = fork();

    if (pid == 0) {
        /* Child process: exit immediately */
        printf("[Child] PID=%d exiting\n", getpid());
        _exit(42);
    }

    /* Parent process: don't call wait() for 5 seconds -> zombie occurs */
    printf("[Parent] Child (PID=%d) has exited. Not calling wait() for 5 seconds...\n", pid);
    printf("[Parent] Check for Z state with 'ps aux | grep zombie' during this time\n");
    sleep(5);

    /* Reap the zombie with wait() */
    int status;
    waitpid(pid, &status, 0);
    printf("[Parent] Child reaped. Exit code=%d\n", WEXITSTATUS(status));
}

/* === Method 2: Auto-reap with SIGCHLD signal handler === */
volatile sig_atomic_t child_count = 0;

void sigchld_handler(int signo)
{
    (void)signo;  /* Suppress unused parameter warning */

    /* Call wait() non-blocking to reap all terminated children.
     * Why a while loop? -> When multiple children terminate simultaneously,
     * SIGCHLD may be delivered only once, so we need to loop
     * to reap all of them. */
    pid_t pid;
    int status;
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
        child_count++;
        /* Only async-signal-safe functions should be used inside signal handlers.
         * printf() is not async-signal-safe, but used here for demo purposes.
         * Production code should use write() instead. */
    }
}

void demo_sigchld_handler(void)
{
    printf("\n=== Method 2: Auto-reap via SIGCHLD handler ===\n");

    /* Set up the SIGCHLD handler */
    struct sigaction sa;
    sa.sa_handler = sigchld_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;  /* Automatic restart of system calls */
    sigaction(SIGCHLD, &sa, NULL);

    /* Create 5 child processes */
    for (int i = 0; i < 5; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            printf("[Child %d] PID=%d exiting\n", i, getpid());
            _exit(0);
        }
        printf("[Parent] Created child %d (PID=%d)\n", i, pid);
    }

    /* Wait for the handler to reap all children */
    sleep(2);
    printf("[Parent] Reaped child process count: %d\n", child_count);
}

/* === Method 3: Set SIGCHLD to SIG_IGN === */
void demo_sigign(void)
{
    printf("\n=== Method 3: Set SIGCHLD to SIG_IGN ===\n");

    /* Setting SIGCHLD to SIG_IGN causes child processes to be
     * automatically reaped upon termination, preventing zombies.
     * However, the child's exit status can no longer be retrieved via wait().
     * This behavior is explicitly defined by POSIX. */
    signal(SIGCHLD, SIG_IGN);

    pid_t pid = fork();
    if (pid == 0) {
        printf("[Child] PID=%d exiting\n", getpid());
        _exit(0);
    }

    sleep(1);
    printf("[Parent] No zombie created (auto-reaped)\n");

    /* Reset to default */
    signal(SIGCHLD, SIG_DFL);
}

int main(void)
{
    printf("Zombie process demo (parent PID=%d)\n", getpid());

    demo_explicit_wait();
    demo_sigchld_handler();
    demo_sigign();

    printf("\nAll demos complete\n");
    return 0;
}
```

---

## 5. CPU Scheduling

### 5.1 The Need for Scheduling

In a multitasking OS, multiple Ready-state processes are waiting for CPU allocation. The scheduler is the OS component that decides which process gets the CPU next. The choice of scheduling algorithm directly affects the system's responsiveness, throughput, and fairness.

### 5.2 Scheduling Evaluation Criteria

| Criterion | Definition | Optimization Direction |
|-----------|-----------|----------------------|
| CPU utilization | Percentage of time the CPU is performing useful work | Maximize (ideal: 100%) |
| Throughput | Number of processes completed per unit time | Maximize |
| Turnaround time | Total time from process arrival to completion | Minimize |
| Waiting time | Total time a process spends waiting in the Ready queue | Minimize |
| Response time | Time from request to first response | Minimize (critical for interactive systems) |

### 5.3 Major Scheduling Algorithms

```
[1] FCFS (First-Come, First-Served)
    Execute in arrival order. The simplest.

    Ready queue: [P1(24ms)] [P2(3ms)] [P3(3ms)]
                   <- front                    tail ->

    Gantt chart:
    | P1                        |P2 |P3 |
    0                          24  27  30

    Waiting time: P1=0, P2=24, P3=27
    Average waiting time: (0+24+27)/3 = 17ms

    Problem: Convoy Effect
    -> If a long CPU-intensive process arrives first,
      short processes are forced to wait a long time.

[2] SJF (Shortest Job First)
    Prioritize the process with the shortest execution time.
    Theoretically minimizes average waiting time (proven).

    Ready queue: [P1(24ms)] [P2(3ms)] [P3(3ms)]

    Execution order under SJF:
    |P2 |P3 | P1                        |
    0   3   6                          30

    Waiting time: P1=6, P2=0, P3=3
    Average waiting time: (6+0+3)/3 = 3ms  <- Major improvement from FCFS's 17ms

    Problem: Execution time prediction is difficult.
    -> Estimated from past execution history using exponential averaging.
    -> Long processes may suffer starvation.

[3] Round Robin (RR)
    Assign each process a time quantum, and switch
    to the next process when that time expires.

    Time quantum q=4ms:

    |P1 |P2 |P3 |P1 |P1 |P1 |P1 |P1 |
    0   4   7  10  14  18  22  26  30

    Properties:
    - If q -> infinity, equivalent to FCFS
    - If q -> 0, processor sharing (ideal fairness,
      but context switch overhead becomes dominant)
    - Appropriate q is typically around 10ms to 100ms
    - Considering context switch cost,
      q should be at least 100 times the switch cost

[4] Priority Scheduling
    Assign each process a priority and execute higher-priority processes first.

    Problem: Starvation of low-priority processes
    Solution: Aging
    -> Gradually increase priority as waiting time grows

[5] Multilevel Feedback Queue (MLFQ)
    Maintains multiple priority queues and moves processes
    between queues based on their behavior. Foundation of modern OS schedulers.

    Queue 0 (highest priority, q=8ms):   [P_new1] [P_new2]
    Queue 1 (medium priority,  q=16ms):  [P_mid1]
    Queue 2 (lowest priority,  q=32ms):  [P_long1] [P_long2]

    Rules:
    1. New processes enter the highest-priority queue
    2. If the time quantum is exhausted, demote to the next lower queue
    3. If the process voluntarily releases the CPU for I/O, it stays in the current queue
    4. Periodically move all processes back to the highest-priority queue
       (Priority Boost for starvation prevention)
```

### 5.4 Scheduling Algorithm Comparison Table

| Algorithm | Preemptive | Starvation | Responsiveness | Implementation Complexity | Application Area |
|-----------|-----------|------------|----------------|--------------------------|-----------------|
| FCFS | No | No | Low | Very Low | Batch processing |
| SJF (non-preemptive) | No | Yes | Low | Medium (prediction needed) | Batch processing |
| SRTF (preemptive SJF) | Yes | Yes | Medium | Medium | Theoretically optimal |
| Round Robin | Yes | No | High | Low | Interactive, general purpose |
| Priority | Yes/No | Yes | Priority-dependent | Low | Real-time |
| MLFQ | Yes | No (with boost) | High | High | General-purpose OS |
| CFS (Linux) | Yes | No | High | High | Linux standard |

### 5.5 Linux CFS (Completely Fair Scheduler)

CFS, adopted since Linux 2.6.23, aims to distribute "fair" CPU time to all processes. It manages Ready-state processes using a red-black tree and executes the process that has used the least CPU time (the process with the smallest virtual runtime, vruntime).

```
CFS virtual runtime (vruntime):

  vruntime = actual execution time * (NICE_0_LOAD / process weight)

  A process with a low nice value (high priority) has a large weight,
  so the same actual execution time results in a smaller vruntime increase,
  and the process consequently gets more CPU time.

  Red-black tree structure:
  +---------------------------------------------+
  |            [vruntime=50]  <- root             |
  |           /             \                    |
  |    [vruntime=30]    [vruntime=70]            |
  |      /    \           /    \               |
  |   [20]    [40]     [60]    [80]             |
  |                                              |
  |  Leftmost node [vruntime=20] runs next       |
  |  -> Accessible in O(1) (cached)              |
  |  -> Insert/delete in the red-black tree is O(log n) |
  +---------------------------------------------+

  CFS time slice calculation:
  Time slice = (process weight / total weight of all Ready processes)
               * scheduling period

  Default scheduling period: 6ms * number of Ready processes
  (minimum granularity is 0.75ms)
```

---

## 6. Inter-Process Communication (IPC)

### 6.1 The Need for and Classification of IPC

Since processes have independent address spaces, they cannot directly access the memory of other processes. IPC (Inter-Process Communication) mechanisms are needed for processes to exchange data or synchronize their operations.

```
Classification of IPC mechanisms:

  +-----------------------------------------------------+
  |                   IPC Mechanisms                      |
  +--------------------+--------------------------------+
  |   Message Passing  |      Shared Memory              |
  |   (data copying)   |    (memory sharing)             |
  +--------------------+--------------------------------+
  | - Pipe             | - POSIX shared memory           |
  | - Named pipe       |   (shm_open)                   |
  |   (FIFO)           | - System V shared memory        |
  | - Message queue    |   (shmget/shmat)                |
  | - Socket           | - mmap (file sharing)           |
  | - Signal           |                                |
  +--------------------+--------------------------------+
  | Features:          | Features:                       |
  | - Kernel mediates  | - Kernel mediates only initially |
  | - Data copying     | - No data copying (fast)        |
  |   occurs (slow)    | - Synchronization must be       |
  | - Easy sync        |   managed manually              |
  |                    |   (semaphores, mutexes, etc.)   |
  +--------------------+--------------------------------+
```

### 6.2 Pipes

A pipe is one of the oldest IPC mechanisms in Unix and is used daily through the `|` symbol in shells. A pipe is a unidirectional byte stream; when the write end is closed, the read end is notified with EOF.

### 6.3 Code Example 5: Inter-Process Communication via Pipes (C)

The following program implements the equivalent of `ls -la | grep ".txt"` in C. It demonstrates how to connect two child processes with a pipe.

```c
/* pipe_demo.c
 * Compile: gcc -Wall -o pipe_demo pipe_demo.c
 * Run: ./pipe_demo
 *
 * Purpose: Understand the mechanism of inter-process communication using pipes.
 *          Reproduce the internal workings of a shell pipeline like ls | grep.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>

int main(void)
{
    int pipefd[2];

    /* pipe() creates a pair of file descriptors.
     * pipefd[0]: for reading (the pipe's exit)
     * pipefd[1]: for writing (the pipe's entrance)
     *
     * Why are two fds needed?
     * -> A pipe is unidirectional communication. To make the direction
     *   of data flow clear, the write end and read end are managed
     *   with separate fds.
     */
    if (pipe(pipefd) < 0) {
        perror("pipe");
        return 1;
    }

    printf("Pipe created: read_fd=%d, write_fd=%d\n", pipefd[0], pipefd[1]);

    /* === First child process: ls -la /tmp === */
    pid_t pid1 = fork();
    if (pid1 < 0) {
        perror("fork");
        return 1;
    }

    if (pid1 == 0) {
        /* Child process 1: the ls command
         *
         * Reassign stdout (fd 1) to the pipe's write end.
         * The output of ls will flow into the pipe.
         */
        close(pipefd[0]);  /* Close the read end since we don't use it.
                            * If not closed, a process holding the read end
                            * persists, and the grep side cannot detect EOF. */

        dup2(pipefd[1], STDOUT_FILENO);  /* fd 1 -> pipe write end */
        close(pipefd[1]);  /* Original fd no longer needed after dup2 */

        execlp("ls", "ls", "-la", "/tmp", NULL);
        perror("execlp ls");
        _exit(1);
    }

    /* === Second child process: grep === */
    pid_t pid2 = fork();
    if (pid2 < 0) {
        perror("fork");
        return 1;
    }

    if (pid2 == 0) {
        /* Child process 2: the grep command
         *
         * Reassign stdin (fd 0) to the pipe's read end.
         * Data flowing from the pipe becomes grep's input.
         */
        close(pipefd[1]);  /* Close the write end since we don't use it */

        dup2(pipefd[0], STDIN_FILENO);  /* fd 0 -> pipe read end */
        close(pipefd[0]);

        /* Search for lines containing "." (matches any file) */
        execlp("grep", "grep", "--color=auto", "\\.", NULL);
        perror("execlp grep");
        _exit(1);
    }

    /* === Parent process === */
    /* The parent process must close both ends of the pipe.
     * In particular, if the write end is not closed, a process that
     * can write to the pipe persists, and the grep side cannot receive EOF. */
    close(pipefd[0]);
    close(pipefd[1]);

    /* Wait for both child processes to finish */
    int status1, status2;
    waitpid(pid1, &status1, 0);
    waitpid(pid2, &status2, 0);

    printf("\n--- Pipeline complete ---\n");
    printf("ls  exit code: %d\n", WEXITSTATUS(status1));
    printf("grep exit code: %d\n", WEXITSTATUS(status2));

    return 0;
}
```

### 6.4 Signals

Signals are software interrupts that asynchronously notify a process of an event. Although the types of signals are limited, they are an essential mechanism for process control and abnormal termination handling.

```
Major signals:

  Signal      Number  Default Action     Trigger
  ----------------------------------------------------------
  SIGHUP       1     Terminate          Controlling terminal hangup
  SIGINT       2     Terminate          Ctrl+C
  SIGQUIT      3     Core dump          Ctrl+\
  SIGILL       4     Core dump          Illegal instruction execution
  SIGABRT      6     Core dump          abort() call
  SIGFPE       8     Core dump          Floating-point exception (division by zero, etc.)
  SIGKILL      9     Terminate          Forced termination (cannot be caught or ignored)
  SIGSEGV     11     Core dump          Illegal memory access
  SIGPIPE     13     Terminate          Writing to a pipe with no reader
  SIGALRM     14     Terminate          alarm() timer expiration
  SIGTERM     15     Terminate          Termination request (can be caught)
  SIGCHLD     17     Ignore             Child process state change
  SIGCONT     18     Resume             Resume a stopped process
  SIGSTOP     19     Stop               Stop process (cannot be caught)
  SIGTSTP     20     Stop               Ctrl+Z

  Only SIGKILL and SIGSTOP cannot be caught, ignored, or blocked.
  This guarantees that an administrator can always stop/terminate any process.
```

### 6.5 IPC Mechanism Comparison Table

| Mechanism | Direction | Related Processes | Speed | Persistence | Typical Use |
|-----------|----------|------------------|-------|-------------|-------------|
| Pipe | Unidirectional | Parent-child | Medium | None | Shell pipelines |
| Named pipe (FIFO) | Unidirectional | Any | Medium | Filesystem | Communication with daemons |
| Message queue | Unidirectional | Any | Medium | Kernel | Structured messages |
| Shared memory | Bidirectional | Any | High | Kernel | Large data sharing |
| Socket (UNIX) | Bidirectional | Any | Medium-High | None | Local server communication |
| Socket (TCP/IP) | Bidirectional | Any (across network) | Low | None | Network communication |
| Signal | Unidirectional | Any | High | None | Event notification |
| File | Bidirectional | Any | Low | Disk | Persistent data sharing |

### 6.6 Code Example 6: Inter-Process Communication via Shared Memory (C)

Shared memory is the fastest IPC mechanism. Since no data copying through the kernel is required, it is suitable for exchanging large amounts of data. However, note that synchronization must be managed manually.

```c
/* shared_memory_demo.c
 * Compile: gcc -Wall -o shared_memory_demo shared_memory_demo.c -lrt -lpthread
 * Run: ./shared_memory_demo
 *
 * Purpose: Understand the mechanism of inter-process communication
 *          using POSIX shared memory and semaphores.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>

/* Data structure placed in shared memory */
typedef struct {
    sem_t sem_producer;  /* Semaphore for writer synchronization */
    sem_t sem_consumer;  /* Semaphore for reader synchronization */
    int   data;          /* Shared data */
    int   done;          /* Termination flag */
} shared_data_t;

#define SHM_NAME "/process_demo_shm"

int main(void)
{
    /* Create a POSIX shared memory object.
     * shm_open() returns a file descriptor.
     * This fd corresponds to a file under /dev/shm (on Linux).
     *
     * Why a file-based API?
     * -> Following the Unix philosophy of "everything is a file,"
     *   existing file operation APIs (ftruncate, mmap, etc.) can
     *   be reused. */
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) {
        perror("shm_open");
        return 1;
    }

    /* Set the size of the shared memory */
    if (ftruncate(shm_fd, sizeof(shared_data_t)) < 0) {
        perror("ftruncate");
        return 1;
    }

    /* Map the shared memory into the process's address space */
    shared_data_t *shm = mmap(NULL, sizeof(shared_data_t),
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED, shm_fd, 0);
    if (shm == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    close(shm_fd);  /* The fd can be closed after mmap */

    /* Initialize semaphores that can be shared between processes.
     * The second argument of 1 means inter-process sharing (0 means thread-only). */
    sem_init(&shm->sem_producer, 1, 1);  /* Writer can proceed initially */
    sem_init(&shm->sem_consumer, 1, 0);  /* Reader waits for data */
    shm->done = 0;

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }

    if (pid == 0) {
        /* --- Child process (Consumer) --- */
        printf("[Consumer PID=%d] Started\n", getpid());

        while (1) {
            sem_wait(&shm->sem_consumer);  /* Wait until data arrives */

            if (shm->done) {
                printf("[Consumer] Received termination signal\n");
                break;
            }

            printf("[Consumer] Received data: %d\n", shm->data);
            sem_post(&shm->sem_producer);  /* Allow writer to proceed */
        }

        _exit(0);
    }

    /* --- Parent process (Producer) --- */
    printf("[Producer PID=%d] Started\n", getpid());

    for (int i = 1; i <= 5; i++) {
        sem_wait(&shm->sem_producer);  /* Wait for write permission */

        shm->data = i * 10;
        printf("[Producer] Sent data: %d\n", shm->data);

        sem_post(&shm->sem_consumer);  /* Notify reader that data has arrived */
        usleep(100000);  /* Wait 100ms (to make behavior visible) */
    }

    /* Send termination notification */
    sem_wait(&shm->sem_producer);
    shm->done = 1;
    sem_post(&shm->sem_consumer);

    /* Wait for child process to finish */
    waitpid(pid, NULL, 0);

    /* Cleanup */
    sem_destroy(&shm->sem_producer);
    sem_destroy(&shm->sem_consumer);
    munmap(shm, sizeof(shared_data_t));
    shm_unlink(SHM_NAME);

    printf("[Producer] All data send/receive complete\n");
    return 0;
}
```

Expected output:
```
[Producer PID=5000] Started
[Consumer PID=5001] Started
[Producer] Sent data: 10
[Consumer] Received data: 10
[Producer] Sent data: 20
[Consumer] Received data: 20
[Producer] Sent data: 30
[Consumer] Received data: 30
[Producer] Sent data: 40
[Consumer] Received data: 40
[Producer] Sent data: 50
[Consumer] Received data: 50
[Consumer] Received termination signal
[Producer] All data send/receive complete
```

### 6.7 Code Example 7: Multiprocess Processing in Python

```python
#!/usr/bin/env python3
"""multiprocess_demo.py
A demo of inter-process communication using Python's multiprocessing module.

Run: python3 multiprocess_demo.py

The multiprocessing module internally abstracts OS mechanisms such as
fork()/exec(), pipes, and shared memory.
"""
import multiprocessing
import os
import time
import sys

def producer(queue, count):
    """A producer process that sends data to a queue.

    multiprocessing.Queue internally uses pipes and locks.
    It is a high-level API for safely passing data between processes.
    """
    print(f"[Producer PID={os.getpid()}] Started")
    for i in range(count):
        item = f"item-{i:03d}"
        queue.put(item)
        print(f"[Producer] Sent: {item}")
        time.sleep(0.1)
    # Send a sentinel value to indicate termination
    queue.put(None)
    print(f"[Producer] All items sent")

def consumer(queue, result_dict, worker_id):
    """A consumer process that receives data from a queue.

    result_dict is a shared dictionary provided by multiprocessing.Manager.
    A Manager process mediates to share dictionaries between processes.
    """
    print(f"[Consumer-{worker_id} PID={os.getpid()}] Started")
    processed = []
    while True:
        item = queue.get()  # Block until data arrives
        if item is None:
            # Received sentinel value; pass it back for other consumers
            queue.put(None)
            break
        result = f"{item} -> processed by worker-{worker_id}"
        processed.append(result)
        print(f"[Consumer-{worker_id}] Processed: {item}")
        time.sleep(0.05)  # Simulate processing

    result_dict[worker_id] = processed
    print(f"[Consumer-{worker_id}] Finished. Items processed={len(processed)}")

def demo_shared_value():
    """Demo of a counter using shared memory.

    multiprocessing.Value internally uses mmap() to create
    a memory region shared between processes.
    It has a built-in lock that automatically prevents race conditions.
    """
    print("\n=== Shared Memory Counter Demo ===")

    counter = multiprocessing.Value('i', 0)  # Shared integer value
    barrier = multiprocessing.Barrier(4)      # Barrier for synchronizing 4 processes

    def increment(shared_counter, sync_barrier, n):
        """Increment the shared counter n times."""
        sync_barrier.wait()  # Wait for all processes to be ready
        for _ in range(n):
            with shared_counter.get_lock():
                # Acquire the lock with get_lock() before modifying the value.
                # Without the lock, another process could intervene between
                # read, add, and write, causing lost updates.
                shared_counter.value += 1

    processes = []
    increments_per_process = 10000
    for i in range(4):
        p = multiprocessing.Process(
            target=increment,
            args=(counter, barrier, increments_per_process)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    expected = 4 * increments_per_process
    actual = counter.value
    print(f"Expected: {expected}, Actual: {actual}")
    if expected == actual:
        print("Mutual exclusion via lock worked correctly")
    else:
        print("Race condition occurred (lock insufficient)")

def main():
    print(f"=== Main process PID={os.getpid()} ===\n")

    # === Producer-Consumer Pattern ===
    print("=== Producer-Consumer Pattern ===")
    queue = multiprocessing.Queue()

    # Share a dictionary between processes using Manager
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    # Start 1 Producer and 2 Consumers
    prod = multiprocessing.Process(target=producer, args=(queue, 10))
    cons1 = multiprocessing.Process(target=consumer, args=(queue, result_dict, 1))
    cons2 = multiprocessing.Process(target=consumer, args=(queue, result_dict, 2))

    prod.start()
    cons1.start()
    cons2.start()

    prod.join()
    cons1.join()
    cons2.join()

    print(f"\nProcessing result summary:")
    for worker_id, items in result_dict.items():
        print(f"  Worker-{worker_id}: {len(items)} items processed")

    # === Shared Memory Demo ===
    demo_shared_value()

    print(f"\n=== All demos complete ===")

if __name__ == "__main__":
    # When using the multiprocessing module,
    # the __name__ == "__main__" guard is mandatory.
    # Why? -> When child processes are created via fork() or spawn(),
    # the module is re-imported, so without the guard,
    # main() would be re-executed in the child process.
    main()
```

---

## 7. Anti-patterns and Pitfalls

### 7.1 Anti-pattern 1: Neglecting Zombie Processes

**Problem**: If you create child processes without calling `wait()` or `waitpid()`, the child's PCB remains even after termination (zombie process). Repeating this in a long-running server process can exhaust the PID space, preventing new process creation.

```c
/* Anti-pattern: A server that forgets wait() */
/* !!!!! This is an example of BAD code !!!!! */
while (1) {
    int client_fd = accept(server_fd, NULL, NULL);
    pid_t pid = fork();
    if (pid == 0) {
        handle_client(client_fd);
        _exit(0);
    }
    close(client_fd);
    /* wait() is not called here!
     * A zombie accumulates every time a child process exits.
     * For a server processing 10,000 requests per day,
     * 10,000 zombies are created per day.
     * Once /proc/sys/kernel/pid_max (usually 32768) is exceeded,
     * fork() starts failing with ENOSPC/EAGAIN. */
}

/* Correct pattern: Auto-reap via SIGCHLD handler */
/* This method reaps zombies without blocking the main loop */
void sigchld_handler(int sig) {
    (void)sig;
    while (waitpid(-1, NULL, WNOHANG) > 0)
        ;  /* Reap all terminated child processes */
}

/* In the initialization part of main(): */
struct sigaction sa;
sa.sa_handler = sigchld_handler;
sigemptyset(&sa.sa_mask);
sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
sigaction(SIGCHLD, &sa, NULL);
```

**Why this is dangerous**: While zombie processes themselves consume almost no CPU or memory, they continue to hold the following resources:
- PID entry (/proc/[pid] directory)
- Part of the kernel's task_struct structure
- The process ID itself (the PID space is finite)

**Detection method**: You can check for zombie processes with `ps aux | grep Z`. They show `Z+` in the STAT column.

### 7.2 Anti-pattern 2: File Descriptor Leaks After fork()

**Problem**: fork() copies all of the parent process's file descriptors to the child process. If unnecessary fds are not closed before exec(), the child process continues to hold unintended files or sockets.

```c
/* Anti-pattern: fd leak */
/* !!!!! This is an example of BAD code !!!!! */
int log_fd = open("/var/log/app.log", O_WRONLY | O_APPEND);
int db_fd = connect_to_database();  /* Database connection socket */

pid_t pid = fork();
if (pid == 0) {
    /* The child process inherits log_fd and db_fd as-is.
     * Even if the program is replaced with exec(), if FD_CLOEXEC
     * is not set, these fds are accessible from the new program.
     *
     * Problems:
     * 1. The child process unintentionally holds the DB connection
     *    -> Connection pool exhaustion
     * 2. Writes to the log file may conflict
     * 3. Security risk (if the child process runs an untrusted program) */
    execvp(cmd[0], cmd);
    _exit(1);
}

/* Correct pattern: close unnecessary fds, or use O_CLOEXEC */

/* Method A: close after fork, before exec */
pid_t pid2 = fork();
if (pid2 == 0) {
    close(log_fd);
    close(db_fd);
    execvp(cmd[0], cmd);
    _exit(1);
}

/* Method B: O_CLOEXEC flag (recommended)
 * If O_CLOEXEC is set when opening a file,
 * it is automatically closed on exec().
 * No risk of forgetting close() after fork(). */
int safe_fd = open("/var/log/app.log",
                   O_WRONLY | O_APPEND | O_CLOEXEC);
```

---

## 8. Edge Case Analysis

### 8.1 Edge Case 1: Combining fork() with Multithreading

Calling fork() from a multithreaded program is one of the most dangerous edge cases. fork() copies only the calling thread into the child process; the other threads do not exist. If another thread was holding a lock at this point, that lock will never be released in the child process (deadlock).

```
Problem with multithreading + fork:

  Parent process:
  +----------------------------------------+
  | Thread A                Thread B       |
  |    |                      |            |
  |    |                 mutex_lock(m)     |
  |    |                      |            |
  |  fork() <-- fork at this moment        |
  |    |                      |            |
  +----------------------------------------+

  Child process:
  +----------------------------------------+
  | Copy of Thread A only                  |
  |    |                                   |
  |    |  mutex_lock(m) <- Deadlock!       |
  |    |  Thread B does not exist, so      |
  |    |  the mutex is never released      |
  |    |                                   |
  |  (blocked forever)                     |
  +----------------------------------------+

  POSIX's pthread_atfork() can manage locks around fork,
  but tracking all locks across all libraries is
  practically infeasible.

  Recommended countermeasures:
  1. Do not fork() from a multithreaded process
     -> Use posix_spawn() instead
  2. After fork(), only call exec() (don't perform complex operations before exec)
  3. If fork() is needed, do it before creating threads
```

### 8.2 Edge Case 2: PID Recycling

PIDs are a finite resource, and when a process terminates, its PID is recycled. This can cause a TOCTOU (Time-of-Check Time-of-Use) problem where a program tracking processes by PID inadvertently sends a signal to an unintended process.

```
Problem caused by PID recycling:

  Time T1: Process X (PID=5000) is running
           The monitoring program records PID=5000

  Time T2: Process X (PID=5000) terminates
           PID 5000 is freed

  Time T3: A new Process Y is created with PID=5000

  Time T4: The monitoring program executes kill(5000, SIGTERM)
           -> Process Y is terminated instead of Process X!

  Countermeasures:
  1. Use pidfd_open() (Linux 5.3+)
     -> Reference the process via a file descriptor instead of PID
     -> The fd is invalidated after the process terminates, so no recycling issue
     int pidfd = pidfd_open(pid, 0);
     pidfd_send_signal(pidfd, SIGTERM, NULL, 0);

  2. For child processes, use waitid() with WNOWAIT to
     verify the process's existence before sending a signal

  3. Use cgroups to manage processes at the process group level

  PID space size:
  Linux: /proc/sys/kernel/pid_max (default 32768, maximum 4194304)
  With pid_max of 32768, PIDs can cycle through in a few minutes
  on a high-load server.
```

### 8.3 Edge Case 3: Signal Delivery During fork()

If a signal is delivered during the execution of the fork() system call, it can cause unexpected behavior. In particular, calling fork() inside a SIGCHLD handler can result in recursive fork() calls, rapidly consuming resources.

```
Principles of safe signal handling:

  1. Use only async-signal-safe functions inside signal handlers
     - Safe: write(), _exit(), signal(), kill(), open(), close()
     - Unsafe: printf(), malloc(), free(), fork(), pthread_mutex_lock()

  2. Keep signal handlers minimal
     - Just set a flag and perform the actual processing in the main loop

  volatile sig_atomic_t got_sigchld = 0;

  void handler(int sig) {
      (void)sig;
      got_sigchld = 1;  /* Just set the flag */
  }

  /* Main loop */
  while (running) {
      if (got_sigchld) {
          got_sigchld = 0;
          while (waitpid(-1, NULL, WNOHANG) > 0)
              ;  /* Safely perform reaping here */
      }
      /* ... normal processing ... */
  }
```

---

## 9. Practical Exercises

### Exercise 1: [Basic] Observing and Retrieving Process Information

**Objective**: Collect information about running processes using OS tools.

```bash
# === Exercise 1-A: Checking the process tree ===
# Display the current system's process tree.
# Why is a tree view useful? -> It shows parent-child relationships
# and which processes originated from which services at a glance.

# Linux:
ps auxf                      # Display all processes in tree format
pstree -p                    # Process tree with PIDs
pstree -p $(pgrep sshd | head -1)  # Only the tree below sshd

# macOS:
ps aux                       # Display all processes
pstree                       # Homebrew: brew install pstree

# === Exercise 1-B: Reading process info from /proc (Linux only) ===
# /proc is a virtual filesystem where the kernel exposes process
# information as files. This allows accessing process information
# with ordinary file operations without special APIs.

cat /proc/self/status       # Info about self (the cat process)
cat /proc/self/maps         # Memory mapping information
ls -la /proc/self/fd        # Open file descriptors
cat /proc/self/cmdline | tr '\0' ' '  # Command line arguments
cat /proc/self/environ | tr '\0' '\n' | head -20  # Environment variables

# Exercise tasks:
# 1. Check your shell process's PID (echo $$)
# 2. Check the process state from /proc/[PID]/status for that PID
# 3. Count the number of open file descriptors in /proc/[PID]/fd
# 4. Open a new file, then re-check the fd count and confirm it increased

# === Exercise 1-C: Process statistics ===
# Check real-time process information with top/htop
top -bn1 | head -20          # Display once in batch mode
# htop                       # Interactive process monitor (requires installation)

# Process resource usage
/usr/bin/time -v ls /tmp 2>&1  # Linux: detailed resource usage
```

### Exercise 2: [Intermediate] Understanding Fork Bombs and Resource Limits

**Objective**: Understand the operating principle of fork bombs and learn how the OS prevents runaway processes.

```bash
# WARNING: Never actually execute a fork bomb. Understand it theoretically only.

# Bash fork bomb:
# :(){ :|:& };:
#
# Expanded:
# bomb() {
#   bomb | bomb &   # Call itself twice, running in the background
# }
# bomb              # Initial call
#
# Why it's dangerous:
# Round 1: 1 process -> 2 processes
# Round 2: 2 processes -> 4 processes
# Round 3: 4 processes -> 8 processes
# ...
# Round n: 2^n processes
# After 15 rounds, 32,768 processes -> reaches pid_max
# Memory and CPU are exhausted, and the system becomes unresponsive

# === Countermeasure 1: Limit process count with ulimit ===
ulimit -u          # Check the current process count limit
ulimit -u 100      # Limit the process count to 100
# This setting applies only to the current shell and its child processes

# === Countermeasure 2: Set permanent limits in /etc/security/limits.conf ===
# student  hard  nproc  200    # student user limited to 200 processes
# @users   hard  nproc  500    # users group limited to 500 processes

# === Countermeasure 3: Limit via cgroups (systemd environment) ===
# systemd automatically assigns a cgroup to each user session
# systemctl status user-1000.slice  # cgroup for UID=1000 user

# === Countermeasure 4: systemd's TasksMax ===
# In a service file:
# [Service]
# TasksMax=512    # Process/thread limit for this service

# Exercise tasks:
# 1. Check the current process count limit with ulimit -u
# 2. With the limit set to 50, write a script that creates many
#    background processes and observe the behavior when the limit is reached
# 3. Locate the cgroups v2 pids.max file and check its value
```

### Exercise 3: [Advanced] Implementing a Multiprocess Pipeline

**Objective**: Implement the pipeline processing performed by a shell yourself, and deepen understanding of the interplay between fork/exec/pipe/dup2.

```c
/* pipeline_exercise.c
 * Exercise: Complete the skeleton code below to implement the
 * 3-stage pipeline "cat /etc/passwd | grep root | wc -l".
 *
 * Hints:
 * - An N-stage pipeline requires N-1 pipes
 * - In each child process, reassign the appropriate fds with dup2
 *   and close all unnecessary fds
 * - The parent process must close all pipe fds and
 *   wait() for all child processes
 *
 * Compile: gcc -Wall -o pipeline_exercise pipeline_exercise.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void)
{
    /* Create 2 pipes */
    int pipe1[2];  /* cat -> grep */
    int pipe2[2];  /* grep -> wc */

    if (pipe(pipe1) < 0 || pipe(pipe2) < 0) {
        perror("pipe");
        return 1;
    }

    /* === Process 1: cat /etc/passwd === */
    pid_t pid1 = fork();
    if (pid1 == 0) {
        /* TODO: Implement the following
         * 1. Reassign stdout to pipe1[1]
         * 2. Close unnecessary fds (pipe1[0], pipe2[0], pipe2[1])
         * 3. Execute execlp("cat", "cat", "/etc/passwd", NULL)
         */

        /* --- Write your code here --- */
        dup2(pipe1[1], STDOUT_FILENO);
        close(pipe1[0]);
        close(pipe1[1]);
        close(pipe2[0]);
        close(pipe2[1]);
        execlp("cat", "cat", "/etc/passwd", NULL);
        perror("execlp cat");
        _exit(1);
    }

    /* === Process 2: grep root === */
    pid_t pid2 = fork();
    if (pid2 == 0) {
        /* TODO: Implement the following
         * 1. Reassign stdin to pipe1[0]
         * 2. Reassign stdout to pipe2[1]
         * 3. Close unnecessary fds (pipe1[1], pipe2[0])
         * 4. Execute execlp("grep", "grep", "root", NULL)
         */

        /* --- Write your code here --- */
        dup2(pipe1[0], STDIN_FILENO);
        dup2(pipe2[1], STDOUT_FILENO);
        close(pipe1[0]);
        close(pipe1[1]);
        close(pipe2[0]);
        close(pipe2[1]);
        execlp("grep", "grep", "root", NULL);
        perror("execlp grep");
        _exit(1);
    }

    /* === Process 3: wc -l === */
    pid_t pid3 = fork();
    if (pid3 == 0) {
        /* TODO: Implement the following
         * 1. Reassign stdin to pipe2[0]
         * 2. Close unnecessary fds (pipe1[0], pipe1[1], pipe2[1])
         * 3. Execute execlp("wc", "wc", "-l", NULL)
         */

        /* --- Write your code here --- */
        dup2(pipe2[0], STDIN_FILENO);
        close(pipe1[0]);
        close(pipe1[1]);
        close(pipe2[0]);
        close(pipe2[1]);
        execlp("wc", "wc", "-l", NULL);
        perror("execlp wc");
        _exit(1);
    }

    /* === Parent process: close all fds and wait for children === */
    close(pipe1[0]);
    close(pipe1[1]);
    close(pipe2[0]);
    close(pipe2[1]);

    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);
    waitpid(pid3, NULL, 0);

    printf("Pipeline complete\n");
    return 0;
}
```

**Advanced task**: Generalize the above to implement a program that connects an arbitrary number of commands with pipes. Design a function `execute_pipeline(char *commands[], int n)` that accepts commands as a string array and dynamically creates pipes.

---

## 10. Modern Developments in Process Management

### 10.1 Containers and Process Namespaces

Container technologies like Docker leverage Linux process namespaces (PID namespaces). By isolating the process namespace, processes inside a container cannot see processes outside the container, making it appear as if they are running on an independent system.

```
PID namespace hierarchy:

  Host's PID namespace:
  +------------------------------------------------------+
  | systemd (PID=1)                                      |
  |   +-- dockerd (PID=500)                              |
  |   |   +-- containerd-shim (PID=600)                  |
  |   |       |                                          |
  |   |   +---+----------------------------------+       |
  |   |   | Container's PID namespace            |       |
  |   |   |                                      |       |
  |   |   | Host PID         Container PID       |       |
  |   |   | PID=601         -> PID=1 (init)      |       |
  |   |   | PID=602         -> PID=2 (app)       |       |
  |   |   | PID=603         -> PID=3 (worker)    |       |
  |   |   |                                      |       |
  |   |   | PID 1 inside the container is        |       |
  |   |   | unrelated to host PID 1 (systemd)    |       |
  |   |   +--------------------------------------+       |
  |   |                                                  |
  |   +-- containerd-shim (PID=700)                      |
  |       |                                              |
  |   +---+----------------------------------+           |
  |   | Another container's PID namespace    |           |
  |   | PID=701 -> PID=1                     |           |
  |   | PID=702 -> PID=2                     |           |
  |   +--------------------------------------+           |
  +------------------------------------------------------+

  Containers are a process isolation technology and, unlike VMs,
  share the kernel. Therefore, if the host's /proc is mounted
  appropriately, the host's processes can also be seen.
  VMs provide a stronger security boundary.
```

### 10.2 Resource Limiting with cgroups (Control Groups)

cgroups is a mechanism for limiting resources (CPU, memory, I/O bandwidth, etc.) at the process group level. systemd automatically creates cgroups for each service and manages resources.

```
cgroups v2 hierarchy structure:

  /sys/fs/cgroup/
  +-- cgroup.controllers    # List of available controllers
  +-- system.slice/         # Services managed by systemd
  |   +-- nginx.service/
  |   |   +-- cgroup.procs  # List of PIDs belonging to this cgroup
  |   |   +-- cpu.max       # CPU usage limit
  |   |   +-- memory.max    # Memory usage limit
  |   |   +-- pids.max      # Process count limit
  |   +-- sshd.service/
  |       +-- cgroup.procs
  |       +-- ...
  +-- user.slice/           # User sessions
      +-- user-1000.slice/
          +-- cgroup.procs
          +-- ...

  Configuration example:
  # Limit nginx CPU usage to 2 cores worth
  echo "200000 100000" > /sys/fs/cgroup/system.slice/nginx.service/cpu.max
  # -> 200ms usable in a 100ms period = 2 cores worth

  # Set memory limit to 512MB
  echo 536870912 > /sys/fs/cgroup/system.slice/nginx.service/memory.max
```

---

## FAQ

### Q1: What is the difference between a process and a thread?


### Q2: Why fork() then exec()? Why not just directly launch a new process?

The separation of fork+exec creates the flexibility of **allowing the parent to pre-configure the child process's environment**. Specifically, the following operations can be performed after fork but before exec:

- **File descriptor reassignment**: `dup2(fd, STDOUT_FILENO)` to redirect stdout to a file. Shell `>` redirection is implemented using this mechanism.
- **Pipe connection**: Connect stdin/stdout of two child processes with a pipe. Shell `|` uses this mechanism.
- **Environment variable setup**: Change the child process's environment with `setenv()`.
- **Permission changes**: Drop privileges with `setuid()` / `setgid()`.
- **Resource limits**: Set the child process's resource limits with `setrlimit()`.
- **Signal mask configuration**: Block/unblock specific signals.

Windows's CreateProcess() is monolithic, making the above flexible configuration difficult (some can be done via CreateProcess arguments, but extensibility is limited). Linux's `posix_spawn()` is a wrapper that internally calls fork+exec, and the above settings can be passed as attributes.

### Q3: Why is context switching slow?

Register save/restore alone completes in tens of nanoseconds. However, the main reasons context switching is considered "slow" are the following indirect costs:

1. **TLB flush**: When the process switches, the page table changes, invalidating the contents of the TLB (Translation Lookaside Buffer). When TLB misses become frequent, the page table must be re-read from memory for address translation, incurring a penalty of tens to hundreds of cycles. Modern CPUs with ASIDs or PCIDs (Process-Context Identifiers) can minimize TLB flushing.

2. **Cache pollution**: Until the new process's data is loaded into L1/L2/L3 caches, cache misses are frequent (the cold cache problem). The impact is greater for processes with large working sets, and cache warm-up can take an estimated tens of microseconds.

3. **Pipeline flush**: The CPU instruction pipeline is cleared, and a bubble of several cycles occurs until the pipeline is refilled.

### Q4: Why do zombie processes exist? Are they harmful?

Zombie processes exist to reliably convey the child process's exit status to the parent process. In Unix design, the child's exit code (whether it exited normally, which signal killed it, etc.) must be retained until the parent collects it with `wait()`. If the OS discarded all information as soon as the child terminates, the parent would have no way to check the child's exit state.

Zombie processes themselves consume almost no CPU time or memory (only minimal task_struct information and the PID entry). However, since PIDs are a finite resource, a massive accumulation of zombies can prevent new process creation. Additionally, `/proc` entries persist, causing administrative confusion.

### Q5: What is the nice value? How do you control process priority?

The nice value is a value for adjusting a process's CPU priority, ranging from -20 (highest priority) to +19 (lowest priority). The default is 0. The name "nice" comes from the meaning of being "nice" (courteous) to other processes, i.e., yielding CPU time.

```bash
# Checking and changing nice values
nice                          # Display current nice value
nice -n 10 ./heavy_task       # Launch program with nice value 10 (low priority)
renice -n -5 -p 1234          # Change PID 1234's nice value to -5 (requires root)

# Effect of nice values in Linux CFS:
# nice 0 process weight: 1024
# nice 1 process weight: 820  (approximately 1.25x slower)
# nice -1 process weight: 1277 (approximately 1.25x faster)
# Each nice value change results in approximately 10% change in relative CPU time
```

### Q6: How do you check how much memory a process is using?

```bash
# Method 1: /proc/[PID]/status (Linux)
grep -E "^(VmSize|VmRSS|VmSwap)" /proc/self/status
# VmSize: Virtual memory size (total of all mapped regions)
# VmRSS:  Size actually in physical memory (Resident Set Size)
# VmSwap: Size swapped out

# Method 2: ps command
ps -o pid,vsz,rss,comm -p 1234
# VSZ: Virtual memory size (KB)
# RSS: Physical memory usage (KB)

# Method 3: /proc/[PID]/smaps_rollup (Linux 4.14+)
# Check detailed memory statistics for a process
cat /proc/self/smaps_rollup

# Note: RSS includes shared library pages, so to know the
# memory usage specific to a process, check PSS
# (Proportional Set Size).
# PSS divides shared pages by the number of processes sharing them.
```

---

## Summary

| Concept | Key Points |
|---------|-----------|
| Process | A running program. Has a PID, independent virtual address space, and PCB |
| Memory layout | Text -> Data -> BSS -> Heap (grows up) -> ... -> Stack (grows down) |
| 5-state model | New -> Ready -> Running -> Blocked -> Terminated. No direct Blocked -> Running transition |
| PCB | The core data structure for OS process management. Saved/restored during context switches |
| fork/exec | Unix-style process creation. Separated design enables flexible environment configuration |
| Copy-on-Write | Dramatically reduces the cost of fork(). Pages are copied only on write |
| Context switch | CPU state save/restore. TLB flush and cache pollution are the main costs |
| Scheduling | FCFS, SJF, RR, MLFQ. Linux uses the red-black tree-based CFS |
| IPC | Pipes, shared memory, signals, sockets. Choose based on use case |
| Zombie/Orphan | Zombies occur when parent neglects wait(). Orphans are adopted by init |
| Namespaces/cgroups | Foundation of container technology. Process isolation and resource limiting |

---

## Suggested Next Reading


---

## References

1. Silberschatz, A., Galvin, P.B., Gagne, G. "Operating System Concepts." 10th Edition, Chapter 3-5, Wiley, 2018. -- A textbook-level explanation of process concepts, scheduling, and inter-process communication. Commonly known as the "dinosaur book."

2. Kerrisk, M. "The Linux Programming Interface: A Linux and UNIX System Programming Handbook." No Starch Press, 2010. -- The definitive reference for Linux system programming. Detailed explanations and extensive code examples for fork/exec, pipes, signals, and shared memory. Chapters 24-28 (process creation), Chapter 44 (pipes), and Chapters 48-54 (shared memory, semaphores) are directly relevant to this chapter.

3. Love, R. "Linux Kernel Development." 3rd Edition, Addison-Wesley, 2010. -- Explains the internal implementation of Linux kernel process management and the CFS scheduler. Contains detailed descriptions of the task_struct structure, kernel-internal process creation processing, and the CFS red-black tree implementation.

4. Tanenbaum, A.S., Bos, H. "Modern Operating Systems." 4th Edition, Pearson, 2014. -- A seminal work on OS theory. Rich in process state transition models, mathematical analysis of scheduling algorithms, and comparisons of IPC mechanisms.

5. Stevens, W.R., Rago, S.A. "Advanced Programming in the UNIX Environment." 3rd Edition, Addison-Wesley, 2013. -- Known as "APUE." A practical reference for Unix/Linux programming. Implementation patterns for fork/exec, pipes, and signal processing are covered in detail.

6. Linux man pages: `fork(2)`, `exec(3)`, `wait(2)`, `pipe(2)`, `signal(7)`, `proc(5)`, `cgroups(7)`. -- Official system call documentation. Contains precise specifications, error codes, and caveats for each system call. Access via `man 2 fork`, etc.
