# Imperative Programming

> Imperative programming is the most intuitive paradigm that "instructs the computer step by step," directly corresponding to the von Neumann architecture. This chapter systematically covers the essence of imperative programming, from structured programming and procedural programming to imperative styles in modern languages.

## What You Will Learn in This Chapter

- [ ] Explain the essence and historical background of imperative programming
- [ ] Correctly distinguish and use the three control structures of structured programming
- [ ] Understand the concepts of state management and side effects, and write safe code
- [ ] Apply modularization design principles in procedural programming
- [ ] Compare and analyze the differences between imperative and other paradigms (declarative, functional, OOP)
- [ ] Recognize anti-patterns specific to imperative programming and practice avoidance strategies


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. What Is Imperative Programming

### 1.1 Definition and Basic Concepts

Imperative Programming is a programming paradigm that describes a program as a "sequence of instructions." The programmer instructs the computer not "What" to do, but "How" to perform the processing step by step.

The word "imperative" derives from the Latin "imperare" (to command). Just as a cooking recipe lists steps such as "crack the egg -> add salt -> stir -> fry in a pan," an imperative program describes each step of computation in order.

The core of imperative programming can be summarized in the following three elements.

```
Three Pillars of Imperative Programming
============================================

  1. State
     - Holds values in memory as variables
     - Represents the "current situation" of the program

  2. Assignment
     - An operation that changes the value of a variable
     - x = x + 1 is mathematically contradictory, but
       in imperative programming it means "increment the value of x by 1"

  3. Control Flow
     - Controls the execution order of instructions
     - Three structures: sequence (sequential execution),
       selection (conditional branching), and
       iteration (loops)
```

### 1.2 Correspondence with the Von Neumann Architecture

To understand why imperative programming is the most "natural" programming style, it is necessary to know its correspondence with the von Neumann architecture, the fundamental structure of modern computers.

```
Correspondence Between Von Neumann Architecture and Imperative Programming
==========================================================

  ┌─────────────────────────────────────────────────────┐
  │              Von Neumann Machine                     │
  │                                                     │
  │  ┌───────────┐    Bus     ┌──────────────────┐     │
  │  │   CPU     │◄─────────►│    Memory         │     │
  │  │           │           │                  │     │
  │  │ PC(next)  │           │ Program Area     │     │
  │  │ ACC(calc) │           │  Instruction 1   │     │
  │  │ IR(instr) │           │  Instruction 2   │     │
  │  │           │           │  Instruction 3...│     │
  │  └───────────┘           │                  │     │
  │                          │ Data Area        │     │
  │                          │  Variable A = 10 │     │
  │                          │  Variable B = 20 │     │
  │                          │  Variable C = ?  │     │
  │                          └──────────────────┘     │
  └─────────────────────────────────────────────────────┘

  Correspondence:
  ┌─────────────────┬──────────────────────────────┐
  │ Hardware         │ Imperative Programming        │
  ├─────────────────┼──────────────────────────────┤
  │ Memory cell      │ Variable                      │
  │ Store instruction│ Assignment (x = 10)           │
  │ Load instruction │ Variable reference (print(x)) │
  │ Program counter  │ Current execution position    │
  │ Branch instruction│ if-else statement             │
  │ Jump instruction │ goto / loop                   │
  │ Subroutine call  │ Function call                 │
  └─────────────────┴──────────────────────────────┘
```

In this architecture proposed by John von Neumann in 1945, programs and data are stored in the same memory, and the CPU fetches instructions from memory one by one, decodes them, and executes them. Imperative programming is nothing other than an abstraction of this fetch-decode-execute cycle.

### 1.3 Basic Example of Imperative Programming: Calculating a Sum

Let us look at the simplest example of an imperative program that calculates the sum of numbers in a list.

**Imperative style in Python:**

```python
def sum_imperative(numbers):
    """
    Calculate the sum of a list in imperative style.
    Uses a state variable total, updated sequentially in a loop.
    """
    total = 0                    # Initialize state
    index = 0                    # Initialize loop counter
    while index < len(numbers):  # Iteration (check termination condition)
        total = total + numbers[index]  # Update state (assignment)
        index = index + 1               # Update counter
    return total                 # Return final state

# Execution example
data = [10, 20, 30, 40, 50]
result = sum_imperative(data)
print(f"Sum: {result}")  # Sum: 150
```

**Equivalent code in C:**

```c
#include <stdio.h>

int sum_imperative(int numbers[], int length) {
    int total = 0;                  /* Initialize state */
    int index = 0;                  /* Initialize loop counter */
    while (index < length) {        /* Iteration */
        total = total + numbers[index];  /* Update state */
        index = index + 1;              /* Update counter */
    }
    return total;                   /* Return final state */
}

int main(void) {
    int data[] = {10, 20, 30, 40, 50};
    int length = sizeof(data) / sizeof(data[0]);
    int result = sum_imperative(data, length);
    printf("Sum: %d\n", result);   /* Sum: 150 */
    return 0;
}
```

In this code, the variable `total` holds the state of the "current sum," and the state is updated by assignment statements in each iteration of the loop. The value of `total` at each point in the program transitions as follows.

```
State Transition Trace (numbers = [10, 20, 30, 40, 50])
=================================================

  Iteration   index   numbers[index]   total (before)   total (after)
  ─────────────────────────────────────────────────────────────────
  Initial      0         -                -                0
  1st          0        10                0               10
  2nd          1        20               10               30
  3rd          2        30               30               60
  4th          3        40               60              100
  5th          4        50              100              150
  ─────────────────────────────────────────────────────────────────
  End          5         -              150        → return 150
```

### 1.4 Imperative vs Declarative: A Fundamental Difference in Thinking

To understand imperative programming, it is helpful to compare it with the contrasting paradigm of Declarative Programming.

```
Fundamental Difference Between Imperative and Declarative
========================================

  ┌──────────────────────────────────────────────────────┐
  │  Imperative: Describes "how" to compute               │
  │                                                      │
  │  Like a procedure manual:                             │
  │  Step 1: Prepare an empty list                        │
  │  Step 2: Examine each element of the original list    │
  │  Step 3: If it meets the condition, add to new list   │
  │  Step 4: Finish when all elements are processed       │
  └──────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────┐
  │  Declarative: Describes "what" is desired              │
  │                                                      │
  │  Like an order form:                                  │
  │  "Give me a list of product names priced 100 or above"│
  │  → The system decides the specific procedure           │
  └──────────────────────────────────────────────────────┘
```

Let us compare with concrete code.

**Imperative (Python):**

```python
# Imperative: Instruct step by step "how" to filter
items = [
    {"name": "Laptop", "price": 150000},
    {"name": "Mouse", "price": 3000},
    {"name": "Keyboard", "price": 12000},
    {"name": "Monitor", "price": 45000},
    {"name": "USB Cable", "price": 800},
]

expensive_items = []                  # Prepare an empty list
for item in items:                    # Process each element sequentially
    if item["price"] >= 10000:        # Evaluate the condition
        expensive_items.append(item["name"])  # Add to the result

print(expensive_items)
# ['Laptop', 'Keyboard', 'Monitor']
```

**Declarative (SQL):**

```sql
-- Declarative: Describe only "what" is desired
SELECT name
FROM items
WHERE price >= 10000;

-- The execution engine automatically determines whether to use an index, table scan, etc.
```

**Declarative (Python list comprehension):**

```python
# Declarative-leaning style
expensive_items = [item["name"] for item in items if item["price"] >= 10000]
```

| Comparison Item | Imperative | Declarative |
|---------|--------|--------|
| Description | Procedure (How) | Result (What) |
| Abstraction Level | Low (close to the machine) | High (close to humans) |
| State Management | Managed explicitly | Managed by the runtime |
| Side Effects | Occur frequently | Minimized |
| Optimization | Performed by the developer | Performed by the runtime |
| Learning Curve | Intuitive and approachable | Requires a shift in thinking |
| Debugging | Easy to trace with step execution | Internal behavior is less visible |
| Representative Languages | C, Pascal, BASIC | SQL, HTML, Haskell |
| Application Domain | Systems programming, control processing | Data queries, UI definition |

---

## 2. History of Imperative Programming

### 2.1 The Era of Machine Code and Assembly Language (1940s-Early 1950s)

The history of imperative programming overlaps with the history of computers themselves. The earliest programming was done in machine code.

In the 1940s, early computers such as ENIAC (1945) and EDVAC (1949) were programmed by changing wiring connections or setting switches. With the introduction of the stored-program concept proposed by von Neumann, programs could be stored in memory just like data.

```
Example of a Machine Code Program (Hypothetical 8-bit CPU)
==========================================

  Address   Machine Code    Meaning
  ─────────────────────────────────────────
  0000:     0001 0001    LOAD  R1, [addr1]    ; Load from memory into R1
  0001:     0001 0010    LOAD  R2, [addr2]    ; Load from memory into R2
  0010:     0010 0001    ADD   R1, R2         ; R1 = R1 + R2
  0011:     0011 0001    STORE R1, [addr3]    ; Store R1 into memory
  0100:     1111 0000    HALT                 ; Halt

  → A sequence of 0s and 1s, extremely difficult for humans to read
```

In the early 1950s, assembly language appeared, assigning mnemonics (human-readable symbolic names) to machine instructions. While readability improved, it remained low-level imperative programming closely tied to hardware.

```nasm
; x86 assembly language example: Adding two numbers
section .data
    num1 dd 10          ; 32-bit integer 10
    num2 dd 20          ; 32-bit integer 20
    result dd 0         ; Storage for result

section .text
    global _start

_start:
    mov eax, [num1]     ; Load the value of num1 into EAX register
    add eax, [num2]     ; Add the value of num2 to EAX
    mov [result], eax   ; Store the result in memory

    ; Exit program (Linux system call)
    mov eax, 1          ; sys_exit
    xor ebx, ebx        ; Exit code 0
    int 0x80            ; Kernel call
```

### 2.2 The Advent of FORTRAN (1957)

In 1957, a team led by John Backus at IBM released FORTRAN (FORmula TRANslation). This was the world's first practical high-level programming language and a groundbreaking turning point in the history of imperative programming.

FORTRAN was designed to allow mathematical formulas to be written almost directly as programs. It spread explosively in the field of scientific and technical computing, enabling programs to be written with about one-twentieth of the effort compared to assembly language.

```fortran
C     Numerical Integration by Trapezoidal Rule in FORTRAN 77
C     A typical example of imperative programming
      PROGRAM TRAPEZOID
      IMPLICIT NONE
      INTEGER N, I
      REAL A, B, H, SUM, X

C     Set integration interval and number of divisions
      A = 0.0
      B = 1.0
      N = 1000

C     Calculate step size
      H = (B - A) / N

C     Numerical integration using trapezoidal rule
      SUM = 0.0
      DO 10 I = 1, N - 1
          X = A + I * H
          SUM = SUM + X * X
   10 CONTINUE

      SUM = H * (A*A/2.0 + SUM + B*B/2.0)

      WRITE(*,*) 'Integral of x^2 from 0 to 1 =', SUM
C     Theoretical value: 1/3 = 0.333333...
      STOP
      END
```

### 2.3 ALGOL and the Seeds of Structured Programming (1958-1960s)

ALGOL 58 was developed in 1958 and ALGOL 60 in 1960 by an international committee. ALGOL (ALGOrithmic Language) was an extremely important language that influenced nearly all subsequent imperative languages.

The innovative concepts introduced by ALGOL 60 were numerous:

- Block structure (begin ... end)
- Lexical scoping (nested scope of variable visibility)
- Recursive calls
- Formal syntax definition using BNF (Backus-Naur Form)
- Call by value and call by name

ALGOL was widely accepted in academia but did not gain widespread adoption on commercial computers. However, its design philosophy greatly influenced subsequent languages such as Pascal, C, and Java.

### 2.4 The Birth and Spread of C (1972)

In 1972, Dennis Ritchie at Bell Labs developed the C language. C was the successor to the B language developed by Ken Thompson and was designed to write the UNIX operating system.

The distinguishing feature of C was that it combined the abstraction capability of a high-level language with low-level operations close to assembly language. It provided the essential features of imperative programming without excess or deficiency: direct memory manipulation through pointers, data organization through structures, and modularization through functions.

```c
/*
 * Dynamic array implementation in C
 * An example that concentrates the characteristics of imperative programming:
 *   - Explicit memory management (malloc/realloc/free)
 *   - Indirect reference through pointers
 *   - State encapsulation through structures
 *   - Operation definition through procedures (functions)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int *data;       /* Pointer to array storing elements */
    int size;        /* Current number of elements */
    int capacity;    /* Allocated capacity */
} DynamicArray;

/* Initialization */
DynamicArray *array_create(int initial_capacity) {
    DynamicArray *arr = malloc(sizeof(DynamicArray));
    if (arr == NULL) return NULL;
    arr->data = malloc(sizeof(int) * initial_capacity);
    if (arr->data == NULL) {
        free(arr);
        return NULL;
    }
    arr->size = 0;
    arr->capacity = initial_capacity;
    return arr;
}

/* Add an element */
int array_push(DynamicArray *arr, int value) {
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        int *new_data = realloc(arr->data, sizeof(int) * new_capacity);
        if (new_data == NULL) return -1;
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    arr->data[arr->size] = value;
    arr->size++;
    return 0;
}

/* Get an element */
int array_get(DynamicArray *arr, int index) {
    if (index < 0 || index >= arr->size) {
        fprintf(stderr, "Index out of bounds: %d\n", index);
        exit(1);
    }
    return arr->data[index];
}

/* Free memory */
void array_destroy(DynamicArray *arr) {
    if (arr != NULL) {
        free(arr->data);
        free(arr);
    }
}

int main(void) {
    DynamicArray *arr = array_create(4);

    for (int i = 0; i < 10; i++) {
        array_push(arr, i * 10);
    }

    printf("Size: %d, Capacity: %d\n", arr->size, arr->capacity);
    /* Size: 10, Capacity: 16 */

    for (int i = 0; i < arr->size; i++) {
        printf("[%d] = %d\n", i, array_get(arr, i));
    }

    array_destroy(arr);
    return 0;
}
```

### 2.5 Genealogy of Imperative Languages

```
Genealogy of Imperative Programming Languages
══════════════════════════════════════════════════════════════

  1940s   Machine Code ───┐
                          │
  1950s   Assembly ───────┤
          FORTRAN(1957)───┤──── Foundation of scientific computing
          COBOL(1959) ────┤──── Business data processing
                          │
  1960s   ALGOL 60 ───────┤──── Ancestor of all structured languages
          BASIC(1964) ────┤──── Educational, for beginners
          PL/I(1964)      │
                          │
  1970s   Pascal(1970)────┤──── Education & structured programming
          C(1972)     ────┤──── Systems programming
          ↓               │
  1980s   C++(1983)   ────┤──── C + object-oriented
          Ada(1983)       │
          Perl(1987)  ────┤──── Text processing & scripting
                          │
  1990s   Python(1991)────┤──── Multi-paradigm
          Java(1995)  ────┤──── OOP + VM
          PHP(1995)   ────┤──── Web development
          Ruby(1995)      │
          JavaScript(1995)
                          │
  2000s   C#(2000)    ────┤──── .NET platform
          Go(2009)    ────┤──── Concurrency-focused
                          │
  2010s   Rust(2010)  ────┤──── Safety + performance
          Kotlin(2011)────┤──── Modern language on JVM
          Swift(2014) ────┤──── Apple platform
          TypeScript(2012)
                          │
  2020s   All languages becoming multi-paradigm
          Fusion of imperative + functional + OOP is mainstream
```

---

## 3. Control Structures

### 3.1 The Structured Program Theorem

In 1966, Corrado Bohm and Giuseppe Jacopini mathematically proved that the control flow of any program can be expressed using only combinations of the following three basic structures. This is the "Structured Program Theorem."

```
Structured Program Theorem: Three Basic Control Structures
═══════════════════════════════════════════════════

  1. Sequence                    2. Selection
  ┌─────────┐                ┌─────────┐
  │ Stmt 1  │                │ Cond P? │
  └────┬────┘                └──┬───┬──┘
       │                    Yes │   │ No
  ┌────▼────┐            ┌─────▼┐ ┌▼─────┐
  │ Stmt 2  │            │ Stmt A│ │Stmt B│
  └────┬────┘            └──┬───┘ └──┬───┘
       │                    │        │
  ┌────▼────┐               └───┬────┘
  │ Stmt 3  │                   │
  └─────────┘                   ▼

  3. Iteration
       │
  ┌────▼────┐
  │ Cond P? │◄──────┐
  └──┬───┬──┘       │
  Yes│   │No    ┌───┴───┐
     │   │      │ Stmt S│
     │   ▼      └───────┘
     │
     └──────►┌───────┐
             │ Stmt S│──────┐
             └───────┘      │
                            │
             ┌──────────────┘
             │
             ▼ (back to Cond P)

  Meaning of the Structured Program Theorem:
  Even without goto statements, any algorithm can be
  expressed using only combinations of these 3 structures.
```

### 3.2 Sequential Structure (Sequence)

Sequential structure is the most basic control structure, executing statements from top to bottom in order. Assignment statements, function calls, I/O operations, and nearly all statements follow sequential structure.

```python
# Example of sequential structure: Calculate area and circumference of a circle
import math

radius = 5.0                                # Stmt 1: Set the radius
area = math.pi * radius ** 2                # Stmt 2: Calculate the area
circumference = 2 * math.pi * radius        # Stmt 3: Calculate the circumference

print(f"Radius: {radius}")                  # Stmt 4: Output the result
print(f"Area: {area:.2f}")                  # Stmt 5
print(f"Circumference: {circumference:.2f}")# Stmt 6

# Output:
# Radius: 5.0
# Area: 78.54
# Circumference: 31.42
```

It is important that in sequential structure, the order of statements affects the result.

```python
# Example where order affects the result
x = 10
y = x + 5    # y = 15
x = 20       # Even though x is changed, y is already computed
print(y)     # 15 (not 20 + 5 = 25)

# Contrast: In a declarative spreadsheet,
# if cell B1 = A1 + 5, changing A1 automatically updates B1
# → This is the fundamental difference between imperative and declarative
```

### 3.3 Selection Structure

Selection structure is a control structure that selects which statement to execute based on the truth value of a conditional expression.

**Basic if-else statement (Python):**

```python
def classify_temperature(celsius):
    """
    Example of selection structure for classifying temperature.
    A pattern of chaining multiple conditions with elif.
    """
    if celsius >= 35:
        category = "Extreme Heat"
        advice = "Stay indoors and hydrate frequently"
    elif celsius >= 30:
        category = "Midsummer Day"
        advice = "Beware of heatstroke"
    elif celsius >= 25:
        category = "Summer Day"
        advice = "Stay moderately hydrated"
    elif celsius >= 15:
        category = "Comfortable"
        advice = "Pleasant temperature"
    elif celsius >= 5:
        category = "Chilly"
        advice = "Bring a jacket"
    elif celsius >= 0:
        category = "Cold"
        advice = "Ensure proper cold-weather protection"
    else:
        category = "Severe Cold"
        advice = "Watch out for freezing conditions"

    return category, advice

# Execution example
for temp in [38, 28, 10, -3]:
    cat, adv = classify_temperature(temp)
    print(f"{temp}°C → {cat}: {adv}")

# Output:
# 38°C → Extreme Heat: Stay indoors and hydrate frequently
# 28°C → Summer Day: Stay moderately hydrated
# 10°C → Chilly: Bring a jacket
# -3°C → Severe Cold: Watch out for freezing conditions
```

**switch-case statement (C):**

```c
#include <stdio.h>

/*
 * Example of multi-way branching with a switch statement.
 * Returns the day name from a day number.
 * Note: Forgetting break causes fall-through.
 */
const char *day_name(int day) {
    switch (day) {
        case 0: return "Sunday";
        case 1: return "Monday";
        case 2: return "Tuesday";
        case 3: return "Wednesday";
        case 4: return "Thursday";
        case 5: return "Friday";
        case 6: return "Saturday";
        default: return "Unknown";
    }
}

/* Example of intentional fall-through */
void classify_day(int day) {
    switch (day) {
        case 0:
        case 6:
            printf("%s is a holiday\n", day_name(day));
            break;
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
            printf("%s is a weekday\n", day_name(day));
            break;
        default:
            printf("Invalid day number: %d\n", day);
            break;
    }
}

int main(void) {
    for (int i = 0; i <= 7; i++) {
        classify_day(i);
    }
    return 0;
}
/*
 * Output:
 * Sunday is a holiday
 * Monday is a weekday
 * Tuesday is a weekday
 * Wednesday is a weekday
 * Thursday is a weekday
 * Friday is a weekday
 * Saturday is a holiday
 * Invalid day number: 7
 */
```

### 3.4 Iteration Structure

Iteration structure is a control structure that repeatedly executes a statement while a condition is satisfied (or until it is satisfied).

**while loop (pre-test iteration):**

```python
def binary_search(sorted_list, target):
    """
    Binary search: A typical example of iteration using a while loop.
    Efficiently searches for a target element in a sorted list.

    - Pre-test loop: Checks the condition before executing the loop body
    - Updates state variables low, high to halve the search range
    """
    low = 0
    high = len(sorted_list) - 1

    while low <= high:                       # Pre-test: While search range exists
        mid = (low + high) // 2              # Calculate middle index
        if sorted_list[mid] == target:       # Selection: Found
            return mid
        elif sorted_list[mid] < target:      # Selection: Search right half
            low = mid + 1                    # Update state
        else:                                # Selection: Search left half
            high = mid - 1                   # Update state

    return -1  # Not found

# Execution example
data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
print(binary_search(data, 23))   # 5 (exists at index 5)
print(binary_search(data, 50))   # -1 (does not exist)
```

**for loop (count-controlled iteration):**

```python
def selection_sort(arr):
    """
    Selection sort: A typical example of nested loops.
    Outer loop: Processes positions to be finalized in order
    Inner loop: Scans the unsorted portion for the minimum value
    """
    n = len(arr)
    for i in range(n - 1):              # Outer loop: Repeat n-1 times
        min_index = i
        for j in range(i + 1, n):       # Inner loop: Scan unsorted portion
            if arr[j] < arr[min_index]:
                min_index = j
        # Swap minimum with the front (state change via assignment)
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

# Execution example
data = [64, 25, 12, 22, 11]
print(f"Before sort: {data}")
result = selection_sort(data)
print(f"After sort: {result}")

# Before sort: [64, 25, 12, 22, 11]
# After sort: [11, 12, 22, 25, 64]
```

**do-while loop (post-test iteration):**

```c
#include <stdio.h>

/*
 * Example of a do-while loop: User input validation
 * Suitable when "at least one execution" is needed.
 * Python does not have a do-while construct, so this is shown in C.
 */
int main(void) {
    int number;

    do {
        printf("Enter an integer from 1 to 100: ");
        scanf("%d", &number);

        if (number < 1 || number > 100) {
            printf("Out of range. Please try again.\n");
        }
    } while (number < 1 || number > 100);  /* Post-test */

    printf("Entered value: %d\n", number);
    return 0;
}
```

### 3.5 Combining Control Structures

In actual programs, the three control structures are freely combined to construct algorithms.

```python
def find_primes_sieve(limit):
    """
    Sieve of Eratosthenes: An example combining three control structures.

    - Sequence: Initialization, result collection
    - Selection: Determining whether a number is prime
    - Iteration: Sieve processing, eliminating multiples
    """
    # Sequence: Initialization
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False

    # Iteration (outer): From 2 to sqrt(limit)
    i = 2
    while i * i <= limit:
        # Selection: Process only if i is still marked as prime
        if is_prime[i]:
            # Iteration (inner): Eliminate multiples of i
            j = i * i
            while j <= limit:
                is_prime[j] = False  # Assignment: Update state
                j += i
        i += 1

    # Sequence: Collect results
    primes = []
    for num in range(2, limit + 1):
        if is_prime[num]:            # Selection
            primes.append(num)

    return primes

# Execution example
result = find_primes_sieve(50)
print(f"Primes up to 50: {result}")
# Primes up to 50: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
print(f"Count: {len(result)}")
# Count: 15
```

---

## 4. State Management and Side Effects

### 4.1 What Is State

"State" in imperative programming refers to the collection of all variable values held in memory during program execution. The state at each point in the program is determined by the results of all assignment statements executed up to that point.

```
Concept of State
════════════════════════════════════════════

  Program:              State Transitions:
  ─────────               ──────────
  x = 10                  S0: { }
      ↓                        ↓
  y = 20                  S1: { x=10 }
      ↓                        ↓
  z = x + y               S2: { x=10, y=20 }
      ↓                        ↓
  x = z * 2               S3: { x=10, y=20, z=30 }
      ↓                        ↓
  (end)                    S4: { x=60, y=20, z=30 }

  Key Points:
  - Each statement execution is a "state→state" transformation (state transition)
  - The same variable x holds different values at different points
  - The "meaning" of a program can be understood as a sequence of state transitions
```

### 4.2 Side Effects

A side effect refers to the modification of external state by a function or subroutine beyond returning a value. Specifically, the following operations are side effects:

1. Modifying global variables
2. Modifying objects passed as arguments (destructive operations)
3. Writing to files
4. Screen output
5. Network communication
6. Database updates

```python
# === Functions with side effects ===

# Example 1: Modifying a global variable
counter = 0

def increment():
    """Modifies the global variable counter each time it is called (side effect)"""
    global counter
    counter += 1
    return counter

print(increment())  # 1
print(increment())  # 2
print(increment())  # 3
# → Returns a different result each time even with the same arguments (none)


# Example 2: Destructive modification of an argument
def remove_negatives(numbers):
    """Directly modifies the original list (side effect)"""
    i = 0
    while i < len(numbers):
        if numbers[i] < 0:
            numbers.pop(i)     # Modifies the original list
        else:
            i += 1
    return numbers

data = [3, -1, 4, -1, 5, -9, 2, 6]
result = remove_negatives(data)
print(result)   # [3, 4, 5, 2, 6]
print(data)     # [3, 4, 5, 2, 6] ← The original list has also changed!


# === Functions without side effects (pure functions) ===

def remove_negatives_pure(numbers):
    """Creates and returns a new list (no side effects)"""
    result = []
    for num in numbers:
        if num >= 0:
            result.append(num)
    return result

data = [3, -1, 4, -1, 5, -9, 2, 6]
result = remove_negatives_pure(data)
print(result)   # [3, 4, 5, 2, 6]
print(data)     # [3, -1, 4, -1, 5, -9, 2, 6] ← The original list is unchanged
```

### 4.3 Referential Transparency

Referential transparency is the property where replacing an expression with the value it evaluates to does not change the behavior of the program. Pure functions without side effects possess referential transparency.

```python
# Referentially transparent function
def add(a, b):
    return a + b

# add(3, 4) can always be replaced with 7
x = add(3, 4) + add(3, 4)   # x = 7 + 7 = 14
y = 7 + 7                    # y = 14
# x and y are always equal → referentially transparent

# Non-referentially-transparent function
import random

def random_add(a, b):
    return a + b + random.randint(0, 10)

# The result of random_add(3, 4) differs with each call
# → Replacing it with a value changes the meaning → not referentially transparent
```

### 4.4 Problems Caused by Mutable State

"Mutable state," the foundation of imperative programming, is also a major source of program complexity.

```python
# === Problem with mutable state: Aliasing ===

def process_orders(orders, vip_orders):
    """
    Process order lists.
    Problem: If orders and vip_orders reference the same object,
    unintended side effects occur.
    """
    # Add shipping to regular orders
    for order in orders:
        order["total"] += 500

    # VIP orders have free shipping
    for order in vip_orders:
        order["total"] -= 500    # Subtract shipping

    return orders, vip_orders

# Problematic case: The same dictionary object exists in both lists
shared_order = {"item": "Book", "total": 3000}
orders = [shared_order]
vip_orders = [shared_order]  # Reference to the same object!

process_orders(orders, vip_orders)
print(shared_order)
# {"item": "Book", "total": 3000}
# → +500 then -500 returns to the original, but
#   this is coincidental; changing the processing order changes the result

# === Solution: Defensive copying ===

import copy

def process_orders_safe(orders, vip_orders):
    """Protects original data through defensive copying"""
    orders_copy = copy.deepcopy(orders)
    vip_copy = copy.deepcopy(vip_orders)

    for order in orders_copy:
        order["total"] += 500

    for order in vip_copy:
        order["total"] -= 500

    return orders_copy, vip_copy
```

### 4.5 Memory Model and Variable Lifecycle

In imperative programming, variables correspond to memory regions. It is important to correctly manage the lifecycle of variables (creation, use, release).

```
Memory Layout (Typical C Program)
═══════════════════════════════════════════════

  High Address
  ┌──────────────────────┐
  │  Command Line Args    │
  │  Environment Vars     │
  ├──────────────────────┤
  │                      │
  │  Stack Area      ↓   │ ← Local variables, function args, return addresses
  │                      │    Grows with each function call (LIFO)
  │  (...free space...)  │
  │                      │
  │  Heap Area       ↑   │ ← Dynamically managed with malloc/free
  │                      │    Programmer's responsibility to free
  ├──────────────────────┤
  │  BSS Segment         │ ← Uninitialized global variables (zero-initialized)
  ├──────────────────────┤
  │  Data Segment        │ ← Initialized global variables, static variables
  ├──────────────────────┤
  │  Text Segment        │ ← Program code (read-only)
  └──────────────────────┘
  Low Address

  Variable Types and Storage Locations:
  ┌──────────────┬─────────────┬─────────────────────┐
  │ Variable Type │ Location    │ Lifecycle            │
  ├──────────────┼─────────────┼─────────────────────┤
  │ Local var    │ Stack       │ During function exec  │
  │ Global var   │ Data/BSS   │ Entire program        │
  │ Dynamic alloc│ Heap       │ malloc to free        │
  │ Static var   │ Data       │ Entire program        │
  └──────────────┴─────────────┴─────────────────────┘
```

```c
#include <stdio.h>
#include <stdlib.h>

int global_var = 100;           /* Data segment */
int uninitialized_global;       /* BSS segment (zero-initialized) */

void demonstrate_memory(void) {
    int local_var = 42;                       /* Stack */
    static int static_var = 0;                /* Data segment */
    int *heap_var = malloc(sizeof(int));       /* Heap */

    static_var++;  /* Value persists across function calls */
    *heap_var = static_var * 10;

    printf("local:  %d (addr: %p)\n", local_var, (void *)&local_var);
    printf("static: %d (addr: %p)\n", static_var, (void *)&static_var);
    printf("heap:   %d (addr: %p)\n", *heap_var, (void *)heap_var);
    printf("global: %d (addr: %p)\n", global_var, (void *)&global_var);

    free(heap_var);  /* Free heap memory */
}

int main(void) {
    printf("=== 1st call ===\n");
    demonstrate_memory();
    printf("\n=== 2nd call ===\n");
    demonstrate_memory();
    return 0;
}
/*
 * Example output:
 * === 1st call ===
 * local:  42 (addr: 0x7ffc...)
 * static: 1  (addr: 0x601...)
 * heap:   10 (addr: 0x1a3...)
 * global: 100 (addr: 0x601...)
 *
 * === 2nd call ===
 * local:  42 (addr: 0x7ffc...)    ← local is reinitialized each time
 * static: 2  (addr: 0x601...)     ← static retains its value
 * heap:   20 (addr: 0x1a3...)
 * global: 100 (addr: 0x601...)
 */
```

---

## 5. Structured Programming

### 5.1 Background of the "Go To Considered Harmful" Argument

In the late 1960s, the scale and complexity of software grew rapidly, leading to a situation called the "Software Crisis." Project delays, budget overruns, and rampant bugs were commonplace.

One cause identified for this crisis was the unstructured control flow resulting from heavy use of goto statements in programs of that era. Programs using goto statements indiscriminately produced entangled control flows, creating code that was extremely difficult to understand and maintain -- so-called "spaghetti code."

```
Example of Spaghetti Code Using goto (Pseudocode)
═══════════════════════════════════════════════════

  10: READ X
  20: IF X < 0 GOTO 80
  30: Y = X * 2
  40: IF Y > 100 GOTO 70
  50: PRINT Y
  60: GOTO 10
  70: PRINT "TOO LARGE"
  75: GOTO 10
  80: IF X = -1 GOTO 100
  90: PRINT "NEGATIVE"
  95: GOTO 10
  100: END

  Control flow diagram:
  10 → 20 → 30 → 40 → 50 → 60 → 10 (loop)
         ↓              ↓
        80 → 90 → 95   70 → 75
         ↓    ↑           ↑
        100   10          10

  → Control flow becomes complexly entangled with jumps back and forth
  → Modifying part of the code can cause unexpected effects elsewhere
```

### 5.2 Dijkstra's Proposal

In 1968, Dutch computer scientist Edsger W. Dijkstra published a letter titled "Go To Statement Considered Harmful" in the Communications of the ACM. This short paper (actually a letter to the editor) had a profound impact on the history of programming.

The core of Dijkstra's argument was the following:

1. **Excessive use of goto statements degrades program quality**: When goto statements are heavily used, tracking program execution state becomes difficult, making it extremely hard to reason about program correctness.

2. **Structured control flow is necessary**: Using only the three control structures -- sequence, selection, and iteration -- makes program structure clear and verification of correctness easier.

3. **Correspondence between static structure and dynamic behavior**: By eliminating goto statements, the runtime behavior (dynamic behavior) of a program can be easily inferred from its source code (static structure).

### 5.3 Principles of Structured Programming

Based on Dijkstra's proposal, structured programming was summarized as the following principles.

```
Principles of Structured Programming
═════════════════════════════════════════

  1. Single Entry, Single Exit
     - Each control structure block has one entry and one exit
     - No jumping into or out of the middle

  2. Use Only Three Basic Control Structures
     - Sequence
     - Selection: if-then-else
     - Iteration: while-do

  3. Top-Down Design
     - Decompose problems from large units to small units
     - Hide details at each level (stepwise refinement)

  4. Stepwise Refinement
     - Systematized by Niklaus Wirth (1971)
     - Gradually expand from abstract descriptions to concrete implementations
```

### 5.4 Comparison of BASIC Code with goto and Structured Code

**BASIC (using goto):**

```basic
10 REM Fibonacci sequence (goto version)
20 LET A = 0
30 LET B = 1
40 LET N = 0
50 PRINT A
60 LET N = N + 1
70 IF N >= 10 THEN GOTO 120
80 LET T = A + B
90 LET A = B
100 LET B = T
110 GOTO 50
120 END
```

**Python (structured programming):**

```python
def fibonacci_sequence(count):
    """
    Structured Fibonacci sequence generation.
    - No goto statements; iteration expressed with while loop
    - Extracted as a function, making it reusable
    - Variable scope is limited to within the function
    """
    a, b = 0, 1
    result = []
    n = 0

    while n < count:       # Iteration: Repeat until specified count
        result.append(a)   # Sequence: Add to result list
        a, b = b, a + b    # Sequence: Calculate next term
        n += 1             # Sequence: Update counter

    return result

# Execution example
fib = fibonacci_sequence(10)
print(fib)
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### 5.5 Impact and Limitations of Structured Programming

Structured programming brought revolutionary changes to software development.

**Impact:**

1. **Dramatic improvement in readability**: Program control flow became predictable
2. **Improved maintainability**: Changing part of the code had limited impact on other parts
3. **Verifiability**: It became easier to logically reason about program correctness
4. **Influence on language design**: Pascal (1970) was designed as an educational language embodying structured programming

**Limitations:**

Structured programming succeeded in eliminating goto statements and organizing control structures, but did not provide sufficient guidance on data organization. As programs grew in scale, managing data structures and their associated operations became a new challenge. One answer to this challenge was object-oriented programming.

Also, the strict "single exit" principle has been relaxed in modern practice. Exception handling (try-catch), early returns (guard clauses), and break statements are not goto statements, but they strictly violate the single exit principle. However, it is widely recognized that when used appropriately, these improve readability.

```python
# Example of early return (guard clause)
# Violates strict single exit but has high readability

def calculate_discount(customer, amount):
    """Example of early return using guard clauses"""
    # Guard clauses: Handle abnormal cases first and return early
    if customer is None:
        return 0
    if amount <= 0:
        return 0
    if not customer.get("is_active", False):
        return 0

    # Normal case processing (nesting is shallow)
    base_rate = 0.05
    if customer.get("is_vip", False):
        base_rate = 0.15
    elif customer.get("years", 0) >= 3:
        base_rate = 0.10

    return amount * base_rate
```

---

## 6. Procedural Programming

### 6.1 Basic Concepts of Procedural Programming

Procedural Programming is a sub-paradigm of imperative programming that organizes programs as collections of "procedures" or "functions."

While imperative programming structures a program as a "sequence of statements," procedural programming names groups of related statements and extracts them as reusable units (procedures).

```
Relationship Between Imperative and Procedural Programming
═══════════════════════════════════════════════════

  ┌──────────────────────────────────────────────┐
  │         Imperative Programming                │
  │  (State + Assignment + Control Flow)          │
  │                                              │
  │  ┌──────────────────────────────────────┐    │
  │  │   Structured Programming              │    │
  │  │   (goto elimination, 3 control structs)│    │
  │  │                                      │    │
  │  │  ┌──────────────────────────────┐   │    │
  │  │  │  Procedural Programming       │   │    │
  │  │  │  (Modularization via          │   │    │
  │  │  │   functions/procedures)       │   │    │
  │  │  └──────────────────────────────┘   │    │
  │  │                                      │    │
  │  │  ┌──────────────────────────────┐   │    │
  │  │  │  Object-Oriented Programming  │   │    │
  │  │  │  (Encapsulation of data +     │   │    │
  │  │  │   operations)                 │   │    │
  │  │  └──────────────────────────────┘   │    │
  │  └──────────────────────────────────────┘    │
  └──────────────────────────────────────────────┘

  Key Point: Procedural and OOP are both forms of structuring
```

### 6.2 Design Principles for Functions (Procedures)

The following principles guide the design of good procedures (functions).

```
Five Principles of Function Design
═════════════════════════════════════════

  1. Single Responsibility Principle
     - One function does only one thing
     - You should be able to describe "what it does" in one sentence

  2. Appropriate Abstraction Level
     - Operations within a function should be at the same abstraction level
     - Do not mix high-level operations with low-level details

  3. Explicit Input/Output
     - Receive input through parameters, return results via return values
     - Minimize dependencies on global variables

  4. Appropriate Granularity
     - Not too long, not too short (guideline: within 20-30 lines)
     - Should fit on one screen

  5. Meaningful Naming
     - Verb + noun format is standard (e.g., calculate_tax)
     - Processing should be inferable from the name
```

### 6.3 Practical Example of Procedural Programming

The following shows a complete implementation example following procedural programming principles. We design a student grade management system step by step.

```python
"""
Student Grade Management System (Practical example of procedural programming)

Design Policy:
- Modularization through functions (each function has a single responsibility)
- Data managed as lists of dictionaries
- No global variables used
- Each function has clear input (parameters) and output (return values)
"""


# ===== Data Operation Functions =====

def create_student(name, student_id):
    """Create student data"""
    return {
        "name": name,
        "id": student_id,
        "scores": {}
    }


def add_score(student, subject, score):
    """Add a subject score to a student (returns a new dictionary)"""
    new_scores = dict(student["scores"])
    new_scores[subject] = score
    return {
        "name": student["name"],
        "id": student["id"],
        "scores": new_scores
    }


def calculate_average(student):
    """Calculate a student's average score"""
    scores = student["scores"]
    if not scores:
        return 0.0
    total = 0
    count = 0
    for score in scores.values():
        total += score
        count += 1
    return total / count


def determine_grade(average):
    """Determine grade from average score"""
    if average >= 90:
        return "A"
    elif average >= 80:
        return "B"
    elif average >= 70:
        return "C"
    elif average >= 60:
        return "D"
    else:
        return "F"


# ===== Aggregation Functions =====

def find_top_students(students, n=3):
    """Get the top n students"""
    averages = []
    for student in students:
        avg = calculate_average(student)
        averages.append((student, avg))

    # Sort by average in descending order (implemented with selection sort)
    for i in range(len(averages)):
        max_idx = i
        for j in range(i + 1, len(averages)):
            if averages[j][1] > averages[max_idx][1]:
                max_idx = j
        averages[i], averages[max_idx] = averages[max_idx], averages[i]

    result = []
    for i in range(min(n, len(averages))):
        result.append(averages[i])
    return result


def calculate_subject_stats(students, subject):
    """Calculate statistics (average, max, min) for a specific subject"""
    scores = []
    for student in students:
        if subject in student["scores"]:
            scores.append(student["scores"][subject])

    if not scores:
        return {"subject": subject, "count": 0, "avg": 0, "max": 0, "min": 0}

    total = 0
    max_score = scores[0]
    min_score = scores[0]
    for score in scores:
        total += score
        if score > max_score:
            max_score = score
        if score < min_score:
            min_score = score

    return {
        "subject": subject,
        "count": len(scores),
        "avg": total / len(scores),
        "max": max_score,
        "min": min_score,
    }


# ===== Display Functions =====

def format_student_report(student):
    """Format a student's grade report as a string"""
    avg = calculate_average(student)
    grade = determine_grade(avg)

    lines = []
    lines.append(f"Student: {student['name']} (ID: {student['id']})")
    lines.append("-" * 40)

    for subject, score in student["scores"].items():
        lines.append(f"  {subject:12s}: {score:5.1f}")

    lines.append("-" * 40)
    lines.append(f"  {'Average':12s}: {avg:5.1f}")
    lines.append(f"  {'Grade':12s}:     {grade}")

    return "\n".join(lines)


def format_ranking(top_students):
    """Format ranking as a string"""
    lines = []
    lines.append("=== Top Students ===")
    for rank, (student, avg) in enumerate(top_students, 1):
        grade = determine_grade(avg)
        lines.append(f"  #{rank}: {student['name']} "
                      f"(Average: {avg:.1f}, Grade: {grade})")
    return "\n".join(lines)


# ===== Main Processing =====

def main():
    """Main processing: Execute data generation, processing, and display in order"""

    # Prepare data
    students = [
        create_student("Taro Sato", "S001"),
        create_student("Hanako Suzuki", "S002"),
        create_student("Ichiro Tanaka", "S003"),
        create_student("Misaki Takahashi", "S004"),
        create_student("Kenta Ito", "S005"),
    ]

    # Set grade data
    score_data = [
        ("S001", [("Math", 85), ("English", 72), ("Physics", 90)]),
        ("S002", [("Math", 92), ("English", 88), ("Physics", 76)]),
        ("S003", [("Math", 68), ("English", 55), ("Physics", 73)]),
        ("S004", [("Math", 95), ("English", 91), ("Physics", 88)]),
        ("S005", [("Math", 78), ("English", 82), ("Physics", 65)]),
    ]

    # Register grades
    for sid, scores in score_data:
        for i in range(len(students)):
            if students[i]["id"] == sid:
                for subject, score in scores:
                    students[i] = add_score(students[i], subject, score)

    # Display individual reports
    print("=" * 40)
    print("      Individual Grade Reports")
    print("=" * 40)
    for student in students:
        print()
        print(format_student_report(student))

    # Display ranking
    print()
    top = find_top_students(students, 3)
    print(format_ranking(top))

    # Display subject statistics
    print()
    print("=== Subject Statistics ===")
    for subject in ["Math", "English", "Physics"]:
        stats = calculate_subject_stats(students, subject)
        print(f"  {stats['subject']}: "
              f"Average={stats['avg']:.1f}, "
              f"Max={stats['max']:.0f}, "
              f"Min={stats['min']:.0f}")


# Entry point
if __name__ == "__main__":
    main()
```

Execution result:

```
========================================
      Individual Grade Reports
========================================

Student: Taro Sato (ID: S001)
----------------------------------------
  Math        :  85.0
  English     :  72.0
  Physics     :  90.0
----------------------------------------
  Average     :  82.3
  Grade       :     B

Student: Hanako Suzuki (ID: S002)
----------------------------------------
  Math        :  92.0
  English     :  88.0
  Physics     :  76.0
----------------------------------------
  Average     :  85.3
  Grade       :     B

...(abbreviated)

=== Top Students ===
  #1: Misaki Takahashi (Average: 91.3, Grade: A)
  #2: Hanako Suzuki (Average: 85.3, Grade: B)
  #3: Taro Sato (Average: 82.3, Grade: B)

=== Subject Statistics ===
  Math: Average=83.6, Max=95, Min=68
  English: Average=77.6, Max=91, Min=55
  Physics: Average=78.4, Max=90, Min=65
```

### 6.4 Modularization and Information Hiding

In procedural programming, related functions and data are grouped into modules (files) to handle program scale. In C, the basic modularization technique involves publishing interfaces through header files (.h) and hiding implementations in source files (.c).

```c
/* === stack.h (Interface: Published declarations) === */
#ifndef STACK_H
#define STACK_H

#define STACK_MAX_SIZE 100

typedef struct {
    int data[STACK_MAX_SIZE];
    int top;
} Stack;

/* Public function declarations (users only need to know these) */
void stack_init(Stack *s);
int  stack_push(Stack *s, int value);
int  stack_pop(Stack *s, int *value);
int  stack_peek(const Stack *s, int *value);
int  stack_is_empty(const Stack *s);
int  stack_size(const Stack *s);

#endif /* STACK_H */
```

```c
/* === stack.c (Implementation: Internal details) === */
#include "stack.h"
#include <stdio.h>

void stack_init(Stack *s) {
    s->top = -1;
}

int stack_push(Stack *s, int value) {
    if (s->top >= STACK_MAX_SIZE - 1) {
        fprintf(stderr, "Stack overflow\n");
        return -1;
    }
    s->top++;
    s->data[s->top] = value;
    return 0;
}

int stack_pop(Stack *s, int *value) {
    if (s->top < 0) {
        fprintf(stderr, "Stack underflow\n");
        return -1;
    }
    *value = s->data[s->top];
    s->top--;
    return 0;
}

int stack_peek(const Stack *s, int *value) {
    if (s->top < 0) {
        return -1;
    }
    *value = s->data[s->top];
    return 0;
}

int stack_is_empty(const Stack *s) {
    return s->top < 0;
}

int stack_size(const Stack *s) {
    return s->top + 1;
}
```

```c
/* === main.c (Client code) === */
#include <stdio.h>
#include "stack.h"

int main(void) {
    Stack s;
    stack_init(&s);

    /* Push values */
    stack_push(&s, 10);
    stack_push(&s, 20);
    stack_push(&s, 30);

    printf("Stack size: %d\n", stack_size(&s));  /* 3 */

    /* Pop values */
    int value;
    while (!stack_is_empty(&s)) {
        stack_pop(&s, &value);
        printf("Popped: %d\n", value);
    }
    /* Popped: 30, Popped: 20, Popped: 10 */

    return 0;
}
```

### 6.5 Strengths and Weaknesses of Procedural Programming

| Aspect | Strengths | Weaknesses |
|------|------|------|
| Learning Cost | Intuitive and beginner-friendly | Design knowledge required at scale |
| Execution Efficiency | Close to hardware, fast | Optimization is the developer's responsibility |
| Code Structure | Clear modularization through functions | Data and operations tend to be separated |
| Reusability | Easy to reuse at the function level | Combinations of data structures and functions are rigid |
| Maintainability | High for small to medium scale | Difficult to maintain data consistency at large scale |
| Testing | Easy to test at the function level | Functions with side effects are hard to test |
| Concurrency | Natural expression of sequential processing | Concurrent access to shared state is problematic |

---

## 7. Comparison of Imperative with Other Paradigms

### 7.1 Imperative vs Functional Programming

Imperative programming and Functional Programming take fundamentally different approaches to computation.

```
Essential Difference Between Imperative and Functional
═══════════════════════════════════════════════

  Imperative: "A sequence of instructions that change state"
  ─────────────────────────────────────
  Program = State Machine

  State S0 → [Instr 1] → State S1 → [Instr 2] → State S2 → ...

  → Program progresses "along the passage of time"
  → The same instruction can produce different results depending on the current state

  Functional: "Transformation through function composition"
  ─────────────────────────────────────
  Program = Function Composition

  Input → f → g → h → Output

  → Program described as "data transformation"
  → Always the same output for the same input (referential transparency)
```

Let us compare with concrete code. We implement a process that squares even numbers in a list and sums them, in both imperative and functional styles.

```python
# === Imperative style ===

def sum_of_even_squares_imperative(numbers):
    """
    Imperative: Sequentially updates the state variable total
    """
    total = 0                         # Initialize state
    for num in numbers:               # Iteration
        if num % 2 == 0:              # Selection
            total += num ** 2         # Update state (assignment)
    return total


# === Functional style ===

def sum_of_even_squares_functional(numbers):
    """
    Functional: Transforms data through function composition
    No intermediate state
    """
    from functools import reduce
    return reduce(
        lambda acc, x: acc + x,       # Aggregation function
        map(
            lambda x: x ** 2,         # Transformation function
            filter(
                lambda x: x % 2 == 0, # Filter function
                numbers
            )
        ),
        0                             # Initial value
    )


# === Pythonic functional style (list comprehension) ===

def sum_of_even_squares_pythonic(numbers):
    """
    Pythonic declarative style
    """
    return sum(x ** 2 for x in numbers if x % 2 == 0)


# Execution example (all produce the same result)
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(sum_of_even_squares_imperative(data))    # 220
print(sum_of_even_squares_functional(data))    # 220
print(sum_of_even_squares_pythonic(data))      # 220

# 2^2 + 4^2 + 6^2 + 8^2 + 10^2 = 4 + 16 + 36 + 64 + 100 = 220
```

| Comparison Item | Imperative | Functional |
|---------|--------|--------|
| Central Concept | State change | Value transformation |
| Variables | Mutable | Immutable by default |
| Loops | for / while | Recursion / Higher-order functions (map, filter, reduce) |
| Side Effects | Common | Avoided (favors pure functions) |
| Concurrency | Requires synchronization via locks, etc. | Safe due to no shared state |
| Debugging | Trace via step execution | Verify input/output of each function |
| Computational Model | Turing Machine | Lambda Calculus |
| Representative Languages | C, Pascal, Go | Haskell, Erlang, Clojure |

### 7.2 Imperative vs Object-Oriented Programming

Object-Oriented Programming (OOP) is often positioned as an extension of imperative programming. OOP encapsulates data (state) and operations (methods) on that data together as objects.

**Procedural approach (data and operations separated):**

```python
# Procedural: Data as dictionaries, operations as independent functions

def create_bank_account(owner, balance=0):
    """Create account data"""
    return {"owner": owner, "balance": balance, "history": []}

def deposit(account, amount):
    """Make a deposit"""
    if amount <= 0:
        print("Deposit amount must be positive")
        return account
    new_balance = account["balance"] + amount
    new_history = list(account["history"])
    new_history.append(f"Deposit: +{amount}")
    return {
        "owner": account["owner"],
        "balance": new_balance,
        "history": new_history,
    }

def withdraw(account, amount):
    """Make a withdrawal"""
    if amount <= 0:
        print("Withdrawal amount must be positive")
        return account
    if amount > account["balance"]:
        print("Insufficient balance")
        return account
    new_balance = account["balance"] - amount
    new_history = list(account["history"])
    new_history.append(f"Withdrawal: -{amount}")
    return {
        "owner": account["owner"],
        "balance": new_balance,
        "history": new_history,
    }

def get_balance_info(account):
    """Get balance information"""
    return f"{account['owner']}'s balance: {account['balance']} yen"

# Usage example
acc = create_bank_account("Taro Sato", 10000)
acc = deposit(acc, 5000)
acc = withdraw(acc, 3000)
print(get_balance_info(acc))  # Taro Sato's balance: 12000 yen
```

**Object-oriented approach (data and operations integrated):**

```python
class BankAccount:
    """
    OOP: Encapsulates data (balance, history) and operations (deposit/withdraw)
    in a single class
    """

    def __init__(self, owner, balance=0):
        self._owner = owner         # Private attribute
        self._balance = balance
        self._history = []

    def deposit(self, amount):
        """Make a deposit"""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
        self._history.append(f"Deposit: +{amount}")

    def withdraw(self, amount):
        """Make a withdrawal"""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient balance")
        self._balance -= amount
        self._history.append(f"Withdrawal: -{amount}")

    @property
    def balance(self):
        return self._balance

    def __str__(self):
        return f"{self._owner}'s balance: {self._balance} yen"


# Usage example
acc = BankAccount("Taro Sato", 10000)
acc.deposit(5000)
acc.withdraw(3000)
print(acc)  # Taro Sato's balance: 12000 yen
```

| Comparison Item | Procedural | Object-Oriented |
|---------|---------|--------------|
| Data and Operations | Separated | Integrated (encapsulated) |
| State Management | Directly accessible from outside functions | Protected by access control |
| Reuse | Function reuse | Reuse through class inheritance/delegation |
| Polymorphism | Achieved through function pointers, etc. | Supported at the language level |
| Design Unit | Function | Object (class) |
| Suitable Scale | Small to medium | Medium to large |
| Complexity Management | Function hierarchies | Object relationship design |

### 7.3 Guidelines for Paradigm Selection

```
Decision Tree for Paradigm Selection
═══════════════════════════════════════════════════

  Analyze the nature of the problem
      │
      ├── Hardware control / embedded → Imperative (C)
      │
      ├── Scripting / automation → Procedural (Python, Shell)
      │
      ├── Data transformation / concurrency → Functional (Haskell, Elixir)
      │
      ├── Complex domain model → OOP (Java, C#)
      │
      ├── Data queries → Declarative (SQL)
      │
      ├── Logical reasoning / constraint satisfaction → Logic (Prolog)
      │
      └── Composite requirements → Multi-paradigm (Python, Rust, TS)

  In real-world projects:
  - Rather than insisting on a single paradigm,
    apply the optimal paradigm to each part of the problem
  - Most modern languages support multiple paradigms
```

---

## 8. Imperative Style in Modern Languages

### 8.1 Fusion of Imperative and Declarative in Python

Python is a representative multi-paradigm language that naturally combines imperative style with functional and declarative styles.

```python
"""
Example of paradigm fusion in modern Python:
Analyze log files and generate error statistics
"""
from collections import defaultdict
from datetime import datetime


# === Imperative style ===

def analyze_logs_imperative(log_lines):
    """Imperative: Explicit loops and state management"""
    error_counts = {}
    error_times = []

    for line in log_lines:
        # Parse each line
        parts = line.strip().split(" ", 3)
        if len(parts) < 4:
            continue

        timestamp_str = parts[0] + " " + parts[1]
        level = parts[2].strip("[]")
        message = parts[3]

        if level == "ERROR":
            # Update error count
            if message in error_counts:
                error_counts[message] += 1
            else:
                error_counts[message] = 1

            # Record timestamp
            try:
                ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                error_times.append(ts)
            except ValueError:
                pass

    # Find the most common error
    most_common_error = None
    max_count = 0
    for error, count in error_counts.items():
        if count > max_count:
            max_count = count
            most_common_error = error

    return {
        "total_errors": len(error_times),
        "unique_errors": len(error_counts),
        "most_common": most_common_error,
        "most_common_count": max_count,
        "error_counts": error_counts,
    }


# === Declarative/functional style ===

def analyze_logs_declarative(log_lines):
    """Declarative: Leveraging higher-order functions and generators"""

    def parse_line(line):
        """Parse a line into structured data"""
        parts = line.strip().split(" ", 3)
        if len(parts) < 4:
            return None
        return {
            "timestamp": parts[0] + " " + parts[1],
            "level": parts[2].strip("[]"),
            "message": parts[3],
        }

    # Data transformation pipeline
    parsed = (parse_line(line) for line in log_lines)           # Lazy evaluation
    valid = (entry for entry in parsed if entry is not None)    # Filter
    errors = [entry for entry in valid if entry["level"] == "ERROR"]

    # Aggregation (concise with defaultdict)
    error_counts = defaultdict(int)
    for error in errors:
        error_counts[error["message"]] += 1

    most_common = max(error_counts.items(), key=lambda x: x[1],
                      default=(None, 0))

    return {
        "total_errors": len(errors),
        "unique_errors": len(error_counts),
        "most_common": most_common[0],
        "most_common_count": most_common[1],
        "error_counts": dict(error_counts),
    }


# Test data
sample_logs = [
    "2024-01-15 10:23:45 [INFO] Application started",
    "2024-01-15 10:24:01 [ERROR] Database connection failed",
    "2024-01-15 10:24:15 [ERROR] Database connection failed",
    "2024-01-15 10:25:30 [WARNING] High memory usage",
    "2024-01-15 10:26:00 [ERROR] File not found: config.yml",
    "2024-01-15 10:27:12 [INFO] Retry successful",
    "2024-01-15 10:28:45 [ERROR] Database connection failed",
    "2024-01-15 10:30:00 [ERROR] Timeout waiting for response",
]

# Both approaches produce the same result
result1 = analyze_logs_imperative(sample_logs)
result2 = analyze_logs_declarative(sample_logs)

print(f"Total errors: {result1['total_errors']}")        # 5
print(f"Unique errors: {result1['unique_errors']}")       # 3
print(f"Most common error: {result1['most_common']}")     # Database connection failed
print(f"Most common count: {result1['most_common_count']}")  # 3
```

### 8.2 Fusion of Imperative and Functional in Rust

Rust is a language that fuses imperative control flow with functional iterators and pattern matching at a high level. Its ownership system guarantees "safety of mutable state" -- the biggest problem of imperative programming -- at compile time.

```rust
// Comparison of imperative and functional styles in Rust

/// Imperative style: Explicit loops and mutable variables
fn word_frequency_imperative(text: &str) -> Vec<(String, usize)> {
    use std::collections::HashMap;

    let mut counts: HashMap<String, usize> = HashMap::new();

    // Imperative loop
    for word in text.split_whitespace() {
        let word_lower = word.to_lowercase();
        // Safely update count with pattern matching
        let count = counts.entry(word_lower).or_insert(0);
        *count += 1;
    }

    // Convert result to vector and sort
    let mut result: Vec<(String, usize)> = counts.into_iter().collect();
    result.sort_by(|a, b| b.1.cmp(&a.1));
    result
}

/// Functional style: Iterator chains
fn word_frequency_functional(text: &str) -> Vec<(String, usize)> {
    use std::collections::HashMap;

    let counts: HashMap<String, usize> = text
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .fold(HashMap::new(), |mut acc, word| {
            *acc.entry(word).or_insert(0) += 1;
            acc
        });

    let mut result: Vec<_> = counts.into_iter().collect();
    result.sort_by(|a, b| b.1.cmp(&a.1));
    result
}

fn main() {
    let text = "the quick brown fox jumps over the lazy dog the fox";

    let result = word_frequency_imperative(text);
    for (word, count) in &result {
        println!("{}: {}", word, count);
    }
    // the: 3
    // fox: 2
    // quick: 1
    // brown: 1
    // ...
}
```

### 8.3 Thoroughgoing Imperative in Go

Go is a language that intentionally adopts a simple imperative style. Before generics (added in Go 1.18), it did not provide higher-order functions like map, filter, and reduce at the language level, adhering to a design philosophy centered on imperative loops.

```go
package main

import (
    "fmt"
    "math"
    "sort"
    "strings"
)

// Point represents a point on a two-dimensional plane
type Point struct {
    X, Y float64
}

// Distance calculates the Euclidean distance between two points
func Distance(p1, p2 Point) float64 {
    dx := p1.X - p2.X
    dy := p1.Y - p2.Y
    return math.Sqrt(dx*dx + dy*dy)
}

// FindClosestPair finds the closest pair of points in a set
// Imperative style: Explicit double loop
func FindClosestPair(points []Point) (Point, Point, float64) {
    if len(points) < 2 {
        return Point{}, Point{}, -1
    }

    minDist := math.Inf(1)
    var closest1, closest2 Point

    // Imperative double loop checking all combinations
    for i := 0; i < len(points); i++ {
        for j := i + 1; j < len(points); j++ {
            dist := Distance(points[i], points[j])
            if dist < minDist {
                minDist = dist
                closest1 = points[i]
                closest2 = points[j]
            }
        }
    }

    return closest1, closest2, minDist
}

// GroupByQuadrant classifies points by quadrant
func GroupByQuadrant(points []Point) map[string][]Point {
    groups := map[string][]Point{
        "Quadrant I":   {},
        "Quadrant II":  {},
        "Quadrant III": {},
        "Quadrant IV":  {},
        "On Axis":      {},
    }

    for _, p := range points {
        switch {
        case p.X > 0 && p.Y > 0:
            groups["Quadrant I"] = append(groups["Quadrant I"], p)
        case p.X < 0 && p.Y > 0:
            groups["Quadrant II"] = append(groups["Quadrant II"], p)
        case p.X < 0 && p.Y < 0:
            groups["Quadrant III"] = append(groups["Quadrant III"], p)
        case p.X > 0 && p.Y < 0:
            groups["Quadrant IV"] = append(groups["Quadrant IV"], p)
        default:
            groups["On Axis"] = append(groups["On Axis"], p)
        }
    }

    return groups
}

func main() {
    points := []Point{
        {1.0, 2.0}, {3.0, 4.0}, {1.5, 2.5},
        {-1.0, 3.0}, {-2.0, -1.0}, {4.0, -2.0},
        {0.0, 5.0},
    }

    // Find closest pair
    p1, p2, dist := FindClosestPair(points)
    fmt.Printf("Closest pair: (%.1f, %.1f) and (%.1f, %.1f)\n", p1.X, p1.Y, p2.X, p2.Y)
    fmt.Printf("Distance: %.4f\n", dist)

    // Classify by quadrant
    groups := GroupByQuadrant(points)
    for quadrant, pts := range groups {
        if len(pts) > 0 {
            strs := make([]string, len(pts))
            for i, p := range pts {
                strs[i] = fmt.Sprintf("(%.1f, %.1f)", p.X, p.Y)
            }
            fmt.Printf("%s: %s\n", quadrant, strings.Join(strs, ", "))
        }
    }

    // Sort (imperative: specify comparison function with sort.Slice)
    sort.Slice(points, func(i, j int) bool {
        d1 := Distance(points[i], Point{0, 0})
        d2 := Distance(points[j], Point{0, 0})
        return d1 < d2
    })

    fmt.Println("\nSorted by distance from origin:")
    for _, p := range points {
        d := Distance(p, Point{0, 0})
        fmt.Printf("  (%.1f, %.1f) -> distance %.4f\n", p.X, p.Y, d)
    }
}
```

### 8.4 Comparison of Imperative and Stream API in Java

Since Java 8, the introduction of the Stream API has made a declarative style available as an alternative to traditional imperative loops.

```java
import java.util.*;
import java.util.stream.*;

public class ImperativeVsStreams {

    record Employee(String name, String department, int salary) {}

    /**
     * Imperative style: Calculate average salary by department
     */
    static Map<String, Double> avgSalaryImperative(List<Employee> employees) {
        // Aggregate totals and counts by department
        Map<String, Integer> totalByDept = new HashMap<>();
        Map<String, Integer> countByDept = new HashMap<>();

        for (Employee emp : employees) {
            String dept = emp.department();
            // Update total
            if (totalByDept.containsKey(dept)) {
                totalByDept.put(dept, totalByDept.get(dept) + emp.salary());
            } else {
                totalByDept.put(dept, emp.salary());
            }
            // Update count
            if (countByDept.containsKey(dept)) {
                countByDept.put(dept, countByDept.get(dept) + 1);
            } else {
                countByDept.put(dept, 1);
            }
        }

        // Calculate averages
        Map<String, Double> result = new HashMap<>();
        for (String dept : totalByDept.keySet()) {
            double avg = (double) totalByDept.get(dept) / countByDept.get(dept);
            result.put(dept, avg);
        }
        return result;
    }

    /**
     * Stream API (declarative style): Same processing described as a pipeline
     */
    static Map<String, Double> avgSalaryStreams(List<Employee> employees) {
        return employees.stream()
            .collect(Collectors.groupingBy(
                Employee::department,
                Collectors.averagingInt(Employee::salary)
            ));
    }

    public static void main(String[] args) {
        List<Employee> employees = List.of(
            new Employee("Sato", "Engineering", 600000),
            new Employee("Suzuki", "Engineering", 550000),
            new Employee("Tanaka", "Sales", 500000),
            new Employee("Takahashi", "Sales", 480000),
            new Employee("Ito", "HR", 520000),
            new Employee("Watanabe", "Engineering", 700000),
            new Employee("Yamamoto", "Sales", 530000)
        );

        // Imperative
        Map<String, Double> result1 = avgSalaryImperative(employees);
        System.out.println("Imperative: " + result1);

        // Stream API
        Map<String, Double> result2 = avgSalaryStreams(employees);
        System.out.println("Stream: " + result2);

        // Both produce the same result:
        // {Engineering=616666.67, Sales=503333.33, HR=520000.0}
    }
}
```

---

## 9. Anti-Patterns

### 9.1 Anti-Pattern 1: Abuse of Global State

Programs that excessively depend on global variables lead to unpredictable behavior and maintenance difficulty.

```python
# ===== Anti-pattern: Abuse of global state =====

# Managing all state with global variables
current_user = None
cart_items = []
total_price = 0
discount_rate = 0
is_logged_in = False
error_message = ""

def login(username, password):
    global current_user, is_logged_in, error_message
    # Authentication processing (omitted)
    current_user = username
    is_logged_in = True
    error_message = ""

def add_to_cart(item, price):
    global cart_items, total_price, error_message
    if not is_logged_in:          # Depends on another global variable
        error_message = "Please log in"
        return
    cart_items.append({"item": item, "price": price})
    total_price += price          # Directly modifies a global variable

def apply_discount(rate):
    global discount_rate, total_price
    discount_rate = rate
    total_price = total_price * (1 - rate)  # No check whether already applied

def checkout():
    global cart_items, total_price, error_message
    if not is_logged_in:
        error_message = "Please log in"
        return
    if total_price <= 0:
        error_message = "Cart is empty"
        return
    # Order processing...
    print(f"{current_user}'s order: {total_price} yen")
    cart_items = []
    total_price = 0

# Problems:
# 1. Calling apply_discount() twice applies the discount twice
# 2. Difficult to track which function modifies which global variable
# 3. Global state must be initialized for testing
# 4. Concurrent execution causes state conflicts


# ===== Improved version: Localize state to an object =====

class ShoppingSession:
    """Confines state within a class and controls operations"""

    def __init__(self):
        self._user = None
        self._cart = []
        self._discount_applied = False

    def login(self, username, password):
        # Authentication processing
        self._user = username

    @property
    def is_logged_in(self):
        return self._user is not None

    def add_to_cart(self, item, price):
        if not self.is_logged_in:
            raise RuntimeError("Please log in")
        self._cart.append({"item": item, "price": price})

    def apply_discount(self, rate):
        if self._discount_applied:
            raise RuntimeError("Discount can only be applied once")
        self._discount_applied = True
        self._discount_rate = rate

    @property
    def total(self):
        subtotal = sum(item["price"] for item in self._cart)
        if self._discount_applied:
            subtotal *= (1 - self._discount_rate)
        return subtotal

    def checkout(self):
        if not self.is_logged_in:
            raise RuntimeError("Please log in")
        if not self._cart:
            raise RuntimeError("Cart is empty")
        order_total = self.total
        print(f"{self._user}'s order: {order_total} yen")
        self._cart = []
        self._discount_applied = False
        return order_total
```

### 9.2 Anti-Pattern 2: Spaghetti Code and Deep Nesting

When complex conditional branches are deeply nested, code readability and maintainability deteriorate significantly.

```python
# ===== Anti-pattern: Spaghetti code with deep nesting =====

def process_order_bad(order):
    """Bad example with deep nesting"""
    if order is not None:
        if order.get("status") == "pending":
            if order.get("items"):
                total = 0
                for item in order["items"]:
                    if item.get("price") is not None:
                        if item["price"] > 0:
                            if item.get("quantity", 0) > 0:
                                subtotal = item["price"] * item["quantity"]
                                if order.get("discount"):
                                    if order["discount"] > 0:
                                        if order["discount"] <= 0.5:
                                            subtotal *= (1 - order["discount"])
                                        else:
                                            print("Discount rate too large")
                                            return None
                                total += subtotal
                            else:
                                print("Invalid quantity")
                                return None
                        else:
                            print("Invalid price")
                            return None
                    else:
                        print("Price not set")
                        return None
                return total
            else:
                print("No items")
                return None
        else:
            print("Invalid status")
            return None
    else:
        print("No order")
        return None


# ===== Improved version: Flatten with guard clauses and helper functions =====

def validate_order(order):
    """Basic order validation"""
    if order is None:
        return False, "No order"
    if order.get("status") != "pending":
        return False, "Invalid status"
    if not order.get("items"):
        return False, "No items"
    return True, ""


def validate_item(item):
    """Validate an order item"""
    if item.get("price") is None:
        return False, "Price not set"
    if item["price"] <= 0:
        return False, "Invalid price"
    if item.get("quantity", 0) <= 0:
        return False, "Invalid quantity"
    return True, ""


def validate_discount(discount):
    """Validate discount rate"""
    if discount is None or discount <= 0:
        return 1.0  # No discount
    if discount > 0.5:
        raise ValueError("Discount rate too large")
    return 1.0 - discount


def calculate_item_subtotal(item, discount_rate):
    """Calculate the subtotal for an item"""
    return item["price"] * item["quantity"] * discount_rate


def process_order_good(order):
    """Improved: Flattened with guard clauses and helper functions"""
    # Guard clauses: Eliminate abnormal cases early
    is_valid, error = validate_order(order)
    if not is_valid:
        print(error)
        return None

    # Validate discount rate
    try:
        discount_multiplier = validate_discount(order.get("discount"))
    except ValueError as e:
        print(str(e))
        return None

    # Main processing: Calculate total
    total = 0
    for item in order["items"]:
        is_valid, error = validate_item(item)
        if not is_valid:
            print(error)
            return None
        total += calculate_item_subtotal(item, discount_multiplier)

    return total
```

### 9.3 Anti-Pattern 3: Magic Numbers and Hard-Coding

A pattern where meaningless numeric or string literals are scattered throughout the code.

```python
# ===== Anti-pattern: Magic numbers =====

def calculate_shipping_bad(weight, distance):
    """Bad example full of magic numbers"""
    if weight < 2.0:
        base = 500
    elif weight < 10.0:
        base = 1200
    elif weight < 30.0:
        base = 2500
    else:
        base = 5000

    if distance > 500:
        base *= 1.5
    if distance > 1000:
        base *= 1.2

    if base > 10000:
        base = 10000

    return int(base * 1.1)  # What is 1.1?


# ===== Improved version: Name the constants =====

# Weight category thresholds (kg)
WEIGHT_LIGHT = 2.0
WEIGHT_MEDIUM = 10.0
WEIGHT_HEAVY = 30.0

# Base rates per weight category (yen)
BASE_RATE_LIGHT = 500
BASE_RATE_MEDIUM = 1200
BASE_RATE_HEAVY = 2500
BASE_RATE_EXTRA_HEAVY = 5000

# Distance surcharge thresholds (km)
LONG_DISTANCE_THRESHOLD = 500
VERY_LONG_DISTANCE_THRESHOLD = 1000

# Distance surcharge multipliers
LONG_DISTANCE_MULTIPLIER = 1.5
VERY_LONG_DISTANCE_MULTIPLIER = 1.2

# Maximum shipping cost (yen)
MAX_SHIPPING_COST = 10000

# Tax rate
TAX_RATE = 0.1

def calculate_shipping_good(weight, distance):
    """Improved example with clear meaning through named constants"""
    # Determine base rate based on weight
    if weight < WEIGHT_LIGHT:
        base = BASE_RATE_LIGHT
    elif weight < WEIGHT_MEDIUM:
        base = BASE_RATE_MEDIUM
    elif weight < WEIGHT_HEAVY:
        base = BASE_RATE_HEAVY
    else:
        base = BASE_RATE_EXTRA_HEAVY

    # Distance surcharge
    if distance > VERY_LONG_DISTANCE_THRESHOLD:
        base *= LONG_DISTANCE_MULTIPLIER * VERY_LONG_DISTANCE_MULTIPLIER
    elif distance > LONG_DISTANCE_THRESHOLD:
        base *= LONG_DISTANCE_MULTIPLIER

    # Apply cap
    base = min(base, MAX_SHIPPING_COST)

    # Tax-inclusive price
    return int(base * (1 + TAX_RATE))
```

### 9.4 Anti-Pattern 4: Bloated Functions and Mixed Responsibilities

A pattern where a single function does far too many things and grows to hundreds of lines. This violates the single responsibility principle, making testing, maintenance, and reuse difficult.

```python
# ===== Anti-pattern: Do-everything function =====

def process_everything(filepath):
    """
    A single function handles file reading, validation,
    calculation, formatting, and file output.
    Testing and reuse are both difficult.
    """
    # File reading (30 lines)
    # Validation (40 lines)
    # Data transformation (50 lines)
    # Calculation processing (60 lines)
    # Result formatting (30 lines)
    # File output (20 lines)
    # Total: A giant function of 230+ lines
    pass


# ===== Improved version: Split by responsibility =====

def read_data(filepath):
    """Read data from a file"""
    pass

def validate_data(raw_data):
    """Validate data"""
    pass

def transform_data(validated_data):
    """Transform data into processing format"""
    pass

def calculate_results(transformed_data):
    """Perform calculations"""
    pass

def format_report(results):
    """Format results as a report"""
    pass

def write_report(report, output_path):
    """Write report to a file"""
    pass

def process_pipeline(input_path, output_path):
    """
    Pipeline: Call each function in order.
    Each step can be tested independently.
    """
    raw = read_data(input_path)
    validated = validate_data(raw)
    transformed = transform_data(validated)
    results = calculate_results(transformed)
    report = format_report(results)
    write_report(report, output_path)
```

---

## 10. Exercises

### 10.1 Basic Exercises

**Exercise 1: Practicing Basic Control Structures**

Implement a function in Python according to the following specification.

```
Specification: Extended FizzBuzz
=========================

Input: A positive integer n
Output: Return a list of strings for each number from 1 to n according to the following rules

  - Multiple of 3 → "Fizz"
  - Multiple of 5 → "Buzz"
  - Multiple of 7 → "Whizz"
  - Multiple of 3 and 5 → "FizzBuzz"
  - Multiple of 3 and 7 → "FizzWhizz"
  - Multiple of 5 and 7 → "BuzzWhizz"
  - Multiple of 3, 5, and 7 → "FizzBuzzWhizz"
  - Otherwise → String representation of the number

Control structures to use: for loop, if-elif-else
```

**Model Answer:**

```python
def fizzbuzz_extended(n):
    """
    Extended FizzBuzz: Checks multiples of 3, 5, and 7.
    Uses a builder pattern technique to construct the string.
    """
    result = []

    for i in range(1, n + 1):
        output = ""

        if i % 3 == 0:
            output += "Fizz"
        if i % 5 == 0:
            output += "Buzz"
        if i % 7 == 0:
            output += "Whizz"

        if output == "":
            output = str(i)

        result.append(output)

    return result


# Test
output = fizzbuzz_extended(105)
for i, val in enumerate(output, 1):
    if val != str(i):  # Show only items where conversion occurred
        print(f"{i:3d}: {val}")

# Example output:
#   3: Fizz
#   5: Buzz
#   6: Fizz
#   7: Whizz
#   9: Fizz
#  10: Buzz
#  12: Fizz
#  14: Whizz
#  15: FizzBuzz
#  21: FizzWhizz
#  35: BuzzWhizz
# 105: FizzBuzzWhizz
```

**Exercise 2: Array Operation Practice**

Implement a function in C according to the following specification.

```
Specification: Array Rotation
================

Input: Integer array arr, array length n, rotation count k
Process: Rotate the array right by k positions
  Example: [1,2,3,4,5] rotated by k=2 → [4,5,1,2,3]

Constraints:
  - Implement with O(1) additional memory without using an extra array
  - Hint: Apply the reverse operation 3 times
```

**Model Answer:**

```c
#include <stdio.h>

/*
 * Helper function to reverse a specified range of an array
 * Characteristics of imperative style: Index manipulation and value swapping
 */
void reverse(int arr[], int start, int end) {
    while (start < end) {
        int temp = arr[start];
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    }
}

/*
 * Rotate an array right by k positions
 * Algorithm:
 *   1. Reverse the entire array
 *   2. Reverse the first k elements
 *   3. Reverse the remaining n-k elements
 *
 * Example: [1,2,3,4,5], k=2
 *   Reverse entire: [5,4,3,2,1]
 *   Reverse first 2: [4,5,3,2,1]
 *   Reverse remaining 3: [4,5,1,2,3]
 */
void rotate_right(int arr[], int n, int k) {
    if (n <= 1) return;

    k = k % n;  /* Handle cases where k >= n */
    if (k == 0) return;

    reverse(arr, 0, n - 1);      /* Reverse entire array */
    reverse(arr, 0, k - 1);      /* Reverse first k elements */
    reverse(arr, k, n - 1);      /* Reverse the rest */
}

void print_array(int arr[], int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        printf("%d", arr[i]);
    }
    printf("]\n");
}

int main(void) {
    int arr[] = {1, 2, 3, 4, 5};
    int n = 5;

    printf("Before rotation: ");
    print_array(arr, n);

    rotate_right(arr, n, 2);

    printf("After rotation: ");
    print_array(arr, n);

    return 0;
}
/* Output:
 * Before rotation: [1, 2, 3, 4, 5]
 * After rotation: [4, 5, 1, 2, 3]
 */
```

### 10.2 Applied Exercises

**Exercise 3: Implementing a State Machine**

Implement a finite state machine that parses numbers from strings according to the following specification.

```
Specification: Simple Number Parser
=====================

Input: String (e.g., "  -123.456  ")
Output: Parse result (floating point number or integer)

State Transition Diagram:
  [Start] --(whitespace)--> [Start]
  [Start] --(+/-)--> [Sign]
  [Start] --(digit)--> [Integer Part]
  [Sign] --(digit)--> [Integer Part]
  [Integer Part] --(digit)--> [Integer Part]
  [Integer Part] --(.)--> [Decimal Point]
  [Integer Part] --(whitespace)--> [Trailing Whitespace]
  [Decimal Point] --(digit)--> [Decimal Part]
  [Decimal Part] --(digit)--> [Decimal Part]
  [Decimal Part] --(whitespace)--> [Trailing Whitespace]
  [Trailing Whitespace] --(whitespace)--> [Trailing Whitespace]
  Any other transition --> [Error]
```

**Model Answer:**

```python
def parse_number(text):
    """
    Number parser using a finite state machine (FSM).

    A typical application of imperative programming: State transitions
    are explicitly managed with variables and conditional branches.
    """
    # State definitions
    STATE_START = "start"
    STATE_SIGN = "sign"
    STATE_INTEGER = "integer"
    STATE_DOT = "dot"
    STATE_DECIMAL = "decimal"
    STATE_TRAILING = "trailing"
    STATE_ERROR = "error"

    state = STATE_START
    result_str = ""

    for ch in text:
        if state == STATE_START:
            if ch == ' ':
                pass  # Skip leading whitespace
            elif ch in ('+', '-'):
                result_str += ch
                state = STATE_SIGN
            elif ch.isdigit():
                result_str += ch
                state = STATE_INTEGER
            else:
                state = STATE_ERROR

        elif state == STATE_SIGN:
            if ch.isdigit():
                result_str += ch
                state = STATE_INTEGER
            else:
                state = STATE_ERROR

        elif state == STATE_INTEGER:
            if ch.isdigit():
                result_str += ch
            elif ch == '.':
                result_str += ch
                state = STATE_DOT
            elif ch == ' ':
                state = STATE_TRAILING
            else:
                state = STATE_ERROR

        elif state == STATE_DOT:
            if ch.isdigit():
                result_str += ch
                state = STATE_DECIMAL
            else:
                state = STATE_ERROR

        elif state == STATE_DECIMAL:
            if ch.isdigit():
                result_str += ch
            elif ch == ' ':
                state = STATE_TRAILING
            else:
                state = STATE_ERROR

        elif state == STATE_TRAILING:
            if ch == ' ':
                pass  # Skip trailing whitespace
            else:
                state = STATE_ERROR

        if state == STATE_ERROR:
            return None, f"Invalid character '{ch}' detected"

    # Validate final state
    if state in (STATE_INTEGER, STATE_DECIMAL, STATE_TRAILING):
        if '.' in result_str:
            return float(result_str), "Floating point number"
        else:
            return int(result_str), "Integer"
    elif state == STATE_START:
        return None, "Empty input"
    else:
        return None, f"Incomplete input (state: {state})"


# Test
test_cases = [
    "  42  ",
    "  -123.456  ",
    "+7.0",
    "  100  ",
    "  12.34.56  ",  # Error
    "  abc  ",       # Error
    "  ",            # Error
    "  +  ",         # Error
]

for text in test_cases:
    value, description = parse_number(text)
    print(f"'{text}' -> {value} ({description})")

# Output:
# '  42  ' -> 42 (Integer)
# '  -123.456  ' -> -123.456 (Floating point number)
# '+7.0' -> 7.0 (Floating point number)
# '  100  ' -> 100 (Integer)
# '  12.34.56  ' -> None (Invalid character '.' detected)
# '  abc  ' -> None (Invalid character 'a' detected)
# '  ' -> None (Empty input)
# '  +  ' -> None (Incomplete input (state: sign))
```

### 10.3 Advanced Exercises

**Exercise 4: Refactoring from Imperative to Functional**

Refactor the following imperative code into functional style (map, filter, reduce / list comprehensions). Behavior must be completely identical.

```python
# ===== Imperative code to refactor =====

def analyze_text_imperative(text):
    """Analyze text and return statistics (imperative)"""
    words = text.lower().split()

    # 1. Remove stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were",
                  "in", "on", "at", "to", "for", "of", "and",
                  "or", "but", "not", "with", "by"}
    filtered = []
    for word in words:
        clean = ""
        for ch in word:
            if ch.isalnum():
                clean += ch
        if clean and clean not in stop_words:
            filtered.append(clean)

    # 2. Group by word length
    groups = {}
    for word in filtered:
        length = len(word)
        if length not in groups:
            groups[length] = []
        groups[length].append(word)

    # 3. Summarize each group
    summary = {}
    for length, word_list in groups.items():
        unique = []
        seen = set()
        for w in word_list:
            if w not in seen:
                unique.append(w)
                seen.add(w)
        summary[length] = {
            "count": len(word_list),
            "unique": len(unique),
            "words": unique
        }

    return summary
```

**Model Answer (functional style):**

```python
from collections import Counter
from itertools import groupby

def analyze_text_functional(text):
    """Analyze text and return statistics (functional)"""
    stop_words = frozenset({
        "the", "a", "an", "is", "are", "was", "were",
        "in", "on", "at", "to", "for", "of", "and",
        "or", "but", "not", "with", "by"
    })

    # Pipeline: Split → Clean → Filter
    clean_word = lambda w: "".join(ch for ch in w if ch.isalnum())
    words = [
        cleaned for w in text.lower().split()
        if (cleaned := clean_word(w)) and cleaned not in stop_words
    ]

    # Generate grouping and summary at once
    sorted_words = sorted(words, key=len)
    summary = {
        length: {
            "count": len(group_words := list(group)),
            "unique": len(unique := list(dict.fromkeys(group_words))),
            "words": unique,
        }
        for length, group in groupby(sorted_words, key=len)
    }

    return summary


# Test: Confirm both functions return the same result
sample = """
The quick brown fox jumps over the lazy dog.
A fox is not a dog, but the fox and the dog are friends.
"""

result_imp = analyze_text_imperative(sample)
result_fun = analyze_text_functional(sample)

for length in sorted(result_imp.keys()):
    r = result_imp[length]
    print(f"  Length {length}: {r['count']} words ({r['unique']} unique) {r['words']}")
```

---

## 11. FAQ (Frequently Asked Questions)

### Q1. Is imperative programming an outdated paradigm that will be replaced by functional programming?

It is unlikely that imperative programming will be completely replaced by functional programming. The reasons are as follows.

**Reasons why imperative will remain important:**

1. **Correspondence with hardware**: Modern computers are based on the von Neumann architecture, and imperative programming directly corresponds to this architecture. Imperative programming is indispensable for low-level programming such as operating systems, device drivers, and embedded systems.

2. **Performance requirements**: In performance-critical scenarios, explicit control of memory layout and cache efficiency is necessary. This is the domain of imperative programming.

3. **Intuitiveness**: For beginners, the imperative way of thinking -- "write steps in order" -- is the most intuitive, making it suitable as an entry point for programming education.

**Realistic trends:**

Major modern languages (Python, JavaScript, Rust, Kotlin, Swift, etc.) are multi-paradigm, fusing imperative and functional elements. Rather than an either/or choice between "imperative or functional," the ability to select the appropriate style for the problem has become important.

### Q2. Should goto statements never be used?

Dijkstra's "Go To Considered Harmful" criticized the "indiscriminate use of goto statements," not advocating for a "total ban on goto." There are cases in modern programming where goto is used reasonably.

**Cases where goto is acceptable:**

1. **Cleanup for error handling (C)**: Since C has no exception handling mechanism, goto is used to centralize resource release processing. This usage is also recommended in the Linux kernel coding style guide.

```c
int process_file(const char *filename) {
    FILE *fp = NULL;
    char *buffer = NULL;
    int result = -1;

    fp = fopen(filename, "r");
    if (fp == NULL) goto cleanup;

    buffer = malloc(1024);
    if (buffer == NULL) goto cleanup;

    /* Processing ... */
    result = 0;

cleanup:
    free(buffer);       /* free is safe even with NULL */
    if (fp) fclose(fp);
    return result;
}
```

2. **Breaking out of nested loops**: In some languages, goto or labeled break is used to exit nested loops at once.

**Why many modern languages have eliminated goto:**

Many modern languages such as Python, Java, JavaScript, Ruby, and Go (limited goto) provide structured alternatives -- exception handling (try-catch), labeled break/continue, defer statements, etc. -- that greatly reduce the need for goto.

### Q3. How can concurrency be safely achieved in imperative programming?

The biggest challenge of concurrent processing in imperative programming is "shared mutable state." When multiple threads or processes simultaneously read and write the same variable, a data race (Race Condition) occurs.

**Main solutions:**

1. **Mutual exclusion (Mutex / Lock)**: Serialize access to shared resources.
2. **Message passing**: Instead of shared state, exchange data through message sending/receiving (Go's goroutine + channel).
3. **Use of immutable data**: Adopt the functional programming approach and make data immutable.
4. **Ownership system**: Like Rust, detect data races at compile time.

```python
import threading

# === Problematic code: Data race ===

counter = 0

def increment_unsafe():
    global counter
    for _ in range(100000):
        counter += 1  # This operation is not atomic
        # It consists of 3 steps: read → add → write,
        # and another thread may interrupt midway

# === Solution: Mutual exclusion via lock ===

counter_safe = 0
lock = threading.Lock()

def increment_safe():
    global counter_safe
    for _ in range(100000):
        with lock:                    # Acquire lock
            counter_safe += 1         # Other threads wait during this
                                      # Lock is automatically released

# Test
threads = [threading.Thread(target=increment_safe) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Expected: 400000, Actual: {counter_safe}")
# With lock: Always 400000
```

### Q4. What constitutes "good code" in imperative programming?

The criteria for "good code" in imperative programming are as follows:

1. **Readability**: Readers can understand the intent. Variable and function names are appropriate, and the flow of processing can be followed.

2. **Predictability**: The behavior of a function can be predicted from its inputs and name. Side effects are minimized.

3. **Testability**: Functions are independent, and unit tests are easy to write. External dependencies are injectable.

4. **Changeability**: The impact scope for requirement changes is limited. Coupling between modules is loose.

5. **Simplicity**: Avoid unnecessarily complex structures. Balance "working code" and "elegant code."

### Q5. What is the optimal order for learning imperative programming?

The following order is recommended.

```
Learning Roadmap
═══════════════════════════════════════

  Phase 1: Foundations (1-2 months)
  ├── Variables and assignment
  ├── Basic data types (integer, float, string, boolean)
  ├── Three control structures: sequence, selection, iteration
  ├── Function definition and calling
  └── Recommended language: Python

  Phase 2: Data Structures and Intermediate Control (2-3 months)
  ├── Arrays, lists, dictionaries
  ├── Nested control structures
  ├── Recursion
  ├── File I/O
  ├── Error handling
  └── Recommended languages: Python + intro to C

  Phase 3: Design Principles (2-3 months)
  ├── Function design (single responsibility, parameters and return values)
  ├── Modularization
  ├── State management best practices
  ├── Basics of test-driven development
  └── Recommended language: Python

  Phase 4: Fusion with Other Paradigms (3+ months)
  ├── Object-oriented programming
  ├── Elements of functional programming
  ├── Concurrent programming
  ├── Low-level programming (C/Rust)
  └── Recommended: Study a multi-paradigm language in depth
```

### Q6. What is the most important debugging technique for imperative code?

The most effective debugging technique for imperative code is "State Tracking." Since imperative programs are chains of state transitions, grasping the state (set of variable values) at each point is the key to finding problems.

**Specific techniques:**

1. **Step Execution**: Use a debugger to execute one statement at a time and observe variable changes.
2. **Watchpoints**: Halt the program when a specific variable is modified.
3. **Assertions**: Embed `assert` statements in the program to detect precondition violations early.
4. **Logging**: Record state transition history in logs for post-mortem analysis.
5. **Invariant Verification**: Explicitly state and verify conditions that should hold at each iteration of a loop (loop invariants).

```python
def binary_search_debug(sorted_list, target):
    """Binary search with embedded debugging techniques"""
    low = 0
    high = len(sorted_list) - 1
    iteration = 0

    while low <= high:
        iteration += 1

        # Verify loop invariant
        assert 0 <= low <= high < len(sorted_list), \
            f"Invariant violation: low={low}, high={high}, len={len(sorted_list)}"

        mid = (low + high) // 2

        # Log state output
        print(f"  Iteration {iteration}: low={low}, mid={mid}, high={high}, "
              f"arr[mid]={sorted_list[mid]}, target={target}")

        if sorted_list[mid] == target:
            print(f"  → Found! index={mid}")
            return mid
        elif sorted_list[mid] < target:
            low = mid + 1
            print(f"  → Search right half (low={low})")
        else:
            high = mid - 1
            print(f"  → Search left half (high={high})")

    print(f"  → Not found ({iteration} iterations)")
    return -1
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying its behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge used in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes especially important during code reviews and architecture design.

---

## 12. Summary and References

### 12.1 Summary

The following table organizes the important concepts of imperative programming covered in this chapter.

| Concept | Key Points |
|------|---------|
| Essence of Imperative | Instructs "how" to do things step by step. State, assignment, and control flow are the three pillars |
| Von Neumann Correspondence | Variables=memory, assignment=store instruction, control flow=jump instruction. Directly corresponds to hardware |
| Historical Background | Machine code→Assembly→FORTRAN→ALGOL→C. Abstraction level improved at each stage |
| Control Structures | Any algorithm can be expressed with only three structures: sequence, selection, and iteration (Structured Program Theorem) |
| State Management | Mutable state is the foundation of imperative programming and simultaneously a major source of complexity |
| Side Effects | Global variable modification, I/O operations, etc. Become a breeding ground for bugs if unmanaged |
| Structured Programming | Goto elimination by Dijkstra. Dramatically improved code readability and maintainability |
| Procedural | Modularization through functions. Principles: single responsibility, explicit I/O, appropriate granularity |
| Paradigm Comparison | Imperative focuses on state change, functional on value transformation, OOP on integrating data and operations |
| Modern Trends | Multi-paradigm. Fusion of imperative and functional is mainstream |
| Anti-Patterns | Global state abuse, deep nesting, magic numbers, bloated functions |

```
Position of Imperative Programming (Final Summary)
═══════════════════════════════════════════════════

  Overview of Programming Paradigms:

  ┌───────────────────────────────────────────┐
  │         Programming Paradigms              │
  │                                           │
  │  ┌─────────────┐   ┌─────────────┐      │
  │  │  Imperative  │   │  Declarative │      │
  │  │  (How)       │   │  (What)      │      │
  │  │              │   │              │      │
  │  │ - Procedural │   │ - Functional │      │
  │  │ - OOP        │   │ - Logic      │      │
  │  │ - Structured │   │ - Dataflow   │      │
  │  └─────────────┘   └─────────────┘      │
  │                                           │
  │  Modern: Multi-paradigm                    │
  │  Python, Rust, Kotlin, TypeScript ...      │
  │  → Freely combine imperative and           │
  │    declarative elements                    │
  └───────────────────────────────────────────┘
```

### 12.2 References

1. Dijkstra, E. W. "Go To Statement Considered Harmful." *Communications of the ACM*, Vol. 11, No. 3, pp. 147-148, March 1968.
   - The historic paper that launched structured programming. Pointed out that indiscriminate use of goto statements degrades program quality.

2. Bohm, C. & Jacopini, G. "Flow Diagrams, Turing Machines and Languages with Only Two Formation Rules." *Communications of the ACM*, Vol. 9, No. 5, pp. 366-371, May 1966.
   - The paper that mathematically proved the Structured Program Theorem (that any program can be expressed with sequence, selection, and iteration).

3. Wirth, N. "Program Development by Stepwise Refinement." *Communications of the ACM*, Vol. 14, No. 4, pp. 221-227, April 1971.
   - The paper that systematized the concept of Stepwise Refinement. Became the foundation for top-down design.

4. Kernighan, B. W. & Ritchie, D. M. *The C Programming Language*, 2nd Edition. Prentice Hall, 1988.
   - The standard textbook for C, demonstrating exemplary imperative and procedural programming style. Known as K&R.

5. Knuth, D. E. "Structured Programming with go to Statements." *Computing Surveys*, Vol. 6, No. 4, pp. 261-301, December 1974.
   - Provided a balanced analysis of the goto-considered-harmful argument and discussed reasonable use cases for goto statements.

6. Abelson, H. & Sussman, G. J. *Structure and Interpretation of Computer Programs*, 2nd Edition. MIT Press, 1996.
   - A renowned MIT textbook that deeply explores fundamental programming concepts including both imperative and functional. Known as SICP.

7. Van Roy, P. & Haridi, S. *Concepts, Techniques, and Models of Computer Programming*. MIT Press, 2004.
   - A comprehensive textbook that covers multiple programming paradigms, including imperative, within a unified framework.

---

## Recommended Next Reads


---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
