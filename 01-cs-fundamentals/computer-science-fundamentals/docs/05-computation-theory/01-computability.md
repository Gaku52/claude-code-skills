# Computability

> "Not every problem can be solved by a computer" -- this fact is one of the greatest discoveries in CS, and the undecidability of the halting problem is its hallmark.

## Learning Objectives

- [ ] Explain the concept of Turing machines and their various variants
- [ ] Understand and reproduce the proof that the halting problem is undecidable
- [ ] Understand the boundary between decidable and undecidable problems
- [ ] Prove undecidability using the concept of reduction
- [ ] Understand the meaning and limitations of the Church-Turing thesis
- [ ] Explain the practical impact of computability theory
- [ ] Understand the recursion theorem and Rice's theorem
- [ ] Grasp the historical background and motivation of computability theory


## Prerequisites

Having the following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Automata and Formal Languages](./00-automata.md)

---

## 1. Historical Background and Motivation

### 1.1 Hilbert's Program and the Decision Problem

```
Hilbert's Program (1900s-1930s):

  David Hilbert's three questions:
  1. Completeness: Can all true mathematical propositions be proven?
  2. Consistency: Is mathematics free of contradictions?
  3. Decidability: Can the truth of any mathematical proposition be mechanically determined? (Entscheidungsproblem)

  Godel's Incompleteness Theorems (1931):
  - First Incompleteness Theorem: In any sufficiently strong consistent formal system,
    there exist propositions that can be neither proved nor disproved
  - Second Incompleteness Theorem: A sufficiently strong consistent formal system
    cannot prove its own consistency

  Turing's Answer (1936):
  - A negative answer to the decision problem
  - Invented the Turing machine to rigorously define the concept of "computation"
  - Proved the undecidability of the halting problem

  Contemporary Contributions:
  - Alonzo Church: Lambda calculus (1936)
  - Emil Post: Post production system (1936)
  - Stephen Kleene: Recursive functions (1936)
  -> All were later proven to have equivalent computational power
```

### 1.2 What is "Computation"?

```
Intuitive Understanding of "Computable":

  Everyday "computation":
  - Addition, multiplication -> Clearly computable
  - Sorting -> Clearly computable
  - Primality testing -> Computable

  Subtle "computation":
  - "Is this mathematical proposition true?" -> Generally not computable
  - "Does this program halt?" -> Not computable
  - "Is this cipher secure?" -> Generally not computable

  Purpose of computability theory:
  -> Identify the boundary between "problems solvable in principle" and "problems unsolvable in principle"

  Practical implications:
  - Avoid wasting effort trying to solve unsolvable problems
  - Guide approaches toward approximations or restricted solutions
  - Understand the limits of software verification
```

---

## 2. Turing Machines

### 2.1 Formal Definition

```
Turing Machine: A theoretical model of computation

  Components:
  +----------------------------------------------+
  | ... | B | 1 | 0 | 1 | 1 | B | B | ...      | <- Infinite tape
  +----------------------------------------------+
                    ^
                 Head (read/write)
                 +-------+
                 | State q| <- Finite control
                 +-------+

  Formal definition M = (Q, Sigma, Gamma, delta, q0, q_accept, q_reject):

  Q: Finite set of states
  Sigma: Input alphabet (does not include blank symbol B)
  Gamma: Tape alphabet (Sigma subset of Gamma, B in Gamma)
  delta: Q x Gamma -> Q x Gamma x {L, R}  Transition function
  q0: Start state (q0 in Q)
  q_accept: Accept state (q_accept in Q)
  q_reject: Reject state (q_reject in Q, q_accept != q_reject)

  Operation: (current state, symbol read) -> (symbol to write, head movement, next state)

  Computation process:
  1. The input string is written on the tape
  2. The head is positioned over the leftmost symbol
  3. Operations are repeated according to the transition function
  4. Reaching the accept state -> "accept" the input
  5. Reaching the reject state -> "reject" the input
  6. May never halt -> "loop"
```

### 2.2 Concrete Example: Palindrome Detection Turing Machine

```
Problem: Determine whether an input string w in {0, 1}* is a palindrome

Algorithm:
1. Read the first character, remember it, and overwrite with X
2. Move to the end of the tape
3. Check if the last character matches the remembered character
4. If it matches, overwrite with X and return to the beginning of the tape
5. If all characters become X, accept

States:
  q0: Start state
  q1: First character was 0, moving to the right end
  q2: First character was 1, moving to the right end
  q3: After confirming 0 at the right end, moving to the left end
  q4: After confirming 1 at the right end, moving to the left end
  q5: Returning to the left end
  q_accept: Accept
  q_reject: Reject

Partial transition table:
  delta(q0, 0) = (q1, X, R)    -- First is 0, remember and overwrite with X
  delta(q0, 1) = (q2, X, R)    -- First is 1, remember and overwrite with X
  delta(q0, X) = (q0, X, R)    -- Skip X
  delta(q0, B) = (q_accept, B, R)  -- All became X, accept

  delta(q1, 0) = (q1, 0, R)    -- Moving to the right end
  delta(q1, 1) = (q1, 1, R)
  delta(q1, X) = (q1, X, R)
  delta(q1, B) = (q3, B, L)    -- Reached the right end, proceed to check

  delta(q3, X) = (q3, X, L)    -- Skip X
  delta(q3, 0) = (q5, X, L)    -- 0 matches, overwrite with X
  delta(q3, 1) = (q_reject, 1, R)  -- Mismatch, reject

Execution example: Input "0110"
  Step 1: [0]110B -> X[1]10B  (q0 -> q1, remember 0)
  Step 2: X[1]10B -> X1[1]0B -> X11[0]B -> X110[B]
  Step 3: X110[B] -> X11[0]B  (q1 -> q3, reached right end)
  Step 4: X11[0]B -> X1[1]XB  (0 matches, overwrite with X)
  Step 5: ... (return to left end, repeat)
  Eventually accepted
```

### 2.3 Turing Machine Simulation (Python)

```python
class TuringMachine:
    """Turing machine simulator"""

    def __init__(self, states, input_alphabet, tape_alphabet,
                 transition_function, start_state,
                 accept_state, reject_state):
        self.states = states
        self.input_alphabet = input_alphabet
        self.tape_alphabet = tape_alphabet
        self.transition = transition_function
        self.start_state = start_state
        self.accept_state = accept_state
        self.reject_state = reject_state

    def run(self, input_string, max_steps=10000):
        """Run the TM on an input string"""
        # Initialize the tape
        tape = list(input_string) + ['B']
        head = 0
        state = self.start_state
        steps = 0

        history = []

        while steps < max_steps:
            # Record the current configuration
            history.append({
                'step': steps,
                'state': state,
                'head': head,
                'tape': ''.join(tape)
            })

            # Check for accept/reject
            if state == self.accept_state:
                return 'accept', history
            if state == self.reject_state:
                return 'reject', history

            # Extend the tape if necessary
            if head < 0:
                tape.insert(0, 'B')
                head = 0
            if head >= len(tape):
                tape.append('B')

            # Execute the transition
            current_symbol = tape[head]
            key = (state, current_symbol)

            if key not in self.transition:
                return 'reject', history

            new_state, write_symbol, direction = self.transition[key]
            tape[head] = write_symbol
            state = new_state
            head += 1 if direction == 'R' else -1
            steps += 1

        return 'loop', history  # Exceeded max_steps


# Concrete example: Turing machine that increments a binary number
def create_binary_increment_tm():
    """TM that performs binary increment"""
    states = {'q0', 'q1', 'q2', 'q_accept'}
    input_alphabet = {'0', '1'}
    tape_alphabet = {'0', '1', 'B'}

    # q0: Move to the right end
    # q1: From right end leftward, carry processing
    # q2: Carry complete, return to left end
    transition = {
        # Move to the right end
        ('q0', '0'): ('q0', '0', 'R'),
        ('q0', '1'): ('q0', '1', 'R'),
        ('q0', 'B'): ('q1', 'B', 'L'),

        # Increment processing (right to left)
        ('q1', '1'): ('q1', '0', 'L'),  # Carry
        ('q1', '0'): ('q2', '1', 'L'),  # Carry stops
        ('q1', 'B'): ('q_accept', '1', 'R'),  # Carry to the front

        # Complete, return to left end
        ('q2', '0'): ('q2', '0', 'L'),
        ('q2', '1'): ('q2', '1', 'L'),
        ('q2', 'B'): ('q_accept', 'B', 'R'),
    }

    return TuringMachine(
        states, input_alphabet, tape_alphabet,
        transition, 'q0', 'q_accept', 'q_reject'
    )


# Execution example
tm = create_binary_increment_tm()

test_cases = ['0', '1', '10', '11', '101', '111', '1111']
for tc in test_cases:
    result, history = tm.run(tc)
    final_tape = history[-1]['tape'].rstrip('B')
    print(f"  {tc} -> {final_tape} ({result})")

# Output:
#   0 -> 1 (accept)
#   1 -> 10 (accept)
#   10 -> 11 (accept)
#   11 -> 100 (accept)
#   101 -> 110 (accept)
#   111 -> 1000 (accept)
#   1111 -> 10000 (accept)
```

### 2.4 Variants of Turing Machines

```
There are many variants of Turing machines, but all have equivalent computational power:

  1. Multi-tape Turing Machine
     +-------------------+  Tape 1 (input)
     +-------------------+
              ^
     +-------------------+  Tape 2 (working)
     +-------------------+
              ^
     +-------------------+  Tape 3 (output)
     +-------------------+
              ^
           +-------+
           | State q|
           +-------+
     -> Can simulate multi-tape with a single tape (O(t^2) steps)
     -> Equivalent to using multiple variables in programming

  2. Nondeterministic Turing Machine (NTM)
     delta: Q x Gamma -> P(Q x Gamma x {L, R})
     -> Transition targets are sets (multiple choices)
     -> Accepts if a "correct choice" exists
     -> Can be simulated by a deterministic TM (exponential time increase)
     -> Closely related to NP problems

  3. Enumerator
     -> Outputs all strings in a language
     -> Equivalent to Turing-recognizable languages

  4. Universal Turing Machine (UTM)
     -> Takes the description of any TM and its input, simulates that TM
     -> Conceptual prototype of modern computers
     -> Theoretical foundation of the stored-program concept

  5. Restricted Turing Machines
     - Linear Bounded Automaton (LBA): Tape length proportional to input length
       -> Recognizes context-sensitive languages
     - Pushdown Automaton (PDA): Can only use a stack
       -> Recognizes context-free languages
     - Finite Automaton (FA): No internal memory
       -> Recognizes regular languages
```

### 2.5 Details of the Universal Turing Machine

```python
class UniversalTuringMachine:
    """Conceptual implementation of a Universal Turing Machine"""

    def __init__(self):
        pass

    def encode_tm(self, tm):
        """Encode a Turing machine (Godel numbering)"""
        # Serialize states, alphabet, and transition function
        encoding = {
            'states': list(tm.states),
            'input_alphabet': list(tm.input_alphabet),
            'tape_alphabet': list(tm.tape_alphabet),
            'transitions': {},
            'start_state': tm.start_state,
            'accept_state': tm.accept_state,
            'reject_state': tm.reject_state
        }

        for (state, symbol), (new_state, write, direction) in tm.transition.items():
            key = f"{state},{symbol}"
            encoding['transitions'][key] = {
                'new_state': new_state,
                'write': write,
                'direction': direction
            }

        return encoding

    def simulate(self, encoded_tm, input_string, max_steps=10000):
        """Simulate an encoded TM on an input string"""
        # Decode
        transitions = {}
        for key, value in encoded_tm['transitions'].items():
            state, symbol = key.split(',')
            transitions[(state, symbol)] = (
                value['new_state'],
                value['write'],
                value['direction']
            )

        # Simulation
        tape = list(input_string) + ['B']
        head = 0
        state = encoded_tm['start_state']

        for step in range(max_steps):
            if state == encoded_tm['accept_state']:
                return 'accept'
            if state == encoded_tm['reject_state']:
                return 'reject'

            if head < 0:
                tape.insert(0, 'B')
                head = 0
            if head >= len(tape):
                tape.append('B')

            key = (state, tape[head])
            if key not in transitions:
                return 'reject'

            new_state, write, direction = transitions[key]
            tape[head] = write
            state = new_state
            head += 1 if direction == 'R' else -1

        return 'timeout'  # Step count exceeded (may not halt)


# Significance of the Universal TM:
# 1. A single machine can execute all programs
# 2. Theoretical foundation of modern computers (von Neumann architecture)
# 3. Prototype of the concept that "programs can be treated as data"
# 4. Theoretical justification for compilers and interpreters
```

### 2.6 Turing Completeness

```
Turing Completeness: A computational system having the same computational power as a Turing machine

  Examples of Turing-complete systems:
  +---------------------------------------------+
  | Programming Languages                        |
  |  - Python, Java, C, C++, Rust, Go            |
  |  - JavaScript, TypeScript, Ruby, PHP         |
  |  - Haskell, OCaml, Lisp, Scheme              |
  |  - Assembly language                          |
  +---------------------------------------------+

  +---------------------------------------------+
  | Surprising Turing-complete systems            |
  |  - CSS (combination of animations + conditionals) |
  |  - SQL (using recursive CTEs)                 |
  |  - Excel (formulas only)                      |
  |  - PowerPoint (animations)                    |
  |  - Minecraft (redstone circuits)              |
  |  - Game of Life (Conway's Game of Life)       |
  |  - LaTeX (macro system)                       |
  |  - sed (stream editor)                        |
  |  - sendmail.cf (configuration file)           |
  |  - TypeScript's type system                   |
  +---------------------------------------------+

  Minimum requirements for Turing completeness:
  1. Conditional branching (if-then-else)
  2. Infinite memory (ability to store arbitrary amounts of data)
  3. Read/write state
  4. Repetition (loops or recursion)

  Examples of non-Turing-complete systems:
  - Regular expressions (standard definition)
  - Finite automata
  - Total recursive functions
  - LOOP language (primitive recursive functions only)
```

```python
# Minimal example of Turing completeness: BF interpreter
# BF is Turing-complete with only 8 instructions

def bf_interpreter(code, input_data=""):
    """Brainf*ck interpreter -- a minimal Turing-complete language"""
    tape = [0] * 30000
    ptr = 0
    pc = 0
    input_pos = 0
    output = []

    # Build bracket matching table
    brackets = {}
    stack = []
    for i, c in enumerate(code):
        if c == '[':
            stack.append(i)
        elif c == ']':
            if stack:
                j = stack.pop()
                brackets[j] = i
                brackets[i] = j

    max_steps = 1000000
    steps = 0

    while pc < len(code) and steps < max_steps:
        cmd = code[pc]
        if cmd == '>':
            ptr = (ptr + 1) % 30000
        elif cmd == '<':
            ptr = (ptr - 1) % 30000
        elif cmd == '+':
            tape[ptr] = (tape[ptr] + 1) % 256
        elif cmd == '-':
            tape[ptr] = (tape[ptr] - 1) % 256
        elif cmd == '.':
            output.append(chr(tape[ptr]))
        elif cmd == ',':
            if input_pos < len(input_data):
                tape[ptr] = ord(input_data[input_pos])
                input_pos += 1
            else:
                tape[ptr] = 0
        elif cmd == '[':
            if tape[ptr] == 0:
                pc = brackets[pc]
        elif cmd == ']':
            if tape[ptr] != 0:
                pc = brackets[pc]

        pc += 1
        steps += 1

    return ''.join(output)


# Hello World in BF
hello_bf = (
    "++++++++[>++++[>++>+++>+++>+<<<<-]"
    ">+>+>->>+[<]<-]>>.>---.+++++++..+++."
    ">>.------------.<<+++++++++++++++.>."
    "+++.------.--------.>>+.>++."
)
print(bf_interpreter(hello_bf))  # "Hello World!\n"

# Why BF is Turing-complete:
# - Infinite (sufficiently large) tape -> Arbitrary memory
# - Loop structure ([...]) -> Repetition
# - Conditional branching ([] for zero check) -> Conditionals
# - Data read/write (+, -, >, <) -> State manipulation
```

---

## 3. The Church-Turing Thesis

### 3.1 Content of the Thesis

```
Church-Turing Thesis:

  Definition:
  "The intuitively computable functions are exactly those functions
   computable by a Turing machine"

  Note: This is a "thesis" (hypothesis), not a mathematical theorem
  -> It can be refuted, but no counterexample has been found so far

  Models proven to be equivalent:
  +---------------------------------------------+
  |                                             |
  |  Turing Machine <-> Lambda Calculus         |
  |       |                  |                  |
  |  Recursive Functions <-> Post Production System |
  |       |                  |                  |
  |  Register Machine <-> Markov Algorithm      |
  |                                             |
  +---------------------------------------------+

  All have the same computational power -> "Robustness" of computability
```

### 3.2 Physical Church-Turing Thesis

```
Physical Church-Turing Thesis:

  "All physically realizable computing devices can be
   simulated by a Turing machine"

  Arguments for and against:

  Arguments in favor:
  - Quantum computers can also be simulated by Turing machines
    (inefficiently, but with equivalent computational power)
  - No physical device serving as a counterexample has been found so far

  Arguments against / skepticism:
  - Possibility of hypercomputation
    -> Zeno machines (executing infinite steps in finite time)
    -> Black hole computers
    -> Infinite precision of analog computation
  - However, these are considered physically unrealizable

  Practical meaning:
  - The choice of programming language does not affect "computational power"
  - All Turing-complete programming languages are equivalent
  - The only differences between languages are "expressiveness," "efficiency," and "safety"
```

### 3.3 Strong Church-Turing Thesis

```
Strong Church-Turing Thesis (Extended Church-Turing Thesis):

  "The efficiently computable functions are exactly those functions
   computable in polynomial time by a probabilistic Turing machine"

  Challenge from quantum computers:
  - Shor's algorithm: Solves integer factorization in polynomial time
  - Whether this is classically solvable in polynomial time is unknown
  -> The strong thesis may be broken by quantum computers

  However:
  - Quantum computers are also not expected to solve NP-complete problems efficiently
  - BQP subset of PSPACE has been proven
  -> Even with quantum, the boundary of "computability" does not change
```

---

## 4. The Halting Problem

### 4.1 Formal Definition of the Halting Problem

```
Halting Problem:

  Input: Description <M> of a Turing machine M and input string w
  Output: Does M halt on w?

  Formally:
  HALT = { <M, w> | M halts on input w }

  Theorem: HALT is undecidable

  Intuitive understanding:
  - A program that always correctly answers "Does this program halt?"
    does not exist
  - For individual programs, it may be possible to determine
  - But there is no universal method to determine for "all" programs
```

### 4.2 Proof of Undecidability (Detailed Version)

```
Proof of the undecidability of the halting problem (by diagonalization):

  Theorem: HALT = { <M, w> | M halts on input w } is undecidable

  Proof:
  Assumption: Suppose a halting decision program H exists
    H(<M>, w) = accept  (if M halts on w)
    H(<M>, w) = reject  (if M does not halt on w)

  Constructing the contradiction:
  Define program D as follows:

    D(<M>):
      Run H(<M>, <M>)
      if H returns accept:
        while True: pass    # Infinite loop
      else:
        return              # Halt

  Give D its own description <D> as input:

  Case 1: Assume D(<D>) halts
    -> H(<D>, <D>) = accept
    -> By D's definition, D enters an infinite loop
    -> D(<D>) does not halt
    -> Contradiction!

  Case 2: Assume D(<D>) does not halt
    -> H(<D>, <D>) = reject
    -> By D's definition, D halts with return
    -> D(<D>) halts
    -> Contradiction!

  Both cases lead to contradictions
  -> The assumption "H exists" is false
  -> HALT is undecidable QED

  Key points of this proof:
  1. Self-reference (D takes itself as input)
  2. Diagonalization (isomorphic to Cantor's diagonal argument)
  3. Paradoxical structure (similar to the "Liar's Paradox")
```

```python
# Python-style pseudocode demonstrating the impossibility of the halting problem

def hypothetical_halts(program_source, input_data):
    """A hypothetical halting decision function (does not actually exist)"""
    # Determines whether program_source halts
    # when executed with input_data
    # ... magical implementation ...
    pass

def diagonal(program_source):
    """A program that creates a contradiction"""
    if hypothetical_halts(program_source, program_source):
        # If determined to halt, enter an infinite loop
        while True:
            pass
    else:
        # If determined not to halt, halt
        return

# Get the source code of diagonal
diagonal_source = inspect.getsource(diagonal)

# What happens when we run diagonal(diagonal_source)?
#
# Case 1: hypothetical_halts(diagonal_source, diagonal_source) == True
#   -> diagonal is determined to halt
#   -> But diagonal enters an infinite loop
#   -> Does not halt -> Contradiction
#
# Case 2: hypothetical_halts(diagonal_source, diagonal_source) == False
#   -> diagonal is determined not to halt
#   -> But diagonal halts with return
#   -> Halts -> Contradiction
#
# Both cases lead to contradiction -> hypothetical_halts cannot exist
```

### 4.3 Relationship with Cantor's Diagonal Argument

```
Structural similarity between Cantor's diagonal argument (1891) and the halting problem proof:

  Cantor: The set of real numbers is "larger" than the set of natural numbers

  Correspondence table between natural numbers and reals (assumption):
  n | Decimal expansion of real number
  --+-------------------
  1 | 0.5 2 3 1 7 ...
  2 | 0.3 1 4 1 5 ...
  3 | 0.7 7 7 8 2 ...
  4 | 0.1 4 9 3 6 ...
  5 | 0.8 2 0 5 1 ...
      v v v v v
  Diagonal: 0.5 1 7 3 1 ...
  New number: 0.6 2 8 4 2 ... (add 1 to each digit)
  -> This number differs from every row in the table -> Contradiction

  Halting problem proof:

  Correspondence table of programs and their inputs (assumption):
        | P1    P2    P3    P4   ...
  ------+----------------------------
  P1    | Halt  Halt  Loop  Halt ...
  P2    | Loop  Halt  Halt  Loop ...
  P3    | Halt  Loop  Loop  Halt ...
  P4    | Halt  Halt  Halt  Loop ...

  D(Pi) = inversion of the diagonal:
  D(P1): Loop (inversion of Halt)
  D(P2): Loop (inversion of Halt)
  D(P3): Halt (inversion of Loop)
  D(P4): Halt (inversion of Loop)
  -> D differs from every row -> Contradiction

  Common structure:
  1. Assume everything can be enumerated
  2. Take the diagonal
  3. Construct a new element by inverting the diagonal
  4. Show that it is not included in the enumeration -> Contradiction
```

### 4.4 Practical Consequences of the Halting Problem

```
Practical implications of the halting problem being undecidable:

  1. A perfect bug detector does not exist
     +--------------------------------------------+
     | A "static analysis tool that detects all    |
     | bugs" is impossible in principle            |
     |                                            |
     | Reason: "Does this program have a bug?" is  |
     | reducible to the halting problem            |
     |                                            |
     | Practical approaches:                       |
     | - Conservative approximation (false positives allowed, no false negatives) |
     | - Optimistic approximation (no false positives, false negatives allowed)   |
     | - Testing (only finite cases)               |
     +--------------------------------------------+

  2. A perfect virus detector does not exist
     - Detecting all malware patterns is impossible
     - Combination of heuristic and signature-based detection
     - Supplemented by behavior-based detection

  3. Perfect dead code elimination is impossible
     - "Will this branch ever be executed?" is generally undecidable
     - Compiler optimizations have limits

  4. A perfect type checker does not exist
     - A type system that detects all type errors while accepting
       all correct programs is impossible
     - Practical type systems are conservative (type-safe but restrictive)

  5. Perfect resource leak detection is impossible
     - Complete detection of memory leaks, file handle leaks, etc. is impossible
     - Ownership systems (Rust) are a good example of a conservative approach
```

```python
# Concrete examples demonstrating consequences of the halting problem

# Example 1: Limits of dead code detection
def is_dead_code(function_body, line_number):
    """
    Completely determining whether a given line is unreachable is impossible

    Reason: Consider code like the following
    """
    pass

def example_with_undecidable_reachability():
    """Example where reachability is undecidable"""
    x = compute_goldbach()  # Halts if Goldbach's conjecture is true
    if x:
        print("Is this line reachable?")  # Depends on an unsolved mathematical problem
        # -> Determining whether this is dead code is
        #   equivalent to proving/disproving Goldbach's conjecture

def compute_goldbach():
    """Search for a counterexample to Goldbach's conjecture"""
    n = 4
    while True:
        if n % 2 == 0 and not is_sum_of_two_primes(n):
            return n  # Halt if a counterexample is found
        n += 2
    # Never halts if no counterexample exists

def is_sum_of_two_primes(n):
    """Check if n can be expressed as the sum of two primes"""
    for i in range(2, n):
        if is_prime(i) and is_prime(n - i):
            return True
    return False

def is_prime(n):
    """Primality test"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


# Example 2: Conservative approximation in static analysis
class ConservativeAnalyzer:
    """Example of a conservative static analyzer"""

    def check_division_by_zero(self, code_ast):
        """
        Conservative check for potential division by zero

        Since perfect detection is impossible:
        - False positives are tolerated
        - False negatives are not tolerated
        """
        warnings = []

        for node in self.walk(code_ast):
            if self.is_division(node):
                divisor = self.get_divisor(node)
                if not self.can_prove_nonzero(divisor):
                    # Cannot prove nonzero -> Warning
                    # (may not actually be zero in practice)
                    warnings.append(f"Line {node.line}: Potential division by zero")

        return warnings

    def can_prove_nonzero(self, expr):
        """Can we prove the expression is nonzero? (conservative)"""
        # Constant case
        if self.is_constant(expr) and self.eval_constant(expr) != 0:
            return True
        # Patterns like abs(x) > 0
        if self.matches_positive_pattern(expr):
            return True
        # Otherwise cannot prove (conservatively return False)
        return False
```

---

## 5. Decidable and Undecidable

### 5.1 Classification System for Problems

```
Hierarchy of languages (problems):

  +-----------------------------------------------------+
  |               All languages                          |
  |  +----------------------------------------------+   |
  |  |         Turing-recognizable                   |   |
  |  |  +--------------------------------------+    |   |
  |  |  |         Decidable                     |    |   |
  |  |  |  +----------------------------+      |    |   |
  |  |  |  |    Context-free             |      |    |   |
  |  |  |  |  +------------------+      |      |    |   |
  |  |  |  |  |   Regular        |      |      |    |   |
  |  |  |  |  +------------------+      |      |    |   |
  |  |  |  +----------------------------+      |    |   |
  |  |  +--------------------------------------+    |   |
  |  |     Semi-decidable (RE)                       |   |
  |  +----------------------------------------------+   |
  |    Includes Turing-unrecognizable (complement of co-RE, etc.) |
  +-----------------------------------------------------+

  Characteristics of each class:

  Regular Languages (REG):
  - Recognizable by finite automata
  - Describable by regular expressions
  - Examples: Email address format checking, identifier patterns

  Context-Free Languages (CFL):
  - Recognizable by pushdown automata
  - Describable by context-free grammars
  - Examples: Programming language syntax, matching parentheses

  Decidable Languages (R):
  - Recognizable by a halting Turing machine
  - Always produces an answer in finite time
  - Examples: Primality testing, graph connectivity

  Turing-Recognizable Languages (RE):
  - Recognizable by a Turing machine (but may not halt)
  - If a string belongs, it can be detected in finite time
  - If a string does not belong, it may wait forever
  - Example: Halting problem (halting cases are detectable)
```

### 5.2 Concrete Examples of Decidable Problems

```python
# Examples of decidable problems

# 1. Acceptance problem for finite automata
# A_DFA = { <B, w> | DFA B accepts string w }
class DFA:
    """Deterministic Finite Automaton"""
    def __init__(self, states, alphabet, transitions, start, accepts):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions  # (state, symbol) -> state
        self.start = start
        self.accepts = accepts

    def accepts_string(self, w):
        """Decidable in O(|w|) -- always halts"""
        state = self.start
        for symbol in w:
            state = self.transitions.get((state, symbol))
            if state is None:
                return False
        return state in self.accepts


# 2. Membership testing for context-free grammars
# A_CFG = { <G, w> | CFG G generates string w }
def cyk_algorithm(grammar, string):
    """CYK algorithm -- decidable in O(n^3|G|)"""
    n = len(string)
    if n == 0:
        return grammar.start in grammar.nullable

    # Initialization
    table = [[set() for _ in range(n)] for _ in range(n)]

    # Substrings of length 1
    for i in range(n):
        for lhs, rhs_list in grammar.rules.items():
            for rhs in rhs_list:
                if len(rhs) == 1 and rhs[0] == string[i]:
                    table[i][i].add(lhs)

    # Substrings of length 2 or more
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                for lhs, rhs_list in grammar.rules.items():
                    for rhs in rhs_list:
                        if len(rhs) == 2:
                            B, C = rhs
                            if B in table[i][k] and C in table[k+1][j]:
                                table[i][j].add(lhs)

    return grammar.start in table[0][n-1]


# 3. Equivalence testing for finite automata
# EQ_DFA = { <A, B> | DFA A and B recognize the same language }
def dfa_equivalence(dfa_a, dfa_b):
    """
    Determine whether two DFAs are equivalent
    Construct the product automaton and check if the symmetric difference is empty
    Always halts -- decidable
    """
    # Symmetric difference = (L(A) - L(B)) union (L(B) - L(A))
    # Symmetric difference is empty <-> L(A) = L(B)
    symmetric_diff = construct_symmetric_difference(dfa_a, dfa_b)
    return is_empty(symmetric_diff)

def is_empty(dfa):
    """Determine if the DFA's language is empty using BFS"""
    visited = set()
    queue = [dfa.start]
    visited.add(dfa.start)

    while queue:
        state = queue.pop(0)
        if state in dfa.accepts:
            return False  # Accept state reachable -> Not empty
        for symbol in dfa.alphabet:
            next_state = dfa.transitions.get((state, symbol))
            if next_state and next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)

    return True  # Accept state unreachable -> Empty


# 4. Graph connectivity
def is_connected(graph):
    """Determine if a graph is connected -- O(V + E)"""
    if not graph:
        return True

    visited = set()
    start = next(iter(graph))

    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return len(visited) == len(graph)
```

### 5.3 Concrete Examples of Undecidable Problems

```
Undecidable problems (representative examples):

  1. Halting Problem (HALT)
     Input: Program P and input x
     Question: Does P halt on x?
     -> Undecidable

  2. Totality Problem (TOTAL)
     Input: Program P
     Question: Does P halt on all inputs?
     -> Undecidable (even harder than the halting problem)

  3. Equivalence Problem (EQ_TM)
     Input: Turing machines M1 and M2
     Question: Is L(M1) = L(M2)?
     -> Undecidable

  4. Post Correspondence Problem (PCP)
     Input: Set of dominoes {(u1,v1), (u2,v2), ..., (un,vn)}
     Question: Do i1, i2, ..., ik exist such that
               u_{i1}u_{i2}...u_{ik} = v_{i1}v_{i2}...v_{ik}?
     -> Undecidable

  5. Tiling Problem
     Input: Set of tiles
     Question: Can the infinite plane be tiled without gaps?
     -> Undecidable

  6. Diophantine Equations (Hilbert's Tenth Problem)
     Input: Multivariate polynomial with integer coefficients
     Question: Does an integer solution exist?
     -> Undecidable (Matiyasevich's theorem, 1970)

  7. Mortality of Matrices Problem
     Input: Set of matrices
     Question: Is there a combination whose product is the zero matrix?
     -> Undecidable
```

```python
# Concrete example of the Post Correspondence Problem (PCP)

def pcp_brute_force(dominoes, max_length=10):
    """
    Brute-force search for PCP (may not halt in general)

    Domino example:
    [(ab, b), (b, ab), (a, aa)]

    Solution: 1,2,1,3 -> ababba = ababba (check)
    """
    from itertools import product

    for length in range(1, max_length + 1):
        for combo in product(range(len(dominoes)), repeat=length):
            top = ''.join(dominoes[i][0] for i in combo)
            bottom = ''.join(dominoes[i][1] for i in combo)
            if top == bottom:
                return list(combo), top

    return None  # No solution found within max_length

# Execution example
dominoes = [('ab', 'b'), ('b', 'ab'), ('a', 'aa')]
result = pcp_brute_force(dominoes, max_length=8)
if result:
    indices, matched = result
    print(f"Solution: {indices}")
    print(f"Matched string: {matched}")

# What it means that PCP is undecidable:
# - No algorithm exists to determine whether a solution exists
# - If a solution exists, brute-force search can find it (semi-decidable)
# - If no solution exists, the search never terminates
```

### 5.4 Semi-decidability

```
Semi-decidable (Recursively Enumerable):

  Definition: Language L is semi-decidable <=>
    If w in L, answer "yes" in finite time
    If w not in L, may never answer

  Example:
  +----------------------------------------------------+
  | The halting problem is semi-decidable               |
  |                                                    |
  | Method: Actually run program P on input x           |
  | - If it halts -> "Yes, it halts"                   |
  | - If it doesn't halt -> Wait forever (cannot answer)|
  +----------------------------------------------------+

  Important properties:
  - L is decidable <=> Both L and L-bar (complement) are semi-decidable
  - If L is semi-decidable but L-bar is not semi-decidable -> L is undecidable
  - The complement of the halting problem (non-halting problem) is not semi-decidable

  Proof: L is decidable <=> Both L and L-bar are semi-decidable

  (=>) If L is decidable, then a TM M that decides L exists.
      Since M always halts, both L and L-bar are semi-decidable.

  (<=) If both L and L-bar are semi-decidable, then
      recognizers M1 for L and M2 for L-bar exist.
      Running M1 and M2 in parallel on input w,
      one of them must halt.
      -> If M1 halts, accept; if M2 halts, reject
      -> Always halts -> L is decidable QED
```

```python
# Implementation examples of semi-decidability

import threading
import time

class SemiDecider:
    """Semi-decision procedure"""

    def check_halts(self, program, input_data, timeout=None):
        """
        Check if a program halts (semi-decidable)

        - If it halts: Returns True (in finite time)
        - If it doesn't halt: Returns None after timeout
          (waits forever without timeout)
        """
        result = [None]
        error = [None]

        def run_program():
            try:
                exec(program, {'input': input_data})
                result[0] = True
            except Exception as e:
                error[0] = e
                result[0] = True  # Terminated with exception = halted

        thread = threading.Thread(target=run_program)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return None  # Timeout (unknown whether it halts)
        return result[0]


# Example of making something decidable through parallel execution
def parallel_decide(recognizer_l, recognizer_l_complement, input_data):
    """
    When recognizers for both L and L-bar exist, a decider can be constructed

    Premise: Both L and L-bar are semi-decidable
    Conclusion: L is decidable
    """
    result = [None]

    def run_l():
        if recognizer_l(input_data):
            result[0] = True  # w in L

    def run_l_complement():
        if recognizer_l_complement(input_data):
            result[0] = False  # w not in L

    t1 = threading.Thread(target=run_l)
    t2 = threading.Thread(target=run_l_complement)
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()

    # Wait until one of them halts (one must halt)
    while result[0] is None:
        time.sleep(0.001)

    return result[0]
```

---

## 6. Reduction

### 6.1 Concept of Reduction

```
Reduction: Converting a solution for problem A into a solution for problem B

  A <= B: "A is reducible to B"

  Meaning:
  - If B is solvable, then A is also solvable
  - If A is unsolvable, then B is also unsolvable (contrapositive)

  Diagram:

  Input w -> [Transformation f] -> f(w) -> [Decider for B] -> Answer

  w in A <=> f(w) in B

  Types of reduction:

  1. Mapping Reduction (Many-One Reduction)
     - A computable function f exists such that w in A <=> f(w) in B
     - Written as A <=_m B

  2. Turing Reduction
     - Decide A using B as an "oracle"
     - Written as A <=_T B
     - More powerful than mapping reduction

  Using reduction:

  Known: HALT is undecidable

  To show that a new problem X is undecidable:
  1. Show HALT <=_m X
  2. That is, show that the halting problem can be transformed into X
  3. If X were decidable, the halting problem would also be decidable -> Contradiction
  -> X is undecidable QED
```

### 6.2 Concrete Examples of Reduction

```python
# Concrete examples of reduction: Halting problem -> Totality problem

# Totality problem: Does program P halt on all inputs?
# TOTAL = { <P> | P halts on all inputs }

# Theorem: TOTAL is undecidable

# Proof: Show HALT <=_m TOTAL-bar (reduce to complement of TOTAL)

def reduce_halt_to_total(program_p, input_w):
    """
    Transform a halting problem instance (P, w) into
    a totality problem instance Q
    """
    # Construct new program Q:
    # Q(x) = run P(w) and halt if it halts;
    #         x is ignored

    q_source = f"""
def Q(x):
    # Ignore x and run P(w)
    {program_p}({input_w})
    return  # If P(w) halts, Q also halts
"""

    # P(w) halts -> Q(x) halts on all inputs x -> Q in TOTAL
    # P(w) doesn't halt -> Q(x) doesn't halt on any input -> Q not in TOTAL

    # Therefore: (P, w) in HALT <=> Q in TOTAL

    return q_source


# Reduction example 2: Halting problem -> Equivalence problem

# Equivalence problem: Do two TMs recognize the same language?
# EQ_TM = { <M1, M2> | L(M1) = L(M2) }

def reduce_halt_to_eq(program_p, input_w):
    """
    Transform a halting problem instance (P, w) into
    an equivalence problem instance (M1, M2)
    """
    # M1: Reject all inputs (empty language)
    m1 = lambda x: False

    # M2: Run P(w); if it halts, reject all inputs
    #      If it doesn't halt, loop
    def m2(x):
        # Run P(w)
        exec(program_p, {'input': input_w})
        # Only reaches here if P(w) halts
        return False  # Reject

    # P(w) halts -> L(M2) = empty = L(M1) -> (M1,M2) in EQ_TM
    # P(w) doesn't halt -> M2 loops on all inputs -> L(M2) is unrecognizable
    #                   -> L(M1) != L(M2) -> (M1,M2) not in EQ_TM

    return m1, m2
```

### 6.3 Hierarchy of Reductions

```
Hierarchy of difficulty for undecidable problems (arithmetical hierarchy):

  Sigma_0^0 = Pi_0^0 = Decidable problems

  Sigma_1^0: Semi-decidable (RE)
    Example: HALT (halting problem)
    -> Can detect when "it halts"

  Pi_1^0: co-RE (complement of RE)
    Example: HALT-bar (non-halting problem)
    -> Can detect when "it doesn't halt"

  Sigma_2^0:
    Example: TOTAL (totality problem)
    -> "Does it halt on all inputs?"

  Pi_2^0:
    Example: TOTAL-bar
    -> "Does it fail to halt on some input?"

  Sigma_3^0, Pi_3^0, ... continues infinitely

  Diagram of the hierarchy:

  Difficulty ->
  ------------------------------------------->
  Sigma_0^0    Sigma_1^0    Sigma_2^0    Sigma_3^0    ...
  (Decidable)   (RE)      (TOTAL etc.)

  At each level, "inherently difficult" problems exist
  -> Their difficulty can be compared through reductions
```

---

## 7. Rice's Theorem

### 7.1 Content of Rice's Theorem

```
Rice's Theorem:

  Theorem: All nontrivial properties of the function
  computed by a Turing machine are undecidable

  Formally:
  Let P be a property of Turing-recognizable languages
  (P is a subset of the set of all languages)

  If P is nontrivial (P != empty and P != all languages), then
  { <M> | L(M) in P } is undecidable

  Intuitively:
  - No general algorithm can answer nontrivial questions about
    "what does this program compute?"

  Concrete examples (all undecidable):

  1. "Does the TM recognize the empty language?"
     -> P = {empty} (only the empty language): Nontrivial

  2. "Does the TM recognize a regular language?"
     -> P = set of regular languages: Nontrivial

  3. "Does the TM accept at least one string?"
     -> P = {L | L != empty}: Nontrivial

  4. "Does the TM accept all strings?"
     -> P = {Sigma*}: Nontrivial

  5. "Does the TM accept a specific string w?"
     -> P = {L | w in L}: Nontrivial

  Examples where Rice's theorem does NOT apply:

  1. "Does the TM have 100 or fewer states?"
     -> This is a property of the TM's structure (not of the computed function)
     -> Decidable (can be determined by examining the TM's description)

  2. "Does the TM accept the empty input within 5 steps?"
     -> Determinable by simulation
     -> Decidable
     -> However, "Does it halt in an arbitrary number of steps?" is undecidable
```

### 7.2 Proof of Rice's Theorem

```
Proof of Rice's Theorem:

  Premises:
  - Let P be a nontrivial property of languages
  - P != empty and P != all languages
  - Assume empty language not in P (if empty in P, argue with complement of P)

  Proof:

  Since P is nontrivial, there exists a language L0 in P
  There exists a TM M0 that recognizes L0

  Reduce the halting problem to Rice's:

  For input <M, w> (does M halt on w?),
  construct a new TM M':

  M'(x):
    1. Run M on w
    2. If M halts, run M0 on x and return the result
    3. If M doesn't halt, M' also doesn't halt

  Analysis:
  - M halts on w -> M' behaves like M0 -> L(M') = L0 in P
  - M doesn't halt on w -> M' accepts nothing -> L(M') = empty not in P

  Therefore: (M, w) in HALT <=> L(M') in P

  If { <M'> | L(M') in P } were decidable,
  HALT would also be decidable -> Contradiction

  -> { <M> | L(M) in P } is undecidable QED
```

```python
# Practical implications of Rice's theorem

class ProgramAnalyzer:
    """
    By Rice's theorem, all of the following analyses are generally impossible
    """

    def does_program_always_return_positive(self, program):
        """Does the program always return a positive value? -> Undecidable"""
        raise UndecidableError("Impossible by Rice's theorem")

    def does_program_use_network(self, program):
        """
        Does the program access the network?

        Note: This is decidable if checked "syntactically"
        (check for imports or socket calls)

        However, "semantic" checking is undecidable
        (when dynamically generated code accesses the network)
        """
        # Syntactic checking (conservative approximation) is possible
        return 'socket' in program or 'requests' in program

    def is_program_equivalent_to(self, program_a, program_b):
        """Do two programs compute the same function? -> Undecidable"""
        raise UndecidableError("Impossible by Rice's theorem")

    def does_program_terminate_for_all_inputs(self, program):
        """Does the program halt on all inputs? -> Undecidable"""
        raise UndecidableError("Impossible by Rice's theorem")


class UndecidableError(Exception):
    """Error for undecidable problems"""
    pass


# Practical approaches:
#
# 1. Conservative Approximation (Sound but Incomplete)
#    - "May be unsafe" -> Issue a warning
#    - May have false positives, but no false negatives
#    Example: Rust's borrow checker
#
# 2. Optimistic Approximation (Complete but Unsound)
#    - "Probably safe" -> Pass it
#    - May have false negatives, but no false positives
#    Example: Runtime checks in dynamically typed languages
#
# 3. Restricted Exact Solutions
#    - Restrict the scope and determine accurately within that scope
#    Example: Bounded model checking
#
# 4. Interactive Verification
#    - Have the user provide hints or proofs
#    Example: Interactive theorem provers (Coq, Isabelle, Lean)
```

---

## 8. The Recursion Theorem

### 8.1 Content of the Recursion Theorem

```
Recursion Theorem (Fixed Point Theorem):

  Theorem: For any computable transformation t,
  there exists a Turing machine M such that
  t(<M>) and M recognize the same language

  Intuitively:
  "For any program transformation, there exists a program
   that is a fixed point of that transformation"

  Also known as Kleene's Fixed Point Theorem:
  For any computable function f,
  there exists a program e such that phi_e = phi_{f(e)}
  (the function computed by e = the function computed by f(e))

  Meaning:
  - Programs can "know their own source code"
  - Guarantees the existence of quines (programs that output themselves)
  - Theoretical foundation for self-replication in computer viruses
  - Computational version of the diagonalization lemma in Godel's incompleteness theorem
```

### 8.2 Quines (Self-Outputting Programs)

```python
# Quine: A program that outputs its own source code
# A constructive proof example of the recursion theorem

# Python quine (minimal version)
s='s=%r;print(s%%s)';print(s%s)

# More readable version
def quine():
    """
    By the recursion theorem, the construction of a quine
    is guaranteed in any programming language
    """
    # Two-part construction: Data part (A) and Code part (B)
    # A = Text representation of B
    # B = Code that reconstructs the whole using A

    A = 'def quine():\n    A = %r\n    B = A %% A\n    print(B)\nquine()'
    B = A % A
    print(B)

# Quine examples in various languages:

# JavaScript
js_quine = '(function q(){console.log("("+q+")()")})()'

# Ruby
ruby_quine = 's="s=%p;puts s%%s";puts s%s'

# C
c_quine = '''
#include <stdio.h>
int main(){char*s="#include <stdio.h>%cint main(){char*s=%c%s%c;printf(s,10,34,s,34);}";printf(s,10,34,s,34);}
'''

# Structure of a quine:
# 1. Data part: A template of the entire program (with some blanks)
# 2. Code part: Output by embedding the data part itself into the blanks
#
# The recursion theorem guarantees that this construction is always possible
```

---

## 9. Computability and Practice

### 9.1 Impact on Software Verification

```
Impact of computability theory on software development:

  +-----------------------------------------------------+
  | Perfect automated verification is impossible -> Practical approaches |
  +-----------------------------------------------------+

  1. Type Systems (a representative example of conservative approximation)

     TypeScript example:
     ```
     function divide(a: number, b: number): number {
       return a / b;  // No check for b=0
     }
     // TypeScript: No type error
     // -> "Detecting all runtime errors via types" is undecidable

     // Safer approach:
     function safeDivide(a: number, b: NonZero<number>): number {
       return a / b;
     }
     // -> Eliminate division by zero at the type level (dependent type approach)
     ```

  2. Static Analysis Tools (Conservative / Optimistic)

     Lint tools such as ESLint, Pylint, Clippy:
     - Syntax pattern-based -> Only decidable portions
     - Tolerates false positives
     - False negatives also exist

     Advanced static analysis:
     - Coverity, PVS-Studio: Path-sensitive analysis
     - Abstract Interpretation: Approximation through abstract interpretation
     - Still not perfect (Rice's theorem)

  3. Testing (Finite Sampling)

     Fundamental limits of testing:
     - "Testing can show the presence of bugs, but not their absence"
       (Dijkstra)
     - Covering an infinite input space with finite test cases
     - Improvable with property-based testing (QuickCheck, etc.)

  4. Formal Verification (Restricted Exact Solutions)

     Model Checking:
     - Completely verifiable for finite-state cases
     - Infinite states -> Approximate with abstraction to finite
     - SPIN, TLA+, Alloy, etc.

     Theorem Proving:
     - Humans guide the proof
     - Coq, Isabelle, Lean, Agda
     - seL4 microkernel: Complete proof of functional correctness
```

### 9.2 Impact on Programming Language Design

```python
# Impact of computability theory on programming language design

# 1. Totality Guarantees vs Turing Completeness
#    A language that guarantees halting on all inputs
#    is not Turing-complete

# Example from Agda (a total functional language) (conceptual):
# - All functions are guaranteed to terminate
# - Recursion requires structurally decreasing arguments
# - Not Turing-complete, but sufficient for practical purposes

# Conceptual example of total recursion
def structural_recursion(n: int) -> int:
    """Structural recursion: argument always decreases -> termination guaranteed"""
    if n <= 0:
        return 0
    return 1 + structural_recursion(n - 1)  # n always decreases

def non_structural(n: int) -> int:
    """Non-structural recursion: termination not guaranteed"""
    if n == 1:
        return 0
    if n % 2 == 0:
        return 1 + non_structural(n // 2)
    else:
        return 1 + non_structural(3 * n + 1)  # Collatz conjecture
    # Whether it halts on all inputs is unknown!


# 2. Decidability of Type Inference
#
# Hindley-Milner type inference (ML, Haskell98): Decidable
# - A principal type exists
# - Complete type inference is possible
#
# Type inference for System F (polymorphic lambda calculus): Undecidable
# - Explicit type annotations are needed
# - Applies to some Haskell extensions
#
# TypeScript's type inference: Intentionally Turing-complete
# - Conditional Types + Recursive Types = Turing-complete
# - Type definitions that cause the type checker to loop forever can be written

# TypeScript type-level computation example (conceptual):
# type Fibonacci<N extends number> = ...
# -> The type checker may enter an infinite loop


# 3. Macro Systems and Computability
#
# C preprocessor: Not Turing-complete
# -> Macro expansion always terminates
#
# Rust macros: Limited Turing completeness
# -> Recursion depth limit (#![recursion_limit = "..."])
#
# Template Haskell: Turing-complete
# -> Arbitrary computation possible at compile time
#
# Lisp macros: Turing-complete
# -> Can loop at compile time (macro expansion time)
```

### 9.3 Limits of Compiler Optimization

```python
# Compiler optimization and the limits of computability

class CompilerOptimizer:
    """Examples showing the limits of compiler optimization"""

    def dead_code_elimination(self, code):
        """
        Dead Code Elimination

        Perfect dead code elimination is undecidable (Rice's theorem)
        Practical compilers use conservative approximations
        """
        # Unreachable code:
        # - Always-false conditional branch -> Detectable (constant folding)
        # - Condition depending on function return value -> Generally undetectable
        pass

    def constant_propagation(self, code):
        """
        Constant Propagation

        "Is this variable always a constant value?" is generally undecidable
        However, many practical cases can be determined
        """
        pass

    def loop_optimization(self, code):
        """
        Loop Optimization

        "Does this loop terminate in a finite number of iterations?" is undecidable
        -> There are also limits to automatic discovery of loop invariants

        However:
        - for i in range(n): ... -> Clearly finite
        - while condition: ... -> Generally unknown
        """
        pass

    def alias_analysis(self, code):
        """
        Alias Analysis

        "Do these two pointers point to the same memory?"
        -> Perfect analysis is undecidable
        -> Conservative approximations (Andersen, Steensgaard, etc.)

        Practical impact:
        - C's restrict keyword: Programmer provides hints
        - Rust's borrowing rules: Compiler can safely track
        """
        pass


# Optimization example: GCC vs. computability limits

# Dead code elimination of the following code:
def optimization_example():
    x = complex_computation()  # Unknown whether it has side effects
    if x > 0:
        print("positive")
    else:
        print("non-positive")

    # What the compiler wants to know:
    # 1. Does complex_computation() always return a positive value? -> Undecidable
    # 2. Does complex_computation() have side effects? -> Undecidable
    # 3. Which branch of this if statement is taken? -> Undecidable

    # Result: The compiler keeps both branches (conservative)
```

### 9.4 Impact on Security

```
Impact of computability theory on security:

  1. Perfect malware detection is impossible
     +--------------------------------------------+
     | Theorem: A program that accurately detects  |
     | all malware does not exist                  |
     |                                            |
     | Proof: Reduce malware detection to the      |
     | halting problem                             |
     | For any program P:                          |
     | "Does P perform malicious behavior?"        |
     | is a nontrivial property -> Undecidable by  |
     | Rice's theorem                              |
     +--------------------------------------------+

  2. Perfect information flow analysis is impossible
     - "Does this program leak confidential data?"
     - Nontrivial semantic property -> Undecidable
     - Countermeasure: Taint tracking (conservative approximation)

  3. Limits of obfuscation
     - Barak et al. (2001): Perfect general obfuscation is impossible
     - "Completely hiding the implementation of a program" is theoretically impossible
     - However, specific classes of obfuscation (iO, etc.) may be possible

  4. Practical approaches
     - Signature-based detection: Known patterns only
     - Heuristic detection: Conservative approximation
     - Sandbox execution: Observe actual behavior
     - Formal verification: Verify limited properties accurately
```

---

## 10. Equivalence of Computational Models

### 10.1 Relationships Among Computational Models

```
Major computational models and their equivalence:

  +----------------------------------------------------+
  |           Turing-equivalent computational models     |
  +----------------------------------------------------+
  |                                                    |
  |  Turing Machine (TM)                               |
  |    - The most standard model                       |
  |    - Tape + Head + Finite control                  |
  |                                                    |
  |  Lambda Calculus                                    |
  |    - Invented by Church                            |
  |    - Only function abstraction and application      |
  |    - Theoretical foundation of functional programming |
  |    -> See 05-lambda-calculus.md for details          |
  |                                                    |
  |  Recursive Functions                                |
  |    - Primitive recursion + mu-operator              |
  |    - The most natural mathematical formulation      |
  |                                                    |
  |  Register Machine                                   |
  |    - Finite number of registers + counters          |
  |    - A model closer to actual CPUs                  |
  |                                                    |
  |  Tag System                                         |
  |    - Invented by Post                              |
  |    - Read from the head and append to the tail      |
  |    - Very simple but Turing-complete               |
  |                                                    |
  |  Cellular Automata                                  |
  |    - Game of Life (Rule 110, etc.)                 |
  |    - Grid cells change according to local rules     |
  |    - Model of natural computational processes       |
  |                                                    |
  +----------------------------------------------------+

  Method for proving equivalence:

  A -> B: Show that A can be simulated by B
  B -> A: Show that B can be simulated by A
  -> A == B (equivalent computational power)

  Chain of actual equivalence proofs:

  TM -> Lambda Calculus:
    Express TM state transitions as lambda reductions

  Lambda Calculus -> TM:
    Implement beta reduction as TM operations

  TM -> Register Machine:
    Encode tape contents in registers

  Register Machine -> TM:
    Record register values on the tape
```

### 10.2 Attempts to Exceed the Limits of Computation

```
Hypercomputation -- Attempts to go beyond Turing machines:

  1. Oracle Machine
     - TM with an oracle for the halting problem
     - Can decide HALT in O(1)
     - However, no known way to realize the oracle
     - Useful as a theoretical tool (relative computability)

  2. Accelerating Machine
     - Step n takes 1/2^n seconds -> All steps complete in 1 second
     - Physically unrealizable (speed of light limit, quantum effects, etc.)

  3. Transfinite Turing Machine
     - Executes steps up to transfinite ordinals
     - Mathematically definable but physically unrealizable

  4. Infinite Precision Analog Computation
     - Exploits infinite precision of real numbers
     - In practice, measurement precision is limited

  Current consensus:
  -> Physically realizable computing devices are believed
    not to exceed the computational power of Turing machines
  -> Quantum computers are no exception
    (Computational "power" is equivalent; only computational "efficiency" differs)
```

---

## 11. Practice Exercises

### Exercise 1: Designing a Turing Machine (Basic)

```
Problem: Design the transition table of a Turing machine that increments a binary number by 1.

Input example: 1011 -> Output: 1100
Input example: 1111 -> Output: 10000

Hints:
1. First move to the right end
2. Perform carry processing from the right end leftward
3. If all digits carry over, add a 1 at the front

Solution:
  States: {q_start, q_right, q_carry, q_done, q_accept}

  Transition table:
  (q_start, 0) -> (q_right, 0, R)  // Move to the right end
  (q_start, 1) -> (q_right, 1, R)
  (q_right, 0) -> (q_right, 0, R)
  (q_right, 1) -> (q_right, 1, R)
  (q_right, B) -> (q_carry, B, L)  // Reached right end, start carry
  (q_carry, 0) -> (q_done, 1, L)   // 0->1, carry stops
  (q_carry, 1) -> (q_carry, 0, L)  // 1->0, carry continues
  (q_carry, B) -> (q_accept, 1, R) // Carry to the front
  (q_done, 0) -> (q_done, 0, L)    // Return to left end
  (q_done, 1) -> (q_done, 1, L)
  (q_done, B) -> (q_accept, B, R)  // Complete
```

### Exercise 2: Reduction from the Halting Problem (Advanced)

```
Problem: Prove that automatically determining "whether a tool returns correct output
for all inputs" is impossible, using reduction from the halting problem.

Proof:
  CORRECT = { <P, S> | Program P is correct for specification S
                        on all inputs }

  Show HALT <=_m CORRECT:

  For any (M, w) (does M halt on w?),
  construct program P' and specification S':

  P'(x):
    1. Run M on w (ignore x)
    2. If M halts, return 0

  S': "Returns 0 for all inputs"

  Analysis:
  - M halts on w -> P' returns 0 on all inputs -> (P', S') in CORRECT
  - M doesn't halt on w -> P' loops on all inputs -> (P', S') not in CORRECT

  If CORRECT were decidable, HALT would also be decidable -> Contradiction
  -> CORRECT is undecidable QED
```

### Exercise 3: Applying Rice's Theorem (Advanced)

```
Problem: For each of the following problems, determine whether Rice's theorem
applies, and if so, show that it is a nontrivial property.

1. "Does TM M accept the empty string epsilon?"
   -> Rice's theorem applies
   -> P = {L | epsilon in L}: Nontrivial (languages that accept epsilon and those that don't both exist)
   -> Undecidable

2. "Does TM M have 100 or more states?"
   -> Rice's theorem does not apply
   -> This is a "structural" property of the TM (not a property of the computed function)
   -> Decidable (can be determined by examining the TM's description)

3. "Is the language recognized by TM M a regular language?"
   -> Rice's theorem applies
   -> P = {L | L is a regular language}: Nontrivial
   -> Undecidable

4. "Does TM M accept input 'hello' within 10 steps?"
   -> Rice's theorem does not apply
   -> Determinable by simulating a finite number of steps
   -> Decidable

5. "Is the function computed by TM M total?" (Does it halt on all inputs?)
   -> Rice's theorem applies
   -> P = {L | L is decidable}: Nontrivial
   -> Undecidable (equivalent to the TOTAL problem)
```

### Exercise 4: Constructing a Quine (Implementation)

```python
"""
Exercise: Construct a quine in Python.

Conditions:
1. Does not read external files
2. Is not an empty program
3. Outputs its own source code exactly

Hint: Use a two-part construction (data part + code part)

Solution:
"""

# Method 1: Using %r format
s='s=%r;print(s%%s)';print(s%s)

# Method 2: Using f-strings (Python 3.12+)
# exec(s:="print(f'exec(s:={chr(34)}{s}{chr(34)})')")

# Method 3: More understandable version
def make_quine():
    """Step-by-step construction for understanding quine structure"""

    # Step 1: Template (skeleton of itself)
    template = 'template = {!r}\nprint(template.format(template))'

    # Step 2: Output by embedding itself into the template
    print(template.format(template))

# Verify that the output matches the input
make_quine()
```

### Exercise 5: Practicing Reduction (Advanced)

```
Problem: Prove that the following problem is undecidable using an appropriate reduction.

Problem: EMPTY = { <M> | L(M) = empty } (Does the TM recognize the empty language?)

Proof Method 1: By Rice's theorem
  P = {empty} (property consisting of only the empty language)
  The empty language is in RE, but not all RE languages are empty -> P is nontrivial
  -> Undecidable by Rice's theorem QED

Proof Method 2: Reduction from the halting problem
  Show HALT <=_m EMPTY-bar (reduce to the complement of EMPTY)

  Input: (M, w)
  Construct TM M':

  M'(x):
    1. Run M on w
    2. If M halts, accept

  Analysis:
  - M halts on w -> M' accepts all inputs -> L(M') = Sigma* != empty -> M' not in EMPTY
  - M doesn't halt on w -> M' accepts nothing -> L(M') = empty -> M' in EMPTY

  Therefore: (M, w) in HALT <=> <M'> not in EMPTY

  If EMPTY were decidable, HALT would also be decidable -> Contradiction
  -> EMPTY is undecidable QED
```

---

## 12. Computability and Modern Topics

### 12.1 Machine Learning and Computability

```
Computability issues in machine learning:

  1. Learnability (PAC Learning)
     - "Can a certain class of functions be learned?"
     - Some problems face computability barriers
     - Ben-David et al. (2019):
       Certain learning problems depend on set-theoretic axioms (independent of ZFC)

  2. Neural Network Verification
     - "Is this NN safe for all inputs?"
     - General property verification of NNs is undecidable
     - Specific properties of ReLU networks are decidable

  3. AutoML and Neural Architecture Search
     - "Automatically find the optimal architecture"
     - Infinite search space -> Approximation by heuristics
     - Connection to the No Free Lunch theorem

  4. LLMs (Large Language Models)
     - Are LLMs Turing-complete?
     - Finite context length -> Strictly not Turing-complete
     - However, they have very broad computational capabilities in practice
     - Combined with external tools (code execution, etc.), they become Turing-complete
```

### 12.2 Quantum Computing and Computability

```
Quantum computers and computability:

  Key conclusion:
  +-------------------------------------------------------+
  | Quantum computers do not exceed the computational      |
  | "power" of Turing machines. However, they may exceed   |
  | computational "efficiency"                             |
  +-------------------------------------------------------+

  Problems accelerated by quantum:
  - Integer factorization: O(n^3) [quantum] vs O(exp(n^{1/3})) [classical]
  - Unstructured search: O(sqrt(N)) [quantum] vs O(N) [classical]
  - Quantum simulation: Exponential speedup

  Problems unsolvable even by quantum:
  - Halting problem -> Still undecidable
  - NP-complete problems -> Probably not solvable in polynomial time
    (BQP not subset of NP is unproven but conjectured)

  Impact on computability:
  - The boundary between decidable/undecidable does not change
  - What changes is only the efficiency aspect (polynomial vs exponential)
  - Parts of computational complexity theory may change
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important aspect. Understanding deepens not just through theory, but by actually writing and testing code.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to applications. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently utilized in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|------|---------|
| Turing Machine | Theoretical model of computation. Represents all computation with tape + head + finite control |
| Church-Turing Thesis | Computable = computable by a Turing machine. No counterexample has been found |
| Halting Problem | Representative example of undecidability. A perfect bug detector does not exist |
| Reduction | Transform problem A into B. Used to prove undecidability |
| Rice's Theorem | All semantic properties of programs are undecidable |
| Recursion Theorem | Programs can refer to themselves. Theoretical guarantee of quines |
| Decidability Hierarchy | Regular subset CFL subset Decidable subset RE subset All languages |
| Practical Impact | Perfection is impossible -> Use conservative/optimistic approximations appropriately |

---

## Recommended Next Guides

---

## References
1. Sipser, M. "Introduction to the Theory of Computation." Chapters 3-5.
2. Turing, A. M. "On Computable Numbers, with an Application to the Entscheidungsproblem." 1936.
3. Church, A. "An Unsolvable Problem of Elementary Number Theory." 1936.
4. Rice, H. G. "Classes of Recursively Enumerable Sets and Their Decision Problems." 1953.
5. Kleene, S. C. "Introduction to Metamathematics." 1952.
6. Godel, K. "Uber formal unentscheidbare Satze der Principia Mathematica und verwandter Systeme I." 1931.
7. Davis, M. "Computability and Unsolvability." 1958.
8. Rogers, H. "Theory of Recursive Functions and Effective Computability." 1967.
9. Cutland, N. "Computability: An Introduction to Recursive Function Theory." 1980.
10. Arora, S. and Barak, B. "Computational Complexity: A Modern Approach." 2009.
