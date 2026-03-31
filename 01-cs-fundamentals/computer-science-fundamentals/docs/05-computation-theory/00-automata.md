# Automata and Formal Languages

> Regular expressions, compilers, protocol verification -- automata theory is a theoretical foundation of CS that also deeply permeates practical applications. By understanding the equivalence of DFA/NFA, the pumping lemma for regular languages, the parsing power of context-free grammars, and the systematic classification of language classes through the Chomsky hierarchy, one can approach the essence of computation.

## What You Will Learn in This Chapter

- [ ] Understand the definition, construction, and equivalence of finite automata (DFA/NFA)
- [ ] Explain the properties and limitations of regular languages (pumping lemma)
- [ ] Implement the subset construction from NFA to DFA
- [ ] Perform mutual conversions between regular expressions and automata
- [ ] Understand the relationship between context-free grammars and pushdown automata
- [ ] Compare and explain the four levels of the Chomsky hierarchy
- [ ] Implement DFA/NFA simulators in Python

## Prerequisites

- Basic programming knowledge (Python fundamentals)
- Basic concepts of set theory (sets, mappings, Cartesian products)
- Basics of graph theory (directed graphs, paths)

---

## 1. Fundamental Concepts of Automata

### 1.1 What Is an Automaton?

An automaton (plural: automata) is an abstract computational model that reads an input string of symbols, transitions through internal states, and ultimately decides whether to "accept" or "reject" the input.

The reasons why automata theory is important are as follows:

1. **Clarifies the limits of computation**: Rigorously defines which class of machines can solve which class of problems
2. **Theoretical foundation for practical tools**: Regular expression engines, compilers, and protocol verifiers are all based on automata theory
3. **Provides design guidance**: Knowing which language class a problem belongs to allows selection of the appropriate algorithm

### 1.2 Basic Terminology of Formal Languages

We define the essential terminology for studying automata theory.

```
Terminology Definitions:

  Alphabet (Σ): A finite set of symbols
    Example: Σ = {0, 1} (binary alphabet)
    Example: Σ = {a, b, c, ..., z} (lowercase English alphabet)

  String (string / word): A finite sequence of symbols from the alphabet
    Example: When Σ = {0, 1}, "0101", "111", "0" are strings
    The empty string is written as ε (epsilon)

  String length |w|: The number of symbols in string w
    |"abc"| = 3,  |ε| = 0

  String concatenation: w₁ · w₂ = w₁w₂
    "ab" · "cd" = "abcd"
    w · ε = ε · w = w

  Σ*: The set of all strings over Σ (including the empty string)
    When Σ = {0, 1}, Σ* = {ε, 0, 1, 00, 01, 10, 11, 000, ...}

  Σ⁺: The set Σ* minus the empty string  Σ⁺ = Σ* \ {ε}

  Language: L ⊆ Σ* (a subset of Σ*)
    Example: L = {w ∈ {0,1}* | w contains an even number of 0s}
    Example: L = {aⁿbⁿ | n ≥ 0} = {ε, ab, aabb, aaabbb, ...}
```

### 1.3 Classification and Computational Power of Automata

```
Classification of Automata (in ascending order of computational power):

  ┌─────────────────────────────────────────────────────────┐
  │  Turing Machine (TM)                                     │
  │  ┌───────────────────────────────────────────────────┐  │
  │  │  Linear Bounded Automaton (LBA)                    │  │
  │  │  ┌─────────────────────────────────────────────┐  │  │
  │  │  │  Pushdown Automaton (PDA)                    │  │  │
  │  │  │  ┌───────────────────────────────────────┐  │  │  │
  │  │  │  │  Finite Automaton (FA)                 │  │  │  │
  │  │  │  │  DFA / NFA                             │  │  │  │
  │  │  │  │  → Recognizes regular languages        │  │  │  │
  │  │  │  └───────────────────────────────────────┘  │  │  │
  │  │  │  → Recognizes context-free languages         │  │  │
  │  │  └─────────────────────────────────────────────┘  │  │
  │  │  → Recognizes context-sensitive languages           │  │
  │  └───────────────────────────────────────────────────┘  │
  │  → Recognizes recursively enumerable languages           │
  └─────────────────────────────────────────────────────────┘

  Each level is a proper inclusion:
    Regular ⊂ Context-free ⊂ Context-sensitive ⊂ Recursively enumerable
```

---

## 2. Finite Automata (DFA and NFA)

### 2.1 Definition of DFA (Deterministic Finite Automaton)

A DFA is the most fundamental automaton, where the transition destination is uniquely determined for each input symbol at each state.

```
Formal Definition of DFA:
  M = (Q, Σ, δ, q₀, F)

  Q : Finite set of states
  Σ : Input alphabet (finite set)
  δ : Transition function  Q × Σ → Q (total function)
  q₀: Initial state  q₀ ∈ Q
  F : Set of accept states  F ⊆ Q

  Operation:
    1. Start from initial state q₀
    2. Read the input string one character at a time from left to right
    3. Transition according to δ based on current state and character read
    4. When all input has been read:
       - Current state ∈ F → accept
       - Current state ∉ F → reject
```

**Example: DFA that accepts strings containing an even number of 'a's**

```
  DFA M₁ = ({q₀, q₁}, {a, b}, δ, q₀, {q₀})

  State transition diagram:

       ┌──b──┐          ┌──b──┐
       │     │          │     │
       ▼     │          ▼     │
    →((q₀))──a──→( q₁ )
       ▲              │
       └──────a───────┘

  ((q₀)): Double circle denotes accept state
  →: Arrow denotes initial state

  Transition table:
  ┌────────┬────────┬────────┐
  │ State  │ a      │ b      │
  ├────────┼────────┼────────┤
  │ →*q₀   │ q₁     │ q₀     │
  │   q₁   │ q₀     │ q₁     │
  └────────┴────────┴────────┘
  →: Initial state  *: Accept state

  Trace examples:
  Input "abba":  q₀ →a→ q₁ →b→ q₁ →b→ q₁ →a→ q₀ → Accept (q₀ ∈ F)
  Input "aba":   q₀ →a→ q₁ →b→ q₁ →a→ q₀         → Accept (q₀ ∈ F)
  Input "a":     q₀ →a→ q₁                          → Reject (q₁ ∉ F)
  Input "bb":    q₀ →b→ q₀ →b→ q₀                  → Accept (q₀ ∈ F)
  Input ε:       q₀                                   → Accept (q₀ ∈ F)
```

### 2.2 Python Implementation of DFA

```python
"""
DFA (Deterministic Finite Automaton) Simulator

Define an arbitrary DFA and determine acceptance/rejection of input strings.
"""

from typing import Dict, Set, Tuple


class DFA:
    """Implementation of a Deterministic Finite Automaton (DFA)"""

    def __init__(
        self,
        states: Set[str],
        alphabet: Set[str],
        transition: Dict[Tuple[str, str], str],
        start_state: str,
        accept_states: Set[str],
    ):
        """
        Initialize the DFA.

        Args:
            states: Set of states Q
            alphabet: Input alphabet Σ
            transition: Transition function δ represented as a dictionary
                        {(state, input_symbol): next_state}
            start_state: Initial state q₀
            accept_states: Set of accept states F
        """
        self.states = states
        self.alphabet = alphabet
        self.transition = transition
        self.start_state = start_state
        self.accept_states = accept_states
        self._validate()

    def _validate(self) -> None:
        """Verify that the DFA definition is correct."""
        # Check if initial state is in the state set
        if self.start_state not in self.states:
            raise ValueError(
                f"Initial state '{self.start_state}' is not in the state set"
            )

        # Check if accept states are a subset of the state set
        if not self.accept_states.issubset(self.states):
            invalid = self.accept_states - self.states
            raise ValueError(
                f"Accept states {invalid} are not in the state set"
            )

        # Check if transition function is total (defined for all state-symbol pairs)
        for state in self.states:
            for symbol in self.alphabet:
                if (state, symbol) not in self.transition:
                    raise ValueError(
                        f"Transition δ({state}, {symbol}) is undefined"
                    )
                next_state = self.transition[(state, symbol)]
                if next_state not in self.states:
                    raise ValueError(
                        f"Transition target '{next_state}' is not in the state set"
                    )

    def process(self, input_string: str) -> Tuple[bool, list]:
        """
        Process the input string and return accept/reject.

        Args:
            input_string: The input string to process

        Returns:
            (whether accepted, history of state transitions)
        """
        current = self.start_state
        history = [current]

        for symbol in input_string:
            if symbol not in self.alphabet:
                raise ValueError(
                    f"Input symbol '{symbol}' is not in the alphabet"
                )
            current = self.transition[(current, symbol)]
            history.append(current)

        accepted = current in self.accept_states
        return accepted, history

    def accepts(self, input_string: str) -> bool:
        """Return whether the input string is accepted."""
        accepted, _ = self.process(input_string)
        return accepted

    def trace(self, input_string: str) -> str:
        """Return a string showing the processing trace of the input."""
        accepted, history = self.process(input_string)
        display_input = input_string if input_string else "ε"
        transitions = []
        for i, state in enumerate(history):
            if i < len(input_string):
                transitions.append(f"{state} →{input_string[i]}→ ")
            else:
                transitions.append(state)
        path = "".join(transitions)
        result = "Accept" if accepted else "Reject"
        return f"Input \"{display_input}\": {path} → {result}"


# --- Example: DFA that accepts strings with an even number of 'a's ---

dfa_even_a = DFA(
    states={"q0", "q1"},
    alphabet={"a", "b"},
    transition={
        ("q0", "a"): "q1",
        ("q0", "b"): "q0",
        ("q1", "a"): "q0",
        ("q1", "b"): "q1",
    },
    start_state="q0",
    accept_states={"q0"},
)

# Test
test_cases = ["", "a", "b", "aa", "ab", "abba", "aba", "aabba"]
for tc in test_cases:
    print(dfa_even_a.trace(tc))

# Output:
# Input "ε": q0 → Accept
# Input "a": q0 →a→ q1 → Reject
# Input "b": q0 →b→ q0 → Accept
# Input "aa": q0 →a→ q1 →a→ q0 → Accept
# Input "ab": q0 →a→ q1 →b→ q1 → Reject
# Input "abba": q0 →a→ q1 →b→ q1 →b→ q1 →a→ q0 → Accept
# Input "aba": q0 →a→ q1 →b→ q1 →a→ q0 → Accept
# Input "aabba": q0 →a→ q1 →a→ q0 →b→ q0 →b→ q0 →a→ q1 → Reject
```

### 2.3 Definition of NFA (Nondeterministic Finite Automaton)

An NFA is an extended model of DFA where multiple transition destinations may exist from a given state for the same input symbol, and furthermore, epsilon transitions that transition without reading input are permitted.

```
Formal Definition of NFA:
  N = (Q, Σ, δ, q₀, F)

  Q : Finite set of states
  Σ : Input alphabet (finite set)
  δ : Transition function  Q × (Σ ∪ {ε}) → P(Q)
      * P(Q) is the power set of Q (the set of all subsets)
  q₀: Initial state  q₀ ∈ Q
  F : Set of accept states  F ⊆ Q

  Differences from DFA:
  ┌──────────────────┬────────────────────┬───────────────────┐
  │ Feature          │ DFA                │ NFA               │
  ├──────────────────┼────────────────────┼───────────────────┤
  │ Transition count │ Exactly 1 per symbol│ 0 or more (multi) │
  │ ε-transitions   │ Not allowed         │ Allowed           │
  │ Transition type  │ Q × Σ → Q          │ Q × (Σ∪{ε}) → P(Q)│
  │ Accept condition │ Single computation │ Any path reaches   │
  │                  │ path determines    │ an accept state    │
  │ Computational    │ Regular languages  │ Regular languages  │
  │ power            │                    │ (same)             │
  └──────────────────┴────────────────────┴───────────────────┘

  Important theorem: DFA and NFA are equivalent in computational power
  → For any NFA, there exists a DFA that accepts the same language
  → However, the DFA may have up to 2^n states (n = number of NFA states)
```

**Example: NFA that accepts strings ending with "abb"**

```
  NFA N₁ = ({q₀, q₁, q₂, q₃}, {a, b}, δ, q₀, {q₃})

  State transition diagram:

       ┌─a,b─┐
       │     │
       ▼     │
    →( q₀ )──a──→( q₁ )──b──→( q₂ )──b──→(( q₃ ))
                                            Accept state

  Transition table:
  ┌────────┬────────────┬────────────┐
  │ State  │ a          │ b          │
  ├────────┼────────────┼────────────┤
  │ →q₀    │ {q₀, q₁}  │ {q₀}      │
  │  q₁    │ ∅          │ {q₂}      │
  │  q₂    │ ∅          │ {q₃}      │
  │ *q₃    │ ∅          │ ∅         │
  └────────┴────────────┴────────────┘

  Trace example (input "aabb"):
  Nondeterministically branching computation tree:
                    q₀
                   / \
           a→    /   \  ←a
               q₀    q₁
              / \     |
       a→   /   \    |←b
           q₀   q₁   q₂
           |    |     |
    b→     |    |←b   |←b (no transition → dies)
           q₀   q₂
           |    |
    b→     |    |←b
           q₀   q₃ ← Reached accept state!

  At least one path reaches an accept state → "aabb" is accepted
```

### 2.4 Python Implementation of NFA

```python
"""
NFA (Nondeterministic Finite Automaton) Simulator

Supports epsilon transitions and performs acceptance checking via subset tracking.
"""

from typing import Dict, FrozenSet, Set, Tuple


class NFA:
    """Implementation of a Nondeterministic Finite Automaton (NFA)"""

    def __init__(
        self,
        states: Set[str],
        alphabet: Set[str],
        transition: Dict[Tuple[str, str], Set[str]],
        start_state: str,
        accept_states: Set[str],
    ):
        """
        Initialize the NFA.

        Args:
            states: Set of states Q
            alphabet: Input alphabet Σ (not including ε)
            transition: Transition function δ
                        {(state, input_symbol_or_ε): set of next states}
                        ε-transitions use ("state", "") as key
            start_state: Initial state q₀
            accept_states: Set of accept states F
        """
        self.states = states
        self.alphabet = alphabet
        self.transition = transition
        self.start_state = start_state
        self.accept_states = accept_states

    def epsilon_closure(self, states: Set[str]) -> Set[str]:
        """
        Compute the epsilon closure of the given set of states.

        Returns the set containing all states reachable via epsilon transitions only.
        """
        closure = set(states)
        stack = list(states)

        while stack:
            state = stack.pop()
            # Get epsilon transition targets (epsilon transitions use empty string "")
            epsilon_targets = self.transition.get((state, ""), set())
            for target in epsilon_targets:
                if target not in closure:
                    closure.add(target)
                    stack.append(target)

        return closure

    def process(self, input_string: str) -> Tuple[bool, list]:
        """
        Process the input string and return accept/reject.

        Implemented by tracking the set of simultaneously reachable states.
        This corresponds to on-the-fly subset construction.

        Returns:
            (whether accepted, history of state sets at each step)
        """
        # Start from the epsilon closure of the initial state
        current_states = self.epsilon_closure({self.start_state})
        history = [frozenset(current_states)]

        for symbol in input_string:
            next_states = set()
            for state in current_states:
                targets = self.transition.get((state, symbol), set())
                next_states.update(targets)
            # Compute epsilon closure of transition targets
            current_states = self.epsilon_closure(next_states)
            history.append(frozenset(current_states))

        # Accept if the current state set contains an accept state
        accepted = bool(current_states & self.accept_states)
        return accepted, history

    def accepts(self, input_string: str) -> bool:
        """Return whether the input string is accepted."""
        accepted, _ = self.process(input_string)
        return accepted

    def trace(self, input_string: str) -> str:
        """Return a string showing the processing trace of the input."""
        accepted, history = self.process(input_string)
        display_input = input_string if input_string else "ε"
        parts = []
        for i, state_set in enumerate(history):
            sorted_states = sorted(state_set)
            states_str = "{" + ", ".join(sorted_states) + "}"
            if i < len(input_string):
                parts.append(f"{states_str} →{input_string[i]}→ ")
            else:
                parts.append(states_str)
        path = "".join(parts)
        result = "Accept" if accepted else "Reject"
        return f"Input \"{display_input}\": {path} → {result}"


# --- Example: NFA that accepts strings ending with "abb" ---

nfa_ends_abb = NFA(
    states={"q0", "q1", "q2", "q3"},
    alphabet={"a", "b"},
    transition={
        ("q0", "a"): {"q0", "q1"},
        ("q0", "b"): {"q0"},
        ("q1", "b"): {"q2"},
        ("q2", "b"): {"q3"},
    },
    start_state="q0",
    accept_states={"q3"},
)

# Test
test_cases_nfa = ["abb", "aabb", "babb", "ab", "abab", "aabbabb"]
for tc in test_cases_nfa:
    print(nfa_ends_abb.trace(tc))

# Output:
# Input "abb": {q0} →a→ {q0, q1} →b→ {q0, q2} →b→ {q0, q3} → Accept
# Input "aabb": {q0} →a→ {q0, q1} →a→ {q0, q1} →b→ {q0, q2} →b→ {q0, q3} → Accept
# Input "babb": {q0} →b→ {q0} →a→ {q0, q1} →b→ {q0, q2} →b→ {q0, q3} → Accept
# Input "ab": {q0} →a→ {q0, q1} →b→ {q0, q2} → Reject
# Input "abab": {q0} →a→ {q0, q1} →b→ {q0, q2} →a→ {q0, q1} →b→ {q0, q2} → Reject
# Input "aabbabb": ... → Accept
```

### 2.5 Conversion from NFA to DFA (Subset Construction)

The constructive proof of the equivalence of NFA and DFA is the subset construction. The power set of the NFA's state set is used as the DFA's states.

```
Subset Construction Algorithm:

  Input: NFA N = (Q_N, Σ, δ_N, q₀_N, F_N)
  Output: DFA D = (Q_D, Σ, δ_D, q₀_D, F_D)

  Procedure:
    1. q₀_D = ε-closure({q₀_N})
    2. Q_D = {q₀_D} (add to worklist)
    3. Repeat until worklist is empty:
       a. Remove state S from worklist
       b. For each input symbol a ∈ Σ:
          T = ε-closure(∪{δ_N(s, a) | s ∈ S})
          δ_D(S, a) = T
          If T is not yet in Q_D, add to Q_D and worklist
    4. F_D = {S ∈ Q_D | S ∩ F_N ≠ ∅}

  Example: Converting the NFA for "ends with abb" to DFA

  NFA states: {q₀, q₁, q₂, q₃}
  (No ε-transitions, so ε-closure is the identity)

  Step 1: Initial state = {q₀}
  Step 2:
    {q₀} →a→ {q₀,q₁}  →b→ {q₀}
    {q₀,q₁} →a→ {q₀,q₁}  →b→ {q₀,q₂}
    {q₀,q₂} →a→ {q₀,q₁}  →b→ {q₀,q₃}
    {q₀,q₃} →a→ {q₀,q₁}  →b→ {q₀}

  DFA states and correspondence:
    A = {q₀}       B = {q₀,q₁}
    C = {q₀,q₂}    D = {q₀,q₃}  ← Accept state

  DFA transition table:
  ┌────────┬────────┬────────┐
  │ State  │ a      │ b      │
  ├────────┼────────┼────────┤
  │ →A     │ B      │ A      │
  │  B     │ B      │ C      │
  │  C     │ B      │ D      │
  │ *D     │ B      │ A      │
  └────────┴────────┴────────┘
```

### 2.6 Python Implementation of Subset Construction

```python
"""
Conversion from NFA to DFA (Subset Construction)

Implementation of the algorithm that converts an NFA into an equivalent DFA.
"""

from typing import Dict, FrozenSet, Set, Tuple


def nfa_to_dfa(nfa: "NFA") -> "DFA":
    """
    Convert an NFA to a DFA using subset construction.

    Args:
        nfa: The source NFA

    Returns:
        An equivalent DFA
    """
    # DFA initial state = epsilon closure of NFA initial state
    dfa_start = frozenset(nfa.epsilon_closure({nfa.start_state}))

    # Build DFA state set and transition function
    dfa_states: Set[FrozenSet[str]] = set()
    dfa_transition: Dict[Tuple[FrozenSet[str], str], FrozenSet[str]] = {}
    worklist = [dfa_start]
    dfa_states.add(dfa_start)

    while worklist:
        current = worklist.pop()

        for symbol in nfa.alphabet:
            # From each NFA state in the current DFA state (= set of NFA states),
            # collect NFA states reachable via symbol, then take epsilon closure
            next_nfa_states = set()
            for nfa_state in current:
                targets = nfa.transition.get((nfa_state, symbol), set())
                next_nfa_states.update(targets)

            next_dfa_state = frozenset(
                nfa.epsilon_closure(next_nfa_states)
            )
            dfa_transition[(current, symbol)] = next_dfa_state

            if next_dfa_state not in dfa_states:
                dfa_states.add(next_dfa_state)
                worklist.append(next_dfa_state)

    # DFA accept states = DFA states that contain NFA accept states
    dfa_accept = {
        s for s in dfa_states
        if s & nfa.accept_states
    }

    # Make state names readable
    state_names = {}
    for i, s in enumerate(sorted(dfa_states, key=lambda x: sorted(x))):
        nfa_label = "{" + ",".join(sorted(s)) + "}"
        state_names[s] = f"S{i}_{nfa_label}"

    # Build DFA object
    named_states = {state_names[s] for s in dfa_states}
    named_transition = {
        (state_names[s], sym): state_names[t]
        for (s, sym), t in dfa_transition.items()
    }
    named_start = state_names[dfa_start]
    named_accept = {state_names[s] for s in dfa_accept}

    return DFA(
        states=named_states,
        alphabet=nfa.alphabet,
        transition=named_transition,
        start_state=named_start,
        accept_states=named_accept,
    )


# --- Example ---

# Convert the NFA that accepts strings ending with "abb" to a DFA
nfa = NFA(
    states={"q0", "q1", "q2", "q3"},
    alphabet={"a", "b"},
    transition={
        ("q0", "a"): {"q0", "q1"},
        ("q0", "b"): {"q0"},
        ("q1", "b"): {"q2"},
        ("q2", "b"): {"q3"},
    },
    start_state="q0",
    accept_states={"q3"},
)

dfa = nfa_to_dfa(nfa)

print("=== Converted DFA ===")
print(f"States: {dfa.states}")
print(f"Initial state: {dfa.start_state}")
print(f"Accept states: {dfa.accept_states}")
print(f"Transition function:")
for (state, symbol), target in sorted(dfa.transition.items()):
    print(f"  δ({state}, {symbol}) = {target}")

# Test with the converted DFA
for tc in ["abb", "aabb", "ab", "babb"]:
    print(dfa.trace(tc))
```

### 2.7 DFA Minimization

Multiple DFAs can accept the same language, but the DFA with the minimum number of states (minimal DFA) is unique up to isomorphism. The Hopcroft algorithm is well known as a minimization algorithm.

```
Basic approach to DFA minimization (equivalence class partitioning):

  1. Removal of unreachable states:
     Remove states unreachable from the initial state

  2. Merging equivalent states:
     If two states p, q are "indistinguishable", merge them

     Definition of distinguishable:
       States p, q are distinguishable ⟺
       There exists a string w such that δ*(p,w) ∈ F and δ*(q,w) ∉ F
       (or vice versa)

  3. Table-filling algorithm:
     a. Create a table of all state pairs (p,q)
     b. Mark pairs where p ∈ F, q ∉ F (or vice versa) as "distinguishable"
     c. For unmarked pairs (p,q):
        If (δ(p,a), δ(q,a)) is distinguishable for some symbol a,
        mark (p,q) as distinguishable too
     d. Repeat until no changes occur
     e. Unmarked pairs are equivalent → merge

  Example:
    Before minimization (5 states) → After minimization (3 states)
    Reduce state count by finding and merging equivalent states
```

---

## 3. Regular Languages and Regular Expressions

### 3.1 Definition and Characterization of Regular Languages

Regular languages are the most fundamental language class, and the following three characterizations are equivalent. This is known as Kleene's theorem.

```
Equivalent characterizations of regular languages (Kleene's theorem):

  The following three define the same language class:

  1. Languages accepted by DFAs
  2. Languages accepted by NFAs
  3. Languages represented by regular expressions

  That is:
    L is accepted by a DFA
    ⟺ L is accepted by an NFA
    ⟺ L is representable by a regular expression
```

### 3.2 Formal Definition of Regular Expressions

```
Inductive definition of regular expressions (over Σ):

  Base cases:
    1. ∅ is a regular expression (represents the empty language ∅)
    2. ε is a regular expression (represents {ε})
    3. For each a ∈ Σ, a is a regular expression (represents {a})

  Inductive step (when R₁, R₂ are regular expressions):
    4. (R₁ | R₂) is a regular expression (union: L(R₁) ∪ L(R₂)）
    5. (R₁ · R₂) is a regular expression (concatenation: L(R₁) · L(R₂)）
    6. (R₁*) is a regular expression (Kleene closure: L(R₁)*）

  Operator precedence: * > · > |
    Example: ab|c* is interpreted as (a·b)|(c*)

  Examples of regular expressions and corresponding languages:
  ┌────────────────┬─────────────────────────────┬──────────────────┐
  │ Regex          │ Language represented                     │ Examples         │
  ├────────────────┼─────────────────────────────┼──────────────────┤
  │ a*             │ {ε, a, aa, aaa, ...}        │ Zero or more a's │
  │ a⁺ = aa*       │ {a, aa, aaa, ...}           │ One or more a's  │
  │ (a|b)*         │ {a,b}all strings over           │ any sequence of a,b      │
  │ a*b*           │ Zero or more a's followed by zero or more b's    │ ε, a, b, aab    │
  │ (ab)*          │ Zero or more repetitions of ab        │ ε, ab, abab     │
  │ (a|b)*abb      │ Strings over {a,b} ending with abb   │ abb, aabb, babb │
  │ a(a|b)*a       │ Length >= 2, starts and ends with a   │ aa, aba, abba   │
  └────────────────┴─────────────────────────────┴──────────────────┘
```

### 3.3 Mutual Conversion Between Regular Expressions and Automata

```
Three directions of conversion:

  Regex ──Thompson's construction──→ NFA
      ↑                       │
      │                   Subset construction
  State elimination              │
      │                       ▼
      └────────────────── DFA

  1. Thompson Construction (Regular Expression → NFA):
     Inductively build an NFA following the structure of the regular expression

     Base cases:
       ε:  →(s)──ε──→((f))
       a:  →(s)──a──→((f))

     Union R₁|R₂:
                  ┌→ NFA(R₁) ─→┐
       →(s)──ε──┤              ├──ε──→((f))
                  └→ NFA(R₂) ─→┘

     Concatenation R₁·R₂:
       →(s)── NFA(R₁) ──ε── NFA(R₂) ──→((f))

     Closure R₁*:
                    ┌──────ε──────┐
                    ↓             │
       →(s)──ε──→ NFA(R₁) ──ε──→((f))
           │                      ↑
           └──────────ε──────────┘

  2. Subset Construction (NFA → DFA):
     → Detailed in Section 2.5

  3. State Elimination Method (DFA → Regular Expression):
     Remove DFA states one by one, generalizing edge labels to regular expressions
     Ultimately obtain the regular expression from initial state to accept state
```

### 3.4 Closure Properties of Regular Languages

The class of regular languages is closed under many operations (the result of the operation is also a regular language).

```
Closure properties of regular languages:

  When L₁, L₂ are regular languages, all of the following are also regular:

  ┌──────────────────────┬─────────────────┬──────────────────────┐
  │ Operation            │ Definition       │ Proof method            │
  ├──────────────────────┼─────────────────┼──────────────────────┤
  │ Union L₁ ∪ L₂      │ Belongs to either │ Parallel composition of NFAs       │
  │ Concatenation L₁ · L₂        │ Splittable into front and back   │ Series composition of NFAs       │
  │ Kleene closure L₁*     │ Zero or more concatenations    │ Add ε-transitions to NFA     │
  │ Complement Σ* \ L₁      │ Not in L₁     │ Swap accept/reject in DFA  │
  │ Intersection L₁ ∩ L₂    │ Belongs to both     │ Product automaton       │
  │ Set difference L₁ \ L₂      │ Belongs only to L₁   │ L₁ ∩ (Σ*\L₂)       │
  │ Reversal L₁ᴿ             │ Reverse the string   │ Reverse NFA edges  │
  │ Homomorphism          │ Symbol substitution│ Replace NFA edge labels   │
  └──────────────────────┴─────────────────┴──────────────────────┘

  Practical applications:
  - Closure under complement → "non-matching" patterns can also be expressed as regular expressions
  - Closure under intersection → AND combination of multiple conditions is possible
  - These closure properties make equivalence testing of regular languages decidable
```

### 3.5 Pumping Lemma for Regular Languages

The pumping lemma is an important tool for proving that a language is **not** regular.

```
Pumping Lemma for Regular Languages:

  If L is a regular language, then there exists a constant p >= 1 such that
  |w| ≥ p for all w ∈ L satisfying this,
  w can be split as w = xyz satisfying the following 3 conditions:

    (1) |y| > 0      (y is non-empty)
    (2) |xy| ≤ p     (xy is within the first p characters)
    (3) For all i ≥ 0, xy^i z ∈ L
        (repeating y any number of times remains in L)

  Intuitive understanding:
    When the DFA has p states, processing a string of length >= p
    must visit some state more than once by the pigeonhole principle.
    That repeated part (loop) corresponds to y.

    →(q₀)──x──→(qᵢ)──y──→(qᵢ)──z──→((qf))
                  ↑    ↓
                  └────┘  ← This loop repeated 0, 1, 2, ...
                             times is still accepted

  How to use the pumping lemma (proof by contradiction):
    To prove "L is not a regular language":
    1. Assume L is a regular language
    2. A pumping constant p exists
    3. Cleverly choose w ∈ L (|w| >= p)
    4. For any split w = xyz (satisfying conditions (1)(2)),
       show xy^i z ∉ L for some i
    5. Contradiction → L is not a regular language
```

**Example: Proof that L = {aⁿbⁿ | n ≥ 0} is not a regular language**

```
  Proof:
    L = {ε, ab, aabb, aaabbb, ...} Assume this is a regular language.

    1. A pumping constant p exists
    2. Choose w = aᵖbᵖ (|w| = 2p >= p, w ∈ L)
    3. Split w = xyz (|y| > 0, |xy| <= p)
       |xy| ≤ p so xy is within the first p characters of w
       → x and y consist only of a's
       x = aˢ, y = aᵗ（t > 0）, z = aᵖ⁻ˢ⁻ᵗbᵖ
    4. When i = 2:
       xy²z = aˢ · a²ᵗ · aᵖ⁻ˢ⁻ᵗbᵖ = aᵖ⁺ᵗbᵖ
       t > 0 so p+t > p → number of a's > number of b's
       → xy²z ∉ L
    5. Contradiction with the pumping lemma
       → L is not a regular language  □

  Significance of this result:
    Verifying matching parentheses (whether the number of "(" and ")" are equal)
    is impossible with finite automata → pushdown automata are needed
```

### 3.6 Practical Aspects of Regular Expressions

```
Implementation methods and performance of regular expression engines:

  ┌──────────────────┬──────────────┬────────────────┬─────────────┐
  │ Method           │ Time complexity│ Languages/Engines │ Features        │
  ├──────────────────┼──────────────┼────────────────┼─────────────┤
  │ DFA-based        │ O(n)         │ grep (partial)  │ Fast, memory-efficient│
  │ NFA simulation   │ O(n×m)       │ Go, Rust, RE2   │ Stable performance│
  │ Backtracking     │ O(2ⁿ) worst  │ Python, JS,     │ Backreferences OK │
  │                  │              │ Java, Ruby      │ ReDoS risk      │
  └──────────────────┴──────────────┴────────────────┴─────────────┘

  n: input string length  m: regex pattern length

  ReDoS（Regular Expression Denial of Service）:
    With backtracking, certain pattern and input combinations
    take exponential time → exploited for denial-of-service attacks

    Dangerous pattern example: (a+)+$
    Input "aaaaaaaaaaaaaaaaX" causes exponential backtracking

    Countermeasures:
    - Use NFA-based engines (RE2, Go's regexp)
    - Set timeouts
    - Limit input length
    - Review patterns (use non-greedy quantifiers, etc.)
```

---

## 4. Context-Free Grammars and Pushdown Automata

### 4.1 Definition of Context-Free Grammar (CFG)

Context-free grammars are a formal description method widely used for defining the syntax of programming languages.

```
Formal definition of Context-Free Grammar (CFG):
  G = (V, Σ, R, S)

  V : Finite set of variables (nonterminal symbols)
  Σ : Finite set of terminal symbols (V ∩ Σ = ∅)
  R : Finite set of production rules  A → α（A ∈ V, α ∈ (V ∪ Σ)*）
  S : Start symbol  S ∈ V

  Meaning of "context-free":
    In production rule A → α, the left side is only a single variable A
    → Regardless of the context (surrounding symbols) in which A appears,
      the same rule can be applied
    (Contrast: context-sensitive grammars allow the form αAβ → αγβ)
```

**Example: The language of matched parentheses**

```
  L = {aⁿbⁿ | n ≥ 0}  Context-free grammar for:

  G₁ = ({S}, {a, b}, R, S)
  R:
    S → aSb    (add one pair of a and b)
    S → ε      (Base case: empty string)

  Derivation example（w = "aaabbb"）:
    S ⟹ aSb ⟹ aaSbb ⟹ aaaSbbb ⟹ aaabbb

  Derivation tree (parse tree):
           S
         / | \
        a  S  b
         / | \
        a  S  b
         / | \
        a  S  b
           |
           ε
```

**Example: Grammar for arithmetic expressions**

```
  G₂ = ({E, T, F}, {+, *, (, ), id}, R, E)
  R:
    E → E + T | T        (Expression is addition of terms)
    T → T * F | F        (Term is multiplication of factors)
    F → ( E ) | id       (Factor is parenthesized expression or identifier)

  Derivation example（w = "id + id * id"）:
    E ⟹ E + T
      ⟹ T + T
      ⟹ F + T
      ⟹ id + T
      ⟹ id + T * F
      ⟹ id + F * F
      ⟹ id + id * F
      ⟹ id + id * id

  Derivation tree:
            E
          / | \
         E  +  T
         |   / | \
         T  T  *  F
         |  |     |
         F  F    id
         |  |
        id id

  This derivation tree reflects operator precedence:
  * has higher precedence than + → * node is positioned lower in the tree
```

### 4.2 Ambiguity

```
Ambiguous grammar:

  When two or more different derivation trees exist for a string w,
  the grammar is called "ambiguous".

  Example: Ambiguous expression grammar
    E → E + E | E * E | ( E ) | id

  The string "id + id * id" has two derivation trees:

  Derivation tree 1 (+ first):        Derivation tree 2 (* first):
        E                         E
      / | \                     / | \
     E  +  E                   E  *  E
     |   / | \               / | \   |
    id  E  *  E             E  +  E  id
        |     |             |     |
       id    id            id    id

  Ambiguity is fatal in programming languages:
  - "2 + 3 * 4" could be either 20 or 14
  - Solutions:
    a. Rewrite the grammar to eliminate ambiguity (as in G₂ above)
    b. Add precedence and associativity rules to the parser

  Important theorem: Whether a given CFG is ambiguous is generally undecidable
```

### 4.3 Chomsky Normal Form and CYK Algorithm

```
Chomsky Normal Form (CNF):

  All production rules are in one of the following forms:
    A → BC    (right side is two variables)
    A → a     (right side is one terminal symbol)
    S → ε     (only the start symbol can produce ε)

  Any CFG can be converted to CNF (except for the empty language)

  Advantages of CNF:
    - String length increases by at most 1 at each derivation step
    - Derivation of a string of length n takes exactly 2n-1 steps
    - Enables efficient parsing via the CYK algorithm

CYK Algorithm (Cocke-Younger-Kasami):

  O(n³) parsing algorithm for CNF grammars

  Input: CNF grammar G and string w = w₁w₂...wₙ
  Output: Whether w ∈ L(G)

  Method: Dynamic programming
    Table T[i][j] = {A ∈ V | A ⟹* wᵢwᵢ₊₁...wⱼ}

    Base cases: T[i][i] = {A | A → wᵢ ∈ R}
    Induction: T[i][j] = ∪{A | A → BC ∈ R,
                          B ∈ T[i][k], C ∈ T[k+1][j],
                          i ≤ k < j}

    Decision: If S ∈ T[1][n], then w ∈ L(G)

  Example: G = ({S, A, B}, {a, b}, R, S)
      R: S → AB, A → a, B → b, S → a

      Input w = "ab"

      T[1][1] = {A | A → a} = {A, S}
      T[2][2] = {B | B → b} = {B}
      T[1][2] = {X | X → YZ, Y ∈ T[1][1], Z ∈ T[2][2]}
              = {S}  （S → AB, A ∈ T[1][1], B ∈ T[2][2]）

      S ∈ T[1][2] → "ab" ∈ L(G) ✓
```

### 4.4 Pushdown Automaton (PDA)

```
Formal definition of PDA:
  P = (Q, Σ, Γ, δ, q₀, Z₀, F)

  Q  : Finite set of states
  Σ  : Input alphabet
  Γ  : Stack alphabet
  δ  : Transition function  Q × (Σ ∪ {ε}) × Γ → P(Q × Γ*)
       (current state, input symbol, stack top) → (next state, stack operation)
  q₀ : Initial state
  Z₀ : Initial stack symbol  Z₀ ∈ Γ
  F  : Set of accept states

  PDA = Finite automaton + Stack (unbounded memory)

  Differences from finite automata:
  ┌──────────────────────┬─────────────────┬──────────────────┐
  │ Feature              │ Finite Automaton │ PDA              │
  ├──────────────────────┼─────────────────┼──────────────────┤
  │ Memory device        │ None (state only)│ Stack (LIFO)  │
  │ Recognized lang class│ Regular languages│ Context-free languages      │
  │ Typical use          │ Lexical analysis │ Parsing          │
  │ Matching parens check│ Impossible       │ Possible              │
  └──────────────────────┴─────────────────┴──────────────────┘
```

**Example: PDA that accepts L = {aⁿbⁿ | n ≥ 0}**

```
  PDA P₁:
    States: {q₀, q₁, q₂}
    Input alphabet: {a, b}
    Stack alphabet: {Z₀, A}
    Initial state: q₀
    Initial stack symbol: Z₀
    Accept states: {q₂}

  Transition rules:
    δ(q₀, a, Z₀) = {(q₀, AZ₀)}    Read a and push A
    δ(q₀, a, A)  = {(q₀, AA)}      Read a and push A
    δ(q₀, b, A)  = {(q₁, ε)}       Read b and pop A
    δ(q₁, b, A)  = {(q₁, ε)}       Read b and pop A
    δ(q₁, ε, Z₀) = {(q₂, Z₀)}     Stack is empty (only Z₀) → Accept

  Trace (input "aabb"):
    State   Remaining   Stack
    q₀      aabb        Z₀
    q₀      abb         AZ₀          ← Read a, push A
    q₀      bb          AAZ₀         ← Read a, push A
    q₁      b           AZ₀          ← Read b, pop A
    q₁      ε           Z₀           ← Read b, pop A
    q₂      ε           Z₀           ← ε-transition to accept state → Accept\!

  Trace (input "aab", rejected example):
    State   Remaining   Stack
    q₀      aab         Z₀
    q₀      ab          AZ₀
    q₀      b           AAZ₀
    q₁      ε           AZ₀          ← A remains on stack
    → Cannot transition to q₂ → Reject
```

### 4.5 Equivalence of CFG and PDA

```
Important theorem:
  Language L is a context-free language
  ⟺ There exists a PDA that accepts L
  ⟺ There exists a CFG that generates L

  Conversion directions:
    CFG → PDA: Convert each production rule to stack operations
    PDA → CFG: Simulate PDA computation with production rules

  Practical significance:
  - A compiler's parser is a type of PDA
  - BNF (Backus-Naur Form) is an alternative notation for CFG
  - Parser generators such as yacc/bison, ANTLR
    automatically generate PDAs from CFG definitions
```

### 4.6 Pumping Lemma for Context-Free Languages

```
Pumping Lemma for Context-Free Languages:

  If L is a context-free language, there exists a constant p >= 1 such that
  |w| ≥ p for all w ∈ L satisfying this,
  w can be split as w = uvxyz satisfying the following 3 conditions:

    (1) |vy| > 0        (at least one of v and y is non-empty)
    (2) |vxy| ≤ p       (the central part has length at most p)
    (3) For all i ≥ 0, uv^i xy^i z ∈ L

  Differences from the regular language version:
    Regular: 3-part split xyz, pump y
    Context-free: 5-part split uvxyz, pump v and y simultaneously

  Intuitive understanding:
    When the derivation tree is sufficiently large, the same variable appears twice on a derivation path.
    The part between them can be repeatedly expanded (pumped).

             S
            /|\
           u  A  z
             /|\
            v  A  y
              /|\
             x

    A → ... A ... part can be repeated 0, 1, 2, ... times

  Example: Proof that L = {aⁿbⁿcⁿ | n ≥ 0} is not a context-free language

    1. Assume L is a context-free language and take constant p
    2. Choose w = aᵖbᵖcᵖ (|w| = 3p >= p)
    3. Split w = uvxyz (|vy| > 0, |vxy| <= p)
    4. Since |vxy| <= p, vxy contains at most 2 types of
       characters among a, b, c
    5. When i = 2, uv²xy²z
       increases the count of at most 2 types while the rest stays the same
       → the counts of a, b, c become unequal → uv²xy²z ∉ L
    6. Contradiction → L is not a context-free language  □
```

---

## 5. Chomsky Hierarchy

### 5.1 Overview of the Chomsky Hierarchy

The Chomsky hierarchy is a hierarchical classification system of formal grammars and formal languages proposed by Noam Chomsky in 1956.

```
Chomsky Hierarchy:

  ┌─────────┬───────────────┬───────────────────┬──────────────────┐
  │ Type    │ Grammar        │ Recognizing machine│ Language class       │
  ├─────────┼───────────────┼───────────────────┼──────────────────┤
  │ Type 0  │ Unrestricted   │ Turing Machine     │ Recursively enumerable   │
  │ Type 1  │ Context-sensitive│ Linear Bounded Auto│ Context-sensitive     │
  │ Type 2  │ Context-free   │ Pushdown           │ Context-free     │
  │         │                │ Automaton          │                  │
  │ Type 3  │ Regular grammar│ Finite Automaton   │ Regular languages         │
  └─────────┴───────────────┴───────────────────┴──────────────────┘

  Inclusion relation (proper inclusion):
    Regular ⊂ Context-free ⊂ Context-sensitive ⊂ Recursively enumerable

  Examples of languages at each level boundary:
    Regular: a*b* (recognizable by finite automata)
    Context-free but not regular: {aⁿbⁿ | n ≥ 0}
    Context-sensitive but not context-free: {aⁿbⁿcⁿ | n ≥ 0}
    Recursively enumerable but not context-sensitive: set of encodings of halting TMs
    Not even recursively enumerable: complement of the halting problem
```

### 5.2 Forms of Grammar Rules for Each Type

```
Classification by grammar rule constraints:

  Type 0 (Unrestricted grammar):
    Rules: α → β（α ∈ (V∪Σ)⁺, β ∈ (V∪Σ)*）
    Constraints: None (left side can include terminal symbols)

  Type 1 (Context-sensitive grammar):
    Rules: αAβ → αγβ（A ∈ V, α,β ∈ (V∪Σ)*, γ ∈ (V∪Σ)⁺）
    Constraints: |α| <= |β| (rules that shorten strings are prohibited)
    Meaning: Rewriting A depends on the surrounding context α, β

  Type 2 (Context-free grammar):
    Rules: A → α（A ∈ V, α ∈ (V∪Σ)*）
    Constraints: Left side is a single variable only
    Meaning: The same rule can be applied to A in any context

  Type 3 (Regular grammar):
    Right-linear grammar: A → aB | a | ε（A,B ∈ V, a ∈ Σ）
    Left-linear grammar: A → Ba | a | ε
    Constraints: At most one variable on the right side, placed at the end

  Strength of constraints: Type 3 > Type 2 > Type 1 > Type 0
  Expressive power:     Type 3 < Type 2 < Type 1 < Type 0
```

### 5.3 Decidability at Each Level

```
Decision problems for each language class:

  ┌─────────────────────┬────────┬──────────┬──────────┬────────┐
  │ Problem             │ Regular│ Context-free│ Context-sensitive│ Type 0 │
  ├─────────────────────┼────────┼──────────┼──────────┼────────┤
  │ Membership problem  │ ○ O(n) │ ○ O(n³)  │ ○        │ ×     │
  │ w ∈ L?              │        │ CYK      │ PSPACE-   │ Semi-decidable │
  ├─────────────────────┼────────┼──────────┼──────────┼────────┤
  │ Emptiness problem   │ ○      │ ○        │ ×        │ ×     │
  │ L = ∅?              │        │          │          │        │
  ├─────────────────────┼────────┼──────────┼──────────┼────────┤
  │ Equivalence problem │ ○      │ ×        │ ×        │ ×     │
  │ L₁ = L₂?            │        │ Undecidable │ Undecidable │ Undecidable│
  ├─────────────────────┼────────┼──────────┼──────────┼────────┤
  │ Inclusion problem   │ ○      │ ×        │ ×        │ ×     │
  │ L₁ ⊆ L₂?            │        │          │          │        │
  ├─────────────────────┼────────┼──────────┼──────────┼────────┤
  │ Finiteness problem  │ ○      │ ○        │ ×        │ ×     │
  │ |L| < ∞?            │        │          │          │        │
  └─────────────────────┴────────┴──────────┴──────────┴────────┘

  ○: Decidable  ×: Undecidable

  Practical significance:
  - Regular languages are the easiest to handle (almost all problems are decidable)
  - For context-free languages, membership and emptiness are solvable but equivalence is undecidable
  - Even the membership problem for Type 0 is undecidable (reduces to the halting problem)
```

### 5.4 Chomsky Hierarchy and Practical Correspondence

```
Practical correspondence of the Chomsky hierarchy:

  ┌─────────┬──────────────────────────────────────────────────┐
  │ Level   │ Practical correspondence                                      │
  ├─────────┼──────────────────────────────────────────────────┤
  │ Type 3  │ - Pattern matching with regular expressions                 │
  │ Regular │ - Lexical analyzers (lexers/tokenizers)             │
  │         │ - Input validation (email addresses, phone numbers, etc.)   │
  │         │ - Network protocol state management                  │
  │         │ - Configuration file keyword recognition                      │
  ├─────────┼──────────────────────────────────────────────────┤
  │ Type 2  │ - Programming language syntax definitions (BNF/EBNF)          │
  │ CF      │ - Parsers (LL, LR, LALR, PEG)       │
  │         │ - Structural analysis of XML/HTML/JSON                          │
  │         │ - Mathematical expression parsers                                      │
  │         │ - Matching parenthesis checking                                │
  ├─────────┼──────────────────────────────────────────────────┤
  │ Type 1  │ - Type checking (variable use depends on declaration)              │
  │ CS      │ - Scope analysis                                      │
  │         │ - Context-dependent parsing due to C language typedef           │
  │         │ - Some natural language syntax                              │
  ├─────────┼──────────────────────────────────────────────────┤
  │ Type 0  │ - General-purpose programming (Turing-complete computation)      │
  │ Unrest. │ - Semantic analysis in general                                      │
  │         │ - Halting problem → The wall of undecidability          │
  └─────────┴──────────────────────────────────────────────────┘
```

### 5.5 Application of the Chomsky Hierarchy in Compilers

```
Compiler frontend and the Chomsky hierarchy:

  Source code
    │
    ▼ ┌──────────────────────────────────────────────────────┐
  Lexical analysis (Lexer) ← Type 3: Regular languages (DFA)              │
    │                  Tokens are defined by regular expressions               │
    │                  Example: ID = [a-zA-Z_][a-zA-Z0-9_]*       │
    │                  Example: NUM = [0-9]+(\.[0-9]+)?           │
    ▼                                                        │
  Token stream                                                 │
    │                                                        │
    ▼ ┌──────────────────────────────────────────────────────┤
  Parsing (Parser) ← Type 2: Context-free languages (PDA)          │
    │                  Syntax rules are defined with BNF/EBNF              │
    │                  Example: if_stmt → 'if' expr 'then' stmt   │
    │                  Methods such as LL(k), LR(k), LALR(1)        │
    ▼                                                        │
  Parse tree (AST)                                              │
    │                                                        │
    ▼ ┌──────────────────────────────────────────────────────┤
  Semantic analysis ← Type 1 and above: Context-sensitive                             │
    │        Type checking, scope analysis,                       │
    │        overload resolution, etc.                             │
    ▼                                                        │
  Annotated AST                                                │
    │                                                        │
    ▼                                                        │
  Intermediate code generation → Optimization → Code generation                       │
  └──────────────────────────────────────────────────────────┘

  Example: "int x = 3 + 5;" Processing of

  Lexical analysis (Type 3):
    "int" → KW_INT
    "x"   → ID("x")
    "="   → ASSIGN
    "3"   → INT(3)
    "+"   → PLUS
    "5"   → INT(5)
    ";"   → SEMICOLON

  Parsing (Type 2):
    declaration
    ├── type: int
    ├── name: x
    └── init_expr
        └── binary_op: +
            ├── left: 3
            └── right: 5

  Semantic analysis (Type 1 and above):
    - x has type int
    - 3 + 5 is an int operation → type consistency ✓
    - x is undeclared in this scope → create new binding
```

---

## 6. Implementation of a Regular Expression Engine

### 6.1 NFA-Based Regular Expression Engine via Thompson Construction

The following is an implementation of an engine that converts regular expressions to NFAs using Thompson construction and performs matching via NFA simulation.

```python
"""
Regular Expression Engine (Thompson Construction + NFA Simulation)

Supported regular expression syntax:
  - Concatenation:   ab
  - Alternation:   a|b
  - Closure:   a*
  - One or more: a+
  - Zero or one: a?
  - Parentheses:   (a|b)*
  - Any character: .

Internally converts to NFA and matches the input string in O(n*m).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class NFAState:
    """NFA state node"""
    id: int
    is_accept: bool = False
    # Epsilon transition targets
    epsilon: List["NFAState"] = field(default_factory=list)
    # Character transitions: {character: [target states]}
    transitions: dict = field(default_factory=dict)


class NFAFragment:
    """NFA fragment (pair of start state and accept state)"""
    def __init__(self, start: NFAState, accept: NFAState):
        self.start = start
        self.accept = accept


class RegexEngine:
    """Thompson construction-based regular expression engine"""

    def __init__(self):
        self._state_counter = 0

    def _new_state(self, is_accept: bool = False) -> NFAState:
        """Generate a new NFA state."""
        state = NFAState(id=self._state_counter, is_accept=is_accept)
        self._state_counter += 1
        return state

    # --- Parser (Regular Expression → Syntax Tree) ---

    def _parse(self, pattern: str) -> "NFAFragment":
        """Parse a regular expression and build an NFA."""
        self._pos = 0
        self._pattern = pattern
        nfa = self._parse_expr()
        nfa.accept.is_accept = True
        return nfa

    def _parse_expr(self) -> NFAFragment:
        """Expression: term ('|' term)*"""
        left = self._parse_term()
        while self._pos < len(self._pattern) and self._peek() == "|":
            self._consume("|")
            right = self._parse_term()
            left = self._build_union(left, right)
        return left

    def _parse_term(self) -> NFAFragment:
        """Term: factor*"""
        # Empty term (corresponding to ε)
        if (self._pos >= len(self._pattern)
                or self._peek() in ("|", ")")):
            s = self._new_state()
            return NFAFragment(s, s)

        left = self._parse_factor()
        while (self._pos < len(self._pattern)
               and self._peek() not in ("|", ")")):
            right = self._parse_factor()
            left = self._build_concat(left, right)
        return left

    def _parse_factor(self) -> NFAFragment:
        """Factor: atom ('*' | '+' | '?')?"""
        atom = self._parse_atom()
        if self._pos < len(self._pattern):
            if self._peek() == "*":
                self._consume("*")
                return self._build_star(atom)
            elif self._peek() == "+":
                self._consume("+")
                return self._build_plus(atom)
            elif self._peek() == "?":
                self._consume("?")
                return self._build_optional(atom)
        return atom

    def _parse_atom(self) -> NFAFragment:
        """Atom: '(' expr ')' | '.' | character"""
        if self._peek() == "(":
            self._consume("(")
            nfa = self._parse_expr()
            self._consume(")")
            return nfa
        elif self._peek() == ".":
            self._consume(".")
            return self._build_dot()
        else:
            ch = self._peek()
            self._pos += 1
            return self._build_char(ch)

    def _peek(self) -> str:
        return self._pattern[self._pos]

    def _consume(self, expected: str) -> None:
        if self._pos < len(self._pattern) and self._peek() == expected:
            self._pos += 1
        else:
            raise ValueError(
                f"Expected '{expected}' at position {self._pos}"
            )

    # --- NFA Construction (Thompson Construction) ---

    def _build_char(self, ch: str) -> NFAFragment:
        """Build an NFA fragment for a character literal."""
        start = self._new_state()
        accept = self._new_state()
        start.transitions[ch] = [accept]
        return NFAFragment(start, accept)

    def _build_dot(self) -> NFAFragment:
        """Build an NFA fragment for any-character '.'."""
        start = self._new_state()
        accept = self._new_state()
        # Use special key "ANY" to represent any-character match
        start.transitions["ANY"] = [accept]
        return NFAFragment(start, accept)

    def _build_concat(
        self, left: NFAFragment, right: NFAFragment
    ) -> NFAFragment:
        """Concatenation: epsilon transition from left accept state to right start state."""
        left.accept.epsilon.append(right.start)
        left.accept.is_accept = False
        return NFAFragment(left.start, right.accept)

    def _build_union(
        self, left: NFAFragment, right: NFAFragment
    ) -> NFAFragment:
        """Alternation (|): ε-transitions from new start state to left and right."""
        start = self._new_state()
        accept = self._new_state()
        start.epsilon.extend([left.start, right.start])
        left.accept.epsilon.append(accept)
        left.accept.is_accept = False
        right.accept.epsilon.append(accept)
        right.accept.is_accept = False
        return NFAFragment(start, accept)

    def _build_star(self, inner: NFAFragment) -> NFAFragment:
        """Kleene closure (*): Zero or more repetitions."""
        start = self._new_state()
        accept = self._new_state()
        start.epsilon.extend([inner.start, accept])
        inner.accept.epsilon.extend([inner.start, accept])
        inner.accept.is_accept = False
        return NFAFragment(start, accept)

    def _build_plus(self, inner: NFAFragment) -> NFAFragment:
        """One or more repetitions (+): inner · inner*"""
        start = self._new_state()
        accept = self._new_state()
        start.epsilon.append(inner.start)
        inner.accept.epsilon.extend([inner.start, accept])
        inner.accept.is_accept = False
        return NFAFragment(start, accept)

    def _build_optional(self, inner: NFAFragment) -> NFAFragment:
        """Zero or one (?): inner | ε"""
        start = self._new_state()
        accept = self._new_state()
        start.epsilon.extend([inner.start, accept])
        inner.accept.epsilon.append(accept)
        inner.accept.is_accept = False
        return NFAFragment(start, accept)

    # --- NFA Simulation ---

    def _epsilon_closure(self, states: Set[NFAState]) -> Set[NFAState]:
        """Compute epsilon closure."""
        closure = set(states)
        stack = list(states)
        while stack:
            state = stack.pop()
            for next_state in state.epsilon:
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)
        return closure

    def _step(
        self, states: Set[NFAState], ch: str
    ) -> Set[NFAState]:
        """Read one character, transition, and return epsilon closure."""
        next_states = set()
        for state in states:
            # Regular character transition
            if ch in state.transitions:
                next_states.update(state.transitions[ch])
            # Any-character match
            if "ANY" in state.transitions:
                next_states.update(state.transitions["ANY"])
        return self._epsilon_closure(next_states)

    def match(self, pattern: str, text: str) -> bool:
        """
        Determine whether the regex pattern matches the entire text.

        Args:
            pattern: Regular expression pattern
            text: String to match against

        Returns:
            True if it matches
        """
        self._state_counter = 0
        nfa = self._parse(pattern)

        current = self._epsilon_closure({nfa.start})
        for ch in text:
            current = self._step(current, ch)
            if not current:
                return False

        return any(s.is_accept for s in current)

    def search(self, pattern: str, text: str) -> Optional[str]:
        """
        Return the first matching substring in the text.

        Args:
            pattern: Regular expression pattern
            text: String to search

        Returns:
            First matching substring, or None if not found
        """
        self._state_counter = 0
        nfa = self._parse(pattern)

        for start_pos in range(len(text)):
            current = self._epsilon_closure({nfa.start})
            # Case where immediately accepted at start position (NFA that accepts ε)
            if any(s.is_accept for s in current):
                return ""
            for end_pos in range(start_pos, len(text)):
                current = self._step(current, text[end_pos])
                if not current:
                    break
                if any(s.is_accept for s in current):
                    return text[start_pos:end_pos + 1]

        return None


# --- Example Usage ---

engine = RegexEngine()

# Basic matching
test_patterns = [
    ("abc",      "abc",    True),
    ("a|b",      "a",      True),
    ("a|b",      "b",      True),
    ("a|b",      "c",      False),
    ("a*",       "",       True),
    ("a*",       "aaa",    True),
    ("a+",       "",       False),
    ("a+",       "aaa",    True),
    ("ab?c",     "ac",     True),
    ("ab?c",     "abc",    True),
    ("(a|b)*abb", "aabb",  True),
    ("(a|b)*abb", "ab",    False),
    (".+",       "hello",  True),
]

print("=== Regular Expression Engine Test ===")
for pattern, text, expected in test_patterns:
    result = engine.match(pattern, text)
    status = "OK" if result == expected else "NG"
    print(f"  [{status}] match('{pattern}', '{text}') = {result}")

# Search
print("\n=== Substring Search ===")
found = engine.search("(a|b)+c", "xxxabcyyy")
print(f"  search('(a|b)+c', 'xxxabcyyy') = '{found}'")

# Output:
# === Regular Expression Engine Test ===
#   [OK] match('abc', 'abc') = True
#   [OK] match('a|b', 'a') = True
#   [OK] match('a|b', 'b') = True
#   [OK] match('a|b', 'c') = False
#   [OK] match('a*', '') = True
#   [OK] match('a*', 'aaa') = True
#   [OK] match('a+', '') = False
#   [OK] match('a+', 'aaa') = True
#   [OK] match('ab?c', 'ac') = True
#   [OK] match('ab?c', 'abc') = True
#   [OK] match('(a|b)*abb', 'aabb') = True
#   [OK] match('(a|b)*abb', 'ab') = False
#   [OK] match('.+', 'hello') = True
#
# === Substring Search ===
#   search('(a|b)+c', 'xxxabcyyy') = 'abc'
```

### 6.2 Comparison of Implementation Methods and Performance Characteristics

```
Comparison of regular expression engine implementation methods:

  ┌────────────────────┬────────────────┬──────────────────────┐
  │ Method              │ Time complexity│ Space complexity           │
  ├────────────────────┼────────────────┼──────────────────────┤
  │ NFA simulation      │ O(n × m)       │ O(m)                 │
  │ (above impl)        │ n: input len   │ m: pattern states   │
  │                     │ m: pattern len │                      │
  ├────────────────────┼────────────────┼──────────────────────┤
  │ DFA post-conversion │ O(n)           │ O(2^m) worst          │
  │                     │ Fastest match  │ State explosion risk     │
  ├────────────────────┼────────────────┼──────────────────────┤
  │ Lazy DFA construct  │ O(n × m) avg   │ O(m) to O(2^m)      │
  │ (RE2 method)        │ Cache-based    │ Cache size limited  │
  ├────────────────────┼────────────────┼──────────────────────┤
  │ Backtracking        │ O(2^n) worst   │ O(n) stack        │
  │ (Perl/Python method) │ Backrefs OK    │ Recursive               │
  └────────────────────┴────────────────┴──────────────────────┘

  Selection guidelines:
  - No backreferences needed, safety-focused → NFA simulation (Go, Rust)
  - Performance priority with small patterns → DFA conversion
  - Backreferences needed → Backtracking (but beware of ReDoS)
  - Balance-focused → Lazy DFA construction (RE2)
```

---

## 7. Practical Applications and Case Studies

### 7.1 Implementation of a Lexical Analyzer (Lexer)

The lexical analyzer, the first stage of a compiler or interpreter, is a direct application of DFA. The following is a DFA-based implementation of a tokenizer for a simple programming language.

```python
"""
DFA-Based Lexical Analyzer (Lexer / Tokenizer)

Decomposes source code of a simple programming language into a token stream.
Each token is recognized based on DFA principles.

Supported tokens:
  - Keywords: if, else, while, return, int, float
  - Identifiers: [a-zA-Z_][a-zA-Z0-9_]*
  - Integer literals: [0-9]+
  - Floating-point literals: [0-9]+\.[0-9]+
  - Operators: +, -, *, /, =, ==, !=, <, >, <=, >=
  - Delimiters: (, ), {, }, ;, ,
  - Whitespace and comments are skipped
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class TokenType(Enum):
    """Token types"""
    # Keywords
    KW_IF = auto()
    KW_ELSE = auto()
    KW_WHILE = auto()
    KW_RETURN = auto()
    KW_INT = auto()
    KW_FLOAT = auto()
    # Literals
    INT_LITERAL = auto()
    FLOAT_LITERAL = auto()
    # Identifiers
    IDENTIFIER = auto()
    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    ASSIGN = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMICOLON = auto()
    COMMA = auto()
    # Special
    EOF = auto()
    ERROR = auto()


@dataclass
class Token:
    """Token"""
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, '{self.value}', L{self.line}:C{self.column})"


# Keywords table
KEYWORDS = {
    "if": TokenType.KW_IF,
    "else": TokenType.KW_ELSE,
    "while": TokenType.KW_WHILE,
    "return": TokenType.KW_RETURN,
    "int": TokenType.KW_INT,
    "float": TokenType.KW_FLOAT,
}

# Single character tokens
SINGLE_CHAR_TOKENS = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    ";": TokenType.SEMICOLON,
    ",": TokenType.COMMA,
}


class Lexer:
    """DFA-based lexical analyzer"""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1

    def _peek(self) -> Optional[str]:
        """Return the character at current position (without consuming)."""
        if self.pos < len(self.source):
            return self.source[self.pos]
        return None

    def _advance(self) -> str:
        """Consume and return the character at current position."""
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _skip_whitespace_and_comments(self) -> None:
        """Skip whitespace and comments."""
        while self.pos < len(self.source):
            ch = self._peek()
            if ch in (" ", "\t", "\n", "\r"):
                self._advance()
            elif ch == "/" and self.pos + 1 < len(self.source):
                if self.source[self.pos + 1] == "/":
                    # Line comment
                    while self.pos < len(self.source) and self._peek() != "\n":
                        self._advance()
                else:
                    break
            else:
                break

    def _read_number(self) -> Token:
        """
        DFA for reading numeric literals:

        State transitions:
          START --[0-9]--> INTEGER --[0-9]--> INTEGER
                                   --[.]--> DOT
          DOT --[0-9]--> FLOAT --[0-9]--> FLOAT

        Accept states: INTEGER, FLOAT
        """
        start_col = self.column
        value = ""

        # State: INTEGER (integer part)
        while self.pos < len(self.source) and self._peek().isdigit():
            value += self._advance()

        # Decimal point check
        if (self.pos < len(self.source)
                and self._peek() == "."
                and self.pos + 1 < len(self.source)
                and self.source[self.pos + 1].isdigit()):
            value += self._advance()  # '.'
            # State: FLOAT (decimal part)
            while self.pos < len(self.source) and self._peek().isdigit():
                value += self._advance()
            return Token(TokenType.FLOAT_LITERAL, value, self.line, start_col)

        return Token(TokenType.INT_LITERAL, value, self.line, start_col)

    def _read_identifier_or_keyword(self) -> Token:
        """
        DFA for reading identifiers/keywords:

        State transitions:
          START --[a-zA-Z_]--> IDENT --[a-zA-Z0-9_]--> IDENT

        Accept state: IDENT
        After acceptance, consult keyword table to determine type
        """
        start_col = self.column
        value = ""

        while (self.pos < len(self.source)
               and (self._peek().isalnum() or self._peek() == "_")):
            value += self._advance()

        token_type = KEYWORDS.get(value, TokenType.IDENTIFIER)
        return Token(token_type, value, self.line, start_col)

    def next_token(self) -> Token:
        """Returns the next token."""
        self._skip_whitespace_and_comments()

        if self.pos >= len(self.source):
            return Token(TokenType.EOF, "", self.line, self.column)

        start_col = self.column
        ch = self._peek()

        # Numeric literal
        if ch.isdigit():
            return self._read_number()

        # Identifiers or keywords
        if ch.isalpha() or ch == "_":
            return self._read_identifier_or_keyword()

        # Processing of two-character operators
        if ch == "=" and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
            self._advance()
            self._advance()
            return Token(TokenType.EQ, "==", self.line, start_col)
        if ch == "!" and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
            self._advance()
            self._advance()
            return Token(TokenType.NEQ, "!=", self.line, start_col)
        if ch == "<" and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
            self._advance()
            self._advance()
            return Token(TokenType.LE, "<=", self.line, start_col)
        if ch == ">" and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
            self._advance()
            self._advance()
            return Token(TokenType.GE, ">=", self.line, start_col)

        # Single character operators
        if ch == "=":
            self._advance()
            return Token(TokenType.ASSIGN, "=", self.line, start_col)
        if ch == "<":
            self._advance()
            return Token(TokenType.LT, "<", self.line, start_col)
        if ch == ">":
            self._advance()
            return Token(TokenType.GT, ">", self.line, start_col)

        # Single character token
        if ch in SINGLE_CHAR_TOKENS:
            self._advance()
            return Token(SINGLE_CHAR_TOKENS[ch], ch, self.line, start_col)

        # Invalid character
        self._advance()
        return Token(TokenType.ERROR, ch, self.line, start_col)

    def tokenize(self) -> List[Token]:
        """Converts the entire source code into a token stream."""
        tokens = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens


# --- Example Usage ---

source_code = """
// Fibonacci sequence
int fib(int n) {
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}
"""

lexer = Lexer(source_code)
tokens = lexer.tokenize()

print("=== Lexical Analysis Results ===")
for token in tokens:
    print(f"  {token}")

# Output:
# === Lexical Analysis Results ===
#   Token(KW_INT, 'int', L3:C1)
#   Token(IDENTIFIER, 'fib', L3:C5)
#   Token(LPAREN, '(', L3:C8)
#   Token(KW_INT, 'int', L3:C9)
#   Token(IDENTIFIER, 'n', L3:C13)
#   Token(RPAREN, ')', L3:C14)
#   Token(LBRACE, '{', L3:C16)
#   Token(KW_IF, 'if', L4:C5)
#   Token(LPAREN, '(', L4:C8)
#   Token(IDENTIFIER, 'n', L4:C9)
#   Token(LE, '<=', L4:C11)
#   Token(INT_LITERAL, '1', L4:C14)
#   Token(RPAREN, ')', L4:C15)
#   Token(LBRACE, '{', L4:C17)
#   Token(KW_RETURN, 'return', L5:C9)
#   Token(IDENTIFIER, 'n', L5:C16)
#   Token(SEMICOLON, ';', L5:C17)
#   Token(RBRACE, '}', L6:C5)
#   Token(KW_RETURN, 'return', L7:C5)
#   Token(IDENTIFIER, 'fib', L7:C12)
#   Token(LPAREN, '(', L7:C15)
#   Token(IDENTIFIER, 'n', L7:C16)
#   Token(MINUS, '-', L7:C18)
#   Token(INT_LITERAL, '1', L7:C20)
#   Token(RPAREN, ')', L7:C21)
#   Token(PLUS, '+', L7:C23)
#   Token(IDENTIFIER, 'fib', L7:C25)
#   Token(LPAREN, '(', L7:C28)
#   Token(IDENTIFIER, 'n', L7:C29)
#   Token(MINUS, '-', L7:C31)
#   Token(INT_LITERAL, '2', L7:C33)
#   Token(RPAREN, ')', L7:C34)
#   Token(SEMICOLON, ';', L7:C35)
#   Token(RBRACE, '}', L8:C1)
#   Token(EOF, '', L9:C1)
```

### 7.2 Application to Protocol Verification

Finite automata are well-suited for modeling state transitions of communication protocols.

```
TCP Connection Management State Transitions (Simplified):

  States: {CLOSED, LISTEN, SYN_SENT, SYN_RCVD, ESTABLISHED,
         FIN_WAIT_1, FIN_WAIT_2, CLOSE_WAIT, LAST_ACK, TIME_WAIT}

  Key transitions:

    ┌─────────┐   passive open    ┌─────────┐
    │ CLOSED  │──────────────────→│ LISTEN  │
    └─────────┘                   └─────────┘
         │ active open                │ recv SYN
         │ send SYN                   │ send SYN+ACK
         ▼                            ▼
    ┌──────────┐                 ┌──────────┐
    │ SYN_SENT │                 │ SYN_RCVD │
    └──────────┘                 └──────────┘
         │ recv SYN+ACK              │ recv ACK
         │ send ACK                  │
         ▼                           ▼
    ┌──────────────────────────────────────┐
    │          ESTABLISHED                  │
    │    (State where data communication is possible)          │
    └──────────────────────────────────────┘
         │ close / send FIN          │ recv FIN
         ▼                           │ send ACK
    ┌────────────┐              ┌────────────┐
    │ FIN_WAIT_1 │              │ CLOSE_WAIT │
    └────────────┘              └────────────┘
         │ recv ACK                  │ close
         ▼                           │ send FIN
    ┌────────────┐              ┌────────────┐
    │ FIN_WAIT_2 │              │  LAST_ACK  │
    └────────────┘              └────────────┘
         │ recv FIN                  │ recv ACK
         │ send ACK                  ▼
         ▼                      ┌─────────┐
    ┌────────────┐              │ CLOSED  │
    │ TIME_WAIT  │──timeout──→  └─────────┘
    └────────────┘

  Use of automata in protocol verification:
  - Detection of unreachable states (finding deadlocks)
  - Safety verification (confirming prohibited states are not reached)
  - Liveness verification (confirming target states are eventually reached)
  - Model checking tools (SPIN, NuSMV, etc.) are based on automata theory
```

### 7.3 Application to Input Validation

```
Input validation strategies in practice:

  Validation targets and appropriate methods:

  ┌──────────────────┬──────────────────┬───────────────────────────┐
  │ Target           │ Language class  │ Appropriate method                │
  ├──────────────────┼──────────────────┼───────────────────────────┤
  │ Email address    │ Regular (simple)│ Regular expression                  │
  │ (simple check)   │                  │ ^[a-zA-Z0-9.]+@[a-zA-Z0-9.]+$│
  ├──────────────────┼──────────────────┼───────────────────────────┤
  │ Phone number     │ Regular         │ Regular expression                  │
  │                  │                  │ ^\d{2,4}-\d{2,4}-\d{4}$  │
  ├──────────────────┼──────────────────┼───────────────────────────┤
  │ IP address       │ Regular         │ Regex + range check   │
  ├──────────────────┼──────────────────┼───────────────────────────┤
  │ JSON             │ Context-free    │ Parser (recursive descent, etc.)    │
  │                  │                 │ Regex is insufficient        │
  ├──────────────────┼──────────────────┼───────────────────────────┤
  │ XML/HTML         │ Context-free    │ Parser                  │
  │                  │                 │ Impossible with regex        │
  ├──────────────────┼──────────────────┼───────────────────────────┤
  │ Program code     │ Context-sensitive│ Compiler frontend   │
  └──────────────────┴──────────────────┴───────────────────────────┘

  Important principle:
    Regular expressions can only correctly process regular languages.
    Nested structures (matching parentheses, nested tags) are outside the scope of regular languages.
    Attempting to parse HTML with regular expressions is theoretically impossible,
    and also causes serious bugs in practice.
```

---

## 8. Anti-Patterns and Pitfalls

### 8.1 Anti-Pattern 1: Parsing HTML with Regular Expressions

```
Anti-pattern: Parsing HTML with regular expressions

  Incorrect approach:
    # Regular expression attempting to extract HTML tag contents
    pattern = r"<div>(.*)</div>"

  Why it is wrong:
    HTML has nested structure → context-free language
    Regular expressions can only process regular languages
    → It is theoretically impossible to correctly parse HTML with regex

  Concrete problem examples:

    Input: "<div><div>content</div></div>"

    pattern = r"<div>(.*)</div>"  match result for:
      Greedy match: "<div>content</div>" → Entire contents of the outer div
      Expected result: Want to correctly recognize the nested structure
      → Regular expressions cannot track nesting depth

    An even worse example:
      Input: "<div class='x'>content</div>"
      pattern = r"<div>(.*)</div>"  → Does not match (has attributes)
      pattern = r"<div[^>]*>(.*)</div>" → Breaks down with nesting

  Correct approach:
    - Use an HTML parser library
      Python: BeautifulSoup, lxml
      JavaScript: DOMParser, cheerio
    - Theoretical basis: HTML structure requires expressive power beyond context-free languages
      → A parser with computational power of PDA or above is needed

  From the perspective of the Chomsky hierarchy:
    Attempting to process context-free language (Type 2) structures with regular expressions (Type 3)
    is a fundamental type mismatch.
```

### 8.2 Anti-Pattern 2: Vulnerable Regular Expression Patterns (ReDoS)

```
Anti-pattern: Patterns that cause ReDoS (Regular expression Denial of Service)

  Characteristics of dangerous patterns:
    Quantifiers (+, *) are nested:
      (a+)+      ← Exponential backtracking
      (a|aa)+    ← Repeated ambiguous alternation
      (.*a){n}   ← Ambiguous boundary between .* and a

  Example: Email address validation

    Dangerous regular expression:
      ^([a-zA-Z0-9]+\.)+[a-zA-Z]{2,}$

    Attack input:
      "aaaaaaaaaaaaaaaaaaaaaaaa!"
      → Processing time increases exponentially in backtracking engines
      → Consumes 100% of server CPU → Service outage

    Attack mechanism:
      1. ([a-zA-Z0-9]+\.)+ In the part "aaa...a" it tries to match
      2. The trailing "!" fails to match at
      3. Engine rearranges a+ boundaries and retries
      4. Number of combinations grows exponentially

  Countermeasures:
    1. Limit the complexity of regular expressions
       - Avoid nested quantifiers
       - Use atomic groups (?>...) or possessive quantifiers a++

    2. Use NFA-based engines
       - Go's regexp package (RE2-based)
       - Rust's regex crate
       - Google's RE2 library
       These guarantee O(n*m) and ReDoS cannot occur in principle

    3. Set timeouts
       - Python: timeout parameter in the regex library
       - .NET: RegexOptions.Timeout

    4. Limit input length

    5. Detect with static analysis tools
       - ESLint's no-misleading-character-class
       - semgrep's ReDoS rules
```

### 8.3 Anti-Pattern 3: Incomplete DFA Design

```
Anti-pattern: DFA design that does not consider error states (dead states)

  Problematic design:
    「"abc" When designing a DFA that accepts strings containing
    not defining transition targets for non-matching cases

  Example:
    Incomplete transition table:
    ┌────────┬────────┬──────────┐
    │ State  │ a      │ b,c,...  │
    ├────────┼────────┼──────────┤
    │ →q0    │ q1     │ ???      │  ← Transition target undefined
    │  q1    │ ???    │ ???      │
    └────────┴────────┴──────────┘

  Correct design:
    - DFA transition function is a total function (must be defined for all state-symbol pairs)
    - Non-matching transitions go to a dead state (trap state)
    - From the dead state, any input transitions to the dead state itself

    Complete transition table:
    ┌────────┬────────┬──────────┐
    │ State  │ a      │ other    │
    ├────────┼────────┼──────────┤
    │ →q0    │ q1     │ q0      │  ← Reset at q0
    │  q1    │ q1     │ ...     │
    │  dead  │ dead   │ dead    │  ← Dead state
    └────────┴────────┴──────────┘

  Implementation lessons:
    - When implementing a DFA, always provide a validation method to check totality
    - The _validate method of the DFA class above is a good example
```

---

## 9. Comparative Analysis of Automata

### 9.1 Comprehensive Comparison of Automata

```
Comprehensive comparison table of automata:

  ┌────────────┬─────────────┬──────────────┬──────────────┬──────────────┐
  │ Property   │ DFA         │ NFA          │ PDA          │ TM           │
  ├────────────┼─────────────┼──────────────┼──────────────┼──────────────┤
  │ Memory     │ None        │ None         │ Stack        │ Tape       │
  │            │(state only) │(state only)  │(LIFO)        │(unbounded)      │
  ├────────────┼─────────────┼──────────────┼──────────────┼──────────────┤
  │ Nondeterm. │ No          │ Yes          │ Yes/No       │ Yes/No    │
  ├────────────┼─────────────┼──────────────┼──────────────┼──────────────┤
  │ Lang class │ Regular     │ Regular      │ Context-free │ Rec. enum.   │
  ├────────────┼─────────────┼──────────────┼──────────────┼──────────────┤
  │ Det. vs    │ Equivalent  │ Equivalent   │ DPDA < NPDA │ DTM = NTM    │
  │ Nondet.    │ DFA = NFA   │              │ (different)  │ (equivalent)     │
  ├────────────┼─────────────┼──────────────┼──────────────┼──────────────┤
  │ Closure    │ ∪,∩,compl,  │ ∪,∩,compl,   │ ∪,concat,    │ ∪,∩,       │
  │            │ Concat,closure│ Concat,closure │ Closure      │ Concat,closure │
  │            │ All closed  │              │ ∩,compl open │ compl open     │
  ├────────────┼─────────────┼──────────────┼──────────────┼──────────────┤
  │ Membership │ O(n)        │ O(n*m)       │ O(n^3) CYK  │ Undecidable     │
  ├────────────┼─────────────┼──────────────┼──────────────┼──────────────┤
  │ Equivalence│ Decidable   │ Decidable    │ Undecidable     │ Undecidable     │
  ├────────────┼─────────────┼──────────────┼──────────────┼──────────────┤
  │ Emptiness  │ Decidable   │ Decidable    │ Decidable    │ Undecidable     │
  ├────────────┼─────────────┼──────────────┼──────────────┼──────────────┤
  │ Practical  │ Regex       │ Regex        │ Parsing      │ General     │
  │ use        │ Lexing      │ engines      │ Parsers      │ Programs   │
  └────────────┴─────────────┴──────────────┴──────────────┴──────────────┘
```

### 9.2 Flowchart for Classifying Language Classes

```
Flow for determining which class a given language belongs to:

  Given a language L:

  Is L a finite set?
    ├── Yes → Regular language (all finite languages are regular)
    └── No  ↓

  Can L be described by a regular expression?
  Or recognized by a DFA/NFA?
    ├── Yes → Regular language
    └── No / Unknown ↓

  Can it be disproven by the pumping lemma for regular languages?
    ├── Yes → Not a regular language
    └── No  ↓ (The pumping lemma is a necessary condition, not sufficient)

  Can L be generated by a CFG?
  Or recognized by a PDA?
    ├── Yes → Context-free language
    └── No / Unknown ↓

  Can it be disproven by the pumping lemma for CFLs?
    ├── Yes → Not a context-free language
    └── No  ↓

  Can L be generated by a context-sensitive grammar?
    ├── Yes → Context-sensitive language
    └── No  ↓

  Can L be recognized by a TM?
    ├── Yes → Recursive language (decidable)
    └── No  ↓

  Is L semi-decidable by a TM?
    ├── Yes → Recursively enumerable language
    └── No  → Not even recursively enumerable

  Commonly encountered language classifications:
    {a^n | n is prime}              → Not context-free, context-sensitive
    {ww | w is any string}       → Not context-free, context-sensitive
    {w | w is a well-formed parenthesis sequence}      → Context-free language (not regular)
    {a^n b^n | n >= 0}            → Context-free language (not regular)
    {a^n b^n c^n | n >= 0}        → Not context-free, context-sensitive
    {a^n | n >= 0}                → Regular language
```

### 9.3 Comparison of Parsing Methods

```
Comparison of parsing methods:

  ┌──────────────┬───────────┬───────────┬──────────────────────────┐
  │ Method       │ Direction │ Complexity│ Features                     │
  ├──────────────┼───────────┼───────────┼──────────────────────────┤
  │ Rec. descent │ Top-down  │ O(n)~    │ Easy to write by hand           │
  │ (LL)         │ Left-right│ O(n^3)   │ Cannot handle left recursion         │
  ├──────────────┼───────────┼───────────┼──────────────────────────┤
  │ LL(k)        │ Top-down  │ O(n)     │ k-token lookahead      │
  │              │           │          │ ANTLR is representative           │
  ├──────────────┼───────────┼───────────┼──────────────────────────┤
  │ LR(0)/SLR    │ Bottom-up │ O(n)     │ Shift-reduce               │
  │              │           │          │ Narrow grammar class     │
  ├──────────────┼───────────┼───────────┼──────────────────────────┤
  │ LALR(1)      │ Bottom-up │ O(n)     │ yacc/bison is representative      │
  │              │           │          │ Sufficient expressive power       │
  ├──────────────┼───────────┼───────────┼──────────────────────────┤
  │ LR(1)        │ Bottom-up │ O(n)     │ More powerful than LALR(1)         │
  │              │           │          │ Tables become large     │
  ├──────────────┼───────────┼───────────┼──────────────────────────┤
  │ Earley       │ General   │ O(n^3)   │ Handles any CFG         │
  │              │           │          │ Can handle ambiguous grammars       │
  ├──────────────┼───────────┼───────────┼──────────────────────────┤
  │ CYK          │ Bottom-up │ O(n^3)   │ Requires CNF               │
  │              │           │          │ Theoretically important             │
  ├──────────────┼───────────┼───────────┼──────────────────────────┤
  │ PEG          │ Top-down  │ O(n)     │ Packrat parser     │
  │              │           │          │ Ordered choice         │
  │              │           │          │ No ambiguity         │
  └──────────────┴───────────┴───────────┴──────────────────────────┘

  Practical selection guidelines:
  - Small languages (config files, etc.) → Handwritten recursive descent
  - General-purpose languages → LALR(1) or LR(1) parser generators
  - DSLs or extensible languages → PEG parser or LL(k) (ANTLR)
  - Natural language processing → Earley parser or GLR
```

---

## 10. Practice Exercises

### Exercise 1: DFA Design and Implementation (Basics)

```
Problem:
  Design a DFA over the alphabet Sigma = {0, 1} that accepts
  strings containing "01" as a substring,
  and implement it in Python.

  Accept examples: "01", "001", "010", "1101", "0100"
  Reject examples: "", "0", "1", "00", "11", "10", "110"

Hint:
  Use 3 states:
    q0: Have not yet seen "0" (initial state)
    q1: Have seen "0" but have not yet completed "01"
    q2: Found "01" (accept state)

Solution:

  State transition diagram:
       ┌──1──┐          ┌──0──┐
       │     │          │     │
       ▼     │          ▼     │
    →( q0 )──0──→( q1 )──1──→(( q2 ))
                                ↑  │
                                │  │
                                └──┘
                              Self-loop on 0,1

  Transition table:
  ┌────────┬────────┬────────┐
  │ State  │ 0      │ 1      │
  ├────────┼────────┼────────┤
  │ →q0    │ q1     │ q0     │
  │  q1    │ q1     │ q2     │
  │ *q2    │ q2     │ q2     │
  └────────┴────────┴────────┘
```

```python
"""Exercise 1 Solution: DFA that accepts strings containing "01" """

# Using the DFA class defined above
dfa_contains_01 = DFA(
    states={"q0", "q1", "q2"},
    alphabet={"0", "1"},
    transition={
        ("q0", "0"): "q1",
        ("q0", "1"): "q0",
        ("q1", "0"): "q1",
        ("q1", "1"): "q2",
        ("q2", "0"): "q2",
        ("q2", "1"): "q2",
    },
    start_state="q0",
    accept_states={"q2"},
)

# Test
accept_cases = ["01", "001", "010", "1101", "0100"]
reject_cases = ["", "0", "1", "00", "11", "10", "110"]

print("=== Cases that should be accepted ===")
for tc in accept_cases:
    result = dfa_contains_01.accepts(tc)
    status = "OK" if result else "NG"
    print(f"  [{status}] \"{tc}\": {result}")

print("\n=== Cases that should be rejected ===")
for tc in reject_cases:
    result = dfa_contains_01.accepts(tc)
    status = "OK" if not result else "NG"
    print(f"  [{status}] \"{tc}\": {result}")
```

### Exercise 2: NFA Design and DFA Conversion (Intermediate)

```
Problem:
  Convert the regular expression (a|b)*aba to an NFA,
  then convert it to a DFA using the subset construction.
  Verify whether "ababa" is accepted by tracing the execution.

Hint:
  First build the NFA using Thompson's construction.
  The (a|b)* part can be simplified so that state q0
  stays in q0 for both a and b.

Solution:

  Simplified NFA:

       ┌─a,b─┐
       │     │
       ▼     │
    →( q0 )──a──→( q1 )──b──→( q2 )──a──→(( q3 ))

  Transition table:
  ┌────────┬────────────┬────────────┐
  │ State  │ a          │ b          │
  ├────────┼────────────┼────────────┤
  │ →q0    │ {q0, q1}   │ {q0}       │
  │  q1    │ (empty)    │ {q2}       │
  │  q2    │ {q3}       │ (empty)    │
  │ *q3    │ (empty)    │ (empty)    │
  └────────┴────────────┴────────────┘

  Subset construction:

  DFA initial state: {q0}

  {q0} →a→ {q0,q1}    →b→ {q0}
  {q0,q1} →a→ {q0,q1}  →b→ {q0,q2}
  {q0,q2} →a→ {q0,q1,q3} →b→ {q0}
  {q0,q1,q3} →a→ {q0,q1} →b→ {q0,q2}

  State mapping:
    A = {q0}         B = {q0,q1}
    C = {q0,q2}      D = {q0,q1,q3}  ← accept state

  DFA transition table:
  ┌────────┬────────┬────────┐
  │ State  │ a      │ b      │
  ├────────┼────────┼────────┤
  │ →A     │ B      │ A      │
  │  B     │ B      │ C      │
  │  C     │ D      │ A      │
  │ *D     │ B      │ C      │
  └────────┴────────┴────────┘

  Trace (input "ababa"):
    A →a→ B →b→ C →a→ D →b→ C →a→ D → Accept
```

### Exercise 3: CYK Algorithm Implementation (Advanced)

```
Problem:
  Implement the CYK algorithm for the following CNF grammar
  and determine whether the string "aabb" can be generated.

  Grammar G (CNF):
    S → AB | BC
    A → BA | a
    B → CC | b
    C → AB | a

  Hint:
    Build the CYK table T[i][j] as a lower triangular matrix.
    T[i][j] is the set of variables that can generate the substring w[i..j].
```

```python
"""
Exercise 3 Solution: CYK Algorithm Implementation

Determines whether a given string can be generated
by a grammar in Chomsky Normal Form (CNF).
"""

from typing import Dict, List, Set, Tuple


def cyk_parse(
    grammar: Dict[str, List[Tuple[str, ...]]],
    start_symbol: str,
    input_string: str,
) -> bool:
    """
    Performs membership testing using the CYK algorithm.

    Args:
        grammar: Production rules in CNF
                 {variable: [(right-hand side tuple), ...]}
                 Example: {"S": [("A", "B"), ("a",)]}
        start_symbol: Start symbol
        input_string: String to test

    Returns:
        True if input_string can be generated by the grammar
    """
    n = len(input_string)
    if n == 0:
        # For empty string, check if S → epsilon exists
        return ("",) in grammar.get(start_symbol, [])

    # CYK table: table[i][j] = set of variables that can generate w[i..j]
    # i, j are 0-indexed
    table: List[List[Set[str]]] = [
        [set() for _ in range(n)] for _ in range(n)
    ]

    # Base case: substrings of length 1
    for i in range(n):
        for var, productions in grammar.items():
            for prod in productions:
                if len(prod) == 1 and prod[0] == input_string[i]:
                    table[i][i].add(var)

    # Inductive step: substrings of length 2 or more
    for length in range(2, n + 1):       # Substring length
        for i in range(n - length + 1):  # Start position
            j = i + length - 1           # End position
            for k in range(i, j):        # Split point
                for var, productions in grammar.items():
                    for prod in productions:
                        if (len(prod) == 2
                                and prod[0] in table[i][k]
                                and prod[1] in table[k + 1][j]):
                            table[i][j].add(var)

    # Debug output: Display the CYK table
    print(f"\n=== CYK Table (input: \"{input_string}\") ===")
    print(f"    ", end="")
    for j in range(n):
        print(f"  {input_string[j]}(j={j})  ", end="")
    print()
    for i in range(n):
        print(f"  i={i}", end="")
        for j in range(n):
            if j < i:
                print("          ", end="")
            else:
                cell = table[i][j]
                cell_str = "{" + ",".join(sorted(cell)) + "}" if cell else "empty"
                print(f"  {cell_str:8s}", end="")
        print()

    return start_symbol in table[0][n - 1]


# --- Example Usage ---

# Define the CNF grammar
grammar = {
    "S": [("A", "B"), ("B", "C")],
    "A": [("B", "A"), ("a",)],
    "B": [("C", "C"), ("b",)],
    "C": [("A", "B"), ("a",)],
}

# Test
test_strings = ["aabb", "ab", "aab", "b", "aaaa"]
for s in test_strings:
    result = cyk_parse(grammar, "S", s)
    print(f"  \"{s}\" in L(G)? -> {result}\n")
```

---

## 11. FAQ (Frequently Asked Questions)

### Q1: If DFA and NFA have the same computational power, why do we need NFA?

```
Answer:

  There are three main reasons why NFAs are practically important.

  1. Conciseness of description:
     NFAs can often represent the same language with fewer states than DFAs.
     For example, the language "the k-th character from the end is a":
       NFA: O(k) states
       DFA: O(2^k) states (exponential growth in the worst case)

  2. Ease of construction:
     Mechanically building an NFA directly from a regular expression,
     as in Thompson's construction, is straightforward, whereas building
     a DFA directly is difficult.
     Union, concatenation, and Kleene closure are naturally expressed with NFAs.

  3. Theoretical tool:
     Nondeterminism in NFAs serves as a powerful tool in theoretical proofs.
     Many closure properties of regular languages can be proved concisely using NFAs.

  In practice:
  - The typical flow is: design with NFA → convert to DFA via subset construction → fast execution with DFA
    (e.g., compiler lexer generators)
  - Alternatively, NFAs can be simulated directly
    (RE2's lazy DFA construction is a representative example)
```

### Q2: Why do backreferences exceed the theoretical power of regular expressions?

```
Answer:

  Theoretical regular expressions (mathematical definition) and practical
  regular expressions (regex engines in programming languages) are different concepts.

  Theoretical regular expressions:
    Only three operations: concatenation, union (|), and Kleene closure (*)
    → Can only express regular languages

  Additional features in practical regular expressions (Perl-compatible regex, etc.):
    - Backreferences: \1, \2, ...
    - Lookahead/lookbehind: (?=...), (?<=...)
    - Conditional patterns
    etc.

  Why backreferences exceed the scope of regular languages:

    Example: Pattern (.+)\1
    This matches "any string w followed by the same w" = {ww | w in Sigma+}
    → {ww | w in Sigma+} is neither a regular language nor a context-free language
      (it belongs to context-sensitive languages)

  Consequence:
    Regex matching with backreferences is an NP-complete problem
    → Exponential time due to backtracking may be inherently unavoidable
    → NFA-based engines (Go, Rust) intentionally do not support backreferences
```

### Q3: Are there programming language syntax elements that cannot be described by context-free grammars?

```
Answer:

  While most syntax of modern programming languages can be described by CFGs,
  some elements exceed the scope of context-free languages.

  Examples of elements that cannot be described by CFG:

  1. Correspondence between variable declaration and usage:
     "Variable x must be declared before use"
     → This is a {wcw | w is arbitrary} type problem (c is a separator)
     → Context-sensitive

  2. Matching number of function arguments:
     "The number of parameters in a function definition must match the number of arguments at the call site"
     → Can be reduced to a {a^n b^n c^n | n >= 0} type problem
     → Context-sensitive

  3. C language typedef:
     typedef allows identifiers to be used as type names
     → Context is needed to determine whether a token is an identifier or type name during parsing
     → Handled by a special technique called the "lexer hack"

  Practical approach:
    Compilers handle these in two stages:
    a. Syntax analysis (CFG-based): Analyze structure only
    b. Semantic analysis: Type checking, scope analysis, name resolution, etc.
       → Constraints that cannot be expressed by CFG are verified here

    In other words, the capabilities of each level of the Chomsky hierarchy
    are appropriately assigned to each phase of the compiler.
```

### Q4: How do you determine the lower bound on the number of states in a finite automaton?

```
Answer:

  The Myhill-Nerode theorem provides a method to determine the minimum
  number of states in a DFA that accepts a given regular language L.

  Myhill-Nerode theorem:
    Define an equivalence relation ≡_L for language L:
      x ≡_L y ⟺ for all z, (xz in L ⟺ yz in L)

    Theorem:
    1. L is a regular language ⟺ the number of equivalence classes of ≡_L is finite
    2. Minimum DFA state count = number of equivalence classes of ≡_L

  Example: L = {w in {a,b}* | w contains an even number of a's}

    Consider the equivalence classes:
      [epsilon] = {w | w contains an even number of a's}  (even number of a's)
      [a]       = {w | w contains an odd number of a's}   (odd number of a's)

    No further splitting is possible → there are 2 equivalence classes
    → minimum DFA state count is 2

  The Myhill-Nerode theorem is useful for:
    - Proving the minimum number of DFA states
    - An alternative proof method that a language is not regular
      (if the number of equivalence classes is infinite, it is not regular)
    - Theoretical foundation for DFA minimization
```

### Q5: How does automata theory relate to machine learning?

```
Answer:

  There are multiple intersections between automata theory and machine learning.

  1. Automaton learning (grammar inference):
     - The problem of learning DFAs from positive and negative examples
     - RPNI (Regular Positive and Negative Inference) algorithm
     - L* algorithm (Angluin, 1987): DFA learning through queries
     - Finding the minimum consistent DFA is NP-complete

  2. Relationship with Recurrent Neural Networks (RNNs):
     - RNNs with finite precision are theoretically equivalent to DFAs
     - With infinite precision, they are Turing-complete
     - Active research on extracting DFA/NFA from RNNs
     - Theoretical analysis of the extent to which Transformers
       can learn regular and context-free languages

  3. Fusion of formal verification and machine learning:
     - Representing learned model behavior as automata
     - Using automaton properties to verify safety
     - Detection of adversarial inputs

  4. Natural language processing:
     - Morphological analysis using finite-state transducers
     - Probabilistic language models using weighted automata
     - Interpretation of n-gram models as finite-state machines
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not only through theory but also by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge used in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes especially important during code reviews and architecture design.

---

## 12. Summary and Learning Roadmap

### 12.1 Chapter Summary

```
Key points of this chapter:

  1. Finite Automata (DFA/NFA):
     - DFA: Transition destination is unique for each input symbol from each state
     - NFA: Multiple transition destinations and epsilon transitions are allowed
     - DFA and NFA have equivalent computational power (convertible via subset construction)
     - However, state count can grow exponentially in the worst case

  2. Regular Languages:
     - Languages accepted by DFA/NFA = languages expressible by regular expressions
     - The pumping lemma can prove that a language is not regular
     - Closed under many operations (including complement and intersection)
     - Used in practice for lexical analysis, pattern matching, and input validation

  3. Context-Free Grammars and PDA:
     - CFG: Production rules with a single variable on the left-hand side
     - PDA: Finite automaton + stack
     - CFG and PDA have equivalent computational power
     - Widely used for programming language syntax definition
     - More expressive than regular languages (e.g., matching parentheses)

  4. Chomsky Hierarchy:
     - Four levels: Regular < Context-Free < Context-Sensitive < Recursively Enumerable
     - Each level has a corresponding automaton (computational model)
     - Each phase of a compiler leverages capabilities from different levels

  5. Practical Applications:
     - Regular expression engine implementation approaches (NFA vs backtracking)
     - ReDoS risks and countermeasures
     - Theoretical basis for why regular expressions should not be used for HTML parsing
     - Application of finite automata to protocol verification
```

### 12.2 List of Important Theorems and Results

```
  ┌────┬─────────────────────────────┬──────────────────────────────┐
  │ No │ Theorem / Result             │ Significance                 │
  ├────┼─────────────────────────────┼──────────────────────────────┤
  │ 1  │ DFA = NFA (equivalence)     │ Nondeterminism does not      │
  │    │                             │ increase FA computational    │
  │    │                             │ power                        │
  ├────┼─────────────────────────────┼──────────────────────────────┤
  │ 2  │ Kleene's theorem            │ DFA/NFA/regex define the     │
  │    │                             │ same language class           │
  ├────┼─────────────────────────────┼──────────────────────────────┤
  │ 3  │ Pumping lemma for regular   │ Proof tool for showing a     │
  │    │ languages                   │ language is not regular       │
  ├────┼─────────────────────────────┼──────────────────────────────┤
  │ 4  │ Myhill-Nerode theorem       │ Uniqueness of minimal DFA    │
  │    │                             │ and state count              │
  │    │                             │ characterization             │
  ├────┼─────────────────────────────┼──────────────────────────────┤
  │ 5  │ CFG = PDA (equivalence)     │ Two characterizations of     │
  │    │                             │ context-free languages       │
  │    │                             │ are equivalent               │
  ├────┼─────────────────────────────┼──────────────────────────────┤
  │ 6  │ Pumping lemma for CFLs      │ Proof tool for showing a     │
  │    │                             │ language is not context-free  │
  ├────┼─────────────────────────────┼──────────────────────────────┤
  │ 7  │ DPDA < NPDA                │ Determinism and nondetermin- │
  │    │                             │ ism differ in computational  │
  │    │                             │ power for PDAs               │
  ├────┼─────────────────────────────┼──────────────────────────────┤
  │ 8  │ CFG ambiguity is            │ Inherent difficulty of       │
  │    │ undecidable                 │ grammar design               │
  ├────┼─────────────────────────────┼──────────────────────────────┤
  │ 9  │ CFL equivalence is          │ Whether two CFGs generate    │
  │    │ undecidable                 │ the same language cannot     │
  │    │                             │ be decided                   │
  ├────┼─────────────────────────────┼──────────────────────────────┤
  │ 10 │ Chomsky hierarchy           │ Systematic classification    │
  │    │                             │ of language expressiveness   │
  └────┴─────────────────────────────┴──────────────────────────────┘
```

### 12.3 Learning Roadmap

```
Recommended learning order:

  Level 1 (Basics: 1-2 weeks)
  ├── Understanding DFA definitions and examples
  ├── Reading and writing state transition diagrams and tables
  ├── Designing simple DFAs (even count, substring containment, etc.)
  └── Basic regular expression syntax

  Level 2 (Intermediate: 2-3 weeks)
  ├── NFA definition and differences from DFA
  ├── Understanding and hand-computing subset construction
  ├── Thompson's construction (regex → NFA)
  ├── Pumping lemma proof techniques
  └── Python implementation of DFA/NFA

  Level 3 (Advanced: 3-4 weeks)
  ├── Context-free grammar definitions and derivations
  ├── Pushdown automata
  ├── Conversion from CFG to PDA
  ├── Conversion to CNF and CYK algorithm
  ├── Overview of the Chomsky hierarchy
  └── Decidability at each level

  Level 4 (Practical: Ongoing)
  ├── Implementing a lexical analyzer
  ├── Implementing a regular expression engine
  ├── Understanding ReDoS and designing safe regular expressions
  ├── Implementing parser combinators
  └── Designing a compiler frontend
```

---

## Recommended Next Guides


---

## References

1. Hopcroft, J. E., Motwani, R., Ullman, J. D. *Introduction to Automata Theory, Languages, and Computation.* 3rd Edition. Pearson, 2006. -- The standard textbook on automata theory. Systematically covers DFA/NFA equivalence, properties of regular languages, context-free grammars, and the Chomsky hierarchy. Excellent balance between mathematical rigor and practical examples.

2. Sipser, M. *Introduction to the Theory of Computation.* 3rd Edition. Cengage Learning, 2012. -- A textbook covering all of computation theory. Provides unified treatment from automata theory to computability and complexity theory. Clear proof style and highly readable. Rich in exercises.

3. Aho, A. V., Lam, M. S., Sethi, R., Ullman, J. D. *Compilers: Principles, Techniques, and Tools.* 2nd Edition. Pearson, 2006. -- Known as the "Dragon Book." A classic textbook on compiler design. Provides detailed coverage of practical applications of lexical analysis (DFA) and syntactic analysis (CFG/PDA).

4. Cox, R. "Regular Expression Matching Can Be Simple And Fast." 2007. https://swtch.com/~rsc/regexp/regexp1.html -- An article explaining the implementation of NFA-based regular expression engines. Covers Thompson's construction implementation, comparison with DFA approach, and problems with backtracking (ReDoS) through practical examples. Background on the design philosophy of Go's regexp package and RE2.

5. Chomsky, N. "Three Models for the Description of Language." *IRE Transactions on Information Theory.* 2(3):113-124, 1956. -- The original paper on the Chomsky hierarchy. Defined four types of formal grammars and clarified the differences in expressive power of each type. A historic publication that had a profound impact on both computer science and linguistics.

6. Thompson, K. "Programming Techniques: Regular expression search algorithm." *Communications of the ACM.* 11(6):419-422, 1968. -- The original paper on Thompson's construction. Proposed the algorithm for building NFAs from regular expressions. Forms the foundation of modern regular expression engines.

7. Angluin, D. "Learning Regular Sets from Queries and Counterexamples." *Information and Computation.* 75(2):87-106, 1987. -- The original paper on the L* algorithm. An algorithm for exactly learning unknown DFAs through queries (membership queries and equivalence queries). Foundation of automaton learning and grammar inference.

