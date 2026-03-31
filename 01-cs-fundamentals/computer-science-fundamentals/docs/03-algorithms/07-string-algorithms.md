# String Algorithms

> Text search is one of the most familiar algorithms. Browser Ctrl+F, the grep command, IDE search — all of them rely on string matching.

## Learning Objectives

- [ ] Understand the problems with naive string search
- [ ] Explain the principles of the KMP algorithm
- [ ] Understand rolling hashes in the Rabin-Karp algorithm
- [ ] Understand the skip strategy in the Boyer-Moore algorithm
- [ ] Implement string data structures such as Tries and suffix arrays
- [ ] Understand the relationship between regular expressions and automata
- [ ] Master practical string processing optimization techniques

## Prerequisites


---

## 1. String Matching

### 1.1 Naive Method

```python
def naive_search(text, pattern):
    """Return all occurrence positions of the pattern in the text"""
    n, m = len(text), len(pattern)
    positions = []
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            positions.append(i)
    return positions

# Complexity: O(n * m) — worst case
# Example: text="AAAAAB", pattern="AAB"
# Almost m characters are compared each time before a mismatch is detected

# Worst-case example:
# text = "A" * 1000000 + "B"
# pattern = "A" * 999 + "B"
# -> ~1 million * 1000 character comparisons = 1 billion comparisons
```

### 1.2 KMP Algorithm (Knuth-Morris-Pratt)

```python
def kmp_search(text, pattern):
    """Eliminate redundant comparisons using the failure function"""
    # Build the failure function (partial match table)
    m = len(pattern)
    failure = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        failure[i] = j

    # Search
    positions = []
    j = 0
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = failure[j - 1]  # Jump using failure function
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            positions.append(i - m + 1)
            j = failure[j - 1]

    return positions

# Complexity: O(n + m) — linear time!
# Failure function construction: O(m)
# Search: O(n)

# Core idea of KMP:
# When a mismatch occurs, leverage the "prefix-suffix match" of the pattern
# to adjust only the pattern pointer without backtracking the text pointer
```

#### Detailed Explanation of the KMP Failure Function

```python
def build_failure_function(pattern):
    """Build the failure function and visualize each step"""
    m = len(pattern)
    failure = [0] * m
    j = 0

    print(f"Pattern: {pattern}")
    print(f"{'i':>3} {'pattern[i]':>10} {'j':>3} {'Match?':>6} {'failure[i]':>10}")
    print("-" * 40)

    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]

        match = pattern[i] == pattern[j]
        if match:
            j += 1

        failure[i] = j
        print(f"{i:>3} {pattern[i]:>10} {j:>3} {'YES' if match else 'NO':>6} {failure[i]:>10}")

    return failure

# Example: pattern "ABABAC"
# failure = [0, 0, 1, 2, 3, 0]
#
# Interpretation:
# failure[i] = length of the longest matching proper prefix and suffix of pattern[0:i+1]
#
# i=0: "A"       -> 0 (trivial)
# i=1: "AB"      -> 0 (prefix "A" != suffix "B")
# i=2: "ABA"     -> 1 (prefix "A" = suffix "A")
# i=3: "ABAB"    -> 2 (prefix "AB" = suffix "AB")
# i=4: "ABABA"   -> 3 (prefix "ABA" = suffix "ABA")
# i=5: "ABABAC"  -> 0 (no match)
```

#### Visualization of KMP Algorithm Execution

```python
def kmp_search_verbose(text, pattern):
    """Visualize the KMP algorithm execution"""
    m = len(pattern)
    failure = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        failure[i] = j

    print(f"Failure function: {failure}")
    print(f"Text:   {text}")
    print()

    positions = []
    j = 0
    comparisons = 0

    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            print(f"  Mismatch text[{i}]='{text[i]}' vs pattern[{j}]='{pattern[j]}'"
                  f" -> rewind j to {failure[j-1]}")
            j = failure[j - 1]
            comparisons += 1

        comparisons += 1
        if text[i] == pattern[j]:
            j += 1
            if j == m:
                positions.append(i - m + 1)
                print(f"  * Match found! Position {i - m + 1}")
                j = failure[j - 1]
        else:
            pass

    print(f"\nComparisons: {comparisons}")
    print(f"Naive worst case: {len(text) * len(pattern)}")
    return positions

# Usage example
kmp_search_verbose("ABABDABABABABAC", "ABABAC")
```

### 1.3 Rabin-Karp Algorithm

```python
def rabin_karp(text, pattern):
    """Fast pattern matching using rolling hash"""
    n, m = len(text), len(pattern)
    base, mod = 256, 10**9 + 7
    positions = []

    # Hash value of the pattern
    pattern_hash = 0
    text_hash = 0
    h = pow(base, m - 1, mod)

    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % mod
        text_hash = (text_hash * base + ord(text[i])) % mod

    for i in range(n - m + 1):
        if text_hash == pattern_hash:
            if text[i:i+m] == pattern:  # Hash collision check
                positions.append(i)
        if i < n - m:
            text_hash = ((text_hash - ord(text[i]) * h) * base
                        + ord(text[i + m])) % mod

    return positions

# Complexity: O(n + m) expected / O(nm) worst case (on hash collisions)
# Advantage: efficient for simultaneous multi-pattern search
```

#### Simultaneous Multi-Pattern Search

```python
def rabin_karp_multi(text, patterns):
    """Search for multiple patterns simultaneously"""
    n = len(text)
    base, mod = 256, 10**9 + 7
    results = {p: [] for p in patterns}

    # Group patterns by length
    by_length = {}
    for p in patterns:
        m = len(p)
        if m not in by_length:
            by_length[m] = {}
        # Compute hash value
        h = 0
        for c in p:
            h = (h * base + ord(c)) % mod
        by_length[m][h] = by_length[m].get(h, []) + [p]

    # Search for each length group
    for m, hash_to_patterns in by_length.items():
        if m > n:
            continue

        h_pow = pow(base, m - 1, mod)

        # Text hash value
        text_hash = 0
        for i in range(m):
            text_hash = (text_hash * base + ord(text[i])) % mod

        for i in range(n - m + 1):
            if text_hash in hash_to_patterns:
                for p in hash_to_patterns[text_hash]:
                    if text[i:i+m] == p:
                        results[p].append(i)

            if i < n - m:
                text_hash = ((text_hash - ord(text[i]) * h_pow) * base
                            + ord(text[i + m])) % mod

    return results

# Usage example
text = "she sells seashells by the seashore"
patterns = ["she", "sea", "sell", "shore"]
result = rabin_karp_multi(text, patterns)
# {'she': [0, 15], 'sea': [10, 27], 'sell': [4], 'shore': [30]}
```

### 1.4 Boyer-Moore Algorithm

```python
def boyer_moore(text, pattern):
    """Boyer-Moore: compare from pattern end, skip large distances on mismatch"""
    n, m = len(text), len(pattern)
    if m > n:
        return []

    # Build Bad Character table
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i  # Last occurrence position of each character in the pattern

    positions = []
    i = 0  # Position in the text

    while i <= n - m:
        j = m - 1  # Compare from pattern end

        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1

        if j < 0:
            # Match!
            positions.append(i)
            # Look for the next match
            i += (m - bad_char.get(text[i + m], -1) if i + m < n else 1)
        else:
            # Bad Character rule
            bad_pos = bad_char.get(text[i + j], -1)
            shift = max(1, j - bad_pos)
            i += shift

    return positions

# Characteristics of Boyer-Moore:
# - Compares from pattern end (right to left)
# - If the mismatched character is not in the pattern, skip by the pattern length
# - Best case: O(n/m) — longer patterns run faster!
# - Worst case: O(nm) — improves to O(n) with the Good Suffix rule
# - Practically the fastest string search algorithm (used by GNU grep)

# Boyer-Moore execution example:
# text = "HERE IS A SIMPLE EXAMPLE"
# pattern = "EXAMPLE"
#
# Step 1: Compare E and S -> mismatch, S is not in pattern -> skip 7 characters
# Step 2: Compare at next position -> ...
# -> Far fewer comparisons than the naive method
```

#### Good Suffix Rule Implementation

```python
def boyer_moore_full(text, pattern):
    """Full Boyer-Moore implementation (Bad Character + Good Suffix)"""
    n, m = len(text), len(pattern)
    if m > n:
        return []

    # Bad Character table
    bad_char = [-1] * 256
    for i in range(m):
        bad_char[ord(pattern[i])] = i

    # Good Suffix table
    good_suffix = [0] * (m + 1)
    border = [0] * (m + 1)

    # Case 1: A match for the good suffix exists within the pattern
    i = m
    j = m + 1
    border[i] = j
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if good_suffix[j] == 0:
                good_suffix[j] = j - i
            j = border[j]
        i -= 1
        j -= 1
        border[i] = j

    # Case 2: A prefix of the good suffix matches a prefix of the pattern
    j = border[0]
    for i in range(m + 1):
        if good_suffix[i] == 0:
            good_suffix[i] = j
        if i == j:
            j = border[j]

    # Search
    positions = []
    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1

        if j < 0:
            positions.append(i)
            i += good_suffix[0]
        else:
            bc_shift = j - bad_char[ord(text[i + j])]
            gs_shift = good_suffix[j + 1]
            i += max(bc_shift, gs_shift)

    return positions

# Complexity: O(n + m) preprocessing + O(n) search
# Best case: O(n/m) — longer patterns run faster
```

### 1.5 Aho-Corasick Algorithm

```python
from collections import deque

class AhoCorasick:
    """Aho-Corasick: simultaneous multi-pattern search in linear time"""

    def __init__(self):
        self.goto = [{}]
        self.failure = [0]
        self.output = [[]]

    def add_pattern(self, pattern, idx):
        """Add a pattern"""
        state = 0
        for char in pattern:
            if char not in self.goto[state]:
                self.goto[state][char] = len(self.goto)
                self.goto.append({})
                self.failure.append(0)
                self.output.append([])
            state = self.goto[state][char]
        self.output[state].append(idx)

    def build(self):
        """Build the failure function (BFS)"""
        queue = deque()

        # Failure function for depth-1 nodes is 0 (root)
        for char, state in self.goto[0].items():
            queue.append(state)

        while queue:
            u = queue.popleft()
            for char, v in self.goto[u].items():
                queue.append(v)

                # Compute failure function
                state = self.failure[u]
                while state != 0 and char not in self.goto[state]:
                    state = self.failure[state]
                self.failure[v] = self.goto[state].get(char, 0)
                if self.failure[v] == v:
                    self.failure[v] = 0

                # Update output function
                self.output[v] = self.output[v] + self.output[self.failure[v]]

    def search(self, text):
        """Search for all patterns in the text"""
        results = []
        state = 0

        for i, char in enumerate(text):
            while state != 0 and char not in self.goto[state]:
                state = self.failure[state]
            state = self.goto[state].get(char, 0)

            for pattern_idx in self.output[state]:
                results.append((i, pattern_idx))

        return results

# Usage example
ac = AhoCorasick()
patterns = ["he", "she", "his", "hers"]
for i, p in enumerate(patterns):
    ac.add_pattern(p, i)
ac.build()

text = "ahishers"
results = ac.search(text)
for pos, idx in results:
    p = patterns[idx]
    print(f"Pattern '{p}' found at position {pos - len(p) + 1}")

# Complexity: O(n + m + z)
#   n: text length, m: total character count of all patterns, z: number of matches
# Applications:
# - Virus scanners (signature search)
# - Network IDS packet inspection
# - Text filtering (banned word detection)
# - DNA sequence multi-motif search
```

### 1.6 Z-Algorithm

```python
def z_function(s):
    """Compute Z-array: z[i] = length of the longest common prefix of s[i:] and s[0:]"""
    n = len(s)
    z = [0] * n
    z[0] = n
    l, r = 0, 0

    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]

    return z

def z_search(text, pattern):
    """Pattern matching using the Z-Algorithm"""
    # Concatenate pattern$text
    concat = pattern + "$" + text
    z = z_function(concat)
    m = len(pattern)

    positions = []
    for i in range(m + 1, len(concat)):
        if z[i] == m:
            positions.append(i - m - 1)

    return positions

# Complexity: O(n + m)
# Same linear time as KMP, but the implementation is more intuitive
# The Z-array is also useful for detecting repeating patterns in strings

# Z-array application: finding the minimum period
def min_period(s):
    """Find the minimum period of a string"""
    z = z_function(s)
    n = len(s)
    for period in range(1, n + 1):
        if n % period == 0 and z[period] == n - period:
            return period
    return n

# Example: "abcabc" -> minimum period 3 ("abc")
# Example: "abab" -> minimum period 2 ("ab")
# Example: "abcde" -> minimum period 5 (entire string)
```

---

## 2. String Data Structures

### 2.1 Trie

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # Number of words with this prefix

class Trie:
    """Prefix tree — efficiently manages a set of strings"""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def count_prefix(self, prefix):
        """Number of words starting with prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.count

    def autocomplete(self, prefix, limit=10):
        """Autocomplete: return words starting with prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        results = []
        self._collect_words(node, prefix, results, limit)
        return results

    def _collect_words(self, node, current, results, limit):
        if len(results) >= limit:
            return
        if node.is_end:
            results.append(current)
        for char in sorted(node.children):
            self._collect_words(node.children[char], current + char, results, limit)

    def delete(self, word):
        """Delete a word"""
        def _delete(node, word, depth):
            if depth == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                return len(node.children) == 0

            char = word[depth]
            if char not in node.children:
                return False

            should_delete = _delete(node.children[char], word, depth + 1)

            if should_delete:
                del node.children[char]
                return not node.is_end and len(node.children) == 0

            node.children[char].count -= 1
            return False

        _delete(self.root, word, 0)

# Applications:
# - Autocomplete (search suggestions)
# - Spell checking
# - IP routing (longest prefix match)
# - Fast dictionary lookup
# - Phone number search
# Complexity: Insert/Search O(m) — m is the string length
# Space: O(Sigma * N) — Sigma is the alphabet size, N is the number of nodes

# Usage example: Autocomplete
trie = Trie()
words = ["apple", "application", "apply", "ape", "banana", "band", "bank"]
for w in words:
    trie.insert(w)

print(trie.autocomplete("app"))  # ["apple", "application", "apply"]
print(trie.count_prefix("app"))  # 3
print(trie.count_prefix("ban"))  # 3
```

#### Compressed Trie (Patricia Trie / Radix Tree)

```python
class CompressedTrieNode:
    def __init__(self, label=""):
        self.label = label       # Edge label (can be multiple characters)
        self.children = {}
        self.is_end = False

class CompressedTrie:
    """Compressed Trie: groups common prefixes instead of individual characters"""

    def __init__(self):
        self.root = CompressedTrieNode()

    def insert(self, word):
        node = self.root
        i = 0

        while i < len(word):
            first_char = word[i]

            if first_char not in node.children:
                new_node = CompressedTrieNode(word[i:])
                new_node.is_end = True
                node.children[first_char] = new_node
                return

            child = node.children[first_char]
            label = child.label

            # Find the length of the common prefix
            j = 0
            while j < len(label) and i + j < len(word) and label[j] == word[i + j]:
                j += 1

            if j == len(label):
                # Entire label matches -> proceed to child node
                i += j
                node = child
            else:
                # Branch in the middle -> split the node
                # New node for the common part
                split = CompressedTrieNode(label[:j])
                split.children[label[j]] = child
                child.label = label[j:]

                if i + j < len(word):
                    new_node = CompressedTrieNode(word[i + j:])
                    new_node.is_end = True
                    split.children[word[i + j]] = new_node
                else:
                    split.is_end = True

                node.children[first_char] = split
                return

        node.is_end = True

    def search(self, word):
        node = self.root
        i = 0

        while i < len(word):
            first_char = word[i]
            if first_char not in node.children:
                return False

            child = node.children[first_char]
            label = child.label

            if not word[i:].startswith(label):
                return False

            i += len(label)
            node = child

        return node.is_end

# Standard Trie vs Compressed Trie:
# Standard: "application" -> a -> p -> p -> l -> i -> c -> a -> t -> i -> o -> n (11 nodes)
# Compressed: "application" -> "application" (1 node)
# Significantly improved space efficiency (especially for long keys)
```

### 2.2 Suffix Array

```python
def build_suffix_array(s):
    """Build a suffix array (O(n log^2 n) version)"""
    n = len(s)
    # Sort each suffix by (rank, next rank, position)
    suffixes = [(ord(s[i]), ord(s[i + 1]) if i + 1 < n else -1, i)
                for i in range(n)]
    suffixes.sort()

    # Update ranks
    rank = [0] * n
    rank[suffixes[0][2]] = 0
    for i in range(1, n):
        rank[suffixes[i][2]] = rank[suffixes[i-1][2]]
        if suffixes[i][:2] != suffixes[i-1][:2]:
            rank[suffixes[i][2]] += 1

    k = 2
    while k < n:
        # Sort by (rank[i], rank[i+k])
        suffixes = [(rank[i], rank[i + k] if i + k < n else -1, i)
                    for i in range(n)]
        suffixes.sort()

        new_rank = [0] * n
        new_rank[suffixes[0][2]] = 0
        for i in range(1, n):
            new_rank[suffixes[i][2]] = new_rank[suffixes[i-1][2]]
            if suffixes[i][:2] != suffixes[i-1][:2]:
                new_rank[suffixes[i][2]] += 1
        rank = new_rank
        k *= 2

    return [s[2] for s in suffixes]

# Suffix array usage example
s = "banana"
sa = build_suffix_array(s)
print(f"Suffix array: {sa}")
# [5, 3, 1, 0, 4, 2] -> a, ana, anana, banana, na, nana

# Pattern search: O(m log n) with binary search
def search_with_suffix_array(text, sa, pattern):
    """Search for a pattern using a suffix array"""
    n = len(text)
    m = len(pattern)
    lo, hi = 0, n - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        suffix = text[sa[mid]:sa[mid] + m]

        if suffix == pattern:
            # Match -> find all occurrences
            start = end = mid
            while start > 0 and text[sa[start-1]:sa[start-1]+m] == pattern:
                start -= 1
            while end < n - 1 and text[sa[end+1]:sa[end+1]+m] == pattern:
                end += 1
            return [sa[i] for i in range(start, end + 1)]
        elif suffix < pattern:
            lo = mid + 1
        else:
            hi = mid - 1

    return []

# LCP Array (Longest Common Prefix)
def build_lcp_array(text, sa):
    """Build LCP array using Kasai's algorithm O(n)"""
    n = len(text)
    rank = [0] * n
    lcp = [0] * n

    for i in range(n):
        rank[sa[i]] = i

    k = 0
    for i in range(n):
        if rank[i] == 0:
            k = 0
            continue
        j = sa[rank[i] - 1]
        while i + k < n and j + k < n and text[i + k] == text[j + k]:
            k += 1
        lcp[rank[i]] = k
        if k > 0:
            k -= 1

    return lcp

# Applications of the LCP array:
# - Finding the longest repeated substring: max(lcp)
# - Counting distinct substrings: n*(n+1)/2 - sum(lcp)
# - Longest common substring (concatenate two strings and build SA + LCP)
```

```
Suffix array: an array of all suffixes sorted lexicographically

  String: "banana"
  Suffixes:            After sorting:
  0: banana           5: a
  1: anana            3: ana
  2: nana             1: anana
  3: ana              0: banana
  4: na               4: na
  5: a                2: nana

  Suffix array: [5, 3, 1, 0, 4, 2]

  Pattern search: O(m log n) with binary search
  Construction: O(n) (SA-IS algorithm)

  Applications:
  - Full-text search engines
  - DNA sequence analysis
  - Data compression (BWT: Burrows-Wheeler Transform)
  - Longest repeated substring
  - Counting distinct substrings
```

### 2.3 Suffix Tree (Overview)

```
Suffix tree: stores all suffixes in a compressed Trie

  Suffix tree for string "banana$":

  root
  |-- "a" -- "na" -- "na$"
  |          +-- "$"
  |-- "banana$"
  |-- "na" -- "na$"
  |          +-- "$"
  +-- "$"

  Characteristics:
  - Construction: O(n) (Ukkonen's algorithm)
  - Pattern search: O(m)
  - Longest repeated substring: O(n)
  - Longest common substring: O(n + m)

  Suffix tree vs Suffix array:
  +----------------+------------------+------------------+
  | Property       | Suffix tree      | Suffix array     |
  +----------------+------------------+------------------+
  | Space          | O(n) but large   | O(n) efficient   |
  |                | constant         |                  |
  | Construction   | O(n) Ukkonen     | O(n) SA-IS       |
  | Pattern search | O(m)             | O(m log n)       |
  | Implementation | High complexity  | Moderate         |
  | complexity     |                  |                  |
  | Cache          | Low              | High             |
  | efficiency     |                  |                  |
  +----------------+------------------+------------------+

  In practice, suffix array + LCP array is the mainstream approach
```

---

## 3. Regular Expressions and Automata

### 3.1 Internals of Regular Expressions

```
Regular expression execution modes:

  1. NFA (Nondeterministic Finite Automaton) mode
     -> Regex -> NFA -> Simulate on string
     -> Complexity: O(n * m) — n is string length, m is regex length
     -> Used by Go, Rust, RE2

  2. Backtracking mode
     -> Execute regex directly with backtracking
     -> Complexity: O(2^n) worst case (exponential time!)
     -> Used by Python, JavaScript, Java, Ruby, Perl
     -> Risk of "catastrophic backtracking"

  Catastrophic backtracking (ReDoS) example:
  Pattern: (a+)+b
  Input: "aaaaaaaaaaaaaaaaaaaac"
  -> Backtracking mode: exponential time, no response!
  -> NFA mode: determines mismatch in linear time

  Countermeasures:
  - Set regex execution timeouts
  - Use RE2/Go regex engine
  - Use static analysis tools for regex
```

### 3.2 Fundamentals of Finite Automata

```python
# DFA (Deterministic Finite Automaton) implementation
class DFA:
    """Deterministic Finite Automaton"""
    def __init__(self, states, alphabet, transitions, start, accepts):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions  # {(state, char): next_state}
        self.start = start
        self.accepts = accepts

    def accepts_string(self, s):
        state = self.start
        for char in s:
            key = (state, char)
            if key not in self.transitions:
                return False
            state = self.transitions[key]
        return state in self.accepts

# Example: DFA that accepts strings containing "ab"
dfa = DFA(
    states={0, 1, 2},
    alphabet={'a', 'b'},
    transitions={
        (0, 'a'): 1, (0, 'b'): 0,
        (1, 'a'): 1, (1, 'b'): 2,
        (2, 'a'): 2, (2, 'b'): 2,
    },
    start=0,
    accepts={2}
)

print(dfa.accepts_string("aab"))     # True
print(dfa.accepts_string("ba"))      # False
print(dfa.accepts_string("bab"))     # True

# NFA -> DFA conversion (subset construction)
# Theoretically, the number of states can grow exponentially
# In practice, it is manageable in most cases
```

### 3.3 Simple Regular Expression Engine

```python
class SimpleRegex:
    """Simple regex engine (Thompson NFA construction)
    Supports: literal characters, '.', '*', '+', '?', '|', '(', ')'
    """

    class State:
        def __init__(self):
            self.transitions = {}  # char -> [State]
            self.epsilon = []      # Epsilon transitions
            self.is_accept = False

    class NFA:
        def __init__(self, start, accept):
            self.start = start
            self.accept = accept

    @staticmethod
    def char_nfa(c):
        """NFA for a single character"""
        start = SimpleRegex.State()
        accept = SimpleRegex.State()
        accept.is_accept = True
        start.transitions[c] = [accept]
        return SimpleRegex.NFA(start, accept)

    @staticmethod
    def concat(nfa1, nfa2):
        """Concatenation"""
        nfa1.accept.is_accept = False
        nfa1.accept.epsilon.append(nfa2.start)
        return SimpleRegex.NFA(nfa1.start, nfa2.accept)

    @staticmethod
    def union(nfa1, nfa2):
        """Alternation (|)"""
        start = SimpleRegex.State()
        accept = SimpleRegex.State()
        accept.is_accept = True

        start.epsilon.extend([nfa1.start, nfa2.start])
        nfa1.accept.is_accept = False
        nfa2.accept.is_accept = False
        nfa1.accept.epsilon.append(accept)
        nfa2.accept.epsilon.append(accept)

        return SimpleRegex.NFA(start, accept)

    @staticmethod
    def star(nfa):
        """Kleene closure (*)"""
        start = SimpleRegex.State()
        accept = SimpleRegex.State()
        accept.is_accept = True

        start.epsilon.extend([nfa.start, accept])
        nfa.accept.is_accept = False
        nfa.accept.epsilon.extend([nfa.start, accept])

        return SimpleRegex.NFA(start, accept)

    @staticmethod
    def match(nfa, text):
        """Match a string using NFA simulation"""
        def epsilon_closure(states):
            stack = list(states)
            closure = set(states)
            while stack:
                state = stack.pop()
                for next_state in state.epsilon:
                    if next_state not in closure:
                        closure.add(next_state)
                        stack.append(next_state)
            return closure

        current = epsilon_closure({nfa.start})

        for char in text:
            next_states = set()
            for state in current:
                if char in state.transitions:
                    next_states.update(state.transitions[char])
                if '.' in state.transitions:  # Wildcard
                    next_states.update(state.transitions['.'])
            current = epsilon_closure(next_states)

        return any(state.is_accept for state in current)

# Advantages of Thompson NFA:
# - Always O(n * m) complexity (no backtracking)
# - No catastrophic backtracking
# - Used by RE2, Go regex engine
```

### 3.4 Dangerous Regex Patterns (ReDoS)

```python
import re
import time

# ReDoS (Regular Expression Denial of Service) examples

# Dangerous patterns: nested quantifiers
dangerous_patterns = [
    r"(a+)+b",           # Nested repetition
    r"(a|aa)+b",         # Alternation + repetition
    r"(.*a){x}",         # .* + repetition
    r"([a-zA-Z]+)*@",    # Character class + nesting
]

# Examples of rewriting to safe patterns:
# Dangerous: (a+)+b   -> Safe: a+b
# Dangerous: (a|aa)+b -> Safe: a+b
# Dangerous: ([a-zA-Z]+)*@ -> Safe: [a-zA-Z]+@

# ReDoS detection checklist:
# 1. Nested quantifiers (X+)+ or (X*)*
# 2. Overlapping alternatives (a|a)+ or (a|ab)+
# 3. Quantifiers followed by patterns that easily fail
# 4. Heavy use of backreferences

# Countermeasures:
# 1. Set regex execution timeouts
# 2. Limit input length
# 3. Use RE2 or Go regex engine
# 4. Use static analysis tools for regex (e.g., redos-checker)
# 5. Replace regex with non-regex string processing where possible

# Regex with timeout in Python
import signal

def regex_with_timeout(pattern, text, timeout=1):
    """Regex match with timeout (Unix-like systems only)"""
    def handler(signum, frame):
        raise TimeoutError("Regex execution timed out")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    try:
        result = re.match(pattern, text)
        signal.alarm(0)
        return result
    except TimeoutError:
        return None
```

### 3.5 Practical String Processing

```python
# Complexity of commonly used string operations in practice

s = "Hello, World!"

# O(n) operations:
s.find("World")      # Linear search
s.replace("o", "0")  # Replace all
s.split(",")          # Split
"".join(parts)        # Join

# Caution: String concatenation
# Bad: O(n^2): String concatenation inside a loop
result = ""
for word in words:
    result += word  # Creates a new string each time -> O(n^2)

# Good: O(n): Use join
result = "".join(words)

# Good: O(n): io.StringIO
from io import StringIO
buf = StringIO()
for word in words:
    buf.write(word)
result = buf.getvalue()
```

#### String Processing Best Practices

```python
import re

# 1. Precompile regex patterns
# Bad: Compiles every time
for line in lines:
    if re.match(r"\d{3}-\d{4}", line):
        pass

# Good: Precompile
pattern = re.compile(r"\d{3}-\d{4}")
for line in lines:
    if pattern.match(line):
        pass

# 2. String formatting
name = "World"
# Bad: Slow
result = "Hello, " + name + "!"
# Good: f-string (Python 3.6+, fastest)
result = f"Hello, {name}!"

# 3. Bulk string operations
# Bad: Repeated string concatenation
s = ""
for i in range(10000):
    s += str(i)

# Good: Accumulate in a list and join
parts = []
for i in range(10000):
    parts.append(str(i))
s = "".join(parts)

# Good: Generator expression
s = "".join(str(i) for i in range(10000))

# 4. String slicing creates a new object
# Python: s[i:j] is a copy O(j-i)
# Avoid heavy slicing -> use memoryview or index-based processing

# 5. String builders in each language
# Python: "".join(list) or io.StringIO
# Java: StringBuilder
# C#: StringBuilder
# Go: strings.Builder
# Rust: String::push_str or format!
# JavaScript: Array.join() or template literals
```

#### Handling Unicode Strings

```python
# Unicode considerations

# 1. Character count vs byte count
s = "こんにちは"
print(len(s))                    # 5 (character count)
print(len(s.encode('utf-8')))    # 15 (UTF-8 byte count)
print(len(s.encode('utf-16')))   # 12 (UTF-16 byte count, including BOM)

# 2. Surrogate pairs
emoji = "😀"
print(len(emoji))                        # 1 (Python)
# In JavaScript: "😀".length = 2 (UTF-16 surrogate pair)

# 3. Combining Characters
# "が" = "か" + "゛" in some cases
import unicodedata
s1 = "が"                              # 1 character
s2 = "か\u3099"                        # か + dakuten = が
print(s1 == s2)                         # False!
print(unicodedata.normalize('NFC', s1) ==
      unicodedata.normalize('NFC', s2)) # True

# 4. Compare strings after normalization
def safe_compare(s1, s2):
    return unicodedata.normalize('NFC', s1) == unicodedata.normalize('NFC', s2)

# 5. Grapheme Clusters
# "👨‍👩‍👧‍👦" is a single emoji, but internally consists of multiple code points
family = "👨‍👩‍👧‍👦"
print(len(family))  # 7 (Python) — includes ZWJ (Zero Width Joiner)
# The grapheme library is needed to correctly treat this as 1 character
```

---

## 4. Advanced String Algorithms

### 4.1 Manacher's Algorithm (Longest Palindromic Substring)

```python
def manacher(s):
    """Find the longest palindrome centered at each position in O(n)"""
    # Insert '#' between characters: "abc" -> "#a#b#c#"
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n  # p[i] = radius of the palindrome centered at position i

    center = right = 0  # Palindrome extending furthest to the right
    for i in range(n):
        # Leverage the mirror
        mirror = 2 * center - i
        if i < right:
            p[i] = min(right - i, p[mirror])

        # Attempt to expand
        while (i + p[i] + 1 < n and i - p[i] - 1 >= 0 and
               t[i + p[i] + 1] == t[i - p[i] - 1]):
            p[i] += 1

        # Update the right boundary
        if i + p[i] > right:
            center = i
            right = i + p[i]

    # Find the longest palindrome
    max_len = max(p)
    max_center = p.index(max_len)
    start = (max_center - max_len) // 2

    return s[start:start + max_len]

# Examples:
print(manacher("babad"))    # "bab" or "aba"
print(manacher("cbbd"))     # "bb"
print(manacher("racecar"))  # "racecar"

# Complexity: O(n) — significant improvement over the O(n^2) naive method
# Mirror utilization is the key to reducing complexity
```

### 4.2 Burrows-Wheeler Transform (BWT)

```python
def bwt_encode(s):
    """Burrows-Wheeler Transform (preprocessing for compression)"""
    s = s + '$'  # Append sentinel character
    n = len(s)

    # Sort all cyclic rotations
    rotations = sorted(range(n), key=lambda i: s[i:] + s[:i])

    # Extract the last column
    bwt = ''.join(s[(i - 1) % n] for i in rotations)
    # Position of the original string
    original_idx = rotations.index(0)

    return bwt, original_idx

def bwt_decode(bwt, idx):
    """Inverse BWT"""
    n = len(bwt)
    table = [''] * n

    for _ in range(n):
        table = sorted(bwt[i] + table[i] for i in range(n))

    for row in table:
        if row.endswith('$'):
            return row[:-1]

# Usage example
text = "banana"
encoded, idx = bwt_encode(text)
print(f"BWT: {encoded}")  # "annb$aa"
decoded = bwt_decode(encoded, idx)
print(f"Decoded: {decoded}")  # "banana"

# Applications of BWT:
# - Core technology in bzip2 compression
# - FM-Index (full-text search index)
# - DNA sequence alignment (BWA, Bowtie)
# BWT clusters identical characters together -> efficiently compressed with Run-Length Encoding
```

### 4.3 Applications of String Hashing

```python
class RollingHash:
    """Rolling Hash: compute substring hash in O(1)"""

    def __init__(self, s, base=131, mod=10**18 + 9):
        self.n = len(s)
        self.base = base
        self.mod = mod

        # Preprocessing: compute hash values and base powers
        self.hash = [0] * (self.n + 1)
        self.power = [1] * (self.n + 1)

        for i in range(self.n):
            self.hash[i + 1] = (self.hash[i] * base + ord(s[i])) % mod
            self.power[i + 1] = (self.power[i] * base) % mod

    def get_hash(self, l, r):
        """Return the hash of substring s[l:r] in O(1)"""
        return (self.hash[r] - self.hash[l] * self.power[r - l]) % self.mod

    def is_equal(self, l1, r1, l2, r2):
        """Check s[l1:r1] == s[l2:r2] in O(1) (probabilistic)"""
        return self.get_hash(l1, r1) == self.get_hash(l2, r2)

# Usage example: Longest repeated substring
def longest_repeated_substring(s):
    """Find the longest repeated substring using binary search + rolling hash"""
    rh = RollingHash(s)
    n = len(s)

    def has_repeat(length):
        """Check if a repeated substring of the given length exists"""
        seen = set()
        for i in range(n - length + 1):
            h = rh.get_hash(i, i + length)
            if h in seen:
                return i
            seen.add(h)
        return -1

    lo, hi = 0, n - 1
    best = ""

    while lo <= hi:
        mid = (lo + hi) // 2
        pos = has_repeat(mid)
        if pos != -1:
            best = s[pos:pos + mid]
            lo = mid + 1
        else:
            hi = mid - 1

    return best

# Complexity: O(n log n) — expected
# Usage example
print(longest_repeated_substring("banana"))  # "ana"
print(longest_repeated_substring("abcabc"))  # "abc"
```

---

## 5. Practical String Processing

### 5.1 Text Processing Pipeline

```python
# Example of a practical text processing pipeline

import re
from collections import Counter

def text_processing_pipeline(text):
    """Preprocess text and compute word frequency"""
    # 1. Normalization
    text = text.lower()

    # 2. Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Tokenization
    words = text.split()

    # 4. Remove stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                  'in', 'on', 'at', 'to', 'of', 'and', 'or', 'for'}
    words = [w for w in words if w not in stop_words]

    # 5. Frequency counting
    freq = Counter(words)

    return freq.most_common(10)

# Fuzzy matching (approximate search)
def fuzzy_search(query, candidates, max_distance=2):
    """Return candidates within max_distance edit distance"""
    results = []

    for candidate in candidates:
        dist = edit_distance(query, candidate)
        if dist <= max_distance:
            results.append((candidate, dist))

    return sorted(results, key=lambda x: x[1])

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp

    return dp[n]

# Usage example
words = ["apple", "application", "apply", "maple", "ape",
         "banana", "bandana", "band"]
results = fuzzy_search("aple", words, max_distance=2)
# [('apple', 1), ('ape', 2), ('maple', 2)]
```

### 5.2 How Full-Text Search Engines Work

```python
# Simple implementation of an Inverted Index

class InvertedIndex:
    """Inverted index for full-text search"""

    def __init__(self):
        self.index = {}        # word -> [(doc_id, position), ...]
        self.documents = {}    # doc_id -> original text

    def add_document(self, doc_id, text):
        """Add a document to the index"""
        self.documents[doc_id] = text
        words = text.lower().split()

        for pos, word in enumerate(words):
            # Stemming (simple version: remove trailing 's')
            word = word.strip('.,!?;:')
            if word not in self.index:
                self.index[word] = []
            self.index[word].append((doc_id, pos))

    def search(self, query):
        """Search for documents containing the word"""
        query = query.lower()
        if query not in self.index:
            return []

        # Simple TF-IDF-like scoring
        import math
        total_docs = len(self.documents)
        results = {}

        postings = self.index[query]
        df = len(set(doc_id for doc_id, _ in postings))
        idf = math.log(total_docs / df) if df > 0 else 0

        for doc_id, pos in postings:
            if doc_id not in results:
                results[doc_id] = 0
            results[doc_id] += 1  # TF

        scored = [(doc_id, tf * idf) for doc_id, tf in results.items()]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def phrase_search(self, phrase):
        """Phrase search (search for consecutive words)"""
        words = phrase.lower().split()
        if not words or words[0] not in self.index:
            return []

        # Occurrence positions of the first word
        candidates = {}
        for doc_id, pos in self.index[words[0]]:
            if doc_id not in candidates:
                candidates[doc_id] = []
            candidates[doc_id].append(pos)

        # Check if subsequent words appear consecutively
        for i, word in enumerate(words[1:], 1):
            if word not in self.index:
                return []

            next_positions = {}
            for doc_id, pos in self.index[word]:
                if doc_id in candidates:
                    next_positions[doc_id] = next_positions.get(doc_id, [])
                    next_positions[doc_id].append(pos)

            new_candidates = {}
            for doc_id, starts in candidates.items():
                if doc_id in next_positions:
                    valid = [s for s in starts
                            if s + i in next_positions[doc_id]]
                    if valid:
                        new_candidates[doc_id] = valid

            candidates = new_candidates

        return list(candidates.keys())

# Usage example
idx = InvertedIndex()
idx.add_document(1, "The quick brown fox jumps over the lazy dog")
idx.add_document(2, "A quick brown dog outpaces the fox")
idx.add_document(3, "The fox and the dog are friends")

print(idx.search("fox"))           # [(1, ...), (2, ...), (3, ...)]
print(idx.phrase_search("quick brown"))  # [1, 2]
```

---

## 6. Practice Exercises

### Exercise 1: Pattern Matching (Basic)
Manually compute the KMP failure function for the pattern "ABABC".

### Exercise 2: Trie (Applied)
Implement an autocomplete feature using a Trie. Given an input prefix, return up to 10 candidates from the dictionary.

### Exercise 3: Boyer-Moore (Applied)
Implement the Bad Character rule of the Boyer-Moore algorithm and measure the difference in comparison count compared to the naive method.

### Exercise 4: Full-Text Search (Applied)
Implement an inverted index that supports AND/OR search. Also implement TF-IDF ranking.

### Exercise 5: Regex Engine (Advanced)
Implement a simple NFA-based regex engine (supporting only `.`, `*`, and `|`).

### Exercise 6: Palindrome Detection (Advanced)
Implement Manacher's algorithm and enumerate all palindromic substrings in a string.

### Exercise 7: Suffix Array (Advanced)
Build a suffix array + LCP array and find the longest repeated substring in a text.

### Exercise 8: String Hashing (Advanced)
Find the longest common substring of two strings in O(n log n) using rolling hash.


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify configuration file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Increased data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access privileges | Verify execution user permissions, review settings |
| Data inconsistency | Concurrency conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Procedure

1. **Check error messages**: Read the stack trace and identify the location of the error
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Incremental verification**: Verify hypotheses using log output and debuggers
5. **Fix and regression test**: After fixing, run tests on related areas as well

```python
# Debugging utilities
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
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}")
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

Steps for diagnosing performance problems:

1. **Identify bottlenecks**: Measure using profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Examine disk and network I/O status
4. **Check concurrent connections**: Inspect connection pool status

| Problem Type | Diagnostic Tool | Countermeasure |
|-------------|----------------|----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper release of references |
| I/O bottleneck | strace, iostat | Asynchronous I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |
---

## FAQ

### Q1: What does grep use internally?
**A**: GNU grep is based on the Boyer-Moore algorithm. It excels with long patterns, maximizing skip distance by comparing from the pattern end. In regex mode, it uses DFA/NFA. fgrep (fixed-string search) sometimes uses Aho-Corasick.

### Q2: What is inside a full-text search engine (e.g., Elasticsearch)?
**A**: Inverted index + BM25 scoring. It indexes the occurrence positions of each word and narrows candidates by merging indices at search time. For Japanese text, morphological analysis using MeCab/kuromoji is performed before building the index. N-gram indices are sometimes used in conjunction.

### Q3: What is the most important optimization for string processing?
**A**: (1) Avoid string concatenation (use join) (2) Precompile regex (re.compile) (3) Avoid unnecessary copies (slicing creates a new string) (4) Choose appropriate data structures (Trie, hashmap, etc.) (5) Consistency in Unicode normalization

### Q4: Which is faster, KMP or Boyer-Moore?
**A**: Generally, Boyer-Moore is faster in practice. It is especially advantageous when the alphabet is large (e.g., ASCII) and the pattern is long. In the best case, it achieves O(n/m), meaning it can complete without even reading the entire text. On the other hand, KMP guarantees O(n+m) even in the worst case and has a relatively simpler implementation. KMP may be advantageous for binary data or small alphabets (e.g., DNA).

### Q5: In what situations is Aho-Corasick used?
**A**: When searching for multiple patterns simultaneously. Examples: virus scanners (searching tens of thousands of signatures simultaneously), network IDS, text filtering (banned word lists), dictionary-based morphological analysis. For single-pattern search, KMP or Boyer-Moore is more efficient.

### Q6: When should you use a suffix array vs a suffix tree?
**A**: In practice, suffix array + LCP array is almost always sufficient. Suffix trees are theoretically powerful but consume more memory (~20n bytes) and are complex to implement. Suffix arrays use about 5n bytes and have better cache efficiency. However, suffix trees (Ukkonen's algorithm) are more suitable when you need to build an index online while incrementally adding characters.

---

## Summary

| Algorithm | Complexity | Application |
|-----------|-----------|-------------|
| Naive | O(nm) | Short texts |
| KMP | O(n+m) | Single pattern search |
| Boyer-Moore | O(n/m) best, O(n+m) | Practically fastest single pattern search |
| Rabin-Karp | O(n+m) expected | Multiple patterns |
| Aho-Corasick | O(n+m+z) | Simultaneous multi-pattern search |
| Z-Algorithm | O(n+m) | Pattern search, period detection |
| Trie | O(m) | Dictionary, autocomplete |
| Suffix Array | O(m log n) | Full-text search |
| Manacher | O(n) | Palindrome detection |
| Rolling Hash | O(1) per query | Substring comparison |

---

## Recommended Next Guides

---

## References
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapter 32: String Matching.
2. Knuth, D. E., Morris, J. H., Pratt, V. R. "Fast Pattern Matching in Strings." 1977.
3. Cox, R. "Regular Expression Matching Can Be Simple And Fast." 2007.
4. Gusfield, D. "Algorithms on Strings, Trees, and Sequences." Cambridge University Press, 1997.
5. Aho, A. V., Corasick, M. J. "Efficient string matching: an aid to bibliographic search." 1975.
6. Crochemore, M., Rytter, W. "Jewels of Stringology." World Scientific, 2002.
