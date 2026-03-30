# String Algorithms

> Systematically understand efficient string pattern searching and matching algorithms: KMP, Rabin-Karp, Z-algorithm, Trie, Aho-Corasick, and Suffix Array

## What You Will Learn in This Chapter

1. Execute single-pattern search in O(n+m) using **KMP and Z-algorithm**
2. Understand probabilistic pattern searching via rolling hash with **Rabin-Karp**
3. Efficiently implement simultaneous multi-pattern search using **Trie and Aho-Corasick**
4. Perform advanced string analysis with **Suffix Array and LCP Array**
5. Implement palindrome detection and string comparison techniques using **Manacher's algorithm and rolling hash**


## Prerequisites

The following knowledge will deepen your understanding of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Segment Tree](./01-segment-tree.md)

---

## 1. Overview of String Search Algorithms

```
┌──────────────────────────────────────────────────────┐
│            String Search Algorithms                   │
├──────────────────┬───────────────────────────────────┤
│  Single Pattern   │  Multiple Patterns                │
├──────────────────┼───────────────────────────────────┤
│ Naive O(nm)      │ Aho-Corasick O(n+m+z)             │
│ KMP O(n+m)       │ (Trie + failure function)          │
│ Rabin-Karp O(n+m)│                                    │
│ Z-algorithm O(n+m)│                                   │
│ Boyer-Moore O(n/m)│                                   │
└──────────────────┴───────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│            Structural String Analysis                 │
├──────────────────┬───────────────────────────────────┤
│  Index Structures │  Special Algorithms               │
├──────────────────┼───────────────────────────────────┤
│ Trie             │ Manacher (palindrome detection)    │
│ Suffix Array     │ Rolling Hash (string comparison)   │
│ Suffix Tree      │ Z-algorithm (general string ops)   │
│ Suffix Automaton │ LCP (Longest Common Prefix)        │
└──────────────────┴───────────────────────────────────┘

  n = text length, m = pattern length, z = number of matches
```

---

## 2. Naive Method (Brute Force)

```
Text:     ABCABCABD
Pattern:  ABCABD

Pos 0: ABCAB[C]ABD   <- C != D, mismatch
       ABCAB[D]

Pos 1: ABCABCABD     <- B != A, mismatch
        A

Pos 2: ABCABCABD     <- C != A, mismatch
         A

Pos 3: ABCABCABD     <- Full match!
          ABCABD
```

```python
def naive_search(text: str, pattern: str) -> list:
    """Naive method - O(nm)"""
    n, m = len(text), len(pattern)
    matches = []
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            matches.append(i)
    return matches

print(naive_search("ABCABCABD", "ABCABD"))  # [3]
```

### Worst Case for the Naive Method

```python
# Worst case: text = "AAA...A" (n chars), pattern = "AA...AB" (m chars)
# Matches m-1 characters at each position before mismatch -> O(nm)
text = "A" * 10000
pattern = "A" * 999 + "B"
# Naive: 9001 * 999 ~ 9 * 10^6 comparisons

# Practical improvement: Python's `in` operator and str.find() use
# a Boyer-Moore-Horspool variant, which is faster than naive
idx = text.find(pattern)  # Internally optimized
```

---

## 3. KMP (Knuth-Morris-Pratt)

Pre-computes "prefix-suffix match" information (failure function / partial match table) of the pattern, and efficiently shifts the pattern on mismatch.

```
Building the failure function (LPS array):
Pattern: A B C A B D
LPS:     [0, 0, 0, 1, 2, 0]

  "A"      -> no prefix = suffix -> 0
  "AB"     -> no prefix = suffix -> 0
  "ABC"    -> no prefix = suffix -> 0
  "ABCA"   -> "A" = "A"         -> 1
  "ABCAB"  -> "AB" = "AB"       -> 2
  "ABCABD" -> no prefix = suffix -> 0

Using on mismatch:
Text:    ...ABCAB|C|ABD...
Pattern:    ABCAB|D|
            ^ LPS[4]=2 -> 2 characters already matched

After shift:
Text:    ...ABCAB|C|ABD...
Pattern:       AB|C|ABD
               ^ Resume from position 2
```

```python
def kmp_search(text: str, pattern: str) -> list:
    """KMP - O(n + m)"""

    def build_lps(pattern: str) -> list:
        """Build the failure function (LPS array) - O(m)"""
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1

        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    n, m = len(text), len(pattern)
    if m == 0:
        return []
    lps = build_lps(pattern)
    matches = []
    i = j = 0  # i: text, j: pattern

    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1

        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]  # Shift pattern
            else:
                i += 1

    return matches

text = "AABAACAADAABAABA"
pattern = "AABA"
print(kmp_search(text, pattern))  # [0, 9, 12]
```

### KMP Application: Minimum Period of a String

```python
def min_period(s: str) -> int:
    """Find the minimum period of a string using KMP's failure function
    Example: "abcabc" has minimum period 3 ("abc")
    Example: "aaaaaa" has minimum period 1 ("a")
    """
    n = len(s)
    lps = [0] * n
    length = 0
    i = 1

    while i < n:
        if s[i] == s[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                i += 1

    # Minimum period = n - lps[n-1]
    period = n - lps[n - 1]

    # Only a complete period if n is divisible by the period
    if n % period == 0:
        return period
    else:
        return n  # No periodic structure (the string itself is the minimum period)

print(min_period("abcabc"))    # 3
print(min_period("aaaaaa"))    # 1
print(min_period("abcab"))     # 5 (not a complete period)
print(min_period("abababab"))  # 2
```

### KMP Application: Occurrence Count of All Prefixes in a String

```python
def count_prefix_occurrences(s: str) -> list:
    """Count how many times each prefix s[0..k] appears in string s
    Return: count[k] = number of occurrences of s[0..k]
    """
    n = len(s)
    lps = [0] * n
    length = 0
    i = 1

    while i < n:
        if s[i] == s[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                i += 1

    # Count occurrences of each prefix
    count = [0] * (n + 1)
    for i in range(n):
        count[lps[i]] += 1

    # Accumulate along the LPS chain
    for i in range(n - 1, 0, -1):
        count[lps[i - 1]] += count[i]

    # Each prefix occurs at least once (as itself)
    for i in range(n + 1):
        count[i] += 1

    return count[1:]  # count[1] through count[n]

print(count_prefix_occurrences("ababa"))
# "a":3, "ab":2, "aba":2, "abab":1, "ababa":1
# -> [3, 2, 2, 1, 1]
```

---

## 4. Rabin-Karp

Uses rolling hash to compare substrings by their hash values. Performs actual string comparison only on hash collision.

```
Text: "ABCDE"  Pattern: "BCD" (hash=h)

Sliding window:
  "ABC" -> hash("ABC") != h -> skip
  "BCD" -> hash("BCD") = h  -> string compare -> match!
  "CDE" -> hash("CDE") != h -> skip

Rolling hash:
  hash("BCD") = (hash("ABC") - 'A' * base^2) * base + 'D'
  -> O(1) to compute next hash
```

```python
def rabin_karp(text: str, pattern: str, prime: int = 101) -> list:
    """Rabin-Karp - average O(n+m), worst O(nm)"""
    n, m = len(text), len(pattern)
    if m > n:
        return []
    base = 256
    matches = []

    # Hash of pattern and first window
    p_hash = 0
    t_hash = 0
    h = pow(base, m - 1, prime)  # base^(m-1) mod prime

    for i in range(m):
        p_hash = (base * p_hash + ord(pattern[i])) % prime
        t_hash = (base * t_hash + ord(text[i])) % prime

    for i in range(n - m + 1):
        if p_hash == t_hash:
            # Hash match -> verify with actual string comparison (eliminate false positives)
            if text[i:i + m] == pattern:
                matches.append(i)

        if i < n - m:
            # Rolling hash: remove leading char, add trailing char
            t_hash = (base * (t_hash - ord(text[i]) * h) +
                      ord(text[i + m])) % prime
            if t_hash < 0:
                t_hash += prime

    return matches

print(rabin_karp("AABAACAADAABAABA", "AABA"))  # [0, 9, 12]
```

### Rolling Hash (General-Purpose Version)

```python
class RollingHash:
    """General-purpose rolling hash (with double hashing support)
    Computes the hash of any substring in O(1)
    """

    def __init__(self, s: str, base1: int = 131, mod1: int = 10**9 + 7,
                 base2: int = 137, mod2: int = 10**9 + 9):
        self.n = len(s)
        self.mod1, self.mod2 = mod1, mod2

        # Precompute hashes
        self.hash1 = [0] * (self.n + 1)
        self.hash2 = [0] * (self.n + 1)
        self.pow1 = [1] * (self.n + 1)
        self.pow2 = [1] * (self.n + 1)

        for i in range(self.n):
            self.hash1[i + 1] = (self.hash1[i] * base1 + ord(s[i])) % mod1
            self.hash2[i + 1] = (self.hash2[i] * base2 + ord(s[i])) % mod2
            self.pow1[i + 1] = self.pow1[i] * base1 % mod1
            self.pow2[i + 1] = self.pow2[i] * base2 % mod2

    def get_hash(self, l: int, r: int) -> tuple:
        """Return the hash of substring s[l:r] (half-open interval) - O(1)"""
        h1 = (self.hash1[r] - self.hash1[l] * self.pow1[r - l]) % self.mod1
        h2 = (self.hash2[r] - self.hash2[l] * self.pow2[r - l]) % self.mod2
        return (h1, h2)

    def lcp(self, i: int, j: int) -> int:
        """Length of longest common prefix starting at positions i and j - O(log n)"""
        lo, hi = 0, min(self.n - i, self.n - j)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.get_hash(i, i + mid) == self.get_hash(j, j + mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

# Usage example
rh = RollingHash("abcabc")
print(rh.get_hash(0, 3) == rh.get_hash(3, 6))  # True ("abc" == "abc")
print(rh.get_hash(0, 3) == rh.get_hash(1, 4))  # False ("abc" != "bca")
print(rh.lcp(0, 3))  # 3 (common prefix of "abc" and "abc")

# Application: palindrome check
def is_palindrome_hash(s: str, l: int, r: int) -> bool:
    """Check if s[l:r] is a palindrome using hashing"""
    rh_forward = RollingHash(s)
    rh_reverse = RollingHash(s[::-1])
    n = len(s)
    h_forward = rh_forward.get_hash(l, r)
    h_reverse = rh_reverse.get_hash(n - r, n - l)
    return h_forward == h_reverse
```

### Rabin-Karp Application: Count of Unique Substrings

```python
def count_unique_substrings(s: str) -> int:
    """Count the number of unique substrings using rolling hash
    O(n^2): add hash of every substring to a set
    """
    rh = RollingHash(s)
    unique = set()
    n = len(s)

    for length in range(1, n + 1):
        for i in range(n - length + 1):
            h = rh.get_hash(i, i + length)
            unique.add(h)

    return len(unique)

print(count_unique_substrings("abc"))   # 6: a,b,c,ab,bc,abc
print(count_unique_substrings("aaa"))   # 3: a,aa,aaa
```

---

## 5. Z-algorithm

The Z-array Z[i] = "the maximum length of the substring starting at position i that matches a prefix of the entire string," computed in O(n).

```
String:   a a b x a a b
Z-array: [-, 1, 0, 0, 3, 1, 0]

Z[1]=1: "abxaab" vs "aabxaab" -> "a" matches -> 1
Z[4]=3: "aab" vs "aabxaab" -> "aab" matches -> 3

Application to pattern search:
  S = pattern + "$" + text
  Positions where Z[i] == len(pattern) are match positions
```

```python
def z_function(s: str) -> list:
    """Compute the Z-array - O(n)"""
    n = len(s)
    z = [0] * n
    z[0] = n
    l, r = 0, 0  # Z-box [l, r)

    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] > r:
            l, r = i, i + z[i]

    return z

def z_search(text: str, pattern: str) -> list:
    """Pattern search using Z-algorithm - O(n+m)"""
    combined = pattern + "$" + text
    z = z_function(combined)
    m = len(pattern)

    matches = []
    for i in range(m + 1, len(combined)):
        if z[i] == m:
            matches.append(i - m - 1)

    return matches

print(z_search("AABAACAADAABAABA", "AABA"))  # [0, 9, 12]
```

### Z-algorithm Applications

```python
def z_min_period(s: str) -> int:
    """Find the minimum period using the Z-array"""
    n = len(s)
    z = z_function(s)

    for i in range(1, n):
        if i + z[i] == n and n % i == 0:
            return i
    return n

print(z_min_period("abcabc"))    # 3
print(z_min_period("abababab"))  # 2

def z_count_distinct_substrings(s: str) -> int:
    """Count distinct substrings using Z-array - O(n^2)
    Uses the Z-array of each suffix to count "newly added substrings"
    """
    n = len(s)
    count = 0

    # Add one character at a time from the right
    current = ""
    for i in range(n - 1, -1, -1):
        current = s[i] + current
        z = z_function(current)
        # Find the maximum Z[j] (j >= 1)
        max_z = max(z[1:]) if len(z) > 1 else 0
        # Number of new substrings = current length - max Z value
        count += len(current) - max_z

    return count

print(z_count_distinct_substrings("abc"))   # 6
print(z_count_distinct_substrings("aaa"))   # 3
```

---

## 6. Trie

A tree structure for efficiently storing and searching a set of strings. Each edge corresponds to a single character.

```
Word set: {"app", "apple", "apply", "apt", "bat", "bath"}

Trie:
        root
       /    \
      a      b
      |      |
      p      a
     / \     |
    p   t    t
   / \       |
  l   [e]   h
  |         |
 [e] [y]  [NULL]

  [] = end of word

Search "app" -> root->a->p->p -> end marker present -> found
Search "api" -> root->a->p->? (no child for 'i') -> not found
Prefix search "ap" -> root->a->p -> return all words in this subtree
```

```python
class TrieNode:
    __slots__ = ['children', 'is_end', 'count', 'word']

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # Number of words with this prefix
        self.word = None  # Stores the original word at terminal nodes

class Trie:
    """Trie"""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert a word - O(m)"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True
        node.word = word

    def search(self, word: str) -> bool:
        """Exact match search - O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """Prefix search - O(m)"""
        return self._find_node(prefix) is not None

    def count_prefix(self, prefix: str) -> int:
        """Count words with the given prefix"""
        node = self._find_node(prefix)
        return node.count if node else 0

    def _find_node(self, prefix: str) -> TrieNode:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def auto_complete(self, prefix: str, limit: int = 10) -> list:
        """Autocomplete"""
        node = self._find_node(prefix)
        if not node:
            return []

        results = []
        def dfs(node, path):
            if len(results) >= limit:
                return
            if node.is_end:
                results.append(prefix + path)
            for char, child in sorted(node.children.items()):
                dfs(child, path + char)

        dfs(node, "")
        return results

    def delete(self, word: str) -> bool:
        """Delete a word - O(m)"""
        def _delete(node, word, depth):
            if depth == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                node.word = None
                return len(node.children) == 0
            char = word[depth]
            if char not in node.children:
                return False

            child = node.children[char]
            child.count -= 1
            should_delete = _delete(child, word, depth + 1)

            if should_delete:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end

            return False

        return _delete(self.root, word, 0)

    def longest_common_prefix(self) -> str:
        """Longest common prefix of all words stored in the Trie"""
        prefix = []
        node = self.root
        while len(node.children) == 1 and not node.is_end:
            char = next(iter(node.children))
            prefix.append(char)
            node = node.children[char]
        return ''.join(prefix)

# Usage example
trie = Trie()
for word in ["app", "apple", "apply", "apt", "bat", "bath"]:
    trie.insert(word)

print(trie.search("apple"))       # True
print(trie.search("api"))         # False
print(trie.starts_with("ap"))     # True
print(trie.count_prefix("ap"))    # 4
print(trie.auto_complete("ap"))   # ['app', 'apple', 'apply', 'apt']
```

### Array-Based High-Speed Trie

```python
class ArrayTrie:
    """Array-based Trie (high-speed version, lowercase English letters only)
    Designed for competitive programming
    """

    def __init__(self, max_nodes: int = 500001):
        self.ALPHA = 26
        self.children = [[0] * self.ALPHA for _ in range(max_nodes)]
        self.is_end = [False] * max_nodes
        self.node_count = 1  # 0 is the root

    def _char_to_idx(self, c: str) -> int:
        return ord(c) - ord('a')

    def insert(self, word: str) -> None:
        node = 0
        for c in word:
            idx = self._char_to_idx(c)
            if self.children[node][idx] == 0:
                self.children[node][idx] = self.node_count
                self.node_count += 1
            node = self.children[node][idx]
        self.is_end[node] = True

    def search(self, word: str) -> bool:
        node = 0
        for c in word:
            idx = self._char_to_idx(c)
            if self.children[node][idx] == 0:
                return False
            node = self.children[node][idx]
        return self.is_end[node]
```

---

## 7. Aho-Corasick

Combines Trie with KMP's failure function to search for multiple patterns simultaneously.

```
Patterns: {"he", "she", "his", "hers"}
Text: "ahishers"

Trie + failure links:
     root --> h --> e --> r --> s
      |        |
      |        +--> i --> s
      |
      +--> s --> h --> e
               (failure link -> root.h)

Search results:
  Position 1: "his" (i=1)
  Position 3: "she" (i=3), "he" (i=4)
  Position 4: "hers" (i=4)
```

```python
from collections import deque, defaultdict

class AhoCorasick:
    """Aho-Corasick - simultaneous multi-pattern search"""

    def __init__(self):
        self.goto = [{}]         # goto function
        self.fail = [0]          # failure function
        self.output = [[]]       # output function
        self.state_count = 1

    def _add_state(self):
        self.goto.append({})
        self.fail.append(0)
        self.output.append([])
        state = self.state_count
        self.state_count += 1
        return state

    def add_pattern(self, pattern: str, pattern_id=None):
        """Add a pattern"""
        if pattern_id is None:
            pattern_id = pattern

        state = 0
        for char in pattern:
            if char not in self.goto[state]:
                self.goto[state][char] = self._add_state()
            state = self.goto[state][char]
        self.output[state].append(pattern_id)

    def build(self):
        """Build the failure function - O(total pattern characters)"""
        queue = deque()

        # Set failure function for depth-1 states
        for char, state in self.goto[0].items():
            self.fail[state] = 0
            queue.append(state)

        # Build via BFS
        while queue:
            r = queue.popleft()
            for char, s in self.goto[r].items():
                queue.append(s)

                state = self.fail[r]
                while state != 0 and char not in self.goto[state]:
                    state = self.fail[state]

                self.fail[s] = self.goto[state].get(char, 0)
                if self.fail[s] == s:
                    self.fail[s] = 0

                self.output[s] = self.output[s] + self.output[self.fail[s]]

    def search(self, text: str) -> list:
        """Search for all patterns in text - O(n + m + z)"""
        state = 0
        results = []

        for i, char in enumerate(text):
            while state != 0 and char not in self.goto[state]:
                state = self.fail[state]

            state = self.goto[state].get(char, 0)

            for pattern in self.output[state]:
                if isinstance(pattern, str):
                    results.append((i - len(pattern) + 1, pattern))
                else:
                    results.append((i, pattern))

        return results

# Usage example
ac = AhoCorasick()
patterns = ["he", "she", "his", "hers"]
for p in patterns:
    ac.add_pattern(p)
ac.build()

results = ac.search("ahishers")
print(results)
# [(1, 'his'), (3, 'she'), (4, 'he'), (4, 'hers')]
```

### Aho-Corasick Application: Forbidden Word Filtering

```python
def censor_text(text: str, forbidden_words: list) -> str:
    """Replace forbidden words in text with '*'
    Uses Aho-Corasick for efficient batch detection
    """
    ac = AhoCorasick()
    for word in forbidden_words:
        ac.add_pattern(word.lower())
    ac.build()

    text_lower = text.lower()
    matches = ac.search(text_lower)

    # Calculate ranges to replace
    censored = list(text)
    for pos, pattern in matches:
        for i in range(pos, pos + len(pattern)):
            censored[i] = '*'

    return ''.join(censored)

text = "This is a bad example with some ugly words"
forbidden = ["bad", "ugly"]
print(censor_text(text, forbidden))
# "This is a *** example with some **** words"
```

---

## 8. Suffix Array

A sorted array of all suffixes of a string. Applicable to many problems including pattern search and longest common substring.

```
String: "banana"

List of suffixes:
  0: banana
  1: anana
  2: nana
  3: ana
  4: na
  5: a

After sorting:
  5: a
  3: ana
  1: anana
  0: banana
  4: na
  2: nana

Suffix Array: [5, 3, 1, 0, 4, 2]
```

```python
def build_suffix_array(s: str) -> list:
    """Build Suffix Array - O(n log^2 n)"""
    n = len(s)
    # Sort by (rank, next rank, index)
    sa = list(range(n))
    rank = [ord(c) for c in s]
    tmp = [0] * n

    k = 1
    while k < n:
        def compare(i, j):
            if rank[i] != rank[j]:
                return -1 if rank[i] < rank[j] else 1
            ri = rank[i + k] if i + k < n else -1
            rj = rank[j + k] if j + k < n else -1
            return -1 if ri < rj else (0 if ri == rj else 1)

        from functools import cmp_to_key
        sa.sort(key=cmp_to_key(compare))

        # Compute new ranks
        tmp[sa[0]] = 0
        for i in range(1, n):
            tmp[sa[i]] = tmp[sa[i-1]]
            if compare(sa[i-1], sa[i]) < 0:
                tmp[sa[i]] += 1

        rank = tmp[:]
        if rank[sa[n-1]] == n - 1:
            break
        k *= 1 if k == 0 else k  # Double k
        k *= 2

    return sa

def build_suffix_array_simple(s: str) -> list:
    """Build Suffix Array (simple version) - O(n log n * m)
    A concise implementation leveraging Python's sort
    """
    n = len(s)
    return sorted(range(n), key=lambda i: s[i:])

# LCP Array (Kasai's Algorithm)
def build_lcp_array(s: str, sa: list) -> list:
    """Build LCP Array (Kasai's Algorithm) - O(n)
    lcp[i] = length of longest common prefix between sa[i] and sa[i+1]
    """
    n = len(s)
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i

    lcp = [0] * (n - 1)
    h = 0

    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i] - 1] = h
            if h > 0:
                h -= 1
        else:
            h = 0

    return lcp

# Usage example
s = "banana"
sa = build_suffix_array_simple(s)
print(f"SA: {sa}")  # [5, 3, 1, 0, 4, 2]

lcp = build_lcp_array(s, sa)
print(f"LCP: {lcp}")  # [1, 3, 0, 0, 2]

# Display suffixes
for i, idx in enumerate(sa):
    prefix = f"lcp={lcp[i]}" if i < len(lcp) else ""
    print(f"  sa[{i}]={idx}: {s[idx]:10s}  {prefix}")
```

### Suffix Array Applications

```python
def search_with_suffix_array(text: str, pattern: str) -> list:
    """Pattern search using Suffix Array - O(m log n)"""
    sa = build_suffix_array_simple(text)
    n = len(text)
    m = len(pattern)

    # Binary search for lower bound
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if text[sa[mid]:sa[mid] + m] < pattern:
            lo = mid + 1
        else:
            hi = mid
    left = lo

    # Binary search for upper bound
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if text[sa[mid]:sa[mid] + m] <= pattern:
            lo = mid + 1
        else:
            hi = mid
    right = lo

    return sorted(sa[left:right])

text = "banana"
print(search_with_suffix_array(text, "ana"))  # [1, 3]
print(search_with_suffix_array(text, "ban"))  # [0]

def longest_repeated_substring(s: str) -> str:
    """Longest repeated substring (longest substring appearing 2+ times)
    Uses the maximum value in the LCP Array
    """
    if len(s) <= 1:
        return ""

    sa = build_suffix_array_simple(s)
    lcp = build_lcp_array(s, sa)

    if not lcp:
        return ""

    max_lcp = max(lcp)
    max_idx = lcp.index(max_lcp)

    return s[sa[max_idx]:sa[max_idx] + max_lcp]

print(longest_repeated_substring("banana"))    # "ana"
print(longest_repeated_substring("abcabc"))    # "abc"
print(longest_repeated_substring("aabaaab"))   # "aab"

def count_distinct_substrings_sa(s: str) -> int:
    """Count distinct substrings using Suffix Array + LCP - O(n log n)
    Total substrings n*(n+1)/2 minus duplicates (sum of LCP)
    """
    n = len(s)
    sa = build_suffix_array_simple(s)
    lcp = build_lcp_array(s, sa)

    total = n * (n + 1) // 2
    duplicates = sum(lcp)
    return total - duplicates

print(count_distinct_substrings_sa("abc"))    # 6
print(count_distinct_substrings_sa("aaa"))    # 3
print(count_distinct_substrings_sa("banana")) # 15
```

---

## 9. Manacher's Algorithm (Palindrome Detection)

An algorithm that finds the longest palindromic substring in O(n).

```python
def manacher(s: str) -> list:
    """Manacher's algorithm - computes the radius of the longest palindrome
    centered at each position. O(n)

    Transforms the input string to handle even-length palindromes uniformly:
    "abba" -> "#a#b#b#a#"
    """
    # Insert dummy characters between each character
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n  # p[i] = radius of palindrome centered at position i

    c = r = 0  # Center and right boundary of the rightmost palindrome

    for i in range(n):
        # Mirror position
        mirror = 2 * c - i

        if i < r:
            p[i] = min(r - i, p[mirror])

        # Attempt to expand
        while (i + p[i] + 1 < n and i - p[i] - 1 >= 0
               and t[i + p[i] + 1] == t[i - p[i] - 1]):
            p[i] += 1

        # Update if past right boundary
        if i + p[i] > r:
            c, r = i, i + p[i]

    return p

def longest_palindrome_substring(s: str) -> str:
    """Longest palindromic substring - O(n)"""
    if not s:
        return ""

    t = '#' + '#'.join(s) + '#'
    p = manacher(s)

    # Find the longest palindrome in the transformed string
    # Compute directly from the p array
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p2 = [0] * n
    c = r = 0

    for i in range(n):
        mirror = 2 * c - i
        if i < r:
            p2[i] = min(r - i, p2[mirror])
        while (i + p2[i] + 1 < n and i - p2[i] - 1 >= 0
               and t[i + p2[i] + 1] == t[i - p2[i] - 1]):
            p2[i] += 1
        if i + p2[i] > r:
            c, r = i, i + p2[i]

    # Position of maximum radius
    max_len = max(p2)
    center = p2.index(max_len)

    # Starting position and length in the original string
    start = (center - max_len) // 2
    return s[start:start + max_len]

print(longest_palindrome_substring("babad"))     # "bab" or "aba"
print(longest_palindrome_substring("cbbd"))      # "bb"
print(longest_palindrome_substring("abacaba"))   # "abacaba"

def count_palindromic_substrings(s: str) -> int:
    """Total number of palindromic substrings - O(n)"""
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n
    c = r = 0

    for i in range(n):
        mirror = 2 * c - i
        if i < r:
            p[i] = min(r - i, p[mirror])
        while (i + p[i] + 1 < n and i - p[i] - 1 >= 0
               and t[i + p[i] + 1] == t[i - p[i] - 1]):
            p[i] += 1
        if i + p[i] > r:
            c, r = i, i + p[i]

    # Palindromes at '#' positions correspond to even-length palindromes
    # Palindromes at character positions correspond to odd-length palindromes
    count = 0
    for i in range(n):
        # Number of palindromes contained within p[i]
        if t[i] == '#':
            count += p[i] // 2  # Even-length
        else:
            count += (p[i] + 1) // 2  # Odd-length

    return count

print(count_palindromic_substrings("abc"))   # 3 (a, b, c)
print(count_palindromic_substrings("aaa"))   # 6 (a,a,a,aa,aa,aaa)
```

---

## 10. Boyer-Moore Overview

The fastest string search algorithm in practice. Compares the pattern from right to left and shifts by large amounts on mismatch.

```python
def boyer_moore_horspool(text: str, pattern: str) -> list:
    """Boyer-Moore-Horspool (simplified version)
    Average O(n/m), worst O(nm)
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []

    # Build Bad Character table
    bad_char = {}
    for i in range(m - 1):
        bad_char[pattern[i]] = m - 1 - i

    matches = []
    i = 0

    while i <= n - m:
        j = m - 1
        while j >= 0 and text[i + j] == pattern[j]:
            j -= 1

        if j < 0:
            matches.append(i)
            i += 1
        else:
            # Shift using Bad Character Rule
            skip = bad_char.get(text[i + m - 1], m)
            i += max(1, skip)

    return matches

print(boyer_moore_horspool("ABCABCABD", "ABCABD"))  # [3]

# Why Boyer-Moore is the fastest in practice:
# - Compares pattern right-to-left -> mismatches detected early
# - Bad Character Rule: if mismatch char not in pattern, shift by m
# - Good Suffix Rule: leverages matching portions within the pattern
# - Average O(n/m) comparisons for English text and similar inputs
```

---

## 11. Algorithm Comparison Table

| Algorithm | Preprocessing | Search | Space | Characteristics |
|:---|:---|:---|:---|:---|
| Naive | O(1) | O(nm) | O(1) | Simplest implementation |
| KMP | O(m) | O(n) | O(m) | Deterministic, worst-case O(n) |
| Rabin-Karp | O(m) | Avg O(n) | O(1) | Supports multiple patterns |
| Z-algorithm | O(n+m) | O(n+m) | O(n+m) | Versatility of Z-array |
| Boyer-Moore | O(m+\|S\|) | Avg O(n/m) | O(m+\|S\|) | Fastest in practice |
| Aho-Corasick | O(Sm) | O(n+z) | O(Sm) | Fastest for multiple patterns |
| Suffix Array | O(n log n) | O(m log n) | O(n) | Strong for multiple searches |

## Selection Guide by Use Case

| Use Case | Recommended | Reason |
|:---|:---|:---|
| Single pattern (guaranteed) | KMP | Worst-case O(n+m) guarantee |
| Single pattern (simple) | Rabin-Karp | Concise implementation |
| Single pattern (fast) | Boyer-Moore | Fastest on average |
| Simultaneous multi-pattern | Aho-Corasick | Optimal with Trie + failure function |
| Prefix matching | Trie | Best for autocomplete |
| Text editor search | Boyer-Moore | Fastest in practice |
| DNA sequence search | Suffix Array | Strong for bulk searches |
| Palindrome detection | Manacher | Detects all palindromes in O(n) |
| String comparison (many) | Rolling Hash | O(1) substring comparison |

---

## 12. Practical Application Patterns

### Log Analysis (Multi-Pattern Detection)

```python
def analyze_logs(log_lines: list, error_patterns: list) -> dict:
    """Batch detection of error patterns in log files"""
    ac = AhoCorasick()
    for i, pattern in enumerate(error_patterns):
        ac.add_pattern(pattern.lower(), i)
    ac.build()

    results = {p: [] for p in error_patterns}

    for line_num, line in enumerate(log_lines, 1):
        matches = ac.search(line.lower())
        for _, pattern_id in matches:
            results[error_patterns[pattern_id]].append(line_num)

    return results

logs = [
    "2024-01-15 INFO Server started",
    "2024-01-15 ERROR Connection timeout",
    "2024-01-15 WARN Memory usage high",
    "2024-01-15 ERROR Disk full",
    "2024-01-15 ERROR Connection refused",
]
patterns = ["error", "timeout", "memory"]
result = analyze_logs(logs, patterns)
print(result)
# {'error': [2, 4, 5], 'timeout': [2], 'memory': [3]}
```

### DNA Sequence Analysis

```python
def find_motifs(dna: str, motifs: list) -> dict:
    """Batch search for motifs (specific patterns) in a DNA sequence"""
    ac = AhoCorasick()
    for motif in motifs:
        ac.add_pattern(motif)
    ac.build()

    results = {m: [] for m in motifs}
    matches = ac.search(dna)

    for pos, motif in matches:
        results[motif].append(pos)

    return results

dna = "ATCGATCGATCGATCG"
motifs = ["ATC", "GAT", "ATCG"]
print(find_motifs(dna, motifs))
# {'ATC': [0, 4, 8, 12], 'GAT': [3, 7, 11], 'ATCG': [0, 4, 8, 12]}
```

### Spell Checker

```python
def spell_checker(dictionary: list, word: str, max_distance: int = 1) -> list:
    """Spell checker using Trie (edit distance based)"""
    trie = Trie()
    for w in dictionary:
        trie.insert(w)

    suggestions = []

    def search_recursive(node, char_idx, word, current_row, results):
        columns = len(word) + 1
        if node.is_end and current_row[-1] <= max_distance:
            results.append((node.word, current_row[-1]))

        for char, child_node in node.children.items():
            new_row = [current_row[0] + 1]
            for col in range(1, columns):
                insert_cost = new_row[col - 1] + 1
                delete_cost = current_row[col] + 1
                replace_cost = current_row[col - 1]
                if word[col - 1] != char:
                    replace_cost += 1
                new_row.append(min(insert_cost, delete_cost, replace_cost))

            if min(new_row) <= max_distance:
                search_recursive(child_node, char_idx, word, new_row, results)

    current_row = list(range(len(word) + 1))
    for char, node in trie.root.children.items():
        new_row = [1]
        for col in range(1, len(word) + 1):
            insert_cost = new_row[col - 1] + 1
            delete_cost = current_row[col] + 1
            replace_cost = current_row[col - 1]
            if word[col - 1] != char:
                replace_cost += 1
            new_row.append(min(insert_cost, delete_cost, replace_cost))

        if min(new_row) <= max_distance:
            search_recursive(node, 0, word, new_row, suggestions)

    return sorted(suggestions, key=lambda x: x[1])

dictionary = ["apple", "apply", "app", "ape", "maple", "application"]
print(spell_checker(dictionary, "aple"))
# [('ape', 1), ('apple', 1)]
```

---

## 13. Anti-Patterns

### Anti-Pattern 1: Overuse of the Naive Method

```python
# BAD: Repeatedly using naive method on large text
text = "A" * 1000000
pattern = "A" * 999
# O(n*m) = O(10^12) -> timeout

# GOOD: Use KMP or Z-algorithm
matches = kmp_search(text, pattern)  # O(n+m) = O(10^6)
```

### Anti-Pattern 2: Searching Multiple Patterns Individually

```python
# BAD: Searching k patterns individually with KMP -> O(k * (n+m))
patterns = ["pattern1", "pattern2", "patternK"]
for p in patterns:
    kmp_search(text, p)  # Repeated k times

# GOOD: Batch search with Aho-Corasick -> O(n + Sm + z)
ac = AhoCorasick()
for p in patterns:
    ac.add_pattern(p)
ac.build()
results = ac.search(text)  # Single pass
```

### Anti-Pattern 3: Ignoring Hash Collisions

```python
# BAD: Using a single hash without verifying false positives
def bad_rabin_karp(text, pattern):
    # Immediately report match when hash matches
    if hash_match:
        matches.append(i)  # Potential false positive!

# GOOD: Use double hashing or verify with string comparison on hash match
def good_rabin_karp(text, pattern):
    if hash_match:
        if text[i:i+m] == pattern:  # Verify
            matches.append(i)
```

### Anti-Pattern 4: Not Considering Trie Memory Consumption

```python
# BAD: Storing many long strings in a dict-based Trie
# Each node holds a dict -> high memory consumption

# GOOD:
# 1. Array-based Trie (when character set is limited)
# 2. Double-Array Trie (memory-efficient)
# 3. Patricia Trie / Radix Tree (compresses common prefixes)
```


---

## Exercises

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
        """Validate input"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main processing logic"""
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
        assert False, "Should have raised an exception"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Applied Patterns

Extend the basic implementation to add the following functionality.

```python
# Exercise 2: Applied patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for applied patterns"""

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
    print("All applied tests passed!")

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

    print(f"Inefficient: {slow_time:.4f}s")
    print(f"Efficient:   {fast_time:.6f}s")
    print(f"Speedup:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be aware of algorithm time complexity
- Choose appropriate data structures
- Measure effectiveness with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Configuration file issues | Check configuration file path and format |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Check user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check error messages**: Read the stack trace to identify the location
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Form hypotheses**: List possible causes
4. **Verify incrementally**: Use logging and debuggers to test hypotheses
5. **Fix and regression test**: After fixing, run tests on related areas

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
    """Decorator that logs function inputs and outputs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception: {func.__name__}: {e}")
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

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Verify absence of memory leaks
3. **Check I/O waits**: Examine disk and network I/O conditions
4. **Check concurrent connections**: Verify connection pool status

| Problem Type | Diagnostic Tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper release of references |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |
---

## 14. FAQ

### Q1: Which is better, KMP or Z-algorithm?

**A:** Both have the same time complexity O(n+m). KMP can perform online processing (stream-compatible) via its failure function. Z-algorithm has a more intuitive array construction and is easier to apply to general string problems. In competitive programming, Z-algorithm is popular for its versatility.

### Q2: How do you mitigate hash collisions in Rabin-Karp?

**A:** (1) Use a large prime p. (2) Use multiple hash functions (double hashing). (3) Verify with string comparison on hash match. The theoretical worst case is O(nm), but with a good hash function, it is practically O(n+m).

### Q3: What are Suffix Array / Suffix Tree?

**A:** A Suffix Array is a sorted array of all suffixes, constructible in O(n log n) with O(m log n) pattern search. A Suffix Tree stores all suffixes in a Trie and can be built in O(n), but has high memory consumption. Both are effective when processing many search queries.

### Q4: What should I watch out for with rolling hash?

**A:** (1) Small MOD values lead to high collision rates (10^9+7 or larger recommended). (2) Ideally, the base should be a random value independent of input. (3) Double hashing reduces collision probability to 1/(mod1 * mod2). (4) Watch for negative hash values (Python handles % to return positive, but other languages require care).

### Q5: What string search algorithm is most commonly used in practice?

**A:** Most programming language standard libraries use a Boyer-Moore variant (Python's `str.find()`, grep's default algorithm, etc.). For security applications (IDS pattern matching, etc.), Aho-Corasick is widely used.

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying its behavior.

### Q2: What common mistakes do beginners make?

Skipping fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## 15. Summary

| Topic | Key Points |
|:---|:---|
| KMP | Optimizes shift distance with failure function. O(n+m) guarantee |
| Rabin-Karp | Fast comparison with rolling hash. Supports multiple patterns |
| Z-algorithm | Computes prefix match lengths with Z-array. Highly versatile |
| Boyer-Moore | Compares right-to-left. Fastest in practice |
| Trie | Tree structure for string sets. Prefix search and autocomplete |
| Aho-Corasick | Trie + failure function for multi-pattern search in O(n+Sm+z) |
| Suffix Array | Sorted array of all suffixes. Powerful combined with LCP |
| Manacher | Palindrome detection in O(n) |
| Rolling Hash | O(1) hash comparison of substrings |

---

## Recommended Next Guides

- [Segment Tree](./01-segment-tree.md) -- Another advanced data structure
- [Network Flow](./03-network-flow.md) -- Optimization on graphs
- [Competitive Programming](../04-practice/01-competitive-programming.md) -- String problems in practice

---

## References

1. Knuth, D. E., Morris, J. H., & Pratt, V. R. (1977). "Fast Pattern Matching in Strings." *SIAM Journal on Computing*.
2. Karp, R. M. & Rabin, M. O. (1987). "Efficient Randomized Pattern-Matching Algorithms." *IBM Journal of Research and Development*.
3. Aho, A. V. & Corasick, M. J. (1975). "Efficient String Matching." *Communications of the ACM*.
4. Gusfield, D. (1997). *Algorithms on Strings, Trees, and Sequences*. Cambridge University Press.
5. Manacher, G. (1975). "A New Linear-Time On-Line Algorithm for Finding the Smallest Initial Palindrome of a String." *JACM*.
6. Kasai, T. et al. (2001). "Linear-Time Longest-Common-Prefix Computation in Suffix Arrays and Its Applications." *CPM*.
