# Performance -- ReDoS, Backtracking Explosion, and Optimization

> Performance issues in regular expressions can have serious consequences, ranging from security vulnerabilities (ReDoS) to complete service outages. This guide explains the principles of backtracking explosion, and how to design safe and fast patterns.

## What You Will Learn in This Chapter

1. **The Principles of ReDoS (Regular Expression Denial of Service)** -- Why certain patterns become exponentially slow
2. **Detecting and Avoiding Backtracking Explosion** -- How to identify dangerous patterns and their safe alternatives
3. **Regex Optimization Techniques** -- Compilation, anchors, negated character classes, possessive quantifiers
4. **Understanding Engine-Specific Characteristics** -- How NFA vs DFA engines work and their tradeoffs
5. **Security Measures in Practice** -- Designing input validation pipelines
6. **Benchmarking Methods** -- Measuring and comparing regex performance


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Unicode Regular Expressions -- \p{Script}, Flags, and Normalization](./02-unicode-regex.md)

---

## 1. Fundamentals of Regex Engines

### 1.1 Differences Between NFA and DFA

```
The Two Major Regex Engine Architectures:

NFA (Non-deterministic Finite Automaton)
  Behavior: Pattern-driven. Tries each element of the pattern in order,
           and backtracks to explore other possibilities on failure.
  Characteristics:
  - Supports backreferences, lookaround, and possessive quantifiers
  - Worst-case exponential time complexity O(2^n)
  - Used by: Python re, JavaScript, Java, Perl, PCRE

DFA (Deterministic Finite Automaton)
  Behavior: Text-driven. Processes each character of the input exactly once,
           determining the next state via a state transition table.
  Characteristics:
  - No backtracking -> Always linear time O(n)
  - Cannot support backreferences or lookaround in principle
  - Used by: RE2, Rust regex, awk, grep (basic)

Hybrid:
  Some engines combine DFA and NFA
  - PCRE2 JIT: NFA-based but accelerated with JIT compilation
  - .NET: NFA-based but includes a timeout mechanism
  - RE2: Primarily DFA, but falls back to NFA for some features
```

### 1.2 How Backtracking Works

```python
import re

# Detailed tracking of how backtracking operates

# Pattern: a+b
# Input: "aaac"

# Engine behavior:
# Position 0: a+ -> matches "aaa" (greedy)
#             b  -> 'c' != 'b' -> fail
#             Backtrack: a+ -> matches "aa"
#             b  -> 'a' != 'b' -> fail
#             Backtrack: a+ -> matches "a"
#             b  -> 'a' != 'b' -> fail
# Position 1: a+ -> matches "aa"
#             b  -> 'c' != 'b' -> fail
#             Backtrack...
# Position 2: a+ -> matches "a"
#             b  -> 'c' != 'b' -> fail
# Position 3: a+ -> no match
# Result: No match

# This pattern is safe -- backtracking is linear

# Dangerous pattern: (a+)+b -- backtracking becomes exponential
```

### 1.3 Backtracking Limits in NFA Engines

```python
# Backtracking limits by language
#
# Python re:      No limit (default)
# Python regex:   Can be limited via timeout parameter
# JavaScript V8:  --regexp-backtracks-limit (default: several million)
# Java:           No limit (default)
# .NET:           Can be limited via Regex.MatchTimeout
# PCRE2:          Can be limited via pcre2_set_match_limit
# Perl:           No limit (default)

# JavaScript V8 example:
# node --regexp-backtracks-limit=10000 script.js

# .NET example:
# var regex = new Regex(pattern, RegexOptions.None,
#                       TimeSpan.FromSeconds(1));
```

---

## 2. What Is ReDoS?

### 2.1 Overview of ReDoS

```
ReDoS (Regular Expression Denial of Service):
A denial-of-service attack that forces a regex engine into
exponential time complexity through malicious input.

Attack flow:
+------------+    Vulnerable pattern    +----------------+
| Attacker   | ----------------------> | Web Server     |
|            |   Malicious input       |                |
|            |                         |  Regex engine  |
|            |                         |  enters        |
|            |                         |  infinite loop |
|            |                         |  -> CPU 100%   |
|            |                         |  -> DoS        |
+------------+                         +----------------+

ReDoS severity levels:
+----------------------------------------------+
| Severity Level | Situation                    |
+----------------+------------------------------+
| Minor          | Single request delay         |
| Moderate       | Worker thread exhaustion     |
| Major          | Server-wide CPU exhaustion   |
| Critical       | Cascading failure, full      |
|                | service outage               |
+----------------+------------------------------+
```

### 2.2 Real-World ReDoS Incidents

```
Notable past ReDoS incidents:

1. Stack Overflow (2016)
   Cause: Regex in HTML sanitizer
   Impact: 34 minutes of downtime
   Pattern: Caused by a whitespace repetition pattern

2. Cloudflare (2019)
   Cause: Regex in WAF rules
   Impact: 27-minute global service disruption
   Pattern: (?:(?:\"|'|\]|\}|\\|\d|(?:nan|infinity|true|false|
             null|undefined|symbol|math)|\`|\-|\+)+[)]*;?((?:\s|
             -|~|!|{}|\|\||\+)*.*(?:.*=.*)))

3. Node.js (2017) -- CVE-2017-15896
   Cause: Regex in HTTP header parser
   Impact: Affected all Node.js applications

4. npm (2018) -- event-stream
   Cause: Regex processing within a package
   Impact: Rippled across the entire npm ecosystem
```

### 2.3 Examples of Vulnerable Patterns

```python
import re
import time

# Vulnerable pattern: (a+)+b
pattern = re.compile(r'(a+)+b')

# Safe input: completes instantly
start = time.time()
pattern.search("aaaaab")
print(f"Safe input: {time.time() - start:.4f}s")

# Malicious input: becomes exponentially slow
# Warning: the following will be extremely slow to execute
for length in [15, 20, 25]:
    malicious = "a" * length + "c"  # Non-matching input
    start = time.time()
    pattern.search(malicious)
    elapsed = time.time() - start
    print(f"Length {length}: {elapsed:.4f}s")

# Example output:
# Length 15: 0.01s
# Length 20: 0.30s
# Length 25: 9.50s    <- Exponential growth!
```

### 2.4 The Mechanism of Backtracking Explosion

```
Pattern: (a+)+b
Input:   "aaaaac" (6 characters, no match)

Engine behavior -- tries all possible partitions:

Attempt 1: (aaaaa)b     -> b != c -> fail
Attempt 2: (aaaa)(a)b   -> b != c -> fail
Attempt 3: (aaa)(aa)b   -> b != c -> fail
Attempt 4: (aaa)(a)(a)b -> b != c -> fail
Attempt 5: (aa)(aaa)b   -> b != c -> fail
Attempt 6: (aa)(aa)(a)b -> b != c -> fail
Attempt 7: (aa)(a)(aa)b -> b != c -> fail
...

For n characters of 'a', approximately 2^n partitions are attempted
-> n=25: approximately 33 million
-> n=30: approximately 1 billion!

Why does it become exponential?
  (a+)+ means "repeat a group of one or more 'a' characters one or more times"
  This means "aaa" can be partitioned as:
    (aaa)         -- 1 group
    (aa)(a)       -- 2 groups
    (a)(aa)       -- 2 groups
    (a)(a)(a)     -- 3 groups
  Each partition pattern is equivalent to integer partitioning -> exponential
```

---

## 3. Classification of Dangerous Patterns

### 3.1 Three Types of ReDoS-Vulnerable Patterns

```
+----------------------------------------------------+
|         Types of ReDoS-Vulnerable Patterns          |
+----------------------------------------------------+
|                                                     |
| 1. Nested Quantifiers                               |
|    (a+)+   (a*)*   (a+)*   (a*)+                   |
|    -> Inner and outer quantifiers match              |
|       the same characters                            |
|                                                     |
| 2. Overlapping Alternation                          |
|    (a|a)+  (a|ab)+  (\w|\d)+                        |
|    -> Alternatives can match the same characters     |
|                                                     |
| 3. Overlapping Quantifiers                          |
|    \d+\d+  a+a+  .*.*                               |
|    -> Consecutive quantifiers compete for            |
|       the same characters                            |
|                                                     |
+----------------------------------------------------+
```

### 3.2 Detailed Analysis by Type

```python
# === Type 1: Nested Quantifiers ===

# Pattern: (a+)+
# Why is it dangerous?
# The inner a+ matches "one or more a's"
# The outer + "repeats the group one or more times"
# -> "aaaa" can be partitioned into (aa)(aa), (a)(aaa), (aaa)(a), (a)(a)(aa), ...
#   exponentially many ways

# Pattern: (\d+)*
# Why is it dangerous?
# The inner \d+ matches "one or more digits"
# The outer * "repeats zero or more times"
# -> "12345" can be partitioned into (12)(345), (1)(2345), (123)(45), ...

# === Type 2: Overlapping Alternation ===

# Pattern: (a|ab)+
# Why is it dangerous?
# For "ab", there are two choices: match as a + b or as ab
# This creates 2 choices at each step
# -> For n repetitions of "ab", there are 2^n possibilities

# Pattern: (\w|\d)+
# Why is it dangerous?
# \d is a subset of \w
# For digits, both \w and \d are attempted
# -> Exponential backtracking with digit-only input

# === Type 3: Overlapping Quantifiers ===

# Pattern: \d+\d+\d+
# Why is it dangerous?
# How to distribute "12345" among each \d+
# (1)(2)(345), (1)(23)(45), (12)(3)(45), ...
# -> Combinatorial explosion
```

### 3.3 Examples of Real-World Vulnerable Patterns

```python
# Catalog of vulnerable patterns (avoid executing these)

vulnerable_patterns = {
    # Nested quantifiers
    "(a+)+":           "aaaaaaaaaaaaaaac",
    "(a+)+b":          "aaaaaaaaaaaaaaac",
    "(a*)*b":          "aaaaaaaaaaaaaaac",
    "([a-z]+)+$":      "aaaaaaaaaaaaaaa!",

    # Overlapping alternation
    "(a|a)+b":         "aaaaaaaaaaaaaaac",
    "(\\w|\\d)+$":     "aaaaaaaaaaaaaaaa!",
    "(.*a){20}":       "aaaaaaaaaaaaaaaaaaaaX",

    # Real-world vulnerable patterns
    # Email validation
    r"^([a-zA-Z0-9])(([\-.]|[_]+)?([a-zA-Z0-9]+))*(@)":
        "aaaaaaaaaaaaaaaaaaaaa!",

    # URL validation
    r"^(https?://)([\w-]+(\.[\w-]+)+)(/[\w-./?%&=]*)*$":
        "http://a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p!",
}
```

### 3.4 Mathematical Analysis of Vulnerable Patterns

```
Calculating backtracking count:

For pattern (a+)+b with input "a^n c":

  The number of ways to partition n characters of 'a'
  into k groups is C(n-1, k-1) (combination)

  Summing over all possible group counts:
  Sum_{k=1}^{n} C(n-1, k-1) = 2^(n-1)

  -> O(2^n) backtracking

Concrete examples:
  n=10:  2^9  =    512 attempts
  n=20:  2^19 = 524,288 attempts
  n=25:  2^24 = 16,777,216 attempts
  n=30:  2^29 = 536,870,912 attempts (over 500 million)

  Assuming 100ns per backtracking step:
  n=25: approximately 1.7 seconds
  n=30: approximately 54 seconds
  n=35: approximately 1,718 seconds (about 29 minutes)

For pattern (a|aa)+ with input "a^n c":
  Grows like the Fibonacci sequence -> O(phi^n) ~ O(1.618^n)
  Exponential but slower growth than (a+)+

For pattern (\d+)+ with input "d^n c":
  Equivalent to (a+)+ -> O(2^n)
```

---

## 4. Designing Safe Patterns

### 4.1 Fundamental Principles of Fixing Patterns

```python
import re

# Principle 1: Eliminate nested quantifiers

# BAD: (a+)+
# GOOD: a+
pattern_safe1 = r'a+b'

# Principle 2: Eliminate overlapping alternatives

# BAD: (\w|\d)+  (\w includes \d)
# GOOD: \w+
pattern_safe2 = r'\w+$'

# Principle 3: Use negated character classes for explicit constraints

# BAD: ".*"  (greedy, matches past the closing quote)
# GOOD: "[^"]*"  (matches only within quotes)
pattern_safe3 = r'"[^"]*"'

# Principle 4: Use atomic groups or possessive quantifiers
# (Python regex module, Java, PCRE)

# BAD: (a+)+b       -> backtracking explosion
# GOOD: (?>a+)+b    -> atomic group (backtracking prohibited)

# Principle 5: Limit input length
# Not just pattern fixes -- limit the input itself

def safe_match(pattern, text, max_length=10000):
    """Safe match with input length limit"""
    if len(text) > max_length:
        raise ValueError(f"Input too long: {len(text)} > {max_length}")
    return re.search(pattern, text)
```

### 4.2 Pattern Fix Examples

```python
import re

# Example 1: Email validation
# BAD: vulnerable
email_bad = r'^([a-zA-Z0-9])(([\-.]|[_]+)?([a-zA-Z0-9]+))*(@)'

# GOOD: safe and practical
email_good = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# Example 2: Extracting HTML tag content
# BAD: vulnerable (nested quantifiers)
tag_bad = r'<(\w+)(\s+\w+="[^"]*")*>'

# GOOD: safe
tag_good = r'<\w+(?:\s+\w+="[^"]*")*>'

# Example 3: CSV fields
# BAD: vulnerable
csv_bad = r'^("(?:[^"]|"")*"|[^,]*)(,("(?:[^"]|"")*"|[^,]*))*$'

# GOOD: process field by field
csv_good = r'"(?:[^"]|"")*"|[^,]+'

text = '"hello, world","test""quote"",normal'
print(re.findall(csv_good, text))

# Example 4: Whitespace handling
# BAD: vulnerable (overlapping like \s+\s*)
whitespace_bad = r'(\s+\w+)*\s*$'

# GOOD: safe (clear delimiters)
whitespace_good = r'(?:\s+\w+)*\s*$'
# Even safer: constrain with negated character class
whitespace_best = r'(?:\s\S+)*\s*$'

# Example 5: Multi-line text processing
# BAD: .* matches across lines
multiline_bad = r'START.*END'

# GOOD: match within a single line only
multiline_good = r'START[^\n]*END'
# Or explicitly control the DOTALL flag
```

### 4.3 Possessive Quantifiers and Atomic Groups

```python
# Possessive Quantifier: X++, X*+, X?+
# Atomic Group: (?>X)
#
# Both prohibit backtracking
# -> Once matched, characters are never relinquished

# Atomic groups with the Python regex module
import regex

# Normal quantifier: backtracks
# Pattern: a+b against "aaac"
# a+ -> "aaa" -> expects b but gets c -> backtracks
# a+ -> "aa" -> expects b but gets a -> backtracks
# ...

# Atomic group: does not backtrack
# Pattern: (?>a+)b against "aaac"
# (?>a+) -> "aaa" (backtracking prohibited) -> expects b but gets c -> immediate failure

pattern_atomic = regex.compile(r'(?>a+)b')
print(pattern_atomic.search("aaab"))   # => match
print(pattern_atomic.search("aaac"))   # => None (immediate failure)

# Possessive quantifier (Java, PCRE2, regex module)
pattern_possessive = regex.compile(r'a++b')
print(pattern_possessive.search("aaab"))   # => match
print(pattern_possessive.search("aaac"))   # => None

# Atomic groups as ReDoS countermeasure
# BAD: (a+)+b -> backtracking explosion
# GOOD: (?>(a+))+b -> inner a+ is atomic -> no explosion
safe_pattern = regex.compile(r'(?>(a+))+b')
import time
start = time.time()
safe_pattern.search("a" * 30 + "c")
print(f"Atomic version: {time.time() - start:.4f}s")
# => completes instantly
```

```java
// Possessive quantifiers in Java
import java.util.regex.*;

public class PossessiveExample {
    public static void main(String[] args) {
        // Normal: backtracking enabled
        Pattern greedy = Pattern.compile("(a+)+b");

        // Possessive: backtracking disabled
        Pattern possessive = Pattern.compile("(a++)+b");

        String input = "a".repeat(30) + "c";

        // Possessive quantifier fails immediately
        long start = System.nanoTime();
        possessive.matcher(input).find();
        long elapsed = System.nanoTime() - start;
        System.out.println("Possessive: " + (elapsed / 1_000_000) + "ms");
        // => Possessive: 0ms
    }
}
```

### 4.4 Timeout Mechanisms

```python
import re
import signal

# Python: timeout using signal (Unix-like systems only)
class RegexTimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise RegexTimeout("Regex timed out")

def safe_search(pattern, text, timeout_sec=1):
    """Regex search with timeout"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = re.search(pattern, text)
        signal.alarm(0)  # Cancel timer
        return result
    except RegexTimeout:
        return None

# Usage example
result = safe_search(r'(a+)+b', "a" * 30 + "c", timeout_sec=2)
if result is None:
    print("Timeout or no match")
```

```python
# Cross-platform timeout using multiprocessing
import re
from multiprocessing import Process, Queue
import time

def regex_worker(pattern_str, text, result_queue):
    """Execute regex in a separate process"""
    try:
        pattern = re.compile(pattern_str)
        result = pattern.search(text)
        result_queue.put(('success', result is not None))
    except Exception as e:
        result_queue.put(('error', str(e)))

def safe_regex_search(pattern_str, text, timeout_sec=2):
    """Process-based regex search with timeout"""
    result_queue = Queue()
    p = Process(target=regex_worker, args=(pattern_str, text, result_queue))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return None, "timeout"

    if not result_queue.empty():
        status, result = result_queue.get()
        return result, status

    return None, "unknown error"

# Usage example
result, status = safe_regex_search(r'(a+)+b', "a" * 30 + "c", timeout_sec=2)
print(f"Result: {result}, Status: {status}")
```

```javascript
// JavaScript: third-party libraries for safe regex
// re2 package (RE2 engine bindings)
// npm install re2

// const RE2 = require('re2');
// const pattern = new RE2('(a+)+b');
// RE2 automatically optimizes this

// Timeout using Worker
function safeRegexTest(pattern, text, timeoutMs = 1000) {
    return new Promise((resolve, reject) => {
        const worker = new Worker(
            URL.createObjectURL(new Blob([`
                self.onmessage = function(e) {
                    const regex = new RegExp(e.data.pattern);
                    const result = regex.test(e.data.text);
                    self.postMessage({ result });
                };
            `]))
        );

        const timer = setTimeout(() => {
            worker.terminate();
            reject(new Error('Regex timeout'));
        }, timeoutMs);

        worker.onmessage = (e) => {
            clearTimeout(timer);
            worker.terminate();
            resolve(e.data.result);
        };

        worker.postMessage({ pattern: pattern.source, text });
    });
}
```

---

## 5. Performance Optimization

### 5.1 Compilation (Pre-compilation)

```python
import re
import time

text = "The quick brown fox jumps over the lazy dog" * 1000

# BAD: Compile on every iteration
start = time.time()
for _ in range(10000):
    re.search(r'\b\w{5}\b', text)
time_uncompiled = time.time() - start

# GOOD: Pre-compile
pattern = re.compile(r'\b\w{5}\b')
start = time.time()
for _ in range(10000):
    pattern.search(text)
time_compiled = time.time() - start

print(f"Uncompiled: {time_uncompiled:.3f}s")
print(f"Compiled:   {time_compiled:.3f}s")
# Pre-compiled is faster (especially with repeated use)

# Note: Python internally caches up to 512 patterns,
# but explicit compilation is recommended
```

### 5.2 Speedup with Anchors

```python
import re
import time

# Anchors let the engine skip unnecessary positions

text = "2024-01-15 Some log entry here" * 10000

# Slow: tries at every position
slow = r'\d{4}-\d{2}-\d{2}'

# Fast: only tries at line start
fast = r'^\d{4}-\d{2}-\d{2}'

# Fast: constrained by word boundaries
fast2 = r'\b\d{4}-\d{2}-\d{2}\b'

# Benchmark
for label, pattern in [("No anchor", slow), ("With ^", fast), ("With \\b", fast2)]:
    compiled = re.compile(pattern)
    start = time.time()
    for _ in range(1000):
        compiled.search(text)
    elapsed = time.time() - start
    print(f"  {label}: {elapsed:.4f}s")
```

### 5.3 Negated Character Class vs Lazy Quantifier

```python
import re
import time

html = '<div class="test">' * 10000 + 'content</div>'

# Slow: lazy quantifier (backtracking occurs)
start = time.time()
re.search(r'<div.*?>', html)
lazy_time = time.time() - start

# Fast: negated character class (no backtracking)
start = time.time()
re.search(r'<div[^>]*>', html)
negated_time = time.time() - start

print(f"Lazy quantifier:        {lazy_time:.6f}s")
print(f"Negated character class: {negated_time:.6f}s")

# Why is the negated character class faster?
#
# Lazy quantifier .*?> behavior:
#   1. .  matches (starting from 0 characters)
#   2. Try > -> fail
#   3. . matches one more character
#   4. Try > -> fail
#   5. Repeat... (2 steps each time)
#
# Negated character class [^>]*> behavior:
#   1. [^>]* matches all non-> characters at once
#   2. Try > -> success
#   (No backtracking occurs)
```

### 5.4 Summary of Optimization Techniques

```python
import re

# 1. Pre-check with fixed strings
text = "long text without the target pattern..."

# Slow: search with regex every time
if re.search(r'target\s+\w+\s+pattern', text):
    pass

# Fast: pre-check with string search
if 'target' in text and 'pattern' in text:
    if re.search(r'target\s+\w+\s+pattern', text):
        pass

# 2. Leverage fixed prefixes
# The engine can optimize patterns with fixed prefixes

# Easy to optimize: starts with a fixed string
good = r'ERROR: \w+'

# Hard to optimize: starts with a variable pattern
bad = r'.*ERROR: \w+'

# 3. Eliminate unnecessary captures
# Capture groups -> non-capture groups
slow = r'(https?)://([\w.]+)/([\w/]+)'
fast = r'(?:https?)://(?:[\w.]+)/(?:[\w/]+)'

# 4. Optimize alternation order
# Put more frequent alternatives first
slow_alt = r'(?:rare_pattern|common_pattern)'
fast_alt = r'(?:common_pattern|rare_pattern)'

# 5. Use specific character classes
# Slow: .* (matches anything -> source of backtracking)
slow_dot = r'<.*>'
# Fast: [^>]* (constrained -> minimizes backtracking)
fast_neg = r'<[^>]*>'

# 6. Match only the required portion
# Slow: match the entire string then extract groups
slow_full = r'^.*?(\d{4}-\d{2}-\d{2}).*$'
# Fast: search only for the needed part
fast_part = r'\d{4}-\d{2}-\d{2}'
```

### 5.5 Optimizing Large Data Processing

```python
import re
import time

# Optimization for processing large log files

def benchmark(name, func, lines, iterations=3):
    """Run benchmark"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(lines)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    print(f"  {name}: {avg:.4f}s (average)")

# Generate test data
log_lines = []
for i in range(100000):
    if i % 3 == 0:
        log_lines.append(f"2024-01-15 10:30:{i%60:02d} [ERROR] Something failed: {i}")
    elif i % 3 == 1:
        log_lines.append(f"2024-01-15 10:30:{i%60:02d} [INFO] Processing item {i}")
    else:
        log_lines.append(f"some other line without timestamp {i}")

# Method 1: Naive (re.search on every line)
def naive_search(lines):
    results = []
    for line in lines:
        m = re.search(r'\[ERROR\]', line)
        if m:
            results.append(line)
    return results

# Method 2: Pre-compiled
error_pattern = re.compile(r'\[ERROR\]')
def compiled_search(lines):
    results = []
    for line in lines:
        if error_pattern.search(line):
            results.append(line)
    return results

# Method 3: Pre-filter with string search
def prefiltered_search(lines):
    results = []
    for line in lines:
        if '[ERROR]' in line:
            results.append(line)
    return results

# Method 4: List comprehension (Python optimization)
def list_comp_search(lines):
    return [line for line in lines if '[ERROR]' in line]

# Run benchmarks
print("Benchmark results (100K lines):")
benchmark("Naive re.search", naive_search, log_lines)
benchmark("Pre-compiled", compiled_search, log_lines)
benchmark("String pre-filter", prefiltered_search, log_lines)
benchmark("List comprehension", list_comp_search, log_lines)
```

### 5.6 Regex vs String Operations Benchmark

```python
import re
import time

# When to use regex vs string operations

text = "user@example.com" * 10000

# Task: Check if "@" is present

# Method 1: re.search
pattern = re.compile(r'@')
start = time.perf_counter()
for _ in range(100000):
    pattern.search(text)
regex_time = time.perf_counter() - start

# Method 2: in operator
start = time.perf_counter()
for _ in range(100000):
    '@' in text
string_time = time.perf_counter() - start

print(f"re.search:   {regex_time:.4f}s")
print(f"in operator: {string_time:.4f}s")
print(f"Speed ratio: {regex_time / string_time:.1f}x")

# General guidelines:
# - Simple string search -> in, find(), startswith(), endswith()
# - Pattern matching -> regex
# - String replacement (fixed) -> str.replace()
# - String replacement (pattern) -> re.sub()
# - String splitting (fixed) -> str.split()
# - String splitting (pattern) -> re.split()
```

---

## 6. Security Design for Input Validation

### 6.1 Defense-in-Depth Architecture

```
Input Validation Pipeline:

+---------------------------------------------------+
|            Input Validation Pipeline               |
+---------------------------------------------------+
|                                                    |
|  Layer 1: Length Limits                            |
|  +-- Limit maximum character count (e.g., 1000)   |
|  +-- Empty string check                           |
|                                                    |
|  Layer 2: Character Type Restrictions              |
|  +-- Allow only permitted character classes        |
|  +-- Remove control and invisible characters       |
|                                                    |
|  Layer 3: Simple Pattern Pre-check                 |
|  +-- Pre-check with string operations              |
|  +-- Early rejection of obviously invalid input    |
|                                                    |
|  Layer 4: Detailed Regex Validation                |
|  +-- Use safe patterns                             |
|  +-- Set timeouts                                  |
|  +-- Prefer DFA engines                            |
|                                                    |
|  Layer 5: Business Logic Validation                |
|  +-- Application-specific rules                    |
|                                                    |
+---------------------------------------------------+
```

```python
import re

class InputValidator:
    """Input validation with defense in depth"""

    def __init__(self, max_length=1000, allowed_chars=None,
                 pattern=None, timeout_sec=1):
        self.max_length = max_length
        self.allowed_chars = allowed_chars
        self.pattern = re.compile(pattern) if pattern else None
        self.timeout_sec = timeout_sec

    def validate(self, value: str) -> tuple[bool, str]:
        """Validate input value"""
        # Layer 1: Length limit
        if not value:
            return False, "Empty input"
        if len(value) > self.max_length:
            return False, f"Input too long ({len(value)} > {self.max_length})"

        # Layer 2: Character type restriction
        if self.allowed_chars:
            invalid = set(value) - set(self.allowed_chars)
            if invalid:
                return False, f"Disallowed characters: {invalid}"

        # Layer 3: Regex check
        if self.pattern:
            if not self.pattern.match(value):
                return False, "Does not match the expected pattern"

        return True, "OK"

# Usage example: email address validator
email_validator = InputValidator(
    max_length=254,
    pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

test_emails = [
    "user@example.com",
    "a" * 300 + "@test.com",
    "invalid-email",
    "<script>alert('xss')</script>@evil.com",
]

for email in test_emails:
    valid, msg = email_validator.validate(email)
    print(f"  {email[:40]:40s} => {msg}")
```

### 6.2 Safe Use of User Input as Regular Expressions

```python
import re

# BAD: Using user input directly as a regex
def search_bad(user_input: str, text: str):
    return re.search(user_input, text)  # ReDoS attack possible!

# Attack example:
# user_input = "(a+)+b"
# text = "a" * 30 + "c"
# -> CPU goes to 100%

# GOOD: Escape user input
def search_safe(user_input: str, text: str):
    escaped = re.escape(user_input)  # Escape all metacharacters
    return re.search(escaped, text)

# GOOD: Or use a DFA engine
# import re2
# def search_safe_re2(pattern: str, text: str):
#     return re2.search(pattern, text)  # O(n) guaranteed

# GOOD: Allow only a restricted regex subset
def search_limited(user_pattern: str, text: str, max_length=100):
    """Allow only a safe subset of regex"""
    if len(user_pattern) > max_length:
        raise ValueError("Pattern too long")

    # Detect dangerous constructs
    dangerous = [
        r'\(.+[+*]\).+[+*]',  # Nested quantifiers
        r'\(\?\:.*\|.*\)[+*]', # Repeated alternation
    ]
    for d in dangerous:
        if re.search(d, user_pattern):
            raise ValueError("Dangerous pattern detected")

    return re.search(user_pattern, text)
```

### 6.3 Pattern Design for WAF (Web Application Firewall)

```python
import re

# Safe WAF rule design

# BAD: Dangerous WAF rules
waf_rules_bad = {
    'sql_injection': r"('.+--)|(--.+')",  # No nesting but .+ is risky
    'xss': r"(<script.*>.*</script.*>)",  # .* is dangerous
}

# GOOD: Safe WAF rules
waf_rules_good = {
    'sql_injection': r"(?:'\s*(?:--|;|/\*)|(?:--|;|/\*)\s*')",
    'xss': r"<script[^>]*>[^<]*</script[^>]*>",
    'path_traversal': r"(?:\.\./|\.\.\\){2,}",
}

def check_waf_rules(input_text: str, rules: dict) -> list[str]:
    """Check input against WAF rules"""
    violations = []
    for rule_name, pattern in rules.items():
        compiled = re.compile(pattern, re.IGNORECASE)
        if compiled.search(input_text):
            violations.append(rule_name)
    return violations

# Test
test_inputs = [
    "normal input",
    "'; DROP TABLE users; --",
    "<script>alert('xss')</script>",
    "../../../etc/passwd",
]

for inp in test_inputs:
    violations = check_waf_rules(inp, waf_rules_good)
    if violations:
        print(f"  [BLOCKED] '{inp[:40]}' => {violations}")
    else:
        print(f"  [PASS]    '{inp[:40]}'")
```

---

## 7. Benchmarking Methods

### 7.1 Regex Benchmark Framework

```python
import re
import time
import statistics

class RegexBenchmark:
    """Performance benchmark for regular expressions"""

    def __init__(self, pattern: str, description: str = ""):
        self.pattern = re.compile(pattern)
        self.pattern_str = pattern
        self.description = description

    def run(self, text: str, iterations: int = 1000,
            warmup: int = 100) -> dict:
        """Run benchmark"""
        # Warmup
        for _ in range(warmup):
            self.pattern.search(text)

        # Measurement
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            self.pattern.search(text)
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed)

        return {
            'pattern': self.pattern_str,
            'description': self.description,
            'iterations': iterations,
            'min_ns': min(times),
            'max_ns': max(times),
            'mean_ns': statistics.mean(times),
            'median_ns': statistics.median(times),
            'stdev_ns': statistics.stdev(times) if len(times) > 1 else 0,
            'p95_ns': sorted(times)[int(len(times) * 0.95)],
            'p99_ns': sorted(times)[int(len(times) * 0.99)],
        }

    def print_result(self, result: dict):
        """Print results"""
        print(f"Pattern: {result['pattern']}")
        print(f"Description: {result['description']}")
        print(f"  Mean:   {result['mean_ns']/1000:.2f}us")
        print(f"  Median: {result['median_ns']/1000:.2f}us")
        print(f"  P95:    {result['p95_ns']/1000:.2f}us")
        print(f"  P99:    {result['p99_ns']/1000:.2f}us")
        print()

# Usage example
text = "The quick brown fox jumps over the lazy dog" * 100

benchmarks = [
    RegexBenchmark(r'\bfox\b', 'Word boundary match'),
    RegexBenchmark(r'fox', 'Simple match'),
    RegexBenchmark(r'(?<=\s)fox(?=\s)', 'Lookaround match'),
    RegexBenchmark(r'.*fox.*', '.* match'),
    RegexBenchmark(r'[^ ]*fox[^ ]*', 'Negated character class match'),
]

print("=== Benchmark Results ===\n")
for bench in benchmarks:
    result = bench.run(text)
    bench.print_result(result)
```

### 7.2 Scalability Testing

```python
import re
import time

def scalability_test(pattern_str: str, char: str = 'a',
                     lengths: list[int] = None):
    """Test scalability against input length"""
    if lengths is None:
        lengths = [10, 50, 100, 500, 1000, 5000, 10000]

    pattern = re.compile(pattern_str)
    print(f"Pattern: {pattern_str}")
    print(f"{'Length':>10s} {'Time(us)':>12s} {'Ratio':>8s}")
    print("-" * 35)

    prev_time = None
    for n in lengths:
        text = char * n
        start = time.perf_counter()
        pattern.search(text)
        elapsed = (time.perf_counter() - start) * 1_000_000  # us

        ratio = f"{elapsed / prev_time:.1f}x" if prev_time else "-"
        prev_time = elapsed

        print(f"{n:>10d} {elapsed:>12.2f} {ratio:>8s}")

        # Safety: abort if over 1 second
        if elapsed > 1_000_000:
            print("  [Aborted: exceeded 1 second]")
            break

# Test
print("=== Safe Pattern ===")
scalability_test(r'[a-z]+$')
print()

print("=== Linear Pattern (negated character class) ===")
scalability_test(r'[^b]*b')
print()

# Warning: the following can be extremely slow with long inputs
# print("=== Dangerous Pattern ===")
# scalability_test(r'(a+)+b', lengths=[10, 15, 20, 25])
```

---

## 8. Security Measures by Language

### 8.1 Python

```python
import re

# ReDoS countermeasures in Python

# Countermeasure 1: regex module's timeout parameter
import regex
try:
    result = regex.search(r'(a+)+b', 'a' * 30 + 'c', timeout=1)
except TimeoutError:
    print("Timeout")

# Countermeasure 2: google-re2 package
# pip install google-re2
# import re2
# result = re2.search(r'(a+)+b', 'a' * 30 + 'c')
# RE2 automatically converts to a safe pattern

# Countermeasure 3: Static analysis of patterns
def audit_pattern(pattern_str: str) -> list[str]:
    """Audit pattern safety"""
    warnings = []

    # Nested quantifiers
    if re.search(r'\([^)]*[+*][^)]*\)[+*]', pattern_str):
        warnings.append("Nested quantifiers")

    # Unanchored .*
    if '.*' in pattern_str and not pattern_str.startswith('^'):
        warnings.append("Unanchored .*")

    # Huge repetition
    large_repeat = re.search(r'\{(\d+)\}', pattern_str)
    if large_repeat and int(large_repeat.group(1)) > 1000:
        warnings.append(f"Large repetition count: {large_repeat.group(1)}")

    return warnings

# Test
patterns = [
    r'(a+)+b',
    r'^[a-z]+$',
    r'.*error.*',
    r'a{10000}',
]

for p in patterns:
    warnings = audit_pattern(p)
    status = "WARN" if warnings else "OK"
    print(f"  [{status}] {p}: {warnings or 'clean'}")
```

### 8.2 JavaScript / Node.js

```javascript
// ReDoS countermeasures in JavaScript / Node.js

// Countermeasure 1: Use the re2 package
// const RE2 = require('re2');
// const safe = new RE2('(a+)+b');

// Countermeasure 2: Static analysis with safe-regex
// const safe = require('safe-regex');
// if (!safe(userPattern)) {
//     throw new Error('Unsafe regex pattern');
// }

// Countermeasure 3: Sandboxed execution using the vm module
// const vm = require('vm');
// const script = new vm.Script(`
//     const result = /${pattern}/.test(input);
// `);
// const context = vm.createContext({ input, pattern });
// script.runInContext(context, { timeout: 1000 });

// Countermeasure 4: Node.js v20+ RegExp timeout (experimental)
// Using --experimental-regexp-engine may enable
// an RE2-style safe engine

// Best practices:
// 1. Always escape user input
function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// 2. Limit pattern complexity
function isPatternSafe(pattern) {
    // Detect nested quantifiers
    if (/\(.+[+*]\).+[+*]/.test(pattern)) return false;
    // Limit pattern length
    if (pattern.length > 200) return false;
    return true;
}

// 3. Limit input length
function safeMatch(pattern, text, maxLength) {
    maxLength = maxLength || 10000;
    if (text.length > maxLength) {
        throw new Error('Input too long');
    }
    return new RegExp(pattern).test(text);
}
```

### 8.3 Java

```java
import java.util.regex.*;
import java.util.concurrent.*;

public class SafeRegex {

    // Countermeasure 1: Regex match with timeout
    public static boolean safeMatch(String pattern, String text,
                                     long timeoutMs)
            throws TimeoutException {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<Boolean> future = executor.submit(() -> {
            Pattern p = Pattern.compile(pattern);
            return p.matcher(text).matches();
        });

        try {
            return future.get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            executor.shutdownNow();
        }
    }

    // Countermeasure 2: Use possessive quantifiers
    public static void possessiveExample() {
        // Normal: backtracking enabled
        Pattern greedy = Pattern.compile("(a+)+b");

        // Possessive: backtracking disabled
        Pattern possessive = Pattern.compile("(a++)+b");

        // Atomic group
        Pattern atomic = Pattern.compile("(?>a+)+b");
    }

    // Countermeasure 3: Pre-validate patterns
    public static boolean isPatternSafe(String pattern) {
        // Detect nested quantifiers
        if (pattern.matches(".*\\(.+[+*]\\).+[+*].*")) {
            return false;
        }
        // Limit pattern length
        if (pattern.length() > 200) {
            return false;
        }
        return true;
    }
}
```

### 8.4 Go (RE2)

```go
package main

import (
    "fmt"
    "regexp"
    "time"
)

func main() {
    // Go's regexp package is RE2-based
    // -> Inherently safe against ReDoS

    pattern := regexp.MustCompile(`(a+)+b`)
    input := string(make([]byte, 30)) + "c"
    for i := range input[:30] {
        input = input[:i] + "a" + input[i+1:]
    }

    start := time.Now()
    pattern.MatchString("a]" + input)
    elapsed := time.Since(start)
    fmt.Printf("RE2: %v\n", elapsed)
    // => Always fast (O(n))

    // However, RE2 has limitations:
    // - No backreferences (\1)
    // - No lookaround (?=) (?<=)
    // - No possessive quantifiers (a++) (unnecessary)
    // - No atomic groups (?>...) (unnecessary)

    // If you need features with these limitations:
    // Use the github.com/dlclark/regexp2 package
    // Note: this reintroduces ReDoS risk
}
```

### 8.5 Rust

```rust
use regex::Regex;
use std::time::Instant;

fn main() {
    // Rust's regex crate is DFA-based, similar to RE2
    // -> Inherently safe against ReDoS

    let re = Regex::new(r"(a+)+b").unwrap();
    // Note: Rust regex automatically converts to a safe pattern

    let input = "a".repeat(100) + "c";
    let start = Instant::now();
    re.is_match(&input);
    let elapsed = start.elapsed();
    println!("Rust regex: {:?}", elapsed);
    // => Always fast (O(n))

    // If you need lookaround:
    // Use the fancy-regex crate
    // use fancy_regex::Regex;
    // Note: this reintroduces ReDoS risk

    // Limitations of the regex crate:
    // - No backreferences
    // - No lookaround
    // - Compile time limits
    //   (overly complex patterns produce compilation errors)
}
```

---

## 9. ASCII Diagrams

### 9.1 Visualizing Backtracking Explosion

```
Pattern: (a+)+b
Input:   "aaac" (4 characters)

Tree of all attempts:

(aaaa) -> b? -> c != b -> fail
|
+-- (aaa)(a) -> b? -> c != b -> fail
|
+-- (aa)(aa) -> b? -> c != b -> fail
|   +-- (aa)(a)(a) -> b? -> c != b -> fail
|
+-- (a)(aaa) -> b? -> c != b -> fail
|   +-- (a)(aa)(a) -> b? -> c != b -> fail
|   +-- (a)(a)(aa) -> b? -> c != b -> fail
|       +-- (a)(a)(a)(a) -> b? -> c != b -> fail
|
Total: 8 attempts (input length n=4)
n=10 -> 512 attempts
n=20 -> 524,288 attempts
n=30 -> 536,870,912 attempts (over 500 million!)
```

### 9.2 NFA vs DFA Behavior Comparison

```
Pattern: a+b
Input: "aaac"

=== NFA Engine (Python re, JavaScript) ===

State transitions + backtracking:

  Position 0:
    a+ -> 'a','a','a' (greedy match)
    b -> 'c' != 'b' -> backtrack
    a+ -> 'a','a'
    b -> 'a' != 'b' -> backtrack
    a+ -> 'a'
    b -> 'a' != 'b' -> fail

  Position 1: similar attempts...
  Position 2: similar attempts...
  Position 3: a+ cannot match -> fail

  Total steps: O(n^2) (in this case)

=== DFA Engine (RE2, Rust regex) ===

State transition table:

  Current State  Input  Next State
  -------------  -----  ----------
  Start          'a'    State1 (a+)
  State1         'a'    State1 (a+ repeat)
  State1         'b'    Accept (match success)
  State1         other  Fail

  Position 0: Start -> 'a' -> State1
  Position 1:         'a' -> State1
  Position 2:         'a' -> State1
  Position 3:         'c' -> Fail

  Total steps: O(n) (always linear)
```

### 9.3 Vulnerable Patterns vs Safe Alternatives

```
Vulnerable Pattern      Safe Alternative         Reason
------------------      ----------------         ------
(a+)+                   a+                       Nesting unnecessary
(a|a)+                  a+                       Overlap removed
(.*a){n}                Max length check         Eliminate nested quantifiers
".*"                    "[^"]*"                  Explicitly constrain range
(\w+\s*)+               [\w\s]+                  Flattened
(.+)+                   .+                       Nesting unnecessary
(a+|b+)+                [ab]+                    Flattened
\d+\d+                  \d{2,}                   Overlap removed
(\w+\.)+                [\w.]+                   Flattened
(.*\n)*                 [^]*                     Flattened (language-dependent)
```

### 9.4 Performance Improvement Flowchart

```
Regex is slow?
    |
    +-- Analyze the pattern
    |   |
    |   +-- Nested quantifiers?
    |   |   +-- YES -> Flatten or make atomic
    |   |
    |   +-- Overlapping alternatives?
    |   |   +-- YES -> Merge into character class
    |   |
    |   +-- Uses .*?
    |   |   +-- YES -> Replace with [^X]*
    |   |
    |   +-- No anchor?
    |   |   +-- YES -> Add ^, \b, etc.
    |   |
    |   +-- Unnecessary captures?
    |       +-- YES -> Change to (?:...)
    |
    +-- Is it compiled?
    |   +-- NO -> Use re.compile()
    |
    +-- Is the input untrusted?
    |   +-- YES -> Consider RE2/DFA engine
    |   +-- YES -> Set a timeout
    |   +-- YES -> Limit input length
    |
    +-- Processing large data?
    |   +-- YES -> Add pre-filter
    |   +-- YES -> Consider batch processing
    |   +-- YES -> Consider parallel processing
    |
    +-- Still slow?
        +-- Consider alternatives to regex
            (string operations, parsers, dedicated libraries, etc.)
```

---

## 10. Comparison Tables

### 10.1 Performance Characteristics by Engine

| Engine | Worst-Case Complexity | ReDoS Resistance | Features | Language |
|--------|----------------------|-----------------|----------|----------|
| Python re | O(2^n) | None | Rich | Python |
| Python regex | O(2^n) | timeout available | Richest | Python |
| JavaScript V8 | O(2^n) | backtracks-limit | Rich | JavaScript |
| Java Pattern | O(2^n) | None | Rich (possessive quantifiers) | Java |
| RE2 | O(n) | Yes | Limited | Go, C++ |
| Rust regex | O(n) | Yes | Limited | Rust |
| PCRE2 JIT | O(2^n) | match_limit | Richest | C |
| .NET | O(2^n) | MatchTimeout | Rich | C# |
| Oniguruma | O(2^n) | None | Rich | Ruby |

### 10.2 Optimization Technique Effectiveness Comparison

| Technique | Impact | Use Case | Implementation Cost |
|-----------|--------|----------|-------------------|
| re.compile() | Low-Medium | Repeated use | Low |
| Adding anchors | Medium | When position is known | Low |
| Negated character class | Medium-High | Tag extraction, etc. | Low |
| Possessive quantifier | High | Preventing backtracking | Medium |
| Atomic group | High | Preventing backtracking | Medium |
| DFA engine (RE2) | Highest | Untrusted input | High (library change) |
| String pre-check | Medium | Low occurrence frequency | Low |
| Pattern splitting | Medium | Complex patterns | Medium |
| Input length limit | High | All cases | Low |
| Timeout setting | High | External input processing | Medium |

### 10.3 Security Countermeasure Priority

| Countermeasure | Priority | Impact | Cost |
|---------------|----------|--------|------|
| Input length limit | Highest | High | Low |
| Pattern static analysis | High | Medium | Low |
| Timeout setting | High | High | Medium |
| DFA engine usage | Medium-High | Highest | High |
| User input escaping | Highest | High | Low |
| Pattern simplification | Medium | Medium | Medium |
| Code review auditing | Medium | Medium | Medium |
| CI/CD automated checks | High | High | Medium |

---

## 11. Vulnerability Detection Tools

### 11.1 Static Analysis Tools

```bash
# recheck: detect regex vulnerabilities
# npm install -g recheck
# recheck "(a+)+b"

# redos-checker (Python)
# pip install redos-checker

# safe-regex (JavaScript)
# npm install safe-regex

# semgrep: static analysis for entire codebase
# semgrep --config "p/regex-dos" .
```

```javascript
// safe-regex usage example
// const safe = require('safe-regex');
// console.log(safe('(a+)+b'));      // => false (vulnerable)
// console.log(safe('[a-z]+'));       // => true  (safe)
// console.log(safe('(a|b|c)+'));     // => true  (safe)
```

### 11.2 Simple Checker in Python

```python
import re

def is_potentially_vulnerable(pattern: str) -> tuple[bool, list[str]]:
    """Simple vulnerability check for regex patterns"""
    warnings = []

    # Detect nested quantifiers
    if re.search(r'\([^)]*[+*][^)]*\)[+*]', pattern):
        warnings.append("Nested quantifiers detected")

    # Detect unrestricted .* usage
    if re.search(r'\.\*(?!\?)', pattern) and '^' not in pattern:
        warnings.append("Unanchored .* detected")

    # Detect overlapping alternation (simple check)
    if re.search(r'\((?:[^)]*\|[^)]*)\)[+*]', pattern):
        warnings.append("Repeated alternation detected")

    # Detect huge repetition count
    large_repeat = re.search(r'\{(\d+)', pattern)
    if large_repeat and int(large_repeat.group(1)) > 1000:
        warnings.append(f"Large repetition count: {large_repeat.group(1)}")

    # Detect backreference usage
    if re.search(r'\\[1-9]', pattern):
        warnings.append("Backreference used (be aware of performance impact)")

    return (len(warnings) > 0, warnings)

# Test
patterns = [
    r'(a+)+b',
    r'(\w|\d)+',
    r'[a-z]+',
    r'.*error.*',
    r'^[a-z]+$',
    r'(\w+\.)+\w+',
    r'(.)\1{100}',
    r'a{10000}',
]

for p in patterns:
    vulnerable, msgs = is_potentially_vulnerable(p)
    status = "VULNERABLE" if vulnerable else "SAFE"
    print(f"  {status}: {p}")
    for msg in msgs:
        print(f"    - {msg}")
```

### 11.3 Integration into CI/CD Pipelines

```python
#!/usr/bin/env python3
"""
regex_audit.py -- Script for auditing regular expressions in CI/CD pipelines

Usage:
    python regex_audit.py path/to/source/

Return codes:
    0: No issues
    1: Warnings found
    2: Dangerous patterns found
"""

import re
import sys
import os

def find_regex_patterns(filepath: str) -> list[tuple[int, str]]:
    """Extract regex patterns from source code"""
    patterns = []

    # Python: re.compile(), re.search(), re.match(), etc.
    regex_call = re.compile(
        r're\.(?:compile|search|match|findall|sub|split)\s*\(\s*'
        r'(?:r)?"\'["\']'
    )

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                for m in regex_call.finditer(line):
                    patterns.append((lineno, m.group(1)))
    except (UnicodeDecodeError, PermissionError):
        pass

    return patterns

def audit_pattern(pattern: str) -> list[str]:
    """Audit pattern safety"""
    warnings = []

    # Nested quantifiers
    if re.search(r'\([^)]*[+*][^)]*\)[+*]', pattern):
        warnings.append("CRITICAL: Nested quantifiers")

    # Unanchored .*
    if '.*' in pattern and not pattern.startswith('^'):
        warnings.append("WARNING: Unanchored .*")

    # Huge repetition
    repeat = re.search(r'\{(\d+)', pattern)
    if repeat and int(repeat.group(1)) > 100:
        warnings.append(f"WARNING: Large repetition count: {repeat.group(1)}")

    return warnings

def main():
    if len(sys.argv) < 2:
        print("Usage: python regex_audit.py <path>")
        sys.exit(1)

    path = sys.argv[1]
    exit_code = 0

    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                patterns = find_regex_patterns(filepath)
                for lineno, pattern in patterns:
                    warnings = audit_pattern(pattern)
                    for w in warnings:
                        print(f"{filepath}:{lineno}: {w}: {pattern}")
                        if 'CRITICAL' in w:
                            exit_code = max(exit_code, 2)
                        elif 'WARNING' in w:
                            exit_code = max(exit_code, 1)

    if exit_code == 0:
        print("All regex patterns passed audit.")
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
```

---

## 12. Best Practices in Practice

### 12.1 Regex Review Checklist

```
Regex Code Review Checklist:

[ ] Is the purpose and intent of the pattern documented in comments?
[ ] Are test cases sufficient? (normal cases, edge cases, boundary values)
[ ] Are there no nested quantifiers?
[ ] Are there no overlapping alternatives?
[ ] Is .* usage appropriate? (consider anchors or negated character classes)
[ ] Is user input not included in the pattern?
[ ] Is input length limited?
[ ] Is a timeout set? (for external input)
[ ] Is the pattern pre-compiled with re.compile()?
[ ] Are there no unnecessary capture groups?
[ ] Is the pattern not overly complex? (consider splitting)
[ ] Can the regex be replaced with non-regex alternatives?
```

### 12.2 Documenting Regular Expressions

```python
import re

# Example of good documentation
EMAIL_PATTERN = re.compile(
    r'''
    ^                       # Start of string
    [a-zA-Z0-9._%+-]+      # Local part: alphanumeric and special chars
    @                       # @ symbol
    [a-zA-Z0-9.-]+          # Domain: alphanumeric, dots, hyphens
    \.                      # Dot separator
    [a-zA-Z]{2,}            # TLD: 2+ alphabetic characters
    $                       # End of string
    ''',
    re.VERBOSE              # Allow whitespace and comments
)

# When not using the VERBOSE flag, supplement with comments
# Simplified version of RFC 5322. Does not support internationalized domains.
# ReDoS safety: OK (no nested quantifiers, uses negated character classes)
# Maximum input length: should be limited to 254 characters
```

---

## 13. Anti-patterns

### 13.1 Anti-pattern: Using User Input Directly as Regex

```python
import re

# BAD: Using user input directly as a regex
def search_bad(user_input: str, text: str):
    return re.search(user_input, text)  # ReDoS attack possible!

# GOOD: Escape user input
def search_safe(user_input: str, text: str):
    escaped = re.escape(user_input)  # Escape all metacharacters
    return re.search(escaped, text)
```

### 13.2 Anti-pattern: Processing Large Data Without Optimization

```python
import re

# BAD: Processing large logs inefficiently
def process_logs_bad(log_lines: list[str]):
    results = []
    for line in log_lines:
        # Running a complex regex on every line
        m = re.search(r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+\[(\w+)\]\s+(.*)', line)
        if m:
            results.append(m.groups())
    return results

# GOOD: Pre-compile + pre-filter
def process_logs_good(log_lines: list[str]):
    pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+\[(\w+)\]\s+(.*)'
    )
    results = []
    for line in log_lines:
        if '[' in line:  # Pre-filter: only process lines containing [
            m = pattern.search(line)
            if m:
                results.append(m.groups())
    return results
```

### 13.3 Anti-pattern: Overusing Regular Expressions

```python
import re

# BAD: Using regex for simple operations
def check_email_bad(email):
    return bool(re.match(r'.+@.+', email))

# GOOD: String operations are sufficient
def check_email_good(email):
    return '@' in email and '.' in email.split('@')[1]

# BAD: Using regex for fixed-string replacement
text = "Hello World"
result_bad = re.sub(r'World', 'Python', text)

# GOOD: Use str.replace()
result_good = text.replace('World', 'Python')

# BAD: Using regex for fixed-string splitting
data = "a,b,c,d"
parts_bad = re.split(r',', data)

# GOOD: Use str.split()
parts_good = data.split(',')
```

---

## 14. FAQ

### Q1: How can I determine if my pattern is vulnerable to ReDoS?

**A**: Follow these 3 steps:

1. **Visual inspection**: Check for nested quantifiers `(X+)+` or overlapping alternation `(a|a)+`
2. **Use static analysis tools**: `safe-regex`, `recheck`, etc.
3. **Stress test**: Measure speed with long non-matching inputs

```python
import re, time

def stress_test(pattern_str, char='a', max_len=30):
    pattern = re.compile(pattern_str)
    for n in range(10, max_len + 1, 5):
        text = char * n + '!'
        start = time.time()
        pattern.search(text)
        elapsed = time.time() - start
        print(f"  n={n}: {elapsed:.4f}s")
        if elapsed > 1.0:
            print("  -> Vulnerability detected!")
            break
```

### Q2: Does using RE2 solve all problems?

**A**: **No**. RE2 guarantees O(n) but cannot use backreferences or lookaround. Choose with an understanding of the tradeoffs:

- Untrusted input -> Strongly recommend RE2
- Backreferences needed -> NFA + timeout + pattern auditing
- Internal processing only -> NFA is usually fine

### Q3: Does Python's `re` module have a timeout feature?

**A**: The standard `re` module does **not** have one. Countermeasures:

1. External timeout with `signal.alarm()` (Unix-like systems only)
2. `timeout` parameter in the `regex` module (v2021.4.4+)
3. Run in a separate process and timeout with `multiprocessing`
4. Use RE2 bindings (`google-re2` package)

```python
# regex module timeout
import regex
try:
    result = regex.search(r'(a+)+b', 'a' * 30 + 'c', timeout=1)
except TimeoutError:
    print("Timeout")
```

### Q4: When can you avoid using regex altogether?

**A**: The following cases have more efficient alternatives:

```python
# Fixed-string search -> in operator
if 'error' in log_line:  # Faster than re.search(r'error', log_line)

# Prefix/suffix matching -> startswith() / endswith()
if filename.endswith('.py'):  # Faster than re.match(r'.*\.py$', filename)

# Simple splitting -> str.split()
fields = line.split(',')  # Faster than re.split(r',', line)

# Simple replacement -> str.replace()
result = text.replace('old', 'new')  # Faster than re.sub(r'old', 'new', text)

# Parsing structured data -> dedicated parsers
import json  # Use json module for JSON
import csv   # Use csv module for CSV
# HTML -> Beautiful Soup / lxml
# XML -> ElementTree
# URL -> urllib.parse
```

### Q5: Should I use a lazy quantifier or a negated character class?

**A**: Always prefer the negated character class when applicable:

```python
# Lazy quantifier: .*? (backtracking occurs)
# Negated character class: [^X]* (no backtracking)

# Example: Extracting HTML tag attribute values
# Slow: <div class=".*?">
# Fast: <div class="[^"]*">

# Example: Extracting quoted strings
# Slow: ".*?"
# Fast: "[^"]*"

# Example: Extracting comments
# Slow: /\*.*?\*/  (requires DOTALL)
# Fast: /\*[^*]*\*+(?:[^/*][^*]*\*+)*/
# ^ When it becomes complex, consider the readability tradeoff
```

### Q6: What is the difference between atomic groups and possessive quantifiers?

**A**: They provide the same functionality with different syntax. Both prohibit backtracking:

```
Atomic group:          (?>pattern)
Possessive quantifier: pattern++, pattern*+, pattern?+

(?>a+)  is equivalent to  a++
(?>a*)  is equivalent to  a*+
(?>a?)  is equivalent to  a?+

However, atomic groups are more versatile:
(?>abc|ab) -- applies atomic behavior to the entire alternation
^ This cannot be expressed with possessive quantifiers

Support status:
  Possessive quantifier: Java, PCRE, Python regex
  Atomic group: Java, PCRE, Python regex, Perl, .NET
  Neither supported: Python re, JavaScript, Go, Rust
```

---


## FAQ

### Q1: What is the most important point to keep in mind when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next steps.

### Q3: How is this knowledge applied in real-world work?

The knowledge from this topic is frequently used in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Item | Description |
|------|-------------|
| ReDoS | Denial-of-service attack via regular expressions |
| Cause | Backtracking explosion (exponential time complexity) |
| Vulnerable patterns | `(a+)+`, `(a\|a)+`, unrestricted `.*` |
| NFA | Backtracking-based, potentially O(2^n) |
| DFA | Linear time guarantee O(n), limited features |
| Defense 1 | Constrain range with negated character class `[^X]*` |
| Defense 2 | Eliminate nested quantifiers |
| Defense 3 | Use a DFA engine (RE2, etc.) |
| Defense 4 | Set timeout mechanisms |
| Defense 5 | Limit input length |
| Optimization | `re.compile()`, anchors, pre-filters |
| Possessive quantifier | `a++` -- prohibits backtracking |
| Atomic group | `(?>...)` -- prohibits backtracking |
| Golden rule | Use DFA for untrusted input; audit patterns with static analysis |

## Recommended Next Reads

- [../02-practical/00-language-specific.md](../02-practical/00-language-specific.md) -- Language-specific regex (engine differences)
- [../02-practical/03-regex-alternatives.md](../02-practical/03-regex-alternatives.md) -- Alternatives to regular expressions

## References

1. **Russ Cox** "Regular Expression Matching Can Be Simple And Fast" https://swtch.com/~rsc/regexp/regexp1.html, 2007 -- Theoretical background of ReDoS and the design philosophy of RE2
2. **OWASP** "Regular expression Denial of Service - ReDoS" https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS -- ReDoS explained from a security perspective
3. **James Davis et al.** "The Impact of Regular Expression Denial of Service (ReDoS) in Practice" FSE 2018 -- Academic study of ReDoS impact in the real world
4. **Google RE2** https://github.com/google/re2 -- Linear-time-guaranteed regex engine
5. **recheck** https://makenowjust-labs.github.io/recheck/ -- ReDoS vulnerability detection tool for regular expressions
6. **Cloudflare Outage Post-mortem** https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/ -- Real-world ReDoS incident case study
7. **PCRE2 Documentation** https://www.pcre.org/current/doc/html/ -- PCRE2 backtracking limits and optimizations
8. **Rust regex crate** https://docs.rs/regex/ -- Rust's O(n)-guaranteed regex engine
