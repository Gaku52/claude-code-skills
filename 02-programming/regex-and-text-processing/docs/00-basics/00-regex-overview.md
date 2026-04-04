# Regular Expression Overview

> A systematic guide to the historical background of Regular Expressions, their major use cases, and the internal workings of NFA/DFA engines.

## What You Will Learn in This Chapter

1. **Mathematical Origins and Historical Evolution of Regular Expressions** -- From Kleene's regular sets to PCRE
2. **The Two Major Regular Expression Engine Types (NFA/DFA)** -- Operating principles, performance characteristics, and selection criteria
3. **Application Domains and Limitations of Regular Expressions** -- From text search to compilers: when to use and when to avoid
4. **Implementation Comparison Across Major Engines** -- Engine characteristics and selection guidelines for each programming language
5. **Debugging and Testing Strategies for Regular Expressions** -- Efficient pattern development methods


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. What Are Regular Expressions?

Regular expressions are a formal language for **pattern matching**. They express sets of strings using finite notation and are used for searching, replacing, extracting, and validating.

```
Pattern: \d{3}-\d{4}
Target string: "The postal code is 100-0001"
Match result: "100-0001"
```

### 1.1 Basic Operating Model

```
Input string ──→ [Regex Engine] ──→ Match result
                    ↑
               Pattern (Regular Expression)
```

The regex engine internally converts the given pattern into an automaton (finite state machine) and processes the input string character by character to perform matching.

### 1.2 Components of Regular Expressions

Regular expression patterns are composed of the following basic elements:

```
Components of Regular Expressions:

1. Literal characters    -- 'a', 'b', '1', etc., match the character directly
2. Metacharacters        -- '.', '*', '+', '?', '|', etc., have special meaning
3. Character classes     -- [abc], [a-z], \d, \w, etc., represent sets of characters
4. Quantifiers           -- {n}, {n,m}, *, +, ?, etc., specify repetition
5. Anchors               -- ^, $, \b, etc., specify positions
6. Grouping              -- (), (?:), (?=), etc., group sub-patterns together
7. Escaping              -- \., \\, \n, etc., disable metacharacters or represent special characters
```

### 1.3 Overall Flow of Regular Expression Processing

```
                 Regex Pattern
                       │
                       ▼
               ┌──────────────┐
               │  Lexical      │  Decompose pattern string into a token sequence
               │  Analysis     │
               └──────┬───────┘
                       │
                       ▼
               ┌──────────────┐
               │  Syntax       │  Convert token sequence into an Abstract Syntax Tree (AST)
               │  Analysis     │
               └──────┬───────┘
                       │
                       ▼
               ┌──────────────┐
               │  Automaton    │  Build NFA/DFA from AST
               │  Construction │
               └──────┬───────┘
                       │
                       ▼
               ┌──────────────┐
               │  Matching     │  Execute automaton against the input string
               │  Execution    │
               └──────┬───────┘
                       │
                       ▼
                 Match Result
```

### 1.4 Types of Regular Expression Notation

```
Major notation systems:

1. POSIX BRE (Basic Regular Expression)
   - Metacharacters require \: \(, \), \{, \}, \+, \?
   - Example: grep 'a\(b\|c\)d' file.txt

2. POSIX ERE (Extended Regular Expression)
   - Metacharacters used directly: (, ), {, }, +, ?
   - Example: grep -E 'a(b|c)d' file.txt

3. PCRE (Perl Compatible Regular Expressions)
   - Extensions to ERE: lookahead/lookbehind, non-greedy quantifiers, named captures
   - Example: grep -P '(?<=prefix)\w+' file.txt

4. ECMAScript (JavaScript)
   - Subset of PCRE + proprietary extensions (u flag, s flag, etc.)
   - Example: /pattern/gimsuvy

5. RE2 Syntax
   - PCRE minus features requiring backtracking
   - Example: no backreferences, no lookahead/lookbehind
```

Since these notation systems are often incompatible, it is important to understand the regex dialect of the tool or programming language you are using.

### 1.5 How to Read Regular Expression Patterns

Here is the procedure for reading complex regular expressions:

```python
# Example: Simple email address pattern
pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# Breaking it down:
# ^                    -- Start of string
# [a-zA-Z0-9._%+-]+   -- Local part (one or more alphanumeric characters and symbols)
# @                    -- Literal '@'
# [a-zA-Z0-9.-]+      -- Domain name (one or more alphanumeric characters, hyphens, and dots)
# \.                   -- Literal '.'
# [a-zA-Z]{2,}        -- TLD (two or more alphabetic characters)
# $                    -- End of string
```

```python
# Example: Japanese phone number pattern
pattern = r'^0\d{1,4}-\d{1,4}-\d{4}$'

# Breaking it down:
# ^           -- Start of string
# 0           -- Literal '0' (beginning of area code)
# \d{1,4}     -- 1 to 4 digits
# -           -- Literal '-'
# \d{1,4}     -- 1 to 4 digits (local exchange number)
# -           -- Literal '-'
# \d{4}       -- 4 digits (subscriber number)
# $           -- End of string
```

---

## 2. Historical Evolution

### 2.1 Timeline

```
1943  McCulloch & Pitts  ─ Mathematical model of neural networks
  │
1956  Stephen Kleene     ─ Formalization of Regular Sets theory
  │
1959  Michael Rabin &    ─ Formalization of Nondeterministic Finite Automata (NFA)
      Dana Scott           Turing Award recipients (1976)
  │
1968  Ken Thompson       ─ Implemented regex in QED/ed editors
  │                        Directly simulated NFA on IBM 7094
  │
1973  Thompson & Ritchie ─ Birth of grep (Unix V4)
  │                        "Global Regular Expression Print"
  │
1975  Alfred Aho         ─ Development of egrep (DFA-based)
  │                        Author of "Compilers: Principles, Techniques, and Tools"
  │
1979  AT&T Unix V7       ─ Introduction of awk (Aho, Weinberger, Kernighan)
  │                        Integrated regex into a programming language
  │
1986  Henry Spencer      ─ First free regex library
  │                        Became the foundation for many UNIX tools
  │
1986  POSIX Standardization ─ Standardized BRE/ERE as IEEE Std 1003.2
  │
1987  Larry Wall         ─ Shipped advanced regex in Perl 1.0
  │                        Introduced backreferences, lookahead, etc.
  │
1994  Perl 5.0           ─ Major regex extensions
  │                        Non-greedy quantifiers, lookahead/lookbehind
  │
1997  Philip Hazel       ─ PCRE (Perl Compatible Regular Expressions)
  │                        Made Perl's regex available as an independent library
  │
2002  .NET Framework     ─ Introduced balancing groups
  │                        Enabled limited matching of nested structures
  │
2006  Russ Cox           ─ RE2 (linear-time guaranteed engine)
  │                        Developed at Google, eliminates ReDoS by design
  │
2012  Rust regex crate   ─ Rust implementation inheriting RE2's philosophy
  │                        Combines safety and performance
  │
2017  ECMAScript 2018    ─ Standardized named captures and lookbehind
  │                        Added s flag (dotAll)
  │
2022  PCRE2 10.40+       ─ Improved JIT compilation
  │                        Significant performance improvements
  │
2024  ECMAScript 2024    ─ Added v flag (Unicode Sets)
                           Supports set operations in character classes
```

### 2.2 Major Milestones

| Year | Person/Project | Contribution |
|------|----------------|--------------|
| 1956 | Stephen Kleene | Mathematically formalized the concept of "regular expression" |
| 1959 | Rabin & Scott | Proved the equivalence of NFA/DFA (Turing Award) |
| 1968 | Ken Thompson | Implemented the first practical regex engine in the QED editor |
| 1973 | grep (Unix) | `g/re/p` -- Global Regular Expression Print |
| 1975 | egrep (Unix) | DFA-based high-speed regex matching |
| 1979 | awk | Integrated regex into a text processing language |
| 1986 | POSIX | Standardized BRE/ERE (Basic/Extended Regular Expressions) |
| 1987 | Perl | Added backreferences, lookahead, etc.; became the de facto standard |
| 1994 | Perl 5 | Non-greedy quantifiers, regex within code blocks |
| 1997 | PCRE | Provided a Perl-compatible engine as an independent library |
| 2002 | .NET | Addressed nested structures with balancing groups |
| 2006 | RE2 (Google) | DFA-based, eliminates ReDoS by design |
| 2017 | ES2018 | Added lookbehind and named captures to JavaScript |

### 2.3 Mathematical Foundations of Regular Expression Theory

Regular expression theory is closely tied to automaton theory, a fundamental area of computer science:

```
Kleene's Theorem (1956):

"Regular languages," "languages describable by regular expressions," and
"languages accepted by finite automata" are all equivalent.

That is:
  Regular expressions ⟺ NFA ⟺ DFA

Conversion directions:
  Regular expression → NFA  : Thompson's construction
  NFA → DFA                : Subset construction
  DFA → Regular expression : State elimination
  DFA → Minimal DFA        : Hopcroft's algorithm
```

```python
# Conceptual implementation of Thompson's construction
# NFA construction corresponding to basic regex operations

class NFAState:
    """State in an NFA"""
    def __init__(self):
        self.transitions = {}  # character -> [next states]
        self.epsilon = []      # epsilon transition targets
        self.is_accept = False

class NFAFragment:
    """Fragment of an NFA (partial NFA under construction)"""
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

def literal(char):
    """NFA for a literal character 'a'"""
    start = NFAState()
    accept = NFAState()
    accept.is_accept = True
    start.transitions[char] = [accept]
    return NFAFragment(start, accept)

def concatenation(frag1, frag2):
    """NFA for concatenation ab"""
    frag1.accept.is_accept = False
    frag1.accept.epsilon.append(frag2.start)
    return NFAFragment(frag1.start, frag2.accept)

def alternation(frag1, frag2):
    """NFA for alternation a|b"""
    start = NFAState()
    accept = NFAState()
    accept.is_accept = True
    start.epsilon.extend([frag1.start, frag2.start])
    frag1.accept.is_accept = False
    frag2.accept.is_accept = False
    frag1.accept.epsilon.append(accept)
    frag2.accept.epsilon.append(accept)
    return NFAFragment(start, accept)

def kleene_star(frag):
    """NFA for Kleene closure a*"""
    start = NFAState()
    accept = NFAState()
    accept.is_accept = True
    start.epsilon.extend([frag.start, accept])
    frag.accept.is_accept = False
    frag.accept.epsilon.extend([frag.start, accept])
    return NFAFragment(start, accept)
```

### 2.4 POSIX Standards and Dialect Divergence

```
The two POSIX regex standards:

┌─────────────────────────────────────────────────────────────┐
│  BRE (Basic Regular Expression)                             │
│                                                             │
│  Characteristics:                                           │
│  - Grouping as metacharacters: \( and \)                    │
│  - Alternation as metacharacter: not supported              │
│    (some implementations support \|)                        │
│  - Quantifiers as metacharacters: \{ and \}                 │
│  - +, ? are literal characters                              │
│    (some implementations support \+, \?)                    │
│                                                             │
│  Used by: grep (default), sed (default)                     │
│                                                             │
│  Example: grep 'a\{2,3\}' file.txt                         │
│      → Matches 'aa' or 'aaa'                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ERE (Extended Regular Expression)                          │
│                                                             │
│  Characteristics:                                           │
│  - Grouping: ( and )                                        │
│  - Alternation: |                                           │
│  - Quantifiers: { and }                                     │
│  - +, ? are metacharacters                                  │
│                                                             │
│  Used by: grep -E (egrep), sed -E, awk                      │
│                                                             │
│  Example: grep -E 'a{2,3}' file.txt                        │
│      → Matches 'aa' or 'aaa'                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Types of Regular Expression Engines

### 3.1 NFA vs DFA -- The Two Major Approaches

```
┌──────────────────────────────────────────────────────┐
│            Regular Expression Engines                  │
├─────────────────────┬────────────────────────────────┤
│   NFA (Nondeterministic) │    DFA (Deterministic)     │
│                     │                                │
│ - Backtracking-based │ - State transition table-based │
│ - Pattern-driven     │ - Text-driven                  │
│ - Supports backreferences │ - No backreference support│
│ - Worst case O(2^n)  │ - Always O(n)                 │
│                     │                                │
│ e.g.: Perl, Python, │ e.g.: awk, grep (some),       │
│     Java, .NET,     │     RE2, Rust regex            │
│     JavaScript      │                                │
└─────────────────────┴────────────────────────────────┘
```

### 3.2 NFA Operation Example

Matching the string `"acd"` against the pattern `a(b|c)d`:

```python
# NFA (Nondeterministic Finite Automaton) operation simulation
import re

pattern = r'a(b|c)d'
text = "acd"

# Internal operation:
# 1. State S0: read 'a' → match → transition to S1
# 2. State S1: read 'c' → try 'b' → fail
#                        → backtrack and try 'c' → match → transition to S2
# 3. State S2: read 'd' → match → accept state

result = re.match(pattern, text)
print(result.group())   # => "acd"
print(result.group(1))  # => "c"
```

### 3.3 DFA Operation Example

```python
# DFA (Deterministic Finite Automaton) expands all states in advance
# Pattern: a(b|c)d

# State transition table:
# Current State | Input 'a' | Input 'b' | Input 'c' | Input 'd'
# --------------|-----------|-----------|-----------|----------
# S0            | S1        | -         | -         | -
# S1            | -         | S2        | S2        | -
# S2            | -         | -         | -         | S3 (accept)

# DFA does not produce backtracking
# Only one state transition per character → O(n)

# Rust's regex crate is DFA-based
# RE2 is also DFA-based
```

### 3.4 Comparison Table: NFA vs DFA

| Property | NFA | DFA |
|----------|-----|-----|
| Time complexity (worst case) | O(2^n) -- exponential | O(n) -- linear |
| Time complexity (average) | O(n) -- practically fast | O(n) -- always linear |
| Space complexity | O(m) pattern size | O(2^m) worst case (state explosion) |
| Backreferences | Supported | Not supported |
| Lookahead/lookbehind | Supported | Limited/not supported |
| Lazy quantifiers | Supported | N/A (leftmost longest match) |
| ReDoS vulnerability | Yes | No |
| Implementation complexity | Relatively simple | Complex state table construction |
| Compile time | Short | Long (state expansion) |
| First match | Fast (left to right) | Varies by pattern |
| Representative implementations | Perl, Python, Java, JS | RE2, Rust regex, awk |

### 3.5 NFA Backtracking in Detail

```
Pattern: a.*b
Text: "axyzb123"

Step 1: 'a' → match
Step 2: '.*' → greedily consumes all characters "xyzb123"
Step 3: 'b' → match failure (end of string)
Step 4: backtrack → '.*' gives back to "xyzb12"
Step 5: 'b' → no match with '3'
Step 6: backtrack → '.*' gives back to "xyzb1"
Step 7: 'b' → no match with '2'
Step 8: backtrack → '.*' gives back to "xyzb"
Step 9: 'b' → no match with '1'
Step 10: backtrack → '.*' gives back to "xyz"
Step 11: 'b' → matches 'b'!

Result: "axyzb"
Backtrack count: 4 times

* Non-greedy version a.*?b:
Step 1: 'a' → match
Step 2: '.*?' → consumes minimum of 0 characters
Step 3: 'b' → no match with 'x'
Step 4: '.*?' → consumes 1 character "x"
Step 5: 'b' → no match with 'y'
Step 6: '.*?' → consumes 2 characters "xy"
Step 7: 'b' → no match with 'z'
Step 8: '.*?' → consumes 3 characters "xyz"
Step 9: 'b' → matches 'b'!

Result: "axyzb" (same result but different path taken)
```

### 3.6 Hybrid Approach

```
┌─────────────────────────────────────────┐
│           Hybrid Engine                  │
│                                         │
│  Pattern Analysis                        │
│      │                                   │
│      ├── No backreferences → Execute     │
│      │   with DFA                        │
│      │                                   │
│      └── Has backreferences → Fall back  │
│          to NFA                          │
│                                         │
│  Examples: .NET, Rust's fancy-regex      │
└─────────────────────────────────────────┘
```

### 3.7 Regex Engine List by Language

| Language/Tool | Engine Type | Library | Notes |
|--------------|-------------|---------|-------|
| Python | NFA | `re` (C implementation) | Extensible via `regex` module |
| JavaScript | NFA | V8 Irregexp | JIT-optimized |
| Java | NFA | `java.util.regex` | No atomic groups (partial support in Java 9+) |
| C# (.NET) | NFA | `System.Text.RegularExpressions` | Balancing groups supported |
| Perl | NFA | Built-in | Most feature-rich NFA implementation |
| Ruby | NFA | Onigmo | Comprehensive Unicode support |
| Go | DFA | `regexp` (RE2-based) | No backreference support |
| Rust | DFA | `regex` crate | Linear-time guarantee |
| PHP | NFA | PCRE2 | `preg_*` function family |
| C/C++ | Both | PCRE2, RE2, std::regex | Selectable |
| awk | DFA | Built-in | ERE-compliant |
| grep | Both | GNU grep | `-G` BRE, `-E` ERE, `-P` PCRE |
| sed | NFA | Built-in | BRE (default), ERE (`-E`) |

### 3.8 Engine Selection Flowchart

```
Does the pattern contain backreferences?
    │
    ├── Yes → Use an NFA engine
    │          │
    │          ├── Is the input untrusted?
    │          │   │
    │          │   ├── Yes → Timeout setting is mandatory
    │          │   │          (.NET: MatchTimeout,
    │          │   │           Java: interrupt,
    │          │   │           Python: signal.alarm)
    │          │   │
    │          │   └── No → Use NFA as-is
    │          │
    │          └── Is performance an issue?
    │              │
    │              ├── Yes → Consider rewriting the pattern
    │              │          (atomic groups, possessive quantifiers)
    │              │
    │              └── No → Use as-is
    │
    └── No → Is a DFA engine available?
                │
                ├── Yes → Use DFA (RE2, Rust regex, Go regexp)
                │          Linear-time guarantee for safety
                │
                └── No → NFA is fine
                             (avoid ReDoS patterns)
```

---

## 4. Major Use Cases for Regular Expressions

### 4.1 Code Examples by Use Case

```bash
# 1. Text search (grep)
grep -E 'ERROR|WARN' /var/log/syslog

# 2. Text replacement (sed)
sed 's/2025/2026/g' document.txt

# 3. Data extraction (Python)
python3 -c "
import re
log = '2026-02-11 10:30:45 [ERROR] Connection timeout (192.168.1.1)'
m = re.search(r'(\d{4}-\d{2}-\d{2}) .* \[(\w+)\] (.+)', log)
print(f'Date: {m.group(1)}, Level: {m.group(2)}, Message: {m.group(3)}')
"

# 4. Input validation (JavaScript)
node -e "
const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
console.log(emailPattern.test('user@example.com'));  // true
console.log(emailPattern.test('invalid@'));           // false
"

# 5. Syntax highlighting -- How editors colorize keywords
# Pattern example: \b(if|else|for|while|return)\b → colored as keywords

# 6. Filename matching (find + grep)
find /var/log -name "*.log" -exec grep -l 'CRITICAL' {} \;

# 7. CSV data transformation (awk)
awk -F',' '/^2026/ {print $1, $3}' data.csv

# 8. Extract TODO comments from code
grep -rn 'TODO\|FIXME\|HACK\|XXX' --include='*.py' src/
```

### 4.2 Practical Log Analysis Example

```python
import re
from collections import Counter, defaultdict
from datetime import datetime

# Apache access log analysis
log_pattern = re.compile(
    r'(?P<ip>\d+\.\d+\.\d+\.\d+)'       # IP address
    r' - - '
    r'\[(?P<timestamp>[^\]]+)\]'           # Timestamp
    r' "(?P<method>\w+)'                   # HTTP method
    r' (?P<path>[^\s]+)'                   # Request path
    r' HTTP/[\d.]+"'                       # HTTP version
    r' (?P<status>\d{3})'                  # Status code
    r' (?P<size>\d+|-)'                    # Response size
    r'(?: "(?P<referer>[^"]*)")?'          # Referer (optional)
    r'(?: "(?P<useragent>[^"]*)")?'        # User agent (optional)
)

def analyze_access_log(log_file: str):
    """Analyze access log and output statistical information"""
    status_counter = Counter()
    path_counter = Counter()
    ip_counter = Counter()
    hourly_access = defaultdict(int)
    error_logs = []

    with open(log_file) as f:
        for line in f:
            m = log_pattern.match(line)
            if not m:
                continue

            data = m.groupdict()
            status = int(data['status'])
            path = data['path']
            ip = data['ip']

            status_counter[status] += 1
            path_counter[path] += 1
            ip_counter[ip] += 1

            # Hourly aggregation
            try:
                dt = datetime.strptime(
                    data['timestamp'],
                    '%d/%b/%Y:%H:%M:%S %z'
                )
                hourly_access[dt.hour] += 1
            except ValueError:
                pass

            # Collect error logs
            if status >= 400:
                error_logs.append({
                    'ip': ip,
                    'path': path,
                    'status': status,
                    'timestamp': data['timestamp']
                })

    return {
        'total_requests': sum(status_counter.values()),
        'status_distribution': dict(status_counter),
        'top_paths': path_counter.most_common(10),
        'top_ips': ip_counter.most_common(10),
        'hourly_distribution': dict(hourly_access),
        'error_count': len(error_logs),
        'recent_errors': error_logs[-10:]
    }
```

### 4.3 Practical Data Cleansing Example

```python
import re

def clean_text(text: str) -> str:
    """Text data cleansing"""
    # Normalize consecutive whitespace to a single space
    text = re.sub(r'\s+', ' ', text)

    # Convert full-width alphanumeric characters to half-width
    text = re.sub(r'[Ａ-Ｚａ-ｚ０-９]',
                  lambda m: chr(ord(m.group()) - 0xFEE0), text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove control characters (preserve newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text

def normalize_phone_number(phone: str) -> str:
    """Phone number normalization"""
    # Remove non-digit characters
    digits = re.sub(r'\D', '', phone)

    # Handle international phone numbers
    if digits.startswith('81') and len(digits) >= 11:
        digits = '0' + digits[2:]

    # Format (landline)
    if len(digits) == 10:
        m = re.match(r'(\d{2,4})(\d{2,4})(\d{4})', digits)
        if m:
            return f'{m.group(1)}-{m.group(2)}-{m.group(3)}'

    # Format (mobile)
    if len(digits) == 11:
        m = re.match(r'(\d{3})(\d{4})(\d{4})', digits)
        if m:
            return f'{m.group(1)}-{m.group(2)}-{m.group(3)}'

    return phone  # Return as-is if conversion is not possible

def extract_urls(text: str) -> list:
    """Extract URLs from text"""
    url_pattern = re.compile(
        r'https?://'                    # Scheme
        r'(?:[a-zA-Z0-9]'              # First character of domain
        r'(?:[a-zA-Z0-9-]{0,61}'       # Domain name body
        r'[a-zA-Z0-9])?\.)'            # End of domain
        r'+[a-zA-Z]{2,}'               # TLD
        r'(?::\d{1,5})?'               # Port (optional)
        r'(?:/[^\s]*)?'                # Path (optional)
    )
    return url_pattern.findall(text)

def mask_personal_info(text: str) -> str:
    """Mask personal information"""
    # Email addresses
    text = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '***@***.***',
        text
    )

    # Phone numbers (Japan)
    text = re.sub(
        r'0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4}',
        '***-****-****',
        text
    )

    # Credit card numbers
    text = re.sub(
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        '****-****-****-****',
        text
    )

    # My Number (Japanese 12-digit identification number)
    text = re.sub(
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        '****-****-****',
        text
    )

    return text
```

### 4.4 Application Domain Map

| Domain | Typical Usage | Recommendation | Notes |
|--------|--------------|----------------|-------|
| Log analysis | Error pattern extraction, aggregation | Optimal | Especially effective for structured logs |
| Input validation | Email, phone numbers, postal codes | Suitable | Don't over-rely; two-stage verification recommended |
| Text editors | Search and replace | Optimal | Essential for bulk replacement in IDEs |
| Web scraping | Information extraction from HTML | Caution | HTML parser recommended; regex as supplementary |
| Compilers/lexical analysis | Token splitting | Suitable | Used with lexer generators |
| Natural language processing | Preprocessing for morphological analysis | Limited | Combine with dedicated libraries |
| Data migration | Format conversion | Suitable | CSV, TSV column operations, etc. |
| Security | WAF rule definitions | Caution | Must consider ReDoS risk |
| Binary analysis | Pattern detection | Unsuitable | Use dedicated binary tools |
| Configuration files | Template expansion | Limited | Dedicated template engines recommended |

### 4.5 Regular Expressions in Text Processing Pipelines

```bash
# Regex usage in Unix pipelines

# Example 1: Aggregate IP addresses from error responses in access logs
cat access.log \
  | grep -E '" [45]\d{2} ' \
  | awk '{print $1}' \
  | sort | uniq -c | sort -rn \
  | head -20

# Example 2: Extract function definitions from source code
grep -rn 'def \w\+(' --include='*.py' src/ \
  | sed 's/.*def \(\w\+\)(.*/\1/' \
  | sort | uniq -c | sort -rn

# Example 3: Extract error messages from JSON logs
cat app.log \
  | grep -oP '"error":\s*"[^"]*"' \
  | sed 's/"error":\s*"\(.*\)"/\1/' \
  | sort | uniq -c | sort -rn

# Example 4: Extract ticket numbers from Git log
git log --oneline \
  | grep -oE '[A-Z]+-[0-9]+' \
  | sort | uniq -c | sort -rn
```

---

## 5. Limitations of Regular Expressions

### 5.1 What Regular Expressions Cannot Express

```
Chomsky Hierarchy:
┌─────────────────────────────────────┐
│ Type 0: Recursively Enumerable     │
│  ┌──────────────────────────────┐   │
│  │ Type 1: Context-Sensitive    │   │
│  │  ┌───────────────────────┐   │   │
│  │  │ Type 2: Context-Free  │   │   │
│  │  │  ┌────────────────┐   │   │   │
│  │  │  │ Type 3: Regular │   │   │   │
│  │  │  │ (Regex)         │   │   │   │
│  │  │  └────────────────┘   │   │   │
│  │  │  e.g.: HTML, JSON,    │   │   │
│  │  │  programming languages│   │   │
│  │  └───────────────────────┘   │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘

Regular expressions (Type 3) cannot count
matching pairs of nested parentheses.
Examples: ((())), {{{}}}, <div><div></div></div>
```

### 5.2 Concrete Examples of Theoretical Limitations

```python
# Patterns that regular expressions fundamentally cannot handle

# 1. Matching nested parentheses
#    a^n b^n (n 'a's followed by n 'b's) is not a regular language
#    Example: ab, aabb, aaabbb are accepted; aab, abbb are rejected
#    → Cannot be expressed with regular expressions

# 2. Palindrome recognition
#    Example: "abcba", "racecar"
#    → Cannot be expressed with regular expressions

# 3. Strings of prime length
#    Match only strings whose length is a prime number
#    → Cannot be expressed with regular expressions

# However, practical "regular expressions" (PCRE, etc.)
# have features that exceed theoretical regular languages:

import re

# .NET's balancing groups can match nested parentheses
# (theoretically in the domain of CFGs)
# (?<open>\()  (?<close-open>\))

# Perl/PCRE recursive patterns can also do this
# \((?:[^()]*|(?R))*\)
```

### 5.3 Practical Limitations

```python
# Cases where regular expressions are not suitable

# 1. Parsing complex grammars
# BAD: Parsing JSON with regex
json_text = '{"name": "John", "address": {"city": "Tokyo"}}'
# → Use a JSON parser instead

# 2. Advanced multibyte character processing
# BAD: Guessing kanji readings with regex
# → Use a morphological analyzer (MeCab, Janome, etc.) instead

# 3. Context-dependent parsing
# BAD: Python's indent-based block structure
# → Use a parser (AST) instead

# 4. Structural analysis of large-scale text
# BAD: Processing hundreds of MB of XML files with regex
# → Use streaming processing with a SAX parser, etc.

# 5. Semantic analysis of natural language
# BAD: Extracting subject and predicate from sentences with regex
# → Use NLP libraries (spaCy, GiNZA, etc.) instead
```

---

## 6. Anti-patterns

### 6.1 Anti-pattern: Parsing HTML with Regular Expressions

```python
# BAD: Attempting to process HTML with regular expressions
import re

html = '<div class="outer"><div class="inner">text</div></div>'

# This pattern cannot correctly handle nested divs
pattern = r'<div[^>]*>(.*?)</div>'
result = re.findall(pattern, html)
print(result)  # => ['<div class="inner">text']  -- inaccurate

# GOOD: Use an HTML parser
from html.parser import HTMLParser
# Or use Beautiful Soup, lxml, etc.
```

**Reason**: HTML is a context-free language, which exceeds the expressive power of regular expressions (regular languages). It cannot correctly track the correspondence of nested elements.

### 6.2 Anti-pattern: All-encompassing Validation Patterns

```python
# BAD: RFC 5322 fully compliant email address pattern (this actually exists)
# A regex spanning thousands of characters → unmaintainable, undebuggable

email_pattern_bad = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@..."""
# (truncated -- actually hundreds of characters or more)

# GOOD: Practical validation + confirmation email
import re

def validate_email(email: str) -> bool:
    """Practical email validation -- format check + confirmation email"""
    # Check basic format only
    if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
        return False
    # The real validation is done by sending a confirmation email
    return True
```

**Reason**: Combining a simple pattern with a separate validation method is more maintainable and reliable than trying to write a perfect regex.

### 6.3 Anti-pattern: ReDoS-Vulnerable Patterns

```python
# BAD: Patterns that cause Catastrophic Backtracking
import re
import time

# Examples of dangerous patterns
dangerous_patterns = [
    r'(a+)+$',           # Nested quantifiers
    r'(a|a)+$',          # Overlapping alternatives
    r'(a+b?)+$',         # Combinatorial explosion
    r'([a-zA-Z]+)*$',    # Nested quantifiers (character class version)
]

# Attack string: many 'a's followed by a non-matching character
evil_input = 'a' * 30 + '!'

for pattern in dangerous_patterns:
    start = time.time()
    try:
        re.match(pattern, evil_input)
    except Exception:
        pass
    elapsed = time.time() - start
    print(f'Pattern: {pattern:30s} Time: {elapsed:.3f}s')
    # Some patterns may take seconds to tens of seconds

# GOOD: Writing ReDoS-resistant patterns
safe_patterns = [
    r'a+$',              # Avoid nesting
    r'(?:a+)+$',         # Non-capturing is equally dangerous → rewrite to a+$
    r'[a-zA-Z]+$',       # Avoid nesting
]

# GOOD: Matching with timeout (Python 3.11+)
# import signal
# signal.alarm(1)  # 1-second timeout
```

### 6.4 Anti-pattern: Low-Readability Patterns

```python
# BAD: A long pattern crammed into a single line
pattern_bad = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'

# GOOD: Using verbose mode for readability
import re

pattern_good = re.compile(r'''
    ^
    (?:
        (?:
            25[0-5]           # 250-255
            | 2[0-4][0-9]     # 200-249
            | [01]?[0-9][0-9]? # 0-199
        )
        \.                     # Dot separator
    ){3}                       # First three octets
    (?:
        25[0-5]               # 250-255
        | 2[0-4][0-9]         # 200-249
        | [01]?[0-9][0-9]?    # 0-199
    )
    $
''', re.VERBOSE)

# Tests
assert pattern_good.match('192.168.1.1')
assert pattern_good.match('255.255.255.255')
assert not pattern_good.match('256.1.1.1')
assert not pattern_good.match('1.2.3.4.5')
```

### 6.5 Anti-pattern: Compilation Without Performance Consideration

```python
import re

# BAD: Compiling on every iteration in a loop
def search_bad(lines, pattern_str):
    results = []
    for line in lines:
        if re.search(pattern_str, line):  # Compiles every time
            results.append(line)
    return results

# GOOD: Pre-compile the pattern
def search_good(lines, pattern_str):
    pattern = re.compile(pattern_str)  # Compile only once
    results = []
    for line in lines:
        if pattern.search(line):       # Reuse compiled object
            results.append(line)
    return results

# GOOD: An even better approach (list comprehension)
def search_best(lines, pattern_str):
    pattern = re.compile(pattern_str)
    return [line for line in lines if pattern.search(line)]

# Note: Python's re module has an internal pattern cache (up to 512 entries),
#    so re.search() is fine for a small number of patterns,
#    but explicit compilation is recommended
```

---

## 7. Debugging and Testing Regular Expressions

### 7.1 Incremental Pattern Construction

```python
import re

# Build complex patterns incrementally

# Goal: Pattern for analyzing Apache logs
# Sample: 192.168.1.1 - - [10/Feb/2026:13:55:36 +0900] "GET /index.html HTTP/1.1" 200 2326

# Step 1: IP address portion
step1 = r'\d+\.\d+\.\d+\.\d+'
assert re.search(step1, '192.168.1.1')

# Step 2: Timestamp portion
step2 = r'\[([^\]]+)\]'
assert re.search(step2, '[10/Feb/2026:13:55:36 +0900]')

# Step 3: Request line
step3 = r'"(\w+) ([^\s]+) HTTP/[\d.]+"'
assert re.search(step3, '"GET /index.html HTTP/1.1"')

# Step 4: Status code and size
step4 = r'(\d{3}) (\d+|-)'
assert re.search(step4, '200 2326')

# Step 5: Combine everything
full_pattern = re.compile(
    rf'({step1})'     # IP
    r' - - '          # ident, auth
    rf'{step2}'       # timestamp
    r' '
    rf'{step3}'       # request
    r' '
    rf'{step4}'       # status, size
)

log_line = '192.168.1.1 - - [10/Feb/2026:13:55:36 +0900] "GET /index.html HTTP/1.1" 200 2326'
m = full_pattern.match(log_line)
assert m is not None
print(f'IP: {m.group(1)}')
print(f'Timestamp: {m.group(2)}')
print(f'Method: {m.group(3)}')
print(f'Path: {m.group(4)}')
print(f'Status: {m.group(5)}')
print(f'Size: {m.group(6)}')
```

### 7.2 Test-Driven Regex Development

```python
import re
import pytest

class TestPhoneNumberPattern:
    """Tests for phone number pattern"""

    pattern = re.compile(r'^0\d{1,4}-\d{1,4}-\d{4}$')

    # Positive cases: should match
    @pytest.mark.parametrize("phone", [
        "03-1234-5678",       # Tokyo
        "06-1234-5678",       # Osaka
        "090-1234-5678",      # Mobile
        "080-1234-5678",      # Mobile
        "0120-123-4567",      # Toll-free
        "0466-12-3456",       # 4-digit area code
    ])
    def test_valid_phones(self, phone):
        assert self.pattern.match(phone), f"{phone} should match"

    # Negative cases: should not match
    @pytest.mark.parametrize("phone", [
        "1234-5678",          # Does not start with 0
        "03-1234-567",        # Subscriber number is 3 digits
        "03-1234-56789",      # Subscriber number is 5 digits
        "abc-defg-hijk",      # Not digits
        "",                   # Empty string
        "03 1234 5678",       # Spaces instead of hyphens
    ])
    def test_invalid_phones(self, phone):
        assert not self.pattern.match(phone), f"{phone} should not match"

class TestEmailPattern:
    """Tests for email address pattern"""

    pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )

    @pytest.mark.parametrize("email", [
        "user@example.com",
        "user.name@example.co.jp",
        "user+tag@example.org",
        "user123@sub.domain.com",
    ])
    def test_valid_emails(self, email):
        assert self.pattern.match(email)

    @pytest.mark.parametrize("email", [
        "user@",
        "@example.com",
        "user@.com",
        "user@com",
        "",
        "user space@example.com",
    ])
    def test_invalid_emails(self, email):
        assert not self.pattern.match(email)
```

### 7.3 Using Debugging Tools

```python
# Use Python's re.DEBUG flag to inspect the internal structure of patterns
import re

# Display the parse result of a pattern
re.compile(r'\d{3}-\d{4}', re.DEBUG)
# Output:
# MAX_REPEAT 3 3
#   IN
#     CATEGORY CATEGORY_DIGIT
# LITERAL 45 ('-')
# MAX_REPEAT 4 4
#   IN
#     CATEGORY CATEGORY_DIGIT

# Verbose mode with commented patterns
pattern = re.compile(r"""
    (?P<year>\d{4})     # Year: 4-digit number
    [-/]                # Separator: hyphen or slash
    (?P<month>\d{2})    # Month: 2-digit number
    [-/]                # Separator
    (?P<day>\d{2})      # Day: 2-digit number
""", re.VERBOSE)

result = pattern.match('2026-02-15')
if result:
    print(result.groupdict())
    # => {'year': '2026', 'month': '02', 'day': '15'}
```

---

## 8. Regular Expression Best Practices

### 8.1 Design Principles

```
Design principles for regular expressions:

1. KISS Principle (Keep It Simple, Stupid)
   - Write the minimum necessary pattern
   - Aim for 80% coverage instead of perfection
   - Handle the remaining 20% with other logic

2. Incremental Refinement
   - Start with a simple pattern and gradually increase precision
   - Add tests at each stage

3. Comments Are Mandatory
   - Add comments to patterns longer than 10 characters
   - Verbose mode is recommended

4. Performance Consideration
   - Pre-compilation is the baseline
   - Avoid ReDoS patterns
   - Use anchors to limit search scope

5. Test-Driven
   - Write tests for positive cases, negative cases, and boundary values
   - Explicitly cover edge cases
```

### 8.2 Naming Conventions

```python
# Maintain consistency in pattern naming

# Validation patterns: is_* or validate_*
PATTERN_IS_EMAIL = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
PATTERN_IS_URL = re.compile(r'^https?://[^\s]+$')
PATTERN_IS_IPV4 = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')

# Extraction patterns: extract_* or parse_*
PATTERN_EXTRACT_DATE = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
PATTERN_EXTRACT_IP = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')

# Replacement patterns: replace_* or clean_*
PATTERN_CLEAN_WHITESPACE = re.compile(r'\s+')
PATTERN_CLEAN_HTML_TAGS = re.compile(r'<[^>]+>')
```


---

## Hands-on Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement appropriate error handling
- Create test code as well

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
        """Main processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
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
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistical information"""
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

    print(f"Inefficient: {slow_time:.4f}s")
    print(f"Efficient:   {fast_time:.6f}s")
    print(f"Speedup:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be mindful of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify configuration file path and format |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access permissions | Verify executing user's permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check error messages**: Read the stack trace and identify the location of occurrence
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify incrementally**: Use log output and debuggers to test hypotheses
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
        logger.debug(f"Called: {func.__name__}(args={args}, kwargs={kwargs})")
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

Diagnostic procedure when performance problems occur:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Examine disk and network I/O conditions
4. **Check concurrent connections**: Examine connection pool status

| Problem Type | Diagnostic Tools | Countermeasures |
|-------------|-----------------|-----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper release of references |
| I/O bottleneck | strace, iostat | Asynchronous I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes the criteria for making technology choices.

| Criterion | Prioritize When | Acceptable to Compromise When |
|-----------|----------------|------------------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal information, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
┌─────────────────────────────────────────────────┐
│          Architecture Selection Flow              │
├─────────────────────────────────────────────────┤
│                                                 │
│  (1) Team size?                                  │
│    ├─ Small (1-5 people) → Monolith              │
│    └─ Large (10+ people) → Go to (2)             │
│                                                 │
│  (2) Deployment frequency?                       │
│    ├─ Once a week or less → Monolith +           │
│    │  modular decomposition                      │
│    └─ Daily/multiple times → Go to (3)           │
│                                                 │
│  (3) Team independence?                          │
│    ├─ High → Microservices                       │
│    └─ Moderate → Modular monolith                │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Cost**
- A quick short-term approach can become technical debt in the long run
- Conversely, over-engineering has high short-term costs and can delay the project

**2. Consistency vs Flexibility**
- A unified technology stack has lower learning costs
- Adopting diverse technologies enables best-fit solutions but increases operational costs

**3. Level of Abstraction**
- High abstraction offers great reusability but can make debugging difficult
- Low abstraction is intuitive but prone to code duplication

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
        md += f"## Background\n{self.context}\n\n"
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

## 9. FAQ

### Q1: What is the best order for learning regular expressions?

**A**: The following order is recommended:

1. Literal matching → Metacharacters → Character classes → Quantifiers
2. Anchors → Groups → Backreferences
3. Lookahead/lookbehind → Unicode → Performance

It is most effective to master the basic syntax first, then practice with real use cases (log analysis, input validation, etc.).

### Q2: Should I choose NFA or DFA?

**A**: Use the following criteria:

- **Backreferences or lookahead required** → NFA (Perl, Python, JavaScript, etc.)
- **Matching against untrusted input** → DFA (RE2, Rust regex) to prevent ReDoS
- **Performance is the top priority** → Choose a DFA-based engine
- **Feature richness is the priority** → Choose an NFA-based engine

In most cases, using the default engine of your language is fine. Only consider DFA when ReDoS is a concern.

### Q3: What tools are available for debugging regular expressions?

**A**: Major tools:

- **regex101.com** -- Real-time pattern testing with explanations (supports PCRE, Python, JS, Go, Java)
- **Debuggex** -- Displays Railroad Diagrams of regular expressions
- **RegExr** -- Interactive regex tester
- **Verbose mode in each language** -- Python's `re.VERBOSE`, Perl's `/x` modifier
- **Debug flags in each language** -- Python's `re.DEBUG`

### Q4: What should I do when regex performance is poor?

**A**: Consider the following in order:

1. **Review the pattern**: Add anchors, eliminate unnecessary backtracking
2. **Pre-compile**: Reuse pattern objects with `re.compile()`
3. **Replace with string operations**: `str.startswith()`, `str.endswith()`, `in` operator
4. **Change the engine**: Migrate to RE2 or Rust regex
5. **Change the algorithm**: Consider solutions that don't use regular expressions

### Q5: How complex should a regular expression be allowed to get?

**A**: As a rule of thumb:

- **20 characters or fewer**: Can be used inline
- **20-50 characters**: Verbose mode with comments recommended
- **50-100 characters**: Split and build incrementally; testing is mandatory
- **100+ characters**: Consider using a parser or library

When readability degrades, it is time to split the regex or consider a different approach.

### Q6: How should I address security risks of regular expressions?

**A**: Major risks and countermeasures:

1. **ReDoS (Regular Expression Denial of Service)**
   - Avoid nested quantifiers: `(a+)+` → `a+`
   - Use a DFA engine for untrusted input
   - Set a timeout for matching

2. **Bypassing input validation**
   - Use `\A` and `\z` instead of just `^` and `$` (considering newline characters)
   - Prevent unintended activation of multiline mode
   - Perform Unicode normalization beforehand

3. **Injection attacks**
   - Do not embed user input directly into patterns
   - When embedding, always escape with `re.escape()`

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important. Understanding deepens not only through theory, but by actually writing code and verifying its behavior.

### Q2: What common mistakes do beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this used in professional practice?

Knowledge of this topic is frequently applied in daily development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Item | Description |
|------|-------------|
| Essence of regex | A formal language for pattern matching |
| Theoretical foundation | Kleene's regular sets (1956) |
| Origin of practical use | Thompson's implementation in QED/ed (1968) |
| Modern standard | PCRE (Perl Compatible Regular Expressions) |
| NFA | Backtracking-based, feature-rich, worst case O(2^n) |
| DFA | State transition-based, fast and stable, O(n) guaranteed |
| Primary uses | Text search, replacement, extraction, validation |
| Primary limitations | Cannot handle nested structures (HTML, etc.) by design |
| Major dialects | POSIX BRE/ERE, PCRE, ECMAScript, RE2 |
| Security | ReDoS countermeasures are essential (especially for NFA engines) |
| Best practices | Incremental construction, test-driven, verbose mode |
| Key to learning | Progress sequentially: basic syntax → advanced syntax → practical patterns |

---

## Recommended Next Guides

- [01-basic-syntax.md](./01-basic-syntax.md) -- Basic Syntax: Literals, Metacharacters, Escaping
- [02-character-classes.md](./02-character-classes.md) -- Character Classes in Detail
- [03-quantifiers-anchors.md](./03-quantifiers-anchors.md) -- Quantifiers and Anchors

---

## References

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions, 3rd Edition" O'Reilly Media, 2006 -- The definitive bible of regular expressions
2. **Russ Cox** "Regular Expression Matching Can Be Simple And Fast" https://swtch.com/~rsc/regexp/regexp1.html, 2007 -- Clear explanation of NFA/DFA theory and implementation
3. **Ken Thompson** "Regular Expression Search Algorithm" Communications of the ACM, 11(6):419-422, 1968 -- The original paper on regex engines
4. **Michael Rabin, Dana Scott** "Finite Automata and Their Decision Problems" IBM Journal of Research and Development, 3(2):114-125, 1959 -- The paper proving NFA/DFA equivalence
5. **PCRE2 Documentation** https://www.pcre.org/current/doc/html/ -- Current PCRE2 reference
6. **RE2 Documentation** https://github.com/google/re2/wiki/Syntax -- RE2 syntax reference
7. **ECMAScript Language Specification** https://tc39.es/ecma262/#sec-regexp-regular-expression-objects -- JavaScript regex specification
8. **POSIX Standard** IEEE Std 1003.1-2017, Section 9 "Regular Expressions" -- Official POSIX regex specification
