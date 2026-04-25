# Language-Specific Regular Expressions -- Differences Across JS/Python/Go/Rust/Java

> Even with the same regular expression, the syntax, flags, Unicode handling, and performance characteristics vary across languages and engines. This guide systematically compares the design philosophies and practical differences of each language's regex API.

## What You Will Learn

1. **Differences across the regex APIs of 5 languages** -- Variations in syntax, flags, return values, and matching models
2. **Understanding engine characteristics and constraints** -- Differences between NFA/DFA, backreference support, and Unicode handling
3. **Points to watch out for when porting between languages** -- Cases where the same pattern produces different results


## Prerequisites

You will get more out of this guide if you are familiar with the following:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. Per-Language Overview

### 1.1 Engine Mapping

```
┌──────────┬──────────────┬──────────────────────────┐
│ Language  │ Engine        │ Characteristics           │
├──────────┼──────────────┼──────────────────────────┤
│ Python   │ re (NFA)     │ PCRE-like, Unicode default│
│ JavaScript│ V8 (NFA)    │ ECMA-262 compliant, ES2018│
│ Java     │ Pattern (NFA)│ PCRE-like, possessive quant│
│ Go       │ RE2 (DFA)    │ Linear time guaranteed     │
│ Rust     │ regex (DFA)  │ Linear time guaranteed     │
└──────────┴──────────────┴──────────────────────────┘
```

### 1.2 Engine Design Philosophy in Detail

The design philosophy of each language's adopted engine has wide-ranging effects, from the API design to the available features.

**NFA (Non-deterministic Finite Automaton) Engine**

NFA engines work based on backtracking. They try each branch of the pattern in order, and if a match fails, they return to the previous choice point and try a different branch. This mechanism enables advanced features such as backreferences and lookarounds, but carries the risk of exponential time complexity in worst-case scenarios. Adopted by Python, JavaScript, and Java.

```
NFA backtracking behavior:

Input:   "aaaaab"
Pattern: a*ab

Step 1: a* greedily matches "aaaaa" → "b" and "ab" don't match
Step 2: Backtrack, a* matches "aaaa" → "ab" matches!
→ Match succeeds

Depending on the pattern:
Input:   "aaaaaaaaaaaaaaaaac"
Pattern: (a+)+b
→ Exponential backtracking occurs (ReDoS)
```

**DFA (Deterministic Finite Automaton) Engine**

DFA engines scan the input string only once, deterministically choosing the next state for each character. Since no backtracking occurs, matching always completes in O(n) linear time. However, features that require backtracking, such as backreferences and lookarounds, cannot be implemented. Adopted by Go and Rust.

```
DFA behavior:

Input:   "aaaaab"
Pattern: a*ab

Simulate all states simultaneously:
Position 0 'a': states {S0, S1}
Position 1 'a': states {S0, S1}
Position 2 'a': states {S0, S1}
Position 3 'a': states {S0, S1}
Position 4 'a': states {S0, S1, S2}
Position 5 'b': states {S3 (accept)}
→ Match succeeds (always O(n))
```

---

## 2. Python

### 2.1 Basic API

```python
import re

text = "2026-02-11 Error: Connection failed at 10:30:45"

# search: returns the first match
m = re.search(r'\d{4}-\d{2}-\d{2}', text)
print(m.group())  # => '2026-02-11'

# findall: returns all matches as a list
print(re.findall(r'\d+', text))
# => ['2026', '02', '11', '10', '30', '45']

# finditer: returns an iterator (memory efficient)
for m in re.finditer(r'\d+', text):
    print(f"  {m.group()} at {m.span()}")

# sub: substitution
result = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3/\2/\1', text)
print(result)  # => '11/02/2026 Error: ...'

# split: splitting
print(re.split(r'\s+', "hello  world\tfoo"))
# => ['hello', 'world', 'foo']

# compile: precompile
pattern = re.compile(r'\d+', re.ASCII)
print(pattern.findall(text))
```

### 2.2 Python-Specific Features

```python
import re

# fullmatch: whether the entire string matches the pattern
print(re.fullmatch(r'\d{4}', '2026'))   # => Match
print(re.fullmatch(r'\d{4}', '2026a'))  # => None

# (?P<name>...) named groups (Python-specific syntax)
m = re.search(r'(?P<year>\d{4})-(?P<month>\d{2})', "2026-02-11")
print(m.groupdict())  # => {'year': '2026', 'month': '02'}

# Conditional patterns (?(id)yes|no)
pattern = r'(\()?\d+(?(1)\))'
print(re.search(pattern, "(42)").group())  # => '(42)'
print(re.search(pattern, "42").group())    # => '42'

# Patterns with comments using re.VERBOSE
pattern = re.compile(r'''
    (?P<year>\d{4})   # year
    -(?P<month>\d{2}) # month
    -(?P<day>\d{2})   # day
''', re.VERBOSE)
```

### 2.3 Advanced Patterns and Practical Techniques in Python

```python
import re

# --- Function-based replacement with sub ---
# When you pass a function as the second argument, it's called with the match object
def celsius_to_fahrenheit(match):
    celsius = float(match.group(1))
    fahrenheit = celsius * 9 / 5 + 32
    return f"{fahrenheit:.1f}F"

text = "The temperature ranges from 20C to 35C"
result = re.sub(r'(\d+(?:\.\d+)?)C', celsius_to_fahrenheit, text)
print(result)  # => 'The temperature ranges from 68.0F to 95.0F'

# --- subn: also returns the number of replacements ---
result, count = re.subn(r'\d+', 'X', "abc 123 def 456")
print(f"result: {result}, replacements: {count}")
# => 'result: abc X def X, replacements: 2'

# --- maxsplit in split ---
print(re.split(r'[,;]', "a,b;c,d", maxsplit=2))
# => ['a', 'b', 'c,d']

# --- split includes separators in result when groups are used ---
print(re.split(r'([,;])', "a,b;c"))
# => ['a', ',', 'b', ';', 'c']

# --- re.escape: escape metacharacters ---
user_input = "price is $100 (USD)"
safe_pattern = re.escape(user_input)
print(safe_pattern)
# => 'price\\ is\\ \\$100\\ \\(USD\\)'
# Essential when incorporating user input into a pattern

# --- Combining multiple flags ---
pattern = re.compile(
    r'''
    (?P<protocol>https?)    # protocol
    ://
    (?P<host>[^/\s]+)       # hostname
    (?P<path>/[^\s]*)?      # path (optional)
    ''',
    re.VERBOSE | re.IGNORECASE
)
m = pattern.search("Visit HTTP://Example.COM/path?q=1 now")
if m:
    print(m.groupdict())
    # => {'protocol': 'HTTP', 'host': 'Example.COM', 'path': '/path?q=1'}
```

### 2.4 Python regex Module (Third-Party)

```python
# pip install regex
# A drop-in upgrade for the standard re, with many additional features
import regex

# Variable-length lookbehind
m = regex.search(r'(?<=ab+)', "abbb_test")
print(m.start())  # Not possible with the standard re

# Unicode category properties
print(regex.findall(r'\p{Han}+', "hello 世界 test"))
# => ['世界']

# Fuzzy matching
# {e<=1} allows edit distance of 1 or less
m = regex.search(r'(?:hello){e<=1}', "helo world")
print(m.group())  # => 'helo'

# Atomic group
m = regex.search(r'(?>a+)b', "aab")
print(m.group())  # => 'aab'

# Possessive quantifier
m = regex.search(r'a++b', "aab")
print(m.group())  # => 'aab'

# POSIX character classes
# => ['hello', 'world']

# Recursive patterns (handle nested parentheses)
pattern = regex.compile(r'\((?:[^()]*|(?R))*\)')
text = "outer (inner (deep) end) rest"
print(pattern.findall(text))
# => ['(inner (deep) end)']
```

### 2.5 Python Performance Optimization

```python
import re
import timeit

# --- Effect of precompiling ---
# When repeating the same pattern in a loop, compile is effective
# However, Python also caches internally (up to 512 patterns),
# so the difference is small in simple cases

compiled = re.compile(r'\d{4}-\d{2}-\d{2}')
text = "date: 2026-02-11"

# With compile:    compiled.search(text)
# Without compile: re.search(r'\d{4}-\d{2}-\d{2}', text)
# → compile becomes effective when there are many patterns (over 512) and cache overflows

# --- findall vs finditer ---
# When there are many matches, finditer is more memory-efficient
large_text = "num " * 100000
# findall:  stores all results in a list (high memory)
# finditer: returns an iterator one at a time (low memory)

# --- Use of non-capturing groups ---
# Use (?:...) for groups that don't need to be captured
# findall returns the contents of groups when groups exist
text = "2026-02-11 and 2025-12-25"
print(re.findall(r'(\d{4})-(\d{2})-(\d{2})', text))
# => [('2026', '02', '11'), ('2025', '12', '25')]  ← list of tuples
print(re.findall(r'\d{4}-\d{2}-\d{2}', text))
# => ['2026-02-11', '2025-12-25']  ← list of strings
print(re.findall(r'(?:\d{4})-(?:\d{2})-(?:\d{2})', text))
# => ['2026-02-11', '2025-12-25']  ← list of strings (same)

# --- Choosing greedy vs lazy ---
html = "<b>bold</b> and <i>italic</i>"
print(re.findall(r'<.+>', html))    # => ['<b>bold</b> and <i>italic</i>'] greedy
print(re.findall(r'<.+?>', html))   # => ['<b>', '</b>', '<i>', '</i>'] lazy
print(re.findall(r'<[^>]+>', html)) # => ['<b>', '</b>', '<i>', '</i>'] negated class (fastest)
```

---

## 3. JavaScript

### 3.1 Basic API

```javascript
const text = "2026-02-11 Error: Connection failed at 10:30:45";

// Literal syntax
const pattern = /\d{4}-\d{2}-\d{2}/;
const match = text.match(pattern);
console.log(match[0]);  // => '2026-02-11'

// g flag: all matches
console.log(text.match(/\d+/g));
// => ['2026', '02', '11', '10', '30', '45']

// matchAll (ES2020): returns an iterator
for (const m of text.matchAll(/\d+/g)) {
    console.log(`  ${m[0]} at index ${m.index}`);
}

// replace: substitution
const result = text.replace(
    /(\d{4})-(\d{2})-(\d{2})/,
    '$3/$2/$1'
);
console.log(result);  // => '11/02/2026 Error: ...'

// replaceAll (ES2021): replace all
console.log("aaa".replaceAll(/a/g, "b"));  // => 'bbb'

// Constructor syntax (dynamic patterns)
const dynamic = new RegExp("\\d{4}", "g");
console.log(text.match(dynamic));
```

### 3.2 JavaScript-Specific Features (ES2018+)

```javascript
// Named groups (ES2018)
const m = "2026-02-11".match(
    /(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})/
);
console.log(m.groups);
// => { year: '2026', month: '02', day: '11' }

// Lookbehind (ES2018)
console.log("$100 €200".match(/(?<=\$)\d+/g));
// => ['100']

// s flag: dotAll (ES2018)
console.log("a\nb".match(/a.b/s));
// => ['a\nb']

// d flag: index info (ES2022)
const result2 = /(?<name>\w+)/.exec("hello");
// result.indices[0] => [0, 5]
// result.indices.groups.name => [0, 5]

// v flag: Unicode set operations (ES2024)
// /[\p{L}&&\p{ASCII}]/v  -- ASCII and letter
```

### 3.3 Advanced Patterns and Practical Techniques in JavaScript

```javascript
// --- Function-based replacement with replace ---
const text2 = "Price: $100, Tax: $15, Total: $115";
const formatted = text2.replace(/\$(\d+)/g, (match, amount) => {
    return `$${Number(amount).toLocaleString()}`;
});
console.log(formatted);
// => 'Price: $100, Tax: $15, Total: $115'

// Function-based replacement using named groups
const dates = "Start: 2026-02-11, End: 2026-03-15";
const converted = dates.replace(
    /(?<y>\d{4})-(?<m>\d{2})-(?<d>\d{2})/g,
    (match, y, m, d, offset, string, groups) => {
        return `${groups.d}/${groups.m}/${groups.y}`;
    }
);
console.log(converted);
// => 'Start: 11/02/2026, End: 15/03/2026'

// --- Sequential matching with exec() ---
const re = /(\w+)=(\w+)/g;
const params = "name=Alice&age=30&city=Tokyo";
let execMatch;
while ((execMatch = re.exec(params)) !== null) {
    console.log(`${execMatch[1]}: ${execMatch[2]}`);
}
// name: Alice
// age: 30
// city: Tokyo

// --- Pitfalls of g flag and lastIndex ---
const reG = /abc/g;
console.log(reG.test("abc def"));  // true
console.log(reG.lastIndex);        // 3
console.log(reG.test("abc def"));  // false! (search starts from lastIndex=3)
reG.lastIndex = 0;                 // reset is required
console.log(reG.test("abc def"));  // true

// --- String.prototype.search() ---
// Returns the index of the match (g flag is ignored)
console.log("hello world".search(/world/));  // => 6
console.log("hello world".search(/xyz/));    // => -1

// --- split limits and caveats ---
console.log("a1b2c3d".split(/(\d)/));
// => ['a', '1', 'b', '2', 'c', '3', 'd']
// With capturing groups, separators are also included in the result

console.log("a,,b,,c".split(/,+/));
// => ['a', 'b', 'c']
```

### 3.4 JavaScript v Flag (ES2024) Details

```javascript
// The v flag is an extended version of the u flag, enabling set operations on character classes

// Difference: \p{L} excluding ASCII → only non-ASCII letters
// /[\p{L}--\p{ASCII}]/v
const nonAsciiLetters = "hello 世界 café".match(/[\p{L}--\p{ASCII}]/gv);
console.log(nonAsciiLetters);
// => ['世', '界', 'é']

// Intersection: both \p{L} and \p{ASCII} → ASCII letters only
// /[\p{L}&&\p{ASCII}]/v
const asciiLetters = "hello 世界 café".match(/[\p{L}&&\p{ASCII}]/gv);
console.log(asciiLetters);
// => ['h', 'e', 'l', 'l', 'o', 'c', 'a', 'f']

// Union (nested character classes)
// /[[\p{Decimal_Number}][\p{L}]]/v
// → matches digits or letters

// Notes when using v flag:
// - The u and v flags cannot be used simultaneously
// - The v flag includes all features of the u flag
// - Special character handling within character classes becomes stricter
```

### 3.5 JavaScript Regex Performance

```javascript
// --- Internal cache of RegExp ---
// The V8 engine caches regexes in literal syntax
// new RegExp() creates a new object each time

// Fast: literal (the same pattern is internally cached)
function matchLiteral(text) {
    return /\d+/g.test(text);
}

// Caution: dynamic patterns cannot be cached
function matchDynamic(text, pattern) {
    return new RegExp(pattern, 'g').test(text);
}

// --- ReDoS countermeasures ---
// V8 has a RegExp timeout mechanism (--regex-timeout in Node.js)
// However, fundamentally it should be prevented through pattern design

// Dangerous patterns:
// /(a+)+b/          → exponential backtracking
// /([a-zA-Z]+)*$/   → exponential backtracking
// /(\w+\s?)+$/      → exponential backtracking

// Rewriting to safe patterns:
// /(a+)+b/  →  /a+b/
// /([a-zA-Z]+)*$/  →  /[a-zA-Z]*$/
// /(\w+\s?)+$/  →  /[\w\s]*$/

// --- Advantages of matchAll ---
// match(g) loses capture group information
const text3 = "2026-02-11 2025-12-25";
console.log(text3.match(/(\d{4})-(\d{2})-(\d{2})/g));
// => ['2026-02-11', '2025-12-25']  ← no group info!

// matchAll preserves group information
for (const m of text3.matchAll(/(\d{4})-(\d{2})-(\d{2})/g)) {
    console.log(`${m[0]} → year: ${m[1]}, month: ${m[2]}, day: ${m[3]}`);
}
// '2026-02-11 → year: 2026, month: 02, day: 11'
// '2025-12-25 → year: 2025, month: 12, day: 25'
```

---

## 4. Go

### 4.1 Basic API

```go
package main

import (
    "fmt"
    "regexp"
)

func main() {
    text := "2026-02-11 Error: Connection failed"

    // Compile: compile the pattern (returns an error)
    re, err := regexp.Compile(`\d{4}-\d{2}-\d{2}`)
    if err != nil {
        panic(err)
    }

    // MustCompile: panicking version (for constant patterns)
    re = regexp.MustCompile(`\d{4}-\d{2}-\d{2}`)

    // FindString: first match
    fmt.Println(re.FindString(text))
    // => "2026-02-11"

    // FindAllString: all matches
    reDigit := regexp.MustCompile(`\d+`)
    fmt.Println(reDigit.FindAllString(text, -1))
    // => [2026 02 11]

    // FindStringSubmatch: with submatches
    re2 := regexp.MustCompile(`(\d{4})-(\d{2})-(\d{2})`)
    matches := re2.FindStringSubmatch(text)
    fmt.Printf("whole: %s, year: %s, month: %s, day: %s\n",
        matches[0], matches[1], matches[2], matches[3])

    // ReplaceAllString: substitution
    result := re2.ReplaceAllString(text, "${3}/${2}/${1}")
    fmt.Println(result)

    // Named groups
    re3 := regexp.MustCompile(`(?P<year>\d{4})-(?P<month>\d{2})`)
    match := re3.FindStringSubmatch(text)
    for i, name := range re3.SubexpNames() {
        if name != "" {
            fmt.Printf("  %s: %s\n", name, match[i])
        }
    }
}
```

### 4.2 Go Constraints

```go
// Features not supported in Go (RE2 engine):
// ✗ Backreferences (\1, \k<name>)
// ✗ Lookahead (?=...), (?!...)
// ✗ Lookbehind (?<=...), (?<!...)
// ✗ Conditional patterns (?(id)yes|no)
// ✗ Possessive quantifiers (*+, ++, ?+)
// ✗ Atomic groups (?>...)
// ✗ Some Unicode properties

// In return, the following is guaranteed:
// ✓ Always O(n) linear time
// ✓ ReDoS is fundamentally impossible
// ✓ Memory usage is predictable
```

### 4.3 Advanced APIs and Practical Techniques in Go

```go
package main

import (
    "fmt"
    "regexp"
    "strings"
)

func main() {
    // --- ReplaceAllStringFunc: replace using a function ---
    re := regexp.MustCompile(`[a-z]+`)
    result := re.ReplaceAllStringFunc("hello WORLD foo BAR", strings.ToUpper)
    fmt.Println(result)
    // => "HELLO WORLD FOO BAR"

    // --- ReplaceAllLiteralString: literal replacement ---
    // No expansion of $1 etc.
    re2 := regexp.MustCompile(`\d+`)
    result2 := re2.ReplaceAllLiteralString("price: 100", "$1")
    fmt.Println(result2)
    // => "price: $1" ($1 is inserted as literal text)

    // --- Split: split ---
    re3 := regexp.MustCompile(`\s*[,;]\s*`)
    parts := re3.Split("a, b; c , d", -1)
    fmt.Println(parts)
    // => [a b c d]

    // The n argument limits the number of splits
    parts2 := re3.Split("a, b; c , d", 2)
    fmt.Println(parts2)
    // => [a b; c , d]

    // --- FindAllStringSubmatchIndex: all matches with position info ---
    re4 := regexp.MustCompile(`(\w+)=(\w+)`)
    text := "name=Alice age=30"
    indices := re4.FindAllStringSubmatchIndex(text, -1)
    for _, idx := range indices {
        // idx[0:2] = start/end of overall match
        // idx[2:4] = start/end of group 1
        // idx[4:6] = start/end of group 2
        key := text[idx[2]:idx[3]]
        val := text[idx[4]:idx[5]]
        fmt.Printf("  %s = %s\n", key, val)
    }

    // --- MatchString: just check whether there is a match ---
    // Faster than Find* (no position calculation needed)
    re5 := regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
    fmt.Println(re5.MatchString("2026-02-11"))  // true
    fmt.Println(re5.MatchString("not a date"))  // false

    // --- []byte version of API ---
    // Process byte sequences directly without string conversion
    reB := regexp.MustCompile(`\d+`)
    data := []byte("hello 123 world 456")
    allBytes := reB.FindAll(data, -1)
    for _, b := range allBytes {
        fmt.Printf("  %s\n", b)
    }

    // --- Expand: template expansion ---
    re6 := regexp.MustCompile(`(?P<first>\w+)\s+(?P<last>\w+)`)
    template := []byte("$last, $first")
    src := []byte("John Smith")
    match := re6.FindSubmatchIndex(src)
    var dst []byte
    dst = re6.Expand(dst, template, src, match)
    fmt.Printf("%s\n", dst)
    // => "Smith, John"
}
```

### 4.4 Alternatives to Lookahead/Lookbehind in Go

```go
package main

import (
    "fmt"
    "regexp"
    "strings"
)

func main() {
    // Since Go does not support lookahead/lookbehind, alternative approaches are required

    // --- Alternative 1: Use capture groups to extract the needed part ---
    // Python: (?<=\$)\d+  (digits after $)
    // Go: use \$(\d+) and refer to group 1
    re := regexp.MustCompile(`\$(\d+)`)
    text := "$100 and $200"
    matches := re.FindAllStringSubmatch(text, -1)
    for _, m := range matches {
        fmt.Println(m[1])  // "100", "200"
    }

    // --- Alternative 2: Process in multiple steps ---
    // Python: (?<=<tag>).*?(?=</tag>)
    // Go: use <tag>(.*?)</tag> and refer to group 1
    re2 := regexp.MustCompile(`<title>(.*?)</title>`)
    html := "<title>My Page</title>"
    if m := re2.FindStringSubmatch(html); m != nil {
        fmt.Println(m[1])  // "My Page"
    }

    // --- Alternative 3: Alternative to negative lookahead ---
    // Python: \b\w+(?!ing)\b  (words not ending in 'ing')
    // Go: filter after matching
    re3 := regexp.MustCompile(`\b\w+\b`)
    text2 := "running jumping hello world coding"
    words := re3.FindAllString(text2, -1)
    for _, w := range words {
        if !strings.HasSuffix(w, "ing") {
            fmt.Printf("  %s\n", w)
        }
    }
    // => hello, world

    // --- Alternative 4: Alternative to backreferences ---
    // Python: <(\w+)>.*?</\1>  (matching open/close tags)
    // Go: process in two passes
    re4 := regexp.MustCompile(`<(\w+)>[^<]*</(\w+)>`)
    html2 := "<div>content</div><span>text</span><div>bad</span>"
    allMatches := re4.FindAllStringSubmatch(html2, -1)
    for _, m := range allMatches {
        if m[1] == m[2] {  // open tag matches close tag
            fmt.Printf("  valid: %s\n", m[0])
        } else {
            fmt.Printf("  invalid: %s\n", m[0])
        }
    }
}
```

### 4.5 Go Performance Optimization

```go
package main

import (
    "regexp"
    "sync"
)

// --- Compile at package level ---
// MustCompile runs only once at program startup
var (
    datePattern  = regexp.MustCompile(`\d{4}-\d{2}-\d{2}`)
    emailPattern = regexp.MustCompile(`[\w.+-]+@[\w-]+\.[\w.]+`)
    urlPattern   = regexp.MustCompile(`https?://[^\s]+`)
)

// --- Reuse Regexp objects with sync.Pool ---
// (usually unnecessary; only when there are many dynamic patterns)
var regexpPool = sync.Pool{
    New: func() interface{} {
        return regexp.MustCompile(`\d+`)
    },
}

func processWithPool(text string) []string {
    re := regexpPool.Get().(*regexp.Regexp)
    defer regexpPool.Put(re)
    return re.FindAllString(text, -1)
}

// --- Regexp is goroutine-safe ---
// The same Regexp object can be safely used from multiple goroutines
// (because it uses internal locks)
func processParallel(texts []string) {
    var wg sync.WaitGroup
    for _, t := range texts {
        wg.Add(1)
        go func(text string) {
            defer wg.Done()
            datePattern.FindString(text)  // safe
        }(t)
    }
    wg.Wait()
}

// --- Avoid lock contention with Copy() ---
// In high-load parallel processing, making a copy with Copy() can be faster
func processHighConcurrency(texts []string) {
    var wg sync.WaitGroup
    for _, t := range texts {
        wg.Add(1)
        go func(text string) {
            defer wg.Done()
            re := datePattern.Copy()  // avoid contention with a copy
            re.FindString(text)
        }(t)
    }
    wg.Wait()
}
```

---

## 5. Rust

### 5.1 Basic API

```rust
use regex::Regex;

fn main() {
    let text = "2026-02-11 Error: Connection failed";

    // Compile
    let re = Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap();

    // find: first match
    if let Some(m) = re.find(text) {
        println!("match: {} (position: {}-{})", m.as_str(), m.start(), m.end());
    }

    // find_iter: iterator over all matches
    let re_digit = Regex::new(r"\d+").unwrap();
    let numbers: Vec<&str> = re_digit.find_iter(text)
        .map(|m| m.as_str())
        .collect();
    println!("{:?}", numbers);
    // => ["2026", "02", "11"]

    // captures: capture groups
    let re2 = Regex::new(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})").unwrap();
    if let Some(caps) = re2.captures(text) {
        println!("year: {}, month: {}, day: {}",
            &caps["year"], &caps["month"], &caps["day"]);
    }

    // replace: substitution
    let result = re2.replace(text, "$day/$month/$year");
    println!("{}", result);
}
```

### 5.2 Rust Characteristics

```rust
// Characteristics of Rust regex:
// ✓ DFA-based: O(n) guaranteed
// ✓ ReDoS impossible
// ✓ Zero-cost lazy compilation (lazy_static!, once_cell)
// ✓ Unicode supported by default
// ✗ No backreferences
// ✗ No lookarounds

// When you need lookarounds: fancy-regex
// use fancy_regex::Regex;
// let re = Regex::new(r"(?<=\$)\d+").unwrap();
// → Falls back to NFA (no O(n) guarantee)

// Performance optimization: RegexSet (matching multiple patterns simultaneously)
use regex::RegexSet;

let set = RegexSet::new(&[
    r"ERROR",
    r"WARN",
    r"INFO",
]).unwrap();

let text = "2026-02-11 ERROR: Something failed";
let matches: Vec<_> = set.matches(text).into_iter().collect();
println!("{:?}", matches);  // => [0] (matches ERROR)
```

### 5.3 Advanced APIs and Practical Techniques in Rust

```rust
use regex::Regex;
use std::borrow::Cow;

fn main() {
    // --- replace_all: replace all ---
    let re = Regex::new(r"\d+").unwrap();
    let result = re.replace_all("abc 123 def 456", "NUM");
    println!("{}", result);
    // => "abc NUM def NUM"

    // replace returns Cow<str>
    // If there is no match, returns a reference to the original string (zero-copy)
    let no_match: Cow<str> = re.replace("no numbers here", "NUM");
    match no_match {
        Cow::Borrowed(_) => println!("no copy"),                 // this branch
        Cow::Owned(_) => println!("new string generated"),
    }

    // --- replace_all with closure ---
    let re2 = Regex::new(r"(?P<word>[a-z]+)").unwrap();
    let result2 = re2.replace_all("hello world", |caps: &regex::Captures| {
        caps["word"].to_uppercase()
    });
    println!("{}", result2);
    // => "HELLO WORLD"

    // --- captures_iter: iterator over all captures ---
    let re3 = Regex::new(r"(?P<key>\w+)=(?P<val>\w+)").unwrap();
    let text = "name=Alice age=30 city=Tokyo";
    for caps in re3.captures_iter(text) {
        println!("  {} = {}", &caps["key"], &caps["val"]);
    }

    // --- split: split ---
    let re4 = Regex::new(r"[,;\s]+").unwrap();
    let parts: Vec<&str> = re4.split("a, b; c d").collect();
    println!("{:?}", parts);
    // => ["a", "b", "c", "d"]

    // --- splitn: limit number of splits ---
    let parts2: Vec<&str> = re4.splitn("a, b; c d", 2).collect();
    println!("{:?}", parts2);
    // => ["a", "b; c d"]

    // --- shortest_match: only the end position of the shortest match ---
    // Faster than find (no need to compute the start position)
    let re5 = Regex::new(r"\d+").unwrap();
    if let Some(end) = re5.shortest_match("abc 123") {
        println!("shortest match end position: {}", end);
    }

    // --- is_match: only whether there is a match ---
    // Faster than find (no need to compute position info)
    println!("{}", re5.is_match("abc 123"));  // true
    println!("{}", re5.is_match("no nums"));  // false
}
```

### 5.4 Lazy Compilation and Performance in Rust

```rust
// --- lazy_static! / once_cell / std::sync::LazyLock ---
// Compiling regexes is expensive. It should be done once globally

// Option 1: once_cell (recommended; expected to be included in std)
use once_cell::sync::Lazy;
use regex::Regex;

static DATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap()
});

// Option 2: std::sync::LazyLock (Rust 1.80+)
// use std::sync::LazyLock;
// static DATE_RE: LazyLock<Regex> = LazyLock::new(|| {
//     Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap()
// });

// Option 3: lazy_static! macro
// use lazy_static::lazy_static;
// lazy_static! {
//     static ref DATE_RE: Regex = Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap();
// }

fn process(text: &str) -> Option<&str> {
    DATE_RE.find(text).map(|m| m.as_str())
}

// --- Practical example of RegexSet: log level classification ---
use regex::RegexSet;

fn classify_log_lines(lines: &[&str]) {
    let set = RegexSet::new(&[
        r"(?i)\berror\b",
        r"(?i)\bwarn(ing)?\b",
        r"(?i)\binfo\b",
        r"(?i)\bdebug\b",
    ]).unwrap();

    let labels = ["ERROR", "WARN", "INFO", "DEBUG"];

    for line in lines {
        let matches: Vec<_> = set.matches(line).into_iter().collect();
        if matches.is_empty() {
            println!("  UNKNOWN: {}", line);
        } else {
            for idx in matches {
                println!("  {}: {}", labels[idx], line);
            }
        }
    }
}

// --- bytes::Regex: directly process byte sequences ---
// When processing non-UTF-8 data (such as binary logs)
use regex::bytes::Regex as BytesRegex;

fn search_binary_log(data: &[u8]) {
    let re = BytesRegex::new(r"ERROR: (.+)").unwrap();
    for caps in re.captures_iter(data) {
        if let Some(msg) = caps.get(1) {
            println!("  error: {:?}", msg.as_bytes());
        }
    }
}

// --- regex-automata: low-level API ---
// Directly manipulate the internal engine of the regex crate
// DFA state tables can be serialized/deserialized
// Useful for embedded systems where you want to reduce compile time
```

### 5.5 Rust fancy-regex Details

```rust
// fancy-regex supports lookarounds and backreferences
// A hybrid engine combining the regex crate's DFA and an NFA
// Parts that can be processed with DFA are O(n); only lookaround parts fall back to NFA

use fancy_regex::Regex;

fn main() {
    // Positive lookahead
    let re = Regex::new(r"\w+(?=\s*=)").unwrap();
    let text = "name = Alice";
    if let Ok(Some(m)) = re.find(text) {
        println!("{}", m.as_str());  // "name"
    }

    // Negative lookahead
    let re2 = Regex::new(r"\b\w+\b(?!\s*=)").unwrap();
    // Matches words not followed by "="

    // Positive lookbehind
    let re3 = Regex::new(r"(?<=\$)\d+").unwrap();
    let text2 = "price: $100";
    if let Ok(Some(m)) = re3.find(text2) {
        println!("{}", m.as_str());  // "100"
    }

    // Backreference
    let re4 = Regex::new(r"<(\w+)>[^<]*</\1>").unwrap();
    let html = "<div>content</div>";
    if let Ok(Some(m)) = re4.find(html) {
        println!("{}", m.as_str());  // "<div>content</div>"
    }

    // Note: fancy_regex::Regex and regex::Regex are different types
    // The APIs are similar but differ in that they return Result
    // Error handling is required:
    // regex:       re.find(text)        → Option<Match>
    // fancy_regex: re.find(text)        → Result<Option<Match>>
}
```

---

## 6. Java

### 6.1 Basic API

```java
import java.util.regex.*;

public class RegexExample {
    public static void main(String[] args) {
        String text = "2026-02-11 Error: Connection failed";

        // Pattern + Matcher
        Pattern pattern = Pattern.compile("(\\d{4})-(\\d{2})-(\\d{2})");
        Matcher matcher = pattern.matcher(text);

        if (matcher.find()) {
            System.out.println("whole: " + matcher.group(0));
            System.out.println("year: " + matcher.group(1));
            System.out.println("month: " + matcher.group(2));
            System.out.println("day: " + matcher.group(3));
        }

        // All matches
        Pattern digits = Pattern.compile("\\d+");
        Matcher dm = digits.matcher(text);
        while (dm.find()) {
            System.out.println("  " + dm.group());
        }

        // Substitution
        String result = pattern.matcher(text)
            .replaceAll("$3/$2/$1");
        System.out.println(result);

        // Named groups
        Pattern named = Pattern.compile(
            "(?<year>\\d{4})-(?<month>\\d{2})-(?<day>\\d{2})"
        );
        Matcher nm = named.matcher(text);
        if (nm.find()) {
            System.out.println("year: " + nm.group("year"));
        }
    }
}
```

### 6.2 Java-Specific Features

```java
// Possessive Quantifiers
// Forbid backtracking for performance
Pattern.compile("a++b");    // possessively consume a's
Pattern.compile("[^\"]*+"); // possessively consume non-" chars

// Atomic groups (not available in Java → use possessive quantifiers as alternative)
// (?>pattern) was added in Java 20+

// String.matches(): whether the entire string matches
boolean valid = "2026-02-11".matches("\\d{4}-\\d{2}-\\d{2}");

// Pattern.UNICODE_CHARACTER_CLASS (Java 7+)
// Extends \w to Unicode characters
Pattern unicode = Pattern.compile("\\w+",
    Pattern.UNICODE_CHARACTER_CLASS);
```

### 6.3 Advanced APIs and Practical Techniques in Java

```java
import java.util.regex.*;
import java.util.stream.*;

public class AdvancedRegex {
    public static void main(String[] args) {
        // --- Matcher.appendReplacement / appendTail ---
        // Functional replacement (the pre-Java 8 approach)
        Pattern p = Pattern.compile("\\d+");
        Matcher m = p.matcher("abc 123 def 456");
        StringBuffer sb = new StringBuffer();
        while (m.find()) {
            int num = Integer.parseInt(m.group());
            m.appendReplacement(sb, String.valueOf(num * 2));
        }
        m.appendTail(sb);
        System.out.println(sb.toString());
        // => "abc 246 def 912"

        // --- Matcher.replaceAll with Function (Java 9+) ---
        String result = Pattern.compile("\\d+")
            .matcher("abc 123 def 456")
            .replaceAll(mr -> String.valueOf(Integer.parseInt(mr.group()) * 2));
        System.out.println(result);
        // => "abc 246 def 912"

        // --- Pattern.splitAsStream (Java 8+) ---
        Pattern sep = Pattern.compile("[,;\\s]+");
        long count = sep.splitAsStream("a, b; c d e")
            .filter(s -> !s.isEmpty())
            .count();
        System.out.println("element count: " + count);  // => 5

        // --- Pattern.asPredicate (Java 8+) ---
        // Convenient for filtering streams
        Pattern emailP = Pattern.compile("\\w+@\\w+\\.\\w+");
        java.util.List<String> items = java.util.List.of(
            "alice@example.com", "not-email", "bob@test.org"
        );
        items.stream()
            .filter(emailP.asPredicate())
            .forEach(System.out::println);
        // => alice@example.com
        //    bob@test.org

        // --- Pattern.asMatchPredicate (Java 11+) ---
        // Equivalent to matches() (whether the entire string matches)
        Pattern dateP = Pattern.compile("\\d{4}-\\d{2}-\\d{2}");
        System.out.println(dateP.asMatchPredicate().test("2026-02-11"));
        // => true
        System.out.println(dateP.asMatchPredicate().test("date: 2026-02-11"));
        // => false (not a full match)

        // --- region: limit the matching range ---
        String text = "hello world foo bar";
        Matcher rm = Pattern.compile("\\w+").matcher(text);
        rm.region(6, 11);  // only the range of "world"
        if (rm.find()) {
            System.out.println(rm.group());  // => "world"
        }

        // --- lookingAt: partial match from the beginning ---
        // (Equivalent to Python's re.match)
        Matcher lm = Pattern.compile("\\d+").matcher("123abc");
        System.out.println(lm.lookingAt());  // true
        System.out.println(lm.group());       // "123"

        // --- hitEnd / requireEnd ---
        // Useful when implementing parsers
        Matcher he = Pattern.compile("abc").matcher("ab");
        he.find();
        System.out.println(he.hitEnd());
        // => true (reached the end of input = a longer input might match)
    }
}
```

### 6.4 Java Possessive Quantifiers and Atomic Groups in Detail

```java
import java.util.regex.*;

public class PossessiveQuantifiers {
    public static void main(String[] args) {
        // --- Greedy vs Lazy vs Possessive ---
        //
        // Greedy:    .*  → match as much as possible, backtrack if needed
        // Lazy:      .*? → match as little as possible, expand if needed
        // Possessive: .*+ → match as much as possible, no backtracking
        String text = "\"hello\" and \"world\"";

        // Greedy: with backtracking
        System.out.println(
            Pattern.compile("\".*\"").matcher(text).results()
                .map(MatchResult::group).collect(java.util.stream.Collectors.toList())
        );
        // => ["hello" and "world"] (maximum range)

        // Lazy: minimum range
        System.out.println(
            Pattern.compile("\".*?\"").matcher(text).results()
                .map(MatchResult::group).collect(java.util.stream.Collectors.toList())
        );
        // => ["hello", "world"]

        // Possessive: no backtracking
        // ".*+" consumes everything and never gives back → cannot match the final "
        System.out.println(
            Pattern.compile("\".*+\"").matcher(text).find()
        );
        // => false (no match!)

        // Correct usage of possessive quantifiers:
        // Combine with negated character class
        System.out.println(
            Pattern.compile("\"[^\"]*+\"").matcher(text).results()
                .map(MatchResult::group).collect(java.util.stream.Collectors.toList())
        );
        // => ["hello", "world"] (fast and accurate)

        // --- Performance differences ---
        // Pattern vulnerable to ReDoS:
        // Pattern.compile("(a+)+b");  → O(2^n)
        //
        // Defense with possessive quantifiers:
        // Pattern.compile("(a++)+b"); → no backtracking, fails fast

        // --- Atomic groups (Java 20+) ---
        // (?>pattern) forbids backtracking for the entire group
        // Pattern.compile("(?>a+)b");
        // Equivalent to possessive a++, but usable for more complex patterns
        // Pattern.compile("(?>abc|ab)c");
        // → After matching "abc", does not backtrack to try "ab"
    }
}
```

### 6.5 Java Performance Optimization

```java
import java.util.regex.*;

public class RegexPerformance {
    // --- Pattern is thread-safe but Matcher is not ---
    // Pattern is compiled and immutable → can be shared with static final
    // Matcher has internal state → create one per thread
    private static final Pattern DATE_PATTERN =
        Pattern.compile("(\\d{4})-(\\d{2})-(\\d{2})");

    public static String findDate(String text) {
        Matcher m = DATE_PATTERN.matcher(text);  // Matcher created each time
        return m.find() ? m.group() : null;
    }

    // --- Pitfalls of String.matches() ---
    // String.matches() calls Pattern.compile() each time
    public static boolean isDateBad(String text) {
        return text.matches("\\d{4}-\\d{2}-\\d{2}");
        // ↑ compiles every time! Very slow when used in a loop
    }

    // Pre-compiled version
    public static boolean isDateGood(String text) {
        return DATE_PATTERN.matcher(text).matches();
        // ↑ reuses the compiled pattern
    }

    // --- Reuse object via Matcher.reset() ---
    // Effective when processing many strings
    public static void processMany(String[] texts) {
        Pattern p = Pattern.compile("\\d+");
        Matcher m = p.matcher("");  // initialize with empty string
        for (String text : texts) {
            m.reset(text);  // reuse Matcher
            while (m.find()) {
                System.out.println(m.group());
            }
        }
    }
}
```

---

## 7. ASCII Diagrams

### 7.1 Differences in Per-Language Match Models

```
Python re.search()  →  match anywhere in the string
Python re.match()   →  match from the beginning of the string
Python re.fullmatch()→  match the entire string

JavaScript .match()  →  match anywhere (only first match without g)
JavaScript .match(g) →  return all matches as an array
JavaScript .test()   →  boolean

Go FindString()     →  match anywhere in the string
Go MatchString()    →  whether there is a match anywhere (boolean)

Java find()         →  match anywhere in the string
Java matches()      →  match the entire string

Rust find()         →  match anywhere in the string
Rust is_match()     →  boolean

┌──────────┬──────────┬──────────┬──────────┐
│ Behavior  │ Partial   │ Beginning │ Full      │
├──────────┼──────────┼──────────┼──────────┤
│ Python   │ search() │ match()  │fullmatch()│
│ JavaScript│ .match()│ /^.../   │ /^...$/ │
│ Go       │ Find()   │ ―       │ Match()※│
│ Java     │ find()   │ ―       │matches()│
│ Rust     │ find()   │ ―       │ ―       │
└──────────┴──────────┴──────────┴──────────┘
※ Go's Match() is a partial match
```

### 7.2 Comparison of Flag Syntax

```
Different syntax for the same effect:

Case-insensitive:
  Python:     re.IGNORECASE / re.I / (?i)
  JavaScript: /pattern/i
  Go:         (?i)pattern
  Java:       Pattern.CASE_INSENSITIVE / (?i)
  Rust:       (?i)pattern

Multiline:
  Python:     re.MULTILINE / re.M / (?m)
  JavaScript: /pattern/m
  Go:         (?m)pattern
  Java:       Pattern.MULTILINE / (?m)
  Rust:       (?m)pattern

Dot-all:
  Python:     re.DOTALL / re.S / (?s)
  JavaScript: /pattern/s
  Go:         (?s)pattern
  Java:       Pattern.DOTALL / (?s)
  Rust:       (?s)pattern
```

### 7.3 Differences in Escaping

```
Backslash escaping:

Python:
  Regular string: "\\d+"  → \d+
  raw string:     r"\d+"  → \d+  (recommended)

JavaScript:
  Literal:       /\d+/          (no escaping needed)
  Constructor:   "\\d+"          (double escape)

Go:
  Backquotes:    `\d+`           (raw string, recommended)
  Regular string:"\\d+"          (double escape)

Java:
  Regular string:"\\d+"          (double escape, the only option)
  Text block:    \"""
    \d+                          (Java 15+ raw-string-like)
  \"""

Rust:
  raw string:    r"\d+"          (no escaping needed)
  Regular string:"\\d+"          (double escape)
```

### 7.4 API Call Flow Diagram

```
Typical regex processing flow per language:

Python:
  pattern string → re.compile() → Pattern object
                                    ├─ .search(text) → Match | None
                                    ├─ .findall(text) → [str, ...]
                                    ├─ .finditer(text) → Iterator[Match]
                                    ├─ .sub(repl, text) → str
                                    └─ .split(text) → [str, ...]

JavaScript:
  /pattern/flags → RegExp object
                    ├─ .test(str) → boolean
                    ├─ .exec(str) → Array | null
                    │
  str.match(re)  → Array | null (without g) / [str, ...] (with g)
  str.matchAll(re) → Iterator
  str.replace(re, repl) → string
  str.split(re) → [string, ...]

Go:
  pattern string → regexp.Compile() → (*Regexp, error)
                   regexp.MustCompile() → *Regexp
                    ├─ .FindString(s) → string
                    ├─ .FindAllString(s, n) → []string
                    ├─ .FindStringSubmatch(s) → []string
                    ├─ .ReplaceAllString(s, repl) → string
                    ├─ .MatchString(s) → bool
                    └─ .Split(s, n) → []string

Rust:
  pattern string → Regex::new() → Result<Regex>
                    ├─ .find(text) → Option<Match>
                    ├─ .find_iter(text) → Iterator<Match>
                    ├─ .captures(text) → Option<Captures>
                    ├─ .captures_iter(text) → Iterator<Captures>
                    ├─ .replace(text, rep) → Cow<str>
                    ├─ .replace_all(text, rep) → Cow<str>
                    ├─ .is_match(text) → bool
                    └─ .split(text) → Iterator<&str>

Java:
  pattern string → Pattern.compile() → Pattern
                    ├─ .matcher(text) → Matcher
                    │   ├─ .find() → boolean
                    │   ├─ .matches() → boolean
                    │   ├─ .group(n) → String
                    │   ├─ .replaceAll(repl) → String
                    │   └─ .results() → Stream<MatchResult>
                    ├─ .split(text) → String[]
                    ├─ .splitAsStream(text) → Stream<String>
                    └─ .asPredicate() → Predicate<String>
```

---

## 8. Comparison Tables

### 8.1 Feature Support Comparison

| Feature | Python | JavaScript | Go | Rust | Java |
|------|--------|------------|-----|------|------|
| Backreferences `\1` | OK | OK | No | No | OK |
| Named groups | `(?P<>)` | `(?<>)` | `(?P<>)` | `(?P<>)` | `(?<>)` |
| Positive lookahead `(?=)` | OK | OK | No | No | OK |
| Negative lookahead `(?!)` | OK | OK | No | No | OK |
| Positive lookbehind `(?<=)` | Fixed-length | Variable-length | No | No | Fixed-length |
| Possessive quantifiers `*+` | No | No | No | No | OK |
| Atomic groups | No | No | No | No | Java 20+ |
| Unicode `\p{}` | regex module | `/u` | Partial | OK | OK |
| VERBOSE/comments | `re.X` | No | No | `(?x)` | `(?x)` |
| O(n) guarantee | No | No | OK | OK | No |

### 8.2 Performance Characteristics Comparison

| Language | Engine | Worst-Case Complexity | Compilation Type | Concurrency Safety |
|------|---------|----------|-------------|---------|
| Python | re (NFA) | O(2^n) | Bytecode | Thread-safe |
| JavaScript | V8 Irregexp | O(2^n) | JIT | -- |
| Go | RE2 (DFA) | O(n) | DFA table | Goroutine-safe |
| Rust | regex (DFA) | O(n) | DFA + NFA hybrid | Send + Sync |
| Java | Pattern (NFA) | O(2^n) | NFA bytecode | Thread-safe |

### 8.3 Detailed Unicode Support Comparison

| Feature | Python re | Python regex | JavaScript | Go | Rust | Java |
|------|-----------|-------------|------------|-----|------|------|
| `\w` Unicode support | Default | Default | `\p{L}` required | ASCII | Unicode | Flag required |
| `\d` Unicode support | Default | Default | `/u` | ASCII | Unicode | Flag required |
| `\p{Script}` | No | OK | `/u` | Partial | OK | OK |
| `\p{General_Category}` | No | OK | `/u` | Partial | OK | OK |
| Grapheme cluster `\X` | No | OK | No | No | No | No |
| Emoji support | Limited | OK | `/v` | Limited | OK | Java 20+ |

```
Differences in \w behavior under Unicode:

Input: "hello 世界 café"

Python 3 re:         \w+ → ['hello', '世界', 'café']    (Unicode)
Python 3 re(ASCII):  \w+ → ['hello', 'caf']             (ASCII)
JavaScript:          \w+ → ['hello', 'caf']              (ASCII)
JavaScript (/u):     \w+ → ['hello', 'caf']              (still ASCII even with u!)
Go:                  \w+ → ['hello', 'caf']              (ASCII)
Rust:                \w+ → ['hello', '世界', 'café']     (Unicode)
Java:                \w+ → ['hello', 'caf']              (ASCII)
Java (UNICODE):      \w+ → ['hello', '世界', 'café']     (Unicode)
```

### 8.4 Comparison of Compilation Cache Mechanisms

```
┌──────────┬────────────────────┬─────────────────────────────┐
│ Language  │ Cache mechanism     │ Recommendation               │
├──────────┼────────────────────┼─────────────────────────────┤
│ Python   │ Internal cache(512)│ compile() helps beyond 512   │
│ JavaScript│ V8 internal cache  │ Literals auto-cached         │
│ Go       │ None (manual)      │ Hold in package-level vars   │
│ Rust     │ None (manual)      │ Hold via once_cell / LazyLock│
│ Java     │ None (manual)      │ Hold Pattern as static final │
└──────────┴────────────────────┴─────────────────────────────┘

Performance impact (10,000 matches in a loop):

Python re.search(r'\d+', text) × 10000
  Cache hit:        ~15ms
  Cache miss:       ~50ms  (when over 512 patterns)
  Using compile():  ~12ms

JavaScript /\d+/.test(text) × 10000
  Literal:          ~5ms
  new RegExp():     ~20ms  (compiles each time)

Go re.FindString(text) × 10000
  MustCompiled:     ~8ms
  Compile each time: ~200ms  (extremely slow!)

Rust re.find(text) × 10000
  Lazy::new() done: ~3ms
  Regex::new() each time: ~300ms  (extremely slow!)

Java pattern.matcher(text).find() × 10000
  Pattern.compile() done: ~6ms
  String.matches():       ~60ms  (compiles each time)

※ Numbers are approximate. Actual values depend on hardware and pattern complexity
```

---

## 9. Anti-Patterns

### 9.1 Anti-Pattern: Porting Patterns As-Is Between Languages

```python
# Pattern that works in Python
import re
pattern_py = r'(?P<date>\d{4}-\d{2}-\d{2})'
# (?P<name>...) is the syntax for Python/Go/Rust

# When porting to JavaScript:
# NG: (?P<date>...) is a SyntaxError in JavaScript
# OK: must convert to (?<date>...)
```

```javascript
// Pattern that works in JavaScript
const pattern_js = /(?<=\$)\d+/;
// Variable-length lookbehind is JavaScript-specific

// When porting to Python:
// NG: variable-length lookbehind is not supported in Python re
// OK: use the regex module or change the pattern
```

### 9.2 Anti-Pattern: Not Considering the Unicode Behavior of `\w`

```python
# Python 3: \w matches Unicode characters (by default)
import re
print(re.findall(r'\w+', "hello 世界"))
# => ['hello', '世界']
```

```javascript
// JavaScript: \w is ASCII-only (the u flag does not change this)
console.log("hello 世界".match(/\w+/g));
// => ['hello']  -- '世界' does not match!

// To include Unicode characters:
console.log("hello 世界".match(/[\p{L}\p{N}_]+/gu));
// => ['hello', '世界']
```

### 9.3 Anti-Pattern: Implicit Assumption of Greedy Matching

```python
# Common pitfall across all languages: greedy matching by default
import re

html = '<div class="a">text1</div><div class="b">text2</div>'

# NG: greedy match grabs the entire thing
print(re.findall(r'<div.*>.*</div>', html))
# => ['<div class="a">text1</div><div class="b">text2</div>']

# OK: lazy match
print(re.findall(r'<div.*?>.*?</div>', html))
# => ['<div class="a">text1</div>', '<div class="b">text2</div>']

# BETTER: negated character class (fastest)
print(re.findall(r'<div[^>]*>[^<]*</div>', html))
# => ['<div class="a">text1</div>', '<div class="b">text2</div>']
```

### 9.4 Anti-Pattern: Patterns That Cause ReDoS

```
ReDoS (Regular Expression Denial of Service) is an attack that exploits
the backtracking of NFA engines. Go and Rust are unaffected because they use DFA.

Common structures of dangerous patterns:
  1. Nested quantifiers:           (a+)+
  2. Overlapping alternatives:     (a|a)*
  3. Overlapping character classes:(\w|\d)+

Examples and fixes:

Dangerous: /([\w.]+)+@/         ← nested quantifiers
Safe:      /[\w.]+@/            ← nesting removed

Dangerous: /([a-zA-Z]|[0-9])+/  ← overlapping alternatives
Safe:      /[a-zA-Z0-9]+/       ← merged

Dangerous: /^(a+)+$/            ← classic ReDoS pattern
Safe:      /^a+$/               ← nesting removed

Visualizing backtracking:

Pattern: (a+)+b
Input:   "aaaaaac"

Try 1: (aaaaa)(a) → b? mismatch
Try 2: (aaaa)(aa) → b? mismatch
Try 3: (aaaa)(a)(a) → b? mismatch
Try 4: (aaa)(aaa) → b? mismatch
Try 5: (aaa)(aa)(a) → b? mismatch
...
→ Tries 2^n partitions (32 for n=6, 1 billion for n=30)
```

### 9.5 Anti-Pattern: Unnecessary Capture Groups

```python
# Common across all languages: unnecessary captures impact performance
import re

# NG: using a group when no capture is needed
pattern_bad = re.compile(r'(https?)://([\w.-]+)(/[\w./]*)?')

# OK: use non-capturing groups
pattern_good = re.compile(r'(?:https?)://(?:[\w.-]+)(?:/[\w./]*)?')

# Even better: remove the group entirely if it isn't needed
pattern_best = re.compile(r'https?://[\w.-]+(?:/[\w./]*)?')
```

```go
// Particularly important in Go: FindAllString vs FindAllStringSubmatch
// Return type differs depending on the presence of capture groups
re1 := regexp.MustCompile(`\d+`)
re2 := regexp.MustCompile(`(\d+)`)

// re1.FindAllString()       → []string (efficient)
// re2.FindAllStringSubmatch() → [][]string (more memory)
```

---

## 10. FAQ

### Q1: What is the difference between Python's `re.match` and `re.search`?

**A**: `re.match` only attempts to match from the **beginning** of the string. `re.search` attempts to match at **any position** in the string:

```python
import re
text = "say hello"
print(re.match(r'hello', text))    # => None (beginning is 'say')
print(re.search(r'hello', text))   # => Match (matches at position 4)
```

There is no `match` equivalent in other languages (use the `^` anchor as a substitute).

### Q2: Is precompilation necessary in JavaScript?

**A**: Literal syntax `/pattern/` is compiled at parse time, so explicit precompilation is usually unnecessary. However, if you generate dynamically with `new RegExp()`, you should generate it outside the loop:

```javascript
// OK: literals are compiled at parse time
for (const line of lines) {
    line.match(/\d+/g);  // fast
}

// NG: compiles every iteration of the loop
for (const line of lines) {
    const re = new RegExp("\\d+", "g");  // compiles every time
    line.match(re);
}

// OK: compile outside the loop
const re = /\d+/g;
for (const line of lines) {
    re.lastIndex = 0;  // reset is required for the g flag
    line.match(re);
}
```

### Q3: What should I do in Go when backreferences are needed?

**A**: Backreferences cannot be used in Go's standard `regexp` package. Alternatives:

1. **Split into multiple patterns and match in stages**: use the first match's result to build a second pattern
2. **Use string operations**: substitute with functions from the `strings` package
3. **Third-party library**: `github.com/dlclark/regexp2` (PCRE-compatible)

```go
// Example with regexp2:
// import "github.com/dlclark/regexp2"
// re := regexp2.MustCompile(`<(\w+)>.*?</\1>`, 0)
// match, _ := re.FindStringMatch("<div>test</div>")
```

### Q4: What is the best practice for holding a regex globally in Rust?

**A**: Since `Regex::new()` is expensive in Rust, you should reuse a compiled pattern. The current recommendation is `std::sync::LazyLock` (Rust 1.80+) or `once_cell::sync::Lazy`:

```rust
// For Rust 1.80+
use std::sync::LazyLock;
use regex::Regex;

static EMAIL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\w.+-]+@[\w-]+\.[\w.]+").unwrap()
});

// For earlier than 1.80
use once_cell::sync::Lazy;
static EMAIL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[\w.+-]+@[\w-]+\.[\w.]+").unwrap()
});
```

### Q5: Why is Java's `String.matches()` slow?

**A**: `String.matches()` internally calls `Pattern.compile()` every time. Using it in a loop causes the same pattern to be compiled repeatedly, causing significant performance degradation:

```java
// NG: 10,000 compilations occur
for (String line : lines) {
    if (line.matches("\\d{4}-\\d{2}-\\d{2}")) { ... }
}

// OK: compiles only once
Pattern p = Pattern.compile("\\d{4}-\\d{2}-\\d{2}");
for (String line : lines) {
    if (p.matcher(line).matches()) { ... }
}
```

### Q6: What is the common subset for using the same pattern across multiple languages?

**A**: The following syntax can be used in common across all five languages:

```
Common subset across 5 languages:
  ✓ Character classes:       [abc], [a-z], [^abc]
  ✓ Metacharacters:          \d, \D, \s, \S, \w (* Unicode behavior differs)
  ✓ Quantifiers:             *, +, ?, {n}, {n,}, {n,m}
  ✓ Lazy quantifiers:        *?, +?, ??
  ✓ Anchors:                 ^, $, \b
  ✓ Alternation:             a|b
  ✓ Groups:                  (...)
  ✓ Non-capturing:           (?:...)
  ✓ Inline flags:            (?i), (?m), (?s) (* some restrictions)

Syntax requiring caution:
  △ Named groups:    (?P<name>...) for Python/Go/Rust
                     (?<name>...) for JavaScript/Java
  △ \w Unicode:      Python/Rust are Unicode, others ASCII
  × Lookarounds:     not available in Go/Rust
  × Backreferences:  not available in Go/Rust
```

### Q7: How does regex debugging differ across languages?

**A**: Each language has its own features useful for debugging regular expressions:

```python
# Python: re.DEBUG flag
import re
re.compile(r'(\d{4})-(\d{2})', re.DEBUG)
# Displays the parse tree of the pattern:
# SUBPATTERN 1 0 0
#   MAX_REPEAT 4 4
#     IN
#       CATEGORY CATEGORY_DIGIT
# LITERAL 45
# SUBPATTERN 2 0 0
#   MAX_REPEAT 2 2
#     IN
#       CATEGORY CATEGORY_DIGIT
```

```go
// Go: regexp.Compile error messages are detailed
_, err := regexp.Compile(`(?<=abc)`)
// err: "error parsing regexp: invalid or unsupported Perl syntax: `(?<`"

// Use the regexp/syntax package for syntactic analysis
import "regexp/syntax"
prog, err := syntax.Parse(`\d+`, syntax.Perl)
// Get the syntax tree to analyze
```

```java
// Java: Matcher.toMatchResult() saves the match state
Pattern p = Pattern.compile("(\\w+)");
Matcher m = p.matcher("hello world");
while (m.find()) {
    MatchResult mr = m.toMatchResult();
    System.out.printf("group=%s start=%d end=%d%n",
        mr.group(), mr.start(), mr.end());
}
```

### Q8: What are the pitfalls of JavaScript's g flag?

**A**: A RegExp with the `g` flag has internal `lastIndex` state, and consecutive `test()` or `exec()` calls start searching from the previous match position. This can cause unexpected `false` returns:

```javascript
const re = /abc/g;

// 1st call: search from lastIndex=0 → match, lastIndex=3
console.log(re.test("abcabc"));  // true

// 2nd call: search from lastIndex=3 → match, lastIndex=6
console.log(re.test("abcabc"));  // true

// 3rd call: search from lastIndex=6 → no match, lastIndex=0
console.log(re.test("abcabc"));  // false!

// Countermeasure 1: reset lastIndex each time
re.lastIndex = 0;

// Countermeasure 2: don't use the g flag with test()
const reNoG = /abc/;
console.log(reNoG.test("abcabc"));  // always true

// Countermeasure 3: use String.prototype.match()
console.log("abcabc".match(/abc/g));  // ['abc', 'abc']
```

---

## 11. Practical Pattern Collection: Cross-Language Reference

### 11.1 Email Address Validation

```python
# Python
import re
email_re = re.compile(r'^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$')
print(email_re.match("user@example.com"))  # Match
print(email_re.match("invalid@"))          # None
```

```javascript
// JavaScript
const emailRe = /^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$/;
console.log(emailRe.test("user@example.com"));  // true
console.log(emailRe.test("invalid@"));           // false
```

```go
// Go
emailRe := regexp.MustCompile(`^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$`)
fmt.Println(emailRe.MatchString("user@example.com"))  // true
fmt.Println(emailRe.MatchString("invalid@"))           // false
```

```rust
// Rust
let email_re = Regex::new(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$").unwrap();
println!("{}", email_re.is_match("user@example.com"));  // true
println!("{}", email_re.is_match("invalid@"));           // false
```

```java
// Java
Pattern emailRe = Pattern.compile("[\\w.+-]+@[\\w-]+(?:\\.[\\w-]+)+");
System.out.println(emailRe.matcher("user@example.com").matches());  // true
System.out.println(emailRe.matcher("invalid@").matches());           // false
```

### 11.2 Extracting IPv4 Addresses

```python
# Python
import re
ipv4_re = re.compile(
    r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
)
text = "Server at 192.168.1.100, gateway 10.0.0.1, invalid 999.999.999.999"
print(ipv4_re.findall(text))
# => ['192.168.1.100', '10.0.0.1']
```

```javascript
// JavaScript
const ipv4Re = /\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b/g;
const text = "Server at 192.168.1.100, gateway 10.0.0.1, invalid 999.999.999.999";
console.log(text.match(ipv4Re));
// => ['192.168.1.100', '10.0.0.1']
```

```go
// Go
ipv4Re := regexp.MustCompile(
    `\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b`,
)
text := "Server at 192.168.1.100, gateway 10.0.0.1"
fmt.Println(ipv4Re.FindAllString(text, -1))
// => [192.168.1.100 10.0.0.1]
```

```rust
// Rust
let ipv4_re = Regex::new(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
).unwrap();
let text = "Server at 192.168.1.100, gateway 10.0.0.1";
let ips: Vec<&str> = ipv4_re.find_iter(text).map(|m| m.as_str()).collect();
println!("{:?}", ips);
// => ["192.168.1.100", "10.0.0.1"]
```

```java
// Java
Pattern ipv4Re = Pattern.compile(
    "\\b(?:(?:25[0-5]|2[0-4]\\d|[01]?\\d\\d?)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d\\d?)\\b"
);
String text = "Server at 192.168.1.100, gateway 10.0.0.1";
Matcher m = ipv4Re.matcher(text);
while (m.find()) {
    System.out.println(m.group());
}
// => 192.168.1.100
//    10.0.0.1
```

### 11.3 Parsing a Log File

```python
# Python: structured log analysis
import re

log_re = re.compile(r'''
    ^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})  # ISO 8601
    \s+\[(?P<level>\w+)\]                                  # log level
    \s+(?P<source>[\w.]+):                                  # source
    \s+(?P<message>.+)$                                     # message
''', re.VERBOSE | re.MULTILINE)

log_text = """2026-02-11T10:30:45 [ERROR] app.server: Connection refused
2026-02-11T10:30:46 [INFO] app.db: Retry attempt 1
2026-02-11T10:30:47 [WARN] app.cache: Cache miss for key 'user:123'"""

for m in log_re.finditer(log_text):
    d = m.groupdict()
    print(f"  [{d['level']}] {d['source']} → {d['message']}")
```

```javascript
// JavaScript: parsing the same log
const logRe = /^(?<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+\[(?<level>\w+)\]\s+(?<source>[\w.]+):\s+(?<message>.+)$/gm;

const logText = `2026-02-11T10:30:45 [ERROR] app.server: Connection refused
2026-02-11T10:30:46 [INFO] app.db: Retry attempt 1
2026-02-11T10:30:47 [WARN] app.cache: Cache miss for key 'user:123'`;

for (const m of logText.matchAll(logRe)) {
    const { level, source, message } = m.groups;
    console.log(`  [${level}] ${source} → ${message}`);
}
```

```go
// Go: parsing the same log
logRe := regexp.MustCompile(
    `(?m)^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+` +
    `\[(?P<level>\w+)\]\s+(?P<source>[\w.]+):\s+(?P<message>.+)$`,
)

logText := `2026-02-11T10:30:45 [ERROR] app.server: Connection refused
2026-02-11T10:30:46 [INFO] app.db: Retry attempt 1`

names := logRe.SubexpNames()
for _, match := range logRe.FindAllStringSubmatch(logText, -1) {
    for i, name := range names {
        if name != "" {
            fmt.Printf("  %s: %s\n", name, match[i])
        }
    }
    fmt.Println()
}
```

### 11.4 Parsing and Decomposing URLs

```python
# Python
import re

url_re = re.compile(r'''
    ^(?P<scheme>https?)://
    (?P<host>[^/:]+)
    (?::(?P<port>\d+))?
    (?P<path>/[^?#]*)?
    (?:\?(?P<query>[^#]*))?
    (?:\#(?P<fragment>.*))?$
''', re.VERBOSE)

urls = [
    "https://example.com:8080/api/v1/users?page=1&limit=10#section",
    "http://localhost/health",
    "https://cdn.example.com/assets/style.css",
]

for url in urls:
    m = url_re.match(url)
    if m:
        d = {k: v for k, v in m.groupdict().items() if v is not None}
        print(f"  {d}")
```

```javascript
// JavaScript
const urlRe = /^(?<scheme>https?):\/\/(?<host>[^\/:]+)(?::(?<port>\d+))?(?<path>\/[^?#]*)?(?:\?(?<query>[^#]*))?(?:#(?<fragment>.*))?$/;

const urls = [
    "https://example.com:8080/api/v1/users?page=1&limit=10#section",
    "http://localhost/health",
];

for (const url of urls) {
    const m = url.match(urlRe);
    if (m) {
        // exclude undefined
        const parts = Object.fromEntries(
            Object.entries(m.groups).filter(([_, v]) => v !== undefined)
        );
        console.log(parts);
    }
}
```

### 11.5 Parsing CSV Fields (Quote-Aware)

```python
# Python: parsing CSV with quoted fields
import re

# Correctly split quoted fields and regular fields
csv_field_re = re.compile(r'''
    (?:                     # start of a field
      "([^"]*(?:""[^"]*)*)" # quoted: "..." (internal "" is an escape)
      |                     # or
      ([^,]*)               # unquoted: until ,
    )
    (?:,|$)                 # , or end of line
''', re.VERBOSE)

line = '"John ""Johnny"" Doe",30,"New York, NY",active'
fields = []
for m in csv_field_re.finditer(line):
    if m.group(1) is not None:
        fields.append(m.group(1).replace('""', '"'))
    else:
        fields.append(m.group(2))
print(fields)
# => ['John "Johnny" Doe', '30', 'New York, NY', 'active']
```

---

## 12. Porting Checklist

Checklist when porting regular expressions between languages:

```
┌─────────────────────────────────────────────────────────────────┐
│ Source → Destination                                            │
├─────────────────────────────────────────────────────────────────┤
│ Python → JavaScript                                            │
│  □ Change (?P<name>...) to (?<name>...)                        │
│  □ Remove re.VERBOSE comments (JS does not support comments)   │
│  □ Change \w Unicode behavior to [\p{L}\p{N}_]                 │
│  □ Change fullmatch() to /^...$/                               │
│  □ Conditional patterns (?(id)yes|no) unavailable, use logic   │
│  □ Change re.sub function callback to replace callback         │
├─────────────────────────────────────────────────────────────────┤
│ Python → Go                                                    │
│  □ Replace lookarounds with capture group + post-processing    │
│  □ Replace backreferences with multi-step processing           │
│  □ \w Unicode behavior is ASCII-only in Go                     │
│  □ Change re.sub function callback to ReplaceAllStringFunc     │
│  □ Change findall to FindAllString (-1 needed as second arg)   │
├─────────────────────────────────────────────────────────────────┤
│ JavaScript → Python                                            │
│  □ Change (?<name>...) to (?P<name>...)                        │
│  □ Variable-length lookbehind → fixed-length or regex module   │
│  □ /pattern/flags → re.compile(r'pattern', flags)              │
│  □ $1, $2 → \1, \2 (in sub replacement string)                │
│  □ g flag → use findall/finditer for all matches               │
│  □ v flag set operations → substitute with the regex module    │
├─────────────────────────────────────────────────────────────────┤
│ Java → Rust                                                    │
│  □ Change (?<name>...) to (?P<name>...)                        │
│  □ Lookarounds → switch to fancy-regex or remove               │
│  □ Backreferences → switch to fancy-regex or remove            │
│  □ Possessive quantifiers *+ → remove (not in Rust regex)      │
│  □ Double escapes \\\\ → change to raw strings r""             │
│  □ Matcher state management → switch to iterator-based API     │
├─────────────────────────────────────────────────────────────────┤
│ From any language to Go/Rust                                    │
│  □ Lookarounds in general → remove and substitute with logic   │
│  □ Backreferences → remove and substitute with logic           │
│  □ Atomic groups → not needed (DFA, no backtracking)           │
│  □ Possessive quantifiers → not needed (DFA, no backtracking)  │
│  □ ReDoS countermeasures → not needed (O(n) guaranteed)        │
└─────────────────────────────────────────────────────────────────┘
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

The most important thing is to gain practical experience. Beyond just theory, your understanding deepens by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping into advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Item | Content |
|------|------|
| Python | NFA, Unicode default, `(?P<>)` syntax, `re.VERBOSE` |
| JavaScript | NFA, significantly extended in ES2018, `/u` `/s` `/d` flags |
| Go | RE2 (DFA), O(n) guaranteed, no backreferences/lookarounds |
| Rust | DFA hybrid, O(n) guaranteed, simultaneous multi-match with `RegexSet` |
| Java | NFA, supports possessive quantifiers, double escapes mandatory |
| Porting caveats | Named group syntax, `\w` Unicode behavior, lookbehind constraints differ |

## Recommended Next Reading

- [01-common-patterns.md](./01-common-patterns.md) -- Collection of frequently used patterns
- [02-text-processing.md](./02-text-processing.md) -- Text processing tools

## References

1. **Python re module** https://docs.python.org/3/library/re.html -- Official reference for Python regular expressions
2. **MDN RegExp** https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/RegExp -- Comprehensive guide to the JavaScript RegExp
3. **Go regexp package** https://pkg.go.dev/regexp -- Specification of Go's regexp package
4. **Rust regex crate** https://docs.rs/regex/latest/regex/ -- Documentation for Rust's regex crate
5. **Java Pattern class** https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/regex/Pattern.html -- Official reference for Java's Pattern class
6. **RE2 Syntax** https://github.com/google/re2/wiki/Syntax -- Syntax specification of the RE2 engine
7. **fancy-regex** https://docs.rs/fancy-regex/latest/fancy_regex/ -- Rust regex crate with lookaround support
8. **Python regex module** https://pypi.org/project/regex/ -- Python's extended regular expression module
9. **ECMAScript 2024 RegExp v flag** https://tc39.es/ecma262/ -- The latest JavaScript RegExp specification
10. **Russ Cox "Regular Expression Matching Can Be Simple And Fast"** https://swtch.com/~rsc/regexp/regexp1.html -- Explanation of the principles of NFA/DFA engines
