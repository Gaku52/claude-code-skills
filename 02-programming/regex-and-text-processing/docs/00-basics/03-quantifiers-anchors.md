# Quantifiers and Anchors -- *+?{n,m}, ^$\b, Greedy/Lazy

> Quantifiers control repetition counts, and anchors constrain positions. Accurately understanding the difference between greedy matching and lazy matching is the key to writing patterns that behave as intended.

## What You Will Learn in This Chapter

1. **Types and behavior of quantifiers** -- Precise meanings and usage of `*` `+` `?` `{n,m}`
2. **Greedy matching vs. lazy matching** -- Why the default behavior is "longest match" and how to control it
3. **Possessive quantifiers and atomic groups** -- Speed optimization by preventing backtracking
4. **Types and applications of anchors** -- All position-specification patterns using `^` `$` `\b` `\A` `\Z` `\G`
5. **Cross-language differences** -- Behavioral comparison across Python / JavaScript / Java / Ruby / Perl / Go / Rust
6. **Impact on performance** -- How quantifier choice affects backtracking count


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Character Classes -- [abc], \d, \w, \s, POSIX](./02-character-classes.md)

---

## 1. Quantifiers

### 1.1 Basic Quantifiers

```python
import re

text = "aaabbbccc"

# * : 0 or more times (matches even with 0 occurrences)
print(re.findall(r'a*', text))
# => ['aaa', '', '', '', '', '', '', '']
# Note: Also matches the empty string (0 repetitions)

# + : 1 or more times (requires at least 1)
print(re.findall(r'a+', text))
# => ['aaa']

# ? : 0 or 1 time
print(re.findall(r'a?', text))
# => ['a', 'a', 'a', '', '', '', '', '', '', '']
```

**Why does `*` match the empty string?**

`*` means "0 or more times," so it matches the empty string as "0 repetitions" even at positions where the target character does not exist. This originates from the mathematical definition of regular expressions. In formal language theory, the empty string (epsilon, ε) is included in the Kleene closure of any language.

```python
import re

# Detailed behavior of empty string matching
text = "XY"
matches = list(re.finditer(r'a*', text))
for m in matches:
    print(f"Position {m.start()}-{m.end()}: '{m.group()}'")
# Position 0-0: ''    <- Before X (no 'a' = 0 matches)
# Position 1-1: ''    <- Before Y (no 'a' = 0 matches)
# Position 2-2: ''    <- End of string (no 'a' = 0 matches)

# Practical issue: situations where + should be used instead of *
text = "abc 123 def"
# NG: Also matches the empty string
print(re.findall(r'\d*', text))
# => ['', '', '', '', '123', '', '', '', '', '']

# OK: Require at least 1 digit
print(re.findall(r'\d+', text))
# => ['123']
```

### 1.2 Range Specification `{n,m}`

```python
import re

# {n}   : exactly n times
# {n,}  : n or more times
# {n,m} : between n and m times (inclusive)
# {,m}  : 0 to m times (some engines only)

text = "1 12 123 1234 12345"

print(re.findall(r'\d{3}', text))     # Exactly 3 digits: ['123', '123', '123']
print(re.findall(r'\b\d{3}\b', text)) # With word boundaries: ['123']
print(re.findall(r'\d{2,4}', text))   # 2-4 digits: ['12', '123', '1234', '1234']
print(re.findall(r'\d{3,}', text))    # 3+ digits: ['123', '1234', '12345']
```

**Syntax notes for `{n,m}`**:

```python
import re

# Spaces are not allowed in {n,m} (engine-dependent)
text = "aaa"

# Python: With a space, it is treated as a literal
print(re.findall(r'a{2,3}', text))   # => ['aaa']  (quantifier)
print(re.findall(r'a{2, 3}', text))  # => []  (literal string "a{2, 3}")

# {n,m} where n > m is an error
try:
    re.compile(r'a{3,2}')
except re.error as e:
    print(f"Error: {e}")
    # => "min repeat greater than max repeat"

# Limits on large repetition counts
try:
    re.compile(r'a{1,65536}')  # Python has an upper limit
except re.error as e:
    print(f"Error: {e}")
```

**Support for `{,m}` across languages**:

```
┌──────────────┬─────────────────────────────────────┐
│ Language/     │ Support for {,m}                     │
│ Engine       │                                      │
├──────────────┼─────────────────────────────────────┤
│ Python       │ ○ Supported (equivalent to {0,m})   │
│ JavaScript   │ ✗ Treated as literal                 │
│ Java         │ ✗ Treated as literal                 │
│ PCRE         │ ○ Supported                          │
│ Ruby         │ ○ Supported                          │
│ Perl         │ ○ Supported                          │
│ Go (RE2)     │ ○ Supported                          │
│ Rust (regex) │ ○ Supported                          │
│ .NET         │ ○ Supported                          │
└──────────────┴─────────────────────────────────────┘
```

### 1.3 Equivalence Relationships of Quantifiers

```
Quantifier syntactic sugar:

  *     ≡  {0,}    0 or more
  +     ≡  {1,}    1 or more
  ?     ≡  {0,1}   0 or 1

  {3}   -> exactly 3 times
  {3,}  -> 3 or more (no upper limit)
  {3,5} -> between 3 and 5 (inclusive)
  {0,5} -> between 0 and 5 (inclusive)
```

### 1.4 What Quantifiers Apply To

A quantifier applies to the "preceding element." It is important to accurately understand what this "preceding element" is.

```python
import re

text = "abcabcabc"

# Applied to a single character: 'c' repeated 0 or more times
print(re.findall(r'abc*', text))      # => ['abc', 'abc', 'abc']
# c* is the repetition of 'c'

# Applied to a group: 'abc' repeated 1 or more times
print(re.findall(r'(?:abc)+', text))  # => ['abcabcabc']
# (?:abc)+ is the repetition of the 'abc' group

# Applied to a character class: [a-c] repeated 2 or more times
print(re.findall(r'[a-c]{2,}', text)) # => ['abcabcabc']

# Applied to an escape sequence: \d repeated 3 times
text2 = "abc123def456"
print(re.findall(r'\d{3}', text2))    # => ['123', '456']
```

**Consecutive (nested) quantifiers**:

```python
import re

# Directly chaining quantifiers causes an error
try:
    re.compile(r'a**')
except re.error as e:
    print(f"Error: {e}")  # multiple repeat

# You can nest them using groups
text = "aaa bbb aaa bbb aaa"
print(re.findall(r'(?:a{3}\s?){2,}', text))
# => ['aaa bbb aaa bbb aaa']

# Practical example: repeated IP address pattern
ip_pattern = r'(?:\d{1,3}\.){3}\d{1,3}'
print(re.findall(ip_pattern, "Server 192.168.1.100 and 10.0.0.1"))
# => ['192.168.1.100', '10.0.0.1']
```

### 1.5 Quantifiers and Empty Matches

```python
import re

# How findall/finditer handle empty matches
text = "abc"

# Python 3.7+ includes a specification change to prevent consecutive empty matches
# Python 3.6 and earlier: retries at the same position after an empty match
# Python 3.7+: advances one character before retrying after an empty match

# Empty matches with *
for m in re.finditer(r'x*', text):
    print(f"Position {m.start()}-{m.end()}: '{m.group()}'")
# Python 3.7+:
# Position 0-0: ''
# Position 1-1: ''
# Position 2-2: ''
# Position 3-3: ''

# How sub handles empty matches
print(re.sub(r'x*', '-', 'abc'))
# Python 3.7+: '-a-b-c-'
# Python 3.6 and earlier: '-a-b-c-' (varies by implementation)
```

```javascript
// Empty matches in JavaScript
const text = "abc";

// ES2020+ matchAll
const matches = [...text.matchAll(/x*/g)];
console.log(matches.map(m => `${m.index}: '${m[0]}'`));
// ["0: ''", "1: ''", "2: ''", "3: ''"]

// Empty matches in replace
console.log("abc".replace(/x*/g, "-"));
// "-a-b-c-"
```

---

## 2. Greedy Matching vs. Lazy Matching

### 2.1 Greedy -- Default Behavior

```python
import re

text = '<div>hello</div><div>world</div>'

# Greedy matching (default): tries to match as much as possible
greedy = re.search(r'<div>.*</div>', text)
print(greedy.group())
# => '<div>hello</div><div>world</div>'
#    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#    Longest match: from the first <div> to the last </div>
```

**Internal workings of greedy matching (backtracking details)**:

```
Pattern: <div>.*</div>
Text:    <div>hello</div><div>world</div>
         0123456789...

Step 1: '<div>' matches text positions 0-4
Step 2: '.*' greedily consumes all remaining characters
        -> Match range: position 5 to position 37 (end)
Step 3: Attempts to match '</div>'
        -> No remaining characters -> Failure -> Backtrack
Step 4: Back up 1 char (pos 36): '>' ≠ '<' -> Failure -> Backtrack
Step 5: Back up 1 char (pos 35): '>' ≠ '<' -> Failure -> Backtrack
  ...
Step N: Back up to position 31: '</div>' matches!
        -> Result: '<div>hello</div><div>world</div>'

Backtrack count: approximately 6 times
```

### 2.2 Lazy -- Appending `?`

```python
import re

text = '<div>hello</div><div>world</div>'

# Lazy matching: tries to match as little as possible
lazy = re.findall(r'<div>.*?</div>', text)
print(lazy)
# => ['<div>hello</div>', '<div>world</div>']
#    Shortest match: from the first <div> to the nearest </div>
```

**Internal workings of lazy matching**:

```
Pattern: <div>.*?</div>
Text:    <div>hello</div><div>world</div>

Step 1: '<div>' matches text positions 0-4
Step 2: '.*?' lazily tries 0 characters first
Step 3: Attempts '</div>' -> pos 5 'h' ≠ '<' -> Failure
        -> Expand .*? by 1 character
Step 4: .*? = 'h', attempt '</div>' from pos 6 -> 'e' ≠ '<' -> Failure
        -> Expand .*? by 1 character
Step 5: .*? = 'he', attempt '</div>' from pos 7 -> 'l' ≠ '<' -> Failure
  ...
Step 8: .*? = 'hello', attempt '</div>' from pos 10 -> Match!
        -> Result: '<div>hello</div>'

Expansion attempts: 5
```

### 2.3 Greedy vs. Lazy -- All Quantifiers

```python
import re

text = "aabab"

# Greedy (default)            Lazy (with ? appended)
print(re.search(r'a.*b', text).group())    # => 'aabab' (longest)
print(re.search(r'a.*?b', text).group())   # => 'aab'   (shortest)

print(re.search(r'a.+b', text).group())    # => 'aabab' (longest)
print(re.search(r'a.+?b', text).group())   # => 'aabab' (.+ requires at least 1 char)

print(re.search(r'a.?b', text).group())    # => 'aab'
print(re.search(r'a.??b', text).group())   # => 'ab' (positions 2-3)
```

**Important note: Lazy matching is "shortest" but not "leftmost shortest"**:

```python
import re

text = "aXbYaZb"

# Lazy matching starts from the left (the leftmost match principle still applies)
print(re.search(r'a.*?b', text).group())  # => 'aXb' (shortest from the left)
# 'aZb' is shorter, but 'aXb' is found first

# findall returns all non-overlapping matches
print(re.findall(r'a.*?b', text))  # => ['aXb', 'aZb']
```

### 2.4 Visualizing the Behavior

```
Greedy match: .*  (Pattern: <.*>, Text: <b>bold</b>)

Step 1: < matches                              <
Step 2: .* swallows all characters             <b>bold</b>
Step 3: > doesn't match (end of string)        -> Backtrack
Step 4: Back up 1 char, try >                  <b>bold</b  -> > ≠ b -> back up
Step 5: Back up 1 more char                    <b>bold</b> -> > = > -> Match!
Result: <b>bold</b>  (longest match)

─────────────────────────────────────────

Lazy match: .*?  (Pattern: <.*?>, Text: <b>bold</b>)

Step 1: < matches                              <
Step 2: .*? tries 0 chars first                <  -> > ≠ b -> expand
Step 3: Expand by 1 char                       <b -> > ≠ o -> expand
Step 4: Expand 1 more... wait                  <b> -> > = > -> Match!
Result: <b>  (shortest match)
```

### 2.5 Possessive Quantifiers -- No Backtracking

```java
// Supported in Java and PCRE (not supported by Python's re)
// Available in Python via the regex module (third-party)

// Greedy:     .*   (with backtracking)
// Lazy:       .*?  (shortest match)
// Possessive: .*+  (no backtracking -> fast but more likely to fail)

// Java example:
String text = "aaaa";
// Greedy:     a*a  -> "aaaa" (a* takes 3, last a takes 1)
// Possessive: a*+a -> No match (a*+ takes all, no backtracking)
```

**Practical use of possessive quantifiers**:

```java
import java.util.regex.*;

public class PossessiveExample {
    public static void main(String[] args) {
        // 1. Fast non-match detection
        // Possessive quantifiers quickly determine "no match"
        String longText = "a".repeat(100000) + "X";

        // Greedy: a*a -> massive backtracking
        long start1 = System.nanoTime();
        Pattern.matches("a*a$", longText);  // Slow
        long time1 = System.nanoTime() - start1;

        // Possessive: a*+X -> immediate determination with no backtracking
        long start2 = System.nanoTime();
        Pattern.matches("a*+X", longText);  // Fast
        long time2 = System.nanoTime() - start2;

        System.out.println("Greedy: " + time1 + "ns");
        System.out.println("Possessive: " + time2 + "ns");

        // 2. Use in CSV parsing
        String csvLine = "field1,field2,\"field with, comma\",field4";

        // Quoted field: possessively consume non-quote characters
        Pattern csvQuoted = Pattern.compile("\"[^\"]*+\"");
        Matcher m = csvQuoted.matcher(csvLine);
        while (m.find()) {
            System.out.println("Match: " + m.group());
        }
        // => Match: "field with, comma"

        // 3. Parsing numeric literals
        // Possessively consume the integer part, handle decimal part separately
        Pattern numPattern = Pattern.compile("\\d++\\.?\\d*+");
        String expr = "3.14 + 42 - 0.5";
        m = numPattern.matcher(expr);
        while (m.find()) {
            System.out.println("Number: " + m.group());
        }
        // => Number: 3.14
        // => Number: 42
        // => Number: 0.5
    }
}
```

**Possessive quantifiers in Python's regex module**:

```python
# pip install regex
import regex

text = "aaaaab"

# Possessive quantifier
m = regex.search(r'a++b', text)
print(m.group())  # => 'aaaaab'

# Case where possessive quantifier causes match failure
m = regex.search(r'a++a', text)
print(m)  # => None (a++ consumes all a's and does not backtrack)

# Atomic group (?>...) provides equivalent functionality
m = regex.search(r'(?>a+)b', text)
print(m.group())  # => 'aaaaab'

m = regex.search(r'(?>a+)a', text)
print(m)  # => None
```

### 2.6 Atomic Groups

An atomic group `(?>...)` is a group that does not backtrack once its internal match is established. It is conceptually equivalent to a possessive quantifier but can be applied to a broader range of constructs.

```java
import java.util.regex.*;

public class AtomicGroupExample {
    public static void main(String[] args) {
        // Possessive quantifier: applies to a single quantifier
        // a*+  ≡  (?>a*)

        // Atomic group: can apply to multiple elements
        // (?>abc|abcd)  -- once abc matches, no backtracking

        String text = "abcd";

        // Normal alternation
        Pattern p1 = Pattern.compile("(?:abc|abcd)d");
        System.out.println(p1.matcher(text).find());  // false
        // abc matches -> expects d -> d is there -> overall match... wait
        // Actually "abcd" is the entire target, abc + d = abcd -> OK

        // A clearer example
        String text2 = "abcde";
        Pattern p2 = Pattern.compile("(?:abc|abcde)$");
        Pattern p3 = Pattern.compile("(?>abc|abcde)$");

        System.out.println(p2.matcher(text2).find());  // true (matches abcde)
        System.out.println(p3.matcher(text2).find());  // false (commits to abc, fails at $ without backtracking)
    }
}
```

```ruby
# Ruby natively supports atomic groups
text = "aaaaab"

# Atomic group
puts text.match(/(?>a+)b/)    # => aaaaab
puts text.match(/(?>a+)a/)    # => nil (no backtracking)

# Practical example: fast email validation
email_pattern = /\A(?>[\w.+-]+)@(?>[\w-]+\.)+\w{2,}\z/
puts "user@example.com".match?(email_pattern)  # => true
puts "invalid@@email".match?(email_pattern)     # => false
```

### 2.7 Quantifier Comparison Table

| Greedy | Lazy | Possessive | Behavior |
|--------|------|-----------|----------|
| `*` | `*?` | `*+` | 0 or more |
| `+` | `+?` | `++` | 1 or more |
| `?` | `??` | `?+` | 0 or 1 |
| `{n,m}` | `{n,m}?` | `{n,m}+` | n to m times |

| Property | Greedy | Lazy | Possessive |
|----------|--------|------|-----------|
| Match strategy | Longest match | Shortest match | Longest (no backtracking on failure) |
| Backtracking | Yes | Yes | No |
| Speed | Normal | Normal | Fast (on match) |
| Use case | Default | Tag extraction, etc. | Performance optimization |
| Support | All engines | All engines | Java/PCRE/regex (Python) |

### 2.8 Comparing Backtrack Counts

Comparing backtrack counts with concrete examples to visualize their performance impact.

```python
import re
import time

# Test case: extracting HTML tags
html = '<div class="container">' + 'x' * 10000 + '</div>'

# Pattern 1: Greedy + negated character class (fastest)
pattern1 = r'<div[^>]*>[^<]*</div>'

# Pattern 2: Lazy (medium speed)
pattern2 = r'<div.*?>.*?</div>'

# Pattern 3: Greedy (slowest -- massive backtracking)
pattern3 = r'<div.*>.*</div>'

for name, pattern in [("Negated class", pattern1), ("Lazy", pattern2), ("Greedy", pattern3)]:
    start = time.perf_counter()
    for _ in range(1000):
        re.search(pattern, html, re.DOTALL)
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f} sec")

# Typical results:
# Negated class: 0.0150 sec
# Lazy:          0.0450 sec
# Greedy:        0.0800 sec
```

```
Estimated backtrack counts:

Pattern: <div>.*</div>  (greedy)
Text:    <div>XXXX...XXXX</div> (10000 X's)

1. .* consumes 10006 characters ('XXXX...XXXX</div>' entirely)
2. Attempts to match '</div>' -> Failure
3. Backtracks 1 character at a time x ~10000 times
4. Reaches the '</div>' position -> Match
-> Backtrack count: ~10000 times

Pattern: <div>.*?</div>  (lazy)
1. .*? starts with 0 characters
2. Attempts to match '</div>' -> Failure -> Expand by 1 character
3. Repeats x ~10000 times
4. After consuming all 'X' characters, '</div>' matches
-> Expansion attempts: ~10000 times (comparable to greedy but in reverse direction)

Pattern: <div>[^<]*</div>  (negated class)
1. [^<]* consumes all non-< characters at once
2. Attempts to match </div> -> Success
-> Backtrack count: 0 times (fastest)
```

---

## 3. Anchors

### 3.1 Line Start/End Anchors

```python
import re

text = """first line
second line
third line"""

# ^ : Line start (by default, matches only the start of the string)
print(re.findall(r'^.+', text))
# => ['first line']

# $ : Line end (by default, matches only the end of the string)
print(re.findall(r'.+$', text))
# => ['third line']

# re.MULTILINE: ^ and $ match the start/end of each line
print(re.findall(r'^.+', text, re.MULTILINE))
# => ['first line', 'second line', 'third line']

print(re.findall(r'^\w+', text, re.MULTILINE))
# => ['first', 'second', 'third']
```

**Subtle behavior of `$` with trailing newlines**:

```python
import re

# Python's $ also matches "before" a trailing newline at the end of the string
text_with_newline = "hello\n"
text_without_newline = "hello"

print(re.search(r'hello$', text_with_newline))       # Matches!
print(re.search(r'hello$', text_without_newline))     # Matches

# \Z matches at the very end (Python-specific)
print(re.search(r'hello\Z', text_with_newline))       # None!
print(re.search(r'hello\Z', text_without_newline))    # Matches

# To match including the trailing newline exactly
print(re.search(r'hello\n?\Z', text_with_newline))    # Matches
```

```javascript
// Behavior of $ in JavaScript
const text = "hello\n";

// $ by default matches only the end of the string
console.log(/hello$/.test(text));        // false (because of the newline)
console.log(/hello$/.test("hello"));     // true

// With m flag, matches each line end
console.log(/hello$/m.test(text));       // true
```

```ruby
# In Ruby, $ always matches the end of a line (just before \n)
text = "hello\nworld"
puts text.scan(/\w+$/)   # => ["hello", "world"]
# In Ruby, $ behaves like MULTILINE by default
# To match only the end of the string, use \z
puts text.match?(/world\z/)  # => true
puts text.match?(/hello\z/)  # => false
```

### 3.2 String Boundary Anchors

```python
import re

text = "hello\nworld"

# \A : Absolute start of the string (not affected by MULTILINE)
print(re.search(r'\Ahello', text).group())  # => 'hello'
print(re.search(r'\Aworld', text))          # => None

# \Z : Absolute end of the string (not affected by MULTILINE)
print(re.search(r'world\Z', text).group())  # => 'world'

# ^ vs \A (MULTILINE mode)
print(re.findall(r'^\w+', text, re.M))   # => ['hello', 'world']
print(re.findall(r'\A\w+', text, re.M))  # => ['hello'] (start only)
```

**Differences between `\z` and `\Z` across languages**:

```
┌──────────────┬──────────────────────────────────────────┐
│ Anchor       │ Behavior                                  │
├──────────────┼──────────────────────────────────────────┤
│ \Z (Python)  │ End of string; does not include trailing  │
│              │ newline                                    │
│ \z (Python)  │ Not supported                             │
│ \Z (Ruby)    │ End of string (also before trailing \n)   │
│ \z (Ruby)    │ Absolute end of string (past trailing \n) │
│ \Z (Java)    │ End of string (also before trailing \n)   │
│ \z (Java)    │ Absolute end of string                    │
│ \Z (Perl)    │ End of string (also before trailing \n)   │
│ \z (Perl)    │ Absolute end of string                    │
└──────────────┴──────────────────────────────────────────┘

Note: Python's \Z is equivalent to \z in other languages.
Python has no anchor equivalent to \Z in other languages.
```

```ruby
# Clear difference between \z and \Z in Ruby
text = "hello\n"

puts text.match?(/hello\Z/)  # => true  (matches before the \n)
puts text.match?(/hello\z/)  # => false (absolute end is \n)
puts text.match?(/hello\n\z/) # => true (including the newline up to the end)
```

### 3.3 Word Boundary `\b`

```python
import re

text = "cat concatenate category caterpillar"

# \b : Word boundary
# The position between a word character (\w) and a non-word character (\W),
# or the start/end of the string

# Search for "cat" as a standalone word
print(re.findall(r'\bcat\b', text))
# => ['cat']  -- Does not match "concatenate", etc.

# Words starting with "cat"
print(re.findall(r'\bcat\w*', text))
# => ['cat', 'concatenate', 'category', 'caterpillar']

# \B : Non-word boundary (inside a word)
print(re.findall(r'\Bcat\B', text))
# => ['cat']  -- Only the "cat" in the middle of "concatenate"
```

**Precise definition of word boundaries**:

```python
import re

# \b matches a "position" (does not consume characters)
# \b matches at the following 4 positions:
# 1. Start of string, if the first character is \w
# 2. End of string, if the last character is \w
# 3. Position immediately after \w followed by \W
# 4. Position immediately after \W followed by \w

text = "Hello, World! 123"
#       ^     ^^ ^^  ^ ^  ^
#       1     34 34  1 34  2

boundaries = []
for i in range(len(text) + 1):
    if re.search(r'\b', text[max(0,i-1):i+1] if i > 0 else text[0:1]):
        pass  # Simplified

# Checking specific boundary positions
for m in re.finditer(r'\b', text):
    print(f"Position {m.start()}: ...{text[max(0,m.start()-1):m.start()+1]}...")
# Position 0:  ...H...
# Position 5:  ...o,...
# Position 7:  ...W...
# Position 12: ...d!...
# Position 14: ...1...
# Position 17: ...3 (end)
```

**Word boundaries and Japanese**:

```python
import re

# \b is based on ASCII word characters (\w = [a-zA-Z0-9_])
# -> Japanese characters are treated as \W, so each character boundary applies

text = "Hello世界World"
print(re.findall(r'\b\w+\b', text))
# => ['Hello', 'World']
# '世界' is not included in \w, so it is not recognized as a standalone word

# For Unicode-aware word boundaries, use the regex module
import regex
# The regex module's UNICODE flag
print(regex.findall(r'\b\w+\b', text, flags=regex.UNICODE))
# => ['Hello世界World']  (Japanese characters are included in \w)
```

```javascript
// JavaScript \b and Unicode
const text = "Hello世界World";

// Even with the ES2015+ u flag, \b is ASCII-based
console.log(text.match(/\b\w+\b/gu));
// => ['Hello', 'World'] ('世界' not included)

// To use Unicode word boundaries:
// ECMAScript 2024 /v flag + Unicode property
console.log(text.match(/[\p{L}\p{N}]+/gu));
// => ['Hello世界World'] (Unicode character properties as an alternative)
```

### 3.4 Conceptual Diagram of Anchors

```
Text: "Hello World"

Position:  ^  H  e  l  l  o     W  o  r  l  d  $
           ↑                                    ↑
           String start (^, \A)                 String end ($, \Z)

Word boundary (\b) positions:
           \b H  e  l  l  o \b  \b W  o  r  l  d \b
           ↑                ↑   ↑                ↑
           Word start        Word end/start       Word end

Non-word boundary (\B) positions:
              \B \B \B \B      \B \B \B \B
              H--e--l--l--o    W--o--r--l--d
              Between characters (both are \w)
```

### 3.5 Complete Anchor Reference

```
┌────────┬──────────────────────────────────────┐
│ Anchor │ Meaning                               │
├────────┼──────────────────────────────────────┤
│ ^      │ Line start (MULTILINE) / String start│
│ $      │ Line end (MULTILINE) / String end    │
│ \A     │ Absolute start of string              │
│ \Z     │ Absolute end of string                │
│ \z     │ Absolute end of string (ignores       │
│        │ trailing newline)                      │
│ \b     │ Word boundary                         │
│ \B     │ Non-word boundary                     │
│ \G     │ End position of previous match        │
│        │ (Java/Perl/.NET)                      │
└────────┴──────────────────────────────────────┘
```

### 3.6 The `\G` Anchor in Detail

`\G` is a special anchor that anchors to "the position where the previous match ended." It is useful for consecutive repeated matches.

```java
import java.util.regex.*;

public class GAnchorExample {
    public static void main(String[] args) {
        // \G anchors to the end position of the previous match
        // On the first match, it is at the same position as the string start (\A)

        String text = "abc123def456ghi";
        Pattern p = Pattern.compile("\\G\\w");
        Matcher m = p.matcher(text);

        StringBuilder result = new StringBuilder();
        while (m.find()) {
            result.append(m.group());
        }
        System.out.println(result);
        // => "abc123def456ghi" (all characters match consecutively)

        // If the match breaks in the middle, \G does not resume
        text = "abc 123 def";
        p = Pattern.compile("\\G\\w");
        m = p.matcher(text);

        result = new StringBuilder();
        while (m.find()) {
            result.append(m.group());
        }
        System.out.println(result);
        // => "abc" (breaks at the space; nothing matches afterward)
    }
}
```

```perl
# Using \G in Perl: a tokenizer
my $text = "3.14 + 42 * 2.0";
my @tokens;

while ($text =~ /\G\s*/gc) {  # Skip whitespace
    if ($text =~ /\G(\d+\.?\d*)/gc) {
        push @tokens, {type => 'NUMBER', value => $1};
    } elsif ($text =~ /\G([+\-*\/])/gc) {
        push @tokens, {type => 'OP', value => $1};
    } else {
        die "Unexpected character at position " . pos($text);
    }
}

for my $token (@tokens) {
    print "$token->{type}: $token->{value}\n";
}
# NUMBER: 3.14
# OP: +
# NUMBER: 42
# OP: *
# NUMBER: 2.0
```

---

## 4. Combining Quantifiers and Anchors

### 4.1 Matching Entire Lines

```python
import re

log = """2026-02-11 10:00 INFO Server started
2026-02-11 10:05 ERROR Connection failed
2026-02-11 10:10 INFO Request received
2026-02-11 10:15 WARN Memory usage high"""

# Extract entire lines containing ERROR
error_lines = re.findall(r'^.*ERROR.*$', log, re.MULTILINE)
print(error_lines)
# => ['2026-02-11 10:05 ERROR Connection failed']

# Lines containing WARN or ERROR
issues = re.findall(r'^.*(ERROR|WARN).*$', log, re.MULTILINE)
print(issues)
# => ['2026-02-11 10:05 ERROR Connection failed',
#     '2026-02-11 10:15 WARN Memory usage high']
```

### 4.2 Full-Match Validation

```python
import re

# Verify that the entire string matches the pattern exactly
# Use ^...$ (or re.fullmatch)

def validate_date(s):
    """Validate date in YYYY-MM-DD format"""
    return bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', s))

print(validate_date("2026-02-11"))      # => True
print(validate_date("2026-02-11 "))     # => False (trailing space)
print(validate_date("date: 2026-02-11"))# => False (text before the date)

def validate_hex_color(s):
    """Validate #RRGGBB color code format"""
    return bool(re.fullmatch(r'#[0-9a-fA-F]{6}', s))

print(validate_hex_color("#FF5733"))  # => True
print(validate_hex_color("#GG5733"))  # => False
```

**Full-match methods across languages**:

```javascript
// JavaScript: Use ^...$ (no fullmatch method)
function validateDate(s) {
    return /^\d{4}-\d{2}-\d{2}$/.test(s);
}
console.log(validateDate("2026-02-11"));  // true
console.log(validateDate("2026-02-11 ")); // false
```

```java
// Java: matches() method implicitly adds ^...$
String date = "2026-02-11";
System.out.println(date.matches("\\d{4}-\\d{2}-\\d{2}"));  // true

// find() does partial matching
Pattern p = Pattern.compile("\\d{4}-\\d{2}-\\d{2}");
Matcher m = p.matcher("date: 2026-02-11");
System.out.println(m.find());     // true (partial match)
System.out.println(m.matches());  // false (full match)
```

```ruby
# Ruby: \A...\z is recommended (^ $ are line-based)
def validate_date(s)
  s.match?(/\A\d{4}-\d{2}-\d{2}\z/)
end

puts validate_date("2026-02-11")       # => true
puts validate_date("2026-02-11\nfoo")  # => false (\z is end of string)
# Using ^ $:
puts "2026-02-11\nfoo".match?(/^\d{4}-\d{2}-\d{2}$/)  # => true (matches line end!)
```

```go
// Go (RE2): MatchString does partial matching; always use ^...$ for full match
package main

import (
    "fmt"
    "regexp"
)

func main() {
    // MatchString does partial matching
    matched, _ := regexp.MatchString(`\d{4}-\d{2}-\d{2}`, "date: 2026-02-11")
    fmt.Println(matched)  // true

    // Use ^...$ explicitly for full match
    matched, _ = regexp.MatchString(`^\d{4}-\d{2}-\d{2}$`, "2026-02-11")
    fmt.Println(matched)  // true
    matched, _ = regexp.MatchString(`^\d{4}-\d{2}-\d{2}$`, "date: 2026-02-11")
    fmt.Println(matched)  // false
}
```

### 4.3 Composite Validation Patterns

```python
import re

# Password strength check: combining multiple conditions with lookaheads
def validate_password(pw):
    """
    Requirements:
    - 8 to 20 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 digit
    - At least 1 special character
    """
    pattern = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*]).{8,20}$'
    return bool(re.match(pattern, pw))

print(validate_password("Abc123!x"))    # True
print(validate_password("abc123!x"))    # False (no uppercase)
print(validate_password("Abc!x"))       # False (fewer than 8 chars)
print(validate_password("Abcdefgh"))    # False (no digits/special chars)

# Validating usernames that may contain Japanese
def validate_username(name):
    """
    Requirements:
    - 2 to 20 characters
    - Only Japanese (Hiragana, Katakana, Kanji), alphanumeric, and underscores
    - Must start with a letter or Japanese character
    """
    pattern = r'^[a-zA-Z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF][\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]{1,19}$'
    return bool(re.match(pattern, name))

print(validate_username("田中太郎"))     # True
print(validate_username("user_123"))     # True
print(validate_username("1abc"))         # False (starts with digit)
print(validate_username("a"))            # False (fewer than 2 chars)
```

### 4.4 Text Replacement Using Boundaries

```python
import re

# Word-level replacement (preventing partial matches)
text = "The cat concatenated the catalog"

# NG: Replaces partial matches
bad = re.sub(r'cat', 'dog', text)
print(bad)  # => 'The dog dogdogenated the dogalog'

# OK: Use word boundaries
good = re.sub(r'\bcat\b', 'dog', text)
print(good)  # => 'The dog concatenated the catalog'

# Variable renaming (useful in programming)
code = """
count = 0
count += 1
account = get_account()
print(count)
"""

# Rename variable count to total
renamed = re.sub(r'\bcount\b', 'total', code)
print(renamed)
# count -> total is changed, but account is left unchanged
```

```javascript
// Text replacement using word boundaries in JavaScript
const code = `
function getData() {
    const data = fetchData();
    const dataMap = new Map();
    return processData(data);
}`;

// Rename data to info (leave getData, fetchData, processData unchanged)
const renamed = code.replace(/\bdata\b/g, "info");
console.log(renamed);
// data -> info is changed; getData/fetchData/processData/dataMap are unchanged
```

### 4.5 Formatting Using Line Start/End

```python
import re

# Removing comment lines
config = """# Database settings
host = localhost
# port = 5432
port = 3306
# Comment
user = admin
"""

# Remove lines starting with #
cleaned = re.sub(r'^#.*$\n?', '', config, flags=re.MULTILINE)
print(cleaned)
# host = localhost
# port = 3306
# user = admin

# Adding indentation to each line
text = """line 1
line 2
line 3"""

indented = re.sub(r'^', '    ', text, flags=re.MULTILINE)
print(indented)
#     line 1
#     line 2
#     line 3

# Removing trailing whitespace from each line
text_with_trailing = "hello   \nworld  \nfoo \n"
trimmed = re.sub(r' +$', '', text_with_trailing, flags=re.MULTILINE)
print(repr(trimmed))
# 'hello\nworld\nfoo\n'

# Removing blank lines
text_with_blanks = """first

second


third"""
no_blanks = re.sub(r'^\s*$\n', '', text_with_blanks, flags=re.MULTILINE)
print(no_blanks)
# first
# second
# third
```

---

## 5. Cross-Language Differences in Quantifiers and Anchors

### 5.1 Quantifier Support Status

```
┌──────────────┬───────┬───────┬──────────┬────────────┐
│ Feature      │ Python│ JS    │ Java     │ Ruby/Perl  │
├──────────────┼───────┼───────┼──────────┼────────────┤
│ Greedy       │ ○     │ ○     │ ○        │ ○          │
│ *+?{n,m}     │       │       │          │            │
│ Lazy *? +?   │ ○     │ ○     │ ○        │ ○          │
│ Possessive   │ ✗(re) │ ✗     │ ○        │ ✗(*1)      │
│ *+ ++        │       │       │          │            │
│ Atomic (?>)  │ ✗(re) │ ✗(*2) │ ○(*3)    │ ○          │
│ {,m} short   │ ○     │ ✗     │ ✗        │ ○          │
└──────────────┴───────┴───────┴──────────┴────────────┘

*1: Partially supported in Ruby 3.0+ (regex literals only)
*2: Under ECMAScript 2025 proposal
*3: Java supports its own (?>...) syntax
```

### 5.2 Anchor Differences Across Languages

```python
# Python
import re
text = "line1\nline2\n"

# ^ $ default to string start/end
# re.MULTILINE changes them to each line's start/end
print(re.findall(r'^\w+', text, re.M))  # => ['line1', 'line2']

# \A \Z are not affected by MULTILINE
print(re.findall(r'\A\w+', text, re.M))  # => ['line1']
```

```javascript
// JavaScript
const text = "line1\nline2\n";

// ^ $ default to string start/end
// m flag changes them to each line's start/end
console.log(text.match(/^\w+/gm));  // => ['line1', 'line2']

// JavaScript does NOT have \A or \Z!
// Use ^ $ without the m flag instead,
// or use lookahead/lookbehind as alternatives
console.log(text.match(/^\w+/g));  // => ['line1']  (no m = string start)
```

```ruby
# Ruby
text = "line1\nline2\n"

# Ruby's ^ $ always work line-by-line (equivalent to MULTILINE)
puts text.scan(/^\w+/)   # => ["line1", "line2"]

# Use \A and \z for string start/end
puts text.scan(/\A\w+/)  # => ["line1"]

# Ruby has both \Z (matches before trailing \n) and \z (absolute end)
puts "hello\n".match?(/hello\Z/)  # => true
puts "hello\n".match?(/hello\z/)  # => false
```

```go
// Go (RE2 engine)
package main

import (
    "fmt"
    "regexp"
)

func main() {
    text := "line1\nline2\n"

    // Go default: ^ $ are string start/end
    re1 := regexp.MustCompile(`^\w+`)
    fmt.Println(re1.FindAllString(text, -1))  // => [line1]

    // (?m) flag for MULTILINE mode
    re2 := regexp.MustCompile(`(?m)^\w+`)
    fmt.Println(re2.FindAllString(text, -1))  // => [line1 line2]

    // Go does NOT have \A, \Z, or \z
    // Use ^ $ with appropriate flag control
}
```

### 5.3 JavaScript-Specific Notes

```javascript
// The dotAll flag (s) introduced in ES2018
const text = "line1\nline2\nline3";

// Without s flag: . does not match newlines
console.log(text.match(/^.+$/));       // => ['line1']
console.log(text.match(/^.+$/m));      // => ['line1'] (first line)
console.log(text.match(/^.+$/gm));     // => ['line1', 'line2', 'line3']

// With s flag: . also matches newlines
console.log(text.match(/^.+$/s));      // => ['line1\nline2\nline3']
console.log(text.match(/^.+$/ms));     // => ['line1\nline2\nline3']
// sm combination: ^ matches line start, but .+ includes newlines so the whole string matches

// The v flag in ES2024 (Unicode Sets)
// The v flag is a superset of the u flag
const emoji = "Hello 🌍 World 🎉";
console.log(emoji.match(/\p{Emoji}/gv));  // => ['🌍', '🎉']
```

---

## 6. Anti-Patterns

### 6.1 Anti-Pattern: Uncontrolled Use of `.*`

```python
import re

# NG: Greedy .* matches an unexpectedly wide range
html = '<span class="a">hello</span><span class="b">world</span>'
bad = re.search(r'<span.*>.*</span>', html)
print(bad.group())
# => '<span class="a">hello</span><span class="b">world</span>'
# The entire string matches!

# OK: Use negated character classes or lazy quantifiers
# Method 1: Lazy quantifier
good1 = re.findall(r'<span.*?>.*?</span>', html)
print(good1)  # => ['<span class="a">hello</span>', ...]

# Method 2: Negated character class (faster)
good2 = re.findall(r'<span[^>]*>[^<]*</span>', html)
print(good2)  # => ['<span class="a">hello</span>', ...]
```

### 6.2 Anti-Pattern: Forgetting the MULTILINE Flag with `^` and `$`

```python
import re

text = """user: admin
user: guest
user: root"""

# NG: Expecting to match each line start without MULTILINE
bad = re.findall(r'^user: (\w+)', text)
print(bad)  # => ['admin']  -- first line only

# OK: Add the MULTILINE flag
good = re.findall(r'^user: (\w+)', text, re.MULTILINE)
print(good)  # => ['admin', 'guest', 'root']
```

### 6.3 Anti-Pattern: ReDoS from Nested Quantifiers

```python
import re
import time

# Dangerous: Nested quantifiers cause exponential backtracking
dangerous_patterns = [
    r'(a+)+b',          # Nested +
    r'(a*)*b',          # Nested *
    r'(a|a)*b',         # Overlapping alternatives + quantifier
    r'(.*a){10}',       # Mass repetition of .*
    r'(\w+\s*)+$',      # Common input validation pattern
]

safe_input = "a" * 20 + "b"      # Matches -> fast
evil_input = "a" * 25             # No match -> exponentially slow

for pattern in dangerous_patterns:
    # Safe input
    start = time.perf_counter()
    re.search(pattern, safe_input)
    safe_time = time.perf_counter() - start

    # Malicious input (with timeout)
    # Warning: actual execution can be extremely slow
    print(f"Pattern: {pattern}")
    print(f"  Safe input: {safe_time:.6f} sec")
    # evil_input test omitted (ReDoS risk)

# Safe alternative patterns
safe_alternatives = {
    r'(a+)+b':     r'a+b',           # Remove nesting
    r'(a*)*b':     r'a*b',           # Remove nesting
    r'(a|a)*b':    r'a*b',           # Remove duplication
    r'(\w+\s*)+$': r'[\w\s]+$',      # Consolidate into character class
}
```

### 6.4 Anti-Pattern: Unnecessary Quantifiers

```python
import re

# NG: Unnecessarily complex quantifiers
bad_patterns = {
    r'\d{1,}':    r'\d+',       # {1,} is the same as +
    r'\d{0,}':    r'\d*',       # {0,} is the same as *
    r'\d{0,1}':   r'\d?',       # {0,1} is the same as ?
    r'[a-z]{1}':  r'[a-z]',     # {1} is unnecessary
    r'(?:ab){1}': r'ab',        # Group with {1} is also unnecessary
}

for bad, good in bad_patterns.items():
    print(f"NG: {bad:20s} -> OK: {good}")

# NG: Unnecessary quantifiers inside groups
# When (?:  )+ has a quantifier inside, check if both inner and outer are needed
text = "hello   world   foo"

# Redundant: group is unnecessary
print(re.split(r'(?:\s)+', text))    # => ['hello', 'world', 'foo']
# Concise:
print(re.split(r'\s+', text))        # => ['hello', 'world', 'foo']
```

### 6.5 Anti-Pattern: Misusing `\b`

```python
import re

# NG: Incorrect assumptions about word boundaries with digit-only sequences
text = "item123"
print(re.findall(r'\b\d+\b', text))
# => [] (empty!) -- The \b before '123' is between 'm' and '1', but
#    there is no non-word character before 123 (\b is the \w->\W transition point)

# Correct understanding:
# In 'item123':
# - \b before 'i' (start of string)
# - \b after '3' (end of string)
# - Between 'm' and '1' is \w->\w so it is NOT \b!
# Therefore \b\d+\b does not find '123' as a standalone word

# OK: Use the correct pattern for the purpose
# 1. Search for standalone numbers (preceded/followed by non-word chars)
print(re.findall(r'(?<!\w)\d+(?!\w)', text))  # => []
# 2. Extract digit portions from a string
print(re.findall(r'\d+', text))  # => ['123']
# 3. Standalone numbers only (whitespace-separated)
text2 = "item123 456 foo"
print(re.findall(r'\b\d+\b', text2))  # => ['456']
```

---

## 7. Practical Pattern Collection

### 7.1 Log Analysis Patterns

```python
import re

log_entries = """
2026-02-11T10:30:45.123Z INFO  [main] Application starting
2026-02-11T10:30:46.456Z DEBUG [db] Connection pool initialized (size=10)
2026-02-11T10:30:47.789Z ERROR [api] Request timeout after 30000ms
2026-02-11T10:30:48.012Z WARN  [mem] Memory usage: 85% (threshold: 80%)
2026-02-11T10:30:49.345Z ERROR [api] Internal server error: NullPointerException
2026-02-11T10:30:50.678Z INFO  [main] Shutdown initiated
"""

# 1. Decompose timestamp, level, component, and message
log_pattern = r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)\s+(INFO|DEBUG|WARN|ERROR)\s+\[(\w+)\]\s+(.+)$'

for match in re.finditer(log_pattern, log_entries, re.MULTILINE):
    ts, level, component, message = match.groups()
    print(f"[{level:5s}] {component:6s} | {message}")

# 2. Extract only ERROR level (entire lines)
errors = re.findall(r'^.*ERROR.*$', log_entries, re.MULTILINE)
for err in errors:
    print(f"ERROR: {err.strip()}")

# 3. Extract numerical values (metrics analysis)
metrics = re.findall(r'(\w+)[=:]\s*(\d+(?:\.\d+)?)', log_entries)
for key, value in metrics:
    print(f"  {key} = {value}")
# size = 10, threshold = 80, etc.

# 4. Time range filter
def filter_time_range(logs, start_time, end_time):
    """Filter logs within a specified time range"""
    pattern = rf'^({re.escape(start_time)}.*?{re.escape(end_time)}.*?)$'
    # More accurate approach: compare each line's timestamp
    result = []
    for line in logs.strip().split('\n'):
        m = re.match(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
        if m:
            ts = m.group(1)
            if start_time <= ts <= end_time:
                result.append(line)
    return result

filtered = filter_time_range(log_entries, "2026-02-11T10:30:47", "2026-02-11T10:30:49")
for line in filtered:
    print(line.strip())
```

### 7.2 Data Cleansing Patterns

```python
import re

# 1. Phone number normalization
def normalize_phone(phone):
    """Convert various phone number formats to a unified format"""
    # Remove non-digits
    digits = re.sub(r'\D', '', phone)

    # Japanese phone number patterns
    # Mobile: 090-XXXX-XXXX / 080-XXXX-XXXX / 070-XXXX-XXXX
    if re.match(r'^0[789]0\d{8}$', digits):
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    # Landline (Tokyo): 03-XXXX-XXXX
    elif re.match(r'^0[3-9]\d{8}$', digits):
        return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}"
    # Landline (other): 0XXX-XX-XXXX
    elif re.match(r'^0\d{9}$', digits):
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return phone  # Unable to convert

tests = [
    "090-1234-5678", "09012345678", "090 1234 5678",
    "03-1234-5678", "0312345678", "(03) 1234-5678"
]
for t in tests:
    print(f"{t:20s} -> {normalize_phone(t)}")

# 2. Currency normalization
def normalize_currency(text):
    """Convert currency notation to a unified format"""
    # Remove comma separators
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # Full-width digits -> half-width
    zen_to_han = str.maketrans('０１２３４５６７８９', '0123456789')
    text = text.translate(zen_to_han)
    # Handle "man" (10,000) and "oku" (100,000,000) units
    text = re.sub(r'(\d+)万(\d*)', lambda m: str(int(m.group(1)) * 10000 + int(m.group(2) or 0)), text)
    text = re.sub(r'(\d+)億', lambda m: str(int(m.group(1)) * 100000000), text)
    return text

print(normalize_currency("1,234,567円"))  # => 1234567円
print(normalize_currency("１２３４円"))     # => 1234円
print(normalize_currency("5万3000"))       # => 53000

# 3. Address normalization
def normalize_address(addr):
    """Standardize notation variations in Japanese addresses"""
    # Full-width digits -> half-width
    zen_to_han = str.maketrans('０１２３４５６７８９', '0123456789')
    addr = addr.translate(zen_to_han)
    # Standardize "chome" "banchi" "go" notation
    addr = re.sub(r'(\d+)丁目(\d+)番地?(\d+)号?', r'\1-\2-\3', addr)
    addr = re.sub(r'(\d+)丁目(\d+)番地?', r'\1-\2', addr)
    addr = re.sub(r'(\d+)丁目', r'\1', addr)
    return addr

print(normalize_address("東京都港区赤坂1丁目2番地3号"))
# => 東京都港区赤坂1-2-3
print(normalize_address("東京都港区赤坂１丁目２番地３号"))
# => 東京都港区赤坂1-2-3
```

### 7.3 Programming Language Token Analysis

```python
import re

def tokenize(source_code):
    """Simple tokenizer: parses Python-like expressions"""
    token_spec = [
        ('NUMBER',    r'\d+\.?\d*(?:[eE][+-]?\d+)?'),  # Integer, decimal, exponent
        ('STRING',    r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''),  # Strings
        ('IDENT',     r'[a-zA-Z_]\w*'),                 # Identifiers
        ('OP',        r'[+\-*/=<>!]=?|[(){}[\],;.]'),   # Operators
        ('NEWLINE',   r'\n'),                            # Newlines
        ('SKIP',      r'[ \t]+'),                        # Whitespace (skip)
        ('COMMENT',   r'#.*'),                           # Comments
        ('MISMATCH',  r'.'),                             # Unknown characters
    ]

    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)
    tokens = []

    for m in re.finditer(tok_regex, source_code):
        kind = m.lastgroup
        value = m.group()
        if kind == 'SKIP' or kind == 'COMMENT':
            continue
        elif kind == 'MISMATCH':
            raise SyntaxError(f"Unexpected character: {value!r}")
        tokens.append((kind, value))

    return tokens

# Test
code = 'x = 3.14 + y * 2  # calculation'
for tok in tokenize(code):
    print(f"  {tok[0]:10s}: {tok[1]}")
# NUMBER    : 3.14
# OP        : +
# IDENT     : y
# OP        : *
# NUMBER    : 2
# etc.
```

### 7.4 URL / URI Parsing

```python
import re

def parse_url(url):
    """URI parsing based on RFC 3986"""
    pattern = r'''
        ^
        (?:(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*):)?  # Scheme
        (?://
            (?:(?P<userinfo>[^@]*)@)?               # User info
            (?P<host>[^/:?#]*)                      # Host
            (?::(?P<port>\d+))?                     # Port
        )?
        (?P<path>[^?#]*)                            # Path
        (?:\?(?P<query>[^#]*))?                     # Query
        (?:\#(?P<fragment>.*))?                      # Fragment
        $
    '''

    m = re.match(pattern, url, re.VERBOSE)
    if not m:
        return None

    return {k: v for k, v in m.groupdict().items() if v is not None}

# Test
urls = [
    "https://user:pass@example.com:8080/path/to/page?q=hello&lang=ja#section",
    "ftp://files.example.com/pub/docs/readme.txt",
    "/api/v2/users?page=1&limit=20",
    "mailto:user@example.com",
]

for url in urls:
    print(f"\nURL: {url}")
    parts = parse_url(url)
    for k, v in parts.items():
        print(f"  {k:12s}: {v}")
```

### 7.5 Parsing Markdown Inline Elements

```python
import re

def parse_markdown_inline(text):
    """Parse Markdown inline elements"""
    patterns = [
        # Bold (**bold** / __bold__)
        (r'\*\*(.+?)\*\*|__(.+?)__', 'bold'),
        # Italic (*italic* / _italic_)
        (r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', 'italic'),
        # Inline code (`code`)
        (r'`([^`]+)`', 'code'),
        # Link ([text](url))
        (r'\[([^\]]+)\]\(([^)]+)\)', 'link'),
        # Image (![alt](url))
        (r'!\[([^\]]*)\]\(([^)]+)\)', 'image'),
        # Strikethrough (~~text~~)
        (r'~~(.+?)~~', 'strikethrough'),
    ]

    elements = []
    for pattern, elem_type in patterns:
        for m in re.finditer(pattern, text):
            elements.append({
                'type': elem_type,
                'match': m.group(),
                'start': m.start(),
                'end': m.end(),
            })

    # Sort by position
    elements.sort(key=lambda x: x['start'])
    return elements

md_text = "This is **bold** and *italic* with `code` and [link](https://example.com)"
for elem in parse_markdown_inline(md_text):
    print(f"  {elem['type']:15s}: {elem['match']}")
```

---

## 8. FAQ

### Q1: Which is faster, `.*` or `[^X]*`?

**A**: In general, **`[^X]*` is faster**. `.*` is prone to backtracking, while `[^X]*` explicitly specifies the stop character, avoiding unnecessary backtracking:

```python
import re

# Slow: .* causes heavy backtracking
slow = r'<tag>.*</tag>'

# Fast: [^<]* matches only non-< characters
fast = r'<tag>[^<]*</tag>'

# Even faster: possessive quantifier (Java, etc.)
# fastest = '<tag>[^<]*+</tag>'
```

### Q2: Is there a difference between `{0,}` and `*`?

**A**: The meaning is completely identical. `*` is syntactic sugar for `{0,}`. They are processed identically inside the engine. Prefer `*` `+` `?` for readability, and use `{n,m}` when specific repetition counts are needed.

### Q3: Is it always safe to use lazy quantifiers?

**A**: **No**. Lazy quantifiers only change the direction of backtracking; they do not eliminate backtracking itself. Certain patterns can still cause ReDoS even with lazy quantifiers:

```python
# Example of a pattern that is slow even when lazy:
# Pattern: (a+?)+b
# Text:    "aaaaaaaaaaaaaaaaac"
# -> Exponential backtracking occurs even with lazy matching

# What IS safe:
# 1. Use negated character classes [^X]*
# 2. Use a DFA engine (RE2, etc.)
# 3. Use possessive quantifiers/atomic groups
```

### Q4: Does `\b` work with Japanese text?

**A**: The standard `\b` is an **ASCII word character boundary** and often does not work as expected with Japanese text:

```python
import re

text = "東京タワーは333mです"

# \b is based on ASCII's \w ([a-zA-Z0-9_])
# Japanese characters are treated as \W -> every character boundary applies
print(re.findall(r'\b.+?\b', text))
# => ['東', '京', 'タ', 'ワ', 'ー', 'は', '333', 'm', 'で', 'す']

# To handle Japanese "words":
# 1. Use morphological analysis like MeCab (recommended)
# 2. Use Unicode word boundaries (ICU)
# 3. Use \b{w} from the regex module
import regex
print(regex.findall(r'\b{w}.+?\b{w}', text))
# Segmentation by Unicode word boundaries
```

### Q5: What is the difference between `re.fullmatch` and `^...$`?

**A**: They are equivalent in most cases, but differ in interaction with the MULTILINE flag:

```python
import re

text = "hello\nworld"

# re.fullmatch always checks the entire string
print(re.fullmatch(r'\w+', text))  # => None (newline present)

# ^...$ behavior changes with MULTILINE
print(re.match(r'^\w+$', text, re.M))  # => <Match: 'hello'> (matches first line)
print(re.fullmatch(r'\w+', "hello"))    # => <Match: 'hello'>

# fullmatch is not affected by MULTILINE (always checks the entire string)
print(re.fullmatch(r'\w+', text, re.M))  # => None
```

### Q6: What are alternatives when possessive quantifiers are unavailable?

**A**: Use atomic groups or in-code countermeasures:

```python
import re

# Python's re module does not support possessive quantifiers
# Alternatives:

# 1. Use the regex module (recommended)
# import regex
# regex.search(r'a++b', text)

# 2. Use negated character classes to avoid backtracking
# NG: .*  (massive backtracking)
# OK: [^<]*  (explicitly specify stop character)

# 3. Set a timeout
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Regex timeout")

def safe_search(pattern, text, timeout_sec=1):
    """Regex search with timeout"""
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        return re.search(pattern, text)
    except TimeoutError:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# 4. Limit input length
def safe_match(pattern, text, max_len=10000):
    """Regex match with input length restriction"""
    if len(text) > max_len:
        raise ValueError(f"Input too long: {len(text)} > {max_len}")
    return re.search(pattern, text)
```

### Q7: Is there a hard limit on quantifier upper bounds?

**A**: It varies by engine:

```
┌──────────────┬──────────────────────────────┐
│ Engine       │ Upper limit for {n,m}        │
├──────────────┼──────────────────────────────┤
│ Python (re)  │ 2^31 - 1 (2147483647)        │
│ JavaScript   │ 2^32 - 1 (browser-dependent) │
│ Java         │ 2^31 - 1                     │
│ PCRE         │ 65535                         │
│ Ruby         │ No limit (memory-dependent)   │
│ Go (RE2)     │ 1000 (default)               │
│ Rust (regex) │ No limit (memory-dependent)   │
└──────────────┴──────────────────────────────┘

# In practice, overly large {n,m} values increase
# compile time and memory usage, so ~100 is a
# reasonable practical upper bound
```

### Q8: How can I use the `\G` anchor in Python?

**A**: Python's `re` module does not have `\G`, but you can use the `regex` module or `re.scanner` as alternatives:

```python
# regex module
import regex

text = "abc123def456"
tokens = []
pos = 0

# Consecutive matching using \G
for m in regex.finditer(r'\G(?:(\w+)|(\d+))', text):
    tokens.append(m.group())

# Alternative with the re module: Scanner
import re
scanner = re.Scanner([
    (r'[a-z]+', lambda s, t: ('WORD', t)),
    (r'\d+',    lambda s, t: ('NUM', t)),
    (r'\s+',    None),  # Skip
])

tokens, remainder = scanner.scan("abc 123 def 456")
for tok in tokens:
    print(tok)
# ('WORD', 'abc')
# ('NUM', '123')
# ('WORD', 'def')
# ('NUM', '456')
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Building practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts covered in this guide before moving to the next step.

### Q3: How is this applied in real-world work?

Knowledge of this topic is frequently applied in everyday development work. It is especially important during code reviews and architecture design.

---

## Summary

| Item | Description |
|------|-------------|
| `*` | 0 or more (greedy) |
| `+` | 1 or more (greedy) |
| `?` | 0 or 1 (greedy) |
| `{n,m}` | n to m times |
| `*?` `+?` `??` | Lazy (shortest match) versions |
| `*+` `++` `?+` | Possessive (no backtracking) versions |
| `(?>...)` | Atomic group |
| `^` | Line start / string start |
| `$` | Line end / string end |
| `\b` | Word boundary |
| `\B` | Non-word boundary |
| `\A` / `\Z` | Absolute string start/end |
| `\z` | Absolute string end (some languages) |
| `\G` | End position of previous match |
| Greedy vs. Lazy | Greedy is longest, lazy is shortest |
| Performance | `[^X]*` > `.*?` > `.*` (generally) |
| ReDoS prevention | Avoid nested quantifiers; use negated classes or possessive quantifiers |

## Recommended Next Guides

- [../01-advanced/00-groups-backreferences.md](../01-advanced/00-groups-backreferences.md) -- Groups and Backreferences
- [../01-advanced/01-lookaround.md](../01-advanced/01-lookaround.md) -- Lookahead and Lookbehind
- [../01-advanced/03-performance.md](../01-advanced/03-performance.md) -- Performance Optimization and ReDoS Prevention

## References

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- Chapter 4 "The Mechanics of Quantifiers" and Chapter 6 "Backtracking" are essential reading
2. **Russ Cox** "Regular Expression Matching: the Virtual Machine Approach" https://swtch.com/~rsc/regexp/regexp2.html, 2009 -- Theoretical analysis of backtracking
3. **Jan Goyvaerts** "Regular-Expressions.info" https://www.regular-expressions.info/repeat.html -- Practical explanation of quantifiers
4. **OWASP** "Regular expression Denial of Service - ReDoS" https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS -- ReDoS attacks and defenses
5. **Python Documentation** "re --- Regular expression operations" https://docs.python.org/3/library/re.html -- Official Python reference
