# Basic Syntax -- Literals, Metacharacters, Escaping

> A comprehensive guide to the operating principles and correct usage of literal characters, metacharacters (special characters), and escape sequences -- the most fundamental building blocks of regular expressions.

## What You Will Learn in This Chapter

1. **Distinguishing Literal Characters from Metacharacters** -- Which characters match directly and which have special meaning
2. **How Escaping Works and Its Pitfalls** -- Disabling metacharacters with backslash and the double-escaping problem
3. **Behavior Changes via Flags (Modifiers)** -- Case-insensitive, multiline mode, dotall mode
4. **Literal Notation Across Languages** -- Differences in how Python, JavaScript, Java, Perl, and Ruby express patterns
5. **Internal Workings of Matching** -- How the engine scans a string


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Regular Expression Overview](./00-regex-overview.md)

---

## 1. Literal Characters

Literal characters match the corresponding character in the pattern directly.

```python
import re

# Pattern consisting only of literal characters
pattern = r'hello'
text = "say hello to the world"

match = re.search(pattern, text)
print(match.group())  # => "hello"
print(match.start())  # => 4
print(match.end())    # => 9
```

Rules for literal character matching:

```
Pattern    Target String        Result
────────   ──────────────────   ──────
cat        "the cat sat"        Match ("cat")
123        "abc123def"          Match ("123")
hello      "Hello World"        No match (case-sensitive)
hello      "Hello World"        Match (with i flag)
```

### 1.1 Detailed Behavior of Literal Matching

```python
import re

# Literal matching succeeds at the first position found, scanning left to right
text = "abcabcabc"
pattern = r'abc'

# search returns the first match
m = re.search(pattern, text)
print(f"First match: position {m.start()}-{m.end()}")  # => position 0-3

# findall returns all matches
all_matches = re.findall(pattern, text)
print(f"All matches: {all_matches}")  # => ['abc', 'abc', 'abc']

# finditer returns an iterator with position information
for m in re.finditer(pattern, text):
    print(f"  position {m.start()}-{m.end()}: '{m.group()}'")
# => position 0-3: 'abc'
# => position 3-6: 'abc'
# => position 6-9: 'abc'
```

### 1.2 Case Handling

```python
import re

text = "Python is Great. PYTHON IS GREAT. python is great."

# Default: case-sensitive
print(re.findall(r'python', text))
# => ['python']

# IGNORECASE flag: case-insensitive
print(re.findall(r'python', text, re.IGNORECASE))
# => ['Python', 'PYTHON', 'python']

# Inline flag: embed the flag within the pattern
print(re.findall(r'(?i)python', text))
# => ['Python', 'PYTHON', 'python']

# Partial flag application (Python 3.6+)
# (?i:pattern) applies case-insensitivity only to that portion
print(re.findall(r'(?i:python) is (?i:great)', text))
# => ['Python is Great', 'PYTHON IS GREAT', 'python is great']
```

### 1.3 Multibyte Character Literal Matching

```python
import re

# Literal matching with Japanese characters
text = "Tokyo is the capital of Japan. Tokyo is the capital of Japan."

# Japanese strings can be matched directly
print(re.findall(r'Tokyo', text))   # => ['Tokyo', 'Tokyo']

# Matching in mixed text
log = "2026-02-15 Error: File not found (error: file not found)"
m = re.search(r'Error', log)
print(m.group())  # => 'Error'

# Emoji can also be matched literally (Python 3)
emoji_text = "Hello! Nice to meet you!"
print(re.findall(r'Nice', emoji_text))  # => ['Nice']
```

### 1.4 Differences in Literal Notation Across Languages

```python
# Python: raw strings recommended
import re
pattern = r'hello\.\*world'
re.search(pattern, text)

# Python: pre-compile with re.compile
compiled = re.compile(r'hello\.\*world')
compiled.search(text)
```

```javascript
// JavaScript: literal notation
const pattern1 = /hello\.\*world/;
pattern1.test(text);

// JavaScript: constructor notation (for dynamic patterns)
const pattern2 = new RegExp('hello\\.\\*world');
pattern2.test(text);
// Note: constructor requires string escaping too, resulting in double escaping
```

```java
// Java: always string literals (no raw strings)
import java.util.regex.*;
Pattern pattern = Pattern.compile("hello\\.\\*world");
Matcher matcher = pattern.matcher(text);

// Java 13+: text blocks make it slightly more readable
// However, backslash escaping is still required
```

```ruby
# Ruby: Regexp literal
pattern = /hello\.\*world/
text =~ pattern

# Ruby: Regexp.new (for dynamic patterns)
pattern = Regexp.new('hello\.\*world')

# Ruby: %r{} notation (convenient for patterns with many slashes)
pattern = %r{http://example\.com/path}
```

```perl
# Perl: pattern match operator
if ($text =~ /hello\.\*world/) {
    print "Match\n";
}

# Perl: pre-compile with qr//
my $pattern = qr/hello\.\*world/;
if ($text =~ $pattern) { ... }
```

---

## 2. Metacharacter Reference

Characters that have special meaning in regular expressions:

```
Metacharacter Reference (12 characters + backslash):

.   Any single character (except newline)
^   Start of line / negation inside character class
$   End of line
*   Repeat the preceding element 0 or more times
+   Repeat the preceding element 1 or more times
?   The preceding element 0 or 1 time
|   Alternation (OR)
()  Grouping and capture
[]  Character class
{}  Quantifier {n,m}
\   Escape character

Metacharacters inside character class []:
]   End of character class
\   Escape
^   Negation (only at the beginning)
-   Range (only between characters)
```

### 2.1 Dot `.` -- Any Single Character

```python
import re

pattern = r'c.t'
texts = ["cat", "cot", "cut", "ct", "coat", "c\nt"]

for t in texts:
    m = re.search(pattern, t)
    result = m.group() if m else "no match"
    print(f"  '{t}' → {result}")

# Output:
#   'cat' → cat
#   'cot' → cot
#   'cut' → cut
#   'ct'  → no match     (dot requires exactly 1 character)
#   'coat' → coa is no match, does not match c.t
#   'c\nt' → no match    (dot does not match newline *except with DOTALL)

# DOTALL flag makes dot match newlines too
m = re.search(r'c.t', "c\nt", re.DOTALL)
print(m.group())  # => "c\nt"
```

```python
# Practical dot patterns

import re

# 1. Pattern including any single character
print(re.findall(r'b.g', "bag big bog bug"))
# => ['bag', 'big', 'bog', 'bug']

# 2. Fixed-length pattern matching
print(re.findall(r'...-....', "Tel: 03-1234-5678"))
# => ['03-1234']  *May not match intent

# 3. Correct use of dot: when a specific single character is unknown
# Filename pattern: any single character before the extension
print(re.findall(r'file.\.txt', "file1.txt file2.txt fileA.txt"))
# => ['file1.txt', 'file2.txt', 'fileA.txt']
```

### 2.2 Pipe `|` -- Alternation (OR)

```python
import re

# Alternation with pipe
pattern = r'cat|dog|bird'
texts = ["I have a cat", "I have a dog", "I have a fish"]

for t in texts:
    m = re.search(pattern, t)
    print(f"  '{t}' → {m.group() if m else 'no match'}")

# Output:
#   'I have a cat' → cat
#   'I have a dog' → dog
#   'I have a fish' → no match

# Note: Pipe precedence
# gr(a|e)y  → "gray" or "grey"     (alternation within group)
# gray|grey → "gray" or "grey"     (equivalent)
# gra|ey    → "gra" or "ey"        (may not match intent)
```

```python
# Understanding pipe precedence

import re

# Pipe is the lowest-precedence operator in regex
# Concatenation (adjacent characters) has higher precedence

# Example 1: abc|def is the same as (abc)|(def)
print(re.findall(r'abc|def', "abc def abdef"))
# => ['abc', 'def']

# Example 2: Restrict scope with groups
print(re.findall(r'gr(a|e)y', "gray grey graey"))
# => ['a', 'e']  *Returns the captured group content

# Non-capturing group returns the full match
print(re.findall(r'gr(?:a|e)y', "gray grey graey"))
# => ['gray', 'grey']

# Example 3: Pattern with multiple alternatives
log_pattern = r'ERROR|WARN|INFO|DEBUG'
log = "2026-02-15 [ERROR] Connection failed"
m = re.search(log_pattern, log)
print(m.group())  # => 'ERROR'

# Example 4: NFA engines try alternatives left to right
# The first matching alternative is selected
print(re.search(r'Java|JavaScript', "JavaScript").group())
# => 'Java' (tried first and matched)

# Place longer alternatives first to handle this
print(re.search(r'JavaScript|Java', "JavaScript").group())
# => 'JavaScript'
```

### 2.3 Asterisk `*`, Plus `+`, Question Mark `?`

```python
import re

# * : 0 or more times
print(re.findall(r'ab*c', "ac abc abbc"))     # => ['ac', 'abc', 'abbc']

# + : 1 or more times
print(re.findall(r'ab+c', "ac abc abbc"))     # => ['abc', 'abbc']

# ? : 0 or 1 time
print(re.findall(r'colou?r', "color colour"))  # => ['color', 'colour']
```

```python
# Detailed behavior of quantifiers

import re

# * (0 or more) -- beware of empty matches
print(re.findall(r'a*', "aaa"))
# => ['aaa', '']  *Empty match occurs at the end

print(re.findall(r'a*', "bbb"))
# => ['', '', '', '']  *0-length match at each position

# + (1 or more) -- no empty matches
print(re.findall(r'a+', "aaa"))
# => ['aaa']

print(re.findall(r'a+', "bbb"))
# => []

# ? (0 or 1) -- optional elements
print(re.findall(r'https?', "http and https"))
# => ['http', 'https']

# Greedy matching of quantifiers
# By default, they consume as many characters as possible
print(re.search(r'a+', "aaaaaa").group())
# => 'aaaaaa' (consumes all 'a's)

# Non-greedy (lazy) matching -- append ?
print(re.search(r'a+?', "aaaaaa").group())
# => 'a' (minimum of 1 character only)

# Practical example of non-greedy: text inside HTML tags
html = "<b>bold</b> and <i>italic</i>"
print(re.findall(r'<.+>', html))    # Greedy: ['<b>bold</b> and <i>italic</i>']
print(re.findall(r'<.+?>', html))   # Non-greedy: ['<b>', '</b>', '<i>', '</i>']
```

### 2.4 Parentheses `()` -- Grouping and Capture

```python
import re

# Basic grouping
pattern = r'(hello) (world)'
m = re.search(pattern, "say hello world")
print(m.group(0))  # => 'hello world' (entire match)
print(m.group(1))  # => 'hello' (group 1)
print(m.group(2))  # => 'world' (group 2)
print(m.groups())  # => ('hello', 'world')

# Non-capturing group (?:...)
# Groups without capturing
pattern = r'(?:hello|hi) (world)'
m = re.search(pattern, "say hello world")
print(m.group(1))  # => 'world' (group numbers are not shifted)

# Named group (?P<name>...)
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
m = re.search(pattern, "Date: 2026-02-15")
print(m.group('year'))   # => '2026'
print(m.group('month'))  # => '02'
print(m.group('day'))    # => '15'
print(m.groupdict())     # => {'year': '2026', 'month': '02', 'day': '15'}
```

### 2.5 Square Brackets `[]` -- Character Class

```python
import re

# Basic character class
print(re.findall(r'[aeiou]', "hello world"))
# => ['e', 'o', 'o']  *Matches only vowels

# Range specification
print(re.findall(r'[a-z]', "Hello 123"))
# => ['e', 'l', 'l', 'o']

print(re.findall(r'[A-Za-z0-9]', "Hello 123!"))
# => ['H', 'e', 'l', 'l', 'o', '1', '2', '3']

# Negated character class
print(re.findall(r'[^a-z]', "hello 123!"))
# => [' ', '1', '2', '3', '!']

# Handling of metacharacters inside character classes
# Most metacharacters are treated as literals
print(re.findall(r'[.+*?]', "a.b+c*d?e"))
# => ['.', '+', '*', '?']

# However, the following are special:
# ] → End of character class (place at beginning or escape: [\]] or []abc])
# \ → Escape
# ^ → Negation only at the beginning
# - → Range only between characters (literal at beginning/end)
```

### 2.6 Curly Braces `{}` -- Quantifiers

```python
import re

# {n} exactly n times
print(re.findall(r'\d{3}', "12 123 1234 12345"))
# => ['123', '123', '123']  *Extracts 123 from 1234, 123 and 45 separate from 12345

# {n,m} between n and m times
print(re.findall(r'\d{2,4}', "1 12 123 1234 12345"))
# => ['12', '123', '1234', '1234']

# {n,} n or more times
print(re.findall(r'\d{3,}', "1 12 123 1234 12345"))
# => ['123', '1234', '12345']

# {,m} 0 to m times (= {0,m})
print(re.findall(r'a{,3}', "aaaa"))
# => ['aaa', 'a', '']  *Matches up to 3 'a's

# Practical example: postal code
print(re.findall(r'\d{3}-\d{4}', "100-0001 Chiyoda-ku, Tokyo"))
# => ['100-0001']

# Practical example: IPv4 address (simplified)
print(re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
                 "Server: 192.168.1.1, Gateway: 10.0.0.1"))
# => ['192.168.1.1', '10.0.0.1']
```

### 2.7 Caret `^` and Dollar Sign `$` -- Anchors

```python
import re

# ^ matches the start of line
print(re.search(r'^hello', "hello world").group())    # => 'hello'
print(re.search(r'^hello', "say hello"))              # => None

# $ matches the end of line
print(re.search(r'world$', "hello world").group())    # => 'world'
print(re.search(r'world$', "world hello"))            # => None

# Combine ^ and $ for full-string matching
print(re.match(r'^\d{3}-\d{4}$', "100-0001"))
# => <re.Match object; ...>  Match succeeded

print(re.match(r'^\d{3}-\d{4}$', "100-0001 Tokyo"))
# => None  (extra characters at the end)

# ^ and $ in multiline mode
text = """line1
line2
line3"""

# Default: ^ matches only the start of the entire string
print(re.findall(r'^line\d', text))
# => ['line1']

# MULTILINE: ^ matches the start of each line
print(re.findall(r'^line\d', text, re.MULTILINE))
# => ['line1', 'line2', 'line3']
```

---

## 3. Escaping

### 3.1 Escaping Metacharacters

```python
import re

# Use a backslash to treat metacharacters as literals
price_pattern = r'\$\d+\.\d{2}'
text = "Price: $19.99 and $5.00"

matches = re.findall(price_pattern, text)
print(matches)  # => ['$19.99', '$5.00']

# Examples of metacharacters that need escaping:
#   \.  → Literal dot
#   \*  → Literal asterisk
#   \+  → Literal plus
#   \?  → Literal question mark
#   \(  → Literal opening parenthesis
#   \)  → Literal closing parenthesis
#   \[  → Literal opening square bracket
#   \{  → Literal opening curly brace
#   \|  → Literal pipe
#   \\  → Literal backslash
#   \^  → Literal caret
#   \$  → Literal dollar sign
```

```python
# Practical patterns requiring escaping

import re

# 1. File path (Windows)
path = r'C:\Users\gaku\Documents\file.txt'
pattern = r'C:\\Users\\(\w+)\\Documents\\(\w+\.txt)'
m = re.search(pattern, path)
if m:
    print(f"User: {m.group(1)}, File: {m.group(2)}")
    # => User: gaku, File: file.txt

# 2. URL pattern
url = "https://example.com/path?key=value&key2=value2"
pattern = r'https?://([^/]+)(/[^?]*)?\?(.+)'
m = re.search(pattern, url)
if m:
    print(f"Host: {m.group(1)}")   # => example.com
    print(f"Path: {m.group(2)}")   # => /path
    print(f"Query: {m.group(3)}")  # => key=value&key2=value2

# 3. Mathematical expression
expr = "f(x) = 3x^2 + 2x + 1"
pattern = r'f\(x\) = (\d+)x\^(\d+)'
m = re.search(pattern, expr)
if m:
    print(f"Coefficient: {m.group(1)}, Exponent: {m.group(2)}")
    # => Coefficient: 3, Exponent: 2

# 4. IP address (escaping dots is important)
ip_text = "Server 192.168.1.1 Port 8080"
pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
m = re.search(pattern, ip_text)
print(m.group())  # => '192.168.1.1'
```

### 3.2 The Double-Escaping Problem

```
Escape flow:

Source Code           Python String        Regex Engine
─────────────       ──────────────      ────────────────
"\\d"          →    \d             →    Matches one digit
"\\\\d"        →    \\d            →    Literal \ + d
r"\d"          →    \d             →    Matches one digit (raw string)
r"\\d"         →    \\d            →    Literal \ + d

* Using raw strings (r"...") avoids double escaping
```

```python
import re

# The double-escaping problem
# When matching Windows paths:

# BAD: Regular string -- backslashes are interpreted at two levels
pattern_bad = "C:\\\\Users\\\\\\w+"
# Python string interpretation: C:\\Users\\\w+
# Regex interpretation: C:\Users\ + word characters

# GOOD: Use raw strings
pattern_good = r"C:\\Users\\\w+"

text = r"C:\Users\gaku"
print(re.search(pattern_good, text).group())  # => C:\Users\gaku
```

```python
# Cases prone to double-escaping problems

import re

# Case 1: Matching a backslash itself
# Goal: Find \ in text
text = "path\\to\\file"

# BAD: Regular string
# "\\" → Python string: \ → Regex: incomplete escape
# GOOD: Raw string
pattern = r'\\'  # Raw string: \\ → Regex: literal \
print(re.findall(pattern, text))  # => ['\\', '\\']

# Case 2: Matching the literal string \n
# Goal: Find the character sequence "\n" (backslash + n) in text
text = r"Newline is represented by \n"

# BAD: "\n" → Python string: newline character → matches newline
# GOOD:
pattern = r'\\n'  # Raw string: \\n → Regex: literal \ + n
print(re.findall(pattern, text))  # => ['\\n']

# Case 3: Extra caution needed in Java / JavaScript
# Java: Pattern.compile("\\\\n") → \\ → literal \ + n
# JavaScript: /\\n/ → literal \ + n
# JavaScript: new RegExp("\\\\n") → string escape + regex escape
```

### 3.3 Special Escape Sequences

```
Escape Sequence Reference:

Character class shorthands:
  \d  → Digit [0-9]
  \D  → Non-digit [^0-9]
  \w  → Word character [a-zA-Z0-9_]
  \W  → Non-word character [^a-zA-Z0-9_]
  \s  → Whitespace [ \t\n\r\f\v]
  \S  → Non-whitespace [^ \t\n\r\f\v]

Anchors:
  \b  → Word boundary
  \B  → Non-word boundary

Special characters:
  \t  → Tab
  \n  → Newline (LF)
  \r  → Carriage return (CR)
  \f  → Form feed
  \v  → Vertical tab
  \0  → NULL character
  \a  → Bell character
  \e  → Escape character (ESC, 0x1B) *Some engines only

Numeric specification:
  \xHH    → Hexadecimal (e.g., \x41 = 'A')
  \uHHHH  → Unicode BMP (e.g., \u3042 = Japanese hiragana 'a')
  \UHHHHHHHH → Unicode (e.g., \U0001F600 = emoji)
  \N{name}   → Unicode name (e.g., \N{SNOWMAN} = snowman character) *Python
  \oOOO   → Octal (e.g., \o101 = 'A')
```

```python
import re

# Verifying shorthand behavior

# \d: digits
print(re.findall(r'\d+', "abc123def456"))
# => ['123', '456']

# \w: word characters
print(re.findall(r'\w+', "hello, world! 123"))
# => ['hello', 'world', '123']

# \s: whitespace characters
text = "hello\tworld\nnext line"
print(re.findall(r'\s', text))
# => ['\t', '\n', ' ']

# \b: word boundary
text = "cat caterpillar concatenate"
print(re.findall(r'\bcat\b', text))
# => ['cat']  *Exact match only

print(re.findall(r'\bcat', text))
# => ['cat', 'cat', 'cat']  *Words starting with cat

# Uppercase versions are negations
print(re.findall(r'\D+', "abc123def"))  # Non-digits
# => ['abc', 'def']

print(re.findall(r'\W+', "hello, world!"))  # Non-word characters
# => [', ', '!']

print(re.findall(r'\S+', "hello world"))  # Non-whitespace
# => ['hello', 'world']
```

### 3.4 Automatic Escaping with re.escape()

```python
import re

# When incorporating user input as a literal into a pattern
user_input = "file (1).txt"

# BAD: Using it directly causes metacharacters to be interpreted
try:
    re.search(user_input, "file (1).txt")  # () interpreted as a group
except re.error as e:
    print(f"Error: {e}")

# GOOD: Escape metacharacters with re.escape()
escaped = re.escape(user_input)
print(escaped)  # => 'file\\ \\(1\\)\\.txt'
m = re.search(escaped, "file (1).txt")
print(m.group())  # => 'file (1).txt'

# Practical example: literal search of user input
def search_literal(text, query):
    """Search for user input as a literal"""
    pattern = re.escape(query)
    return re.findall(pattern, text)

print(search_literal("price is $10.00", "$10.00"))
# => ['$10.00']

# Practical example: extract text between literal delimiters
def extract_between(text, start, end):
    """Extract strings between start and end"""
    pattern = re.escape(start) + r'(.+?)' + re.escape(end)
    return re.findall(pattern, text)

print(extract_between("value = [hello]", "[", "]"))
# => ['hello']
```

### 3.5 Escape Functions in Various Languages

```python
# Python
import re
re.escape("hello.world")  # => 'hello\\.world'
```

```javascript
// JavaScript (not in the standard, but a commonly used utility)
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
escapeRegExp("hello.world");  // => "hello\\.world"
```

```java
// Java
java.util.regex.Pattern.quote("hello.world");
// => "\\Qhello.world\\E" (wrapped in a literal block)
```

```ruby
# Ruby
Regexp.escape("hello.world")  # => "hello\\.world"
```

---

## 4. Flags (Modifiers)

### 4.1 Major Flags Reference

```python
import re

text = """Hello World
hello python
HELLO REGEX"""

# i flag: ignore case
print(re.findall(r'hello', text, re.IGNORECASE))
# => ['Hello', 'hello', 'HELLO']

# m flag: multiline mode (^ and $ apply to each line)
print(re.findall(r'^hello', text, re.MULTILINE | re.IGNORECASE))
# => ['Hello', 'hello', 'HELLO']

# s flag: dot matches newlines too
print(re.search(r'Hello.+REGEX', text, re.DOTALL).group())
# => 'Hello World\nhello python\nHELLO REGEX'

# x flag: verbose mode (ignores whitespace and comments)
pattern = re.compile(r'''
    \d{4}       # Year (4 digits)
    -            # Hyphen separator
    \d{2}       # Month (2 digits)
    -            # Hyphen separator
    \d{2}       # Day (2 digits)
''', re.VERBOSE)
print(pattern.search("Date: 2026-02-11").group())
# => '2026-02-11'
```

### 4.2 Flag Comparison Table

| Flag | Python | JavaScript | Perl | Java | Effect |
|------|--------|------------|------|------|--------|
| Case-insensitive | `re.IGNORECASE` / `re.I` | `/i` | `/i` | `CASE_INSENSITIVE` | Ignore case differences |
| Multiline | `re.MULTILINE` / `re.M` | `/m` | `/m` | `MULTILINE` | `^` `$` match start/end of each line |
| Dotall | `re.DOTALL` / `re.S` | `/s` | `/s` | `DOTALL` | `.` matches newlines too |
| Verbose | `re.VERBOSE` / `re.X` | Not supported | `/x` | `COMMENTS` | Ignores whitespace and comments |
| Unicode | `re.UNICODE` / `re.U` | `/u` | Default | `UNICODE_CHARACTER_CLASS` | Unicode support |
| Global | N/A (`findall`) | `/g` | `/g` | N/A (`Matcher.find()`) | Return all matches |
| Sticky | N/A | `/y` | N/A | N/A | Match only from lastIndex position |
| ASCII | `re.ASCII` / `re.A` | N/A | `/a` | N/A | Restrict \d \w \s to ASCII only |

### 4.3 Inline Flags

```python
import re

# Embed flags within the pattern
# (?flags) format

# (?i) ignore case
print(re.findall(r'(?i)hello', "Hello HELLO hello"))
# => ['Hello', 'HELLO', 'hello']

# (?m) multiline mode
text = "line1\nline2\nline3"
print(re.findall(r'(?m)^\w+', text))
# => ['line1', 'line2', 'line3']

# (?s) dotall
print(re.search(r'(?s)line1.+line3', text).group())
# => 'line1\nline2\nline3'

# (?x) verbose mode
pattern = r'''(?x)
    (\d{4})     # Year
    -(\d{2})    # Month
    -(\d{2})    # Day
'''
m = re.search(pattern, "2026-02-15")
print(m.groups())  # => ('2026', '02', '15')

# Combining multiple flags
print(re.findall(r'(?im)^hello', "Hello\nhello\nHELLO"))
# => ['Hello', 'hello', 'HELLO']

# Scoped flags (Python 3.6+)
# (?i:pattern) applies the flag only to that portion
pattern = r'(?i:hello) world'  # hello is case-insensitive, world is case-sensitive
print(re.findall(pattern, "Hello world HELLO world hello World"))
# => ['Hello world', 'hello world']  *'HELLO world' matches, 'hello World' does not
# In practice:
# 'Hello world' → match
# 'HELLO world' → match
# 'hello World' → no match (world is uppercase)
```

### 4.4 Practical Flag Patterns

```python
import re

# 1. Log file analysis (multiline + case-insensitive)
log = """
2026-02-15 10:30:00 [ERROR] Database connection failed
2026-02-15 10:31:00 [Warning] High memory usage
2026-02-15 10:32:00 [error] Disk space low
"""

# Matches ERROR, Warning, and error
errors = re.findall(
    r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[(?:error|warning)\] (.+)$',
    log,
    re.MULTILINE | re.IGNORECASE
)
print(errors)
# => ['Database connection failed', 'High memory usage', 'Disk space low']

# 2. Improving readability of complex patterns (verbose mode)
email_pattern = re.compile(r'''
    ^                       # Start of string
    [a-zA-Z0-9._%+-]+      # Local part
    @                       # At sign
    [a-zA-Z0-9.-]+          # Domain name
    \.                      # Dot
    [a-zA-Z]{2,}            # TLD
    $                       # End of string
''', re.VERBOSE)

# 3. Multiline text analysis (dotall + multiline)
html = """<div class="content">
    <p>First paragraph</p>
    <p>Second paragraph</p>
</div>"""

# Extract the entire contents of the div
m = re.search(r'<div[^>]*>(.*?)</div>', html, re.DOTALL)
if m:
    print(m.group(1).strip())
```

---

## 5. ASCII Diagrams: Pattern Matching Flow

### 5.1 Basic Matching Procedure

```
Pattern: h.llo
Text: "say hello world"

Position: s a y   h e l l o   w o r l d
          0 1 2 3 4 5 6 7 8 9 ...

Attempt 1: position 0, 's' != 'h' → fail, move to position 1
Attempt 2: position 1, 'a' != 'h' → fail, move to position 2
Attempt 3: position 2, 'y' != 'h' → fail, move to position 3
Attempt 4: position 3, ' ' != 'h' → fail, move to position 4
Attempt 5: position 4, 'h' = 'h' → match
           position 5, 'e' = '.' → match (any single character)
           position 6, 'l' = 'l' → match
           position 7, 'l' = 'l' → match
           position 8, 'o' = 'o' → match
           → Match succeeded: "hello" (position 4-8)
```

### 5.2 Metacharacter Meaning Map

```
┌─────────────────────────────────────────────────┐
│         Components of Regular Expressions         │
├────────────┬────────────────────────────────────┤
│ Literals   │  a b c 1 2 3 etc.                   │
│ (as-is)    │  → Matches the character itself       │
├────────────┼────────────────────────────────────┤
│ Metacharacters │  . ^ $ * + ? | ( ) [ ] { } \   │
│ (special)  │  → Instructs special behavior         │
├────────────┼────────────────────────────────────┤
│ Escapes    │  \. \* \+ \? \( \) \[ \{ \\ etc.    │
│ (disable)  │  → Reverts metacharacters to literals  │
├────────────┼────────────────────────────────────┤
│ Shorthands │  \d \w \s \b \t \n etc.              │
│ (abbreviations) │  → Abbreviated character class    │
│            │    notation                            │
└────────────┴────────────────────────────────────┘
```

### 5.3 Structure of Escape Layers

```
  Source Code Layer      Language Layer        Regex Engine Layer
 ┌──────────┐      ┌──────────┐      ┌──────────────┐
 │ "\\d+"   │ ───→ │  \d+     │ ───→ │ 1+ digits     │
 │ r"\d+"   │ ───→ │  \d+     │ ───→ │ 1+ digits     │
 │ "\\\\n"  │ ───→ │  \\n     │ ───→ │ \ + n (2 chars)│
 │ r"\\n"   │ ───→ │  \\n     │ ───→ │ \ + n (2 chars)│
 │ "\n"     │ ───→ │  newline │ ───→ │ newline char   │
 │ r"\n"    │ ───→ │  \n      │ ───→ │ newline char   │
 └──────────┘      └──────────┘      └──────────────┘

 Key point: Raw strings (r"...") disable escaping
            at the language layer.
            Regex engine layer escaping is separate.
```

### 5.4 Quantifier Behavior Comparison

```
Pattern: a{2,4}
Text: "aaaaaa"

Greedy match (default):
  Position 0: a{2,4} → "aaaa" (consumes maximum 4 characters)
  Position 4: a{2,4} → "aa"   (consumes remaining 2 characters)
  Result: ["aaaa", "aa"]

Non-greedy match (a{2,4}?):
  Position 0: a{2,4}? → "aa" (consumes minimum 2 characters)
  Position 2: a{2,4}? → "aa" (consumes minimum 2 characters)
  Position 4: a{2,4}? → "aa" (consumes minimum 2 characters)
  Result: ["aa", "aa", "aa"]
```

```
Pattern: <.+> vs <.+?>
Text: "<b>bold</b>"

Greedy <.+>:
  < matches → . greedily consumes "b>bold</b" → > matches
  Result: "<b>bold</b>" (one large match)

Non-greedy <.+?>:
  < matches → . consumes minimum "b" → > matches
  Result: "<b>" (minimum match)
  Continues: "<" matches → "..." → Result: "</b>"
```

---

## 6. Practical Pattern Examples

### 6.1 Basic Validation Patterns

```python
import re

# Japanese postal code
postal_code = re.compile(r'^\d{3}-\d{4}$')
assert postal_code.match('100-0001')
assert not postal_code.match('1000001')
assert not postal_code.match('100-000')

# Japanese mobile phone number
mobile_phone = re.compile(r'^0[789]0-\d{4}-\d{4}$')
assert mobile_phone.match('090-1234-5678')
assert mobile_phone.match('080-1234-5678')
assert not mobile_phone.match('03-1234-5678')

# Date in ISO format (YYYY-MM-DD)
date_pattern = re.compile(r'^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$')
assert date_pattern.match('2026-02-15')
assert date_pattern.match('2026-12-31')
assert not date_pattern.match('2026-13-01')
assert not date_pattern.match('2026-00-15')

# Time (HH:MM:SS)
time_pattern = re.compile(r'^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$')
assert time_pattern.match('00:00:00')
assert time_pattern.match('23:59:59')
assert not time_pattern.match('24:00:00')
assert not time_pattern.match('12:60:00')
```

### 6.2 Text Extraction Patterns

```python
import re

# Extract links from Markdown
text = "See the [official site](https://example.com) and [FAQ](https://example.com/faq) for details"
links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)
for label, url in links:
    print(f"  {label} → {url}")
# => official site → https://example.com
# => FAQ → https://example.com/faq

# Extract hashtags
tweet = "Beautiful weather today #sunny #tokyo Amazing!"
tags = re.findall(r'#(\w+)', tweet)
print(tags)  # => ['sunny', 'tokyo']

# Quoted strings
text = 'She said "hello" and "goodbye"'
quoted = re.findall(r'"([^"]*)"', text)
print(quoted)  # => ['hello', 'goodbye']

# Key-value pair extraction
config = "host=localhost port=3306 db=mydb user=admin"
pairs = re.findall(r'(\w+)=(\S+)', config)
print(dict(pairs))  # => {'host': 'localhost', 'port': '3306', 'db': 'mydb', 'user': 'admin'}
```

### 6.3 Text Replacement Patterns

```python
import re

# 1. Snake case → camel case
def snake_to_camel(name):
    return re.sub(r'_([a-z])', lambda m: m.group(1).upper(), name)

print(snake_to_camel('hello_world'))    # => 'helloWorld'
print(snake_to_camel('my_var_name'))    # => 'myVarName'

# 2. Camel case → snake case
def camel_to_snake(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

print(camel_to_snake('helloWorld'))    # => 'hello_world'
print(camel_to_snake('myVarName'))     # => 'my_var_name'

# 3. Normalize consecutive whitespace
text = "hello   world\t\t\tnext"
print(re.sub(r'\s+', ' ', text))  # => 'hello world next'

# 4. Remove HTML tags
html = "<p>Hello <b>World</b></p>"
print(re.sub(r'<[^>]+>', '', html))  # => 'Hello World'

# 5. Date format conversion (YYYY/MM/DD → YYYY-MM-DD)
date = "2026/02/15"
print(re.sub(r'(\d{4})/(\d{2})/(\d{2})', r'\1-\2-\3', date))
# => '2026-02-15'

# 6. Detect and fix duplicate words using backreferences
text = "the the quick brown fox fox"
print(re.sub(r'\b(\w+)\s+\1\b', r'\1', text))
# => 'the quick brown fox'
```

---

## 7. Anti-patterns

### 7.1 Anti-pattern: Not Using Raw Strings

```python
import re

# BAD: Writing regex without raw strings
pattern_bad = "\\b\\w+\\b"  # Hard to read, prone to escape mistakes

# GOOD: Using raw strings
pattern_good = r"\b\w+\b"   # Clear and less error-prone

text = "hello world"
print(re.findall(pattern_good, text))  # => ['hello', 'world']

# A particularly dangerous example:
# "\b" in Python is the backspace character (0x08)
# r"\b" is the regex word boundary
print("\b" == "\x08")   # => True  -- backspace!
print(r"\b" == "\\b")   # => True  -- regex \b
```

### 7.2 Anti-pattern: Overuse of Dot

```python
import re

# BAD: Matching anything with dot
pattern_bad = r'\d+.\d+.\d+'

# This matches not only "192.168.1.1" but also:
texts = ["192.168.1.1", "192-168-1-1", "192x168x1x1", "192 168 1 1"]
for t in texts:
    m = re.search(pattern_bad, t)
    if m:
        print(f"  Match: {m.group()}")  # All match

# GOOD: Escape the dot to be explicit
pattern_good = r'\d+\.\d+\.\d+\.\d+'
for t in texts:
    m = re.search(pattern_good, t)
    if m:
        print(f"  Match: {m.group()}")  # Only "192.168.1.1" matches
```

### 7.3 Anti-pattern: Unnecessarily Complex Patterns

```python
import re

# BAD: Cases where regex is unnecessary

# Simple string search is sufficient with the in operator
text = "hello world"

# BAD
if re.search(r'hello', text):
    pass

# GOOD (fast and clear)
if 'hello' in text:
    pass

# BAD: Using regex to check start/end
if re.match(r'^hello', text):
    pass

# GOOD
if text.startswith('hello'):
    pass

# BAD: Using regex for fixed string replacement
re.sub(r'hello', 'hi', text)

# GOOD
text.replace('hello', 'hi')
```

### 7.4 Anti-pattern: Confusing match() and search()

```python
import re

text = "say hello world"

# match() only attempts matching from the start of the string
m = re.match(r'hello', text)
print(m)  # => None  *Start is 'say', so no match

# search() searches the entire string
m = re.search(r'hello', text)
print(m.group())  # => 'hello'

# fullmatch() checks if the entire string matches
m = re.fullmatch(r'hello', "hello")
print(m.group())  # => 'hello'

m = re.fullmatch(r'hello', "hello world")
print(m)  # => None  *Extra characters at the end

# Best practices for full matching:
# Use fullmatch() for input validation (Python 3.4+)
# Use search() for text searching
# Use match() for line-start matching
```

---

## 8. Performance Tips

### 8.1 Pattern Pre-compilation

```python
import re
import time

text_lines = [f"line {i}: some text here" for i in range(100000)]

# BAD: Using string pattern in every iteration
start = time.time()
for line in text_lines:
    re.search(r'\d+', line)
print(f"Not compiled: {time.time() - start:.3f}s")

# GOOD: Pre-compile
pattern = re.compile(r'\d+')
start = time.time()
for line in text_lines:
    pattern.search(line)
print(f"Pre-compiled: {time.time() - start:.3f}s")

# Note: Python's re module has an internal cache (up to 512 patterns),
#    so the difference is small when reusing a few patterns repeatedly,
#    but explicit compilation makes the intent clear
```

### 8.2 Writing Efficient Patterns

```python
import re

# 1. Character classes are faster than alternatives
# BAD
pattern_slow = r'a|b|c|d|e'
# GOOD
pattern_fast = r'[a-e]'

# 2. Non-capturing groups save memory
# BAD (when capture is not needed)
pattern_slow = r'(foo|bar|baz)+'
# GOOD
pattern_fast = r'(?:foo|bar|baz)+'

# 3. Anchors limit the search scope
# BAD
pattern_slow = r'error'  # Scans the entire string
# GOOD (when error is at the start of line)
pattern_fast = r'^error'  # Only checks the start of each line

# 4. Specific patterns are preferred
# BAD
pattern_slow = r'.+@.+\..+'  # Too vague
# GOOD
pattern_fast = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# 5. Place longer alternatives first
# BAD (shorter alternative matches first, causing issues)
pattern_slow = r'Java|JavaScript'
# GOOD
pattern_fast = r'JavaScript|Java'
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

### Q1: What if it's tedious to escape every metacharacter?

**A**: Most languages have a function to "escape the entire string":

```python
import re
user_input = "price is $10.00 (tax+)"
escaped = re.escape(user_input)
print(escaped)  # => 'price\\ is\\ \\$10\\.00\\ \\(tax\\+\\)'

# Can be safely used as a regex pattern
pattern = re.compile(escaped)
```

In JavaScript, you can achieve the same with `string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')`.

### Q2: How do I make `.` match newlines too?

**A**: Use the DOTALL (Python) / `s` flag (JavaScript ES2018+):

```python
import re
text = "line1\nline2\nline3"
# Default: dot does not match newlines
print(re.search(r'line1.+line3', text))  # => None

# DOTALL: dot matches newlines too
m = re.search(r'line1.+line3', text, re.DOTALL)
print(m.group())  # => 'line1\nline2\nline3'
```

Alternatively, you can use `[\s\S]` which works in any language.

### Q3: Can multiple regex flags be used simultaneously?

**A**: Yes. In Python, combine them with bitwise OR (`|`):

```python
import re
pattern = re.compile(r'^hello.+world$', re.IGNORECASE | re.MULTILINE | re.DOTALL)
# JavaScript: /^hello.+world$/ims
# Perl: /^hello.+world$/ims
```

In Python, inline flags `(?ims)` can also be used:
```python
pattern = re.compile(r'(?ims)^hello.+world$')
```

### Q4: Does \d match full-width digits?

**A**: It depends on the engine and flags:

```python
import re

# Python default (Unicode mode)
print(re.findall(r'\d+', "half-width 123 full-width \uff11\uff12\uff13"))
# => ['123', '\uff11\uff12\uff13']  *Also matches full-width digits!

# Restrict to ASCII mode
print(re.findall(r'\d+', "half-width 123 full-width \uff11\uff12\uff13", re.ASCII))
# => ['123']  *Half-width digits only

# JavaScript /u flag
# /\d+/u matches Unicode digits
# /\d+/ matches ASCII digits only (varies by engine)
```

### Q5: What is the difference between match(), search(), and fullmatch()?

**A**:

```python
import re

text = "hello world"

# match(): attempts matching from the start of the string only
re.match(r'hello', text)       # => match
re.match(r'world', text)       # => None

# search(): searches the entire string
re.search(r'hello', text)      # => match
re.search(r'world', text)      # => match

# fullmatch(): checks if the entire string matches the pattern
re.fullmatch(r'hello', text)   # => None
re.fullmatch(r'hello world', text)  # => match
```

### Q6: How do I write comments in regex?

**A**: Use verbose mode (`re.VERBOSE` / `re.X`):

```python
import re

pattern = re.compile(r'''
    ^                   # Start of string
    (?P<protocol>       # Protocol portion
        https?          #   http or https
    )
    ://                 # Scheme separator
    (?P<host>           # Host portion
        [^/]+           #   One or more non-slash characters
    )
    (?P<path>           # Path portion (optional)
        /[^\s]*         #   Starts with a slash
    )?
    $                   # End of string
''', re.VERBOSE)

m = pattern.match('https://example.com/path/to/page')
if m:
    print(m.groupdict())
    # => {'protocol': 'https', 'host': 'example.com', 'path': '/path/to/page'}
```

When a literal space is needed inside verbose mode, use `\ ` or the class `[ ]`.

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
| Literal characters | Match the corresponding character directly |
| Metacharacters | 13 types: `. ^ $ * + ? \| ( ) [ ] { } \` |
| Escaping | `\` reverts metacharacters to literals |
| Raw strings | `r"..."` disables language-level escaping (Python) |
| Shorthands | Abbreviated notations such as `\d` `\w` `\s` `\b` |
| Flags | `i` (case-insensitive), `m` (multiline), `s` (dotall), `x` (verbose) |
| Double escaping | Two layers of escaping occur at the source code and regex engine levels |
| Inline flags | Embed flags within patterns using `(?i)` `(?m)` `(?s)` `(?x)` |
| Pre-compilation | Create pattern objects with `re.compile()` for reuse |
| re.escape() | Automatically escapes metacharacters in user input |
| Golden rule | Always use raw strings, and use dot only when necessary |

---

## Recommended Next Guides

- [02-character-classes.md](./02-character-classes.md) -- Character Classes: `[abc]`, `\d`, `\w`, `\s`, POSIX classes
- [03-quantifiers-anchors.md](./03-quantifiers-anchors.md) -- Quantifiers and Anchors in Detail

---

## References

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions, 3rd Edition" O'Reilly Media, 2006 -- Chapter 3 "Basic Syntax" is especially helpful
2. **Python re module documentation** https://docs.python.org/3/library/re.html -- Official reference for Python regular expressions
3. **MDN Web Docs - Regular Expressions** https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions -- Comprehensive guide to JavaScript regular expressions
4. **Java Pattern class documentation** https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/regex/Pattern.html -- Official reference for Java regular expressions
5. **Ruby Regexp documentation** https://docs.ruby-lang.org/en/3.2/Regexp.html -- Official reference for Ruby regular expressions
