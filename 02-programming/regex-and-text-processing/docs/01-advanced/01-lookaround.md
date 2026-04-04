# Lookaround -- (?=)(?!)(?<=)(?<!)

> Lookahead and lookbehind are zero-width assertions that specify positional conditions without consuming characters. They are powerful features that enable constraints difficult to express with ordinary patterns, such as password strength validation, complex extraction conditions, and limiting replacement targets.

## What You Will Learn

1. **Syntax and behavior of all 4 lookaround types** -- The precise meaning of positive/negative lookahead/lookbehind
2. **The concept of zero-width assertions** -- How matching without consuming characters works and its applications
3. **Practical use cases** -- Password validation, number formatting, compound condition extraction
4. **Behavioral differences and constraints across languages** -- Implementation differences in Python, JavaScript, Java, Go, and Rust
5. **Performance impact and optimization** -- The cost of lookaround and how to mitigate it
6. **Nested lookaround** -- Advanced techniques for complex conditions
7. **Test-driven pattern development** -- Methods for safely developing lookaround patterns


## Prerequisites

The following knowledge will help you get the most out of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Groups and Backreferences -- Capturing, Named Groups, Lookahead/Lookbehind](./00-groups-backreferences.md)

---

## 1. The 4 Types of Lookaround

### 1.1 Overview

```
+------------------------------------------------------+
|              Lookaround Overview                      |
+------------+--------------+--------------------------+
|            | Positive (=) | Negative (!)              |
+------------+--------------+--------------------------+
| Lookahead  | (?=pattern)  | (?!pattern)              |
| (forward)  | Followed by  | NOT followed by          |
+------------+--------------+--------------------------+
| Lookbehind | (?<=pattern) | (?<!pattern)             |
| (backward) | Preceded by  | NOT preceded by          |
+------------+--------------+--------------------------+
```

### 1.2 Conceptual Diagram

```
Text: "price: $100"

         p r i c e :   $ 1 0 0
                        ^
                    Current position

Lookahead (?=...):  "To the right of this position, ... exists"
Lookbehind (?<=...): "To the left of this position, ... exists"

Example: (?<=\$)\d+
  -> "A digit sequence starting from a position with $ to its left"
  -> Matches "100" ($ is not included)
```

### 1.3 Internal Mechanism of Lookaround

To accurately understand how lookaround works, let's look at how the regex engine processes it internally.

```
NFA engine lookaround processing flow:

1. Engine records current position (position = P)
2. Attempts to match the pattern inside the lookaround
   - Positive: match success -> assertion success
   - Negative: match failure -> assertion success
3. Restores current position to P (zero-width: position doesn't advance)
4. Proceeds to the next element of the main pattern

Concrete example: Pattern (?<=\$)\d+ applied to "price: $100"

Step 1: Position 0 'p' -- (?<=\$) check
  Nothing to the left -> failure -> advance position by 1

Step 2: Position 1 'r' -- (?<=\$) check
  Left side 'p' is not '$' -> failure -> advance position by 1

...

Step 8: Position 8 '1' -- (?<=\$) check
  Left side '$' is '$' -> success!
  -> Try \d+ from position 8
  -> '1','0','0' match
  -> Result: "100" (position 8-11)
```

### 1.4 Comparison of Lookaround with Other Zero-Width Assertions

```
Zero-width assertion list:

Assertion           Syntax     Meaning
--------------     ------     --------------------------
Line start         ^          Beginning of line
Line end           $          End of line
Word boundary      \b         Boundary between word and non-word chars
Non-word boundary  \B         Position that is NOT \b
String start       \A         Beginning of entire string (even in multiline)
String end         \Z, \z     End of entire string
Positive lookahead (?=...)    Position where pattern exists to the right
Negative lookahead (?!...)    Position where pattern does NOT exist to the right
Positive lookbehind(?<=...)   Position where pattern exists to the left
Negative lookbehind(?<!...)   Position where pattern does NOT exist to the left

Common trait: all check a "position" and do not consume characters
```

```python
import re

# Comparison of various zero-width assertions
text = "Hello World 123"

# ^ -- beginning of line
print(re.findall(r'^.', text))          # => ['H']

# \b -- word boundary
print(re.findall(r'\b\w', text))        # => ['H', 'W', '1']

# (?=...) -- positive lookahead
print(re.findall(r'\w(?=\s)', text))    # => ['o', 'd']

# (?<=...) -- positive lookbehind
print(re.findall(r'(?<=\s)\w', text))   # => ['W', '1']

# All are zero-width: they do not consume characters
```

---

## 2. Positive Lookahead `(?=pattern)`

### 2.1 Basic Behavior

```python
import re

# Extract "numbers followed by 円"
pattern = r'\d+(?=円)'
text = "Product A: 1000円, Product B: 2500円, Product C: 30 dollars"

print(re.findall(pattern, text))
# => ['1000', '2500']
# Note: "30" does not match (followed by "dollars", not "円")
# Note: "円" itself is not included in the match (zero-width)
```

### 2.2 Proof of Zero-Width

```python
import re

text = "100円"

# Without lookahead: includes digits + 円
m1 = re.search(r'\d+円', text)
print(m1.group())   # => '100円' (includes 円)
print(m1.end())     # => 4

# With lookahead: digits only (円 is not consumed)
m2 = re.search(r'\d+(?=円)', text)
print(m2.group())   # => '100' (doesn't include 円)
print(m2.end())     # => 3 (zero-width: position is right before 円)
```

### 2.3 Compound Conditions with Lookahead

Multiple lookaheads can be chained to create AND conditions.

```python
import re

# Build AND conditions with multiple positive lookaheads
# "Words that contain uppercase AND a digit AND are 6+ characters"
pattern = r'\b(?=\w*[A-Z])(?=\w*\d)\w{6,}\b'
text = "Hello World Pass1word abc123 Test99 MyPW short A1"

print(re.findall(pattern, text))
# => ['Pass1word', 'Test99']
# "Hello" -- no digit -> NG
# "abc123" -- no uppercase -> NG
# "MyPW" -- 4 chars < 6 -> NG
# "A1" -- 2 chars < 6 -> NG
```

### 2.4 Overlapping Matches with Lookahead

Normal `findall` returns only non-overlapping matches, but lookahead can detect overlapping patterns.

```python
import re

# Normal pattern: no overlap
text = "abcabc"
print(re.findall(r'ab', text))
# => ['ab', 'ab']

# Detect overlapping matches with lookahead
text = "aaaa"
# Normal: detect consecutive "aa" (no overlap)
print(re.findall(r'aa', text))
# => ['aa', 'aa']  -- positions 0 and 2

# Lookahead to detect all "aa" start positions (with overlap)
print(re.findall(r'(?=aa)', text))
# => ['', '', '']  -- positions 0, 1, 2 (3 locations)

# Practical example: detect overlapping substrings in text
text = "abcabcabc"
positions = [(m.start(), m.end()) for m in re.finditer(r'(?=(abc))', text)]
print(positions)
# => [(0, 0), (3, 3), (6, 6)]
# Get content via capture group inside the lookahead
print(re.findall(r'(?=(abc))', text))
# => ['abc', 'abc', 'abc']

# Practical example: overlapping motif detection in DNA sequences
dna = "ATGATGATG"
# Normal: "ATG" appears 3 times without overlap
print(re.findall(r'ATG', dna))  # => ['ATG', 'ATG', 'ATG']
# Overlapping pattern detection: positions where "ATGATG" appears
overlapping = [m.start() for m in re.finditer(r'(?=ATGATG)', dna)]
print(overlapping)  # => [0, 3]
```

### 2.5 Interaction Between Lookahead and Quantifiers

```python
import re

# Lookahead and greedy/non-greedy behavior
text = "abc123def456"

# Greedy: matches as long as possible
print(re.findall(r'\w+(?=\d)', text))
# => ['abc12', 'def45']
# Note: greedy \w+ consumes up to just before the last digit

# Non-greedy: matches as short as possible
print(re.findall(r'\w+?(?=\d)', text))
# => ['abc', '1', '2', 'def', '4', '5']
# Note: tries to match one character at a time

# Proper pattern design is needed for intended results
# "alphabetic characters followed by digits"
print(re.findall(r'[a-z]+(?=\d)', text))
# => ['abc', 'def']
```

---

## 3. Negative Lookahead `(?!pattern)`

### 3.1 Basic Behavior

```python
import re

# "Numbers NOT followed by ドル"
pattern = r'\d+(?!ドル|\d)'
text = "100円 200ドル 300ユーロ"

print(re.findall(pattern, text))
# => ['100', '300']
# "200" doesn't match (followed by "ドル")
```

### 3.2 Exclusion Patterns

```python
import re

# Exclude specific words from matches
# Extract words that don't start with "test"
pattern = r'\b(?!test)\w+'
text = "testing hello testcase world testify"

print(re.findall(pattern, text))
# => ['hello', 'world']

# JavaScript identifiers excluding reserved words
reserved = r'\b(?!if|else|for|while|return|function\b)\w+'
code = "function hello if world return value"
print(re.findall(reserved, code))
# => ['hello', 'world', 'value']
```

### 3.3 Practical Negative Lookahead Patterns

```python
import re

# Pattern 1: Match filenames excluding specific extensions
files = "main.py config.yaml app.js test.pyc utils.py data.json"
# Extract .py files excluding .pyc
pattern = r'\b\w+\.py(?!c)\b'
print(re.findall(pattern, files))
# => ['main.py', 'utils.py']

# Pattern 2: URL extraction excluding specific domains
urls = "http://example.com http://spam.evil.com http://good-site.org"
# Extract URLs that don't contain "spam"
pattern = r'https?://(?!spam)\S+'
print(re.findall(pattern, urls))
# => ['http://example.com', 'http://good-site.org']

# Pattern 3: Extract non-comment lines
lines = """
# This is a comment
data = 123
// This is also a comment
result = data + 1
"""
# Non-empty lines not starting with # or //
pattern = r'^(?!#|//)(?!\s*$).+'
print(re.findall(pattern, lines.strip(), re.MULTILINE))
# => ['data = 123', 'result = data + 1']
```

### 3.4 Prohibited Patterns in Passwords via Negative Lookahead

```python
import re

def validate_no_common_patterns(password: str) -> tuple[bool, list[str]]:
    """Validate that the password doesn't contain common weak patterns"""
    errors = []

    # Prohibit consecutive identical characters (aaa, 111, etc.)
    if re.search(r'(.)\1{2,}', password):
        errors.append("Contains 3 or more consecutive identical characters")

    # Prohibit sequential character patterns (abc, 123, etc.)
    sequential_patterns = [
        'abc', 'bcd', 'cde', 'def', 'efg', 'fgh',
        '123', '234', '345', '456', '567', '678', '789',
        'qwerty', 'asdf', 'zxcv'
    ]
    for seq in sequential_patterns:
        if seq in password.lower():
            errors.append(f"Contains sequential pattern '{seq}'")

    # Strong password: all conditions in one pattern
    # - 8+ characters
    # - No 3 consecutive identical characters
    # - Does not contain "password"
    # - Does not contain "12345"
    strong_pattern = re.compile(
        r'^(?!.*(.)\1{2})'     # No 3 consecutive identical chars
        r'(?!.*password)'       # Does not contain "password"
        r'(?!.*12345)'          # Does not contain "12345"
        r'.{8,}$',              # 8+ characters
        re.IGNORECASE
    )

    if not strong_pattern.match(password):
        errors.append("Password is too weak")

    return (len(errors) == 0, errors)

# Test
test_passwords = [
    "Str0ng!Pass",    # OK
    "password123!",   # Contains "password"
    "aaabbb1234!",   # Consecutive characters
    "P@ss12345word",  # Contains "12345"
    "Sh0rt!",         # Too short
]

for pw in test_passwords:
    valid, errs = validate_no_common_patterns(pw)
    status = "OK" if valid else "NG"
    print(f"  {pw}: {status} {errs if errs else ''}")
```

### 3.5 Common Pitfalls with Negative Lookahead

```python
import re

# Pitfall 1: Watch the position of negative lookahead
# Intent: exclude only the word "test"
text = "testing test tested"

# Wrong: \b(?!test)\w+ also matches parts of "test"
print(re.findall(r'\b(?!test)\w+', text))
# => ['esting', 'ed']  -- "esting" from "testing" matches!

# Correct: use \b to exclude the whole word
print(re.findall(r'\b(?!test\b)\w+', text))
# => ['testing', 'tested']
# Only exact "test" is excluded; "testing" and "tested" are OK

# Pitfall 2: Combining negative lookahead with quantifiers
text = "foobar foobaz foo"

# Intent: extract "foo" not followed by "bar"
# Wrong: match range becomes unexpected
print(re.findall(r'foo(?!bar)', text))
# => ['foo', 'foo']  -- "foo" from "foobaz" and standalone "foo"

# If you also want the following part
print(re.findall(r'foo(?!bar)\w*', text))
# => ['foobaz', 'foo']

# Pitfall 3: Watch for empty string matches
text = "abc"
print(re.findall(r'(?!abc)', text))
# => ['', '', '']  -- matches at positions 1, 2, 3 (position 0 matches 'abc' so excluded)
# Zero-width, so empty strings match
```

---

## 4. Positive Lookbehind `(?<=pattern)`

### 4.1 Basic Behavior

```python
import re

# Extract "numbers preceded by $"
pattern = r'(?<=\$)\d+'
text = "Price: $100, Tax: $15, Total: 115"

print(re.findall(pattern, text))
# => ['100', '15']
# "115" doesn't match (no $ before it)
```

### 4.2 Lookbehind Constraints

```
Lookbehind width constraints (by engine):

Engine          Variable-length   Constraint
----------      ---------------  ----------
Python re       Not allowed      Fixed-length only
JavaScript      Allowed (ES2018+) No restriction
Java            Not allowed      Fixed-length only
.NET            Allowed          No restriction
Perl            Not allowed      Fixed-length only
PCRE2           Allowed          No restriction
Ruby            Not allowed      Fixed-length only (Onigmo)
PHP (PCRE)      Not allowed      Fixed-length only (PCRE1 era)

Fixed-length constraints:
  (?<=abc)    OK  -- 3 characters fixed
  (?<=ab|cd)  OK  -- each alternative is the same length
  (?<=a{3})   OK  -- fixed repetition count
  (?<=a+)     NG  -- variable-length (not allowed in Python, Java, Perl)
  (?<=a*)     NG  -- variable-length

  Note: (?<=ab|cde) can be used in Python even with different-length
        alternatives, as long as each alternative is fixed-length (Python 3.6+)
```

```python
import re

# Fixed-length: OK
print(re.findall(r'(?<=\$)\d+', "$100 $200"))
# => ['100', '200']

# Variable-length: error (Python)
try:
    re.findall(r'(?<=\$+)\d+', "$100 $$200")
except re.error as e:
    print(f"Error: {e}")
# => Error: look-behind requires fixed-width pattern

# Workaround: use the regex module (third-party)
# import regex
# regex.findall(r'(?<=\$+)\d+', "$100 $$200")
# => ['100', '200']
```

### 4.3 Lookbehind Alternative Behavior in Python 3.6+

```python
import re

# In Python 3.6+, alternatives with different fixed lengths
# in each branch are allowed
text = "USD100 JPY200 EUR300"

# Each alternative is fixed-length (3 characters) -- OK
pattern = r'(?<=USD|JPY|EUR)\d+'
print(re.findall(pattern, text))
# => ['100', '200', '300']

# Alternatives with different lengths, but each is fixed -- OK in Python 3.6+
text = "$100 USD200 EURO300"
pattern = r'(?<=\$|USD|EURO)\d+'
print(re.findall(pattern, text))
# => ['100', '200', '300']

# However, quantifiers within alternatives are not allowed
try:
    re.findall(r'(?<=\$+|USD)\d+', text)
except re.error as e:
    print(f"Error: {e}")
# => Error: look-behind requires fixed-width pattern
```

### 4.4 Data Extraction with Lookbehind

```python
import re

# Extract HTML tag attribute values
html = '<div class="main" id="content" data-value="42">'

# Extract class attribute value
pattern = r'(?<=class=")\w+'
print(re.findall(pattern, html))
# => ['main']

# Extract id attribute value
pattern = r'(?<=id=")\w+'
print(re.findall(pattern, html))
# => ['content']

# Extract data-value
pattern = r'(?<=data-value=")\d+'
print(re.findall(pattern, html))
# => ['42']

# Extract request paths after IP addresses from log files
log_lines = [
    '192.168.1.1 - - [01/Jan/2024] "GET /api/users HTTP/1.1" 200',
    '10.0.0.5 - - [01/Jan/2024] "POST /api/login HTTP/1.1" 401',
    '172.16.0.1 - - [01/Jan/2024] "GET /index.html HTTP/1.1" 200',
]
for line in log_lines:
    # Extract path after "GET " or "POST "
    m = re.search(r'(?<=(?:GET|POST) )/\S+', line)
    if m:
        print(f"  Path: {m.group()}")
# => Path: /api/users
# => Path: /api/login
# => Path: /index.html
```

### 4.5 Combining Lookbehind and Lookahead

```python
import re

# Extract content enclosed by specific delimiters
text = "[Important] This is an important message [Info] This is information"

# Extract strings inside [] (lookbehind + lookahead)
pattern = r'(?<=\[)[^\]]+(?=\])'
print(re.findall(pattern, text))
# => ['Important', 'Info']

# Extract content enclosed in quotes
text = 'name="Alice" age="30" city="Tokyo"'
pattern = r'(?<=")[^"]+(?=")'
print(re.findall(pattern, text))
# => ['Alice', '30', 'Tokyo']

# Extract specific column values from CSV
csv_line = "Alice,30,Tokyo,Engineer"
# After the 2nd comma, before the 3rd comma
pattern = r'(?<=,)[^,]+(?=,)'
print(re.findall(pattern, csv_line))
# => ['30', 'Tokyo']  -- all fields except first and last
```

---

## 5. Negative Lookbehind `(?<!pattern)`

### 5.1 Basic Behavior

```python
import re

# Extract "numbers NOT preceded by $"
pattern = r'(?<!\$)\b\d+'
text = "Price: $100, Qty: 5, Tax: $15, Count: 42"

print(re.findall(pattern, text))
# => ['5', '42']
# "$100" and "$15" don't match (preceded by $)
```

### 5.2 Compound Conditions

```python
import re

# Negative lookbehind + negative lookahead combined
# "Numbers not enclosed in quotes"
pattern = r'(?<!["\'`])\b\d+\b(?!["\'`])'
text = 'value is 42 and "100" and \'200\''

print(re.findall(pattern, text))
# => ['42']
```

### 5.3 Practical Negative Lookbehind Examples

```python
import re

# Detect unescaped special characters
text = r'Hello\nWorld\tTab\\Backslash\xHex'

# Detect 'n' not preceded by backslash
# (position where it's not an escape sequence)
# * Note: raw string required
pattern = r'(?<!\\)n'
# Note: raw string must be used in this example

# Extract code parts excluding comments
code_lines = [
    "x = 10  # Variable initialization",
    "# This is a complete comment line",
    "y = x + 1  # Addition",
    "print(y)",
]

for line in code_lines:
    # Remove everything after # as a comment (simplified version)
    code_part = re.sub(r'\s*#.*$', '', line)
    if code_part.strip():
        print(f"  Code: {code_part.strip()}")
# => Code: x = 10
# => Code: y = x + 1
# => Code: print(y)

# Escape character handling with negative lookbehind
# Detect unescaped quotes
text = r'He said "hello" and "it\'s \"fine\""'
# Detect " that is not \"
unescaped_quotes = re.findall(r'(?<!\\)"', text)
print(f"Unescaped quotes: {len(unescaped_quotes)}")
```

### 5.4 Conditional Replacement with Negative Lookbehind

```python
import re

# Perform replacement only in specific contexts

# Example 1: Convert & that is not already HTML-entity-encoded
text = "Tom & Jerry &amp; Friends &lt;tag&gt;"
# Don't convert "&amp;", "&lt;", etc. as they're already escaped
result = re.sub(r'&(?!amp;|lt;|gt;|quot;|#\d+;)', '&amp;', text)
print(result)
# => "Tom &amp; Jerry &amp; Friends &lt;tag&gt;"

# Example 2: Auto-link URLs that are not already inside Markdown links
text = "Visit http://example.com or [click here](http://other.com)"
# Don't convert URLs already inside links
pattern = r'(?<!\()(https?://\S+)(?!\))'
result = re.sub(pattern, r'<a href="\1">\1</a>', text)
print(result)

# Example 3: Link-ify email addresses not already inside tags
text = "Contact: user@example.com <a>admin@example.com</a>"
pattern = r'(?<!>)\b[\w.+-]+@[\w-]+\.[\w.]+\b(?!<)'
result = re.sub(pattern, r'<a href="mailto:\g<0>">\g<0></a>', text)
print(result)
```

---

## 6. Lookaround Combination Patterns

### 6.1 AND Condition: Chaining Multiple Lookaheads

Placing multiple lookaheads at the same position requires all conditions to be satisfied simultaneously.

```python
import re

# Example: a string satisfying all of the following conditions
# - 8-20 characters
# - Contains uppercase
# - Contains lowercase
# - Contains a digit
# - Contains a symbol
# - No same character repeated 3+ times consecutively
pattern = re.compile(
    r'^'
    r'(?=.{8,20}$)'            # 8-20 characters
    r'(?=.*[A-Z])'              # Contains uppercase
    r'(?=.*[a-z])'              # Contains lowercase
    r'(?=.*\d)'                 # Contains a digit
    r'(?=.*[!@#$%^&*()_+=-])'  # Contains a symbol
    r'(?!.*(.)\1{2})'          # No 3 consecutive identical chars
    r'.*$'                      # Match entire string
)

test_cases = [
    ("Passw0rd!", True),
    ("weakpass", False),        # No digit/symbol/uppercase
    ("ALLCAPS1!", False),       # No lowercase
    ("Short1!", False),         # Less than 8 chars
    ("Tooooo0long!password!!", False),  # Over 20 chars
    ("Paaass0rd!", False),      # 'a' repeated 3 times
    ("C0mpl3x!Pwd", True),
]

for pw, expected in test_cases:
    result = bool(pattern.match(pw))
    status = "PASS" if result == expected else "FAIL"
    print(f"  [{status}] '{pw}' => {result} (expected: {expected})")
```

### 6.2 NOT Condition: Exclusion with Negative Lookahead

```python
import re

# Extract lines not containing specific patterns
text = """
DEBUG: Starting process
INFO: User logged in
ERROR: Connection failed
DEBUG: Processing data
WARN: Low memory
INFO: Task completed
ERROR: Timeout exceeded
"""

# Lines not containing ERROR or DEBUG
pattern = r'^(?!.*(ERROR|DEBUG)).*$'
lines = re.findall(pattern, text.strip(), re.MULTILINE)
print("Filtered results:")
for line in lines:
    if line.strip():
        print(f"  {line.strip()}")
# => INFO: User logged in
# => WARN: Low memory
# => INFO: Task completed
```

### 6.3 Position Sandwiching: Lookbehind + Lookahead

```python
import re

# Pattern: replace only words in a specific context
text = "The quick brown fox jumps over the lazy dog"

# Replace "the" with "THE", but leave "The" at the beginning as-is
# Lookbehind: preceded by a space, Lookahead: followed by a space
result = re.sub(r'(?<=\s)the(?=\s)', 'THE', text)
print(result)
# => "The quick brown fox jumps over THE lazy dog"

# Convert JSON key names (snake_case -> camelCase)
json_text = '{"user_name": "Alice", "first_name": "Alice", "last_name": "Smith"}'
# Detect underscores within keys using lookbehind, capitalize next character
def snake_to_camel(match):
    return match.group(1).upper()

result = re.sub(r'(?<="[a-z_]*)_([a-z])', snake_to_camel, json_text)
print(result)
```

### 6.4 Practical Multi-Condition Pattern Collection

```python
import re

# Pattern 1: Extract amounts by currency symbol
text = "Items: $100, EUR200, JPY15000, GBP50"
currencies = {
    'USD': re.findall(r'(?<=\$)\d+', text),
    'EUR': re.findall(r'(?<=EUR)\d+', text),
    'JPY': re.findall(r'(?<=JPY)\d+', text),
    'GBP': re.findall(r'(?<=GBP)\d+', text),
}
for currency, values in currencies.items():
    if values:
        print(f"  {currency}: {values}")

# Pattern 2: Extract content between XML/HTML tags
html = "<title>My Page</title><p>Hello World</p><span>Test</span>"

# Generic pattern: capture tag name, get content until matching closing tag
pattern = r'(?<=<(\w+)>).*?(?=</\1>)'
# Note: using capture groups inside lookbehind can cause issues in Python re
# Alternative approach:
tags = re.findall(r'<(\w+)>(.*?)</\1>', html)
for tag, content in tags:
    print(f"  <{tag}>: {content}")
# => <title>: My Page
# => <p>: Hello World
# => <span>: Test

# Pattern 3: Conditional email address extraction
# Detect email addresses outside the internal domain (@company.com)
emails = "alice@company.com bob@gmail.com carol@company.com dave@yahoo.co.jp"
external = re.findall(r'\b[\w.+-]+@(?!company\.com\b)[\w.-]+\.\w+', emails)
print(f"  External emails: {external}")
# => ['bob@gmail.com', 'dave@yahoo.co.jp']
```

---

## 7. Practical Use Cases

### 7.1 Password Strength Validation

```python
import re

def validate_password(password: str) -> tuple[bool, list[str]]:
    """Validate password strength (using lookaround)"""
    errors = []

    # 8 or more characters
    if len(password) < 8:
        errors.append("Must be at least 8 characters")

    # Contains uppercase (positive lookahead)
    if not re.search(r'(?=.*[A-Z])', password):
        errors.append("Must contain at least 1 uppercase letter")

    # Contains lowercase
    if not re.search(r'(?=.*[a-z])', password):
        errors.append("Must contain at least 1 lowercase letter")

    # Contains a digit
    if not re.search(r'(?=.*\d)', password):
        errors.append("Must contain at least 1 digit")

    # Contains a symbol
    if not re.search(r'(?=.*[!@#$%^&*])', password):
        errors.append("Must contain at least 1 symbol (!@#$%^&*)")

    return (len(errors) == 0, errors)

# As a single pattern:
strong_password = re.compile(
    r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*]).{8,}$'
)

print(strong_password.match("Passw0rd!"))  # => match
print(strong_password.match("password"))   # => None
```

### 7.2 Digit Grouping

```python
import re

# Insert commas every 3 digits
def add_commas(n: str) -> str:
    """Insert digit separators using lookahead/lookbehind"""
    return re.sub(
        r'(?<=\d)(?=(?:\d{3})+(?!\d))',
        ',',
        n
    )

print(add_commas("1234567"))     # => '1,234,567'
print(add_commas("1234567890"))  # => '1,234,567,890'
print(add_commas("42"))          # => '42' (no change)

# Pattern explanation:
# (?<=\d)           -- position preceded by a digit
# (?=(?:\d{3})+     -- followed by one or more groups of 3 digits
#   (?!\d))         -- NOT followed by another digit
# -> Insert a comma at that position
```

### 7.3 Context-Specific Replacement

```python
import re

# Replace "foo" with "bar", but exclude occurrences inside quotes
text = 'Use foo here, but "foo" stays unchanged'

# Method: negative lookbehind + negative lookahead
# (Note: this has limitations for complete in-quote detection)
result = re.sub(r'(?<!")foo(?!")', 'bar', text)
print(result)
# => 'Use bar here, but "foo" stays unchanged'
```

### 7.4 Advanced Extraction in Log Analysis

```python
import re

# Apache/Nginx access log analysis
log_line = '192.168.1.100 - admin [10/Oct/2024:13:55:36 -0700] "GET /api/v2/users?page=1 HTTP/1.1" 200 2326'

# Extract each field using lookaround
ip = re.search(r'^\S+', log_line).group()
user = re.search(r'(?<=- )\w+', log_line).group()
timestamp = re.search(r'(?<=\[)[^\]]+(?=\])', log_line).group()
method = re.search(r'(?<=")\w+', log_line).group()
path = re.search(r'(?<=(?:GET|POST|PUT|DELETE|PATCH) )\S+', log_line).group()
status = re.search(r'(?<=" )\d{3}(?= )', log_line).group()
size = re.search(r'\d+$', log_line).group()

print(f"IP: {ip}")
print(f"User: {user}")
print(f"Timestamp: {timestamp}")
print(f"Method: {method}")
print(f"Path: {path}")
print(f"Status: {status}")
print(f"Size: {size}")
```

### 7.5 Text Conversion Utilities

```python
import re

# CamelCase -> snake_case conversion
def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case using lookaround"""
    # Step 1: Insert underscore at boundaries between lowercase and uppercase
    s1 = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
    # Step 2: Boundary where consecutive uppercase is followed by lowercase
    s2 = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', s1)
    return s2.lower()

test_cases = [
    "camelCase",           # => "camel_case"
    "CamelCase",           # => "camel_case"
    "getHTTPResponse",     # => "get_http_response"
    "XMLParser",           # => "xml_parser"
    "parseJSON",           # => "parse_json"
    "myURLHandler",        # => "my_url_handler"
    "simpleTest",          # => "simple_test"
]

for tc in test_cases:
    print(f"  {tc:25s} => {camel_to_snake(tc)}")

# snake_case -> CamelCase conversion
def snake_to_camel(name: str, upper_first: bool = True) -> str:
    """Convert snake_case to CamelCase"""
    components = name.split('_')
    if upper_first:
        return ''.join(x.title() for x in components)
    else:
        return components[0] + ''.join(x.title() for x in components[1:])

test_cases_snake = [
    "camel_case",           # => "CamelCase"
    "get_http_response",    # => "GetHttpResponse"
    "xml_parser",           # => "XmlParser"
    "my_url_handler",       # => "MyUrlHandler"
]

for tc in test_cases_snake:
    print(f"  {tc:25s} => {snake_to_camel(tc)}")
```

### 7.6 Markdown Text Processing

```python
import re

# Transform text while protecting content inside Markdown inline code
markdown = "Use `foo` to call `bar()`, but foo outside code should change"

# Method: replace "foo" only outside inline code
# Step 1: Temporarily replace inline code with placeholders
placeholders = {}
counter = [0]

def save_code(match):
    key = f"\x00CODE{counter[0]}\x00"
    placeholders[key] = match.group()
    counter[0] += 1
    return key

protected = re.sub(r'`[^`]+`', save_code, markdown)

# Step 2: Replace "foo" outside placeholders
result = re.sub(r'foo', 'baz', protected)

# Step 3: Restore placeholders
for key, value in placeholders.items():
    result = result.replace(key, value)

print(result)
# => "Use `foo` to call `bar()`, but baz outside code should change"
```

### 7.7 Conditional Comment Removal

```python
import re

# Remove comments from source code (preserve comment symbols inside strings)
code = '''
x = "hello # world"  # This part is a comment
y = 'test // data'  // This is also a comment
z = 42  # A number
url = "http://example.com"  # // inside URL is preserved
'''

# Simplified version: comment removal considering string literals
def remove_comments(text: str) -> str:
    """Remove comments while preserving # and // inside string literals"""
    result = []
    for line in text.split('\n'):
        # Track whether we're inside a string literal
        in_string = False
        string_char = None
        comment_start = -1

        for i, ch in enumerate(line):
            if not in_string:
                if ch in ('"', "'"):
                    in_string = True
                    string_char = ch
                elif ch == '#':
                    comment_start = i
                    break
                elif i + 1 < len(line) and line[i:i+2] == '//':
                    comment_start = i
                    break
            else:
                if ch == string_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False

        if comment_start >= 0:
            result.append(line[:comment_start].rstrip())
        else:
            result.append(line)

    return '\n'.join(result)

print(remove_comments(code))
```

---

## 8. Lookaround in JavaScript

### 8.1 Lookbehind Support Since ES2018

```javascript
// Lookbehind available since ES2018+

// Positive lookbehind
const text1 = "Price: $100, EUR200";
console.log(text1.match(/(?<=\$)\d+/g));
// => ['100']

// Negative lookbehind
const text2 = "$100 200 $300 400";
console.log(text2.match(/(?<!\$)\b\d+/g));
// => ['200', '400']

// JavaScript supports variable-length lookbehind
const text3 = "http://example.com https://secure.example.com";
console.log(text3.match(/(?<=https?:\/\/)\w+/g));
// => ['example', 'secure']
// This pattern would cause an error in Python re
```

### 8.2 Practical Examples in JavaScript

```javascript
// Number formatting
function formatNumber(num) {
    return num.toString().replace(/(?<=\d)(?=(\d{3})+(?!\d))/g, ',');
}

console.log(formatNumber(1234567));     // => "1,234,567"
console.log(formatNumber(1234567890));  // => "1,234,567,890"

// Detect variables inside template literals
const template = "Hello ${name}, your balance is ${balance}";
const variables = template.match(/(?<=\$\{)\w+(?=\})/g);
console.log(variables);
// => ['name', 'balance']

// Password strength check
function checkPasswordStrength(password) {
    const checks = {
        length: /.{8,}/.test(password),
        uppercase: /(?=.*[A-Z])/.test(password),
        lowercase: /(?=.*[a-z])/.test(password),
        number: /(?=.*\d)/.test(password),
        special: /(?=.*[!@#$%^&*])/.test(password),
    };

    const score = Object.values(checks).filter(Boolean).length;
    return { checks, score, strong: score >= 4 };
}

console.log(checkPasswordStrength("Passw0rd!"));
// => { checks: { length: true, uppercase: true, ... }, score: 5, strong: true }
```

### 8.3 Combining Named Capture Groups with Lookaround

```javascript
// Combining ES2018 named capture groups with lookaround
const logLine = '2024-01-15T10:30:45 [ERROR] Database connection failed: timeout';

const pattern = /(?<=\[)(?<level>\w+)(?=\])/;
const match = logLine.match(pattern);
console.log(match.groups.level);
// => 'ERROR'

// Bulk extraction of multiple log levels
const logs = `
2024-01-15 [INFO] Server started
2024-01-15 [ERROR] Connection failed
2024-01-15 [WARN] Low disk space
2024-01-15 [DEBUG] Processing request
`;

const levels = [...logs.matchAll(/(?<=\[)(?<level>\w+)(?=\])/g)];
levels.forEach(m => console.log(m.groups.level));
// => INFO, ERROR, WARN, DEBUG
```

---

## 9. Lookaround in Java

### 9.1 Basic Usage

```java
import java.util.regex.*;
import java.util.*;

public class LookaroundExample {
    public static void main(String[] args) {
        // Positive lookahead
        Pattern p1 = Pattern.compile("\\d+(?=円)");
        Matcher m1 = p1.matcher("Product A: 1000円, Product B: 2500円");
        while (m1.find()) {
            System.out.println("Amount: " + m1.group());
        }
        // => Amount: 1000
        // => Amount: 2500

        // Positive lookbehind (fixed-length only)
        Pattern p2 = Pattern.compile("(?<=\\$)\\d+");
        Matcher m2 = p2.matcher("$100 $200 300");
        while (m2.find()) {
            System.out.println("USD: " + m2.group());
        }
        // => USD: 100
        // => USD: 200

        // Digit grouping
        String number = "1234567890";
        String formatted = number.replaceAll(
            "(?<=\\d)(?=(\\d{3})+(?!\\d))", ","
        );
        System.out.println(formatted);
        // => 1,234,567,890
    }
}
```

### 9.2 Java-Specific Notes

```java
// Java lookbehind is fixed-length only
// The following causes a compile error

try {
    Pattern.compile("(?<=\\w+)\\d+");
    // => PatternSyntaxException: Look-behind group does not have
    //    an obvious maximum length
} catch (PatternSyntaxException e) {
    System.out.println("Error: " + e.getMessage());
}

// Workaround: use fixed-length alternatives
// (?<=\\w{1}|\\w{2}|\\w{3})\\d+  -- lookbehind of 1-3 characters
// However, this approach is impractical,
// so using capture groups is better

// Java 13+ has some improvements for variable-length lookbehind
// But officially only fixed-length is supported
```

---

## 10. ASCII Diagrams

### 10.1 The 4 Types of Lookaround in Action

```
Text: "$100"

Positive lookahead (?=\d):
  Position: $ [here] 1 0 0
  "Is there \d to the right?" -> 1 is there -> success

Negative lookahead (?!\$):
  Position: [here] $ 1 0 0
  "Is there no \$ to the right?" -> $ is there -> failure
  Position: $ [here] 1 0 0
  "Is there no \$ to the right?" -> 1 is not $ -> success

Positive lookbehind (?<=\$):
  Position: $ [here] 1 0 0
  "Is there \$ to the left?" -> $ is there -> success

Negative lookbehind (?<!\$):
  Position: [here] $ 1 0 0
  "Is there no \$ to the left?" -> nothing there -> success
  Position: $ [here] 1 0 0
  "Is there no \$ to the left?" -> $ is there -> failure
```

### 10.2 Password Validation Lookahead Chain

```
Pattern: ^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*]).{8,}$

Input: "Passw0rd!"

Position 0 (start of string):
  (?=.*[A-Z])     -> lookahead: "P" is uppercase -> success (position resets)
  (?=.*[a-z])     -> lookahead: "a" is lowercase -> success (position resets)
  (?=.*\d)        -> lookahead: "0" is a digit -> success (position resets)
  (?=.*[!@#$%^&*])-> lookahead: "!" is a symbol -> success (position resets)
  .{8,}$          -> "Passw0rd!" is 9 chars >= 8 -> success

All lookaheads start from the same position 0
(zero-width, so position does not advance)
```

### 10.3 Digit Grouping Lookahead in Action

```
Input: "1234567"
Pattern: (?<=\d)(?=(?:\d{3})+(?!\d))

Checking each position:

  1 | 2 3 4 5 6 7
    ^
  (?<=\d): 1 is there -> OK
  (?=(?:\d{3})+(?!\d)): "234567" = 3 digits x 2 + no digit after -> OK
  -> Comma insertion point!

  1 2 | 3 4 5 6 7
      ^
  (?<=\d): 2 is there -> OK
  (?=(?:\d{3})+(?!\d)): "34567" = 3 digits x 1 + "67" remainder -> NG
  -> Skip

  1 2 3 | 4 5 6 7
        ^
  (?<=\d): 3 is there -> OK
  (?=(?:\d{3})+(?!\d)): "4567" = 3 digits x 1 + "7" remainder -> NG
  -> Skip

  1 2 3 4 | 5 6 7
          ^
  (?<=\d): 4 is there -> OK
  (?=(?:\d{3})+(?!\d)): "567" = 3 digits x 1 + no digit after -> OK
  -> Comma insertion point!

Result: "1,234,567"
```

### 10.4 Nested Lookaround in Action

```
Pattern: (?<=(?<=[A-Z])[a-z])\d
Text: "Ab1Cd2ef3"

Checking: 3 conditions checked at each position

Position 2 '1':
  Outer lookbehind (?<=...): check position 1
    Position 1 'b' is [a-z] -> OK
    Inner lookbehind (?<=[A-Z]): check position 0
      Position 0 'A' is [A-Z] -> OK
  -> All conditions passed! '1' matches

Position 5 '2':
  Outer lookbehind (?<=...): check position 4
    Position 4 'd' is [a-z] -> OK
    Inner lookbehind (?<=[A-Z]): check position 3
      Position 3 'C' is [A-Z] -> OK
  -> All conditions passed! '2' matches

Position 8 '3':
  Outer lookbehind (?<=...): check position 7
    Position 7 'f' is [a-z] -> OK
    Inner lookbehind (?<=[A-Z]): check position 6
      Position 6 'e' is [A-Z]? -> NO!
  -> Inner lookbehind fails! '3' does not match

Result: ['1', '2']
Meaning: the digit part of "uppercase -> lowercase -> digit" patterns
```

---

## 11. Comparison Tables

### 11.1 Complete Lookaround Comparison

| Type | Syntax | Meaning | Example | Match |
|------|--------|---------|---------|-------|
| Positive lookahead | `X(?=Y)` | X followed by Y | `\d+(?=円)` | "100" in "100円" |
| Negative lookahead | `X(?!Y)` | X NOT followed by Y | `\d+(?!円)` | "200" in "200ドル" |
| Positive lookbehind | `(?<=Y)X` | X preceded by Y | `(?<=\$)\d+` | "100" in "$100" |
| Negative lookbehind | `(?<!Y)X` | X NOT preceded by Y | `(?<!\$)\d+` | "42" in "count: 42" |

### 11.2 Language Support Status

| Feature | Python | JavaScript | Java | Go(RE2) | Rust | .NET | Perl | Ruby |
|---------|--------|------------|------|---------|------|------|------|------|
| Positive lookahead `(?=)` | OK | OK | OK | N/A | N/A | OK | OK | OK |
| Negative lookahead `(?!)` | OK | OK | OK | N/A | N/A | OK | OK | OK |
| Positive lookbehind `(?<=)` | OK(fixed) | OK(variable) | OK(fixed) | N/A | N/A | OK(variable) | OK(fixed) | OK(fixed) |
| Negative lookbehind `(?<!)` | OK(fixed) | OK(variable) | OK(fixed) | N/A | N/A | OK(variable) | OK(fixed) | OK(fixed) |
| Variable-length lookbehind | regex module | ES2018+ | N/A | N/A | fancy-regex | Standard | N/A | N/A |

### 11.3 Lookaround vs Capture Groups

| Comparison | Lookaround | Capture Groups |
|------------|-----------|----------------|
| Included in match? | No (zero-width) | Yes |
| Behavior in replacement | Preserves surrounding text | Can be referenced as a group |
| Performance | Slightly slower (backtracking) | Generally faster |
| Readability | Tends to be complex | Relatively readable |
| Use case | Positional condition specification | Substring extraction |
| AND conditions | Achievable by chaining | Not possible alone |
| Engine compatibility | Highly engine-dependent | Nearly all engines support |

```python
import re

# Comparison example: two approaches for the same result

text = "Price: $100, $200, $300"

# Approach 1: lookbehind (get number without $)
result1 = re.findall(r'(?<=\$)\d+', text)
print(f"Lookbehind: {result1}")   # => ['100', '200', '300']

# Approach 2: capture group
result2 = re.findall(r'\$(\d+)', text)
print(f"Capture:    {result2}")   # => ['100', '200', '300']

# Results are the same, but replacement behavior differs:
# Lookbehind replacement: $ is preserved
result3 = re.sub(r'(?<=\$)\d+', 'XXX', text)
print(f"Lookbehind replace: {result3}")
# => "Price: $XXX, $XXX, $XXX"

# Capture group replacement: $ must also be specified
result4 = re.sub(r'\$\d+', '$XXX', text)
print(f"Group replace:      {result4}")
# => "Price: $XXX, $XXX, $XXX"
```

---

## 12. Performance Considerations

### 12.1 Cost of Lookaround

```python
import re
import time

# Lookaround requires sub-pattern evaluation at each position
# This can affect performance on large texts

def benchmark(name, pattern, text, iterations=10000):
    compiled = re.compile(pattern)
    start = time.perf_counter()
    for _ in range(iterations):
        compiled.findall(text)
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed:.4f}s ({iterations} iterations)")

text = "The quick brown fox jumps over the lazy dog " * 100

# Simple pattern vs lookaround
benchmark("Simple match  ", r'\b\w+\b', text)
benchmark("With lookahead", r'\b\w+(?=\s)', text)
benchmark("With lookbehind", r'(?<=\s)\w+', text)
benchmark("Both combined ", r'(?<=\s)\w+(?=\s)', text)
```

### 12.2 Catastrophic Backtracking with Lookaround

```python
import re

# Examples of dangerous patterns

# Dangerous: nested lookahead with repetition
# pattern = r'(?=.*a)(?=.*b)(?=.*c).+'
# Can become exponentially slow with longer inputs

# Safe: place specific patterns after each lookahead
safe_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$'
# Each lookahead is evaluated independently, .{8,} runs in linear time

# Patterns with ReDoS risk
# Quantifiers inside lookaheads like (?=.*a+)(?=.*b+)
# can cause backtracking

# Countermeasure: make patterns inside lookaheads as specific as possible
# BAD: (?=.*a+)
# OK:  (?=.*a)
# OK:  (?=[^a]*a)  -- use negated character class for efficiency
```

### 12.3 Optimization Techniques

```python
import re

# Technique 1: Speed up lookahead with negated character classes
text = "abc123def456ghi789"

# Slow: .* tries the entire string then backtracks
slow = r'(?=.*\d)\w+'
# Fast: [^\d]* skips only non-digit characters
fast = r'(?=[^\d]*\d)\w+'

# Technique 2: Atomic groups (for supported engines)
# Not supported in Python re; available in the regex module
# (?>pattern) prevents backtracking

# Technique 3: Optimize lookahead order
# Place conditions most likely to fail first for early failure
# Example: in password validation, check symbols first
# (since passwords without symbols are most common)
password_pattern = re.compile(
    r'^'
    r'(?=.*[!@#$%^&*])'  # Symbol check (most likely to fail)
    r'(?=.*\d)'            # Digit check
    r'(?=.*[A-Z])'         # Uppercase check
    r'(?=.*[a-z])'         # Lowercase check
    r'.{8,}$'
)

# Technique 4: Reuse compiled patterns
# Patterns with lookaround have especially high compilation costs
# Always pre-compile with re.compile()
compiled = re.compile(r'(?<=\$)\d+(?=\.\d{2})')
# Use compiled.findall(text) inside loops
```

---

## 13. Advanced Techniques

### 13.1 Conditional Patterns (Lookaround Applications)

```python
import re

# Context-dependent replacement
text = "foo_bar baz_qux FOO_BAR"

# If first character is uppercase, convert entire word to uppercase; if lowercase, capitalize
def context_aware_replace(match):
    word = match.group()
    parts = word.split('_')
    if parts[0][0].isupper():
        return ''.join(p.upper() for p in parts)
    else:
        return ''.join(p.capitalize() for p in parts)

result = re.sub(r'\w+(?:_\w+)+', context_aware_replace, text)
print(result)
# => "FooBar BazQux FOOBAR"
```

### 13.2 Simulating Recursive Patterns

```python
import re

# Python re doesn't support recursive patterns,
# but some can be simulated with lookaround

# Split at commas outside nested parentheses
text = "a(b,c),d,(e,f(g,h)),i"

# Step 1: Track parenthesis nesting level and split
def split_at_top_level(text: str, delimiter: str = ',') -> list[str]:
    """Split at top-level delimiters (ignoring content inside parentheses)"""
    result = []
    current = []
    depth = 0

    for ch in text:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == delimiter and depth == 0:
            result.append(''.join(current))
            current = []
        else:
            current.append(ch)

    result.append(''.join(current))
    return result

print(split_at_top_level(text))
# => ['a(b,c)', 'd', '(e,f(g,h))', 'i']
```

### 13.3 Complex Validation with Multiple Lookarounds

```python
import re

def validate_credit_card(number: str) -> dict:
    """Credit card number validation (using lookaround)"""
    # Remove spaces and hyphens
    clean = re.sub(r'[\s-]', '', number)

    result = {
        'number': clean,
        'valid_format': False,
        'card_type': 'Unknown',
        'luhn_valid': False,
    }

    # Card type detection (checking number patterns with lookahead)
    card_patterns = {
        'Visa': r'^4\d{12}(?:\d{3})?$',
        'MasterCard': r'^5[1-5]\d{14}$',
        'AmEx': r'^3[47]\d{13}$',
        'Discover': r'^6(?:011|5\d{2})\d{12}$',
        'JCB': r'^(?:2131|1800|35\d{3})\d{11}$',
    }

    for card_type, pattern in card_patterns.items():
        if re.match(pattern, clean):
            result['card_type'] = card_type
            result['valid_format'] = True
            break

    # Luhn algorithm validation
    if result['valid_format']:
        digits = [int(d) for d in clean]
        checksum = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
        result['luhn_valid'] = (checksum % 10 == 0)

    return result

# Test
test_cards = [
    "4111 1111 1111 1111",   # Visa test card
    "5500 0000 0000 0004",   # MasterCard test card
    "3400 000000 00009",     # AmEx test card
    "1234 5678 9012 3456",   # Invalid
]

for card in test_cards:
    result = validate_credit_card(card)
    print(f"  {card}: {result['card_type']} "
          f"(format: {result['valid_format']}, luhn: {result['luhn_valid']})")
```

### 13.4 Using Lookaround for Text Tokenization

```python
import re

# Advanced tokenization using lookaround
# Split at character type boundaries

def tokenize_mixed(text: str) -> list[str]:
    """Split at boundaries between alphabetic, numeric, and Japanese characters"""
    # Insert spaces at character type boundaries
    # Boundary between letters and digits
    result = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)
    result = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', result)
    # Boundary between alphanumeric and Japanese characters
    result = re.sub(r'(?<=[a-zA-Z0-9])(?=[\u3040-\u9fff])', ' ', result)
    result = re.sub(r'(?<=[\u3040-\u9fff])(?=[a-zA-Z0-9])', ' ', result)

    return result.split()

test_cases = [
    "Hello123World",        # => ['Hello', '123', 'World']
    "test42data",           # => ['test', '42', 'data']
    "Hello世界2024",        # => ['Hello', '世界', '2024']
    "Python3プログラミング", # => ['Python', '3', 'プログラミング']
]

for tc in test_cases:
    tokens = tokenize_mixed(tc)
    print(f"  '{tc}' => {tokens}")
```

### 13.5 CSV Parser Using Lookaround

```python
import re

def parse_csv_field(line: str) -> list[str]:
    """CSV field parsing leveraging lookaround

    Correctly handles commas inside double-quoted fields
    """
    fields = []
    # Handle both quoted and unquoted fields
    pattern = re.compile(
        r'"([^"]*(?:""[^"]*)*)"|'  # Inside double quotes ("" is escape)
        r'([^,]*)'                  # Unquoted field
    )

    pos = 0
    while pos <= len(line):
        m = pattern.match(line, pos)
        if m:
            if m.group(1) is not None:
                # Quoted field: convert "" to "
                fields.append(m.group(1).replace('""', '"'))
            else:
                fields.append(m.group(2))
            pos = m.end()
            # Skip comma
            if pos < len(line) and line[pos] == ',':
                pos += 1
            elif pos >= len(line):
                break
        else:
            break

    return fields

# Test
test_lines = [
    'Alice,30,Tokyo',
    '"Bob ""Jr""",25,"New York, NY"',
    '"contains,comma",normal,"also ""quoted"""',
]

for line in test_lines:
    fields = parse_csv_field(line)
    print(f"  Input:  {line}")
    print(f"  Result: {fields}")
    print()
```

---

## 14. Lookaround Alternatives for Go and Rust

### 14.1 Alternative Approaches in Go (RE2)

```go
package main

import (
    "fmt"
    "regexp"
    "strings"
)

func main() {
    // Go's RE2 engine does not support lookaround
    // Use capture groups as an alternative

    // Alternative 1: instead of (?<=\$)\d+
    text := "Price: $100, $200, 300"
    re := regexp.MustCompile(`\$(\d+)`)
    matches := re.FindAllStringSubmatch(text, -1)
    for _, m := range matches {
        fmt.Println("Amount:", m[1]) // Capture group 1
    }
    // => Amount: 100
    // => Amount: 200

    // Alternative 2: digit grouping (without lookaround)
    number := "1234567890"
    formatted := addCommas(number)
    fmt.Println("Formatted:", formatted)
    // => Formatted: 1,234,567,890

    // Alternative 3: password validation (individual checks)
    password := "Passw0rd!"
    fmt.Println("Password strength:", validatePassword(password))
}

func addCommas(s string) string {
    // Implement digit grouping without lookaround
    n := len(s)
    if n <= 3 {
        return s
    }

    var result strings.Builder
    remainder := n % 3
    if remainder > 0 {
        result.WriteString(s[:remainder])
        if remainder < n {
            result.WriteByte(',')
        }
    }
    for i := remainder; i < n; i += 3 {
        if i > remainder {
            result.WriteByte(',')
        }
        result.WriteString(s[i : i+3])
    }
    return result.String()
}

func validatePassword(pw string) bool {
    // Check each condition individually (alternative to lookahead)
    hasUpper := regexp.MustCompile(`[A-Z]`).MatchString(pw)
    hasLower := regexp.MustCompile(`[a-z]`).MatchString(pw)
    hasDigit := regexp.MustCompile(`\d`).MatchString(pw)
    hasSpecial := regexp.MustCompile(`[!@#$%^&*]`).MatchString(pw)
    hasLength := len(pw) >= 8

    return hasUpper && hasLower && hasDigit && hasSpecial && hasLength
}
```

### 14.2 Alternative Approaches in Rust

```rust
use regex::Regex;

fn main() {
    // Rust's regex crate does not support lookaround
    // fancy-regex crate can be used for support

    // Standard regex alternative: capture groups
    let re = Regex::new(r"\$(\d+)").unwrap();
    let text = "Price: $100, $200, 300";

    for cap in re.captures_iter(text) {
        println!("Amount: {}", &cap[1]);
    }
    // => Amount: 100
    // => Amount: 200

    // Password validation: individual checks
    let password = "Passw0rd!";
    println!("Valid: {}", validate_password(password));
}

fn validate_password(pw: &str) -> bool {
    let has_upper = Regex::new(r"[A-Z]").unwrap().is_match(pw);
    let has_lower = Regex::new(r"[a-z]").unwrap().is_match(pw);
    let has_digit = Regex::new(r"\d").unwrap().is_match(pw);
    let has_special = Regex::new(r"[!@#$%^&*]").unwrap().is_match(pw);
    let has_length = pw.len() >= 8;

    has_upper && has_lower && has_digit && has_special && has_length
}

// To use fancy-regex:
// [dependencies]
// fancy-regex = "0.11"
//
// use fancy_regex::Regex;
//
// fn with_lookaround() {
//     let re = Regex::new(r"(?<=\$)\d+").unwrap();
//     let text = "Price: $100, $200";
//     for m in re.find_iter(text) {
//         if let Ok(m) = m {
//             println!("Amount: {}", m.as_str());
//         }
//     }
// }
```

---

## 15. Test-Driven Pattern Development

### 15.1 Unit Testing Lookaround Patterns

```python
import re
import unittest

class TestLookaroundPatterns(unittest.TestCase):
    """Test suite for lookaround patterns"""

    def test_positive_lookahead_yen(self):
        """Positive lookahead: extract numbers before yen"""
        pattern = re.compile(r'\d+(?=円)')
        self.assertEqual(pattern.findall("1000円"), ['1000'])
        self.assertEqual(pattern.findall("1000ドル"), [])
        self.assertEqual(pattern.findall("1000円と2000円"), ['1000', '2000'])
        self.assertEqual(pattern.findall("円1000"), [])

    def test_negative_lookahead_exclusion(self):
        """Negative lookahead: exclude specific words"""
        pattern = re.compile(r'\b(?!test\b)\w+')
        words = pattern.findall("test hello testing world")
        self.assertIn('hello', words)
        self.assertIn('testing', words)
        self.assertIn('world', words)
        self.assertNotIn('test', words)

    def test_positive_lookbehind_dollar(self):
        """Positive lookbehind: extract numbers after $"""
        pattern = re.compile(r'(?<=\$)\d+')
        self.assertEqual(pattern.findall("$100 $200 300"), ['100', '200'])
        self.assertEqual(pattern.findall("100 200"), [])

    def test_negative_lookbehind_no_dollar(self):
        """Negative lookbehind: extract numbers without $"""
        pattern = re.compile(r'(?<!\$)\b\d+')
        self.assertEqual(pattern.findall("$100 200 $300 400"), ['200', '400'])

    def test_password_validation(self):
        """Password validation pattern"""
        pattern = re.compile(
            r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*]).{8,}$'
        )
        self.assertIsNotNone(pattern.match("Passw0rd!"))
        self.assertIsNone(pattern.match("password"))
        self.assertIsNone(pattern.match("SHORT1!"))
        self.assertIsNone(pattern.match("nouppercase1!"))
        self.assertIsNone(pattern.match("NOLOWERCASE1!"))
        self.assertIsNone(pattern.match("NoDigits!!"))
        self.assertIsNone(pattern.match("NoSpecial1a"))

    def test_comma_formatting(self):
        """Digit grouping format"""
        pattern = re.compile(r'(?<=\d)(?=(?:\d{3})+(?!\d))')

        def add_commas(n):
            return pattern.sub(',', n)

        self.assertEqual(add_commas("1234567"), "1,234,567")
        self.assertEqual(add_commas("42"), "42")
        self.assertEqual(add_commas("1000"), "1,000")
        self.assertEqual(add_commas("1234567890"), "1,234,567,890")

    def test_camel_to_snake(self):
        """CamelCase -> snake_case conversion"""
        def camel_to_snake(name):
            s1 = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
            s2 = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', s1)
            return s2.lower()

        self.assertEqual(camel_to_snake("camelCase"), "camel_case")
        self.assertEqual(camel_to_snake("CamelCase"), "camel_case")
        self.assertEqual(camel_to_snake("getHTTPResponse"), "get_http_response")
        self.assertEqual(camel_to_snake("XMLParser"), "xml_parser")
        self.assertEqual(camel_to_snake("simple"), "simple")

if __name__ == '__main__':
    unittest.main()
```

### 15.2 Testing Edge Cases

```python
import re

# Comprehensive lookaround edge cases
def test_edge_cases():
    """Comprehensively test edge cases"""

    # 1. Empty string
    pattern = re.compile(r'(?=\d)\d+')
    assert pattern.findall("") == []

    # 2. Lookaround at string start/end
    # Lookbehind at start: always fails (nothing before)
    assert re.findall(r'(?<=x)\w+', "abc") == []
    # When matching character is at the start
    assert re.findall(r'(?<=x)\w+', "xabc") == ['abc']

    # 3. Multiline support
    text = "line1\nline2\nline3"
    # ^ matches each line start (with re.MULTILINE)
    assert re.findall(r'(?<=^)line\d', text, re.MULTILINE) == ['line1', 'line2', 'line3']

    # 4. Lookaround with Unicode characters
    text = "Price: 100円, 200 dollars"
    assert re.findall(r'\d+(?=円)', text) == ['100']
    assert re.findall(r'(?<=: )\d+', text) == ['100']

    # 5. Overlapping lookaround conditions
    # Contradictory conditions at the same position: always fails
    assert re.findall(r'(?=a)(?=b)', "ab") == []
    # Conditions that can both be satisfied at the same position
    assert len(re.findall(r'(?=a)(?!b)', "a")) == 1

    # 6. Consecutive zero-width matches
    text = "abc"
    # Matches at all positions (4 positions: 0,1,2,3)
    assert len(re.findall(r'(?=.)?', text)) >= 3

    print("All edge case tests passed!")

test_edge_cases()
```

---

## 16. Anti-Patterns

### 16.1 Anti-Pattern: Overuse of Lookaround

```python
import re

# BAD: Using lookaround for a simple condition
pattern_bad = r'(?<=price: )\d+'
# ^ Extraction is possible without lookbehind

# GOOD: A capture group is sufficient
pattern_good = r'price: (\d+)'
match = re.search(pattern_good, "price: 100")
print(match.group(1))  # => '100'

# Scenarios where lookaround is genuinely needed:
# - When you want to preserve surrounding text during replacement
# - When you need to combine multiple positional conditions with AND
# - When you don't want specific strings included in the match result
```

### 16.2 Anti-Pattern: Assuming Variable-Length Lookbehind

```python
import re

# BAD: Using variable-length lookbehind in Python re
try:
    re.search(r'(?<=https?://)\w+', "https://example.com")
except re.error as e:
    print(f"Error: {e}")
    # => look-behind requires fixed-width pattern
    # "https?" is 4 or 5 characters -> variable-length

# OK: Enumerate each fixed length with OR
pattern = r'(?<=http://|https://)\w+'
# This also causes an error in Python re (different-length alternatives)
# * In Python 3.6+, alternatives with different fixed lengths are allowed

# Workaround 1: Different approach
pattern = r'https?://(\w+)'
match = re.search(pattern, "https://example.com")
print(match.group(1))  # => 'example'

# Workaround 2: regex module (supports variable-length lookbehind)
# import regex
# regex.search(r'(?<=https?://)\w+', "https://example.com")
```

### 16.3 Anti-Pattern: Trying to Solve Everything with Lookaround

```python
import re

# BAD: Overly complex lookaround chain
# "Words starting with uppercase, containing a digit, 5-10 chars, not containing 'test'"
bad_pattern = r'\b(?=[A-Z])(?=\w*\d)(?!\w*test)(?=\w{5,10}\b)\w+'

# GOOD: Process in stages
def find_valid_words(text: str) -> list[str]:
    """Apply multiple conditions in stages"""
    words = re.findall(r'\b\w+\b', text)
    result = []
    for word in words:
        if not (5 <= len(word) <= 10):
            continue
        if not word[0].isupper():
            continue
        if not re.search(r'\d', word):
            continue
        if 'test' in word.lower():
            continue
        result.append(word)
    return result

# Staged processing is:
# - More readable
# - Easier to debug
# - Each condition can be tested independently
# - Performance is comparable (for short texts)
```

### 16.4 Anti-Pattern: Getting the Lookaround Direction Wrong

```python
import re

# Common beginner mistakes

# Wrong: want to extract "100" from "100円"
# Lookahead and lookbehind are reversed
try:
    wrong = re.findall(r'(?=円)\d+', "100円")
    print(f"Wrong: {wrong}")  # => [] -- nothing matches!
except:
    pass

# Correct: "円" is to the right of the number -> use lookahead
correct = re.findall(r'\d+(?=円)', "100円")
print(f"Correct: {correct}")  # => ['100']

# How to remember:
# Lookahead  = look forward  = check the right side  = (?=...) or (?!...)
# Lookbehind = look backward = check the left side   = (?<=...) or (?<!...)
#
# "ahead" = the direction we're going   = right side
# "behind" = the direction we came from = left side
```

### 16.5 Anti-Pattern: Group Reference Mistakes in Replacement

```python
import re

# Capture groups in lookaround can be referenced, but be careful

text = "old_value: 100"

# BAD: trying to reference captures inside lookaround
# Lookaround doesn't consume, so it's not part of the replacement target
result = re.sub(r'(?<=old_value: )(\d+)', r'NEW_\1', text)
print(result)
# => "old_value: NEW_100"  -- this works, but...

# Note: putting capture groups inside lookbehind
# can lead to unexpected behavior
# It's safer to capture in the main pattern instead
result = re.sub(r'(old_value: )(\d+)', r'\1NEW_\2', text)
print(result)
# => "old_value: NEW_100"  -- more explicit
```

---

## 17. FAQ

### Q1: Why is lookaround called "zero-width"?

**A**: Lookaround only checks a "position" in the string and does not "consume" characters. This means it is not included in the match result and the engine's current position does not advance. It is a type of "assertion" just like `^` and `\b`:

```python
import re
text = "100円200ドル"
# (?=円) only checks the position -- 円 itself is available for the next match
for m in re.finditer(r'\d+(?=円)', text):
    print(f"Position {m.start()}-{m.end()}: '{m.group()}'")
# => Position 0-3: '100'
# "円" is not included in the match
```

### Q2: Can lookaround be nested?

**A**: Yes. It is possible to put another lookaround inside a lookaround:

```python
import re
# "Character at a position where the previous char is uppercase and the next is a digit"
pattern = r'(?<=(?<=[A-Z])\w)(?=\d)'
# This is complex, so it's usually recommended to decompose into simpler patterns
```

However, nesting significantly reduces readability, so it is recommended to avoid complex lookaround nesting and instead split into multiple patterns or handle it with program logic.

### Q3: Why is lookaround not available in Go and Rust?

**A**: Go's RE2 and Rust's regex crate use **DFA-based engines** that prioritize O(n) linear time guarantees. Lookaround can require backtracking, which is incompatible with linear time guarantees. In Rust, the `fancy-regex` crate provides an NFA engine with lookaround support, but the O(n) guarantee is lost:

```rust
// Rust standard regex: lookaround not available
// use regex::Regex;

// fancy-regex: lookaround supported
// use fancy_regex::Regex;
// let re = Regex::new(r"(?<=\$)\d+").unwrap();
```

### Q4: Can capture groups be used inside lookaround?

**A**: Yes, but with caveats:

```python
import re

# Capture group inside lookahead
text = "100円 200ドル 300ユーロ"
pattern = r'\d+(?=(円|ドル|ユーロ))'
matches = re.findall(pattern, text)
print(matches)
# => ['円', 'ドル', 'ユーロ']
# Note: findall returns the capture group contents

# When you need both the number and the group
pattern = r'(\d+)(?=(円|ドル|ユーロ))'
matches = re.findall(pattern, text)
print(matches)
# => [('100', '円'), ('200', 'ドル'), ('300', 'ユーロ')]
```

### Q5: What is the difference between lookaround and `\b`?

**A**: `\b` is a fixed positional condition for "word boundaries", while lookaround is a general-purpose assertion that can use any pattern as a positional condition:

```python
import re

text = "hello world 123"

# \b: detects boundaries between word and non-word characters
print(re.findall(r'\b\w+\b', text))
# => ['hello', 'world', '123']

# (?=...): checks any condition to the right
# Example: "words at positions followed by a space or end"
print(re.findall(r'\w+(?=\s|$)', text))
# => ['hello', 'world', '123']

# \b is fixed but fast
# Lookaround is flexible but slightly slower
```

### Q6: Can lookaround be used in POSIX regular expressions?

**A**: No. POSIX BRE/ERE does not support lookaround. It is a PCRE (Perl Compatible Regular Expressions) feature. Use `grep -P` to access PCRE:

```bash
# POSIX ERE (grep -E): lookaround not available
# grep -E '(?<=\$)\d+' file.txt  # Error

# PCRE (grep -P): lookaround available
grep -P '(?<=\$)\d+' file.txt

# macOS grep may not support -P
# In that case, install ggrep (GNU grep):
# brew install grep
# ggrep -P '(?<=\$)\d+' file.txt
```

### Q7: Can lookahead be placed inside another lookahead?

**A**: Yes. Lookaround can be nested freely:

```python
import re

# Lookahead inside a lookahead
# "Position where there is a digit to the right, and to the right of that digit is a letter"
text = "a1b c2d e3 4f"
pattern = r'(?=\d(?=[a-z]))\d'
print(re.findall(pattern, text))
# => ['1', '2']
# '3' doesn't match (followed by a space)
# '4' doesn't match (the lookahead (?=\d...) checks the position,
# needing \d(?=[a-z]) to the right of that position)

# Practical use cases are rare, but theoretically nestable to any depth
# However, it's recommended to avoid for readability
```

### Q8: What is the relationship between lookaround and atomic groups?

**A**: Atomic groups `(?>...)` are groups that prohibit backtracking, and they are related to the internal behavior of lookaround. Matching inside lookaround is essentially atomic (once success/failure is determined, it is not reversed):

```python
# Atomic groups not supported in Python re
# Available in the regex module:
# import regex
# pattern = regex.compile(r'(?>abc|ab)c')
# regex.search(pattern, "abc")  # Doesn't match
# Normal (abc|ab)c would match "abc"

# Lookaround internals are always atomic:
# (?=abc|ab) -- once it succeeds with "abc", it doesn't try "ab"
# This is beneficial for performance but can cause unexpected behavior
```

---

## 18. Debugging and Troubleshooting

### 18.1 Debugging Lookaround

```python
import re

# Method 1: Build patterns incrementally
text = "Price: $100.50, Tax: $15.00, Total: 115.50"

# Step 1: First verify matches without lookaround
print("Step 1:", re.findall(r'\d+\.\d{2}', text))
# => ['100.50', '15.00', '115.50']

# Step 2: Add lookbehind
print("Step 2:", re.findall(r'(?<=\$)\d+\.\d{2}', text))
# => ['100.50', '15.00']

# Step 3: Add lookahead
print("Step 3:", re.findall(r'(?<=\$)\d+(?=\.\d{2})', text))
# => ['100', '15']

# Method 2: Commented patterns with verbose mode
pattern = re.compile(r'''
    (?<=\$)         # Position preceded by $ (lookbehind)
    \d+             # 1+ digits (integer part)
    (?=\.\d{2})     # Position followed by .XX (lookahead)
''', re.VERBOSE)

matches = pattern.findall(text)
print("Verbose:", matches)

# Method 3: Verify position info with finditer
for m in re.finditer(r'(?<=\$)\d+\.\d{2}', text):
    print(f"  Match: '{m.group()}' at [{m.start()}:{m.end()}]")
```

### 18.2 Common Errors and Solutions

```python
import re

# Error 1: look-behind requires fixed-width pattern
try:
    re.compile(r'(?<=\w+)\d+')
except re.error as e:
    print(f"Error 1: {e}")
# Solution: use capture groups
# re.findall(r'\w+(\d+)', text)

# Error 2: lookahead not working as intended
text = "abc123"
# Intent: get alphabetic characters before a digit
wrong = re.findall(r'(?=\d)[a-z]+', text)
print(f"Wrong: {wrong}")  # => []
# Reason: at the position of (?=\d), [a-z] cannot match
correct = re.findall(r'[a-z]+(?=\d)', text)
print(f"Correct: {correct}")  # => ['abc']

# Error 3: negative lookahead matching too much
text = "foo foobar foobaz"
wrong = re.findall(r'(?!foo)\w+', text)
print(f"Wrong: {wrong}")  # => unexpected results
# Reason: from position 1, 'oo', 'oobar', etc. also match
correct = re.findall(r'\b(?!foo)\w+', text)
print(f"Improved: {correct}")  # => [] -- all words start with foo

# Error 4: lookaround range mistake
text = "12ab34cd56"
# Intent: extract digits sandwiched between alphabetic chars
wrong = re.findall(r'(?<=[a-z])\d+(?=[a-z])', text)
print(f"Result: {wrong}")  # => ['34']
# "12" has no alphabetic char before it, "56" has none after
# If intended, this is fine; adjust conditions if "12" should also be included
```

### 18.3 Debugging with regex101

```
regex101.com is extremely useful for debugging lookaround:

1. Go to https://regex101.com/
2. Select the language (Python, JavaScript, Java, etc.) in the top left
3. Enter the pattern
4. Enter test strings
5. Check the pattern explanation in "EXPLANATION" on the right
6. Check match details in "MATCH INFORMATION" at the bottom
7. Trace step-by-step behavior in "REGEX DEBUGGER"

The REGEX DEBUGGER is particularly useful as it lets you visually
confirm success/failure at each position during lookaround evaluation.
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in real-world work?

Knowledge of this topic is frequently used in day-to-day development work. It is particularly important during code reviews and architecture design.

---

## Summary

| Item | Description |
|------|-------------|
| `(?=X)` | Positive lookahead -- position where X is to the right |
| `(?!X)` | Negative lookahead -- position where X is NOT to the right |
| `(?<=X)` | Positive lookbehind -- position where X is to the left |
| `(?<!X)` | Negative lookbehind -- position where X is NOT to the left |
| Zero-width | Does not consume characters (checks position only) |
| Lookbehind constraint | Fixed-length only in many engines |
| AND condition | Achieved by chaining multiple lookaheads |
| NOT condition | Achieved with negative lookahead/lookbehind |
| Primary use cases | Password validation, digit grouping, conditional extraction/replacement |
| DFA engines | Lookaround not supported (RE2, Rust regex) |
| Design guideline | Avoid when a simpler pattern can achieve the same result |
| Performance | Optimize lookahead order; leverage negated character classes |
| Testing | Incremental pattern construction; comprehensive edge case coverage |

## Recommended Next Guides

- [02-unicode-regex.md](./02-unicode-regex.md) -- Unicode Regular Expressions
- [03-performance.md](./03-performance.md) -- Performance and ReDoS Prevention

## References

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- Chapter 5: "Lookaround"
2. **MDN - Lookahead assertion** https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions/Lookahead_assertion -- JavaScript lookaround specification
3. **Regular-Expressions.info - Lookaround** https://www.regular-expressions.info/lookaround.html -- Comprehensive lookaround explanation and cross-engine comparison
4. **Python re module documentation** https://docs.python.org/3/library/re.html -- Python standard library lookaround specification
5. **TC39 Proposal - Lookbehind Assertions** https://github.com/tc39/proposal-regexp-lookbehind -- JavaScript ES2018 lookbehind proposal
6. **RE2 Syntax** https://github.com/google/re2/wiki/Syntax -- RE2 (Go) supported syntax list and reasons for no lookaround support
7. **fancy-regex crate** https://docs.rs/fancy-regex/ -- Rust crate for using lookaround
8. **regex101.com** https://regex101.com/ -- Online tool useful for debugging lookaround
