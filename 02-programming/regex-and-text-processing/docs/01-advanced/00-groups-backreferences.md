# Groups and Backreferences -- Capturing, Named Groups, Lookahead/Lookbehind

> Grouping bundles subexpressions of a pattern, and backreferences reuse matched substrings. Accurately understand the differences between capture groups, non-capture groups, and named groups, and apply them in replacement, extraction, and validation. This guide also covers advanced group syntax including lookahead/lookbehind, atomic groups, and conditional branching patterns.

## What You Will Learn

1. **Capture Groups and Non-Capture Groups** -- The difference between `(...)` and `(?:...)`, and their impact on performance
2. **Named Groups** -- Designing readable patterns with `(?P<name>...)`
3. **Backreferences and Replacement** -- Reusing match results with `\1` and `\k<name>`
4. **Lookahead and Lookbehind (Lookaround)** -- Zero-width assertions with `(?=...)`, `(?!...)`, `(?<=...)`, `(?<!...)`
5. **Atomic Groups** -- Suppressing backtracking with `(?>...)` for performance optimization
6. **Conditional Branching Patterns** -- Conditional matching with `(?(id)yes|no)`


## Prerequisites

The following knowledge will help you get the most out of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Capture Groups `(...)`

### 1.1 Basic Usage

```python
import re

# Extract year, month, and day individually from a date pattern
pattern = r'(\d{4})-(\d{2})-(\d{2})'
text = "Today is 2026-02-11"

match = re.search(pattern, text)
print(match.group(0))  # => '2026-02-11' (entire match)
print(match.group(1))  # => '2026'       (group 1: year)
print(match.group(2))  # => '02'         (group 2: month)
print(match.group(3))  # => '11'         (group 3: day)
print(match.groups())  # => ('2026', '02', '11')
```

### 1.2 Group Numbering

```
Pattern: ((A)(B(C)))

Group number assignment (in order of opening parenthesis appearance):

  (  (  A  )  (  B  (  C  )  )  )
  ^  ^        ^     ^
  1  2        3     4

  Group 0: entire match
  Group 1: ((A)(B(C)))  = "ABC"
  Group 2: (A)          = "A"
  Group 3: (B(C))       = "BC"
  Group 4: (C)          = "C"
```

```python
import re

pattern = r'((A)(B(C)))'
match = re.search(pattern, "ABC")

print(match.group(0))  # => 'ABC'
print(match.group(1))  # => 'ABC'
print(match.group(2))  # => 'A'
print(match.group(3))  # => 'BC'
print(match.group(4))  # => 'C'
```

### 1.3 Combining Groups with Alternation

```python
import re

# Alternation (|) inside a group
pattern = r'(cat|dog|bird)s?'
text = "I have 2 cats and 3 dogs"

matches = re.findall(pattern, text)
print(matches)  # => ['cat', 'dog']
# Note: findall returns group contents when groups are present

# When you need the full match rather than just group contents
matches_full = re.finditer(pattern, text)
for m in matches_full:
    print(f"  Full: {m.group(0)}, Group 1: {m.group(1)}")
# => Full: cats, Group 1: cat
# => Full: dogs, Group 1: dog
```

### 1.4 Practical Example with Nested Groups

```python
import re

# Extracting HTML attributes: class="value" or class='value'
html = '<div class="main container" id="app" data-role=\'admin\'>'
pattern = r'(\w+)=((["\'])(.*?)\3)'

for m in re.finditer(pattern, html):
    print(f"  Attribute: {m.group(1)}, Value: {m.group(4)}, Quote: {m.group(3)}")
# => Attribute: class, Value: main container, Quote: "
# => Attribute: id, Value: app, Quote: "
# => Attribute: data-role, Value: admin, Quote: '
```

### 1.5 Using Multiple Groups Simultaneously

```python
import re

# Log analysis: extract timestamp, level, and message in one pass
log_pattern = re.compile(
    r'\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]\s+'  # Group 1: timestamp
    r'\[(DEBUG|INFO|WARN|ERROR|FATAL)\]\s+'              # Group 2: log level
    r'\[(\w+)\]\s+'                                      # Group 3: module name
    r'(.*)'                                              # Group 4: message
)

log_lines = [
    "[2026-02-11 10:30:45] [ERROR] [AuthModule] Login failed for user admin",
    "[2026-02-11 10:31:00] [INFO] [Database] Connection pool initialized (size=10)",
    "[2026-02-11 10:31:15] [WARN] [Cache] Cache miss rate exceeds 50%",
]

for line in log_lines:
    m = log_pattern.search(line)
    if m:
        ts, level, module, msg = m.groups()
        print(f"  {ts} | {level:5s} | {module:12s} | {msg}")
```

### 1.6 Capture Groups in JavaScript

```javascript
// Capture groups in JavaScript
const text = "2026-02-11 Error at 10:30:45";
const pattern = /(\d{4})-(\d{2})-(\d{2})/;

const match = text.match(pattern);
console.log(match[0]);  // '2026-02-11' (full match)
console.log(match[1]);  // '2026' (group 1)
console.log(match[2]);  // '02' (group 2)
console.log(match[3]);  // '11' (group 3)

// Retrieving all groups with matchAll (ES2020)
const logPattern = /\[(\w+)\]\s+(\w+)/g;
const logText = "[ERROR] AuthFailed [WARN] HighLoad";

for (const m of logText.matchAll(logPattern)) {
    console.log(`Level: ${m[1]}, Message: ${m[2]}`);
}
// => Level: ERROR, Message: AuthFailed
// => Level: WARN, Message: HighLoad
```

### 1.7 Capture Groups in Go

```go
package main

import (
    "fmt"
    "regexp"
)

func main() {
    text := "2026-02-11 Error at 10:30:45"
    re := regexp.MustCompile(`(\d{4})-(\d{2})-(\d{2})`)

    // FindStringSubmatch: returns result with submatches
    match := re.FindStringSubmatch(text)
    if match != nil {
        fmt.Printf("Full: %s, Year: %s, Month: %s, Day: %s\n",
            match[0], match[1], match[2], match[3])
    }

    // FindAllStringSubmatch: submatches for all matches
    allMatches := re.FindAllStringSubmatch(text, -1)
    for _, m := range allMatches {
        fmt.Printf("  Match: %v\n", m)
    }
}
```

### 1.8 Capture Groups in Rust

```rust
use regex::Regex;

fn main() {
    let text = "2026-02-11 Error at 10:30:45";
    let re = Regex::new(r"(\d{4})-(\d{2})-(\d{2})").unwrap();

    // captures: match with capture groups
    if let Some(caps) = re.captures(text) {
        println!("Full: {}", &caps[0]);
        println!("Year: {}", &caps[1]);
        println!("Month: {}", &caps[2]);
        println!("Day: {}", &caps[3]);
    }

    // captures_iter: captures for all matches
    for caps in re.captures_iter(text) {
        println!("  Match: {}", &caps[0]);
    }
}
```

---

## 2. Non-Capture Groups `(?:...)`

### 2.1 Grouping Without Capturing

```python
import re

# Capture group -- a group number is assigned
pattern_capture = r'(https?)://([\w.]+)'
match = re.search(pattern_capture, "https://example.com")
print(match.group(1))  # => 'https'
print(match.group(2))  # => 'example.com'

# Non-capture group -- no group number is assigned
pattern_noncapture = r'(?:https?)://([\w.]+)'
match = re.search(pattern_noncapture, "https://example.com")
print(match.group(1))  # => 'example.com' (numbering doesn't shift)
# match.group(2) -> error (group 2 does not exist)
```

### 2.2 Guidelines for Choosing Between Them

```
When to use capture groups:
  * You want to use the matched substring later (extraction)
  * You need backreferences (\1, \2)
  * You want to reference in replacement (\1 or $1)

When to use non-capture groups:
  * Grouping is needed for quantifiers or alternation but the value is not needed
  * You want to gain a slight performance improvement
  * You don't want group numbers to shift
```

```python
import re

# BAD: Unnecessary capturing -- group numbers increase needlessly
pattern_bad = r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{2}) (Jan|Feb|Mar) (\d{4})'
# Groups: 1=weekday, 2=day, 3=month, 4=year
# If weekday (group 1) is not needed, numbers shift unnecessarily

# GOOD: Non-capture for unnecessary parts
pattern_good = r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{2}) (Jan|Feb|Mar) (\d{4})'
# Groups: 1=day, 2=month, 3=year -- only needed parts get numbered
```

### 2.3 Performance Measurement

```python
import re
import timeit

text = "The quick brown fox jumps over the lazy dog" * 1000

# Capture group version
pattern_capture = re.compile(r'(\w+)\s+(\w+)\s+(\w+)\s+(\w+)')

# Non-capture group version
pattern_noncapture = re.compile(r'(?:\w+)\s+(?:\w+)\s+(?:\w+)\s+(?:\w+)')

# Benchmark
t_capture = timeit.timeit(
    lambda: pattern_capture.findall(text), number=1000
)
t_noncapture = timeit.timeit(
    lambda: pattern_noncapture.findall(text), number=1000
)

print(f"Capture version:     {t_capture:.4f}s")
print(f"Non-capture version: {t_noncapture:.4f}s")
print(f"Speed difference: {(t_capture - t_noncapture) / t_capture * 100:.1f}%")
# Non-capture version is typically 5-15% faster
```

### 2.4 Using Non-Capture Groups in Complex Patterns

```python
import re

# DateTime pattern: structured with non-capture, capturing only needed parts
datetime_pattern = re.compile(r'''
    (?P<date>                          # Full date (named capture)
        (?P<year>\d{4})                # Year (named capture)
        [-/]                           # Separator (no grouping needed)
        (?P<month>\d{2})               # Month (named capture)
        [-/]                           # Separator
        (?P<day>\d{2})                 # Day (named capture)
    )
    (?:\s+|T)                          # Date-time separator (non-capture)
    (?P<time>                          # Full time (named capture)
        (?P<hour>\d{2})                # Hour (named capture)
        :(?P<minute>\d{2})             # Minute (named capture)
        (?::(?P<second>\d{2}))?        # Second (optional, named capture)
    )
    (?:\s*(?P<tz>[A-Z]{3,4}|[+-]\d{2}:?\d{2}))?  # Timezone (optional)
''', re.VERBOSE)

test_strings = [
    "2026-02-11 10:30:45 JST",
    "2026/02/11T10:30",
    "2026-02-11 10:30:45+09:00",
]

for s in test_strings:
    m = datetime_pattern.search(s)
    if m:
        print(f"  Input: {s}")
        print(f"    Date: {m.group('date')}, Time: {m.group('time')}")
        print(f"    Year: {m.group('year')}, Month: {m.group('month')}, Day: {m.group('day')}")
        tz = m.group('tz')
        if tz:
            print(f"    TZ: {tz}")
```

---

## 3. Named Groups

### 3.1 Syntax (by Language)

```
+------------+----------------------+------------------+
| Language   | Definition           | Reference        |
+------------+----------------------+------------------+
| Python     | (?P<name>...)        | (?P=name), \g<name>|
| Perl       | (?<name>...)         | \k<name>          |
| Java       | (?<name>...)         | \k<name>          |
| .NET       | (?<name>...)         | \k<name>          |
| JavaScript | (?<name>...)         | \k<name>          |
| Go (RE2)   | (?P<name>...)        | (no backreference)|
| Rust       | (?P<name>...)        | (no backreference)|
+------------+----------------------+------------------+
```

### 3.2 Named Groups in Python

```python
import re

# Parse a date using named groups
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
text = "Date: 2026-02-11"

match = re.search(pattern, text)

# Access by name
print(match.group('year'))   # => '2026'
print(match.group('month'))  # => '02'
print(match.group('day'))    # => '11'

# Get as dictionary with groupdict()
print(match.groupdict())
# => {'year': '2026', 'month': '02', 'day': '11'}

# Access by number is also possible
print(match.group(1))  # => '2026'
```

### 3.3 Named Groups in JavaScript (ES2018+)

```javascript
// Named groups in JavaScript
const pattern = /(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})/;
const text = "Date: 2026-02-11";

const match = text.match(pattern);
console.log(match.groups);
// => { year: '2026', month: '02', day: '11' }
console.log(match.groups.year);   // => '2026'

// Combining with destructuring
const { year, month, day } = match.groups;
console.log(`${year}/${month}/${day}`);
// => '2026/02/11'

// Named group reference in replace
const result = "2026-02-11".replace(
    /(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})/,
    '$<day>/$<month>/$<year>'
);
console.log(result);  // => '11/02/2026'

// String.prototype.replaceAll + named groups (ES2021)
const multiDates = "Start: 2026-02-11, End: 2026-03-15";
const formatted = multiDates.replaceAll(
    /(?<y>\d{4})-(?<m>\d{2})-(?<d>\d{2})/g,
    '$<d>/$<m>/$<y>'
);
console.log(formatted);
// => 'Start: 11/02/2026, End: 15/03/2026'
```

### 3.4 Named Groups in Go

```go
package main

import (
    "fmt"
    "regexp"
)

func main() {
    // Named groups in Go: (?P<name>...)
    re := regexp.MustCompile(`(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})`)
    text := "Date: 2026-02-11"

    match := re.FindStringSubmatch(text)
    if match == nil {
        return
    }

    // Get names with SubexpNames()
    result := make(map[string]string)
    for i, name := range re.SubexpNames() {
        if i != 0 && name != "" {
            result[name] = match[i]
        }
    }

    fmt.Printf("Year: %s, Month: %s, Day: %s\n",
        result["year"], result["month"], result["day"])
    // => Year: 2026, Month: 02, Day: 11

    // Organized as a helper function
    fmt.Println(extractNamedGroups(re, text))
}

// Generic helper function
func extractNamedGroups(re *regexp.Regexp, text string) map[string]string {
    match := re.FindStringSubmatch(text)
    if match == nil {
        return nil
    }
    result := make(map[string]string)
    for i, name := range re.SubexpNames() {
        if i != 0 && name != "" {
            result[name] = match[i]
        }
    }
    return result
}
```

### 3.5 Named Groups in Rust

```rust
use regex::Regex;
use std::collections::HashMap;

fn main() {
    let re = Regex::new(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})").unwrap();
    let text = "Date: 2026-02-11";

    if let Some(caps) = re.captures(text) {
        // Access by name
        println!("Year: {}", &caps["year"]);    // => Year: 2026
        println!("Month: {}", &caps["month"]);   // => Month: 02
        println!("Day: {}", &caps["day"]);     // => Day: 11

        // Get Option<Match> with the name() method
        if let Some(year) = caps.name("year") {
            println!("Year position: {}-{}", year.start(), year.end());
        }
    }

    // Convert named groups from all matches to HashMap
    let results: Vec<HashMap<&str, &str>> = re.captures_iter(text)
        .map(|caps| {
            re.capture_names()
                .flatten()
                .filter_map(|name| {
                    caps.name(name).map(|m| (name, m.as_str()))
                })
                .collect()
        })
        .collect();

    println!("{:?}", results);
}
```

### 3.6 Naming Conventions for Named Groups

```
Recommended naming conventions:
+--------------------------------------------------------+
| * Lowercase letters + underscores (snake_case)         |
|   e.g., (?P<first_name>...) (?P<area_code>...)        |
|                                                        |
| * Clear, meaningful names                              |
|   e.g., (?P<protocol>https?) (?P<port>\d{1,5})        |
|                                                        |
| x Avoid:                                               |
|   - Names too short: (?P<p>...) (?P<x>...)             |
|   - Number-like names: (?P<group1>...) (?P<g2>...)     |
|   - Hyphens: (?P<first-name>...) -> syntax error       |
|   - Reserved-word-like: (?P<class>...) (?P<type>...)   |
+--------------------------------------------------------+
```

---

## 4. Backreferences

### 4.1 Backreferences Within a Pattern

```python
import re

# \1 references the match from group 1
# Detect repetitions of the same string

# Matching HTML opening and closing tags
pattern = r'<(\w+)>.*?</\1>'
text = '<div>hello</div> <span>world</span>'

matches = re.findall(pattern, text)
print(matches)  # => ['div', 'span']

# Detecting duplicate words
pattern = r'\b(\w+)\s+\1\b'
text = "the the quick brown fox fox"
print(re.findall(pattern, text))  # => ['the', 'fox']
```

### 4.2 Named Backreferences

```python
import re

# (?P=name) backreferences a named group
pattern = r'(?P<quote>["\']).*?(?P=quote)'
text = """He said "hello" and 'world'"""

matches = re.findall(pattern, text)
print(matches)  # => ['"', "'"]

# Use finditer to get the full match
for m in re.finditer(pattern, text):
    print(m.group())
# => "hello"
# => 'world'
```

### 4.3 Backreferences in Replacement

```python
import re

# \1 or \g<1> references a group during replacement
text = "2026-02-11"

# Date format conversion: YYYY-MM-DD -> DD/MM/YYYY
result = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3/\2/\1', text)
print(result)  # => '11/02/2026'

# Replacement with named groups
result = re.sub(
    r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})',
    r'\g<day>/\g<month>/\g<year>',
    text
)
print(result)  # => '11/02/2026'

# Advanced replacement using a function
def format_date(match):
    y, m, d = match.group('year'), match.group('month'), match.group('day')
    return f"{y}/{m}/{d}"

result = re.sub(
    r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})',
    format_date,
    text
)
print(result)  # => '2026/02/11'
```

### 4.4 Applied Backreference Patterns

```python
import re

# 1. Palindrome detection (3-5 character palindromes)
palindrome_3 = r'\b(\w)(\w)\2\1\b'  # 4-character palindrome
text = "abba deed noon hello"
for m in re.finditer(palindrome_3, text):
    print(f"  Palindrome: {m.group()}")
# => Palindrome: abba
# => Palindrome: deed
# => Palindrome: noon

# 2. XML/HTML matching tag detection (no nesting)
xml_tag = r'<(?P<tag>\w+)(?:\s[^>]*)?>(?P<content>.*?)</(?P=tag)>'
html = '<p class="intro">Hello</p> <div>World</div>'
for m in re.finditer(xml_tag, html):
    print(f"  Tag: {m.group('tag')}, Content: {m.group('content')}")

# 3. Quoted fields in CSV
csv_field = r'(?P<quote>["\'])(?P<value>(?:(?!(?P=quote)).)*|(?:(?P=quote){2})*)(?P=quote)'
csv_line = '"hello","world","it""s a test"'
for m in re.finditer(csv_field, csv_line):
    print(f"  Value: {m.group('value')}")

# 4. Detecting repeated characters (password checks, etc.)
repeated_char = r'(.)\1{2,}'  # Same character repeated 3+ times
passwords = ["abc", "aabbc", "aaabbb", "password111"]
for pwd in passwords:
    m = re.search(repeated_char, pwd)
    if m:
        print(f"  FAIL: {pwd} ('{m.group(1)}' repeated {len(m.group())} times)")
```

### 4.5 Backreferences in JavaScript

```javascript
// Backreferences in JavaScript

// 1. Backreference within a pattern
const html = '<div>hello</div> <span>world</span>';
const tagPattern = /<(\w+)>.*?<\/\1>/g;
let m;
while ((m = tagPattern.exec(html)) !== null) {
    console.log(`  Match: ${m[0]}, Tag: ${m[1]}`);
}

// 2. Named backreference (ES2018)
const quotePattern = /(?<q>["']).*?\k<q>/g;
const text = `He said "hello" and 'world'`;
for (const match of text.matchAll(quotePattern)) {
    console.log(`  Match: ${match[0]}`);
}

// 3. Backreference in replacement
const date = "2026-02-11";
console.log(date.replace(
    /(\d{4})-(\d{2})-(\d{2})/,
    '$3/$2/$1'
));  // => '11/02/2026'

// Named group replacement
console.log(date.replace(
    /(?<y>\d{4})-(?<m>\d{2})-(?<d>\d{2})/,
    '$<d>/$<m>/$<y>'
));  // => '11/02/2026'
```

---

## 5. Lookahead and Lookbehind (Lookaround)

Lookahead and lookbehind are called zero-width assertions -- they check conditions without consuming characters.

### 5.1 Lookaround Syntax Overview

```
+--------------------+--------------+------------------------------+
| Type               | Syntax       | Meaning                      |
+--------------------+--------------+------------------------------+
| Positive lookahead | (?=...)      | Matches position followed by ...|
| Negative lookahead | (?!...)      | Matches position NOT followed by ...|
| Positive lookbehind| (?<=...)     | Matches position preceded by ...|
| Negative lookbehind| (?<!...)     | Matches position NOT preceded by ...|
+--------------------+--------------+------------------------------+

Note: Go (RE2) and Rust (regex) do NOT support lookahead/lookbehind.
      -> fancy-regex (Rust) or regexp2 (Go) can be used as alternatives.
```

### 5.2 Positive Lookahead `(?=...)`

```python
import re

# Match positions "followed by a specific pattern"

# 1. Detect currency symbols before amounts (numbers are not consumed)
text = "$100 EUR200 JPY300"
pattern = r'$\u20ac\u00a5'
for m in re.finditer(pattern, text):
    print(f"  Currency symbol: {m.group()} at {m.start()}")

# 2. Password strength check: simultaneously checking multiple conditions
# Lookahead lets you check multiple conditions without consuming the pattern
password_pattern = re.compile(r'''
    ^
    (?=.*[A-Z])        # Contains uppercase
    (?=.*[a-z])        # Contains lowercase
    (?=.*\d)           # Contains a digit
    (?=.*[!@#$%^&*])   # Contains a symbol
    .{8,}              # 8 or more characters
    $
''', re.VERBOSE)

passwords = ["MyP@ss1", "MyP@ssw0rd", "password", "PASSWORD1!", "Ab1!abcd"]
for pwd in passwords:
    result = "OK" if password_pattern.match(pwd) else "FAIL"
    print(f"  {result}: {pwd}")

# 3. Inserting commas every 3 digits
def add_commas(number_str):
    """Insert commas every 3 digits using lookahead"""
    return re.sub(r'(?<=\d)(?=(\d{3})+(?!\d))', ',', number_str)

print(add_commas("1234567890"))   # => '1,234,567,890'
print(add_commas("12345"))        # => '12,345'
print(add_commas("123"))          # => '123'
```

### 5.3 Negative Lookahead `(?!...)`

```python
import re

# Match positions "NOT followed by a specific pattern"

# 1. Filenames with extensions other than .exe
filenames = ["report.pdf", "virus.exe", "photo.jpg", "setup.exe", "data.csv"]
pattern = r'\w+\.(?!exe\b)\w+'
for f in filenames:
    m = re.fullmatch(pattern, f)
    if m:
        print(f"  Safe: {f}")
# => Safe: report.pdf
# => Safe: photo.jpg
# => Safe: data.csv

# 2. Identifiers that are not reserved words
reserved = "if|else|for|while|return|class|def"
identifier_pattern = re.compile(rf'\b(?!(?:{reserved})\b)[a-zA-Z_]\w*\b')
code = "if x > 0: return calculate(x) else: count = 0"
identifiers = identifier_pattern.findall(code)
print(f"  Identifiers: {identifiers}")
# => Identifiers: ['x', 'calculate', 'x', 'count']

# 3. URL extraction excluding specific domains
urls = [
    "https://example.com/page",
    "https://spam.example.net/malware",
    "https://trusted.org/resource",
    "https://ads.tracker.com/pixel",
]
blocked_domains = r'spam\.example\.net|ads\.tracker\.com'
safe_url_pattern = re.compile(rf'https?://(?!{blocked_domains})[^\s]+')
for url in urls:
    m = safe_url_pattern.match(url)
    if m:
        print(f"  Allowed: {url}")
```

### 5.4 Positive Lookbehind `(?<=...)`

```python
import re

# Match positions "preceded by a specific pattern"

# 1. Extract amounts after currency symbols
text = "$100 EUR200 JPY300 free"
dollar_amounts = re.findall(r'(?<=\$)\d+', text)
print(f"  Dollar amounts: {dollar_amounts}")  # => ['100']

# 2. Extract @mentions
tweet = "Hello @alice and @bob, check out @charlie's work"
mentions = re.findall(r'(?<=@)\w+', tweet)
print(f"  Mentions: {mentions}")  # => ['alice', 'bob', 'charlie']

# 3. Extract JSON values (value after a key name)
json_text = '"name": "Alice", "age": 30, "city": "Tokyo"'
name_value = re.search(r'(?<="name":\s*")[^"]+', json_text)
if name_value:
    print(f"  name value: {name_value.group()}")  # => Alice
```

### 5.5 Negative Lookbehind `(?<!...)`

```python
import re

# Match positions "NOT preceded by a specific pattern"

# 1. Detect unescaped quotes
text = r'He said \"hello\" and "world"'
# Exclude \" and match only "
unescaped_quotes = re.findall(r'(?<!\\)"', text)
print(f"  Unescaped quote count: {len(unescaped_quotes)}")

# 2. URL paths without protocol prefix (i.e., not preceded by http://)
paths = ["/api/users", "http://example.com/api", "/api/items", "https://x.com"]
path_only = re.compile(r'(?<!https?)(?<!:)/\w[\w/]*')

# 3. Detect extra spaces that are not leading indentation
code = "def foo():\n    return bar  + baz"
# 2+ consecutive spaces (excluding line beginnings)
extra_spaces = re.findall(r'(?<=\S)\s{2,}(?=\S)', code)
print(f"  Extra space occurrences: {len(extra_spaces)}")
```

### 5.6 Combining Lookahead and Lookbehind

```python
import re

# Combine lookahead and lookbehind to extract only enclosed content

# 1. Extract only the content inside HTML tags (excluding the tags themselves)
html = "<b>bold</b> and <i>italic</i>"
content = re.findall(r'(?<=<\w+>).*?(?=</\w+>)', html)
print(f"  Tag content: {content}")  # => ['bold', 'italic']

# 2. Extract only content inside parentheses
text = "Function foo(x, y) calls bar(z)"
args = re.findall(r'(?<=\()[^)]+(?=\))', text)
print(f"  Arguments: {args}")  # => ['x, y', 'z']

# 3. Insert spaces at word boundaries in camelCase
camel = "getUserNameFromDatabase"
result = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', camel)
print(f"  Converted: {result}")  # => 'get User Name From Database'

# Convert to snake_case
snake = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', camel).lower()
print(f"  snake_case: {snake}")  # => 'get_user_name_from_database'
```

### 5.7 Lookaround Language Support

```
+--------------------+--------+------------+-----+------+------+
| Feature            | Python | JavaScript | Go  | Rust | Java |
+--------------------+--------+------------+-----+------+------+
| Positive lookahead (?=)  | OK | OK     | N/A | N/A  | OK   |
| Negative lookahead (?!)  | OK | OK     | N/A | N/A  | OK   |
| Positive lookbehind(fixed)| OK | OK    | N/A | N/A  | OK   |
| Positive lookbehind(var) | N/A*| OK (V8)| N/A | N/A  | N/A  |
| Negative lookbehind(fixed)| OK | OK    | N/A | N/A  | OK   |
| Negative lookbehind(var) | N/A*| OK (V8)| N/A | N/A  | N/A  |
+--------------------+--------+------------+-----+------+------+
| Alt. crate/package | regex  | --         |regexp2|fancy-| --  |
|                    | module |            |      |regex |      |
+--------------------+--------+------------+-----+------+------+

* Python's third-party regex module supports variable-length lookbehind
```

### 5.8 Lookaround in JavaScript (ES2018+)

```javascript
// JavaScript (ES2018) also supports variable-length lookbehind

// Positive lookbehind (variable-length)
const text1 = "USD100 EUR200 JPY3000";
const amounts = text1.match(/(?<=USD|EUR|JPY)\d+/g);
console.log(amounts);  // => ['100', '200', '3000']
// Note: USD (3 chars) and JPY (3 chars) are the same length,
// but JavaScript allows variable-length lookbehind

// Negative lookbehind + positive lookahead combined
const code = "let x = 10; const y = 20; var z = 30;";
// Get only variable names declared with const/let (exclude var)
const modernVars = code.match(/(?<=(?:const|let)\s+)\w+/g);
console.log(modernVars);  // => ['x', 'y']
```

---

## 6. Atomic Groups `(?>...)`

### 6.1 What Are Atomic Groups?

Atomic groups prohibit backtracking once a portion has matched. This prevents catastrophic backtracking.

```
Normal group:
  Pattern: (a+)b
  Input:   aaac

  Attempt 1: capture "aaa" -> b not found
  Backtrack: capture "aa" -> b not found
  Backtrack: capture "a" -> b not found
  -> Failure (3 backtracks)

Atomic group:
  Pattern: (?>a+)b
  Input:   aaac

  Attempt 1: capture "aaa" -> b not found
  -> Immediate failure (no backtracking)
  -> Completed in 1 attempt
```

### 6.2 Support Status

```
+----------+------------------------------------------------------+
| Language | Atomic Group Support                                  |
+----------+------------------------------------------------------+
| Perl     | (?>...) supported                                    |
| Java 20+ | (?>...) supported (added in Java 20)                 |
| Java <20 | Not supported -> use possessive quantifiers (*+, ++, ?+) |
| .NET     | (?>...) supported                                    |
| Python   | re: not supported / regex module: supported           |
| JavaScript| Not supported                                       |
| Go       | Not supported (not needed due to RE2 base)            |
| Rust     | Not supported (not needed due to DFA base)            |
+----------+------------------------------------------------------+
```

### 6.3 Relationship with Possessive Quantifiers

```
Atomic groups and possessive quantifiers are equivalent:

  (?>a+)    ==  a++     (1 or more, no backtracking)
  (?>a*)    ==  a*+     (0 or more, no backtracking)
  (?>a?)    ==  a?+     (0 or 1, no backtracking)
  (?>a{2,5}) == a{2,5}+ (2-5 times, no backtracking)

Languages supporting possessive quantifiers:
  * Java
  * Perl 5.10+
  x Python (re)
  x JavaScript
  x Go
  x Rust
```

### 6.4 Catastrophic Backtracking: Examples and Countermeasures

```python
import re
import time

# Dangerous pattern: (a+)+ causes catastrophic backtracking
dangerous_pattern = re.compile(r'(a+)+b')

# Short input: fast
text_short = "aaaaab"
start = time.time()
dangerous_pattern.search(text_short)
print(f"  Short input: {time.time() - start:.4f}s")

# Long non-matching input: exponential time
# WARNING: the line below is extremely slow to execute (n=25 takes several seconds)
# text_long = "a" * 25 + "c"
# dangerous_pattern.search(text_long)  # Dangerous! Takes seconds to minutes

# Safe alternative pattern
safe_pattern = re.compile(r'a+b')  # Remove nested groups
# Or use non-capture + possessive quantifiers (in languages that support them)
```

```java
// Possessive quantifier countermeasure in Java
import java.util.regex.*;

public class AtomicExample {
    public static void main(String[] args) {
        // Dangerous: catastrophic backtracking
        // Pattern dangerous = Pattern.compile("(a+)+b");

        // Safe: possessive quantifier
        Pattern safe = Pattern.compile("a++b");

        String input = "a".repeat(30) + "c";
        long start = System.nanoTime();
        safe.matcher(input).find();
        long elapsed = System.nanoTime() - start;
        System.out.printf("  Elapsed: %.3f ms%n", elapsed / 1e6);
        // => Completes instantly
    }
}
```

---

## 7. Conditional Branching Pattern `(?(id)yes|no)`

### 7.1 Basic Syntax

```
(?(id)yes-pattern|no-pattern)

id:           Group number or name to reference
yes-pattern:  Pattern to use if the group matched
no-pattern:   Pattern to use if the group did not match (optional)
```

### 7.2 Practical Examples

```python
import re

# 1. If there is an opening parenthesis, require a closing parenthesis
pattern = r'(\()?hello(?(1)\))'
# (?(1)\)) = if group 1 matched, require \)

print(re.search(pattern, "hello").group())    # => 'hello'
print(re.search(pattern, "(hello)").group())  # => '(hello)'
print(re.search(pattern, "(hello").group())   # => 'hello' (matches the unparenthesized version)

# 2. Quote matching check
# If there is an opening quote, require the same closing quote
quote_pattern = r'(?P<q>["\'])?(?P<content>\w+)(?(q)(?P=q))'
test_strings = ['"hello"', "'world'", 'plain', '"mismatch\'']
for s in test_strings:
    m = re.search(quote_pattern, s)
    if m:
        print(f"  {s} -> content: {m.group('content')}")

# 3. Format changes based on optional prefix
# If +81 is present, international format; otherwise, domestic format
phone_pattern = r'(\+81)?-?(?(1)\d{1,4}-\d{1,4}-\d{4}|0\d{1,4}-\d{1,4}-\d{4})'
phones = ["+81-90-1234-5678", "090-1234-5678", "+81-3-1234-5678", "03-1234-5678"]
for p in phones:
    m = re.match(phone_pattern, p)
    if m:
        print(f"  Match: {m.group()}")
```

### 7.3 Conditional Branching with Named Groups

```python
import re

# Conditional branching using named groups
# Email display name: "Name <email>" or standalone "email"
pattern = r'(?:(?P<display_name>[^<]+)\s+)?<(?P<email>[^>]+)>(?(display_name)|\s*(?P<email_only>[^\s]+))?'

# More practical example: tag format or plain format
# <tag attr="val">content</tag> or plain text
tag_or_plain = r'(?P<open><(?P<tagname>\w+)[^>]*>)?(?(open)(?P<content>.*?)</(?P=tagname)>|(?P<plain>.+))'
tests = ["<b>bold text</b>", "plain text", "<a href='url'>link</a>"]
for t in tests:
    m = re.match(tag_or_plain, t)
    if m:
        if m.group('open'):
            print(f"  Tag: {m.group('tagname')}, Content: {m.group('content')}")
        else:
            print(f"  Plain: {m.group('plain')}")
```

### 7.4 Conditional Branching Support Status

```
+----------+--------------------------------------+
| Language | (?(id)yes|no) Support                |
+----------+--------------------------------------+
| Python   | OK (standard support in re module)   |
| Perl     | OK                                   |
| .NET     | OK                                   |
| Java     | Not supported                        |
| JavaScript| Not supported                       |
| Go       | Not supported                        |
| Rust     | Not supported                        |
+----------+--------------------------------------+
```

---

## 8. ASCII Diagrams

### 8.1 Nested Group Structure

```
Pattern: ((\d{4})-(\d{2})-(\d{2}))T((\d{2}):(\d{2}):(\d{2}))

Input:   2026-02-11T10:30:45

Group structure:
+--- Group 1: 2026-02-11 ------------------------------------------+
| +- Group 2: 2026 --+   +- Group 3: 02 --+  +- Group 4: 11 --+   |
| |    \d{4}          | - |    \d{2}        |- |    \d{2}        |  |
| |    2026           |   |    02           |  |    11           |  |
| +-------------------+   +----------------+  +----------------+   |
+-------------------------------------------------------------------+
                           T
+--- Group 5: 10:30:45 --------------------------------------------+
| +- Group 6: 10 --+   +- Group 7: 30 --+  +- Group 8: 45 --+    |
| |    \d{2}        | : |    \d{2}        |: |    \d{2}        |   |
| |    10           |   |    30           |  |    45           |   |
| +-----------------+   +----------------+  +----------------+    |
+-------------------------------------------------------------------+
```

### 8.2 How Backreferences Work

```
Pattern: <(\w+)>.*?</\1>
Input:   <div>hello</div>

Step 1: < matches
Step 2: (\w+) captures "div" -> Group 1 = "div"
Step 3: > matches
Step 4: .*? matches "hello" (lazy)
Step 5: </ matches
Step 6: \1 -> compare with group 1 ("div") -> "div" matches
Step 7: > matches

Result: <div>hello</div>

If the input were <div>hello</span>:
  Step 6: \1 ("div") != "span" -> failure -> backtrack
```

### 8.3 Internal Behavior: Capture vs Non-Capture

```
Capture group (pattern):
+------------------------------------------+
|  Pattern: (a)(b)(c)                      |
|                                          |
|  Engine internal state:                  |
|  +----------------------+               |
|  | Group array:          |               |
|  |  [0] = "abc" (full)  |  <- allocated  |
|  |  [1] = "a"           |  <- allocated  |
|  |  [2] = "b"           |  <- allocated  |
|  |  [3] = "c"           |  <- allocated  |
|  +----------------------+               |
+------------------------------------------+

Non-capture group (?:pattern):
+------------------------------------------+
|  Pattern: (?:a)(b)(?:c)                  |
|                                          |
|  Engine internal state:                  |
|  +----------------------+               |
|  | Group array:          |               |
|  |  [0] = "abc" (full)  |  <- allocated  |
|  |  [1] = "b"           |  <- allocated  |
|  +----------------------+               |
|  -> Less memory usage                    |
+------------------------------------------+
```

### 8.4 Lookahead Operation Flow

```
Pattern: \d+(?=円)
Input:   "100円 200ドル 300円"

+------------------------------------------------------+
| Position 0: "1"                                      |
|   \d+ -> matches "100"                               |
|   (?=円) -> next char is "円" -> lookahead success!  |
|   -> "100" added to results (* "円" is not consumed) |
|                                                      |
| Position 5: "2"                                      |
|   \d+ -> matches "200"                               |
|   (?=円) -> next char is "ド" -> lookahead fails     |
|   -> backtrack -> "20" -> fail -> "2" -> fail        |
|                                                      |
| Position 10: "3"                                     |
|   \d+ -> matches "300"                               |
|   (?=円) -> next char is "円" -> lookahead success!  |
|   -> "300" added to results                          |
+------------------------------------------------------+

Result: ["100", "300"]
```

### 8.5 Lookbehind Operation Flow

```
Pattern: (?<=\$)\d+
Input:   "$100 €200 $300"

+------------------------------------------------------+
| Position 0: "$"                                      |
|   -> not a digit -> skip                             |
|                                                      |
| Position 1: "1"                                      |
|   (?<=\$) -> preceding char is "$" -> success!       |
|   \d+ -> matches "100"                               |
|   -> "100" added to results                          |
|                                                      |
| Position 5: "€"                                      |
|   -> not a digit -> skip                             |
|                                                      |
| Position 7: "2"                                      |
|   (?<=\$) -> preceding char is "€" -> failure        |
|                                                      |
| Position 12: "3"                                     |
|   (?<=\$) -> preceding char is "$" -> success!       |
|   \d+ -> matches "300"                               |
|   -> "300" added to results                          |
+------------------------------------------------------+

Result: ["100", "300"]
```

### 8.6 Conditional Branching Operation Flow

```
Pattern: (\()?hello(?(1)\))
Input 1: "(hello)"
Input 2: "hello"

Processing input 1:
+------------------------------------------+
| (\()? -> matches "(" -> Group 1 = "("   |
| hello -> matches "hello"                 |
| (?(1)\)) -> Group 1 exists -> \) needed  |
| \) -> matches ")"                        |
| -> Success: "(hello)"                    |
+------------------------------------------+

Processing input 2:
+------------------------------------------+
| (\()? -> no match -> Group 1 = none      |
| hello -> matches "hello"                 |
| (?(1)\)) -> Group 1 absent -> nothing needed |
| -> Success: "hello"                      |
+------------------------------------------+
```

---

## 9. Comparison Tables

### 9.1 Group Type Comparison

| Type | Syntax | Captures | Number | Name | Purpose |
|------|--------|----------|--------|------|---------|
| Capture | `(...)` | Yes | Yes | No | Extraction/backreference |
| Non-capture | `(?:...)` | No | No | No | Grouping only |
| Named | `(?P<n>...)` | Yes | Yes | Yes | Readable extraction |
| Atomic | `(?>...)` | No | No | No | Backtrack suppression |
| Conditional | `(?(id)yes\|no)` | -- | -- | -- | Conditional branching |
| Positive lookahead | `(?=...)` | No | No | No | Zero-width assertion |
| Negative lookahead | `(?!...)` | No | No | No | Zero-width assertion |
| Positive lookbehind | `(?<=...)` | No | No | No | Zero-width assertion |
| Negative lookbehind | `(?<!...)` | No | No | No | Zero-width assertion |

### 9.2 Backreference Syntax (by Language)

| Language | In-pattern reference | Replacement reference | Named reference |
|----------|---------------------|----------------------|-----------------|
| Python | `\1`, `(?P=name)` | `\1`, `\g<1>`, `\g<name>` | `(?P<name>...)` |
| JavaScript | `\1`, `\k<name>` | `$1`, `$<name>` | `(?<name>...)` |
| Java | `\1`, `\k<name>` | `$1`, `${name}` | `(?<name>...)` |
| Perl | `\1`, `\k<name>` | `$1`, `$+{name}` | `(?<name>...)` |
| Go (RE2) | Not supported | `${1}`, `${name}` | `(?P<name>...)` |
| Rust | Not supported | `$1`, `$name` | `(?P<name>...)` |

### 9.3 Lookaround Constraint Comparison

| Constraint | Python re | JavaScript | Java | Perl | .NET |
|------------|----------|------------|------|------|------|
| Lookbehind length | Fixed-length only | Variable OK | Fixed-length only | Variable OK | Variable OK |
| Nestable | OK | OK | OK | OK | OK |
| Capture inside lookahead | OK | OK | OK | OK | OK |
| Capture inside lookbehind | OK | OK | OK | OK | OK |
| Quantifiers inside lookahead | OK | OK | OK | OK | OK |

### 9.4 Comprehensive Group Feature Comparison

| Feature | Python re | Python regex | JavaScript | Java | Go | Rust |
|---------|----------|-------------|------------|------|----|------|
| Capture `()` | OK | OK | OK | OK | OK | OK |
| Non-capture `(?:)` | OK | OK | OK | OK | OK | OK |
| Named | `(?P<>)` | `(?P<>)` | `(?<>)` | `(?<>)` | `(?P<>)` | `(?P<>)` |
| Backreference | OK | OK | OK | OK | N/A | N/A |
| Lookahead | OK | OK | OK | OK | N/A | N/A |
| Lookbehind | Fixed | Variable | Variable | Fixed | N/A | N/A |
| Atomic | N/A | OK | N/A | Java 20+ | N/A | N/A |
| Conditional | OK | OK | N/A | N/A | N/A | N/A |
| Branch reset | N/A | `(?|)` | N/A | N/A | N/A | N/A |

---

## 10. Anti-Patterns

### 10.1 Anti-Pattern: Excessive Capture Groups

```python
import re

# BAD: Making everything a capture group
pattern_bad = r'(https?)://(www\.)?(\w+)\.(\w+)/(\w+)/(\w+)'
# 6 groups -> group numbers become difficult to manage

# GOOD: Capture only needed parts + named groups
pattern_good = r'(?:https?)://(?:www\.)?(?P<domain>\w+\.\w+)/(?P<path>\w+/\w+)'

text = "https://www.example.com/api/users"
match = re.search(pattern_good, text)
if match:
    print(match.group('domain'))  # => 'example.com'
    print(match.group('path'))    # => 'api/users'
```

### 10.2 Anti-Pattern: Hard-Coding Backreference Group Numbers

```python
import re

# BAD: Hard-coded group numbers -- causes bugs when patterns change
pattern = r'(\w+)\s+(\d+)\s+(\w+)'
text = "item 42 completed"
match = re.search(pattern, text)
name = match.group(1)    # Shifts when pattern changes
count = match.group(2)   # Shifts when pattern changes

# GOOD: Named groups -- resilient to pattern changes
pattern = r'(?P<name>\w+)\s+(?P<count>\d+)\s+(?P<status>\w+)'
match = re.search(pattern, text)
name = match.group('name')      # Referenced by name -> doesn't shift
count = match.group('count')    # Referenced by name -> doesn't shift
```

### 10.3 Anti-Pattern: Overuse of Lookahead

```python
import re

# BAD: Validating password with lookahead only (hard to understand)
password_bad = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*])(?=.{8,})(?!.*(.)\1{2,})(?!.*(?:123|abc|password)).*$'
# -> Crammed into one line and unmaintainable

# GOOD: Structured with VERBOSE flag + comments
password_good = re.compile(r'''
    ^
    (?=.*[A-Z])            # Condition 1: contains uppercase
    (?=.*[a-z])            # Condition 2: contains lowercase
    (?=.*\d)               # Condition 3: contains digit
    (?=.*[!@#$%^&*])       # Condition 4: contains symbol
    (?!.*(.)\1{2,})        # Condition 5: no 3 consecutive identical chars
    (?!.*(?:123|abc|pwd))  # Condition 6: no weak patterns
    .{8,}                  # Body: 8 or more characters
    $
''', re.VERBOSE)

# Even better: individual check functions
def validate_password(pwd: str) -> list[str]:
    """Password validation: checks each condition individually, returns failure reasons"""
    errors = []
    if len(pwd) < 8:
        errors.append("Must be 8 or more characters")
    if not re.search(r'[A-Z]', pwd):
        errors.append("Must contain an uppercase letter")
    if not re.search(r'[a-z]', pwd):
        errors.append("Must contain a lowercase letter")
    if not re.search(r'\d', pwd):
        errors.append("Must contain a digit")
    if not re.search(r'[!@#$%^&*]', pwd):
        errors.append("Must contain a symbol")
    if re.search(r'(.)\1{2,}', pwd):
        errors.append("Must not have 3 or more consecutive identical characters")
    return errors
```

### 10.4 Anti-Pattern: Attempting Lookaround in Go/Rust

```go
// BAD: Lookahead is not available in Go
// re := regexp.MustCompile(`\d+(?=円)`)  // Panic!

// GOOD: Alternative approach - capture with groups and post-process
package main

import (
    "fmt"
    "regexp"
)

func main() {
    re := regexp.MustCompile(`(\d+)円`)
    text := "100円 200ドル 300円"

    matches := re.FindAllStringSubmatch(text, -1)
    for _, m := range matches {
        fmt.Printf("  Amount: %s\n", m[1])
    }
    // => Amount: 100
    // => Amount: 300
}
```

```rust
// BAD: Lookahead is not available in the Rust regex crate
// let re = Regex::new(r"\d+(?=円)").unwrap();  // Error!

// GOOD 1: Use capture groups as an alternative
use regex::Regex;

fn main() {
    let re = Regex::new(r"(\d+)円").unwrap();
    let text = "100円 200ドル 300円";

    for caps in re.captures_iter(text) {
        println!("  Amount: {}", &caps[1]);
    }

    // GOOD 2: Use fancy-regex (supports lookahead/lookbehind)
    // use fancy_regex::Regex;
    // let re = Regex::new(r"\d+(?=円)").unwrap();
}
```

---

## 11. Best Practices

### 11.1 Group Design Principles

```
+----------------------------------------------------------------+
| Principle 1: Minimal Capture Principle                         |
|   Capture only the needed parts; use (?:...) for everything else|
|                                                                |
| Principle 2: Prefer Named Groups                               |
|   Use named groups when there are 3 or more groups             |
|                                                                |
| Principle 3: Use VERBOSE Mode                                  |
|   Structure complex patterns with re.VERBOSE + comments        |
|                                                                |
| Principle 4: Test-Driven Pattern Design                        |
|   List cases that should/shouldn't match before writing the    |
|   pattern                                                      |
|                                                                |
| Principle 5: When to Use Lookaround                            |
|   - Simultaneous multi-condition checks (passwords, etc.)      |
|   - Pinpointing positions without consuming (comma insertion)  |
|   - Context-dependent matching (amounts after currency symbols)|
+----------------------------------------------------------------+
```

### 11.2 Designing Patterns for Readability

```python
import re

# BAD: A single unreadable line
bad = r'((?:\+81|0)[\d-]{9,13})|(\d{3}-?\d{4})|([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'

# GOOD: VERBOSE + named groups + comments
good = re.compile(r'''
    (?P<phone>                              # Phone number
        (?:\+81|0)                          #   Country code or leading 0
        [\d-]{9,13}                         #   Digits and hyphens
    )
    |
    (?P<postal>                             # Postal code
        \d{3}-?\d{4}                        #   XXX-XXXX format
    )
    |
    (?P<email>                              # Email address
        [a-zA-Z0-9._%+-]+                   #   Local part
        @                                   #   @
        [a-zA-Z0-9.-]+                      #   Domain
        \.[a-zA-Z]{2,}                      #   TLD
    )
''', re.VERBOSE)

text = "Contact: 090-1234-5678, Postal 100-0001, info@example.com"
for m in good.finditer(text):
    for name in ['phone', 'postal', 'email']:
        if m.group(name):
            print(f"  {name}: {m.group(name)}")
```

### 11.3 Designing Patterns for Performance

```python
import re

# 1. Limit lookahead to where it is truly needed
# BAD: Pointless lookahead
bad1 = r'(?=\d)\d+'  # Lookahead followed by the same pattern -> pointless

# GOOD: Same result without lookahead
good1 = r'\d+'

# 2. Suppress backtracking with non-capture groups
# BAD: Nested capture groups + quantifiers
bad2 = r'((\w+)\s*)+'  # Risk of catastrophic backtracking

# GOOD: Flat structure
good2 = r'\w+(?:\s+\w+)*'

# 3. Optimize alternation (|) order
# Place shorter patterns first when short strings are more common
# BAD: Starting with long patterns
bad3 = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'

# GOOD: Group by first character (some engines auto-optimize this)
good3 = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
```

---

## 12. FAQ

### Q1: How do I get both the group and the full match with `findall`?

**A**: `findall` returns group contents when groups are present. If you also need the full match, use `finditer` or add an enclosing group:

```python
import re

text = "cats and dogs"
# findall + group -> returns group contents only
print(re.findall(r'(cat|dog)s', text))  # => ['cat', 'dog']

# Method 1: Use finditer
for m in re.finditer(r'(cat|dog)s', text):
    print(f"Full: {m.group(0)}, Animal: {m.group(1)}")

# Method 2: Use a non-capture group
print(re.findall(r'(?:cat|dog)s', text))  # => ['cats', 'dogs']
```

### Q2: What is a conditional group `(?(id)yes|no)`?

**A**: A pattern that branches based on whether a group matched:

```python
import re

# If there is an opening parenthesis, require a closing parenthesis
pattern = r'(\()?hello(?(1)\))'
# (?(1)\)) = if group 1 matched, require \)

print(re.search(pattern, "hello").group())    # => 'hello'
print(re.search(pattern, "(hello)").group())  # => '(hello)'
print(re.search(pattern, "(hello").group())   # => 'hello' (matches the unparenthesized version)
```

### Q3: Can the same name be used for multiple named groups?

**A**: It depends on the language. Python's `re` module does **not** allow it. .NET allows same-named groups on both sides of a pipe. Python's third-party `regex` module enables similar behavior with the `(?|...)` branch reset group:

```python
# .NET example (conceptual):
# (?<digit>\d+)|(?<digit>[a-f]+)  -- both are "digit" group

# Python regex module branch reset:
# (?|(\d+)|([a-f]+))  -- both are group 1
```

### Q4: Can capture groups be used inside a lookahead?

**A**: Yes. Capture groups inside a lookahead are included in the match result when the lookahead succeeds:

```python
import re

# Capture group inside a lookahead
pattern = r'(?=(\d+)円)\d+'
text = "100円 200ドル 300円"

for m in re.finditer(pattern, text):
    print(f"  Number: {m.group()}, Lookahead group: {m.group(1)}")
# => Number: 100, Lookahead group: 100
# => Number: 300, Lookahead group: 300
```

### Q5: What are alternatives when lookaround is not available in Go or Rust?

**A**: There are three approaches:

1. **Capture group alternative**: Capture the surrounding context and exclude it in post-processing
2. **Two-pass matching**: Match with a broad pattern first, then filter the results further
3. **Third-party libraries**: `regexp2` for Go, `fancy-regex` for Rust

```rust
// Rust: Example using fancy-regex for lookahead
// Cargo.toml: fancy-regex = "0.13"
use fancy_regex::Regex;

fn main() {
    // Positive lookahead: numbers followed by "円"
    let re = Regex::new(r"\d+(?=円)").unwrap();
    let text = "100円 200ドル 300円";

    for m in re.find_iter(text) {
        if let Ok(m) = m {
            println!("  Amount: {}", m.as_str());
        }
    }
}
```

### Q6: Should I use atomic groups or possessive quantifiers?

**A**: They are functionally equivalent, so choose based on language support. In Java, possessive quantifiers (`a++`) are more concise. In Perl/.NET, both are available. They are worth considering for performance-critical cases in languages using NFA engines (Python, JavaScript, Java). In Go/Rust (DFA-based), catastrophic backtracking cannot occur by design, so they are unnecessary.

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
| `(...)` | Capture group -- saves results with a number |
| `(?:...)` | Non-capture group -- grouping only |
| `(?P<name>...)` | Named group -- accessible by name |
| `\1`, `\2` | Backreference -- reuses a group within the pattern |
| `(?P=name)` | Named backreference (Python) |
| `\g<1>`, `\g<name>` | Group reference in replacement (Python) |
| `$1`, `$<name>` | Group reference in replacement (JavaScript) |
| `(?=...)`, `(?!...)` | Lookahead -- zero-width check ahead |
| `(?<=...)`, `(?<!...)` | Lookbehind -- zero-width check behind |
| `(?>...)` | Atomic group -- suppresses backtracking |
| `(?(id)yes\|no)` | Conditional branch -- branches based on group presence |
| Group numbering | Assigned from 1 in order of opening parenthesis |
| Design guideline | Capture only needed parts; prefer named groups |

---

## 13. Exercises

### Exercise 1: HTML Tag Attribute Extraction

Write a regular expression using named groups to extract all tag names and attributes from the following HTML.

```html
<div class="container" id="main">
<img src="photo.jpg" alt="Photo">
<a href="https://example.com" target="_blank">Link</a>
```

Expected output: a list of tag names, attribute names, and attribute values

### Exercise 2: Detecting Duplicate Lines

Write a regular expression that detects consecutive identical lines in a text file (ignoring whitespace differences). Use backreferences.

```
Input:
Hello World
Hello World
Foo Bar
  Foo Bar
Baz
```

### Exercise 3: Password Validation with Lookahead

Build a password pattern using lookahead that satisfies all of the following conditions:
- 10 to 20 characters
- Contains at least one uppercase letter, one lowercase letter, one digit, and one symbol
- No 4 or more consecutive identical characters
- Does not contain "password", "12345", or "qwerty" (case-insensitive)

### Exercise 4: Conditional Branching Pattern

Build a phone number validation pattern using conditional branching:
- If it starts with `+81`, international format (`+81-XX-XXXX-XXXX`)
- If it starts with `0`, domestic format (`0XX-XXXX-XXXX`)
- Otherwise, invalid

### Exercise 5: CamelCase to snake_case Conversion

Write a regular expression using lookahead/lookbehind to convert camelCase identifiers to snake_case.

```
Input: getUserNameFromDatabase
Output: get_user_name_from_database

Input: XMLParser
Output: xml_parser

Input: getHTTPResponse
Output: get_http_response
```

Hint: Pay attention to how consecutive uppercase letters are handled.

### Exercise 6: Matching the Outermost Nested Parentheses

Extract the outermost parenthesized portions from the following string (using recursive patterns or an iterative approach):

```
Input: "func(a, (b + c), d) + other(x)"
Expected: ["func(a, (b + c), d)", "other(x)"]
```

Hint: Fully handling nesting with regular expressions alone is difficult. Consider recursive patterns in `.NET` or `Perl`, or a combination of regex and program logic.

### Exercise 7: Structured Log File Analysis

Build a pattern using named groups that fully parses the following log format:

```
[2026-02-11T10:30:45.123+09:00] [ERROR] [com.example.auth.LoginService] [req-id=abc123] User login failed: invalid credentials for user "admin" from IP 192.168.1.100
```

Fields to extract: timestamp, log level, class name, request ID, message body

---

## Recommended Next Guides

- [01-lookaround.md](./01-lookaround.md) -- Lookahead and Lookbehind in Detail
- [02-unicode-regex.md](./02-unicode-regex.md) -- Unicode Regular Expressions

## References

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- Chapter 7: "Groups and Backreferences"
2. **Python re module - Grouping** https://docs.python.org/3/library/re.html#regular-expression-syntax -- Official Python group syntax reference
3. **TC39 Named Capture Groups Proposal** https://tc39.es/proposal-regexp-named-groups/ -- JavaScript named capture groups specification
4. **MDN Lookahead and Lookbehind** https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions/Assertions -- JavaScript lookahead/lookbehind documentation
5. **RE2 Syntax** https://github.com/google/re2/wiki/Syntax -- Syntax specification of the RE2 engine used in Go/Rust
6. **fancy-regex** https://docs.rs/fancy-regex/ -- Rust crate for lookahead/lookbehind support
7. **regex module for Python** https://pypi.org/project/regex/ -- Extended regular expression module for Python
