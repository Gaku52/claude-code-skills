# Character Classes -- [abc], \d, \w, \s, POSIX

> Character classes are a core feature of regular expressions, defining "the set of characters that may match at a given position." This guide covers bracket notation, shorthand classes, and POSIX classes comprehensively.

## What You Will Learn in This Chapter

1. **Bracket character class `[...]` syntax and behavior** -- Precise rules for positive, negative, and range specifications
2. **Shorthand class meanings and cross-language differences** -- How `\d` `\w` `\s` differ in Unicode support across languages
3. **POSIX character classes and practical selection criteria** -- When to use `[:alpha:]`, `[:digit:]`, etc.
4. **Leveraging Unicode character properties** -- Specifying Unicode categories with `\p{L}`, `\p{N}`, etc.
5. **Set operations on character classes** -- Implementing intersection, subtraction, and union


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Basic Syntax -- Literals, Metacharacters, Escaping](./01-basic-syntax.md)

---

## 1. Bracket Character Class `[...]`

### 1.1 Basic Form

```python
import re

# [abc] -- matches any single character: a, b, or c
print(re.findall(r'[abc]', "abcdef"))
# => ['a', 'b', 'c']

# [aeiou] -- matches a single vowel
text = "Hello World"
print(re.findall(r'[aeiou]', text, re.IGNORECASE))
# => ['e', 'o', 'o']

# Character order within a class has no significance
# [abc], [cba], and [bac] are all equivalent
```

```python
# Basic usage of character classes

import re

# 1. Selecting from a specific character set
vowels = re.compile(r'[aeiouAEIOU]')
text = "Hello Beautiful World"
print(vowels.findall(text))
# => ['e', 'o', 'E', 'a', 'u', 'i', 'u', 'o']

# 2. Note that a character class matches "one character" at a time
print(re.findall(r'[abc]', "aabbcc"))
# => ['a', 'a', 'b', 'b', 'c', 'c']  # Each character matches individually

# 3. Combine with quantifiers to match multiple characters
print(re.findall(r'[abc]+', "aabbcc def aab"))
# => ['aabbcc', 'aab']

# 4. Order within a character class is irrelevant
assert re.findall(r'[abc]', "abc") == re.findall(r'[cba]', "abc")
assert re.findall(r'[abc]', "abc") == re.findall(r'[bca]', "abc")
```

### 1.2 Range Specification `-`

```python
import re

# [a-z]  lowercase alphabetic characters
# [A-Z]  uppercase alphabetic characters
# [0-9]  digits
# [a-zA-Z0-9]  alphanumeric characters

text = "Item-42: Price $9.99"

# Alphabetic characters only
print(re.findall(r'[a-zA-Z]+', text))
# => ['Item', 'Price']

# Digits only
print(re.findall(r'[0-9]+', text))
# => ['42', '9', '99']

# Combining multiple ranges
print(re.findall(r'[a-zA-Z0-9]+', text))
# => ['Item', '42', 'Price', '9', '99']

# To include a literal hyphen:
# At the start: [-abc]  At the end: [abc-]  Escaped: [a\-c]
print(re.findall(r'[-+*/]', "3+4-2*1/5"))
# => ['+', '-', '*', '/']
```

```python
# Range specification details

import re

# Multiple contiguous ranges
# Alphanumeric characters and underscore
print(re.findall(r'[a-zA-Z0-9_]+', "hello_world 123"))
# => ['hello_world', '123']

# Hexadecimal characters
print(re.findall(r'[0-9a-fA-F]+', "0xFF 0xAB 0xGG"))
# => ['0', 'xFF', '0', 'xAB', '0', 'xGG']  # xGG is invalid hex

# A more precise hexadecimal pattern
print(re.findall(r'0x[0-9a-fA-F]+', "0xFF 0xAB 0xGG"))
# => ['0xFF', '0xAB']  # 0xGG does not partially match

# Partial ranges
print(re.findall(r'[a-f]+', "abcdefghij"))
# => ['abcdef']

print(re.findall(r'[2-7]+', "0123456789"))
# => ['234567']

# Multiple independent ranges
print(re.findall(r'[a-cm-o1-3]+', "abcmnop123456"))
# => ['abc', 'mno', '123']
```

### 1.3 Negated Character Class `[^...]`

```python
import re

# [^abc] -- matches any single character except a, b, or c
print(re.findall(r'[^abc]', "abcdef"))
# => ['d', 'e', 'f']

# [^0-9] -- anything except digits
print(re.findall(r'[^0-9]+', "abc123def456"))
# => ['abc', 'def']

# ^ means negation only when at the start
# [a^b] -- matches a, ^, or b (^ is treated as a literal)
print(re.findall(r'[a^b]', "a^b"))
# => ['a', '^', 'b']
```

```python
# Practical uses of negated character classes

import re

# 1. Extract content inside quotes (excluding the quotes themselves)
text = '"hello" and "world"'
print(re.findall(r'"([^"]*)"', text))
# => ['hello', 'world']

# 2. Extract HTML tag attribute values
html = '<a href="https://example.com" class="link">'
print(re.findall(r'(\w+)="([^"]*)"', html))
# => [('href', 'https://example.com'), ('class', 'link')]

# 3. Fields separated by commas (excluding the commas)
csv_line = "field1,field2,field3"
print(re.findall(r'[^,]+', csv_line))
# => ['field1', 'field2', 'field3']

# 4. Last component of a path (after the final slash)
path = "/usr/local/bin/python3"
print(re.findall(r'[^/]+$', path))
# => ['python3']

# 5. Extract file extension (after the dot)
filename = "document.backup.tar.gz"
print(re.findall(r'[^.]+', filename))
# => ['document', 'backup', 'tar', 'gz']

# 6. Consecutive non-whitespace characters
text = "  hello   world  "
print(re.findall(r'[^\s]+', text))
# => ['hello', 'world']

# 7. Match excluding specific characters
# Exclude control characters and special characters
text = "hello\x00world\x1b[31mred"
print(re.findall(r'[^\x00-\x1f\x7f]+', text))
# => ['hello', 'world', '[31mred']
```

### 1.4 Metacharacter Rules Inside Character Classes

```
Characters with special meaning inside brackets [...]:

Char   Meaning                    How to use as literal
----   -------                    ---------------------
]      End of class               Place at start: []abc] or escape: [\]]
\      Escape                     Escape: [\\]
^      Negation (start only)      Place anywhere but start: [a^b]
-      Range (between chars only) Place at start/end: [-abc] [abc-]

Inside brackets, . * + ? | ( ) { } are treated as literals:
  [.*+?]  -> dot, asterisk, plus, question mark
```

```python
# Verifying metacharacter behavior

import re

# Most metacharacters are literals inside a character class
print(re.findall(r'[.+*?|(){}]', "a.b+c*d?e|f(g)h{i}"))
# => ['.', '+', '*', '?', '|', '(', ')', '{', '}']

# How to include ] in a character class
# Method 1: Place at the start
print(re.findall(r'[]ab]', "a]b"))
# => ['a', ']', 'b']

# Method 2: Escape
print(re.findall(r'[a\]b]', "a]b"))
# => ['a', ']', 'b']

# Including \ in a character class
print(re.findall(r'[a\\b]', r"a\b"))
# => ['a', '\\', 'b']

# How to include - in a character class
# Method 1: At the start
print(re.findall(r'[-ab]', "a-b"))   # => ['a', '-', 'b']
# Method 2: At the end
print(re.findall(r'[ab-]', "a-b"))   # => ['a', '-', 'b']
# Method 3: Escape
print(re.findall(r'[a\-b]', "a-b"))  # => ['a', '-', 'b']

# Including ^ for purposes other than negation
# Place anywhere but the start
print(re.findall(r'[a^b]', "a^b"))   # => ['a', '^', 'b']
# Escape
print(re.findall(r'[\^ab]', "a^b"))  # => ['a', '^', 'b']
```

### 1.5 Character Class Combination Techniques

```python
import re

# 1. Combining shorthands with character classes
# Digits, hyphens, and dots (for phone numbers or IP addresses)
print(re.findall(r'[\d.-]+', "IP: 192.168.1.1 Tel: 03-1234-5678"))
# => ['192.168.1.1', '03-1234-5678']

# 2. Word characters and hyphens (CSS class names or slugs)
print(re.findall(r'[\w-]+', "my-class another_class third-class-name"))
# => ['my-class', 'another_class', 'third-class-name']

# 3. Combining negated shorthands with character classes
# Non-whitespace and non-comma
print(re.findall(r'[^\s,]+', "apple, banana, cherry"))
# => ['apple', 'banana', 'cherry']

# 4. Japanese-related character classes
# Hiragana
print(re.findall(r'[\u3040-\u309F]+', "こんにちは Hello 世界"))
# => ['こんにちは']

# Katakana
print(re.findall(r'[\u30A0-\u30FF]+', "カタカナ ひらがな ABC"))
# => ['カタカナ']

# Kanji (basic CJK Unified Ideographs block)
print(re.findall(r'[\u4E00-\u9FFF]+', "漢字テスト hello 東京タワー"))
# => ['漢字', '東京']

# Hiragana + Katakana + Kanji
print(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+',
                 "東京タワーへ行こう ABC 123"))
# => ['東京タワーへ行こう']

# 5. Detecting full-width characters
# Full-width alphanumeric
print(re.findall(r'[Ａ-Ｚａ-ｚ０-９]+', "Ｈｅｌｌｏ 123 Ｗｏｒｌｄ"))
# => ['Ｈｅｌｌｏ', 'Ｗｏｒｌｄ']
```

---

## 2. Shorthand Classes

### 2.1 List and Equivalent Expressions

```
┌──────────┬───────────┬─────────────────────────────────┐
│ Shorthand│  Negation │  Equivalent class (ASCII)         │
├──────────┼───────────┼─────────────────────────────────┤
│ \d       │ \D        │ [0-9]                           │
│ \w       │ \W        │ [a-zA-Z0-9_]                    │
│ \s       │ \S        │ [ \t\n\r\f\v]                   │
│ \b       │ \B        │ (Anchor: word/non-word boundary) │
└──────────┴───────────┴─────────────────────────────────┘

* In Unicode mode, the range expands significantly (see below)
```

### 2.2 Code Examples

```python
import re

text = "User: 田中太郎, Age: 25, Email: tanaka@example.com"

# \d -- digits
print(re.findall(r'\d+', text))
# => ['25']

# \w -- word characters (Unicode-aware in Python 3)
print(re.findall(r'\w+', text))
# => ['User', '田中太郎', 'Age', '25', 'Email', 'tanaka', 'example', 'com']

# \s -- whitespace characters
print(re.split(r'\s+', "hello   world\tfoo\nbar"))
# => ['hello', 'world', 'foo', 'bar']

# \D, \W, \S -- negated forms
print(re.findall(r'\D+', "abc123def456"))
# => ['abc', 'def']
```

### 2.3 Differences in \w Under Unicode Mode

```python
import re

text = "Hello 世界 café 123"

# Python 3: \w is Unicode-aware by default
print(re.findall(r'\w+', text))
# => ['Hello', '世界', 'café', '123']

# To restrict to ASCII mode
print(re.findall(r'\w+', text, re.ASCII))
# => ['Hello', 'caf', '123']  -- 'é' and '世界' do not match
```

```javascript
// JavaScript: Unicode support with the u flag
const text = "Hello 世界 café 123";

// Without u flag: \w is ASCII only
console.log(text.match(/\w+/g));
// => ['Hello', 'caf', '123']

// Unicode property escape (ES2018+)
console.log(text.match(/[\p{L}\p{N}]+/gu));
// => ['Hello', '世界', 'café', '123']
```

### 2.4 Detailed Unicode Behavior of \d

```python
import re

# Python 3's \d matches all Unicode digits
# Examples beyond ASCII digits:

test_strings = [
    "Half-width: 0123456789",             # ASCII digits
    "Full-width: ０１２３４５６７８９",     # Full-width digits
    "Arabic-Indic: ٠١٢٣٤٥٦٧٨٩",         # Arabic-Indic digits
    "Devanagari: ०१२३",                   # Devanagari digits
    "Thai: ๐๑๒๓๔๕๖๗๘๙",                # Thai digits
]

for s in test_strings:
    matches = re.findall(r'\d+', s)
    ascii_matches = re.findall(r'\d+', s, re.ASCII)
    print(f"  {s}")
    print(f"    Unicode \\d: {matches}")
    print(f"    ASCII \\d:   {ascii_matches}")

# Three ways to restrict to ASCII digits only:
# 1. re.ASCII flag
print(re.findall(r'\d+', "123 ０１２", re.ASCII))
# => ['123']

# 2. Explicit character class [0-9]
print(re.findall(r'[0-9]+', "123 ０１２"))
# => ['123']

# 3. Inline flag (?a)
print(re.findall(r'(?a)\d+', "123 ０１２"))
# => ['123']
```

### 2.5 Details of \s: Types of Whitespace Characters

```python
import re

# Characters matched by \s (ASCII mode)
whitespace_chars = {
    ' ':  'Space (0x20)',
    '\t': 'Tab (0x09)',
    '\n': 'Line Feed LF (0x0A)',
    '\r': 'Carriage Return CR (0x0D)',
    '\f': 'Form Feed (0x0C)',
    '\v': 'Vertical Tab (0x0B)',
}

for char, desc in whitespace_chars.items():
    matches = bool(re.match(r'\s', char))
    print(f"  {desc}: {'Match' if matches else 'No match'}")

# Additional whitespace characters in Unicode mode
unicode_spaces = {
    '\u00A0': 'No-Break Space (NBSP)',
    '\u2000': 'En Quad',
    '\u2001': 'Em Quad',
    '\u2002': 'En Space',
    '\u2003': 'Em Space',
    '\u2004': 'Three-Per-Em Space',
    '\u2005': 'Four-Per-Em Space',
    '\u2006': 'Six-Per-Em Space',
    '\u2007': 'Figure Space',
    '\u2008': 'Punctuation Space',
    '\u2009': 'Thin Space',
    '\u200A': 'Hair Space',
    '\u2028': 'Line Separator',
    '\u2029': 'Paragraph Separator',
    '\u202F': 'Narrow No-Break Space',
    '\u205F': 'Medium Mathematical Space',
    '\u3000': 'Ideographic Space (Full-width Space)',
    '\uFEFF': 'BOM (Byte Order Mark)',
}

for char, desc in unicode_spaces.items():
    # Python 3 uses Unicode mode by default
    matches_unicode = bool(re.match(r'\s', char))
    matches_ascii = bool(re.match(r'\s', char, re.ASCII))
    print(f"  {desc}: Unicode={matches_unicode}, ASCII={matches_ascii}")

# Practical example: handling full-width spaces
text = "Hello　World"  # Contains a full-width space
print(re.split(r'\s+', text))
# => ['Hello', 'World']  # Full-width space is also matched by \s

# In ASCII mode, full-width spaces are ignored
print(re.split(r'\s+', text, flags=re.ASCII))
# => ['Hello\u3000World']  # Full-width space is not matched
```

### 2.6 Detailed Look at \b Word Boundary

```python
import re

# \b matches a "position" (zero-width assertion)
# It does not consume characters

# Word boundary definition:
# The position between \w and \W
# The position at the start of the string if followed by \w
# The position at the end of the string if preceded by \w

text = "cat caterpillar concatenate category the_cat"

# \bcat\b: only the complete word "cat"
print(re.findall(r'\bcat\b', text))
# => ['cat']

# \bcat: words starting with "cat"
print(re.findall(r'\bcat\w*', text))
# => ['cat', 'caterpillar', 'concatenate', 'category']

# cat\b: words ending with "cat"
print(re.findall(r'\w*cat\b', text))
# => ['cat', 'the_cat']

# \B: non-word boundary (inside a word)
print(re.findall(r'\Bcat\B', text))
# => ['cat']  # The "cat" inside "concatenate"

# Practical example: exact word search
def find_exact_word(text, word):
    """Search for an exact word match"""
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.findall(pattern, text)

print(find_exact_word("Java JavaScript JavaEE", "Java"))
# => ['Java']

# Word boundaries with Unicode
text = "東京は首都です。Tokyo is capital."
print(re.findall(r'\b\w+\b', text))
# => ['東京は首都です', 'Tokyo', 'is', 'capital']
# Since Japanese has no spaces between words, consecutive \w characters match together
```

### 2.7 Cross-Language Differences in Shorthands

```
Behavior of \w across languages:

┌──────────────┬──────────────────────────────────────────┐
│ Language     │ Range of \w                               │
├──────────────┼──────────────────────────────────────────┤
│ Python 3     │ Unicode Letters + Digits + _             │
│ (default)    │ → Matches Japanese, Chinese, etc.        │
├──────────────┼──────────────────────────────────────────┤
│ Python 3     │ [a-zA-Z0-9_]                             │
│ (re.ASCII)   │ → ASCII only                             │
├──────────────┼──────────────────────────────────────────┤
│ JavaScript   │ [a-zA-Z0-9_]                             │
│ (default)    │ → ASCII only                             │
├──────────────┼──────────────────────────────────────────┤
│ JavaScript   │ Use \p{L} for Unicode support            │
│ (/u flag)    │ → \w itself does not change              │
├──────────────┼──────────────────────────────────────────┤
│ Java         │ [a-zA-Z0-9_]                             │
│ (default)    │ → ASCII only                             │
├──────────────┼──────────────────────────────────────────┤
│ Java         │ Unicode Letters + Digits + _             │
│ (UNICODE_    │ → Matches Japanese, etc.                 │
│  CHARACTER_  │                                          │
│  CLASS)      │                                          │
├──────────────┼──────────────────────────────────────────┤
│ Perl         │ Unicode Letters + Digits + _             │
│              │ → Unicode-aware by default               │
├──────────────┼──────────────────────────────────────────┤
│ Ruby         │ Unicode Letters + Digits + _             │
│              │ → Unicode-aware by default               │
├──────────────┼──────────────────────────────────────────┤
│ Go           │ [0-9A-Za-z_]                             │
│ (RE2)        │ → ASCII only                             │
├──────────────┼──────────────────────────────────────────┤
│ Rust         │ Unicode-aware (regex crate)              │
│              │ → ASCII mode specified separately        │
└──────────────┴──────────────────────────────────────────┘
```

---

## 3. POSIX Character Classes

### 3.1 List

```
┌──────────────┬──────────────────────┬──────────────────┐
│ POSIX Class  │ Equivalent (ASCII)    │ Meaning          │
├──────────────┼──────────────────────┼──────────────────┤
│ [:alpha:]    │ [a-zA-Z]             │ Alphabetic chars │
│ [:digit:]    │ [0-9]                │ Digits           │
│ [:alnum:]    │ [a-zA-Z0-9]          │ Alphanumeric     │
│ [:upper:]    │ [A-Z]                │ Uppercase        │
│ [:lower:]    │ [a-z]                │ Lowercase        │
│ [:space:]    │ [ \t\n\r\f\v]        │ Whitespace       │
│ [:blank:]    │ [ \t]                │ Space & tab only │
│ [:punct:]    │ [!"#$%&'()*+,-./:;  │ Punctuation      │
│              │  <=>?@[\]^_`{|}~]    │                  │
│ [:print:]    │ [ -~]                │ Printable chars  │
│ [:graph:]    │ [!-~]                │ Printable (no sp)│
│ [:cntrl:]    │ [\x00-\x1f\x7f]     │ Control chars    │
│ [:xdigit:]   │ [0-9a-fA-F]          │ Hex digits       │
│ [:ascii:]    │ [\x00-\x7f]          │ ASCII chars      │
└──────────────┴──────────────────────┴──────────────────┘
```

### 3.2 Using POSIX Classes

```bash
# POSIX classes are primarily used in grep, sed, and awk

# Extract alphabetic characters only
# => Hello
# => World

# Extract digits only
# => 19
# => 99

# Extract hexadecimal digits
# => FF00AA

# Negating POSIX classes
echo "abc123" | grep -oE '[^[:digit:]]+'
# => abc
```

```bash
# Practical examples of POSIX classes

# 1. Extract alphanumeric characters and underscores (variable name pattern)
echo "hello-world my_var 123abc" | grep -oE '[[:alnum:]_]+'
# => hello
# => world
# => my_var
# => 123abc

# 2. Extract punctuation
# => ,
# => !
# => ?

# 3. Detect non-printable characters (control characters)
# => 2 (two control characters)

# 4. Split fields by whitespace (blank matches space and tab only)
# => 3

# 5. Using POSIX classes in sed
# => Hello  World

# 6. Convert uppercase to lowercase (POSIX class based)
# => hello world

# 7. Keep only safe filename characters
echo "my file (1).txt" | sed 's/[^[:alnum:]._-]/_/g'
# => my_file__1_.txt

# 8. Remove blank lines
# => Only non-empty lines are displayed
```

### 3.3 POSIX vs Shorthand Comparison

| Purpose | POSIX | Shorthand | Available Environments |
|---------|-------|-----------|----------------------|
| Word characters | None | `\w` | Shorthand only |

### 3.4 Notes on POSIX Classes

```bash
# Note 1: POSIX classes must always be used inside brackets
# NG: [:digit:] -- matches individual characters :, d, i, g, t

# Note 2: POSIX classes can be combined with other characters
echo "abc-123_def" | grep -oE '[[:alnum:]_-]+'
# => abc-123_def

# Note 3: Behavior changes depending on locale
# LC_ALL=C matches ASCII only
# LC_ALL=ja_JP.UTF-8 also matches Japanese
# => a, b (é does not match)

# => aéb (é also matches)

# Note 4: grep -P (PCRE) may not support POSIX classes
# grep -E (ERE) or grep (BRE) is recommended
```

---

## 4. Unicode Character Properties

### 4.1 Unicode General Category

```python
# Using Unicode properties with Python's regex module (third-party)
# pip install regex

# Usage examples with the regex module
try:
    import regex

    text = "Hello 世界 café 123 !@#"

    # \p{L} -- Unicode "Letter"
    print(regex.findall(r'\p{L}+', text))
    # => ['Hello', '世界', 'café']

    # \p{N} -- Unicode "Number"
    print(regex.findall(r'\p{N}+', text))
    # => ['123']

    # \p{P} -- Unicode "Punctuation"
    print(regex.findall(r'\p{P}', text))
    # => ['!']

    # \p{S} -- Unicode "Symbol"
    print(regex.findall(r'\p{S}', text))
    # => ['@', '#']

    # \p{Z} -- Unicode "Separator"
    # Spaces, etc.

except ImportError:
    # Fallback when the regex module is not available
    import re

    # Python's standard re does not directly support Unicode properties
    # Alternative: specify Unicode categories using ranges

    # Japanese characters (Hiragana, Katakana, Kanji)
    print(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+', text))
```

### 4.2 Major Unicode Categories

```
Unicode General Category:

L  (Letter)       -- Letters
├── Lu (Uppercase) -- Uppercase letters (A, B, C, ...)
├── Ll (Lowercase) -- Lowercase letters (a, b, c, ...)
├── Lt (Titlecase) -- Titlecase letters (Dž, Lj, ...)
├── Lm (Modifier)  -- Modifier letters
└── Lo (Other)     -- Other letters (Kanji, Hiragana, ...)

M  (Mark)          -- Marks (combining characters)
├── Mn (Nonspacing)
├── Mc (Spacing Combining)
└── Me (Enclosing)

N  (Number)        -- Numbers
├── Nd (Decimal)   -- Decimal digits (0-9, ０-９, ...)
├── Nl (Letter)    -- Letterlike numbers (Ⅰ, Ⅱ, ...)
└── No (Other)     -- Other numbers (½, ⅓, ...)

P  (Punctuation)   -- Punctuation
├── Pc (Connector) -- Connector punctuation (_)
├── Pd (Dash)      -- Dashes (-, –, —)
├── Ps (Open)      -- Opening brackets ((, [, {)
├── Pe (Close)     -- Closing brackets (), ], })
├── Pi (Initial)   -- Opening quotation marks («, ', ")
├── Pf (Final)     -- Closing quotation marks (», ', ")
└── Po (Other)     -- Other punctuation (., ,, !, ?)

S  (Symbol)        -- Symbols
├── Sm (Math)      -- Mathematical symbols (+, =, <, >)
├── Sc (Currency)  -- Currency symbols ($, €, ¥, £)
├── Sk (Modifier)  -- Modifier symbols
└── So (Other)     -- Other symbols (©, ®, ™)

Z  (Separator)     -- Separators
├── Zs (Space)     -- Space separators
├── Zl (Line)      -- Line separators
└── Zp (Paragraph) -- Paragraph separators

C  (Other)         -- Other
├── Cc (Control)   -- Control characters
├── Cf (Format)    -- Format characters (BOM, etc.)
├── Cs (Surrogate) -- Surrogates
├── Co (Private)   -- Private use characters
└── Cn (Unassigned)-- Unassigned
```

### 4.3 Character Classes by Unicode Script

```javascript
// JavaScript ES2018+ Unicode Property Escape

const text = "Hello こんにちは 世界 Привет مرحبا";

// Japanese Hiragana
console.log(text.match(/\p{Script=Hiragana}+/gu));
// => ['こんにちは']

// Kanji (Han)
console.log(text.match(/\p{Script=Han}+/gu));
// => ['世界']

// Cyrillic
console.log(text.match(/\p{Script=Cyrillic}+/gu));
// => ['Привет']

// Arabic
console.log(text.match(/\p{Script=Arabic}+/gu));
// => ['مرحبا']

// Latin
console.log(text.match(/\p{Script=Latin}+/gu));
// => ['Hello']

// Emoji
const emoji_text = "Hello! Nice day!";
console.log(emoji_text.match(/\p{Emoji}/gu));
// => ['', '']
```

```python
# Unicode Script with Python's regex module

try:
    import regex

    text = "Hello こんにちは 世界 カタカナ"

    # Hiragana
    print(regex.findall(r'\p{Hiragana}+', text))
    # => ['こんにちは']

    # Katakana
    print(regex.findall(r'\p{Katakana}+', text))
    # => ['カタカナ']

    # Kanji
    print(regex.findall(r'\p{Han}+', text))
    # => ['世界']

    # All Japanese (Hiragana + Katakana + Kanji)
    print(regex.findall(r'[\p{Hiragana}\p{Katakana}\p{Han}]+', text))
    # => ['こんにちは', '世界', 'カタカナ']

except ImportError:
    pass
```

### 4.4 ECMAScript 2024 v Flag (Unicode Sets)

```javascript
// The v flag in ECMAScript 2024 enables set operations on character classes

// Intersection (&&) -- characters in both sets
// /[\p{Script=Latin}&&\p{Letter}]/v

// Subtraction (--) -- left set minus right set
// /[\p{Letter}--\p{Script=Latin}]/v

// Union -- same as traditional character classes
// /[\p{Script=Latin}\p{Script=Greek}]/v

// Example: Latin characters excluding ASCII (accented characters only)
// /[\p{Script=Latin}--[a-zA-Z]]/v

// Example: Alphanumeric minus digits = letters only
// /[\p{Alnum}--\p{Number}]/v
```

---

## 5. Combination Patterns

### 5.1 Combining Character Classes

```python
import re

# Alphanumeric, hyphens, and underscores
slug_pattern = r'[a-zA-Z0-9_-]+'
print(re.findall(slug_pattern, "my-page_title 2026"))
# => ['my-page_title', '2026']

# Japanese characters (Unicode ranges)
jp_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+'
print(re.findall(jp_pattern, "Hello 東京タワーへ行こう"))
# => ['東京タワーへ行こう']

# Character class subtraction (.NET only)
# [a-z-[aeiou]] -- lowercase consonants only

# Character class intersection (Java)
# [a-z&&[^aeiou]] -- lowercase consonants only
```

### 5.2 Common Character Class Patterns

```python
import re

# Characters allowed in filenames
filename_pattern = r'[a-zA-Z0-9._-]+'
print(re.findall(filename_pattern, "report_2026-02.pdf"))
# => ['report_2026-02.pdf']

# Hexadecimal color codes
hex_color = r'#[0-9a-fA-F]{6}\b'
print(re.findall(hex_color, "color: #FF5733; bg: #00aaff;"))
# => ['#FF5733', '#00aaff']

# Quoted strings (excluding the quotes themselves)
quoted = r'"[^"]*"'
print(re.findall(quoted, 'name="John" age="25"'))
# => ['"John"', '"25"']

# Printable characters excluding control characters
printable = r'[^\x00-\x1f\x7f]+'
print(re.findall(printable, "hello\x00world\x1b[31m"))
# => ['hello', 'world', '[31m']
```

### 5.3 Advanced Character Class Patterns

```python
import re

# 1. Characters allowed in email local parts
local_part = r'[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+'
print(re.findall(local_part, "user.name+tag@example.com"))
# => ["user.name+tag"]

# 2. URL-safe characters (RFC 3986)
url_safe = r'[a-zA-Z0-9._~:/?#\[\]@!$&\'()*+,;=-]+'
print(re.findall(url_safe, "https://example.com/path?q=hello&lang=ja"))

# 3. Characters allowed in CSS selectors
css_selector = r'[a-zA-Z0-9_-]+'

# 4. Shell-safe filename characters
safe_filename = r'[a-zA-Z0-9._-]+'

# 5. SQL injection prevention: allow only alphanumeric and spaces
safe_input = r'^[a-zA-Z0-9 ]+$'

# 6. Base64 encoded strings
base64_pattern = r'[A-Za-z0-9+/]+=*'
print(re.findall(base64_pattern, "SGVsbG8gV29ybGQ= next"))
# => ['SGVsbG8gV29ybGQ=']

# 7. UUID pattern
uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
print(re.findall(uuid_pattern, "id: 550e8400-e29b-41d4-a716-446655440000", re.IGNORECASE))
# => ['550e8400-e29b-41d4-a716-446655440000']

# 8. Semantic versioning
semver_pattern = r'[0-9]+\.[0-9]+\.[0-9]+(?:-[a-zA-Z0-9.]+)?(?:\+[a-zA-Z0-9.]+)?'
print(re.findall(semver_pattern, "v1.2.3-beta.1+build.123"))
# => ['1.2.3-beta.1+build.123']
```

---

## 6. ASCII Diagrams

### 6.1 Conceptual Diagram of Character Classes

```
Entire character space (Unicode: ~150,000 characters)
┌─────────────────────────────────────────┐
│                                         │
│   [a-z]  ┌─────────┐                   │
│          │a b c ... z│  26 characters   │
│          └─────────┘                    │
│                                         │
│   \d     ┌────────────┐                 │
│          │0 1 2 ... 9  │  10 characters │
│          │(Unicode: hundreds)│           │
│          └────────────┘                 │
│                                         │
│   \w     ┌──────────────────────┐       │
│          │a-z A-Z 0-9 _         │       │
│          │(Unicode: tens of thousands)│  │
│          └──────────────────────┘       │
│                                         │
│   \s     ┌──────────────┐               │
│          │Space TAB LF CR│  6 characters│
│          │FF VT          │              │
│          └──────────────┘               │
│                                         │
│   [^a-z] = complement of [a-z] above   │
│   \D     = complement of \d             │
│   \W     = complement of \w             │
│   \S     = complement of \s             │
└─────────────────────────────────────────┘
```

### 6.2 How Negated Character Classes Work

```
Pattern: [^aeiou]  (non-vowels)
Text:    "regex"

  r -> match [^aeiou]? -> 'r' is not a vowel -> Match
  e -> match [^aeiou]? -> 'e' is a vowel -> No match
  g -> match [^aeiou]? -> 'g' is not a vowel -> Match
  e -> match [^aeiou]? -> 'e' is a vowel -> No match
  x -> match [^aeiou]? -> 'x' is not a vowel -> Match

Result: r, g, x match
```

### 6.3 Range Specification Based on ASCII Codes

```
Ranges based on ASCII codes:

[0-9]  = ASCII 48-57
  48: '0'  49: '1'  50: '2' ... 57: '9'

[A-Z]  = ASCII 65-90
  65: 'A'  66: 'B'  67: 'C' ... 90: 'Z'

[a-z]  = ASCII 97-122
  97: 'a'  98: 'b'  99: 'c' ... 122: 'z'

Warning: [A-z] includes unintended characters!
  65: 'A' ... 90: 'Z'
  91: '['  92: '\'  93: ']'  94: '^'  95: '_'  96: '`'
  97: 'a' ... 122: 'z'

  -> [ \ ] ^ _ ` are also included!
  -> Use [A-Za-z] instead
```

### 6.4 Set Operations on Character Classes

```
Set operation concepts:

Union:        [a-z0-9]  = [a-z] ∪ [0-9]
Complement:   [^a-z]    = U \ [a-z]
Intersection: Java: [a-z&&[aeiou]]  = [a-z] ∩ [aeiou] = [aeiou]
Subtraction:  .NET: [a-z-[aeiou]]   = [a-z] \ [aeiou] = consonants

Visual representation:

     [a-z]           [aeiou]
  ┌──────────┐    ┌─────────┐
  │ bcdfgh...│    │ a e i   │
  │ jklmnp...│ ∩  │ o u     │
  │  a e i   │    │         │
  │  o u     │    │         │
  └──────────┘    └─────────┘

  Intersection [a-z&&[aeiou]] = {a, e, i, o, u}
  Subtraction  [a-z-[aeiou]]  = {b, c, d, f, g, h, ...}
```

---

## 7. Anti-Patterns

### 7.1 Anti-Pattern: Using [A-z]

```python
import re

# NG: [A-z] includes unexpected characters
pattern_bad = r'[A-z]+'
text = "Hello[World]_test"
print(re.findall(pattern_bad, text))
# => ['Hello[World]_test']  -- [ ] _ also match!

# OK: Use [A-Za-z]
pattern_good = r'[A-Za-z]+'
print(re.findall(pattern_good, text))
# => ['Hello', 'World', 'test']
```

### 7.2 Anti-Pattern: Ignoring Unicode Behavior of Shorthands

```python
import re

# NG: Forgetting that \d also matches Unicode digits
text = "Price: ١٢٣ yen"  # Arabic digits (U+0661, U+0662, U+0663)
print(re.findall(r'\d+', text))
# => ['١٢٣']  -- Python 3 matches Unicode digits

# This can be a security issue (unexpected values during numeric parsing)

# OK: Be explicit when targeting ASCII digits only
print(re.findall(r'[0-9]+', text))
# => []  -- ASCII digits only

# Or use the re.ASCII flag
print(re.findall(r'\d+', text, re.ASCII))
# => []
```

### 7.3 Anti-Pattern: Unnecessary Character Classes

```python
import re

# NG: Character class with only one character
pattern_bad = r'[a]'   # Same as 'a' but needlessly verbose
# NG: Wrapping a shorthand in a character class adds nothing
pattern_bad2 = r'[\d]'  # Same as \d

# OK: Write simply
pattern_good = r'a'
pattern_good2 = r'\d'

# However, character classes are needed when combining:
pattern_ok = r'[\d_-]'  # Digits, underscores, and hyphens
```

### 7.4 Anti-Pattern: Confusing Negated Character Classes with Dot

```python
import re

# NG: [^...] matches newlines but . does not
text = "hello\nworld"

# . does not match newlines by default
print(re.findall(r'.+', text))
# => ['hello', 'world']  # Split at the newline

# [^\n] is everything except newlines (equivalent to . but explicit)
print(re.findall(r'[^\n]+', text))
# => ['hello', 'world']

# [^a] also matches newlines!
print(re.findall(r'[^a]+', text))
# => ['hello\nworld']  # Includes the newline

# Not understanding this difference can cause bugs
```

### 7.5 Anti-Pattern: Overly Broad Character Classes

```python
import re

# NG: Using \d for numeric validation
# Port number validation
port = "65536"
if re.match(r'^\d+$', port):
    print("Valid port?")  # NG: 65536 is not a valid port number

# OK: Perform range checks in code, not with regex alone
def is_valid_port(s):
    if not re.match(r'^[0-9]+$', s):
        return False
    return 0 <= int(s) <= 65535

# NG: Using only \d{1,3} for IP address validation
ip_bad = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
# Also matches 999.999.999.999

# OK: Validate each octet's range
ip_good = re.compile(r'''
    ^
    (?:
        (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)  # 0-255
        \.
    ){3}
    (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)      # 0-255
    $
''', re.VERBOSE)
```

---

## 8. Practical Pattern Collection

### 8.1 Japanese Text Processing

```python
import re

# Detecting Hiragana
hiragana = re.compile(r'[\u3040-\u309F]+')
print(hiragana.findall("東京タワーへ行こう"))
# => ['へ', 'こう']  # Hiragana parts: particles and verb endings

# Detecting Katakana
katakana = re.compile(r'[\u30A0-\u30FF]+')
print(katakana.findall("東京タワーへ行こう"))
# => ['タワー']

# Converting full-width alphanumeric to half-width
def zen_to_han(text):
    """Convert full-width alphanumeric characters to half-width"""
    return re.sub(r'[Ａ-Ｚａ-ｚ０-９]',
                  lambda m: chr(ord(m.group()) - 0xFEE0), text)

print(zen_to_han("Ｈｅｌｌｏ ０１２３"))
# => "Hello 0123"

# Mapping for half-width to full-width Katakana conversion
han_to_zen_map = str.maketrans(
    'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ',
    'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン'
)

def han_kata_to_zen(text):
    """Convert half-width Katakana to full-width"""
    return text.translate(han_to_zen_map)

# Japanese sentence splitting
sentences = re.split(r'[。！？\n]+', "今日は天気がいい。明日も晴れるだろう！楽しみだ。")
print([s for s in sentences if s])
# => ['今日は天気がいい', '明日も晴れるだろう', '楽しみだ']
```

### 8.2 Numeric Character Class Patterns

```python
import re

# Integers (positive/negative)
integer_pattern = r'[+-]?[0-9]+'
print(re.findall(integer_pattern, "x=42, y=-17, z=+3"))
# => ['+42', '-17', '+3']  # Leading + may also be an operator

# More precise integer pattern
integer_strict = r'(?<![0-9])[+-]?[0-9]+(?![0-9.])'

# Decimals (fixed-point)
decimal_pattern = r'[+-]?[0-9]+\.[0-9]+'
print(re.findall(decimal_pattern, "pi=3.14159, e=2.71828"))
# => ['3.14159', '2.71828']

# Scientific notation
scientific = r'[+-]?[0-9]+\.?[0-9]*[eE][+-]?[0-9]+'
print(re.findall(scientific, "speed=3.0e8 tiny=1.6e-19"))
# => ['3.0e8', '1.6e-19']

# Comma-separated numbers
comma_number = r'[0-9]{1,3}(?:,[0-9]{3})*'
print(re.findall(comma_number, "Population: 1,234,567 Area: 377,975"))
# => ['1,234,567', '377,975']

# Currency notation
currency = r'[¥$€£][0-9,]+(?:\.[0-9]{2})?'
print(re.findall(currency, "Price: $1,299.99 and ¥150,000"))
# => ['$1,299.99', '¥150,000']
```


---

## Hands-On Exercises

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
        """Validate the input value"""
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

    print(f"Inefficient version: {slow_time:.4f} sec")
    print(f"Efficient version:   {fast_time:.6f} sec")
    print(f"Speedup: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be mindful of algorithmic complexity
- Choose appropriate data structures
- Measure effectiveness with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Check configuration file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Increased data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access permissions | Check execution user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check the error message**: Read the stack trace to identify the location
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify incrementally**: Use logging or debuggers to test hypotheses
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
    """Decorator that logs function inputs and outputs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Called: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return value: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception in: {func.__name__}: {e}")
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

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Look for memory leaks
3. **Check I/O waits**: Examine disk and network I/O status
4. **Check concurrent connections**: Examine connection pool status

| Problem Type | Diagnostic Tool | Solution |
|-------------|----------------|----------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference cleanup |
| I/O bottleneck | strace, iostat | Asynchronous I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes criteria for making technology choices.

| Criterion | When to prioritize | When to compromise |
|-----------|-------------------|-------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services with expected growth | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time to market | Quality-focused, mission-critical |

### Choosing an Architecture Pattern

```
┌─────────────────────────────────────────────────┐
│          Architecture Selection Flow              │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. Team size?                                  │
│    ├─ Small (1-5) -> Monolith                   │
│    └─ Large (10+) -> Go to 2                    │
│                                                 │
│  2. Deployment frequency?                       │
│    ├─ Weekly or less -> Monolith + modules      │
│    └─ Daily/multiple -> Go to 3                 │
│                                                 │
│  3. Team independence?                          │
│    ├─ High -> Microservices                     │
│    └─ Moderate -> Modular monolith              │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs. Long-term Cost**
- A fast approach in the short term can become technical debt in the long term
- Conversely, over-engineering incurs high short-term costs and delays the project

**2. Consistency vs. Flexibility**
- A unified technology stack reduces learning costs
- Adopting diverse technologies enables best-fit choices but increases operational costs

**3. Level of Abstraction**
- High abstraction increases reusability but can make debugging harder
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
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## 9. FAQ

### Q1: How do I use a hyphen as a literal inside a character class?

**A**: There are three methods:

```python
import re

# Method 1: Place at the start
print(re.findall(r'[-abc]', "a-b"))  # => ['a', '-', 'b']

# Method 2: Place at the end
print(re.findall(r'[abc-]', "a-b"))  # => ['a', '-', 'b']

# Method 3: Escape it
print(re.findall(r'[a\-c]', "a-b"))  # => ['a', '-']
```

Placing it at the start is the most common and readable approach.

### Q2: Are `\w` and `[a-zA-Z0-9_]` always the same?

**A**: **No, they are not**. When Unicode mode is enabled, `\w` also matches characters from various scripts (Kanji, Hiragana, etc.):

```python
import re

text = "hello_世界"

# Unicode mode (Python 3 default)
print(re.findall(r'\w+', text))            # => ['hello_世界']
print(re.findall(r'[a-zA-Z0-9_]+', text))  # => ['hello_']

# ASCII mode
print(re.findall(r'\w+', text, re.ASCII))  # => ['hello_']
```

### Q3: Can POSIX classes be used in Python?

**A**: Python's `re` module does **not directly support** POSIX character classes. Alternatives:

```python
import re

# Alternative for [:alpha:]
# Method 1: Use Unicode categories (regex module)
# pip install regex
# import regex
# regex.findall(r'\p{Alpha}+', text)

# Method 2: Specify ranges explicitly
alpha_ascii = r'[a-zA-Z]'

# Method 3: Combine with str.isalpha()
text = "Hello 123 World"
words = re.findall(r'\S+', text)
alpha_words = [w for w in words if w.isalpha()]
print(alpha_words)  # => ['Hello', 'World']
```

### Q4: What are Unicode Property Escapes?

**A**: `\p{...}` allows specifying Unicode categories or scripts (support varies by engine):

```javascript
// JavaScript (ES2018+ with /u flag)
const text = "Hello 世界 café";

// All Unicode "letters"
console.log(text.match(/\p{L}+/gu));
// => ['Hello', '世界', 'café']

// Japanese script
console.log(text.match(/\p{Script=Hiragana}+/gu));
// => (none)

// Kanji
console.log(text.match(/\p{Script=Han}+/gu));
// => ['世界']
```

### Q5: What about the performance of character classes?

**A**: Character classes are generally fast, but note the following:

```python
import re

# 1. Character classes are faster than alternation (|)
# Slow: a|b|c|d|e
# Fast: [a-e]

# 2. Negated character classes may be slightly slower than positive ones
# [^abc] internally checks "all characters except abc"

# 3. Unicode character classes are slower than ASCII-only
# \d (Unicode) > [0-9] (ASCII only)
# If speed matters, consider re.ASCII

# 4. Character class optimization is engine-dependent
# Many engines optimize [a-z] into a bitmap
# Large Unicode ranges may use tree-based lookups
```

### Q6: Can shorthands be used inside character classes?

**A**: Yes, they can. Shorthands are expanded inside character classes:

```python
import re

# Digits, underscores, and hyphens
print(re.findall(r'[\d_-]+', "hello_123-world"))
# => ['_123-']

# Whitespace and punctuation
print(re.findall(r'[\s,.!?]+', "hello, world! foo"))
# => [', ', '! ']

# Word characters and dots (for domain names)
print(re.findall(r'[\w.]+', "example.com hello"))
# => ['example.com', 'hello']

# Negated shorthands work too
print(re.findall(r'[\D]+', "abc123def"))  # Non-digits
# => ['abc', 'def']
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
| `[abc]` | Any single character: a, b, or c |
| `[a-z]` | Range from a to z |
| `[^abc]` | Any single character except a, b, c (negation) |
| `\d` / `\D` | Digit / Non-digit |
| `\w` / `\W` | Word character / Non-word character |
| `\s` / `\S` | Whitespace / Non-whitespace |
| `\b` / `\B` | Word boundary / Non-word boundary (zero-width) |
| `\p{L}` | Unicode character property (supported engines only) |
| `\p{Script=...}` | Unicode script specification |
| Unicode note | `\d` `\w` ranges vary by language and mode |
| Set operations | Java: `&&` (intersection), .NET: `-` (subtraction), ES2024: `v` flag |
| Golden rule | Do not use `[A-z]`; understand Unicode behavior |

---

## Recommended Next Guides

- [03-quantifiers-anchors.md](./03-quantifiers-anchors.md) -- Quantifiers and Anchors
- [../01-advanced/00-groups-backreferences.md](../01-advanced/00-groups-backreferences.md) -- Groups and Backreferences
- [../01-advanced/02-unicode-regex.md](../01-advanced/02-unicode-regex.md) -- Unicode Regular Expressions in Detail

---

## References

1. **Unicode Technical Standard #18** "Unicode Regular Expressions" https://unicode.org/reports/tr18/ -- International standard for Unicode regular expressions
2. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- Detailed coverage of character classes in Chapter 5
3. **POSIX.1-2017** "Regular Expressions" https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap09.html -- Official POSIX regular expression specification
4. **ECMAScript Language Specification** -- Unicode Property Escapes specification
5. **Python regex module** https://pypi.org/project/regex/ -- Advanced regular expression module for Python
