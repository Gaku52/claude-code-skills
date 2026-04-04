# Unicode Regular Expressions -- \p{Script}, Flags, and Normalization

> In global text processing, Unicode-aware regular expressions are essential. This guide systematically explains Unicode property escapes (`\p{...}`), normalization forms (NFC/NFD), and matching by writing system (Script).

## What You Will Learn in This Chapter

1. **The Unicode Property Escape System** -- Category and property classification of `\p{L}`, `\p{Script=Han}`, etc.
2. **The Relationship Between Unicode Normalization and Regular Expressions** -- How NFC/NFD/NFKC/NFKD affect search results
3. **Practical Multilingual Text Processing** -- Matching techniques for Japanese, Chinese, Arabic, and more
4. **Regex Processing of Emoji** -- Handling compound emoji and grapheme clusters
5. **Unicode Support Differences Across Languages** -- Differences between Python, JavaScript, Java, Go, and Rust
6. **Building Normalization Pipelines in Practice** -- Pre-processing for search, comparison, and validation


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Lookahead and Lookbehind -- (?=)(?!)(?<=)(?<!)](./01-lookaround.md)

---

## 1. Unicode Fundamentals

### 1.1 The Structure of Unicode

```
Unicode code point space:

U+0000 - U+007F    Basic Latin (ASCII)              128 characters
U+0080 - U+07FF    Latin, Greek, Cyrillic, etc.     ~1,920 characters
U+0800 - U+FFFF    CJK, Hiragana, Katakana, etc.   ~63,488 characters
U+10000 - U+10FFFF Emoji, ancient scripts, etc.     ~1,048,576 characters

Total: approximately 149,000 characters assigned (Unicode 16.0)

UTF-8 encoding:
+-------------------+-----------+----------------+
| Code Point        | Bytes     | Example        |
+-------------------+-----------+----------------+
| U+0000-U+007F    | 1 byte    | 'A' = 0x41     |
| U+0080-U+07FF    | 2 bytes   | 'e' = C3 A9    |
| U+0800-U+FFFF    | 3 bytes   | 'Han' = E6 BC A2|
| U+10000-U+10FFFF | 4 bytes   | '😀' = F0 9F 98 80|
+-------------------+-----------+----------------+
```

### 1.2 Unicode Categories (General Category)

```
+---------------------------------------------------+
|              Unicode General Category               |
+------+--------------------------------------------+
| L    | Letter                                      |
|  Lu  |  Uppercase Letter                           |
|  Ll  |  Lowercase Letter                           |
|  Lt  |  Titlecase Letter                           |
|  Lm  |  Modifier Letter                            |
|  Lo  |  Other Letter (CJK characters, kana, etc.)  |
+------+--------------------------------------------+
| M    | Mark (combining characters)                  |
|  Mn  |  Nonspacing Mark                            |
|  Mc  |  Spacing Combining Mark                     |
|  Me  |  Enclosing Mark                             |
+------+--------------------------------------------+
| N    | Number                                       |
|  Nd  |  Decimal Digit Number                       |
|  Nl  |  Letter Number (Roman numerals, etc.)       |
|  No  |  Other Number (fractions, etc.)             |
+------+--------------------------------------------+
| P    | Punctuation                                  |
|  Pc  |  Connector Punctuation (_ etc.)             |
|  Pd  |  Dash Punctuation (- - -- etc.)             |
|  Ps  |  Open Punctuation (( [ { etc.)              |
|  Pe  |  Close Punctuation () ] } etc.)             |
|  Pi  |  Initial Quote (« ' " etc.)                 |
|  Pf  |  Final Quote (» ' " etc.)                   |
|  Po  |  Other Punctuation (. , ; : ! ? etc.)       |
+------+--------------------------------------------+
| S    | Symbol                                       |
|  Sc  |  Currency Symbol ($ € ¥ £ etc.)             |
|  Sk  |  Modifier Symbol (^ ` ´ ¨ etc.)            |
|  Sm  |  Math Symbol (+ = < > ± × ÷ etc.)          |
|  So  |  Other Symbol (© ® ™ ° etc.)               |
+------+--------------------------------------------+
| Z    | Separator                                    |
|  Zs  |  Space Separator                            |
|  Zl  |  Line Separator                             |
|  Zp  |  Paragraph Separator                        |
+------+--------------------------------------------+
| C    | Other (control characters, etc.)              |
|  Cc  |  Control                                    |
|  Cf  |  Format (ZWJ, BOM, etc.)                    |
|  Cs  |  Surrogate                                  |
|  Co  |  Private Use                                |
|  Cn  |  Unassigned                                 |
+------+--------------------------------------------+
```

### 1.3 Major Unicode Writing Systems (Script)

```
+------------------------------------------------------------+
|              Major Unicode Script List                       |
+------------------+-----------------------------------------+
| Script Name      | Example Characters                       |
+------------------+-----------------------------------------+
| Latin            | A-Z a-z À-ÿ (Latin characters)           |
| Han              | 漢 字 東 京 (CJK Unified Ideographs)     |
| Hiragana         | あ い う え お (Hiragana)                 |
| Katakana         | ア イ ウ エ オ (Katakana)                 |
| Hangul           | 가 나 다 라 (Korean)                     |
| Cyrillic         | А Б В Г (Russian, etc.)                  |
| Arabic           | ا ب ت ث (Arabic)                        |
| Devanagari       | अ आ इ ई (Hindi, etc.)                   |
| Greek            | Α Β Γ Δ (Greek)                          |
| Thai             | ก ข ค ง (Thai)                           |
| Hebrew           | א ב ג ד (Hebrew)                        |
| Bengali          | অ আ ই ঈ (Bengali)                       |
| Tamil            | அ ஆ இ ஈ (Tamil)                         |
| Ethiopic         | ሀ ለ ሐ መ (Amharic, etc.)                 |
| Common           | 0-9 , . ! ? @ (shared across scripts)    |
| Inherited        | Combining characters (inherit parent      |
|                  | script)                                   |
+------------------+-----------------------------------------+
```

### 1.4 Unicode Binary Properties

```python
import regex

# Unicode binary properties (have true/false values)
text = "Hello! 123 café Α Β"

# Alphabetic -- alphabetic characters
print(regex.findall(r'\p{Alphabetic}+', text))
# => ['Hello', 'café', 'Α', 'Β']

# White_Space -- whitespace characters
print(regex.findall(r'\p{White_Space}+', text))
# => [' ', ' ', ' ', ' ']

# Uppercase / Lowercase
print(regex.findall(r'\p{Uppercase}+', text))
# => ['H', 'Α', 'Β']

print(regex.findall(r'\p{Lowercase}+', text))
# => ['ello', 'caf', 'é']

# ID_Start / ID_Continue -- programming language identifiers
# ID_Start: characters valid at the start of an identifier
# ID_Continue: characters valid from the 2nd position onward
identifier_pattern = regex.compile(r'\p{ID_Start}\p{ID_Continue}*')
code_text = "変数名 variable_1 _private 42invalid"
print(identifier_pattern.findall(code_text))
# => ['変数名', 'variable_1', '_private']

# Emoji properties
emoji_text = "Hello 👋 World 🌍 Test 1️⃣ #️⃣"
print(regex.findall(r'\p{Emoji_Presentation}', emoji_text))
# => ['👋', '🌍']

# Extended_Pictographic -- broader emoji range
print(regex.findall(r'\p{Extended_Pictographic}', emoji_text))
```

---

## 2. Unicode Property Escapes `\p{...}`

### 2.1 Basic Syntax

```python
# Python: requires the regex module (third-party)
# pip install regex
import regex

text = "Hello 世界 café 123 ١٢٣"

# \p{L} -- all letters
print(regex.findall(r'\p{L}+', text))
# => ['Hello', '世界', 'café']

# \p{N} -- all numbers
print(regex.findall(r'\p{N}+', text))
# => ['123', '١٢٣']

# \p{Lu} -- uppercase only
print(regex.findall(r'\p{Lu}', text))
# => ['H']

# \P{L} -- non-letters (negation)
print(regex.findall(r'\P{L}+', text))
# => [' ', ' ', ' ', ' ١٢٣']

# \p{Ll} -- lowercase only
print(regex.findall(r'\p{Ll}+', text))
# => ['ello', '世界', 'café']
# Note: CJK characters are Lo (Other Letter), not Ll

# \p{Lo} -- other letters (CJK characters, kana, etc.)
print(regex.findall(r'\p{Lo}+', text))
# => ['世界']
```

### 2.2 Unicode Properties in JavaScript (ES2018+)

```javascript
const text = "Hello 世界 café 123 ١٢٣";

// \p{L} -- all letters
console.log(text.match(/\p{L}+/gu));
// => ['Hello', '世界', 'café']

// \p{Script=Han} -- CJK characters only
console.log(text.match(/\p{Script=Han}+/gu));
// => ['世界']

// \p{Emoji} -- emoji
const emojiText = "Hello 👋 World 🌍!";
console.log(emojiText.match(/\p{Emoji}/gu));
// => ['👋', '🌍']

// The u flag is required
// /\p{L}/g  -> SyntaxError (without u flag)
// /\p{L}/gu -> OK

// v flag (ES2024): extended version of u
// Enables set operations
// /[\p{L}&&\p{ASCII}]/gv  -- ASCII characters that are also Letters
// /[\p{L}--\p{Script=Latin}]/gv  -- Letters other than Latin
```

### 2.3 Script (Writing System) Properties

```python
import regex

text = "日本語テスト English Русский العربية"

# Extract each writing system individually
print(regex.findall(r'\p{Script=Han}+', text))
# => ['日本語']  (CJK characters)

print(regex.findall(r'\p{Script=Hiragana}+', text))
# => []  (no hiragana in this example)

print(regex.findall(r'\p{Script=Katakana}+', text))
# => ['テスト']

print(regex.findall(r'\p{Script=Latin}+', text))
# => ['English']

print(regex.findall(r'\p{Script=Cyrillic}+', text))
# => ['Русский']

print(regex.findall(r'\p{Script=Arabic}+', text))
# => ['العربية']
```

### 2.4 The Script_Extensions Property

```python
import regex

# Difference between Script and Script_Extensions
# Script: belongs to only one writing system
# Script_Extensions: includes characters used in multiple writing systems

# Example: the prolonged sound mark "ー" (U+30FC)
# Script=Katakana, but Script_Extensions also includes Hiragana

text = "カタカナ ひらがなー"

# Script=Katakana only
print(regex.findall(r'\p{Script=Katakana}+', text))
# => ['カタカナ', 'ー']  -- ー has Script of Katakana

# CJK number characters have Script=Han
text2 = "一二三 123"
print(regex.findall(r'\p{Script=Han}+', text2))
# => ['一二三']

# CJK punctuation has Common Script
text3 = "日本語。English"
print(regex.findall(r'\p{Script=Common}', text3))
# => ['。']  -- Punctuation is Common
```

### 2.5 Processing Japanese Text

```python
import regex

text = "東京都は Tokyo とも呼ばれ、人口は約1400万人です。"

# Kanji (CJK characters)
kanji = regex.findall(r'\p{Script=Han}+', text)
print(f"Kanji: {kanji}")
# => Kanji: ['東京都', '呼', '人口', '約', '万人']

# Hiragana
hiragana = regex.findall(r'\p{Script=Hiragana}+', text)
print(f"Hiragana: {hiragana}")
# => Hiragana: ['は', 'とも', 'ばれ', 'は', 'です']

# Katakana
katakana = regex.findall(r'\p{Script=Katakana}+', text)
print(f"Katakana: {katakana}")
# => Katakana: []

# All Japanese characters (Kanji + Hiragana + Katakana)
japanese = regex.findall(r'[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]+', text)
print(f"Japanese: {japanese}")
# => Japanese: ['東京都は', 'とも呼ばれ', '人口は約', '万人です']

# Numbers (both full-width and half-width)
numbers = regex.findall(r'[\p{Nd}]+', text)
print(f"Numbers: {numbers}")
# => Numbers: ['1400']
```

### 2.6 Japanese-Specific Character Ranges and Regular Expressions

```python
import regex
import re

# Unicode blocks/ranges commonly used in Japanese processing
# These can be used with re (no regex module needed)

# Hiragana: U+3040 - U+309F
# Katakana: U+30A0 - U+30FF
# CJK Unified Ideographs: U+4E00 - U+9FFF
# CJK Unified Ideographs Extension A: U+3400 - U+4DBF
# Fullwidth Alphanumerics: U+FF01 - U+FF5E
# Halfwidth Katakana: U+FF65 - U+FF9F
# CJK Symbols and Punctuation: U+3000 - U+303F

text = "東京タワー（とうきょうタワー）は高さ333mの電波塔です。"

# Match CJK characters with re module
# Kanji
print(re.findall(r'[\u4e00-\u9fff]+', text))
# => ['東京', '高', '電波塔']

# Hiragana
print(re.findall(r'[\u3040-\u309f]+', text))
# => ['とうきょう', 'は', 'さ', 'の', 'です']

# Katakana
print(re.findall(r'[\u30a0-\u30ff]+', text))
# => ['タワー', 'タワー']

# All Japanese (Kanji + Hiragana + Katakana)
print(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', text))
# => ['東京タワー', 'とうきょうタワー', 'は高さ', 'の電波塔です']

# Estimating morphological boundaries from Japanese text
# (Simplified: split at character type boundaries)
def split_japanese(text: str) -> list[str]:
    """Split Japanese text at character type boundaries"""
    # Boundaries like Kanji->Hiragana, Katakana->Kanji, etc.
    pattern = regex.compile(
        r'[\p{Script=Han}]+'
        r'|[\p{Script=Hiragana}]+'
        r'|[\p{Script=Katakana}ー]+'  # Include prolonged sound mark
        r'|[\p{Script=Latin}]+'
        r'|[\p{Nd}]+'
        r'|\S'
    )
    return pattern.findall(text)

result = split_japanese("東京タワーは高さ333mの電波塔です")
print(result)
# => ['東京', 'タワー', 'は', '高', 'さ', '333', 'm', 'の', '電波塔', 'です']
```

### 2.7 Processing Various Number Systems

```python
import regex

# Unicode has various number systems
text = "Latin: 123, Arabic: ١٢٣, Devanagari: १२३, Thai: ๑๒๓, CJK: ３２１"

# \p{Nd} -- all decimal digits
all_digits = regex.findall(r'\p{Nd}+', text)
print(f"All digits: {all_digits}")
# => ['123', '١٢٣', '१२३', '๑๒๓', '３２１']

# Specific script digits only
# Latin digits only
print(regex.findall(r'[0-9]+', text))
# => ['123']

# Arabic-Indic digits
print(regex.findall(r'[\u0660-\u0669]+', text))
# => ['١٢٣']

# Fullwidth digits
print(regex.findall(r'[\uff10-\uff19]+', text))
# => ['３２１']

# Convert all Unicode digits to ASCII digits
import unicodedata

def normalize_digits(text: str) -> str:
    """Normalize all Unicode digits to ASCII digits"""
    result = []
    for ch in text:
        if unicodedata.category(ch) == 'Nd':
            # Get numeric value via digit_value
            result.append(str(unicodedata.digit(ch)))
        else:
            result.append(ch)
    return ''.join(result)

normalized = normalize_digits(text)
print(f"After normalization: {normalized}")
# => "Latin: 123, Arabic: 123, Devanagari: 123, Thai: 123, CJK: 321"
```

---

## 3. Unicode Normalization

### 3.1 The Four Normalization Forms

```
NFC  (Canonical Decomposition + Canonical Composition)
NFD  (Canonical Decomposition)
NFKC (Compatibility Decomposition + Canonical Composition)
NFKD (Compatibility Decomposition)

Example: ways to represent "cafe"

NFC:  c a f e        (4 characters -- e is 1 code point U+00E9)
NFD:  c a f e ◌́      (5 characters -- e + combining acute U+0301)

Both look the same, but the byte sequences differ!

NFKC/NFKD additionally decompose compatibility characters:
  "ﬁ" (U+FB01) -> "fi" (2 characters)
  "①" (U+2460) -> "1"
  "Ｈｅｌｌｏ" (fullwidth) -> "Hello" (halfwidth)
```

### 3.2 How Normalization Affects Regular Expressions

```python
import unicodedata
import re

# Example where NFD and NFC produce different search results
cafe_nfc = "café"                    # NFC: é = U+00E9
cafe_nfd = "cafe\u0301"             # NFD: e + ◌́ = U+0065 + U+0301

print(f"NFC: {repr(cafe_nfc)}")     # => 'caf\xe9'
print(f"NFD: {repr(cafe_nfd)}")     # => 'cafe\u0301'
print(f"Visually identical: {cafe_nfc} == {cafe_nfd}")  # Look the same

# Search for "é" with regex
pattern = r'café'
print(bool(re.search(pattern, cafe_nfc)))  # => True
print(bool(re.search(pattern, cafe_nfd)))  # => False!

# Solution: normalize before searching
normalized = unicodedata.normalize('NFC', cafe_nfd)
print(bool(re.search(pattern, normalized)))  # => True
```

### 3.3 Detailed Behavior of Normalization Forms

```python
import unicodedata

# Detailed differences between each normalization form

# Test case 1: Accented character
e_acute = '\u00e9'          # é (NFC form)
e_combining = 'e\u0301'     # e + combining acute (NFD form)

print("=== Accented character ===")
print(f"NFC:  {repr(unicodedata.normalize('NFC', e_acute))}")   # => '\xe9'
print(f"NFD:  {repr(unicodedata.normalize('NFD', e_acute))}")   # => 'e\u0301'
print(f"NFKC: {repr(unicodedata.normalize('NFKC', e_acute))}")  # => '\xe9'
print(f"NFKD: {repr(unicodedata.normalize('NFKD', e_acute))}")  # => 'e\u0301'

# Test case 2: Ligature
fi_ligature = '\ufb01'  # ﬁ

print("\n=== Ligature ===")
print(f"NFC:  {repr(unicodedata.normalize('NFC', fi_ligature))}")   # => '\ufb01'
print(f"NFD:  {repr(unicodedata.normalize('NFD', fi_ligature))}")   # => '\ufb01'
print(f"NFKC: {repr(unicodedata.normalize('NFKC', fi_ligature))}")  # => 'fi'
print(f"NFKD: {repr(unicodedata.normalize('NFKD', fi_ligature))}")  # => 'fi'

# Test case 3: Fullwidth alphanumerics
fullwidth = '\uff28\uff45\uff4c\uff4c\uff4f'  # Ｈｅｌｌｏ

print("\n=== Fullwidth alphanumerics ===")
print(f"NFC:  {unicodedata.normalize('NFC', fullwidth)}")   # => Ｈｅｌｌｏ
print(f"NFD:  {unicodedata.normalize('NFD', fullwidth)}")   # => Ｈｅｌｌｏ
print(f"NFKC: {unicodedata.normalize('NFKC', fullwidth)}")  # => Hello
print(f"NFKD: {unicodedata.normalize('NFKD', fullwidth)}")  # => Hello

# Test case 4: Circled numbers
circled = '\u2460\u2461\u2462'  # ①②③

print("\n=== Circled numbers ===")
print(f"NFC:  {unicodedata.normalize('NFC', circled)}")   # => ①②③
print(f"NFKC: {unicodedata.normalize('NFKC', circled)}")  # => 123

# Test case 5: Roman numerals
roman = '\u2160\u2161\u2162'  # ⅠⅡⅢ

print("\n=== Roman numerals ===")
print(f"NFC:  {unicodedata.normalize('NFC', roman)}")   # => ⅠⅡⅢ
print(f"NFKC: {unicodedata.normalize('NFKC', roman)}")  # => III
```

### 3.4 Practical Normalization Pipeline

```python
import unicodedata
import re

def normalize_and_search(pattern: str, text: str, form: str = 'NFC') -> list:
    """Normalize, then search"""
    norm_text = unicodedata.normalize(form, text)
    norm_pattern = unicodedata.normalize(form, pattern)
    return re.findall(norm_pattern, norm_text)

# Handle mixed fullwidth/halfwidth text (NFKC)
text = "Ｈｅｌｌｏ　Ｗｏｒｌｄ　１２３"  # Fullwidth
normalized = unicodedata.normalize('NFKC', text)
print(normalized)         # => "Hello World 123"
print(re.findall(r'\w+', normalized))
# => ['Hello', 'World', '123']
```

### 3.5 Japanese Text Normalization Pipeline

```python
import unicodedata
import re

def normalize_japanese_text(text: str) -> str:
    """Comprehensive normalization of Japanese text

    Performs the following:
    1. NFKC normalization (fullwidth alphanumerics -> halfwidth, unify compatibility chars)
    2. Fullwidth space -> halfwidth space
    3. Consolidate consecutive whitespace
    4. Strip leading/trailing whitespace
    """
    # Step 1: NFKC normalization
    text = unicodedata.normalize('NFKC', text)

    # Step 2: Fullwidth space -> halfwidth space
    text = text.replace('\u3000', ' ')

    # Step 3: Collapse consecutive whitespace to one
    text = re.sub(r'\s+', ' ', text)

    # Step 4: Strip leading/trailing whitespace
    text = text.strip()

    return text

# Test
test_cases = [
    "Ｈｅｌｌｏ　Ｗｏｒｌｄ",          # Fullwidth alphanumerics and fullwidth spaces
    "テスト　　テスト",                  # Consecutive fullwidth spaces
    "①②③の手順",                       # Circled numbers
    "ﬁnally　ﬁnished",               # Ligatures
    "  前後に  スペース  ",             # Leading/trailing whitespace
]

for tc in test_cases:
    result = normalize_japanese_text(tc)
    print(f"  '{tc}' => '{result}'")

# Pre-search normalization pipeline
def search_normalized(pattern: str, text: str, flags=0) -> list:
    """Search in normalized text"""
    norm_text = normalize_japanese_text(text)
    norm_pattern = normalize_japanese_text(pattern)
    return re.findall(norm_pattern, norm_text, flags)

# Usage: search text with mixed fullwidth/halfwidth
text = "電話番号は０３−１２３４−５６７８です"
results = search_normalized(r'\d{2,4}[-−]\d{4}[-−]\d{4}', text)
print(f"Phone number: {results}")
# => ['03-1234-5678']
```

### 3.6 Normalization Caveats and Edge Cases

```python
import unicodedata

# Caveat 1: NFKC includes irreversible transformations
# Circled number ① -> 1 is irreversible
circled_1 = '\u2460'  # ①
nfkc = unicodedata.normalize('NFKC', circled_1)
print(f"NFKC of ①: '{nfkc}' (U+{ord(nfkc):04X})")
# => '1' (U+0031) -- becomes a regular digit

# Caveat 2: Normalization of CJK Compatibility Ideographs
# Some CJK compatibility ideographs are unified by NFKC
# Example: U+FA30 (CJK compat) -> may remain U+FA30 (not always changed)

# Caveat 3: Katakana "ヴ" and combining characters
# "ヴ" (U+30F4) stays as a single character in both NFC and NFD
vu = '\u30f4'  # ヴ
print(f"NFC: {repr(unicodedata.normalize('NFC', vu))}")   # => '\u30f4'
print(f"NFD: {repr(unicodedata.normalize('NFD', vu))}")   # => '\u30f4'

# Caveat 4: Normalization of halfwidth katakana
# NFKC converts halfwidth katakana to fullwidth katakana
halfwidth_katakana = '\uff76\uff80\uff76\uff85'  # ｶﾀｶﾅ
nfkc = unicodedata.normalize('NFKC', halfwidth_katakana)
print(f"NFKC of halfwidth katakana: '{nfkc}'")
# => 'カタカナ' (converted to fullwidth katakana)

# Caveat 5: Handling voiced/semi-voiced marks
# Halfwidth katakana "ガ" = ｶ + ﾞ (2 characters)
# NFKC combines them into fullwidth "ガ" (1 character)
ga_halfwidth = '\uff76\uff9e'  # ｶﾞ
ga_nfkc = unicodedata.normalize('NFKC', ga_halfwidth)
print(f"NFKC of ｶﾞ: '{ga_nfkc}' (len={len(ga_nfkc)})")
# => 'ガ' (len=1)
```

---

## 4. Unicode Flags and Modes

### 4.1 Unicode Flags by Language

```python
import re

text = "café CAFÉ"

# Python 3: Unicode-aware by default
# \w matches Unicode characters
print(re.findall(r'\w+', text))
# => ['café', 'CAFÉ']

# re.ASCII: restrict to ASCII only
print(re.findall(r'\w+', text, re.ASCII))
# => ['caf', 'CAF']   # é does not match

# re.IGNORECASE + Unicode
print(re.findall(r'café', text, re.IGNORECASE))
# => ['café', 'CAFÉ']
```

```javascript
// JavaScript: u flag (ES2015+)
const text = "café CAFÉ";

// Without u flag: surrogate pair issues
console.log("😀".match(/^.$/));   // => null (2 code units)
console.log("😀".match(/^.$/u));  // => ['😀'] (1 code point)

// v flag (ES2024): extension of u
// Set operations: intersection, difference
console.log("aéあ".match(/[\p{L}&&\p{ASCII}]/gv));
// => ['a']  (ASCII AND Letter)
```

### 4.2 Differences in Unicode Support Across Languages

```python
# Python 3 Unicode support
import re

text = "café naïve résumé"

# In Python 3, \w, \d, \s are Unicode-aware by default
print(re.findall(r'\w+', text))
# => ['café', 'naïve', 'résumé']

# \b is also a Unicode word boundary
print(re.findall(r'\b\w+\b', text))
# => ['café', 'naïve', 'résumé']

# re.ASCII flag restricts to ASCII mode
print(re.findall(r'\w+', text, re.ASCII))
# => ['caf', 'na', 've', 'r', 'sum']

# Can also be specified as an inline flag
print(re.findall(r'(?a)\w+', text))  # (?a) = re.ASCII
# => ['caf', 'na', 've', 'r', 'sum']
```

```java
// Java Unicode support
import java.util.regex.*;

public class UnicodeJava {
    public static void main(String[] args) {
        String text = "café naïve 東京";

        // Default: \w is [a-zA-Z_0-9] only (not Unicode-aware)
        Pattern p1 = Pattern.compile("\\w+");
        Matcher m1 = p1.matcher(text);
        while (m1.find()) {
            System.out.println(m1.group());
        }
        // => "caf", "na", "ve", (東京 does not match)

        // UNICODE_CHARACTER_CLASS flag enables Unicode support
        Pattern p2 = Pattern.compile("\\w+",
            Pattern.UNICODE_CHARACTER_CLASS);
        Matcher m2 = p2.matcher(text);
        while (m2.find()) {
            System.out.println(m2.group());
        }
        // => "café", "naïve", "東京"

        // \p{L} works in Java without any flag
        Pattern p3 = Pattern.compile("\\p{L}+");
        Matcher m3 = p3.matcher(text);
        while (m3.find()) {
            System.out.println(m3.group());
        }
        // => "café", "naïve", "東京"
    }
}
```

```go
// Go (RE2) Unicode support
package main

import (
    "fmt"
    "regexp"
)

func main() {
    text := "café naïve 東京"

    // Go's \w is ASCII only
    re1 := regexp.MustCompile(`\w+`)
    fmt.Println(re1.FindAllString(text, -1))
    // => ["caf", "na", "ve"]

    // Unicode character properties are available
    // \p{L} -- Unicode Letter
    re2 := regexp.MustCompile(`\p{L}+`)
    fmt.Println(re2.FindAllString(text, -1))
    // => ["café", "naïve", "東京"]

    // \p{Han} -- CJK characters
    re3 := regexp.MustCompile(`\p{Han}+`)
    fmt.Println(re3.FindAllString(text, -1))
    // => ["東京"]

    // \p{Hiragana}, \p{Katakana} are also available
    text2 := "ひらがなカタカナ漢字"
    re4 := regexp.MustCompile(`\p{Hiragana}+`)
    fmt.Println(re4.FindAllString(text2, -1))
    // => ["ひらがな"]
}
```

### 4.3 Unicode Case Conversion Issues

```python
import re

# Unicode case conversion is not always 1:1
# German ß -> SS (1 character becomes 2)
text = "straße STRASSE"

print(re.findall(r'stra(?:ße|sse)', text, re.IGNORECASE))
# => ['straße', 'STRASSE']

# Turkish i/I problem
# Turkish: İ (U+0130) <-> i, I <-> ı (U+0131)
# English: I <-> i
# -> IGNORECASE results vary by locale

# Case Folding
# Unicode-standard case unification transformation
text = "Straße straße STRASSE"
for word in text.split():
    print(f"  {word} -> casefold: {word.casefold()}")
# => straße -> casefold: strasse
# => straße -> casefold: strasse
# => STRASSE -> casefold: strasse
# casefold() converts more aggressively than lower()

# Unicode case-insensitive regex
# Python's re.IGNORECASE is Unicode-aware
print(re.findall(r'straße', text, re.IGNORECASE))
# => ['Straße', 'straße']
# Note: 'STRASSE' may not match in some cases (implementation-dependent)
```

### 4.4 Comprehensive Comparison of Unicode Regex Flags

```
Unicode flags by language:

Language      Flag/Option                       Effect
--------      -----------                       ------
Python        re.UNICODE (default)              \w, \d, \s are Unicode-aware
              re.ASCII (or (?a))                Restrict to ASCII only
              re.IGNORECASE                     Unicode case-insensitive

JavaScript    /u                                Unicode-aware (code point level)
              /v (ES2024)                       /u + set operations
              /i                                Case-insensitive

Java          Pattern.UNICODE_CHARACTER_CLASS   \w, \d are Unicode-aware
              Pattern.CASE_INSENSITIVE          Case-insensitive
              Pattern.UNICODE_CASE              Unicode case (requires CASE_INSENSITIVE)

Go            (default)                         \p{...} is Unicode-aware
                                                \w, \d are ASCII only

Rust          (?u) (default)                    Unicode-aware
              (?-u)                             ASCII only
```

---

## 5. Emoji Regular Expressions

### 5.1 Challenges of Emoji Matching

```python
import regex

text = "Hello 👋🏽 World 🇯🇵 Nice 👨‍👩‍👧‍👦"

# Emoji structure:
# 👋🏽 = 👋 (U+1F44B) + 🏽 (U+1FFFE, skin tone modifier) -> 2 code points
# 🇯🇵 = 🇯 (U+1F1EF) + 🇵 (U+1F1F5)                    -> 2 code points (flag)
# 👨‍👩‍👧‍👦 = 👨 + ZWJ + 👩 + ZWJ + 👧 + ZWJ + 👦         -> 7 code points

# Python regex module
emojis = regex.findall(r'\p{Emoji_Presentation}', text)
print(emojis)

# More accurate emoji pattern (grapheme clusters)
graphemes = regex.findall(r'\X', text)  # \X = grapheme cluster
print([g for g in graphemes if regex.match(r'\p{Emoji}', g)])
```

```javascript
// JavaScript (ES2024 v flag)
const text = "Hello 👋 World 🌍!";
const emojis = text.match(/\p{Emoji_Presentation}/gu);
console.log(emojis);
// => ['👋', '🌍']
```

### 5.2 Types and Structure of Emoji

```
Types of emoji:

1. Basic Emoji
   😀 = U+1F600 (1 code point)

2. Text/Emoji Presentation Toggle
   ☺️ = ☺ (U+263A) + VS16 (U+FE0F)  -- emoji presentation
   ☺ = U+263A                         -- text presentation

3. Skin Tone Modifier
   👋🏽 = 👋 (U+1F44B) + 🏽 (U+1F3FD)

4. Flags (Regional Indicator)
   🇯🇵 = 🇯 (U+1F1EF) + 🇵 (U+1F1F5)

5. ZWJ Sequences (Zero Width Joiner)
   👨‍💻 = 👨 (U+1F468) + ZWJ (U+200D) + 💻 (U+1F4BB)
   👨‍👩‍👧‍👦 = 👨 + ZWJ + 👩 + ZWJ + 👧 + ZWJ + 👦

6. Keycap
   1️⃣ = 1 (U+0031) + VS16 (U+FE0F) + ⃣ (U+20E3)

7. Tag Sequence -- sub-regional flags
   🏴󠁧󠁢󠁥󠁮󠁧󠁿 = 🏴 + TAG_g + TAG_b + TAG_e + TAG_n + TAG_g + CANCEL_TAG
```

### 5.3 Comprehensive Emoji Regex Pattern

```python
import regex

def extract_emojis(text: str) -> list[str]:
    """Extract all emoji from text as grapheme cluster units"""
    # Split into grapheme clusters with \X
    graphemes = regex.findall(r'\X', text)

    # Determine if each is an emoji
    emoji_pattern = regex.compile(
        r'[\p{Emoji_Presentation}\p{Extended_Pictographic}]'
    )

    emojis = []
    for g in graphemes:
        if emoji_pattern.search(g):
            # Exclude false positives like digits and hash
            if not regex.match(r'^[\d#*]$', g):
                emojis.append(g)

    return emojis

# Test
test_texts = [
    "Hello 😀 World",
    "Flag: 🇯🇵 🇺🇸 🇬🇧",
    "Family: 👨‍👩‍👧‍👦",
    "Skin: 👋🏻 👋🏽 👋🏿",
    "Mix: テスト🎉テスト",
    "Numbers: 1️⃣2️⃣3️⃣",
]

for text in test_texts:
    emojis = extract_emojis(text)
    print(f"  '{text}' => {emojis}")
```

### 5.4 Removing and Replacing Emoji

```python
import regex
import re

def remove_emojis(text: str) -> str:
    """Remove all emoji from text"""
    # Method 1: Use the regex module (recommended)
    return regex.sub(
        r'[\p{Emoji_Presentation}\p{Extended_Pictographic}]'
        r'[\p{Emoji_Modifier}\p{Emoji_Component}\u200d\ufe0f\ufe0e]*',
        '',
        text
    )

def replace_emojis_with_text(text: str) -> str:
    """Replace emoji with their text representation"""
    import unicodedata
    result = []
    for grapheme in regex.findall(r'\X', text):
        name = None
        for ch in grapheme:
            try:
                n = unicodedata.name(ch, None)
                if n and 'EMOJI' not in n.upper():
                    name = n
                    break
            except ValueError:
                continue
        if name and regex.search(r'[\p{Emoji_Presentation}]', grapheme):
            result.append(f'[{name}]')
        else:
            result.append(grapheme)
    return ''.join(result)

# Test
text = "素晴らしい! 🎉 今日は天気がいい ☀️ 散歩に行こう 🚶"
print(f"Original: {text}")
print(f"After removal: {remove_emojis(text)}")
```

### 5.5 Emoji Processing in JavaScript

```javascript
// Grapheme cluster segmentation using ES2024 Intl.Segmenter
function extractEmojis(text) {
    const segmenter = new Intl.Segmenter('ja', { granularity: 'grapheme' });
    const segments = [...segmenter.segment(text)];

    return segments
        .filter(seg => /\p{Emoji_Presentation}/u.test(seg.segment))
        .map(seg => seg.segment);
}

console.log(extractEmojis("Hello 😀 World 🌍 Family 👨‍👩‍👧‍👦"));
// => ['😀', '🌍', '👨‍👩‍👧‍👦']

// Character counting (by grapheme cluster)
function graphemeLength(text) {
    const segmenter = new Intl.Segmenter('ja', { granularity: 'grapheme' });
    return [...segmenter.segment(text)].length;
}

console.log("👨‍👩‍👧‍👦".length);           // => 11 (UTF-16 code unit count)
console.log(graphemeLength("👨‍👩‍👧‍👦"));  // => 1 (visual character count)
```

---

## 6. Grapheme Clusters

### 6.1 What Are Grapheme Clusters?

```
Difference between code points and grapheme clusters:

Text: "café"
  Code points (NFC): c a f é  -> 4
  Grapheme clusters:  c a f é  -> 4 (same)

Text: "cafe\u0301" (NFD)
  Code points:       c a f e ́  -> 5
  Grapheme clusters: c a f é  -> 4 (matches visual appearance)

Text: "👨‍👩‍👧‍👦"
  Code points:       👨 ZWJ 👩 ZWJ 👧 ZWJ 👦  -> 7
  Grapheme clusters: 👨‍👩‍👧‍👦                       -> 1 (visually 1 character)

Text: "🇯🇵"
  Code points:       🇯 🇵  -> 2
  Grapheme clusters: 🇯🇵   -> 1

Text: "ก้" (Thai "kor" + mai tho tone mark)
  Code points:       ก ้  -> 2
  Grapheme clusters: ก้   -> 1
```

### 6.2 Regular Expressions with Grapheme Clusters

```python
import regex

# \X -- matches a grapheme cluster (regex module)
text = "café 👨‍👩‍👧‍👦 🇯🇵 naïve"

# By code point (normal .)
import re
print(f"Code point count: {len(text)}")

# By grapheme cluster (\X)
graphemes = regex.findall(r'\X', text)
print(f"Grapheme cluster count: {len(graphemes)}")
print(f"Graphemes: {graphemes}")

# Get the first N characters by grapheme cluster
def truncate_graphemes(text: str, max_graphemes: int) -> str:
    """Truncate by visual character count"""
    graphemes = regex.findall(r'\X', text)
    return ''.join(graphemes[:max_graphemes])

# Test
long_text = "こんにちは👨‍👩‍👧‍👦世界🌍テスト"
for n in [5, 8, 10]:
    truncated = truncate_graphemes(long_text, n)
    print(f"  First {n} graphemes: '{truncated}'")
```

### 6.3 Accurate Character Count Implementation

```python
import regex
import unicodedata

def count_characters(text: str) -> dict:
    """Count various character metrics for text"""
    graphemes = regex.findall(r'\X', text)

    return {
        'bytes_utf8': len(text.encode('utf-8')),
        'bytes_utf16': len(text.encode('utf-16-le')),
        'codepoints': len(text),
        'grapheme_clusters': len(graphemes),
        'nfc_codepoints': len(unicodedata.normalize('NFC', text)),
        'nfd_codepoints': len(unicodedata.normalize('NFD', text)),
    }

# Test
test_texts = [
    ("ASCII", "Hello"),
    ("Japanese", "こんにちは"),
    ("Accent (NFC)", "caf\u00e9"),
    ("Accent (NFD)", "cafe\u0301"),
    ("Emoji", "👨‍👩‍👧‍👦"),
    ("Flag", "🇯🇵"),
    ("Mixed", "Hello世界😀"),
]

for label, text in test_texts:
    counts = count_characters(text)
    print(f"\n{label}: '{text}'")
    for key, value in counts.items():
        print(f"  {key}: {value}")
```

---

## 7. Practical Multilingual Text Processing

### 7.1 Script Detection in Multilingual Text

```python
import regex

def detect_scripts(text: str) -> dict[str, int]:
    """Detect and count writing systems present in the text"""
    scripts = {}

    script_patterns = {
        'Latin': r'\p{Script=Latin}',
        'Han': r'\p{Script=Han}',
        'Hiragana': r'\p{Script=Hiragana}',
        'Katakana': r'\p{Script=Katakana}',
        'Cyrillic': r'\p{Script=Cyrillic}',
        'Arabic': r'\p{Script=Arabic}',
        'Devanagari': r'\p{Script=Devanagari}',
        'Hangul': r'\p{Script=Hangul}',
        'Thai': r'\p{Script=Thai}',
        'Greek': r'\p{Script=Greek}',
    }

    for script_name, pattern in script_patterns.items():
        count = len(regex.findall(pattern, text))
        if count > 0:
            scripts[script_name] = count

    return scripts

# Test
test_texts = [
    "Hello World",
    "こんにちは世界",
    "Hello 世界 Мир العالم",
    "東京タワー Tokyo Tower",
    "한국어 테스트",
]

for text in test_texts:
    scripts = detect_scripts(text)
    print(f"  '{text}' => {scripts}")
```

### 7.2 Processing Arabic and Hebrew (RTL) Text

```python
import regex

# Regex processing of RTL (right-to-left) text

# Arabic text
arabic_text = "مرحبا بالعالم"  # "Hello World" in Arabic
print(regex.findall(r'\p{Script=Arabic}+', arabic_text))
# => ['مرحبا', 'بالعالم']

# Hebrew text
hebrew_text = "שלום עולם"  # "Hello World" in Hebrew
print(regex.findall(r'\p{Script=Hebrew}+', hebrew_text))
# => ['שלום', 'עולם']

# Text with mixed RTL and LTR
mixed = "Hello مرحبا World عالم"
# Extract Latin and Arabic characters separately
latin = regex.findall(r'\p{Script=Latin}+', mixed)
arabic = regex.findall(r'\p{Script=Arabic}+', mixed)
print(f"Latin: {latin}, Arabic: {arabic}")
# => Latin: ['Hello', 'World'], Arabic: ['مرحبا', 'عالم']

# BiDi (bidirectional) text processing
# Unicode directional control characters
# U+200E: LEFT-TO-RIGHT MARK (LRM)
# U+200F: RIGHT-TO-LEFT MARK (RLM)
# U+202A-U+202E: Directional control characters
# These are invisible but affect text processing

# Regex to remove directional control characters
bidi_cleanup = regex.compile(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]')
clean_text = bidi_cleanup.sub('', mixed)
print(f"After BiDi removal: {clean_text}")
```

### 7.3 Processing Chinese (Simplified/Traditional)

```python
import regex

# Distinguishing simplified and traditional Chinese
# In Unicode, they often share the same CJK Unified Ideograph code point
# Complete distinction requires specialized libraries

simplified_text = "简体中文测试"  # Simplified Chinese
traditional_text = "繁體中文測試"  # Traditional Chinese
japanese_text = "日本語漢字テスト"

# Extract CJK Unified Ideographs
for label, text in [("Simplified", simplified_text),
                     ("Traditional", traditional_text),
                     ("Japanese", japanese_text)]:
    cjk = regex.findall(r'\p{Script=Han}+', text)
    print(f"  {label}: {cjk}")

# CJK Unified Ideograph code point ranges
# U+4E00-U+9FFF: CJK Unified Ideographs (basic)
# U+3400-U+4DBF: CJK Unified Ideographs Extension A
# U+20000-U+2A6DF: CJK Unified Ideographs Extension B
# U+2A700-U+2B73F: CJK Unified Ideographs Extension C
# U+2B740-U+2B81F: CJK Unified Ideographs Extension D
# U+2B820-U+2CEAF: CJK Unified Ideographs Extension E
# U+2CEB0-U+2EBEF: CJK Unified Ideographs Extension F
# U+30000-U+3134F: CJK Unified Ideographs Extension G
```

### 7.4 Processing Korean (Hangul)

```python
import regex

korean_text = "한국어 테스트 123 Hello"

# Extract Hangul syllables
hangul = regex.findall(r'\p{Script=Hangul}+', korean_text)
print(f"Hangul: {hangul}")
# => ['한국어', '테스트']

# Hangul structure:
# Hangul syllable = Initial consonant + Medial vowel + Final consonant (optional)
# U+AC00-U+D7AF: Hangul Syllables (11,172 characters)
# U+1100-U+11FF: Hangul Jamo
# U+3130-U+318F: Hangul Compatibility Jamo

# Decompose a Hangul syllable character into Jamo
def decompose_hangul(ch: str) -> tuple[str, str, str]:
    """Decompose a Hangul syllable into initial, medial, and final"""
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return (ch, '', '')

    # Initial: 19 types, Medial: 21 types, Final: 28 types (including none)
    initial = code // (21 * 28)
    medial = (code % (21 * 28)) // 28
    final = code % 28

    initials = "ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒ"
    medials = "ᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵ"
    finals = "\0ᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ"

    i = initials[initial]
    m = medials[medial]
    f = finals[final] if final > 0 else ''

    return (i, m, f)

# Test
for ch in "한국":
    i, m, f = decompose_hangul(ch)
    print(f"  {ch} => Initial:{i} Medial:{m} Final:{f}")
```

### 7.5 Processing Indic Scripts (Devanagari)

```python
import regex

# Hindi text processing
hindi_text = "नमस्ते दुनिया 123"  # "Hello World" in Hindi

# Extract Devanagari characters
devanagari = regex.findall(r'\p{Script=Devanagari}+', hindi_text)
print(f"Devanagari: {devanagari}")
# => ['नमस्ते', 'दुनिया']

# Devanagari numerals
hindi_numbers = "१२३४५६७८९०"  # 1234567890 in Devanagari
print(regex.findall(r'\p{Nd}+', hindi_numbers))
# => ['१२३४५६७८९०']

# Handling combining characters
# In Devanagari, consonants are joined with virama (्)
# "स्ते" = स + ् + त + े (4 code points, 1 grapheme cluster)
graphemes = regex.findall(r'\X', "नमस्ते")
print(f"Grapheme clusters: {graphemes} (count: {len(graphemes)})")
```

---

## 8. ASCII Diagrams

### 8.1 Unicode Property Hierarchy

```
\p{L}  Letter (all letters)
+-- \p{Lu}  Uppercase    A B C ... Z  Á É  А Б В
+-- \p{Ll}  Lowercase    a b c ... z  á é  а б в
+-- \p{Lt}  Titlecase    ǅ ǈ ǋ (rare)
+-- \p{Lm}  Modifier     ʰ ʲ ˈ
+-- \p{Lo}  Other        漢 字 あ い う ア イ ウ

\p{N}  Number (all numbers)
+-- \p{Nd}  Decimal      0-9  ٠-٩  ०-९  ０-９
+-- \p{Nl}  Letter Num   Ⅰ Ⅱ Ⅲ Ⅳ Ⅴ
+-- \p{No}  Other Num    ½ ¼ ① ②

\p{P}  Punctuation
+-- \p{Pc}  Connector    _
+-- \p{Pd}  Dash         - – —
+-- \p{Ps}  Open         ( [ {
+-- \p{Pe}  Close        ) ] }
+-- ...

\p{S}  Symbol
+-- \p{Sc}  Currency     $ € ¥ £
+-- \p{Sm}  Math         + = < > ≤ ≥
+-- ...
```

### 8.2 Normalization Form Relationship Diagram

```
         Canonical decomposition
  NFC <---------------------------> NFD
   |                                 |
   |Compatibility                    |Compatibility
   |composition                      |decomposition
   v                                 v
  NFKC <-------------------------> NFKD
         Canonical decomposition

Example: "ﬁ" (U+FB01 LATIN SMALL LIGATURE FI)

NFC:  ﬁ (unchanged)
NFD:  ﬁ (unchanged -- no canonical decomposition)
NFKC: fi (decomposed to 2 characters)
NFKD: fi (decomposed to 2 characters)

Example: "é" (U+00E9 LATIN SMALL LETTER E WITH ACUTE)

NFC:  é        (1 character: U+00E9)
NFD:  e + ◌́    (2 characters: U+0065 + U+0301)
NFKC: é        (1 character: U+00E9)
NFKD: e + ◌́    (2 characters: U+0065 + U+0301)

Normalization selection flow:

  Storing/exchanging text?
  +-- Yes -> NFC (Web standard, most common)
  +-- No
      Searching/collating/comparing?
      +-- Yes -> NFKC (unify compatibility characters)
      +-- No
          Processing accent marks individually?
          +-- Yes -> NFD (decomposed form)
          +-- No -> NFC (recommended default)
```

### 8.3 How Surrogate Pairs Work

```
Code point representation in UTF-16:

BMP (U+0000 - U+FFFF): Represented directly as 16 bits
  'A' = U+0041 -> 0x0041 (1 code unit)
  '漢' = U+6F22 -> 0x6F22 (1 code unit)

Supplementary Planes (U+10000+): Surrogate pair (two 16-bit values)
  '😀' = U+1F600
  -> 0xD83D 0xDE00 (2 code units = surrogate pair)

  Calculation:
  code = 0x1F600 - 0x10000 = 0xF600
  high = (0xF600 >> 10) + 0xD800 = 0xD83D
  low  = (0xF600 & 0x3FF) + 0xDC00 = 0xDE00

JavaScript . (without u flag):
  "😀".length      -> 2 (surrogate pair)
  "😀".match(/./)  -> "\uD83D" (high surrogate only)

JavaScript . (with u flag):
  "😀".match(/./u) -> "😀" (correctly treated as 1 character)
```

### 8.4 How UTF-8 Encoding Works

```
UTF-8 byte structure:

1-byte character (U+0000-U+007F):
  0xxxxxxx
  Example: 'A' = 01000001 = 0x41

2-byte character (U+0080-U+07FF):
  110xxxxx 10xxxxxx
  Example: 'é' (U+00E9) = 11000011 10101001 = 0xC3 0xA9

3-byte character (U+0800-U+FFFF):
  1110xxxx 10xxxxxx 10xxxxxx
  Example: '漢' (U+6F22) = 11100110 10111100 10100010 = 0xE6 0xBC 0xA2

4-byte character (U+10000-U+10FFFF):
  11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
  Example: '😀' (U+1F600) = 11110000 10011111 10011000 10000000
                           = 0xF0 0x9F 0x98 0x80

Regex engines operate at the code point level,
so you generally do not need to be aware of UTF-8 byte structure.
However, care is needed with byte-level pattern matching (e.g., grep -P).
```

---

## 9. Comparison Tables

### 9.1 Unicode Property Support by Language

| Property | Python re | Python regex | JavaScript | Java | Go | Rust | Perl |
|----------|----------|-------------|------------|------|----|------|------|
| `\p{L}` | No | Yes | Yes (ES2018+u) | Yes | Yes | Yes | Yes |
| `\p{Lu}` | No | Yes | Yes | Yes | Yes | Yes | Yes |
| `\p{Script=Han}` | No | Yes | Yes | No | No | No | Yes |
| `\p{Han}` (short form) | No | Yes | No | No | Yes | Yes | Yes |
| `\p{Emoji}` | No | Yes | Yes | No | No | No | Yes |
| `\p{Block=CJK}` | No | Yes | No | Yes | No | No | Yes |
| `\X` (grapheme) | No | Yes | No | No | No | No | Yes |
| Unicode `\w` | Default | Default | Requires `/u` | Requires flag | No (use `\p{L}`) | Default | Default |

### 9.2 When to Use Each Normalization Form

| Form | Use Case | Characteristics | Recommended Scenarios |
|------|----------|----------------|----------------------|
| NFC | Standard for text storage/exchange | Composed form. Recommended by web standards | HTML, JSON, database storage |
| NFD | When you need to process decomposed text | Separates accent marks | Text analysis, sorting |
| NFKC | Search/collation | Unifies compatibility characters (fullwidth -> halfwidth, etc.) | Full-text search, user input normalization |
| NFKD | Search pre-processing | Maximum decomposition | Index construction |

### 9.3 Cross-Language Comparison of \w

| Language | Default `\w` Range | How to Enable Unicode |
|----------|-------------------|----------------------|
| Python 3 | Unicode characters + digits + `_` | Unicode-aware by default |
| JavaScript (no flag) | `[a-zA-Z0-9_]` | No expansion with `/u` flag |
| JavaScript `/u` | `[a-zA-Z0-9_]` | Use `\p{L}` instead |
| Java | `[a-zA-Z_0-9]` | `UNICODE_CHARACTER_CLASS` flag |
| Go | `[0-9A-Za-z_]` | Use `\p{L}` instead |
| Rust | Unicode characters + digits + `_` | `(?-u)` to restrict to ASCII |
| Perl | Unicode characters + digits + `_` | Unicode-aware by default |

---

## 10. Practical Pattern Collection

### 10.1 Internationalized Email Addresses

```python
import regex

# Internationalized email addresses (RFC 6531)
# Allow Unicode characters in local part and domain

# Basic email pattern (ASCII)
ascii_email = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# Unicode-aware email pattern
unicode_email = r'[\p{L}\p{N}._%+-]+@[\p{L}\p{N}.-]+\.[\p{L}]{2,}'

test_emails = [
    "user@example.com",           # Normal
    "田中@例え.jp",                # Japanese (internationalized)
    "пользователь@пример.рф",     # Russian (internationalized)
    "user@例え.com",               # Mixed
]

import re
for email in test_emails:
    ascii_match = bool(re.match(ascii_email, email))
    unicode_match = bool(regex.match(unicode_email, email))
    print(f"  {email}: ASCII={ascii_match}, Unicode={unicode_match}")
```

### 10.2 International Phone Numbers

```python
import re

# International phone number pattern (E.164 format)
# +[country code][number] with max 15 digits
e164_pattern = r'\+[1-9]\d{1,14}'

# Country-specific phone number formats
phone_patterns = {
    'JP': r'(?:0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4}|\+81\s?\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4})',
    'US': r'(?:\+1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}',
    'UK': r'(?:\+44[-\s]?)?\d{2,5}[-\s]?\d{3,8}',
}

test_numbers = [
    "+81-90-1234-5678",   # Japan
    "03-1234-5678",        # Japan (with area code)
    "+1 (555) 123-4567",   # US
    "+44 20 7123 4567",    # UK
]

for num in test_numbers:
    for country, pattern in phone_patterns.items():
        if re.search(pattern, num):
            print(f"  {num} => {country}")
            break
```

### 10.3 Unicode-Aware Validation

```python
import regex
import unicodedata

def validate_username(username: str) -> tuple[bool, list[str]]:
    """Unicode-aware username validation

    Rules:
    - 3 to 20 characters (by grapheme cluster count)
    - Only Unicode letters, digits, underscores, and hyphens
    - Must start with a letter
    - No control or invisible characters
    - Prohibit mixing multiple writing systems to avoid visual confusion
    """
    errors = []

    # Count grapheme clusters
    graphemes = regex.findall(r'\X', username)
    if len(graphemes) < 3:
        errors.append("Must be at least 3 characters")
    if len(graphemes) > 20:
        errors.append("Must be 20 characters or fewer")

    # Allowed characters check
    if not regex.match(r'^[\p{L}\p{N}_-]+$', username):
        errors.append("Only letters, digits, _, and - are allowed")

    # First character check
    if username and not regex.match(r'^\p{L}', username):
        errors.append("Must start with a letter")

    # Control character check
    if regex.search(r'\p{C}', username):
        errors.append("Control characters are not allowed")

    # Mixed script check (confusable attack prevention)
    scripts = set()
    for ch in username:
        cat = unicodedata.category(ch)
        if cat.startswith('L'):
            try:
                script = unicodedata.script(ch) if hasattr(unicodedata, 'script') else 'Unknown'
            except:
                script = 'Unknown'
            if script not in ('Common', 'Inherited', 'Unknown'):
                scripts.add(script)

    if len(scripts) > 1:
        errors.append(f"Mixing multiple writing systems is prohibited: {scripts}")

    return (len(errors) == 0, errors)

# Test
test_usernames = [
    "alice",           # OK: ASCII
    "田中太郎",         # OK: Japanese
    "Алексей",         # OK: Cyrillic
    "ab",              # NG: too short
    "alice_tanaka",    # OK: ASCII + underscore
    "123start",        # NG: starts with digit
]

for username in test_usernames:
    valid, errors = validate_username(username)
    status = "OK" if valid else "NG"
    print(f"  '{username}': {status} {errors if errors else ''}")
```

### 10.4 Detecting Confusable Characters

```python
import regex

# Examples of confusable characters
# These can be exploited for phishing and impersonation

confusable_pairs = [
    ('a', 'а'),      # Latin 'a' vs Cyrillic 'а' (U+0430)
    ('e', 'е'),      # Latin 'e' vs Cyrillic 'е' (U+0435)
    ('o', 'о'),      # Latin 'o' vs Cyrillic 'о' (U+043E)
    ('p', 'р'),      # Latin 'p' vs Cyrillic 'р' (U+0440)
    ('c', 'с'),      # Latin 'c' vs Cyrillic 'с' (U+0441)
    ('x', 'х'),      # Latin 'x' vs Cyrillic 'х' (U+0445)
    ('0', 'О'),      # Digit '0' vs Cyrillic 'О' (U+041E)
    ('1', 'l'),      # Digit '1' vs Latin 'l'
    ('I', 'l'),      # Latin 'I' vs Latin 'l'
]

# Detect mixed scripts
def detect_mixed_scripts(text: str) -> bool:
    """Detect mixed scripts that could indicate a confusable attack"""
    has_latin = bool(regex.search(r'\p{Script=Latin}', text))
    has_cyrillic = bool(regex.search(r'\p{Script=Cyrillic}', text))
    has_greek = bool(regex.search(r'\p{Script=Greek}', text))

    # Mixing Latin with Cyrillic or Greek characters is dangerous
    scripts = sum([has_latin, has_cyrillic, has_greek])
    return scripts > 1

# Test
suspicious_urls = [
    "example.com",      # Normal
    "ехаmple.com",      # Cyrillic 'е' and 'х' mixed in
    "gооgle.com",       # Cyrillic 'о' mixed in
    "paypal.com",       # Normal
    "раypal.com",       # Cyrillic 'р' and 'а' mixed in
]

for url in suspicious_urls:
    is_mixed = detect_mixed_scripts(url)
    if is_mixed:
        print(f"  [WARNING] '{url}' -- Mixed script detected!")
    else:
        print(f"  [OK] '{url}'")
```

---

## 11. Anti-patterns

### 11.1 Anti-pattern: Hardcoding Unicode Ranges

```python
import re
import regex

# BAD: Manually specifying Unicode ranges
pattern_bad = r'[\u3040-\u309F]+'  # Hardcoded hiragana range
# Unicode version updates may change the range

# GOOD: Use Unicode properties
pattern_good = r'\p{Script=Hiragana}+'  # regex module

text = "こんにちは"
print(regex.findall(pattern_good, text))
# => ['こんにちは']
```

### 11.2 Anti-pattern: Comparing Without Normalization

```python
import unicodedata
import re

# BAD: Comparing strings without normalization
text_nfc = "caf\u00e9"      # NFC: é (1 character)
text_nfd = "cafe\u0301"     # NFD: e + ́ (2 characters)

# They look the same, but...
print(text_nfc == text_nfd)             # => False!
print(re.search(r'café', text_nfd))     # => None!

# GOOD: Normalize before comparing
text_normalized = unicodedata.normalize('NFC', text_nfd)
print(text_nfc == text_normalized)      # => True
print(re.search(r'café', text_normalized))  # => match
```

### 11.3 Anti-pattern: Assuming . Matches All Characters

```python
import re

# BAD: . does not match newlines, and
# without the u flag (JavaScript), it matches half of a surrogate pair

# In Python this is not an issue, but...
text = "Hello 😀 World"
print(re.findall(r'.', text))
# Python 3: correctly matches one code point at a time

# BAD: combined emoji sequences get broken
text = "👨‍👩‍👧‍👦"
print(re.findall(r'.', text))
# => ['👨', '\u200d', '👩', '\u200d', '👧', '\u200d', '👦']
# ZWJ-joined emoji gets decomposed

# GOOD: process by grapheme cluster
import regex
print(regex.findall(r'\X', text))
# => ['👨\u200d👩\u200d👧\u200d👦']  -- one grapheme cluster
```

### 11.4 Anti-pattern: Using len() to Determine Character Count

```python
# BAD: len() returns the code point count
text1 = "café"           # NFC: 4 code points
text2 = "cafe\u0301"     # NFD: 5 code points
text3 = "👨‍👩‍👧‍👦"          # 7 code points

print(f"len(text1) = {len(text1)}")  # => 4
print(f"len(text2) = {len(text2)}")  # => 5 (visually 4 characters!)
print(f"len(text3) = {len(text3)}")  # => 7 (visually 1 character!)

# GOOD: Count by grapheme cluster
import regex

def visual_length(text: str) -> int:
    """Return the visual character count"""
    return len(regex.findall(r'\X', text))

print(f"visual_length(text1) = {visual_length(text1)}")  # => 4
print(f"visual_length(text2) = {visual_length(text2)}")  # => 4
print(f"visual_length(text3) = {visual_length(text3)}")  # => 1
```

### 11.5 Anti-pattern: Assuming a Specific Encoding

```python
import re

# BAD: Processing characters at the byte level
text = "漢字"
# bad: expecting text.encode('utf-8')[0:3] to get 1 character
# CJK characters are 3 bytes in UTF-8, but different in other encodings

# GOOD: Process at the string level
first_char = text[0]  # '漢'
print(first_char)

# BAD: Using byte patterns in regex
# bad: re.findall(rb'\xe6[\x80-\xbf][\x80-\xbf]', text.encode())
# This matches 3-byte UTF-8 characters but is fragile

# GOOD: Use string-level regex
print(re.findall(r'[\u4e00-\u9fff]', text))
# => ['漢', '字']

# Even better: Use Unicode properties
import regex
print(regex.findall(r'\p{Script=Han}', text))
# => ['漢', '字']
```

---

## 12. FAQ

### Q1: How can I use `\p{L}` in Python's `re` module?

**A**: The standard `re` module does not support it. Use the third-party `regex` module:

```bash
pip install regex
```

```python
import regex

text = "Hello 世界"
print(regex.findall(r'\p{L}+', text))
# => ['Hello', '世界']

# Alternative with the re module:
import re
# Method 1: Unicode category flag workaround
print(re.findall(r'[^\W\d_]+', text))  # Negate \W and exclude digits and _
# => ['Hello', '世界']
```

### Q2: What is the best way to accurately detect emoji?

**A**: Emoji consist of multiple code points, so simple patterns are insufficient. Using grapheme clusters (`\X`) is the best approach:

```python
import regex

text = "Hi 👨‍👩‍👧‍👦 there 🇯🇵"

# Split into grapheme clusters with \X
graphemes = regex.findall(r'\X', text)
emoji_graphemes = [g for g in graphemes if regex.search(r'\p{Emoji}', g) and not regex.match(r'[\d#*]', g)]
print(emoji_graphemes)
```

In JavaScript, you can also use `Intl.Segmenter` (ES2022).

### Q3: How do I unify fullwidth and halfwidth characters for searching?

**A**: Apply NFKC normalization as a pre-processing step:

```python
import unicodedata
import re

text = "Ｈｅｌｌｏ　Ｗｏｒｌｄ　１２３"

# NFKC normalization: convert fullwidth alphanumerics to halfwidth
normalized = unicodedata.normalize('NFKC', text)
print(normalized)  # => "Hello World 123"

# After normalization, you can search with normal regex
print(re.findall(r'[A-Za-z]+', normalized))
# => ['Hello', 'World']

print(re.findall(r'\d+', normalized))
# => ['123']
```

### Q4: Can Unicode version upgrades change regex behavior?

**A**: Yes. When new characters are added in a new Unicode version, the match range of `\p{L}` or `\p{Script=Han}` may change:

```python
# Example: characters added in Unicode 15.0 will not
# be recognized by older versions of Python/regex

# Countermeasures:
# 1. Regularly update the runtime version
# 2. Explicitly document the Unicode version dependency
# 3. Verify character range coverage with unit tests
```

### Q5: Is there a way to automatically perform Unicode normalization in regex?

**A**: Some engines support "canonical equivalence matching," but it is not common:

```python
# ICU (International Components for Unicode)-based engines:
# CANONICAL_EQUIVALENCE flag enables canonically equivalent pattern matching

# In Java:
# Pattern.compile("café", Pattern.CANON_EQ)
# This matches both "café" (NFC) and "cafe\u0301" (NFD)

# In Python, pre-normalizing is the standard approach:
import unicodedata
import re

def canonical_search(pattern: str, text: str):
    """Canonical equivalence match"""
    nfc_text = unicodedata.normalize('NFC', text)
    nfc_pattern = unicodedata.normalize('NFC', pattern)
    return re.findall(nfc_pattern, nfc_text)
```

### Q6: What is the difference between `\w` and `\p{L}`?

**A**: `\w` is "word characters" equivalent to `[\p{L}\p{N}\p{Pc}]` (letters, numbers, and connector punctuation). `\p{L}` is "letters" only:

```python
import regex

text = "hello_123_世界"

# \w: letters + digits + _
print(regex.findall(r'\w+', text))
# => ['hello_123_世界']

# \p{L}: letters only
print(regex.findall(r'\p{L}+', text))
# => ['hello', '世界']

# \p{N}: digits only
print(regex.findall(r'\p{N}+', text))
# => ['123']
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
| `\p{L}` | Matches all Unicode letters |
| `\p{N}` | Matches all Unicode numbers |
| `\p{Script=Han}` | Matches CJK characters only |
| `\p{Script=Hiragana}` | Matches hiragana only |
| `\p{Script=Katakana}` | Matches katakana only |
| `\p{Emoji}` | Matches emoji |
| `\X` | Grapheme cluster (regex module) |
| NFC | Composed form (web standard, text storage) |
| NFD | Decomposed form (accent processing) |
| NFKC | Compatibility decomposition + composition (for search, fullwidth -> halfwidth) |
| NFKD | Compatibility decomposition (maximum decomposition) |
| `/u` flag | Enables Unicode support in JavaScript |
| `/v` flag | JavaScript ES2024 set operations |
| Golden rule | Normalize before searching; do not hardcode properties |
| Character count | Count by grapheme cluster, not len() |

## Recommended Next Reads

- [03-performance.md](./03-performance.md) -- Performance and ReDoS countermeasures
- [../02-practical/00-language-specific.md](../02-practical/00-language-specific.md) -- Language-specific regex differences

## References

1. **Unicode Technical Standard #18** "Unicode Regular Expressions" https://unicode.org/reports/tr18/ -- The international standard specification for Unicode regular expressions
2. **Unicode Technical Report #15** "Unicode Normalization Forms" https://unicode.org/reports/tr15/ -- Official specification for normalization forms
3. **Unicode Technical Standard #51** "Unicode Emoji" https://unicode.org/reports/tr51/ -- Official specification for emoji
4. **Mathias Bynens** "JavaScript has a Unicode problem" https://mathiasbynens.be/notes/javascript-unicode -- Unicode issues and solutions in JavaScript
5. **Python regex module** https://github.com/mrabarnett/mrab-regex -- Feature-rich regex module for Python
6. **TC39 RegExp v flag proposal** https://github.com/tc39/proposal-regexp-v-flag -- JavaScript ES2024 v flag specification
7. **Unicode CLDR** https://cldr.unicode.org/ -- Unicode Common Locale Data Repository (locale-specific character processing)
8. **Unicode Confusables** https://unicode.org/reports/tr39/ -- Unicode Security Mechanisms (confusable character detection)
