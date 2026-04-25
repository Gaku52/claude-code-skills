# Common Patterns -- Email, URL, Date, Phone Number

> This guide explains frequently used regular expression patterns from both a "practical level" and a "strict level" perspective. Understand the limits of each pattern correctly and learn robust validation designs that don't rely solely on regular expressions.

## What You Will Learn in This Chapter

1. **Practical implementations of common patterns** -- email, URL, date, phone number, IP address, etc.
2. **Trade-offs between strict specification compliance and practicality** -- why full RFC compliance is usually unnecessary
3. **Design patterns combining regex + additional validation** -- validation that cannot be completed by pattern matching alone


## Prerequisites

To deepen your understanding of this guide, the following knowledge is helpful:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Language-specific Regex -- Differences across JS/Python/Go/Rust/Java](./00-language-specific.md)

---

## 1. Email Address

### 1.1 Practical Pattern

```python
import re

# Practical level: matches most real-world email addresses
email_pattern = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

test_emails = [
    "user@example.com",           # OK
    "user.name+tag@domain.co.jp", # OK
    "user@sub.domain.example.com",# OK
    "user@",                      # NG
    "@domain.com",                # NG
    "user@domain",                # NG (no TLD)
    "user name@domain.com",       # NG (whitespace)
]

for email in test_emails:
    result = "OK" if email_pattern.match(email) else "NG"
    print(f"  {result}: {email}")
```

### 1.2 Why You Should Not Use a Fully RFC 5322 Compliant Pattern

```
RFC 5322 compliant pattern: a regex of thousands of characters
-> Unmaintainable, undebuggable, performance risk

Practical approach:
+-----------------------------------------+
| 1. Check basic format with regex        |
|    (has @, has domain, has TLD)         |
|                                         |
| 2. Send a confirmation email to verify  |
|    existence (this is the only          |
|    correct verification)                |
+-----------------------------------------+

Reasons:
- "Valid" but non-existent addresses cannot be detected by regex
- Some existing addresses don't comply with RFC
- The job of regex is only to "reject obvious typos"
```

### 1.3 Email Address Edge Cases in Detail

This section explains in detail the boundary cases of email addresses encountered in practice.

```python
import re

email_pattern = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

# Edge cases: understand the limits of the practical pattern
edge_cases = {
    # --- Cases that are valid per RFC but NG with the practical pattern ---
    '"user name"@example.com': "RFC valid: quoted local part",
    'user@[192.168.1.1]':      "RFC valid: IP literal domain",
    '(comment)user@example.com': "RFC valid: with comment",
    'user@example':            "RFC valid: local domain without TLD",

    # --- Cases that are correctly rejected by the practical pattern ---
    'user@.example.com':       "NG: domain starts with a dot",
    'user@example..com':       "NG: consecutive dots in domain",
    '.user@example.com':       "NG: local part starts with a dot",
    'user.@example.com':       "NG: local part ends with a dot",

    # --- Support for new TLDs ---
    'user@example.photography': "OK: long TLD (.photography)",
    'user@example.museum':      "OK: .museum",
    'user@example.co.uk':       "OK: double TLD",
    'user@example.xn--p1ai':    "OK: internationalized TLD (Punycode)",
}

for email, description in edge_cases.items():
    result = "OK" if email_pattern.match(email) else "NG"
    print(f"  {result}: {email:<40} -- {description}")
```

### 1.4 Improved Email Pattern

This shows an improved version that strengthens the weaknesses of the basic pattern.

```python
import re

# Improved version: rejects consecutive dots and dots at the start/end
email_improved = re.compile(
    r'^'
    r'(?![.])'                      # does not start with a dot
    r'[a-zA-Z0-9]'                  # starts with alphanumeric
    r'(?:[a-zA-Z0-9._%+-]*'         # middle part
    r'[a-zA-Z0-9_%+-])?'            # does not end with a dot (optional for single character)
    r'@'
    r'(?![.-])'                     # domain does not start with a dot or hyphen
    r'[a-zA-Z0-9]'                  # domain head
    r'(?:[a-zA-Z0-9.-]*'            # domain middle
    r'[a-zA-Z0-9])?'               # domain does not end with a hyphen
    r'\.[a-zA-Z]{2,}$'             # TLD
)

test_improved = [
    ("user@example.com",        True),
    (".user@example.com",       False),  # starts with dot
    ("user.@example.com",       False),  # ends with dot
    ("user..name@example.com",  False),  # consecutive dots
    ("u@example.com",           True),   # single-character local part
    ("user@-domain.com",        False),  # domain starting with hyphen
    ("user@domain-.com",        False),  # domain ending with hyphen
]

for email, expected in test_improved:
    result = bool(email_improved.match(email))
    status = "PASS" if result == expected else "FAIL"
    print(f"  {status}: {email:<30} expected={expected}, got={result}")
```

### 1.5 Email Validation in Multiple Languages

```javascript
// JavaScript version
const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

// HTML5 input[type="email"] uses its own pattern
// Leveraging the browser's standard validation is the best approach
const form = document.createElement('form');
const input = document.createElement('input');
input.type = 'email';
input.value = 'test@example.com';
console.log(input.checkValidity()); // true
```

```go
// Go version
package main

import (
    "fmt"
    "net/mail"
    "regexp"
)

func main() {
    // Basic check via regex
    pattern := regexp.MustCompile(
        `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`,
    )
    fmt.Println(pattern.MatchString("user@example.com")) // true

    // Recommended: use the net/mail package
    _, err := mail.ParseAddress("user@example.com")
    fmt.Println(err == nil) // true
}
```

```ruby
# Ruby version
# Use the standard library URI::MailTo
require 'uri'

email = "user@example.com"
pattern = /\A[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\z/

puts email.match?(pattern)  # => true

# URI::MailTo::EMAIL_REGEXP is also available
puts email.match?(URI::MailTo::EMAIL_REGEXP)  # => true
```

---

## 2. URL

### 2.1 Practical Pattern

```python
import re

# Practical pattern for HTTP/HTTPS URLs
url_pattern = re.compile(
    r'https?://'                # protocol
    r'(?:[a-zA-Z0-9]'          # domain head
    r'(?:[a-zA-Z0-9-]{0,61}'   # domain middle
    r'[a-zA-Z0-9])?\.)'        # domain tail + dot
    r'+[a-zA-Z]{2,}'           # TLD
    r'(?:/[^\s]*)?'            # path (optional)
)

test_urls = [
    "https://example.com",
    "https://www.example.com/path/to/page",
    "http://sub.domain.example.co.jp/path?q=1&p=2#hash",
    "ftp://example.com",          # NG (only http/https)
    "not a url",                  # NG
]

for url in test_urls:
    m = url_pattern.search(url)
    result = m.group() if m else "NG"
    print(f"  {result}")
```

### 2.2 Extracting URLs from Text

```python
import re

text = """
Official site: https://example.com/docs
Reference: http://sub.domain.co.jp/path?key=value
Contact: mailto:info@example.com (this should not match)
"""

# Extract HTTP URLs from text
url_extract = re.compile(r'https?://[^\s<>"]+')

urls = url_extract.findall(text)
for url in urls:
    print(f"  {url}")
# => https://example.com/docs
# => http://sub.domain.co.jp/path?key=value
```

### 2.3 Decomposing and Extracting URL Components

This shows a pattern that decomposes a URL into capture groups so each component can be obtained individually.

```python
import re

# Extract each URL component using named capture groups
url_decompose = re.compile(
    r'^'
    r'(?P<scheme>https?)'              # scheme
    r'://'
    r'(?:(?P<user>[^:@]+)'             # user (optional)
    r'(?::(?P<password>[^@]+))?@)?'    # password (optional)
    r'(?P<host>[a-zA-Z0-9.-]+)'        # host
    r'(?::(?P<port>\d{1,5}))?'         # port (optional)
    r'(?P<path>/[^?#]*)?'              # path (optional)
    r'(?:\?(?P<query>[^#]*))?'         # query (optional)
    r'(?:#(?P<fragment>.*))?'          # fragment (optional)
    r'$'
)

test_urls_decompose = [
    "https://www.example.com/path/to/page?key=value&lang=ja#section1",
    "http://user:pass@host.example.com:8080/api/v1?format=json",
    "https://example.co.jp",
    "http://localhost:3000/dashboard",
]

for url in test_urls_decompose:
    m = url_decompose.match(url)
    if m:
        print(f"\n  URL: {url}")
        for name in ['scheme', 'user', 'password', 'host', 'port', 'path', 'query', 'fragment']:
            value = m.group(name)
            if value:
                print(f"    {name:>10}: {value}")
```

Example output:

```
  URL: https://www.example.com/path/to/page?key=value&lang=ja#section1
      scheme: https
        host: www.example.com
        path: /path/to/page
       query: key=value&lang=ja
    fragment: section1

  URL: http://user:pass@host.example.com:8080/api/v1?format=json
      scheme: http
        user: user
    password: pass
        host: host.example.com
        port: 8080
        path: /api/v1
       query: format=json
```

### 2.4 Parsing and Extracting Query Parameters

```python
import re
from urllib.parse import urlparse, parse_qs

url = "https://example.com/search?q=python+regex&page=2&lang=ja&sort=date"

# Method 1: extract individual query parameters using regex
param_pattern = re.compile(r'?&=([^&]*)')
params_regex = param_pattern.findall(url)
print("Regex:")
for key, value in params_regex:
    print(f"  {key} = {value}")

# Method 2: use urllib.parse (recommended)
parsed = urlparse(url)
params_lib = parse_qs(parsed.query)
print("\nurllib.parse:")
for key, values in params_lib.items():
    print(f"  {key} = {values}")

# Note: regex is insufficient for handling encoded parameters
# Use a library for decoding things like %E6%97%A5%E6%9C%AC -> "日本"
```

### 2.5 Extracting Links from Markdown or HTML

```python
import re

# Extract markdown links: text
markdown_link = re.compile(
    r'\[([^\]]+)\]'          # link text
    r'\(([^)]+)\)'           # URL
)

md_text = """
For details, see [Official Documentation](https://docs.example.com/guide).
Also check [API Reference](https://api.example.com/v2/docs).
Image: ![alt text](https://img.example.com/logo.png)
"""

for m in markdown_link.finditer(md_text):
    print(f"  Text: {m.group(1)}")
    print(f"  URL:  {m.group(2)}\n")

# Extract links from HTML <a> tags
html_link = re.compile(
    r'<a\s+[^>]*href="\'["\'][^>]*>'   # href attribute
    r'(.*?)'                                        # link text
    r'</a>',
    re.DOTALL
)

html_text = """
<a href="https://example.com" class="link">Example</a>
<a href="/relative/path" target="_blank">Relative Link</a>
"""

for m in html_link.finditer(html_text):
    print(f"  href: {m.group(1)}, text: {m.group(2)}")

# Note: for serious HTML parsing, use BeautifulSoup
```

---

## 3. Date

### 3.1 Patterns for Various Formats

```python
import re

# YYYY-MM-DD
iso_date = re.compile(r'\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b')

# YYYY/MM/DD
slash_date = re.compile(r'\b(\d{4})/(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])\b')

# DD/MM/YYYY (European style)
eu_date = re.compile(r'\b(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/(\d{4})\b')

# MM/DD/YYYY (American style)
us_date = re.compile(r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(\d{4})\b')

# Japanese date
jp_date = re.compile(r'(\d{4})年(0?[1-9]|1[0-2])月(0?[1-9]|[12]\d|3[01])日')

# Test
texts = [
    "2026-02-11",
    "2026/02/11",
    "11/02/2026",
    "02/11/2026",
    "2026年2月11日",
]

patterns = {
    "ISO": iso_date,
    "Slash": slash_date,
    "EU": eu_date,
    "US": us_date,
    "Japanese": jp_date,
}

for text in texts:
    for name, pat in patterns.items():
        m = pat.search(text)
        if m:
            print(f"  {text} -> {name}: {m.groups()}")
```

### 3.2 Caveats for Date Validation

```python
import re
from datetime import datetime

# Examples where regex alone is insufficient:
# "2026-02-30" -- format is correct, but February 30 does not exist
# "2025-02-29" -- not a leap year, so it does not exist
# "2024-02-29" -- a leap year, so it exists

def validate_date(date_str: str) -> bool:
    """Validate a date with regex + datetime"""
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return False
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

print(validate_date("2026-02-11"))  # => True
print(validate_date("2026-02-30"))  # => False (day 30 does not exist)
print(validate_date("2025-02-29"))  # => False (not a leap year)
print(validate_date("2024-02-29"))  # => True  (leap year)
```

### 3.3 DateTime Patterns

This shows patterns that include time as well as date.

```python
import re
from datetime import datetime

# ISO 8601 datetime format: YYYY-MM-DDTHH:MM:SS
iso_datetime = re.compile(
    r'\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])'  # date part
    r'[T ]'                                                # separator (T or space)
    r'([01]\d|2[0-3]):([0-5]\d):([0-5]\d)'                # time part
    r'(?:\.(\d{1,6}))?'                                   # microseconds (optional)
    r'(?:Z|([+-])([01]\d|2[0-3]):?([0-5]\d))?\b'          # timezone (optional)
)

test_datetimes = [
    "2026-02-11T14:30:00",           # local time
    "2026-02-11T14:30:00Z",          # UTC
    "2026-02-11T14:30:00+09:00",     # JST
    "2026-02-11 14:30:00.123456",    # with microseconds
    "2026-02-11T14:30:00-05:00",     # EST
    "2026-02-11T25:00:00",           # NG: hour 25 does not exist
]

for dt_str in test_datetimes:
    m = iso_datetime.search(dt_str)
    if m:
        print(f"  OK: {dt_str}")
        print(f"      Date: {m.group(1)}-{m.group(2)}-{m.group(3)}")
        print(f"      Time: {m.group(4)}:{m.group(5)}:{m.group(6)}")
    else:
        print(f"  NG: {dt_str}")
```

### 3.4 Parsing Relative Date Expressions

Extract relative date expressions like "3 days ago" or "2 weeks from now" from logs and text.

```python
import re
from datetime import datetime, timedelta

# Japanese relative date expressions
relative_date_jp = re.compile(
    r'(\d+)\s*'
    r'(秒|分|時間|日|週間?|ヶ月|か月|カ月|年)'
    r'\s*(前|後|先|以内)'
)

# English relative date expressions
relative_date_en = re.compile(
    r'(\d+)\s+'
    r'(seconds?|minutes?|hours?|days?|weeks?|months?|years?)'
    r'\s+(ago|later|from now)',
    re.IGNORECASE
)

test_relative = [
    "このイベントは3日前に発生しました",
    "レポートは2週間以内に提出してください",
    "5年前のデータを参照",
    "The error occurred 30 minutes ago",
    "Delivery expected 2 weeks from now",
]

for text in test_relative:
    m = relative_date_jp.search(text) or relative_date_en.search(text)
    if m:
        print(f"  '{text}'")
        print(f"    Extracted: {m.group(0)}")
        print(f"    Number: {m.group(1)}, Unit: {m.group(2)}, Direction: {m.group(3)}")
```

### 3.5 Date Range Validation

```python
import re
from datetime import datetime

def validate_date_range(start_str: str, end_str: str) -> dict:
    """Validate a date range"""
    date_pattern = re.compile(r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$')

    result = {"valid": True, "errors": []}

    # Format check
    if not date_pattern.match(start_str):
        result["valid"] = False
        result["errors"].append(f"Invalid start date format: {start_str}")
    if not date_pattern.match(end_str):
        result["valid"] = False
        result["errors"].append(f"Invalid end date format: {end_str}")

    if not result["valid"]:
        return result

    # Date validity check
    try:
        start = datetime.strptime(start_str, '%Y-%m-%d')
        end = datetime.strptime(end_str, '%Y-%m-%d')
    except ValueError as e:
        result["valid"] = False
        result["errors"].append(f"Date does not exist: {e}")
        return result

    # Logical check: start <= end
    if start > end:
        result["valid"] = False
        result["errors"].append("Start date is after end date")

    # Range check: within 365 days
    if (end - start).days > 365:
        result["valid"] = False
        result["errors"].append("Date range exceeds 365 days")

    return result

# Test
cases = [
    ("2026-01-01", "2026-12-31"),  # OK
    ("2026-12-31", "2026-01-01"),  # NG: reversed
    ("2026-02-29", "2026-03-01"),  # NG: 2026-02-29 does not exist
    ("2025-01-01", "2026-12-31"),  # NG: exceeds 365 days
]

for start, end in cases:
    r = validate_date_range(start, end)
    status = "OK" if r["valid"] else "NG"
    print(f"  {status}: {start} ~ {end}")
    for err in r["errors"]:
        print(f"       {err}")
```

---

## 4. Phone Number

### 4.1 Patterns by Country

```python
import re

# Japanese phone numbers
jp_phone_patterns = {
    # Mobile: 090/080/070-XXXX-XXXX
    "Mobile": re.compile(r'0[789]0-?\d{4}-?\d{4}'),
    # Landline (Tokyo): 03-XXXX-XXXX
    "Landline (Tokyo)": re.compile(r'03-?\d{4}-?\d{4}'),
    # Landline (Osaka): 06-XXXX-XXXX
    "Landline (Osaka)": re.compile(r'06-?\d{4}-?\d{4}'),
    # Toll-free: 0120-XXX-XXX
    "Toll-free": re.compile(r'0120-?\d{3}-?\d{3}'),
    # International format: +81-XX-XXXX-XXXX
    "International": re.compile(r'\+81-?\d{1,4}-?\d{1,4}-?\d{4}'),
}

# General Japanese phone number pattern
jp_phone_general = re.compile(
    r'(?:\+81|0)'           # +81 or 0
    r'[\d-]{9,13}'          # 9-13 characters of digits and hyphens
)

test_phones = [
    "090-1234-5678",
    "09012345678",
    "03-1234-5678",
    "0120-123-456",
    "+81-90-1234-5678",
]

for phone in test_phones:
    m = jp_phone_general.search(phone)
    print(f"  {phone}: {'OK' if m else 'NG'}")
```

### 4.2 International Phone Number (E.164)

```python
import re

# E.164 format: +[country code][phone number] (max 15 digits)
e164_pattern = re.compile(r'^\+[1-9]\d{1,14}$')

test_numbers = [
    "+819012345678",     # Japan
    "+14155551234",      # USA
    "+442012345678",     # UK
    "+0123456789",       # NG (no country code starts with 0)
    "+123456789012345",  # NG (16 digits -- exceeds upper limit)
]

for num in test_numbers:
    result = "OK" if e164_pattern.match(num) else "NG"
    print(f"  {result}: {num}")

# Note: for strict phone number validation,
# Google's libphonenumber library is recommended
```

### 4.3 Phone Number Patterns by Country in Detail

```python
import re

# Collection of phone number patterns by country
international_phone_patterns = {
    # USA/Canada (NANP): +1-NXX-NXX-XXXX
    "US/CA": re.compile(
        r'(?:\+1[-.\s]?)?'            # country code (optional)
        r'\(?[2-9]\d{2}\)?'            # area code
        r'[-.\s]?'
        r'[2-9]\d{2}'                  # exchange
        r'[-.\s]?'
        r'\d{4}'                       # subscriber number
    ),

    # UK: +44 XXXX XXXXXX
    "UK": re.compile(
        r'(?:\+44[-.\s]?|0)'           # country code or trunk prefix
        r'[1-9]\d{1,4}'               # area code
        r'[-.\s]?'
        r'\d{4,8}'                     # subscriber number
    ),

    # Germany: +49 XXXX XXXXXXX
    "DE": re.compile(
        r'(?:\+49[-.\s]?|0)'
        r'[1-9]\d{1,4}'
        r'[-.\s]?'
        r'\d{3,8}'
    ),

    # China: +86 1XX XXXX XXXX (mobile)
    "CN_mobile": re.compile(
        r'(?:\+86[-.\s]?)?'
        r'1[3-9]\d'                    # mobile prefix
        r'[-.\s]?'
        r'\d{4}'
        r'[-.\s]?'
        r'\d{4}'
    ),

    # Korea: +82 01X-XXXX-XXXX (mobile)
    "KR_mobile": re.compile(
        r'(?:\+82[-.\s]?|0)'
        r'1[016789]'                   # mobile prefix
        r'[-.\s]?'
        r'\d{3,4}'
        r'[-.\s]?'
        r'\d{4}'
    ),
}

test_international = [
    ("US/CA", "(415) 555-1234"),
    ("US/CA", "+1-415-555-1234"),
    ("UK",    "+44 20 7946 0958"),
    ("UK",    "020 7946 0958"),
    ("DE",    "+49 30 12345678"),
    ("CN_mobile", "+86 138 1234 5678"),
    ("KR_mobile", "010-1234-5678"),
]

for country, phone in test_international:
    pattern = international_phone_patterns[country]
    m = pattern.search(phone)
    print(f"  {country:>10}: {phone:<25} {'OK' if m else 'NG'}")
```

### 4.4 Bulk Extraction of Phone Numbers from Text

```python
import re

# General-purpose pattern for extracting phone-number-like strings from text
phone_extractor = re.compile(
    r'(?:'
    r'\+\d{1,3}[-.\s]?'               # with country code
    r'|'
    r'0'                               # domestic number
    r')'
    r'(?:\d[-.\s]?){8,13}'            # sequence of 8 to 13 digits
)

document = """
Contact Information:
  Tokyo HQ: 03-1234-5678
  Osaka Branch: 06-9876-5432
  Mobile (direct line): 090-1111-2222
  Toll-free: 0120-456-789
  International inquiries: +81-3-1234-5678

* Business hours: 9:00-18:00 (digits but not a phone number)
* FAX: 03-1234-5679
"""

phones = phone_extractor.findall(document)
print("Extracted phone numbers:")
for phone in phones:
    # Normalize: remove hyphens and spaces
    normalized = re.sub(r'[-.\s]', '', phone)
    print(f"  Original: {phone:<25} Normalized: {normalized}")
```

---

## 5. IP Address

### 5.1 IPv4

```python
import re

# IPv4: from 0.0.0.0 to 255.255.255.255
ipv4_pattern = re.compile(
    r'\b'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'   # 1st octet
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'   # 2nd octet
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'   # 3rd octet
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'     # 4th octet
    r'\b'
)

test_ips = [
    "192.168.1.1",      # OK
    "10.0.0.0",         # OK
    "255.255.255.255",  # OK
    "0.0.0.0",          # OK
    "256.1.1.1",        # NG (256 is out of range)
    "192.168.1",        # NG (only 3 octets)
    "192.168.1.1.1",    # NG (5 octets)
]

for ip in test_ips:
    result = "OK" if ipv4_pattern.fullmatch(ip) else "NG"
    print(f"  {result}: {ip}")
```

### 5.2 IPv6 (Simplified)

```python
import re
import ipaddress

# Strict IPv6 matching with regex alone is difficult
# Recommendation: simple pattern + library validation

ipv6_simple = re.compile(
    r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}'  # full form only
)

# Recommended: use the ipaddress module
def validate_ip(addr: str) -> str:
    try:
        obj = ipaddress.ip_address(addr)
        return f"IPv{obj.version}"
    except ValueError:
        return "Invalid"

print(validate_ip("192.168.1.1"))                # => IPv4
print(validate_ip("2001:db8::1"))                 # => IPv6
print(validate_ip("fe80::1%eth0"))                # => Invalid
```

### 5.3 CIDR Notation Pattern

```python
import re
import ipaddress

# IPv4 CIDR notation: 192.168.1.0/24
ipv4_cidr = re.compile(
    r'\b'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'
    r'/([12]?\d|3[0-2])'      # subnet mask: 0-32
    r'\b'
)

test_cidrs = [
    "192.168.1.0/24",      # OK: /24 subnet
    "10.0.0.0/8",          # OK: Class A
    "172.16.0.0/12",       # OK: private address
    "192.168.1.0/33",      # NG: /33 is out of range
    "256.0.0.0/24",        # NG: 256 is out of range
]

for cidr in test_cidrs:
    m = ipv4_cidr.fullmatch(cidr)
    if m:
        # Also validate with the library
        try:
            network = ipaddress.ip_network(cidr, strict=False)
            print(f"  OK: {cidr:<20} network={network.network_address}, "
                  f"hosts={network.num_addresses}")
        except ValueError as e:
            print(f"  NG: {cidr:<20} {e}")
    else:
        print(f"  NG: {cidr}")
```

### 5.4 Identifying Private IP Addresses

```python
import re

# Private IP address ranges
# 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16

private_ipv4 = re.compile(
    r'\b(?:'
    r'10\.\d{1,3}\.\d{1,3}\.\d{1,3}'            # 10.0.0.0/8
    r'|'
    r'172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}'  # 172.16.0.0/12
    r'|'
    r'192\.168\.\d{1,3}\.\d{1,3}'               # 192.168.0.0/16
    r'|'
    r'127\.\d{1,3}\.\d{1,3}\.\d{1,3}'           # 127.0.0.0/8 (loopback)
    r')\b'
)

test_private = [
    "10.0.0.1",        # private
    "172.16.0.1",      # private
    "172.31.255.255",  # private
    "172.32.0.1",      # public (172.32 is out of range)
    "192.168.1.1",     # private
    "127.0.0.1",       # loopback
    "8.8.8.8",         # public
    "203.0.113.1",     # public (documentation)
]

for ip in test_private:
    is_private = bool(private_ipv4.match(ip))
    print(f"  {'Private' if is_private else 'Public ':>7}: {ip}")
```

### 5.5 Extracting and Aggregating IP Addresses from Log Files

```python
import re
from collections import Counter

# Extract IPs from Apache/Nginx access logs
ipv4_pattern = re.compile(
    r'\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
)

sample_log = """
192.168.1.100 - - [11/Feb/2026:14:30:00 +0900] "GET /index.html HTTP/1.1" 200 1234
10.0.0.50 - - [11/Feb/2026:14:30:01 +0900] "POST /api/data HTTP/1.1" 201 567
192.168.1.100 - - [11/Feb/2026:14:30:02 +0900] "GET /style.css HTTP/1.1" 200 890
203.0.113.42 - - [11/Feb/2026:14:30:03 +0900] "GET /admin HTTP/1.1" 403 0
192.168.1.100 - - [11/Feb/2026:14:30:04 +0900] "GET /favicon.ico HTTP/1.1" 404 0
203.0.113.42 - - [11/Feb/2026:14:30:05 +0900] "GET /admin HTTP/1.1" 403 0
10.0.0.50 - - [11/Feb/2026:14:30:06 +0900] "GET /api/health HTTP/1.1" 200 15
"""

# Extract IPs and aggregate
ips = ipv4_pattern.findall(sample_log)
ip_counts = Counter(ips)

print("Access counts per IP address:")
for ip, count in ip_counts.most_common():
    print(f"  {ip:<16} {count} times")

# Identify IPs with 403 errors
error_pattern = re.compile(
    r'(\d+\.\d+\.\d+\.\d+).*?"(?:GET|POST|PUT|DELETE)\s+\S+\s+HTTP/\d\.\d"\s+403'
)

error_ips = error_pattern.findall(sample_log)
error_counts = Counter(error_ips)
print("\nIPs with 403 errors:")
for ip, count in error_counts.most_common():
    print(f"  {ip:<16} {count} times -- possibly unauthorized access")
```

---

## 6. Other Common Patterns

### 6.1 Postal Code (Japan)

```python
import re

# Japanese postal code: XXX-XXXX
jp_postal = re.compile(r'\b\d{3}-?\d{4}\b')

test_codes = ["100-0001", "1000001", "100-001"]
for code in test_codes:
    result = "OK" if jp_postal.fullmatch(code) else "NG"
    print(f"  {result}: {code}")
# => OK: 100-0001
# => OK: 1000001
# => NG: 100-001
```

### 6.2 Credit Card Number (with Luhn Check)

```python
import re

# Patterns for major card brands
card_patterns = {
    "Visa":       re.compile(r'^4\d{12}(?:\d{3})?$'),
    "Mastercard": re.compile(r'^5[1-5]\d{14}$'),
    "AMEX":       re.compile(r'^3[47]\d{13}$'),
    "JCB":        re.compile(r'^(?:2131|1800|35\d{3})\d{11}$'),
}

def luhn_check(number: str) -> bool:
    """Validate the check digit using the Luhn algorithm"""
    digits = [int(d) for d in number]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0

def validate_card(number: str) -> tuple[str, bool]:
    """Validate a card number (format + Luhn)"""
    clean = number.replace(' ', '').replace('-', '')
    for brand, pattern in card_patterns.items():
        if pattern.match(clean):
            return brand, luhn_check(clean)
    return "Unknown", False

print(validate_card("4111 1111 1111 1111"))  # => ('Visa', True)
```

### 6.3 Password Strength

```python
import re

def check_password_strength(password: str) -> dict:
    """Check password strength against multiple criteria"""
    checks = {
        "8 characters or more": len(password) >= 8,
        "Contains uppercase":   bool(re.search(r'[A-Z]', password)),
        "Contains lowercase":   bool(re.search(r'[a-z]', password)),
        "Contains digit":       bool(re.search(r'\d', password)),
        "Contains symbol":      bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)),
        "No consecutive chars": not bool(re.search(r'(.)\1{2,}', password)),
    }
    strength = sum(checks.values())
    return {"checks": checks, "score": f"{strength}/6"}

result = check_password_strength("MyP@ssw0rd")
for check, passed in result["checks"].items():
    print(f"  {'OK' if passed else 'NG'}: {check}")
print(f"  Score: {result['score']}")
```

### 6.4 Username Validation

This shows a typical username validation pattern for web services.

```python
import re

# Username requirements:
# - 3 to 20 characters
# - Only alphanumerics, underscores, and hyphens
# - Must start with a letter
# - No consecutive underscores or hyphens

username_pattern = re.compile(
    r'^'
    r'[a-zA-Z]'                  # starts with a letter
    r'(?!.*[-_]{2})'             # forbid consecutive symbols via negative lookahead
    r'[a-zA-Z0-9_-]{2,19}'      # remaining 2-19 characters (3-20 total)
    r'$'
)

test_usernames = [
    ("alice",           True,  "OK: basic name"),
    ("user_name",       True,  "OK: with underscore"),
    ("user-name",       True,  "OK: with hyphen"),
    ("a1b2c3",          True,  "OK: alphanumeric mix"),
    ("ab",              False, "NG: 2 chars (min 3)"),
    ("1user",           False, "NG: starts with digit"),
    ("_user",           False, "NG: starts with underscore"),
    ("user__name",      False, "NG: consecutive underscores"),
    ("user--name",      False, "NG: consecutive hyphens"),
    ("user name",       False, "NG: contains space"),
    ("user@name",       False, "NG: contains special char"),
    ("a" * 21,          False, "NG: 21 chars (max 20)"),
]

for username, expected, description in test_usernames:
    result = bool(username_pattern.match(username))
    status = "PASS" if result == expected else "FAIL"
    display_name = username if len(username) <= 20 else username[:17] + "..."
    print(f"  {status}: {display_name:<20} -- {description}")
```

### 6.5 File Path Patterns

```python
import re

# Unix/Linux file path
unix_path = re.compile(
    r'^(/[a-zA-Z0-9._-]+)+/?$'
)

# Windows file path
windows_path = re.compile(
    r'^[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$'
)

# Extract file extension
extension = re.compile(r'\.([a-zA-Z0-9]+)$')

test_paths = {
    "/home/user/file.txt":           "Unix",
    "/var/log/syslog":               "Unix",
    "C:\\Users\\user\\file.txt":     "Windows",
    "C:\\Program Files\\app.exe":    "Windows",
}

for path, os_type in test_paths.items():
    pattern = unix_path if os_type == "Unix" else windows_path
    valid = bool(pattern.match(path))
    ext_match = extension.search(path)
    ext = ext_match.group(1) if ext_match else "(none)"
    print(f"  {os_type:>7}: {path:<40} valid={valid}, ext={ext}")
```

### 6.6 Hexadecimal Color Codes

```python
import re

# CSS color codes
hex_color = re.compile(
    r'^#(?:'
    r'[0-9a-fA-F]{3}'    # short form: #RGB
    r'|'
    r'[0-9a-fA-F]{4}'    # short form + alpha: #RGBA
    r'|'
    r'[0-9a-fA-F]{6}'    # full form: #RRGGBB
    r'|'
    r'[0-9a-fA-F]{8}'    # full form + alpha: #RRGGBBAA
    r')$'
)

# CSS rgb()/rgba() function form
rgb_color = re.compile(
    r'^rgba?\(\s*'
    r'(\d{1,3})\s*,\s*'     # R: 0-255
    r'(\d{1,3})\s*,\s*'     # G: 0-255
    r'(\d{1,3})'             # B: 0-255
    r'(?:\s*,\s*'
    r'([01]?\.?\d*)'         # A: 0-1 (optional)
    r')?\s*\)$'
)

# HSL form
hsl_color = re.compile(
    r'^hsla?\(\s*'
    r'(\d{1,3})\s*,\s*'     # H: 0-360
    r'(\d{1,3})%\s*,\s*'    # S: 0-100%
    r'(\d{1,3})%'            # L: 0-100%
    r'(?:\s*,\s*'
    r'([01]?\.?\d*)'         # A: 0-1 (optional)
    r')?\s*\)$'
)

test_colors = [
    "#fff",                 # OK: short form
    "#FF5733",              # OK: full form
    "#FF573380",            # OK: with alpha
    "#GGHHII",              # NG: invalid hex
    "rgb(255, 87, 51)",     # OK: RGB
    "rgba(255, 87, 51, 0.5)", # OK: RGBA
    "hsl(9, 100%, 60%)",    # OK: HSL
]

for color in test_colors:
    matched = (
        hex_color.match(color) or
        rgb_color.match(color) or
        hsl_color.match(color)
    )
    print(f"  {'OK' if matched else 'NG'}: {color}")
```

### 6.7 UUID Validation

```python
import re

# UUID v4: xxxxxxxx-xxxx-4xxx-[89ab]xxx-xxxxxxxxxxxx
uuid_v4 = re.compile(
    r'^[0-9a-f]{8}-'
    r'[0-9a-f]{4}-'
    r'4[0-9a-f]{3}-'          # version 4
    r'[89ab][0-9a-f]{3}-'     # variant 1
    r'[0-9a-f]{12}$',
    re.IGNORECASE
)

# UUID of any version
uuid_any = re.compile(
    r'^[0-9a-f]{8}-'
    r'[0-9a-f]{4}-'
    r'[1-5][0-9a-f]{3}-'      # versions 1-5
    r'[89ab][0-9a-f]{3}-'
    r'[0-9a-f]{12}$',
    re.IGNORECASE
)

test_uuids = [
    "550e8400-e29b-41d4-a716-446655440000",  # OK: v4
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8",  # OK: v1
    "not-a-uuid",                              # NG
    "550e8400-e29b-61d4-a716-446655440000",   # NG: version 6 (only v1-5 supported)
    "550e8400-e29b-41d4-c716-446655440000",   # NG: invalid variant
]

for uuid_str in test_uuids:
    v4 = "v4" if uuid_v4.match(uuid_str) else "--"
    any_v = "OK" if uuid_any.match(uuid_str) else "NG"
    print(f"  {any_v}({v4}): {uuid_str}")
```

### 6.8 Japanese Text Patterns

```python
import re

# Hiragana
hiragana = re.compile(r'^[\u3040-\u309F]+$')

# Katakana
katakana = re.compile(r'^[\u30A0-\u30FF]+$')

# Kanji (CJK Unified Ideographs)
kanji = re.compile(r'^[\u4E00-\u9FFF]+$')

# Full-width characters
zenkaku = re.compile(r'^[\uFF01-\uFF5E]+$')

# Japanese name (family given): kanji + space + kanji
jp_name = re.compile(r'^[\u4E00-\u9FFF\u3040-\u309F]{1,10}\s[\u4E00-\u9FFF\u3040-\u309F]{1,10}$')

# Furigana (katakana)
furigana = re.compile(r'^[\u30A0-\u30FF\s]{2,20}$')

test_japanese = [
    ("あいうえお",     hiragana,  "Hiragana"),
    ("アイウエオ",     katakana,  "Katakana"),
    ("漢字",           kanji,     "Kanji"),
    ("山田 太郎",      jp_name,   "Japanese name"),
    ("ヤマダ タロウ",  furigana,  "Furigana"),
    ("ABC",            hiragana,  "Latin letters (NG)"),
]

for text, pattern, description in test_japanese:
    result = "OK" if pattern.match(text) else "NG"
    print(f"  {result}: {text:<15} -- {description}")
```

### 6.9 Numeric Format Patterns

```python
import re

# Integer (with comma separators)
integer_comma = re.compile(r'^-?(?:\d{1,3}(?:,\d{3})*|\d+)$')

# Decimal number
decimal_number = re.compile(r'^-?\d+(?:\.\d+)?$')

# Scientific notation
scientific = re.compile(r'^-?\d+(?:\.\d+)?[eE][+-]?\d+$')

# Percentage
percentage = re.compile(r'^-?\d+(?:\.\d+)?%$')

# Currency (Japanese yen)
yen = re.compile(r'^[¥￥]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$')

# Currency (US dollar)
usd = re.compile(r'^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$')

test_numbers = [
    ("1,234,567",     integer_comma,  "Comma-separated integer"),
    ("-42",           integer_comma,  "Negative integer"),
    ("3.14159",       decimal_number, "Decimal"),
    ("6.022e23",      scientific,     "Scientific notation"),
    ("1.5E-10",       scientific,     "Scientific notation (negative exponent)"),
    ("85.5%",         percentage,     "Percentage"),
    ("¥1,234,567",    yen,            "Japanese yen"),
    ("$99.99",        usd,            "US dollar"),
    ("1,23,456",      integer_comma,  "Invalid comma separation (NG)"),
]

for text, pattern, description in test_numbers:
    result = "OK" if pattern.match(text) else "NG"
    print(f"  {result}: {text:<20} -- {description}")
```

---

## 7. ASCII Diagrams

### 7.1 Validation Design Flow

```
User input
    |
    v
+-------------------------+
| Step 1: Format check    |
| (regex)                 |
| e.g.: is it email shape?|
| -> reject obvious typos |
+------------+------------+
             | pass
             v
+-------------------------+
| Step 2: Logical check   |
| (program logic)         |
| e.g.: does the date     |
|       actually exist?   |
| e.g.: is the value      |
|       within range?     |
+------------+------------+
             | pass
             v
+-------------------------+
| Step 3: Existence check |
| (external service)      |
| e.g.: email deliverable |
| e.g.: address API check |
+------------+------------+
             | pass
             v
        Accept input
```

### 7.2 Structure of the Email Address Pattern

```
Pattern: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$

^                              $
|                              |
|  +---- local part --------+  |
|  | [a-zA-Z0-9._%+-]+      |  |
|  | alphanumerics + . _ % + - |
|  | one or more characters |  |
|  +------------------------+  |
|              @               |
|  +---- domain part -------+  |
|  | [a-zA-Z0-9.-]+         |  |
|  | alphanumerics + . -    |  |
|  | one or more characters |  |
|  +------------------------+  |
|              .               |
|  +---- TLD ---------------+  |
|  | [a-zA-Z]{2,}           |  |
|  | letters only, 2 or more|  |
|  +------------------------+  |

Example: user.name+tag@sub.domain.co.jp
         ^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^ ^^
         local          domain         TLD
```

### 7.3 Decomposition of Each Octet in the IPv4 Pattern

```
IPv4 octet: 0 to 255

Pattern: (?:25[0-5]|2[0-4]\d|[01]?\d\d?)

Branch 1: 25[0-5]     -> 250, 251, 252, 253, 254, 255
Branch 2: 2[0-4]\d    -> 200-249
Branch 3: [01]?\d\d?  -> 0-199

The order of branches is important:
  25[0-5]  -> first check 250-255
  2[0-4]\d -> next check 200-249
  [01]?\d\d? -> finally check 0-199

If you reverse the order:
  [01]?\d\d? -> matches "25" -> "5" is left over
  -> may not match correctly
```

### 7.4 URL Structure Decomposition Diagram

```
URL structure (RFC 3986):

  https://user:pass@www.example.com:443/path/to/page?key=val&k2=v2#section
  +-+-+   +-+-+ +-----+--------+++-++----+-+------++----+-+----+
  scheme   userinfo     host     port    path         query     fragment

Regex for each component:
  scheme:    [a-zA-Z][a-zA-Z0-9+.-]*
  userinfo:  [^@]+
  host:      [a-zA-Z0-9.-]+
  port:      \d{1,5}
  path:      /[^?#]*
  query:     [^#]*
  fragment:  .*

Full decomposition pattern:
  ^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*)://
   (?:(?P<userinfo>[^@]+)@)?
   (?P<host>[a-zA-Z0-9.-]+)
   (?::(?P<port>\d{1,5}))?
   (?P<path>/[^?#]*)?
   (?:\?(?P<query>[^#]*))?
   (?:#(?P<fragment>.*))?$
```

### 7.5 International Comparison of Phone Number Patterns

```
Phone number formats by country:

Japan (+81):
  Mobile:        0[789]0-XXXX-XXXX     e.g.: 090-1234-5678
  Landline:      0X-XXXX-XXXX          e.g.: 03-1234-5678
  International: +81-X0-XXXX-XXXX      e.g.: +81-90-1234-5678

  Pattern: 0[789]0-?\d{4}-?\d{4}
           +-+ +-+  +----+  +----+
           mobile hyphen  4 digits 4 digits
           prefix optional

USA (+1):
  Format:        (NXX) NXX-XXXX        e.g.: (415) 555-1234
  International: +1-NXX-NXX-XXXX       e.g.: +1-415-555-1234

  Pattern: \(?[2-9]\d{2}\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}
           +-----------+          +------+        +----+
           area code              exchange         subscriber

UK (+44):
  Mobile:        07XXX XXXXXX          e.g.: 07911 123456
  Landline:      0XX XXXX XXXX         e.g.: 020 7946 0958
  International: +44 XXXX XXXXXX       e.g.: +44 20 7946 0958
```

---

## 8. Comparison Tables

### 8.1 Pattern Accuracy vs. Complexity

| Pattern | Simple version | Practical version | Strict version | Recommended |
|---------|--------|--------|--------|------|
| Email | `.+@.+\..+` | `[a-zA-Z0-9._%+-]+@...` | RFC 5322 (thousands of chars) | Practical + delivery confirmation |
| URL | `https?://\S+` | with domain validation | full RFC 3986 compliance | Practical |
| Date | `\d{4}-\d{2}-\d{2}` | month/day range check | leap year support | Practical + datetime |
| Phone number | `[\d-+]+` | per-country/region patterns | libphonenumber | Practical or library |
| IPv4 | `\d+\.\d+\.\d+\.\d+` | 0-255 range check | full validation | Practical |

### 8.2 Regex vs. Dedicated Libraries

| Validation target | Is regex sufficient? | Recommended library |
|---------|---------------|--------------|
| Email format | Sufficient at practical level | -- |
| Email existence | Not possible | SMTP verification / confirmation email |
| URL format | Sufficient at practical level | urllib.parse (Python) |
| Date validity | Not possible (leap years, etc.) | datetime (Python) |
| Phone number | Basic format possible | libphonenumber |
| Credit card | Format possible | Luhn + payment API |
| HTML | Not possible | BeautifulSoup, lxml |
| JSON | Not possible | json.loads() |

### 8.3 Comparison of Regex Support by Language

| Feature | Python (`re`) | Python (`regex`) | JavaScript | Go | Ruby | Java |
|------|--------------|-----------------|------------|-----|------|------|
| Named groups | `(?P<name>...)` | `(?P<name>...)` | `(?<name>...)` | `(?P<name>...)` | `(?<name>...)` | `(?<name>...)` |
| Lookahead | Supported | Supported | Supported | Not supported | Supported | Supported |
| Lookbehind | Fixed-length only | Variable-length supported | Supported | Not supported | Supported | Supported |
| Unicode properties | `\p{...}` not supported | Supported | Supported | Supported | Supported | Supported |
| Recursive patterns | Not supported | Supported | Not supported | Not supported | Not supported | Not supported |
| Atomic groups | Not supported | Supported | Not supported | Not supported | Supported | Supported |
| POSIX character classes | Not supported | Supported | Not supported | Supported | Supported | Supported |

---

## 9. Anti-patterns

### 9.1 Anti-pattern: Fully Validating Dates with Regex Alone

```python
import re

# NG: trying to handle leap years in regex
# (the pattern becomes extremely complex and unmaintainable)
leap_year_pattern = r"""...(pattern of hundreds of characters)..."""

# OK: regex only checks the format; logical checks are done in code
def validate_date(s: str) -> bool:
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return False
    from datetime import datetime
    try:
        datetime.strptime(s, '%Y-%m-%d')
        return True
    except ValueError:
        return False
```

### 9.2 Anti-pattern: Copy-Pasting Patterns

```python
import re

# NG: using a pattern copy-pasted from StackOverflow as-is
# Reason: the context (language, Unicode settings) may differ
email_copied = r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08...]"
# -> unknown source, unmaintainable, unknown edge cases

# OK: design the pattern yourself to match your requirements and write tests
email_own = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Always prepare test cases
assert email_own.match("user@example.com")
assert email_own.match("a.b+c@d.co.jp")
assert not email_own.match("user@")
assert not email_own.match("@domain.com")
```

### 9.3 Anti-pattern: Rejecting Valid Input with an Overly Strict Pattern

```python
import re

# NG: restricting the TLD to 2-3 characters
email_strict_tld = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,3}$'
)

# With this restriction, the following valid addresses will be rejected:
rejected_valid = [
    "user@example.museum",       # .museum (6 chars)
    "user@example.photography",  # .photography (11 chars)
    "user@example.technology",   # .technology (10 chars)
    "user@example.international",# .international (13 chars)
]

for email in rejected_valid:
    result = "OK" if email_strict_tld.match(email) else "NG"
    print(f"  {result}: {email}  -- valid but rejected!")

# OK: set TLD to 2 or more characters (no upper limit)
email_flexible_tld = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)
```

### 9.4 Anti-pattern: Parsing HTML with Regex

```python
import re

# NG: trying to parse HTML with regex
# As the famous StackOverflow answer shows, this is impossible
html = '<div class="outer"><div class="inner">text</div></div>'

# This regex cannot correctly handle nested tags
bad_tag_extract = re.compile(r'<div[^>]*>(.*?)</div>')
# -> the match ends at the first </div>, breaking the nested structure

# OK: regex can be used for limited purposes
# e.g.: extracting self-contained tags
img_tag = re.compile(r'<img\s+[^>]*src="\'["\'][^>]*/?>')
link_href = re.compile(r'<a\s+[^>]*href="\'["\'][^>]*>')

# OK: for serious HTML parsing, use a library
# from bs4 import BeautifulSoup
# soup = BeautifulSoup(html, 'html.parser')
# divs = soup.find_all('div', class_='inner')
```

### 9.5 Anti-pattern: Patterns Vulnerable to ReDoS (Regular Expression Denial of Service)

```python
import re
import time

# NG: patterns that cause catastrophic backtracking
# Using these against user input poses a DoS attack risk

vulnerable_patterns = [
    # Pattern 1: nested quantifiers
    (r'(a+)+$', 'a' * 25 + 'b'),
    # Pattern 2: overlapping character classes
    (r'([a-zA-Z]+)*@', 'a' * 25 + '!'),
    # Pattern 3: alternation with overlap
    (r'(a|aa)+$', 'a' * 25 + 'b'),
]

for pattern, evil_input in vulnerable_patterns:
    print(f"\n  Pattern: {pattern}")
    print(f"  Input:   '{evil_input[:30]}...' ({len(evil_input)} chars)")
    start = time.time()
    try:
        # Run with a timeout (in real code, set a timeout)
        re.match(pattern, evil_input[:20])  # delay is visible even with short input
        elapsed = time.time() - start
        print(f"  Time:    {elapsed:.4f}s")
        print(f"  Warning: time grows exponentially as input length increases!")
    except Exception as e:
        print(f"  Error:   {e}")

# OK: safe patterns that prevent ReDoS
safe_patterns = [
    r'[a-zA-Z]+$',          # non-nested quantifier
    r'[a-zA-Z]+@',          # remove group quantifier
    r'a+$',                 # simple quantifier
]

# Mitigations:
# 1. Avoid nested quantifiers like (a+)+
# 2. Limit input length
# 3. Use atomic groups (?>...) (only in supported languages)
# 4. Set a timeout
# 5. Consider regex engines with guarantees, such as re2
```

---

## 10. FAQ

### Q1: How strict should an email regex be?

**A**: In practice, "basic format check + sending a confirmation email" is sufficient. Full RFC 5322 compliance is discouraged for maintainability and performance reasons:

```python
# Practical criteria:
# 1. Has exactly one @
# 2. Has a local part and a domain part
# 3. The TLD is 2 or more characters
# -> Anything beyond this is guaranteed by the confirmation email
```

### Q2: What is the most easily overlooked case in URL validation?

**A**: The following cases are easy to miss:

- **Internationalized Domain Names (IDN)**: `https://日本語.jp` -- Punycode conversion required
- **Port numbers**: `http://localhost:3000`
- **Authentication info**: `http://user:pass@host.com` -- security risk
- **Fragment**: `https://example.com/page#section`
- **Encoded query parameters**: `?q=%E6%97%A5%E6%9C%AC`

Using a library (`urllib.parse`, `URL` API) is recommended.

### Q3: What is the recommended approach for international phone number support?

**A**: Using Google's **libphonenumber** is the best option. Phone number rules vary by country and change frequently, so handling them completely with regex is impractical:

```python
# pip install phonenumbers
import phonenumbers

number = phonenumbers.parse("+819012345678", None)
print(phonenumbers.is_valid_number(number))  # => True
print(phonenumbers.format_number(
    number,
    phonenumbers.PhoneNumberFormat.INTERNATIONAL
))
# => '+81 90-1234-5678'
```

### Q4: How can I improve regex pattern performance?

**A**: The following techniques are effective:

```python
import re

# 1. Compile patterns and reuse them
# NG: compiling inside the loop every iteration
for line in lines:
    if re.match(r'^\d{4}-\d{2}-\d{2}', line):  # compiles every time
        pass

# OK: compile in advance
date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}')
for line in lines:
    if date_pattern.match(line):  # reuse compiled pattern
        pass

# 2. Avoid unnecessary capture groups
# NG: grouping when capture is not needed
pattern_capture = re.compile(r'(https?)://([\w.]+)')

# OK: use non-capturing groups
pattern_noncapture = re.compile(r'(?:https?)://(?:[\w.]+)')

# 3. Put more specific patterns first
# NG: generic pattern first
pattern_slow = re.compile(r'.*error.*fatal')

# OK: start with anchors or specific characters
pattern_fast = re.compile(r'^.*?error.*?fatal', re.MULTILINE)

# 4. Minimize quantifiers
# NG: greedy matching
greedy = re.compile(r'<.*>')       # longest match

# OK: lazy matching (depending on use case)
lazy = re.compile(r'<.*?>')       # shortest match
```

### Q5: What is the difference between input sanitization and regex validation?

**A**: They serve different purposes; neither alone is sufficient:

```python
import re
import html

user_input = '<script>alert("XSS")</script>Hello, World!'

# Validation: judge whether input is in an acceptable format
# -> reject invalid input
is_safe = bool(re.match(r'^[a-zA-Z0-9\s,.!?]+$', user_input))
print(f"Validation: {'OK' if is_safe else 'NG'}")
# => NG (contains HTML tags)

# Sanitization: remove or neutralize dangerous elements from input
# -> transform input into a safe form
sanitized = html.escape(user_input)
print(f"After sanitization: {sanitized}")
# => &lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;Hello, World!

# To strip tags entirely
stripped = re.sub(r'<[^>]+>', '', user_input)
print(f"After tag removal: {stripped}")
# => alert("XSS")Hello, World!

# Iron rules:
# 1. First, reject invalid input via validation
# 2. Sanitize the passing input for safe handling
# 3. Also escape on output (defense in depth)
```

### Q6: Should I do numeric range checks with regex?

**A**: Generally, do them in code. Range checks via regex are less readable and bug-prone:

```python
import re

# NG: expressing the 0-255 range with regex (IPv4 example)
# It works but is hard to read and maintain
range_regex = re.compile(r'^(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$')

# OK: regex only checks "is a digit string"; do range in code
def validate_range(value_str: str, min_val: int, max_val: int) -> bool:
    if not re.match(r'^\d+$', value_str):
        return False
    value = int(value_str)
    return min_val <= value <= max_val

# Guidelines:
# - Simple range (0-9, 0-99, etc.) -> regex is fine
# - Complex range (0-255, 1-366, etc.) -> handle in code
# - Well-known patterns like IPv4 -> regex (because they are widely known)
```

### Q7: How should I design test cases?

**A**: Combine boundary value testing with equivalence partitioning:

```python
import re

def create_test_cases(pattern_name: str, pattern: re.Pattern) -> list:
    """Test case design guidelines for regex patterns"""

    # Test case categories:
    categories = {
        "Normal (typical)":        "the most common input",
        "Normal (boundary)":       "input at the pattern's boundary",
        "Normal (minimum)":        "the shortest matching input",
        "Normal (maximum)":        "the longest matching input",
        "Abnormal (format viol.)": "obviously non-matching input",
        "Abnormal (boundary)":     "non-matching input near the boundary",
        "Abnormal (empty)":        "empty string",
        "Abnormal (special)":      "control chars, Unicode, newlines, etc.",
    }
    return categories

# Example: email address test cases
email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

test_matrix = [
    # (input, expected, category)
    ("user@example.com",          True,  "Normal (typical)"),
    ("a@b.cd",                    True,  "Normal (minimum)"),
    ("a" * 64 + "@example.com",   True,  "Normal (near boundary)"),
    ("user.name+tag@domain.co.jp",True,  "Normal (special chars)"),

    ("",                          False, "Abnormal (empty)"),
    ("user",                      False, "Abnormal (no @)"),
    ("user@",                     False, "Abnormal (no domain)"),
    ("@domain.com",               False, "Abnormal (no local part)"),
    ("user@domain",               False, "Abnormal (no TLD)"),
    ("user @domain.com",          False, "Abnormal (contains space)"),
    ("user@domain.c",             False, "Abnormal (TLD 1 char)"),
]

print(f"Email pattern tests ({len(test_matrix)} cases):")
all_passed = True
for email, expected, category in test_matrix:
    result = bool(email_pattern.match(email))
    passed = result == expected
    if not passed:
        all_passed = False
    status = "PASS" if passed else "FAIL"
    display = email if email else "(empty string)"
    print(f"  {status}: {display:<40} -- {category}")

print(f"\nResult: {'All tests passed' if all_passed else 'Some tests failed'}")
```

---

## 11. Practical Scenarios

### 11.1 Bulk Validation of Form Input

This shows a comprehensive validation function for use in real-world web forms.

```python
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class ValidationResult:
    field: str
    valid: bool
    value: str
    error: Optional[str] = None

class FormValidator:
    """Bulk validation of form input"""

    PATTERNS = {
        "email": re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        ),
        "phone_jp": re.compile(
            r'^(?:0[789]0-?\d{4}-?\d{4}|0\d{1,4}-?\d{1,4}-?\d{4})$'
        ),
        "postal_jp": re.compile(r'^\d{3}-?\d{4}$'),
        "date_iso": re.compile(
            r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$'
        ),
        "url": re.compile(r'^https?://[^\s<>"]+$'),
        "username": re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{2,19}$'),
    }

    MESSAGES = {
        "email":     "Please enter a valid email address",
        "phone_jp":  "Please enter a valid phone number (e.g., 090-1234-5678)",
        "postal_jp": "Please enter a valid postal code (e.g., 100-0001)",
        "date_iso":  "Please enter a valid date (e.g., 2026-01-01)",
        "url":       "Please enter a valid URL",
        "username":  "Please enter 3-20 alphanumeric characters (must start with a letter)",
    }

    def validate_field(
        self, field_name: str, value: str, field_type: str, required: bool = True
    ) -> ValidationResult:
        """Validate a single field"""
        if not value.strip():
            if required:
                return ValidationResult(field_name, False, value, "This field is required")
            return ValidationResult(field_name, True, value)

        pattern = self.PATTERNS.get(field_type)
        if pattern and not pattern.match(value.strip()):
            return ValidationResult(
                field_name, False, value,
                self.MESSAGES.get(field_type, "Invalid input format")
            )

        return ValidationResult(field_name, True, value)

    def validate_form(self, form_data: dict, schema: dict) -> list:
        """Validate the entire form"""
        results = []
        for field_name, config in schema.items():
            value = form_data.get(field_name, "")
            result = self.validate_field(
                field_name, value,
                config["type"],
                config.get("required", True)
            )
            results.append(result)
        return results


# Usage example
validator = FormValidator()

form_data = {
    "name":    "山田太郎",
    "email":   "yamada@example.com",
    "phone":   "090-1234-5678",
    "postal":  "100-0001",
    "website": "https://yamada.example.com",
}

schema = {
    "email":   {"type": "email",     "required": True},
    "phone":   {"type": "phone_jp",  "required": True},
    "postal":  {"type": "postal_jp", "required": True},
    "website": {"type": "url",       "required": False},
}

results = validator.validate_form(form_data, schema)
for r in results:
    status = "OK" if r.valid else "NG"
    print(f"  {status}: {r.field:<10} = {r.value}")
    if r.error:
        print(f"         Error: {r.error}")
```

### 11.2 Structured Parsing of Log Files

```python
import re
from datetime import datetime
from collections import defaultdict

# Parser for the Apache Combined Log Format
apache_log = re.compile(
    r'(?P<ip>\d+\.\d+\.\d+\.\d+)\s+'           # client IP
    r'(?P<ident>\S+)\s+'                         # identd
    r'(?P<user>\S+)\s+'                          # username
    r'\[(?P<datetime>[^\]]+)\]\s+'               # datetime
    r'"(?P<method>GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+'  # HTTP method
    r'(?P<path>\S+)\s+'                          # request path
    r'(?P<protocol>HTTP/\d\.\d)"\s+'             # protocol
    r'(?P<status>\d{3})\s+'                      # status code
    r'(?P<size>\d+|-)\s+'                        # response size
    r'"(?P<referer>[^"]*)"\s+'                   # referer
    r'"(?P<useragent>[^"]*)"'                    # user agent
)

sample_logs = [
    '192.168.1.100 - admin [11/Feb/2026:14:30:00 +0900] "GET /dashboard HTTP/1.1" 200 5432 "https://example.com/" "Mozilla/5.0"',
    '10.0.0.50 - - [11/Feb/2026:14:30:01 +0900] "POST /api/users HTTP/1.1" 201 128 "-" "curl/7.68.0"',
    '203.0.113.42 - - [11/Feb/2026:14:30:02 +0900] "GET /admin/login HTTP/1.1" 401 0 "-" "Python-urllib/3.9"',
    '192.168.1.100 - admin [11/Feb/2026:14:30:03 +0900] "DELETE /api/users/42 HTTP/1.1" 204 0 "https://example.com/admin" "Mozilla/5.0"',
]

stats = defaultdict(int)
for line in sample_logs:
    m = apache_log.match(line)
    if m:
        data = m.groupdict()
        status_class = f"{data['status'][0]}xx"
        stats[status_class] += 1
        print(f"  {data['method']:>6} {data['path']:<25} "
              f"{data['status']} from {data['ip']}")

print("\nStatus aggregation:")
for status_class, count in sorted(stats.items()):
    print(f"  {status_class}: {count} entries")
```

### 11.3 Parsing Configuration Files

```python
import re

# INI-format configuration file parser
section_pattern = re.compile(r'^\[([^\]]+)\]$')
kv_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*$')
comment_pattern = re.compile(r'^\s*[;#]')
empty_pattern = re.compile(r'^\s*$')

ini_content = """
; Database settings
[database]
host = localhost
port = 5432
name = myapp_production
user = dbadmin
password = s3cret!

# Application settings
[application]
debug = false
log_level = INFO
max_connections = 100
timeout = 30

[email]
smtp_host = smtp.example.com
smtp_port = 587
from_address = noreply@example.com
"""

config = {}
current_section = None

for line in ini_content.strip().split('\n'):
    # Skip empty lines and comments
    if empty_pattern.match(line) or comment_pattern.match(line):
        continue

    # Section header
    section_match = section_pattern.match(line)
    if section_match:
        current_section = section_match.group(1)
        config[current_section] = {}
        continue

    # key=value
    kv_match = kv_pattern.match(line)
    if kv_match and current_section:
        key, value = kv_match.groups()
        config[current_section][key] = value

# Display results
for section, values in config.items():
    print(f"\n  [{section}]")
    for key, value in values.items():
        print(f"    {key} = {value}")
```

### 11.4 CSV Field Extraction (with Quote Support)

```python
import re

# Regex-based CSV field parsing
# Supports quoted fields (which may contain commas)
csv_field = re.compile(
    r'(?:'
    r'"([^"]*(?:""[^"]*)*)"'   # quoted field (supports "" escape)
    r'|'
    r'([^,]*)'                  # unquoted field
    r')'
)

def parse_csv_line(line: str) -> list:
    """Decompose a CSV line into a list of fields"""
    fields = []
    for m in csv_field.finditer(line):
        quoted = m.group(1)
        unquoted = m.group(2)
        if quoted is not None:
            # Convert "" -> "
            fields.append(quoted.replace('""', '"'))
        elif unquoted is not None:
            fields.append(unquoted)
    return fields

test_csv_lines = [
    'John,Doe,30,New York',
    '"Smith, Jr.",Jane,25,"Los Angeles, CA"',
    'Alice,"She said ""hello""",28,Tokyo',
]

for line in test_csv_lines:
    fields = parse_csv_line(line)
    print(f"\n  Input: {line}")
    for i, field in enumerate(fields):
        print(f"    [{i}] {field}")

# Note: for serious CSV parsing, use the csv module
# import csv
# reader = csv.reader(io.StringIO(line))
```

---

## 12. Performance Optimization

### 12.1 Pattern Compilation and Reuse

```python
import re
import time

# Benchmark: precompiled vs. compiled each time
test_data = ["user@example.com"] * 10000

# Method 1: call re.match() each iteration (uses internal cache)
start = time.time()
for email in test_data:
    re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email)
method1_time = time.time() - start

# Method 2: use a precompiled pattern
pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
start = time.time()
for email in test_data:
    pattern.match(email)
method2_time = time.time() - start

print(f"  re.match() each time: {method1_time:.4f}s")
print(f"  Precompiled:          {method2_time:.4f}s")
print(f"  Speed ratio:          {method1_time / method2_time:.1f}x")

# Note: Python's re module caches recently used patterns internally,
# so the difference is small, but using precompiled patterns is best practice
```

### 12.2 Efficient Matching on Large Datasets

```python
import re
from typing import Iterator

def efficient_search(pattern: re.Pattern, lines: Iterator[str]) -> list:
    """Efficient regex search for large data sets"""
    results = []

    # Use match() rather than search() (faster for anchored matches)
    # Use search() only when leading match is not needed

    for line in lines:
        m = pattern.match(line)
        if m:
            results.append(m.group())
    return results

# Process line-by-line without loading the whole file
def search_large_file(filepath: str, pattern: re.Pattern):
    """Search a large file line-by-line (memory-efficient)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            m = pattern.search(line)
            if m:
                yield (line_num, m.group(), line.rstrip())

# Bulk matching of multiple patterns
def multi_pattern_search(patterns: dict, text: str) -> dict:
    """Search multiple patterns in one pass"""
    # Combine individual patterns with | so a single match suffices
    combined = '|'.join(f'(?P<{name}>{pat.pattern})'
                        for name, pat in patterns.items())
    combined_re = re.compile(combined)

    results = {}
    for m in combined_re.finditer(text):
        for name in patterns:
            if m.group(name):
                results.setdefault(name, []).append(m.group(name))
    return results
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining hands-on experience is the most important thing. Beyond theory, writing actual code and verifying behavior deepens your understanding.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. We recommend firmly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

The knowledge in this topic is frequently applied in everyday development work. It is especially important during code reviews and architecture design.

---

## Summary

| Pattern | Practical regex | Additional validation |
|---------|-------------|---------|
| Email | `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$` | Send confirmation email |
| URL | `https?://[^\s<>"]+` | `urllib.parse` / `URL` API |
| ISO date | `\d{4}-(0[1-9]\|1[0-2])-(0[1-9]\|[12]\d\|3[01])` | Parse with datetime |
| Phone (Japan) | `0[789]0-?\d{4}-?\d{4}` | libphonenumber |
| IPv4 | `(?:25[0-5]\|2[0-4]\d\|[01]?\d\d?)\.{4 parts}` | `ipaddress` module |
| Postal code (Japan) | `\d{3}-?\d{4}` | API verification |
| Username | `^[a-zA-Z][a-zA-Z0-9_-]{2,19}$` | Duplicate check (DB) |
| UUID v4 | `^[0-9a-f]{8}-...-[0-9a-f]{12}$` | Library validation |
| Color code | `^#[0-9a-fA-F]{3,8}$` | CSS parser |
| Iron rule | Regex is for format only. Logical and existence checks belong to code/libraries |

### Reaffirming Design Principles

```
+--------------------------------------------------------------+
|        Five Principles of Regex Pattern Design                |
|                                                              |
|  1. Principle of necessary sufficiency                       |
|     Avoid being overly strict; stay at a practical level     |
|     The regex's job is to "reject obviously invalid input"   |
|                                                              |
|  2. Principle of defense in depth                            |
|     Validate in three stages: regex -> logic -> external     |
|     Don't load all responsibility onto one layer             |
|                                                              |
|  3. Principle of testability                                 |
|     Always prepare test cases for your patterns              |
|     Cover normal, abnormal, and boundary cases               |
|                                                              |
|  4. Principle of maintainability                             |
|     Don't use unreadable patterns                            |
|     Express intent with comments and named groups            |
|                                                              |
|  5. Principle of safety                                      |
|     Consider ReDoS risk                                      |
|     Set input length limits for user input                   |
+--------------------------------------------------------------+
```

## Recommended Next Reading

- [02-text-processing.md](./02-text-processing.md) -- Text processing (sed/awk/grep)
- [03-regex-alternatives.md](./03-regex-alternatives.md) -- Alternatives to regular expressions

## References

1. **RFC 5322** "Internet Message Format" https://tools.ietf.org/html/rfc5322 -- Official specification of email addresses
2. **RFC 3986** "Uniform Resource Identifier (URI): Generic Syntax" https://tools.ietf.org/html/rfc3986 -- Official specification of URIs
3. **Google libphonenumber** https://github.com/google/libphonenumber -- The de facto standard library for phone number validation
4. **OWASP Input Validation Cheat Sheet** https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html -- Security best practices for input validation
5. **Regular Expression Denial of Service (ReDoS)** https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS -- Explanation and mitigation of ReDoS attacks
6. **Python re module documentation** https://docs.python.org/3/library/re.html -- Python standard library regex reference
7. **regex101.com** https://regex101.com/ -- Online regex testing and debugging tool
