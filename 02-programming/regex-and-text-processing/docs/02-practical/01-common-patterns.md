# よく使うパターン -- メール、URL、日付、電話番号

> 実務で頻出する正規表現パターンを「実用レベル」と「厳密レベル」の両面から解説する。各パターンの限界を正しく理解し、正規表現だけに頼らない堅牢なバリデーション設計を示す。

## この章で学ぶこと

1. **頻出パターンの実用的な実装** -- メール、URL、日付、電話番号、IPアドレス等
2. **厳密な仕様準拠と実用性のトレードオフ** -- RFC 完全準拠が不要な理由
3. **正規表現+追加検証の設計パターン** -- パターンマッチだけで完結しないバリデーション

---

## 1. メールアドレス

### 1.1 実用パターン

```python
import re

# 実用レベル: ほとんどの実際のメールアドレスにマッチ
email_pattern = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

test_emails = [
    "user@example.com",           # OK
    "user.name+tag@domain.co.jp", # OK
    "user@sub.domain.example.com",# OK
    "user@",                      # NG
    "@domain.com",                # NG
    "user@domain",                # NG (TLD なし)
    "user name@domain.com",       # NG (空白)
]

for email in test_emails:
    result = "OK" if email_pattern.match(email) else "NG"
    print(f"  {result}: {email}")
```

### 1.2 なぜ RFC 5322 完全準拠パターンを使わないのか

```
RFC 5322 準拠パターン: 数千文字に及ぶ正規表現
→ 保守不能、デバッグ不能、パフォーマンスリスク

実用的なアプローチ:
┌─────────────────────────────────────────┐
│ 1. 正規表現で基本形式をチェック          │
│    (@ がある、ドメインがある、TLD がある) │
│                                         │
│ 2. 確認メールを送信して実在確認          │
│    (これが唯一の正しい検証)             │
└─────────────────────────────────────────┘

理由:
- "valid" だが存在しないアドレスは正規表現で検出不能
- 存在するが RFC 非準拠のアドレスもある
- 正規表現の仕事は「明らかな誤入力を弾く」だけ
```

---

## 2. URL

### 2.1 実用パターン

```python
import re

# HTTP/HTTPS URL の実用パターン
url_pattern = re.compile(
    r'https?://'                # プロトコル
    r'(?:[a-zA-Z0-9]'          # ドメイン先頭
    r'(?:[a-zA-Z0-9-]{0,61}'   # ドメイン中間
    r'[a-zA-Z0-9])?\.)'        # ドメイン末尾 + ドット
    r'+[a-zA-Z]{2,}'           # TLD
    r'(?:/[^\s]*)?'            # パス(任意)
)

test_urls = [
    "https://example.com",
    "https://www.example.com/path/to/page",
    "http://sub.domain.example.co.jp/path?q=1&p=2#hash",
    "ftp://example.com",          # NG (http/https のみ)
    "not a url",                  # NG
]

for url in test_urls:
    m = url_pattern.search(url)
    result = m.group() if m else "NG"
    print(f"  {result}")
```

### 2.2 テキストからURLを抽出

```python
import re

text = """
公式サイト: https://example.com/docs
参考: http://sub.domain.co.jp/path?key=value
連絡先: mailto:info@example.com (これはマッチしない)
"""

# テキスト中からHTTP URLを抽出
url_extract = re.compile(r'https?://[^\s<>"]+')

urls = url_extract.findall(text)
for url in urls:
    print(f"  {url}")
# => https://example.com/docs
# => http://sub.domain.co.jp/path?key=value
```

---

## 3. 日付

### 3.1 各形式のパターン

```python
import re

# YYYY-MM-DD
iso_date = re.compile(r'\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b')

# YYYY/MM/DD
slash_date = re.compile(r'\b(\d{4})/(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])\b')

# DD/MM/YYYY (ヨーロッパ式)
eu_date = re.compile(r'\b(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/(\d{4})\b')

# MM/DD/YYYY (アメリカ式)
us_date = re.compile(r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(\d{4})\b')

# 日本語日付
jp_date = re.compile(r'(\d{4})年(0?[1-9]|1[0-2])月(0?[1-9]|[12]\d|3[01])日')

# テスト
texts = [
    "2026-02-11",
    "2026/02/11",
    "11/02/2026",
    "02/11/2026",
    "2026年2月11日",
]

patterns = {
    "ISO": iso_date,
    "スラッシュ": slash_date,
    "EU": eu_date,
    "US": us_date,
    "日本語": jp_date,
}

for text in texts:
    for name, pat in patterns.items():
        m = pat.search(text)
        if m:
            print(f"  {text} → {name}: {m.groups()}")
```

### 3.2 日付バリデーションの注意

```python
import re
from datetime import datetime

# 正規表現だけでは不十分な例:
# "2026-02-30" -- 形式は正しいが、2月30日は存在しない
# "2025-02-29" -- 閏年ではないので存在しない
# "2024-02-29" -- 閏年なので存在する

def validate_date(date_str: str) -> bool:
    """正規表現 + datetime で日付を検証"""
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return False
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

print(validate_date("2026-02-11"))  # => True
print(validate_date("2026-02-30"))  # => False (30日は存在しない)
print(validate_date("2025-02-29"))  # => False (閏年でない)
print(validate_date("2024-02-29"))  # => True  (閏年)
```

---

## 4. 電話番号

### 4.1 各国のパターン

```python
import re

# 日本の電話番号
jp_phone_patterns = {
    # 携帯電話: 090/080/070-XXXX-XXXX
    "携帯": re.compile(r'0[789]0-?\d{4}-?\d{4}'),
    # 固定電話(東京): 03-XXXX-XXXX
    "固定(東京)": re.compile(r'03-?\d{4}-?\d{4}'),
    # 固定電話(大阪): 06-XXXX-XXXX
    "固定(大阪)": re.compile(r'06-?\d{4}-?\d{4}'),
    # フリーダイヤル: 0120-XXX-XXX
    "フリーダイヤル": re.compile(r'0120-?\d{3}-?\d{3}'),
    # 国際形式: +81-XX-XXXX-XXXX
    "国際": re.compile(r'\+81-?\d{1,4}-?\d{1,4}-?\d{4}'),
}

# 汎用的な日本の電話番号パターン
jp_phone_general = re.compile(
    r'(?:\+81|0)'           # +81 または 0
    r'[\d-]{9,13}'          # 数字とハイフンで9-13文字
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

### 4.2 国際的な電話番号(E.164)

```python
import re

# E.164 形式: +[国番号][電話番号] (最大15桁)
e164_pattern = re.compile(r'^\+[1-9]\d{1,14}$')

test_numbers = [
    "+819012345678",     # 日本
    "+14155551234",      # アメリカ
    "+442012345678",     # イギリス
    "+0123456789",       # NG (0で始まる国番号はない)
    "+123456789012345",  # NG (16桁 -- 上限超過)
]

for num in test_numbers:
    result = "OK" if e164_pattern.match(num) else "NG"
    print(f"  {result}: {num}")

# 注: 電話番号の厳密なバリデーションには
# Google の libphonenumber ライブラリを推奨
```

---

## 5. IPアドレス

### 5.1 IPv4

```python
import re

# IPv4: 0.0.0.0 から 255.255.255.255
ipv4_pattern = re.compile(
    r'\b'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'   # 第1オクテット
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'   # 第2オクテット
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'   # 第3オクテット
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'     # 第4オクテット
    r'\b'
)

test_ips = [
    "192.168.1.1",      # OK
    "10.0.0.0",         # OK
    "255.255.255.255",  # OK
    "0.0.0.0",          # OK
    "256.1.1.1",        # NG (256は範囲外)
    "192.168.1",        # NG (3オクテットのみ)
    "192.168.1.1.1",    # NG (5オクテット)
]

for ip in test_ips:
    result = "OK" if ipv4_pattern.fullmatch(ip) else "NG"
    print(f"  {result}: {ip}")
```

### 5.2 IPv6(簡易版)

```python
import re
import ipaddress

# IPv6 は正規表現だけで厳密にマッチするのは困難
# 簡易パターン + ライブラリでの検証を推奨

ipv6_simple = re.compile(
    r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}'  # フル形式のみ
)

# 推奨: ipaddress モジュールを使用
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

---

## 6. その他の頻出パターン

### 6.1 郵便番号(日本)

```python
import re

# 日本の郵便番号: XXX-XXXX
jp_postal = re.compile(r'\b\d{3}-?\d{4}\b')

test_codes = ["100-0001", "1000001", "100-001"]
for code in test_codes:
    result = "OK" if jp_postal.fullmatch(code) else "NG"
    print(f"  {result}: {code}")
# => OK: 100-0001
# => OK: 1000001
# => NG: 100-001
```

### 6.2 クレジットカード番号(Luhnチェック付き)

```python
import re

# 主要カードブランドのパターン
card_patterns = {
    "Visa":       re.compile(r'^4\d{12}(?:\d{3})?$'),
    "Mastercard": re.compile(r'^5[1-5]\d{14}$'),
    "AMEX":       re.compile(r'^3[47]\d{13}$'),
    "JCB":        re.compile(r'^(?:2131|1800|35\d{3})\d{11}$'),
}

def luhn_check(number: str) -> bool:
    """Luhn アルゴリズムでチェックディジットを検証"""
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
    """カード番号を検証(形式 + Luhn)"""
    clean = number.replace(' ', '').replace('-', '')
    for brand, pattern in card_patterns.items():
        if pattern.match(clean):
            return brand, luhn_check(clean)
    return "Unknown", False

print(validate_card("4111 1111 1111 1111"))  # => ('Visa', True)
```

### 6.3 パスワード強度

```python
import re

def check_password_strength(password: str) -> dict:
    """パスワード強度を複数基準でチェック"""
    checks = {
        "8文字以上": len(password) >= 8,
        "大文字含む": bool(re.search(r'[A-Z]', password)),
        "小文字含む": bool(re.search(r'[a-z]', password)),
        "数字含む":   bool(re.search(r'\d', password)),
        "記号含む":   bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)),
        "連続文字なし": not bool(re.search(r'(.)\1{2,}', password)),
    }
    strength = sum(checks.values())
    return {"checks": checks, "score": f"{strength}/6"}

result = check_password_strength("MyP@ssw0rd")
for check, passed in result["checks"].items():
    print(f"  {'OK' if passed else 'NG'}: {check}")
print(f"  スコア: {result['score']}")
```

---

## 7. ASCII 図解

### 7.1 バリデーション設計のフロー

```
ユーザー入力
    │
    ▼
┌─────────────────────────┐
│ ステップ1: 形式チェック   │
│ (正規表現)               │
│ 例: メール形式か?        │
│ → 明らかな誤入力を弾く   │
└─────────────┬───────────┘
              │ 通過
              ▼
┌─────────────────────────┐
│ ステップ2: 論理チェック   │
│ (プログラムロジック)      │
│ 例: 日付は実在するか?    │
│ 例: 値の範囲は適切か?    │
└─────────────┬───────────┘
              │ 通過
              ▼
┌─────────────────────────┐
│ ステップ3: 実在チェック   │
│ (外部サービス)           │
│ 例: メール送達確認       │
│ 例: API で住所検証       │
└─────────────┬───────────┘
              │ 通過
              ▼
        入力を受理
```

### 7.2 メールアドレスパターンの構造

```
パターン: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$

^                              $
│                              │
│  ┌──── ローカルパート ────┐   │
│  │ [a-zA-Z0-9._%+-]+    │   │
│  │ 英数字と . _ % + -   │   │
│  │ 1文字以上            │   │
│  └───────────────────────┘   │
│              @               │
│  ┌──── ドメインパート ────┐   │
│  │ [a-zA-Z0-9.-]+       │   │
│  │ 英数字と . -          │   │
│  │ 1文字以上            │   │
│  └───────────────────────┘   │
│              .               │
│  ┌──── TLD ──────────────┐   │
│  │ [a-zA-Z]{2,}         │   │
│  │ 英字のみ、2文字以上   │   │
│  └───────────────────────┘   │

例: user.name+tag@sub.domain.co.jp
    ^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^ ^^
    ローカル       ドメイン       TLD
```

### 7.3 IPv4パターンの各オクテット分解

```
IPv4 オクテット: 0 から 255

パターン: (?:25[0-5]|2[0-4]\d|[01]?\d\d?)

分岐1: 25[0-5]     → 250, 251, 252, 253, 254, 255
分岐2: 2[0-4]\d    → 200-249
分岐3: [01]?\d\d?  → 0-199

分岐の優先順序が重要:
  25[0-5]  → まず 250-255 をチェック
  2[0-4]\d → 次に 200-249 をチェック
  [01]?\d\d? → 残り 0-199 をチェック

もし順序を逆にすると:
  [01]?\d\d? → "25" にマッチ → "5" が余る
  → 正しくマッチしない可能性
```

---

## 8. 比較表

### 8.1 パターン精度 vs 複雑さ

| パターン | 簡易版 | 実用版 | 厳密版 | 推奨 |
|---------|--------|--------|--------|------|
| メール | `.+@.+\..+` | `[a-zA-Z0-9._%+-]+@...` | RFC 5322 (数千文字) | 実用版 + 送達確認 |
| URL | `https?://\S+` | ドメイン検証付き | RFC 3986 完全準拠 | 実用版 |
| 日付 | `\d{4}-\d{2}-\d{2}` | 月/日の範囲チェック | 閏年対応 | 実用版 + datetime |
| 電話番号 | `[\d-+]+` | 国/地域別パターン | libphonenumber | 実用版 or ライブラリ |
| IPv4 | `\d+\.\d+\.\d+\.\d+` | 0-255 範囲チェック | 完全検証 | 実用版 |

### 8.2 正規表現 vs 専用ライブラリ

| 検証対象 | 正規表現で十分か | 推奨ライブラリ |
|---------|---------------|--------------|
| メールの形式 | 実用レベルなら可 | -- |
| メールの実在 | 不可 | SMTP検証 / 確認メール |
| URL の形式 | 実用レベルなら可 | urllib.parse (Python) |
| 日付の妥当性 | 不可(閏年等) | datetime (Python) |
| 電話番号 | 基本形式は可 | libphonenumber |
| クレジットカード | 形式は可 | Luhn + 決済API |
| HTML | 不可 | BeautifulSoup, lxml |
| JSON | 不可 | json.loads() |

---

## 9. アンチパターン

### 9.1 アンチパターン: 正規表現だけで日付を完全検証する

```python
import re

# NG: 閏年を正規表現で処理しようとする
# (パターンが極めて複雑になり保守不能)
leap_year_pattern = r"""...(数百文字のパターン)..."""

# OK: 正規表現は形式チェックのみ、論理チェックはコードで
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

### 9.2 アンチパターン: パターンのコピペ

```python
import re

# NG: StackOverflow からコピペしたパターンをそのまま使う
# 理由: コンテキスト(言語、Unicode設定)が異なる場合がある
email_copied = r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08...]"
# → 出典不明、メンテナンス不能、エッジケース不明

# OK: 要件に合わせて自分で設計し、テストを書く
email_own = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# テストケースを必ず用意
assert email_own.match("user@example.com")
assert email_own.match("a.b+c@d.co.jp")
assert not email_own.match("user@")
assert not email_own.match("@domain.com")
```

---

## 10. FAQ

### Q1: メールアドレスの正規表現はどこまで厳密にすべきか？

**A**: 実務では「基本形式チェック + 確認メール送信」で十分。RFC 5322 完全準拠は保守性とパフォーマンスの面で非推奨:

```python
# 実用的な基準:
# 1. @ が1つある
# 2. ローカルパートとドメインパートがある
# 3. TLD が2文字以上ある
# → これ以上の厳密性は確認メールで担保する
```

### Q2: URL バリデーションで最も見落としがちなケースは？

**A**: 以下のケースが見落としやすい:

- **国際化ドメイン名(IDN)**: `https://日本語.jp` -- Punycode 変換が必要
- **ポート番号**: `http://localhost:3000`
- **認証情報**: `http://user:pass@host.com` -- セキュリティリスク
- **フラグメント**: `https://example.com/page#section`
- **クエリパラメータのエンコード**: `?q=%E6%97%A5%E6%9C%AC`

ライブラリ(`urllib.parse`、`URL` API)の使用を推奨。

### Q3: 電話番号の国際対応で推奨されるアプローチは？

**A**: Google の **libphonenumber** ライブラリを使うのが最善。各国の電話番号ルールは複雑で頻繁に変更されるため、正規表現での完全対応は非現実的:

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

---

## まとめ

| パターン | 実用正規表現 | 追加検証 |
|---------|-------------|---------|
| メール | `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$` | 確認メール送信 |
| URL | `https?://[^\s<>"]+` | `urllib.parse` / `URL` API |
| ISO日付 | `\d{4}-(0[1-9]\|1[0-2])-(0[1-9]\|[12]\d\|3[01])` | datetime でパース |
| 電話番号(日本) | `0[789]0-?\d{4}-?\d{4}` | libphonenumber |
| IPv4 | `(?:25[0-5]\|2[0-4]\d\|[01]?\d\d?)\.{4部分}` | `ipaddress` モジュール |
| 郵便番号(日本) | `\d{3}-?\d{4}` | API検証 |
| 鉄則 | 正規表現は形式チェックのみ。論理・実在チェックはコード/ライブラリで |

## 次に読むべきガイド

- [02-text-processing.md](./02-text-processing.md) -- テキスト処理(sed/awk/grep)
- [03-regex-alternatives.md](./03-regex-alternatives.md) -- 正規表現の代替手法

## 参考文献

1. **RFC 5322** "Internet Message Format" https://tools.ietf.org/html/rfc5322 -- メールアドレスの公式仕様
2. **RFC 3986** "Uniform Resource Identifier (URI): Generic Syntax" https://tools.ietf.org/html/rfc3986 -- URI の公式仕様
3. **Google libphonenumber** https://github.com/google/libphonenumber -- 電話番号検証の事実上の標準ライブラリ
4. **OWASP Input Validation Cheat Sheet** https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html -- 入力検証のセキュリティベストプラクティス
