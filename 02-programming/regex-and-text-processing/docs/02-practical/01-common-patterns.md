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

### 1.3 メールアドレスのエッジケース詳細

実務で遭遇するメールアドレスの境界ケースを詳しく解説する。

```python
import re

email_pattern = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

# エッジケース集: 実用パターンの限界を理解する
edge_cases = {
    # --- RFC 的には有効だが、実用パターンでは NG になるケース ---
    '"user name"@example.com': "RFC有効: クォート付きローカルパート",
    'user@[192.168.1.1]':      "RFC有効: IPリテラルドメイン",
    '(comment)user@example.com': "RFC有効: コメント付き",
    'user@example':            "RFC有効: TLD なしのローカルドメイン",

    # --- 実用パターンで正しく NG になるケース ---
    'user@.example.com':       "NG: ドメインがドットで始まる",
    'user@example..com':       "NG: ドメイン内に連続ドット",
    '.user@example.com':       "NG: ローカルパートがドットで始まる",
    'user.@example.com':       "NG: ローカルパートがドットで終わる",

    # --- 新しい TLD への対応 ---
    'user@example.photography': "OK: 長い TLD (.photography)",
    'user@example.museum':      "OK: .museum",
    'user@example.co.uk':       "OK: 二重 TLD",
    'user@example.xn--p1ai':    "OK: 国際化 TLD (Punycode)",
}

for email, description in edge_cases.items():
    result = "OK" if email_pattern.match(email) else "NG"
    print(f"  {result}: {email:<40} -- {description}")
```

### 1.4 改良版メールパターン

基本パターンの弱点を補強した改良版を示す。

```python
import re

# 改良版: 連続ドットやドット始まり/終わりを弾く
email_improved = re.compile(
    r'^'
    r'(?![.])'                      # ドットで始まらない
    r'[a-zA-Z0-9]'                  # 英数字で開始
    r'(?:[a-zA-Z0-9._%+-]*'         # 中間部分
    r'[a-zA-Z0-9_%+-])?'            # ドットで終わらない（1文字の場合は省略可）
    r'@'
    r'(?![.-])'                     # ドメインがドットやハイフンで始まらない
    r'[a-zA-Z0-9]'                  # ドメイン先頭
    r'(?:[a-zA-Z0-9.-]*'            # ドメイン中間
    r'[a-zA-Z0-9])?'               # ドメインがハイフンで終わらない
    r'\.[a-zA-Z]{2,}$'             # TLD
)

test_improved = [
    ("user@example.com",        True),
    (".user@example.com",       False),  # ドット始まり
    ("user.@example.com",       False),  # ドット終わり
    ("user..name@example.com",  False),  # 連続ドット
    ("u@example.com",           True),   # 1文字ローカルパート
    ("user@-domain.com",        False),  # ハイフン始まりドメイン
    ("user@domain-.com",        False),  # ハイフン終わりドメイン
]

for email, expected in test_improved:
    result = bool(email_improved.match(email))
    status = "PASS" if result == expected else "FAIL"
    print(f"  {status}: {email:<30} expected={expected}, got={result}")
```

### 1.5 複数言語でのメールバリデーション

```javascript
// JavaScript 版
const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

// HTML5 の input[type="email"] は独自のパターンを使用
// ブラウザ標準のバリデーションを活用するのが最善
const form = document.createElement('form');
const input = document.createElement('input');
input.type = 'email';
input.value = 'test@example.com';
console.log(input.checkValidity()); // true
```

```go
// Go 版
package main

import (
    "fmt"
    "net/mail"
    "regexp"
)

func main() {
    // 正規表現による基本チェック
    pattern := regexp.MustCompile(
        `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`,
    )
    fmt.Println(pattern.MatchString("user@example.com")) // true

    // 推奨: net/mail パッケージを使用
    _, err := mail.ParseAddress("user@example.com")
    fmt.Println(err == nil) // true
}
```

```ruby
# Ruby 版
# 標準ライブラリの URI::MailTo を活用
require 'uri'

email = "user@example.com"
pattern = /\A[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\z/

puts email.match?(pattern)  # => true

# URI::MailTo::EMAIL_REGEXP も利用可能
puts email.match?(URI::MailTo::EMAIL_REGEXP)  # => true
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

### 2.3 URL の各構成要素を分解して抽出

URLの構造をキャプチャグループで分解し、各要素を個別に取得するパターンを示す。

```python
import re

# URL の各構成要素をキャプチャグループで抽出
url_decompose = re.compile(
    r'^'
    r'(?P<scheme>https?)'              # スキーム
    r'://'
    r'(?:(?P<user>[^:@]+)'             # ユーザー（任意）
    r'(?::(?P<password>[^@]+))?@)?'    # パスワード（任意）
    r'(?P<host>[a-zA-Z0-9.-]+)'        # ホスト
    r'(?::(?P<port>\d{1,5}))?'         # ポート（任意）
    r'(?P<path>/[^?#]*)?'              # パス（任意）
    r'(?:\?(?P<query>[^#]*))?'         # クエリ（任意）
    r'(?:#(?P<fragment>.*))?'          # フラグメント（任意）
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

出力例:

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

### 2.4 クエリパラメータのパースと抽出

```python
import re
from urllib.parse import urlparse, parse_qs

url = "https://example.com/search?q=python+regex&page=2&lang=ja&sort=date"

# 方法1: 正規表現でクエリパラメータを個別に抽出
param_pattern = re.compile(r'[?&]([^=]+)=([^&]*)')
params_regex = param_pattern.findall(url)
print("正規表現:")
for key, value in params_regex:
    print(f"  {key} = {value}")

# 方法2: urllib.parse を使用（推奨）
parsed = urlparse(url)
params_lib = parse_qs(parsed.query)
print("\nurllib.parse:")
for key, values in params_lib.items():
    print(f"  {key} = {values}")

# 注意: 正規表現はエンコード済みパラメータの処理が不十分
# %E6%97%A5%E6%9C%AC → "日本" のようなデコードにはライブラリを使用する
```

### 2.5 マークダウンや HTML からリンクを抽出

```python
import re

# マークダウンのリンクを抽出: [text](url)
markdown_link = re.compile(
    r'\[([^\]]+)\]'          # リンクテキスト
    r'\(([^)]+)\)'           # URL
)

md_text = """
詳細は[公式ドキュメント](https://docs.example.com/guide)を参照。
[APIリファレンス](https://api.example.com/v2/docs)も確認してください。
画像: ![alt text](https://img.example.com/logo.png)
"""

for m in markdown_link.finditer(md_text):
    print(f"  テキスト: {m.group(1)}")
    print(f"  URL:     {m.group(2)}\n")

# HTML の <a> タグからリンクを抽出
html_link = re.compile(
    r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>'   # href 属性
    r'(.*?)'                                        # リンクテキスト
    r'</a>',
    re.DOTALL
)

html_text = """
<a href="https://example.com" class="link">Example</a>
<a href="/relative/path" target="_blank">Relative Link</a>
"""

for m in html_link.finditer(html_text):
    print(f"  href: {m.group(1)}, text: {m.group(2)}")

# 注意: 本格的な HTML パースには BeautifulSoup を使うべき
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

### 3.3 日時（DateTime）パターン

日付だけでなく時刻まで含むパターンを示す。

```python
import re
from datetime import datetime

# ISO 8601 日時形式: YYYY-MM-DDTHH:MM:SS
iso_datetime = re.compile(
    r'\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])'  # 日付部分
    r'[T ]'                                                # 区切り（T またはスペース）
    r'([01]\d|2[0-3]):([0-5]\d):([0-5]\d)'                # 時刻部分
    r'(?:\.(\d{1,6}))?'                                   # マイクロ秒（任意）
    r'(?:Z|([+-])([01]\d|2[0-3]):?([0-5]\d))?\b'          # タイムゾーン（任意）
)

test_datetimes = [
    "2026-02-11T14:30:00",           # ローカル時刻
    "2026-02-11T14:30:00Z",          # UTC
    "2026-02-11T14:30:00+09:00",     # JST
    "2026-02-11 14:30:00.123456",    # マイクロ秒付き
    "2026-02-11T14:30:00-05:00",     # EST
    "2026-02-11T25:00:00",           # NG: 25時は存在しない
]

for dt_str in test_datetimes:
    m = iso_datetime.search(dt_str)
    if m:
        print(f"  OK: {dt_str}")
        print(f"      日付: {m.group(1)}-{m.group(2)}-{m.group(3)}")
        print(f"      時刻: {m.group(4)}:{m.group(5)}:{m.group(6)}")
    else:
        print(f"  NG: {dt_str}")
```

### 3.4 相対日付表現のパース

ログやテキストから「3日前」「2週間後」のような相対日付表現を抽出する。

```python
import re
from datetime import datetime, timedelta

# 日本語の相対日付表現
relative_date_jp = re.compile(
    r'(\d+)\s*'
    r'(秒|分|時間|日|週間?|ヶ月|か月|カ月|年)'
    r'\s*(前|後|先|以内)'
)

# 英語の相対日付表現
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
        print(f"    抽出: {m.group(0)}")
        print(f"    数値: {m.group(1)}, 単位: {m.group(2)}, 方向: {m.group(3)}")
```

### 3.5 日付範囲のバリデーション

```python
import re
from datetime import datetime

def validate_date_range(start_str: str, end_str: str) -> dict:
    """日付範囲のバリデーション"""
    date_pattern = re.compile(r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$')

    result = {"valid": True, "errors": []}

    # 形式チェック
    if not date_pattern.match(start_str):
        result["valid"] = False
        result["errors"].append(f"開始日の形式が不正: {start_str}")
    if not date_pattern.match(end_str):
        result["valid"] = False
        result["errors"].append(f"終了日の形式が不正: {end_str}")

    if not result["valid"]:
        return result

    # 日付の妥当性チェック
    try:
        start = datetime.strptime(start_str, '%Y-%m-%d')
        end = datetime.strptime(end_str, '%Y-%m-%d')
    except ValueError as e:
        result["valid"] = False
        result["errors"].append(f"日付が存在しない: {e}")
        return result

    # 論理チェック: 開始日 <= 終了日
    if start > end:
        result["valid"] = False
        result["errors"].append("開始日が終了日より後です")

    # 範囲チェック: 最大365日以内
    if (end - start).days > 365:
        result["valid"] = False
        result["errors"].append("日付範囲が365日を超えています")

    return result

# テスト
cases = [
    ("2026-01-01", "2026-12-31"),  # OK
    ("2026-12-31", "2026-01-01"),  # NG: 逆順
    ("2026-02-29", "2026-03-01"),  # NG: 2026年2月29日は存在しない
    ("2025-01-01", "2026-12-31"),  # NG: 365日超過
]

for start, end in cases:
    r = validate_date_range(start, end)
    status = "OK" if r["valid"] else "NG"
    print(f"  {status}: {start} ~ {end}")
    for err in r["errors"]:
        print(f"       {err}")
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

### 4.3 各国の電話番号パターン詳細

```python
import re

# 各国の電話番号パターン集
international_phone_patterns = {
    # アメリカ/カナダ (NANP): +1-NXX-NXX-XXXX
    "US/CA": re.compile(
        r'(?:\+1[-.\s]?)?'            # 国番号（任意）
        r'\(?[2-9]\d{2}\)?'            # エリアコード
        r'[-.\s]?'
        r'[2-9]\d{2}'                  # 局番
        r'[-.\s]?'
        r'\d{4}'                       # 加入者番号
    ),

    # イギリス: +44 XXXX XXXXXX
    "UK": re.compile(
        r'(?:\+44[-.\s]?|0)'           # 国番号またはトランクプレフィックス
        r'[1-9]\d{1,4}'               # エリアコード
        r'[-.\s]?'
        r'\d{4,8}'                     # 加入者番号
    ),

    # ドイツ: +49 XXXX XXXXXXX
    "DE": re.compile(
        r'(?:\+49[-.\s]?|0)'
        r'[1-9]\d{1,4}'
        r'[-.\s]?'
        r'\d{3,8}'
    ),

    # 中国: +86 1XX XXXX XXXX (携帯)
    "CN_mobile": re.compile(
        r'(?:\+86[-.\s]?)?'
        r'1[3-9]\d'                    # 携帯プレフィックス
        r'[-.\s]?'
        r'\d{4}'
        r'[-.\s]?'
        r'\d{4}'
    ),

    # 韓国: +82 01X-XXXX-XXXX (携帯)
    "KR_mobile": re.compile(
        r'(?:\+82[-.\s]?|0)'
        r'1[016789]'                   # 携帯プレフィックス
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

### 4.4 テキストから電話番号を一括抽出

```python
import re

# テキスト中から電話番号らしい文字列を抽出する汎用パターン
phone_extractor = re.compile(
    r'(?:'
    r'\+\d{1,3}[-.\s]?'               # 国番号付き
    r'|'
    r'0'                               # 国内番号
    r')'
    r'(?:\d[-.\s]?){8,13}'            # 8〜13桁の数字列
)

document = """
お問い合わせ先:
  東京本社: 03-1234-5678
  大阪支社: 06-9876-5432
  携帯（担当者直通）: 090-1111-2222
  フリーダイヤル: 0120-456-789
  海外からのお問い合わせ: +81-3-1234-5678

※ 営業時間: 9:00-18:00（数字だが電話番号ではない）
※ FAX: 03-1234-5679
"""

phones = phone_extractor.findall(document)
print("抽出された電話番号:")
for phone in phones:
    # 正規化: ハイフンとスペースを除去
    normalized = re.sub(r'[-.\s]', '', phone)
    print(f"  原文: {phone:<25} 正規化: {normalized}")
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

### 5.3 CIDR 表記のパターン

```python
import re
import ipaddress

# IPv4 CIDR 表記: 192.168.1.0/24
ipv4_cidr = re.compile(
    r'\b'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'
    r'/([12]?\d|3[0-2])'      # サブネットマスク: 0-32
    r'\b'
)

test_cidrs = [
    "192.168.1.0/24",      # OK: /24 サブネット
    "10.0.0.0/8",          # OK: クラスA
    "172.16.0.0/12",       # OK: プライベートアドレス
    "192.168.1.0/33",      # NG: /33は範囲外
    "256.0.0.0/24",        # NG: 256は範囲外
]

for cidr in test_cidrs:
    m = ipv4_cidr.fullmatch(cidr)
    if m:
        # ライブラリでも検証
        try:
            network = ipaddress.ip_network(cidr, strict=False)
            print(f"  OK: {cidr:<20} ネットワーク={network.network_address}, "
                  f"ホスト数={network.num_addresses}")
        except ValueError as e:
            print(f"  NG: {cidr:<20} {e}")
    else:
        print(f"  NG: {cidr}")
```

### 5.4 プライベート IP アドレスの識別

```python
import re

# プライベート IP アドレスの範囲
# 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16

private_ipv4 = re.compile(
    r'\b(?:'
    r'10\.\d{1,3}\.\d{1,3}\.\d{1,3}'            # 10.0.0.0/8
    r'|'
    r'172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}'  # 172.16.0.0/12
    r'|'
    r'192\.168\.\d{1,3}\.\d{1,3}'               # 192.168.0.0/16
    r'|'
    r'127\.\d{1,3}\.\d{1,3}\.\d{1,3}'           # 127.0.0.0/8 (ループバック)
    r')\b'
)

test_private = [
    "10.0.0.1",        # プライベート
    "172.16.0.1",      # プライベート
    "172.31.255.255",  # プライベート
    "172.32.0.1",      # パブリック（172.32 は範囲外）
    "192.168.1.1",     # プライベート
    "127.0.0.1",       # ループバック
    "8.8.8.8",         # パブリック
    "203.0.113.1",     # パブリック（ドキュメント用）
]

for ip in test_private:
    is_private = bool(private_ipv4.match(ip))
    print(f"  {'Private' if is_private else 'Public ':>7}: {ip}")
```

### 5.5 ログファイルからの IP アドレス抽出と集計

```python
import re
from collections import Counter

# Apache/Nginx アクセスログからの IP 抽出
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

# IP を抽出して集計
ips = ipv4_pattern.findall(sample_log)
ip_counts = Counter(ips)

print("IP アドレス別アクセス数:")
for ip, count in ip_counts.most_common():
    print(f"  {ip:<16} {count} 回")

# 403 エラーの IP を特定
error_pattern = re.compile(
    r'(\d+\.\d+\.\d+\.\d+).*?"(?:GET|POST|PUT|DELETE)\s+\S+\s+HTTP/\d\.\d"\s+403'
)

error_ips = error_pattern.findall(sample_log)
error_counts = Counter(error_ips)
print("\n403 エラーの IP:")
for ip, count in error_counts.most_common():
    print(f"  {ip:<16} {count} 回 -- 不正アクセスの可能性")
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

### 6.4 ユーザー名のバリデーション

Webサービスで一般的なユーザー名のバリデーションパターンを示す。

```python
import re

# ユーザー名の要件:
# - 3〜20文字
# - 英数字、アンダースコア、ハイフンのみ
# - 先頭は英字
# - 連続するアンダースコアやハイフンは不可

username_pattern = re.compile(
    r'^'
    r'[a-zA-Z]'                  # 先頭は英字
    r'(?!.*[-_]{2})'             # 連続記号を否定先読みで禁止
    r'[a-zA-Z0-9_-]{2,19}'      # 残り2〜19文字（合計3〜20文字）
    r'$'
)

test_usernames = [
    ("alice",           True,  "OK: 基本的な名前"),
    ("user_name",       True,  "OK: アンダースコア入り"),
    ("user-name",       True,  "OK: ハイフン入り"),
    ("a1b2c3",          True,  "OK: 英数字混合"),
    ("ab",              False, "NG: 2文字（最低3文字）"),
    ("1user",           False, "NG: 数字で始まる"),
    ("_user",           False, "NG: アンダースコアで始まる"),
    ("user__name",      False, "NG: 連続アンダースコア"),
    ("user--name",      False, "NG: 連続ハイフン"),
    ("user name",       False, "NG: スペース含む"),
    ("user@name",       False, "NG: 特殊文字含む"),
    ("a" * 21,          False, "NG: 21文字（最大20文字）"),
]

for username, expected, description in test_usernames:
    result = bool(username_pattern.match(username))
    status = "PASS" if result == expected else "FAIL"
    display_name = username if len(username) <= 20 else username[:17] + "..."
    print(f"  {status}: {display_name:<20} -- {description}")
```

### 6.5 ファイルパスのパターン

```python
import re

# Unix/Linux ファイルパス
unix_path = re.compile(
    r'^(/[a-zA-Z0-9._-]+)+/?$'
)

# Windows ファイルパス
windows_path = re.compile(
    r'^[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$'
)

# ファイル拡張子の抽出
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
    ext = ext_match.group(1) if ext_match else "(なし)"
    print(f"  {os_type:>7}: {path:<40} valid={valid}, ext={ext}")
```

### 6.6 16進カラーコード

```python
import re

# CSS カラーコード
hex_color = re.compile(
    r'^#(?:'
    r'[0-9a-fA-F]{3}'    # 短縮形: #RGB
    r'|'
    r'[0-9a-fA-F]{4}'    # 短縮形+アルファ: #RGBA
    r'|'
    r'[0-9a-fA-F]{6}'    # フル形式: #RRGGBB
    r'|'
    r'[0-9a-fA-F]{8}'    # フル形式+アルファ: #RRGGBBAA
    r')$'
)

# CSS の rgb()/rgba() 関数形式
rgb_color = re.compile(
    r'^rgba?\(\s*'
    r'(\d{1,3})\s*,\s*'     # R: 0-255
    r'(\d{1,3})\s*,\s*'     # G: 0-255
    r'(\d{1,3})'             # B: 0-255
    r'(?:\s*,\s*'
    r'([01]?\.?\d*)'         # A: 0-1（任意）
    r')?\s*\)$'
)

# HSL 形式
hsl_color = re.compile(
    r'^hsla?\(\s*'
    r'(\d{1,3})\s*,\s*'     # H: 0-360
    r'(\d{1,3})%\s*,\s*'    # S: 0-100%
    r'(\d{1,3})%'            # L: 0-100%
    r'(?:\s*,\s*'
    r'([01]?\.?\d*)'         # A: 0-1（任意）
    r')?\s*\)$'
)

test_colors = [
    "#fff",                 # OK: 短縮形
    "#FF5733",              # OK: フル形式
    "#FF573380",            # OK: アルファ付き
    "#GGHHII",              # NG: 無効な16進
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

### 6.7 UUID のバリデーション

```python
import re

# UUID v4: xxxxxxxx-xxxx-4xxx-[89ab]xxx-xxxxxxxxxxxx
uuid_v4 = re.compile(
    r'^[0-9a-f]{8}-'
    r'[0-9a-f]{4}-'
    r'4[0-9a-f]{3}-'          # バージョン4
    r'[89ab][0-9a-f]{3}-'     # バリアント1
    r'[0-9a-f]{12}$',
    re.IGNORECASE
)

# 任意バージョンの UUID
uuid_any = re.compile(
    r'^[0-9a-f]{8}-'
    r'[0-9a-f]{4}-'
    r'[1-5][0-9a-f]{3}-'      # バージョン1-5
    r'[89ab][0-9a-f]{3}-'
    r'[0-9a-f]{12}$',
    re.IGNORECASE
)

test_uuids = [
    "550e8400-e29b-41d4-a716-446655440000",  # OK: v4
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8",  # OK: v1
    "not-a-uuid",                              # NG
    "550e8400-e29b-61d4-a716-446655440000",   # NG: バージョン6（v1-5のみ対応）
    "550e8400-e29b-41d4-c716-446655440000",   # NG: 無効なバリアント
]

for uuid_str in test_uuids:
    v4 = "v4" if uuid_v4.match(uuid_str) else "--"
    any_v = "OK" if uuid_any.match(uuid_str) else "NG"
    print(f"  {any_v}({v4}): {uuid_str}")
```

### 6.8 日本語テキストのパターン

```python
import re

# ひらがな
hiragana = re.compile(r'^[\u3040-\u309F]+$')

# カタカナ
katakana = re.compile(r'^[\u30A0-\u30FF]+$')

# 漢字（CJK統合漢字）
kanji = re.compile(r'^[\u4E00-\u9FFF]+$')

# 全角文字
zenkaku = re.compile(r'^[\uFF01-\uFF5E]+$')

# 日本語の名前（姓 名）: 漢字+スペース+漢字
jp_name = re.compile(r'^[\u4E00-\u9FFF\u3040-\u309F]{1,10}\s[\u4E00-\u9FFF\u3040-\u309F]{1,10}$')

# フリガナ（カタカナ）
furigana = re.compile(r'^[\u30A0-\u30FF\s]{2,20}$')

test_japanese = [
    ("あいうえお",     hiragana,  "ひらがな"),
    ("アイウエオ",     katakana,  "カタカナ"),
    ("漢字",           kanji,     "漢字"),
    ("山田 太郎",      jp_name,   "日本語名"),
    ("ヤマダ タロウ",  furigana,  "フリガナ"),
    ("ABC",            hiragana,  "英字（NG）"),
]

for text, pattern, description in test_japanese:
    result = "OK" if pattern.match(text) else "NG"
    print(f"  {result}: {text:<15} -- {description}")
```

### 6.9 数値フォーマットのパターン

```python
import re

# 整数（カンマ区切り対応）
integer_comma = re.compile(r'^-?(?:\d{1,3}(?:,\d{3})*|\d+)$')

# 小数点数
decimal_number = re.compile(r'^-?\d+(?:\.\d+)?$')

# 科学的記数法
scientific = re.compile(r'^-?\d+(?:\.\d+)?[eE][+-]?\d+$')

# パーセンテージ
percentage = re.compile(r'^-?\d+(?:\.\d+)?%$')

# 通貨（日本円）
yen = re.compile(r'^[¥￥]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$')

# 通貨（米ドル）
usd = re.compile(r'^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$')

test_numbers = [
    ("1,234,567",     integer_comma,  "カンマ区切り整数"),
    ("-42",           integer_comma,  "負の整数"),
    ("3.14159",       decimal_number, "小数"),
    ("6.022e23",      scientific,     "科学的記数法"),
    ("1.5E-10",       scientific,     "科学的記数法（負の指数）"),
    ("85.5%",         percentage,     "パーセンテージ"),
    ("¥1,234,567",    yen,            "日本円"),
    ("$99.99",        usd,            "米ドル"),
    ("1,23,456",      integer_comma,  "不正なカンマ区切り（NG）"),
]

for text, pattern, description in test_numbers:
    result = "OK" if pattern.match(text) else "NG"
    print(f"  {result}: {text:<20} -- {description}")
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

### 7.4 URL 構造の分解図

```
URL の構造 (RFC 3986):

  https://user:pass@www.example.com:443/path/to/page?key=val&k2=v2#section
  └─┬──┘   └─┬──┘ └─────┬────────┘└┬┘└────┬──────┘└────┬──────┘└──┬───┘
  scheme   userinfo     host     port    path         query     fragment

各コンポーネントの正規表現:
  scheme:    [a-zA-Z][a-zA-Z0-9+.-]*
  userinfo:  [^@]+
  host:      [a-zA-Z0-9.-]+
  port:      \d{1,5}
  path:      /[^?#]*
  query:     [^#]*
  fragment:  .*

完全な分解パターン:
  ^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*)://
   (?:(?P<userinfo>[^@]+)@)?
   (?P<host>[a-zA-Z0-9.-]+)
   (?::(?P<port>\d{1,5}))?
   (?P<path>/[^?#]*)?
   (?:\?(?P<query>[^#]*))?
   (?:#(?P<fragment>.*))?$
```

### 7.5 電話番号パターンの国際比較

```
各国の電話番号形式:

日本 (+81):
  携帯:     0[789]0-XXXX-XXXX     例: 090-1234-5678
  固定:     0X-XXXX-XXXX          例: 03-1234-5678
  国際:     +81-X0-XXXX-XXXX      例: +81-90-1234-5678

  パターン: 0[789]0-?\d{4}-?\d{4}
            ├─┘ ├─┘  ├────┘  ├────┘
            携帯  ハイフン  4桁   4桁
            prefix  任意

アメリカ (+1):
  形式:     (NXX) NXX-XXXX        例: (415) 555-1234
  国際:     +1-NXX-NXX-XXXX       例: +1-415-555-1234

  パターン: \(?[2-9]\d{2}\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}
            ├───────────┘          ├──────┘        ├────┘
            エリアコード            局番           加入者番号

イギリス (+44):
  携帯:     07XXX XXXXXX          例: 07911 123456
  固定:     0XX XXXX XXXX         例: 020 7946 0958
  国際:     +44 XXXX XXXXXX       例: +44 20 7946 0958
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

### 8.3 言語別の正規表現サポート比較

| 機能 | Python (`re`) | Python (`regex`) | JavaScript | Go | Ruby | Java |
|------|--------------|-----------------|------------|-----|------|------|
| 名前付きグループ | `(?P<name>...)` | `(?P<name>...)` | `(?<name>...)` | `(?P<name>...)` | `(?<name>...)` | `(?<name>...)` |
| 先読み | 対応 | 対応 | 対応 | 非対応 | 対応 | 対応 |
| 後読み | 固定長のみ | 可変長対応 | 対応 | 非対応 | 対応 | 対応 |
| Unicode プロパティ | `\p{...}` 非対応 | 対応 | 対応 | 対応 | 対応 | 対応 |
| 再帰パターン | 非対応 | 対応 | 非対応 | 非対応 | 非対応 | 非対応 |
| アトミックグループ | 非対応 | 対応 | 非対応 | 非対応 | 対応 | 対応 |
| POSIX 文字クラス | 非対応 | 対応 | 非対応 | 対応 | 対応 | 対応 |

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

### 9.3 アンチパターン: 過度に厳密なパターンで正当な入力を拒否する

```python
import re

# NG: TLD を2-3文字に制限
email_strict_tld = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,3}$'
)

# この制限だと以下の有効なアドレスが NG になる:
rejected_valid = [
    "user@example.museum",       # .museum (6文字)
    "user@example.photography",  # .photography (11文字)
    "user@example.technology",   # .technology (10文字)
    "user@example.international",# .international (13文字)
]

for email in rejected_valid:
    result = "OK" if email_strict_tld.match(email) else "NG"
    print(f"  {result}: {email}  -- 有効だが拒否される!")

# OK: TLD は2文字以上に設定（上限なし）
email_flexible_tld = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)
```

### 9.4 アンチパターン: HTML を正規表現でパースする

```python
import re

# NG: 正規表現で HTML をパースしようとする
# 有名な StackOverflow の回答が示す通り、これは不可能
html = '<div class="outer"><div class="inner">text</div></div>'

# この正規表現はネストされたタグを正しく処理できない
bad_tag_extract = re.compile(r'<div[^>]*>(.*?)</div>')
# → 最初の </div> でマッチが終わり、ネスト構造が壊れる

# OK: 限定的な用途なら正規表現を使える
# 例: 自己完結するタグの抽出
img_tag = re.compile(r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*/?>')
link_href = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>')

# OK: 本格的な HTML パースにはライブラリを使う
# from bs4 import BeautifulSoup
# soup = BeautifulSoup(html, 'html.parser')
# divs = soup.find_all('div', class_='inner')
```

### 9.5 アンチパターン: ReDoS（正規表現サービス拒否攻撃）に脆弱なパターン

```python
import re
import time

# NG: 壊滅的バックトラッキングが発生するパターン
# ユーザー入力に対して使用すると DoS 攻撃の危険がある

vulnerable_patterns = [
    # パターン1: ネストした量指定子
    (r'(a+)+$', 'a' * 25 + 'b'),
    # パターン2: 重複する文字クラス
    (r'([a-zA-Z]+)*@', 'a' * 25 + '!'),
    # パターン3: 交互に重複
    (r'(a|aa)+$', 'a' * 25 + 'b'),
]

for pattern, evil_input in vulnerable_patterns:
    print(f"\n  パターン: {pattern}")
    print(f"  入力:     '{evil_input[:30]}...' ({len(evil_input)}文字)")
    start = time.time()
    try:
        # タイムアウト付きで実行（実際のコードではタイムアウトを設定すべき）
        re.match(pattern, evil_input[:20])  # 短い入力でも遅延が見える
        elapsed = time.time() - start
        print(f"  時間:     {elapsed:.4f}秒")
        print(f"  警告:     入力長が増えると指数関数的に遅くなる!")
    except Exception as e:
        print(f"  エラー:   {e}")

# OK: ReDoS を防ぐ安全なパターン
safe_patterns = [
    r'[a-zA-Z]+$',          # ネストしない量指定子
    r'[a-zA-Z]+@',          # グループの量指定子を排除
    r'a+$',                 # 単純な量指定子
]

# 対策:
# 1. ネストした量指定子 (a+)+ を避ける
# 2. 入力長の上限を設定する
# 3. アトミックグループ (?>...) を使う（対応言語のみ）
# 4. タイムアウトを設定する
# 5. re2 などの保証付き正規表現エンジンを検討する
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

### Q4: 正規表現パターンのパフォーマンスを改善するにはどうすればよいか？

**A**: 以下の手法が有効:

```python
import re

# 1. パターンをコンパイルして再利用する
# NG: ループ内で毎回コンパイル
for line in lines:
    if re.match(r'^\d{4}-\d{2}-\d{2}', line):  # 毎回コンパイル
        pass

# OK: 事前にコンパイル
date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}')
for line in lines:
    if date_pattern.match(line):  # コンパイル済みを再利用
        pass

# 2. 不要なキャプチャグループを避ける
# NG: キャプチャ不要なのにグループ化
pattern_capture = re.compile(r'(https?)://([\w.]+)')

# OK: 非キャプチャグループを使用
pattern_noncapture = re.compile(r'(?:https?)://(?:[\w.]+)')

# 3. 具体的なパターンを先に書く
# NG: 汎用的なパターンが先
pattern_slow = re.compile(r'.*error.*fatal')

# OK: アンカーや具体的な文字から始める
pattern_fast = re.compile(r'^.*?error.*?fatal', re.MULTILINE)

# 4. 量指定子を最小限にする
# NG: 貪欲マッチ
greedy = re.compile(r'<.*>')       # 最長一致

# OK: 非貪欲マッチ（用途による）
lazy = re.compile(r'<.*?>')       # 最短一致
```

### Q5: 入力サニタイズと正規表現バリデーションの違いは？

**A**: 両者は目的が異なり、どちらか一方では不十分:

```python
import re
import html

user_input = '<script>alert("XSS")</script>Hello, World!'

# バリデーション: 入力が許容される形式かどうかを判定
# → 不正な入力を拒否する
is_safe = bool(re.match(r'^[a-zA-Z0-9\s,.!?]+$', user_input))
print(f"バリデーション: {'OK' if is_safe else 'NG'}")
# => NG (HTMLタグが含まれる)

# サニタイズ: 入力から危険な要素を除去または無害化する
# → 入力を安全な形に変換する
sanitized = html.escape(user_input)
print(f"サニタイズ後: {sanitized}")
# => &lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;Hello, World!

# タグを完全に除去する場合
stripped = re.sub(r'<[^>]+>', '', user_input)
print(f"タグ除去後: {stripped}")
# => alert("XSS")Hello, World!

# 鉄則:
# 1. まずバリデーションで不正入力を拒否
# 2. 通過した入力をサニタイズして安全に処理
# 3. 出力時にもエスケープ処理を行う（多層防御）
```

### Q6: 正規表現で数値の範囲チェックを行うべきか？

**A**: 基本的にはコードで行うべき。正規表現での範囲チェックは可読性が低く、バグが入りやすい:

```python
import re

# NG: 正規表現で 0-255 の範囲を表現（IPv4の例）
# 動作するが、読みにくく保守しにくい
range_regex = re.compile(r'^(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$')

# OK: 正規表現は「数字である」ことだけチェックし、範囲はコードで
def validate_range(value_str: str, min_val: int, max_val: int) -> bool:
    if not re.match(r'^\d+$', value_str):
        return False
    value = int(value_str)
    return min_val <= value <= max_val

# 使い分けの基準:
# - 範囲が単純（0-9, 0-99 など）→ 正規表現でも可
# - 範囲が複雑（0-255, 1-366 など）→ コードで処理
# - IPv4 のような定番パターン → 正規表現（広く知られているため）
```

### Q7: テストケースはどのように設計すべきか？

**A**: 境界値テストと等価クラス分割を組み合わせる:

```python
import re

def create_test_cases(pattern_name: str, pattern: re.Pattern) -> list:
    """正規表現パターンのテストケース設計ガイドライン"""

    # テストケースの分類:
    categories = {
        "正常系（典型）":     "最も一般的な入力",
        "正常系（境界）":     "パターンの境界に位置する入力",
        "正常系（最小）":     "マッチする最短の入力",
        "正常系（最大）":     "マッチする最長の入力",
        "異常系（形式違反）": "明らかにマッチしない入力",
        "異常系（境界）":     "マッチしないが境界に近い入力",
        "異常系（空入力）":   "空文字列",
        "異常系（特殊文字）": "制御文字、Unicode、改行等",
    }
    return categories

# 例: メールアドレスのテストケース
email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

test_matrix = [
    # (入力, 期待結果, カテゴリ)
    ("user@example.com",          True,  "正常系（典型）"),
    ("a@b.cd",                    True,  "正常系（最小）"),
    ("a" * 64 + "@example.com",   True,  "正常系（境界付近）"),
    ("user.name+tag@domain.co.jp",True,  "正常系（特殊文字）"),

    ("",                          False, "異常系（空入力）"),
    ("user",                      False, "異常系（@なし）"),
    ("user@",                     False, "異常系（ドメインなし）"),
    ("@domain.com",               False, "異常系（ローカルパートなし）"),
    ("user@domain",               False, "異常系（TLDなし）"),
    ("user @domain.com",          False, "異常系（空白含む）"),
    ("user@domain.c",             False, "異常系（TLD 1文字）"),
]

print(f"メールアドレスパターンのテスト ({len(test_matrix)} ケース):")
all_passed = True
for email, expected, category in test_matrix:
    result = bool(email_pattern.match(email))
    passed = result == expected
    if not passed:
        all_passed = False
    status = "PASS" if passed else "FAIL"
    display = email if email else "(空文字列)"
    print(f"  {status}: {display:<40} -- {category}")

print(f"\n結果: {'全テスト合格' if all_passed else 'テスト失敗あり'}")
```

---

## 11. 実践シナリオ

### 11.1 フォーム入力の一括バリデーション

実際のWebフォームで使用する包括的なバリデーション関数を示す。

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
    """フォーム入力の一括バリデーション"""

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
        "email":     "有効なメールアドレスを入力してください",
        "phone_jp":  "有効な電話番号を入力してください（例: 090-1234-5678）",
        "postal_jp": "有効な郵便番号を入力してください（例: 100-0001）",
        "date_iso":  "有効な日付を入力してください（例: 2026-01-01）",
        "url":       "有効なURLを入力してください",
        "username":  "3-20文字の英数字（先頭は英字）で入力してください",
    }

    def validate_field(
        self, field_name: str, value: str, field_type: str, required: bool = True
    ) -> ValidationResult:
        """単一フィールドのバリデーション"""
        if not value.strip():
            if required:
                return ValidationResult(field_name, False, value, "必須項目です")
            return ValidationResult(field_name, True, value)

        pattern = self.PATTERNS.get(field_type)
        if pattern and not pattern.match(value.strip()):
            return ValidationResult(
                field_name, False, value,
                self.MESSAGES.get(field_type, "入力形式が不正です")
            )

        return ValidationResult(field_name, True, value)

    def validate_form(self, form_data: dict, schema: dict) -> list:
        """フォーム全体のバリデーション"""
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


# 使用例
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
        print(f"         エラー: {r.error}")
```

### 11.2 ログファイルの構造化パース

```python
import re
from datetime import datetime
from collections import defaultdict

# Apache Combined Log Format のパーサー
apache_log = re.compile(
    r'(?P<ip>\d+\.\d+\.\d+\.\d+)\s+'           # クライアントIP
    r'(?P<ident>\S+)\s+'                         # identd
    r'(?P<user>\S+)\s+'                          # ユーザー名
    r'\[(?P<datetime>[^\]]+)\]\s+'               # 日時
    r'"(?P<method>GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+'  # HTTPメソッド
    r'(?P<path>\S+)\s+'                          # リクエストパス
    r'(?P<protocol>HTTP/\d\.\d)"\s+'             # プロトコル
    r'(?P<status>\d{3})\s+'                      # ステータスコード
    r'(?P<size>\d+|-)\s+'                        # レスポンスサイズ
    r'"(?P<referer>[^"]*)"\s+'                   # リファラー
    r'"(?P<useragent>[^"]*)"'                    # ユーザーエージェント
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

print("\nステータス集計:")
for status_class, count in sorted(stats.items()):
    print(f"  {status_class}: {count} 件")
```

### 11.3 設定ファイルのパース

```python
import re

# INI 形式の設定ファイルパーサー
section_pattern = re.compile(r'^\[([^\]]+)\]$')
kv_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*$')
comment_pattern = re.compile(r'^\s*[;#]')
empty_pattern = re.compile(r'^\s*$')

ini_content = """
; データベース設定
[database]
host = localhost
port = 5432
name = myapp_production
user = dbadmin
password = s3cret!

# アプリケーション設定
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
    # 空行とコメントをスキップ
    if empty_pattern.match(line) or comment_pattern.match(line):
        continue

    # セクションヘッダー
    section_match = section_pattern.match(line)
    if section_match:
        current_section = section_match.group(1)
        config[current_section] = {}
        continue

    # キー=値
    kv_match = kv_pattern.match(line)
    if kv_match and current_section:
        key, value = kv_match.groups()
        config[current_section][key] = value

# 結果表示
for section, values in config.items():
    print(f"\n  [{section}]")
    for key, value in values.items():
        print(f"    {key} = {value}")
```

### 11.4 CSVデータのフィールド抽出（引用符対応）

```python
import re

# CSV フィールドの正規表現パース
# 引用符付きフィールド（カンマを含む）に対応
csv_field = re.compile(
    r'(?:'
    r'"([^"]*(?:""[^"]*)*)"'   # 引用符付きフィールド（""エスケープ対応）
    r'|'
    r'([^,]*)'                  # 引用符なしフィールド
    r')'
)

def parse_csv_line(line: str) -> list:
    """CSVの1行をフィールドリストに分解"""
    fields = []
    for m in csv_field.finditer(line):
        quoted = m.group(1)
        unquoted = m.group(2)
        if quoted is not None:
            # "" → " に変換
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
    print(f"\n  入力: {line}")
    for i, field in enumerate(fields):
        print(f"    [{i}] {field}")

# 注意: 本格的な CSV パースには csv モジュールを使用すべき
# import csv
# reader = csv.reader(io.StringIO(line))
```

---

## 12. パフォーマンス最適化

### 12.1 パターンのコンパイルと再利用

```python
import re
import time

# ベンチマーク: コンパイル済み vs 毎回コンパイル
test_data = ["user@example.com"] * 10000

# 方法1: 毎回 re.match() を呼ぶ（内部キャッシュあり）
start = time.time()
for email in test_data:
    re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email)
method1_time = time.time() - start

# 方法2: コンパイル済みパターンを使う
pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
start = time.time()
for email in test_data:
    pattern.match(email)
method2_time = time.time() - start

print(f"  毎回 re.match():   {method1_time:.4f}秒")
print(f"  コンパイル済み:     {method2_time:.4f}秒")
print(f"  速度比:            {method1_time / method2_time:.1f}x")

# 注: Python の re モジュールは内部で最近使ったパターンをキャッシュするため、
# 差は小さいが、コンパイル済みを使うのがベストプラクティス
```

### 12.2 大量データでの効率的なマッチング

```python
import re
from typing import Iterator

def efficient_search(pattern: re.Pattern, lines: Iterator[str]) -> list:
    """大量データに対する効率的な正規表現検索"""
    results = []

    # search() ではなく match() を使う（先頭マッチで高速）
    # 先頭マッチが必要ない場合のみ search() を使用

    for line in lines:
        m = pattern.match(line)
        if m:
            results.append(m.group())
    return results

# ファイル全体を読み込まず、行単位で処理
def search_large_file(filepath: str, pattern: re.Pattern):
    """大きなファイルを行単位で検索（メモリ効率的）"""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            m = pattern.search(line)
            if m:
                yield (line_num, m.group(), line.rstrip())

# 複数パターンの一括マッチ
def multi_pattern_search(patterns: dict, text: str) -> dict:
    """複数パターンを一度に検索"""
    # 個別パターンを | で結合して1回のマッチで済ませる
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

## まとめ

| パターン | 実用正規表現 | 追加検証 |
|---------|-------------|---------|
| メール | `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$` | 確認メール送信 |
| URL | `https?://[^\s<>"]+` | `urllib.parse` / `URL` API |
| ISO日付 | `\d{4}-(0[1-9]\|1[0-2])-(0[1-9]\|[12]\d\|3[01])` | datetime でパース |
| 電話番号(日本) | `0[789]0-?\d{4}-?\d{4}` | libphonenumber |
| IPv4 | `(?:25[0-5]\|2[0-4]\d\|[01]?\d\d?)\.{4部分}` | `ipaddress` モジュール |
| 郵便番号(日本) | `\d{3}-?\d{4}` | API検証 |
| ユーザー名 | `^[a-zA-Z][a-zA-Z0-9_-]{2,19}$` | 重複チェック（DB） |
| UUID v4 | `^[0-9a-f]{8}-...-[0-9a-f]{12}$` | ライブラリ検証 |
| カラーコード | `^#[0-9a-fA-F]{3,8}$` | CSS パーサー |
| 鉄則 | 正規表現は形式チェックのみ。論理・実在チェックはコード/ライブラリで |

### 設計原則の再確認

```
┌──────────────────────────────────────────────────────────────┐
│                正規表現パターン設計の5原則                      │
│                                                              │
│  1. 必要十分の原則                                            │
│     過度に厳密にせず、実用上十分なレベルに留める                │
│     正規表現の仕事は「明らかに不正な入力を弾くこと」           │
│                                                              │
│  2. 多層防御の原則                                            │
│     正規表現 → ロジック → 外部検証 の3段階で検証               │
│     1つのレイヤーに全責任を負わせない                          │
│                                                              │
│  3. テスタビリティの原則                                      │
│     パターンには必ずテストケースを用意する                      │
│     正常系・異常系・境界値を網羅する                           │
│                                                              │
│  4. 保守性の原則                                              │
│     読めないパターンは使わない                                 │
│     コメントや名前付きグループで意図を明示する                  │
│                                                              │
│  5. 安全性の原則                                              │
│     ReDoS のリスクを考慮する                                   │
│     ユーザー入力に対しては入力長制限を設ける                    │
└──────────────────────────────────────────────────────────────┘
```

## 次に読むべきガイド

- [02-text-processing.md](./02-text-processing.md) -- テキスト処理(sed/awk/grep)
- [03-regex-alternatives.md](./03-regex-alternatives.md) -- 正規表現の代替手法

## 参考文献

1. **RFC 5322** "Internet Message Format" https://tools.ietf.org/html/rfc5322 -- メールアドレスの公式仕様
2. **RFC 3986** "Uniform Resource Identifier (URI): Generic Syntax" https://tools.ietf.org/html/rfc3986 -- URI の公式仕様
3. **Google libphonenumber** https://github.com/google/libphonenumber -- 電話番号検証の事実上の標準ライブラリ
4. **OWASP Input Validation Cheat Sheet** https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html -- 入力検証のセキュリティベストプラクティス
5. **Regular Expression Denial of Service (ReDoS)** https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS -- ReDoS 攻撃の解説と対策
6. **Python re module documentation** https://docs.python.org/3/library/re.html -- Python 標準ライブラリの正規表現リファレンス
7. **regex101.com** https://regex101.com/ -- 正規表現のオンラインテスト・デバッグツール
