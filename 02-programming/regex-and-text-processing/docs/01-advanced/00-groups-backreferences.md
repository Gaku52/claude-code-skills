# グループ・後方参照 -- キャプチャ、名前付きグループ、先読み・後読み

> グループ化はパターンの部分式をまとめ、後方参照はマッチした部分文字列を再利用する。キャプチャグループ、非キャプチャグループ、名前付きグループの使い分けを正確に理解し、置換・抽出・検証で活用する。さらに先読み・後読み（lookahead/lookbehind）、アトミックグループ、条件分岐パターンといった高度なグループ構文も網羅する。

## この章で学ぶこと

1. **キャプチャグループと非キャプチャグループ** -- `(...)` と `(?:...)` の違い、パフォーマンスへの影響
2. **名前付きグループ** -- `(?P<name>...)` による可読性の高いパターン設計
3. **後方参照と置換** -- `\1`、`\k<name>` によるマッチ結果の再利用
4. **先読みと後読み（Lookaround）** -- `(?=...)`, `(?!...)`, `(?<=...)`, `(?<!...)` によるゼロ幅アサーション
5. **アトミックグループ** -- `(?>...)` によるバックトラック抑制と性能最適化
6. **条件分岐パターン** -- `(?(id)yes|no)` による条件付きマッチング

---

## 1. キャプチャグループ `(...)`

### 1.1 基本的な使い方

```python
import re

# 日付パターンから年月日を個別に抽出
pattern = r'(\d{4})-(\d{2})-(\d{2})'
text = "今日は 2026-02-11 です"

match = re.search(pattern, text)
print(match.group(0))  # => '2026-02-11' (マッチ全体)
print(match.group(1))  # => '2026'       (グループ1: 年)
print(match.group(2))  # => '02'         (グループ2: 月)
print(match.group(3))  # => '11'         (グループ3: 日)
print(match.groups())  # => ('2026', '02', '11')
```

### 1.2 グループの番号付け

```
パターン: ((A)(B(C)))

グループ番号の割り当て(左括弧の出現順):

  (  (  A  )  (  B  (  C  )  )  )
  ↑  ↑        ↑     ↑
  1  2        3     4

  グループ0: 全体のマッチ
  グループ1: ((A)(B(C)))  = "ABC"
  グループ2: (A)          = "A"
  グループ3: (B(C))       = "BC"
  グループ4: (C)          = "C"
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

### 1.3 グループと選択の組み合わせ

```python
import re

# グループ内で選択 (|)
pattern = r'(cat|dog|bird)s?'
text = "I have 2 cats and 3 dogs"

matches = re.findall(pattern, text)
print(matches)  # => ['cat', 'dog']
# 注意: findall はグループがあるとグループの中身を返す

# グループの中身ではなく全体マッチが必要な場合
matches_full = re.finditer(pattern, text)
for m in matches_full:
    print(f"  全体: {m.group(0)}, グループ1: {m.group(1)}")
# => 全体: cats, グループ1: cat
# => 全体: dogs, グループ1: dog
```

### 1.4 ネストされたグループの実践例

```python
import re

# HTML属性の抽出: class="value" または class='value'
html = '<div class="main container" id="app" data-role=\'admin\'>'
pattern = r'(\w+)=((["\'])(.*?)\3)'

for m in re.finditer(pattern, html):
    print(f"  属性名: {m.group(1)}, 値: {m.group(4)}, 引用符: {m.group(3)}")
# => 属性名: class, 値: main container, 引用符: "
# => 属性名: id, 値: app, 引用符: "
# => 属性名: data-role, 値: admin, 引用符: '
```

### 1.5 複数グループの同時使用パターン

```python
import re

# ログ解析: タイムスタンプ、レベル、メッセージを一括抽出
log_pattern = re.compile(
    r'\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]\s+'  # グループ1: タイムスタンプ
    r'\[(DEBUG|INFO|WARN|ERROR|FATAL)\]\s+'              # グループ2: ログレベル
    r'\[(\w+)\]\s+'                                      # グループ3: モジュール名
    r'(.*)'                                              # グループ4: メッセージ
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

### 1.6 JavaScript でのキャプチャグループ

```javascript
// JavaScript でのキャプチャグループ
const text = "2026-02-11 Error at 10:30:45";
const pattern = /(\d{4})-(\d{2})-(\d{2})/;

const match = text.match(pattern);
console.log(match[0]);  // '2026-02-11' (全体)
console.log(match[1]);  // '2026' (グループ1)
console.log(match[2]);  // '02' (グループ2)
console.log(match[3]);  // '11' (グループ3)

// matchAll を使った全グループの取得 (ES2020)
const logPattern = /\[(\w+)\]\s+(\w+)/g;
const logText = "[ERROR] AuthFailed [WARN] HighLoad";

for (const m of logText.matchAll(logPattern)) {
    console.log(`Level: ${m[1]}, Message: ${m[2]}`);
}
// => Level: ERROR, Message: AuthFailed
// => Level: WARN, Message: HighLoad
```

### 1.7 Go でのキャプチャグループ

```go
package main

import (
    "fmt"
    "regexp"
)

func main() {
    text := "2026-02-11 Error at 10:30:45"
    re := regexp.MustCompile(`(\d{4})-(\d{2})-(\d{2})`)

    // FindStringSubmatch: サブマッチ付きで返す
    match := re.FindStringSubmatch(text)
    if match != nil {
        fmt.Printf("全体: %s, 年: %s, 月: %s, 日: %s\n",
            match[0], match[1], match[2], match[3])
    }

    // FindAllStringSubmatch: 全マッチのサブマッチ
    allMatches := re.FindAllStringSubmatch(text, -1)
    for _, m := range allMatches {
        fmt.Printf("  マッチ: %v\n", m)
    }
}
```

### 1.8 Rust でのキャプチャグループ

```rust
use regex::Regex;

fn main() {
    let text = "2026-02-11 Error at 10:30:45";
    let re = Regex::new(r"(\d{4})-(\d{2})-(\d{2})").unwrap();

    // captures: キャプチャグループ付きマッチ
    if let Some(caps) = re.captures(text) {
        println!("全体: {}", &caps[0]);
        println!("年: {}", &caps[1]);
        println!("月: {}", &caps[2]);
        println!("日: {}", &caps[3]);
    }

    // captures_iter: 全マッチのキャプチャ
    for caps in re.captures_iter(text) {
        println!("  マッチ: {}", &caps[0]);
    }
}
```

---

## 2. 非キャプチャグループ `(?:...)`

### 2.1 キャプチャ不要なグループ化

```python
import re

# キャプチャグループ -- グループ番号が割り当てられる
pattern_capture = r'(https?)://([\w.]+)'
match = re.search(pattern_capture, "https://example.com")
print(match.group(1))  # => 'https'
print(match.group(2))  # => 'example.com'

# 非キャプチャグループ -- グループ番号が割り当てられない
pattern_noncapture = r'(?:https?)://([\w.]+)'
match = re.search(pattern_noncapture, "https://example.com")
print(match.group(1))  # => 'example.com' (番号がずれない)
# match.group(2) → エラー (グループ2は存在しない)
```

### 2.2 使い分けの基準

```
キャプチャグループを使う場面:
  ✓ マッチした部分文字列を後で使いたい(抽出)
  ✓ 後方参照が必要 (\1, \2)
  ✓ 置換で参照したい (\1 や $1)

非キャプチャグループを使う場面:
  ✓ 量指定子や選択のためにグループ化が必要だが値は不要
  ✓ パフォーマンスを少しでも上げたい
  ✓ グループ番号をずらしたくない
```

```python
import re

# NG: 不要なキャプチャ -- グループ番号が無駄に増える
pattern_bad = r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{2}) (Jan|Feb|Mar) (\d{4})'
# グループ: 1=曜日, 2=日, 3=月, 4=年
# 曜日(グループ1)を使わない場合、番号が無駄にずれる

# OK: 不要な部分は非キャプチャ
pattern_good = r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{2}) (Jan|Feb|Mar) (\d{4})'
# グループ: 1=日, 2=月, 3=年 -- 必要なものだけ番号が付く
```

### 2.3 パフォーマンス測定

```python
import re
import timeit

text = "The quick brown fox jumps over the lazy dog" * 1000

# キャプチャグループ版
pattern_capture = re.compile(r'(\w+)\s+(\w+)\s+(\w+)\s+(\w+)')

# 非キャプチャグループ版
pattern_noncapture = re.compile(r'(?:\w+)\s+(?:\w+)\s+(?:\w+)\s+(?:\w+)')

# 測定
t_capture = timeit.timeit(
    lambda: pattern_capture.findall(text), number=1000
)
t_noncapture = timeit.timeit(
    lambda: pattern_noncapture.findall(text), number=1000
)

print(f"キャプチャ版:    {t_capture:.4f}s")
print(f"非キャプチャ版:  {t_noncapture:.4f}s")
print(f"速度差: {(t_capture - t_noncapture) / t_capture * 100:.1f}%")
# 非キャプチャ版は通常 5-15% 高速
```

### 2.4 複雑なパターンでの非キャプチャグループ活用

```python
import re

# 日時パターン: 非キャプチャで構造化しつつ、必要な部分だけキャプチャ
datetime_pattern = re.compile(r'''
    (?P<date>                          # 日付全体(名前付きキャプチャ)
        (?P<year>\d{4})                # 年(名前付きキャプチャ)
        [-/]                           # セパレータ(グループ化不要)
        (?P<month>\d{2})               # 月(名前付きキャプチャ)
        [-/]                           # セパレータ
        (?P<day>\d{2})                 # 日(名前付きキャプチャ)
    )
    (?:\s+|T)                          # 日付と時刻の区切り(非キャプチャ)
    (?P<time>                          # 時刻全体(名前付きキャプチャ)
        (?P<hour>\d{2})                # 時(名前付きキャプチャ)
        :(?P<minute>\d{2})             # 分(名前付きキャプチャ)
        (?::(?P<second>\d{2}))?        # 秒(任意、名前付きキャプチャ)
    )
    (?:\s*(?P<tz>[A-Z]{3,4}|[+-]\d{2}:?\d{2}))?  # タイムゾーン(任意)
''', re.VERBOSE)

test_strings = [
    "2026-02-11 10:30:45 JST",
    "2026/02/11T10:30",
    "2026-02-11 10:30:45+09:00",
]

for s in test_strings:
    m = datetime_pattern.search(s)
    if m:
        print(f"  入力: {s}")
        print(f"    日付: {m.group('date')}, 時刻: {m.group('time')}")
        print(f"    年: {m.group('year')}, 月: {m.group('month')}, 日: {m.group('day')}")
        tz = m.group('tz')
        if tz:
            print(f"    TZ: {tz}")
```

---

## 3. 名前付きグループ

### 3.1 構文(言語別)

```
┌──────────┬──────────────────────┬──────────────────┐
│ 言語      │ 定義                  │ 参照              │
├──────────┼──────────────────────┼──────────────────┤
│ Python   │ (?P<name>...)        │ (?P=name), \g<name>│
│ Perl     │ (?<name>...)         │ \k<name>          │
│ Java     │ (?<name>...)         │ \k<name>          │
│ .NET     │ (?<name>...)         │ \k<name>          │
│ JavaScript│ (?<name>...)        │ \k<name>          │
│ Go (RE2) │ (?P<name>...)        │ (後方参照なし)     │
│ Rust     │ (?P<name>...)        │ (後方参照なし)     │
└──────────┴──────────────────────┴──────────────────┘
```

### 3.2 Python での名前付きグループ

```python
import re

# 名前付きグループで日付をパース
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
text = "Date: 2026-02-11"

match = re.search(pattern, text)

# 名前でアクセス
print(match.group('year'))   # => '2026'
print(match.group('month'))  # => '02'
print(match.group('day'))    # => '11'

# groupdict() で辞書として取得
print(match.groupdict())
# => {'year': '2026', 'month': '02', 'day': '11'}

# 番号でもアクセス可能
print(match.group(1))  # => '2026'
```

### 3.3 JavaScript での名前付きグループ (ES2018+)

```javascript
// JavaScript の名前付きグループ
const pattern = /(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})/;
const text = "Date: 2026-02-11";

const match = text.match(pattern);
console.log(match.groups);
// => { year: '2026', month: '02', day: '11' }
console.log(match.groups.year);   // => '2026'

// 分割代入との組み合わせ
const { year, month, day } = match.groups;
console.log(`${year}年${month}月${day}日`);
// => '2026年02月11日'

// replace での名前付きグループ参照
const result = "2026-02-11".replace(
    /(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})/,
    '$<day>/$<month>/$<year>'
);
console.log(result);  // => '11/02/2026'

// String.prototype.replaceAll + 名前付きグループ (ES2021)
const multiDates = "Start: 2026-02-11, End: 2026-03-15";
const formatted = multiDates.replaceAll(
    /(?<y>\d{4})-(?<m>\d{2})-(?<d>\d{2})/g,
    '$<d>/$<m>/$<y>'
);
console.log(formatted);
// => 'Start: 11/02/2026, End: 15/03/2026'
```

### 3.4 Go での名前付きグループ

```go
package main

import (
    "fmt"
    "regexp"
)

func main() {
    // Go での名前付きグループ: (?P<name>...)
    re := regexp.MustCompile(`(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})`)
    text := "Date: 2026-02-11"

    match := re.FindStringSubmatch(text)
    if match == nil {
        return
    }

    // SubexpNames() で名前を取得
    result := make(map[string]string)
    for i, name := range re.SubexpNames() {
        if i != 0 && name != "" {
            result[name] = match[i]
        }
    }

    fmt.Printf("年: %s, 月: %s, 日: %s\n",
        result["year"], result["month"], result["day"])
    // => 年: 2026, 月: 02, 日: 11

    // ヘルパー関数として整理
    fmt.Println(extractNamedGroups(re, text))
}

// 汎用ヘルパー関数
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

### 3.5 Rust での名前付きグループ

```rust
use regex::Regex;
use std::collections::HashMap;

fn main() {
    let re = Regex::new(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})").unwrap();
    let text = "Date: 2026-02-11";

    if let Some(caps) = re.captures(text) {
        // 名前でアクセス
        println!("年: {}", &caps["year"]);    // => 年: 2026
        println!("月: {}", &caps["month"]);   // => 月: 02
        println!("日: {}", &caps["day"]);     // => 日: 11

        // name() メソッドで Option<Match> を取得
        if let Some(year) = caps.name("year") {
            println!("年の位置: {}-{}", year.start(), year.end());
        }
    }

    // 全マッチから名前付きグループを HashMap に変換
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

### 3.6 名前付きグループの命名規則

```
推奨する命名規則:
┌────────────────────────────────────────────────────────┐
│ ✓ 英小文字 + アンダースコア (snake_case)                 │
│   例: (?P<first_name>...) (?P<area_code>...)           │
│                                                        │
│ ✓ 意味が明確な名前                                      │
│   例: (?P<protocol>https?) (?P<port>\d{1,5})           │
│                                                        │
│ ✗ 避けるべき:                                           │
│   - 短すぎる名前: (?P<p>...) (?P<x>...)                │
│   - 番号風の名前: (?P<group1>...) (?P<g2>...)          │
│   - ハイフン入り: (?P<first-name>...) → 構文エラー      │
│   - 予約語風: (?P<class>...) (?P<type>...)             │
└────────────────────────────────────────────────────────┘
```

---

## 4. 後方参照(Backreference)

### 4.1 パターン内での後方参照

```python
import re

# \1 でグループ1のマッチを参照
# 同じ文字列の繰り返しを検出

# HTML開始タグと終了タグの対応
pattern = r'<(\w+)>.*?</\1>'
text = '<div>hello</div> <span>world</span>'

matches = re.findall(pattern, text)
print(matches)  # => ['div', 'span']

# 重複単語の検出
pattern = r'\b(\w+)\s+\1\b'
text = "the the quick brown fox fox"
print(re.findall(pattern, text))  # => ['the', 'fox']
```

### 4.2 名前付き後方参照

```python
import re

# (?P=name) で名前付きグループを後方参照
pattern = r'(?P<quote>["\']).*?(?P=quote)'
text = """He said "hello" and 'world'"""

matches = re.findall(pattern, text)
print(matches)  # => ['"', "'"]

# finditer で全体を取得
for m in re.finditer(pattern, text):
    print(m.group())
# => "hello"
# => 'world'
```

### 4.3 置換での後方参照

```python
import re

# \1 または \g<1> で置換時にグループを参照
text = "2026-02-11"

# 日付形式の変換: YYYY-MM-DD → DD/MM/YYYY
result = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3/\2/\1', text)
print(result)  # => '11/02/2026'

# 名前付きグループで置換
result = re.sub(
    r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})',
    r'\g<day>/\g<month>/\g<year>',
    text
)
print(result)  # => '11/02/2026'

# 関数を使った高度な置換
def format_date(match):
    y, m, d = match.group('year'), match.group('month'), match.group('day')
    return f"{y}年{m}月{d}日"

result = re.sub(
    r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})',
    format_date,
    text
)
print(result)  # => '2026年02月11日'
```

### 4.4 後方参照の応用パターン

```python
import re

# 1. 回文検出（3〜5文字の回文）
palindrome_3 = r'\b(\w)(\w)\2\1\b'  # 4文字の回文
text = "abba deed noon hello"
for m in re.finditer(palindrome_3, text):
    print(f"  回文: {m.group()}")
# => 回文: abba
# => 回文: deed
# => 回文: noon

# 2. XML/HTML の対応タグ検出（ネストなし）
xml_tag = r'<(?P<tag>\w+)(?:\s[^>]*)?>(?P<content>.*?)</(?P=tag)>'
html = '<p class="intro">Hello</p> <div>World</div>'
for m in re.finditer(xml_tag, html):
    print(f"  タグ: {m.group('tag')}, 内容: {m.group('content')}")

# 3. CSV の引用符付きフィールド
csv_field = r'(?P<quote>["\'])(?P<value>(?:(?!(?P=quote)).)*|(?:(?P=quote){2})*)(?P=quote)'
csv_line = '"hello","world","it""s a test"'
for m in re.finditer(csv_field, csv_line):
    print(f"  値: {m.group('value')}")

# 4. 同じ文字の繰り返し検出（パスワードチェック等）
repeated_char = r'(.)\1{2,}'  # 同じ文字が3回以上連続
passwords = ["abc", "aabbc", "aaabbb", "password111"]
for pwd in passwords:
    m = re.search(repeated_char, pwd)
    if m:
        print(f"  NG: {pwd} ('{m.group(1)}' が {len(m.group())} 回連続)")
```

### 4.5 JavaScript での後方参照

```javascript
// JavaScript での後方参照

// 1. パターン内後方参照
const html = '<div>hello</div> <span>world</span>';
const tagPattern = /<(\w+)>.*?<\/\1>/g;
let m;
while ((m = tagPattern.exec(html)) !== null) {
    console.log(`  マッチ: ${m[0]}, タグ: ${m[1]}`);
}

// 2. 名前付き後方参照 (ES2018)
const quotePattern = /(?<q>["']).*?\k<q>/g;
const text = `He said "hello" and 'world'`;
for (const match of text.matchAll(quotePattern)) {
    console.log(`  マッチ: ${match[0]}`);
}

// 3. 置換での後方参照
const date = "2026-02-11";
console.log(date.replace(
    /(\d{4})-(\d{2})-(\d{2})/,
    '$3/$2/$1'
));  // => '11/02/2026'

// 名前付きグループでの置換
console.log(date.replace(
    /(?<y>\d{4})-(?<m>\d{2})-(?<d>\d{2})/,
    '$<d>/$<m>/$<y>'
));  // => '11/02/2026'
```

---

## 5. 先読みと後読み（Lookaround）

先読み・後読みはゼロ幅アサーションと呼ばれ、文字を消費せずに条件をチェックする。

### 5.1 先読み・後読みの構文一覧

```
┌────────────────────┬──────────────┬──────────────────────────────┐
│ 種類                │ 構文          │ 意味                          │
├────────────────────┼──────────────┼──────────────────────────────┤
│ 肯定先読み          │ (?=...)      │ 後に...が続く位置にマッチ      │
│ 否定先読み          │ (?!...)      │ 後に...が続かない位置にマッチ  │
│ 肯定後読み          │ (?<=...)     │ 前に...がある位置にマッチ      │
│ 否定後読み          │ (?<!...)     │ 前に...がない位置にマッチ      │
└────────────────────┴──────────────┴──────────────────────────────┘

注意: Go (RE2) と Rust (regex) は先読み・後読みを「サポートしない」
      → fancy-regex (Rust) や regexp2 (Go) で代替可能
```

### 5.2 肯定先読み `(?=...)`

```python
import re

# 「の後に特定パターンが続く」位置をマッチ

# 1. 金額の前の通貨記号を検出（数字は消費しない）
text = "$100 €200 ¥300"
pattern = r'[$€¥](?=\d+)'
for m in re.finditer(pattern, text):
    print(f"  通貨記号: {m.group()} at {m.start()}")

# 2. パスワード強度チェック: 複数条件の同時検査
# 先読みを使うと、パターン全体を消費せずに複数条件をチェックできる
password_pattern = re.compile(r'''
    ^
    (?=.*[A-Z])        # 大文字を含む
    (?=.*[a-z])        # 小文字を含む
    (?=.*\d)           # 数字を含む
    (?=.*[!@#$%^&*])   # 記号を含む
    .{8,}              # 8文字以上
    $
''', re.VERBOSE)

passwords = ["MyP@ss1", "MyP@ssw0rd", "password", "PASSWORD1!", "Ab1!abcd"]
for pwd in passwords:
    result = "OK" if password_pattern.match(pwd) else "NG"
    print(f"  {result}: {pwd}")

# 3. 3桁ごとのカンマ区切り
def add_commas(number_str):
    """先読みを使って3桁ごとにカンマを挿入"""
    return re.sub(r'(?<=\d)(?=(\d{3})+(?!\d))', ',', number_str)

print(add_commas("1234567890"))   # => '1,234,567,890'
print(add_commas("12345"))        # => '12,345'
print(add_commas("123"))          # => '123'
```

### 5.3 否定先読み `(?!...)`

```python
import re

# 「の後に特定パターンが続かない」位置をマッチ

# 1. 拡張子が .exe でないファイル名
filenames = ["report.pdf", "virus.exe", "photo.jpg", "setup.exe", "data.csv"]
pattern = r'\w+\.(?!exe\b)\w+'
for f in filenames:
    m = re.fullmatch(pattern, f)
    if m:
        print(f"  安全: {f}")
# => 安全: report.pdf
# => 安全: photo.jpg
# => 安全: data.csv

# 2. 予約語でない識別子
reserved = "if|else|for|while|return|class|def"
identifier_pattern = re.compile(rf'\b(?!(?:{reserved})\b)[a-zA-Z_]\w*\b')
code = "if x > 0: return calculate(x) else: count = 0"
identifiers = identifier_pattern.findall(code)
print(f"  識別子: {identifiers}")
# => 識別子: ['x', 'calculate', 'x', 'count']

# 3. 特定のドメインを除外したURL抽出
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
        print(f"  許可: {url}")
```

### 5.4 肯定後読み `(?<=...)`

```python
import re

# 「の前に特定パターンがある」位置をマッチ

# 1. 通貨記号の後の金額を抽出
text = "$100 €200 ¥300 free"
dollar_amounts = re.findall(r'(?<=\$)\d+', text)
print(f"  ドル金額: {dollar_amounts}")  # => ['100']

# 2. @メンション の抽出
tweet = "Hello @alice and @bob, check out @charlie's work"
mentions = re.findall(r'(?<=@)\w+', tweet)
print(f"  メンション: {mentions}")  # => ['alice', 'bob', 'charlie']

# 3. JSON 値の抽出（キー名の後の値）
json_text = '"name": "Alice", "age": 30, "city": "Tokyo"'
name_value = re.search(r'(?<="name":\s*")[^"]+', json_text)
if name_value:
    print(f"  name の値: {name_value.group()}")  # => Alice
```

### 5.5 否定後読み `(?<!...)`

```python
import re

# 「の前に特定パターンがない」位置をマッチ

# 1. エスケープされていない引用符を検出
text = r'He said \"hello\" and "world"'
# \" は除外して、" だけをマッチ
unescaped_quotes = re.findall(r'(?<!\\)"', text)
print(f"  非エスケープ引用符の数: {len(unescaped_quotes)}")

# 2. http:// ではない（つまりプロトコル付きでない）URL パス
paths = ["/api/users", "http://example.com/api", "/api/items", "https://x.com"]
path_only = re.compile(r'(?<!https?)(?<!:)/\w[\w/]*')

# 3. 行頭のインデントではないスペースを検出
code = "def foo():\n    return bar  + baz"
# 連続する2つ以上のスペース（ただし行頭を除く）
extra_spaces = re.findall(r'(?<=\S)\s{2,}(?=\S)', code)
print(f"  余分なスペース箇所: {len(extra_spaces)}")
```

### 5.6 先読み・後読みの組み合わせ

```python
import re

# 先読みと後読みの組み合わせで、囲まれた部分だけを取得

# 1. HTML タグの中身だけ抽出（タグ自体は含めない）
html = "<b>bold</b> and <i>italic</i>"
content = re.findall(r'(?<=<\w+>).*?(?=</\w+>)', html)
print(f"  タグの中身: {content}")  # => ['bold', 'italic']

# 2. 括弧の中身だけ抽出
text = "関数 foo(x, y) は bar(z) を呼ぶ"
args = re.findall(r'(?<=\()[^)]+(?=\))', text)
print(f"  引数: {args}")  # => ['x, y', 'z']

# 3. キャメルケースの単語境界に空白を挿入
camel = "getUserNameFromDatabase"
result = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', camel)
print(f"  変換: {result}")  # => 'get User Name From Database'

# snake_case に変換
snake = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', camel).lower()
print(f"  snake_case: {snake}")  # => 'get_user_name_from_database'
```

### 5.7 先読み・後読みの言語サポート状況

```
┌────────────────────┬────────┬────────────┬─────┬──────┬──────┐
│ 機能                │ Python │ JavaScript │ Go  │ Rust │ Java │
├────────────────────┼────────┼────────────┼─────┼──────┼──────┤
│ 肯定先読み (?=)     │ OK     │ OK         │ 不可 │ 不可  │ OK   │
│ 否定先読み (?!)     │ OK     │ OK         │ 不可 │ 不可  │ OK   │
│ 肯定後読み(固定長)   │ OK     │ OK         │ 不可 │ 不可  │ OK   │
│ 肯定後読み(可変長)   │ 不可※  │ OK (V8)    │ 不可 │ 不可  │ 不可  │
│ 否定後読み(固定長)   │ OK     │ OK         │ 不可 │ 不可  │ OK   │
│ 否定後読み(可変長)   │ 不可※  │ OK (V8)    │ 不可 │ 不可  │ 不可  │
├────────────────────┼────────┼────────────┼─────┼──────┼──────┤
│ 代替クレート/パッケージ│ regex  │ --       │ regexp2│fancy-│ --   │
│                    │ モジュール│           │      │regex │      │
└────────────────────┴────────┴────────────┴─────┴──────┴──────┘

※ Python の regex モジュール(サードパーティ)では可変長後読みも対応
```

### 5.8 JavaScript(ES2018+) での先読み・後読み

```javascript
// JavaScript (ES2018) は可変長の後読みもサポート

// 肯定後読み (可変長)
const text1 = "USD100 EUR200 JPY3000";
const amounts = text1.match(/(?<=USD|EUR|JPY)\d+/g);
console.log(amounts);  // => ['100', '200', '3000']
// 注意: USD(3文字) と JPY(3文字) は同じ長さだが、
// JavaScript は可変長後読みも許可

// 否定後読み + 肯定先読み の組み合わせ
const code = "let x = 10; const y = 20; var z = 30;";
// const/let で宣言された変数名のみ取得 (var は除外)
const modernVars = code.match(/(?<=(?:const|let)\s+)\w+/g);
console.log(modernVars);  // => ['x', 'y']
```

---

## 6. アトミックグループ `(?>...)`

### 6.1 アトミックグループとは

アトミックグループは、一度マッチした部分についてバックトラックを禁止する。これにより壊滅的なバックトラッキング（catastrophic backtracking）を防止できる。

```
通常のグループ:
  パターン: (a+)b
  入力:     aaac

  試行1: "aaa" をキャプチャ → b が見つからない
  バックトラック: "aa" をキャプチャ → b が見つからない
  バックトラック: "a" をキャプチャ → b が見つからない
  → 失敗（3回のバックトラック）

アトミックグループ:
  パターン: (?>a+)b
  入力:     aaac

  試行1: "aaa" をキャプチャ → b が見つからない
  → 即座に失敗（バックトラックしない）
  → 1回の試行で完了
```

### 6.2 サポート状況

```
┌──────────┬──────────────────────────────────────────────────┐
│ 言語      │ アトミックグループのサポート                       │
├──────────┼──────────────────────────────────────────────────┤
│ Perl     │ (?>...) サポートあり                              │
│ Java 20+ │ (?>...) サポートあり (Java 20 で追加)              │
│ Java <20 │ 非サポート → 独占的量指定子 (*+, ++, ?+) で代替    │
│ .NET     │ (?>...) サポートあり                              │
│ Python   │ re: 非サポート / regex モジュール: サポート         │
│ JavaScript│ 非サポート                                       │
│ Go       │ 非サポート（RE2 ベースのため不要）                  │
│ Rust     │ 非サポート（DFA ベースのため不要）                  │
└──────────┴──────────────────────────────────────────────────┘
```

### 6.3 独占的量指定子との関係

```
アトミックグループと独占的量指定子は等価:

  (?>a+)    ≡  a++     (1回以上、バックトラックなし)
  (?>a*)    ≡  a*+     (0回以上、バックトラックなし)
  (?>a?)    ≡  a?+     (0または1回、バックトラックなし)
  (?>a{2,5}) ≡ a{2,5}+ (2〜5回、バックトラックなし)

独占的量指定子をサポートする言語:
  ✓ Java
  ✓ Perl 5.10+
  ✗ Python (re)
  ✗ JavaScript
  ✗ Go
  ✗ Rust
```

### 6.4 壊滅的バックトラッキングの例と対策

```python
import re
import time

# 危険なパターン: (a+)+ は壊滅的バックトラッキングを引き起こす
dangerous_pattern = re.compile(r'(a+)+b')

# 短い入力: 高速
text_short = "aaaaab"
start = time.time()
dangerous_pattern.search(text_short)
print(f"  短い入力: {time.time() - start:.4f}s")

# マッチしない長い入力: 指数時間
# 注意: 下の行は実行すると非常に遅い（n=25 でも数秒かかる）
# text_long = "a" * 25 + "c"
# dangerous_pattern.search(text_long)  # 危険！数秒〜数分かかる

# 安全な代替パターン
safe_pattern = re.compile(r'a+b')  # グループのネストを排除
# または非キャプチャ + 独占的量指定子（サポートされている言語で）
```

```java
// Java での独占的量指定子による対策
import java.util.regex.*;

public class AtomicExample {
    public static void main(String[] args) {
        // 危険: 壊滅的バックトラッキング
        // Pattern dangerous = Pattern.compile("(a+)+b");

        // 安全: 独占的量指定子
        Pattern safe = Pattern.compile("a++b");

        String input = "a".repeat(30) + "c";
        long start = System.nanoTime();
        safe.matcher(input).find();
        long elapsed = System.nanoTime() - start;
        System.out.printf("  所要時間: %.3f ms%n", elapsed / 1e6);
        // => 即座に完了
    }
}
```

---

## 7. 条件分岐パターン `(?(id)yes|no)`

### 7.1 基本構文

```
(?(id)yes-pattern|no-pattern)

id:           参照するグループ番号または名前
yes-pattern:  グループがマッチした場合に使うパターン
no-pattern:   グループがマッチしなかった場合に使うパターン（省略可）
```

### 7.2 実践例

```python
import re

# 1. 開き括弧があれば閉じ括弧を要求
pattern = r'(\()?hello(?(1)\))'
# (?(1)\)) = グループ1がマッチしていれば \) を要求

print(re.search(pattern, "hello").group())    # => 'hello'
print(re.search(pattern, "(hello)").group())  # => '(hello)'
print(re.search(pattern, "(hello").group())   # => 'hello' (括弧なし版にマッチ)

# 2. 引用符の対応チェック
# 開き引用符があれば同じ閉じ引用符を要求
quote_pattern = r'(?P<q>["\'])?(?P<content>\w+)(?(q)(?P=q))'
test_strings = ['"hello"', "'world'", 'plain', '"mismatch\'']
for s in test_strings:
    m = re.search(quote_pattern, s)
    if m:
        print(f"  {s} → content: {m.group('content')}")

# 3. オプショナルなプレフィックスに応じた形式変更
# +81 があれば国際形式、なければ国内形式
phone_pattern = r'(\+81)?-?(?(1)\d{1,4}-\d{1,4}-\d{4}|0\d{1,4}-\d{1,4}-\d{4})'
phones = ["+81-90-1234-5678", "090-1234-5678", "+81-3-1234-5678", "03-1234-5678"]
for p in phones:
    m = re.match(phone_pattern, p)
    if m:
        print(f"  マッチ: {m.group()}")
```

### 7.3 名前付きグループでの条件分岐

```python
import re

# 名前付きグループを使った条件分岐
# メールアドレスの表示名: "Name <email>" または単独 "email"
pattern = r'(?:(?P<display_name>[^<]+)\s+)?<(?P<email>[^>]+)>(?(display_name)|\s*(?P<email_only>[^\s]+))?'

# より実用的な例: タグ形式またはプレーン形式
# <tag attr="val">content</tag> または プレーンテキスト
tag_or_plain = r'(?P<open><(?P<tagname>\w+)[^>]*>)?(?(open)(?P<content>.*?)</(?P=tagname)>|(?P<plain>.+))'
tests = ["<b>bold text</b>", "plain text", "<a href='url'>link</a>"]
for t in tests:
    m = re.match(tag_or_plain, t)
    if m:
        if m.group('open'):
            print(f"  タグ: {m.group('tagname')}, 内容: {m.group('content')}")
        else:
            print(f"  プレーン: {m.group('plain')}")
```

### 7.4 条件分岐のサポート状況

```
┌──────────┬──────────────────────────────────────┐
│ 言語      │ 条件分岐 (?(id)yes|no) サポート       │
├──────────┼──────────────────────────────────────┤
│ Python   │ OK (re モジュール標準サポート)          │
│ Perl     │ OK                                   │
│ .NET     │ OK                                   │
│ Java     │ 不可                                  │
│ JavaScript│ 不可                                 │
│ Go       │ 不可                                  │
│ Rust     │ 不可                                  │
└──────────┴──────────────────────────────────────┘
```

---

## 8. ASCII 図解

### 8.1 グループのネスト構造

```
パターン: ((\d{4})-(\d{2})-(\d{2}))T((\d{2}):(\d{2}):(\d{2}))

入力:     2026-02-11T10:30:45

グループ構造:
┌─── グループ1: 2026-02-11 ───────────────────────────────────────┐
│ ┌─ グループ2: 2026 ─┐   ┌─ グループ3: 02 ─┐  ┌─ グループ4: 11 ─┐│
│ │    \d{4}           │ - │    \d{2}         │- │    \d{2}         ││
│ │    2026            │   │    02            │  │    11            ││
│ └────────────────────┘   └─────────────────┘  └─────────────────┘│
└──────────────────────────────────────────────────────────────────┘
                           T
┌─── グループ5: 10:30:45 ────────────────────────────────────────┐
│ ┌─ グループ6: 10 ─┐   ┌─ グループ7: 30 ─┐  ┌─ グループ8: 45 ─┐│
│ │    \d{2}         │ : │    \d{2}         │: │    \d{2}         ││
│ │    10            │   │    30            │  │    45            ││
│ └──────────────────┘   └─────────────────┘  └─────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 後方参照の動作

```
パターン: <(\w+)>.*?</\1>
入力:     <div>hello</div>

ステップ1: < にマッチ
ステップ2: (\w+) が "div" をキャプチャ → グループ1 = "div"
ステップ3: > にマッチ
ステップ4: .*? が "hello" をマッチ(怠惰)
ステップ5: </ にマッチ
ステップ6: \1 → グループ1("div")と照合 → "div" にマッチ
ステップ7: > にマッチ

結果: <div>hello</div>

もし入力が <div>hello</span> の場合:
  ステップ6で \1("div") ≠ "span" → 失敗 → バックトラック
```

### 8.3 キャプチャ vs 非キャプチャの内部動作

```
キャプチャグループ (pattern):
┌─────────────────────────────────────────┐
│  パターン: (a)(b)(c)                     │
│                                         │
│  エンジン内部の状態:                      │
│  ┌─────────────────────┐                │
│  │ グループ配列:         │                │
│  │  [0] = "abc" (全体)  │  ← メモリ割当 │
│  │  [1] = "a"           │  ← メモリ割当 │
│  │  [2] = "b"           │  ← メモリ割当 │
│  │  [3] = "c"           │  ← メモリ割当 │
│  └─────────────────────┘                │
└─────────────────────────────────────────┘

非キャプチャグループ (?:pattern):
┌─────────────────────────────────────────┐
│  パターン: (?:a)(b)(?:c)                 │
│                                         │
│  エンジン内部の状態:                      │
│  ┌─────────────────────┐                │
│  │ グループ配列:         │                │
│  │  [0] = "abc" (全体)  │  ← メモリ割当 │
│  │  [1] = "b"           │  ← メモリ割当 │
│  └─────────────────────┘                │
│  → メモリ使用量が少ない                   │
└─────────────────────────────────────────┘
```

### 8.4 先読みの動作フロー

```
パターン: \d+(?=円)
入力:     "100円 200ドル 300円"

┌──────────────────────────────────────────────────────┐
│ 位置 0: "1"                                          │
│   \d+ → "100" にマッチ                               │
│   (?=円) → 次の文字は "円" → 先読み成功!             │
│   → "100" を結果に (※ "円" は消費しない)             │
│                                                      │
│ 位置 5: "2"                                          │
│   \d+ → "200" にマッチ                               │
│   (?=円) → 次の文字は "ド" → 先読み失敗              │
│   → バックトラック → "20" → 失敗 → "2" → 失敗       │
│                                                      │
│ 位置 10: "3"                                         │
│   \d+ → "300" にマッチ                               │
│   (?=円) → 次の文字は "円" → 先読み成功!             │
│   → "300" を結果に                                   │
└──────────────────────────────────────────────────────┘

結果: ["100", "300"]
```

### 8.5 後読みの動作フロー

```
パターン: (?<=\$)\d+
入力:     "$100 €200 $300"

┌──────────────────────────────────────────────────────┐
│ 位置 0: "$"                                          │
│   → 数字でない → スキップ                            │
│                                                      │
│ 位置 1: "1"                                          │
│   (?<=\$) → 前の文字は "$" → 後読み成功!             │
│   \d+ → "100" にマッチ                               │
│   → "100" を結果に                                   │
│                                                      │
│ 位置 5: "€"                                          │
│   → 数字でない → スキップ                            │
│                                                      │
│ 位置 7: "2"                                          │
│   (?<=\$) → 前の文字は "€" → 後読み失敗              │
│                                                      │
│ 位置 12: "3"                                         │
│   (?<=\$) → 前の文字は "$" → 後読み成功!             │
│   \d+ → "300" にマッチ                               │
│   → "300" を結果に                                   │
└──────────────────────────────────────────────────────┘

結果: ["100", "300"]
```

### 8.6 条件分岐の動作フロー

```
パターン: (\()?hello(?(1)\))
入力1:    "(hello)"
入力2:    "hello"

入力1の処理:
┌──────────────────────────────────────────┐
│ (\()? → "(" にマッチ → グループ1 = "("   │
│ hello → "hello" にマッチ                │
│ (?(1)\)) → グループ1は存在する → \) 必要 │
│ \) → ")" にマッチ                        │
│ → 成功: "(hello)"                        │
└──────────────────────────────────────────┘

入力2の処理:
┌──────────────────────────────────────────┐
│ (\()? → マッチなし → グループ1 = なし     │
│ hello → "hello" にマッチ                │
│ (?(1)\)) → グループ1は不在 → 何も不要   │
│ → 成功: "hello"                          │
└──────────────────────────────────────────┘
```

---

## 9. 比較表

### 9.1 グループ種別比較

| 種別 | 構文 | キャプチャ | 番号 | 名前 | 用途 |
|------|------|-----------|------|------|------|
| キャプチャ | `(...)` | あり | あり | なし | 抽出・後方参照 |
| 非キャプチャ | `(?:...)` | なし | なし | なし | グループ化のみ |
| 名前付き | `(?P<n>...)` | あり | あり | あり | 可読性の高い抽出 |
| アトミック | `(?>...)` | なし | なし | なし | バックトラック抑制 |
| 条件付き | `(?(id)yes\|no)` | -- | -- | -- | 条件分岐 |
| 肯定先読み | `(?=...)` | なし | なし | なし | ゼロ幅アサーション |
| 否定先読み | `(?!...)` | なし | なし | なし | ゼロ幅アサーション |
| 肯定後読み | `(?<=...)` | なし | なし | なし | ゼロ幅アサーション |
| 否定後読み | `(?<!...)` | なし | なし | なし | ゼロ幅アサーション |

### 9.2 後方参照の構文(言語別)

| 言語 | パターン内参照 | 置換時参照 | 名前付き参照 |
|------|--------------|-----------|-------------|
| Python | `\1`, `(?P=name)` | `\1`, `\g<1>`, `\g<name>` | `(?P<name>...)` |
| JavaScript | `\1`, `\k<name>` | `$1`, `$<name>` | `(?<name>...)` |
| Java | `\1`, `\k<name>` | `$1`, `${name}` | `(?<name>...)` |
| Perl | `\1`, `\k<name>` | `$1`, `$+{name}` | `(?<name>...)` |
| Go (RE2) | 非サポート | `${1}`, `${name}` | `(?P<name>...)` |
| Rust | 非サポート | `$1`, `$name` | `(?P<name>...)` |

### 9.3 先読み・後読みの制約比較

| 制約 | Python re | JavaScript | Java | Perl | .NET |
|------|----------|------------|------|------|------|
| 後読みの長さ | 固定長のみ | 可変長 OK | 固定長のみ | 可変長 OK | 可変長 OK |
| ネスト可能 | OK | OK | OK | OK | OK |
| 先読み内のキャプチャ | OK | OK | OK | OK | OK |
| 後読み内のキャプチャ | OK | OK | OK | OK | OK |
| 先読み内の量指定子 | OK | OK | OK | OK | OK |

### 9.4 グループ機能の総合比較表

| 機能 | Python re | Python regex | JavaScript | Java | Go | Rust |
|------|----------|-------------|------------|------|----|------|
| キャプチャ `()` | OK | OK | OK | OK | OK | OK |
| 非キャプチャ `(?:)` | OK | OK | OK | OK | OK | OK |
| 名前付き | `(?P<>)` | `(?P<>)` | `(?<>)` | `(?<>)` | `(?P<>)` | `(?P<>)` |
| 後方参照 | OK | OK | OK | OK | 不可 | 不可 |
| 先読み | OK | OK | OK | OK | 不可 | 不可 |
| 後読み | 固定長 | 可変長 | 可変長 | 固定長 | 不可 | 不可 |
| アトミック | 不可 | OK | 不可 | Java 20+ | 不可 | 不可 |
| 条件分岐 | OK | OK | 不可 | 不可 | 不可 | 不可 |
| ブランチリセット | 不可 | `(?|)` | 不可 | 不可 | 不可 | 不可 |

---

## 10. アンチパターン

### 10.1 アンチパターン: 過剰なキャプチャグループ

```python
import re

# NG: 全てをキャプチャグループにする
pattern_bad = r'(https?)://(www\.)?(\w+)\.(\w+)/(\w+)/(\w+)'
# グループが6個 → 番号の管理が困難

# OK: 必要な部分のみキャプチャ + 名前付き
pattern_good = r'(?:https?)://(?:www\.)?(?P<domain>\w+\.\w+)/(?P<path>\w+/\w+)'

text = "https://www.example.com/api/users"
match = re.search(pattern_good, text)
if match:
    print(match.group('domain'))  # => 'example.com'
    print(match.group('path'))    # => 'api/users'
```

### 10.2 アンチパターン: 後方参照のグループ番号ハードコード

```python
import re

# NG: グループ番号のハードコード -- パターン変更時にバグを生む
pattern = r'(\w+)\s+(\d+)\s+(\w+)'
text = "item 42 completed"
match = re.search(pattern, text)
name = match.group(1)    # パターン変更でずれる
count = match.group(2)   # パターン変更でずれる

# OK: 名前付きグループ -- パターン変更に強い
pattern = r'(?P<name>\w+)\s+(?P<count>\d+)\s+(?P<status>\w+)'
match = re.search(pattern, text)
name = match.group('name')      # 名前で参照 → ずれない
count = match.group('count')    # 名前で参照 → ずれない
```

### 10.3 アンチパターン: 先読みの過剰使用

```python
import re

# NG: 先読みだけでパスワードを検証（理解困難）
password_bad = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*])(?=.{8,})(?!.*(.)\1{2,})(?!.*(?:123|abc|password)).*$'
# → 一行に詰め込みすぎて保守不能

# OK: VERBOSE フラグ + コメントで構造化
password_good = re.compile(r'''
    ^
    (?=.*[A-Z])            # 条件1: 大文字を含む
    (?=.*[a-z])            # 条件2: 小文字を含む
    (?=.*\d)               # 条件3: 数字を含む
    (?=.*[!@#$%^&*])       # 条件4: 記号を含む
    (?!.*(.)\1{2,})        # 条件5: 同じ文字の3連続を禁止
    (?!.*(?:123|abc|pwd))  # 条件6: 弱いパターンを禁止
    .{8,}                  # 本体: 8文字以上
    $
''', re.VERBOSE)

# さらに良い方法: 個別チェック関数
def validate_password(pwd: str) -> list[str]:
    """パスワード検証: 各条件を個別にチェックし、失敗理由を返す"""
    errors = []
    if len(pwd) < 8:
        errors.append("8文字以上必要")
    if not re.search(r'[A-Z]', pwd):
        errors.append("大文字を含めてください")
    if not re.search(r'[a-z]', pwd):
        errors.append("小文字を含めてください")
    if not re.search(r'\d', pwd):
        errors.append("数字を含めてください")
    if not re.search(r'[!@#$%^&*]', pwd):
        errors.append("記号を含めてください")
    if re.search(r'(.)\1{2,}', pwd):
        errors.append("同じ文字を3回以上連続させないでください")
    return errors
```

### 10.4 アンチパターン: Go/Rust で先読み・後読みを使おうとする

```go
// NG: Go では先読みは使えない
// re := regexp.MustCompile(`\d+(?=円)`)  // パニック!

// OK: 代替手法 - キャプチャグループで取得して後処理
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
        fmt.Printf("  金額: %s\n", m[1])
    }
    // => 金額: 100
    // => 金額: 300
}
```

```rust
// NG: Rust の regex クレートでは先読みは使えない
// let re = Regex::new(r"\d+(?=円)").unwrap();  // エラー!

// OK1: キャプチャグループで代替
use regex::Regex;

fn main() {
    let re = Regex::new(r"(\d+)円").unwrap();
    let text = "100円 200ドル 300円";

    for caps in re.captures_iter(text) {
        println!("  金額: {}", &caps[1]);
    }

    // OK2: fancy-regex を使う（先読み・後読み対応）
    // use fancy_regex::Regex;
    // let re = Regex::new(r"\d+(?=円)").unwrap();
}
```

---

## 11. ベストプラクティス

### 11.1 グループ設計の原則

```
┌────────────────────────────────────────────────────────────────┐
│ 原則1: 最小キャプチャの原則                                      │
│   必要な部分だけキャプチャし、それ以外は (?:...) を使う          │
│                                                                │
│ 原則2: 名前付きグループ優先                                      │
│   3個以上のグループがある場合は名前付きを使う                    │
│                                                                │
│ 原則3: VERBOSE モードの活用                                      │
│   複雑なパターンは re.VERBOSE + コメントで構造化                 │
│                                                                │
│ 原則4: テスト駆動パターン設計                                    │
│   パターンを書く前に、マッチすべき/しないケースを列挙する        │
│                                                                │
│ 原則5: 先読み・後読みの使いどころ                                │
│   - 複数条件の同時チェック（パスワード等）                       │
│   - 文字列を消費せずに位置を特定（カンマ挿入等）                │
│   - 前後の文脈に依存したマッチ（通貨記号の後の金額等）           │
└────────────────────────────────────────────────────────────────┘
```

### 11.2 可読性を高めるパターン設計

```python
import re

# NG: 一行の読みにくいパターン
bad = r'((?:\+81|0)[\d-]{9,13})|(\d{3}-?\d{4})|([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'

# OK: VERBOSE + 名前付きグループ + コメント
good = re.compile(r'''
    (?P<phone>                              # 電話番号
        (?:\+81|0)                          #   国番号または先頭の0
        [\d-]{9,13}                         #   数字とハイフン
    )
    |
    (?P<postal>                             # 郵便番号
        \d{3}-?\d{4}                        #   XXX-XXXX形式
    )
    |
    (?P<email>                              # メールアドレス
        [a-zA-Z0-9._%+-]+                   #   ローカルパート
        @                                   #   @
        [a-zA-Z0-9.-]+                      #   ドメイン
        \.[a-zA-Z]{2,}                      #   TLD
    )
''', re.VERBOSE)

text = "連絡先: 090-1234-5678, 〒100-0001, info@example.com"
for m in good.finditer(text):
    for name in ['phone', 'postal', 'email']:
        if m.group(name):
            print(f"  {name}: {m.group(name)}")
```

### 11.3 パフォーマンスを意識したパターン設計

```python
import re

# 1. 先読みは必要な場所だけに限定する
# NG: 無意味な先読み
bad1 = r'(?=\d)\d+'  # 先読みの後に同じパターン → 無意味

# OK: 先読みなしで同じ結果
good1 = r'\d+'

# 2. 非キャプチャグループでバックトラックを抑制
# NG: ネストしたキャプチャグループ + 量指定子
bad2 = r'((\w+)\s*)+'  # 壊滅的バックトラッキングのリスク

# OK: フラットな構造
good2 = r'\w+(?:\s+\w+)*'

# 3. 交替(|)の順序を最適化
# 短い文字列が多い場合、短いパターンを先に
# NG: 長いパターンから
bad3 = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'

# OK: 先頭文字でグループ化（エンジンによっては自動最適化される）
good3 = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
```

---

## 12. FAQ

### Q1: `findall` でグループとマッチ全体の両方を取得するには？

**A**: `findall` はグループがあるとグループの中身を返す。全体マッチも必要な場合は `finditer` を使うか、全体を囲むグループを追加する:

```python
import re

text = "cats and dogs"
# findall + グループ → グループの中身のみ
print(re.findall(r'(cat|dog)s', text))  # => ['cat', 'dog']

# 方法1: finditer を使う
for m in re.finditer(r'(cat|dog)s', text):
    print(f"全体: {m.group(0)}, 動物: {m.group(1)}")

# 方法2: 非キャプチャグループにする
print(re.findall(r'(?:cat|dog)s', text))  # => ['cats', 'dogs']
```

### Q2: 条件付きグループ `(?(id)yes|no)` とは？

**A**: グループがマッチしたかどうかで分岐するパターン:

```python
import re

# 開き括弧があれば閉じ括弧を要求
pattern = r'(\()?hello(?(1)\))'
# (?(1)\)) = グループ1がマッチしていれば \) を要求

print(re.search(pattern, "hello").group())    # => 'hello'
print(re.search(pattern, "(hello)").group())  # => '(hello)'
print(re.search(pattern, "(hello").group())   # => 'hello' (括弧なし版にマッチ)
```

### Q3: 名前付きグループに同じ名前を複数回使えるか？

**A**: 言語による。Python の `re` モジュールでは **不可**。.NET では同名グループをパイプの両側で使える。Python の `regex` モジュール(サードパーティ)では `(?|...)` ブランチリセットグループで同様のことが可能:

```python
# .NET での例(概念):
# (?<digit>\d+)|(?<digit>[a-f]+)  -- 両方とも "digit" グループ

# Python regex モジュールでのブランチリセット:
# (?|(\d+)|([a-f]+))  -- どちらもグループ1
```

### Q4: 先読みの中でキャプチャグループは使えるか？

**A**: 使える。先読み内のキャプチャグループは、先読みが成功した場合にマッチ結果に含まれる:

```python
import re

# 先読み内のキャプチャグループ
pattern = r'(?=(\d+)円)\d+'
text = "100円 200ドル 300円"

for m in re.finditer(pattern, text):
    print(f"  数値: {m.group()}, 先読み内グループ: {m.group(1)}")
# => 数値: 100, 先読み内グループ: 100
# => 数値: 300, 先読み内グループ: 300
```

### Q5: Go や Rust で先読み・後読みが使えない場合の代替手段は？

**A**: 以下の3つの方法がある:

1. **キャプチャグループで代替**: 前後の文脈もキャプチャして後処理で除外
2. **2段階マッチ**: 大まかなパターンでマッチしてから、結果をさらにフィルタ
3. **サードパーティライブラリ**: Go は `regexp2`、Rust は `fancy-regex`

```rust
// Rust: fancy-regex を使った先読みの例
// Cargo.toml: fancy-regex = "0.13"
use fancy_regex::Regex;

fn main() {
    // 肯定先読み: 数値の後に「円」が続くもの
    let re = Regex::new(r"\d+(?=円)").unwrap();
    let text = "100円 200ドル 300円";

    for m in re.find_iter(text) {
        if let Ok(m) = m {
            println!("  金額: {}", m.as_str());
        }
    }
}
```

### Q6: アトミックグループと独占的量指定子はどちらを使うべきか？

**A**: 機能的には等価だが、言語のサポート状況に応じて選択する。Java では独占的量指定子(`a++`)がシンプルに書ける。Perl/.NET ではどちらも使える。パフォーマンスが重要な場合で、NFA エンジンを使う言語（Python, JavaScript, Java）では意識する価値がある。Go/Rust（DFA ベース）では壊滅的バックトラッキングが原理的に発生しないので不要。

---

## まとめ

| 項目 | 内容 |
|------|------|
| `(...)` | キャプチャグループ -- 番号付きで結果を保存 |
| `(?:...)` | 非キャプチャグループ -- グループ化のみ |
| `(?P<name>...)` | 名前付きグループ -- 名前でアクセス可能 |
| `\1`, `\2` | 後方参照 -- パターン内でグループを再利用 |
| `(?P=name)` | 名前付き後方参照(Python) |
| `\g<1>`, `\g<name>` | 置換時のグループ参照(Python) |
| `$1`, `$<name>` | 置換時のグループ参照(JavaScript) |
| `(?=...)`, `(?!...)` | 先読み -- 前方の条件をゼロ幅で検査 |
| `(?<=...)`, `(?<!...)` | 後読み -- 後方の条件をゼロ幅で検査 |
| `(?>...)` | アトミックグループ -- バックトラック抑制 |
| `(?(id)yes\|no)` | 条件分岐 -- グループの有無で分岐 |
| グループ番号 | 左括弧の出現順に1から割り当て |
| 設計指針 | 必要な部分のみキャプチャ、名前付き推奨 |

---

## 13. 演習問題

### 演習1: HTMLタグの属性抽出

以下のHTMLから、全てのタグ名と属性を名前付きグループで抽出する正規表現を書いてください。

```html
<div class="container" id="main">
<img src="photo.jpg" alt="写真">
<a href="https://example.com" target="_blank">リンク</a>
```

期待出力: タグ名、属性名、属性値のリスト

### 演習2: 重複行の検出

テキストファイル内で連続する同一行（空白の違いは無視）を検出する正規表現を書いてください。後方参照を使用すること。

```
入力:
Hello World
Hello World
Foo Bar
  Foo Bar
Baz
```

### 演習3: 先読みを使ったパスワード検証

以下の条件を全て満たすパスワードパターンを先読みで構築してください:
- 10文字以上20文字以下
- 大文字、小文字、数字、記号をそれぞれ1つ以上含む
- 同じ文字の4連続以上を禁止
- 「password」「12345」「qwerty」を含まない（大文字小文字無視）

### 演習4: 条件分岐パターン

電話番号を検証するパターンを条件分岐で構築してください:
- `+81` で始まる場合は国際形式（`+81-XX-XXXX-XXXX`）
- `0` で始まる場合は国内形式（`0XX-XXXX-XXXX`）
- どちらでもない場合は不正

### 演習5: キャメルケースからスネークケースへの変換

先読み・後読みを使って、キャメルケースの識別子をスネークケースに変換する正規表現を書いてください。

```
入力: getUserNameFromDatabase
出力: get_user_name_from_database

入力: XMLParser
出力: xml_parser

入力: getHTTPResponse
出力: get_http_response
```

ヒント: 連続する大文字の扱いに注意が必要です。

### 演習6: ネストされた括弧の最外側マッチ

以下の文字列から、最外側の括弧で囲まれた部分を抽出してください（再帰パターンまたは反復的アプローチで）:

```
入力: "func(a, (b + c), d) + other(x)"
期待: ["func(a, (b + c), d)", "other(x)"]
```

ヒント: 正規表現だけでネストを完全に処理するのは困難です。`.NET` や `Perl` の再帰パターン、または正規表現+プログラムの組み合わせを検討してください。

### 演習7: ログファイルの構造化解析

以下のログ形式を名前付きグループで完全に解析するパターンを構築してください:

```
[2026-02-11T10:30:45.123+09:00] [ERROR] [com.example.auth.LoginService] [req-id=abc123] User login failed: invalid credentials for user "admin" from IP 192.168.1.100
```

抽出すべきフィールド: タイムスタンプ、ログレベル、クラス名、リクエストID、メッセージ本文

---

## 次に読むべきガイド

- [01-lookaround.md](./01-lookaround.md) -- 先読み・後読きの詳細
- [02-unicode-regex.md](./02-unicode-regex.md) -- Unicode 正規表現

## 参考文献

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- 第7章「グループと後方参照」
2. **Python re module - Grouping** https://docs.python.org/3/library/re.html#regular-expression-syntax -- Python 公式のグループ構文リファレンス
3. **TC39 Named Capture Groups Proposal** https://tc39.es/proposal-regexp-named-groups/ -- JavaScript の名前付きキャプチャグループ仕様
4. **MDN Lookahead and Lookbehind** https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions/Assertions -- JavaScript の先読み・後読みドキュメント
5. **RE2 Syntax** https://github.com/google/re2/wiki/Syntax -- Go/Rust で採用されている RE2 エンジンの構文仕様
6. **fancy-regex** https://docs.rs/fancy-regex/ -- Rust で先読み・後読みを使うためのクレート
7. **regex module for Python** https://pypi.org/project/regex/ -- Python の拡張正規表現モジュール
