# グループ・後方参照 -- キャプチャ、名前付きグループ

> グループ化はパターンの部分式をまとめ、後方参照はマッチした部分文字列を再利用する。キャプチャグループ、非キャプチャグループ、名前付きグループの使い分けを正確に理解し、置換・抽出・検証で活用する。

## この章で学ぶこと

1. **キャプチャグループと非キャプチャグループ** -- `(...)` と `(?:...)` の違い、パフォーマンスへの影響
2. **名前付きグループ** -- `(?P<name>...)` による可読性の高いパターン設計
3. **後方参照と置換** -- `\1`、`\k<name>` によるマッチ結果の再利用

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

---

## 5. ASCII 図解

### 5.1 グループのネスト構造

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

### 5.2 後方参照の動作

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

### 5.3 キャプチャ vs 非キャプチャの内部動作

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

---

## 6. 比較表

### 6.1 グループ種別比較

| 種別 | 構文 | キャプチャ | 番号 | 名前 | 用途 |
|------|------|-----------|------|------|------|
| キャプチャ | `(...)` | あり | あり | なし | 抽出・後方参照 |
| 非キャプチャ | `(?:...)` | なし | なし | なし | グループ化のみ |
| 名前付き | `(?P<n>...)` | あり | あり | あり | 可読性の高い抽出 |
| アトミック | `(?>...)` | なし | なし | なし | バックトラック抑制 |
| 条件付き | `(?(id)yes\|no)` | -- | -- | -- | 条件分岐 |

### 6.2 後方参照の構文(言語別)

| 言語 | パターン内参照 | 置換時参照 | 名前付き参照 |
|------|--------------|-----------|-------------|
| Python | `\1`, `(?P=name)` | `\1`, `\g<1>`, `\g<name>` | `(?P<name>...)` |
| JavaScript | `\1`, `\k<name>` | `$1`, `$<name>` | `(?<name>...)` |
| Java | `\1`, `\k<name>` | `$1`, `${name}` | `(?<name>...)` |
| Perl | `\1`, `\k<name>` | `$1`, `$+{name}` | `(?<name>...)` |
| Go (RE2) | 非サポート | `${1}`, `${name}` | `(?P<name>...)` |
| Rust | 非サポート | `$1`, `$name` | `(?P<name>...)` |

---

## 7. アンチパターン

### 7.1 アンチパターン: 過剰なキャプチャグループ

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

### 7.2 アンチパターン: 後方参照のグループ番号ハードコード

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

---

## 8. FAQ

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
| グループ番号 | 左括弧の出現順に1から割り当て |
| 設計指針 | 必要な部分のみキャプチャ、名前付き推奨 |

## 次に読むべきガイド

- [01-lookaround.md](./01-lookaround.md) -- 先読み・後読み
- [02-unicode-regex.md](./02-unicode-regex.md) -- Unicode 正規表現

## 参考文献

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- 第7章「グループと後方参照」
2. **Python re module - Grouping** https://docs.python.org/3/library/re.html#regular-expression-syntax -- Python 公式のグループ構文リファレンス
3. **TC39 Named Capture Groups Proposal** https://tc39.es/proposal-regexp-named-groups/ -- JavaScript の名前付きキャプチャグループ仕様
