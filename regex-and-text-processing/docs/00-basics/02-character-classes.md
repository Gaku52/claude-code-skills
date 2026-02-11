# 文字クラス -- [abc]、\d、\w、\s、POSIX

> 文字クラス(Character Class)は正規表現の核心的機能であり、「この位置にマッチしてよい文字の集合」を定義する。角括弧記法、ショートハンドクラス、POSIX クラスの全体像を解説する。

## この章で学ぶこと

1. **角括弧文字クラス `[...]` の構文と動作** -- 肯定・否定・範囲指定の正確な規則
2. **ショートハンドクラスの意味と言語差異** -- `\d` `\w` `\s` が言語ごとに異なるUnicode対応
3. **POSIX文字クラスと実用的な選択基準** -- `[:alpha:]` `[:digit:]` 等の使いどころ

---

## 1. 角括弧文字クラス `[...]`

### 1.1 基本形

```python
import re

# [abc] -- a, b, c のいずれか1文字にマッチ
print(re.findall(r'[abc]', "abcdef"))
# => ['a', 'b', 'c']

# [aeiou] -- 母音1文字にマッチ
text = "Hello World"
print(re.findall(r'[aeiou]', text, re.IGNORECASE))
# => ['e', 'o', 'o']

# 文字クラス内の文字順序は意味を持たない
# [abc] と [cba] と [bac] は全て同じ
```

### 1.2 範囲指定 `-`

```python
import re

# [a-z]  小文字アルファベット
# [A-Z]  大文字アルファベット
# [0-9]  数字
# [a-zA-Z0-9]  英数字

text = "Item-42: Price $9.99"

# 英字のみ
print(re.findall(r'[a-zA-Z]+', text))
# => ['Item', 'Price']

# 数字のみ
print(re.findall(r'[0-9]+', text))
# => ['42', '9', '99']

# 複数範囲の組み合わせ
print(re.findall(r'[a-zA-Z0-9]+', text))
# => ['Item', '42', 'Price', '9', '99']

# ハイフンをリテラルとして含める場合:
# 先頭: [-abc]  末尾: [abc-]  エスケープ: [a\-c]
print(re.findall(r'[-+*/]', "3+4-2*1/5"))
# => ['+', '-', '*', '/']
```

### 1.3 否定文字クラス `[^...]`

```python
import re

# [^abc] -- a, b, c 以外の1文字にマッチ
print(re.findall(r'[^abc]', "abcdef"))
# => ['d', 'e', 'f']

# [^0-9] -- 数字以外
print(re.findall(r'[^0-9]+', "abc123def456"))
# => ['abc', 'def']

# ^ は先頭にあるときのみ否定の意味
# [a^b] -- a, ^, b のいずれか(^はリテラル)
print(re.findall(r'[a^b]', "a^b"))
# => ['a', '^', 'b']
```

### 1.4 文字クラス内のメタ文字規則

```
角括弧 [...] 内で特殊な意味を持つ文字:

文字   意味              リテラルにする方法
────   ────              ──────────────────
]      クラスの終了      先頭に置く: []abc] またはエスケープ: [\]]
\      エスケープ        エスケープ: [\\]
^      否定(先頭のみ)    先頭以外に置く: [a^b]
-      範囲(文字間のみ)  先頭/末尾に置く: [-abc] [abc-]

角括弧内では . * + ? | ( ) { } はリテラル扱い:
  [.*+?]  → ドット、アスタリスク、プラス、クエスチョンマーク
```

---

## 2. ショートハンドクラス

### 2.1 一覧と等価表現

```
┌──────────┬───────────┬─────────────────────────────────┐
│ショートハンド│  否定形   │  等価な文字クラス (ASCII)          │
├──────────┼───────────┼─────────────────────────────────┤
│ \d       │ \D        │ [0-9]                           │
│ \w       │ \W        │ [a-zA-Z0-9_]                    │
│ \s       │ \S        │ [ \t\n\r\f\v]                   │
│ \b       │ \B        │ (アンカー: 単語境界/非単語境界)    │
└──────────┴───────────┴─────────────────────────────────┘

※ Unicode モードでは範囲が大幅に拡大する(後述)
```

### 2.2 コード例

```python
import re

text = "User: 田中太郎, Age: 25, Email: tanaka@example.com"

# \d -- 数字
print(re.findall(r'\d+', text))
# => ['25']

# \w -- 単語文字 (Python 3 では Unicode 対応)
print(re.findall(r'\w+', text))
# => ['User', '田中太郎', 'Age', '25', 'Email', 'tanaka', 'example', 'com']

# \s -- 空白文字
print(re.split(r'\s+', "hello   world\tfoo\nbar"))
# => ['hello', 'world', 'foo', 'bar']

# \D, \W, \S -- 否定形
print(re.findall(r'\D+', "abc123def456"))
# => ['abc', 'def']
```

### 2.3 Unicode モードでの \w の違い

```python
import re

text = "Hello 世界 café 123"

# Python 3: \w はデフォルトで Unicode 対応
print(re.findall(r'\w+', text))
# => ['Hello', '世界', 'café', '123']

# ASCII モードに限定する場合
print(re.findall(r'\w+', text, re.ASCII))
# => ['Hello', 'caf', '123']  -- 'é' と '世界' がマッチしない
```

```javascript
// JavaScript: u フラグでUnicode対応
const text = "Hello 世界 café 123";

// u フラグなし: \w は ASCII のみ
console.log(text.match(/\w+/g));
// => ['Hello', 'caf', '123']

// Unicode property escape (ES2018+)
console.log(text.match(/[\p{L}\p{N}]+/gu));
// => ['Hello', '世界', 'café', '123']
```

---

## 3. POSIX 文字クラス

### 3.1 一覧

```
┌──────────────┬──────────────────────┬──────────────────┐
│ POSIX クラス  │ 等価表現 (ASCII)      │ 意味              │
├──────────────┼──────────────────────┼──────────────────┤
│ [:alpha:]    │ [a-zA-Z]             │ 英字              │
│ [:digit:]    │ [0-9]                │ 数字              │
│ [:alnum:]    │ [a-zA-Z0-9]          │ 英数字            │
│ [:upper:]    │ [A-Z]                │ 大文字            │
│ [:lower:]    │ [a-z]                │ 小文字            │
│ [:space:]    │ [ \t\n\r\f\v]        │ 空白文字          │
│ [:blank:]    │ [ \t]                │ 空白・タブのみ     │
│ [:punct:]    │ [!"#$%&'()*+,-./:;  │ 句読点            │
│              │  <=>?@[\]^_`{|}~]    │                  │
│ [:print:]    │ [ -~]                │ 印刷可能文字      │
│ [:graph:]    │ [!-~]                │ 印刷可能(空白除く) │
│ [:cntrl:]    │ [\x00-\x1f\x7f]     │ 制御文字          │
│ [:xdigit:]   │ [0-9a-fA-F]          │ 16進数字          │
│ [:ascii:]    │ [\x00-\x7f]          │ ASCII文字         │
└──────────────┴──────────────────────┴──────────────────┘
```

### 3.2 POSIX クラスの使い方

```bash
# POSIX クラスは主に grep, sed, awk で使用
# 注意: 角括弧で囲んで使う [[:alpha:]]

# 英字のみを抽出
echo "Hello 123 World" | grep -oE '[[:alpha:]]+'
# => Hello
# => World

# 数字のみを抽出
echo "Price: $19.99" | grep -oE '[[:digit:]]+'
# => 19
# => 99

# 16進数を抽出
echo "Color: #FF00AA" | grep -oE '[[:xdigit:]]+'
# => FF00AA

# POSIX クラスの否定
echo "abc123" | grep -oE '[^[:digit:]]+'
# => abc
```

### 3.3 POSIX vs ショートハンド 比較表

| 用途 | POSIX | ショートハンド | 使える環境 |
|------|-------|--------------|-----------|
| 数字 | `[[:digit:]]` | `\d` | POSIX: grep/sed/awk、ショートハンド: ほぼ全言語 |
| 英字 | `[[:alpha:]]` | なし(`\p{L}` で代替) | POSIX: UNIX系ツール |
| 英数字 | `[[:alnum:]]` | なし | POSIX: UNIX系ツール |
| 空白 | `[[:space:]]` | `\s` | 両方とも広くサポート |
| 単語文字 | なし | `\w` | ショートハンドのみ |
| 大文字 | `[[:upper:]]` | `\p{Lu}` | POSIX: UNIX、Unicode: Python/Perl |
| 小文字 | `[[:lower:]]` | `\p{Ll}` | POSIX: UNIX、Unicode: Python/Perl |

---

## 4. 組み合わせパターン

### 4.1 文字クラスの組み合わせ

```python
import re

# 英数字とハイフン、アンダースコア
slug_pattern = r'[a-zA-Z0-9_-]+'
print(re.findall(slug_pattern, "my-page_title 2026"))
# => ['my-page_title', '2026']

# 日本語文字 (Unicodeレンジ)
jp_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+'
print(re.findall(jp_pattern, "Hello 東京タワーへ行こう"))
# => ['東京タワーへ行こう']

# 文字クラスの減算 (.NETのみ)
# [a-z-[aeiou]] -- 小文字子音のみ

# 文字クラスの交差 (Java)
# [a-z&&[^aeiou]] -- 小文字子音のみ
```

### 4.2 よくある文字クラスパターン

```python
import re

# ファイル名に使える文字
filename_pattern = r'[a-zA-Z0-9._-]+'
print(re.findall(filename_pattern, "report_2026-02.pdf"))
# => ['report_2026-02.pdf']

# 16進カラーコード
hex_color = r'#[0-9a-fA-F]{6}\b'
print(re.findall(hex_color, "color: #FF5733; bg: #00aaff;"))
# => ['#FF5733', '#00aaff']

# 引用符で囲まれた文字列 (引用符自体を除外)
quoted = r'"[^"]*"'
print(re.findall(quoted, 'name="John" age="25"'))
# => ['"John"', '"25"']

# 制御文字を除外した印刷可能文字
printable = r'[^\x00-\x1f\x7f]+'
print(re.findall(printable, "hello\x00world\x1b[31m"))
# => ['hello', 'world', '[31m']
```

---

## 5. ASCII 図解

### 5.1 文字クラスの概念図

```
文字空間全体 (Unicode: 約15万文字)
┌─────────────────────────────────────────┐
│                                         │
│   [a-z]  ┌─────────┐                   │
│          │a b c ... z│  26文字          │
│          └─────────┘                    │
│                                         │
│   \d     ┌────────────┐                 │
│          │0 1 2 ... 9  │  10文字        │
│          │(Unicode:数百)│                │
│          └────────────┘                 │
│                                         │
│   \w     ┌──────────────────────┐       │
│          │a-z A-Z 0-9 _         │       │
│          │(Unicode: 数万文字)    │       │
│          └──────────────────────┘       │
│                                         │
│   \s     ┌──────────────┐               │
│          │空白 TAB LF CR│  6文字        │
│          │FF VT         │               │
│          └──────────────┘               │
│                                         │
│   [^a-z] = 上記 [a-z] の補集合          │
│   \D     = \d の補集合                   │
│   \W     = \w の補集合                   │
│   \S     = \s の補集合                   │
└─────────────────────────────────────────┘
```

### 5.2 否定文字クラスの動作

```
パターン: [^aeiou]  (母音以外)
テキスト: "regex"

  r → [^aeiou] にマッチ? → 'r'は母音ではない → マッチ
  e → [^aeiou] にマッチ? → 'e'は母音 → 不一致
  g → [^aeiou] にマッチ? → 'g'は母音ではない → マッチ
  e → [^aeiou] にマッチ? → 'e'は母音 → 不一致
  x → [^aeiou] にマッチ? → 'x'は母音ではない → マッチ

結果: r, g, x がマッチ
```

### 5.3 範囲指定のASCIIコード基盤

```
ASCII コードによる範囲:

[0-9]  = ASCII 48-57
  48: '0'  49: '1'  50: '2' ... 57: '9'

[A-Z]  = ASCII 65-90
  65: 'A'  66: 'B'  67: 'C' ... 90: 'Z'

[a-z]  = ASCII 97-122
  97: 'a'  98: 'b'  99: 'c' ... 122: 'z'

注意: [A-z] は意図しない文字を含む!
  65: 'A' ... 90: 'Z'
  91: '['  92: '\'  93: ']'  94: '^'  95: '_'  96: '`'
  97: 'a' ... 122: 'z'

  → [ \ ] ^ _ ` も含まれてしまう!
  → 正しくは [A-Za-z] を使う
```

---

## 6. アンチパターン

### 6.1 アンチパターン: [A-z] を使う

```python
import re

# NG: [A-z] は予期しない文字を含む
pattern_bad = r'[A-z]+'
text = "Hello[World]_test"
print(re.findall(pattern_bad, text))
# => ['Hello[World]_test']  -- [ ] _ もマッチしてしまう!

# OK: [A-Za-z] を使う
pattern_good = r'[A-Za-z]+'
print(re.findall(pattern_good, text))
# => ['Hello', 'World', 'test']
```

### 6.2 アンチパターン: ショートハンドのUnicode挙動を無視する

```python
import re

# NG: \d が Unicode 数字にもマッチすることを忘れる
text = "価格: ١٢٣ 円"  # アラビア数字 (U+0661, U+0662, U+0663)
print(re.findall(r'\d+', text))
# => ['١٢٣']  -- Python 3 では Unicode 数字にもマッチ

# セキュリティ上問題になる場合がある(数値パース時に予期しない値)

# OK: ASCII 数字のみを対象にする場合は明示する
print(re.findall(r'[0-9]+', text))
# => []  -- ASCII 数字のみ

# または re.ASCII フラグを使う
print(re.findall(r'\d+', text, re.ASCII))
# => []
```

### 6.3 アンチパターン: 不要な文字クラス

```python
import re

# NG: 1文字しかない文字クラス
pattern_bad = r'[a]'   # \a と同じだが無駄に冗長
# NG: ショートハンドを文字クラスに入れる意味なし
pattern_bad2 = r'[\d]'  # \d と同じ

# OK: シンプルに書く
pattern_good = r'a'
pattern_good2 = r'\d'

# ただし組み合わせる場合は文字クラスが必要:
pattern_ok = r'[\d_-]'  # 数字、アンダースコア、ハイフン
```

---

## 7. FAQ

### Q1: 文字クラス内でハイフンをリテラルとして使うには？

**A**: 3つの方法がある:

```python
import re

# 方法1: 先頭に置く
print(re.findall(r'[-abc]', "a-b"))  # => ['a', '-', 'b']

# 方法2: 末尾に置く
print(re.findall(r'[abc-]', "a-b"))  # => ['a', '-', 'b']

# 方法3: エスケープする
print(re.findall(r'[a\-c]', "a-b"))  # => ['a', '-']
```

先頭に置く方法が最も一般的で読みやすい。

### Q2: `\w` と `[a-zA-Z0-9_]` は常に同じか？

**A**: **同じではない**。Unicode モードが有効な場合、`\w` は各言語の文字(漢字、ひらがな等)にもマッチする:

```python
import re

text = "hello_世界"

# Unicode モード (Python 3 デフォルト)
print(re.findall(r'\w+', text))            # => ['hello_世界']
print(re.findall(r'[a-zA-Z0-9_]+', text))  # => ['hello_']

# ASCII モード
print(re.findall(r'\w+', text, re.ASCII))  # => ['hello_']
```

### Q3: POSIX クラスは Python で使えるか？

**A**: Python の `re` モジュールは POSIX 文字クラスを **直接サポートしない**。代替手段:

```python
import re
import unicodedata

# [:alpha:] の代替
# 方法1: Unicode カテゴリを使う (regex モジュール)
# pip install regex
# import regex
# regex.findall(r'\p{Alpha}+', text)

# 方法2: 明示的に範囲を指定
alpha_ascii = r'[a-zA-Z]'

# 方法3: str.isalpha() と組み合わせ
text = "Hello 123 World"
words = re.findall(r'\S+', text)
alpha_words = [w for w in words if w.isalpha()]
print(alpha_words)  # => ['Hello', 'World']
```

### Q4: Unicode Property Escape とは何か？

**A**: `\p{...}` で Unicode のカテゴリやスクリプトを指定できる(サポートはエンジンによる):

```javascript
// JavaScript (ES2018+ with /u flag)
const text = "Hello 世界 café";

// Unicodeの「文字」全般
console.log(text.match(/\p{L}+/gu));
// => ['Hello', '世界', 'café']

// 日本語スクリプト
console.log(text.match(/\p{Script=Hiragana}+/gu));
// => (なし)

// 漢字
console.log(text.match(/\p{Script=Han}+/gu));
// => ['世界']
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| `[abc]` | a, b, c のいずれか1文字 |
| `[a-z]` | a から z の範囲 |
| `[^abc]` | a, b, c 以外の1文字(否定) |
| `\d` / `\D` | 数字 / 非数字 |
| `\w` / `\W` | 単語文字 / 非単語文字 |
| `\s` / `\S` | 空白 / 非空白 |
| POSIX `[[:alpha:]]` | 英字(UNIX系ツール用) |
| `\p{L}` | Unicode文字プロパティ(対応エンジンのみ) |
| Unicode注意 | `\d` `\w` は言語とモードで範囲が変わる |
| 鉄則 | `[A-z]` は使わない、Unicode挙動を把握する |

## 次に読むべきガイド

- [03-quantifiers-anchors.md](./03-quantifiers-anchors.md) -- 量指定子とアンカー
- [../01-advanced/00-groups-backreferences.md](../01-advanced/00-groups-backreferences.md) -- グループと後方参照

## 参考文献

1. **Unicode Technical Standard #18** "Unicode Regular Expressions" https://unicode.org/reports/tr18/ -- Unicode正規表現の国際標準
2. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- 第5章「文字クラス」の詳細解説
3. **POSIX.1-2017** "Regular Expressions" https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap09.html -- POSIX正規表現の公式仕様
