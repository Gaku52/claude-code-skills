# 量指定子・アンカー -- *+?{n,m}、^$\b、貪欲/怠惰

> 量指定子(Quantifier)は繰り返し回数を制御し、アンカー(Anchor)は位置を制約する。貪欲(greedy)マッチと怠惰(lazy)マッチの違いを正確に理解することが、意図通りのパターンを書くための鍵である。

## この章で学ぶこと

1. **量指定子の種類と動作** -- `*` `+` `?` `{n,m}` の正確な意味と使い分け
2. **貪欲マッチと怠惰マッチ** -- デフォルト動作が「最長一致」である理由とその制御方法
3. **アンカーの種類と応用** -- `^` `$` `\b` `\A` `\Z` による位置指定の全パターン

---

## 1. 量指定子(Quantifiers)

### 1.1 基本量指定子

```python
import re

text = "aaabbbccc"

# * : 0回以上(0個でもマッチする)
print(re.findall(r'a*', text))
# => ['aaa', '', '', '', '', '', '', '']
# 注: 空文字列にもマッチする(0回の繰り返し)

# + : 1回以上(最低1個必要)
print(re.findall(r'a+', text))
# => ['aaa']

# ? : 0回または1回
print(re.findall(r'a?', text))
# => ['a', 'a', 'a', '', '', '', '', '', '', '']
```

### 1.2 範囲指定 `{n,m}`

```python
import re

# {n}   : ちょうど n 回
# {n,}  : n 回以上
# {n,m} : n 回以上 m 回以下
# {,m}  : 0 回以上 m 回以下 (一部のエンジンのみ)

text = "1 12 123 1234 12345"

print(re.findall(r'\d{3}', text))     # ちょうど3桁: ['123', '123', '123']
print(re.findall(r'\b\d{3}\b', text)) # 単語境界付き: ['123']
print(re.findall(r'\d{2,4}', text))   # 2-4桁: ['12', '123', '1234', '1234']
print(re.findall(r'\d{3,}', text))    # 3桁以上: ['123', '1234', '12345']
```

### 1.3 量指定子の等価関係

```
量指定子の糖衣構文:

  *     ≡  {0,}    0回以上
  +     ≡  {1,}    1回以上
  ?     ≡  {0,1}   0回または1回

  {3}   → ちょうど3回
  {3,}  → 3回以上(上限なし)
  {3,5} → 3回以上5回以下
  {0,5} → 0回以上5回以下
```

---

## 2. 貪欲マッチと怠惰マッチ

### 2.1 貪欲(Greedy) -- デフォルト

```python
import re

text = '<div>hello</div><div>world</div>'

# 貪欲マッチ(デフォルト): できるだけ多くマッチしようとする
greedy = re.search(r'<div>.*</div>', text)
print(greedy.group())
# => '<div>hello</div><div>world</div>'
#    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#    最長一致: 最初の <div> から最後の </div> まで
```

### 2.2 怠惰(Lazy) -- `?` を付加

```python
import re

text = '<div>hello</div><div>world</div>'

# 怠惰マッチ: できるだけ少なくマッチしようとする
lazy = re.findall(r'<div>.*?</div>', text)
print(lazy)
# => ['<div>hello</div>', '<div>world</div>']
#    最短一致: 最初の <div> から最も近い </div> まで
```

### 2.3 貪欲 vs 怠惰 -- 全量指定子

```python
import re

text = "aabab"

# 貪欲(デフォルト)         怠惰(? 付加)
print(re.search(r'a.*b', text).group())    # => 'aabab' (最長)
print(re.search(r'a.*?b', text).group())   # => 'aab'   (最短)

print(re.search(r'a.+b', text).group())    # => 'aabab' (最長)
print(re.search(r'a.+?b', text).group())   # => 'aabab' (.+は最低1文字)

print(re.search(r'a.?b', text).group())    # => 'aab'
print(re.search(r'a.??b', text).group())   # => 'ab' (位置2-3)
```

### 2.4 動作の可視化

```
貪欲マッチ: .*  (パターン: <.*>、テキスト: <b>bold</b>)

ステップ1: < にマッチ                         <
ステップ2: .* が全文字を飲み込む              <b>bold</b>
ステップ3: > にマッチしない(文字列末尾)        → バックトラック
ステップ4: 1文字戻して > を試行               <b>bold</b  → > ≠ b → 戻る
ステップ5: さらに1文字戻す                    <b>bold</b> → > = > → マッチ!
結果: <b>bold</b>  (最長一致)

─────────────────────────────────────────

怠惰マッチ: .*?  (パターン: <.*?>、テキスト: <b>bold</b>)

ステップ1: < にマッチ                         <
ステップ2: .*? が0文字でまず試行              <  → > ≠ b → 拡張
ステップ3: 1文字拡張                          <b → > ≠ o → 拡張
ステップ4: もう1文字拡張... いや待て           <b> → > = > → マッチ!
結果: <b>  (最短一致)
```

### 2.5 独占的量指定子(Possessive) -- バックトラック禁止

```java
// Java, PCRE でサポート (Python の re では非サポート)
// Python では regex モジュール(サードパーティ)で利用可能

// 貪欲:     .*   (バックトラックあり)
// 怠惰:     .*?  (最短一致)
// 独占的:   .*+  (バックトラックなし → 高速だがマッチ失敗しやすい)

// Java の例:
String text = "aaaa";
// 貪欲:  a*a  → "aaaa" (a* が3つ取り、最後のaが1つ)
// 独占的: a*+a → マッチ失敗 (a*+ が全て取り、バックトラックしない)
```

### 2.6 量指定子比較表

| 貪欲 | 怠惰 | 独占的 | 動作 |
|------|------|--------|------|
| `*` | `*?` | `*+` | 0回以上 |
| `+` | `+?` | `++` | 1回以上 |
| `?` | `??` | `?+` | 0回or1回 |
| `{n,m}` | `{n,m}?` | `{n,m}+` | n回以上m回以下 |

| 特性 | 貪欲 | 怠惰 | 独占的 |
|------|------|------|--------|
| マッチ方針 | 最長一致 | 最短一致 | 最長(失敗時バックトラックなし) |
| バックトラック | あり | あり | なし |
| 速度 | 普通 | 普通 | 高速(マッチ時) |
| 用途 | デフォルト | タグ抽出等 | パフォーマンス最適化 |
| サポート | 全エンジン | 全エンジン | Java/PCRE/regex(Python) |

---

## 3. アンカー(Anchors)

### 3.1 行頭・行末アンカー

```python
import re

text = """first line
second line
third line"""

# ^ : 行頭(デフォルトでは文字列先頭のみ)
print(re.findall(r'^.+', text))
# => ['first line']

# $ : 行末(デフォルトでは文字列末尾のみ)
print(re.findall(r'.+$', text))
# => ['third line']

# re.MULTILINE: ^$ が各行の先頭・末尾にマッチ
print(re.findall(r'^.+', text, re.MULTILINE))
# => ['first line', 'second line', 'third line']

print(re.findall(r'^\w+', text, re.MULTILINE))
# => ['first', 'second', 'third']
```

### 3.2 文字列境界アンカー

```python
import re

text = "hello\nworld"

# \A : 文字列の絶対先頭(MULTILINEの影響を受けない)
print(re.search(r'\Ahello', text).group())  # => 'hello'
print(re.search(r'\Aworld', text))          # => None

# \Z : 文字列の絶対末尾(MULTILINEの影響を受けない)
print(re.search(r'world\Z', text).group())  # => 'world'

# ^ vs \A (MULTILINEモード)
print(re.findall(r'^\w+', text, re.M))   # => ['hello', 'world']
print(re.findall(r'\A\w+', text, re.M))  # => ['hello'] (先頭のみ)
```

### 3.3 単語境界 `\b`

```python
import re

text = "cat concatenate category caterpillar"

# \b : 単語境界
# 単語文字(\w)と非単語文字(\W)の間、または文字列の先頭/末尾

# "cat" を独立した単語として検索
print(re.findall(r'\bcat\b', text))
# => ['cat']  -- "concatenate" 等にはマッチしない

# "cat" で始まる単語
print(re.findall(r'\bcat\w*', text))
# => ['cat', 'concatenate', 'category', 'caterpillar']

# \B : 非単語境界(単語の途中)
print(re.findall(r'\Bcat\B', text))
# => ['cat']  -- "concatenate" の途中の cat のみ
```

### 3.4 アンカーの概念図

```
テキスト: "Hello World"

位置:  ^  H  e  l  l  o     W  o  r  l  d  $
       ↑                                    ↑
       文字列先頭(^, \A)                    文字列末尾($, \Z)

単語境界(\b)の位置:
       \b H  e  l  l  o \b  \b W  o  r  l  d \b
       ↑                ↑   ↑                ↑
       単語開始          単語終了/開始         単語終了

非単語境界(\B)の位置:
          \B \B \B \B      \B \B \B \B
          H--e--l--l--o    W--o--r--l--d
          文字と文字の間(両方が\w)
```

### 3.5 各種アンカー一覧

```
┌────────┬──────────────────────────────────────┐
│ アンカー │ 意味                                  │
├────────┼──────────────────────────────────────┤
│ ^      │ 行頭 (MULTILINE) / 文字列先頭         │
│ $      │ 行末 (MULTILINE) / 文字列末尾         │
│ \A     │ 文字列の絶対先頭                       │
│ \Z     │ 文字列の絶対末尾                       │
│ \z     │ 文字列の絶対末尾(末尾改行も無視)       │
│ \b     │ 単語境界                               │
│ \B     │ 非単語境界                             │
│ \G     │ 前回マッチの終了位置(Java/Perl/.NET)   │
└────────┴──────────────────────────────────────┘
```

---

## 4. 実践パターン

### 4.1 行全体のマッチ

```python
import re

log = """2026-02-11 10:00 INFO Server started
2026-02-11 10:05 ERROR Connection failed
2026-02-11 10:10 INFO Request received
2026-02-11 10:15 WARN Memory usage high"""

# ERROR を含む行を完全に取得
error_lines = re.findall(r'^.*ERROR.*$', log, re.MULTILINE)
print(error_lines)
# => ['2026-02-11 10:05 ERROR Connection failed']

# WARN または ERROR を含む行
issues = re.findall(r'^.*(ERROR|WARN).*$', log, re.MULTILINE)
print(issues)
# => ['2026-02-11 10:05 ERROR Connection failed',
#     '2026-02-11 10:15 WARN Memory usage high']
```

### 4.2 完全一致バリデーション

```python
import re

# 文字列全体が完全にパターンに一致するか検証
# ^...$ を使う(または re.fullmatch)

def validate_date(s):
    """YYYY-MM-DD 形式の日付を検証"""
    return bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', s))

print(validate_date("2026-02-11"))      # => True
print(validate_date("2026-02-11 "))     # => False (末尾空白)
print(validate_date("date: 2026-02-11"))# => False (先頭にテキスト)

def validate_hex_color(s):
    """#RRGGBB 形式のカラーコードを検証"""
    return bool(re.fullmatch(r'#[0-9a-fA-F]{6}', s))

print(validate_hex_color("#FF5733"))  # => True
print(validate_hex_color("#GG5733"))  # => False
```

---

## 5. アンチパターン

### 5.1 アンチパターン: `.*` の無秩序な使用

```python
import re

# NG: 貪欲な .* が予期しない範囲をマッチ
html = '<span class="a">hello</span><span class="b">world</span>'
bad = re.search(r'<span.*>.*</span>', html)
print(bad.group())
# => '<span class="a">hello</span><span class="b">world</span>'
# 全体がマッチしてしまう!

# OK: 否定文字クラスまたは怠惰量指定子を使う
# 方法1: 怠惰量指定子
good1 = re.findall(r'<span.*?>.*?</span>', html)
print(good1)  # => ['<span class="a">hello</span>', ...]

# 方法2: 否定文字クラス(より高速)
good2 = re.findall(r'<span[^>]*>[^<]*</span>', html)
print(good2)  # => ['<span class="a">hello</span>', ...]
```

### 5.2 アンチパターン: `^` `$` のMULTILINEフラグ忘れ

```python
import re

text = """user: admin
user: guest
user: root"""

# NG: MULTILINE なしで各行の先頭を期待
bad = re.findall(r'^user: (\w+)', text)
print(bad)  # => ['admin']  -- 最初の行のみ

# OK: MULTILINE フラグを付ける
good = re.findall(r'^user: (\w+)', text, re.MULTILINE)
print(good)  # => ['admin', 'guest', 'root']
```

---

## 6. FAQ

### Q1: `.*` と `[^X]*` はどちらが高速か？

**A**: 一般的に **`[^X]*` のほうが高速**。`.*` はバックトラックが発生しやすいが、`[^X]*` は明示的に停止文字を指定するため不要なバックトラックを回避できる:

```python
import re

# 遅い: .* がバックトラックを多用
slow = r'<tag>.*</tag>'

# 速い: [^<]* で < 以外だけマッチ
fast = r'<tag>[^<]*</tag>'

# さらに速い: 独占的量指定子(Javaなど)
# fastest = '<tag>[^<]*+</tag>'
```

### Q2: `{0,}` と `*` に違いはあるか？

**A**: 意味は完全に同じ。`*` は `{0,}` の糖衣構文(シンタックスシュガー)である。エンジン内部では同一の処理となる。可読性のために `*` `+` `?` を優先的に使い、`{n,m}` は具体的な回数指定が必要な場合に使う。

### Q3: 怠惰量指定子を使えば常に安全か？

**A**: **いいえ**。怠惰量指定子はバックトラックの方向を変えるだけで、バックトラック自体はなくならない。特定のパターンでは怠惰でもReDoSが発生しうる:

```python
# 怠惰でも遅いパターンの例:
# パターン: (a+?)+b
# テキスト: "aaaaaaaaaaaaaaaaac"
# → 怠惰でも指数的バックトラックが発生

# 安全なのは:
# 1. 否定文字クラス [^X]* を使う
# 2. DFA エンジン(RE2等)を使う
# 3. 独占的量指定子/アトミックグループを使う
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| `*` | 0回以上(貪欲) |
| `+` | 1回以上(貪欲) |
| `?` | 0回or1回(貪欲) |
| `{n,m}` | n回以上m回以下 |
| `*?` `+?` `??` | 怠惰(最短一致)版 |
| `*+` `++` `?+` | 独占的(バックトラック禁止)版 |
| `^` | 行頭 / 文字列先頭 |
| `$` | 行末 / 文字列末尾 |
| `\b` | 単語境界 |
| `\A` / `\Z` | 文字列の絶対先頭/末尾 |
| 貪欲 vs 怠惰 | 貪欲が最長、怠惰が最短 |
| パフォーマンス | `[^X]*` > `.*?` > `.*`(一般的に) |

## 次に読むべきガイド

- [../01-advanced/00-groups-backreferences.md](../01-advanced/00-groups-backreferences.md) -- グループと後方参照
- [../01-advanced/01-lookaround.md](../01-advanced/01-lookaround.md) -- 先読み・後読み
- [../01-advanced/03-performance.md](../01-advanced/03-performance.md) -- パフォーマンス最適化とReDoS対策

## 参考文献

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- 第4章「量指定子の力学」と第6章「バックトラック」が必読
2. **Russ Cox** "Regular Expression Matching: the Virtual Machine Approach" https://swtch.com/~rsc/regexp/regexp2.html, 2009 -- バックトラックの理論的分析
3. **Jan Goyvaerts** "Regular-Expressions.info" https://www.regular-expressions.info/repeat.html -- 量指定子の実用的な解説
