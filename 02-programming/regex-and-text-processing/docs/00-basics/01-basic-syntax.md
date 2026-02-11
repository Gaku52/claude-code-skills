# 基本構文 -- リテラル、メタ文字、エスケープ

> 正規表現の最も基礎的な構成要素であるリテラル文字、メタ文字(特殊文字)、エスケープシーケンスの動作原理と正しい使い方を網羅的に解説する。

## この章で学ぶこと

1. **リテラル文字とメタ文字の区別** -- どの文字がそのまま一致し、どの文字が特殊な意味を持つか
2. **エスケープの仕組みと落とし穴** -- バックスラッシュによるメタ文字の無効化と二重エスケープ問題
3. **フラグ(修飾子)による挙動変更** -- 大文字小文字無視、複数行モード、ドットオールモード

---

## 1. リテラル文字

リテラル文字はパターン中でそのまま対応する文字にマッチする。

```python
import re

# リテラル文字のみのパターン
pattern = r'hello'
text = "say hello to the world"

match = re.search(pattern, text)
print(match.group())  # => "hello"
print(match.start())  # => 4
print(match.end())    # => 9
```

リテラル文字マッチの規則:

```
パターン   対象文字列        結果
────────   ──────────────   ──────
cat        "the cat sat"    マッチ ("cat")
123        "abc123def"      マッチ ("123")
hello      "Hello World"    不一致 (大文字小文字区別)
hello      "Hello World"    マッチ (i フラグ使用時)
```

---

## 2. メタ文字一覧

正規表現において特殊な意味を持つ文字群:

```
メタ文字一覧 (12文字 + バックスラッシュ):

.   任意の1文字 (改行を除く)
^   行頭 / 文字クラス内で否定
$   行末
*   直前の要素を0回以上繰り返し
+   直前の要素を1回以上繰り返し
?   直前の要素を0回または1回
|   選択 (OR)
()  グループ化・キャプチャ
[]  文字クラス
{}  量指定子 {n,m}
\   エスケープ文字

文字クラス [] 内でのメタ文字:
]   文字クラスの終了
\   エスケープ
^   否定 (先頭のみ)
-   範囲指定 (文字間のみ)
```

### 2.1 ドット `.` -- 任意の1文字

```python
import re

pattern = r'c.t'
texts = ["cat", "cot", "cut", "ct", "coat", "c\nt"]

for t in texts:
    m = re.search(pattern, t)
    result = m.group() if m else "不一致"
    print(f"  '{t}' → {result}")

# 出力:
#   'cat' → cat
#   'cot' → cot
#   'cut' → cut
#   'ct'  → 不一致     (ドットは1文字必須)
#   'coat' → coa は不一致、c.t にはマッチしない
#   'c\nt' → 不一致    (ドットは改行にマッチしない ※DOTALL除く)

# DOTALL フラグで改行にもマッチ
m = re.search(r'c.t', "c\nt", re.DOTALL)
print(m.group())  # => "c\nt"
```

### 2.2 パイプ `|` -- 選択(OR)

```python
import re

# パイプによる選択
pattern = r'cat|dog|bird'
texts = ["I have a cat", "I have a dog", "I have a fish"]

for t in texts:
    m = re.search(pattern, t)
    print(f"  '{t}' → {m.group() if m else '不一致'}")

# 出力:
#   'I have a cat' → cat
#   'I have a dog' → dog
#   'I have a fish' → 不一致

# 注意: パイプの優先順位
# gr(a|e)y  → "gray" or "grey"     (グループ内で選択)
# gray|grey → "gray" or "grey"     (同等)
# gra|ey    → "gra" or "ey"        (意図と異なる可能性)
```

### 2.3 アスタリスク `*`、プラス `+`、クエスチョン `?`

```python
import re

# * : 0回以上
print(re.findall(r'ab*c', "ac abc abbc"))     # => ['ac', 'abc', 'abbc']

# + : 1回以上
print(re.findall(r'ab+c', "ac abc abbc"))     # => ['abc', 'abbc']

# ? : 0回または1回
print(re.findall(r'colou?r', "color colour"))  # => ['color', 'colour']
```

---

## 3. エスケープ

### 3.1 メタ文字のエスケープ

```python
import re

# メタ文字をリテラルとして扱うにはバックスラッシュでエスケープ
price_pattern = r'\$\d+\.\d{2}'
text = "Price: $19.99 and $5.00"

matches = re.findall(price_pattern, text)
print(matches)  # => ['$19.99', '$5.00']

# エスケープが必要なメタ文字の例:
#   \.  → リテラルのドット
#   \*  → リテラルのアスタリスク
#   \+  → リテラルのプラス
#   \?  → リテラルのクエスチョンマーク
#   \(  → リテラルの開き括弧
#   \)  → リテラルの閉じ括弧
#   \[  → リテラルの開き角括弧
#   \{  → リテラルの開き波括弧
#   \|  → リテラルのパイプ
#   \\  → リテラルのバックスラッシュ
#   \^  → リテラルのキャレット
#   \$  → リテラルのドル記号
```

### 3.2 二重エスケープ問題

```
エスケープの流れ:

ソースコード          Python文字列        正規表現エンジン
─────────────       ──────────────      ────────────────
"\\d"          →    \d             →    数字1文字にマッチ
"\\\\d"        →    \\d            →    リテラル \ + d
r"\d"          →    \d             →    数字1文字にマッチ (raw string)
r"\\d"         →    \\d            →    リテラル \ + d

※ raw string (r"...") を使えば二重エスケープを回避できる
```

```python
import re

# 二重エスケープの問題
# Windowsパスをマッチする場合:

# NG: 通常文字列 -- バックスラッシュが二重に解釈される
pattern_bad = "C:\\\\Users\\\\\\w+"
# Python文字列としての解釈: C:\\Users\\\w+
# 正規表現としての解釈: C:\Users\ + 単語文字列

# OK: raw string を使う
pattern_good = r"C:\\Users\\\w+"

text = r"C:\Users\gaku"
print(re.search(pattern_good, text).group())  # => C:\Users\gaku
```

### 3.3 特殊エスケープシーケンス

```
エスケープシーケンス一覧:

文字クラス系:
  \d  → 数字 [0-9]
  \D  → 非数字 [^0-9]
  \w  → 単語文字 [a-zA-Z0-9_]
  \W  → 非単語文字 [^a-zA-Z0-9_]
  \s  → 空白文字 [ \t\n\r\f\v]
  \S  → 非空白文字 [^ \t\n\r\f\v]

アンカー系:
  \b  → 単語境界
  \B  → 非単語境界

特殊文字:
  \t  → タブ
  \n  → 改行 (LF)
  \r  → 復帰 (CR)
  \0  → NULL文字
  \xHH  → 16進数で指定 (例: \x41 = 'A')
  \uHHHH → Unicode (例: \u3042 = 'あ')
```

---

## 4. フラグ(修飾子)

### 4.1 主要フラグ一覧

```python
import re

text = """Hello World
hello python
HELLO REGEX"""

# i フラグ: 大文字小文字を無視
print(re.findall(r'hello', text, re.IGNORECASE))
# => ['Hello', 'hello', 'HELLO']

# m フラグ: 複数行モード (^$ が各行に作用)
print(re.findall(r'^hello', text, re.MULTILINE | re.IGNORECASE))
# => ['Hello', 'hello', 'HELLO']

# s フラグ: ドットが改行にもマッチ
print(re.search(r'Hello.+REGEX', text, re.DOTALL).group())
# => 'Hello World\nhello python\nHELLO REGEX'

# x フラグ: 冗長モード (空白・コメントを無視)
pattern = re.compile(r'''
    \d{4}       # 年 (4桁)
    -            # ハイフン区切り
    \d{2}       # 月 (2桁)
    -            # ハイフン区切り
    \d{2}       # 日 (2桁)
''', re.VERBOSE)
print(pattern.search("Date: 2026-02-11").group())
# => '2026-02-11'
```

### 4.2 フラグ比較表

| フラグ | Python | JavaScript | Perl | 効果 |
|--------|--------|------------|------|------|
| 大文字小文字無視 | `re.IGNORECASE` / `re.I` | `/i` | `/i` | 大文字小文字を区別しない |
| 複数行 | `re.MULTILINE` / `re.M` | `/m` | `/m` | `^` `$` が各行の先頭・末尾にマッチ |
| ドットオール | `re.DOTALL` / `re.S` | `/s` | `/s` | `.` が改行にもマッチ |
| 冗長モード | `re.VERBOSE` / `re.X` | 非対応(ES2025で `/x` 提案中) | `/x` | 空白・コメントを無視 |
| Unicode | `re.UNICODE` / `re.U` | `/u` | デフォルト | Unicode対応 |
| グローバル | N/A (`findall`) | `/g` | `/g` | 全マッチを返す |
| スティッキー | N/A | `/y` | N/A | lastIndex位置からのみマッチ |

---

## 5. ASCII 図解: パターンマッチングの流れ

### 5.1 基本的なマッチング手順

```
パターン: h.llo
テキスト: "say hello world"

位置: s a y   h e l l o   w o r l d
      0 1 2 3 4 5 6 7 8 9 ...

試行1: 位置0 's' ≠ 'h' → 失敗、位置1へ
試行2: 位置1 'a' ≠ 'h' → 失敗、位置2へ
試行3: 位置2 'y' ≠ 'h' → 失敗、位置3へ
試行4: 位置3 ' ' ≠ 'h' → 失敗、位置4へ
試行5: 位置4 'h' = 'h' → 一致
        位置5 'e' = '.' → 一致 (任意の1文字)
        位置6 'l' = 'l' → 一致
        位置7 'l' = 'l' → 一致
        位置8 'o' = 'o' → 一致
        → マッチ成功: "hello" (位置4-8)
```

### 5.2 メタ文字の意味マップ

```
┌─────────────────────────────────────────────────┐
│              正規表現の構成要素                    │
├────────────┬────────────────────────────────────┤
│ リテラル    │  a b c 1 2 3 あ い う              │
│ (そのまま)  │  → その文字自身にマッチ              │
├────────────┼────────────────────────────────────┤
│ メタ文字    │  . ^ $ * + ? | ( ) [ ] { } \      │
│ (特殊意味)  │  → 特別な動作を指示                  │
├────────────┼────────────────────────────────────┤
│ エスケープ  │  \. \* \+ \? \( \) \[ \{ \\ 等     │
│ (無効化)    │  → メタ文字をリテラルに戻す           │
├────────────┼────────────────────────────────────┤
│ ショートハンド│  \d \w \s \b \t \n 等             │
│ (略記法)    │  → 文字クラスの省略表記               │
└────────────┴────────────────────────────────────┘
```

### 5.3 エスケープ層の構造

```
  ソースコード層        言語処理系層       正規表現エンジン層
 ┌──────────┐      ┌──────────┐      ┌──────────────┐
 │ "\\d+"   │ ───→ │  \d+     │ ───→ │ 1桁以上の数字  │
 │ r"\d+"   │ ───→ │  \d+     │ ───→ │ 1桁以上の数字  │
 │ "\\\\n"  │ ───→ │  \\n     │ ───→ │ \ + n (2文字) │
 │ r"\\n"   │ ───→ │  \\n     │ ───→ │ \ + n (2文字) │
 │ "\n"     │ ───→ │  改行    │ ───→ │ 改行文字       │
 │ r"\n"    │ ───→ │  \n      │ ───→ │ 改行文字       │
 └──────────┘      └──────────┘      └──────────────┘

 ポイント: raw string (r"...") は言語処理系層の
           エスケープを無効化する。
           正規表現エンジン層のエスケープは別物。
```

---

## 6. アンチパターン

### 6.1 アンチパターン: raw string を使わない

```python
import re

# NG: raw string を使わずに正規表現を書く
pattern_bad = "\\b\\w+\\b"  # 読みにくい、エスケープミスしやすい

# OK: raw string を使う
pattern_good = r"\b\w+\b"   # 明快で間違いにくい

text = "hello world"
print(re.findall(pattern_good, text))  # => ['hello', 'world']

# 特に危険な例:
# "\b" は Python では バックスペース文字(0x08)
# r"\b" は正規表現の単語境界
print("\b" == "\x08")   # => True  -- バックスペース!
print(r"\b" == "\\b")   # => True  -- 正規表現の \b
```

### 6.2 アンチパターン: ドットの過度な使用

```python
import re

# NG: ドットで何でもマッチさせる
pattern_bad = r'\d+.\d+.\d+'

# これは "192.168.1.1" だけでなく以下にもマッチしてしまう:
texts = ["192.168.1.1", "192-168-1-1", "192x168x1x1", "192 168 1 1"]
for t in texts:
    m = re.search(pattern_bad, t)
    if m:
        print(f"  マッチ: {m.group()}")  # 全てマッチしてしまう

# OK: ドットをエスケープして明示的にする
pattern_good = r'\d+\.\d+\.\d+\.\d+'
for t in texts:
    m = re.search(pattern_good, t)
    if m:
        print(f"  マッチ: {m.group()}")  # "192.168.1.1" のみマッチ
```

---

## 7. FAQ

### Q1: メタ文字を全部エスケープするのが面倒な場合は？

**A**: 多くの言語に「文字列全体をエスケープする」関数がある:

```python
import re
user_input = "price is $10.00 (tax+)"
escaped = re.escape(user_input)
print(escaped)  # => 'price\\ is\\ \\$10\\.00\\ \\(tax\\+\\)'

# そのまま正規表現として安全に使える
pattern = re.compile(escaped)
```

JavaScript なら `string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')` で同等の処理ができる。

### Q2: `.` を改行にもマッチさせたいときは？

**A**: DOTALL(Python)/ `s` フラグ(JavaScript ES2018+)を使う:

```python
import re
text = "line1\nline2\nline3"
# デフォルト: ドットは改行にマッチしない
print(re.search(r'line1.+line3', text))  # => None

# DOTALL: ドットが改行にもマッチ
m = re.search(r'line1.+line3', text, re.DOTALL)
print(m.group())  # => 'line1\nline2\nline3'
```

代替手段として `[\s\S]` を使う方法もある(どの言語でも動作する)。

### Q3: 正規表現のフラグは複数同時に使えるか？

**A**: 使える。Python ではビット OR(`|`)で結合する:

```python
import re
pattern = re.compile(r'^hello.+world$', re.IGNORECASE | re.MULTILINE | re.DOTALL)
# JavaScript: /^hello.+world$/ims
# Perl: /^hello.+world$/ims
```

Python ではインラインフラグ `(?ims)` も使用可能:
```python
pattern = re.compile(r'(?ims)^hello.+world$')
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| リテラル文字 | そのまま対応する文字にマッチ |
| メタ文字 | `. ^ $ * + ? \| ( ) [ ] { } \` の13種類 |
| エスケープ | `\` でメタ文字をリテラルに戻す |
| raw string | `r"..."` で言語レベルのエスケープを無効化(Python) |
| ショートハンド | `\d` `\w` `\s` `\b` 等の省略記法 |
| フラグ | `i`(大文字小文字無視)、`m`(複数行)、`s`(ドットオール)、`x`(冗長) |
| 二重エスケープ | ソースコード層と正規表現層で2段階のエスケープが発生 |
| 鉄則 | 常に raw string を使い、ドットは必要なときだけ使う |

## 次に読むべきガイド

- [02-character-classes.md](./02-character-classes.md) -- 文字クラス `[abc]`、`\d`、`\w`、`\s`、POSIX クラス
- [03-quantifiers-anchors.md](./03-quantifiers-anchors.md) -- 量指定子・アンカーの詳細

## 参考文献

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions, 3rd Edition" O'Reilly Media, 2006 -- 第3章「基本構文」が特に参考になる
2. **Python re module documentation** https://docs.python.org/3/library/re.html -- Python 正規表現の公式リファレンス
3. **MDN Web Docs - Regular Expressions** https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions -- JavaScript 正規表現の包括的ガイド
