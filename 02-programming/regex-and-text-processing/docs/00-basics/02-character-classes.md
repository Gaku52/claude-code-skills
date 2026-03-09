# 文字クラス -- [abc]、\d、\w、\s、POSIX

> 文字クラス(Character Class)は正規表現の核心的機能であり、「この位置にマッチしてよい文字の集合」を定義する。角括弧記法、ショートハンドクラス、POSIX クラスの全体像を解説する。

## この章で学ぶこと

1. **角括弧文字クラス `[...]` の構文と動作** -- 肯定・否定・範囲指定の正確な規則
2. **ショートハンドクラスの意味と言語差異** -- `\d` `\w` `\s` が言語ごとに異なるUnicode対応
3. **POSIX文字クラスと実用的な選択基準** -- `[:alpha:]` `[:digit:]` 等の使いどころ
4. **Unicode文字プロパティの活用** -- `\p{L}` `\p{N}` 等のUnicodeカテゴリ指定
5. **文字クラスの集合演算** -- 交差・減算・和集合の実現方法


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [基本構文 -- リテラル、メタ文字、エスケープ](./01-basic-syntax.md) の内容を理解していること

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

```python
# 文字クラスの基本的な使い方

import re

# 1. 特定の文字集合からの選択
vowels = re.compile(r'[aeiouAEIOU]')
text = "Hello Beautiful World"
print(vowels.findall(text))
# => ['e', 'o', 'E', 'a', 'u', 'i', 'u', 'o']

# 2. 文字クラスは「1文字」にマッチする点に注意
print(re.findall(r'[abc]', "aabbcc"))
# => ['a', 'a', 'b', 'b', 'c', 'c']  ※ 各文字が個別にマッチ

# 3. 量指定子と組み合わせて複数文字にマッチ
print(re.findall(r'[abc]+', "aabbcc def aab"))
# => ['aabbcc', 'aab']

# 4. 文字クラスの中で順序は無関係
assert re.findall(r'[abc]', "abc") == re.findall(r'[cba]', "abc")
assert re.findall(r'[abc]', "abc") == re.findall(r'[bca]', "abc")
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

```python
# 範囲指定の詳細

import re

# 連続した範囲を複数指定
# 英数字とアンダースコア
print(re.findall(r'[a-zA-Z0-9_]+', "hello_world 123"))
# => ['hello_world', '123']

# 16進数の文字
print(re.findall(r'[0-9a-fA-F]+', "0xFF 0xAB 0xGG"))
# => ['0', 'xFF', '0', 'xAB', '0', 'xGG'] ※ xGG は不正な16進数

# より正確な16進数パターン
print(re.findall(r'0x[0-9a-fA-F]+', "0xFF 0xAB 0xGG"))
# => ['0xFF', '0xAB']  ※ 0xGG は部分マッチしない

# 部分的な範囲
print(re.findall(r'[a-f]+', "abcdefghij"))
# => ['abcdef']

print(re.findall(r'[2-7]+', "0123456789"))
# => ['234567']

# 複数の独立した範囲
print(re.findall(r'[a-cm-o1-3]+', "abcmnop123456"))
# => ['abc', 'mno', '123']
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

```python
# 否定文字クラスの実践的な使い方

import re

# 1. 引用符の中身を抽出（引用符自体を除外）
text = '"hello" and "world"'
print(re.findall(r'"([^"]*)"', text))
# => ['hello', 'world']

# 2. HTMLタグの属性値を抽出
html = '<a href="https://example.com" class="link">'
print(re.findall(r'(\w+)="([^"]*)"', html))
# => [('href', 'https://example.com'), ('class', 'link')]

# 3. カンマで区切られたフィールド（カンマ自体を除外）
csv_line = "field1,field2,field3"
print(re.findall(r'[^,]+', csv_line))
# => ['field1', 'field2', 'field3']

# 4. パスの最後のコンポーネント（スラッシュ以降）
path = "/usr/local/bin/python3"
print(re.findall(r'[^/]+$', path))
# => ['python3']

# 5. ファイル拡張子の抽出（ドット以降）
filename = "document.backup.tar.gz"
print(re.findall(r'[^.]+', filename))
# => ['document', 'backup', 'tar', 'gz']

# 6. 空白以外の連続文字
text = "  hello   world  "
print(re.findall(r'[^\s]+', text))
# => ['hello', 'world']

# 7. 特定の文字を除外したマッチ
# 制御文字と特殊文字を除外
text = "hello\x00world\x1b[31mred"
print(re.findall(r'[^\x00-\x1f\x7f]+', text))
# => ['hello', 'world', '[31mred']
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

```python
# メタ文字の扱いを確認

import re

# 文字クラス内ではほとんどのメタ文字はリテラル
print(re.findall(r'[.+*?|(){}]', "a.b+c*d?e|f(g)h{i}"))
# => ['.', '+', '*', '?', '|', '(', ')', '{', '}']

# ] を文字クラスに含める方法
# 方法1: 先頭に置く
print(re.findall(r'[]ab]', "a]b"))
# => ['a', ']', 'b']

# 方法2: エスケープ
print(re.findall(r'[a\]b]', "a]b"))
# => ['a', ']', 'b']

# \ を文字クラスに含める
print(re.findall(r'[a\\b]', r"a\b"))
# => ['a', '\\', 'b']

# - を文字クラスに含める方法
# 方法1: 先頭
print(re.findall(r'[-ab]', "a-b"))   # => ['a', '-', 'b']
# 方法2: 末尾
print(re.findall(r'[ab-]', "a-b"))   # => ['a', '-', 'b']
# 方法3: エスケープ
print(re.findall(r'[a\-b]', "a-b"))  # => ['a', '-', 'b']

# ^ を否定以外の目的で含める
# 先頭以外に置く
print(re.findall(r'[a^b]', "a^b"))   # => ['a', '^', 'b']
# エスケープ
print(re.findall(r'[\^ab]', "a^b"))  # => ['a', '^', 'b']
```

### 1.5 文字クラスの組み合わせテクニック

```python
import re

# 1. ショートハンドと文字クラスの組み合わせ
# 数字とハイフンとドット（電話番号やIPアドレス用）
print(re.findall(r'[\d.-]+', "IP: 192.168.1.1 Tel: 03-1234-5678"))
# => ['192.168.1.1', '03-1234-5678']

# 2. 単語文字とハイフン（CSS クラス名やスラグ）
print(re.findall(r'[\w-]+', "my-class another_class third-class-name"))
# => ['my-class', 'another_class', 'third-class-name']

# 3. 否定ショートハンドと文字クラスの組み合わせ
# 空白以外かつカンマ以外
print(re.findall(r'[^\s,]+', "apple, banana, cherry"))
# => ['apple', 'banana', 'cherry']

# 4. 日本語関連の文字クラス
# ひらがな
print(re.findall(r'[\u3040-\u309F]+', "こんにちは Hello 世界"))
# => ['こんにちは']

# カタカナ
print(re.findall(r'[\u30A0-\u30FF]+', "カタカナ ひらがな ABC"))
# => ['カタカナ']

# 漢字（CJK統合漢字の基本ブロック）
print(re.findall(r'[\u4E00-\u9FFF]+', "漢字テスト hello 東京タワー"))
# => ['漢字', '東京']

# ひらがな + カタカナ + 漢字
print(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+',
                 "東京タワーへ行こう ABC 123"))
# => ['東京タワーへ行こう']

# 5. 全角文字の検出
# 全角英数字
print(re.findall(r'[Ａ-Ｚａ-ｚ０-９]+', "Ｈｅｌｌｏ 123 Ｗｏｒｌｄ"))
# => ['Ｈｅｌｌｏ', 'Ｗｏｒｌｄ']
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

### 2.4 \d の Unicode 挙動の詳細

```python
import re

# Python 3 の \d は Unicode 数字全般にマッチする
# ASCII 数字以外の例:

test_strings = [
    "半角: 0123456789",           # ASCII 数字
    "全角: ０１２３４５６７８９",   # 全角数字
    "アラビア: ٠١٢٣٤٥٦٧٨٩",     # アラビア・インド数字
    "デーヴァナーガリー: ०१२३",    # デーヴァナーガリー数字
    "タイ: ๐๑๒๓๔๕๖๗๘๙",        # タイ数字
]

for s in test_strings:
    matches = re.findall(r'\d+', s)
    ascii_matches = re.findall(r'\d+', s, re.ASCII)
    print(f"  {s}")
    print(f"    Unicode \\d: {matches}")
    print(f"    ASCII \\d:   {ascii_matches}")

# ASCII のみにしたい場合の3つの方法:
# 1. re.ASCII フラグ
print(re.findall(r'\d+', "123 ０１２", re.ASCII))
# => ['123']

# 2. 明示的な文字クラス [0-9]
print(re.findall(r'[0-9]+', "123 ０１２"))
# => ['123']

# 3. インラインフラグ (?a)
print(re.findall(r'(?a)\d+', "123 ０１２"))
# => ['123']
```

### 2.5 \s の詳細: 空白文字の種類

```python
import re

# \s がマッチする文字の一覧 (ASCII モード)
whitespace_chars = {
    ' ':  'スペース (0x20)',
    '\t': 'タブ (0x09)',
    '\n': '改行 LF (0x0A)',
    '\r': '復帰 CR (0x0D)',
    '\f': 'フォームフィード (0x0C)',
    '\v': '垂直タブ (0x0B)',
}

for char, desc in whitespace_chars.items():
    matches = bool(re.match(r'\s', char))
    print(f"  {desc}: {'マッチ' if matches else '不一致'}")

# Unicode モードの追加空白文字
unicode_spaces = {
    '\u00A0': 'ノーブレークスペース (NBSP)',
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
    '\u3000': '全角スペース (Ideographic Space)',
    '\uFEFF': 'BOM (Byte Order Mark)',
}

for char, desc in unicode_spaces.items():
    # Python 3 ではデフォルトで Unicode モード
    matches_unicode = bool(re.match(r'\s', char))
    matches_ascii = bool(re.match(r'\s', char, re.ASCII))
    print(f"  {desc}: Unicode={matches_unicode}, ASCII={matches_ascii}")

# 実践例: 全角スペースの処理
text = "Hello　World"  # 全角スペースが含まれる
print(re.split(r'\s+', text))
# => ['Hello', 'World']  ※ 全角スペースも \s にマッチ

# ASCII モードでは全角スペースを無視
print(re.split(r'\s+', text, flags=re.ASCII))
# => ['Hello\u3000World']  ※ 全角スペースにマッチしない
```

### 2.6 \b 単語境界の詳細

```python
import re

# \b は「位置」にマッチする（ゼロ幅アサーション）
# 文字を消費しない

# 単語境界の定義:
# \w と \W の間の位置
# 文字列の先頭で直後が \w の位置
# 文字列の末尾で直前が \w の位置

text = "cat caterpillar concatenate category the_cat"

# \bcat\b: "cat" という完全な単語のみ
print(re.findall(r'\bcat\b', text))
# => ['cat']

# \bcat: "cat" で始まる単語の位置
print(re.findall(r'\bcat\w*', text))
# => ['cat', 'caterpillar', 'concatenate', 'category']

# cat\b: "cat" で終わる単語の位置
print(re.findall(r'\w*cat\b', text))
# => ['cat', 'the_cat']

# \B: 非単語境界（単語の途中）
print(re.findall(r'\Bcat\B', text))
# => ['cat']  ※ concatenate の中の cat

# 実践例: 単語の完全一致検索
def find_exact_word(text, word):
    """単語の完全一致を検索"""
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.findall(pattern, text)

print(find_exact_word("Java JavaScript JavaEE", "Java"))
# => ['Java']

# Unicode での単語境界
text = "東京は首都です。Tokyo is capital."
print(re.findall(r'\b\w+\b', text))
# => ['東京は首都です', 'Tokyo', 'is', 'capital']
# ※ 日本語には単語間の空白がないため、連続した \w がまとめてマッチ
```

### 2.7 ショートハンドの言語間差異

```
各言語での \w の挙動:

┌──────────────┬──────────────────────────────────────────┐
│ 言語         │ \w の範囲                                 │
├──────────────┼──────────────────────────────────────────┤
│ Python 3     │ Unicode Letters + Digits + _             │
│ (デフォルト)  │ → 日本語、中国語等もマッチ               │
├──────────────┼──────────────────────────────────────────┤
│ Python 3     │ [a-zA-Z0-9_]                             │
│ (re.ASCII)   │ → ASCII のみ                             │
├──────────────┼──────────────────────────────────────────┤
│ JavaScript   │ [a-zA-Z0-9_]                             │
│ (デフォルト)  │ → ASCII のみ                             │
├──────────────┼──────────────────────────────────────────┤
│ JavaScript   │ Unicode対応は \p{L} を使用               │
│ (/u フラグ)   │ → \w 自体は変わらない                    │
├──────────────┼──────────────────────────────────────────┤
│ Java         │ [a-zA-Z0-9_]                             │
│ (デフォルト)  │ → ASCII のみ                             │
├──────────────┼──────────────────────────────────────────┤
│ Java         │ Unicode Letters + Digits + _             │
│ (UNICODE_    │ → 日本語等もマッチ                       │
│  CHARACTER_  │                                          │
│  CLASS)      │                                          │
├──────────────┼──────────────────────────────────────────┤
│ Perl         │ Unicode Letters + Digits + _             │
│              │ → デフォルトで Unicode 対応               │
├──────────────┼──────────────────────────────────────────┤
│ Ruby         │ Unicode Letters + Digits + _             │
│              │ → デフォルトで Unicode 対応               │
├──────────────┼──────────────────────────────────────────┤
│ Go           │ [0-9A-Za-z_]                             │
│ (RE2)        │ → ASCII のみ                             │
├──────────────┼──────────────────────────────────────────┤
│ Rust         │ Unicode対応（regex クレート）              │
│              │ → ASCII モードは別途指定                   │
└──────────────┴──────────────────────────────────────────┘
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

# 英字のみを抽出
# => Hello
# => World

# 数字のみを抽出
# => 19
# => 99

# 16進数を抽出
# => FF00AA

# POSIX クラスの否定
echo "abc123" | grep -oE '[^[:digit:]]+'
# => abc
```

```bash
# POSIX クラスの実践的な使用例

# 1. 英数字とアンダースコアのみを抽出（変数名パターン）
echo "hello-world my_var 123abc" | grep -oE '[[:alnum:]_]+'
# => hello
# => world
# => my_var
# => 123abc

# 2. 句読点の抽出
# => ,
# => !
# => ?

# 3. 印刷不可能文字（制御文字）の検出
# => 2 (制御文字が2つ)

# 4. 空白でフィールドを分割（blank はスペースとタブのみ）
# => 3

# 5. sed でのPOSIXクラス使用
# => Hello  World

# 6. 大文字を小文字に変換（POSIX クラスベース）
# => hello world

# 7. ファイル名の安全な文字のみを残す
echo "my file (1).txt" | sed 's/[^[:alnum:]._-]/_/g'
# => my_file__1_.txt

# 8. 空行の除去
# => 空でない行のみ表示
```

### 3.3 POSIX vs ショートハンド 比較表

| 用途 | POSIX | ショートハンド | 使える環境 |
|------|-------|--------------|-----------|
| 単語文字 | なし | `\w` | ショートハンドのみ |

### 3.4 POSIX クラスの注意事項

```bash
# 注意1: POSIX クラスは必ず角括弧内で使う
# NG: [:digit:] -- 個別の文字 :, d, i, g, t としてマッチ

# 注意2: POSIX クラスと他の文字を組み合わせ可能
echo "abc-123_def" | grep -oE '[[:alnum:]_-]+'
# => abc-123_def

# 注意3: ロケールによって挙動が変わる
# LC_ALL=C では ASCII のみ
# LC_ALL=ja_JP.UTF-8 では日本語もマッチ
# => a, b (é はマッチしない)

# => aéb (é もマッチ)

# 注意4: grep -P (PCRE) では POSIX クラスは使えない場合がある
# grep -E (ERE) または grep (BRE) を推奨
```

---

## 4. Unicode 文字プロパティ

### 4.1 Unicode General Category

```python
# Python の regex モジュール（サードパーティ）で Unicode プロパティを使用
# pip install regex

# regex モジュールの使用例
try:
    import regex

    text = "Hello 世界 café 123 !@#"

    # \p{L} -- Unicode の「文字」(Letter)
    print(regex.findall(r'\p{L}+', text))
    # => ['Hello', '世界', 'café']

    # \p{N} -- Unicode の「数字」(Number)
    print(regex.findall(r'\p{N}+', text))
    # => ['123']

    # \p{P} -- Unicode の「句読点」(Punctuation)
    print(regex.findall(r'\p{P}', text))
    # => ['!']

    # \p{S} -- Unicode の「記号」(Symbol)
    print(regex.findall(r'\p{S}', text))
    # => ['@', '#']

    # \p{Z} -- Unicode の「区切り」(Separator)
    # スペース等

except ImportError:
    # regex モジュールがない場合の代替
    import re

    # Python 標準 re では Unicode プロパティは直接使えない
    # 代替: Unicode カテゴリをレンジで指定

    # 日本語文字（ひらがな・カタカナ・漢字）
    print(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+', text))
```

### 4.2 主要な Unicode カテゴリ

```
Unicode General Category:

L  (Letter)       -- 文字
├── Lu (Uppercase) -- 大文字 (A, B, C, ...)
├── Ll (Lowercase) -- 小文字 (a, b, c, ...)
├── Lt (Titlecase) -- タイトルケース (Dž, Lj, ...)
├── Lm (Modifier)  -- 修飾文字
└── Lo (Other)     -- その他の文字 (漢字, ひらがな, ...)

M  (Mark)          -- マーク(結合文字)
├── Mn (Nonspacing)
├── Mc (Spacing Combining)
└── Me (Enclosing)

N  (Number)        -- 数字
├── Nd (Decimal)   -- 10進数字 (0-9, ０-９, ...)
├── Nl (Letter)    -- 文字としての数字 (Ⅰ, Ⅱ, ...)
└── No (Other)     -- その他の数字 (½, ⅓, ...)

P  (Punctuation)   -- 句読点
├── Pc (Connector) -- 接続句読点 (_)
├── Pd (Dash)      -- ダッシュ (-, –, —)
├── Ps (Open)      -- 開き括弧 ((, [, {)
├── Pe (Close)     -- 閉じ括弧 (), ], })
├── Pi (Initial)   -- 開始引用符 («, ', ")
├── Pf (Final)     -- 終了引用符 (», ', ")
└── Po (Other)     -- その他の句読点 (., ,, !, ?)

S  (Symbol)        -- 記号
├── Sm (Math)      -- 数学記号 (+, =, <, >)
├── Sc (Currency)  -- 通貨記号 ($, €, ¥, £)
├── Sk (Modifier)  -- 修飾記号
└── So (Other)     -- その他の記号 (©, ®, ™)

Z  (Separator)     -- 区切り
├── Zs (Space)     -- 空白区切り
├── Zl (Line)      -- 行区切り
└── Zp (Paragraph) -- 段落区切り

C  (Other)         -- その他
├── Cc (Control)   -- 制御文字
├── Cf (Format)    -- 書式文字 (BOM等)
├── Cs (Surrogate) -- サロゲート
├── Co (Private)   -- 私用文字
└── Cn (Unassigned)-- 未割り当て
```

### 4.3 Unicode Script による文字クラス

```javascript
// JavaScript ES2018+ の Unicode Property Escape

const text = "Hello こんにちは 世界 Привет مرحبا";

// 日本語のひらがな
console.log(text.match(/\p{Script=Hiragana}+/gu));
// => ['こんにちは']

// 漢字 (Han)
console.log(text.match(/\p{Script=Han}+/gu));
// => ['世界']

// キリル文字
console.log(text.match(/\p{Script=Cyrillic}+/gu));
// => ['Привет']

// アラビア文字
console.log(text.match(/\p{Script=Arabic}+/gu));
// => ['مرحبا']

// ラテン文字
console.log(text.match(/\p{Script=Latin}+/gu));
// => ['Hello']

// 絵文字
const emoji_text = "Hello! Nice day!";
console.log(emoji_text.match(/\p{Emoji}/gu));
// => ['', '']
```

```python
# Python の regex モジュールでの Unicode Script

try:
    import regex

    text = "Hello こんにちは 世界 カタカナ"

    # ひらがな
    print(regex.findall(r'\p{Hiragana}+', text))
    # => ['こんにちは']

    # カタカナ
    print(regex.findall(r'\p{Katakana}+', text))
    # => ['カタカナ']

    # 漢字
    print(regex.findall(r'\p{Han}+', text))
    # => ['世界']

    # 日本語全般 (ひらがな + カタカナ + 漢字)
    print(regex.findall(r'[\p{Hiragana}\p{Katakana}\p{Han}]+', text))
    # => ['こんにちは', '世界', 'カタカナ']

except ImportError:
    pass
```

### 4.4 ECMAScript 2024 の v フラグ (Unicode Sets)

```javascript
// ECMAScript 2024 の v フラグでは文字クラスの集合演算が可能

// 交差 (&&) -- 両方に含まれる文字
// /[\p{Script=Latin}&&\p{Letter}]/v

// 減算 (--) -- 左から右を除く
// /[\p{Letter}--\p{Script=Latin}]/v

// 和集合 -- 従来の文字クラスと同じ
// /[\p{Script=Latin}\p{Script=Greek}]/v

// 例: ASCII文字を除くラテン文字（アクセント付き文字のみ）
// /[\p{Script=Latin}--[a-zA-Z]]/v

// 例: 数字を除く英数字 = 英字のみ
// /[\p{Alnum}--\p{Number}]/v
```

---

## 5. 組み合わせパターン

### 5.1 文字クラスの組み合わせ

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

### 5.2 よくある文字クラスパターン

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

### 5.3 高度な文字クラスパターン

```python
import re

# 1. メールアドレスのローカルパートに使える文字
local_part = r'[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+'
print(re.findall(local_part, "user.name+tag@example.com"))
# => ["user.name+tag"]

# 2. URL セーフな文字（RFC 3986）
url_safe = r'[a-zA-Z0-9._~:/?#\[\]@!$&\'()*+,;=-]+'
print(re.findall(url_safe, "https://example.com/path?q=hello&lang=ja"))

# 3. CSS セレクタに使える文字
css_selector = r'[a-zA-Z0-9_-]+'

# 4. シェルで安全なファイル名文字
safe_filename = r'[a-zA-Z0-9._-]+'

# 5. SQLインジェクション防止: 英数字とスペースのみ許可
safe_input = r'^[a-zA-Z0-9 ]+$'

# 6. Base64 エンコードされた文字列
base64_pattern = r'[A-Za-z0-9+/]+=*'
print(re.findall(base64_pattern, "SGVsbG8gV29ybGQ= next"))
# => ['SGVsbG8gV29ybGQ=']

# 7. UUID パターン
uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
print(re.findall(uuid_pattern, "id: 550e8400-e29b-41d4-a716-446655440000", re.IGNORECASE))
# => ['550e8400-e29b-41d4-a716-446655440000']

# 8. セマンティックバージョニング
semver_pattern = r'[0-9]+\.[0-9]+\.[0-9]+(?:-[a-zA-Z0-9.]+)?(?:\+[a-zA-Z0-9.]+)?'
print(re.findall(semver_pattern, "v1.2.3-beta.1+build.123"))
# => ['1.2.3-beta.1+build.123']
```

---

## 6. ASCII 図解

### 6.1 文字クラスの概念図

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

### 6.2 否定文字クラスの動作

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

### 6.3 範囲指定のASCIIコード基盤

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

### 6.4 文字クラスの集合演算

```
集合演算の概念:

和集合 (Union):       [a-z0-9]  = [a-z] ∪ [0-9]
否定 (Complement):    [^a-z]    = U \ [a-z]
交差 (Intersection):  Java: [a-z&&[aeiou]]  = [a-z] ∩ [aeiou] = [aeiou]
減算 (Subtraction):   .NET: [a-z-[aeiou]]   = [a-z] \ [aeiou] = 子音

視覚的な表現:

     [a-z]           [aeiou]
  ┌──────────┐    ┌─────────┐
  │ bcdfgh...│    │ a e i   │
  │ jklmnp...│ ∩  │ o u     │
  │  a e i   │    │         │
  │  o u     │    │         │
  └──────────┘    └─────────┘

  交差 [a-z&&[aeiou]] = {a, e, i, o, u}
  減算 [a-z-[aeiou]]  = {b, c, d, f, g, h, ...}
```

---

## 7. アンチパターン

### 7.1 アンチパターン: [A-z] を使う

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

### 7.2 アンチパターン: ショートハンドのUnicode挙動を無視する

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

### 7.3 アンチパターン: 不要な文字クラス

```python
import re

# NG: 1文字しかない文字クラス
pattern_bad = r'[a]'   # a と同じだが無駄に冗長
# NG: ショートハンドを文字クラスに入れる意味なし
pattern_bad2 = r'[\d]'  # \d と同じ

# OK: シンプルに書く
pattern_good = r'a'
pattern_good2 = r'\d'

# ただし組み合わせる場合は文字クラスが必要:
pattern_ok = r'[\d_-]'  # 数字、アンダースコア、ハイフン
```

### 7.4 アンチパターン: 否定文字クラスとドットの混同

```python
import re

# NG: [^...] は改行にもマッチするが . はマッチしない
text = "hello\nworld"

# . はデフォルトで改行にマッチしない
print(re.findall(r'.+', text))
# => ['hello', 'world']  ※ 改行で分断される

# [^\n] は改行以外全て（. と同等だが明示的）
print(re.findall(r'[^\n]+', text))
# => ['hello', 'world']

# [^a] は改行にもマッチする！
print(re.findall(r'[^a]+', text))
# => ['hello\nworld']  ※ 改行を含む

# この違いを理解していないとバグの原因になる
```

### 7.5 アンチパターン: 過度に広い文字クラス

```python
import re

# NG: 数値バリデーションに \d を使う
# ポート番号の検証
port = "65536"
if re.match(r'^\d+$', port):
    print("Valid port?")  # NG: 65536 は不正なポート番号

# OK: 数値の範囲チェックは正規表現ではなくコードで行う
def is_valid_port(s):
    if not re.match(r'^[0-9]+$', s):
        return False
    return 0 <= int(s) <= 65535

# NG: IPアドレスの検証に \d{1,3} だけを使う
ip_bad = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
# 999.999.999.999 にもマッチしてしまう

# OK: 各オクテットの範囲を検証
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

## 8. 実践パターン集

### 8.1 日本語テキスト処理

```python
import re

# ひらがなの検出
hiragana = re.compile(r'[\u3040-\u309F]+')
print(hiragana.findall("東京タワーへ行こう"))
# => ['へ', 'こう']  ※ 助詞や動詞のひらがな部分

# カタカナの検出
katakana = re.compile(r'[\u30A0-\u30FF]+')
print(katakana.findall("東京タワーへ行こう"))
# => ['タワー']

# 全角英数字の半角変換
def zen_to_han(text):
    """全角英数字を半角に変換"""
    return re.sub(r'[Ａ-Ｚａ-ｚ０-９]',
                  lambda m: chr(ord(m.group()) - 0xFEE0), text)

print(zen_to_han("Ｈｅｌｌｏ ０１２３"))
# => "Hello 0123"

# 半角カタカナの全角変換用マッピング
han_to_zen_map = str.maketrans(
    'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ',
    'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン'
)

def han_kata_to_zen(text):
    """半角カタカナを全角に変換"""
    return text.translate(han_to_zen_map)

# 日本語の文区切り
sentences = re.split(r'[。！？\n]+', "今日は天気がいい。明日も晴れるだろう！楽しみだ。")
print([s for s in sentences if s])
# => ['今日は天気がいい', '明日も晴れるだろう', '楽しみだ']
```

### 8.2 数値の文字クラスパターン

```python
import re

# 整数（正負）
integer_pattern = r'[+-]?[0-9]+'
print(re.findall(integer_pattern, "x=42, y=-17, z=+3"))
# => ['+42', '-17', '+3']  ※ 先頭の+は演算子の場合もある

# より正確な整数パターン
integer_strict = r'(?<![0-9])[+-]?[0-9]+(?![0-9.])'

# 小数（固定小数点）
decimal_pattern = r'[+-]?[0-9]+\.[0-9]+'
print(re.findall(decimal_pattern, "pi=3.14159, e=2.71828"))
# => ['3.14159', '2.71828']

# 科学表記
scientific = r'[+-]?[0-9]+\.?[0-9]*[eE][+-]?[0-9]+'
print(re.findall(scientific, "speed=3.0e8 tiny=1.6e-19"))
# => ['3.0e8', '1.6e-19']

# カンマ区切りの数値
comma_number = r'[0-9]{1,3}(?:,[0-9]{3})*'
print(re.findall(comma_number, "Population: 1,234,567 Area: 377,975"))
# => ['1,234,567', '377,975']

# 通貨表記
currency = r'[¥$€£][0-9,]+(?:\.[0-9]{2})?'
print(re.findall(currency, "Price: $1,299.99 and ¥150,000"))
# => ['$1,299.99', '¥150,000']
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
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

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## 9. FAQ

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

### Q5: 文字クラスのパフォーマンスは？

**A**: 文字クラスは一般的に高速だが、以下の点に注意:

```python
import re

# 1. 文字クラスは選択(|)より高速
# 遅い: a|b|c|d|e
# 速い: [a-e]

# 2. 否定文字クラスは肯定より若干遅い場合がある
# [^abc] は内部的に「abc 以外の全文字」をチェック

# 3. Unicode 文字クラスは ASCII のみより遅い
# \d (Unicode) > [0-9] (ASCII のみ)
# 速度が重要なら re.ASCII を検討

# 4. 文字クラスの最適化はエンジン依存
# 多くのエンジンは [a-z] をビットマップに最適化
# 大きな Unicode 範囲はツリー検索になる場合がある
```

### Q6: 文字クラス内でショートハンドを使えるか？

**A**: はい、使える。文字クラス内でショートハンドは展開される:

```python
import re

# 数字とアンダースコアとハイフン
print(re.findall(r'[\d_-]+', "hello_123-world"))
# => ['_123-']

# 空白と句読点
print(re.findall(r'[\s,.!?]+', "hello, world! foo"))
# => [', ', '! ']

# 単語文字とドット（ドメイン名用）
print(re.findall(r'[\w.]+', "example.com hello"))
# => ['example.com', 'hello']

# 否定ショートハンドも使える
print(re.findall(r'[\D]+', "abc123def"))  # 非数字
# => ['abc', 'def']
```

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

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
| `\b` / `\B` | 単語境界 / 非単語境界 (ゼロ幅) |
| `\p{L}` | Unicode文字プロパティ(対応エンジンのみ) |
| `\p{Script=...}` | Unicodeスクリプト指定 |
| Unicode注意 | `\d` `\w` は言語とモードで範囲が変わる |
| 集合演算 | Java: `&&`(交差), .NET: `-`(減算), ES2024: `v`フラグ |
| 鉄則 | `[A-z]` は使わない、Unicode挙動を把握する |

---

## 次に読むべきガイド

- [03-quantifiers-anchors.md](./03-quantifiers-anchors.md) -- 量指定子とアンカー
- [../01-advanced/00-groups-backreferences.md](../01-advanced/00-groups-backreferences.md) -- グループと後方参照
- [../01-advanced/02-unicode-regex.md](../01-advanced/02-unicode-regex.md) -- Unicode正規表現の詳細

---

## 参考文献

1. **Unicode Technical Standard #18** "Unicode Regular Expressions" https://unicode.org/reports/tr18/ -- Unicode正規表現の国際標準
2. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- 第5章「文字クラス」の詳細解説
3. **POSIX.1-2017** "Regular Expressions" https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap09.html -- POSIX正規表現の公式仕様
4. **ECMAScript Language Specification** -- Unicode Property Escapes の仕様
5. **Python regex module** https://pypi.org/project/regex/ -- Python 用高機能正規表現モジュール
