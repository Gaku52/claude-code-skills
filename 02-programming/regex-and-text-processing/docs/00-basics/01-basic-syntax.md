# 基本構文 -- リテラル、メタ文字、エスケープ

> 正規表現の最も基礎的な構成要素であるリテラル文字、メタ文字(特殊文字)、エスケープシーケンスの動作原理と正しい使い方を網羅的に解説する。

## この章で学ぶこと

1. **リテラル文字とメタ文字の区別** -- どの文字がそのまま一致し、どの文字が特殊な意味を持つか
2. **エスケープの仕組みと落とし穴** -- バックスラッシュによるメタ文字の無効化と二重エスケープ問題
3. **フラグ(修飾子)による挙動変更** -- 大文字小文字無視、複数行モード、ドットオールモード
4. **各言語でのリテラル表記** -- Python, JavaScript, Java, Perl, Ruby での書き方の違い
5. **マッチングの内部動作** -- エンジンが文字列をどう走査するか

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

### 1.1 リテラルマッチの詳細動作

```python
import re

# リテラルマッチは左から右へ最初に見つかった位置で成功する
text = "abcabcabc"
pattern = r'abc'

# search は最初のマッチを返す
m = re.search(pattern, text)
print(f"最初のマッチ: 位置{m.start()}-{m.end()}")  # => 位置0-3

# findall は全てのマッチを返す
all_matches = re.findall(pattern, text)
print(f"全マッチ: {all_matches}")  # => ['abc', 'abc', 'abc']

# finditer はイテレータで位置情報付き
for m in re.finditer(pattern, text):
    print(f"  位置{m.start()}-{m.end()}: '{m.group()}'")
# => 位置0-3: 'abc'
# => 位置3-6: 'abc'
# => 位置6-9: 'abc'
```

### 1.2 大文字・小文字の扱い

```python
import re

text = "Python is Great. PYTHON IS GREAT. python is great."

# デフォルト: 大文字小文字を区別
print(re.findall(r'python', text))
# => ['python']

# IGNORECASE フラグ: 区別しない
print(re.findall(r'python', text, re.IGNORECASE))
# => ['Python', 'PYTHON', 'python']

# インラインフラグ: パターン内にフラグを埋め込む
print(re.findall(r'(?i)python', text))
# => ['Python', 'PYTHON', 'python']

# 部分的にフラグを適用（Python 3.6+）
# (?i:pattern) でその部分だけ大文字小文字無視
print(re.findall(r'(?i:python) is (?i:great)', text))
# => ['Python is Great', 'PYTHON IS GREAT', 'python is great']
```

### 1.3 マルチバイト文字のリテラルマッチ

```python
import re

# 日本語のリテラルマッチ
text = "東京は日本の首都です。Tokyo is the capital of Japan."

# 日本語文字列もそのままマッチ可能
print(re.findall(r'東京', text))   # => ['東京']
print(re.findall(r'Tokyo', text))  # => ['Tokyo']
print(re.findall(r'首都', text))   # => ['首都']

# 混在テキストでのマッチ
log = "2026-02-15 エラー: ファイルが見つかりません (error: file not found)"
m = re.search(r'エラー', log)
print(m.group())  # => 'エラー'

# 絵文字もリテラルマッチ可能（Python 3）
emoji_text = "Hello! Nice to meet you!"
print(re.findall(r'Nice', emoji_text))  # => ['Nice']
```

### 1.4 各言語でのリテラル表記の違い

```python
# Python: raw string を推奨
import re
pattern = r'hello\.\*world'
re.search(pattern, text)

# Python: re.compile でプリコンパイル
compiled = re.compile(r'hello\.\*world')
compiled.search(text)
```

```javascript
// JavaScript: リテラル記法
const pattern1 = /hello\.\*world/;
pattern1.test(text);

// JavaScript: コンストラクタ記法（動的パターン用）
const pattern2 = new RegExp('hello\\.\\*world');
pattern2.test(text);
// ※ コンストラクタでは文字列のエスケープも必要なため二重エスケープ
```

```java
// Java: 常に文字列リテラル（raw string なし）
import java.util.regex.*;
Pattern pattern = Pattern.compile("hello\\.\\*world");
Matcher matcher = pattern.matcher(text);

// Java 13+: テキストブロックで若干読みやすくなる
// ただしバックスラッシュのエスケープは依然必要
```

```ruby
# Ruby: Regexp リテラル
pattern = /hello\.\*world/
text =~ pattern

# Ruby: Regexp.new（動的パターン用）
pattern = Regexp.new('hello\.\*world')

# Ruby: %r{} 記法（スラッシュが多いパターンに便利）
pattern = %r{http://example\.com/path}
```

```perl
# Perl: パターンマッチ演算子
if ($text =~ /hello\.\*world/) {
    print "マッチ\n";
}

# Perl: qr// でプリコンパイル
my $pattern = qr/hello\.\*world/;
if ($text =~ $pattern) { ... }
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

```python
# ドットの実用パターン

import re

# 1. 任意の1文字を含むパターン
print(re.findall(r'b.g', "bag big bog bug"))
# => ['bag', 'big', 'bog', 'bug']

# 2. 固定長パターンのマッチ
print(re.findall(r'...-....', "Tel: 03-1234-5678"))
# => ['03-1234']  ※意図と異なる可能性

# 3. ドットの正しい使い方: 特定の1文字が不明な場合
# ファイル名パターン: 拡張子の前の任意の1文字
print(re.findall(r'file.\.txt', "file1.txt file2.txt fileA.txt"))
# => ['file1.txt', 'file2.txt', 'fileA.txt']
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

```python
# パイプの優先順位を理解する

import re

# パイプは正規表現で最も優先度が低い演算子
# 連結(隣接文字)の方が優先される

# 例1: abc|def は (abc)|(def) と同じ
print(re.findall(r'abc|def', "abc def abdef"))
# => ['abc', 'def']

# 例2: グループで範囲を制限
print(re.findall(r'gr(a|e)y', "gray grey graey"))
# => ['a', 'e']  ※ キャプチャグループの内容が返る

# 非キャプチャグループを使えばマッチ全体が返る
print(re.findall(r'gr(?:a|e)y', "gray grey graey"))
# => ['gray', 'grey']

# 例3: 複数の選択肢を持つパターン
log_pattern = r'ERROR|WARN|INFO|DEBUG'
log = "2026-02-15 [ERROR] Connection failed"
m = re.search(log_pattern, log)
print(m.group())  # => 'ERROR'

# 例4: NFA エンジンでは左から順に試行される
# 最初にマッチした選択肢で確定
print(re.search(r'Java|JavaScript', "JavaScript").group())
# => 'Java' (先に試行されてマッチ)

# 長い選択肢を先に書くことで対処
print(re.search(r'JavaScript|Java', "JavaScript").group())
# => 'JavaScript'
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

```python
# 量指定子の詳細な動作

import re

# * (0回以上) -- 空マッチに注意
print(re.findall(r'a*', "aaa"))
# => ['aaa', '']  ※ 末尾で空マッチが発生

print(re.findall(r'a*', "bbb"))
# => ['', '', '', '']  ※ 各位置で0回マッチ

# + (1回以上) -- 空マッチは発生しない
print(re.findall(r'a+', "aaa"))
# => ['aaa']

print(re.findall(r'a+', "bbb"))
# => []

# ? (0回または1回) -- オプショナルな要素
print(re.findall(r'https?', "http and https"))
# => ['http', 'https']

# 量指定子の貪欲(greedy)マッチ
# デフォルトでは可能な限り多くの文字を消費する
print(re.search(r'a+', "aaaaaa").group())
# => 'aaaaaa' (全ての 'a' を消費)

# 非貪欲(lazy)マッチ -- ? を付ける
print(re.search(r'a+?', "aaaaaa").group())
# => 'a' (最小限の1文字のみ)

# 非貪欲の実用例: HTML タグ内のテキスト
html = "<b>bold</b> and <i>italic</i>"
print(re.findall(r'<.+>', html))    # 貪欲: ['<b>bold</b> and <i>italic</i>']
print(re.findall(r'<.+?>', html))   # 非貪欲: ['<b>', '</b>', '<i>', '</i>']
```

### 2.4 括弧 `()` -- グループ化とキャプチャ

```python
import re

# 基本的なグループ化
pattern = r'(hello) (world)'
m = re.search(pattern, "say hello world")
print(m.group(0))  # => 'hello world' (マッチ全体)
print(m.group(1))  # => 'hello' (グループ1)
print(m.group(2))  # => 'world' (グループ2)
print(m.groups())  # => ('hello', 'world')

# 非キャプチャグループ (?:...)
# グループ化するがキャプチャしない
pattern = r'(?:hello|hi) (world)'
m = re.search(pattern, "say hello world")
print(m.group(1))  # => 'world' (グループ番号がずれない)

# 名前付きグループ (?P<name>...)
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
m = re.search(pattern, "Date: 2026-02-15")
print(m.group('year'))   # => '2026'
print(m.group('month'))  # => '02'
print(m.group('day'))    # => '15'
print(m.groupdict())     # => {'year': '2026', 'month': '02', 'day': '15'}
```

### 2.5 角括弧 `[]` -- 文字クラス

```python
import re

# 基本的な文字クラス
print(re.findall(r'[aeiou]', "hello world"))
# => ['e', 'o', 'o']  ※ 母音だけをマッチ

# 範囲指定
print(re.findall(r'[a-z]', "Hello 123"))
# => ['e', 'l', 'l', 'o']

print(re.findall(r'[A-Za-z0-9]', "Hello 123!"))
# => ['H', 'e', 'l', 'l', 'o', '1', '2', '3']

# 否定文字クラス
print(re.findall(r'[^a-z]', "hello 123!"))
# => [' ', '1', '2', '3', '!']

# 文字クラス内でのメタ文字の扱い
# ほとんどのメタ文字はリテラルとして扱われる
print(re.findall(r'[.+*?]', "a.b+c*d?e"))
# => ['.', '+', '*', '?']

# ただし以下は特殊:
# ] → 文字クラスの終了（先頭に置くかエスケープ: [\]] or []abc]）
# \ → エスケープ
# ^ → 先頭にある場合のみ否定
# - → 文字間にある場合のみ範囲指定（先頭/末尾ならリテラル）
```

### 2.6 波括弧 `{}` -- 量指定子

```python
import re

# {n} 正確にn回
print(re.findall(r'\d{3}', "12 123 1234 12345"))
# => ['123', '123', '123']  ※ 1234 から 123 を抽出、12345 から 123 と 45は別

# {n,m} n回以上m回以下
print(re.findall(r'\d{2,4}', "1 12 123 1234 12345"))
# => ['12', '123', '1234', '1234']

# {n,} n回以上
print(re.findall(r'\d{3,}', "1 12 123 1234 12345"))
# => ['123', '1234', '12345']

# {,m} 0回以上m回以下 (= {0,m})
print(re.findall(r'a{,3}', "aaaa"))
# => ['aaa', 'a', '']  ※ 最大3つの 'a' にマッチ

# 実用例: 郵便番号
print(re.findall(r'\d{3}-\d{4}', "〒100-0001 東京都千代田区"))
# => ['100-0001']

# 実用例: IPv4 アドレス（簡易版）
print(re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
                 "Server: 192.168.1.1, Gateway: 10.0.0.1"))
# => ['192.168.1.1', '10.0.0.1']
```

### 2.7 キャレット `^` とドル記号 `$` -- アンカー

```python
import re

# ^ 行頭にマッチ
print(re.search(r'^hello', "hello world").group())    # => 'hello'
print(re.search(r'^hello', "say hello"))              # => None

# $ 行末にマッチ
print(re.search(r'world$', "hello world").group())    # => 'world'
print(re.search(r'world$', "world hello"))            # => None

# ^ と $ を組み合わせて全体マッチ
print(re.match(r'^\d{3}-\d{4}$', "100-0001"))
# => <re.Match object; ...>  マッチ成功

print(re.match(r'^\d{3}-\d{4}$', "100-0001 東京"))
# => None  (末尾に余分な文字があるため)

# 複数行モードでの ^ と $
text = """line1
line2
line3"""

# デフォルト: ^ は文字列全体の先頭のみ
print(re.findall(r'^line\d', text))
# => ['line1']

# MULTILINE: ^ が各行の先頭にマッチ
print(re.findall(r'^line\d', text, re.MULTILINE))
# => ['line1', 'line2', 'line3']
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

```python
# エスケープが必要な実用パターン

import re

# 1. ファイルパス（Windows）
path = r'C:\Users\gaku\Documents\file.txt'
pattern = r'C:\\Users\\(\w+)\\Documents\\(\w+\.txt)'
m = re.search(pattern, path)
if m:
    print(f"ユーザー: {m.group(1)}, ファイル: {m.group(2)}")
    # => ユーザー: gaku, ファイル: file.txt

# 2. URL パターン
url = "https://example.com/path?key=value&key2=value2"
pattern = r'https?://([^/]+)(/[^?]*)?\?(.+)'
m = re.search(pattern, url)
if m:
    print(f"ホスト: {m.group(1)}")  # => example.com
    print(f"パス: {m.group(2)}")    # => /path
    print(f"クエリ: {m.group(3)}")  # => key=value&key2=value2

# 3. 数学的な式
expr = "f(x) = 3x^2 + 2x + 1"
pattern = r'f\(x\) = (\d+)x\^(\d+)'
m = re.search(pattern, expr)
if m:
    print(f"係数: {m.group(1)}, 指数: {m.group(2)}")
    # => 係数: 3, 指数: 2

# 4. IPアドレス（ドットのエスケープ重要）
ip_text = "Server 192.168.1.1 Port 8080"
pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
m = re.search(pattern, ip_text)
print(m.group())  # => '192.168.1.1'
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

```python
# 二重エスケープ問題が発生しやすいケース

import re

# ケース1: バックスラッシュ自体をマッチ
# 目的: テキスト中の \ を見つける
text = "path\\to\\file"

# NG: 通常文字列
# "\\" → Python文字列: \ → 正規表現: エスケープ不完全
# OK: raw string
pattern = r'\\'  # raw string: \\ → 正規表現: リテラル \
print(re.findall(pattern, text))  # => ['\\', '\\']

# ケース2: \n をリテラル文字列としてマッチ
# 目的: テキスト中の文字列 "\n" (バックスラッシュ+n) を見つける
text = r"改行は\nで表します"

# NG: "\n" → Python文字列: 改行文字 → 改行にマッチしてしまう
# OK:
pattern = r'\\n'  # raw string: \\n → 正規表現: リテラル \ + n
print(re.findall(pattern, text))  # => ['\\n']

# ケース3: Java / JavaScript では更に注意
# Java: Pattern.compile("\\\\n") → \\ → リテラル \ + n
# JavaScript: /\\n/ → リテラル \ + n
# JavaScript: new RegExp("\\\\n") → 文字列エスケープ + 正規表現エスケープ
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
  \f  → フォームフィード
  \v  → 垂直タブ
  \0  → NULL文字
  \a  → ベル文字
  \e  → エスケープ文字 (ESC, 0x1B) ※一部エンジンのみ

数値指定:
  \xHH    → 16進数で指定 (例: \x41 = 'A')
  \uHHHH  → Unicode BMP (例: \u3042 = 'あ')
  \UHHHHHHHH → Unicode (例: \U0001F600 = 絵文字)
  \N{name}   → Unicode名 (例: \N{SNOWMAN} = '☃') ※Python
  \oOOO   → 8進数で指定 (例: \o101 = 'A')
```

```python
import re

# ショートハンドの動作確認

# \d: 数字
print(re.findall(r'\d+', "abc123def456"))
# => ['123', '456']

# \w: 単語文字
print(re.findall(r'\w+', "hello, world! 123"))
# => ['hello', 'world', '123']

# \s: 空白文字
text = "hello\tworld\nnext line"
print(re.findall(r'\s', text))
# => ['\t', '\n', ' ']

# \b: 単語境界
text = "cat caterpillar concatenate"
print(re.findall(r'\bcat\b', text))
# => ['cat']  ※ 完全一致のみ

print(re.findall(r'\bcat', text))
# => ['cat', 'cat', 'cat']  ※ cat で始まる単語

# 大文字版は否定
print(re.findall(r'\D+', "abc123def"))  # 非数字
# => ['abc', 'def']

print(re.findall(r'\W+', "hello, world!"))  # 非単語文字
# => [', ', '!']

print(re.findall(r'\S+', "hello world"))  # 非空白文字
# => ['hello', 'world']
```

### 3.4 re.escape() による自動エスケープ

```python
import re

# ユーザー入力をリテラルとしてパターンに組み込む場合
user_input = "file (1).txt"

# NG: そのまま使うとメタ文字が解釈される
try:
    re.search(user_input, "file (1).txt")  # () がグループとして解釈
except re.error as e:
    print(f"エラー: {e}")

# OK: re.escape() でメタ文字をエスケープ
escaped = re.escape(user_input)
print(escaped)  # => 'file\\ \\(1\\)\\.txt'
m = re.search(escaped, "file (1).txt")
print(m.group())  # => 'file (1).txt'

# 実用例: ユーザー入力のリテラル検索
def search_literal(text, query):
    """ユーザー入力をリテラルとして検索"""
    pattern = re.escape(query)
    return re.findall(pattern, text)

print(search_literal("price is $10.00", "$10.00"))
# => ['$10.00']

# 実用例: リテラルで囲まれた部分を抽出
def extract_between(text, start, end):
    """start と end の間の文字列を抽出"""
    pattern = re.escape(start) + r'(.+?)' + re.escape(end)
    return re.findall(pattern, text)

print(extract_between("value = [hello]", "[", "]"))
# => ['hello']
```

### 3.5 各言語でのエスケープ関数

```python
# Python
import re
re.escape("hello.world")  # => 'hello\\.world'
```

```javascript
// JavaScript (標準にはないが、よく使われるユーティリティ)
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
escapeRegExp("hello.world");  // => "hello\\.world"
```

```java
// Java
java.util.regex.Pattern.quote("hello.world");
// => "\\Qhello.world\\E" (リテラルブロックで囲む)
```

```ruby
# Ruby
Regexp.escape("hello.world")  # => "hello\\.world"
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

| フラグ | Python | JavaScript | Perl | Java | 効果 |
|--------|--------|------------|------|------|------|
| 大文字小文字無視 | `re.IGNORECASE` / `re.I` | `/i` | `/i` | `CASE_INSENSITIVE` | 大文字小文字を区別しない |
| 複数行 | `re.MULTILINE` / `re.M` | `/m` | `/m` | `MULTILINE` | `^` `$` が各行の先頭・末尾にマッチ |
| ドットオール | `re.DOTALL` / `re.S` | `/s` | `/s` | `DOTALL` | `.` が改行にもマッチ |
| 冗長モード | `re.VERBOSE` / `re.X` | 非対応 | `/x` | `COMMENTS` | 空白・コメントを無視 |
| Unicode | `re.UNICODE` / `re.U` | `/u` | デフォルト | `UNICODE_CHARACTER_CLASS` | Unicode対応 |
| グローバル | N/A (`findall`) | `/g` | `/g` | N/A (`Matcher.find()`) | 全マッチを返す |
| スティッキー | N/A | `/y` | N/A | N/A | lastIndex位置からのみマッチ |
| ASCII | `re.ASCII` / `re.A` | N/A | `/a` | N/A | \d \w \s を ASCII のみに限定 |

### 4.3 インラインフラグ

```python
import re

# パターン内にフラグを埋め込む
# (?flags) の形式

# (?i) 大文字小文字無視
print(re.findall(r'(?i)hello', "Hello HELLO hello"))
# => ['Hello', 'HELLO', 'hello']

# (?m) 複数行モード
text = "line1\nline2\nline3"
print(re.findall(r'(?m)^\w+', text))
# => ['line1', 'line2', 'line3']

# (?s) ドットオール
print(re.search(r'(?s)line1.+line3', text).group())
# => 'line1\nline2\nline3'

# (?x) 冗長モード
pattern = r'''(?x)
    (\d{4})     # 年
    -(\d{2})    # 月
    -(\d{2})    # 日
'''
m = re.search(pattern, "2026-02-15")
print(m.groups())  # => ('2026', '02', '15')

# 複数フラグの組み合わせ
print(re.findall(r'(?im)^hello', "Hello\nhello\nHELLO"))
# => ['Hello', 'hello', 'HELLO']

# スコープ付きフラグ (Python 3.6+)
# (?i:pattern) でその部分だけフラグ適用
pattern = r'(?i:hello) world'  # hello は大文字小文字無視、world は区別
print(re.findall(pattern, "Hello world HELLO world hello World"))
# => ['Hello world', 'hello world']  ※ 'HELLO world' はマッチ、'hello World' は不一致
# 実際には:
# 'Hello world' → マッチ
# 'HELLO world' → マッチ
# 'hello World' → 不一致（world が大文字のため）
```

### 4.4 フラグの実用パターン

```python
import re

# 1. ログファイルの解析（複数行 + 大文字小文字無視）
log = """
2026-02-15 10:30:00 [ERROR] Database connection failed
2026-02-15 10:31:00 [Warning] High memory usage
2026-02-15 10:32:00 [error] Disk space low
"""

# ERROR も Warning も error もマッチ
errors = re.findall(
    r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[(?:error|warning)\] (.+)$',
    log,
    re.MULTILINE | re.IGNORECASE
)
print(errors)
# => ['Database connection failed', 'High memory usage', 'Disk space low']

# 2. 複雑なパターンの可読性向上（冗長モード）
email_pattern = re.compile(r'''
    ^                       # 文字列の先頭
    [a-zA-Z0-9._%+-]+      # ローカルパート
    @                       # アットマーク
    [a-zA-Z0-9.-]+          # ドメイン名
    \.                      # ドット
    [a-zA-Z]{2,}            # TLD
    $                       # 文字列の末尾
''', re.VERBOSE)

# 3. 複数行テキストの解析（ドットオール + 複数行）
html = """<div class="content">
    <p>First paragraph</p>
    <p>Second paragraph</p>
</div>"""

# div の中身全体を抽出
m = re.search(r'<div[^>]*>(.*?)</div>', html, re.DOTALL)
if m:
    print(m.group(1).strip())
```

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

### 5.4 量指定子の動作比較

```
パターン: a{2,4}
テキスト: "aaaaaa"

貪欲マッチ (デフォルト):
  位置0: a{2,4} → "aaaa" (最大4文字を消費)
  位置4: a{2,4} → "aa"   (残り2文字を消費)
  結果: ["aaaa", "aa"]

非貪欲マッチ (a{2,4}?):
  位置0: a{2,4}? → "aa" (最小2文字を消費)
  位置2: a{2,4}? → "aa" (最小2文字を消費)
  位置4: a{2,4}? → "aa" (最小2文字を消費)
  結果: ["aa", "aa", "aa"]
```

```
パターン: <.+> vs <.+?>
テキスト: "<b>bold</b>"

貪欲 <.+>:
  < にマッチ → . が貪欲に "b>bold</b" を消費 → > にマッチ
  結果: "<b>bold</b>" (1つの大きなマッチ)

非貪欲 <.+?>:
  < にマッチ → . が最小の "b" を消費 → > にマッチ
  結果: "<b>" (最小のマッチ)
  続行: "<" にマッチ → "..." → 結果: "</b>"
```

---

## 6. 実践的なパターン例

### 6.1 基本的なバリデーションパターン

```python
import re

# 日本の郵便番号
postal_code = re.compile(r'^\d{3}-\d{4}$')
assert postal_code.match('100-0001')
assert not postal_code.match('1000001')
assert not postal_code.match('100-000')

# 日本の携帯電話番号
mobile_phone = re.compile(r'^0[789]0-\d{4}-\d{4}$')
assert mobile_phone.match('090-1234-5678')
assert mobile_phone.match('080-1234-5678')
assert not mobile_phone.match('03-1234-5678')

# 西暦日付 (YYYY-MM-DD)
date_pattern = re.compile(r'^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$')
assert date_pattern.match('2026-02-15')
assert date_pattern.match('2026-12-31')
assert not date_pattern.match('2026-13-01')
assert not date_pattern.match('2026-00-15')

# 時刻 (HH:MM:SS)
time_pattern = re.compile(r'^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$')
assert time_pattern.match('00:00:00')
assert time_pattern.match('23:59:59')
assert not time_pattern.match('24:00:00')
assert not time_pattern.match('12:60:00')
```

### 6.2 テキスト抽出パターン

```python
import re

# Markdown のリンクを抽出
text = "詳細は[公式サイト](https://example.com)と[FAQ](https://example.com/faq)を参照"
links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)
for label, url in links:
    print(f"  {label} → {url}")
# => 公式サイト → https://example.com
# => FAQ → https://example.com/faq

# ハッシュタグの抽出
tweet = "今日は天気がいい #sunny #tokyo 最高！"
tags = re.findall(r'#(\w+)', tweet)
print(tags)  # => ['sunny', 'tokyo']

# 引用符で囲まれた文字列
text = 'She said "hello" and "goodbye"'
quoted = re.findall(r'"([^"]*)"', text)
print(quoted)  # => ['hello', 'goodbye']

# キーバリューペアの抽出
config = "host=localhost port=3306 db=mydb user=admin"
pairs = re.findall(r'(\w+)=(\S+)', config)
print(dict(pairs))  # => {'host': 'localhost', 'port': '3306', 'db': 'mydb', 'user': 'admin'}
```

### 6.3 テキスト置換パターン

```python
import re

# 1. スネークケース → キャメルケース
def snake_to_camel(name):
    return re.sub(r'_([a-z])', lambda m: m.group(1).upper(), name)

print(snake_to_camel('hello_world'))    # => 'helloWorld'
print(snake_to_camel('my_var_name'))    # => 'myVarName'

# 2. キャメルケース → スネークケース
def camel_to_snake(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

print(camel_to_snake('helloWorld'))    # => 'hello_world'
print(camel_to_snake('myVarName'))     # => 'my_var_name'

# 3. 連続する空白の正規化
text = "hello   world\t\t\tnext"
print(re.sub(r'\s+', ' ', text))  # => 'hello world next'

# 4. HTMLタグの除去
html = "<p>Hello <b>World</b></p>"
print(re.sub(r'<[^>]+>', '', html))  # => 'Hello World'

# 5. 日付フォーマットの変換 (YYYY/MM/DD → YYYY-MM-DD)
date = "2026/02/15"
print(re.sub(r'(\d{4})/(\d{2})/(\d{2})', r'\1-\2-\3', date))
# => '2026-02-15'

# 6. 後方参照を使った重複語の検出と修正
text = "the the quick brown fox fox"
print(re.sub(r'\b(\w+)\s+\1\b', r'\1', text))
# => 'the quick brown fox'
```

---

## 7. アンチパターン

### 7.1 アンチパターン: raw string を使わない

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

### 7.2 アンチパターン: ドットの過度な使用

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

### 7.3 アンチパターン: 不必要に複雑なパターン

```python
import re

# NG: 正規表現で書く必要がないケース

# 単純な文字列検索は in 演算子で十分
text = "hello world"

# NG
if re.search(r'hello', text):
    pass

# OK (高速かつ明快)
if 'hello' in text:
    pass

# NG: 先頭/末尾のチェックに正規表現
if re.match(r'^hello', text):
    pass

# OK
if text.startswith('hello'):
    pass

# NG: 固定文字列の置換
re.sub(r'hello', 'hi', text)

# OK
text.replace('hello', 'hi')
```

### 7.4 アンチパターン: match() と search() の混同

```python
import re

text = "say hello world"

# match() は文字列の先頭からのみマッチを試みる
m = re.match(r'hello', text)
print(m)  # => None  ※ 先頭が 'say' なので不一致

# search() は文字列全体を検索する
m = re.search(r'hello', text)
print(m.group())  # => 'hello'

# fullmatch() は文字列全体が一致するかを確認
m = re.fullmatch(r'hello', "hello")
print(m.group())  # => 'hello'

m = re.fullmatch(r'hello', "hello world")
print(m)  # => None  ※ 末尾に余分な文字があるため

# 全体マッチのベストプラクティス:
# 入力検証には fullmatch() を使用（Python 3.4+）
# テキスト検索には search() を使用
# 行頭マッチには match() を使用
```

---

## 8. パフォーマンスのヒント

### 8.1 パターンのプリコンパイル

```python
import re
import time

text_lines = [f"line {i}: some text here" for i in range(100000)]

# NG: ループ内で毎回文字列パターンを使用
start = time.time()
for line in text_lines:
    re.search(r'\d+', line)
print(f"未コンパイル: {time.time() - start:.3f}s")

# OK: プリコンパイル
pattern = re.compile(r'\d+')
start = time.time()
for line in text_lines:
    pattern.search(line)
print(f"コンパイル済み: {time.time() - start:.3f}s")

# ※ Python の re モジュールは内部キャッシュ(最大512パターン)を持つため
#    少数のパターンを繰り返し使う場合は差が小さいが、
#    明示的なコンパイルは意図を明確にする
```

### 8.2 効率的なパターンの書き方

```python
import re

# 1. 文字クラスは選択肢より高速
# NG
pattern_slow = r'a|b|c|d|e'
# OK
pattern_fast = r'[a-e]'

# 2. 非キャプチャグループでメモリ節約
# NG (キャプチャが不要な場合)
pattern_slow = r'(foo|bar|baz)+'
# OK
pattern_fast = r'(?:foo|bar|baz)+'

# 3. アンカーで検索範囲を限定
# NG
pattern_slow = r'error'  # 文字列全体を走査
# OK (エラーが行頭にある場合)
pattern_fast = r'^error'  # 各行の先頭のみチェック

# 4. 具体的なパターンが優先
# NG
pattern_slow = r'.+@.+\..+'  # 曖昧すぎる
# OK
pattern_fast = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# 5. 選択肢は長いものから先に
# NG (短い選択肢が先にマッチして問題になるケース)
pattern_slow = r'Java|JavaScript'
# OK
pattern_fast = r'JavaScript|Java'
```

---

## 9. FAQ

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

### Q4: \d は全角数字にもマッチするか？

**A**: エンジンとフラグによる:

```python
import re

# Python のデフォルト(Unicode モード)
print(re.findall(r'\d+', "半角123 全角１２３"))
# => ['123', '１２３']  ※ 全角数字にもマッチする！

# ASCII モードに制限
print(re.findall(r'\d+', "半角123 全角１２３", re.ASCII))
# => ['123']  ※ 半角数字のみ

# JavaScript の /u フラグ
# /\d+/u は Unicode 数字にもマッチ
# /\d+/ は ASCII 数字のみ（エンジンによる）
```

### Q5: match() と search() と fullmatch() の違いは？

**A**:

```python
import re

text = "hello world"

# match(): 文字列の先頭からマッチを試みる
re.match(r'hello', text)       # => マッチ
re.match(r'world', text)       # => None

# search(): 文字列全体を検索
re.search(r'hello', text)      # => マッチ
re.search(r'world', text)      # => マッチ

# fullmatch(): 文字列全体がパターンと一致するか
re.fullmatch(r'hello', text)   # => None
re.fullmatch(r'hello world', text)  # => マッチ
```

### Q6: 正規表現でコメントを書くには？

**A**: verbose モード (`re.VERBOSE` / `re.X`) を使用:

```python
import re

pattern = re.compile(r'''
    ^                   # 文字列の先頭
    (?P<protocol>       # プロトコル部分
        https?          #   http または https
    )
    ://                 # スキーム区切り
    (?P<host>           # ホスト部分
        [^/]+           #   スラッシュ以外の1文字以上
    )
    (?P<path>           # パス部分 (オプション)
        /[^\s]*         #   スラッシュから始まる
    )?
    $                   # 文字列の末尾
''', re.VERBOSE)

m = pattern.match('https://example.com/path/to/page')
if m:
    print(m.groupdict())
    # => {'protocol': 'https', 'host': 'example.com', 'path': '/path/to/page'}
```

verbose モード内でリテラルの空白が必要な場合は `\ ` またはクラス `[ ]` を使用する。

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
| インラインフラグ | `(?i)` `(?m)` `(?s)` `(?x)` でパターン内にフラグを埋め込む |
| プリコンパイル | `re.compile()` でパターンオブジェクトを作成し再利用 |
| re.escape() | ユーザー入力のメタ文字を自動エスケープ |
| 鉄則 | 常に raw string を使い、ドットは必要なときだけ使う |

---

## 次に読むべきガイド

- [02-character-classes.md](./02-character-classes.md) -- 文字クラス `[abc]`、`\d`、`\w`、`\s`、POSIX クラス
- [03-quantifiers-anchors.md](./03-quantifiers-anchors.md) -- 量指定子・アンカーの詳細

---

## 参考文献

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions, 3rd Edition" O'Reilly Media, 2006 -- 第3章「基本構文」が特に参考になる
2. **Python re module documentation** https://docs.python.org/3/library/re.html -- Python 正規表現の公式リファレンス
3. **MDN Web Docs - Regular Expressions** https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions -- JavaScript 正規表現の包括的ガイド
4. **Java Pattern class documentation** https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/regex/Pattern.html -- Java 正規表現の公式リファレンス
5. **Ruby Regexp documentation** https://docs.ruby-lang.org/en/3.2/Regexp.html -- Ruby 正規表現の公式リファレンス
