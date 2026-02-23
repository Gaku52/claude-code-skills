# 先読み・後読み -- (?=)(?!)(?<=)(?<!)

> 先読み(Lookahead)と後読み(Lookbehind)はゼロ幅アサーションであり、文字を消費せずに位置条件を指定する。パスワード強度検証、複雑な抽出条件、置換対象の限定など、通常のパターンでは表現しにくい制約を可能にする強力な機能である。

## この章で学ぶこと

1. **4種類のルックアラウンドの構文と動作** -- 肯定/否定の先読み/後読みの正確な意味
2. **ゼロ幅アサーションの概念** -- 文字を消費しないマッチの仕組みと応用
3. **実践的なユースケース** -- パスワード検証、数値フォーマット、複合条件抽出
4. **各言語での動作差異と制約** -- Python, JavaScript, Java, Go, Rust における実装の違い
5. **パフォーマンスへの影響と最適化** -- ルックアラウンドがもたらすコストと回避策
6. **ネストされたルックアラウンド** -- 複雑な条件を実現する高度なテクニック
7. **テスト駆動でのパターン構築** -- ルックアラウンドパターンを安全に開発する手法

---

## 1. ルックアラウンドの4種類

### 1.1 一覧

```
┌──────────────────────────────────────────────────┐
│              ルックアラウンド一覧                   │
├────────────┬──────────────┬───────────────────────┤
│            │  肯定 (=)    │  否定 (!)              │
├────────────┼──────────────┼───────────────────────┤
│ 先読み(→)  │ (?=pattern)  │ (?!pattern)           │
│ Lookahead  │ 後ろにある   │ 後ろにない             │
├────────────┼──────────────┼───────────────────────┤
│ 後読み(←)  │ (?<=pattern) │ (?<!pattern)          │
│ Lookbehind │ 前にある     │ 前にない               │
└────────────┴──────────────┴───────────────────────┘
```

### 1.2 概念図

```
テキスト: "price: $100"

         p r i c e :   $ 1 0 0
                        ↑
                      現在位置

先読み (?=...):  「この位置の右側(後ろ)に...がある」
後読み (?<=...): 「この位置の左側(前)に...がある」

例: (?<=\$)\d+
  → 「左側に $ がある位置から始まる数字列」
  → "100" にマッチ ($は含まない)
```

### 1.3 ルックアラウンドの内部動作メカニズム

ルックアラウンドの動作を正確に理解するために、正規表現エンジンが内部でどのように処理しているかを詳しく見てみる。

```
NFAエンジンのルックアラウンド処理フロー:

1. エンジンが現在位置を記録 (position = P)
2. ルックアラウンド内のパターンをマッチ試行
   - 肯定: マッチ成功 → アサーション成功
   - 否定: マッチ失敗 → アサーション成功
3. 現在位置を P に復元 (ゼロ幅: 位置が進まない)
4. メインパターンの次の要素に進む

具体例: パターン (?<=\$)\d+ を "price: $100" に適用

Step 1: 位置0 'p' -- (?<=\$) チェック
  左側に何もない → 失敗 → 位置を1つ進める

Step 2: 位置1 'r' -- (?<=\$) チェック
  左側 'p' は '$' ではない → 失敗 → 位置を1つ進める

...

Step 8: 位置8 '1' -- (?<=\$) チェック
  左側 '$' は '$' である → 成功!
  → \d+ を位置8から試行
  → '1','0','0' がマッチ
  → 結果: "100" (位置8-11)
```

### 1.4 ルックアラウンドと他のゼロ幅アサーションの比較

```
ゼロ幅アサーション一覧:

アサーション        構文        意味
──────────────     ──────     ──────────────────────────
行頭              ^           行の先頭位置
行末              $           行の末尾位置
単語境界          \b          単語文字と非単語文字の境界
非単語境界        \B          \b でない位置
文字列先頭        \A          文字列全体の先頭(複数行でも)
文字列末尾        \Z, \z     文字列全体の末尾
肯定先読み        (?=...)     右側にパターンがある位置
否定先読み        (?!...)     右側にパターンがない位置
肯定後読み        (?<=...)    左側にパターンがある位置
否定後読み        (?<!...)    左側にパターンがない位置

共通の特徴: すべて「位置」をチェックし、文字を消費しない
```

```python
import re

# 各種ゼロ幅アサーションの比較
text = "Hello World 123"

# ^ -- 行の先頭
print(re.findall(r'^.', text))          # => ['H']

# \b -- 単語境界
print(re.findall(r'\b\w', text))        # => ['H', 'W', '1']

# (?=...) -- 肯定先読み
print(re.findall(r'\w(?=\s)', text))    # => ['o', 'd']

# (?<=...) -- 肯定後読み
print(re.findall(r'(?<=\s)\w', text))   # => ['W', '1']

# すべてゼロ幅: 文字を消費しない
```

---

## 2. 肯定先読み `(?=pattern)`

### 2.1 基本動作

```python
import re

# 「後ろに "円" が続く数字」を抽出
pattern = r'\d+(?=円)'
text = "商品A: 1000円、商品B: 2500円、商品C: 30ドル"

print(re.findall(pattern, text))
# => ['1000', '2500']
# 注: "30" はマッチしない(後ろが「ドル」)
# 注: "円" 自体はマッチに含まれない(ゼロ幅)
```

### 2.2 ゼロ幅の証明

```python
import re

text = "100円"

# 先読みなし: 数字 + 円 を含む
m1 = re.search(r'\d+円', text)
print(m1.group())   # => '100円' (円を含む)
print(m1.end())     # => 4

# 先読みあり: 数字のみ(円は消費しない)
m2 = re.search(r'\d+(?=円)', text)
print(m2.group())   # => '100' (円を含まない)
print(m2.end())     # => 3 (ゼロ幅: 位置は円の直前)
```

### 2.3 先読みの複合条件

複数の先読みを連鎖させることで、AND 条件を実現できる。

```python
import re

# 複数の肯定先読みで AND 条件を構築
# 「大文字を含み、かつ数字を含み、かつ6文字以上の単語」
pattern = r'\b(?=\w*[A-Z])(?=\w*\d)\w{6,}\b'
text = "Hello World Pass1word abc123 Test99 MyPW short A1"

print(re.findall(pattern, text))
# => ['Pass1word', 'Test99']
# "Hello" -- 数字を含まない → NG
# "abc123" -- 大文字を含まない → NG
# "MyPW" -- 4文字 < 6 → NG
# "A1" -- 2文字 < 6 → NG
```

### 2.4 先読みを使った重複マッチ

通常の `findall` では重複しないマッチしか返さないが、先読みを使えば重複するパターンも検出できる。

```python
import re

# 通常のパターン: 重複なし
text = "abcabc"
print(re.findall(r'ab', text))
# => ['ab', 'ab']

# 先読みで重複マッチを検出
text = "aaaa"
# 通常: 連続する "aa" を検出（重複なし）
print(re.findall(r'aa', text))
# => ['aa', 'aa']  -- 位置0と位置2

# 先読みで全ての "aa" の開始位置を検出（重複あり）
print(re.findall(r'(?=aa)', text))
# => ['', '', '']  -- 位置0, 1, 2 の3箇所

# 実用例: テキスト内の重複部分文字列を検出
text = "abcabcabc"
positions = [(m.start(), m.end()) for m in re.finditer(r'(?=(abc))', text)]
print(positions)
# => [(0, 0), (3, 3), (6, 6)]
# 先読みの中のキャプチャグループで内容を取得
print(re.findall(r'(?=(abc))', text))
# => ['abc', 'abc', 'abc']

# 実用例: DNA配列の重複モチーフ検出
dna = "ATGATGATG"
# 通常: "ATG" は重複なしで3つ
print(re.findall(r'ATG', dna))  # => ['ATG', 'ATG', 'ATG']
# 重複パターンの検出: "ATGATG" の出現位置
overlapping = [m.start() for m in re.finditer(r'(?=ATGATG)', dna)]
print(overlapping)  # => [0, 3]
```

### 2.5 先読みと量指定子の相互作用

```python
import re

# 先読みと貪欲/非貪欲の動作
text = "abc123def456"

# 貪欲: できるだけ長くマッチ
print(re.findall(r'\w+(?=\d)', text))
# => ['abc12', 'def45']
# 注: \w+ が貪欲なので、最後の数字の直前まで食べる

# 非貪欲: できるだけ短くマッチ
print(re.findall(r'\w+?(?=\d)', text))
# => ['abc', '1', '2', 'def', '4', '5']
# 注: 1文字ずつマッチしようとする

# 意図通りの結果を得るには適切なパターン設計が必要
# 「英字の後に数字が続く」場合
print(re.findall(r'[a-z]+(?=\d)', text))
# => ['abc', 'def']
```

---

## 3. 否定先読み `(?!pattern)`

### 3.1 基本動作

```python
import re

# 「後ろに "ドル" が続かない数字」
pattern = r'\d+(?!ドル|\d)'
text = "100円 200ドル 300ユーロ"

print(re.findall(pattern, text))
# => ['100', '300']
# "200" はマッチしない(後ろが「ドル」)
```

### 3.2 除外パターン

```python
import re

# 特定の単語を除外してマッチ
# "test" で始まらない単語を抽出
pattern = r'\b(?!test)\w+'
text = "testing hello testcase world testify"

print(re.findall(pattern, text))
# => ['hello', 'world']

# JavaScript 予約語以外の識別子
reserved = r'\b(?!if|else|for|while|return|function\b)\w+'
code = "function hello if world return value"
print(re.findall(reserved, code))
# => ['hello', 'world', 'value']
```

### 3.3 否定先読みの実践的活用パターン

```python
import re

# パターン1: 特定の拡張子を除外したファイル名のマッチ
files = "main.py config.yaml app.js test.pyc utils.py data.json"
# .pyc 以外の .py ファイルを抽出
pattern = r'\b\w+\.py(?!c)\b'
print(re.findall(pattern, files))
# => ['main.py', 'utils.py']

# パターン2: 特定のドメインを除外したURL抽出
urls = "http://example.com http://spam.evil.com http://good-site.org"
# spam を含まない URL を抽出
pattern = r'https?://(?!spam)\S+'
print(re.findall(pattern, urls))
# => ['http://example.com', 'http://good-site.org']

# パターン3: コメント行でない行を抽出
lines = """
# これはコメント
data = 123
// これもコメント
result = data + 1
"""
# # や // で始まらない非空行
pattern = r'^(?!#|//)(?!\s*$).+'
print(re.findall(pattern, lines.strip(), re.MULTILINE))
# => ['data = 123', 'result = data + 1']
```

### 3.4 否定先読みによるパスワードの禁止パターン

```python
import re

def validate_no_common_patterns(password: str) -> tuple[bool, list[str]]:
    """パスワードに一般的な弱いパターンが含まれていないか検証"""
    errors = []

    # 連続する同一文字を禁止 (aaa, 111 など)
    if re.search(r'(.)\1{2,}', password):
        errors.append("同じ文字が3回以上連続しています")

    # 連続する順序文字を禁止 (abc, 123 など)
    sequential_patterns = [
        'abc', 'bcd', 'cde', 'def', 'efg', 'fgh',
        '123', '234', '345', '456', '567', '678', '789',
        'qwerty', 'asdf', 'zxcv'
    ]
    for seq in sequential_patterns:
        if seq in password.lower():
            errors.append(f"連続パターン '{seq}' が含まれています")

    # 強力なパスワード: 全条件を1パターンで表現
    # - 8文字以上
    # - 連続3文字の同一文字を含まない
    # - "password" を含まない
    # - "12345" を含まない
    strong_pattern = re.compile(
        r'^(?!.*(.)\1{2})'     # 同一文字3連続なし
        r'(?!.*password)'       # "password" を含まない
        r'(?!.*12345)'          # "12345" を含まない
        r'.{8,}$',              # 8文字以上
        re.IGNORECASE
    )

    if not strong_pattern.match(password):
        errors.append("パスワードが弱すぎます")

    return (len(errors) == 0, errors)

# テスト
test_passwords = [
    "Str0ng!Pass",    # OK
    "password123!",   # "password" を含む
    "aaabbb1234!",   # 連続文字
    "P@ss12345word",  # "12345" を含む
    "Sh0rt!",         # 短すぎる
]

for pw in test_passwords:
    valid, errs = validate_no_common_patterns(pw)
    status = "OK" if valid else "NG"
    print(f"  {pw}: {status} {errs if errs else ''}")
```

### 3.5 否定先読みのよくある落とし穴

```python
import re

# 落とし穴1: 否定先読みの位置に注意
# 意図: "test" という単語だけを除外
text = "testing test tested"

# 間違い: \b(?!test)\w+ は "test" の一部もマッチする
print(re.findall(r'\b(?!test)\w+', text))
# => ['esting', 'ed']  -- "testing" の "esting" がマッチ!

# 正解: 単語全体を除外するには \b も使う
print(re.findall(r'\b(?!test\b)\w+', text))
# => ['testing', 'tested']
# "test" 完全一致のみ除外、"testing" と "tested" はOK

# 落とし穴2: 否定先読みと量指定子の組み合わせ
text = "foobar foobaz foo"

# 意図: "foo" の後に "bar" が続かないものを抽出
# 間違い: マッチの範囲がおかしくなる
print(re.findall(r'foo(?!bar)', text))
# => ['foo', 'foo']  -- "foobaz" の "foo" と "foo"

# 続く部分も含めたい場合
print(re.findall(r'foo(?!bar)\w*', text))
# => ['foobaz', 'foo']

# 落とし穴3: 空文字列マッチに注意
text = "abc"
print(re.findall(r'(?!abc)', text))
# => ['', '', '']  -- 位置1,2,3でマッチ(位置0は 'abc' と一致するので除外)
# ゼロ幅なので空文字列がマッチする
```

---

## 4. 肯定後読み `(?<=pattern)`

### 4.1 基本動作

```python
import re

# 「前に "$" がある数字」を抽出
pattern = r'(?<=\$)\d+'
text = "Price: $100, Tax: $15, Total: 115"

print(re.findall(pattern, text))
# => ['100', '15']
# "115" はマッチしない(前に $ がない)
```

### 4.2 後読みの制約

```
後読みの幅制約(エンジンによる):

エンジン        可変長後読み    制約
──────────      ────────────   ──────
Python re       不可           固定長のみ
JavaScript      可能(ES2018+)  制限なし
Java            不可           固定長のみ
.NET            可能           制限なし
Perl            不可           固定長のみ
PCRE2           可能           制限なし
Ruby            不可           固定長のみ (Onigmo)
PHP (PCRE)      不可           固定長のみ (PCRE1時代)

固定長の制約:
  (?<=abc)    OK  -- 3文字固定
  (?<=ab|cd)  OK  -- 各選択肢が同じ長さ
  (?<=a{3})   OK  -- 固定回数
  (?<=a+)     NG  -- 可変長 (Python, Java, Perl で不可)
  (?<=a*)     NG  -- 可変長

  注意: (?<=ab|cde) は Python で使えるが、選択肢の長さが
        異なっていても各選択肢が固定長ならOK(Python 3.6+)
```

```python
import re

# 固定長: OK
print(re.findall(r'(?<=\$)\d+', "$100 $200"))
# => ['100', '200']

# 可変長: エラー(Python)
try:
    re.findall(r'(?<=\$+)\d+', "$100 $$200")
except re.error as e:
    print(f"エラー: {e}")
# => エラー: look-behind requires fixed-width pattern

# 回避策: regex モジュール(サードパーティ)を使う
# import regex
# regex.findall(r'(?<=\$+)\d+', "$100 $$200")
# => ['100', '200']
```

### 4.3 Python 3.6+ の後読み選択肢の挙動

```python
import re

# Python 3.6+ では、選択肢の各ブランチが固定長であれば
# 異なる長さの選択肢も使える
text = "USD100 JPY200 EUR300"

# 各選択肢が固定長（3文字）-- OK
pattern = r'(?<=USD|JPY|EUR)\d+'
print(re.findall(pattern, text))
# => ['100', '200', '300']

# 選択肢の長さが異なるが、各選択肢は固定長 -- Python 3.6+ でOK
text = "$100 USD200 EURO300"
pattern = r'(?<=\$|USD|EURO)\d+'
print(re.findall(pattern, text))
# => ['100', '200', '300']

# ただし選択肢内に量指定子は不可
try:
    re.findall(r'(?<=\$+|USD)\d+', text)
except re.error as e:
    print(f"エラー: {e}")
# => エラー: look-behind requires fixed-width pattern
```

### 4.4 後読みを使ったデータ抽出

```python
import re

# HTMLタグの属性値を抽出
html = '<div class="main" id="content" data-value="42">'

# class 属性の値を抽出
pattern = r'(?<=class=")\w+'
print(re.findall(pattern, html))
# => ['main']

# id 属性の値を抽出
pattern = r'(?<=id=")\w+'
print(re.findall(pattern, html))
# => ['content']

# data-value の値を抽出
pattern = r'(?<=data-value=")\d+'
print(re.findall(pattern, html))
# => ['42']

# ログファイルからIPアドレスの後のリクエストパスを抽出
log_lines = [
    '192.168.1.1 - - [01/Jan/2024] "GET /api/users HTTP/1.1" 200',
    '10.0.0.5 - - [01/Jan/2024] "POST /api/login HTTP/1.1" 401',
    '172.16.0.1 - - [01/Jan/2024] "GET /index.html HTTP/1.1" 200',
]
for line in log_lines:
    # "GET " or "POST " の後のパスを抽出
    m = re.search(r'(?<=(?:GET|POST) )/\S+', line)
    if m:
        print(f"  パス: {m.group()}")
# => パス: /api/users
# => パス: /api/login
# => パス: /index.html
```

### 4.5 後読みと先読みの組み合わせ

```python
import re

# 特定の区切り文字に囲まれた内容を抽出
text = "[重要] これは重要なメッセージです [情報] これは情報です"

# [] 内の文字列を抽出（後読み + 先読み）
pattern = r'(?<=\[)[^\]]+(?=\])'
print(re.findall(pattern, text))
# => ['重要', '情報']

# 引用符で囲まれた内容を抽出
text = 'name="Alice" age="30" city="Tokyo"'
pattern = r'(?<=")[^"]+(?=")'
print(re.findall(pattern, text))
# => ['Alice', '30', 'Tokyo']

# CSV の特定カラムの値を抽出
csv_line = "Alice,30,Tokyo,Engineer"
# 2番目のカンマの後、3番目のカンマの前
pattern = r'(?<=,)[^,]+(?=,)'
print(re.findall(pattern, csv_line))
# => ['30', 'Tokyo']  -- 最初と最後のフィールド以外
```

---

## 5. 否定後読み `(?<!pattern)`

### 5.1 基本動作

```python
import re

# 「前に "$" がない数字」を抽出
pattern = r'(?<!\$)\b\d+'
text = "Price: $100, Qty: 5, Tax: $15, Count: 42"

print(re.findall(pattern, text))
# => ['5', '42']
# "$100" と "$15" はマッチしない(前に $ がある)
```

### 5.2 複合条件

```python
import re

# 否定後読み + 否定先読みの組み合わせ
# 「引用符で囲まれていない数字」
pattern = r'(?<!["\'`])\b\d+\b(?!["\'`])'
text = 'value is 42 and "100" and \'200\''

print(re.findall(pattern, text))
# => ['42']
```

### 5.3 否定後読みの実践例

```python
import re

# エスケープされていない特殊文字を検出
text = r'Hello\nWorld\tTab\\Backslash\xHex'

# バックスラッシュの後でない 'n' を検出
# (エスケープシーケンスでない位置の 'n')
# ※ raw string に注意
pattern = r'(?<!\\)n'
# 注: この例では raw string を使う必要がある

# コメント以外のコード部分を抽出
code_lines = [
    "x = 10  # 変数の初期化",
    "# これは完全なコメント行",
    "y = x + 1  # 加算",
    "print(y)",
]

for line in code_lines:
    # # 以降をコメントとして除去（ただし文字列内の # は除く）
    # 簡易版: 行頭から # の前までを取得
    code_part = re.sub(r'\s*#.*$', '', line)
    if code_part.strip():
        print(f"  コード: {code_part.strip()}")
# => コード: x = 10
# => コード: y = x + 1
# => コード: print(y)

# 否定後読みを使ったエスケープ文字の処理
# エスケープされていない引用符を検出
text = r'He said "hello" and "it\'s \"fine\""'
# \" でない " を検出
unescaped_quotes = re.findall(r'(?<!\\)"', text)
print(f"エスケープされていない引用符: {len(unescaped_quotes)}個")
```

### 5.4 否定後読みによる条件付き置換

```python
import re

# 特定の文脈でのみ置換を行う

# 例1: HTML エンティティ化されていない & を変換
text = "Tom & Jerry &amp; Friends &lt;tag&gt;"
# "&amp;" や "&lt;" などは既にエスケープ済みなので変換しない
result = re.sub(r'&(?!amp;|lt;|gt;|quot;|#\d+;)', '&amp;', text)
print(result)
# => "Tom &amp; Jerry &amp; Friends &lt;tag&gt;"

# 例2: Markdown のリンク内でない URL を自動リンク化
text = "Visit http://example.com or [click here](http://other.com)"
# 既にリンク内にある URL は変換しない
pattern = r'(?<!\()(https?://\S+)(?!\))'
result = re.sub(pattern, r'<a href="\1">\1</a>', text)
print(result)

# 例3: 既にタグ付けされていないメールアドレスをリンク化
text = "連絡: user@example.com <a>admin@example.com</a>"
pattern = r'(?<!>)\b[\w.+-]+@[\w-]+\.[\w.]+\b(?!<)'
result = re.sub(pattern, r'<a href="mailto:\g<0>">\g<0></a>', text)
print(result)
```

---

## 6. ルックアラウンドの組み合わせパターン

### 6.1 AND条件: 複数の先読みを連鎖

複数の先読みを同じ位置に配置することで、全ての条件を同時に満たすことを要求できる。

```python
import re

# 例: 以下の全条件を満たす文字列
# - 8文字以上20文字以下
# - 大文字を含む
# - 小文字を含む
# - 数字を含む
# - 記号を含む
# - 同じ文字が3回以上連続しない
pattern = re.compile(
    r'^'
    r'(?=.{8,20}$)'            # 8-20文字
    r'(?=.*[A-Z])'              # 大文字を含む
    r'(?=.*[a-z])'              # 小文字を含む
    r'(?=.*\d)'                 # 数字を含む
    r'(?=.*[!@#$%^&*()_+=-])'  # 記号を含む
    r'(?!.*(.)\1{2})'          # 同一文字3連続なし
    r'.*$'                      # 全体にマッチ
)

test_cases = [
    ("Passw0rd!", True),
    ("weakpass", False),        # 数字・記号・大文字なし
    ("ALLCAPS1!", False),       # 小文字なし
    ("Short1!", False),         # 8文字未満
    ("Tooooo0long!password!!", False),  # 20文字超
    ("Paaass0rd!", False),      # 'a' が3連続
    ("C0mpl3x!Pwd", True),
]

for pw, expected in test_cases:
    result = bool(pattern.match(pw))
    status = "PASS" if result == expected else "FAIL"
    print(f"  [{status}] '{pw}' => {result} (expected: {expected})")
```

### 6.2 NOT条件: 否定先読みで除外

```python
import re

# 特定のパターンを含まない行を抽出
text = """
DEBUG: Starting process
INFO: User logged in
ERROR: Connection failed
DEBUG: Processing data
WARN: Low memory
INFO: Task completed
ERROR: Timeout exceeded
"""

# ERROR と DEBUG を含まない行
pattern = r'^(?!.*(ERROR|DEBUG)).*$'
lines = re.findall(pattern, text.strip(), re.MULTILINE)
print("フィルタ結果:")
for line in lines:
    if line.strip():
        print(f"  {line.strip()}")
# => INFO: User logged in
# => WARN: Low memory
# => INFO: Task completed
```

### 6.3 位置の挟み込み: 後読み + 先読み

```python
import re

# パターン: 特定の文脈にある単語のみを置換
text = "The quick brown fox jumps over the lazy dog"

# "the" を大文字の "THE" に置換するが、文頭の "The" はそのまま
# 後読み: 前にスペースがある、先読み: 後にスペースがある
result = re.sub(r'(?<=\s)the(?=\s)', 'THE', text)
print(result)
# => "The quick brown fox jumps over THE lazy dog"

# JSON のキー名を変換（snake_case → camelCase）
json_text = '{"user_name": "Alice", "first_name": "Alice", "last_name": "Smith"}'
# 後読みでキー内のアンダースコアを検出し、次の文字を大文字に
def snake_to_camel(match):
    return match.group(1).upper()

result = re.sub(r'(?<="[a-z_]*)_([a-z])', snake_to_camel, json_text)
print(result)
```

### 6.4 複数条件の実用パターン集

```python
import re

# パターン1: 通貨記号の後の数値を通貨ごとに抽出
text = "Items: $100, EUR200, JPY15000, GBP50"
currencies = {
    'USD': re.findall(r'(?<=\$)\d+', text),
    'EUR': re.findall(r'(?<=EUR)\d+', text),
    'JPY': re.findall(r'(?<=JPY)\d+', text),
    'GBP': re.findall(r'(?<=GBP)\d+', text),
}
for currency, values in currencies.items():
    if values:
        print(f"  {currency}: {values}")

# パターン2: XML/HTMLタグの中身を抽出（開始タグと終了タグの間）
html = "<title>My Page</title><p>Hello World</p><span>Test</span>"

# 汎用パターン: タグ名をキャプチャし、対応する終了タグまでを取得
pattern = r'(?<=<(\w+)>).*?(?=</\1>)'
# ※ Python re では後読み内にキャプチャグループを使うと問題がある場合がある
# 代替アプローチ:
tags = re.findall(r'<(\w+)>(.*?)</\1>', html)
for tag, content in tags:
    print(f"  <{tag}>: {content}")
# => <title>: My Page
# => <p>: Hello World
# => <span>: Test

# パターン3: 条件付きメールアドレス抽出
# 社内ドメイン（@company.com）以外のメールアドレスを検出
emails = "alice@company.com bob@gmail.com carol@company.com dave@yahoo.co.jp"
external = re.findall(r'\b[\w.+-]+@(?!company\.com\b)[\w.-]+\.\w+', emails)
print(f"  外部メール: {external}")
# => ['bob@gmail.com', 'dave@yahoo.co.jp']
```

---

## 7. 実践的なユースケース

### 7.1 パスワード強度検証

```python
import re

def validate_password(password: str) -> tuple[bool, list[str]]:
    """パスワード強度を検証する(ルックアラウンド活用)"""
    errors = []

    # 8文字以上
    if len(password) < 8:
        errors.append("8文字以上必要")

    # 大文字を含む(肯定先読み)
    if not re.search(r'(?=.*[A-Z])', password):
        errors.append("大文字を1文字以上含む必要あり")

    # 小文字を含む
    if not re.search(r'(?=.*[a-z])', password):
        errors.append("小文字を1文字以上含む必要あり")

    # 数字を含む
    if not re.search(r'(?=.*\d)', password):
        errors.append("数字を1文字以上含む必要あり")

    # 記号を含む
    if not re.search(r'(?=.*[!@#$%^&*])', password):
        errors.append("記号(!@#$%^&*)を1文字以上含む必要あり")

    return (len(errors) == 0, errors)

# 一つのパターンにまとめる場合:
strong_password = re.compile(
    r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*]).{8,}$'
)

print(strong_password.match("Passw0rd!"))  # => マッチ
print(strong_password.match("password"))   # => None
```

### 7.2 数値の桁区切り

```python
import re

# 3桁ごとにカンマを挿入
def add_commas(n: str) -> str:
    """先読み・後読みで桁区切りを挿入"""
    return re.sub(
        r'(?<=\d)(?=(?:\d{3})+(?!\d))',
        ',',
        n
    )

print(add_commas("1234567"))     # => '1,234,567'
print(add_commas("1234567890"))  # => '1,234,567,890'
print(add_commas("42"))          # => '42' (変化なし)

# パターン解説:
# (?<=\d)           -- 前に数字がある位置
# (?=(?:\d{3})+     -- 後ろに3桁の数字が1回以上続き
#   (?!\d))         -- その後に数字が続かない位置
# → その位置にカンマを挿入
```

### 7.3 特定コンテキストの置換

```python
import re

# "foo" を "bar" に置換するが、引用符内は除外
text = 'Use foo here, but "foo" stays unchanged'

# 方法: 否定後読み + 否定先読み
# (注: 完全な引用符内判定には限界がある)
result = re.sub(r'(?<!")foo(?!")', 'bar', text)
print(result)
# => 'Use bar here, but "foo" stays unchanged'
```

### 7.4 ログ解析における高度な抽出

```python
import re

# Apache/Nginx アクセスログの解析
log_line = '192.168.1.100 - admin [10/Oct/2024:13:55:36 -0700] "GET /api/v2/users?page=1 HTTP/1.1" 200 2326'

# 各フィールドをルックアラウンドで抽出
ip = re.search(r'^\S+', log_line).group()
user = re.search(r'(?<=- )\w+', log_line).group()
timestamp = re.search(r'(?<=\[)[^\]]+(?=\])', log_line).group()
method = re.search(r'(?<=")\w+', log_line).group()
path = re.search(r'(?<=(?:GET|POST|PUT|DELETE|PATCH) )\S+', log_line).group()
status = re.search(r'(?<=" )\d{3}(?= )', log_line).group()
size = re.search(r'\d+$', log_line).group()

print(f"IP: {ip}")
print(f"ユーザー: {user}")
print(f"日時: {timestamp}")
print(f"メソッド: {method}")
print(f"パス: {path}")
print(f"ステータス: {status}")
print(f"サイズ: {size}")
```

### 7.5 テキスト変換ユーティリティ

```python
import re

# CamelCase → snake_case 変換
def camel_to_snake(name: str) -> str:
    """ルックアラウンドでCamelCaseをsnake_caseに変換"""
    # Step 1: 大文字と小文字の境界にアンダースコアを挿入
    s1 = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
    # Step 2: 連続する大文字の後に小文字が来る境界
    s2 = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', s1)
    return s2.lower()

test_cases = [
    "camelCase",           # => "camel_case"
    "CamelCase",           # => "camel_case"
    "getHTTPResponse",     # => "get_http_response"
    "XMLParser",           # => "xml_parser"
    "parseJSON",           # => "parse_json"
    "myURLHandler",        # => "my_url_handler"
    "simpleTest",          # => "simple_test"
]

for tc in test_cases:
    print(f"  {tc:25s} => {camel_to_snake(tc)}")

# snake_case → CamelCase 変換
def snake_to_camel(name: str, upper_first: bool = True) -> str:
    """snake_caseをCamelCaseに変換"""
    components = name.split('_')
    if upper_first:
        return ''.join(x.title() for x in components)
    else:
        return components[0] + ''.join(x.title() for x in components[1:])

test_cases_snake = [
    "camel_case",           # => "CamelCase"
    "get_http_response",    # => "GetHttpResponse"
    "xml_parser",           # => "XmlParser"
    "my_url_handler",       # => "MyUrlHandler"
]

for tc in test_cases_snake:
    print(f"  {tc:25s} => {snake_to_camel(tc)}")
```

### 7.6 Markdown テキストの処理

```python
import re

# Markdown のインラインコード内を保護しつつテキストを変換
markdown = "Use `foo` to call `bar()`, but foo outside code should change"

# 方法: インラインコード外の "foo" のみを置換
# Step 1: インラインコード部分を一時的にプレースホルダーに置換
placeholders = {}
counter = [0]

def save_code(match):
    key = f"\x00CODE{counter[0]}\x00"
    placeholders[key] = match.group()
    counter[0] += 1
    return key

protected = re.sub(r'`[^`]+`', save_code, markdown)

# Step 2: プレースホルダー以外の "foo" を置換
result = re.sub(r'foo', 'baz', protected)

# Step 3: プレースホルダーを元に戻す
for key, value in placeholders.items():
    result = result.replace(key, value)

print(result)
# => "Use `foo` to call `bar()`, but baz outside code should change"
```

### 7.7 条件付きコメント除去

```python
import re

# ソースコードからコメントを除去（文字列内のコメント記号は保持）
code = '''
x = "hello # world"  # この部分はコメント
y = 'test // data'  // これもコメント
z = 42  # 数値
url = "http://example.com"  # URL内の // は保持
'''

# 簡易版: 文字列リテラルを考慮したコメント除去
def remove_comments(text: str) -> str:
    """文字列リテラル内の # や // を保持しつつコメントを除去"""
    result = []
    for line in text.split('\n'):
        # 文字列リテラル内かどうかを追跡
        in_string = False
        string_char = None
        comment_start = -1

        for i, ch in enumerate(line):
            if not in_string:
                if ch in ('"', "'"):
                    in_string = True
                    string_char = ch
                elif ch == '#':
                    comment_start = i
                    break
                elif i + 1 < len(line) and line[i:i+2] == '//':
                    comment_start = i
                    break
            else:
                if ch == string_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False

        if comment_start >= 0:
            result.append(line[:comment_start].rstrip())
        else:
            result.append(line)

    return '\n'.join(result)

print(remove_comments(code))
```

---

## 8. JavaScript での先読み・後読み

### 8.1 ES2018 以降の後読みサポート

```javascript
// ES2018+ で後読みが利用可能

// 肯定後読み
const text1 = "Price: $100, EUR200";
console.log(text1.match(/(?<=\$)\d+/g));
// => ['100']

// 否定後読み
const text2 = "$100 200 $300 400";
console.log(text2.match(/(?<!\$)\b\d+/g));
// => ['200', '400']

// JavaScript では可変長後読みが可能
const text3 = "http://example.com https://secure.example.com";
console.log(text3.match(/(?<=https?:\/\/)\w+/g));
// => ['example', 'secure']
// Python re ではこのパターンはエラーになる
```

### 8.2 JavaScript での実践例

```javascript
// 数値のフォーマット
function formatNumber(num) {
    return num.toString().replace(/(?<=\d)(?=(\d{3})+(?!\d))/g, ',');
}

console.log(formatNumber(1234567));     // => "1,234,567"
console.log(formatNumber(1234567890));  // => "1,234,567,890"

// テンプレートリテラル内の変数を検出
const template = "Hello ${name}, your balance is ${balance}";
const variables = template.match(/(?<=\$\{)\w+(?=\})/g);
console.log(variables);
// => ['name', 'balance']

// パスワード強度チェック
function checkPasswordStrength(password) {
    const checks = {
        length: /.{8,}/.test(password),
        uppercase: /(?=.*[A-Z])/.test(password),
        lowercase: /(?=.*[a-z])/.test(password),
        number: /(?=.*\d)/.test(password),
        special: /(?=.*[!@#$%^&*])/.test(password),
    };

    const score = Object.values(checks).filter(Boolean).length;
    return { checks, score, strong: score >= 4 };
}

console.log(checkPasswordStrength("Passw0rd!"));
// => { checks: { length: true, uppercase: true, ... }, score: 5, strong: true }
```

### 8.3 名前付きキャプチャグループとルックアラウンドの組み合わせ

```javascript
// ES2018 の名前付きキャプチャグループとルックアラウンドを組み合わせ
const logLine = '2024-01-15T10:30:45 [ERROR] Database connection failed: timeout';

const pattern = /(?<=\[)(?<level>\w+)(?=\])/;
const match = logLine.match(pattern);
console.log(match.groups.level);
// => 'ERROR'

// 複数のログレベルを一括抽出
const logs = `
2024-01-15 [INFO] Server started
2024-01-15 [ERROR] Connection failed
2024-01-15 [WARN] Low disk space
2024-01-15 [DEBUG] Processing request
`;

const levels = [...logs.matchAll(/(?<=\[)(?<level>\w+)(?=\])/g)];
levels.forEach(m => console.log(m.groups.level));
// => INFO, ERROR, WARN, DEBUG
```

---

## 9. Java でのルックアラウンド

### 9.1 基本的な使い方

```java
import java.util.regex.*;
import java.util.*;

public class LookaroundExample {
    public static void main(String[] args) {
        // 肯定先読み
        Pattern p1 = Pattern.compile("\\d+(?=円)");
        Matcher m1 = p1.matcher("商品A: 1000円、商品B: 2500円");
        while (m1.find()) {
            System.out.println("金額: " + m1.group());
        }
        // => 金額: 1000
        // => 金額: 2500

        // 肯定後読み（固定長のみ）
        Pattern p2 = Pattern.compile("(?<=\\$)\\d+");
        Matcher m2 = p2.matcher("$100 $200 300");
        while (m2.find()) {
            System.out.println("USD: " + m2.group());
        }
        // => USD: 100
        // => USD: 200

        // 数値の桁区切り
        String number = "1234567890";
        String formatted = number.replaceAll(
            "(?<=\\d)(?=(\\d{3})+(?!\\d))", ","
        );
        System.out.println(formatted);
        // => 1,234,567,890
    }
}
```

### 9.2 Java 固有の注意点

```java
// Java の後読みは固定長のみ
// 以下はコンパイルエラーになる

try {
    Pattern.compile("(?<=\\w+)\\d+");
    // => PatternSyntaxException: Look-behind group does not have
    //    an obvious maximum length
} catch (PatternSyntaxException e) {
    System.out.println("エラー: " + e.getMessage());
}

// 回避策: 固定長の選択肢を使う
// (?<=\\w{1}|\\w{2}|\\w{3})\\d+  -- 1〜3文字の後読み
// ただしこのアプローチは非実用的なので、
// キャプチャグループを使う方が良い

// Java 13+ では一部の可変長後読みが改善されている
// ただし公式には固定長のみサポート
```

---

## 10. ASCII 図解

### 10.1 4種類のルックアラウンド動作

```
テキスト: "$100"

肯定先読み (?=\d):
  位置: $ [ここ] 1 0 0
  「右側に \d があるか?」 → 1 がある → 成功

否定先読み (?!\$):
  位置: [ここ] $ 1 0 0
  「右側に \$ がないか?」 → $ がある → 失敗
  位置: $ [ここ] 1 0 0
  「右側に \$ がないか?」 → 1 は $ でない → 成功

肯定後読み (?<=\$):
  位置: $ [ここ] 1 0 0
  「左側に \$ があるか?」 → $ がある → 成功

否定後読み (?<!\$):
  位置: [ここ] $ 1 0 0
  「左側に \$ がないか?」 → 何もない → 成功
  位置: $ [ここ] 1 0 0
  「左側に \$ がないか?」 → $ がある → 失敗
```

### 10.2 パスワード検証の先読みチェーン

```
パターン: ^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*]).{8,}$

入力: "Passw0rd!"

位置0(文字列先頭):
  (?=.*[A-Z])     → 先読み: "P" が大文字 → 成功 (位置は戻る)
  (?=.*[a-z])     → 先読み: "a" が小文字 → 成功 (位置は戻る)
  (?=.*\d)        → 先読み: "0" が数字 → 成功 (位置は戻る)
  (?=.*[!@#$%^&*])→ 先読み: "!" が記号 → 成功 (位置は戻る)
  .{8,}$          → "Passw0rd!" 9文字 ≥ 8 → 成功

全ての先読みが同じ位置0から開始される
(ゼロ幅なので位置が進まない)
```

### 10.3 桁区切りの先読み動作

```
入力: "1234567"
パターン: (?<=\d)(?=(?:\d{3})+(?!\d))

位置を一つずつ検査:

  1 | 2 3 4 5 6 7
    ↑
  (?<=\d): 1がある → OK
  (?=(?:\d{3})+(?!\d)): "234567" = 3桁×2 + 末尾に数字なし → OK
  → カンマ挿入位置!

  1 2 | 3 4 5 6 7
      ↑
  (?<=\d): 2がある → OK
  (?=(?:\d{3})+(?!\d)): "34567" = 3桁×1 + "67"余り → NG
  → スキップ

  1 2 3 | 4 5 6 7
        ↑
  (?<=\d): 3がある → OK
  (?=(?:\d{3})+(?!\d)): "4567" = 3桁×1 + "7"余り → NG
  → スキップ

  1 2 3 4 | 5 6 7
          ↑
  (?<=\d): 4がある → OK
  (?=(?:\d{3})+(?!\d)): "567" = 3桁×1 + 末尾に数字なし → OK
  → カンマ挿入位置!

結果: "1,234,567"
```

### 10.4 ネストされたルックアラウンドの動作

```
パターン: (?<=(?<=[A-Z])[a-z])\d
テキスト: "Ab1Cd2ef3"

検査: 各位置で3つの条件をチェック

位置2 '1':
  外側の後読み (?<=...): 位置1を検査
    位置1 'b' は [a-z] → OK
    内側の後読み (?<=[A-Z]): 位置0を検査
      位置0 'A' は [A-Z] → OK
  → 全条件成功! '1' がマッチ

位置5 '2':
  外側の後読み (?<=...): 位置4を検査
    位置4 'd' は [a-z] → OK
    内側の後読み (?<=[A-Z]): 位置3を検査
      位置3 'C' は [A-Z] → OK
  → 全条件成功! '2' がマッチ

位置8 '3':
  外側の後読み (?<=...): 位置7を検査
    位置7 'f' は [a-z] → OK
    内側の後読み (?<=[A-Z]): 位置6を検査
      位置6 'e' は [A-Z]? → NO!
  → 内側の後読み失敗! '3' はマッチしない

結果: ['1', '2']
意味: 「大文字→小文字→数字」のパターンの数字部分
```

---

## 11. 比較表

### 11.1 ルックアラウンド完全比較

| 種類 | 構文 | 意味 | 例 | マッチ |
|------|------|------|-----|--------|
| 肯定先読み | `X(?=Y)` | XのあとにYがある | `\d+(?=円)` | "100" in "100円" |
| 否定先読み | `X(?!Y)` | XのあとにYがない | `\d+(?!円)` | "200" in "200ドル" |
| 肯定後読み | `(?<=Y)X` | Xの前にYがある | `(?<=\$)\d+` | "100" in "$100" |
| 否定後読み | `(?<!Y)X` | Xの前にYがない | `(?<!\$)\d+` | "42" in "count: 42" |

### 11.2 言語サポート状況

| 機能 | Python | JavaScript | Java | Go(RE2) | Rust | .NET | Perl | Ruby |
|------|--------|------------|------|---------|------|------|------|------|
| 肯定先読み `(?=)` | OK | OK | OK | 不可 | 不可 | OK | OK | OK |
| 否定先読み `(?!)` | OK | OK | OK | 不可 | 不可 | OK | OK | OK |
| 肯定後読み `(?<=)` | OK(固定長) | OK(可変長) | OK(固定長) | 不可 | 不可 | OK(可変長) | OK(固定長) | OK(固定長) |
| 否定後読み `(?<!)` | OK(固定長) | OK(可変長) | OK(固定長) | 不可 | 不可 | OK(可変長) | OK(固定長) | OK(固定長) |
| 可変長後読み | regex モジュール | ES2018+ | 不可 | N/A | fancy-regex | 標準 | 不可 | 不可 |

### 11.3 ルックアラウンド vs キャプチャグループ

| 比較項目 | ルックアラウンド | キャプチャグループ |
|----------|------------------|-------------------|
| マッチに含まれるか | 含まれない（ゼロ幅） | 含まれる |
| 置換時の扱い | 周囲のテキストを保持 | グループとして参照可能 |
| パフォーマンス | やや遅い（バックトラック） | 一般的に速い |
| 可読性 | 複雑になりやすい | 比較的読みやすい |
| 用途 | 位置の条件指定 | 部分文字列の抽出 |
| AND条件 | 連鎖で実現可能 | 単独では不可 |
| エンジン互換性 | エンジン依存が大きい | ほぼ全エンジン対応 |

```python
import re

# 比較例: 同じ結果を得る2つのアプローチ

text = "Price: $100, $200, $300"

# アプローチ1: 後読み（$を含まずに数値を取得）
result1 = re.findall(r'(?<=\$)\d+', text)
print(f"後読み:     {result1}")   # => ['100', '200', '300']

# アプローチ2: キャプチャグループ
result2 = re.findall(r'\$(\d+)', text)
print(f"キャプチャ: {result2}")   # => ['100', '200', '300']

# 結果は同じだが、置換時に違いが出る:
# 後読みを使った置換: $は保持される
result3 = re.sub(r'(?<=\$)\d+', 'XXX', text)
print(f"後読み置換: {result3}")
# => "Price: $XXX, $XXX, $XXX"

# キャプチャグループを使った置換: $も含めて指定が必要
result4 = re.sub(r'\$\d+', '$XXX', text)
print(f"グループ置換: {result4}")
# => "Price: $XXX, $XXX, $XXX"
```

---

## 12. パフォーマンス考慮事項

### 12.1 ルックアラウンドのコスト

```python
import re
import time

# ルックアラウンドは各位置でサブパターンの評価が必要
# 大量のテキストでは性能に影響する

def benchmark(name, pattern, text, iterations=10000):
    compiled = re.compile(pattern)
    start = time.perf_counter()
    for _ in range(iterations):
        compiled.findall(text)
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed:.4f}秒 ({iterations}回)")

text = "The quick brown fox jumps over the lazy dog " * 100

# 単純なパターン vs ルックアラウンド
benchmark("単純マッチ", r'\b\w+\b', text)
benchmark("先読み付き", r'\b\w+(?=\s)', text)
benchmark("後読み付き", r'(?<=\s)\w+', text)
benchmark("両方付き  ", r'(?<=\s)\w+(?=\s)', text)
```

### 12.2 ルックアラウンドによる破壊的バックトラッキング

```python
import re

# 危険なパターンの例

# 危険: ネストされた先読みと繰り返し
# pattern = r'(?=.*a)(?=.*b)(?=.*c).+'
# 入力が長くなると指数的に遅くなる可能性がある

# 安全: 各先読みの後に具体的なパターンを置く
safe_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$'
# これは各先読みが独立に評価され、.{8,} は線形時間

# ReDoS のリスクがあるパターン
# (?=.*a+)(?=.*b+) のような先読みの中の量指定子は
# バックトラックの原因になりうる

# 対策: 先読み内のパターンをできるだけ具体的に
# NG: (?=.*a+)
# OK: (?=.*a)
# OK: (?=[^a]*a)  -- 否定文字クラスで効率化
```

### 12.3 最適化テクニック

```python
import re

# テクニック1: 否定文字クラスで先読みを高速化
text = "abc123def456ghi789"

# 遅い: .* が全文字列を試行してからバックトラック
slow = r'(?=.*\d)\w+'
# 速い: [^\d]* が数字でない文字だけスキップ
fast = r'(?=[^\d]*\d)\w+'

# テクニック2: アトミックグループ（対応エンジンの場合）
# Python re では非対応、regex モジュールで利用可能
# (?>pattern) はバックトラックを防止

# テクニック3: 先読みの順序を最適化
# 失敗しやすい条件を先に置くことで早期に失敗できる
# 例: パスワード検証で、記号チェックは最初に
# （記号を含まないパスワードが多いため）
password_pattern = re.compile(
    r'^'
    r'(?=.*[!@#$%^&*])'  # 記号チェック（最も失敗しやすい）
    r'(?=.*\d)'            # 数字チェック
    r'(?=.*[A-Z])'         # 大文字チェック
    r'(?=.*[a-z])'         # 小文字チェック
    r'.{8,}$'
)

# テクニック4: コンパイル済みパターンの再利用
# ルックアラウンドを含むパターンは特にコンパイルコストが高い
# 必ず re.compile() で事前コンパイルする
compiled = re.compile(r'(?<=\$)\d+(?=\.\d{2})')
# ループ内では compiled.findall(text) を使う
```

---

## 13. 高度なテクニック

### 13.1 条件付きパターン（ルックアラウンドの応用）

```python
import re

# 文脈依存の置換
text = "foo_bar baz_qux FOO_BAR"

# 先頭が大文字なら全体を大文字に、小文字なら小文字に変換
def context_aware_replace(match):
    word = match.group()
    parts = word.split('_')
    if parts[0][0].isupper():
        return ''.join(p.upper() for p in parts)
    else:
        return ''.join(p.capitalize() for p in parts)

result = re.sub(r'\w+(?:_\w+)+', context_aware_replace, text)
print(result)
# => "FooBar BazQux FOOBAR"
```

### 13.2 再帰的パターンのシミュレーション

```python
import re

# Python re では再帰パターンはサポートされないが、
# ルックアラウンドで一部をシミュレートできる

# ネストされた括弧の外側のカンマで分割
text = "a(b,c),d,(e,f(g,h)),i"

# Step 1: 括弧のネストレベルを追跡して分割
def split_at_top_level(text: str, delimiter: str = ',') -> list[str]:
    """トップレベルのデリミタで分割（括弧内は無視）"""
    result = []
    current = []
    depth = 0

    for ch in text:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == delimiter and depth == 0:
            result.append(''.join(current))
            current = []
        else:
            current.append(ch)

    result.append(''.join(current))
    return result

print(split_at_top_level(text))
# => ['a(b,c)', 'd', '(e,f(g,h))', 'i']
```

### 13.3 複数のルックアラウンドを使った複雑な検証

```python
import re

def validate_credit_card(number: str) -> dict:
    """クレジットカード番号の検証（ルックアラウンド活用）"""
    # スペースとハイフンを除去
    clean = re.sub(r'[\s-]', '', number)

    result = {
        'number': clean,
        'valid_format': False,
        'card_type': 'Unknown',
        'luhn_valid': False,
    }

    # カードタイプの判定（先読みで番号パターンを検査）
    card_patterns = {
        'Visa': r'^4\d{12}(?:\d{3})?$',
        'MasterCard': r'^5[1-5]\d{14}$',
        'AmEx': r'^3[47]\d{13}$',
        'Discover': r'^6(?:011|5\d{2})\d{12}$',
        'JCB': r'^(?:2131|1800|35\d{3})\d{11}$',
    }

    for card_type, pattern in card_patterns.items():
        if re.match(pattern, clean):
            result['card_type'] = card_type
            result['valid_format'] = True
            break

    # Luhn アルゴリズムによる検証
    if result['valid_format']:
        digits = [int(d) for d in clean]
        checksum = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
        result['luhn_valid'] = (checksum % 10 == 0)

    return result

# テスト
test_cards = [
    "4111 1111 1111 1111",   # Visa テストカード
    "5500 0000 0000 0004",   # MasterCard テストカード
    "3400 000000 00009",     # AmEx テストカード
    "1234 5678 9012 3456",   # 無効
]

for card in test_cards:
    result = validate_credit_card(card)
    print(f"  {card}: {result['card_type']} "
          f"(format: {result['valid_format']}, luhn: {result['luhn_valid']})")
```

### 13.4 テキストのトークナイズにルックアラウンドを活用

```python
import re

# ルックアラウンドを使った高度なトークナイズ
# 文字種の境界で分割する

def tokenize_mixed(text: str) -> list[str]:
    """英数字、日本語、記号の境界で分割"""
    # 文字種の境界にスペースを挿入
    # 英字→数字、数字→英字の境界
    result = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)
    result = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', result)
    # 英数字→日本語、日本語→英数字の境界
    result = re.sub(r'(?<=[a-zA-Z0-9])(?=[\u3040-\u9fff])', ' ', result)
    result = re.sub(r'(?<=[\u3040-\u9fff])(?=[a-zA-Z0-9])', ' ', result)

    return result.split()

test_cases = [
    "Hello123World",        # => ['Hello', '123', 'World']
    "test42data",           # => ['test', '42', 'data']
    "Hello世界2024",        # => ['Hello', '世界', '2024']
    "Python3プログラミング", # => ['Python', '3', 'プログラミング']
]

for tc in test_cases:
    tokens = tokenize_mixed(tc)
    print(f"  '{tc}' => {tokens}")
```

### 13.5 ルックアラウンドを使ったCSVパーサー

```python
import re

def parse_csv_field(line: str) -> list[str]:
    """ルックアラウンドを活用したCSVフィールドのパース

    ダブルクォートで囲まれたフィールド内のカンマを正しく処理する
    """
    fields = []
    # クォートされたフィールドとされていないフィールドの両方に対応
    pattern = re.compile(
        r'"([^"]*(?:""[^"]*)*)"|'  # ダブルクォート内（"" はエスケープ）
        r'([^,]*)'                  # クォートなしフィールド
    )

    pos = 0
    while pos <= len(line):
        m = pattern.match(line, pos)
        if m:
            if m.group(1) is not None:
                # クォートされたフィールド: "" を " に変換
                fields.append(m.group(1).replace('""', '"'))
            else:
                fields.append(m.group(2))
            pos = m.end()
            # カンマをスキップ
            if pos < len(line) and line[pos] == ',':
                pos += 1
            elif pos >= len(line):
                break
        else:
            break

    return fields

# テスト
test_lines = [
    'Alice,30,Tokyo',
    '"Bob ""Jr""",25,"New York, NY"',
    '"contains,comma",normal,"also ""quoted"""',
]

for line in test_lines:
    fields = parse_csv_field(line)
    print(f"  入力: {line}")
    print(f"  結果: {fields}")
    print()
```

---

## 14. Go と Rust でのルックアラウンド代替策

### 14.1 Go (RE2) での代替手法

```go
package main

import (
    "fmt"
    "regexp"
    "strings"
)

func main() {
    // Go の RE2 エンジンはルックアラウンド非サポート
    // キャプチャグループで代替する

    // 代替例1: (?<=\$)\d+ の代わり
    text := "Price: $100, $200, 300"
    re := regexp.MustCompile(`\$(\d+)`)
    matches := re.FindAllStringSubmatch(text, -1)
    for _, m := range matches {
        fmt.Println("金額:", m[1]) // キャプチャグループ1
    }
    // => 金額: 100
    // => 金額: 200

    // 代替例2: 数値の桁区切り（ルックアラウンドなし）
    number := "1234567890"
    formatted := addCommas(number)
    fmt.Println("桁区切り:", formatted)
    // => 桁区切り: 1,234,567,890

    // 代替例3: パスワード検証（個別チェック）
    password := "Passw0rd!"
    fmt.Println("パスワード強度:", validatePassword(password))
}

func addCommas(s string) string {
    // ルックアラウンドなしで桁区切りを実装
    n := len(s)
    if n <= 3 {
        return s
    }

    var result strings.Builder
    remainder := n % 3
    if remainder > 0 {
        result.WriteString(s[:remainder])
        if remainder < n {
            result.WriteByte(',')
        }
    }
    for i := remainder; i < n; i += 3 {
        if i > remainder {
            result.WriteByte(',')
        }
        result.WriteString(s[i : i+3])
    }
    return result.String()
}

func validatePassword(pw string) bool {
    // 各条件を個別にチェック（先読みの代替）
    hasUpper := regexp.MustCompile(`[A-Z]`).MatchString(pw)
    hasLower := regexp.MustCompile(`[a-z]`).MatchString(pw)
    hasDigit := regexp.MustCompile(`\d`).MatchString(pw)
    hasSpecial := regexp.MustCompile(`[!@#$%^&*]`).MatchString(pw)
    hasLength := len(pw) >= 8

    return hasUpper && hasLower && hasDigit && hasSpecial && hasLength
}
```

### 14.2 Rust での代替手法

```rust
use regex::Regex;

fn main() {
    // Rust の regex クレートはルックアラウンド非サポート
    // fancy-regex クレートを使えば利用可能

    // 標準 regex での代替: キャプチャグループ
    let re = Regex::new(r"\$(\d+)").unwrap();
    let text = "Price: $100, $200, 300";

    for cap in re.captures_iter(text) {
        println!("金額: {}", &cap[1]);
    }
    // => 金額: 100
    // => 金額: 200

    // パスワード検証: 個別チェック
    let password = "Passw0rd!";
    println!("有効: {}", validate_password(password));
}

fn validate_password(pw: &str) -> bool {
    let has_upper = Regex::new(r"[A-Z]").unwrap().is_match(pw);
    let has_lower = Regex::new(r"[a-z]").unwrap().is_match(pw);
    let has_digit = Regex::new(r"\d").unwrap().is_match(pw);
    let has_special = Regex::new(r"[!@#$%^&*]").unwrap().is_match(pw);
    let has_length = pw.len() >= 8;

    has_upper && has_lower && has_digit && has_special && has_length
}

// fancy-regex を使う場合:
// [dependencies]
// fancy-regex = "0.11"
//
// use fancy_regex::Regex;
//
// fn with_lookaround() {
//     let re = Regex::new(r"(?<=\$)\d+").unwrap();
//     let text = "Price: $100, $200";
//     for m in re.find_iter(text) {
//         if let Ok(m) = m {
//             println!("金額: {}", m.as_str());
//         }
//     }
// }
```

---

## 15. テスト駆動パターン開発

### 15.1 ルックアラウンドパターンのユニットテスト

```python
import re
import unittest

class TestLookaroundPatterns(unittest.TestCase):
    """ルックアラウンドパターンのテストスイート"""

    def test_positive_lookahead_yen(self):
        """肯定先読み: 円の前の数値を抽出"""
        pattern = re.compile(r'\d+(?=円)')
        self.assertEqual(pattern.findall("1000円"), ['1000'])
        self.assertEqual(pattern.findall("1000ドル"), [])
        self.assertEqual(pattern.findall("1000円と2000円"), ['1000', '2000'])
        self.assertEqual(pattern.findall("円1000"), [])

    def test_negative_lookahead_exclusion(self):
        """否定先読み: 特定の単語を除外"""
        pattern = re.compile(r'\b(?!test\b)\w+')
        words = pattern.findall("test hello testing world")
        self.assertIn('hello', words)
        self.assertIn('testing', words)
        self.assertIn('world', words)
        self.assertNotIn('test', words)

    def test_positive_lookbehind_dollar(self):
        """肯定後読み: $の後の数値を抽出"""
        pattern = re.compile(r'(?<=\$)\d+')
        self.assertEqual(pattern.findall("$100 $200 300"), ['100', '200'])
        self.assertEqual(pattern.findall("100 200"), [])

    def test_negative_lookbehind_no_dollar(self):
        """否定後読み: $がない数値を抽出"""
        pattern = re.compile(r'(?<!\$)\b\d+')
        self.assertEqual(pattern.findall("$100 200 $300 400"), ['200', '400'])

    def test_password_validation(self):
        """パスワード検証パターン"""
        pattern = re.compile(
            r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*]).{8,}$'
        )
        self.assertIsNotNone(pattern.match("Passw0rd!"))
        self.assertIsNone(pattern.match("password"))
        self.assertIsNone(pattern.match("SHORT1!"))
        self.assertIsNone(pattern.match("nouppercase1!"))
        self.assertIsNone(pattern.match("NOLOWERCASE1!"))
        self.assertIsNone(pattern.match("NoDigits!!"))
        self.assertIsNone(pattern.match("NoSpecial1a"))

    def test_comma_formatting(self):
        """桁区切りフォーマット"""
        pattern = re.compile(r'(?<=\d)(?=(?:\d{3})+(?!\d))')

        def add_commas(n):
            return pattern.sub(',', n)

        self.assertEqual(add_commas("1234567"), "1,234,567")
        self.assertEqual(add_commas("42"), "42")
        self.assertEqual(add_commas("1000"), "1,000")
        self.assertEqual(add_commas("1234567890"), "1,234,567,890")

    def test_camel_to_snake(self):
        """CamelCase → snake_case 変換"""
        def camel_to_snake(name):
            s1 = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
            s2 = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', s1)
            return s2.lower()

        self.assertEqual(camel_to_snake("camelCase"), "camel_case")
        self.assertEqual(camel_to_snake("CamelCase"), "camel_case")
        self.assertEqual(camel_to_snake("getHTTPResponse"), "get_http_response")
        self.assertEqual(camel_to_snake("XMLParser"), "xml_parser")
        self.assertEqual(camel_to_snake("simple"), "simple")

if __name__ == '__main__':
    unittest.main()
```

### 15.2 エッジケースのテスト

```python
import re

# ルックアラウンドのエッジケース集
def test_edge_cases():
    """エッジケースを網羅的にテスト"""

    # 1. 空文字列
    pattern = re.compile(r'(?=\d)\d+')
    assert pattern.findall("") == []

    # 2. 文字列の先頭/末尾でのルックアラウンド
    # 先頭での後読み: 常に失敗（前に何もない）
    assert re.findall(r'(?<=x)\w+', "abc") == []
    # 先頭にマッチする文字がある場合
    assert re.findall(r'(?<=x)\w+', "xabc") == ['abc']

    # 3. マルチライン対応
    text = "line1\nline2\nline3"
    # ^ は各行の先頭（re.MULTILINE使用時）
    assert re.findall(r'(?<=^)line\d', text, re.MULTILINE) == ['line1', 'line2', 'line3']

    # 4. Unicode文字でのルックアラウンド
    text = "価格：100円、200ドル"
    assert re.findall(r'\d+(?=円)', text) == ['100']
    assert re.findall(r'(?<=：)\d+', text) == ['100']

    # 5. 重複するルックアラウンド条件
    # 同じ位置で矛盾する条件: 常に失敗
    assert re.findall(r'(?=a)(?=b)', "ab") == []
    # 同じ位置で両方満たせる条件
    assert len(re.findall(r'(?=a)(?!b)', "a")) == 1

    # 6. ゼロ幅マッチの連続
    text = "abc"
    # 全位置でマッチ（4位置: 0,1,2,3）
    assert len(re.findall(r'(?=.)?', text)) >= 3

    print("全エッジケーステスト合格!")

test_edge_cases()
```

---

## 16. アンチパターン

### 16.1 アンチパターン: ルックアラウンドの過剰使用

```python
import re

# NG: 単純な条件にルックアラウンドを使う
pattern_bad = r'(?<=price: )\d+'
# ↑ 後読みを使わなくても抽出可能

# OK: キャプチャグループで十分
pattern_good = r'price: (\d+)'
match = re.search(pattern_good, "price: 100")
print(match.group(1))  # => '100'

# ルックアラウンドが真に必要な場面:
# ・置換で周囲のテキストを保持したい場合
# ・複数の位置条件を AND で組み合わせたい場合
# ・マッチ結果に特定の文字列を含めたくない場合
```

### 16.2 アンチパターン: 可変長後読みを想定する

```python
import re

# NG: Python re で可変長後読みを使う
try:
    re.search(r'(?<=https?://)\w+', "https://example.com")
except re.error as e:
    print(f"エラー: {e}")
    # => look-behind requires fixed-width pattern
    # "https?" は4文字または5文字 → 可変長

# OK: 各長さを OR で列挙
pattern = r'(?<=http://|https://)\w+'
# これも Python re ではエラー（選択肢の長さが異なる）
# ※ Python 3.6+ では異なる固定長の選択肢は許可される

# 回避策1: 別のアプローチ
pattern = r'https?://(\w+)'
match = re.search(pattern, "https://example.com")
print(match.group(1))  # => 'example'

# 回避策2: regex モジュール(可変長後読みをサポート)
# import regex
# regex.search(r'(?<=https?://)\w+', "https://example.com")
```

### 16.3 アンチパターン: ルックアラウンドで全てを解決しようとする

```python
import re

# NG: 複雑すぎるルックアラウンドチェーン
# 「大文字で始まり、数字を含み、5-10文字で、'test'を含まない単語」
bad_pattern = r'\b(?=[A-Z])(?=\w*\d)(?!\w*test)(?=\w{5,10}\b)\w+'

# OK: 段階的に処理する
def find_valid_words(text: str) -> list[str]:
    """複数条件を段階的に適用"""
    words = re.findall(r'\b\w+\b', text)
    result = []
    for word in words:
        if not (5 <= len(word) <= 10):
            continue
        if not word[0].isupper():
            continue
        if not re.search(r'\d', word):
            continue
        if 'test' in word.lower():
            continue
        result.append(word)
    return result

# 段階的処理の方が:
# - 可読性が高い
# - デバッグしやすい
# - 各条件を独立にテストできる
# - パフォーマンスも大差ない（短いテキストの場合）
```

### 16.4 アンチパターン: ルックアラウンドの方向を間違える

```python
import re

# 初心者がよくする間違い

# 間違い: "100円" の "100" を取り出したい
# 先読みと後読みを逆に使っている
try:
    wrong = re.findall(r'(?=円)\d+', "100円")
    print(f"間違い: {wrong}")  # => [] -- 何もマッチしない!
except:
    pass

# 正解: "円" は数字の右側にある → 先読みを使う
correct = re.findall(r'\d+(?=円)', "100円")
print(f"正解: {correct}")  # => ['100']

# 覚え方:
# 先読み (Lookahead)  = 前方を見る = 右側をチェック = (?=...) or (?!...)
# 後読み (Lookbehind) = 後方を見る = 左側をチェック = (?<=...) or (?<!...)
#
# "先" = これから進む方向 = 右側
# "後" = 既に通過した方向 = 左側
```

### 16.5 アンチパターン: 置換時のグループ参照ミス

```python
import re

# ルックアラウンドのキャプチャグループは参照できるが注意が必要

text = "old_value: 100"

# NG: ルックアラウンド内のキャプチャを参照しようとする
# ルックアラウンドは消費しないので置換対象に含まれない
result = re.sub(r'(?<=old_value: )(\d+)', r'NEW_\1', text)
print(result)
# => "old_value: NEW_100"  -- これは動くが...

# 注意: 後読みの中にキャプチャグループを入れると
# 予期しない動作になることがある
# 代わりにメインパターン内でキャプチャする方が安全
result = re.sub(r'(old_value: )(\d+)', r'\1NEW_\2', text)
print(result)
# => "old_value: NEW_100"  -- より明確
```

---

## 17. FAQ

### Q1: ルックアラウンドはなぜ「ゼロ幅」と呼ばれるのか？

**A**: ルックアラウンドは文字列中の「位置」をチェックするだけで、文字を「消費」しない。つまりマッチの結果に含まれず、エンジンの現在位置も進まない。これは `^` や `\b` と同じ「アサーション」の一種である:

```python
import re
text = "100円200ドル"
# (?=円) は位置のみチェック -- 円自体は次のマッチで再び利用可能
for m in re.finditer(r'\d+(?=円)', text):
    print(f"位置 {m.start()}-{m.end()}: '{m.group()}'")
# => 位置 0-3: '100'
# "円" はマッチに含まれない
```

### Q2: ルックアラウンドをネストできるか？

**A**: できる。ルックアラウンドの中に別のルックアラウンドを入れることが可能:

```python
import re
# 「前の文字が大文字で、後ろに数字が続く」位置の文字
pattern = r'(?<=(?<=[A-Z])\w)(?=\d)'
# これは複雑なので、通常はシンプルなパターンに分解することを推奨
```

ただし可読性が著しく低下するため、複雑なルックアラウンドのネストは避け、複数のパターンに分割するか、プログラムロジックで処理することを推奨する。

### Q3: Go や Rust でルックアラウンドが使えないのはなぜか？

**A**: Go の RE2 と Rust の regex クレートは **DFA ベースのエンジン** を採用しており、O(n) の線形時間保証を重視している。ルックアラウンドはバックトラックを必要とする場合があり、線形時間保証と相容れない。Rust では `fancy-regex` クレートを使えばルックアラウンド対応のNFAエンジンが利用できるが、O(n) 保証は失われる:

```rust
// Rust 標準 regex: ルックアラウンド不可
// use regex::Regex;

// fancy-regex: ルックアラウンド対応
// use fancy_regex::Regex;
// let re = Regex::new(r"(?<=\$)\d+").unwrap();
```

### Q4: ルックアラウンド内でキャプチャグループを使えるか？

**A**: 使える。ただし注意点がある:

```python
import re

# 先読み内のキャプチャグループ
text = "100円 200ドル 300ユーロ"
pattern = r'\d+(?=(円|ドル|ユーロ))'
matches = re.findall(pattern, text)
print(matches)
# => ['円', 'ドル', 'ユーロ']
# 注: findall はキャプチャグループの内容を返す

# 数値とグループの両方が欲しい場合
pattern = r'(\d+)(?=(円|ドル|ユーロ))'
matches = re.findall(pattern, text)
print(matches)
# => [('100', '円'), ('200', 'ドル'), ('300', 'ユーロ')]
```

### Q5: ルックアラウンドと `\b` の違いは？

**A**: `\b` は「単語境界」という固定的な位置条件だが、ルックアラウンドは任意のパターンを位置条件として使える汎用的なアサーション:

```python
import re

text = "hello world 123"

# \b: 単語文字と非単語文字の境界を検出
print(re.findall(r'\b\w+\b', text))
# => ['hello', 'world', '123']

# (?=...): 任意の条件を右側にチェック
# 例: 「右側にスペースまたは末尾がある位置の単語」
print(re.findall(r'\w+(?=\s|$)', text))
# => ['hello', 'world', '123']

# \b は固定的だが高速
# ルックアラウンドは柔軟だがやや遅い
```

### Q6: ルックアラウンドは POSIX 正規表現で使えるか？

**A**: 使えない。POSIX BRE/ERE はルックアラウンドをサポートしていない。PCRE (Perl Compatible Regular Expressions) の機能である。`grep -P` を使えば PCRE が利用可能:

```bash
# POSIX ERE (grep -E): ルックアラウンド不可
# grep -E '(?<=\$)\d+' file.txt  # エラー

# PCRE (grep -P): ルックアラウンド可能
grep -P '(?<=\$)\d+' file.txt

# macOS の grep は -P をサポートしていない場合がある
# その場合は ggrep (GNU grep) をインストール:
# brew install grep
# ggrep -P '(?<=\$)\d+' file.txt
```

### Q7: 先読みの中に先読みを入れられるか？

**A**: 入れられる。ルックアラウンドは自由にネスト可能:

```python
import re

# 先読みの中に先読み
# 「右側に数字があり、その数字の右側にアルファベットがある位置」
text = "a1b c2d e3 4f"
pattern = r'(?=\d(?=[a-z]))\d'
print(re.findall(pattern, text))
# => ['1', '2']
# '3' はマッチしない（右側がスペース）
# '4' はマッチしない（先に \d にマッチする必要があるが、
# (?=\d...) は位置チェックなのでその位置の右側に \d(?=[a-z]) がある位置）

# 実用例は少ないが、理論的には任意の深さでネスト可能
# ただし可読性のため、できるだけ避けることを推奨
```

### Q8: ルックアラウンドとアトミックグループの関係は？

**A**: アトミックグループ `(?>...)` はバックトラックを禁止するグループで、ルックアラウンドの内部動作と関連がある。ルックアラウンド内のマッチは本質的にアトミック（一度成功/失敗が決まると覆さない）:

```python
# Python re ではアトミックグループ非サポート
# regex モジュールでは利用可能:
# import regex
# pattern = regex.compile(r'(?>abc|ab)c')
# regex.search(pattern, "abc")  # マッチしない
# 通常の (abc|ab)c なら "abc" にマッチ

# ルックアラウンド内は常にアトミック:
# (?=abc|ab) は "abc" で成功したら "ab" は試行しない
# これはパフォーマンスに有利だが、意図しない動作の原因にもなる
```

---

## 18. デバッグとトラブルシューティング

### 18.1 ルックアラウンドのデバッグ手法

```python
import re

# 手法1: 段階的にパターンを構築
text = "Price: $100.50, Tax: $15.00, Total: 115.50"

# Step 1: まずルックアラウンドなしでマッチを確認
print("Step 1:", re.findall(r'\d+\.\d{2}', text))
# => ['100.50', '15.00', '115.50']

# Step 2: 後読みを追加
print("Step 2:", re.findall(r'(?<=\$)\d+\.\d{2}', text))
# => ['100.50', '15.00']

# Step 3: 先読みを追加
print("Step 3:", re.findall(r'(?<=\$)\d+(?=\.\d{2})', text))
# => ['100', '15']

# 手法2: verbose モードでコメント付きパターン
pattern = re.compile(r'''
    (?<=\$)         # 前に $ がある位置（後読み）
    \d+             # 1桁以上の数字（整数部分）
    (?=\.\d{2})     # 後ろに .XX がある位置（先読み）
''', re.VERBOSE)

matches = pattern.findall(text)
print("Verbose:", matches)

# 手法3: finditer で位置情報を確認
for m in re.finditer(r'(?<=\$)\d+\.\d{2}', text):
    print(f"  マッチ: '{m.group()}' at [{m.start()}:{m.end()}]")
```

### 18.2 よくあるエラーと解決策

```python
import re

# エラー1: look-behind requires fixed-width pattern
try:
    re.compile(r'(?<=\w+)\d+')
except re.error as e:
    print(f"エラー1: {e}")
# 解決策: キャプチャグループを使う
# re.findall(r'\w+(\d+)', text)

# エラー2: 先読みが意図通りに動かない
text = "abc123"
# 意図: 数字の前のアルファベットを取得
wrong = re.findall(r'(?=\d)[a-z]+', text)
print(f"間違い: {wrong}")  # => []
# 理由: (?=\d) の位置では [a-z] はマッチしない
correct = re.findall(r'[a-z]+(?=\d)', text)
print(f"正解: {correct}")  # => ['abc']

# エラー3: 否定先読みが多くマッチしすぎる
text = "foo foobar foobaz"
wrong = re.findall(r'(?!foo)\w+', text)
print(f"間違い: {wrong}")  # => 予期しない結果
# 理由: 位置1から 'oo', 'oobar' なども マッチする
correct = re.findall(r'\b(?!foo)\w+', text)
print(f"改善: {correct}")  # => [] -- 全ての単語がfooで始まる

# エラー4: ルックアラウンドの範囲ミス
text = "12ab34cd56"
# 意図: アルファベットに挟まれた数字を抽出
wrong = re.findall(r'(?<=[a-z])\d+(?=[a-z])', text)
print(f"結果: {wrong}")  # => ['34']
# "12" は前にアルファベットがない、"56" は後にアルファベットがない
# 意図通りならOKだが、"12" も含めたい場合は条件を変える
```

### 18.3 regex101 でのデバッグ

```
ルックアラウンドのデバッグには regex101.com が非常に有用:

1. https://regex101.com/ にアクセス
2. 左上で使用する言語(Python, JavaScript, Java など)を選択
3. パターンを入力
4. テスト文字列を入力
5. 右側の "EXPLANATION" でパターンの解説を確認
6. 下部の "MATCH INFORMATION" でマッチの詳細を確認
7. "REGEX DEBUGGER" でステップバイステップの動作を追跡

特に有用なのは REGEX DEBUGGER で、ルックアラウンドの
各位置での成功/失敗を視覚的に確認できる。
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| `(?=X)` | 肯定先読み -- 右側にXがある位置 |
| `(?!X)` | 否定先読み -- 右側にXがない位置 |
| `(?<=X)` | 肯定後読み -- 左側にXがある位置 |
| `(?<!X)` | 否定後読み -- 左側にXがない位置 |
| ゼロ幅 | 文字を消費しない(位置のみチェック) |
| 後読み制約 | 多くのエンジンで固定長のみ |
| AND条件 | 複数の先読みを連鎖して実現 |
| NOT条件 | 否定先読み/否定後読みで実現 |
| 主な用途 | パスワード検証、桁区切り、条件付き抽出/置換 |
| DFAエンジン | ルックアラウンド非サポート(RE2, Rust regex) |
| 設計指針 | シンプルなパターンで代替可能なら避ける |
| パフォーマンス | 先読みの順序最適化、否定文字クラスの活用 |
| テスト | 段階的パターン構築、エッジケース網羅 |

## 次に読むべきガイド

- [02-unicode-regex.md](./02-unicode-regex.md) -- Unicode 正規表現
- [03-performance.md](./03-performance.md) -- パフォーマンスとReDoS対策

## 参考文献

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- 第5章「ルックアラウンド」
2. **MDN - Lookahead assertion** https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions/Lookahead_assertion -- JavaScript のルックアラウンド仕様
3. **Regular-Expressions.info - Lookaround** https://www.regular-expressions.info/lookaround.html -- ルックアラウンドの包括的解説と全エンジン比較
4. **Python re module documentation** https://docs.python.org/3/library/re.html -- Python 標準ライブラリのルックアラウンド仕様
5. **TC39 Proposal - Lookbehind Assertions** https://github.com/tc39/proposal-regexp-lookbehind -- JavaScript ES2018 後読みの提案書
6. **RE2 Syntax** https://github.com/google/re2/wiki/Syntax -- RE2(Go)のサポート構文一覧とルックアラウンド非サポートの理由
7. **fancy-regex crate** https://docs.rs/fancy-regex/ -- Rust でルックアラウンドを使うためのクレート
8. **regex101.com** https://regex101.com/ -- ルックアラウンドのデバッグに有用なオンラインツール
