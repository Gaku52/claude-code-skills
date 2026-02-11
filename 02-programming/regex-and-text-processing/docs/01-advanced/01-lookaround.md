# 先読み・後読み -- (?=)(?!)(?<=)(?<!)

> 先読み(Lookahead)と後読み(Lookbehind)はゼロ幅アサーションであり、文字を消費せずに位置条件を指定する。パスワード強度検証、複雑な抽出条件、置換対象の限定など、通常のパターンでは表現しにくい制約を可能にする強力な機能である。

## この章で学ぶこと

1. **4種類のルックアラウンドの構文と動作** -- 肯定/否定の先読み/後読みの正確な意味
2. **ゼロ幅アサーションの概念** -- 文字を消費しないマッチの仕組みと応用
3. **実践的なユースケース** -- パスワード検証、数値フォーマット、複合条件抽出

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

固定長の制約:
  (?<=abc)    OK  -- 3文字固定
  (?<=ab|cd)  OK  -- 各選択肢が同じ長さ
  (?<=a{3})   OK  -- 固定回数
  (?<=a+)     NG  -- 可変長 (Python, Java, Perl で不可)
  (?<=a*)     NG  -- 可変長
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

---

## 6. 実践的なユースケース

### 6.1 パスワード強度検証

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

### 6.2 数値の桁区切り

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

### 6.3 特定コンテキストの置換

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

---

## 7. ASCII 図解

### 7.1 4種類のルックアラウンド動作

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

### 7.2 パスワード検証の先読みチェーン

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

### 7.3 桁区切りの先読み動作

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

---

## 8. 比較表

### 8.1 ルックアラウンド完全比較

| 種類 | 構文 | 意味 | 例 | マッチ |
|------|------|------|-----|--------|
| 肯定先読み | `X(?=Y)` | XのあとにYがある | `\d+(?=円)` | "100" in "100円" |
| 否定先読み | `X(?!Y)` | XのあとにYがない | `\d+(?!円)` | "200" in "200ドル" |
| 肯定後読み | `(?<=Y)X` | Xの前にYがある | `(?<=\$)\d+` | "100" in "$100" |
| 否定後読み | `(?<!Y)X` | Xの前にYがない | `(?<!\$)\d+` | "42" in "count: 42" |

### 8.2 言語サポート状況

| 機能 | Python | JavaScript | Java | Go(RE2) | Rust |
|------|--------|------------|------|---------|------|
| 肯定先読み `(?=)` | OK | OK | OK | 不可 | 不可 |
| 否定先読み `(?!)` | OK | OK | OK | 不可 | 不可 |
| 肯定後読み `(?<=)` | OK(固定長) | OK(可変長) | OK(固定長) | 不可 | 不可 |
| 否定後読み `(?<!)` | OK(固定長) | OK(可変長) | OK(固定長) | 不可 | 不可 |
| 可変長後読み | regex モジュール | ES2018+ | 不可 | N/A | fancy-regex |

---

## 9. アンチパターン

### 9.1 アンチパターン: ルックアラウンドの過剰使用

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

### 9.2 アンチパターン: 可変長後読みを想定する

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

# 回避策1: 別のアプローチ
pattern = r'https?://(\w+)'
match = re.search(pattern, "https://example.com")
print(match.group(1))  # => 'example'

# 回避策2: regex モジュール(可変長後読みをサポート)
# import regex
# regex.search(r'(?<=https?://)\w+', "https://example.com")
```

---

## 10. FAQ

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
| 主な用途 | パスワード検証、桁区切り、条件付き抽出/置換 |
| DFAエンジン | ルックアラウンド非サポート(RE2, Rust regex) |
| 設計指針 | シンプルなパターンで代替可能なら避ける |

## 次に読むべきガイド

- [02-unicode-regex.md](./02-unicode-regex.md) -- Unicode 正規表現
- [03-performance.md](./03-performance.md) -- パフォーマンスとReDoS対策

## 参考文献

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- 第5章「ルックアラウンド」
2. **MDN - Lookahead assertion** https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions/Lookahead_assertion -- JavaScript のルックアラウンド仕様
3. **Regular-Expressions.info - Lookaround** https://www.regular-expressions.info/lookaround.html -- ルックアラウンドの包括的解説と全エンジン比較
