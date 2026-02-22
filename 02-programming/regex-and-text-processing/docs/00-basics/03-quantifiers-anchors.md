# 量指定子・アンカー -- *+?{n,m}、^$\b、貪欲/怠惰

> 量指定子(Quantifier)は繰り返し回数を制御し、アンカー(Anchor)は位置を制約する。貪欲(greedy)マッチと怠惰(lazy)マッチの違いを正確に理解することが、意図通りのパターンを書くための鍵である。

## この章で学ぶこと

1. **量指定子の種類と動作** -- `*` `+` `?` `{n,m}` の正確な意味と使い分け
2. **貪欲マッチと怠惰マッチ** -- デフォルト動作が「最長一致」である理由とその制御方法
3. **独占的量指定子とアトミックグループ** -- バックトラック禁止による高速化
4. **アンカーの種類と応用** -- `^` `$` `\b` `\A` `\Z` `\G` による位置指定の全パターン
5. **言語ごとの差異** -- Python / JavaScript / Java / Ruby / Perl / Go / Rust の動作比較
6. **パフォーマンスへの影響** -- 量指定子の選択がバックトラック回数に与える影響

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

**なぜ `*` は空文字列にマッチするのか**:

`*` は「0回以上」を意味するため、対象文字が存在しない位置でも「0回の繰り返し」として空文字列にマッチする。これは正規表現の数学的定義に由来する。形式言語理論では、空文字列(epsilon, ε)は任意の言語の Kleene 閉包に含まれる。

```python
import re

# 空文字列マッチの詳細な動作
text = "XY"
matches = list(re.finditer(r'a*', text))
for m in matches:
    print(f"位置{m.start()}-{m.end()}: '{m.group()}'")
# 位置0-0: ''    ← X の前(aがない = 0回マッチ)
# 位置1-1: ''    ← Y の前(aがない = 0回マッチ)
# 位置2-2: ''    ← 文字列末尾(aがない = 0回マッチ)

# 実用上の問題: * の代わりに + を使うべき場面
text = "abc 123 def"
# NG: 空文字列にもマッチ
print(re.findall(r'\d*', text))
# => ['', '', '', '', '123', '', '', '', '', '']

# OK: 最低1桁を要求
print(re.findall(r'\d+', text))
# => ['123']
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

**`{n,m}` の構文上の注意点**:

```python
import re

# {n,m} のスペースは許されない(エンジンによる)
text = "aaa"

# Python: スペースありはリテラルとして扱われる
print(re.findall(r'a{2,3}', text))   # => ['aaa']  (量指定子)
print(re.findall(r'a{2, 3}', text))  # => []  (リテラル文字列 "a{2, 3}")

# {n,m} で n > m はエラー
try:
    re.compile(r'a{3,2}')
except re.error as e:
    print(f"エラー: {e}")
    # => "min repeat greater than max repeat"

# 大きな繰り返し回数の制限
try:
    re.compile(r'a{1,65536}')  # Python は上限あり
except re.error as e:
    print(f"エラー: {e}")
```

**各言語での `{,m}` サポート状況**:

```
┌──────────────┬─────────────────────────────────────┐
│ 言語/エンジン │ {,m} のサポート                       │
├──────────────┼─────────────────────────────────────┤
│ Python       │ ○ 対応({0,m} と同等)               │
│ JavaScript   │ ✗ リテラルとして扱う                  │
│ Java         │ ✗ リテラルとして扱う                  │
│ PCRE         │ ○ 対応                              │
│ Ruby         │ ○ 対応                              │
│ Perl         │ ○ 対応                              │
│ Go (RE2)     │ ○ 対応                              │
│ Rust (regex) │ ○ 対応                              │
│ .NET         │ ○ 対応                              │
└──────────────┴─────────────────────────────────────┘
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

### 1.4 量指定子の適用対象

量指定子は「直前の要素」に対して適用される。この「直前の要素」が何であるかを正確に理解することが重要である。

```python
import re

text = "abcabcabc"

# 1文字に適用: 'c' が0回以上
print(re.findall(r'abc*', text))      # => ['abc', 'abc', 'abc']
# c* は 'c' の繰り返し

# グループに適用: 'abc' が1回以上
print(re.findall(r'(?:abc)+', text))  # => ['abcabcabc']
# (?:abc)+ は 'abc' グループの繰り返し

# 文字クラスに適用: [a-c] が2回以上
print(re.findall(r'[a-c]{2,}', text)) # => ['abcabcabc']

# エスケープシーケンスに適用: \d が3回
text2 = "abc123def456"
print(re.findall(r'\d{3}', text2))    # => ['123', '456']
```

**量指定子の連続使用(ネスト)**:

```python
import re

# 量指定子を直接連続させるとエラー
try:
    re.compile(r'a**')
except re.error as e:
    print(f"エラー: {e}")  # multiple repeat

# グループを使えばネストできる
text = "aaa bbb aaa bbb aaa"
print(re.findall(r'(?:a{3}\s?){2,}', text))
# => ['aaa bbb aaa bbb aaa']

# 実用例: 繰り返しのIPアドレスパターン
ip_pattern = r'(?:\d{1,3}\.){3}\d{1,3}'
print(re.findall(ip_pattern, "Server 192.168.1.100 and 10.0.0.1"))
# => ['192.168.1.100', '10.0.0.1']
```

### 1.5 量指定子と空マッチの関係

```python
import re

# findall/finditer における空マッチの扱い
text = "abc"

# Python 3.7+ では連続する空マッチを防ぐ仕様変更あり
# Python 3.6以前: 空マッチの直後に同じ位置で再試行
# Python 3.7以降: 空マッチの直後は1文字進めてから再試行

# * による空マッチ
for m in re.finditer(r'x*', text):
    print(f"位置{m.start()}-{m.end()}: '{m.group()}'")
# Python 3.7+:
# 位置0-0: ''
# 位置1-1: ''
# 位置2-2: ''
# 位置3-3: ''

# sub における空マッチの扱い
print(re.sub(r'x*', '-', 'abc'))
# Python 3.7+: '-a-b-c-'
# Python 3.6以前: '-a-b-c-'(実装により異なる）
```

```javascript
// JavaScript における空マッチ
const text = "abc";

// ES2020+ の matchAll
const matches = [...text.matchAll(/x*/g)];
console.log(matches.map(m => `${m.index}: '${m[0]}'`));
// ["0: ''", "1: ''", "2: ''", "3: ''"]

// replace における空マッチ
console.log("abc".replace(/x*/g, "-"));
// "-a-b-c-"
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

**貪欲マッチの内部動作(バックトラッキングの詳細)**:

```
パターン: <div>.*</div>
テキスト: <div>hello</div><div>world</div>
          0123456789...

ステップ 1: '<div>' がテキスト位置0-4にマッチ
ステップ 2: '.*' が貪欲に全残り文字を消費
            → マッチ範囲: 位置5 から 位置37(末尾)まで
ステップ 3: '</div>' のマッチを試行
            → 残り文字がない → 失敗 → バックトラック
ステップ 4: 1文字戻す(位置36): '>' ≠ '<' → 失敗 → バックトラック
ステップ 5: 1文字戻す(位置35): '>' ≠ '<' → 失敗 → バックトラック
  ...
ステップ N: 位置31まで戻す: '</div>' がマッチ!
            → 結果: '<div>hello</div><div>world</div>'

バックトラック回数: 約6回
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

**怠惰マッチの内部動作**:

```
パターン: <div>.*?</div>
テキスト: <div>hello</div><div>world</div>

ステップ 1: '<div>' がテキスト位置0-4にマッチ
ステップ 2: '.*?' が怠惰に0文字でまず試行
ステップ 3: '</div>' のマッチを試行 → 位置5 'h' ≠ '<' → 失敗
            → .*? を1文字拡張
ステップ 4: .*? = 'h'、'</div>' を位置6から試行 → 'e' ≠ '<' → 失敗
            → .*? を1文字拡張
ステップ 5: .*? = 'he'、'</div>' を位置7から試行 → 'l' ≠ '<' → 失敗
  ...
ステップ 8: .*? = 'hello'、'</div>' を位置10から試行 → マッチ!
            → 結果: '<div>hello</div>'

ステップ拡張回数: 5回
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

**重要な注意: 怠惰マッチは「最短」であって「最左最短」ではない**:

```python
import re

text = "aXbYaZb"

# 怠惰マッチは左端から始まる(最左一致の原則は変わらない)
print(re.search(r'a.*?b', text).group())  # => 'aXb'(左端から最短)
# 'aZb' のほうが短いが、'aXb' が先に見つかる

# findall は全ての非重複マッチを返す
print(re.findall(r'a.*?b', text))  # => ['aXb', 'aZb']
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

**独占的量指定子の実践的な使い方**:

```java
import java.util.regex.*;

public class PossessiveExample {
    public static void main(String[] args) {
        // 1. 高速な不一致検出
        // 独占的量指定子は「マッチしない」ことを高速に判定する
        String longText = "a".repeat(100000) + "X";

        // 貪欲: a*a → バックトラックが大量に発生
        long start1 = System.nanoTime();
        Pattern.matches("a*a$", longText);  // 遅い
        long time1 = System.nanoTime() - start1;

        // 独占的: a*+X → バックトラックなしで即座に判定
        long start2 = System.nanoTime();
        Pattern.matches("a*+X", longText);  // 高速
        long time2 = System.nanoTime() - start2;

        System.out.println("貪欲: " + time1 + "ns");
        System.out.println("独占的: " + time2 + "ns");

        // 2. CSV解析での活用
        String csvLine = "field1,field2,\"field with, comma\",field4";

        // 引用符で囲まれたフィールド: 引用符以外を独占的に消費
        Pattern csvQuoted = Pattern.compile("\"[^\"]*+\"");
        Matcher m = csvQuoted.matcher(csvLine);
        while (m.find()) {
            System.out.println("マッチ: " + m.group());
        }
        // => マッチ: "field with, comma"

        // 3. 数値リテラルの解析
        // 整数部分を独占的に消費し、小数点以下は別途処理
        Pattern numPattern = Pattern.compile("\\d++\\.?\\d*+");
        String expr = "3.14 + 42 - 0.5";
        m = numPattern.matcher(expr);
        while (m.find()) {
            System.out.println("数値: " + m.group());
        }
        // => 数値: 3.14
        // => 数値: 42
        // => 数値: 0.5
    }
}
```

**Python regex モジュールでの独占的量指定子**:

```python
# pip install regex
import regex

text = "aaaaab"

# 独占的量指定子
m = regex.search(r'a++b', text)
print(m.group())  # => 'aaaaab'

# 独占的でマッチ失敗するケース
m = regex.search(r'a++a', text)
print(m)  # => None (a++ が全ての a を消費し、バックトラックしない)

# アトミックグループ (?>...) も同等の機能
m = regex.search(r'(?>a+)b', text)
print(m.group())  # => 'aaaaab'

m = regex.search(r'(?>a+)a', text)
print(m)  # => None
```

### 2.6 アトミックグループ

アトミックグループ `(?>...)` は、グループ内のマッチが確定した後はバックトラックしないグループである。独占的量指定子と概念的に同等だが、より柔軟な範囲に適用できる。

```java
import java.util.regex.*;

public class AtomicGroupExample {
    public static void main(String[] args) {
        // 独占的量指定子: 単一の量指定子に適用
        // a*+  ≡  (?>a*)

        // アトミックグループ: 複数の要素に適用可能
        // (?>abc|abcd)  -- abc がマッチしたらバックトラックしない

        String text = "abcd";

        // 通常の選択
        Pattern p1 = Pattern.compile("(?:abc|abcd)d");
        System.out.println(p1.matcher(text).find());  // false
        // abc がマッチ → d を期待 → d がある → 全体マッチ... いや待て
        // 実際は "abcd" 全体が対象で、abc + d = abcd → OK

        // より明確な例
        String text2 = "abcde";
        Pattern p2 = Pattern.compile("(?:abc|abcde)$");
        Pattern p3 = Pattern.compile("(?>abc|abcde)$");

        System.out.println(p2.matcher(text2).find());  // true (abcde にマッチ)
        System.out.println(p3.matcher(text2).find());  // false (abc で確定、$に失敗してもバックトラックしない)
    }
}
```

```ruby
# Ruby はアトミックグループをネイティブサポート
text = "aaaaab"

# アトミックグループ
puts text.match(/(?>a+)b/)    # => aaaaab
puts text.match(/(?>a+)a/)    # => nil (バックトラックしない)

# 実用例: メールアドレスの高速検証
email_pattern = /\A(?>[\w.+-]+)@(?>[\w-]+\.)+\w{2,}\z/
puts "user@example.com".match?(email_pattern)  # => true
puts "invalid@@email".match?(email_pattern)     # => false
```

### 2.7 量指定子比較表

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

### 2.8 バックトラック回数の比較

具体的な例でバックトラック回数を比較し、パフォーマンスへの影響を可視化する。

```python
import re
import time

# テストケース: HTML タグの抽出
html = '<div class="container">' + 'x' * 10000 + '</div>'

# パターン1: 貪欲 + 否定文字クラス(最速)
pattern1 = r'<div[^>]*>[^<]*</div>'

# パターン2: 怠惰(中速)
pattern2 = r'<div.*?>.*?</div>'

# パターン3: 貪欲(最遅 -- 大量バックトラック)
pattern3 = r'<div.*>.*</div>'

for name, pattern in [("否定クラス", pattern1), ("怠惰", pattern2), ("貪欲", pattern3)]:
    start = time.perf_counter()
    for _ in range(1000):
        re.search(pattern, html, re.DOTALL)
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}秒")

# 典型的な結果:
# 否定クラス: 0.0150秒
# 怠惰:      0.0450秒
# 貪欲:      0.0800秒
```

```
バックトラック回数の概算:

パターン: <div>.*</div>  (貪欲)
テキスト: <div>XXXX...XXXX</div> (Xが10000個)

1. .* が10006文字を消費（'XXXX...XXXX</div>' 全体）
2. '</div>' のマッチを試行 → 失敗
3. 1文字ずつバックトラック × 約10000回
4. '</div>' の位置に到達 → マッチ
→ バックトラック回数: 約10000回

パターン: <div>.*?</div>  (怠惰)
1. .*? が0文字で開始
2. '</div>' のマッチを試行 → 失敗 → 1文字拡張
3. 繰り返し × 約10000回
4. 'X' を全て消費した位置で '</div>' がマッチ
→ 拡張試行回数: 約10000回（貪欲と同程度だが方向が逆）

パターン: <div>[^<]*</div>  (否定クラス)
1. [^<]* が < 以外の全文字を一度に消費
2. </div> のマッチを試行 → 成功
→ バックトラック回数: 0回（最速）
```

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

**`$` と末尾改行の微妙な挙動**:

```python
import re

# Python の $ は文字列末尾の改行の「前」にもマッチする
text_with_newline = "hello\n"
text_without_newline = "hello"

print(re.search(r'hello$', text_with_newline))       # マッチする!
print(re.search(r'hello$', text_without_newline))     # マッチする

# \Z は末尾改行の前にもマッチ(Python固有)
print(re.search(r'hello\Z', text_with_newline))       # None!
print(re.search(r'hello\Z', text_without_newline))    # マッチする

# 末尾改行を含めて完全にマッチさせるには
print(re.search(r'hello\n?\Z', text_with_newline))    # マッチ
```

```javascript
// JavaScript の $ の挙動
const text = "hello\n";

// $ はデフォルトで文字列末尾のみ
console.log(/hello$/.test(text));        // false (改行があるため)
console.log(/hello$/.test("hello"));     // true

// m フラグで各行末尾にマッチ
console.log(/hello$/m.test(text));       // true
```

```ruby
# Ruby の $ は常に行末(\n の直前)にマッチする
text = "hello\nworld"
puts text.scan(/\w+$/)   # => ["hello", "world"]
# Ruby では $ はデフォルトで MULTILINE 的動作
# 文字列末尾だけにマッチさせるには \z を使う
puts text.match?(/world\z/)  # => true
puts text.match?(/hello\z/)  # => false
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

**各言語での `\z` vs `\Z` の違い**:

```
┌──────────────┬──────────────────────────────────────────┐
│ アンカー      │ 動作                                      │
├──────────────┼──────────────────────────────────────────┤
│ \Z (Python)  │ 文字列末尾。末尾の改行は含まない            │
│ \z (Python)  │ 非サポート                                 │
│ \Z (Ruby)    │ 文字列末尾(末尾改行の前にもマッチ)          │
│ \z (Ruby)    │ 文字列の絶対末尾(末尾改行も超えない)        │
│ \Z (Java)    │ 文字列末尾(末尾改行の前にもマッチ)          │
│ \z (Java)    │ 文字列の絶対末尾                            │
│ \Z (Perl)    │ 文字列末尾(末尾改行の前にもマッチ)          │
│ \z (Perl)    │ 文字列の絶対末尾                            │
└──────────────┴──────────────────────────────────────────┘

注意: Python の \Z は他の言語の \z に相当する。
Python には他言語の \Z に相当するアンカーがない。
```

```ruby
# Ruby での \z と \Z の明確な違い
text = "hello\n"

puts text.match?(/hello\Z/)  # => true  (\n の前にマッチ)
puts text.match?(/hello\z/)  # => false (絶対末尾は \n)
puts text.match?(/hello\n\z/) # => true (改行含めて末尾)
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

**単語境界の正確な定義**:

```python
import re

# \b は「位置」にマッチする(文字を消費しない)
# 以下の4つの位置で \b がマッチする:
# 1. 文字列先頭で、最初の文字が \w の場合
# 2. 文字列末尾で、最後の文字が \w の場合
# 3. \w の直後に \W が続く位置
# 4. \W の直後に \w が続く位置

text = "Hello, World! 123"
#       ^     ^^ ^^  ^ ^  ^
#       1     34 34  1 34  2

boundaries = []
for i in range(len(text) + 1):
    if re.search(r'\b', text[max(0,i-1):i+1] if i > 0 else text[0:1]):
        pass  # 簡略化

# 具体的な境界位置の確認
for m in re.finditer(r'\b', text):
    print(f"位置{m.start()}: ...{text[max(0,m.start()-1):m.start()+1]}...")
# 位置0:  ...H...
# 位置5:  ...o,...
# 位置7:  ...W...
# 位置12: ...d!...
# 位置14: ...1...
# 位置17: ...3 (末尾)
```

**単語境界と日本語**:

```python
import re

# \b は ASCII 単語文字(\w = [a-zA-Z0-9_])に基づく
# → 日本語文字は \W とみなされるため、各文字の前後が境界になる

text = "Hello世界World"
print(re.findall(r'\b\w+\b', text))
# => ['Hello', 'World']
# '世界' は \w に含まれないため単独の単語として認識されない

# Unicode 対応の単語境界には regex モジュールを使用
import regex
# regex モジュールの UNICODE フラグ
print(regex.findall(r'\b\w+\b', text, flags=regex.UNICODE))
# => ['Hello世界World']  (日本語文字も \w に含まれる)
```

```javascript
// JavaScript の \b と Unicode
const text = "Hello世界World";

// ES2015+ の u フラグでも \b は ASCII ベース
console.log(text.match(/\b\w+\b/gu));
// => ['Hello', 'World']（'世界' は含まれない）

// Unicode 単語境界を使うには
// ECMAScript 2024 の /v フラグ + Unicode property
console.log(text.match(/[\p{L}\p{N}]+/gu));
// => ['Hello世界World'] (Unicode文字プロパティで代替)
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

### 3.6 `\G` アンカーの詳細

`\G` は「前回のマッチが終了した位置」にアンカーする特殊なアンカーである。連続する繰り返しマッチに有用。

```java
import java.util.regex.*;

public class GAnchorExample {
    public static void main(String[] args) {
        // \G は前回マッチの終了位置にアンカーする
        // 最初のマッチでは文字列先頭(\A)と同じ位置

        String text = "abc123def456ghi";
        Pattern p = Pattern.compile("\\G\\w");
        Matcher m = p.matcher(text);

        StringBuilder result = new StringBuilder();
        while (m.find()) {
            result.append(m.group());
        }
        System.out.println(result);
        // => "abc123def456ghi" (全文字が連続してマッチ)

        // 途中でマッチが途切れると \G は再開しない
        text = "abc 123 def";
        p = Pattern.compile("\\G\\w");
        m = p.matcher(text);

        result = new StringBuilder();
        while (m.find()) {
            result.append(m.group());
        }
        System.out.println(result);
        // => "abc" (スペースで途切れ、以降はマッチしない)
    }
}
```

```perl
# Perl での \G の活用: トークナイザ
my $text = "3.14 + 42 * 2.0";
my @tokens;

while ($text =~ /\G\s*/gc) {  # 空白をスキップ
    if ($text =~ /\G(\d+\.?\d*)/gc) {
        push @tokens, {type => 'NUMBER', value => $1};
    } elsif ($text =~ /\G([+\-*\/])/gc) {
        push @tokens, {type => 'OP', value => $1};
    } else {
        die "Unexpected character at position " . pos($text);
    }
}

for my $token (@tokens) {
    print "$token->{type}: $token->{value}\n";
}
# NUMBER: 3.14
# OP: +
# NUMBER: 42
# OP: *
# NUMBER: 2.0
```

---

## 4. 量指定子とアンカーの組み合わせパターン

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

**各言語での完全一致の方法**:

```javascript
// JavaScript: ^...$ を使う(fullmatch がない)
function validateDate(s) {
    return /^\d{4}-\d{2}-\d{2}$/.test(s);
}
console.log(validateDate("2026-02-11"));  // true
console.log(validateDate("2026-02-11 ")); // false
```

```java
// Java: matches() メソッドは暗黙的に ^...$ を追加
String date = "2026-02-11";
System.out.println(date.matches("\\d{4}-\\d{2}-\\d{2}"));  // true

// find() は部分一致
Pattern p = Pattern.compile("\\d{4}-\\d{2}-\\d{2}");
Matcher m = p.matcher("date: 2026-02-11");
System.out.println(m.find());     // true (部分一致)
System.out.println(m.matches());  // false (完全一致)
```

```ruby
# Ruby: \A...\z を推奨(^ $ は行単位のため)
def validate_date(s)
  s.match?(/\A\d{4}-\d{2}-\d{2}\z/)
end

puts validate_date("2026-02-11")       # => true
puts validate_date("2026-02-11\nfoo")  # => false (\z は文字列末尾)
# ^ $ を使うと:
puts "2026-02-11\nfoo".match?(/^\d{4}-\d{2}-\d{2}$/)  # => true (行末にマッチ!)
```

```go
// Go (RE2): 常に完全一致には \A...\z は不要(MatchString は部分一致)
package main

import (
    "fmt"
    "regexp"
)

func main() {
    // MatchString は部分一致
    matched, _ := regexp.MatchString(`\d{4}-\d{2}-\d{2}`, "date: 2026-02-11")
    fmt.Println(matched)  // true

    // 完全一致には ^...$ を明示
    matched, _ = regexp.MatchString(`^\d{4}-\d{2}-\d{2}$`, "2026-02-11")
    fmt.Println(matched)  // true
    matched, _ = regexp.MatchString(`^\d{4}-\d{2}-\d{2}$`, "date: 2026-02-11")
    fmt.Println(matched)  // false
}
```

### 4.3 複合バリデーションパターン

```python
import re

# パスワード強度チェック: 複数条件を先読みで組み合わせ
def validate_password(pw):
    """
    要件:
    - 8文字以上20文字以下
    - 大文字を1文字以上含む
    - 小文字を1文字以上含む
    - 数字を1文字以上含む
    - 特殊文字を1文字以上含む
    """
    pattern = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*]).{8,20}$'
    return bool(re.match(pattern, pw))

print(validate_password("Abc123!x"))    # True
print(validate_password("abc123!x"))    # False (大文字なし)
print(validate_password("Abc!x"))       # False (8文字未満)
print(validate_password("Abcdefgh"))    # False (数字・特殊文字なし)

# 日本語を含むユーザー名の検証
def validate_username(name):
    """
    要件:
    - 2文字以上20文字以下
    - 日本語(ひらがな・カタカナ・漢字)、英数字、アンダースコアのみ
    - 先頭は英字または日本語文字
    """
    pattern = r'^[a-zA-Z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF][\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]{1,19}$'
    return bool(re.match(pattern, name))

print(validate_username("田中太郎"))     # True
print(validate_username("user_123"))     # True
print(validate_username("1abc"))         # False (先頭が数字)
print(validate_username("a"))            # False (2文字未満)
```

### 4.4 境界を活用したテキスト置換

```python
import re

# 単語単位の置換(部分一致を防ぐ)
text = "The cat concatenated the catalog"

# NG: 部分一致で置換してしまう
bad = re.sub(r'cat', 'dog', text)
print(bad)  # => 'The dog dogdogenated the dogalog'

# OK: 単語境界を使用
good = re.sub(r'\bcat\b', 'dog', text)
print(good)  # => 'The dog concatenated the catalog'

# 変数名のリネーム(プログラミングで有用)
code = """
count = 0
count += 1
account = get_account()
print(count)
"""

# 変数 count を total にリネーム
renamed = re.sub(r'\bcount\b', 'total', code)
print(renamed)
# count → total に変更されるが、account は変更されない
```

```javascript
// JavaScript での単語境界を活用した置換
const code = `
function getData() {
    const data = fetchData();
    const dataMap = new Map();
    return processData(data);
}`;

// data を info にリネーム(getData, fetchData, processData は変更しない)
const renamed = code.replace(/\bdata\b/g, "info");
console.log(renamed);
// data → info に変更、getData/fetchData/processData/dataMap は変更されない
```

### 4.5 行頭・行末を活用したフォーマット処理

```python
import re

# コメント行の除去
config = """# Database settings
host = localhost
# port = 5432
port = 3306
# Comment
user = admin
"""

# # で始まる行を除去
cleaned = re.sub(r'^#.*$\n?', '', config, flags=re.MULTILINE)
print(cleaned)
# host = localhost
# port = 3306
# user = admin

# 各行にインデントを追加
text = """line 1
line 2
line 3"""

indented = re.sub(r'^', '    ', text, flags=re.MULTILINE)
print(indented)
#     line 1
#     line 2
#     line 3

# 各行末の空白を除去
text_with_trailing = "hello   \nworld  \nfoo \n"
trimmed = re.sub(r' +$', '', text_with_trailing, flags=re.MULTILINE)
print(repr(trimmed))
# 'hello\nworld\nfoo\n'

# 空行の除去
text_with_blanks = """first

second


third"""
no_blanks = re.sub(r'^\s*$\n', '', text_with_blanks, flags=re.MULTILINE)
print(no_blanks)
# first
# second
# third
```

---

## 5. 言語別の量指定子・アンカーの差異

### 5.1 量指定子のサポート状況

```
┌──────────────┬───────┬───────┬──────────┬────────────┐
│ 機能          │ Python│ JS    │ Java     │ Ruby/Perl  │
├──────────────┼───────┼───────┼──────────┼────────────┤
│ 貪欲 *+?{n,m}│ ○     │ ○     │ ○        │ ○          │
│ 怠惰 *? +?   │ ○     │ ○     │ ○        │ ○          │
│ 独占的 *+ ++  │ ✗(re) │ ✗     │ ○        │ ✗(※1)     │
│ アトミック(?>)│ ✗(re) │ ✗(※2)│ ○(※3)   │ ○          │
│ {,m} 省略形  │ ○     │ ✗     │ ✗        │ ○          │
└──────────────┴───────┴───────┴──────────┴────────────┘

※1: Ruby 3.0+ で一部サポート(正規表現リテラルのみ)
※2: ECMAScript 2025 提案段階
※3: Java 独自構文 (?>...) をサポート
```

### 5.2 アンカーの言語差異

```python
# Python
import re
text = "line1\nline2\n"

# ^ $ はデフォルトで文字列先頭/末尾
# re.MULTILINE で各行の先頭/末尾に変更
print(re.findall(r'^\w+', text, re.M))  # => ['line1', 'line2']

# \A \Z は MULTILINE の影響を受けない
print(re.findall(r'\A\w+', text, re.M))  # => ['line1']
```

```javascript
// JavaScript
const text = "line1\nline2\n";

// ^ $ はデフォルトで文字列先頭/末尾
// m フラグで各行の先頭/末尾に変更
console.log(text.match(/^\w+/gm));  // => ['line1', 'line2']

// JavaScript には \A \Z がない!
// 代わりに m フラグなしの ^ $ を使うか、
// 先読み・後読みで代替する
console.log(text.match(/^\w+/g));  // => ['line1']  (m なし = 文字列先頭)
```

```ruby
# Ruby
text = "line1\nline2\n"

# Ruby の ^ $ は常に行単位(MULTILINE相当)
puts text.scan(/^\w+/)   # => ["line1", "line2"]

# 文字列先頭/末尾には \A \z を使う
puts text.scan(/\A\w+/)  # => ["line1"]

# Ruby には \Z(末尾改行の前にもマッチ)と \z(絶対末尾)の両方がある
puts "hello\n".match?(/hello\Z/)  # => true
puts "hello\n".match?(/hello\z/)  # => false
```

```go
// Go (RE2エンジン)
package main

import (
    "fmt"
    "regexp"
)

func main() {
    text := "line1\nline2\n"

    // Go のデフォルト: ^ $ は文字列先頭/末尾
    re1 := regexp.MustCompile(`^\w+`)
    fmt.Println(re1.FindAllString(text, -1))  // => [line1]

    // (?m) フラグで MULTILINE モード
    re2 := regexp.MustCompile(`(?m)^\w+`)
    fmt.Println(re2.FindAllString(text, -1))  // => [line1 line2]

    // Go には \A \Z \z がない
    // ^ $ を適切にフラグ制御して使う
}
```

### 5.3 JavaScript 固有の注意点

```javascript
// ES2018 で導入された dotAll フラグ (s)
const text = "line1\nline2\nline3";

// s フラグなし: . は改行にマッチしない
console.log(text.match(/^.+$/));       // => ['line1']
console.log(text.match(/^.+$/m));      // => ['line1'] (最初の行)
console.log(text.match(/^.+$/gm));     // => ['line1', 'line2', 'line3']

// s フラグあり: . が改行にもマッチ
console.log(text.match(/^.+$/s));      // => ['line1\nline2\nline3']
console.log(text.match(/^.+$/ms));     // => ['line1\nline2\nline3']
// sm の組み合わせ: ^ は行頭にマッチするが、.+ が改行も含むため全体がマッチ

// ES2024 の v フラグ（Unicode Sets）
// v フラグは u フラグの上位互換
const emoji = "Hello 🌍 World 🎉";
console.log(emoji.match(/\p{Emoji}/gv));  // => ['🌍', '🎉']
```

---

## 6. アンチパターン

### 6.1 アンチパターン: `.*` の無秩序な使用

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

### 6.2 アンチパターン: `^` `$` のMULTILINEフラグ忘れ

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

### 6.3 アンチパターン: ネストした量指定子によるReDoS

```python
import re
import time

# 危険: ネストした量指定子は指数的バックトラックを引き起こす
dangerous_patterns = [
    r'(a+)+b',          # ネストした +
    r'(a*)*b',          # ネストした *
    r'(a|a)*b',         # 重複する選択肢 + 量指定子
    r'(.*a){10}',       # .* の大量繰り返し
    r'(\w+\s*)+$',      # よくある入力検証パターン
]

safe_input = "a" * 20 + "b"      # マッチする → 高速
evil_input = "a" * 25             # マッチしない → 指数的に遅い

for pattern in dangerous_patterns:
    # 安全な入力
    start = time.perf_counter()
    re.search(pattern, safe_input)
    safe_time = time.perf_counter() - start

    # 悪意のある入力(タイムアウト付き)
    # 注意: 実際の実行は非常に遅くなる可能性あり
    print(f"パターン: {pattern}")
    print(f"  安全な入力: {safe_time:.6f}秒")
    # evil_input のテストは省略(ReDoSの危険性)

# 安全な代替パターン
safe_alternatives = {
    r'(a+)+b':     r'a+b',           # ネスト除去
    r'(a*)*b':     r'a*b',           # ネスト除去
    r'(a|a)*b':    r'a*b',           # 重複除去
    r'(\w+\s*)+$': r'[\w\s]+$',      # 文字クラスに統合
}
```

### 6.4 アンチパターン: 不要な量指定子

```python
import re

# NG: 不要に複雑な量指定子
bad_patterns = {
    r'\d{1,}':    r'\d+',       # {1,} は + と同じ
    r'\d{0,}':    r'\d*',       # {0,} は * と同じ
    r'\d{0,1}':   r'\d?',       # {0,1} は ? と同じ
    r'[a-z]{1}':  r'[a-z]',     # {1} は不要
    r'(?:ab){1}': r'ab',        # {1} のグループも不要
}

for bad, good in bad_patterns.items():
    print(f"NG: {bad:20s} → OK: {good}")

# NG: 不要なグループ内の量指定子
# (?: )+  の中に量指定子がある場合、外側と内側の両方が必要か確認
text = "hello   world   foo"

# 冗長: グループ不要
print(re.split(r'(?:\s)+', text))    # => ['hello', 'world', 'foo']
# 簡潔:
print(re.split(r'\s+', text))        # => ['hello', 'world', 'foo']
```

### 6.5 アンチパターン: `\b` の誤用

```python
import re

# NG: 数字のみの単語境界の想定が間違っている
text = "item123"
print(re.findall(r'\b\d+\b', text))
# => [] (空!) -- '123' の前の \b は m と 1 の間だが、
#    123 の前に非単語文字がない(\b は \w→\W の遷移点)

# 正確な理解:
# 'item123' では:
# - 'i' の前に \b (文字列先頭)
# - '3' の後に \b (文字列末尾)
# - 'm' と '1' の間は \w→\w なので \b ではない!
# したがって \b\d+\b は '123' を独立した単語として見つけない

# OK: 目的に応じた正確なパターン
# 1. 独立した数値を検索(前後が非数字)
print(re.findall(r'(?<!\w)\d+(?!\w)', text))  # => []
# 2. 文字列中の数字部分を抽出
print(re.findall(r'\d+', text))  # => ['123']
# 3. 独立した数値のみ(空白区切り)
text2 = "item123 456 foo"
print(re.findall(r'\b\d+\b', text2))  # => ['456']
```

---

## 7. 実践パターン集

### 7.1 ログ解析パターン

```python
import re

log_entries = """
2026-02-11T10:30:45.123Z INFO  [main] Application starting
2026-02-11T10:30:46.456Z DEBUG [db] Connection pool initialized (size=10)
2026-02-11T10:30:47.789Z ERROR [api] Request timeout after 30000ms
2026-02-11T10:30:48.012Z WARN  [mem] Memory usage: 85% (threshold: 80%)
2026-02-11T10:30:49.345Z ERROR [api] Internal server error: NullPointerException
2026-02-11T10:30:50.678Z INFO  [main] Shutdown initiated
"""

# 1. タイムスタンプ、レベル、コンポーネント、メッセージの分解
log_pattern = r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)\s+(INFO|DEBUG|WARN|ERROR)\s+\[(\w+)\]\s+(.+)$'

for match in re.finditer(log_pattern, log_entries, re.MULTILINE):
    ts, level, component, message = match.groups()
    print(f"[{level:5s}] {component:6s} | {message}")

# 2. ERROR レベルのみ抽出(行全体)
errors = re.findall(r'^.*ERROR.*$', log_entries, re.MULTILINE)
for err in errors:
    print(f"ERROR: {err.strip()}")

# 3. 数値の抽出(メトリクス解析)
metrics = re.findall(r'(\w+)[=:]\s*(\d+(?:\.\d+)?)', log_entries)
for key, value in metrics:
    print(f"  {key} = {value}")
# size = 10, threshold = 80, etc.

# 4. 時間範囲フィルタ
def filter_time_range(logs, start_time, end_time):
    """指定時間範囲のログをフィルタ"""
    pattern = rf'^({re.escape(start_time)}.*?{re.escape(end_time)}.*?)$'
    # より正確なアプローチ: 各行のタイムスタンプを比較
    result = []
    for line in logs.strip().split('\n'):
        m = re.match(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
        if m:
            ts = m.group(1)
            if start_time <= ts <= end_time:
                result.append(line)
    return result

filtered = filter_time_range(log_entries, "2026-02-11T10:30:47", "2026-02-11T10:30:49")
for line in filtered:
    print(line.strip())
```

### 7.2 データクレンジングパターン

```python
import re

# 1. 電話番号の正規化
def normalize_phone(phone):
    """各種形式の電話番号を統一フォーマットに変換"""
    # 数字以外を除去
    digits = re.sub(r'\D', '', phone)

    # 日本の電話番号パターン
    # 携帯: 090-XXXX-XXXX / 080-XXXX-XXXX / 070-XXXX-XXXX
    if re.match(r'^0[789]0\d{8}$', digits):
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    # 固定電話(東京): 03-XXXX-XXXX
    elif re.match(r'^0[3-9]\d{8}$', digits):
        return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}"
    # 固定電話(その他): 0XXX-XX-XXXX
    elif re.match(r'^0\d{9}$', digits):
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return phone  # 変換不能

tests = [
    "090-1234-5678", "09012345678", "090 1234 5678",
    "03-1234-5678", "0312345678", "(03) 1234-5678"
]
for t in tests:
    print(f"{t:20s} → {normalize_phone(t)}")

# 2. 金額の正規化
def normalize_currency(text):
    """金額表記を統一フォーマットに変換"""
    # カンマ区切り → 除去
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # 全角数字 → 半角
    zen_to_han = str.maketrans('０１２３４５６７８９', '0123456789')
    text = text.translate(zen_to_han)
    # 「万」「億」の処理
    text = re.sub(r'(\d+)万(\d*)', lambda m: str(int(m.group(1)) * 10000 + int(m.group(2) or 0)), text)
    text = re.sub(r'(\d+)億', lambda m: str(int(m.group(1)) * 100000000), text)
    return text

print(normalize_currency("1,234,567円"))  # => 1234567円
print(normalize_currency("１２３４円"))     # => 1234円
print(normalize_currency("5万3000"))       # => 53000

# 3. 住所の正規化
def normalize_address(addr):
    """日本語住所の表記揺れを統一"""
    # 全角数字 → 半角
    zen_to_han = str.maketrans('０１２３４５６７８９', '0123456789')
    addr = addr.translate(zen_to_han)
    # 「丁目」「番地」「号」の表記統一
    addr = re.sub(r'(\d+)丁目(\d+)番地?(\d+)号?', r'\1-\2-\3', addr)
    addr = re.sub(r'(\d+)丁目(\d+)番地?', r'\1-\2', addr)
    addr = re.sub(r'(\d+)丁目', r'\1', addr)
    return addr

print(normalize_address("東京都港区赤坂1丁目2番地3号"))
# => 東京都港区赤坂1-2-3
print(normalize_address("東京都港区赤坂１丁目２番地３号"))
# => 東京都港区赤坂1-2-3
```

### 7.3 プログラミング言語のトークン解析

```python
import re

def tokenize(source_code):
    """簡易トークナイザ: Python風の式を解析"""
    token_spec = [
        ('NUMBER',    r'\d+\.?\d*(?:[eE][+-]?\d+)?'),  # 整数・小数・指数
        ('STRING',    r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''),  # 文字列
        ('IDENT',     r'[a-zA-Z_]\w*'),                 # 識別子
        ('OP',        r'[+\-*/=<>!]=?|[(){}[\],;.]'),   # 演算子
        ('NEWLINE',   r'\n'),                            # 改行
        ('SKIP',      r'[ \t]+'),                        # 空白(スキップ)
        ('COMMENT',   r'#.*'),                           # コメント
        ('MISMATCH',  r'.'),                             # 不明な文字
    ]

    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)
    tokens = []

    for m in re.finditer(tok_regex, source_code):
        kind = m.lastgroup
        value = m.group()
        if kind == 'SKIP' or kind == 'COMMENT':
            continue
        elif kind == 'MISMATCH':
            raise SyntaxError(f"Unexpected character: {value!r}")
        tokens.append((kind, value))

    return tokens

# テスト
code = 'x = 3.14 + y * 2  # calculation'
for tok in tokenize(code):
    print(f"  {tok[0]:10s}: {tok[1]}")
# NUMBER    : 3.14
# OP        : +
# IDENT     : y
# OP        : *
# NUMBER    : 2
# etc.
```

### 7.4 URL / URI の解析

```python
import re

def parse_url(url):
    """RFC 3986 に基づく URI 解析"""
    pattern = r'''
        ^
        (?:(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*):)?  # スキーム
        (?://
            (?:(?P<userinfo>[^@]*)@)?               # ユーザー情報
            (?P<host>[^/:?#]*)                      # ホスト
            (?::(?P<port>\d+))?                     # ポート
        )?
        (?P<path>[^?#]*)                            # パス
        (?:\?(?P<query>[^#]*))?                     # クエリ
        (?:\#(?P<fragment>.*))?                      # フラグメント
        $
    '''

    m = re.match(pattern, url, re.VERBOSE)
    if not m:
        return None

    return {k: v for k, v in m.groupdict().items() if v is not None}

# テスト
urls = [
    "https://user:pass@example.com:8080/path/to/page?q=hello&lang=ja#section",
    "ftp://files.example.com/pub/docs/readme.txt",
    "/api/v2/users?page=1&limit=20",
    "mailto:user@example.com",
]

for url in urls:
    print(f"\nURL: {url}")
    parts = parse_url(url)
    for k, v in parts.items():
        print(f"  {k:12s}: {v}")
```

### 7.5 マークダウンのインライン要素解析

```python
import re

def parse_markdown_inline(text):
    """Markdown のインライン要素を解析"""
    patterns = [
        # 強調(**bold** / __bold__)
        (r'\*\*(.+?)\*\*|__(.+?)__', 'bold'),
        # 斜体(*italic* / _italic_)
        (r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', 'italic'),
        # インラインコード(`code`)
        (r'`([^`]+)`', 'code'),
        # リンク([text](url))
        (r'\[([^\]]+)\]\(([^)]+)\)', 'link'),
        # 画像(![alt](url))
        (r'!\[([^\]]*)\]\(([^)]+)\)', 'image'),
        # 取り消し線(~~text~~)
        (r'~~(.+?)~~', 'strikethrough'),
    ]

    elements = []
    for pattern, elem_type in patterns:
        for m in re.finditer(pattern, text):
            elements.append({
                'type': elem_type,
                'match': m.group(),
                'start': m.start(),
                'end': m.end(),
            })

    # 位置順にソート
    elements.sort(key=lambda x: x['start'])
    return elements

md_text = "This is **bold** and *italic* with `code` and [link](https://example.com)"
for elem in parse_markdown_inline(md_text):
    print(f"  {elem['type']:15s}: {elem['match']}")
```

---

## 8. FAQ

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

### Q4: `\b` は日本語テキストで機能するか？

**A**: 標準の `\b` は **ASCII 単語文字境界** であり、日本語テキストでは期待通りに動作しないことが多い:

```python
import re

text = "東京タワーは333mです"

# \b は ASCII の \w([a-zA-Z0-9_])に基づく
# 日本語文字は \W 扱い → 各文字の前後が全て境界
print(re.findall(r'\b.+?\b', text))
# => ['東', '京', 'タ', 'ワ', 'ー', 'は', '333', 'm', 'で', 'す']

# 日本語の「単語」を扱うには:
# 1. MeCab 等の形態素解析を使う(推奨)
# 2. Unicode 単語境界(ICU)を使う
# 3. regex モジュールの \b{w} を使う
import regex
print(regex.findall(r'\b{w}.+?\b{w}', text))
# Unicode 単語境界による分割
```

### Q5: `re.fullmatch` と `^...$` の違いは？

**A**: 多くの場合同等だが、MULTILINE フラグとの相互作用が異なる:

```python
import re

text = "hello\nworld"

# re.fullmatch は常に文字列全体をチェック
print(re.fullmatch(r'\w+', text))  # => None (改行があるため)

# ^...$ は MULTILINE で挙動が変わる
print(re.match(r'^\w+$', text, re.M))  # => <Match: 'hello'> (1行目にマッチ)
print(re.fullmatch(r'\w+', "hello"))    # => <Match: 'hello'>

# fullmatch は MULTILINE の影響を受けない(常に文字列全体)
print(re.fullmatch(r'\w+', text, re.M))  # => None
```

### Q6: 独占的量指定子が使えない言語での代替手段は？

**A**: アトミックグループまたはコード内での対策を使う:

```python
import re

# Python の re モジュールでは独占的量指定子が使えない
# 代替手段:

# 1. regex モジュールを使う(推奨)
# import regex
# regex.search(r'a++b', text)

# 2. 否定文字クラスでバックトラックを回避
# NG: .*  (バックトラック大量)
# OK: [^<]*  (停止文字を明示)

# 3. タイムアウトを設定
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Regex timeout")

def safe_search(pattern, text, timeout_sec=1):
    """タイムアウト付きの正規表現検索"""
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        return re.search(pattern, text)
    except TimeoutError:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# 4. 入力長を制限する
def safe_match(pattern, text, max_len=10000):
    """入力長制限付きの正規表現マッチ"""
    if len(text) > max_len:
        raise ValueError(f"Input too long: {len(text)} > {max_len}")
    return re.search(pattern, text)
```

### Q7: 量指定子の上限値にハードリミットはあるか？

**A**: エンジンによって異なる:

```
┌──────────────┬──────────────────────────────┐
│ エンジン      │ {n,m} の上限                  │
├──────────────┼──────────────────────────────┤
│ Python (re)  │ 2^31 - 1 (2147483647)        │
│ JavaScript   │ 2^32 - 1 (ブラウザ依存)       │
│ Java         │ 2^31 - 1                     │
│ PCRE         │ 65535                         │
│ Ruby         │ 制限なし(メモリ依存)          │
│ Go (RE2)     │ 1000 (デフォルト)             │
│ Rust (regex) │ 制限なし(メモリ依存)          │
└──────────────┴──────────────────────────────┘

# 実用上は {n,m} の値が大きすぎると
# コンパイル時間やメモリ使用量が増大するため
# 100程度を上限の目安とする
```

### Q8: `\G` アンカーを Python で使うには？

**A**: Python の `re` モジュールには `\G` がないが、`regex` モジュールまたは `re.scanner` で代替できる:

```python
# regex モジュール
import regex

text = "abc123def456"
tokens = []
pos = 0

# \G を使った連続マッチ
for m in regex.finditer(r'\G(?:(\w+)|(\d+))', text):
    tokens.append(m.group())

# re モジュールでの代替: Scanner
import re
scanner = re.Scanner([
    (r'[a-z]+', lambda s, t: ('WORD', t)),
    (r'\d+',    lambda s, t: ('NUM', t)),
    (r'\s+',    None),  # スキップ
])

tokens, remainder = scanner.scan("abc 123 def 456")
for tok in tokens:
    print(tok)
# ('WORD', 'abc')
# ('NUM', '123')
# ('WORD', 'def')
# ('NUM', '456')
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
| `(?>...)` | アトミックグループ |
| `^` | 行頭 / 文字列先頭 |
| `$` | 行末 / 文字列末尾 |
| `\b` | 単語境界 |
| `\B` | 非単語境界 |
| `\A` / `\Z` | 文字列の絶対先頭/末尾 |
| `\z` | 文字列の絶対末尾(一部言語) |
| `\G` | 前回マッチ終了位置 |
| 貪欲 vs 怠惰 | 貪欲が最長、怠惰が最短 |
| パフォーマンス | `[^X]*` > `.*?` > `.*`(一般的に) |
| ReDoS対策 | ネスト量指定子を避け、否定クラスか独占的量指定子を使用 |

## 次に読むべきガイド

- [../01-advanced/00-groups-backreferences.md](../01-advanced/00-groups-backreferences.md) -- グループと後方参照
- [../01-advanced/01-lookaround.md](../01-advanced/01-lookaround.md) -- 先読み・後読み
- [../01-advanced/03-performance.md](../01-advanced/03-performance.md) -- パフォーマンス最適化とReDoS対策

## 参考文献

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions" O'Reilly, 2006 -- 第4章「量指定子の力学」と第6章「バックトラック」が必読
2. **Russ Cox** "Regular Expression Matching: the Virtual Machine Approach" https://swtch.com/~rsc/regexp/regexp2.html, 2009 -- バックトラックの理論的分析
3. **Jan Goyvaerts** "Regular-Expressions.info" https://www.regular-expressions.info/repeat.html -- 量指定子の実用的な解説
4. **OWASP** "Regular expression Denial of Service - ReDoS" https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS -- ReDoS攻撃とその防御
5. **Python Documentation** "re --- Regular expression operations" https://docs.python.org/3/library/re.html -- Python公式リファレンス
