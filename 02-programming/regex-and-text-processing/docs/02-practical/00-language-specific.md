# 言語別正規表現 -- JS/Python/Go/Rust/Java の違い

> 同じ正規表現でも言語・エンジンによって構文、フラグ、Unicode対応、パフォーマンス特性が異なる。各言語の正規表現APIの設計思想と実用上の差異を体系的に比較する。

## この章で学ぶこと

1. **5言語の正規表現APIの違い** -- 構文、フラグ、返り値、マッチモデルの差異
2. **エンジン特性と制約の把握** -- NFA/DFA、後方参照サポート、Unicode対応の違い
3. **言語間の移植時に注意すべきポイント** -- 同じパターンでも結果が異なるケース

---

## 1. 言語別概要

### 1.1 エンジンマッピング

```
┌──────────┬──────────────┬──────────────────────────┐
│ 言語      │ エンジン      │ 特徴                      │
├──────────┼──────────────┼──────────────────────────┤
│ Python   │ re (NFA)     │ PCRE風、Unicode デフォルト │
│ JavaScript│ V8 (NFA)    │ ECMA-262 準拠、ES2018 拡張│
│ Java     │ Pattern (NFA)│ PCRE風、独占的量指定子あり  │
│ Go       │ RE2 (DFA)    │ 線形時間保証、機能制限     │
│ Rust     │ regex (DFA)  │ 線形時間保証、機能制限     │
└──────────┴──────────────┴──────────────────────────┘
```

### 1.2 エンジン設計思想の詳細

各言語が採用するエンジンの設計思想は、API 設計から利用できる機能まで広く影響する。

**NFA (Non-deterministic Finite Automaton) エンジン**

NFA エンジンはバックトラッキングベースで動作する。パターンの各分岐を順に試行し、マッチしなければ前の選択点に戻って別の分岐を試す。この仕組みにより後方参照やルックアラウンドなど高度な機能を実現できるが、最悪ケースで指数関数的な計算量になるリスクがある。Python、JavaScript、Java が採用している。

```
NFA のバックトラッキング動作:

入力: "aaaaab"
パターン: a*ab

ステップ 1: a* が "aaaaa" を貪欲にマッチ → "b" と "ab" が不一致
ステップ 2: バックトラック、a* が "aaaa" をマッチ → "ab" が一致!
→ マッチ成功

パターンによっては:
入力: "aaaaaaaaaaaaaaaaac"
パターン: (a+)+b
→ 指数的なバックトラッキングが発生 (ReDoS)
```

**DFA (Deterministic Finite Automaton) エンジン**

DFA エンジンは入力文字列を一度だけスキャンし、各文字に対して決定的に次の状態を選択する。バックトラッキングが発生しないため、常に O(n) の線形時間でマッチが完了する。ただし、後方参照やルックアラウンドなどバックトラッキングが必要な機能は実現できない。Go と Rust が採用している。

```
DFA の動作:

入力: "aaaaab"
パターン: a*ab

全状態を同時にシミュレート:
位置 0 'a': 状態{S0, S1}
位置 1 'a': 状態{S0, S1}
位置 2 'a': 状態{S0, S1}
位置 3 'a': 状態{S0, S1}
位置 4 'a': 状態{S0, S1, S2}
位置 5 'b': 状態{S3(受理)}
→ マッチ成功 (常に O(n))
```

---

## 2. Python

### 2.1 基本API

```python
import re

text = "2026-02-11 Error: Connection failed at 10:30:45"

# search: 最初のマッチを返す
m = re.search(r'\d{4}-\d{2}-\d{2}', text)
print(m.group())  # => '2026-02-11'

# findall: 全マッチをリストで返す
print(re.findall(r'\d+', text))
# => ['2026', '02', '11', '10', '30', '45']

# finditer: イテレータで返す(メモリ効率的)
for m in re.finditer(r'\d+', text):
    print(f"  {m.group()} at {m.span()}")

# sub: 置換
result = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3/\2/\1', text)
print(result)  # => '11/02/2026 Error: ...'

# split: 分割
print(re.split(r'\s+', "hello  world\tfoo"))
# => ['hello', 'world', 'foo']

# compile: プリコンパイル
pattern = re.compile(r'\d+', re.ASCII)
print(pattern.findall(text))
```

### 2.2 Python 固有の機能

```python
import re

# fullmatch: 文字列全体がパターンに一致するか
print(re.fullmatch(r'\d{4}', '2026'))   # => Match
print(re.fullmatch(r'\d{4}', '2026a'))  # => None

# (?P<name>...) 名前付きグループ (Python 独自構文)
m = re.search(r'(?P<year>\d{4})-(?P<month>\d{2})', "2026-02-11")
print(m.groupdict())  # => {'year': '2026', 'month': '02'}

# 条件付きパターン (?(id)yes|no)
pattern = r'(\()?\d+(?(1)\))'
print(re.search(pattern, "(42)").group())  # => '(42)'
print(re.search(pattern, "42").group())    # => '42'

# re.VERBOSE でコメント付きパターン
pattern = re.compile(r'''
    (?P<year>\d{4})   # 年
    -(?P<month>\d{2}) # 月
    -(?P<day>\d{2})   # 日
''', re.VERBOSE)
```

### 2.3 Python の高度なパターンと実践テクニック

```python
import re

# --- sub での関数置換 ---
# 第2引数に関数を渡すと、マッチオブジェクトを引数に呼ばれる
def celsius_to_fahrenheit(match):
    celsius = float(match.group(1))
    fahrenheit = celsius * 9 / 5 + 32
    return f"{fahrenheit:.1f}F"

text = "気温は 20C から 35C の範囲です"
result = re.sub(r'(\d+(?:\.\d+)?)C', celsius_to_fahrenheit, text)
print(result)  # => '気温は 68.0F から 95.0F の範囲です'

# --- subn: 置換回数も返す ---
result, count = re.subn(r'\d+', 'X', "abc 123 def 456")
print(f"結果: {result}, 置換回数: {count}")
# => '結果: abc X def X, 置換回数: 2'

# --- split の maxsplit ---
print(re.split(r'[,;]', "a,b;c,d", maxsplit=2))
# => ['a', 'b', 'c,d']

# --- split でグループを含むとセパレータも結果に含まれる ---
print(re.split(r'([,;])', "a,b;c"))
# => ['a', ',', 'b', ';', 'c']

# --- re.escape: メタ文字をエスケープ ---
user_input = "price is $100 (USD)"
safe_pattern = re.escape(user_input)
print(safe_pattern)
# => 'price\\ is\\ \\$100\\ \\(USD\\)'
# ユーザー入力をパターンに組み込むとき必須

# --- 複数フラグの組み合わせ ---
pattern = re.compile(
    r'''
    (?P<protocol>https?)    # プロトコル
    ://
    (?P<host>[^/\s]+)       # ホスト名
    (?P<path>/[^\s]*)?      # パス (オプション)
    ''',
    re.VERBOSE | re.IGNORECASE
)
m = pattern.search("Visit HTTP://Example.COM/path?q=1 now")
if m:
    print(m.groupdict())
    # => {'protocol': 'HTTP', 'host': 'Example.COM', 'path': '/path?q=1'}
```

### 2.4 Python regex モジュール (サードパーティ)

```python
# pip install regex
# 標準 re の上位互換、追加機能が多い
import regex

# 可変長の後読み
m = regex.search(r'(?<=ab+)', "abbb_test")
print(m.start())  # 標準 re では不可能

# Unicode カテゴリプロパティ
print(regex.findall(r'\p{Han}+', "hello 世界 test"))
# => ['世界']

# あいまい一致 (fuzzy matching)
# {e<=1} は編集距離1以内を許容
m = regex.search(r'(?:hello){e<=1}', "helo world")
print(m.group())  # => 'helo'

# アトミックグループ
m = regex.search(r'(?>a+)b', "aab")
print(m.group())  # => 'aab'

# 独占的量指定子
m = regex.search(r'a++b', "aab")
print(m.group())  # => 'aab'

# POSIX 文字クラス
print(regex.findall(r'[[:alpha:]]+', "hello 123 world"))
# => ['hello', 'world']

# 再帰パターン (括弧のネスト対応)
pattern = regex.compile(r'\((?:[^()]*|(?R))*\)')
text = "outer (inner (deep) end) rest"
print(pattern.findall(text))
# => ['(inner (deep) end)']
```

### 2.5 Python のパフォーマンス最適化

```python
import re
import timeit

# --- プリコンパイルの効果 ---
# ループ内で同じパターンを繰り返す場合、compile が有効
# ただし Python 内部でもキャッシュ(最大512パターン)があるため
# 単純なケースでは差は小さい

compiled = re.compile(r'\d{4}-\d{2}-\d{2}')
text = "date: 2026-02-11"

# compile あり: compiled.search(text)
# compile なし: re.search(r'\d{4}-\d{2}-\d{2}', text)
# → 大量パターン(512超)でキャッシュ溢れがある場合に compile が有効

# --- findall vs finditer ---
# 大量のマッチがある場合、finditer がメモリ効率的
large_text = "num " * 100000
# findall: 全結果をリストに格納 (メモリ大)
# finditer: イテレータで1つずつ返す (メモリ小)

# --- 非キャプチャグループの活用 ---
# キャプチャ不要なグループは (?:...) にする
# findall はグループがあるとグループの内容を返す
text = "2026-02-11 and 2025-12-25"
print(re.findall(r'(\d{4})-(\d{2})-(\d{2})', text))
# => [('2026', '02', '11'), ('2025', '12', '25')]  ← タプルのリスト
print(re.findall(r'\d{4}-\d{2}-\d{2}', text))
# => ['2026-02-11', '2025-12-25']  ← 文字列のリスト
print(re.findall(r'(?:\d{4})-(?:\d{2})-(?:\d{2})', text))
# => ['2026-02-11', '2025-12-25']  ← 文字列のリスト (同じ)

# --- 貪欲 vs 怠惰の選択 ---
html = "<b>bold</b> and <i>italic</i>"
print(re.findall(r'<.+>', html))    # => ['<b>bold</b> and <i>italic</i>'] 貪欲
print(re.findall(r'<.+?>', html))   # => ['<b>', '</b>', '<i>', '</i>'] 怠惰
print(re.findall(r'<[^>]+>', html)) # => ['<b>', '</b>', '<i>', '</i>'] 否定文字クラス(最速)
```

---

## 3. JavaScript

### 3.1 基本API

```javascript
const text = "2026-02-11 Error: Connection failed at 10:30:45";

// リテラル構文
const pattern = /\d{4}-\d{2}-\d{2}/;
const match = text.match(pattern);
console.log(match[0]);  // => '2026-02-11'

// g フラグ: 全マッチ
console.log(text.match(/\d+/g));
// => ['2026', '02', '11', '10', '30', '45']

// matchAll (ES2020): イテレータで返す
for (const m of text.matchAll(/\d+/g)) {
    console.log(`  ${m[0]} at index ${m.index}`);
}

// replace: 置換
const result = text.replace(
    /(\d{4})-(\d{2})-(\d{2})/,
    '$3/$2/$1'
);
console.log(result);  // => '11/02/2026 Error: ...'

// replaceAll (ES2021): 全置換
console.log("aaa".replaceAll(/a/g, "b"));  // => 'bbb'

// コンストラクタ構文(動的パターン)
const dynamic = new RegExp("\\d{4}", "g");
console.log(text.match(dynamic));
```

### 3.2 JavaScript 固有の機能 (ES2018+)

```javascript
// 名前付きグループ (ES2018)
const m = "2026-02-11".match(
    /(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})/
);
console.log(m.groups);
// => { year: '2026', month: '02', day: '11' }

// 後読み (ES2018)
console.log("$100 €200".match(/(?<=\$)\d+/g));
// => ['100']

// s フラグ: dotAll (ES2018)
console.log("a\nb".match(/a.b/s));
// => ['a\nb']

// d フラグ: インデックス情報 (ES2022)
const result2 = /(?<name>\w+)/.exec("hello");
// result.indices[0] => [0, 5]
// result.indices.groups.name => [0, 5]

// v フラグ: Unicode集合演算 (ES2024)
// /[\p{L}&&\p{ASCII}]/v  -- ASCII かつ文字
```

### 3.3 JavaScript の高度なパターンと実践テクニック

```javascript
// --- replace での関数置換 ---
const text2 = "Price: $100, Tax: $15, Total: $115";
const formatted = text2.replace(/\$(\d+)/g, (match, amount) => {
    return `$${Number(amount).toLocaleString()}`;
});
console.log(formatted);
// => 'Price: $100, Tax: $15, Total: $115'

// 名前付きグループを使った関数置換
const dates = "Start: 2026-02-11, End: 2026-03-15";
const converted = dates.replace(
    /(?<y>\d{4})-(?<m>\d{2})-(?<d>\d{2})/g,
    (match, y, m, d, offset, string, groups) => {
        return `${groups.d}/${groups.m}/${groups.y}`;
    }
);
console.log(converted);
// => 'Start: 11/02/2026, End: 15/03/2026'

// --- exec() の逐次マッチ ---
const re = /(\w+)=(\w+)/g;
const params = "name=Alice&age=30&city=Tokyo";
let execMatch;
while ((execMatch = re.exec(params)) !== null) {
    console.log(`${execMatch[1]}: ${execMatch[2]}`);
}
// name: Alice
// age: 30
// city: Tokyo

// --- g フラグと lastIndex の罠 ---
const reG = /abc/g;
console.log(reG.test("abc def"));  // true
console.log(reG.lastIndex);        // 3
console.log(reG.test("abc def"));  // false! (lastIndex=3 から検索)
reG.lastIndex = 0;                 // リセットが必要
console.log(reG.test("abc def"));  // true

// --- String.prototype.search() ---
// マッチ位置のインデックスを返す (g フラグは無視)
console.log("hello world".search(/world/));  // => 6
console.log("hello world".search(/xyz/));    // => -1

// --- split での制限と注意点 ---
console.log("a1b2c3d".split(/(\d)/));
// => ['a', '1', 'b', '2', 'c', '3', 'd']
// キャプチャグループ付きだとセパレータも結果に含まれる

console.log("a,,b,,c".split(/,+/));
// => ['a', 'b', 'c']
```

### 3.4 JavaScript の v フラグ (ES2024) 詳細

```javascript
// v フラグは u フラグの拡張版。文字クラスの集合演算を可能にする

// 差集合: \p{L} から ASCII を除外 → 非ASCII文字のみ
// /[\p{L}--\p{ASCII}]/v
const nonAsciiLetters = "hello 世界 café".match(/[\p{L}--\p{ASCII}]/gv);
console.log(nonAsciiLetters);
// => ['世', '界', 'é']

// 積集合: \p{L} と \p{ASCII} の両方 → ASCII文字のみ
// /[\p{L}&&\p{ASCII}]/v
const asciiLetters = "hello 世界 café".match(/[\p{L}&&\p{ASCII}]/gv);
console.log(asciiLetters);
// => ['h', 'e', 'l', 'l', 'o', 'c', 'a', 'f']

// 和集合(ネストされた文字クラス)
// /[[\p{Decimal_Number}][\p{L}]]/v
// → 数字または文字にマッチ

// v フラグ使用時の注意:
// - u フラグと v フラグは同時に使用不可
// - v フラグは u フラグの全機能を含む
// - 文字クラス内での特殊文字の扱いが厳格になる
```

### 3.5 JavaScript の正規表現パフォーマンス

```javascript
// --- RegExp の内部キャッシュ ---
// V8 エンジンはリテラル構文の正規表現をキャッシュする
// new RegExp() は毎回新しいオブジェクトを生成する

// 高速: リテラル (同一パターンは内部キャッシュされる)
function matchLiteral(text) {
    return /\d+/g.test(text);
}

// 注意: 動的パターンの場合はキャッシュ不可
function matchDynamic(text, pattern) {
    return new RegExp(pattern, 'g').test(text);
}

// --- ReDoS 対策 ---
// V8 は RegExp のタイムアウト機構を持つ(Node.js では --regex-timeout)
// しかし根本的にはパターン設計で防ぐべき

// 危険なパターン:
// /(a+)+b/          → 指数的バックトラッキング
// /([a-zA-Z]+)*$/   → 指数的バックトラッキング
// /(\w+\s?)+$/      → 指数的バックトラッキング

// 安全なパターンへの書き換え:
// /(a+)+b/  →  /a+b/
// /([a-zA-Z]+)*$/  →  /[a-zA-Z]*$/
// /(\w+\s?)+$/  →  /[\w\s]*$/

// --- matchAll の利点 ---
// match(g) はキャプチャグループ情報を失う
const text3 = "2026-02-11 2025-12-25";
console.log(text3.match(/(\d{4})-(\d{2})-(\d{2})/g));
// => ['2026-02-11', '2025-12-25']  ← グループ情報なし!

// matchAll はグループ情報を保持
for (const m of text3.matchAll(/(\d{4})-(\d{2})-(\d{2})/g)) {
    console.log(`${m[0]} → 年: ${m[1]}, 月: ${m[2]}, 日: ${m[3]}`);
}
// '2026-02-11 → 年: 2026, 月: 02, 日: 11'
// '2025-12-25 → 年: 2025, 月: 12, 日: 25'
```

---

## 4. Go

### 4.1 基本API

```go
package main

import (
    "fmt"
    "regexp"
)

func main() {
    text := "2026-02-11 Error: Connection failed"

    // Compile: パターンをコンパイル(エラーを返す)
    re, err := regexp.Compile(`\d{4}-\d{2}-\d{2}`)
    if err != nil {
        panic(err)
    }

    // MustCompile: パニックする版(定数パターン用)
    re = regexp.MustCompile(`\d{4}-\d{2}-\d{2}`)

    // FindString: 最初のマッチ
    fmt.Println(re.FindString(text))
    // => "2026-02-11"

    // FindAllString: 全マッチ
    reDigit := regexp.MustCompile(`\d+`)
    fmt.Println(reDigit.FindAllString(text, -1))
    // => [2026 02 11]

    // FindStringSubmatch: サブマッチ付き
    re2 := regexp.MustCompile(`(\d{4})-(\d{2})-(\d{2})`)
    matches := re2.FindStringSubmatch(text)
    fmt.Printf("全体: %s, 年: %s, 月: %s, 日: %s\n",
        matches[0], matches[1], matches[2], matches[3])

    // ReplaceAllString: 置換
    result := re2.ReplaceAllString(text, "${3}/${2}/${1}")
    fmt.Println(result)

    // 名前付きグループ
    re3 := regexp.MustCompile(`(?P<year>\d{4})-(?P<month>\d{2})`)
    match := re3.FindStringSubmatch(text)
    for i, name := range re3.SubexpNames() {
        if name != "" {
            fmt.Printf("  %s: %s\n", name, match[i])
        }
    }
}
```

### 4.2 Go の制約

```go
// Go (RE2 エンジン) でサポートされない機能:
// ✗ 後方参照 (\1, \k<name>)
// ✗ 先読み (?=...), (?!...)
// ✗ 後読み (?<=...), (?<!...)
// ✗ 条件付きパターン (?(id)yes|no)
// ✗ 独占的量指定子 (*+, ++, ?+)
// ✗ アトミックグループ (?>...)
// ✗ Unicode プロパティの一部

// 代わりに保証されること:
// ✓ 常に O(n) の線形時間
// ✓ ReDoS が原理的に不可能
// ✓ メモリ使用量が予測可能
```

### 4.3 Go の高度なAPIと実践テクニック

```go
package main

import (
    "fmt"
    "regexp"
    "strings"
)

func main() {
    // --- ReplaceAllStringFunc: 関数による置換 ---
    re := regexp.MustCompile(`[a-z]+`)
    result := re.ReplaceAllStringFunc("hello WORLD foo BAR", strings.ToUpper)
    fmt.Println(result)
    // => "HELLO WORLD FOO BAR"

    // --- ReplaceAllLiteralString: リテラル置換 ---
    // $1 などの展開を行わない
    re2 := regexp.MustCompile(`\d+`)
    result2 := re2.ReplaceAllLiteralString("price: 100", "$1")
    fmt.Println(result2)
    // => "price: $1" ($1 がそのまま文字列として挿入される)

    // --- Split: 分割 ---
    re3 := regexp.MustCompile(`\s*[,;]\s*`)
    parts := re3.Split("a, b; c , d", -1)
    fmt.Println(parts)
    // => [a b c d]

    // n 引数で分割数を制限
    parts2 := re3.Split("a, b; c , d", 2)
    fmt.Println(parts2)
    // => [a b; c , d]

    // --- FindAllStringSubmatchIndex: 位置情報付き全マッチ ---
    re4 := regexp.MustCompile(`(\w+)=(\w+)`)
    text := "name=Alice age=30"
    indices := re4.FindAllStringSubmatchIndex(text, -1)
    for _, idx := range indices {
        // idx[0:2] = 全体マッチの開始/終了
        // idx[2:4] = グループ1の開始/終了
        // idx[4:6] = グループ2の開始/終了
        key := text[idx[2]:idx[3]]
        val := text[idx[4]:idx[5]]
        fmt.Printf("  %s = %s\n", key, val)
    }

    // --- MatchString: マッチの有無だけを確認 ---
    // Find系より高速(マッチ位置の計算が不要)
    re5 := regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
    fmt.Println(re5.MatchString("2026-02-11"))  // true
    fmt.Println(re5.MatchString("not a date"))  // false

    // --- []byte 版 API ---
    // 文字列変換を避けてバイト列を直接処理
    reB := regexp.MustCompile(`\d+`)
    data := []byte("hello 123 world 456")
    allBytes := reB.FindAll(data, -1)
    for _, b := range allBytes {
        fmt.Printf("  %s\n", b)
    }

    // --- Expand: テンプレート展開 ---
    re6 := regexp.MustCompile(`(?P<first>\w+)\s+(?P<last>\w+)`)
    template := []byte("$last, $first")
    src := []byte("John Smith")
    match := re6.FindSubmatchIndex(src)
    var dst []byte
    dst = re6.Expand(dst, template, src, match)
    fmt.Printf("%s\n", dst)
    // => "Smith, John"
}
```

### 4.4 Go での先読み/後読みの代替手法

```go
package main

import (
    "fmt"
    "regexp"
    "strings"
)

func main() {
    // Go では先読み/後読みが使えないため、代替手法が必要

    // --- 代替手法 1: キャプチャグループで必要部分を取得 ---
    // Python: (?<=\$)\d+  ($ の後ろの数字)
    // Go: \$(\d+) としてグループ1を使う
    re := regexp.MustCompile(`\$(\d+)`)
    text := "$100 and $200"
    matches := re.FindAllStringSubmatch(text, -1)
    for _, m := range matches {
        fmt.Println(m[1])  // "100", "200"
    }

    // --- 代替手法 2: 複数ステップで処理 ---
    // Python: (?<=<tag>).*?(?=</tag>)
    // Go: <tag>(.*?)</tag> としてグループ1を使う
    re2 := regexp.MustCompile(`<title>(.*?)</title>`)
    html := "<title>My Page</title>"
    if m := re2.FindStringSubmatch(html); m != nil {
        fmt.Println(m[1])  // "My Page"
    }

    // --- 代替手法 3: 否定先読みの代替 ---
    // Python: \b\w+(?!ing)\b  (ing で終わらない単語)
    // Go: マッチ後にフィルタリング
    re3 := regexp.MustCompile(`\b\w+\b`)
    text2 := "running jumping hello world coding"
    words := re3.FindAllString(text2, -1)
    for _, w := range words {
        if !strings.HasSuffix(w, "ing") {
            fmt.Printf("  %s\n", w)
        }
    }
    // => hello, world

    // --- 代替手法 4: 後方参照の代替 ---
    // Python: <(\w+)>.*?</\1>  (同じタグの開閉)
    // Go: 2パスで処理
    re4 := regexp.MustCompile(`<(\w+)>[^<]*</(\w+)>`)
    html2 := "<div>content</div><span>text</span><div>bad</span>"
    allMatches := re4.FindAllStringSubmatch(html2, -1)
    for _, m := range allMatches {
        if m[1] == m[2] {  // 開始タグと終了タグが一致
            fmt.Printf("  有効: %s\n", m[0])
        } else {
            fmt.Printf("  無効: %s\n", m[0])
        }
    }
}
```

### 4.5 Go のパフォーマンス最適化

```go
package main

import (
    "regexp"
    "sync"
)

// --- パッケージレベルでコンパイル ---
// MustCompile はプログラム起動時に一度だけ実行される
var (
    datePattern  = regexp.MustCompile(`\d{4}-\d{2}-\d{2}`)
    emailPattern = regexp.MustCompile(`[\w.+-]+@[\w-]+\.[\w.]+`)
    urlPattern   = regexp.MustCompile(`https?://[^\s]+`)
)

// --- sync.Pool で Regexp オブジェクトを再利用 ---
// (通常は不要。動的パターンが大量にある場合のみ)
var regexpPool = sync.Pool{
    New: func() interface{} {
        return regexp.MustCompile(`\d+`)
    },
}

func processWithPool(text string) []string {
    re := regexpPool.Get().(*regexp.Regexp)
    defer regexpPool.Put(re)
    return re.FindAllString(text, -1)
}

// --- Regexp は goroutine 安全 ---
// 同じ Regexp オブジェクトを複数の goroutine で安全に使用可能
// (内部でロックを使用しているため)
func processParallel(texts []string) {
    var wg sync.WaitGroup
    for _, t := range texts {
        wg.Add(1)
        go func(text string) {
            defer wg.Done()
            datePattern.FindString(text)  // 安全
        }(t)
    }
    wg.Wait()
}

// --- Copy() でロック競合を回避 ---
// 高負荷な並列処理では Copy() でコピーを作ると速くなる場合がある
func processHighConcurrency(texts []string) {
    var wg sync.WaitGroup
    for _, t := range texts {
        wg.Add(1)
        go func(text string) {
            defer wg.Done()
            re := datePattern.Copy()  // コピーで競合回避
            re.FindString(text)
        }(t)
    }
    wg.Wait()
}
```

---

## 5. Rust

### 5.1 基本API

```rust
use regex::Regex;

fn main() {
    let text = "2026-02-11 Error: Connection failed";

    // コンパイル
    let re = Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap();

    // find: 最初のマッチ
    if let Some(m) = re.find(text) {
        println!("マッチ: {} (位置: {}-{})", m.as_str(), m.start(), m.end());
    }

    // find_iter: 全マッチのイテレータ
    let re_digit = Regex::new(r"\d+").unwrap();
    let numbers: Vec<&str> = re_digit.find_iter(text)
        .map(|m| m.as_str())
        .collect();
    println!("{:?}", numbers);
    // => ["2026", "02", "11"]

    // captures: キャプチャグループ
    let re2 = Regex::new(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})").unwrap();
    if let Some(caps) = re2.captures(text) {
        println!("年: {}, 月: {}, 日: {}",
            &caps["year"], &caps["month"], &caps["day"]);
    }

    // replace: 置換
    let result = re2.replace(text, "$day/$month/$year");
    println!("{}", result);
}
```

### 5.2 Rust の特徴

```rust
// Rust regex の特徴:
// ✓ DFA ベース: O(n) 保証
// ✓ ReDoS 不可能
// ✓ ゼロコストの遅延コンパイル (lazy_static!, once_cell)
// ✓ Unicode デフォルト対応
// ✗ 後方参照なし
// ✗ ルックアラウンドなし

// ルックアラウンドが必要な場合: fancy-regex
// use fancy_regex::Regex;
// let re = Regex::new(r"(?<=\$)\d+").unwrap();
// → NFA にフォールバック (O(n) 保証なし)

// パフォーマンス最適化: RegexSet (複数パターンの同時マッチ)
use regex::RegexSet;

let set = RegexSet::new(&[
    r"ERROR",
    r"WARN",
    r"INFO",
]).unwrap();

let text = "2026-02-11 ERROR: Something failed";
let matches: Vec<_> = set.matches(text).into_iter().collect();
println!("{:?}", matches);  // => [0] (ERROR にマッチ)
```

### 5.3 Rust の高度なAPIと実践テクニック

```rust
use regex::Regex;
use std::borrow::Cow;

fn main() {
    // --- replace_all: 全置換 ---
    let re = Regex::new(r"\d+").unwrap();
    let result = re.replace_all("abc 123 def 456", "NUM");
    println!("{}", result);
    // => "abc NUM def NUM"

    // replace は Cow<str> を返す
    // マッチがなければ元の文字列への参照(ゼロコピー)
    let no_match: Cow<str> = re.replace("no numbers here", "NUM");
    match no_match {
        Cow::Borrowed(_) => println!("コピーなし"),    // こちら
        Cow::Owned(_) => println!("新しい文字列を生成"),
    }

    // --- replace_all with closure ---
    let re2 = Regex::new(r"(?P<word>[a-z]+)").unwrap();
    let result2 = re2.replace_all("hello world", |caps: &regex::Captures| {
        caps["word"].to_uppercase()
    });
    println!("{}", result2);
    // => "HELLO WORLD"

    // --- captures_iter: 全キャプチャのイテレータ ---
    let re3 = Regex::new(r"(?P<key>\w+)=(?P<val>\w+)").unwrap();
    let text = "name=Alice age=30 city=Tokyo";
    for caps in re3.captures_iter(text) {
        println!("  {} = {}", &caps["key"], &caps["val"]);
    }

    // --- split: 分割 ---
    let re4 = Regex::new(r"[,;\s]+").unwrap();
    let parts: Vec<&str> = re4.split("a, b; c d").collect();
    println!("{:?}", parts);
    // => ["a", "b", "c", "d"]

    // --- splitn: 分割数の制限 ---
    let parts2: Vec<&str> = re4.splitn("a, b; c d", 2).collect();
    println!("{:?}", parts2);
    // => ["a", "b; c d"]

    // --- shortest_match: 最短マッチの終了位置のみ ---
    // find より高速(開始位置の計算が不要)
    let re5 = Regex::new(r"\d+").unwrap();
    if let Some(end) = re5.shortest_match("abc 123") {
        println!("最短マッチ終了位置: {}", end);
    }

    // --- is_match: マッチの有無のみ ---
    // find より高速(位置情報の計算が不要)
    println!("{}", re5.is_match("abc 123"));  // true
    println!("{}", re5.is_match("no nums"));  // false
}
```

### 5.4 Rust の遅延コンパイルとパフォーマンス

```rust
// --- lazy_static! / once_cell / std::sync::LazyLock ---
// 正規表現のコンパイルは高コスト。グローバルに一度だけ行うべき

// 方法1: once_cell (推奨、std に含まれる見込み)
use once_cell::sync::Lazy;
use regex::Regex;

static DATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap()
});

// 方法2: std::sync::LazyLock (Rust 1.80+)
// use std::sync::LazyLock;
// static DATE_RE: LazyLock<Regex> = LazyLock::new(|| {
//     Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap()
// });

// 方法3: lazy_static! マクロ
// use lazy_static::lazy_static;
// lazy_static! {
//     static ref DATE_RE: Regex = Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap();
// }

fn process(text: &str) -> Option<&str> {
    DATE_RE.find(text).map(|m| m.as_str())
}

// --- RegexSet の実践例: ログレベル分類 ---
use regex::RegexSet;

fn classify_log_lines(lines: &[&str]) {
    let set = RegexSet::new(&[
        r"(?i)\berror\b",
        r"(?i)\bwarn(ing)?\b",
        r"(?i)\binfo\b",
        r"(?i)\bdebug\b",
    ]).unwrap();

    let labels = ["ERROR", "WARN", "INFO", "DEBUG"];

    for line in lines {
        let matches: Vec<_> = set.matches(line).into_iter().collect();
        if matches.is_empty() {
            println!("  UNKNOWN: {}", line);
        } else {
            for idx in matches {
                println!("  {}: {}", labels[idx], line);
            }
        }
    }
}

// --- bytes::Regex: バイト列の直接処理 ---
// 非UTF-8 データ(バイナリログなど)を処理する場合
use regex::bytes::Regex as BytesRegex;

fn search_binary_log(data: &[u8]) {
    let re = BytesRegex::new(r"ERROR: (.+)").unwrap();
    for caps in re.captures_iter(data) {
        if let Some(msg) = caps.get(1) {
            println!("  エラー: {:?}", msg.as_bytes());
        }
    }
}

// --- regex-automata: 低レベルAPI ---
// regex クレートの内部エンジンを直接操作
// DFA の状態テーブルをシリアライズ・デシリアライズ可能
// 組み込みシステムなどコンパイル時間を削減したい場合に有用
```

### 5.5 Rust fancy-regex の詳細

```rust
// fancy-regex はルックアラウンドと後方参照をサポート
// regex クレートの DFA と NFA のハイブリッドエンジン
// DFA で処理できる部分は O(n)、ルックアラウンド部分だけ NFA にフォールバック

use fancy_regex::Regex;

fn main() {
    // 肯定先読み
    let re = Regex::new(r"\w+(?=\s*=)").unwrap();
    let text = "name = Alice";
    if let Ok(Some(m)) = re.find(text) {
        println!("{}", m.as_str());  // "name"
    }

    // 否定先読み
    let re2 = Regex::new(r"\b\w+\b(?!\s*=)").unwrap();
    // "=" の前にない単語にマッチ

    // 肯定後読み
    let re3 = Regex::new(r"(?<=\$)\d+").unwrap();
    let text2 = "price: $100";
    if let Ok(Some(m)) = re3.find(text2) {
        println!("{}", m.as_str());  // "100"
    }

    // 後方参照
    let re4 = Regex::new(r"<(\w+)>[^<]*</\1>").unwrap();
    let html = "<div>content</div>";
    if let Ok(Some(m)) = re4.find(html) {
        println!("{}", m.as_str());  // "<div>content</div>"
    }

    // 注意: fancy_regex::Regex と regex::Regex は別の型
    // API は似ているが Result を返す点が異なる
    // エラーハンドリングが必要:
    // regex:       re.find(text)        → Option<Match>
    // fancy_regex: re.find(text)        → Result<Option<Match>>
}
```

---

## 6. Java

### 6.1 基本API

```java
import java.util.regex.*;

public class RegexExample {
    public static void main(String[] args) {
        String text = "2026-02-11 Error: Connection failed";

        // Pattern + Matcher
        Pattern pattern = Pattern.compile("(\\d{4})-(\\d{2})-(\\d{2})");
        Matcher matcher = pattern.matcher(text);

        if (matcher.find()) {
            System.out.println("全体: " + matcher.group(0));
            System.out.println("年: " + matcher.group(1));
            System.out.println("月: " + matcher.group(2));
            System.out.println("日: " + matcher.group(3));
        }

        // 全マッチ
        Pattern digits = Pattern.compile("\\d+");
        Matcher dm = digits.matcher(text);
        while (dm.find()) {
            System.out.println("  " + dm.group());
        }

        // 置換
        String result = pattern.matcher(text)
            .replaceAll("$3/$2/$1");
        System.out.println(result);

        // 名前付きグループ
        Pattern named = Pattern.compile(
            "(?<year>\\d{4})-(?<month>\\d{2})-(?<day>\\d{2})"
        );
        Matcher nm = named.matcher(text);
        if (nm.find()) {
            System.out.println("年: " + nm.group("year"));
        }
    }
}
```

### 6.2 Java 固有の機能

```java
// 独占的量指定子 (Possessive Quantifiers)
// バックトラックを禁止して高速化
Pattern.compile("a++b");    // a を独占的に取得
Pattern.compile("[^\"]*+"); // " 以外を独占的に取得

// アトミックグループ (Java 不可 → 独占的量指定子で代替)
// (?>pattern) は Java 20+ で追加

// String.matches(): 文字列全体がマッチするか
boolean valid = "2026-02-11".matches("\\d{4}-\\d{2}-\\d{2}");

// Pattern.UNICODE_CHARACTER_CLASS (Java 7+)
// \w が Unicode 文字に拡張される
Pattern unicode = Pattern.compile("\\w+",
    Pattern.UNICODE_CHARACTER_CLASS);
```

### 6.3 Java の高度なAPIと実践テクニック

```java
import java.util.regex.*;
import java.util.stream.*;

public class AdvancedRegex {
    public static void main(String[] args) {
        // --- Matcher.appendReplacement / appendTail ---
        // 関数的な置換(Java 8 以前の方法)
        Pattern p = Pattern.compile("\\d+");
        Matcher m = p.matcher("abc 123 def 456");
        StringBuffer sb = new StringBuffer();
        while (m.find()) {
            int num = Integer.parseInt(m.group());
            m.appendReplacement(sb, String.valueOf(num * 2));
        }
        m.appendTail(sb);
        System.out.println(sb.toString());
        // => "abc 246 def 912"

        // --- Matcher.replaceAll with Function (Java 9+) ---
        String result = Pattern.compile("\\d+")
            .matcher("abc 123 def 456")
            .replaceAll(mr -> String.valueOf(Integer.parseInt(mr.group()) * 2));
        System.out.println(result);
        // => "abc 246 def 912"

        // --- Pattern.splitAsStream (Java 8+) ---
        Pattern sep = Pattern.compile("[,;\\s]+");
        long count = sep.splitAsStream("a, b; c d e")
            .filter(s -> !s.isEmpty())
            .count();
        System.out.println("要素数: " + count);  // => 5

        // --- Pattern.asPredicate (Java 8+) ---
        // Stream のフィルタリングに便利
        Pattern emailP = Pattern.compile("\\w+@\\w+\\.\\w+");
        java.util.List<String> items = java.util.List.of(
            "alice@example.com", "not-email", "bob@test.org"
        );
        items.stream()
            .filter(emailP.asPredicate())
            .forEach(System.out::println);
        // => alice@example.com
        //    bob@test.org

        // --- Pattern.asMatchPredicate (Java 11+) ---
        // matches() と同等(文字列全体がマッチするか)
        Pattern dateP = Pattern.compile("\\d{4}-\\d{2}-\\d{2}");
        System.out.println(dateP.asMatchPredicate().test("2026-02-11"));
        // => true
        System.out.println(dateP.asMatchPredicate().test("date: 2026-02-11"));
        // => false (全体一致ではない)

        // --- region: マッチ範囲の制限 ---
        String text = "hello world foo bar";
        Matcher rm = Pattern.compile("\\w+").matcher(text);
        rm.region(6, 11);  // "world" の範囲のみ
        if (rm.find()) {
            System.out.println(rm.group());  // => "world"
        }

        // --- lookingAt: 先頭からの部分一致 ---
        // (Python の re.match に相当)
        Matcher lm = Pattern.compile("\\d+").matcher("123abc");
        System.out.println(lm.lookingAt());  // true
        System.out.println(lm.group());       // "123"

        // --- hitEnd / requireEnd ---
        // パーサー実装時に有用
        Matcher he = Pattern.compile("abc").matcher("ab");
        he.find();
        System.out.println(he.hitEnd());
        // => true (入力の末尾に到達した = より長い入力ならマッチする可能性)
    }
}
```

### 6.4 Java の独占的量指定子とアトミックグループの詳細

```java
import java.util.regex.*;

public class PossessiveQuantifiers {
    public static void main(String[] args) {
        // --- 貪欲 vs 怠惰 vs 独占的 ---
        //
        // 貪欲 (greedy):  .*  → 最大限マッチし、必要ならバックトラック
        // 怠惰 (lazy):    .*? → 最小限マッチし、必要なら拡大
        // 独占的 (possessive): .*+ → 最大限マッチし、バックトラック禁止

        String text = "\"hello\" and \"world\"";

        // 貪欲: バックトラックあり
        System.out.println(
            Pattern.compile("\".*\"").matcher(text).results()
                .map(MatchResult::group).collect(java.util.stream.Collectors.toList())
        );
        // => ["hello" and "world"] (最大範囲)

        // 怠惰: 最小範囲
        System.out.println(
            Pattern.compile("\".*?\"").matcher(text).results()
                .map(MatchResult::group).collect(java.util.stream.Collectors.toList())
        );
        // => ["hello", "world"]

        // 独占的: バックトラックなし
        // ".*+" は全てを消費して戻らない → 最後の " にマッチしない
        System.out.println(
            Pattern.compile("\".*+\"").matcher(text).find()
        );
        // => false (マッチしない!)

        // 独占的量指定子の正しい使い方:
        // 否定文字クラスと組み合わせる
        System.out.println(
            Pattern.compile("\"[^\"]*+\"").matcher(text).results()
                .map(MatchResult::group).collect(java.util.stream.Collectors.toList())
        );
        // => ["hello", "world"] (高速かつ正確)

        // --- パフォーマンスの違い ---
        // ReDoS に弱いパターン:
        // Pattern.compile("(a+)+b");  → O(2^n)
        //
        // 独占的量指定子で防御:
        // Pattern.compile("(a++)+b"); → バックトラックなし、高速に失敗

        // --- アトミックグループ (Java 20+) ---
        // (?>pattern) はグループ全体のバックトラックを禁止
        // Pattern.compile("(?>a+)b");
        // 独占的量指定子 a++ と同等だが、より複雑なパターンに使える
        // Pattern.compile("(?>abc|ab)c");
        // → "abc" にマッチした後、"ab" へのバックトラックは行わない
    }
}
```

### 6.5 Java のパフォーマンス最適化

```java
import java.util.regex.*;

public class RegexPerformance {
    // --- Pattern はスレッドセーフだが Matcher はそうでない ---
    // Pattern はコンパイル済みで不変 → static final で共有可能
    // Matcher は内部状態を持つ → スレッドごとに生成
    private static final Pattern DATE_PATTERN =
        Pattern.compile("(\\d{4})-(\\d{2})-(\\d{2})");

    public static String findDate(String text) {
        Matcher m = DATE_PATTERN.matcher(text);  // Matcher は毎回生成
        return m.find() ? m.group() : null;
    }

    // --- String.matches() の罠 ---
    // String.matches() は毎回 Pattern.compile() を呼ぶ
    public static boolean isDateBad(String text) {
        return text.matches("\\d{4}-\\d{2}-\\d{2}");
        // ↑ 毎回コンパイル! ループ内で使うと非常に遅い
    }

    // 事前コンパイル版
    public static boolean isDateGood(String text) {
        return DATE_PATTERN.matcher(text).matches();
        // ↑ コンパイル済みパターンを再利用
    }

    // --- Matcher.reset() でオブジェクト再利用 ---
    // 大量の文字列を処理する場合に有効
    public static void processMany(String[] texts) {
        Pattern p = Pattern.compile("\\d+");
        Matcher m = p.matcher("");  // 空文字列で初期化
        for (String text : texts) {
            m.reset(text);  // Matcher を再利用
            while (m.find()) {
                System.out.println(m.group());
            }
        }
    }
}
```

---

## 7. ASCII 図解

### 7.1 言語別マッチモデルの違い

```
Python re.search()  →  文字列のどこかでマッチ
Python re.match()   →  文字列の先頭からマッチ
Python re.fullmatch()→  文字列全体がマッチ

JavaScript .match()  →  文字列のどこかで(gなしで最初のみ)
JavaScript .match(g) →  全マッチを配列で返す
JavaScript .test()   →  boolean

Go FindString()     →  文字列のどこかでマッチ
Go MatchString()    →  文字列のどこかにマッチがあるか(boolean)

Java find()         →  文字列のどこかでマッチ
Java matches()      →  文字列全体がマッチ

Rust find()         →  文字列のどこかでマッチ
Rust is_match()     →  boolean

┌──────────┬──────────┬──────────┬──────────┐
│ 動作      │ 部分一致  │ 先頭一致  │ 完全一致  │
├──────────┼──────────┼──────────┼──────────┤
│ Python   │ search() │ match()  │fullmatch()│
│ JavaScript│ .match()│ /^.../   │ /^...$/ │
│ Go       │ Find()   │ ―       │ Match()※│
│ Java     │ find()   │ ―       │matches()│
│ Rust     │ find()   │ ―       │ ―       │
└──────────┴──────────┴──────────┴──────────┘
※ Go の Match() は部分一致
```

### 7.2 フラグ構文の対比

```
同じ効果の異なる書き方:

大文字小文字無視:
  Python:     re.IGNORECASE / re.I / (?i)
  JavaScript: /pattern/i
  Go:         (?i)pattern
  Java:       Pattern.CASE_INSENSITIVE / (?i)
  Rust:       (?i)pattern

複数行:
  Python:     re.MULTILINE / re.M / (?m)
  JavaScript: /pattern/m
  Go:         (?m)pattern
  Java:       Pattern.MULTILINE / (?m)
  Rust:       (?m)pattern

ドットオール:
  Python:     re.DOTALL / re.S / (?s)
  JavaScript: /pattern/s
  Go:         (?s)pattern
  Java:       Pattern.DOTALL / (?s)
  Rust:       (?s)pattern
```

### 7.3 エスケープの違い

```
バックスラッシュのエスケープ:

Python:
  通常文字列: "\\d+"  → \d+
  raw string: r"\d+"  → \d+  (推奨)

JavaScript:
  リテラル:   /\d+/          (エスケープ不要)
  コンストラクタ: "\\d+"     (二重エスケープ)

Go:
  バッククォート: `\d+`       (raw string、推奨)
  通常文字列: "\\d+"          (二重エスケープ)

Java:
  通常文字列: "\\d+"          (二重エスケープ、唯一の方法)
  テキストブロック: \"""
    \d+                       (Java 15+ raw string 風)
  \"""

Rust:
  raw string: r"\d+"          (エスケープ不要)
  通常文字列: "\\d+"          (二重エスケープ)
```

### 7.4 API 呼び出しフロー図

```
各言語の典型的な正規表現処理フロー:

Python:
  パターン文字列 → re.compile() → Pattern オブジェクト
                                    ├─ .search(text) → Match | None
                                    ├─ .findall(text) → [str, ...]
                                    ├─ .finditer(text) → Iterator[Match]
                                    ├─ .sub(repl, text) → str
                                    └─ .split(text) → [str, ...]

JavaScript:
  /pattern/flags → RegExp オブジェクト
                    ├─ .test(str) → boolean
                    ├─ .exec(str) → Array | null
                    │
  str.match(re)  → Array | null (g なし) / [str, ...] (g あり)
  str.matchAll(re) → Iterator
  str.replace(re, repl) → string
  str.split(re) → [string, ...]

Go:
  パターン文字列 → regexp.Compile() → (*Regexp, error)
                   regexp.MustCompile() → *Regexp
                    ├─ .FindString(s) → string
                    ├─ .FindAllString(s, n) → []string
                    ├─ .FindStringSubmatch(s) → []string
                    ├─ .ReplaceAllString(s, repl) → string
                    ├─ .MatchString(s) → bool
                    └─ .Split(s, n) → []string

Rust:
  パターン文字列 → Regex::new() → Result<Regex>
                    ├─ .find(text) → Option<Match>
                    ├─ .find_iter(text) → Iterator<Match>
                    ├─ .captures(text) → Option<Captures>
                    ├─ .captures_iter(text) → Iterator<Captures>
                    ├─ .replace(text, rep) → Cow<str>
                    ├─ .replace_all(text, rep) → Cow<str>
                    ├─ .is_match(text) → bool
                    └─ .split(text) → Iterator<&str>

Java:
  パターン文字列 → Pattern.compile() → Pattern
                    ├─ .matcher(text) → Matcher
                    │   ├─ .find() → boolean
                    │   ├─ .matches() → boolean
                    │   ├─ .group(n) → String
                    │   ├─ .replaceAll(repl) → String
                    │   └─ .results() → Stream<MatchResult>
                    ├─ .split(text) → String[]
                    ├─ .splitAsStream(text) → Stream<String>
                    └─ .asPredicate() → Predicate<String>
```

---

## 8. 比較表

### 8.1 機能サポート比較

| 機能 | Python | JavaScript | Go | Rust | Java |
|------|--------|------------|-----|------|------|
| 後方参照 `\1` | OK | OK | 不可 | 不可 | OK |
| 名前付きグループ | `(?P<>)` | `(?<>)` | `(?P<>)` | `(?P<>)` | `(?<>)` |
| 肯定先読み `(?=)` | OK | OK | 不可 | 不可 | OK |
| 否定先読み `(?!)` | OK | OK | 不可 | 不可 | OK |
| 肯定後読み `(?<=)` | 固定長 | 可変長 | 不可 | 不可 | 固定長 |
| 独占的量指定子 `*+` | 不可 | 不可 | 不可 | 不可 | OK |
| アトミックグループ | 不可 | 不可 | 不可 | 不可 | Java 20+ |
| Unicode `\p{}` | regex モジュール | `/u` | 一部 | OK | OK |
| VERBOSE/コメント | `re.X` | 不可 | 不可 | `(?x)` | `(?x)` |
| O(n) 保証 | 不可 | 不可 | OK | OK | 不可 |

### 8.2 パフォーマンス特性比較

| 言語 | エンジン | 最悪計算量 | コンパイル方式 | 並列安全 |
|------|---------|----------|-------------|---------|
| Python | re (NFA) | O(2^n) | バイトコード | スレッド安全 |
| JavaScript | V8 Irregexp | O(2^n) | JIT | -- |
| Go | RE2 (DFA) | O(n) | DFA テーブル | goroutine 安全 |
| Rust | regex (DFA) | O(n) | DFA + NFA hybrid | Send + Sync |
| Java | Pattern (NFA) | O(2^n) | NFA バイトコード | スレッド安全 |

### 8.3 Unicode サポート詳細比較

| 機能 | Python re | Python regex | JavaScript | Go | Rust | Java |
|------|-----------|-------------|------------|-----|------|------|
| `\w` Unicode 対応 | デフォルト | デフォルト | `\p{L}` 必要 | ASCII | Unicode | フラグ必要 |
| `\d` Unicode 対応 | デフォルト | デフォルト | `/u` | ASCII | Unicode | フラグ必要 |
| `\p{Script}` | 不可 | OK | `/u` | 一部 | OK | OK |
| `\p{General_Category}` | 不可 | OK | `/u` | 一部 | OK | OK |
| 書記素クラスタ `\X` | 不可 | OK | 不可 | 不可 | 不可 | 不可 |
| Emoji 対応 | 限定的 | OK | `/v` | 限定的 | OK | Java 20+ |

```
Unicode の \w の挙動差異:

入力: "hello 世界 café"

Python 3 re:         \w+ → ['hello', '世界', 'café']    (Unicode)
Python 3 re(ASCII):  \w+ → ['hello', 'caf']             (ASCII)
JavaScript:          \w+ → ['hello', 'caf']              (ASCII)
JavaScript (/u):     \w+ → ['hello', 'caf']              (u でも ASCII!)
Go:                  \w+ → ['hello', 'caf']              (ASCII)
Rust:                \w+ → ['hello', '世界', 'café']     (Unicode)
Java:                \w+ → ['hello', 'caf']              (ASCII)
Java (UNICODE):      \w+ → ['hello', '世界', 'café']     (Unicode)
```

### 8.4 コンパイルキャッシュ機構の比較

```
┌──────────┬────────────────────┬─────────────────────────────┐
│ 言語      │ キャッシュ機構       │ 推奨事項                     │
├──────────┼────────────────────┼─────────────────────────────┤
│ Python   │ 内部キャッシュ(512) │ compile() は512超で有効      │
│ JavaScript│ V8 内部キャッシュ   │ リテラルは自動キャッシュ       │
│ Go       │ なし(手動管理)     │ パッケージレベル変数で保持     │
│ Rust     │ なし(手動管理)     │ once_cell / LazyLock で保持  │
│ Java     │ なし(手動管理)     │ static final で Pattern 保持 │
└──────────┴────────────────────┴─────────────────────────────┘

パフォーマンス影響(ループ内で 10,000 回マッチの場合):

Python re.search(r'\d+', text) × 10000
  キャッシュヒット時: ~15ms
  キャッシュミス時:   ~50ms  (512パターン超の場合)
  compile() 使用時:   ~12ms

JavaScript /\d+/.test(text) × 10000
  リテラル:          ~5ms
  new RegExp():     ~20ms  (毎回コンパイル)

Go re.FindString(text) × 10000
  MustCompile済み:   ~8ms
  毎回Compile:       ~200ms  (極端に遅い!)

Rust re.find(text) × 10000
  Lazy::new()済み:   ~3ms
  毎回Regex::new():  ~300ms  (極端に遅い!)

Java pattern.matcher(text).find() × 10000
  Pattern.compile済み: ~6ms
  String.matches():    ~60ms  (毎回コンパイル)

※ 数値は概算。実際の値はハードウェアとパターンの複雑さに依存
```

---

## 9. アンチパターン

### 9.1 アンチパターン: 言語間でパターンをそのまま移植する

```python
# Python で動作するパターン
import re
pattern_py = r'(?P<date>\d{4}-\d{2}-\d{2})'
# (?P<name>...) は Python/Go/Rust の構文

# JavaScript に移植する際:
# NG: (?P<date>...) は JavaScript では SyntaxError
# OK: (?<date>...) に変換が必要
```

```javascript
// JavaScript で動作するパターン
const pattern_js = /(?<=\$)\d+/;
// 可変長の後読みは JavaScript 固有

// Python に移植する際:
// NG: 可変長後読みは Python re では不可
// OK: regex モジュールを使うか、パターンを変更
```

### 9.2 アンチパターン: `\w` の Unicode 挙動を想定しない

```python
# Python 3: \w は Unicode 文字にマッチ (デフォルト)
import re
print(re.findall(r'\w+', "hello 世界"))
# => ['hello', '世界']
```

```javascript
// JavaScript: \w は ASCII のみ (u フラグでも変わらない)
console.log("hello 世界".match(/\w+/g));
// => ['hello']  -- '世界' はマッチしない!

// Unicode 文字を含めるには:
console.log("hello 世界".match(/[\p{L}\p{N}_]+/gu));
// => ['hello', '世界']
```

### 9.3 アンチパターン: 貪欲マッチの暗黙的な想定

```python
# 全言語共通の罠: 貪欲マッチのデフォルト
import re

html = '<div class="a">text1</div><div class="b">text2</div>'

# NG: 貪欲マッチで全体を取得してしまう
print(re.findall(r'<div.*>.*</div>', html))
# => ['<div class="a">text1</div><div class="b">text2</div>']

# OK: 怠惰マッチ
print(re.findall(r'<div.*?>.*?</div>', html))
# => ['<div class="a">text1</div>', '<div class="b">text2</div>']

# BETTER: 否定文字クラス(最も高速)
print(re.findall(r'<div[^>]*>[^<]*</div>', html))
# => ['<div class="a">text1</div>', '<div class="b">text2</div>']
```

### 9.4 アンチパターン: ReDoS を引き起こすパターン

```
ReDoS (Regular Expression Denial of Service) は NFA エンジンの
バックトラッキングを悪用する攻撃。Go と Rust は DFA のため影響なし。

危険なパターンの共通構造:
  1. ネストされた量指定子: (a+)+
  2. 重複する選択肢: (a|a)*
  3. 重複する文字クラス: (\w|\d)+

具体例と修正:

危険: /([\w.]+)+@/         ← ネストされた量指定子
安全: /[\w.]+@/            ← ネスト解消

危険: /([a-zA-Z]|[0-9])+/ ← 重複する選択肢
安全: /[a-zA-Z0-9]+/      ← 統合

危険: /^(a+)+$/            ← 典型的な ReDoS パターン
安全: /^a+$/               ← ネスト解消

バックトラッキングの可視化:

パターン: (a+)+b
入力:     "aaaaaac"

試行 1: (aaaaa)(a) → b? 不一致
試行 2: (aaaa)(aa) → b? 不一致
試行 3: (aaaa)(a)(a) → b? 不一致
試行 4: (aaa)(aaa) → b? 不一致
試行 5: (aaa)(aa)(a) → b? 不一致
...
→ 2^n 通りの分割を試行 (n=6 で 32通り、n=30 で 10億通り)
```

### 9.5 アンチパターン: 不要なキャプチャグループ

```python
# 全言語共通: 不要なキャプチャはパフォーマンスに影響
import re

# NG: キャプチャ不要なのにグループを使用
pattern_bad = re.compile(r'(https?)://([\w.-]+)(/[\w./]*)?')

# OK: 非キャプチャグループを使用
pattern_good = re.compile(r'(?:https?)://(?:[\w.-]+)(?:/[\w./]*)?')

# さらに良い: グループ自体が不要なら除去
pattern_best = re.compile(r'https?://[\w.-]+(?:/[\w./]*)?')
```

```go
// Go では特に重要: FindAllString vs FindAllStringSubmatch
// キャプチャグループの有無で返り値の型が異なる
re1 := regexp.MustCompile(`\d+`)
re2 := regexp.MustCompile(`(\d+)`)

// re1.FindAllString()       → []string (効率的)
// re2.FindAllStringSubmatch() → [][]string (メモリ増)
```

---

## 10. FAQ

### Q1: Python の `re.match` と `re.search` の違いは？

**A**: `re.match` は文字列の **先頭** からのみマッチを試行する。`re.search` は文字列の **任意の位置** でマッチを試行する:

```python
import re
text = "say hello"
print(re.match(r'hello', text))    # => None (先頭は 'say')
print(re.search(r'hello', text))   # => Match (位置4でマッチ)
```

他の言語に `match` 相当はない(`^` アンカーで代替)。

### Q2: JavaScript でプリコンパイルは必要か？

**A**: リテラル構文 `/pattern/` はパース時にコンパイルされるため、通常は明示的なプリコンパイルは不要。ただし `new RegExp()` で動的に生成する場合はループ外で生成すべき:

```javascript
// OK: リテラルはパース時にコンパイル済み
for (const line of lines) {
    line.match(/\d+/g);  // 高速
}

// NG: ループ内で毎回コンパイル
for (const line of lines) {
    const re = new RegExp("\\d+", "g");  // 毎回コンパイル
    line.match(re);
}

// OK: ループ外でコンパイル
const re = /\d+/g;
for (const line of lines) {
    re.lastIndex = 0;  // g フラグの場合はリセット必要
    line.match(re);
}
```

### Q3: Go で後方参照が必要な場合はどうするか？

**A**: Go の標準 `regexp` パッケージでは後方参照は使えない。代替策:

1. **パターンを分割して複数回マッチ**: 最初のマッチ結果を使って2回目のパターンを構築
2. **文字列操作で処理**: `strings` パッケージの関数で代替
3. **サードパーティライブラリ**: `github.com/dlclark/regexp2` (PCRE互換)

```go
// regexp2 の例:
// import "github.com/dlclark/regexp2"
// re := regexp2.MustCompile(`<(\w+)>.*?</\1>`, 0)
// match, _ := re.FindStringMatch("<div>test</div>")
```

### Q4: Rust で正規表現をグローバルに保持するベストプラクティスは？

**A**: Rust では `Regex::new()` のコストが高いため、一度コンパイルしたパターンを再利用する。現在の推奨は `std::sync::LazyLock`（Rust 1.80+）または `once_cell::sync::Lazy`:

```rust
// Rust 1.80+ の場合
use std::sync::LazyLock;
use regex::Regex;

static EMAIL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\w.+-]+@[\w-]+\.[\w.]+").unwrap()
});

// 1.80 未満の場合
use once_cell::sync::Lazy;
static EMAIL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[\w.+-]+@[\w-]+\.[\w.]+").unwrap()
});
```

### Q5: Java の `String.matches()` はなぜ遅いのか？

**A**: `String.matches()` は内部で毎回 `Pattern.compile()` を呼ぶ。ループ内で使用すると、同じパターンを繰り返しコンパイルすることになり大幅な性能低下が起きる:

```java
// NG: 10,000回コンパイルが発生
for (String line : lines) {
    if (line.matches("\\d{4}-\\d{2}-\\d{2}")) { ... }
}

// OK: 1回だけコンパイル
Pattern p = Pattern.compile("\\d{4}-\\d{2}-\\d{2}");
for (String line : lines) {
    if (p.matcher(line).matches()) { ... }
}
```

### Q6: 複数言語で同じパターンを使いたい場合の共通サブセットは？

**A**: 以下の構文は5言語全てで共通して使用可能:

```
5言語共通サブセット:
  ✓ 文字クラス:     [abc], [a-z], [^abc]
  ✓ メタ文字:       \d, \D, \s, \S, \w (※Unicode挙動は異なる)
  ✓ 量指定子:       *, +, ?, {n}, {n,}, {n,m}
  ✓ 怠惰量指定子:   *?, +?, ??
  ✓ アンカー:       ^, $, \b
  ✓ 選択:          a|b
  ✓ グループ:       (...)
  ✓ 非キャプチャ:    (?:...)
  ✓ インラインフラグ: (?i), (?m), (?s) (※一部制限あり)

注意が必要な構文:
  △ 名前付きグループ: (?P<name>...) は Python/Go/Rust
                      (?<name>...) は JavaScript/Java
  △ \w のUnicode:    Python/Rust はUnicode、他はASCII
  × ルックアラウンド: Go/Rust では不可
  × 後方参照:       Go/Rust では不可
```

### Q7: 正規表現のデバッグ方法は言語ごとにどう違うか？

**A**: 各言語には正規表現のデバッグに役立つ固有の機能がある:

```python
# Python: re.DEBUG フラグ
import re
re.compile(r'(\d{4})-(\d{2})', re.DEBUG)
# パターンの解析木を表示:
# SUBPATTERN 1 0 0
#   MAX_REPEAT 4 4
#     IN
#       CATEGORY CATEGORY_DIGIT
# LITERAL 45
# SUBPATTERN 2 0 0
#   MAX_REPEAT 2 2
#     IN
#       CATEGORY CATEGORY_DIGIT
```

```go
// Go: regexp.Compile のエラーメッセージが詳細
_, err := regexp.Compile(`(?<=abc)`)
// err: "error parsing regexp: invalid or unsupported Perl syntax: `(?<`"

// regexp/syntax パッケージで構文解析
import "regexp/syntax"
prog, err := syntax.Parse(`\d+`, syntax.Perl)
// 構文木を取得して分析
```

```java
// Java: Matcher.toMatchResult() でマッチ状態を保存
Pattern p = Pattern.compile("(\\w+)");
Matcher m = p.matcher("hello world");
while (m.find()) {
    MatchResult mr = m.toMatchResult();
    System.out.printf("group=%s start=%d end=%d%n",
        mr.group(), mr.start(), mr.end());
}
```

### Q8: JavaScript の g フラグの落とし穴は？

**A**: `g` フラグ付きの RegExp は内部に `lastIndex` 状態を持ち、連続する `test()` や `exec()` 呼び出しで前回のマッチ位置から検索を開始する。これにより予期しない false が返ることがある:

```javascript
const re = /abc/g;

// 1回目: lastIndex=0 から検索 → マッチ、lastIndex=3
console.log(re.test("abcabc"));  // true

// 2回目: lastIndex=3 から検索 → マッチ、lastIndex=6
console.log(re.test("abcabc"));  // true

// 3回目: lastIndex=6 から検索 → マッチなし、lastIndex=0
console.log(re.test("abcabc"));  // false!

// 対策1: 毎回 lastIndex をリセット
re.lastIndex = 0;

// 対策2: test() には g フラグを使わない
const reNoG = /abc/;
console.log(reNoG.test("abcabc"));  // 常に true

// 対策3: String.prototype.match() を使う
console.log("abcabc".match(/abc/g));  // ['abc', 'abc']
```

---

## 11. 実践パターン集: 言語横断リファレンス

### 11.1 メールアドレスのバリデーション

```python
# Python
import re
email_re = re.compile(r'^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$')
print(email_re.match("user@example.com"))  # Match
print(email_re.match("invalid@"))          # None
```

```javascript
// JavaScript
const emailRe = /^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$/;
console.log(emailRe.test("user@example.com"));  // true
console.log(emailRe.test("invalid@"));           // false
```

```go
// Go
emailRe := regexp.MustCompile(`^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$`)
fmt.Println(emailRe.MatchString("user@example.com"))  // true
fmt.Println(emailRe.MatchString("invalid@"))           // false
```

```rust
// Rust
let email_re = Regex::new(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$").unwrap();
println!("{}", email_re.is_match("user@example.com"));  // true
println!("{}", email_re.is_match("invalid@"));           // false
```

```java
// Java
Pattern emailRe = Pattern.compile("[\\w.+-]+@[\\w-]+(?:\\.[\\w-]+)+");
System.out.println(emailRe.matcher("user@example.com").matches());  // true
System.out.println(emailRe.matcher("invalid@").matches());           // false
```

### 11.2 IPv4 アドレスの抽出

```python
# Python
import re
ipv4_re = re.compile(
    r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
)
text = "Server at 192.168.1.100, gateway 10.0.0.1, invalid 999.999.999.999"
print(ipv4_re.findall(text))
# => ['192.168.1.100', '10.0.0.1']
```

```javascript
// JavaScript
const ipv4Re = /\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b/g;
const text = "Server at 192.168.1.100, gateway 10.0.0.1, invalid 999.999.999.999";
console.log(text.match(ipv4Re));
// => ['192.168.1.100', '10.0.0.1']
```

```go
// Go
ipv4Re := regexp.MustCompile(
    `\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b`,
)
text := "Server at 192.168.1.100, gateway 10.0.0.1"
fmt.Println(ipv4Re.FindAllString(text, -1))
// => [192.168.1.100 10.0.0.1]
```

```rust
// Rust
let ipv4_re = Regex::new(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
).unwrap();
let text = "Server at 192.168.1.100, gateway 10.0.0.1";
let ips: Vec<&str> = ipv4_re.find_iter(text).map(|m| m.as_str()).collect();
println!("{:?}", ips);
// => ["192.168.1.100", "10.0.0.1"]
```

```java
// Java
Pattern ipv4Re = Pattern.compile(
    "\\b(?:(?:25[0-5]|2[0-4]\\d|[01]?\\d\\d?)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d\\d?)\\b"
);
String text = "Server at 192.168.1.100, gateway 10.0.0.1";
Matcher m = ipv4Re.matcher(text);
while (m.find()) {
    System.out.println(m.group());
}
// => 192.168.1.100
//    10.0.0.1
```

### 11.3 ログファイルのパース

```python
# Python: 構造化ログの解析
import re

log_re = re.compile(r'''
    ^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})  # ISO 8601
    \s+\[(?P<level>\w+)\]                                  # ログレベル
    \s+(?P<source>[\w.]+):                                  # ソース
    \s+(?P<message>.+)$                                     # メッセージ
''', re.VERBOSE | re.MULTILINE)

log_text = """2026-02-11T10:30:45 [ERROR] app.server: Connection refused
2026-02-11T10:30:46 [INFO] app.db: Retry attempt 1
2026-02-11T10:30:47 [WARN] app.cache: Cache miss for key 'user:123'"""

for m in log_re.finditer(log_text):
    d = m.groupdict()
    print(f"  [{d['level']}] {d['source']} → {d['message']}")
```

```javascript
// JavaScript: 同じログの解析
const logRe = /^(?<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+\[(?<level>\w+)\]\s+(?<source>[\w.]+):\s+(?<message>.+)$/gm;

const logText = `2026-02-11T10:30:45 [ERROR] app.server: Connection refused
2026-02-11T10:30:46 [INFO] app.db: Retry attempt 1
2026-02-11T10:30:47 [WARN] app.cache: Cache miss for key 'user:123'`;

for (const m of logText.matchAll(logRe)) {
    const { level, source, message } = m.groups;
    console.log(`  [${level}] ${source} → ${message}`);
}
```

```go
// Go: 同じログの解析
logRe := regexp.MustCompile(
    `(?m)^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+` +
    `\[(?P<level>\w+)\]\s+(?P<source>[\w.]+):\s+(?P<message>.+)$`,
)

logText := `2026-02-11T10:30:45 [ERROR] app.server: Connection refused
2026-02-11T10:30:46 [INFO] app.db: Retry attempt 1`

names := logRe.SubexpNames()
for _, match := range logRe.FindAllStringSubmatch(logText, -1) {
    for i, name := range names {
        if name != "" {
            fmt.Printf("  %s: %s\n", name, match[i])
        }
    }
    fmt.Println()
}
```

### 11.4 URL のパースと分解

```python
# Python
import re

url_re = re.compile(r'''
    ^(?P<scheme>https?)://
    (?P<host>[^/:]+)
    (?::(?P<port>\d+))?
    (?P<path>/[^?#]*)?
    (?:\?(?P<query>[^#]*))?
    (?:\#(?P<fragment>.*))?$
''', re.VERBOSE)

urls = [
    "https://example.com:8080/api/v1/users?page=1&limit=10#section",
    "http://localhost/health",
    "https://cdn.example.com/assets/style.css",
]

for url in urls:
    m = url_re.match(url)
    if m:
        d = {k: v for k, v in m.groupdict().items() if v is not None}
        print(f"  {d}")
```

```javascript
// JavaScript
const urlRe = /^(?<scheme>https?):\/\/(?<host>[^\/:]+)(?::(?<port>\d+))?(?<path>\/[^?#]*)?(?:\?(?<query>[^#]*))?(?:#(?<fragment>.*))?$/;

const urls = [
    "https://example.com:8080/api/v1/users?page=1&limit=10#section",
    "http://localhost/health",
];

for (const url of urls) {
    const m = url.match(urlRe);
    if (m) {
        // undefined を除外
        const parts = Object.fromEntries(
            Object.entries(m.groups).filter(([_, v]) => v !== undefined)
        );
        console.log(parts);
    }
}
```

### 11.5 CSVフィールドの解析(引用符対応)

```python
# Python: 引用符で囲まれたフィールドを含む CSV の解析
import re

# 引用符付きフィールドと通常フィールドを正しく分割
csv_field_re = re.compile(r'''
    (?:                     # フィールドの開始
      "([^"]*(?:""[^"]*)*)" # 引用符付き: "..." (内部の "" はエスケープ)
      |                     # または
      ([^,]*)               # 引用符なし: , まで
    )
    (?:,|$)                 # , または行末
''', re.VERBOSE)

line = '"John ""Johnny"" Doe",30,"New York, NY",active'
fields = []
for m in csv_field_re.finditer(line):
    if m.group(1) is not None:
        fields.append(m.group(1).replace('""', '"'))
    else:
        fields.append(m.group(2))
print(fields)
# => ['John "Johnny" Doe', '30', 'New York, NY', 'active']
```

---

## 12. 移植チェックリスト

言語間で正規表現を移植する際のチェックリスト:

```
┌─────────────────────────────────────────────────────────────────┐
│ 移植元 → 移植先                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Python → JavaScript                                            │
│  □ (?P<name>...) → (?<name>...) に変更                         │
│  □ re.VERBOSE のコメント → 除去(JS はコメント不可)              │
│  □ \w の Unicode 挙動 → [\p{L}\p{N}_] に変更                  │
│  □ fullmatch() → /^...$/ に変更                                │
│  □ 条件付きパターン (?(id)yes|no) → 使用不可、ロジックで代替    │
│  □ re.sub の関数置換 → replace のコールバック関数に変更         │
├─────────────────────────────────────────────────────────────────┤
│ Python → Go                                                    │
│  □ ルックアラウンド → キャプチャグループ+後処理で代替           │
│  □ 後方参照 → 複数ステップ処理で代替                           │
│  □ \w の Unicode 挙動 → Go では ASCII のみ                     │
│  □ re.sub の関数置換 → ReplaceAllStringFunc に変更             │
│  □ findall → FindAllString に変更 (第2引数 -1 が必要)          │
├─────────────────────────────────────────────────────────────────┤
│ JavaScript → Python                                            │
│  □ (?<name>...) → (?P<name>...) に変更                         │
│  □ 可変長後読み → 固定長に変更 or regex モジュール使用          │
│  □ /pattern/flags → re.compile(r'pattern', flags) に変更       │
│  □ $1, $2 → \1, \2 (sub の置換文字列)                         │
│  □ g フラグ → findall/finditer で全マッチ取得                  │
│  □ v フラグの集合演算 → regex モジュールで代替                  │
├─────────────────────────────────────────────────────────────────┤
│ Java → Rust                                                    │
│  □ (?<name>...) → (?P<name>...) に変更                         │
│  □ ルックアラウンド → fancy-regex に切り替え or 除去            │
│  □ 後方参照 → fancy-regex に切り替え or 除去                   │
│  □ 独占的量指定子 *+ → 除去(Rust regex では不可)               │
│  □ 二重エスケープ \\\\ → raw string r"" に変更                 │
│  □ Matcher の状態管理 → イテレータベースの API に変更           │
├─────────────────────────────────────────────────────────────────┤
│ どの言語からでも Go/Rust へ                                     │
│  □ ルックアラウンド全般 → 除去してロジックで代替                │
│  □ 後方参照 → 除去してロジックで代替                           │
│  □ アトミックグループ → 不要(DFA なのでバックトラックなし)      │
│  □ 独占的量指定子 → 不要(DFA なのでバックトラックなし)          │
│  □ ReDoS 対策 → 不要(O(n) 保証)                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| Python | NFA、Unicode デフォルト、`(?P<>)` 構文、`re.VERBOSE` |
| JavaScript | NFA、ES2018 で大幅拡張、`/u` `/s` `/d` フラグ |
| Go | RE2(DFA)、O(n) 保証、後方参照/ルックアラウンドなし |
| Rust | DFA hybrid、O(n) 保証、`RegexSet` で複数同時マッチ |
| Java | NFA、独占的量指定子あり、二重エスケープが必須 |
| 移植の注意 | 名前付きグループ構文、`\w` の Unicode 挙動、後読みの制約が異なる |

## 次に読むべきガイド

- [01-common-patterns.md](./01-common-patterns.md) -- よく使うパターン集
- [02-text-processing.md](./02-text-processing.md) -- テキスト処理ツール

## 参考文献

1. **Python re module** https://docs.python.org/3/library/re.html -- Python 正規表現公式リファレンス
2. **MDN RegExp** https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/RegExp -- JavaScript RegExp の完全ガイド
3. **Go regexp package** https://pkg.go.dev/regexp -- Go 正規表現パッケージ仕様
4. **Rust regex crate** https://docs.rs/regex/latest/regex/ -- Rust regex クレートのドキュメント
5. **Java Pattern class** https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/regex/Pattern.html -- Java Pattern クラス公式リファレンス
6. **RE2 Syntax** https://github.com/google/re2/wiki/Syntax -- RE2 エンジンの構文仕様
7. **fancy-regex** https://docs.rs/fancy-regex/latest/fancy_regex/ -- Rust のルックアラウンド対応 regex クレート
8. **Python regex module** https://pypi.org/project/regex/ -- Python の拡張正規表現モジュール
9. **ECMAScript 2024 RegExp v flag** https://tc39.es/ecma262/ -- JavaScript の最新 RegExp 仕様
10. **Russ Cox "Regular Expression Matching Can Be Simple And Fast"** https://swtch.com/~rsc/regexp/regexp1.html -- NFA/DFA エンジンの原理解説
