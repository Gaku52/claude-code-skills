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
