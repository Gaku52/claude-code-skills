# 正規表現の代替

> パーサーコンビネータ、PEG、構造化テキスト処理など、正規表現では困難なテキスト解析タスクの代替手法を習得する

## この章で学ぶこと

1. **正規表現の限界** — 再帰構造、ネスト、文脈依存文法が扱えない理由
2. **パーサーコンビネータ** — 小さなパーサーを組み合わせて複雑な文法を解析する手法
3. **PEG と実用ツール** — Parsing Expression Grammar の特性と tree-sitter 等のツール

---

## 1. 正規表現の限界

```
チョムスキー階層とパーサーの対応
=================================

Type 0: 句構造文法    <-- チューリングマシン
Type 1: 文脈依存文法  <-- 線形拘束オートマトン
Type 2: 文脈自由文法  <-- プッシュダウンオートマトン ★
Type 3: 正規文法      <-- 有限オートマトン (正規表現)

正規表現が扱えないもの:
  - 対応する括弧:  ((()))     ★ 文脈自由文法
  - HTML のネスト: <div><div></div></div>
  - プログラミング言語の構文
  - 再帰的な構造全般

正規表現で「やってはいけない」こと:
  - HTML/XML のパース
  - JSON のパース
  - プログラミング言語の解析
  - ネストした括弧のマッチング
```

### コード例 1: 正規表現の限界を示す例

```python
import re

# [NG] HTML を正規表現でパース
html = '<div class="outer"><div class="inner">text</div></div>'
# この正規表現はネストに対応できない
pattern = r'<div[^>]*>(.*?)</div>'
matches = re.findall(pattern, html)
# 期待: inner div の中身 → 実際: 最短マッチで不正確な結果

# [OK] 専用パーサーを使う
from html.parser import HTMLParser
# または BeautifulSoup, lxml 等

# [NG] JSON を正規表現でパース
json_str = '{"key": {"nested": [1, 2, 3]}}'
# 再帰構造は正規表現では不可能

# [OK] json モジュールを使う
import json
data = json.loads(json_str)
```

---

## 2. パーサーコンビネータ

```
パーサーコンビネータの考え方
==============================

小さなパーサーを組み合わせて大きなパーサーを構築

基本パーサー:
  digit    : "0"-"9" を1文字パース
  letter   : "a"-"z" を1文字パース
  string   : 固定文字列をパース

コンビネータ:
  seq(a, b)    : a の後に b   (逐次)
  alt(a, b)    : a または b   (選択)
  many(a)      : a の0回以上の繰り返し
  map(a, f)    : a の結果に f を適用

例: 整数パーサー
  integer = map(many1(digit), digits => parseInt(digits.join('')))

例: 四則演算パーサー
  expr   = alt(addExpr, term)
  term   = alt(mulExpr, factor)
  factor = alt(number, parens(expr))  ← 再帰!
```

### コード例 2: TypeScript でのパーサーコンビネータ

```typescript
// パーサーの型定義
type Parser<T> = (input: string, pos: number) => ParseResult<T>;
type ParseResult<T> =
  | { success: true; value: T; pos: number }
  | { success: false; expected: string; pos: number };

// 基本パーサー
function char(c: string): Parser<string> {
  return (input, pos) =>
    input[pos] === c
      ? { success: true, value: c, pos: pos + 1 }
      : { success: false, expected: `'${c}'`, pos };
}

function regex(pattern: RegExp): Parser<string> {
  return (input, pos) => {
    const match = input.slice(pos).match(pattern);
    if (match && match.index === 0) {
      return { success: true, value: match[0], pos: pos + match[0].length };
    }
    return { success: false, expected: pattern.toString(), pos };
  };
}

// コンビネータ
function seq<A, B>(pa: Parser<A>, pb: Parser<B>): Parser<[A, B]> {
  return (input, pos) => {
    const ra = pa(input, pos);
    if (!ra.success) return ra as any;
    const rb = pb(input, ra.pos);
    if (!rb.success) return rb as any;
    return { success: true, value: [ra.value, rb.value], pos: rb.pos };
  };
}

function alt<T>(...parsers: Parser<T>[]): Parser<T> {
  return (input, pos) => {
    for (const p of parsers) {
      const r = p(input, pos);
      if (r.success) return r;
    }
    return { success: false, expected: "one of alternatives", pos };
  };
}

function many<T>(parser: Parser<T>): Parser<T[]> {
  return (input, pos) => {
    const results: T[] = [];
    let current = pos;
    while (true) {
      const r = parser(input, current);
      if (!r.success) break;
      results.push(r.value);
      current = r.pos;
    }
    return { success: true, value: results, pos: current };
  };
}

function map<A, B>(parser: Parser<A>, fn: (a: A) => B): Parser<B> {
  return (input, pos) => {
    const r = parser(input, pos);
    if (!r.success) return r as any;
    return { success: true, value: fn(r.value), pos: r.pos };
  };
}

// 使用例: 四則演算パーサー
const digit = regex(/[0-9]+/);
const number = map(digit, s => parseInt(s, 10));
const ws = regex(/\s*/);

function token<T>(p: Parser<T>): Parser<T> {
  return (input, pos) => {
    const r = ws(input, pos);
    return p(input, r.success ? r.pos : pos);
  };
}

// "123 + 456" をパース
const addExpr = (input: string, pos: number): ParseResult<number> => {
  const left = token(number)(input, pos);
  if (!left.success) return left;
  const op = token(char('+'))(input, left.pos);
  if (!op.success) return left;  // 加算なし → 数値のみ
  const right = addExpr(input, op.pos);  // 再帰
  if (!right.success) return right;
  return { success: true, value: left.value + right.value, pos: right.pos };
};
```

---

## 3. PEG（Parsing Expression Grammar）

### コード例 3: PEG.js / Peggy による文法定義

```javascript
// grammar.pegjs (Peggy 形式)
// JSON パーサーの PEG 文法

Value
  = Object / Array / String / Number / Boolean / Null

Object
  = "{" _ head:Pair tail:("," _ p:Pair { return p; })* _ "}"
    { return Object.fromEntries([head, ...tail]); }
  / "{" _ "}" { return {}; }

Pair
  = key:String _ ":" _ value:Value { return [key, value]; }

Array
  = "[" _ head:Value tail:("," _ v:Value { return v; })* _ "]"
    { return [head, ...tail]; }
  / "[" _ "]" { return []; }

String
  = '"' chars:[^"]* '"' { return chars.join(""); }

Number
  = digits:[0-9]+ { return parseInt(digits.join(""), 10); }

Boolean
  = "true" { return true; }
  / "false" { return false; }

Null
  = "null" { return null; }

_ = [ \t\n\r]*
```

```bash
# Peggy でパーサーを生成
npx peggy grammar.pegjs --output parser.js

# 使用
node -e "const p = require('./parser'); console.log(p.parse('{\"a\": 1}'))"
```

---

## 4. 実用的な代替ツール

### コード例 4: Python の pyparsing

```python
from pyparsing import (
    Word, alphas, alphanums, nums, Suppress, Group,
    Forward, Optional, ZeroOrMore, Literal, quotedString
)

# SQL の SELECT 文パーサー（簡易版）
identifier = Word(alphas + "_", alphanums + "_")
number = Word(nums)
string_literal = quotedString

# SELECT column1, column2 FROM table WHERE condition
select_stmt = (
    Suppress(Literal("SELECT")) +
    Group(identifier + ZeroOrMore(Suppress(",") + identifier))("columns") +
    Suppress(Literal("FROM")) +
    identifier("table") +
    Optional(
        Suppress(Literal("WHERE")) +
        identifier("where_col") +
        Literal("=") +
        (number | string_literal)("where_val")
    )
)

# パース実行
result = select_stmt.parseString("SELECT name, age FROM users WHERE status = 'active'")
print(result.columns.asList())  # ['name', 'age']
print(result.table)             # 'users'
print(result.where_col)         # 'status'
```

### コード例 5: Rust の nom パーサーコンビネータ

```rust
use nom::{
    IResult,
    bytes::complete::{tag, take_while1},
    character::complete::{char, digit1, space0},
    combinator::{map, map_res},
    multi::separated_list1,
    sequence::{delimited, preceded, tuple},
    branch::alt,
};

// 整数パーサー
fn integer(input: &str) -> IResult<&str, i64> {
    map_res(digit1, |s: &str| s.parse::<i64>())(input)
}

// カンマ区切りの整数リスト
fn integer_list(input: &str) -> IResult<&str, Vec<i64>> {
    separated_list1(
        delimited(space0, char(','), space0),
        integer,
    )(input)
}

// "[1, 2, 3]" のパース
fn bracketed_list(input: &str) -> IResult<&str, Vec<i64>> {
    delimited(
        char('['),
        delimited(space0, integer_list, space0),
        char(']'),
    )(input)
}

fn main() {
    let (remaining, result) = bracketed_list("[1, 2, 3, 42]").unwrap();
    assert_eq!(result, vec![1, 2, 3, 42]);
    assert_eq!(remaining, "");
}
```

---

## 手法比較表

| 手法 | 表現力 | 性能 | 学習コスト | ユースケース |
|---|---|---|---|---|
| **正規表現** | 正規文法 | 高速 | 低 | パターンマッチ、検索・置換 |
| **パーサーコンビネータ** | 文脈自由文法 | 中〜高 | 中 | DSL、設定ファイル、プロトコル |
| **PEG** | 文脈自由文法+ | 高 | 中 | 言語処理、構文解析 |
| **ANTLR/yacc** | 文脈自由文法 | 高 | 高 | プログラミング言語、SQL |
| **tree-sitter** | 文脈自由文法 | 非常に高 | 中 | エディタのシンタックスハイライト |
| **専用パーサー** | 任意 | 最高 | 低 | JSON、HTML、CSV 等の標準形式 |

### 選択指針比較表

| 要件 | 推奨手法 |
|---|---|
| 単純なパターンマッチ | 正規表現 |
| 標準フォーマット（JSON/HTML/CSV） | 専用パーサーライブラリ |
| カスタム DSL の設計 | パーサーコンビネータ or PEG |
| プログラミング言語の解析 | ANTLR / tree-sitter |
| エディタ統合（ハイライト等） | tree-sitter |
| 高性能なバイナリプロトコル | nom (Rust) / 手書きパーサー |

---

## アンチパターン

### 1. HTML を正規表現でパース

**問題**: HTML はネスト構造を持つため正規文法では表現できない。タグの属性値にクォートが含まれるケースや、自己終了タグなど、正規表現では処理できないエッジケースが無数に存在する。

**対策**: BeautifulSoup、cheerio、lxml 等の専用パーサーを使用する。

### 2. パーサーコンビネータで全て解決しようとする

**問題**: JSON、CSV、YAML 等の標準フォーマットに対して独自パーサーを書くと、エッジケースの対応漏れやセキュリティ問題が発生する。

**対策**: 標準フォーマットには実績のあるライブラリを使用する。パーサーコンビネータはカスタム DSL や独自プロトコルに限定して使用する。

---

## FAQ

### Q1: パーサーコンビネータの性能は正規表現と比べてどうですか？

**A**: 単純なパターンマッチでは正規表現の方が高速です。ただし、正規表現でバックトラッキングが多発する複雑なパターンでは、PEG やパーサーコンビネータの方が予測可能な性能を発揮します。nom（Rust）は正規表現と同等の性能を達成する場合もあります。

### Q2: tree-sitter はエディタ以外でも使えますか？

**A**: はい。tree-sitter はインクリメンタルパーサーとして、コード解析ツール、リンター、コード変換ツールでも活用できます。GitHub のコード検索やセキュリティスキャンでも使用されています。

### Q3: どの言語のパーサーコンビネータライブラリが最も成熟していますか？

**A**: Haskell の `parsec`/`megaparsec` が最も理論的に洗練されています。実用面では Rust の `nom`/`winnow`（高性能）、Python の `pyparsing`（読みやすさ）、JavaScript の `Peggy`（Web 統合）がそれぞれの領域で成熟しています。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 正規表現の限界 | ネスト構造・再帰・文脈依存は扱えない |
| パーサーコンビネータ | 小さなパーサーを合成して複雑な文法を解析 |
| PEG | 文法定義からパーサーを自動生成。優先順位付き選択 |
| nom (Rust) | ゼロコピーの高性能パーサーコンビネータ |
| tree-sitter | インクリメンタルパーサー。エディタ統合に最適 |
| 選択基準 | 標準形式には専用ライブラリ、カスタム文法にはコンビネータ/PEG |

## 次に読むべきガイド

- [正規表現基礎](../01-basics/00-regex-syntax.md) — 正規表現が適切な場面での活用法
- [テキスト処理実践](../02-practical/01-text-processing.md) — 実務でのテキスト処理パターン

## 参考文献

1. **Bryan Ford**: [Parsing Expression Grammars (2004)](https://bford.info/pub/lang/peg.pdf) — PEG の原論文
2. **nom 公式**: [nom Documentation](https://docs.rs/nom/latest/nom/) — Rust パーサーコンビネータの包括的ドキュメント
3. **tree-sitter 公式**: [tree-sitter](https://tree-sitter.github.io/tree-sitter/) — インクリメンタルパーサーフレームワーク
