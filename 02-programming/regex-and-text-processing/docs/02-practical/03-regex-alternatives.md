# 正規表現の代替

> パーサーコンビネータ、PEG、構造化テキスト処理など、正規表現では困難なテキスト解析タスクの代替手法を習得する

## この章で学ぶこと

1. **正規表現の限界** -- 再帰構造、ネスト、文脈依存文法が扱えない理由
2. **パーサーコンビネータ** -- 小さなパーサーを組み合わせて複雑な文法を解析する手法
3. **PEG と実用ツール** -- Parsing Expression Grammar の特性と tree-sitter 等のツール
4. **ANTLR と yacc/bison** -- パーサージェネレータによる本格的な言語処理
5. **構造化データの専用パーサー** -- JSON, HTML, XML, YAML 等の実務的な処理手法
6. **tree-sitter の活用** -- インクリメンタルパーサーの実践的な利用法

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

### 1.1 正規表現の限界を示す具体例

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

### 1.2 正規表現が失敗する典型的なケース

```python
import re

# ケース1: ネストした括弧のマッチング
# 正規表現では対応する括弧のペアを認識できない
text = "f(g(x, y), h(z))"
# 「f の引数全体」を正しく抽出することは不可能
# 以下は誤った結果になる
match = re.search(r'f\((.+)\)', text)
# match.group(1) = "g(x, y), h(z)" -- たまたま正しいが以下は失敗
text2 = "f(g(x), y) + f(a)"
match2 = re.search(r'f\((.+)\)', text2)
# match2.group(1) = "g(x), y) + f(a" -- 壊れる（貪欲マッチ）

# ケース2: 文字列リテラル内のエスケープ処理
code = 'print("He said \\"hello\\"", end="")'
# 正規表現ではエスケープされた引用符を含む文字列の正確な抽出が困難
# 以下は不正確
strings = re.findall(r'"([^"]*)"', code)
# エスケープされた \" を正しく処理できない

# ケース3: コメント内のコード風テキスト
code2 = """
# print("this is a comment")
print("this is real code")
"""
# コメント行か実行行かを正規表現だけで判断するのは
# 言語構文の理解が必要で困難

# ケース4: ヒアドキュメントやテンプレートリテラル
ruby_code = '''
text = <<~HEREDOC
  This contains "quotes" and #{interpolation}
  And even regex: /pattern/
HEREDOC
'''
# ヒアドキュメントの開始と終了を正しく追跡するには
# ステートマシンが必要
```

### 1.3 ポンピングレンマ -- 正規言語の限界の数学的証明

```
ポンピングレンマ（Pumping Lemma）:
=================================

定理: 言語 L が正規言語であれば、ある定数 p が存在し、
     |w| >= p なる任意の文字列 w ∈ L に対して、
     w = xyz と分解でき、以下が成立する:
       1. |y| > 0
       2. |xy| <= p
       3. 任意の i >= 0 で xy^iz ∈ L

反例: L = { a^n b^n | n >= 0 } は正規言語でない
     「a が n 個、b が n 個」を正規表現では表現できない

実用上の意味:
  - 「対応する括弧」は正規言語ではない
  - 「対応するタグ」は正規言語ではない
  - これらを正規表現で完全に処理しようとするのは理論的に不可能

ただし注意:
  - Perl/PCRE の「拡張」正規表現は理論的な正規言語を超える
  - (?R) 等の再帰パターンで一部の文脈自由言語を扱える
  - しかし可読性・保守性の観点からパーサーを使うべき
```

### 1.4 PCRE の再帰パターン -- 正規表現の拡張

```python
import regex  # Python の regex モジュール（re の拡張）

# PCRE の再帰パターンで対応する括弧をマッチ
# (?R) は全体パターンの再帰呼び出し
pattern = r'\((?:[^()]*|(?R))*\)'

text = "f(g(x, y), h(z))"
matches = regex.findall(pattern, text)
# matches = ['(g(x, y), h(z))', '(x, y)', '(z)']

# 名前付きグループの再帰
# (?P<name>...) と (?&name) を使用
pattern2 = r'(?P<brackets>\{(?:[^{}]*|(?&brackets))*\})'
json_like = '{"a": {"b": {"c": 1}}}'
match = regex.search(pattern2, json_like)
# マッチ: {"a": {"b": {"c": 1}}}

# 注意: これは「可能」であって「推奨」ではない
# 可読性・保守性の観点から、複雑な再帰パターンには
# パーサーコンビネータや PEG を使うべき
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

### 2.1 TypeScript でのパーサーコンビネータ

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

### 2.2 TypeScript パーサーコンビネータの拡張: エラーレポート

```typescript
// より実用的なパーサーコンビネータ: エラー位置と期待値のレポート

interface ParseError {
  pos: number;
  line: number;
  column: number;
  expected: string[];
  found: string;
}

type BetterParser<T> = (input: string, pos: number) => BetterParseResult<T>;
type BetterParseResult<T> =
  | { success: true; value: T; pos: number }
  | { success: false; error: ParseError };

// エラー位置から行と列を計算
function getLineAndColumn(input: string, pos: number): { line: number; column: number } {
  const lines = input.slice(0, pos).split('\n');
  return { line: lines.length, column: lines[lines.length - 1].length + 1 };
}

// エラーメッセージの生成
function formatError(input: string, error: ParseError): string {
  const lines = input.split('\n');
  const line = lines[error.line - 1] || '';
  const pointer = ' '.repeat(error.column - 1) + '^';
  return [
    `Parse error at line ${error.line}, column ${error.column}:`,
    `  ${line}`,
    `  ${pointer}`,
    `Expected: ${error.expected.join(' or ')}`,
    `Found: ${error.found || 'end of input'}`,
  ].join('\n');
}

// 実用例: 設定ファイルパーサー
// key = value 形式の設定ファイルをパースする
function configParser(input: string): Map<string, string> {
  const result = new Map<string, string>();
  const lines = input.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === '' || line.startsWith('#')) continue;

    const keyParser = regex(/[a-zA-Z_][a-zA-Z0-9_]*/);
    const r = keyParser(line, 0);
    if (!r.success) {
      const { line: ln, column } = getLineAndColumn(input, i);
      throw new Error(`Invalid key at line ${ln + 1}`);
    }

    const key = r.value;
    const eqParser = seq(ws, char('='));
    const eq = eqParser(line, r.pos);
    if (!eq.success) {
      throw new Error(`Expected '=' after key '${key}' at line ${i + 1}`);
    }

    const valueStart = eq.success ? eq.pos : r.pos;
    const wsResult = ws(line, valueStart);
    const value = line.slice(wsResult.success ? wsResult.pos : valueStart).trim();
    result.set(key, value);
  }

  return result;
}
```

### 2.3 Haskell のパーサーコンビネータ (Parsec / Megaparsec)

```haskell
-- Megaparsec は Haskell の最も成熟したパーサーコンビネータライブラリ
-- 正規表現では不可能な文脈自由文法の解析を型安全に実現

import Text.Megaparsec
import Text.Megaparsec.Char
import qualified Text.Megaparsec.Char.Lexer as L
import Data.Void (Void)

type Parser = Parsec Void String

-- 空白とコメントのスキップ
sc :: Parser ()
sc = L.space space1 (L.skipLineComment "//") (L.skipBlockComment "/*" "*/")

-- レキサーヘルパー
lexeme :: Parser a -> Parser a
lexeme = L.lexeme sc

symbol :: String -> Parser String
symbol = L.symbol sc

-- 整数リテラル
integer :: Parser Integer
integer = lexeme L.decimal

-- 識別子
identifier :: Parser String
identifier = lexeme $ do
  first <- letterChar
  rest  <- many (alphaNumChar <|> char '_')
  return (first : rest)

-- JSON パーサーの例
data JsonValue
  = JsonNull
  | JsonBool Bool
  | JsonNumber Double
  | JsonString String
  | JsonArray [JsonValue]
  | JsonObject [(String, JsonValue)]
  deriving (Show)

jsonValue :: Parser JsonValue
jsonValue = sc *> choice
  [ JsonNull   <$ symbol "null"
  , JsonBool True  <$ symbol "true"
  , JsonBool False <$ symbol "false"
  , JsonNumber <$> lexeme L.float
  , JsonString <$> stringLiteral
  , JsonArray  <$> brackets (jsonValue `sepBy` symbol ",")
  , JsonObject <$> braces (keyValue `sepBy` symbol ",")
  ]
  where
    stringLiteral = lexeme $ char '"' *> manyTill L.charLiteral (char '"')
    brackets = between (symbol "[") (symbol "]")
    braces   = between (symbol "{") (symbol "}")
    keyValue = do
      key <- stringLiteral
      _   <- symbol ":"
      val <- jsonValue
      return (key, val)

-- 使用例
-- parse jsonValue "" "{\"name\": \"Alice\", \"scores\": [95, 87, 92]}"
-- Right (JsonObject [("name", JsonString "Alice"), ("scores", JsonArray [...])])
```

### 2.4 Python のパーサーコンビネータ (lark)

```python
from lark import Lark, Transformer, v_args

# lark は Python で最も使いやすいパーサーライブラリの一つ
# EBNF 風の文法定義からパーサーを自動生成する

# 四則演算の文法定義
calc_grammar = """
    ?start: expr

    ?expr: term
        | expr "+" term   -> add
        | expr "-" term   -> sub

    ?term: factor
        | term "*" factor -> mul
        | term "/" factor -> div

    ?factor: NUMBER       -> number
        | "-" factor      -> neg
        | "(" expr ")"

    %import common.NUMBER
    %import common.WS
    %ignore WS
"""

# AST を計算結果に変換するトランスフォーマー
@v_args(inline=True)
class CalcTransformer(Transformer):
    from operator import add, sub, mul, truediv as div

    def number(self, n):
        return float(n)

    def neg(self, n):
        return -n

# パーサーの生成と使用
calc_parser = Lark(calc_grammar, parser='lalr', transformer=CalcTransformer())

# 計算の実行
result = calc_parser.parse("(1 + 2) * 3 - 4 / 2")
print(result)  # 7.0

# より複雑な例: SQL の SELECT 文パーサー
sql_grammar = """
    start: select_stmt

    select_stmt: "SELECT"i column_list "FROM"i table_name where_clause?
                 order_clause? limit_clause?

    column_list: "*" | column ("," column)*
    column: IDENTIFIER ("." IDENTIFIER)? alias?
    alias: "AS"i IDENTIFIER

    table_name: IDENTIFIER alias?

    where_clause: "WHERE"i condition
    condition: comparison (("AND"i | "OR"i) comparison)*
    comparison: column_ref operator value
    column_ref: IDENTIFIER ("." IDENTIFIER)?
    operator: "=" | "!=" | "<" | ">" | "<=" | ">=" | "LIKE"i | "IN"i
    value: STRING | NUMBER | "NULL"i | "(" value ("," value)* ")"

    order_clause: "ORDER"i "BY"i order_item ("," order_item)*
    order_item: column_ref ("ASC"i | "DESC"i)?

    limit_clause: "LIMIT"i NUMBER ("OFFSET"i NUMBER)?

    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
    STRING: "'" /[^']*/ "'"
    NUMBER: /[0-9]+(\.[0-9]+)?/

    %import common.WS
    %ignore WS
"""

sql_parser = Lark(sql_grammar, parser='earley')
tree = sql_parser.parse("SELECT name, age FROM users WHERE status = 'active' ORDER BY age DESC LIMIT 10")
print(tree.pretty())
```

---

## 3. PEG（Parsing Expression Grammar）

### 3.1 PEG.js / Peggy による文法定義

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

### 3.2 PEG の特性と正規表現との違い

```
PEG vs 正規表現 vs CFG:
========================

正規表現:
  - 選択は「最長マッチ」か「最短マッチ」
  - バックトラッキングによる指数的な実行時間のリスク
  - 再帰が不可能（PCRE 拡張を除く）

PEG:
  - 選択は「優先順位付き」（最初にマッチした選択肢を採用）
  - 曖昧性がない（文法が一意のパースツリーを生成）
  - 再帰が可能（文脈自由文法を扱える）
  - packrat パーサーで線形時間を保証可能

CFG (Context-Free Grammar):
  - 選択は「非決定的」（複数のパースツリーの可能性）
  - 曖昧性を含む可能性がある
  - LR, LL, Earley 等のアルゴリズムで解析

PEG の選択演算子 "/" は「順序付き選択」:
  rule = A / B
  → まず A を試す
  → A が成功 → A の結果を採用（B は試さない）
  → A が失敗 → B を試す

これにより:
  - 文法の曖昧性が排除される
  - パーサーの動作が予測可能
  - ただし「見落とし」のリスク（順序が重要）
```

### 3.3 PEG の実践的な文法定義パターン

```javascript
// Peggy での実践的な文法パターン集

// パターン1: 設定ファイルパーサー (INI 形式)
// ファイル: ini_parser.pegjs

IniFile
  = sections:Section* { return Object.fromEntries(sections); }

Section
  = _ "[" name:SectionName "]" _ "\n" entries:Entry*
    { return [name, Object.fromEntries(entries)]; }

SectionName
  = chars:[a-zA-Z0-9._-]+ { return chars.join(""); }

Entry
  = _ key:Key _ "=" _ value:Value _ "\n"?
    { return [key, value]; }
  / Comment { return null; }

Key
  = chars:[a-zA-Z0-9._-]+ { return chars.join(""); }

Value
  = QuotedString / UnquotedValue

QuotedString
  = '"' chars:[^"]* '"' { return chars.join(""); }

UnquotedValue
  = chars:[^\n#;]* { return chars.join("").trim(); }

Comment
  = _ [#;] [^\n]* "\n"?

_ = [ \t]*

// パターン2: Markdown のインラインフォーマットパーサー
// ファイル: markdown_inline.pegjs

InlineContent
  = elements:InlineElement* { return elements; }

InlineElement
  = Bold / Italic / Code / Link / Text

Bold
  = "**" content:$[^*]+ "**"
    { return { type: "bold", content }; }

Italic
  = "*" content:$[^*]+ "*"
    { return { type: "italic", content }; }

Code
  = "`" content:$[^`]+ "`"
    { return { type: "code", content }; }

Link
  = "[" text:$[^\]]+ "]" "(" url:$[^)]+ ")"
    { return { type: "link", text, url }; }

Text
  = chars:$[^*`\[]+ { return { type: "text", content: chars }; }

// パターン3: URL パーサー
// ファイル: url_parser.pegjs

URL
  = scheme:Scheme "://" authority:Authority path:Path? query:Query? fragment:Fragment?
    { return { scheme, ...authority, path: path || "/", query, fragment }; }

Scheme
  = chars:[a-zA-Z]+ { return chars.join(""); }

Authority
  = userinfo:(Userinfo "@")? host:Host port:(":" Port)?
    { return { userinfo: userinfo?.[0], host, port: port?.[1] }; }

Userinfo
  = chars:[a-zA-Z0-9._~!$&'()*+,;=:-]+ { return chars.join(""); }

Host
  = chars:[a-zA-Z0-9.-]+ { return chars.join(""); }

Port
  = digits:[0-9]+ { return parseInt(digits.join(""), 10); }

Path
  = segments:("/" PathSegment)* { return segments.map(s => "/" + s[1]).join(""); }

PathSegment
  = chars:[a-zA-Z0-9._~!$&'()*+,;=:@-]* { return chars.join(""); }

Query
  = "?" params:QueryParam* { return Object.fromEntries(params); }

QueryParam
  = key:$[^=&#]+ "=" value:$[^&#]* "&"?
    { return [decodeURIComponent(key), decodeURIComponent(value)]; }

Fragment
  = "#" chars:$[a-zA-Z0-9._~!$&'()*+,;=:@/?-]* { return chars; }
```

---

## 4. 実用的な代替ツール

### 4.1 Python の pyparsing

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

### 4.2 Rust の nom パーサーコンビネータ

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

### 4.3 Rust の nom -- 実践的なログパーサー

```rust
use nom::{
    IResult,
    bytes::complete::{tag, take_while, take_until, take_while1},
    character::complete::{char, digit1, space0, space1},
    combinator::{map, map_res, opt},
    sequence::{delimited, tuple, preceded},
    branch::alt,
};
use std::net::Ipv4Addr;

// Apache ログの1行をパースする構造体
#[derive(Debug)]
struct AccessLogEntry {
    ip: Ipv4Addr,
    timestamp: String,
    method: String,
    path: String,
    protocol: String,
    status: u16,
    size: u64,
}

// IP アドレスパーサー
fn ip_address(input: &str) -> IResult<&str, Ipv4Addr> {
    map_res(
        take_while1(|c: char| c.is_ascii_digit() || c == '.'),
        |s: &str| s.parse::<Ipv4Addr>(),
    )(input)
}

// タイムスタンプパーサー: [10/Oct/2000:13:55:36 -0700]
fn timestamp(input: &str) -> IResult<&str, String> {
    map(
        delimited(char('['), take_until("]"), char(']')),
        |s: &str| s.to_string(),
    )(input)
}

// リクエスト行パーサー: "GET /path HTTP/1.1"
fn request_line(input: &str) -> IResult<&str, (String, String, String)> {
    let (input, _) = char('"')(input)?;
    let (input, method) = take_while1(|c: char| c.is_ascii_alphabetic())(input)?;
    let (input, _) = space1(input)?;
    let (input, path) = take_while1(|c: char| c != ' ')(input)?;
    let (input, _) = space1(input)?;
    let (input, protocol) = take_until("\"")(input)?;
    let (input, _) = char('"')(input)?;
    Ok((input, (method.to_string(), path.to_string(), protocol.to_string())))
}

// ステータスコードパーサー
fn status_code(input: &str) -> IResult<&str, u16> {
    map_res(digit1, |s: &str| s.parse::<u16>())(input)
}

// ログ行全体のパーサー
fn log_entry(input: &str) -> IResult<&str, AccessLogEntry> {
    let (input, ip) = ip_address(input)?;
    let (input, _) = tag(" - - ")(input)?;
    let (input, ts) = timestamp(input)?;
    let (input, _) = space1(input)?;
    let (input, (method, path, protocol)) = request_line(input)?;
    let (input, _) = space1(input)?;
    let (input, status) = status_code(input)?;
    let (input, _) = space1(input)?;
    let (input, size) = map_res(digit1, |s: &str| s.parse::<u64>())(input)?;

    Ok((input, AccessLogEntry {
        ip, timestamp: ts, method, path, protocol, status, size,
    }))
}

// 使用例
fn main() {
    let line = r#"192.168.1.1 - - [11/Feb/2026:10:30:45 +0900] "GET /api/users HTTP/1.1" 200 1234"#;
    match log_entry(line) {
        Ok((_, entry)) => {
            println!("IP: {}", entry.ip);
            println!("Status: {}", entry.status);
            println!("Path: {}", entry.path);
        }
        Err(e) => eprintln!("Parse error: {:?}", e),
    }
}
```

### 4.4 Go の participle パーサーライブラリ

```go
package main

import (
    "fmt"
    "github.com/alecthomas/participle/v2"
    "github.com/alecthomas/participle/v2/lexer"
)

// Go の participle は構造体のタグからパーサーを自動生成する
// 正規表現では不可能な再帰構造を型安全にパースできる

// 四則演算の AST 定義
type Expression struct {
    Left  *Term   `@@`
    Op    string  `@("+" | "-")?`
    Right *Term   `@@?`
}

type Term struct {
    Left  *Factor `@@`
    Op    string  `@("*" | "/")?`
    Right *Factor `@@?`
}

type Factor struct {
    Number *float64    `  @Float | @Int`
    Sub    *Expression `| "(" @@ ")"`
}

func main() {
    parser := participle.MustBuild[Expression]()

    expr := &Expression{}
    err := parser.ParseString("", "1 + 2 * (3 - 4)", expr)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Parsed: %+v\n", expr)
}

// 設定ファイルパーサーの例
type Config struct {
    Sections []*Section `@@*`
}

type Section struct {
    Name    string   `"[" @Ident "]"`
    Entries []*Entry `@@*`
}

type Entry struct {
    Key   string `@Ident "="`
    Value string `@(String | Ident | Int)`
}

func parseConfig(input string) (*Config, error) {
    configLexer := lexer.MustSimple([]lexer.SimpleRule{
        {Name: "String", Pattern: `"[^"]*"`},
        {Name: "Ident", Pattern: `[a-zA-Z_][a-zA-Z0-9_]*`},
        {Name: "Int", Pattern: `[0-9]+`},
        {Name: "Punct", Pattern: `[\[\]=]`},
        {Name: "whitespace", Pattern: `[\s]+`},
        {Name: "comment", Pattern: `#[^\n]*`},
    })

    parser := participle.MustBuild[Config](
        participle.Lexer(configLexer),
    )

    config := &Config{}
    err := parser.ParseString("", input, config)
    return config, err
}
```

---

## 5. ANTLR -- パーサージェネレータ

### 5.1 ANTLR の文法定義と使用法

```antlr
// Calculator.g4 -- ANTLR 文法ファイル
grammar Calculator;

// パーサールール
prog: stat+ ;

stat: expr NEWLINE          # printExpr
    | ID '=' expr NEWLINE   # assign
    | NEWLINE               # blank
    ;

expr: expr op=('*'|'/') expr   # MulDiv
    | expr op=('+'|'-') expr   # AddSub
    | INT                      # int
    | ID                       # id
    | '(' expr ')'             # parens
    ;

// レキサールール
MUL : '*' ;
DIV : '/' ;
ADD : '+' ;
SUB : '-' ;
ID  : [a-zA-Z]+ ;
INT : [0-9]+ ;
NEWLINE : '\r'? '\n' ;
WS  : [ \t]+ -> skip ;
```

```java
// ANTLR で生成されたパーサーの使用例（Java）
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;

public class CalcApp {
    public static void main(String[] args) throws Exception {
        // 入力ストリームの準備
        CharStream input = CharStreams.fromString("x = 1 + 2 * 3\n");

        // レキサー → トークンストリーム → パーサー
        CalculatorLexer lexer = new CalculatorLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CalculatorParser parser = new CalculatorParser(tokens);

        // パースツリーの取得
        ParseTree tree = parser.prog();

        // ビジターパターンで AST を走査
        CalcVisitor visitor = new CalcVisitor();
        visitor.visit(tree);
    }
}

// ビジターの実装
class CalcVisitor extends CalculatorBaseVisitor<Integer> {
    Map<String, Integer> memory = new HashMap<>();

    @Override
    public Integer visitAssign(CalculatorParser.AssignContext ctx) {
        String id = ctx.ID().getText();
        int value = visit(ctx.expr());
        memory.put(id, value);
        return value;
    }

    @Override
    public Integer visitMulDiv(CalculatorParser.MulDivContext ctx) {
        int left = visit(ctx.expr(0));
        int right = visit(ctx.expr(1));
        if (ctx.op.getType() == CalculatorParser.MUL) return left * right;
        return left / right;
    }

    @Override
    public Integer visitAddSub(CalculatorParser.AddSubContext ctx) {
        int left = visit(ctx.expr(0));
        int right = visit(ctx.expr(1));
        if (ctx.op.getType() == CalculatorParser.ADD) return left + right;
        return left - right;
    }
}
```

```bash
# ANTLR の使い方
# 1. 文法ファイルの作成
# 2. パーサーの生成
antlr4 Calculator.g4 -Dlanguage=Python3  # Python 向け
antlr4 Calculator.g4 -Dlanguage=Java     # Java 向け
antlr4 Calculator.g4 -Dlanguage=Go       # Go 向け

# 3. grun でテスト（GUI でパースツリーを可視化）
grun Calculator prog -gui
```

---

## 6. tree-sitter -- インクリメンタルパーサー

### 6.1 tree-sitter の概要と活用法

```
tree-sitter の特徴:
====================

1. インクリメンタルパース
   - 変更された部分のみを再パースする
   - エディタでのリアルタイム構文解析に最適
   - O(log n) の編集後再パース

2. エラー回復
   - 構文エラーがあっても可能な限りパースを続行
   - エディタが壊れたコードでもハイライトを維持

3. 多言語対応
   - 200以上のプログラミング言語の文法が利用可能
   - JavaScript, Python, Rust, Go, Java, C/C++, ...

4. クエリシステム
   - S式によるパターンマッチング
   - コードのセマンティックな検索が可能

利用場面:
  - エディタのシンタックスハイライト（Neovim, Helix, Zed 等）
  - コードナビゲーション（定義/参照ジャンプ）
  - リンター・フォーマッターの実装
  - コード変換ツール
  - GitHub のコード検索・セキュリティスキャン
```

### 6.2 tree-sitter のクエリシステム

```scheme
;; tree-sitter のクエリ: S 式でコード構造をマッチング

;; Python の関数定義を全てマッチ
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  body: (block) @function.body)

;; クラス定義内のメソッドのみをマッチ
(class_definition
  name: (identifier) @class.name
  body: (block
    (function_definition
      name: (identifier) @method.name)))

;; import 文のマッチ
(import_statement
  name: (dotted_name) @import.module)

(import_from_statement
  module_name: (dotted_name) @import.from
  name: (dotted_name) @import.name)

;; 特定のパターンを持つ関数呼び出し
;; 例: logging.error("...") のような呼び出し
(call
  function: (attribute
    object: (identifier) @object (#eq? @object "logging")
    attribute: (identifier) @method (#eq? @method "error"))
  arguments: (argument_list
    (string) @message))

;; TypeScript の型定義をマッチ
(type_alias_declaration
  name: (type_identifier) @type.name
  value: (_) @type.definition)

(interface_declaration
  name: (type_identifier) @interface.name
  body: (object_type) @interface.body)
```

### 6.3 tree-sitter を使ったコード解析ツール (Python)

```python
# tree-sitter の Python バインディングを使ったコード解析

from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# パーサーのセットアップ
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

# ソースコードのパース
source_code = b"""
class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int) -> dict:
        query = "SELECT * FROM users WHERE id = ?"
        return self.db.execute(query, (user_id,))

    def create_user(self, name: str, email: str) -> int:
        query = "INSERT INTO users (name, email) VALUES (?, ?)"
        return self.db.execute(query, (name, email))

def helper_function():
    pass
"""

tree = parser.parse(source_code)
root_node = tree.root_node

# 関数定義の抽出
def find_functions(node, depth=0):
    """再帰的に関数定義を探索"""
    if node.type == 'function_definition':
        name_node = node.child_by_field_name('name')
        params_node = node.child_by_field_name('parameters')
        return_type = node.child_by_field_name('return_type')

        info = {
            'name': name_node.text.decode(),
            'params': params_node.text.decode(),
            'return_type': return_type.text.decode() if return_type else None,
            'line': node.start_point[0] + 1,
            'depth': depth,
        }
        print(f"{'  ' * depth}関数: {info['name']}{info['params']}"
              f"{' -> ' + info['return_type'] if info['return_type'] else ''}"
              f" (行 {info['line']})")

    for child in node.children:
        find_functions(child, depth + (1 if node.type == 'class_definition' else 0))

find_functions(root_node)

# クラス定義の抽出
def find_classes(node):
    """クラス定義とそのメソッドを抽出"""
    if node.type == 'class_definition':
        name = node.child_by_field_name('name').text.decode()
        methods = []
        body = node.child_by_field_name('body')
        if body:
            for child in body.children:
                if child.type == 'function_definition':
                    method_name = child.child_by_field_name('name').text.decode()
                    methods.append(method_name)
        print(f"クラス: {name}")
        print(f"  メソッド: {', '.join(methods)}")

    for child in node.children:
        find_classes(child)

find_classes(root_node)

# SQL インジェクション脆弱性の検出（簡易版）
def find_sql_injection_risks(node):
    """文字列フォーマットで SQL を構築している箇所を検出"""
    if node.type == 'binary_operator':
        # 文字列連結（+）による SQL 構築を検出
        left = node.children[0]
        if left.type == 'string' and any(
            keyword in left.text.decode().upper()
            for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
        ):
            print(f"警告: SQL文字列連結 (行 {node.start_point[0] + 1})")
            print(f"  {node.text.decode()}")

    if node.type == 'call':
        # f-string や .format() による SQL 構築を検出
        func = node.child_by_field_name('function')
        if func and func.type == 'attribute' and func.text.decode().endswith('.format'):
            # .format() の呼び出し元が SQL を含むか確認
            pass

    for child in node.children:
        find_sql_injection_risks(child)

find_sql_injection_risks(root_node)
```

---

## 7. 構造化データの専用パーサー

### 7.1 HTML 処理 -- Beautiful Soup と lxml

```python
from bs4 import BeautifulSoup
import lxml.html

# === Beautiful Soup ===
html = """
<html>
<body>
  <div class="container">
    <h1>タイトル</h1>
    <ul class="items">
      <li class="item active">項目1</li>
      <li class="item">項目2</li>
      <li class="item">項目3</li>
    </ul>
    <div class="nested">
      <div class="deep">
        <p>深くネストされたテキスト</p>
      </div>
    </div>
  </div>
</body>
</html>
"""

soup = BeautifulSoup(html, 'html.parser')

# CSS セレクタで検索（正規表現よりはるかに信頼性が高い）
items = soup.select('ul.items li.item')
for item in items:
    print(item.text)  # "項目1", "項目2", "項目3"

# ネストされた要素の取得
deep_text = soup.select_one('.nested .deep p').text
print(deep_text)  # "深くネストされたテキスト"

# 属性でフィルタ
active = soup.find('li', class_='active')
print(active.text)  # "項目1"

# 正規表現では不可能なケース: 自己閉じタグ、属性値の引用符等
complex_html = '<img src="photo.jpg" alt="He said &quot;hello&quot;" />'
soup2 = BeautifulSoup(complex_html, 'html.parser')
img = soup2.find('img')
print(img['alt'])  # 'He said "hello"' -- エンティティも正しくデコード

# === lxml (XPath) ===
doc = lxml.html.fromstring(html)

# XPath で検索
titles = doc.xpath('//h1/text()')
print(titles)  # ['タイトル']

items_xpath = doc.xpath('//ul[@class="items"]/li/text()')
print(items_xpath)  # ['項目1', '項目2', '項目3']

# 複雑な条件
active_xpath = doc.xpath('//li[contains(@class, "active")]/text()')
print(active_xpath)  # ['項目1']
```

### 7.2 JSON 処理 -- jq と Python

```bash
# jq: コマンドラインの JSON プロセッサ
# 正規表現では不可能なネストした JSON の処理

# 基本的なフィールド抽出
echo '{"name": "Alice", "age": 30}' | jq '.name'
# "Alice"

# ネストしたフィールド
echo '{"user": {"name": "Alice", "address": {"city": "Tokyo"}}}' | \
    jq '.user.address.city'
# "Tokyo"

# 配列の処理
echo '[{"name": "Alice"}, {"name": "Bob"}]' | jq '.[].name'
# "Alice"
# "Bob"

# フィルタリング
echo '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]' | \
    jq '.[] | select(.age > 28)'
# {"name": "Alice", "age": 30}

# 変換
echo '[{"name": "Alice", "scores": [90, 85, 92]}]' | \
    jq '.[] | {name, avg_score: (.scores | add / length)}'
# {"name": "Alice", "avg_score": 89}

# JSON Lines のストリーム処理
cat events.jsonl | jq -c 'select(.level == "ERROR") | {timestamp, message}'

# CSV への変換
echo '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]' | \
    jq -r '.[] | [.name, .age] | @csv'
# "Alice",30
# "Bob",25
```

```python
# Python での JSON 処理
import json
from pathlib import Path

# JSON ファイルの読み込みと変換
with open('data.json') as f:
    data = json.load(f)

# ネストしたデータの安全なアクセス
def safe_get(data, *keys, default=None):
    """ネストしたキーに安全にアクセスする"""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and isinstance(key, int):
            current = current[key] if key < len(current) else None
        else:
            return default
        if current is None:
            return default
    return current

# 使用例
config = {"database": {"host": "localhost", "port": 5432}}
host = safe_get(config, "database", "host")  # "localhost"
missing = safe_get(config, "database", "timeout", default=30)  # 30

# JSON スキーマによるバリデーション
from jsonschema import validate, ValidationError

schema = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "format": "email"},
    },
}

try:
    validate(instance={"name": "Alice", "age": 30}, schema=schema)
    print("バリデーション成功")
except ValidationError as e:
    print(f"バリデーション失敗: {e.message}")
```

### 7.3 YAML 処理

```python
import yaml
from pathlib import Path

# YAML パーサー（正規表現では処理不可能な構造）
yaml_content = """
server:
  host: localhost
  port: 8080
  ssl:
    enabled: true
    cert: /path/to/cert.pem

database:
  primary:
    host: db-primary.example.com
    port: 5432
    credentials:
      username: admin
      password: ${DB_PASSWORD}  # 環境変数の参照
  replicas:
    - host: db-replica-1.example.com
      port: 5432
    - host: db-replica-2.example.com
      port: 5432

logging:
  level: INFO
  handlers:
    - type: console
      format: "%(asctime)s [%(levelname)s] %(message)s"
    - type: file
      path: /var/log/app.log
      rotation: daily
"""

config = yaml.safe_load(yaml_content)

# ネストしたアクセス
print(config['server']['ssl']['enabled'])  # True
print(config['database']['replicas'][0]['host'])  # db-replica-1.example.com

# YAML のアンカーとエイリアス（正規表現では処理不可能）
yaml_with_anchors = """
defaults: &defaults
  adapter: postgres
  host: localhost
  port: 5432

development:
  <<: *defaults
  database: myapp_dev

production:
  <<: *defaults
  host: db.production.com
  database: myapp_prod
"""

envs = yaml.safe_load(yaml_with_anchors)
print(envs['production']['host'])  # db.production.com（オーバーライド）
print(envs['production']['port'])  # 5432（デフォルトから継承）
```

### 7.4 XML 処理

```python
import xml.etree.ElementTree as ET
from lxml import etree

# XML パーサー（名前空間やネストを正しく処理）
xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<bookstore xmlns:bk="http://example.com/books">
  <bk:book category="programming">
    <bk:title lang="ja">プログラミング入門</bk:title>
    <bk:author>山田太郎</bk:author>
    <bk:price currency="JPY">3000</bk:price>
  </bk:book>
  <bk:book category="science">
    <bk:title lang="en">Introduction to Physics</bk:title>
    <bk:author>John Smith</bk:author>
    <bk:price currency="USD">45</bk:price>
  </bk:book>
</bookstore>
"""

# ElementTree での処理
root = ET.fromstring(xml_content)
ns = {'bk': 'http://example.com/books'}

for book in root.findall('bk:book', ns):
    title = book.find('bk:title', ns).text
    author = book.find('bk:author', ns).text
    price = book.find('bk:price', ns)
    print(f"{title} by {author} - {price.get('currency')} {price.text}")

# lxml + XPath（より強力なクエリ）
doc = etree.fromstring(xml_content.encode())
nsmap = {'bk': 'http://example.com/books'}

# プログラミングカテゴリの本のタイトル
titles = doc.xpath('//bk:book[@category="programming"]/bk:title/text()', namespaces=nsmap)
print(titles)  # ['プログラミング入門']

# 日本語の本を検索
ja_books = doc.xpath('//bk:title[@lang="ja"]/text()', namespaces=nsmap)
print(ja_books)  # ['プログラミング入門']
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
| **lark (Python)** | 文脈自由文法 | 中 | 低〜中 | DSL、カスタム文法、プロトタイプ |
| **nom (Rust)** | 文脈自由文法 | 非常に高 | 中〜高 | バイナリプロトコル、高性能パーサー |

### 選択指針比較表

| 要件 | 推奨手法 |
|---|---|
| 単純なパターンマッチ | 正規表現 |
| 標準フォーマット（JSON/HTML/CSV） | 専用パーサーライブラリ |
| カスタム DSL の設計 | パーサーコンビネータ or PEG |
| プログラミング言語の解析 | ANTLR / tree-sitter |
| エディタ統合（ハイライト等） | tree-sitter |
| 高性能なバイナリプロトコル | nom (Rust) / 手書きパーサー |
| プロトタイプの高速開発 | lark (Python) / Peggy (JS) |
| 型安全な文法定義 | participle (Go) / nom (Rust) |

### 言語別の推奨パーサーライブラリ

| 言語 | ライブラリ | 特徴 |
|---|---|---|
| **Python** | lark | EBNF 風文法、Earley/LALR 対応 |
| **Python** | pyparsing | 直感的な API、学習コスト低 |
| **Rust** | nom | ゼロコピー、高性能 |
| **Rust** | winnow | nom の後継、エラーメッセージ改善 |
| **Haskell** | megaparsec | 最も理論的に洗練 |
| **TypeScript** | Peggy | PEG ベース、ブラウザ対応 |
| **Go** | participle | 構造体タグベース、型安全 |
| **Java** | ANTLR | 産業標準、ビジュアルデバッガ |
| **C/C++** | tree-sitter | インクリメンタル、エラー回復 |

---

## アンチパターン

### 1. HTML を正規表現でパース

**問題**: HTML はネスト構造を持つため正規文法では表現できない。タグの属性値にクォートが含まれるケースや、自己終了タグなど、正規表現では処理できないエッジケースが無数に存在する。

```python
# [NG] 正規表現で HTML をパース
import re
html = '<div class="outer"><div class="inner">text</div></div>'
# この正規表現はネストに対応できない
divs = re.findall(r'<div[^>]*>(.*?)</div>', html)
# 期待: ['text'] → 実際: ['<div class="inner">text'] (壊れている)

# [OK] Beautiful Soup を使う
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')
inner = soup.find('div', class_='inner')
print(inner.text)  # "text"
```

**対策**: BeautifulSoup、cheerio、lxml 等の専用パーサーを使用する。

### 2. パーサーコンビネータで全て解決しようとする

**問題**: JSON、CSV、YAML 等の標準フォーマットに対して独自パーサーを書くと、エッジケースの対応漏れやセキュリティ問題が発生する。

```python
# [NG] JSON パーサーを自作
def parse_json(text):
    # 文字列エスケープ、Unicode サロゲートペア、数値の精度、
    # BOM の処理、再帰の深さ制限... 全て自分で実装する必要がある
    pass

# [OK] 標準ライブラリの json モジュールを使う
import json
data = json.loads('{"key": "value"}')
```

**対策**: 標準フォーマットには実績のあるライブラリを使用する。パーサーコンビネータはカスタム DSL や独自プロトコルに限定して使用する。

### 3. 正規表現とパーサーの中間地点を無視する

**問題**: 「正規表現では足りないが、パーサーは大げさ」という場面で、どちらも選ばずに手書きのステートマシンを書いてしまう。

```python
# [NG] 手書きのステートマシン（保守困難）
def parse_key_value(text):
    state = 'KEY'
    key = ''
    value = ''
    results = {}
    for ch in text:
        if state == 'KEY':
            if ch == '=':
                state = 'VALUE'
            else:
                key += ch
        elif state == 'VALUE':
            if ch == '\n':
                results[key.strip()] = value.strip()
                key = value = ''
                state = 'KEY'
            else:
                value += ch
    if key:
        results[key.strip()] = value.strip()
    return results

# [OK] 正規表現で十分なケースもある
import re
def parse_key_value_regex(text):
    return dict(re.findall(r'([^=\n]+)=([^\n]*)', text))

# [OK] より複雑なら軽量パーサーを使う
from configparser import ConfigParser
parser = ConfigParser()
parser.read_string('[DEFAULT]\n' + text)
```

### 4. パフォーマンスを考慮せずにパーサーを選択する

**問題**: 大量データの処理に、パース速度の遅いライブラリを選択してしまう。

```python
# [注意] Beautiful Soup は便利だが大量 HTML には遅い
# 100万件の HTML フラグメントをパースする場合

# 遅い: Beautiful Soup (pure Python パーサー)
from bs4 import BeautifulSoup
for html_fragment in million_fragments:
    soup = BeautifulSoup(html_fragment, 'html.parser')  # 遅い

# 速い: lxml (C 実装)
from lxml import html
for html_fragment in million_fragments:
    doc = html.fromstring(html_fragment)  # 数倍高速

# さらに速い: 正規表現で事前フィルタ + パーサー
import re
pattern = re.compile(r'class="target"')
for html_fragment in million_fragments:
    if pattern.search(html_fragment):  # 正規表現で候補を絞る
        doc = html.fromstring(html_fragment)  # パーサーは候補のみ
```

---

## FAQ

### Q1: パーサーコンビネータの性能は正規表現と比べてどうですか？

**A**: 単純なパターンマッチでは正規表現の方が高速です。ただし、正規表現でバックトラッキングが多発する複雑なパターンでは、PEG やパーサーコンビネータの方が予測可能な性能を発揮します。nom（Rust）は正規表現と同等の性能を達成する場合もあります。

```
ベンチマーク目安（相対値）:
  正規表現（単純パターン）  : 1.0x
  正規表現（複雑パターン）  : 1.0x 〜 100x（バックトラック依存）
  nom (Rust)               : 1.0x 〜 2.0x
  Megaparsec (Haskell)     : 2.0x 〜 5.0x
  lark (Python, LALR)      : 5.0x 〜 20x
  pyparsing (Python)       : 10x 〜 50x
  PEG (Peggy, JavaScript)  : 3.0x 〜 10x
```

### Q2: tree-sitter はエディタ以外でも使えますか？

**A**: はい。tree-sitter はインクリメンタルパーサーとして、コード解析ツール、リンター、コード変換ツールでも活用できます。GitHub のコード検索やセキュリティスキャンでも使用されています。

具体的な活用例:

1. **GitHub Code Search**: tree-sitter でコードの構文構造を理解し、セマンティックな検索を実現
2. **Semgrep**: tree-sitter ベースの静的解析ツールで、セキュリティ脆弱性を検出
3. **Difftastic**: tree-sitter を使った構文認識 diff ツール
4. **Neovim / Helix**: tree-sitter ベースのシンタックスハイライト、折りたたみ、テキストオブジェクト

### Q3: どの言語のパーサーコンビネータライブラリが最も成熟していますか？

**A**: Haskell の `parsec`/`megaparsec` が最も理論的に洗練されています。実用面では Rust の `nom`/`winnow`（高性能）、Python の `pyparsing`（読みやすさ）、JavaScript の `Peggy`（Web 統合）がそれぞれの領域で成熟しています。

### Q4: 正規表現とパーサーの使い分けの具体的な判断基準は？

**A**: 以下のフローチャートで判断する:

```
対象は標準フォーマット（JSON/HTML/CSV/XML/YAML）か？
  → Yes: 専用パーサーライブラリを使う
  → No: ↓

ネスト構造や再帰がある か？
  → Yes: パーサーコンビネータ / PEG を使う
  → No: ↓

パターンが1行に収まるか？
  → Yes: 正規表現で十分
  → No: ↓

複数行にまたがるパターンか？
  → Yes: 正規表現の multiline / dotall フラグで対応可能なら正規表現
  → No: パーサーを検討

正規表現パターンが50文字を超えるか？
  → Yes: パーサーに切り替えることを強く推奨
  → No: 正規表現で OK（ただしコメント付きで）
```

### Q5: PEG と CFG（文脈自由文法）の違いは何ですか？

**A**: 主な違いは「選択」の扱い方です:

- **CFG の選択 (|)**: 非決定的。複数の選択肢が同時に考慮され、曖昧性が生じうる
- **PEG の選択 (/)**: 優先順位付き。最初にマッチした選択肢が採用される（曖昧性なし）

```
例: 「if-then-else」の曖昧性

CFG:
  stmt = "if" expr "then" stmt "else" stmt
       | "if" expr "then" stmt
  → "if a then if b then x else y" は2通りの解釈が可能（ダングリング else）

PEG:
  stmt = "if" expr "then" stmt "else" stmt
       / "if" expr "then" stmt
  → 最初のルール（else 付き）が優先され、曖昧性がない
```

### Q6: WASM でパーサーを使いたい場合はどうすればよいですか？

**A**: 以下の選択肢がある:

1. **tree-sitter**: WASM ビルドが公式サポートされている。ブラウザ上でコード解析が可能
2. **Peggy**: JavaScript ネイティブのため、ブラウザでそのまま動作
3. **nom (Rust)**: `wasm-pack` でコンパイルして WASM モジュールとして使用可能
4. **ANTLR**: JavaScript ターゲットを使えばブラウザで動作

```javascript
// tree-sitter の WASM 使用例
import Parser from 'web-tree-sitter';

async function initParser() {
    await Parser.init();
    const parser = new Parser();
    const Lang = await Parser.Language.load('/tree-sitter-python.wasm');
    parser.setLanguage(Lang);

    const tree = parser.parse('def hello(): pass');
    console.log(tree.rootNode.toString());
    // (module (function_definition name: (identifier) parameters: (parameters) body: (block (pass_statement))))
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| 正規表現の限界 | ネスト構造・再帰・文脈依存は扱えない |
| パーサーコンビネータ | 小さなパーサーを合成して複雑な文法を解析 |
| PEG | 文法定義からパーサーを自動生成。優先順位付き選択 |
| ANTLR | 産業標準のパーサージェネレータ。多言語対応 |
| nom (Rust) | ゼロコピーの高性能パーサーコンビネータ |
| lark (Python) | EBNF 風文法で手軽にパーサーを構築 |
| tree-sitter | インクリメンタルパーサー。エディタ統合に最適 |
| 選択基準 | 標準形式には専用ライブラリ、カスタム文法にはコンビネータ/PEG |
| パフォーマンス | 正規表現 > nom > PEG > pyparsing（一般的な傾向） |

## 次に読むべきガイド

- [正規表現基礎](../01-basics/00-regex-syntax.md) -- 正規表現が適切な場面での活用法
- [テキスト処理実践](../02-practical/02-text-processing.md) -- sed/awk/grep による実務でのテキスト処理パターン

## 参考文献

1. **Bryan Ford**: [Parsing Expression Grammars (2004)](https://bford.info/pub/lang/peg.pdf) -- PEG の原論文
2. **nom 公式**: [nom Documentation](https://docs.rs/nom/latest/nom/) -- Rust パーサーコンビネータの包括的ドキュメント
3. **tree-sitter 公式**: [tree-sitter](https://tree-sitter.github.io/tree-sitter/) -- インクリメンタルパーサーフレームワーク
4. **ANTLR 公式**: [ANTLR](https://www.antlr.org/) -- パーサージェネレータの公式サイト
5. **lark 公式**: [lark-parser](https://lark-parser.readthedocs.io/) -- Python のモダンなパーサーライブラリ
6. **Peggy 公式**: [Peggy](https://peggyjs.org/) -- JavaScript の PEG パーサージェネレータ
7. **megaparsec 公式**: [megaparsec](https://hackage.haskell.org/package/megaparsec) -- Haskell の産業強度パーサーコンビネータ
8. **Terence Parr**: "The Definitive ANTLR 4 Reference" Pragmatic Bookshelf, 2013 -- ANTLR の定番書
9. **Semgrep**: [semgrep.dev](https://semgrep.dev/) -- tree-sitter ベースの静的解析ツール
