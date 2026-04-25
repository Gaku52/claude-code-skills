# Alternatives to Regular Expressions

> Master alternative approaches to text analysis tasks where regex falls short, including parser combinators, PEG, and structured text processing

## What You Will Learn

1. **Limits of regex** -- Why recursive structures, nesting, and context-sensitive grammars cannot be handled
2. **Parser combinators** -- Composing small parsers to analyze complex grammars
3. **PEG and practical tools** -- Properties of Parsing Expression Grammar and tools like tree-sitter
4. **ANTLR and yacc/bison** -- Full-fledged language processing with parser generators
5. **Dedicated parsers for structured data** -- Practical processing techniques for JSON, HTML, XML, YAML, etc.
6. **Leveraging tree-sitter** -- Practical use of incremental parsers


## Prerequisites

Reading the following beforehand will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Text Processing -- sed/awk/grep, Log Analysis, CSV](./02-text-processing.md)

---

## 1. Limits of Regular Expressions

```
Chomsky Hierarchy and Corresponding Parsers
=================================

Type 0: Phrase structure grammar    <-- Turing machine
Type 1: Context-sensitive grammar  <-- Linear bounded automaton
Type 2: Context-free grammar       <-- Pushdown automaton ★
Type 3: Regular grammar            <-- Finite automaton (regex)

What regex cannot handle:
  - Matching parentheses:  ((()))     ★ Context-free grammar
  - Nested HTML: <div><div></div></div>
  - Programming language syntax
  - Recursive structures in general

Things you should NOT do with regex:
  - Parsing HTML/XML
  - Parsing JSON
  - Parsing programming languages
  - Matching nested parentheses
```

### 1.1 Concrete Examples Showing the Limits of Regex

```python
import re

# [NG] Parsing HTML with regex
html = '<div class="outer"><div class="inner">text</div></div>'
# This regex cannot handle nesting
pattern = r'<div[^>]*>(.*?)</div>'
matches = re.findall(pattern, html)
# Expected: contents of inner div -> Actual: incorrect result due to shortest match

# [OK] Use a dedicated parser
from html.parser import HTMLParser
# Or BeautifulSoup, lxml, etc.

# [NG] Parsing JSON with regex
json_str = '{"key": {"nested": [1, 2, 3]}}'
# Recursive structures are impossible with regex

# [OK] Use the json module
import json
data = json.loads(json_str)
```

### 1.2 Typical Cases Where Regex Fails

```python
import re

# Case 1: Matching nested parentheses
# Regex cannot recognize matching parenthesis pairs
text = "f(g(x, y), h(z))"
# Correctly extracting "the entire arguments of f" is impossible
# The following yields incorrect results
match = re.search(r'f\((.+)\)', text)
# match.group(1) = "g(x, y), h(z)" -- happens to be correct, but the next case fails
text2 = "f(g(x), y) + f(a)"
match2 = re.search(r'f\((.+)\)', text2)
# match2.group(1) = "g(x), y) + f(a" -- broken (greedy match)

# Case 2: Handling escapes inside string literals
code = 'print("He said \\"hello\\"", end="")'
# Regex makes it difficult to accurately extract strings containing escaped quotes
# The following is inaccurate
strings = re.findall(r'"([^"]*)"', code)
# Cannot correctly handle escaped \"

# Case 3: Code-like text inside comments
code2 = """
# print("this is a comment")
print("this is real code")
"""
# Determining whether a line is a comment or executable code with regex alone
# is difficult because it requires understanding the language syntax

# Case 4: Heredocs and template literals
ruby_code = '''
text = <<~HEREDOC
  This contains "quotes" and #{interpolation}
  And even regex: /pattern/
HEREDOC
'''
# Correctly tracking the start and end of a heredoc
# requires a state machine
```

### 1.3 The Pumping Lemma -- Mathematical Proof of the Limits of Regular Languages

```
Pumping Lemma:
=================================

Theorem: If a language L is regular, there exists a constant p such that
     for any string w in L with |w| >= p,
     w can be decomposed as w = xyz, satisfying:
       1. |y| > 0
       2. |xy| <= p
       3. For any i >= 0, xy^iz is in L

Counterexample: L = { a^n b^n | n >= 0 } is not a regular language
     "n a's followed by n b's" cannot be expressed by regex

Practical implications:
  - "Matching parentheses" is not a regular language
  - "Matching tags" is not a regular language
  - Trying to fully process these with regex is theoretically impossible

However, note:
  - The "extended" regex of Perl/PCRE goes beyond theoretical regular languages
  - Recursive patterns like (?R) can handle some context-free languages
  - But from a readability/maintainability perspective, you should use a parser
```

### 1.4 PCRE Recursive Patterns -- Extensions of Regex

```python
import regex  # Python's regex module (an extension of re)

# Match matching parentheses with PCRE recursive patterns
# (?R) is a recursive call to the entire pattern
pattern = r'\((?:[^()]*|(?R))*\)'

text = "f(g(x, y), h(z))"
matches = regex.findall(pattern, text)
# matches = ['(g(x, y), h(z))', '(x, y)', '(z)']

# Recursion with named groups
# Use (?P<name>...) and (?&name)
pattern2 = r'(?P<brackets>\{(?:[^{}]*|(?&brackets))*\})'
json_like = '{"a": {"b": {"c": 1}}}'
match = regex.search(pattern2, json_like)
# Match: {"a": {"b": {"c": 1}}}

# Note: This is "possible" but not "recommended"
# From a readability/maintainability standpoint, complex recursive patterns
# should use parser combinators or PEG
```

---

## 2. Parser Combinators

```
The Concept of Parser Combinators
==============================

Build large parsers by combining small parsers

Basic parsers:
  digit    : Parses a single character "0"-"9"
  letter   : Parses a single character "a"-"z"
  string   : Parses a fixed string

Combinators:
  seq(a, b)    : a followed by b   (sequencing)
  alt(a, b)    : a or b   (choice)
  many(a)      : zero or more repetitions of a
  map(a, f)    : apply f to the result of a

Example: integer parser
  integer = map(many1(digit), digits => parseInt(digits.join('')))

Example: arithmetic expression parser
  expr   = alt(addExpr, term)
  term   = alt(mulExpr, factor)
  factor = alt(number, parens(expr))  <- recursion!
```

### 2.1 Parser Combinators in TypeScript

```typescript
// Parser type definition
type Parser<T> = (input: string, pos: number) => ParseResult<T>;
type ParseResult<T> =
  | { success: true; value: T; pos: number }
  | { success: false; expected: string; pos: number };

// Basic parsers
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

// Combinators
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

// Usage example: arithmetic expression parser
const digit = regex(/[0-9]+/);
const number = map(digit, s => parseInt(s, 10));
const ws = regex(/\s*/);

function token<T>(p: Parser<T>): Parser<T> {
  return (input, pos) => {
    const r = ws(input, pos);
    return p(input, r.success ? r.pos : pos);
  };
}

// Parse "123 + 456"
const addExpr = (input: string, pos: number): ParseResult<number> => {
  const left = token(number)(input, pos);
  if (!left.success) return left;
  const op = token(char('+'))(input, left.pos);
  if (!op.success) return left;  // No addition -> just a number
  const right = addExpr(input, op.pos);  // Recursion
  if (!right.success) return right;
  return { success: true, value: left.value + right.value, pos: right.pos };
};
```

### 2.2 Extending the TypeScript Parser Combinator: Error Reporting

```typescript
// More practical parser combinators: report error position and expected values

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

// Compute line and column from an error position
function getLineAndColumn(input: string, pos: number): { line: number; column: number } {
  const lines = input.slice(0, pos).split('\n');
  return { line: lines.length, column: lines[lines.length - 1].length + 1 };
}

// Generate an error message
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

// Practical example: configuration file parser
// Parse a configuration file in key = value format
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

### 2.3 Haskell Parser Combinators (Parsec / Megaparsec)

```haskell
-- Megaparsec is the most mature parser combinator library in Haskell
-- It enables type-safe parsing of context-free grammars that are impossible with regex

import Text.Megaparsec
import Text.Megaparsec.Char
import qualified Text.Megaparsec.Char.Lexer as L
import Data.Void (Void)

type Parser = Parsec Void String

-- Skip whitespace and comments
sc :: Parser ()
sc = L.space space1 (L.skipLineComment "//") (L.skipBlockComment "/*" "*/")

-- Lexer helpers
lexeme :: Parser a -> Parser a
lexeme = L.lexeme sc

symbol :: String -> Parser String
symbol = L.symbol sc

-- Integer literal
integer :: Parser Integer
integer = lexeme L.decimal

-- Identifier
identifier :: Parser String
identifier = lexeme $ do
  first <- letterChar
  rest  <- many (alphaNumChar <|> char '_')
  return (first : rest)

-- JSON parser example
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

-- Usage example
-- parse jsonValue "" "{\"name\": \"Alice\", \"scores\": [95, 87, 92]}"
-- Right (JsonObject [("name", JsonString "Alice"), ("scores", JsonArray [...])])
```

### 2.4 Python Parser Combinator (lark)

```python
from lark import Lark, Transformer, v_args

# lark is one of the most user-friendly parser libraries in Python
# It auto-generates a parser from an EBNF-style grammar definition

# Grammar definition for arithmetic expressions
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

# Transformer that converts the AST into the computed result
@v_args(inline=True)
class CalcTransformer(Transformer):
    from operator import add, sub, mul, truediv as div

    def number(self, n):
        return float(n)

    def neg(self, n):
        return -n

# Generate and use the parser
calc_parser = Lark(calc_grammar, parser='lalr', transformer=CalcTransformer())

# Run the calculation
result = calc_parser.parse("(1 + 2) * 3 - 4 / 2")
print(result)  # 7.0

# More complex example: SQL SELECT statement parser
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

## 3. PEG (Parsing Expression Grammar)

### 3.1 Defining Grammars with PEG.js / Peggy

```javascript
// grammar.pegjs (Peggy format)
// PEG grammar for a JSON parser

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
# Generate a parser with Peggy
npx peggy grammar.pegjs --output parser.js

# Usage
node -e "const p = require('./parser'); console.log(p.parse('{\"a\": 1}'))"
```

### 3.2 Properties of PEG and Differences from Regex

```
PEG vs Regex vs CFG:
========================

Regex:
  - Choice is "longest match" or "shortest match"
  - Risk of exponential execution time due to backtracking
  - Recursion is impossible (except in PCRE extensions)

PEG:
  - Choice is "prioritized" (the first matching alternative is chosen)
  - No ambiguity (the grammar produces a unique parse tree)
  - Recursion is possible (handles context-free grammars)
  - Linear time can be guaranteed with packrat parsers

CFG (Context-Free Grammar):
  - Choice is "non-deterministic" (multiple parse trees possible)
  - May contain ambiguity
  - Parsed with algorithms like LR, LL, Earley

PEG's choice operator "/" is "ordered choice":
  rule = A / B
  -> Try A first
  -> If A succeeds -> use A's result (don't try B)
  -> If A fails -> try B

This results in:
  - Eliminating grammar ambiguity
  - Predictable parser behavior
  - But there's risk of "oversights" (order matters)
```

### 3.3 Practical Grammar Definition Patterns in PEG

```javascript
// Practical grammar pattern collection in Peggy

// Pattern 1: Configuration file parser (INI format)
// File: ini_parser.pegjs

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

// Pattern 2: Markdown inline format parser
// File: markdown_inline.pegjs

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

// Pattern 3: URL parser
// File: url_parser.pegjs

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

## 4. Practical Alternative Tools

### 4.1 Python's pyparsing

```python
from pyparsing import (
    Word, alphas, alphanums, nums, Suppress, Group,
    Forward, Optional, ZeroOrMore, Literal, quotedString
)

# SQL SELECT statement parser (simplified)
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

# Run the parse
result = select_stmt.parseString("SELECT name, age FROM users WHERE status = 'active'")
print(result.columns.asList())  # ['name', 'age']
print(result.table)             # 'users'
print(result.where_col)         # 'status'
```

### 4.2 Rust's nom Parser Combinator

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

// Integer parser
fn integer(input: &str) -> IResult<&str, i64> {
    map_res(digit1, |s: &str| s.parse::<i64>())(input)
}

// Comma-separated list of integers
fn integer_list(input: &str) -> IResult<&str, Vec<i64>> {
    separated_list1(
        delimited(space0, char(','), space0),
        integer,
    )(input)
}

// Parse "[1, 2, 3]"
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

### 4.3 Rust's nom -- A Practical Log Parser

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

// Struct for parsing one line of an Apache log
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

// IP address parser
fn ip_address(input: &str) -> IResult<&str, Ipv4Addr> {
    map_res(
        take_while1(|c: char| c.is_ascii_digit() || c == '.'),
        |s: &str| s.parse::<Ipv4Addr>(),
    )(input)
}

// Timestamp parser: [10/Oct/2000:13:55:36 -0700]
fn timestamp(input: &str) -> IResult<&str, String> {
    map(
        delimited(char('['), take_until("]"), char(']')),
        |s: &str| s.to_string(),
    )(input)
}

// Request line parser: "GET /path HTTP/1.1"
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

// Status code parser
fn status_code(input: &str) -> IResult<&str, u16> {
    map_res(digit1, |s: &str| s.parse::<u16>())(input)
}

// Parser for a full log line
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

// Usage example
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

### 4.4 Go's participle Parser Library

```go
package main

import (
    "fmt"
    "github.com/alecthomas/participle/v2"
    "github.com/alecthomas/participle/v2/lexer"
)

// Go's participle auto-generates parsers from struct tags
// It can type-safely parse recursive structures that are impossible with regex

// AST definition for arithmetic expressions
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

// Configuration file parser example
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

    parser := participle.MustBuildConfig,
    )

    config := &Config{}
    err := parser.ParseString("", input, config)
    return config, err
}
```

---

## 5. ANTLR -- A Parser Generator

### 5.1 Defining and Using Grammars in ANTLR

```antlr
// Calculator.g4 -- ANTLR grammar file
grammar Calculator;

// Parser rules
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

// Lexer rules
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
// Example of using a parser generated by ANTLR (Java)
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;

public class CalcApp {
    public static void main(String[] args) throws Exception {
        // Prepare the input stream
        CharStream input = CharStreams.fromString("x = 1 + 2 * 3\n");

        // Lexer -> token stream -> parser
        CalculatorLexer lexer = new CalculatorLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CalculatorParser parser = new CalculatorParser(tokens);

        // Get the parse tree
        ParseTree tree = parser.prog();

        // Walk the AST with the visitor pattern
        CalcVisitor visitor = new CalcVisitor();
        visitor.visit(tree);
    }
}

// Visitor implementation
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
# How to use ANTLR
# 1. Create the grammar file
# 2. Generate the parser
antlr4 Calculator.g4 -Dlanguage=Python3  # For Python
antlr4 Calculator.g4 -Dlanguage=Java     # For Java
antlr4 Calculator.g4 -Dlanguage=Go       # For Go

# 3. Test with grun (visualize the parse tree in a GUI)
grun Calculator prog -gui
```

---

## 6. tree-sitter -- An Incremental Parser

### 6.1 Overview and Use of tree-sitter

```
Features of tree-sitter:
====================

1. Incremental parsing
   - Re-parses only the modified parts
   - Optimal for real-time syntax analysis in editors
   - O(log n) re-parsing after edits

2. Error recovery
   - Continues parsing as much as possible despite syntax errors
   - The editor can keep highlighting even broken code

3. Multi-language support
   - Grammars for 200+ programming languages are available
   - JavaScript, Python, Rust, Go, Java, C/C++, ...

4. Query system
   - Pattern matching with S-expressions
   - Enables semantic code search

Use cases:
  - Editor syntax highlighting (Neovim, Helix, Zed, etc.)
  - Code navigation (jump to definition/reference)
  - Implementing linters and formatters
  - Code transformation tools
  - GitHub code search and security scanning
```

### 6.2 The tree-sitter Query System

```scheme
;; tree-sitter queries: match code structures using S-expressions

;; Match all Python function definitions
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  body: (block) @function.body)

;; Match only methods inside class definitions
(class_definition
  name: (identifier) @class.name
  body: (block
    (function_definition
      name: (identifier) @method.name)))

;; Match import statements
(import_statement
  name: (dotted_name) @import.module)

(import_from_statement
  module_name: (dotted_name) @import.from
  name: (dotted_name) @import.name)

;; Function calls with a specific pattern
;; Example: calls like logging.error("...")
(call
  function: (attribute
    object: (identifier) @object (#eq? @object "logging")
    attribute: (identifier) @method (#eq? @method "error"))
  arguments: (argument_list
    (string) @message))

;; Match TypeScript type definitions
(type_alias_declaration
  name: (type_identifier) @type.name
  value: (_) @type.definition)

(interface_declaration
  name: (type_identifier) @interface.name
  body: (object_type) @interface.body)
```

### 6.3 A Code Analysis Tool Using tree-sitter (Python)

```python
# Code analysis using tree-sitter's Python bindings

from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# Set up the parser
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

# Parse the source code
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

# Extract function definitions
def find_functions(node, depth=0):
    """Recursively explore function definitions"""
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
        print(f"{'  ' * depth}Function: {info['name']}{info['params']}"
              f"{' -> ' + info['return_type'] if info['return_type'] else ''}"
              f" (line {info['line']})")

    for child in node.children:
        find_functions(child, depth + (1 if node.type == 'class_definition' else 0))

find_functions(root_node)

# Extract class definitions
def find_classes(node):
    """Extract class definitions and their methods"""
    if node.type == 'class_definition':
        name = node.child_by_field_name('name').text.decode()
        methods = []
        body = node.child_by_field_name('body')
        if body:
            for child in body.children:
                if child.type == 'function_definition':
                    method_name = child.child_by_field_name('name').text.decode()
                    methods.append(method_name)
        print(f"Class: {name}")
        print(f"  Methods: {', '.join(methods)}")

    for child in node.children:
        find_classes(child)

find_classes(root_node)

# Detect SQL injection vulnerabilities (simplified)
def find_sql_injection_risks(node):
    """Detect places where SQL is built using string formatting"""
    if node.type == 'binary_operator':
        # Detect SQL constructed via string concatenation (+)
        left = node.children[0]
        if left.type == 'string' and any(
            keyword in left.text.decode().upper()
            for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
        ):
            print(f"Warning: SQL string concatenation (line {node.start_point[0] + 1})")
            print(f"  {node.text.decode()}")

    if node.type == 'call':
        # Detect SQL built with f-strings or .format()
        func = node.child_by_field_name('function')
        if func and func.type == 'attribute' and func.text.decode().endswith('.format'):
            # Check whether the caller of .format() contains SQL
            pass

    for child in node.children:
        find_sql_injection_risks(child)

find_sql_injection_risks(root_node)
```

---

## 7. Dedicated Parsers for Structured Data

### 7.1 HTML Processing -- Beautiful Soup and lxml

```python
from bs4 import BeautifulSoup
import lxml.html

# === Beautiful Soup ===
html = """
<html>
<body>
  <div class="container">
    <h1>Title</h1>
    <ul class="items">
      <li class="item active">Item 1</li>
      <li class="item">Item 2</li>
      <li class="item">Item 3</li>
    </ul>
    <div class="nested">
      <div class="deep">
        <p>Deeply nested text</p>
      </div>
    </div>
  </div>
</body>
</html>
"""

soup = BeautifulSoup(html, 'html.parser')

# Search with CSS selectors (far more reliable than regex)
items = soup.select('ul.items li.item')
for item in items:
    print(item.text)  # "Item 1", "Item 2", "Item 3"

# Get nested elements
deep_text = soup.select_one('.nested .deep p').text
print(deep_text)  # "Deeply nested text"

# Filter by attribute
active = soup.find('li', class_='active')
print(active.text)  # "Item 1"

# Cases impossible with regex: self-closing tags, attribute value quotes, etc.
complex_html = '<img src="photo.jpg" alt="He said &quot;hello&quot;" />'
soup2 = BeautifulSoup(complex_html, 'html.parser')
img = soup2.find('img')
print(img['alt'])  # 'He said "hello"' -- entities are correctly decoded too

# === lxml (XPath) ===
doc = lxml.html.fromstring(html)

# Search with XPath
titles = doc.xpath('//h1/text()')
print(titles)  # ['Title']

items_xpath = doc.xpath('//ul[@class="items"]/li/text()')
print(items_xpath)  # ['Item 1', 'Item 2', 'Item 3']

# Complex conditions
active_xpath = doc.xpath('//li[contains(@class, "active")]/text()')
print(active_xpath)  # ['Item 1']
```

### 7.2 JSON Processing -- jq and Python

```bash
# jq: a command-line JSON processor
# Handles nested JSON that is impossible with regex

# Basic field extraction
echo '{"name": "Alice", "age": 30}' | jq '.name'
# "Alice"

# Nested fields
echo '{"user": {"name": "Alice", "address": {"city": "Tokyo"}}}' | \
    jq '.user.address.city'
# "Tokyo"

# Array processing
echo '[{"name": "Alice"}, {"name": "Bob"}]' | jq '.[].name'
# "Alice"
# "Bob"

# Filtering
echo '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]' | \
    jq '.[] | select(.age > 28)'
# {"name": "Alice", "age": 30}

# Transformation
echo '[{"name": "Alice", "scores": [90, 85, 92]}]' | \
    jq '.[] | {name, avg_score: (.scores | add / length)}'
# {"name": "Alice", "avg_score": 89}

# Stream processing of JSON Lines
cat events.jsonl | jq -c 'select(.level == "ERROR") | {timestamp, message}'

# Conversion to CSV
echo '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]' | \
    jq -r '.[] | [.name, .age] | @csv'
# "Alice",30
# "Bob",25
```

```python
# JSON processing in Python
import json
from pathlib import Path

# Load and transform a JSON file
with open('data.json') as f:
    data = json.load(f)

# Safe access to nested data
def safe_get(data, *keys, default=None):
    """Safely access nested keys"""
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

# Usage example
config = {"database": {"host": "localhost", "port": 5432}}
host = safe_get(config, "database", "host")  # "localhost"
missing = safe_get(config, "database", "timeout", default=30)  # 30

# Validation with JSON Schema
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
    print("Validation succeeded")
except ValidationError as e:
    print(f"Validation failed: {e.message}")
```

### 7.3 YAML Processing

```python
import yaml
from pathlib import Path

# YAML parser (structures impossible to process with regex)
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
      password: ${DB_PASSWORD}  # Environment variable reference
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

# Nested access
print(config['server']['ssl']['enabled'])  # True
print(config['database']['replicas'][0]['host'])  # db-replica-1.example.com

# YAML anchors and aliases (impossible to process with regex)
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
print(envs['production']['host'])  # db.production.com (overridden)
print(envs['production']['port'])  # 5432 (inherited from defaults)
```

### 7.4 XML Processing

```python
import xml.etree.ElementTree as ET
from lxml import etree

# XML parser (correctly handles namespaces and nesting)
xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<bookstore xmlns:bk="http://example.com/books">
  <bk:book category="programming">
    <bk:title lang="ja">Introduction to Programming</bk:title>
    <bk:author>Taro Yamada</bk:author>
    <bk:price currency="JPY">3000</bk:price>
  </bk:book>
  <bk:book category="science">
    <bk:title lang="en">Introduction to Physics</bk:title>
    <bk:author>John Smith</bk:author>
    <bk:price currency="USD">45</bk:price>
  </bk:book>
</bookstore>
"""

# Processing with ElementTree
root = ET.fromstring(xml_content)
ns = {'bk': 'http://example.com/books'}

for book in root.findall('bk:book', ns):
    title = book.find('bk:title', ns).text
    author = book.find('bk:author', ns).text
    price = book.find('bk:price', ns)
    print(f"{title} by {author} - {price.get('currency')} {price.text}")

# lxml + XPath (more powerful queries)
doc = etree.fromstring(xml_content.encode())
nsmap = {'bk': 'http://example.com/books'}

# Title of books in the programming category
titles = doc.xpath('//bk:book[@category="programming"]/bk:title/text()', namespaces=nsmap)
print(titles)  # ['Introduction to Programming']

# Search for Japanese-language books
ja_books = doc.xpath('//bk:title[@lang="ja"]/text()', namespaces=nsmap)
print(ja_books)  # ['Introduction to Programming']
```

---

## Comparison Table of Approaches

| Approach | Expressiveness | Performance | Learning Cost | Use Cases |
|---|---|---|---|---|
| **Regex** | Regular grammar | Fast | Low | Pattern matching, search & replace |
| **Parser combinators** | Context-free grammar | Medium-High | Medium | DSLs, config files, protocols |
| **PEG** | Context-free grammar+ | High | Medium | Language processing, syntax analysis |
| **ANTLR/yacc** | Context-free grammar | High | High | Programming languages, SQL |
| **tree-sitter** | Context-free grammar | Very High | Medium | Editor syntax highlighting |
| **Dedicated parsers** | Arbitrary | Highest | Low | Standard formats like JSON, HTML, CSV |
| **lark (Python)** | Context-free grammar | Medium | Low-Medium | DSLs, custom grammars, prototypes |
| **nom (Rust)** | Context-free grammar | Very High | Medium-High | Binary protocols, high-performance parsers |

### Selection Criteria Comparison

| Requirement | Recommended Approach |
|---|---|
| Simple pattern matching | Regex |
| Standard formats (JSON/HTML/CSV) | Dedicated parser library |
| Designing a custom DSL | Parser combinators or PEG |
| Parsing programming languages | ANTLR / tree-sitter |
| Editor integration (highlighting, etc.) | tree-sitter |
| High-performance binary protocols | nom (Rust) / hand-written parser |
| Rapid prototyping | lark (Python) / Peggy (JS) |
| Type-safe grammar definition | participle (Go) / nom (Rust) |

### Recommended Parser Libraries by Language

| Language | Library | Features |
|---|---|---|
| **Python** | lark | EBNF-style grammar, supports Earley/LALR |
| **Python** | pyparsing | Intuitive API, low learning cost |
| **Rust** | nom | Zero-copy, high performance |
| **Rust** | winnow | Successor to nom, improved error messages |
| **Haskell** | megaparsec | Most theoretically refined |
| **TypeScript** | Peggy | PEG-based, browser support |
| **Go** | participle | Struct-tag based, type-safe |
| **Java** | ANTLR | Industry standard, visual debugger |
| **C/C++** | tree-sitter | Incremental, error recovery |

---

## Anti-patterns

### 1. Parsing HTML with Regex

**Problem**: HTML has nested structure and cannot be expressed by a regular grammar. Countless edge cases exist that regex cannot handle, such as cases where attribute values contain quotes and self-closing tags.

```python
# [NG] Parsing HTML with regex
import re
html = '<div class="outer"><div class="inner">text</div></div>'
# This regex cannot handle nesting
divs = re.findall(r'<div[^>]*>(.*?)</div>', html)
# Expected: ['text'] -> Actual: ['<div class="inner">text'] (broken)

# [OK] Use Beautiful Soup
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')
inner = soup.find('div', class_='inner')
print(inner.text)  # "text"
```

**Solution**: Use dedicated parsers like BeautifulSoup, cheerio, or lxml.

### 2. Trying to Solve Everything with Parser Combinators

**Problem**: Writing a custom parser for standard formats like JSON, CSV, or YAML leads to missed edge cases and security issues.

```python
# [NG] Building your own JSON parser
def parse_json(text):
    # String escaping, Unicode surrogate pairs, numeric precision,
    # BOM handling, recursion depth limits... you have to implement them all
    pass

# [OK] Use the standard library's json module
import json
data = json.loads('{"key": "value"}')
```

**Solution**: Use battle-tested libraries for standard formats. Limit parser combinators to custom DSLs and proprietary protocols.

### 3. Ignoring the Middle Ground Between Regex and Parsers

**Problem**: In situations where "regex isn't enough but a parser is overkill," people fail to choose either and end up writing a hand-rolled state machine.

```python
# [NG] Hand-rolled state machine (hard to maintain)
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

# [OK] Sometimes regex is sufficient
import re
def parse_key_value_regex(text):
    return dict(re.findall(r'([^=\n]+)=([^\n]*)', text))

# [OK] If more complex, use a lightweight parser
from configparser import ConfigParser
parser = ConfigParser()
parser.read_string('[DEFAULT]\n' + text)
```

### 4. Choosing a Parser Without Considering Performance

**Problem**: Choosing a slow library for processing large amounts of data.

```python
# [Caution] Beautiful Soup is convenient but slow for huge HTML
# When parsing a million HTML fragments

# Slow: Beautiful Soup (pure Python parser)
from bs4 import BeautifulSoup
for html_fragment in million_fragments:
    soup = BeautifulSoup(html_fragment, 'html.parser')  # Slow

# Fast: lxml (C implementation)
from lxml import html
for html_fragment in million_fragments:
    doc = html.fromstring(html_fragment)  # Several times faster

# Even faster: Pre-filter with regex + parser
import re
pattern = re.compile(r'class="target"')
for html_fragment in million_fragments:
    if pattern.search(html_fragment):  # Narrow candidates with regex
        doc = html.fromstring(html_fragment)  # Run parser only on candidates
```

---

## FAQ

### Q1: How does the performance of parser combinators compare to regex?

**A**: For simple pattern matching, regex is faster. However, for complex patterns where regex frequently backtracks, PEG and parser combinators deliver more predictable performance. nom (Rust) can sometimes achieve performance on par with regex.

```
Benchmark guide (relative values):
  Regex (simple pattern)    : 1.0x
  Regex (complex pattern)   : 1.0x to 100x (depends on backtracking)
  nom (Rust)                : 1.0x to 2.0x
  Megaparsec (Haskell)      : 2.0x to 5.0x
  lark (Python, LALR)       : 5.0x to 20x
  pyparsing (Python)        : 10x to 50x
  PEG (Peggy, JavaScript)   : 3.0x to 10x
```

### Q2: Can tree-sitter be used outside of editors?

**A**: Yes. As an incremental parser, tree-sitter is also useful in code analysis tools, linters, and code transformation tools. It is also used in GitHub's code search and security scanning.

Concrete examples:

1. **GitHub Code Search**: Uses tree-sitter to understand code's syntactic structure and enable semantic search
2. **Semgrep**: A tree-sitter-based static analysis tool that detects security vulnerabilities
3. **Difftastic**: A syntax-aware diff tool using tree-sitter
4. **Neovim / Helix**: tree-sitter-based syntax highlighting, folding, and text objects

### Q3: Which language has the most mature parser combinator library?

**A**: Haskell's `parsec`/`megaparsec` is the most theoretically refined. In practical terms, Rust's `nom`/`winnow` (high performance), Python's `pyparsing` (readability), and JavaScript's `Peggy` (web integration) are mature in their respective domains.

### Q4: What are concrete criteria for choosing between regex and a parser?

**A**: Decide using the following flowchart:

```
Is the target a standard format (JSON/HTML/CSV/XML/YAML)?
  -> Yes: Use a dedicated parser library
  -> No: v

Is there nested structure or recursion?
  -> Yes: Use a parser combinator / PEG
  -> No: v

Does the pattern fit on one line?
  -> Yes: Regex is sufficient
  -> No: v

Is it a multi-line pattern?
  -> Yes: If regex's multiline / dotall flag can handle it, use regex
  -> No: Consider a parser

Does the regex pattern exceed 50 characters?
  -> Yes: Strongly recommend switching to a parser
  -> No: Regex is OK (but with comments)
```

### Q5: What is the difference between PEG and CFG (context-free grammar)?

**A**: The main difference is how "choice" is handled:

- **CFG choice (|)**: Non-deterministic. Multiple alternatives are considered simultaneously, and ambiguity may arise
- **PEG choice (/)**: Prioritized. The first matching alternative is chosen (no ambiguity)

```
Example: ambiguity of "if-then-else"

CFG:
  stmt = "if" expr "then" stmt "else" stmt
       | "if" expr "then" stmt
  -> "if a then if b then x else y" has two possible interpretations (dangling else)

PEG:
  stmt = "if" expr "then" stmt "else" stmt
       / "if" expr "then" stmt
  -> The first rule (with else) takes priority, eliminating ambiguity
```

### Q6: What should I do if I want to use a parser in WASM?

**A**: There are several options:

1. **tree-sitter**: WASM builds are officially supported. Code analysis is possible in the browser
2. **Peggy**: Native JavaScript, so it runs as-is in the browser
3. **nom (Rust)**: Can be compiled with `wasm-pack` and used as a WASM module
4. **ANTLR**: The JavaScript target enables it to run in the browser

```javascript
// Example of using tree-sitter in WASM
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

## Summary

| Topic | Key Point |
|---|---|
| Limits of regex | Cannot handle nested structures, recursion, or context dependence |
| Parser combinators | Compose small parsers to analyze complex grammars |
| PEG | Auto-generate parsers from grammar definitions. Prioritized choice |
| ANTLR | Industry-standard parser generator. Multi-language support |
| nom (Rust) | Zero-copy, high-performance parser combinator |
| lark (Python) | Easily build parsers with EBNF-style grammar |
| tree-sitter | Incremental parser. Ideal for editor integration |
| Selection criteria | Use dedicated libraries for standard formats; combinators/PEG for custom grammars |
| Performance | Regex > nom > PEG > pyparsing (general trend) |

## Recommended Next Reading

- Regex Fundamentals -- How to use regex effectively in appropriate situations
- [Practical Text Processing](../02-practical/02-text-processing.md) -- Practical text-processing patterns with sed/awk/grep

## References

1. **Bryan Ford**: [Parsing Expression Grammars (2004)](https://bford.info/pub/lang/peg.pdf) -- The original PEG paper
2. **nom official**: [nom Documentation](https://docs.rs/nom/latest/nom/) -- Comprehensive documentation for the Rust parser combinator
3. **tree-sitter official**: [tree-sitter](https://tree-sitter.github.io/tree-sitter/) -- Incremental parser framework
4. **ANTLR official**: [ANTLR](https://www.antlr.org/) -- Official site of the parser generator
5. **lark official**: [lark-parser](https://lark-parser.readthedocs.io/) -- Modern parser library for Python
6. **Peggy official**: [Peggy](https://peggyjs.org/) -- PEG parser generator for JavaScript
7. **megaparsec official**: [megaparsec](https://hackage.haskell.org/package/megaparsec) -- Industrial-strength parser combinator for Haskell
8. **Terence Parr**: "The Definitive ANTLR 4 Reference" Pragmatic Bookshelf, 2013 -- The standard book on ANTLR
9. **Semgrep**: [semgrep.dev](https://semgrep.dev/) -- tree-sitter-based static analysis tool
