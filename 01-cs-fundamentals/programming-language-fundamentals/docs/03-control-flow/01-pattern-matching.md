# パターンマッチ

> パターンマッチは「データの構造に基づいて分岐する」強力な制御構造。switch文の進化版であり、関数型プログラミングの中心的な機能。

## この章で学ぶこと

- [ ] パターンマッチの種類と表現力を理解する
- [ ] 網羅性チェックの重要性を理解する
- [ ] 各言語のパターンマッチ機能を比較できる
- [ ] 実務でのパターンマッチ活用パターンを習得する
- [ ] パターンマッチのアンチパターンを回避できる
- [ ] ADT（代数的データ型）との組み合わせを理解する

---

## 1. パターンマッチの基本概念

### 1.1 パターンマッチとは何か

```
パターンマッチ = 値の「構造」を調べて、合致するパターンに応じた処理を実行する仕組み

switch文との違い:
  switch: 値の「等値比較」のみ
  match:  値の「構造分解」+ 「条件」+ 「束縛」が可能

パターンマッチの構成要素:
  1. リテラルパターン     — 具体的な値との一致
  2. 変数パターン        — 任意の値を束縛
  3. ワイルドカードパターン — 任意の値にマッチ（束縛なし）
  4. 構造体パターン       — データ構造の分解
  5. タプルパターン       — タプルの分解
  6. 列挙型パターン       — バリアントの分解
  7. ガードパターン       — 追加条件の指定
  8. OR パターン         — 複数パターンの論理和
  9. 範囲パターン        — 値の範囲指定
  10. 束縛パターン       — マッチした値に名前を付ける
```

### 1.2 Rust のパターンマッチ（最も完成度が高い）

```rust
// Rust: match による構造的パターンマッチ

// ========================================
// リテラルパターン
// ========================================
match x {
    1 => println!("one"),
    2 | 3 => println!("two or three"),  // OR パターン
    4..=9 => println!("four to nine"),  // 範囲パターン
    _ => println!("other"),             // ワイルドカード
}

// ========================================
// 構造体の分解（Destructuring）
// ========================================
struct Point { x: i32, y: i32 }

match point {
    Point { x: 0, y: 0 } => println!("origin"),
    Point { x, y: 0 } => println!("on x-axis at {}", x),
    Point { x: 0, y } => println!("on y-axis at {}", y),
    Point { x, y } if x == y => println!("on diagonal at {}", x),
    Point { x, y } => println!("({}, {})", x, y),
}

// フィールドの部分マッチ（残りを無視）
struct Config {
    host: String,
    port: u16,
    debug: bool,
    timeout: u64,
}

match config {
    Config { debug: true, .. } => println!("Debug mode enabled"),
    Config { port: 443, host, .. } => println!("HTTPS on {}", host),
    Config { port: 80, host, .. } => println!("HTTP on {}", host),
    _ => println!("Custom config"),
}

// ========================================
// 列挙型の分解
// ========================================
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    Color(u8, u8, u8),
}

match msg {
    Message::Quit => println!("quit"),
    Message::Move { x, y } => println!("move to ({}, {})", x, y),
    Message::Write(text) => println!("text: {}", text),
    Message::Color(r, g, b) => println!("color: #{:02x}{:02x}{:02x}", r, g, b),
}

// ========================================
// ネストしたパターン
// ========================================
match value {
    Some(Some(x)) if x > 0 => println!("positive: {}", x),
    Some(Some(x)) => println!("non-positive: {}", x),
    Some(None) => println!("inner none"),
    None => println!("outer none"),
}

// ネストした列挙型
enum Expr {
    Num(f64),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
}

fn eval(expr: &Expr) -> f64 {
    match expr {
        Expr::Num(n) => *n,
        Expr::Add(a, b) => eval(a) + eval(b),
        Expr::Mul(a, b) => eval(a) * eval(b),
        Expr::Neg(e) => -eval(e),
    }
}

// ========================================
// ガード条件（match guard）
// ========================================
match num {
    n if n < 0 => println!("negative"),
    n if n == 0 => println!("zero"),
    n if n % 2 == 0 => println!("positive even: {}", n),
    n => println!("positive odd: {}", n),
}

// 外部変数を参照するガード
let threshold = 100;
match value {
    v if v > threshold => println!("above threshold"),
    v if v == threshold => println!("at threshold"),
    v => println!("below threshold: {}", v),
}

// ========================================
// 束縛（@ パターン）— 値にマッチしつつ名前を付ける
// ========================================
match age {
    n @ 0..=12 => println!("child: {}", n),
    n @ 13..=17 => println!("teen: {}", n),
    n @ 18..=64 => println!("adult: {}", n),
    n @ 65.. => println!("senior: {}", n),
    _ => unreachable!(),
}

// 列挙型の中身に名前を付ける
match msg {
    Message::Write(ref text @ _) if text.len() > 100 => {
        println!("Long message: {}...", &text[..100]);
    }
    Message::Write(text) => println!("Message: {}", text),
    _ => {}
}

// ========================================
// スライスパターン
// ========================================
match slice {
    [] => println!("empty"),
    [x] => println!("single: {}", x),
    [x, y] => println!("pair: {}, {}", x, y),
    [first, .., last] => println!("first={}, last={}", first, last),
}

// スライスパターンの実務的な使用例
fn parse_command(parts: &[&str]) -> Command {
    match parts {
        ["help"] => Command::Help,
        ["quit" | "exit"] => Command::Quit,
        ["get", key] => Command::Get(key.to_string()),
        ["set", key, value] => Command::Set(key.to_string(), value.to_string()),
        ["del", keys @ ..] => Command::Delete(keys.iter().map(|s| s.to_string()).collect()),
        _ => Command::Unknown,
    }
}

// ========================================
// 参照パターン
// ========================================
let reference = &42;
match reference {
    &val => println!("Got a value: {}", val),
}

// ref キーワード（借用を作る）
let value = String::from("hello");
match value {
    ref s => println!("Borrowed: {}", s),
    // value はまだ使える
}

// ref mut キーワード
let mut value = vec![1, 2, 3];
match value {
    ref mut v => v.push(4),
}
```

### 1.3 パターンマッチが使える場所（Rust）

```rust
// match 式以外でもパターンマッチが使える

// 1. let 束縛
let (x, y, z) = (1, 2, 3);
let Point { x, y } = point;
let Some(value) = optional else { return; };  // let-else (Rust 1.65+)

// 2. if let（単一パターンの簡潔なマッチ）
if let Some(value) = optional {
    println!("Got: {}", value);
}

// 3. while let
while let Some(item) = stack.pop() {
    process(item);
}

// 4. for ループ
for (index, value) in collection.iter().enumerate() {
    println!("{}: {}", index, value);
}

// 5. 関数の引数
fn print_point(&Point { x, y }: &Point) {
    println!("({}, {})", x, y);
}

// 6. クロージャの引数
let points: Vec<Point> = get_points();
let sum_x: i32 = points.iter()
    .map(|&Point { x, .. }| x)
    .sum();

// 7. let-else パターン（Rust 1.65+）
fn parse_config(input: &str) -> Result<Config, Error> {
    let Some(line) = input.lines().next() else {
        return Err(Error::EmptyInput);
    };

    let [key, value] = line.splitn(2, '=').collect::<Vec<_>>()[..] else {
        return Err(Error::InvalidFormat);
    };

    Ok(Config { key: key.to_string(), value: value.to_string() })
}
```

---

## 2. 各言語のパターンマッチ

### 2.1 Python（3.10+ match文）

```python
# Python 3.10: Structural Pattern Matching（PEP 634）

# ========================================
# リテラルパターンとOR パターン
# ========================================
match command:
    case "quit":
        sys.exit()
    case "hello" | "hi" | "hey":
        print("Hello!")
    case str(s) if s.startswith("/"):
        handle_command(s)
    case _:
        print("Unknown")

# ========================================
# シーケンスパターン（タプル、リスト）
# ========================================
match point:
    case (0, 0):
        print("Origin")
    case (x, 0):
        print(f"X-axis at {x}")
    case (0, y):
        print(f"Y-axis at {y}")
    case (x, y) if x == y:
        print(f"On diagonal at ({x}, {y})")
    case (x, y):
        print(f"({x}, {y})")

# 可変長シーケンスパターン
match sequence:
    case []:
        print("Empty")
    case [x]:
        print(f"Single: {x}")
    case [x, y]:
        print(f"Pair: {x}, {y}")
    case [first, *rest]:
        print(f"First: {first}, rest: {rest}")
    case [first, *middle, last]:
        print(f"First: {first}, last: {last}, middle: {middle}")

# ========================================
# マッピングパターン（辞書）
# ========================================
match config:
    case {"type": "postgres", "host": host, "port": port}:
        connect_postgres(host, port)
    case {"type": "sqlite", "path": path}:
        connect_sqlite(path)
    case {"type": "redis", **rest}:
        connect_redis(**rest)  # 残りのキーを捕捉
    case _:
        raise ValueError("Unknown database type")

# ========================================
# クラスパターン
# ========================================
from dataclasses import dataclass

@dataclass
class Click:
    position: tuple[int, int]
    button: str

@dataclass
class KeyPress:
    key: str
    modifiers: list[str]

@dataclass
class Scroll:
    direction: str
    amount: int

match event:
    case Click(position=(x, y), button="left") if x > 100:
        print(f"Left click at ({x}, {y})")
    case Click(position=(x, y), button="right"):
        show_context_menu(x, y)
    case KeyPress(key="Enter", modifiers=[]):
        submit_form()
    case KeyPress(key="Enter", modifiers=["Ctrl"]):
        new_line()
    case KeyPress(key=k, modifiers=mods) if "Ctrl" in mods:
        handle_shortcut(k, mods)
    case KeyPress(key=k):
        type_character(k)
    case Scroll(direction="up", amount=n):
        scroll_up(n)
    case Scroll(direction="down", amount=n):
        scroll_down(n)

# ========================================
# 型パターン
# ========================================
match value:
    case bool():        # bool は int のサブクラスなので先にチェック
        print(f"Boolean: {value}")
    case int(n) if n > 0:
        print(f"Positive int: {n}")
    case int(n):
        print(f"Non-positive int: {n}")
    case float(f):
        print(f"Float: {f}")
    case str(s) if len(s) > 100:
        print(f"Long string: {s[:100]}...")
    case str(s):
        print(f"String: {s}")
    case list() as lst if len(lst) > 0:
        print(f"Non-empty list of {len(lst)} items")
    case _:
        print(f"Other: {value}")

# ========================================
# ガード条件（if）
# ========================================
match point:
    case (x, y) if x > 0 and y > 0:
        print("First quadrant")
    case (x, y) if x < 0 and y > 0:
        print("Second quadrant")
    case (x, y) if x < 0 and y < 0:
        print("Third quadrant")
    case (x, y) if x > 0 and y < 0:
        print("Fourth quadrant")
    case (x, y):
        print("On axis")

# ========================================
# 実務的な例: JSON API レスポンスの処理
# ========================================
def handle_api_response(response: dict) -> str:
    match response:
        case {"status": "success", "data": {"users": [*users]}}:
            return f"Found {len(users)} users"
        case {"status": "success", "data": {"user": {"name": name, "email": email}}}:
            return f"User: {name} ({email})"
        case {"status": "error", "code": 404, "message": msg}:
            return f"Not found: {msg}"
        case {"status": "error", "code": code, "message": msg} if code >= 500:
            log_error(f"Server error {code}: {msg}")
            return f"Server error: {msg}"
        case {"status": "error", "code": code, "message": msg}:
            return f"Error {code}: {msg}"
        case _:
            return "Unknown response format"

# ========================================
# 実務的な例: AST（抽象構文木）の評価
# ========================================
def evaluate(expr):
    match expr:
        case {"type": "number", "value": n}:
            return n
        case {"type": "string", "value": s}:
            return s
        case {"type": "binary", "op": "+", "left": left, "right": right}:
            return evaluate(left) + evaluate(right)
        case {"type": "binary", "op": "-", "left": left, "right": right}:
            return evaluate(left) - evaluate(right)
        case {"type": "binary", "op": "*", "left": left, "right": right}:
            return evaluate(left) * evaluate(right)
        case {"type": "binary", "op": "/", "left": left, "right": right}:
            divisor = evaluate(right)
            if divisor == 0:
                raise ValueError("Division by zero")
            return evaluate(left) / divisor
        case {"type": "unary", "op": "-", "operand": operand}:
            return -evaluate(operand)
        case {"type": "call", "name": name, "args": args}:
            evaluated_args = [evaluate(arg) for arg in args]
            return call_function(name, evaluated_args)
        case _:
            raise ValueError(f"Unknown expression: {expr}")
```

### 2.2 TypeScript（判別ユニオン + switch）

```typescript
// TypeScript: 判別ユニオンで擬似パターンマッチ

// ========================================
// 基本的な判別ユニオン
// ========================================
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rect"; width: number; height: number }
    | { kind: "triangle"; base: number; height: number };

function area(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            return Math.PI * shape.radius ** 2;
        case "rect":
            return shape.width * shape.height;
        case "triangle":
            return (shape.base * shape.height) / 2;
    }
    // TypeScript は網羅性をチェック（kind の全値を処理しないとエラー）
}

// ========================================
// ts-pattern ライブラリでより強力なパターンマッチ
// ========================================
import { match, P } from 'ts-pattern';

const result = match(shape)
    .with({ kind: "circle", radius: P.when(r => r > 10) }, s =>
        `Large circle: ${s.radius}`)
    .with({ kind: "circle" }, s =>
        `Small circle: ${s.radius}`)
    .with({ kind: "rect" }, s =>
        `Rectangle: ${s.width}x${s.height}`)
    .with({ kind: "triangle" }, s =>
        `Triangle: base=${s.base}`)
    .exhaustive();  // 網羅性チェック

// ========================================
// ts-pattern の高度な使用例
// ========================================
type ApiResponse =
    | { status: "loading" }
    | { status: "success"; data: unknown }
    | { status: "error"; error: string; retryable: boolean };

function renderResponse(response: ApiResponse): string {
    return match(response)
        .with({ status: "loading" }, () => "Loading...")
        .with({ status: "success", data: P.nullish }, () => "No data")
        .with({ status: "success", data: P.array() }, (r) =>
            `Found ${(r.data as unknown[]).length} items`)
        .with({ status: "success" }, (r) => `Data: ${JSON.stringify(r.data)}`)
        .with({ status: "error", retryable: true }, (r) =>
            `Error: ${r.error} (retryable)`)
        .with({ status: "error", retryable: false }, (r) =>
            `Fatal error: ${r.error}`)
        .exhaustive();
}

// ========================================
// TypeScript: 型ガードとパターンマッチの組み合わせ
// ========================================
type Result<T, E> =
    | { ok: true; value: T }
    | { ok: false; error: E };

function processResult<T, E>(result: Result<T, E>): string {
    if (result.ok) {
        // result.value にアクセス可能（型が絞り込まれる）
        return `Success: ${result.value}`;
    } else {
        // result.error にアクセス可能
        return `Error: ${result.error}`;
    }
}

// ========================================
// 複雑な判別ユニオンの実務例: Redux アクション
// ========================================
type Action =
    | { type: "INCREMENT"; amount: number }
    | { type: "DECREMENT"; amount: number }
    | { type: "RESET" }
    | { type: "SET"; value: number }
    | { type: "FETCH_START" }
    | { type: "FETCH_SUCCESS"; data: number[] }
    | { type: "FETCH_ERROR"; error: string };

interface State {
    count: number;
    loading: boolean;
    error: string | null;
    data: number[];
}

function reducer(state: State, action: Action): State {
    switch (action.type) {
        case "INCREMENT":
            return { ...state, count: state.count + action.amount };
        case "DECREMENT":
            return { ...state, count: state.count - action.amount };
        case "RESET":
            return { ...state, count: 0 };
        case "SET":
            return { ...state, count: action.value };
        case "FETCH_START":
            return { ...state, loading: true, error: null };
        case "FETCH_SUCCESS":
            return { ...state, loading: false, data: action.data };
        case "FETCH_ERROR":
            return { ...state, loading: false, error: action.error };
        default: {
            const _exhaustive: never = action;
            return state;
        }
    }
}

// ========================================
// never 型による網羅性チェックのユーティリティ
// ========================================
function assertNever(value: never): never {
    throw new Error(`Unexpected value: ${value}`);
}

function handleShape(shape: Shape): string {
    switch (shape.kind) {
        case "circle": return "Circle";
        case "rect": return "Rectangle";
        case "triangle": return "Triangle";
        default: return assertNever(shape);
        // 新しい Shape を追加すると、ここでコンパイルエラー
    }
}
```

### 2.3 Haskell（パターンマッチの元祖）

```haskell
-- Haskell: 関数定義でのパターンマッチ

-- ========================================
-- 基本的な関数パターンマッチ
-- ========================================
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- ガード条件
bmiCategory :: Double -> String
bmiCategory bmi
  | bmi < 18.5 = "Underweight"
  | bmi < 25.0 = "Normal"
  | bmi < 30.0 = "Overweight"
  | otherwise   = "Obese"

-- ========================================
-- case 式
-- ========================================
describe :: [a] -> String
describe xs = case xs of
    []     -> "empty"
    [_]    -> "singleton"
    [_,_]  -> "pair"
    _      -> "many"

-- ========================================
-- リストのパターン（cons パターン）
-- ========================================
head' :: [a] -> a
head' (x:_) = x
head' []    = error "empty list"

-- リストの再帰処理
sum' :: Num a => [a] -> a
sum' []     = 0
sum' (x:xs) = x + sum' xs

-- リストパターンの応用
zip' :: [a] -> [b] -> [(a, b)]
zip' _ []          = []
zip' [] _          = []
zip' (x:xs) (y:ys) = (x, y) : zip' xs ys

-- ========================================
-- タプルのパターン
-- ========================================
addVectors :: (Double, Double) -> (Double, Double) -> (Double, Double)
addVectors (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)

fst3 :: (a, b, c) -> a
fst3 (x, _, _) = x

-- ========================================
-- 代数的データ型のパターンマッチ
-- ========================================
data Shape = Circle Double
           | Rectangle Double Double
           | Triangle Double Double Double
           deriving (Show)

area :: Shape -> Double
area (Circle r)        = pi * r * r
area (Rectangle w h)   = w * h
area (Triangle a b c)  = let s = (a + b + c) / 2
                          in sqrt (s * (s-a) * (s-b) * (s-c))

-- ========================================
-- Maybe 型のパターンマッチ
-- ========================================
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing
safeDivide a b = Just (a / b)

fromMaybe :: a -> Maybe a -> a
fromMaybe def Nothing  = def
fromMaybe _   (Just x) = x

-- ========================================
-- Either 型のパターンマッチ
-- ========================================
data AppError = NotFound String
              | Unauthorized
              | ValidationError [String]
              deriving (Show)

handleError :: Either AppError a -> String
handleError (Left (NotFound resource))    = "Not found: " ++ resource
handleError (Left Unauthorized)           = "Unauthorized"
handleError (Left (ValidationError errs)) = "Validation: " ++ unwords errs
handleError (Right _)                     = "Success"

-- ========================================
-- レコード構文のパターンマッチ
-- ========================================
data User = User
    { userName  :: String
    , userAge   :: Int
    , userEmail :: String
    } deriving (Show)

greetUser :: User -> String
greetUser User { userName = name, userAge = age }
    | age < 18  = "Hi, " ++ name ++ "!"
    | otherwise = "Hello, " ++ name ++ "."

-- ========================================
-- as パターン（@）
-- ========================================
firstLetter :: String -> String
firstLetter ""       = "Empty"
firstLetter all@(x:_) = "First letter of " ++ all ++ " is " ++ [x]

-- ========================================
-- 二分木のパターンマッチ
-- ========================================
data Tree a = Leaf | Node (Tree a) a (Tree a)
              deriving (Show)

-- 木の要素数
treeSize :: Tree a -> Int
treeSize Leaf         = 0
treeSize (Node l _ r) = 1 + treeSize l + treeSize r

-- 木の深さ
treeDepth :: Tree a -> Int
treeDepth Leaf         = 0
treeDepth (Node l _ r) = 1 + max (treeDepth l) (treeDepth r)

-- 木のフラット化
flatten :: Tree a -> [a]
flatten Leaf         = []
flatten (Node l x r) = flatten l ++ [x] ++ flatten r

-- 要素の検索（二分探索木）
search :: Ord a => a -> Tree a -> Bool
search _ Leaf = False
search target (Node left value right)
    | target == value = True
    | target < value  = search target left
    | otherwise       = search target right

-- 要素の挿入（二分探索木）
insert :: Ord a => a -> Tree a -> Tree a
insert x Leaf = Node Leaf x Leaf
insert x (Node left value right)
    | x < value  = Node (insert x left) value right
    | x > value  = Node left value (insert x right)
    | otherwise   = Node left value right  -- 重複は無視
```

### 2.4 Scala

```scala
// Scala: 最も表現力豊かなパターンマッチの1つ

// ========================================
// 基本的なパターンマッチ
// ========================================
val result = x match {
  case 1 => "one"
  case n if n > 0 => s"positive: $n"
  case _ => "other"
}

// ========================================
// case class の分解
// ========================================
case class Person(name: String, age: Int)
case class Address(city: String, country: String)

person match {
  case Person("Alice", _) => "Found Alice"
  case Person(name, age) if age >= 18 => s"$name is an adult"
  case Person(name, age) => s"$name is $age years old"
}

// ========================================
// sealed trait（代数的データ型）
// ========================================
sealed trait Expr
case class Num(value: Double) extends Expr
case class Add(left: Expr, right: Expr) extends Expr
case class Mul(left: Expr, right: Expr) extends Expr
case class Neg(expr: Expr) extends Expr
case class Var(name: String) extends Expr

def eval(expr: Expr, env: Map[String, Double]): Double = expr match {
  case Num(n) => n
  case Add(l, r) => eval(l, env) + eval(r, env)
  case Mul(l, r) => eval(l, env) * eval(r, env)
  case Neg(e) => -eval(e, env)
  case Var(name) => env.getOrElse(name, throw new RuntimeException(s"Undefined: $name"))
}
// sealed trait → 網羅性チェック

// ========================================
// 型パターン
// ========================================
def describe(x: Any): String = x match {
  case i: Int if i > 0 => s"positive int: $i"
  case i: Int => s"non-positive int: $i"
  case s: String if s.nonEmpty => s"non-empty string: $s"
  case s: String => "empty string"
  case _: Boolean => "boolean"
  case l: List[_] => s"list of ${l.size} elements"
  case _ => "unknown"
}

// ========================================
// 抽出子（Extractor）パターン — unapply メソッド
// ========================================
object Email {
  def unapply(s: String): Option[(String, String)] = {
    val parts = s.split("@")
    if (parts.length == 2) Some((parts(0), parts(1)))
    else None
  }
}

"user@example.com" match {
  case Email(user, domain) => s"User: $user, Domain: $domain"
  case _ => "Not an email"
}

// カスタム抽出子
object Even {
  def unapply(n: Int): Boolean = n % 2 == 0
}

object Positive {
  def unapply(n: Int): Boolean = n > 0
}

42 match {
  case n if Even.unapply(n) && Positive.unapply(n) => s"$n is positive and even"
  case _ => "other"
}

// ========================================
// パーシャル関数（PartialFunction）
// ========================================
val handler: PartialFunction[Int, String] = {
  case 200 => "OK"
  case 404 => "Not Found"
  case 500 => "Internal Server Error"
}

// isDefinedAt でチェック
handler.isDefinedAt(200) // true
handler.isDefinedAt(302) // false

// collect で安全に適用
val codes = List(200, 301, 404, 500)
val messages = codes.collect(handler) // List("OK", "Not Found", "Internal Server Error")

// ========================================
// for 内包表記でのパターンマッチ
// ========================================
val pairs = List((1, "one"), (2, "two"), (3, "three"))
for {
  (num, name) <- pairs
  if num > 1
} yield s"$num = $name"
// List("2 = two", "3 = three")

// Option のパターンマッチ in for
val users = Map("alice" -> 30, "bob" -> 25)
val emails = Map("alice" -> "alice@example.com")

for {
  (name, age) <- users
  email <- emails.get(name)
} yield s"$name ($age): $email"
// List("alice (30): alice@example.com")
```

### 2.5 Elixir

```elixir
# Elixir: パターンマッチが言語の核心

# ========================================
# 基本的なパターンマッチ（= は束縛演算子）
# ========================================
{:ok, result} = {:ok, 42}       # result = 42
[head | tail] = [1, 2, 3, 4]    # head = 1, tail = [2, 3, 4]
%{name: name} = %{name: "Gaku", age: 30}  # name = "Gaku"

# ========================================
# case 式
# ========================================
case File.read("config.txt") do
  {:ok, content} ->
    IO.puts("Content: #{content}")
  {:error, :enoent} ->
    IO.puts("File not found")
  {:error, reason} ->
    IO.puts("Error: #{reason}")
end

# ========================================
# 関数定義でのパターンマッチ（複数の関数ヘッド）
# ========================================
defmodule Math do
  def factorial(0), do: 1
  def factorial(n) when n > 0, do: n * factorial(n - 1)

  def fibonacci(0), do: 0
  def fibonacci(1), do: 1
  def fibonacci(n) when n > 1, do: fibonacci(n - 1) + fibonacci(n - 2)
end

# ========================================
# マップのパターンマッチ
# ========================================
defmodule UserHandler do
  def process(%{role: "admin", name: name}) do
    IO.puts("Admin: #{name}")
  end

  def process(%{role: "user", name: name, verified: true}) do
    IO.puts("Verified user: #{name}")
  end

  def process(%{role: "user", name: name}) do
    IO.puts("Unverified user: #{name}")
  end

  def process(_) do
    IO.puts("Unknown role")
  end
end

# ========================================
# ピン演算子（^）— 再束縛の防止
# ========================================
x = 1
case {1, 2, 3} do
  {^x, y, z} -> "x is 1, y=#{y}, z=#{z}"  # ^x は x の現在の値（1）にマッチ
  _ -> "no match"
end

# ========================================
# with 式（パイプライン的なパターンマッチ）
# ========================================
def create_user(params) do
  with {:ok, name} <- validate_name(params["name"]),
       {:ok, email} <- validate_email(params["email"]),
       {:ok, user} <- insert_user(%{name: name, email: email}) do
    {:ok, user}
  else
    {:error, :invalid_name} -> {:error, "Invalid name"}
    {:error, :invalid_email} -> {:error, "Invalid email"}
    {:error, reason} -> {:error, "Database error: #{reason}"}
  end
end
```

### 2.6 OCaml / F#

```ocaml
(* OCaml: ML系のパターンマッチ *)

(* 基本的なパターンマッチ *)
let rec factorial = function
  | 0 -> 1
  | n -> n * factorial (n - 1)

(* リストのパターンマッチ *)
let rec sum = function
  | [] -> 0
  | x :: xs -> x + sum xs

(* 代数的データ型 *)
type shape =
  | Circle of float
  | Rectangle of float * float
  | Triangle of float * float * float

let area = function
  | Circle r -> Float.pi *. r *. r
  | Rectangle (w, h) -> w *. h
  | Triangle (a, b, c) ->
    let s = (a +. b +. c) /. 2.0 in
    Float.sqrt (s *. (s -. a) *. (s -. b) *. (s -. c))

(* Option 型 *)
let safe_divide a b =
  match b with
  | 0.0 -> None
  | _ -> Some (a /. b)

let unwrap_or default = function
  | None -> default
  | Some x -> x

(* ネストしたパターン *)
type expr =
  | Num of float
  | Add of expr * expr
  | Mul of expr * expr
  | Neg of expr

let rec eval = function
  | Num n -> n
  | Add (a, b) -> eval a +. eval b
  | Mul (a, b) -> eval a *. eval b
  | Neg e -> -.(eval e)

(* as パターン *)
let describe_list = function
  | [] -> "empty"
  | [_] -> "singleton"
  | (_ :: _ :: _ as lst) -> Printf.sprintf "list of %d" (List.length lst)

(* when ガード *)
let categorize n = match n with
  | n when n < 0 -> "negative"
  | 0 -> "zero"
  | n when n mod 2 = 0 -> "positive even"
  | _ -> "positive odd"
```

---

## 3. 網羅性チェック（Exhaustiveness Checking）

### 3.1 なぜ網羅性チェックが重要か

```
網羅性チェック = 「全てのケースが処理されているか」をコンパイル時に検証

なぜ重要か？
  1. 新しいバリアントを追加した時、処理漏れをコンパイルエラーで検出
  2. 実行時の予期しない動作を防止
  3. デッドコード（到達不能なパターン）の検出
  4. リファクタリング時の安全性を保証

網羅性チェックの仕組み:
  - コンパイラがパターンの集合を分析
  - 全ての可能な値がカバーされているか検証
  - カバーされていない場合はコンパイルエラー（または警告）
```

### 3.2 各言語の網羅性チェック

```rust
// Rust: 網羅性チェック（最も厳密）
enum Color { Red, Green, Blue }

fn describe(c: Color) -> &'static str {
    match c {
        Color::Red => "red",
        Color::Green => "green",
        // Color::Blue が未処理 → コンパイルエラー:
        // error[E0004]: non-exhaustive patterns: `Blue` not covered
    }
}

// 新しいバリアント追加時の安全性
enum Color { Red, Green, Blue, Yellow }  // Yellow 追加
// → 全ての match 文でコンパイルエラーが発生
// → 処理漏れを確実に検出

// Option 型の網羅性
fn process(opt: Option<i32>) -> String {
    match opt {
        Some(n) if n > 0 => format!("positive: {}", n),
        Some(n) => format!("non-positive: {}", n),
        None => "nothing".to_string(),
        // 全パターンを網羅 → OK
    }
}

// Result 型の網羅性
fn handle(result: Result<String, AppError>) -> String {
    match result {
        Ok(value) => value,
        Err(AppError::NotFound(msg)) => format!("Not found: {}", msg),
        Err(AppError::Unauthorized) => "Unauthorized".to_string(),
        Err(AppError::Validation(errors)) => format!("Invalid: {:?}", errors),
        // 全エラーバリアントを網羅する必要がある
    }
}

// 数値型の網羅性（ワイルドカードが必要）
fn categorize(n: i32) -> &'static str {
    match n {
        0 => "zero",
        1..=100 => "small positive",
        // i32 の全範囲をカバーする必要がある → _ が必要
        _ => "other",
    }
}
```

```typescript
// TypeScript: never 型による網羅性チェック
type Color = "red" | "green" | "blue";

function describe(c: Color): string {
    switch (c) {
        case "red": return "Red";
        case "green": return "Green";
        case "blue": return "Blue";
        default:
            const _exhaustive: never = c;
            // "yellow" を追加すると、ここでコンパイルエラー
            return _exhaustive;
    }
}

// satisfies 演算子での網羅性チェック（TypeScript 4.9+）
type EventType = "click" | "hover" | "scroll";

const handlers = {
    click: (e: MouseEvent) => { /* ... */ },
    hover: (e: MouseEvent) => { /* ... */ },
    scroll: (e: Event) => { /* ... */ },
} satisfies Record<EventType, Function>;
// EventType に新しい値を追加すると、handlers でエラーになる

// マップ型での網羅性チェック
type StatusCode = 200 | 201 | 400 | 404 | 500;

const statusMessages: Record<StatusCode, string> = {
    200: "OK",
    201: "Created",
    400: "Bad Request",
    404: "Not Found",
    500: "Internal Server Error",
    // StatusCode に値を追加すると、ここにも追加が必要
};
```

```haskell
-- Haskell: コンパイル時の網羅性チェック（-Wall オプション）
data Color = Red | Green | Blue

describe :: Color -> String
describe Red   = "red"
describe Green = "green"
-- Blue が未処理
-- GHC: warning: [-Wincomplete-patterns]
--   Pattern match(es) are non-exhaustive
--   In an equation for 'describe': Patterns not matched: Blue
```

```scala
// Scala: sealed trait で網羅性チェック
sealed trait Color
case object Red extends Color
case object Green extends Color
case object Blue extends Color

def describe(c: Color): String = c match {
  case Red => "red"
  case Green => "green"
  // Blue が未処理
  // warning: match may not be exhaustive
  // It would fail on the following input: Blue
}
```

### 3.3 網羅性チェックがない言語での対策

```python
# Python: match 文に網羅性チェックはない
# → mypy のプラグインで部分的にチェック可能

from enum import Enum
from typing import assert_never

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

def describe(c: Color) -> str:
    match c:
        case Color.RED:
            return "Red"
        case Color.GREEN:
            return "Green"
        case Color.BLUE:
            return "Blue"
        case _ as unreachable:
            assert_never(unreachable)  # mypy が網羅性をチェック
```

```go
// Go: 網羅性チェックはないが、exhaustive リンターが利用可能
// go install github.com/nishanths/exhaustive/cmd/exhaustive@latest

type Color int

const (
    Red Color = iota
    Green
    Blue
)

func describe(c Color) string {
    switch c {
    case Red:
        return "red"
    case Green:
        return "green"
    // Blue が未処理 → exhaustive リンターが警告
    default:
        return "unknown"
    }
}
```

---

## 4. パターンマッチと代数的データ型（ADT）

### 4.1 ADT とパターンマッチの相性

```
代数的データ型（Algebraic Data Types）:
  直積型（Product Type）: 複数のフィールドを持つ（構造体、タプル）
  直和型（Sum Type）: 複数のバリアントのいずれか（列挙型、ユニオン型）

パターンマッチは直和型と完璧に組み合わさる:
  → 各バリアントに対してパターンを定義
  → 網羅性チェックで安全性を保証
  → データの構造分解で内部の値にアクセス
```

```rust
// Rust: ADT + パターンマッチの実務例

// ========================================
// HTTPレスポンスの型安全な表現
// ========================================
enum HttpResponse {
    Ok { body: String, headers: HashMap<String, String> },
    Created { id: String, location: String },
    NoContent,
    BadRequest { errors: Vec<ValidationError> },
    NotFound { resource: String },
    Unauthorized { reason: String },
    InternalError { message: String, trace: Option<String> },
}

fn render_response(response: HttpResponse) -> (u16, String) {
    match response {
        HttpResponse::Ok { body, .. } => (200, body),
        HttpResponse::Created { id, location } => {
            (201, format!(r#"{{"id": "{}", "location": "{}"}}"#, id, location))
        }
        HttpResponse::NoContent => (204, String::new()),
        HttpResponse::BadRequest { errors } => {
            let msgs: Vec<String> = errors.iter()
                .map(|e| format!("{}: {}", e.field, e.message))
                .collect();
            (400, format!(r#"{{"errors": [{}]}}"#, msgs.join(", ")))
        }
        HttpResponse::NotFound { resource } => {
            (404, format!(r#"{{"error": "{} not found"}}"#, resource))
        }
        HttpResponse::Unauthorized { reason } => {
            (401, format!(r#"{{"error": "{}"}}"#, reason))
        }
        HttpResponse::InternalError { message, trace } => {
            if let Some(t) = trace {
                eprintln!("Internal error trace: {}", t);
            }
            (500, format!(r#"{{"error": "{}"}}"#, message))
        }
    }
}

// ========================================
// コンパイラの AST 表現
// ========================================
enum Type {
    Int,
    Float,
    Bool,
    String,
    Array(Box<Type>),
    Function { params: Vec<Type>, returns: Box<Type> },
    Optional(Box<Type>),
}

fn type_to_string(ty: &Type) -> String {
    match ty {
        Type::Int => "int".to_string(),
        Type::Float => "float".to_string(),
        Type::Bool => "bool".to_string(),
        Type::String => "string".to_string(),
        Type::Array(inner) => format!("{}[]", type_to_string(inner)),
        Type::Function { params, returns } => {
            let param_strs: Vec<String> = params.iter()
                .map(type_to_string)
                .collect();
            format!("({}) -> {}", param_strs.join(", "), type_to_string(returns))
        }
        Type::Optional(inner) => format!("{}?", type_to_string(inner)),
    }
}
```

### 4.2 Option / Maybe パターン

```rust
// Rust: Option<T> の活用パターン

// map — Some の中身を変換
let length: Option<usize> = name.map(|n| n.len());

// and_then — ネストした Option をフラット化
let first_char: Option<char> = name.and_then(|n| n.chars().next());

// unwrap_or — デフォルト値
let display_name = name.unwrap_or("Anonymous");

// unwrap_or_else — 遅延評価のデフォルト値
let display_name = name.unwrap_or_else(|| generate_default_name());

// ? 演算子（Option の連鎖）
fn get_street_name(user: &User) -> Option<String> {
    let address = user.address.as_ref()?;
    let street = address.street.as_ref()?;
    Some(street.clone())
}

// filter
let even_number: Option<i32> = some_number.filter(|n| n % 2 == 0);

// zip — 2つの Option を結合
let full_name: Option<String> = first_name.zip(last_name)
    .map(|(first, last)| format!("{} {}", first, last));

// ok_or — Option を Result に変換
let value: Result<i32, AppError> = optional_value
    .ok_or(AppError::NotFound("value".to_string()))?;
```

```haskell
-- Haskell: Maybe の活用パターン

-- fmap（Functor）
length' :: Maybe String -> Maybe Int
length' = fmap length

-- >>= （Monad bind）
firstChar :: Maybe String -> Maybe Char
firstChar name = name >>= safeHead
  where
    safeHead []    = Nothing
    safeHead (x:_) = Just x

-- do 記法
getStreetName :: User -> Maybe String
getStreetName user = do
  address <- userAddress user
  street <- addressStreet address
  return (streetName street)

-- fromMaybe（デフォルト値）
displayName :: Maybe String -> String
displayName = fromMaybe "Anonymous"
```

---

## 5. パターンマッチのアンチパターン

### 5.1 よくある間違い

```
❌ ワイルドカードの過剰使用
match color {
    Color::Red => "red",
    _ => "other",  // Green, Blue の個別処理を忘れる可能性
}
// 新しいバリアントが追加されても気づかない

✅ 全バリアントを明示的に処理
match color {
    Color::Red => "red",
    Color::Green => "green",
    Color::Blue => "blue",
}
// 新しいバリアントが追加されるとコンパイルエラー

❌ パターンの順序ミス
match n {
    _ => "any",     // 全てマッチ → 以下は到達不能
    1 => "one",     // 到達不能（unreachable pattern）
}

❌ ガード条件の網羅性の罠
match n {
    x if x > 0 => "positive",
    x if x < 0 => "negative",
    // x == 0 のケースが抜けている！
    // Rust はガード条件の網羅性を保証できない → _ が必要
}

✅ ガード条件を使う場合は最後にキャッチオール
match n {
    x if x > 0 => "positive",
    x if x < 0 => "negative",
    _ => "zero",  // 0 をキャッチ
}
```

### 5.2 パフォーマンスの考慮

```
パターンマッチのコンパイル:
  1. 決定木（Decision Tree）に変換される
  2. 各パターンを順番にチェックするのではなく、最適化される
  3. 一般的に O(1) 〜 O(log n) の計算量

最適化のヒント:
  - 頻出パターンを先に配置
  - 不必要なガード条件を避ける
  - ワイルドカードは最後に配置

Rust のマッチの最適化:
  - 整数のマッチ → ジャンプテーブルまたは二分探索
  - 列挙型のマッチ → タグの比較（O(1)）
  - 文字列のマッチ → ハッシュまたは逐次比較
```

### 5.3 複雑すぎるパターンの回避

```rust
// ❌ 複雑すぎるパターン（読みにくい）
match response {
    Ok(Response { status: 200, body: Some(Body { content_type: "json", data, .. }), .. })
        if data.len() > 0 => {
        parse_json(data)
    }
    Ok(Response { status: 200, body: Some(Body { content_type: "xml", data, .. }), .. })
        if data.len() > 0 => {
        parse_xml(data)
    }
    Ok(Response { status: code @ 200..=299, .. }) => {
        handle_success(code)
    }
    Ok(Response { status: code @ 400..=499, .. }) => {
        handle_client_error(code)
    }
    Err(e) => handle_error(e),
    _ => handle_unknown(),
}

// ✅ ヘルパー関数に分解
fn process_response(response: Result<Response, Error>) -> Output {
    match response {
        Ok(resp) => process_ok_response(resp),
        Err(e) => handle_error(e),
    }
}

fn process_ok_response(resp: Response) -> Output {
    match resp.status {
        200 => process_body(resp.body),
        code @ 200..=299 => handle_success(code),
        code @ 400..=499 => handle_client_error(code),
        _ => handle_unknown(),
    }
}

fn process_body(body: Option<Body>) -> Output {
    match body {
        Some(Body { content_type: "json", data, .. }) if !data.is_empty() => {
            parse_json(&data)
        }
        Some(Body { content_type: "xml", data, .. }) if !data.is_empty() => {
            parse_xml(&data)
        }
        Some(_) => Output::Empty,
        None => Output::NoBody,
    }
}
```

---

## 6. 実務でのパターンマッチ活用

### 6.1 コマンドパーサー

```rust
// Rust: CLI コマンドパーサー
enum Command {
    Get { key: String },
    Set { key: String, value: String, ttl: Option<u64> },
    Delete { keys: Vec<String> },
    List { pattern: Option<String>, limit: usize },
    Help,
    Quit,
}

fn parse_command(input: &str) -> Result<Command, ParseError> {
    let parts: Vec<&str> = input.trim().split_whitespace().collect();
    match parts.as_slice() {
        ["get", key] => Ok(Command::Get { key: key.to_string() }),
        ["set", key, value] => Ok(Command::Set {
            key: key.to_string(),
            value: value.to_string(),
            ttl: None,
        }),
        ["set", key, value, "ttl", ttl_str] => {
            let ttl = ttl_str.parse::<u64>()
                .map_err(|_| ParseError::InvalidTtl)?;
            Ok(Command::Set {
                key: key.to_string(),
                value: value.to_string(),
                ttl: Some(ttl),
            })
        }
        ["del", keys @ ..] if !keys.is_empty() => Ok(Command::Delete {
            keys: keys.iter().map(|s| s.to_string()).collect(),
        }),
        ["list"] => Ok(Command::List { pattern: None, limit: 100 }),
        ["list", pattern] => Ok(Command::List {
            pattern: Some(pattern.to_string()),
            limit: 100,
        }),
        ["list", pattern, "limit", n] => {
            let limit = n.parse::<usize>()
                .map_err(|_| ParseError::InvalidLimit)?;
            Ok(Command::List {
                pattern: Some(pattern.to_string()),
                limit,
            })
        }
        ["help" | "?"] => Ok(Command::Help),
        ["quit" | "exit" | "q"] => Ok(Command::Quit),
        [] => Err(ParseError::Empty),
        _ => Err(ParseError::Unknown(input.to_string())),
    }
}
```

### 6.2 JSONバリデーション

```python
# Python: JSON レスポンスのバリデーション
def validate_user_data(data: dict) -> list[str]:
    errors = []

    match data:
        case {"name": str(name)} if len(name) < 2:
            errors.append("Name too short")
        case {"name": str()}:
            pass  # OK
        case {"name": _}:
            errors.append("Name must be a string")
        case _:
            errors.append("Name is required")

    match data:
        case {"email": str(email)} if "@" not in email:
            errors.append("Invalid email format")
        case {"email": str()}:
            pass  # OK
        case {"email": _}:
            errors.append("Email must be a string")
        case _:
            errors.append("Email is required")

    match data:
        case {"age": int(age)} if age < 0 or age > 150:
            errors.append("Age must be between 0 and 150")
        case {"age": int()}:
            pass  # OK
        case {"age": _}:
            errors.append("Age must be an integer")
        case _:
            pass  # age is optional

    return errors
```

### 6.3 イベント処理システム

```typescript
// TypeScript: イベント駆動アーキテクチャ
import { match } from 'ts-pattern';

type DomainEvent =
    | { type: "UserCreated"; userId: string; email: string; timestamp: Date }
    | { type: "UserUpdated"; userId: string; changes: Partial<UserProfile>; timestamp: Date }
    | { type: "UserDeleted"; userId: string; reason: string; timestamp: Date }
    | { type: "OrderPlaced"; orderId: string; userId: string; items: OrderItem[]; timestamp: Date }
    | { type: "OrderShipped"; orderId: string; trackingId: string; timestamp: Date }
    | { type: "OrderCancelled"; orderId: string; reason: string; timestamp: Date }
    | { type: "PaymentReceived"; orderId: string; amount: number; currency: string; timestamp: Date }
    | { type: "PaymentFailed"; orderId: string; error: string; timestamp: Date };

async function handleEvent(event: DomainEvent): Promise<void> {
    await match(event)
        .with({ type: "UserCreated" }, async (e) => {
            await sendWelcomeEmail(e.email);
            await createUserProfile(e.userId);
            await publishToAnalytics("user_signup", { userId: e.userId });
        })
        .with({ type: "UserDeleted" }, async (e) => {
            await anonymizeUserData(e.userId);
            await cancelPendingOrders(e.userId);
            await publishToAnalytics("user_churn", { userId: e.userId, reason: e.reason });
        })
        .with({ type: "OrderPlaced" }, async (e) => {
            await reserveInventory(e.items);
            await notifyWarehouse(e.orderId);
            await sendOrderConfirmation(e.userId, e.orderId);
        })
        .with({ type: "PaymentFailed" }, async (e) => {
            await notifyPaymentTeam(e.orderId, e.error);
            await schedulePaymentRetry(e.orderId);
        })
        .otherwise(async (e) => {
            console.log(`Unhandled event: ${e.type}`);
        });
}
```

---

## まとめ

| 言語 | パターンマッチ | 網羅性チェック | 特徴 |
|------|-------------|-------------|------|
| Rust | match（式） | コンパイル時 | 最も安全、スライスパターン |
| Haskell | case / 関数定義 | コンパイル時（-Wall） | 元祖、ガード条件 |
| Scala | match（式） | sealed trait で保証 | 抽出子、表現力最大 |
| OCaml/F# | match / function | コンパイル時 | ML系の伝統 |
| Elixir | case / 関数ヘッド | なし | ピン演算子、with 式 |
| Python | match（3.10+） | なし（mypy で部分的） | 構造的パターン |
| TypeScript | switch + never | 型レベル | 判別ユニオン、ts-pattern |
| Java | switch（21+） | sealed class で保証 | パターンマッチング switch |
| Kotlin | when（式） | sealed class で保証 | 範囲、型チェック |

### パターンの種類と対応言語

| パターン種類 | Rust | Haskell | Scala | Python | TypeScript |
|------------|------|---------|-------|--------|-----------|
| リテラル | ✅ | ✅ | ✅ | ✅ | ✅ |
| 変数束縛 | ✅ | ✅ | ✅ | ✅ | ✅ |
| ワイルドカード | ✅ | ✅ | ✅ | ✅ | ✅ |
| OR パターン | ✅ | - | ✅ | ✅ | - |
| 範囲 | ✅ | - | - | - | - |
| 構造体分解 | ✅ | ✅ | ✅ | ✅ | ✅(ts-pattern) |
| ネスト | ✅ | ✅ | ✅ | ✅ | ✅(ts-pattern) |
| ガード条件 | ✅ | ✅ | ✅ | ✅ | ✅(ts-pattern) |
| スライス | ✅ | ✅ | - | ✅ | - |
| @ 束縛 | ✅ | ✅ | ✅ | ✅(as) | - |
| 参照 | ✅ | - | - | - | - |
| 抽出子 | - | - | ✅ | - | - |

---

## 次に読むべきガイド
→ [[02-error-handling.md]] — エラーハンドリング

---

## 参考文献
1. "Rust By Example: Pattern matching." doc.rust-lang.org.
2. "PEP 634: Structural Pattern Matching." python.org, 2021.
3. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.18, 2023.
4. Lipovaca, M. "Learn You a Haskell for Great Good!" Ch.4, No Starch Press, 2011.
5. Odersky, M., Spoon, L. & Venners, B. "Programming in Scala." 5th Ed, Ch.15, Artima, 2021.
6. Thomas, D. & Hunt, A. "Programming Elixir." Ch.12, Pragmatic Bookshelf, 2018.
7. "TypeScript Handbook: Narrowing." typescriptlang.org.
8. "OCaml Manual: Pattern matching." ocaml.org.
9. Van Rossum, G. "PEP 636: Structural Pattern Matching: Tutorial." python.org, 2021.
10. Jemerov, D. & Isakova, S. "Kotlin in Action." Ch.2, Manning, 2017.
