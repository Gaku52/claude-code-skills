# 代数的データ型（Algebraic Data Types）

> ADT は「直積型（AND）と直和型（OR）の組み合わせ」でデータを正確にモデリングする手法。不正な状態を型で表現不可能にする。

## この章で学ぶこと

- [ ] 直積型と直和型の概念を理解する
- [ ] パターンマッチとの組み合わせを活用できる
- [ ] 「不正な状態を表現不可能にする」設計ができる
- [ ] Null 問題と Option/Maybe 型の意義を理解する
- [ ] 実務でのドメインモデリングに ADT を活用できる
- [ ] 再帰的データ型とジェネリック ADT を実装できる
- [ ] 各言語の ADT サポートの違いを把握する

---

## 1. 直積型（Product Types）

```
直積型 = 「AかつB」（AND）
  → 全てのフィールドを同時に持つ

  構造体 / レコード / タプル / クラス

  なぜ「積」と呼ぶか:
    型 A が a 通り、型 B が b 通りの値を持つとき
    (A, B) は a × b 通りの値を持つ
    例: (Bool, Bool) = 2 × 2 = 4 通り
        (True, True), (True, False), (False, True), (False, False)
```

### Rust

```rust
// Rust: 構造体（直積型）
struct User {
    name: String,      // AND
    age: u32,          // AND
    email: String,     // AND
}
// User = String × u32 × String
// 取りうる値の数 = name の値の数 × age の値の数 × email の値の数

// タプル（無名の直積型）
let point: (f64, f64) = (3.0, 4.0);

// タプル構造体（名前付きタプル）
struct Color(u8, u8, u8);
let red = Color(255, 0, 0);
println!("R: {}", red.0);

// ニュータイプパターン（単一フィールドのタプル構造体）
struct UserId(u64);
struct OrderId(u64);
// UserId と OrderId は異なる型 → 混同できない

fn get_user(id: UserId) -> Option<User> { /* ... */ }
fn get_order(id: OrderId) -> Option<Order> { /* ... */ }

let user_id = UserId(42);
let order_id = OrderId(42);
get_user(user_id);    // OK
// get_user(order_id); // コンパイルエラー: 型が違う

// ユニット構造体（フィールドなし）
struct Marker;
// サイズ 0。型レベルのマーカーとして使う

// 構造体の更新構文
struct Config {
    host: String,
    port: u16,
    max_connections: u32,
    timeout_ms: u64,
    debug: bool,
}

impl Config {
    fn default() -> Self {
        Config {
            host: "localhost".to_string(),
            port: 8080,
            max_connections: 100,
            timeout_ms: 5000,
            debug: false,
        }
    }
}

let config = Config {
    port: 3000,
    debug: true,
    ..Config::default()  // 残りのフィールドはデフォルト値
};
```

### TypeScript

```typescript
// TypeScript: インターフェース（直積型）
interface User {
    name: string;     // AND
    age: number;      // AND
    email: string;    // AND
}

// タプル型
type Point = [number, number];
type RGB = [r: number, g: number, b: number]; // ラベル付きタプル

// ブランド型（ニュータイプパターンの TypeScript 版）
type UserId = string & { readonly __brand: unique symbol };
type OrderId = string & { readonly __brand: unique symbol };

function createUserId(id: string): UserId {
    return id as UserId;
}

function createOrderId(id: string): OrderId {
    return id as OrderId;
}

function getUser(id: UserId): User | null { /* ... */ return null; }
function getOrder(id: OrderId): Order | null { /* ... */ return null; }

const userId = createUserId("u-123");
const orderId = createOrderId("o-456");
getUser(userId);    // OK
// getUser(orderId); // 型エラー

// レコード型
type Config = {
    readonly host: string;
    readonly port: number;
    readonly maxConnections: number;
    readonly timeoutMs: number;
    readonly debug: boolean;
};

// Readonly ユーティリティ型
type ReadonlyConfig = Readonly<Config>;

// 交差型による直積の合成
type HasId = { id: string };
type HasTimestamps = { createdAt: Date; updatedAt: Date };
type HasSoftDelete = { deletedAt: Date | null };

type Entity = HasId & HasTimestamps;
type SoftDeletableEntity = Entity & HasSoftDelete;
```

### Haskell

```haskell
-- Haskell: data 宣言による直積型
data User = User
    { userName  :: String
    , userAge   :: Int
    , userEmail :: String
    }
-- レコード構文で自動的にフィールドアクセサ関数が生成される

-- タプル
type Point = (Double, Double)

-- newtype（ゼロコストの型ラッパー）
newtype UserId = UserId Int deriving (Eq, Ord, Show)
newtype OrderId = OrderId Int deriving (Eq, Ord, Show)
-- newtype は実行時にはラップなし（コンパイル時のみの区別）
```

### 直積型の値の数と情報量

```
型の代数:

  Void（値なし）      = 0
  Unit / ()           = 1
  Bool                = 2
  u8 / byte           = 256
  (Bool, Bool)        = 2 × 2 = 4
  (Bool, u8)          = 2 × 256 = 512
  (u8, u8)            = 256 × 256 = 65,536
  (Bool, Bool, Bool)  = 2 × 2 × 2 = 8

  直積型の情報量（ビット数）:
    log₂(a × b) = log₂(a) + log₂(b)
    つまり、フィールドを増やすと情報量が「加算」される
```

---

## 2. 直和型（Sum Types / Tagged Unions）

```
直和型 = 「AまたはB」（OR）
  → 複数の候補のうち1つだけを持つ

  列挙型 / ユニオン型 / バリアント

  なぜ「和」と呼ぶか:
    型 A が a 通り、型 B が b 通りの値を持つとき
    A | B は a + b 通りの値を持つ
    例: Bool | Unit = 2 + 1 = 3 通り
        True, False, ()
```

### Rust

```rust
// Rust: enum（直和型の最も洗練された形）
enum Shape {
    Circle(f64),                    // 半径
    Rectangle(f64, f64),            // 幅, 高さ
    Triangle(f64, f64, f64),        // 3辺
}
// Shape = Circle(f64) + Rectangle(f64, f64) + Triangle(f64, f64, f64)
// いずれか1つだけ

// Option: 「値があるかないか」を型で表現
enum Option<T> {
    Some(T),
    None,
}

// Result: 「成功か失敗か」を型で表現
enum Result<T, E> {
    Ok(T),
    Err(E),
}

// 名前付きフィールドを持つバリアント
enum Event {
    Click { x: f64, y: f64, button: MouseButton },
    KeyPress { key: char, modifiers: Modifiers },
    Scroll { delta_x: f64, delta_y: f64 },
    Resize { width: u32, height: u32 },
    Close,
}

enum MouseButton {
    Left,
    Right,
    Middle,
}

struct Modifiers {
    shift: bool,
    ctrl: bool,
    alt: bool,
    meta: bool,
}

// 再帰的な直和型（木構造）
enum BinaryTree<T> {
    Leaf(T),
    Node {
        left: Box<BinaryTree<T>>,
        value: T,
        right: Box<BinaryTree<T>>,
    },
}

impl<T: Ord + Clone> BinaryTree<T> {
    fn insert(self, new_value: T) -> BinaryTree<T> {
        match self {
            BinaryTree::Leaf(v) => {
                if new_value < v {
                    BinaryTree::Node {
                        left: Box::new(BinaryTree::Leaf(new_value)),
                        value: v.clone(),
                        right: Box::new(BinaryTree::Leaf(v)),
                    }
                } else {
                    BinaryTree::Node {
                        left: Box::new(BinaryTree::Leaf(v.clone())),
                        value: v,
                        right: Box::new(BinaryTree::Leaf(new_value)),
                    }
                }
            }
            BinaryTree::Node { left, value, right } => {
                if new_value < value {
                    BinaryTree::Node {
                        left: Box::new(left.insert(new_value)),
                        value,
                        right,
                    }
                } else {
                    BinaryTree::Node {
                        left,
                        value,
                        right: Box::new(right.insert(new_value)),
                    }
                }
            }
        }
    }
}

// JSON 値の表現
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
}
```

### TypeScript

```typescript
// TypeScript: ユニオン型 + 判別フィールド
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; a: number; b: number; c: number };

// 判別ユニオン（Discriminated Union）
function area(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            return shape.width * shape.height;
        case "triangle": {
            const s = (shape.a + shape.b + shape.c) / 2;
            return Math.sqrt(s * (s-shape.a) * (s-shape.b) * (s-shape.c));
        }
    }
}

// イベント型の定義
type UIEvent =
    | { type: "click"; x: number; y: number; button: "left" | "right" | "middle" }
    | { type: "keypress"; key: string; modifiers: { shift: boolean; ctrl: boolean; alt: boolean } }
    | { type: "scroll"; deltaX: number; deltaY: number }
    | { type: "resize"; width: number; height: number }
    | { type: "close" };

function handleEvent(event: UIEvent): void {
    switch (event.type) {
        case "click":
            console.log(`Click at (${event.x}, ${event.y}) with ${event.button}`);
            break;
        case "keypress":
            console.log(`Key: ${event.key}`);
            break;
        case "scroll":
            console.log(`Scroll: (${event.deltaX}, ${event.deltaY})`);
            break;
        case "resize":
            console.log(`Resize to ${event.width}x${event.height}`);
            break;
        case "close":
            console.log("Window closed");
            break;
    }
}

// JSON 値の型定義
type JsonValue =
    | null
    | boolean
    | number
    | string
    | JsonValue[]
    | { [key: string]: JsonValue };

// 再帰的な式の木（AST）
type Expr =
    | { type: "literal"; value: number }
    | { type: "variable"; name: string }
    | { type: "binary"; op: "+" | "-" | "*" | "/"; left: Expr; right: Expr }
    | { type: "unary"; op: "-" | "!"; operand: Expr }
    | { type: "call"; name: string; args: Expr[] }
    | { type: "if"; condition: Expr; then: Expr; else: Expr };

function evaluate(expr: Expr, env: Record<string, number>): number {
    switch (expr.type) {
        case "literal":
            return expr.value;
        case "variable":
            if (!(expr.name in env)) throw new Error(`Undefined: ${expr.name}`);
            return env[expr.name];
        case "binary": {
            const left = evaluate(expr.left, env);
            const right = evaluate(expr.right, env);
            switch (expr.op) {
                case "+": return left + right;
                case "-": return left - right;
                case "*": return left * right;
                case "/": return left / right;
            }
        }
        case "unary": {
            const operand = evaluate(expr.operand, env);
            switch (expr.op) {
                case "-": return -operand;
                case "!": return operand === 0 ? 1 : 0;
            }
        }
        case "call":
            throw new Error("Function calls not implemented");
        case "if":
            return evaluate(expr.condition, env) !== 0
                ? evaluate(expr.then, env)
                : evaluate(expr.else, env);
    }
}
```

### Haskell

```haskell
-- Haskell: data 宣言（ADTの本家）
data Shape
    = Circle Double
    | Rectangle Double Double
    | Triangle Double Double Double

area :: Shape -> Double
area (Circle r)        = pi * r * r
area (Rectangle w h)   = w * h
area (Triangle a b c)  = let s = (a + b + c) / 2
                          in sqrt (s * (s-a) * (s-b) * (s-c))

-- 再帰的なデータ型
data List a = Nil | Cons a (List a)
  deriving (Show, Eq)

-- 使用例
myList :: List Int
myList = Cons 1 (Cons 2 (Cons 3 Nil))

-- リストの長さ
length' :: List a -> Int
length' Nil         = 0
length' (Cons _ xs) = 1 + length' xs

-- リストの畳み込み
foldList :: (a -> b -> b) -> b -> List a -> b
foldList _ acc Nil         = acc
foldList f acc (Cons x xs) = f x (foldList f acc xs)

-- 二分木
data Tree a
    = Empty
    | Branch (Tree a) a (Tree a)
  deriving (Show, Eq)

-- 木の挿入
insert :: (Ord a) => a -> Tree a -> Tree a
insert x Empty = Branch Empty x Empty
insert x (Branch left val right)
    | x < val   = Branch (insert x left) val right
    | x > val   = Branch left val (insert x right)
    | otherwise  = Branch left val right

-- 木の探索
search :: (Ord a) => a -> Tree a -> Bool
search _ Empty = False
search x (Branch left val right)
    | x == val  = True
    | x < val   = search x left
    | otherwise  = search x right

-- JSON 値
data JsonValue
    = JsonNull
    | JsonBool Bool
    | JsonNumber Double
    | JsonString String
    | JsonArray [JsonValue]
    | JsonObject [(String, JsonValue)]
  deriving (Show, Eq)

-- JSON の表示
showJson :: JsonValue -> String
showJson JsonNull        = "null"
showJson (JsonBool True) = "true"
showJson (JsonBool False) = "false"
showJson (JsonNumber n)  = show n
showJson (JsonString s)  = "\"" ++ s ++ "\""
showJson (JsonArray xs)  = "[" ++ intercalate ", " (map showJson xs) ++ "]"
showJson (JsonObject ps) = "{" ++ intercalate ", " (map showPair ps) ++ "}"
  where showPair (k, v) = "\"" ++ k ++ "\": " ++ showJson v

-- Either: 2つの型のどちらかを持つ
data Either a b = Left a | Right b

-- 慣例: Left はエラー、Right は成功
safeDivide :: Double -> Double -> Either String Double
safeDivide _ 0 = Left "Division by zero"
safeDivide x y = Right (x / y)
```

### Go

```go
// Go: インターフェースで直和型を模倣（sealed interface パターン）
type Shape interface {
    isShape()  // プライベートメソッドで外部からの実装を防止
    Area() float64
}

type Circle struct {
    Radius float64
}

type Rectangle struct {
    Width  float64
    Height float64
}

type Triangle struct {
    A, B, C float64
}

func (c Circle) isShape()    {}
func (r Rectangle) isShape() {}
func (t Triangle) isShape()  {}

func (c Circle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (t Triangle) Area() float64 {
    s := (t.A + t.B + t.C) / 2
    return math.Sqrt(s * (s - t.A) * (s - t.B) * (s - t.C))
}

// 型スイッチ（パターンマッチの代替）
func describe(s Shape) string {
    switch v := s.(type) {
    case Circle:
        return fmt.Sprintf("Circle(r=%.1f)", v.Radius)
    case Rectangle:
        return fmt.Sprintf("Rectangle(%.1f x %.1f)", v.Width, v.Height)
    case Triangle:
        return fmt.Sprintf("Triangle(%.1f, %.1f, %.1f)", v.A, v.B, v.C)
    default:
        return "Unknown shape"
    }
}
```

---

## 3. パターンマッチ

### Rust の高度なパターンマッチ

```rust
// Rust: match による網羅的パターンマッチ
fn describe(shape: &Shape) -> String {
    match shape {
        Shape::Circle(r) => format!("Circle with radius {}", r),
        Shape::Rectangle(w, h) => format!("{}x{} rectangle", w, h),
        Shape::Triangle(a, b, c) => format!("Triangle ({}, {}, {})", a, b, c),
    }
    // 全バリアントを処理しないとコンパイルエラー（網羅性チェック）
}

// Option のパターンマッチ
fn greet(name: Option<&str>) -> String {
    match name {
        Some(n) => format!("Hello, {}!", n),
        None => "Hello, stranger!".to_string(),
    }
}

// if let（単一パターン）
if let Some(name) = get_name() {
    println!("Found: {}", name);
}

// while let（ループとパターンマッチの組み合わせ）
let mut stack = vec![1, 2, 3];
while let Some(top) = stack.pop() {
    println!("{}", top);
}

// let else（Rust 1.65+）
fn process_input(input: &str) -> Result<u32, String> {
    let Ok(number) = input.parse::<u32>() else {
        return Err(format!("Failed to parse: {}", input));
    };
    Ok(number * 2)
}

// ネストしたパターン
match result {
    Ok(Some(value)) if value > 0 => println!("Positive: {}", value),
    Ok(Some(value)) => println!("Non-positive: {}", value),
    Ok(None) => println!("No value"),
    Err(e) => println!("Error: {}", e),
}

// 構造体のパターンマッチ
struct Point { x: f64, y: f64 }

fn classify_point(p: &Point) -> &str {
    match p {
        Point { x: 0.0, y: 0.0 } => "origin",
        Point { x, y: 0.0 } => "on x-axis",
        Point { x: 0.0, y } => "on y-axis",
        Point { x, y } if x == y => "on diagonal",
        _ => "elsewhere",
    }
}

// 範囲パターン
fn classify_age(age: u32) -> &'static str {
    match age {
        0..=2 => "乳児",
        3..=5 => "幼児",
        6..=11 => "小学生",
        12..=14 => "中学生",
        15..=17 => "高校生",
        18..=64 => "成人",
        65.. => "高齢者",
    }
}

// OR パターン
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U')
}

// バインディング（@ パターン）
fn classify_number(n: i32) -> String {
    match n {
        n @ 1..=9 => format!("small positive: {}", n),
        n @ 10..=99 => format!("medium positive: {}", n),
        n @ 100.. => format!("large positive: {}", n),
        0 => "zero".to_string(),
        n => format!("negative: {}", n),
    }
}

// スライスパターン
fn describe_slice(slice: &[i32]) -> String {
    match slice {
        [] => "empty".to_string(),
        [x] => format!("single: {}", x),
        [first, .., last] => format!("from {} to {}", first, last),
    }
}

// 参照パターン
fn process_references(values: &[Option<String>]) {
    for value in values {
        match value {
            Some(ref s) if s.starts_with("A") => println!("Starts with A: {}", s),
            Some(ref s) => println!("Other: {}", s),
            None => println!("Missing"),
        }
    }
}
```

### TypeScript の網羅性チェック

```typescript
// TypeScript: exhaustiveness check（網羅性チェック）
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; a: number; b: number; c: number };

// never を使った網羅性チェック
function assertNever(x: never): never {
    throw new Error(`Unexpected value: ${x}`);
}

function area(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            return shape.width * shape.height;
        case "triangle": {
            const s = (shape.a + shape.b + shape.c) / 2;
            return Math.sqrt(s * (s-shape.a) * (s-shape.b) * (s-shape.c));
        }
        default:
            return assertNever(shape); // 新しい kind を追加したらコンパイルエラー
    }
}

// 型ガード関数
function isCircle(shape: Shape): shape is Extract<Shape, { kind: "circle" }> {
    return shape.kind === "circle";
}

function isRectangle(shape: Shape): shape is Extract<Shape, { kind: "rectangle" }> {
    return shape.kind === "rectangle";
}

// カスタム型ガード
function hasLength(value: unknown): value is { length: number } {
    return typeof value === "object" && value !== null && "length" in value;
}

// in 演算子による型の絞り込み
type Fish = { swim: () => void };
type Bird = { fly: () => void };
type Pet = Fish | Bird;

function move(pet: Pet): void {
    if ("swim" in pet) {
        pet.swim(); // Fish として扱われる
    } else {
        pet.fly();  // Bird として扱われる
    }
}
```

### Haskell のパターンマッチ

```haskell
-- Haskell: 高度なパターンマッチ

-- ガード条件
bmi :: Double -> String
bmi x
    | x < 18.5  = "やせ型"
    | x < 25.0  = "普通体重"
    | x < 30.0  = "肥満(1度)"
    | otherwise  = "肥満(2度以上)"

-- as パターン（全体と部分を同時に束縛）
firstLetter :: String -> String
firstLetter ""         = "空文字列"
firstLetter all@(x:_)  = "'" ++ all ++ "' の先頭は '" ++ [x] ++ "'"

-- ビューパターン（GHC拡張）
-- {-# LANGUAGE ViewPatterns #-}
-- isEven :: Int -> Bool
-- process (isEven -> True) = "偶数"
-- process _                = "奇数"

-- case 式
describeList :: [a] -> String
describeList xs = "The list is " ++ case xs of
    []  -> "empty."
    [_] -> "a singleton."
    _   -> "a longer list of " ++ show (length xs) ++ " elements."

-- where 句との組み合わせ
calcTriangleType :: Double -> Double -> Double -> String
calcTriangleType a b c
    | a == b && b == c = "正三角形"
    | a == b || b == c || a == c = "二等辺三角形"
    | isRight = "直角三角形"
    | otherwise = "不等辺三角形"
  where
    sides = sort [a, b, c]
    isRight = abs (sides!!0^2 + sides!!1^2 - sides!!2^2) < 1e-10
```

---

## 4. 「不正な状態を表現不可能にする」

### 悪い設計（不正な状態が可能）

```typescript
// 不正な状態が表現可能
interface Connection {
    status: "disconnected" | "connecting" | "connected" | "error";
    socket?: WebSocket;       // connected の時だけ存在
    error?: Error;            // error の時だけ存在
    retryCount?: number;      // error の時だけ意味がある
}

// 問題: status: "disconnected" なのに socket がある状態が作れてしまう
const invalid: Connection = {
    status: "disconnected",
    socket: new WebSocket("ws://..."),  // 不正だが型エラーにならない
};

// 別の問題: Optional フィールドの組み合わせ爆発
// socket の有無 × error の有無 × retryCount の有無 = 8 通り
// 有効な組み合わせは 4 通りだけ → 4 通りが不正な状態
```

### 良い設計（不正な状態が表現不可能）

```typescript
// 不正な状態が型で表現できない
type Connection =
    | { status: "disconnected" }
    | { status: "connecting" }
    | { status: "connected"; socket: WebSocket }
    | { status: "error"; error: Error; retryCount: number };

// status: "disconnected" に socket を持たせることが型レベルで不可能

// 状態遷移関数も型安全に
function connect(conn: Extract<Connection, { status: "disconnected" }>): Extract<Connection, { status: "connecting" }> {
    return { status: "connecting" };
}

function onConnected(
    conn: Extract<Connection, { status: "connecting" }>,
    socket: WebSocket
): Extract<Connection, { status: "connected" }> {
    return { status: "connected", socket };
}

function onError(
    conn: Connection,
    error: Error
): Extract<Connection, { status: "error" }> {
    const retryCount = conn.status === "error" ? conn.retryCount + 1 : 0;
    return { status: "error", error, retryCount };
}
```

```rust
// Rust: enum で状態遷移を厳密にモデリング
enum ConnectionState {
    Disconnected,
    Connecting,
    Connected { socket: TcpStream },
    Error { error: io::Error, retry_count: u32 },
}

// Connected 状態でしか socket にアクセスできない
fn send_data(state: &ConnectionState, data: &[u8]) -> Result<(), String> {
    match state {
        ConnectionState::Connected { socket } => {
            // socket を安全に使用
            Ok(())
        }
        _ => Err("Not connected".to_string()),
    }
}

// 型状態パターンによるさらに厳密な表現
struct Disconnected;
struct Connecting;
struct Connected { socket: TcpStream }
struct ErrorState { error: io::Error, retry_count: u32 }

struct Connection<S> {
    state: S,
    config: ConnectionConfig,
}

impl Connection<Disconnected> {
    fn connect(self) -> Connection<Connecting> {
        Connection {
            state: Connecting,
            config: self.config,
        }
    }
}

impl Connection<Connecting> {
    fn on_connected(self, socket: TcpStream) -> Connection<Connected> {
        Connection {
            state: Connected { socket },
            config: self.config,
        }
    }

    fn on_error(self, error: io::Error) -> Connection<ErrorState> {
        Connection {
            state: ErrorState { error, retry_count: 0 },
            config: self.config,
        }
    }
}

impl Connection<Connected> {
    fn send(&mut self, data: &[u8]) -> io::Result<usize> {
        self.state.socket.write(data)
    }

    fn disconnect(self) -> Connection<Disconnected> {
        Connection {
            state: Disconnected,
            config: self.config,
        }
    }
}
// Connected 状態でのみ send が呼べる → コンパイル時に保証
```

### 実践例: API レスポンス

```typescript
// ローディング状態を ADT で表現
type AsyncData<T, E = Error> =
    | { state: "idle" }
    | { state: "loading"; abortController?: AbortController }
    | { state: "success"; data: T; fetchedAt: Date }
    | { state: "error"; error: E; retryCount: number };

// ヘルパー関数
function idle<T>(): AsyncData<T> {
    return { state: "idle" };
}

function loading<T>(abortController?: AbortController): AsyncData<T> {
    return { state: "loading", abortController };
}

function success<T>(data: T): AsyncData<T> {
    return { state: "success", data, fetchedAt: new Date() };
}

function error<T>(err: Error, retryCount: number = 0): AsyncData<T> {
    return { state: "error", error: err, retryCount };
}

// React コンポーネントでの使用
function renderUser(user: AsyncData<User>) {
    switch (user.state) {
        case "idle":
            return <div>Press load</div>;
        case "loading":
            return <Spinner />;
        case "success":
            return <UserCard user={user.data} />;
        case "error":
            return <ErrorMessage error={user.error} retryCount={user.retryCount} />;
    }
}

// マップ関数
function mapAsyncData<T, U, E = Error>(
    data: AsyncData<T, E>,
    fn: (value: T) => U
): AsyncData<U, E> {
    if (data.state === "success") {
        return { ...data, data: fn(data.data) };
    }
    return data as AsyncData<U, E>;
}
```

### 実践例: フォームバリデーション

```typescript
// フォームフィールドの状態
type FieldState<T> =
    | { status: "pristine" }
    | { status: "touched"; value: T }
    | { status: "valid"; value: T }
    | { status: "invalid"; value: T; errors: string[] };

// フォーム全体の状態
type FormState<T extends Record<string, unknown>> = {
    fields: { [K in keyof T]: FieldState<T[K]> };
    submitted: boolean;
};

// フォームがsubmit可能かどうかの判定
function canSubmit<T extends Record<string, unknown>>(form: FormState<T>): boolean {
    return Object.values(form.fields).every(
        (field) => (field as FieldState<unknown>).status === "valid"
    );
}

// バリデーション規則の ADT
type ValidationRule<T> =
    | { type: "required"; message: string }
    | { type: "minLength"; min: number; message: string }
    | { type: "maxLength"; max: number; message: string }
    | { type: "pattern"; regex: RegExp; message: string }
    | { type: "custom"; validate: (value: T) => boolean; message: string };

function validateField<T>(value: T, rules: ValidationRule<T>[]): string[] {
    const errors: string[] = [];
    for (const rule of rules) {
        switch (rule.type) {
            case "required":
                if (value === null || value === undefined || value === "") {
                    errors.push(rule.message);
                }
                break;
            case "minLength":
                if (typeof value === "string" && value.length < rule.min) {
                    errors.push(rule.message);
                }
                break;
            case "maxLength":
                if (typeof value === "string" && value.length > rule.max) {
                    errors.push(rule.message);
                }
                break;
            case "pattern":
                if (typeof value === "string" && !rule.regex.test(value)) {
                    errors.push(rule.message);
                }
                break;
            case "custom":
                if (!rule.validate(value)) {
                    errors.push(rule.message);
                }
                break;
        }
    }
    return errors;
}
```

### 実践例: 権限モデル

```rust
// Rust: ADT による権限モデル
enum Permission {
    Read,
    Write,
    Delete,
    Admin,
}

enum Role {
    Anonymous,
    User { id: UserId, permissions: Vec<Permission> },
    Moderator { id: UserId, managed_areas: Vec<String> },
    Admin { id: UserId },
}

enum AccessResult {
    Allowed,
    Denied { reason: String },
    RequiresAuthentication,
    RequiresElevation { required_role: String },
}

fn check_access(role: &Role, resource: &str, action: &Permission) -> AccessResult {
    match (role, action) {
        (Role::Admin { .. }, _) => AccessResult::Allowed,
        (Role::Anonymous, Permission::Read) => AccessResult::Allowed,
        (Role::Anonymous, _) => AccessResult::RequiresAuthentication,
        (Role::User { permissions, .. }, action) => {
            if permissions.iter().any(|p| std::mem::discriminant(p) == std::mem::discriminant(action)) {
                AccessResult::Allowed
            } else {
                AccessResult::Denied {
                    reason: format!("Insufficient permissions for {:?}", action),
                }
            }
        }
        (Role::Moderator { managed_areas, .. }, _) => {
            if managed_areas.iter().any(|a| resource.starts_with(a)) {
                AccessResult::Allowed
            } else {
                AccessResult::Denied {
                    reason: "Outside managed area".to_string(),
                }
            }
        }
    }
}
```

---

## 5. Null の問題と Option/Maybe

```
「10億ドルの間違い」（Tony Hoare, null の発明者）

問題: null は型システムの穴
  Java:    String name = null;  // NullPointerException の温床
  JS:      let x = null;       // TypeError: Cannot read property ...
  C:       char *p = NULL;     // セグメンテーションフォルト

解決: Option型 / Maybe型
  Rust:    Option<String>  → Some("Gaku") | None
  Haskell: Maybe String    → Just "Gaku" | Nothing
  Swift:   String?         → "Gaku" | nil
  Scala:   Option[String]  → Some("Gaku") | None
  Kotlin:  String?         → "Gaku" | null（コンパイラが追跡）

  値がないことを「型で明示」し、
  パターンマッチで安全に処理を強制する
```

### Rust の Option 詳細

```rust
// Rust: null がない。Option で明示
fn find_user(id: u32) -> Option<User> {
    if id == 1 {
        Some(User { name: "Gaku".into(), age: 30 })
    } else {
        None
    }
}

// 使う側は None の可能性を必ず処理する
match find_user(1) {
    Some(user) => println!("Found: {}", user.name),
    None => println!("Not found"),
}

// メソッドチェーン
let name = find_user(1)
    .map(|u| u.name)
    .unwrap_or("Unknown".into());

// ? 演算子（エラー伝播）
fn get_user_name(id: u32) -> Option<String> {
    let user = find_user(id)?;  // None なら早期リターン
    Some(user.name)
}

// Option のコンビネータ群
fn process_user(id: u32) {
    let user = find_user(id);

    // map: Some の中身を変換
    let name: Option<String> = user.as_ref().map(|u| u.name.clone());

    // and_then (flatMap): ネストした Option を平坦化
    let email: Option<String> = find_user(id)
        .and_then(|u| find_email(u.id));

    // or_else: None の場合の代替
    let backup_user: Option<User> = find_user(id)
        .or_else(|| find_user_by_name("default"));

    // filter: 条件を満たさなければ None
    let adult: Option<User> = find_user(id)
        .filter(|u| u.age >= 18);

    // zip: 2つの Option を組み合わせ
    let pair: Option<(User, Config)> = find_user(id)
        .zip(load_config());

    // unwrap_or_else: None の場合にクロージャで値を生成
    let user_or_default: User = find_user(id)
        .unwrap_or_else(|| User::default());

    // is_some, is_none: 存在チェック
    if find_user(id).is_some() {
        println!("User exists");
    }
}

// Option<Option<T>> のフラット化
fn find_setting(key: &str) -> Option<Option<String>> {
    // 設定キーが存在しない → None
    // 設定キーが存在するが値が空 → Some(None)
    // 設定キーが存在し値がある → Some(Some(value))
    todo!()
}

let flat: Option<String> = find_setting("key").flatten();
```

### Result との組み合わせ

```rust
// Option と Result の変換
fn find_user_or_error(id: u32) -> Result<User, String> {
    find_user(id).ok_or(format!("User {} not found", id))
}

fn try_find_user(id: u32) -> Option<User> {
    query_database(id).ok() // Result<User, DbError> → Option<User>
}

// 複数の Option/Result を組み合わせるパターン
fn create_full_profile(user_id: u32) -> Result<FullProfile, String> {
    let user = find_user(user_id)
        .ok_or("User not found")?;
    let address = find_address(user_id)
        .ok_or("Address not found")?;
    let preferences = find_preferences(user_id)
        .unwrap_or_default();

    Ok(FullProfile { user, address, preferences })
}

// collect で Vec<Result<T, E>> → Result<Vec<T>, E>
fn parse_all_numbers(inputs: &[&str]) -> Result<Vec<i32>, std::num::ParseIntError> {
    inputs.iter()
        .map(|s| s.parse::<i32>())
        .collect()
}
```

### 各言語の Null 安全性

```kotlin
// Kotlin: Null 安全性
fun findUser(id: Int): User? {
    return if (id == 1) User("Gaku", 30) else null
}

// 安全呼び出し演算子 (?.)
val name: String? = findUser(1)?.name

// エルビス演算子 (?:)
val nameOrDefault: String = findUser(1)?.name ?: "Unknown"

// 非 null アサーション (!!) — 危険
val forceUnwrap: String = findUser(1)!!.name // NullPointerException の可能性

// let でスコープを限定
findUser(1)?.let { user ->
    println("Found: ${user.name}")
    println("Age: ${user.age}")
}

// スマートキャスト
fun processUser(user: User?) {
    if (user != null) {
        // ここでは user は User（非null）として扱われる
        println(user.name)
    }
}
```

```swift
// Swift: Optional
func findUser(id: Int) -> User? {
    return id == 1 ? User(name: "Gaku", age: 30) : nil
}

// Optional バインディング
if let user = findUser(id: 1) {
    print("Found: \(user.name)")
}

// guard let（早期リターン）
func processUser(id: Int) -> String {
    guard let user = findUser(id: id) else {
        return "Not found"
    }
    return "User: \(user.name)"
}

// Optional チェイニング
let name = findUser(id: 1)?.name

// nil 合体演算子
let nameOrDefault = findUser(id: 1)?.name ?? "Unknown"

// map / flatMap
let uppercaseName = findUser(id: 1).map { $0.name.uppercased() }
```

---

## 6. 再帰的データ型

```rust
// Rust: リンクリスト
enum List<T> {
    Cons(T, Box<List<T>>),
    Nil,
}

impl<T: std::fmt::Display> List<T> {
    fn new() -> Self {
        List::Nil
    }

    fn prepend(self, value: T) -> Self {
        List::Cons(value, Box::new(self))
    }

    fn len(&self) -> usize {
        match self {
            List::Nil => 0,
            List::Cons(_, tail) => 1 + tail.len(),
        }
    }

    fn to_string(&self) -> String {
        match self {
            List::Nil => "Nil".to_string(),
            List::Cons(head, tail) => format!("{} -> {}", head, tail.to_string()),
        }
    }
}

let list = List::new()
    .prepend(3)
    .prepend(2)
    .prepend(1);
// 1 -> 2 -> 3 -> Nil

// 式の木（インタープリタの核心）
enum Expr {
    Num(f64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Let { name: String, value: Box<Expr>, body: Box<Expr> },
    If { cond: Box<Expr>, then_: Box<Expr>, else_: Box<Expr> },
    Lambda { param: String, body: Box<Expr> },
    Apply(Box<Expr>, Box<Expr>),
}

fn eval(expr: &Expr, env: &HashMap<String, f64>) -> Result<f64, String> {
    match expr {
        Expr::Num(n) => Ok(*n),
        Expr::Var(name) => env.get(name).copied()
            .ok_or_else(|| format!("Undefined variable: {}", name)),
        Expr::Add(left, right) => {
            Ok(eval(left, env)? + eval(right, env)?)
        }
        Expr::Mul(left, right) => {
            Ok(eval(left, env)? * eval(right, env)?)
        }
        Expr::Let { name, value, body } => {
            let val = eval(value, env)?;
            let mut new_env = env.clone();
            new_env.insert(name.clone(), val);
            eval(body, &new_env)
        }
        Expr::If { cond, then_, else_ } => {
            let c = eval(cond, env)?;
            if c != 0.0 { eval(then_, env) } else { eval(else_, env) }
        }
        _ => Err("Not implemented".to_string()),
    }
}
```

---

## 7. ジェネリック ADT

```haskell
-- Haskell: ファンクタとしてのジェネリック ADT
data Tree a = Leaf | Node (Tree a) a (Tree a)

-- Functor インスタンス
instance Functor Tree where
    fmap _ Leaf         = Leaf
    fmap f (Node l x r) = Node (fmap f l) (f x) (fmap f r)

-- 使用例: 木の全要素を2倍にする
doubleTree :: Tree Int -> Tree Int
doubleTree = fmap (* 2)

-- Foldable インスタンス
instance Foldable Tree where
    foldMap _ Leaf         = mempty
    foldMap f (Node l x r) = foldMap f l <> f x <> foldMap f r

-- 木の要素の合計
sumTree :: Tree Int -> Int
sumTree = sum  -- Foldable のおかげで sum がそのまま使える

-- Free モナド（ADT の究極形）
data Free f a = Pure a | Free (f (Free f a))

-- Free モナドで DSL を構築
data ConsoleF next
    = ReadLine (String -> next)
    | PrintLine String next

type Console = Free ConsoleF

readLine :: Console String
readLine = Free (ReadLine Pure)

printLine :: String -> Console ()
printLine msg = Free (PrintLine msg (Pure ()))

-- DSL の使用例
greetProgram :: Console ()
greetProgram = do
    printLine "What is your name?"
    name <- readLine
    printLine ("Hello, " ++ name ++ "!")
```

```typescript
// TypeScript: ジェネリック ADT
type Tree<T> =
    | { type: "leaf" }
    | { type: "node"; left: Tree<T>; value: T; right: Tree<T> };

function mapTree<T, U>(tree: Tree<T>, fn: (value: T) => U): Tree<U> {
    switch (tree.type) {
        case "leaf":
            return { type: "leaf" };
        case "node":
            return {
                type: "node",
                left: mapTree(tree.left, fn),
                value: fn(tree.value),
                right: mapTree(tree.right, fn),
            };
    }
}

function foldTree<T, U>(tree: Tree<T>, leaf: U, node: (left: U, value: T, right: U) => U): U {
    switch (tree.type) {
        case "leaf":
            return leaf;
        case "node":
            return node(
                foldTree(tree.left, leaf, node),
                tree.value,
                foldTree(tree.right, leaf, node)
            );
    }
}

// 使用例
const numTree: Tree<number> = {
    type: "node",
    left: { type: "node", left: { type: "leaf" }, value: 1, right: { type: "leaf" } },
    value: 2,
    right: { type: "node", left: { type: "leaf" }, value: 3, right: { type: "leaf" } },
};

const doubled = mapTree(numTree, x => x * 2);
const sum = foldTree(numTree, 0, (l, v, r) => l + v + r); // 6
```

---

## 実践演習

### 演習1: [基礎] -- 信号機をADTでモデリング
交通信号機の状態（赤・黄・青）を Rust の enum または TypeScript の判別ユニオンで実装する。各状態に持続時間を持たせ、次の状態への遷移関数を実装する。

### 演習2: [基礎] -- JSON パーサーの型定義
JSON の値（null, bool, number, string, array, object）を ADT で定義し、以下の関数を実装する:
- `stringify`: JsonValue → String
- `get`: JsonValue → path → Option<JsonValue>
- `merge`: JsonValue → JsonValue → JsonValue

### 演習3: [応用] -- 状態機械の実装
HTTP リクエストの状態遷移（Idle → Sending → Success/Error → Idle）を ADT で実装し、不正な遷移を型で防止する。リトライロジックも含める。

### 演習4: [応用] -- 式の評価器
四則演算・変数・let 束縛をサポートする小さな式言語のインタープリタを ADT で実装する。

### 演習5: [発展] -- 型安全なステートマシンライブラリ
Rust の型状態パターンを使って、任意の状態遷移図を型レベルで表現できる汎用的なステートマシンライブラリを実装する。

---

## まとめ

| 概念 | 説明 | 例 |
|------|------|------|
| 直積型 | A かつ B（全フィールド） | struct, interface |
| 直和型 | A または B（1つだけ） | enum, union type |
| パターンマッチ | 網羅的な分岐処理 | match, switch |
| Option/Maybe | null の安全な代替 | Some/None |
| 状態モデリング | 不正な状態を型で防止 | 判別ユニオン |
| ニュータイプ | 型安全なラッパー | newtype, ブランド型 |
| 再帰的 ADT | 自己参照するデータ構造 | List, Tree, Expr |
| ジェネリック ADT | 型パラメータ付き ADT | Tree<T>, Result<T,E> |

| 言語 | 直和型サポート | パターンマッチ | 網羅性チェック |
|------|-------------|-------------|-------------|
| Rust | enum（最高レベル） | match, if let | コンパイル時 |
| Haskell | data（最高レベル） | case, 関数定義 | コンパイル時（警告） |
| TypeScript | 判別ユニオン | switch + 型ガード | never チェック |
| Kotlin | sealed class | when | コンパイル時 |
| Swift | enum + associated values | switch | コンパイル時 |
| Scala | sealed trait | match | コンパイル時（警告） |
| Go | インターフェース + 型スイッチ | switch v.(type) | なし |
| Java | sealed interface（17+） | switch（21+ パターン） | コンパイル時 |
| Python | dataclass + Union | match（3.10+） | なし |

---

## 次に読むべきガイド
→ [[../02-memory-models/00-stack-and-heap.md]] -- メモリモデル

---

## 参考文献
1. Pierce, B. "Types and Programming Languages." MIT Press, 2002.
2. Wlaschin, S. "Domain Modeling Made Functional." Pragmatic Bookshelf, 2018.
3. Swierstra, W. "Data Types a la Carte." JFP, 2008.
4. Yorgey, B. "The Typeclassopedia." The Monad.Reader, 2009.
5. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.6 (Enums and Pattern Matching), 2023.
6. Hoare, C.A.R. "Null References: The Billion Dollar Mistake." QCon, 2009.
7. Bloch, J. "Effective Java." 3rd Ed, Item 55 (Return optionals judiciously), 2018.
8. Rust RFC 2005: "Match Ergonomics." 2017.
