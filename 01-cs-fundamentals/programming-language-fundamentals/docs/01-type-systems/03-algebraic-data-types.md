# Algebraic Data Types (ADTs)

> ADTs are a technique for precisely modeling data through "combinations of product types (AND) and sum types (OR)." They make invalid states unrepresentable at the type level.

## Learning Objectives

- [ ] Understand the concepts of product types and sum types
- [ ] Use pattern matching in combination with ADTs effectively
- [ ] Design systems that "make invalid states unrepresentable"
- [ ] Understand the null problem and the significance of Option/Maybe types
- [ ] Apply ADTs to domain modeling in practice
- [ ] Implement recursive data types and generic ADTs
- [ ] Understand the differences in ADT support across languages


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content in [Generics and Polymorphism](./02-generics-and-polymorphism.md)

---

## 1. Product Types

```
Product Type = "A and B" (AND)
  -> Holds all fields simultaneously

  Structs / Records / Tuples / Classes

  Why "product"?
    If type A has a possible values and type B has b possible values,
    (A, B) has a * b possible values.
    Example: (Bool, Bool) = 2 * 2 = 4 possible values
        (True, True), (True, False), (False, True), (False, False)
```

### Rust

```rust
// Rust: Struct (product type)
struct User {
    name: String,      // AND
    age: u32,          // AND
    email: String,     // AND
}
// User = String * u32 * String
// Number of possible values = number of name values * number of age values * number of email values

// Tuple (unnamed product type)
let point: (f64, f64) = (3.0, 4.0);

// Tuple struct (named tuple)
struct Color(u8, u8, u8);
let red = Color(255, 0, 0);
println!("R: {}", red.0);

// Newtype pattern (single-field tuple struct)
struct UserId(u64);
struct OrderId(u64);
// UserId and OrderId are different types -> cannot be confused

fn get_user(id: UserId) -> Option<User> { /* ... */ }
fn get_order(id: OrderId) -> Option<Order> { /* ... */ }

let user_id = UserId(42);
let order_id = OrderId(42);
get_user(user_id);    // OK
// get_user(order_id); // Compile error: type mismatch

// Unit struct (no fields)
struct Marker;
// Size 0. Used as a type-level marker.

// Struct update syntax
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
    ..Config::default()  // Remaining fields use default values
};
```

### TypeScript

```typescript
// TypeScript: Interface (product type)
interface User {
    name: string;     // AND
    age: number;      // AND
    email: string;    // AND
}

// Tuple type
type Point = [number, number];
type RGB = [r: number, g: number, b: number]; // Labeled tuple

// Branded type (TypeScript version of the newtype pattern)
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
// getUser(orderId); // Type error

// Record type
type Config = {
    readonly host: string;
    readonly port: number;
    readonly maxConnections: number;
    readonly timeoutMs: number;
    readonly debug: boolean;
};

// Readonly utility type
type ReadonlyConfig = Readonly<Config>;

// Product composition via intersection types
type HasId = { id: string };
type HasTimestamps = { createdAt: Date; updatedAt: Date };
type HasSoftDelete = { deletedAt: Date | null };

type Entity = HasId & HasTimestamps;
type SoftDeletableEntity = Entity & HasSoftDelete;
```

### Haskell

```haskell
-- Haskell: Product type via data declaration
data User = User
    { userName  :: String
    , userAge   :: Int
    , userEmail :: String
    }
-- Record syntax automatically generates field accessor functions

-- Tuple
type Point = (Double, Double)

-- newtype (zero-cost type wrapper)
newtype UserId = UserId Int deriving (Eq, Ord, Show)
newtype OrderId = OrderId Int deriving (Eq, Ord, Show)
-- newtype has no wrapping at runtime (distinction is compile-time only)
```

### Number of Values and Information Content of Product Types

```
Type algebra:

  Void (no values)      = 0
  Unit / ()             = 1
  Bool                  = 2
  u8 / byte             = 256
  (Bool, Bool)          = 2 * 2 = 4
  (Bool, u8)            = 2 * 256 = 512
  (u8, u8)              = 256 * 256 = 65,536
  (Bool, Bool, Bool)    = 2 * 2 * 2 = 8

  Information content (bits) of a product type:
    log2(a * b) = log2(a) + log2(b)
    In other words, adding fields "adds" information content.
```

---

## 2. Sum Types (Tagged Unions)

```
Sum Type = "A or B" (OR)
  -> Holds exactly one of several alternatives

  Enums / Union types / Variants

  Why "sum"?
    If type A has a possible values and type B has b possible values,
    A | B has a + b possible values.
    Example: Bool | Unit = 2 + 1 = 3 possible values
        True, False, ()
```

### Rust

```rust
// Rust: enum (the most refined form of sum types)
enum Shape {
    Circle(f64),                    // radius
    Rectangle(f64, f64),            // width, height
    Triangle(f64, f64, f64),        // 3 sides
}
// Shape = Circle(f64) + Rectangle(f64, f64) + Triangle(f64, f64, f64)
// Exactly one of these

// Option: Representing "presence or absence of a value" at the type level
enum Option<T> {
    Some(T),
    None,
}

// Result: Representing "success or failure" at the type level
enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Variants with named fields
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

// Recursive sum type (tree structure)
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

// JSON value representation
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
// TypeScript: Union type + discriminant field
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; a: number; b: number; c: number };

// Discriminated union
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

// Event type definition
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

// JSON value type definition
type JsonValue =
    | null
    | boolean
    | number
    | string
    | JsonValue[]
    | { [key: string]: JsonValue };

// Recursive expression tree (AST)
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
-- Haskell: data declaration (the original home of ADTs)
data Shape
    = Circle Double
    | Rectangle Double Double
    | Triangle Double Double Double

area :: Shape -> Double
area (Circle r)        = pi * r * r
area (Rectangle w h)   = w * h
area (Triangle a b c)  = let s = (a + b + c) / 2
                          in sqrt (s * (s-a) * (s-b) * (s-c))

-- Recursive data type
data List a = Nil | Cons a (List a)
  deriving (Show, Eq)

-- Usage example
myList :: List Int
myList = Cons 1 (Cons 2 (Cons 3 Nil))

-- Length of a list
length' :: List a -> Int
length' Nil         = 0
length' (Cons _ xs) = 1 + length' xs

-- Folding a list
foldList :: (a -> b -> b) -> b -> List a -> b
foldList _ acc Nil         = acc
foldList f acc (Cons x xs) = f x (foldList f acc xs)

-- Binary tree
data Tree a
    = Empty
    | Branch (Tree a) a (Tree a)
  deriving (Show, Eq)

-- Tree insertion
insert :: (Ord a) => a -> Tree a -> Tree a
insert x Empty = Branch Empty x Empty
insert x (Branch left val right)
    | x < val   = Branch (insert x left) val right
    | x > val   = Branch left val (insert x right)
    | otherwise  = Branch left val right

-- Tree search
search :: (Ord a) => a -> Tree a -> Bool
search _ Empty = False
search x (Branch left val right)
    | x == val  = True
    | x < val   = search x left
    | otherwise  = search x right

-- JSON value
data JsonValue
    = JsonNull
    | JsonBool Bool
    | JsonNumber Double
    | JsonString String
    | JsonArray [JsonValue]
    | JsonObject [(String, JsonValue)]
  deriving (Show, Eq)

-- JSON display
showJson :: JsonValue -> String
showJson JsonNull        = "null"
showJson (JsonBool True) = "true"
showJson (JsonBool False) = "false"
showJson (JsonNumber n)  = show n
showJson (JsonString s)  = "\"" ++ s ++ "\""
showJson (JsonArray xs)  = "[" ++ intercalate ", " (map showJson xs) ++ "]"
showJson (JsonObject ps) = "{" ++ intercalate ", " (map showPair ps) ++ "}"
  where showPair (k, v) = "\"" ++ k ++ "\": " ++ showJson v

-- Either: Holds one of two types
data Either a b = Left a | Right b

-- Convention: Left is error, Right is success
safeDivide :: Double -> Double -> Either String Double
safeDivide _ 0 = Left "Division by zero"
safeDivide x y = Right (x / y)
```

### Go

```go
// Go: Simulating sum types with interfaces (sealed interface pattern)
type Shape interface {
    isShape()  // Private method prevents external implementation
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

// Type switch (alternative to pattern matching)
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

## 3. Pattern Matching

### Advanced Pattern Matching in Rust

```rust
// Rust: Exhaustive pattern matching with match
fn describe(shape: &Shape) -> String {
    match shape {
        Shape::Circle(r) => format!("Circle with radius {}", r),
        Shape::Rectangle(w, h) => format!("{}x{} rectangle", w, h),
        Shape::Triangle(a, b, c) => format!("Triangle ({}, {}, {})", a, b, c),
    }
    // Compile error if not all variants are handled (exhaustiveness check)
}

// Pattern matching on Option
fn greet(name: Option<&str>) -> String {
    match name {
        Some(n) => format!("Hello, {}!", n),
        None => "Hello, stranger!".to_string(),
    }
}

// if let (single pattern)
if let Some(name) = get_name() {
    println!("Found: {}", name);
}

// while let (combining loops and pattern matching)
let mut stack = vec![1, 2, 3];
while let Some(top) = stack.pop() {
    println!("{}", top);
}

// let else (Rust 1.65+)
fn process_input(input: &str) -> Result<u32, String> {
    let Ok(number) = input.parse::<u32>() else {
        return Err(format!("Failed to parse: {}", input));
    };
    Ok(number * 2)
}

// Nested patterns
match result {
    Ok(Some(value)) if value > 0 => println!("Positive: {}", value),
    Ok(Some(value)) => println!("Non-positive: {}", value),
    Ok(None) => println!("No value"),
    Err(e) => println!("Error: {}", e),
}

// Struct pattern matching
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

// Range patterns
fn classify_age(age: u32) -> &'static str {
    match age {
        0..=2 => "infant",
        3..=5 => "toddler",
        6..=11 => "elementary school",
        12..=14 => "middle school",
        15..=17 => "high school",
        18..=64 => "adult",
        65.. => "senior",
    }
}

// OR patterns
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U')
}

// Binding (@ pattern)
fn classify_number(n: i32) -> String {
    match n {
        n @ 1..=9 => format!("small positive: {}", n),
        n @ 10..=99 => format!("medium positive: {}", n),
        n @ 100.. => format!("large positive: {}", n),
        0 => "zero".to_string(),
        n => format!("negative: {}", n),
    }
}

// Slice patterns
fn describe_slice(slice: &[i32]) -> String {
    match slice {
        [] => "empty".to_string(),
        [x] => format!("single: {}", x),
        [first, .., last] => format!("from {} to {}", first, last),
    }
}

// Reference patterns
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

### Exhaustiveness Checking in TypeScript

```typescript
// TypeScript: Exhaustiveness check
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; a: number; b: number; c: number };

// Exhaustiveness check using never
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
            return assertNever(shape); // Compile error if a new kind is added
    }
}

// Type guard functions
function isCircle(shape: Shape): shape is Extract<Shape, { kind: "circle" }> {
    return shape.kind === "circle";
}

function isRectangle(shape: Shape): shape is Extract<Shape, { kind: "rectangle" }> {
    return shape.kind === "rectangle";
}

// Custom type guard
function hasLength(value: unknown): value is { length: number } {
    return typeof value === "object" && value !== null && "length" in value;
}

// Type narrowing with the in operator
type Fish = { swim: () => void };
type Bird = { fly: () => void };
type Pet = Fish | Bird;

function move(pet: Pet): void {
    if ("swim" in pet) {
        pet.swim(); // Treated as Fish
    } else {
        pet.fly();  // Treated as Bird
    }
}
```

### Pattern Matching in Haskell

```haskell
-- Haskell: Advanced pattern matching

-- Guard conditions
bmi :: Double -> String
bmi x
    | x < 18.5  = "Underweight"
    | x < 25.0  = "Normal weight"
    | x < 30.0  = "Overweight"
    | otherwise  = "Obese"

-- As pattern (bind both the whole and parts simultaneously)
firstLetter :: String -> String
firstLetter ""         = "Empty string"
firstLetter all@(x:_)  = "'" ++ all ++ "' starts with '" ++ [x] ++ "'"

-- View patterns (GHC extension)
-- {-# LANGUAGE ViewPatterns #-}
-- isEven :: Int -> Bool
-- process (isEven -> True) = "Even"
-- process _                = "Odd"

-- case expression
describeList :: [a] -> String
describeList xs = "The list is " ++ case xs of
    []  -> "empty."
    [_] -> "a singleton."
    _   -> "a longer list of " ++ show (length xs) ++ " elements."

-- Combining with where clause
calcTriangleType :: Double -> Double -> Double -> String
calcTriangleType a b c
    | a == b && b == c = "Equilateral triangle"
    | a == b || b == c || a == c = "Isosceles triangle"
    | isRight = "Right triangle"
    | otherwise = "Scalene triangle"
  where
    sides = sort [a, b, c]
    isRight = abs (sides!!0^2 + sides!!1^2 - sides!!2^2) < 1e-10
```

---

## 4. "Making Invalid States Unrepresentable"

### Bad Design (Invalid States Are Possible)

```typescript
// Invalid states are representable
interface Connection {
    status: "disconnected" | "connecting" | "connected" | "error";
    socket?: WebSocket;       // Only exists when connected
    error?: Error;            // Only exists when error
    retryCount?: number;      // Only meaningful when error
}

// Problem: A state where status is "disconnected" but socket exists can be created
const invalid: Connection = {
    status: "disconnected",
    socket: new WebSocket("ws://..."),  // Invalid, but no type error
};

// Another problem: Combinatorial explosion of optional fields
// socket presence * error presence * retryCount presence = 8 combinations
// Only 4 are valid -> 4 combinations are invalid states
```

### Good Design (Invalid States Are Unrepresentable)

```typescript
// Invalid states cannot be expressed at the type level
type Connection =
    | { status: "disconnected" }
    | { status: "connecting" }
    | { status: "connected"; socket: WebSocket }
    | { status: "error"; error: Error; retryCount: number };

// Adding a socket to status: "disconnected" is impossible at the type level

// State transition functions are also type-safe
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
// Rust: Strict state transition modeling with enum
enum ConnectionState {
    Disconnected,
    Connecting,
    Connected { socket: TcpStream },
    Error { error: io::Error, retry_count: u32 },
}

// Socket can only be accessed in the Connected state
fn send_data(state: &ConnectionState, data: &[u8]) -> Result<(), String> {
    match state {
        ConnectionState::Connected { socket } => {
            // Safely use socket
            Ok(())
        }
        _ => Err("Not connected".to_string()),
    }
}

// Even stricter representation using the typestate pattern
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
// send can only be called in the Connected state -> guaranteed at compile time
```

### Practical Example: API Response

```typescript
// Representing loading state with ADTs
type AsyncData<T, E = Error> =
    | { state: "idle" }
    | { state: "loading"; abortController?: AbortController }
    | { state: "success"; data: T; fetchedAt: Date }
    | { state: "error"; error: E; retryCount: number };

// Helper functions
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

// Usage in a React component
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

// Map function
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

### Practical Example: Form Validation

```typescript
// Form field state
type FieldState<T> =
    | { status: "pristine" }
    | { status: "touched"; value: T }
    | { status: "valid"; value: T }
    | { status: "invalid"; value: T; errors: string[] };

// Overall form state
type FormState<T extends Record<string, unknown>> = {
    fields: { [K in keyof T]: FieldState<T[K]> };
    submitted: boolean;
};

// Determine whether the form can be submitted
function canSubmit<T extends Record<string, unknown>>(form: FormState<T>): boolean {
    return Object.values(form.fields).every(
        (field) => (field as FieldState<unknown>).status === "valid"
    );
}

// Validation rule ADT
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

### Practical Example: Permission Model

```rust
// Rust: Permission model with ADTs
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

## 5. The Null Problem and Option/Maybe

```
"The Billion Dollar Mistake" (Tony Hoare, inventor of null)

Problem: null is a hole in the type system
  Java:    String name = null;  // A breeding ground for NullPointerException
  JS:      let x = null;       // TypeError: Cannot read property ...
  C:       char *p = NULL;     // Segmentation fault

Solution: Option type / Maybe type
  Rust:    Option<String>  -> Some("Gaku") | None
  Haskell: Maybe String    -> Just "Gaku" | Nothing
  Swift:   String?         -> "Gaku" | nil
  Scala:   Option[String]  -> Some("Gaku") | None
  Kotlin:  String?         -> "Gaku" | null (tracked by the compiler)

  Explicitly indicate the absence of a value at the type level,
  and force safe handling via pattern matching.
```

### Rust's Option in Detail

```rust
// Rust: No null. Use Option to be explicit.
fn find_user(id: u32) -> Option<User> {
    if id == 1 {
        Some(User { name: "Gaku".into(), age: 30 })
    } else {
        None
    }
}

// The caller must always handle the None possibility
match find_user(1) {
    Some(user) => println!("Found: {}", user.name),
    None => println!("Not found"),
}

// Method chaining
let name = find_user(1)
    .map(|u| u.name)
    .unwrap_or("Unknown".into());

// ? operator (error propagation)
fn get_user_name(id: u32) -> Option<String> {
    let user = find_user(id)?;  // Early return if None
    Some(user.name)
}

// Option combinator methods
fn process_user(id: u32) {
    let user = find_user(id);

    // map: Transform the contents of Some
    let name: Option<String> = user.as_ref().map(|u| u.name.clone());

    // and_then (flatMap): Flatten nested Options
    let email: Option<String> = find_user(id)
        .and_then(|u| find_email(u.id));

    // or_else: Alternative when None
    let backup_user: Option<User> = find_user(id)
        .or_else(|| find_user_by_name("default"));

    // filter: Returns None if condition is not met
    let adult: Option<User> = find_user(id)
        .filter(|u| u.age >= 18);

    // zip: Combine two Options
    let pair: Option<(User, Config)> = find_user(id)
        .zip(load_config());

    // unwrap_or_else: Generate a value via closure when None
    let user_or_default: User = find_user(id)
        .unwrap_or_else(|| User::default());

    // is_some, is_none: Existence check
    if find_user(id).is_some() {
        println!("User exists");
    }
}

// Flattening Option<Option<T>>
fn find_setting(key: &str) -> Option<Option<String>> {
    // Setting key does not exist -> None
    // Setting key exists but value is empty -> Some(None)
    // Setting key exists and has a value -> Some(Some(value))
    todo!()
}

let flat: Option<String> = find_setting("key").flatten();
```

### Combining with Result

```rust
// Converting between Option and Result
fn find_user_or_error(id: u32) -> Result<User, String> {
    find_user(id).ok_or(format!("User {} not found", id))
}

fn try_find_user(id: u32) -> Option<User> {
    query_database(id).ok() // Result<User, DbError> -> Option<User>
}

// Pattern for combining multiple Option/Result values
fn create_full_profile(user_id: u32) -> Result<FullProfile, String> {
    let user = find_user(user_id)
        .ok_or("User not found")?;
    let address = find_address(user_id)
        .ok_or("Address not found")?;
    let preferences = find_preferences(user_id)
        .unwrap_or_default();

    Ok(FullProfile { user, address, preferences })
}

// collect to convert Vec<Result<T, E>> -> Result<Vec<T>, E>
fn parse_all_numbers(inputs: &[&str]) -> Result<Vec<i32>, std::num::ParseIntError> {
    inputs.iter()
        .map(|s| s.parse::<i32>())
        .collect()
}
```

### Null Safety Across Languages

```kotlin
// Kotlin: Null safety
fun findUser(id: Int): User? {
    return if (id == 1) User("Gaku", 30) else null
}

// Safe call operator (?.)
val name: String? = findUser(1)?.name

// Elvis operator (?:)
val nameOrDefault: String = findUser(1)?.name ?: "Unknown"

// Non-null assertion (!!) -- dangerous
val forceUnwrap: String = findUser(1)!!.name // Potential NullPointerException

// Scope limitation with let
findUser(1)?.let { user ->
    println("Found: ${user.name}")
    println("Age: ${user.age}")
}

// Smart cast
fun processUser(user: User?) {
    if (user != null) {
        // Here user is treated as User (non-null)
        println(user.name)
    }
}
```

```swift
// Swift: Optional
func findUser(id: Int) -> User? {
    return id == 1 ? User(name: "Gaku", age: 30) : nil
}

// Optional binding
if let user = findUser(id: 1) {
    print("Found: \(user.name)")
}

// guard let (early return)
func processUser(id: Int) -> String {
    guard let user = findUser(id: id) else {
        return "Not found"
    }
    return "User: \(user.name)"
}

// Optional chaining
let name = findUser(id: 1)?.name

// Nil coalescing operator
let nameOrDefault = findUser(id: 1)?.name ?? "Unknown"

// map / flatMap
let uppercaseName = findUser(id: 1).map { $0.name.uppercased() }
```

---

## 6. Recursive Data Types

```rust
// Rust: Linked list
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

// Expression tree (core of an interpreter)
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

## 7. Generic ADTs

```haskell
-- Haskell: Generic ADTs as functors
data Tree a = Leaf | Node (Tree a) a (Tree a)

-- Functor instance
instance Functor Tree where
    fmap _ Leaf         = Leaf
    fmap f (Node l x r) = Node (fmap f l) (f x) (fmap f r)

-- Usage: Double all elements in a tree
doubleTree :: Tree Int -> Tree Int
doubleTree = fmap (* 2)

-- Foldable instance
instance Foldable Tree where
    foldMap _ Leaf         = mempty
    foldMap f (Node l x r) = foldMap f l <> f x <> foldMap f r

-- Sum of tree elements
sumTree :: Tree Int -> Int
sumTree = sum  -- Works directly thanks to Foldable

-- Free monad (the ultimate form of ADTs)
data Free f a = Pure a | Free (f (Free f a))

-- Building a DSL with the Free monad
data ConsoleF next
    = ReadLine (String -> next)
    | PrintLine String next

type Console = Free ConsoleF

readLine :: Console String
readLine = Free (ReadLine Pure)

printLine :: String -> Console ()
printLine msg = Free (PrintLine msg (Pure ()))

-- DSL usage example
greetProgram :: Console ()
greetProgram = do
    printLine "What is your name?"
    name <- readLine
    printLine ("Hello, " ++ name ++ "!")
```

```typescript
// TypeScript: Generic ADTs
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

// Usage
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

## Practical Exercises

### Exercise 1: [Basics] -- Modeling a Traffic Light with ADTs
Implement traffic light states (red, yellow, green) using Rust enums or TypeScript discriminated unions. Give each state a duration and implement a transition function to the next state.

### Exercise 2: [Basics] -- Type Definition for a JSON Parser
Define JSON values (null, bool, number, string, array, object) as ADTs and implement the following functions:
- `stringify`: JsonValue -> String
- `get`: JsonValue -> path -> Option<JsonValue>
- `merge`: JsonValue -> JsonValue -> JsonValue

### Exercise 3: [Intermediate] -- State Machine Implementation
Implement HTTP request state transitions (Idle -> Sending -> Success/Error -> Idle) with ADTs, preventing invalid transitions at the type level. Include retry logic.

### Exercise 4: [Intermediate] -- Expression Evaluator
Implement an interpreter for a small expression language supporting arithmetic, variables, and let bindings using ADTs.

### Exercise 5: [Advanced] -- Type-Safe State Machine Library
Use Rust's typestate pattern to implement a general-purpose state machine library that can represent arbitrary state transition diagrams at the type level.


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Configuration file issues | Verify the configuration file path and format |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access permissions | Verify the executing user's permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Procedure

1. **Check the error message**: Read the stack trace and identify where the error occurs
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use log output and debuggers to verify hypotheses
5. **Fix and regression test**: After fixing, also run tests for related areas

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function inputs and outputs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Diagnostic steps when performance issues occur:

1. **Identify the bottleneck**: Measure using profiling tools
2. **Check memory usage**: Verify presence of memory leaks
3. **Check I/O waits**: Examine disk and network I/O conditions
4. **Check concurrent connections**: Verify connection pool status

| Issue Type | Diagnostic Tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper release of references |
| I/O bottleneck | strace, iostat | Asynchronous I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes criteria for making technology choices.

| Criterion | When Prioritized | When Compromisable |
|---------|------------|-------------|
| Performance | Real-time processing, large-scale data | Admin dashboards, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expecting growth | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Choosing Architecture Patterns

```
+---------------------------------------------------+
|          Architecture Selection Flow               |
+---------------------------------------------------+
|                                                    |
|  (1) Team size?                                    |
|    +-- Small (1-5) -> Monolith                     |
|    +-- Large (10+) -> Go to (2)                    |
|                                                    |
|  (2) Deployment frequency?                         |
|    +-- Once a week or less -> Monolith + modules   |
|    +-- Daily / multiple times -> Go to (3)         |
|                                                    |
|  (3) Independence between teams?                   |
|    +-- High -> Microservices                       |
|    +-- Medium -> Modular monolith                  |
|                                                    |
+---------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs. Long-term Cost**
- A fast short-term approach may become technical debt in the long run
- Conversely, over-engineering has high short-term costs and can cause project delays

**2. Consistency vs. Flexibility**
- A unified technology stack has lower learning costs
- Adopting diverse technologies allows best-fit choices but increases operational costs

**3. Level of Abstraction**
- High abstraction offers high reusability but can make debugging more difficult
- Low abstraction is intuitive but prone to code duplication

```python
# Design decision recording template
class ArchitectureDecisionRecord:
    """Creating an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe background and challenges"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decision"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """Add a consequence"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """Add a rejected alternative"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Output in Markdown format"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Background\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "+" if c['type'] == 'positive' else "!"
            md += f"- [{icon}] {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this used in professional practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Description | Example |
|------|------|------|
| Product type | A and B (all fields) | struct, interface |
| Sum type | A or B (exactly one) | enum, union type |
| Pattern matching | Exhaustive branching | match, switch |
| Option/Maybe | Safe alternative to null | Some/None |
| State modeling | Prevent invalid states via types | Discriminated unions |
| Newtype | Type-safe wrapper | newtype, branded type |
| Recursive ADT | Self-referencing data structures | List, Tree, Expr |
| Generic ADT | ADTs with type parameters | Tree<T>, Result<T,E> |

| Language | Sum Type Support | Pattern Matching | Exhaustiveness Check |
|------|-------------|-------------|-------------|
| Rust | enum (best-in-class) | match, if let | Compile-time |
| Haskell | data (best-in-class) | case, function definitions | Compile-time (warning) |
| TypeScript | Discriminated unions | switch + type guards | never check |
| Kotlin | sealed class | when | Compile-time |
| Swift | enum + associated values | switch | Compile-time |
| Scala | sealed trait | match | Compile-time (warning) |
| Go | Interface + type switch | switch v.(type) | None |
| Java | sealed interface (17+) | switch (21+ patterns) | Compile-time |
| Python | dataclass + Union | match (3.10+) | None |

---

## Recommended Next Guides

---

## References
1. Pierce, B. "Types and Programming Languages." MIT Press, 2002.
2. Wlaschin, S. "Domain Modeling Made Functional." Pragmatic Bookshelf, 2018.
3. Swierstra, W. "Data Types a la Carte." JFP, 2008.
4. Yorgey, B. "The Typeclassopedia." The Monad.Reader, 2009.
5. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.6 (Enums and Pattern Matching), 2023.
6. Hoare, C.A.R. "Null References: The Billion Dollar Mistake." QCon, 2009.
7. Bloch, J. "Effective Java." 3rd Ed, Item 55 (Return optionals judiciously), 2018.
8. Rust RFC 2005: "Match Ergonomics." 2017.
