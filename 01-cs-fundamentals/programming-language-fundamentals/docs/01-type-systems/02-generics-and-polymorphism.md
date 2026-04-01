# Generics and Polymorphism

> Polymorphism is the ability to "handle different types with the same code." It is the key to achieving both code reusability and type safety.

## Learning Objectives

- [ ] Understand the three kinds of polymorphism
- [ ] Master the implementation and usage of generics
- [ ] Use bounded polymorphism (type constraints) effectively
- [ ] Understand the differences in generics implementation across languages
- [ ] Grasp advanced concepts such as higher-kinded types and GADTs
- [ ] Learn patterns for effectively leveraging generics in practice


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content in [Type Inference](./01-type-inference.md)

---

## 1. Kinds of Polymorphism

```
Polymorphism = "having many forms"

  1. Parametric Polymorphism (Generics)
     -> Parameterize types to write generic code
     Example: List<T>, Map<K, V>

  2. Subtype Polymorphism (Inheritance / Interfaces)
     -> Handle subtypes through a parent type's interface
     Example: A function taking Animal can accept both Dog and Cat

  3. Ad-hoc Polymorphism (Overloading / Type Classes)
     -> Different implementations per type
     Example: + works as numeric addition and string concatenation

  4. Row Polymorphism
     -> Handle any record that has specific fields
     Example: OCaml objects, TypeScript structural subtyping

  5. Kind Polymorphism (Higher-Kinded Polymorphism)
     -> Abstract over type constructors themselves
     Example: Haskell's Functor, Monad
```

### Historical Background of Polymorphism

```
1967: Strachey distinguishes "parametric polymorphism" from "ad-hoc polymorphism"
1972: Girard proposes System F (second-order polymorphic lambda calculus)
1978: Milner realizes parametric polymorphism with type inference in ML
1984: Cardelli & Wegner formalize subtype polymorphism
1989: Wadler & Blott propose type classes (a unified framework for ad-hoc polymorphism)
2004: Java 5 introduces generics
2012: Rust's trait system matures
2022: Go 1.18 introduces generics
```

---

## 2. Parametric Polymorphism (Generics)

### TypeScript

```typescript
// Generic function
function identity<T>(value: T): T {
    return value;
}

identity<number>(42);     // T = number
identity<string>("hello"); // T = string
identity(42);             // T = number (type inference)

// Generic type
interface Box<T> {
    value: T;
    map<U>(fn: (v: T) => U): Box<U>;
}

function createBox<T>(value: T): Box<T> {
    return {
        value,
        map: (fn) => createBox(fn(value)),
    };
}

const numBox = createBox(42);
const strBox = numBox.map(n => n.toString());  // Box<string>

// Multiple type parameters
function zip<A, B>(a: A[], b: B[]): [A, B][] {
    return a.map((val, i) => [val, b[i]]);
}

zip([1, 2], ["a", "b"]);  // [[1, "a"], [2, "b"]]

// Generic class
class Stack<T> {
    private items: T[] = [];

    push(item: T): void {
        this.items.push(item);
    }

    pop(): T | undefined {
        return this.items.pop();
    }

    peek(): T | undefined {
        return this.items[this.items.length - 1];
    }

    isEmpty(): boolean {
        return this.items.length === 0;
    }

    size(): number {
        return this.items.length;
    }
}

const numStack = new Stack<number>();
numStack.push(1);
numStack.push(2);
numStack.pop(); // 2 (type is number | undefined)

// Practical generic interface example: Repository pattern
interface Repository<T, ID> {
    findById(id: ID): Promise<T | null>;
    findAll(): Promise<T[]>;
    save(entity: T): Promise<T>;
    delete(id: ID): Promise<boolean>;
    count(): Promise<number>;
}

interface User {
    id: string;
    name: string;
    email: string;
}

class UserRepository implements Repository<User, string> {
    private users: Map<string, User> = new Map();

    async findById(id: string): Promise<User | null> {
        return this.users.get(id) ?? null;
    }

    async findAll(): Promise<User[]> {
        return Array.from(this.users.values());
    }

    async save(user: User): Promise<User> {
        this.users.set(user.id, user);
        return user;
    }

    async delete(id: string): Promise<boolean> {
        return this.users.delete(id);
    }

    async count(): Promise<number> {
        return this.users.size;
    }
}

// Generic utility functions
function groupBy<T, K extends string | number>(
    items: T[],
    keyFn: (item: T) => K
): Record<K, T[]> {
    return items.reduce((acc, item) => {
        const key = keyFn(item);
        if (!acc[key]) acc[key] = [];
        acc[key].push(item);
        return acc;
    }, {} as Record<K, T[]>);
}

function unique<T>(items: T[], keyFn?: (item: T) => unknown): T[] {
    if (!keyFn) return [...new Set(items)];
    const seen = new Set();
    return items.filter(item => {
        const key = keyFn(item);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

function partition<T>(items: T[], predicate: (item: T) => boolean): [T[], T[]] {
    const pass: T[] = [];
    const fail: T[] = [];
    for (const item of items) {
        (predicate(item) ? pass : fail).push(item);
    }
    return [pass, fail];
}

// Usage example
const users = [
    { name: "Alice", dept: "Engineering" },
    { name: "Bob", dept: "Marketing" },
    { name: "Charlie", dept: "Engineering" },
];

const byDept = groupBy(users, u => u.dept);
// { Engineering: [...], Marketing: [...] }

const [engineers, others] = partition(users, u => u.dept === "Engineering");
```

### Rust

```rust
// Generic function
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in &list[1..] {
        if item > largest {
            largest = item;
        }
    }
    largest
}

// Generic struct
struct Point<T> {
    x: T,
    y: T,
}

impl<T: std::fmt::Display> Point<T> {
    fn show(&self) {
        println!("({}, {})", self.x, self.y);
    }
}

// Point with different type parameters
struct Point2<T, U> {
    x: T,
    y: U,
}

impl<T, U> Point2<T, U> {
    fn mixup<V, W>(self, other: Point2<V, W>) -> Point2<T, W> {
        Point2 {
            x: self.x,
            y: other.y,
        }
    }
}

let int_point = Point { x: 5, y: 10 };
let float_point = Point { x: 1.0, y: 4.0 };

// Generic enum (core of Rust)
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Generic Iterator implementation
struct Counter {
    start: u32,
    end: u32,
    current: u32,
}

impl Counter {
    fn new(start: u32, end: u32) -> Self {
        Counter { start, end, current: start }
    }
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let val = self.current;
            self.current += 1;
            Some(val)
        } else {
            None
        }
    }
}

// Generic HashMap wrapper (cache implementation)
use std::collections::HashMap;
use std::hash::Hash;

struct Cache<K, V> {
    store: HashMap<K, V>,
    max_size: usize,
}

impl<K: Eq + Hash + Clone, V: Clone> Cache<K, V> {
    fn new(max_size: usize) -> Self {
        Cache {
            store: HashMap::new(),
            max_size,
        }
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.store.get(key)
    }

    fn put(&mut self, key: K, value: V) -> Option<V> {
        if self.store.len() >= self.max_size && !self.store.contains_key(&key) {
            // Remove the oldest entry (simplified)
            if let Some(first_key) = self.store.keys().next().cloned() {
                self.store.remove(&first_key);
            }
        }
        self.store.insert(key, value)
    }

    fn contains(&self, key: &K) -> bool {
        self.store.contains_key(key)
    }

    fn size(&self) -> usize {
        self.store.len()
    }
}
```

### Go

```go
// Go 1.18+ generics
func Map[T any, U any](slice []T, fn func(T) U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}

doubled := Map([]int{1, 2, 3}, func(n int) int { return n * 2 })
// -> [2, 4, 6]

// Type constraints
type Number interface {
    ~int | ~int64 | ~float64
}

func Sum[T Number](numbers []T) T {
    var sum T
    for _, n := range numbers {
        sum += n
    }
    return sum
}

// Generic stack
type Stack[T any] struct {
    items []T
}

func NewStack[T any]() *Stack[T] {
    return &Stack[T]{items: make([]T, 0)}
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    var zero T
    if len(s.items) == 0 {
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

func (s *Stack[T]) Peek() (T, bool) {
    var zero T
    if len(s.items) == 0 {
        return zero, false
    }
    return s.items[len(s.items)-1], true
}

func (s *Stack[T]) Size() int {
    return len(s.items)
}

// Generic Result type (Go has no native Result)
type Result[T any] struct {
    value T
    err   error
    ok    bool
}

func Ok[T any](value T) Result[T] {
    return Result[T]{value: value, ok: true}
}

func Err[T any](err error) Result[T] {
    return Result[T]{err: err, ok: false}
}

func (r Result[T]) Unwrap() T {
    if !r.ok {
        panic(r.err)
    }
    return r.value
}

func (r Result[T]) UnwrapOr(defaultValue T) T {
    if !r.ok {
        return defaultValue
    }
    return r.value
}

// Generic Pair type
type Pair[T any, U any] struct {
    First  T
    Second U
}

func NewPair[T any, U any](first T, second U) Pair[T, U] {
    return Pair[T, U]{First: first, Second: second}
}

// Generic Filter and Reduce
func Filter[T any](slice []T, predicate func(T) bool) []T {
    result := make([]T, 0)
    for _, v := range slice {
        if predicate(v) {
            result = append(result, v)
        }
    }
    return result
}

func Reduce[T any, U any](slice []T, initial U, fn func(U, T) U) U {
    acc := initial
    for _, v := range slice {
        acc = fn(acc, v)
    }
    return acc
}

// Advanced type constraint interface examples
type Ordered interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
    ~float32 | ~float64 | ~string
}

func Min[T Ordered](a, b T) T {
    if a < b {
        return a
    }
    return b
}

func Max[T Ordered](a, b T) T {
    if a > b {
        return a
    }
    return b
}

func SortSlice[T Ordered](slice []T) {
    sort.Slice(slice, func(i, j int) bool {
        return slice[i] < slice[j]
    })
}
```

### Java

```java
// Java generics
public class Pair<A, B> {
    private final A first;
    private final B second;

    public Pair(A first, B second) {
        this.first = first;
        this.second = second;
    }

    public A getFirst() { return first; }
    public B getSecond() { return second; }

    public <C> Pair<A, C> mapSecond(Function<B, C> fn) {
        return new Pair<>(first, fn.apply(second));
    }
}

// Wildcard types
// ? extends T (upper bounded wildcard): accepts subtypes of T (covariant)
public static double sumOfList(List<? extends Number> list) {
    double sum = 0.0;
    for (Number n : list) {
        sum += n.doubleValue();
    }
    return sum;
}

sumOfList(List.of(1, 2, 3));           // List<Integer> -> OK
sumOfList(List.of(1.0, 2.0, 3.0));     // List<Double> -> OK

// ? super T (lower bounded wildcard): accepts supertypes of T (contravariant)
public static void addNumbers(List<? super Integer> list) {
    list.add(1);
    list.add(2);
}

List<Number> numbers = new ArrayList<>();
addNumbers(numbers);  // List<Number> is a kind of List<? super Integer>

// PECS (Producer Extends, Consumer Super)
// Read-only -> extends, Write-only -> super
public static <T> void copy(List<? extends T> src, List<? super T> dst) {
    for (T item : src) {
        dst.add(item);
    }
}

// Generic method bounds
public static <T extends Comparable<T>> T max(T a, T b) {
    return a.compareTo(b) >= 0 ? a : b;
}

// Intersection types
public static <T extends Serializable & Comparable<T>> void process(T item) {
    // T is both Serializable and Comparable
}
```

---

## 3. Subtype Polymorphism

### TypeScript: Structural Subtyping

```typescript
// TypeScript: Subtype polymorphism via interfaces
interface Printable {
    print(): string;
}

class User implements Printable {
    constructor(private name: string) {}
    print(): string {
        return `User: ${this.name}`;
    }
}

class Product implements Printable {
    constructor(private title: string) {}
    print(): string {
        return `Product: ${this.title}`;
    }
}

// Anything implementing Printable can be passed
function display(item: Printable): void {
    console.log(item.print());
}

display(new User("Gaku"));      // User: Gaku
display(new Product("Book"));   // Product: Book

// TypeScript uses structural subtyping
// Even without explicit implements, matching structure is sufficient
const plainObj = {
    print(): string {
        return "Plain object";
    }
};
display(plainObj); // OK! Because it has a print() method

// Practical use of structural subtyping
interface HasId {
    id: string;
}

interface HasName {
    name: string;
}

interface HasTimestamps {
    createdAt: Date;
    updatedAt: Date;
}

// Combining multiple interfaces
type Entity = HasId & HasName & HasTimestamps;

// Function accepting a partial type
function getDisplayName(item: HasName): string {
    return item.name;
}

// Entity includes HasName, so it can be passed
const entity: Entity = {
    id: "1",
    name: "Alice",
    createdAt: new Date(),
    updatedAt: new Date(),
};
getDisplayName(entity); // "Alice"

// Combining the Visitor pattern with subtype polymorphism
interface ASTNode {
    accept<T>(visitor: ASTVisitor<T>): T;
}

interface ASTVisitor<T> {
    visitLiteral(node: LiteralNode): T;
    visitBinaryOp(node: BinaryOpNode): T;
    visitUnaryOp(node: UnaryOpNode): T;
    visitVariable(node: VariableNode): T;
}

class LiteralNode implements ASTNode {
    constructor(public value: number) {}
    accept<T>(visitor: ASTVisitor<T>): T {
        return visitor.visitLiteral(this);
    }
}

class BinaryOpNode implements ASTNode {
    constructor(
        public op: "+" | "-" | "*" | "/",
        public left: ASTNode,
        public right: ASTNode
    ) {}
    accept<T>(visitor: ASTVisitor<T>): T {
        return visitor.visitBinaryOp(this);
    }
}

class UnaryOpNode implements ASTNode {
    constructor(public op: "-" | "!", public operand: ASTNode) {}
    accept<T>(visitor: ASTVisitor<T>): T {
        return visitor.visitUnaryOp(this);
    }
}

class VariableNode implements ASTNode {
    constructor(public name: string) {}
    accept<T>(visitor: ASTVisitor<T>): T {
        return visitor.visitVariable(this);
    }
}

// Visitor that evaluates expressions
class Evaluator implements ASTVisitor<number> {
    private env: Map<string, number>;

    constructor(env: Map<string, number>) {
        this.env = env;
    }

    visitLiteral(node: LiteralNode): number {
        return node.value;
    }

    visitBinaryOp(node: BinaryOpNode): number {
        const left = node.left.accept(this);
        const right = node.right.accept(this);
        switch (node.op) {
            case "+": return left + right;
            case "-": return left - right;
            case "*": return left * right;
            case "/": return left / right;
        }
    }

    visitUnaryOp(node: UnaryOpNode): number {
        const operand = node.operand.accept(this);
        switch (node.op) {
            case "-": return -operand;
            case "!": return operand === 0 ? 1 : 0;
        }
    }

    visitVariable(node: VariableNode): number {
        const val = this.env.get(node.name);
        if (val === undefined) throw new Error(`Undefined: ${node.name}`);
        return val;
    }
}
```

### Rust: Polymorphism via Traits

```rust
// Rust: Polymorphism via traits
trait Drawable {
    fn draw(&self);
    fn bounding_box(&self) -> (f64, f64, f64, f64); // (x, y, width, height)
}

struct Circle { x: f64, y: f64, radius: f64 }
struct Rectangle { x: f64, y: f64, width: f64, height: f64 }
struct Triangle { points: [(f64, f64); 3] }

impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle at ({}, {}) with radius {}", self.x, self.y, self.radius);
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        (self.x - self.radius, self.y - self.radius, self.radius * 2.0, self.radius * 2.0)
    }
}

impl Drawable for Rectangle {
    fn draw(&self) {
        println!("Drawing {}x{} rectangle at ({}, {})", self.width, self.height, self.x, self.y);
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        (self.x, self.y, self.width, self.height)
    }
}

impl Drawable for Triangle {
    fn draw(&self) {
        println!("Drawing triangle with vertices {:?}", self.points);
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let min_x = self.points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let min_y = self.points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_x = self.points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let max_y = self.points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        (min_x, min_y, max_x - min_x, max_y - min_y)
    }
}

// Static dispatch (resolved at compile time, fast)
fn draw_static(shape: &impl Drawable) {
    shape.draw();
}

// Dynamic dispatch (resolved at runtime, flexible)
fn draw_dynamic(shape: &dyn Drawable) {
    shape.draw();
}

// Trait objects (different types in a single collection)
let shapes: Vec<Box<dyn Drawable>> = vec![
    Box::new(Circle { x: 0.0, y: 0.0, radius: 5.0 }),
    Box::new(Rectangle { x: 1.0, y: 1.0, width: 3.0, height: 4.0 }),
];

// Draw all shapes
for shape in &shapes {
    shape.draw();
}

// Trait inheritance (supertraits)
trait Shape: Drawable + std::fmt::Debug {
    fn area(&self) -> f64;
    fn perimeter(&self) -> f64;
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }

    fn perimeter(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }
}

// Default methods
trait Summary {
    fn title(&self) -> String;
    fn author(&self) -> String;
    fn content(&self) -> String;

    // Default implementation
    fn summarize(&self) -> String {
        format!("{} by {} - {}", self.title(), self.author(), &self.content()[..50])
    }

    fn word_count(&self) -> usize {
        self.content().split_whitespace().count()
    }
}

// Associated types
trait Container {
    type Item;
    type Error;

    fn get(&self, index: usize) -> Result<&Self::Item, Self::Error>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
```

---

## 4. Ad-hoc Polymorphism

### Overloading

```java
// Java: Method overloading
class Calculator {
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
    String add(String a, String b) { return a + b; }

    // Overload resolution priority
    void print(Object obj) { System.out.println("Object: " + obj); }
    void print(String str) { System.out.println("String: " + str); }
    void print(int num)    { System.out.println("int: " + num); }

    // print("hello") -> String version is called (most specific type)
    // print(42)      -> int version is called
    // print(null)    -> Compile error (ambiguous)
}

// C++: Template specialization (a form of ad-hoc polymorphism)
template<typename T>
std::string serialize(const T& value) {
    // Generic version
    return std::to_string(value);
}

template<>
std::string serialize<std::string>(const std::string& value) {
    // string-specific version
    return "\"" + value + "\"";
}

template<>
std::string serialize<bool>(const bool& value) {
    // bool-specific version
    return value ? "true" : "false";
}
```

### Overloading in TypeScript

```typescript
// TypeScript: Function overloading
function createElement(tag: "div"): HTMLDivElement;
function createElement(tag: "span"): HTMLSpanElement;
function createElement(tag: "input"): HTMLInputElement;
function createElement(tag: string): HTMLElement;
function createElement(tag: string): HTMLElement {
    return document.createElement(tag);
}

const div = createElement("div");   // type is HTMLDivElement
const span = createElement("span"); // type is HTMLSpanElement
const input = createElement("input"); // type is HTMLInputElement
const p = createElement("p");       // type is HTMLElement

// Practical method overloading example
class EventEmitter<Events extends Record<string, unknown[]>> {
    private handlers: Map<string, Function[]> = new Map();

    on<K extends keyof Events>(event: K, handler: (...args: Events[K]) => void): void {
        const existing = this.handlers.get(event as string) ?? [];
        existing.push(handler);
        this.handlers.set(event as string, existing);
    }

    emit<K extends keyof Events>(event: K, ...args: Events[K]): void {
        const handlers = this.handlers.get(event as string) ?? [];
        for (const handler of handlers) {
            handler(...args);
        }
    }
}

// Type-safe event emitter
interface AppEvents {
    "user:login": [userId: string, timestamp: Date];
    "user:logout": [userId: string];
    "error": [error: Error, context: string];
}

const emitter = new EventEmitter<AppEvents>();
emitter.on("user:login", (userId, timestamp) => {
    // userId: string, timestamp: Date are inferred
    console.log(`${userId} logged in at ${timestamp}`);
});
```

### Type Classes (Haskell / Rust Traits)

```haskell
-- Haskell: Type classes (the most refined form of ad-hoc polymorphism)
class Eq a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool
    x /= y = not (x == y)  -- Default implementation

instance Eq Int where
    x == y = eqInt x y

instance Eq String where
    x == y = eqString x y

-- Each type has a different implementation of ==

-- Type class hierarchy
class (Eq a) => Ord a where
    compare :: a -> a -> Ordering
    (<)  :: a -> a -> Bool
    (>)  :: a -> a -> Bool
    (<=) :: a -> a -> Bool
    (>=) :: a -> a -> Bool
    min  :: a -> a -> a
    max  :: a -> a -> a

-- Show: types that can be displayed
class Show a where
    show :: a -> String

-- Read: types that can be parsed from a string
class Read a where
    read :: String -> a

-- Compound type class constraints
printSorted :: (Show a, Ord a) => [a] -> String
printSorted xs = show (sort xs)

-- Functor: a mappable type constructor
class Functor f where
    fmap :: (a -> b) -> f a -> f b

instance Functor [] where
    fmap = map

instance Functor Maybe where
    fmap _ Nothing  = Nothing
    fmap f (Just x) = Just (f x)

-- Type class instances for custom types
data Color = Red | Green | Blue

instance Show Color where
    show Red   = "Red"
    show Green = "Green"
    show Blue  = "Blue"

instance Eq Color where
    Red   == Red   = True
    Green == Green = True
    Blue  == Blue  = True
    _     == _     = False

instance Ord Color where
    compare Red   Red   = EQ
    compare Red   _     = LT
    compare Green Red   = GT
    compare Green Green = EQ
    compare Green Blue  = LT
    compare Blue  Blue  = EQ
    compare Blue  _     = GT
```

```rust
// Rust: Ad-hoc polymorphism via traits
use std::fmt;

// Implementing the Display trait enables printing with println!
impl fmt::Display for Point<f64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// Operator overloading
use std::ops::Add;

impl Add for Point<f64> {
    type Output = Point<f64>;
    fn add(self, other: Point<f64>) -> Point<f64> {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

// Implementing multiple operators
use std::ops::{Sub, Mul, Neg};

impl Sub for Point<f64> {
    type Output = Point<f64>;
    fn sub(self, other: Point<f64>) -> Point<f64> {
        Point {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul<f64> for Point<f64> {
    type Output = Point<f64>;
    fn mul(self, scalar: f64) -> Point<f64> {
        Point {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl Neg for Point<f64> {
    type Output = Point<f64>;
    fn neg(self) -> Point<f64> {
        Point {
            x: -self.x,
            y: -self.y,
        }
    }
}

// From / Into traits (ad-hoc polymorphism for type conversion)
struct Celsius(f64);
struct Fahrenheit(f64);

impl From<Celsius> for Fahrenheit {
    fn from(c: Celsius) -> Self {
        Fahrenheit(c.0 * 9.0 / 5.0 + 32.0)
    }
}

impl From<Fahrenheit> for Celsius {
    fn from(f: Fahrenheit) -> Self {
        Celsius((f.0 - 32.0) * 5.0 / 9.0)
    }
}

let boiling = Celsius(100.0);
let f: Fahrenheit = boiling.into(); // Fahrenheit(212.0)

// Ad-hoc polymorphism of the Iterator trait
// Makes any type usable with for loops
struct Fibonacci {
    a: u64,
    b: u64,
}

impl Fibonacci {
    fn new() -> Self {
        Fibonacci { a: 0, b: 1 }
    }
}

impl Iterator for Fibonacci {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.a + self.b;
        self.a = self.b;
        self.b = next;
        Some(self.a)
    }
}

// Usage
for fib in Fibonacci::new().take(10) {
    println!("{}", fib);
}
```

---

## 5. Bounded Polymorphism (Type Constraints)

### TypeScript

```typescript
// TypeScript: Type constraints with extends
function getLength<T extends { length: number }>(item: T): number {
    return item.length;
}

getLength("hello");     // OK: string has length
getLength([1, 2, 3]);   // OK: array has length
// getLength(42);        // NG: number does not have length

// keyof constraint
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
    return obj[key];
}

const user = { name: "Gaku", age: 30 };
getProperty(user, "name");  // string
// getProperty(user, "foo"); // NG: "foo" is not in keyof User

// Combining conditional types with type constraints
type IsArray<T> = T extends unknown[] ? true : false;

type A = IsArray<number[]>;  // true
type B = IsArray<string>;    // false

// Type extraction with conditional types
type ElementType<T> = T extends (infer E)[] ? E : never;

type C = ElementType<number[]>;   // number
type D = ElementType<string[]>;   // string
type E = ElementType<number>;     // never

// Mapped types and type constraints
type Readonly<T> = { readonly [K in keyof T]: T[K] };
type Partial<T> = { [K in keyof T]?: T[K] };
type Required<T> = { [K in keyof T]-?: T[K] };
type Pick<T, K extends keyof T> = { [P in K]: T[P] };
type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

// Practical type constraint example: Validation
type Validator<T> = {
    [K in keyof T]: (value: T[K]) => string | null;
};

interface UserForm {
    name: string;
    age: number;
    email: string;
}

const userValidator: Validator<UserForm> = {
    name: (value) => value.length > 0 ? null : "Name is required",
    age: (value) => value >= 0 ? null : "Age must be 0 or greater",
    email: (value) => value.includes("@") ? null : "Invalid email address",
};

function validate<T>(data: T, validator: Validator<T>): Record<keyof T, string | null> {
    const result = {} as Record<keyof T, string | null>;
    for (const key in validator) {
        result[key] = validator[key](data[key]);
    }
    return result;
}

// Recursive type constraints
type DeepReadonly<T> = {
    readonly [K in keyof T]: T[K] extends object ? DeepReadonly<T[K]> : T[K];
};

type DeepPartial<T> = {
    [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

// Template literal types and constraints
type EventName<T extends string> = `on${Capitalize<T>}`;
type ClickEvent = EventName<"click">;     // "onClick"
type SubmitEvent = EventName<"submit">;   // "onSubmit"

// Type-safe path specification
type PathOf<T, Prefix extends string = ""> = {
    [K in keyof T & string]: T[K] extends object
        ? `${Prefix}${K}` | PathOf<T[K], `${Prefix}${K}.`>
        : `${Prefix}${K}`;
}[keyof T & string];

interface Config {
    database: {
        host: string;
        port: number;
    };
    cache: {
        ttl: number;
    };
}

type ConfigPath = PathOf<Config>;
// "database" | "database.host" | "database.port" | "cache" | "cache.ttl"
```

### Rust

```rust
// Rust: Trait bounds
fn print_all<T: Display + Debug>(items: &[T]) {
    for item in items {
        println!("{} ({:?})", item, item);
    }
}

// where clause (for complex constraints)
fn complex<T, U>(t: T, u: U) -> String
where
    T: Display + Clone,
    U: Debug + Into<String>,
{
    format!("{}: {:?}", t, u)
}

// Constraints with associated types
fn sum_iterator<I>(iter: I) -> I::Item
where
    I: Iterator,
    I::Item: std::ops::Add<Output = I::Item> + Default,
{
    iter.fold(I::Item::default(), |acc, x| acc + x)
}

// Combining lifetimes and trait bounds
fn longest_displayable<'a, T>(x: &'a T, y: &'a T) -> &'a T
where
    T: Display + PartialOrd,
{
    if x >= y { x } else { y }
}

// Negative constraints are not directly supported, but marker traits can be used
trait NotSend {}
impl !Send for SomeType {}  // nightly only

// impl Trait (shorthand syntax)
fn make_iterator(start: i32, end: i32) -> impl Iterator<Item = i32> {
    (start..end).filter(|x| x % 2 == 0)
}

// Conditional method implementation
struct Wrapper<T>(T);

impl<T> Wrapper<T> {
    fn new(value: T) -> Self {
        Wrapper(value)
    }
}

// The show method is only available when T implements Display
impl<T: Display> Wrapper<T> {
    fn show(&self) {
        println!("Value: {}", self.0);
    }
}

// Only available when T implements Clone + Debug
impl<T: Clone + Debug> Wrapper<T> {
    fn clone_and_debug(&self) -> T {
        let cloned = self.0.clone();
        println!("Cloned: {:?}", cloned);
        cloned
    }
}
```

### Go Type Constraints

```go
// Go: Interface constraints
type Stringer interface {
    String() string
}

// Type set constraints
type Numeric interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
    ~float32 | ~float64
}

// Combining methods and union type constraints
type StringableNumeric interface {
    Numeric
    String() string
}

// comparable constraint (types that support == and !=)
func Contains[T comparable](slice []T, target T) bool {
    for _, v := range slice {
        if v == target {
            return true
        }
    }
    return false
}

// Compound constraints
type OrderedStringer interface {
    Ordered
    fmt.Stringer
}
```

---

## 6. Differences in Implementation Approaches

```
Monomorphization -- Rust, C++
  Generates type-specific code at compile time
  Vec<i32> and Vec<String> become separate code
  Advantage: Zero-cost abstraction (no runtime overhead)
  Disadvantage: Binary size and compile time increase

Type Erasure -- Java, TypeScript
  Removes generic type information after compilation
  List<Integer> and List<String> are the same List at runtime
  Advantage: Smaller binary size, backward compatibility
  Disadvantage: Type information is lost at runtime

Boxing + vtable -- Rust's dyn Trait
  Resolved at runtime via virtual function table
  Advantage: Different types can be stored in the same collection
  Disadvantage: Overhead from indirection

Dictionary Passing -- Haskell
  Implicitly passes method tables for type classes as arguments
  Advantage: High flexibility
  Disadvantage: Overhead from indirect calls (can be mitigated by inlining)

Reification -- C#
  Preserves generic type information at runtime
  typeof(T) and typeof(List<int>) are available
  Advantage: Runtime type inspection is possible
  Disadvantage: Increased runtime complexity
```

### Monomorphization in Detail

```rust
// Rust monomorphization
fn add<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {
    a + b
}

// Calls
add(1i32, 2i32);
add(1.0f64, 2.0f64);

// Code generated by the compiler (conceptual):
fn add_i32(a: i32, b: i32) -> i32 { a + b }
fn add_f64(a: f64, b: f64) -> f64 { a + b }

// Advantage: Direct call, can be inlined
// Disadvantage: A function is generated for each type
```

### Type Erasure in Detail

```java
// Java type erasure
List<String> strings = new ArrayList<>();
List<Integer> integers = new ArrayList<>();

// After compilation, both become ArrayList
// No type parameter information at runtime
strings.getClass() == integers.getClass(); // true!

// Limitations of type erasure
// 1. Cannot use instanceof with generic types
// if (obj instanceof List<String>) {} // Compile error
if (obj instanceof List<?>) {} // OK (wildcards only)

// 2. Cannot create arrays of generic types
// T[] arr = new T[10]; // Compile error
Object[] arr = new Object[10]; // Workaround

// 3. Cannot use primitive types with generics
// List<int> list = ...; // Compile error
List<Integer> list = new ArrayList<>(); // Boxing required

// Comparison with C# (reification)
// In C#, generic type information is preserved at runtime
// typeof(List<string>) != typeof(List<int>) // true
```

### Dynamic Dispatch in Detail

```rust
// Rust: Static dispatch vs dynamic dispatch
trait Animal {
    fn speak(&self) -> String;
    fn name(&self) -> &str;
}

struct Dog { name: String }
struct Cat { name: String }

impl Animal for Dog {
    fn speak(&self) -> String { format!("{}: Woof!", self.name) }
    fn name(&self) -> &str { &self.name }
}

impl Animal for Cat {
    fn speak(&self) -> String { format!("{}: Meow!", self.name) }
    fn name(&self) -> &str { &self.name }
}

// Static dispatch (monomorphization)
// The concrete type is determined at compile time
fn greet_static(animal: &impl Animal) {
    println!("{}", animal.speak());
}
// The compiler generates greet_static_Dog and greet_static_Cat

// Dynamic dispatch (vtable)
// Methods are called by looking up the vtable at runtime
fn greet_dynamic(animal: &dyn Animal) {
    println!("{}", animal.speak());
}

// Memory layout of vtable (conceptual):
// +-------------------+
// | &dyn Animal        | = fat pointer
// |   data: *const T   | -> actual data
// |   vtable: *const   | -> vtable
// +-------------------+
//
// vtable:
// +-------------------+
// | drop_fn            | -> destructor
// | size               | -> data size
// | align              | -> data alignment
// | speak_fn           | -> pointer to speak method
// | name_fn            | -> pointer to name method
// +-------------------+

// Guidelines for choosing between them
// Static: When performance matters, when types are known at compile time
// Dynamic: Heterogeneous collections, plugin systems, runtime type determination
```

---

## 7. Variance

```
Variance determines how subtype relationships propagate to generic types:

Covariant: If A <: B then F<A> <: F<B>
  Example: List<Dog> is a subtype of List<Animal> (when read-only)
  Rust: &T is covariant over T
  Java: ? extends T (upper bounded wildcard)

Contravariant: If A <: B then F<B> <: F<A> (reversed)
  Example: (Animal) => void is a subtype of (Dog) => void
  Rust: fn(T) is contravariant over T
  Java: ? super T (lower bounded wildcard)

Invariant: No conversion possible
  Example: List<Dog> and List<Animal> are incompatible (when reading and writing)
  Rust: &mut T is invariant over T
  Java: List<T> (without wildcards)
```

```typescript
// TypeScript variance
// Function parameters are contravariant (when strictFunctionTypes: true)
type Handler<T> = (event: T) => void;

interface MouseEvent { x: number; y: number; }
interface ClickEvent extends MouseEvent { button: number; }

// Can Handler<MouseEvent> be assigned to Handler<ClickEvent>?
// Contravariant: ClickEvent <: MouseEvent -> Handler<MouseEvent> <: Handler<ClickEvent>
const mouseHandler: Handler<MouseEvent> = (e) => console.log(e.x, e.y);
const clickHandler: Handler<ClickEvent> = mouseHandler; // OK (contravariant)

// Arrays are invariant (but TypeScript treats them as covariant -- a type safety hole)
const dogs: Dog[] = [new Dog()];
const animals: Animal[] = dogs; // OK in TypeScript (but not safe)
animals.push(new Cat()); // Type check passes, but a Cat ends up in dogs
```

```java
// Java variance and wildcards
// Covariant: ? extends T (read-only)
List<? extends Animal> animals = new ArrayList<Dog>();
Animal a = animals.get(0);  // OK: reading is safe
// animals.add(new Dog());  // NG: writing is unsafe

// Contravariant: ? super T (write-only)
List<? super Dog> dogs = new ArrayList<Animal>();
dogs.add(new Dog());        // OK: writing is safe
// Dog d = dogs.get(0);     // NG: reading is unsafe (only Object can be obtained)

// PECS: Producer Extends, Consumer Super
public static <T> void copy(
    List<? extends T> src,  // Reading (Producer) -> extends
    List<? super T> dst     // Writing (Consumer) -> super
) {
    for (T item : src) {
        dst.add(item);
    }
}
```

---

## 8. Higher-Kinded Types

```
Higher-Kinded Types = the ability to "abstract over type constructors"

Regular generics: T is a type (kind: *)
Higher-kinded types: F is a type constructor (kind: * -> *)

Example: F can be substituted with List or Option

Haskell: Native support
Scala: Native support
Rust: Approximated via associated types in traits
TypeScript: Partially expressible through type-level tricks
Java/Go: Not supported
```

```haskell
-- Haskell: Functor (the canonical example of higher-kinded types)
class Functor f where
    fmap :: (a -> b) -> f a -> f b

-- f is a type constructor (kind: * -> *)
-- List, Maybe, IO, Either e, etc. can be Functors

instance Functor [] where
    fmap = map

instance Functor Maybe where
    fmap _ Nothing  = Nothing
    fmap f (Just x) = Just (f x)

-- Applicative (extension of Functor)
class Functor f => Applicative f where
    pure  :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b

-- Monad (extension of Applicative)
class Applicative m => Monad m where
    return :: a -> m a
    (>>=)  :: m a -> (a -> m b) -> m b

-- This provides a unified interface for
-- different type constructors like List, Maybe, IO, Either, etc.
```

```rust
// Rust: Approximation of higher-kinded types (GAT: Generic Associated Types)
trait Functor {
    type Unwrapped;
    type Wrapped<U>: Functor;

    fn map<U, F>(self, f: F) -> Self::Wrapped<U>
    where
        F: FnOnce(Self::Unwrapped) -> U;
}

impl<T> Functor for Option<T> {
    type Unwrapped = T;
    type Wrapped<U> = Option<U>;

    fn map<U, F>(self, f: F) -> Option<U>
    where
        F: FnOnce(T) -> U,
    {
        self.map(f)
    }
}

impl<T> Functor for Vec<T> {
    type Unwrapped = T;
    type Wrapped<U> = Vec<U>;

    fn map<U, F>(self, f: F) -> Vec<U>
    where
        F: FnOnce(T) -> U,
    {
        self.into_iter().map(f).collect()
    }
}
```

---

## 9. Practical Pattern Collection

### Builder Pattern (Leveraging Generics)

```rust
// Rust: Safe Builder using the Typestate Pattern
struct NoName;
struct HasName(String);
struct NoEmail;
struct HasEmail(String);

struct UserBuilder<N, E> {
    name: N,
    email: E,
    age: Option<u32>,
}

impl UserBuilder<NoName, NoEmail> {
    fn new() -> Self {
        UserBuilder {
            name: NoName,
            email: NoEmail,
            age: None,
        }
    }
}

impl<E> UserBuilder<NoName, E> {
    fn name(self, name: String) -> UserBuilder<HasName, E> {
        UserBuilder {
            name: HasName(name),
            email: self.email,
            age: self.age,
        }
    }
}

impl<N> UserBuilder<N, NoEmail> {
    fn email(self, email: String) -> UserBuilder<N, HasEmail> {
        UserBuilder {
            name: self.name,
            email: HasEmail(email),
            age: self.age,
        }
    }
}

impl<N, E> UserBuilder<N, E> {
    fn age(mut self, age: u32) -> Self {
        self.age = Some(age);
        self
    }
}

// build can only be called when both name and email are set
impl UserBuilder<HasName, HasEmail> {
    fn build(self) -> User {
        User {
            name: self.name.0,
            email: self.email.0,
            age: self.age,
        }
    }
}

// Usage
let user = UserBuilder::new()
    .name("Gaku".into())
    .email("gaku@example.com".into())
    .age(30)
    .build();

// This is a compile error (email not set)
// let invalid = UserBuilder::new().name("Gaku".into()).build();
```

### Strategy Pattern

```typescript
// TypeScript: Strategy pattern leveraging generics
interface SortStrategy<T> {
    sort(items: T[]): T[];
    readonly name: string;
}

class QuickSort<T> implements SortStrategy<T> {
    readonly name = "QuickSort";

    constructor(private compare: (a: T, b: T) => number) {}

    sort(items: T[]): T[] {
        if (items.length <= 1) return items;
        const pivot = items[Math.floor(items.length / 2)];
        const left = items.filter(x => this.compare(x, pivot) < 0);
        const middle = items.filter(x => this.compare(x, pivot) === 0);
        const right = items.filter(x => this.compare(x, pivot) > 0);
        return [...this.sort(left), ...middle, ...this.sort(right)];
    }
}

class MergeSort<T> implements SortStrategy<T> {
    readonly name = "MergeSort";

    constructor(private compare: (a: T, b: T) => number) {}

    sort(items: T[]): T[] {
        if (items.length <= 1) return items;
        const mid = Math.floor(items.length / 2);
        const left = this.sort(items.slice(0, mid));
        const right = this.sort(items.slice(mid));
        return this.merge(left, right);
    }

    private merge(left: T[], right: T[]): T[] {
        const result: T[] = [];
        let i = 0, j = 0;
        while (i < left.length && j < right.length) {
            if (this.compare(left[i], right[j]) <= 0) {
                result.push(left[i++]);
            } else {
                result.push(right[j++]);
            }
        }
        return [...result, ...left.slice(i), ...right.slice(j)];
    }
}

// Usage
class Sorter<T> {
    constructor(private strategy: SortStrategy<T>) {}

    setStrategy(strategy: SortStrategy<T>): void {
        this.strategy = strategy;
    }

    sort(items: T[]): T[] {
        console.log(`Sorting with ${this.strategy.name}`);
        return this.strategy.sort(items);
    }
}

const numberCompare = (a: number, b: number) => a - b;
const sorter = new Sorter(new QuickSort<number>(numberCompare));
sorter.sort([3, 1, 4, 1, 5, 9, 2, 6]);

sorter.setStrategy(new MergeSort<number>(numberCompare));
sorter.sort([3, 1, 4, 1, 5, 9, 2, 6]);
```

### Type-Safe Error Handling with Result Types

```typescript
// TypeScript: Result monadic pattern
type Result<T, E> =
    | { ok: true; value: T }
    | { ok: false; error: E };

function ok<T>(value: T): Result<T, never> {
    return { ok: true, value };
}

function err<E>(error: E): Result<never, E> {
    return { ok: false, error };
}

function map<T, U, E>(result: Result<T, E>, fn: (value: T) => U): Result<U, E> {
    return result.ok ? ok(fn(result.value)) : result;
}

function flatMap<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => Result<U, E>
): Result<U, E> {
    return result.ok ? fn(result.value) : result;
}

function mapError<T, E, F>(
    result: Result<T, E>,
    fn: (error: E) => F
): Result<T, F> {
    return result.ok ? result : err(fn(result.error));
}

// Pipeline-style usage
type ParseError = { type: "parse"; message: string };
type ValidationError = { type: "validation"; field: string; message: string };
type AppError = ParseError | ValidationError;

function parseAge(input: string): Result<number, ParseError> {
    const n = parseInt(input, 10);
    return isNaN(n) ? err({ type: "parse", message: `Invalid number: ${input}` }) : ok(n);
}

function validateAge(age: number): Result<number, ValidationError> {
    if (age < 0) return err({ type: "validation", field: "age", message: "Age must be non-negative" });
    if (age > 150) return err({ type: "validation", field: "age", message: "Age seems unrealistic" });
    return ok(age);
}

const result = flatMap(
    mapError(parseAge("25"), (e): AppError => e),
    (age) => mapError(validateAge(age), (e): AppError => e)
);
```

---

## 10. Anti-patterns and Caveats

```
1. Over-generification
   -> Do not sacrifice readability. Use concrete types when they suffice.
   -> Consider generics when you have written similar code 3+ times.

2. Abuse of type parameters
   -> Reconsider your design when you have 4 or more type parameters.
   -> Consider whether associated types can reduce the count.

3. Problems caused by Java's type erasure
   -> Generic types cannot be inspected at runtime.
   -> instanceof List<String> is impossible.
   -> Workaround: Pass a Class<T> token as an argument.

4. Limitations of Go's interface constraints
   -> There are restrictions on mixing method and union type constraints.
   -> Complex constraints may need to be supplemented with type assertions.

5. Covariant array pitfall (Java, TypeScript)
   -> Java: String[] is a subtype of Object[] -> potential ArrayStoreException
   -> TypeScript: Arrays are covariant -> a type safety hole

6. Rust's object safety
   -> Methods returning Self or methods with type parameters cannot be used with dyn Trait.
   -> Workaround: Use associated types or generic wrappers.
```

```rust
// Rust: A trait that is not object-safe
trait Cloneable {
    fn clone(&self) -> Self;  // Returns Self -> cannot be used as dyn Cloneable
}

// Workaround: A wrapper that erases the type
trait CloneBox {
    fn clone_box(&self) -> Box<dyn CloneBox>;
}

impl<T: Clone + 'static> CloneBox for T {
    fn clone_box(&self) -> Box<dyn CloneBox> {
        Box::new(self.clone())
    }
}

// Now it can be used as dyn CloneBox
let items: Vec<Box<dyn CloneBox>> = vec![
    Box::new(42),
    Box::new(String::from("hello")),
];
```

---

## Practical Exercises

### Exercise 1: [Basics] -- Generic Collection
Implement a generic double-ended queue (Deque) in TypeScript or Rust. It should have push_front, push_back, pop_front, pop_back, peek_front, peek_back, size, and is_empty methods.

### Exercise 2: [Intermediate] -- Type-Safe Event System
Implement a type-safe event emitter in TypeScript that strictly manages event names and argument types via type parameters. It should have on, off, emit, and once methods, and cause compile errors when argument types do not match the event name.

### Exercise 3: [Intermediate] -- Choosing Between Trait Objects and Generics
Design a plugin system in Rust. Implement both static dispatch and dynamic dispatch versions, and compare their performance differences with benchmarks.

### Exercise 4: [Advanced] -- Type-Level Programming
Use TypeScript's conditional types and mapped types to implement a type utility that automatically generates TypeScript types from type definitions equivalent to JSON Schema.

### Exercise 5: [Advanced] -- Simulating Higher-Kinded Types
Use Rust's GAT (Generic Associated Types) to define traits equivalent to Functor and Monad, and implement them for Option and Result.

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

| Type of Polymorphism | Mechanism | Example |
|------------|--------|------|
| Parametric | Parameterize types | `List<T>`, `Vec<T>` |
| Subtype | Inheritance / Interfaces | `impl Trait`, `extends` |
| Ad-hoc | Different implementations per type | Overloading, type classes |
| Bounded | Constrain type parameters | `T extends X`, `T: Trait` |
| Row | Types with specific fields | TypeScript structural subtyping |
| Kind | Abstract over type constructors | `Functor f`, `Monad m` |

| Implementation Approach | Characteristics | Representative Languages |
|---------|------|---------|
| Monomorphization | Compile-time expansion, zero-cost | Rust, C++ |
| Type Erasure | No runtime type info | Java, TypeScript |
| Reification | Runtime type info preserved | C# |
| vtable | Dynamic dispatch | Rust dyn, Java virtual methods |
| Dictionary Passing | Type class method table | Haskell |

| Variance | Meaning | Example |
|------|------|-----|
| Covariant | Subtype relationship preserved | `&T`, `? extends T` |
| Contravariant | Subtype relationship reversed | `fn(T)`, `? super T` |
| Invariant | No subtype relationship | `&mut T`, `List<T>` |

---

## Recommended Next Guides

---

## References
1. Pierce, B. "Types and Programming Languages." Ch.23-26, MIT Press, 2002.
2. Wadler, P. & Blott, S. "How to make ad-hoc polymorphism less ad hoc." 1989.
3. Cardelli, L. & Wegner, P. "On Understanding Types, Data Abstraction, and Polymorphism." Computing Surveys, 1985.
4. Odersky, M. & Zenger, C. "Scalable Component Abstractions." OOPSLA, 2005.
5. Strachey, C. "Fundamental Concepts in Programming Languages." 1967.
6. Crary, K. et al. "Intensional Polymorphism in Type-Erasure Semantics." JFP, 2002.
7. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.10, 2023.
8. Bloch, J. "Effective Java." 3rd Ed, Ch.5 (Generics), Addison-Wesley, 2018.
9. Rust RFC 1598: "Generic Associated Types." 2023.
10. Go Team. "Type Parameters Proposal." 2022.
