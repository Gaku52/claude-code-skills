# ジェネリクスと多態性（Polymorphism）

> 多態性は「同じコードで異なる型を扱う」能力。コードの再利用性と型安全性を両立する鍵。

## この章で学ぶこと

- [ ] 多態性の3つの種類を理解する
- [ ] ジェネリクスの実装と使い方を習得する
- [ ] 型制約（Bounded Polymorphism）を活用できる
- [ ] 各言語のジェネリクス実装方式の違いを理解する
- [ ] 高カインド型・GADTs などの発展的な概念を把握する
- [ ] 実務でジェネリクスを効果的に活用するパターンを身につける

---

## 1. 多態性の種類

```
多態性（Polymorphism）= 「多くの形を持つ」

  1. パラメトリック多態性（ジェネリクス）
     → 型をパラメータ化して汎用的なコードを書く
     例: List<T>, Map<K, V>

  2. サブタイプ多態性（継承・インターフェース）
     → 親型のインターフェースで子型を扱う
     例: Animal を引数に取る関数で Dog も Cat も渡せる

  3. アドホック多態性（オーバーロード・型クラス）
     → 型ごとに異なる実装を持つ
     例: + が数値の加算にも文字列の結合にもなる

  4. 行多態性（Row Polymorphism）
     → 特定のフィールドを持つ任意のレコードを扱う
     例: OCaml のオブジェクト、TypeScript の構造的部分型

  5. カインド多態性（Higher-Kinded Polymorphism）
     → 型コンストラクタ自体を抽象化する
     例: Haskell の Functor, Monad
```

### 多態性の歴史的背景

```
1967年: Strachey が「パラメトリック多態性」と「アドホック多態性」を区別
1972年: Girard が System F（二階多態型ラムダ計算）を提案
1978年: Milner が ML 言語で型推論付きパラメトリック多態性を実現
1984年: Cardelli & Wegner がサブタイプ多態性を形式化
1989年: Wadler & Blott が型クラス（アドホック多態性の統一的枠組み）を提案
2004年: Java 5 でジェネリクスが導入
2012年: Rust のトレイトシステムが成熟
2022年: Go 1.18 でジェネリクスが導入
```

---

## 2. パラメトリック多態性（ジェネリクス）

### TypeScript

```typescript
// ジェネリック関数
function identity<T>(value: T): T {
    return value;
}

identity<number>(42);     // T = number
identity<string>("hello"); // T = string
identity(42);             // T = number（型推論）

// ジェネリック型
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

// 複数の型パラメータ
function zip<A, B>(a: A[], b: B[]): [A, B][] {
    return a.map((val, i) => [val, b[i]]);
}

zip([1, 2], ["a", "b"]);  // [[1, "a"], [2, "b"]]

// ジェネリッククラス
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
numStack.pop(); // 2（型は number | undefined）

// ジェネリックインターフェースの実務的な例: Repository パターン
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

// ジェネリックユーティリティ関数群
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

// 使用例
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
// ジェネリック関数
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in &list[1..] {
        if item > largest {
            largest = item;
        }
    }
    largest
}

// ジェネリック構造体
struct Point<T> {
    x: T,
    y: T,
}

impl<T: std::fmt::Display> Point<T> {
    fn show(&self) {
        println!("({}, {})", self.x, self.y);
    }
}

// 異なる型パラメータを持つ Point
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

// ジェネリック列挙型（Rust の核心）
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

// ジェネリックな Iterator 実装
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

// ジェネリックな HashMap ラッパー（キャッシュ実装）
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
            // 最も古いエントリを削除（簡略化）
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
// Go 1.18+ のジェネリクス
func Map[T any, U any](slice []T, fn func(T) U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}

doubled := Map([]int{1, 2, 3}, func(n int) int { return n * 2 })
// → [2, 4, 6]

// 型制約
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

// ジェネリックなスタック
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

// ジェネリックな Result 型（Go にはネイティブの Result がない）
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

// ジェネリックな Pair 型
type Pair[T any, U any] struct {
    First  T
    Second U
}

func NewPair[T any, U any](first T, second U) Pair[T, U] {
    return Pair[T, U]{First: first, Second: second}
}

// ジェネリックな Filter, Reduce
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

// 型制約インターフェースの高度な例
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
// Java のジェネリクス
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

// ワイルドカード型
// ? extends T（上限ワイルドカード）: T のサブタイプを受け入れる（共変）
public static double sumOfList(List<? extends Number> list) {
    double sum = 0.0;
    for (Number n : list) {
        sum += n.doubleValue();
    }
    return sum;
}

sumOfList(List.of(1, 2, 3));           // List<Integer> → OK
sumOfList(List.of(1.0, 2.0, 3.0));     // List<Double> → OK

// ? super T（下限ワイルドカード）: T のスーパータイプを受け入れる（反変）
public static void addNumbers(List<? super Integer> list) {
    list.add(1);
    list.add(2);
}

List<Number> numbers = new ArrayList<>();
addNumbers(numbers);  // List<Number> は List<? super Integer> の一種

// PECS（Producer Extends, Consumer Super）
// 読み取り専用 → extends、書き込み専用 → super
public static <T> void copy(List<? extends T> src, List<? super T> dst) {
    for (T item : src) {
        dst.add(item);
    }
}

// ジェネリックメソッドのバウンド
public static <T extends Comparable<T>> T max(T a, T b) {
    return a.compareTo(b) >= 0 ? a : b;
}

// 型の交差（intersection types）
public static <T extends Serializable & Comparable<T>> void process(T item) {
    // T は Serializable かつ Comparable
}
```

---

## 3. サブタイプ多態性

### TypeScript: 構造的部分型

```typescript
// TypeScript: インターフェースによるサブタイプ多態性
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

// Printable を実装していれば何でも渡せる
function display(item: Printable): void {
    console.log(item.print());
}

display(new User("Gaku"));      // User: Gaku
display(new Product("Book"));   // Product: Book

// TypeScript は構造的部分型（Structural Subtyping）
// 明示的な implements がなくても、構造が一致すればOK
const plainObj = {
    print(): string {
        return "Plain object";
    }
};
display(plainObj); // OK! print() メソッドを持っているから

// 構造的部分型の実践的な利用
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

// 複数のインターフェースを組み合わせ
type Entity = HasId & HasName & HasTimestamps;

// 部分的な型を受け入れる関数
function getDisplayName(item: HasName): string {
    return item.name;
}

// Entity は HasName を含むので渡せる
const entity: Entity = {
    id: "1",
    name: "Alice",
    createdAt: new Date(),
    updatedAt: new Date(),
};
getDisplayName(entity); // "Alice"

// Visitor パターンとサブタイプ多態性の組み合わせ
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

// 式を評価する Visitor
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

### Rust: トレイトによる多態性

```rust
// Rust: トレイトによる多態性
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

// 静的ディスパッチ（コンパイル時に解決、高速）
fn draw_static(shape: &impl Drawable) {
    shape.draw();
}

// 動的ディスパッチ（実行時に解決、柔軟）
fn draw_dynamic(shape: &dyn Drawable) {
    shape.draw();
}

// トレイトオブジェクト（異なる型を1つのコレクションに）
let shapes: Vec<Box<dyn Drawable>> = vec![
    Box::new(Circle { x: 0.0, y: 0.0, radius: 5.0 }),
    Box::new(Rectangle { x: 1.0, y: 1.0, width: 3.0, height: 4.0 }),
];

// 全ての図形を描画
for shape in &shapes {
    shape.draw();
}

// トレイト継承（スーパートレイト）
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

// デフォルトメソッド
trait Summary {
    fn title(&self) -> String;
    fn author(&self) -> String;
    fn content(&self) -> String;

    // デフォルト実装
    fn summarize(&self) -> String {
        format!("{} by {} - {}", self.title(), self.author(), &self.content()[..50])
    }

    fn word_count(&self) -> usize {
        self.content().split_whitespace().count()
    }
}

// 関連型（Associated Types）
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

## 4. アドホック多態性

### オーバーロード

```java
// Java: メソッドオーバーロード
class Calculator {
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
    String add(String a, String b) { return a + b; }

    // オーバーロード解決の優先順位
    void print(Object obj) { System.out.println("Object: " + obj); }
    void print(String str) { System.out.println("String: " + str); }
    void print(int num)    { System.out.println("int: " + num); }

    // print("hello") → String 版が呼ばれる（最も具体的な型）
    // print(42)      → int 版が呼ばれる
    // print(null)    → コンパイルエラー（曖昧）
}

// C++: テンプレート特殊化（アドホック多態性の一種）
template<typename T>
std::string serialize(const T& value) {
    // 汎用版
    return std::to_string(value);
}

template<>
std::string serialize<std::string>(const std::string& value) {
    // string 専用版
    return "\"" + value + "\"";
}

template<>
std::string serialize<bool>(const bool& value) {
    // bool 専用版
    return value ? "true" : "false";
}
```

### TypeScript でのオーバーロード

```typescript
// TypeScript: 関数オーバーロード
function createElement(tag: "div"): HTMLDivElement;
function createElement(tag: "span"): HTMLSpanElement;
function createElement(tag: "input"): HTMLInputElement;
function createElement(tag: string): HTMLElement;
function createElement(tag: string): HTMLElement {
    return document.createElement(tag);
}

const div = createElement("div");   // 型は HTMLDivElement
const span = createElement("span"); // 型は HTMLSpanElement
const input = createElement("input"); // 型は HTMLInputElement
const p = createElement("p");       // 型は HTMLElement

// メソッドオーバーロードの実践例
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

// 型安全なイベントエミッター
interface AppEvents {
    "user:login": [userId: string, timestamp: Date];
    "user:logout": [userId: string];
    "error": [error: Error, context: string];
}

const emitter = new EventEmitter<AppEvents>();
emitter.on("user:login", (userId, timestamp) => {
    // userId: string, timestamp: Date が推論される
    console.log(`${userId} logged in at ${timestamp}`);
});
```

### 型クラス（Haskell / Rust のトレイト）

```haskell
-- Haskell: 型クラス（アドホック多態性の最も洗練された形）
class Eq a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool
    x /= y = not (x == y)  -- デフォルト実装

instance Eq Int where
    x == y = eqInt x y

instance Eq String where
    x == y = eqString x y

-- 型ごとに異なる == の実装を持つ

-- 型クラスの階層
class (Eq a) => Ord a where
    compare :: a -> a -> Ordering
    (<)  :: a -> a -> Bool
    (>)  :: a -> a -> Bool
    (<=) :: a -> a -> Bool
    (>=) :: a -> a -> Bool
    min  :: a -> a -> a
    max  :: a -> a -> a

-- Show: 表示可能な型
class Show a where
    show :: a -> String

-- Read: 文字列から解析可能な型
class Read a where
    read :: String -> a

-- 複合的な型クラス制約
printSorted :: (Show a, Ord a) => [a] -> String
printSorted xs = show (sort xs)

-- Functor: 写像可能な型コンストラクタ
class Functor f where
    fmap :: (a -> b) -> f a -> f b

instance Functor [] where
    fmap = map

instance Functor Maybe where
    fmap _ Nothing  = Nothing
    fmap f (Just x) = Just (f x)

-- 自作の型に対する型クラスインスタンス
data Color = Red | Green | Blue

instance Show Color where
    show Red   = "赤"
    show Green = "緑"
    show Blue  = "青"

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
// Rust: トレイトでのアドホック多態性
use std::fmt;

// Display トレイトを実装すると println! で表示可能
impl fmt::Display for Point<f64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// 演算子オーバーロード
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

// 複数の演算子を実装
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

// From / Into トレイト（型変換のアドホック多態性）
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

// Iterator トレイトのアドホック多態性
// 任意の型に対して for ループを使えるようにする
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

// 使用
for fib in Fibonacci::new().take(10) {
    println!("{}", fib);
}
```

---

## 5. 型制約（Bounded Polymorphism）

### TypeScript

```typescript
// TypeScript: extends による型制約
function getLength<T extends { length: number }>(item: T): number {
    return item.length;
}

getLength("hello");     // OK: string は length を持つ
getLength([1, 2, 3]);   // OK: array は length を持つ
// getLength(42);        // NG: number は length を持たない

// keyof 制約
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
    return obj[key];
}

const user = { name: "Gaku", age: 30 };
getProperty(user, "name");  // string
// getProperty(user, "foo"); // NG: "foo" は keyof User にない

// 条件型と型制約の組み合わせ
type IsArray<T> = T extends unknown[] ? true : false;

type A = IsArray<number[]>;  // true
type B = IsArray<string>;    // false

// 条件型での型の抽出
type ElementType<T> = T extends (infer E)[] ? E : never;

type C = ElementType<number[]>;   // number
type D = ElementType<string[]>;   // string
type E = ElementType<number>;     // never

// マップ型と型制約
type Readonly<T> = { readonly [K in keyof T]: T[K] };
type Partial<T> = { [K in keyof T]?: T[K] };
type Required<T> = { [K in keyof T]-?: T[K] };
type Pick<T, K extends keyof T> = { [P in K]: T[P] };
type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

// 実務的な型制約の例: バリデーション
type Validator<T> = {
    [K in keyof T]: (value: T[K]) => string | null;
};

interface UserForm {
    name: string;
    age: number;
    email: string;
}

const userValidator: Validator<UserForm> = {
    name: (value) => value.length > 0 ? null : "名前は必須です",
    age: (value) => value >= 0 ? null : "年齢は0以上です",
    email: (value) => value.includes("@") ? null : "メールアドレスが不正です",
};

function validate<T>(data: T, validator: Validator<T>): Record<keyof T, string | null> {
    const result = {} as Record<keyof T, string | null>;
    for (const key in validator) {
        result[key] = validator[key](data[key]);
    }
    return result;
}

// 再帰的な型制約
type DeepReadonly<T> = {
    readonly [K in keyof T]: T[K] extends object ? DeepReadonly<T[K]> : T[K];
};

type DeepPartial<T> = {
    [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

// テンプレートリテラル型と制約
type EventName<T extends string> = `on${Capitalize<T>}`;
type ClickEvent = EventName<"click">;     // "onClick"
type SubmitEvent = EventName<"submit">;   // "onSubmit"

// 型安全なパス指定
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
// Rust: トレイト境界
fn print_all<T: Display + Debug>(items: &[T]) {
    for item in items {
        println!("{} ({:?})", item, item);
    }
}

// where 句（複雑な制約の場合）
fn complex<T, U>(t: T, u: U) -> String
where
    T: Display + Clone,
    U: Debug + Into<String>,
{
    format!("{}: {:?}", t, u)
}

// 関連型による制約
fn sum_iterator<I>(iter: I) -> I::Item
where
    I: Iterator,
    I::Item: std::ops::Add<Output = I::Item> + Default,
{
    iter.fold(I::Item::default(), |acc, x| acc + x)
}

// ライフタイムとトレイト境界の組み合わせ
fn longest_displayable<'a, T>(x: &'a T, y: &'a T) -> &'a T
where
    T: Display + PartialOrd,
{
    if x >= y { x } else { y }
}

// 否定制約は直接サポートされないが、マーカートレイトで制御可能
trait NotSend {}
impl !Send for SomeType {}  // nightly のみ

// impl Trait（簡略構文）
fn make_iterator(start: i32, end: i32) -> impl Iterator<Item = i32> {
    (start..end).filter(|x| x % 2 == 0)
}

// 条件付きメソッド実装
struct Wrapper<T>(T);

impl<T> Wrapper<T> {
    fn new(value: T) -> Self {
        Wrapper(value)
    }
}

// T が Display を実装している場合のみ show メソッドが使える
impl<T: Display> Wrapper<T> {
    fn show(&self) {
        println!("Value: {}", self.0);
    }
}

// T が Clone + Debug を実装している場合のみ
impl<T: Clone + Debug> Wrapper<T> {
    fn clone_and_debug(&self) -> T {
        let cloned = self.0.clone();
        println!("Cloned: {:?}", cloned);
        cloned
    }
}
```

### Go の型制約

```go
// Go: インターフェース制約
type Stringer interface {
    String() string
}

// 型セット制約
type Numeric interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
    ~float32 | ~float64
}

// メソッドとユニオン型の組み合わせ
type StringableNumeric interface {
    Numeric
    String() string
}

// comparable 制約（== と != が使える型）
func Contains[T comparable](slice []T, target T) bool {
    for _, v := range slice {
        if v == target {
            return true
        }
    }
    return false
}

// 複合制約
type OrderedStringer interface {
    Ordered
    fmt.Stringer
}
```

---

## 6. 実装方式の違い

```
単相化（Monomorphization）— Rust, C++
  コンパイル時に型ごとにコードを生成
  Vec<i32> と Vec<String> は別のコードになる
  利点: ゼロコスト抽象化（実行時オーバーヘッドなし）
  欠点: バイナリサイズが増加、コンパイル時間が増加

型消去（Type Erasure）— Java, TypeScript
  コンパイル後にジェネリクスの型情報を削除
  List<Integer> と List<String> は実行時に同じ List
  利点: バイナリサイズが小さい、後方互換性
  欠点: 実行時に型情報が失われる

ボックス化 + vtable — Rust の dyn Trait
  実行時に仮想関数テーブルで解決
  利点: 異なる型を同一コレクションに格納可能
  欠点: 間接参照のオーバーヘッド

辞書渡し（Dictionary Passing）— Haskell
  型クラスのメソッドテーブルを暗黙的に引数として渡す
  利点: 柔軟性が高い
  欠点: 間接呼び出しのオーバーヘッド（インライン化で軽減可能）

具体化（Reification）— C#
  実行時にもジェネリクスの型情報を保持
  typeof(T) や typeof(List<int>) が使える
  利点: 実行時の型検査が可能
  欠点: ランタイムの複雑化
```

### 単相化の詳細

```rust
// Rust の単相化
fn add<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {
    a + b
}

// 呼び出し
add(1i32, 2i32);
add(1.0f64, 2.0f64);

// コンパイラが生成するコード（概念的）:
fn add_i32(a: i32, b: i32) -> i32 { a + b }
fn add_f64(a: f64, b: f64) -> f64 { a + b }

// 利点: 直接呼び出し、インライン化可能
// 欠点: 型の数だけ関数が生成される
```

### 型消去の詳細

```java
// Java の型消去
List<String> strings = new ArrayList<>();
List<Integer> integers = new ArrayList<>();

// コンパイル後は両方とも ArrayList になる
// 実行時には型パラメータの情報がない
strings.getClass() == integers.getClass(); // true!

// 型消去の制限
// 1. instanceof でジェネリック型をチェックできない
// if (obj instanceof List<String>) {} // コンパイルエラー
if (obj instanceof List<?>) {} // OK（ワイルドカードのみ）

// 2. ジェネリック型の配列を作れない
// T[] arr = new T[10]; // コンパイルエラー
Object[] arr = new Object[10]; // 代替手段

// 3. プリミティブ型をジェネリックに使えない
// List<int> list = ...; // コンパイルエラー
List<Integer> list = new ArrayList<>(); // ボクシングが必要

// C# との比較（具体化）
// C# ではジェネリック型情報が実行時にも保持される
// typeof(List<string>) != typeof(List<int>) // true
```

### 動的ディスパッチの詳細

```rust
// Rust: 静的ディスパッチ vs 動的ディスパッチ
trait Animal {
    fn speak(&self) -> String;
    fn name(&self) -> &str;
}

struct Dog { name: String }
struct Cat { name: String }

impl Animal for Dog {
    fn speak(&self) -> String { format!("{}: ワンワン!", self.name) }
    fn name(&self) -> &str { &self.name }
}

impl Animal for Cat {
    fn speak(&self) -> String { format!("{}: ニャー!", self.name) }
    fn name(&self) -> &str { &self.name }
}

// 静的ディスパッチ（単相化）
// コンパイル時に具体的な型が決定される
fn greet_static(animal: &impl Animal) {
    println!("{}", animal.speak());
}
// コンパイラは greet_static_Dog と greet_static_Cat を生成

// 動的ディスパッチ（vtable）
// 実行時に vtable を参照してメソッドを呼び出す
fn greet_dynamic(animal: &dyn Animal) {
    println!("{}", animal.speak());
}

// vtable のメモリレイアウト（概念的）:
// ┌─────────────────┐
// │ &dyn Animal      │ = ファットポインタ
// │   data: *const T │ → 実際のデータ
// │   vtable: *const │ → vtable
// └─────────────────┘
//
// vtable:
// ┌─────────────────┐
// │ drop_fn          │ → デストラクタ
// │ size             │ → データのサイズ
// │ align            │ → データのアライメント
// │ speak_fn         │ → speak メソッドのポインタ
// │ name_fn          │ → name メソッドのポインタ
// └─────────────────┘

// 使い分けの指針
// 静的: パフォーマンス重視、コンパイル時に型が分かる場合
// 動的: 異種コレクション、プラグインシステム、実行時の型決定
```

---

## 7. 変性（Variance）

```
型パラメータの変性は、サブタイプ関係がジェネリック型にどう伝播するかを決める:

共変（Covariant）: A <: B ならば F<A> <: F<B>
  例: List<Dog> は List<Animal> のサブタイプ（読み取り専用の場合）
  Rust: &T は T に対して共変
  Java: ? extends T（上限ワイルドカード）

反変（Contravariant）: A <: B ならば F<B> <: F<A>（逆転）
  例: (Animal) => void は (Dog) => void のサブタイプ
  Rust: fn(T) は T に対して反変
  Java: ? super T（下限ワイルドカード）

不変（Invariant）: 変換不可
  例: List<Dog> と List<Animal> に互換性なし（読み書きする場合）
  Rust: &mut T は T に対して不変
  Java: List<T>（ワイルドカードなし）
```

```typescript
// TypeScript の変性
// 関数パラメータは反変（strictFunctionTypes: true の場合）
type Handler<T> = (event: T) => void;

interface MouseEvent { x: number; y: number; }
interface ClickEvent extends MouseEvent { button: number; }

// Handler<MouseEvent> は Handler<ClickEvent> に代入可能?
// 反変なので: ClickEvent <: MouseEvent → Handler<MouseEvent> <: Handler<ClickEvent>
const mouseHandler: Handler<MouseEvent> = (e) => console.log(e.x, e.y);
const clickHandler: Handler<ClickEvent> = mouseHandler; // OK（反変）

// 配列は不変（実際は TypeScript では共変として扱われる ← 型安全性の穴）
const dogs: Dog[] = [new Dog()];
const animals: Animal[] = dogs; // TypeScript では OK（安全ではない）
animals.push(new Cat()); // 型チェックは通るが、dogs に Cat が入ってしまう
```

```java
// Java の変性とワイルドカード
// 共変: ? extends T（読み取り専用）
List<? extends Animal> animals = new ArrayList<Dog>();
Animal a = animals.get(0);  // OK: 読み取りは安全
// animals.add(new Dog());  // NG: 書き込みは不安全

// 反変: ? super T（書き込み専用）
List<? super Dog> dogs = new ArrayList<Animal>();
dogs.add(new Dog());        // OK: 書き込みは安全
// Dog d = dogs.get(0);     // NG: 読み取りは不安全（Object しか得られない）

// PECS: Producer Extends, Consumer Super
public static <T> void copy(
    List<? extends T> src,  // 読み取り（Producer）→ extends
    List<? super T> dst     // 書き込み（Consumer）→ super
) {
    for (T item : src) {
        dst.add(item);
    }
}
```

---

## 8. 高カインド型（Higher-Kinded Types）

```
高カインド型 = 「型コンストラクタを抽象化する」能力

通常のジェネリクス: T は型（kind: *）
高カインド型: F は型コンストラクタ（kind: * -> *）

例: F を List や Option に差し替えられる

Haskell: ネイティブサポート
Scala: ネイティブサポート
Rust: トレイトの関連型で擬似的に表現
TypeScript: 型レベルの工夫で部分的に表現
Java/Go: サポートなし
```

```haskell
-- Haskell: Functor（高カインド型の典型例）
class Functor f where
    fmap :: (a -> b) -> f a -> f b

-- f は型コンストラクタ（kind: * -> *）
-- List, Maybe, IO, Either e などが Functor になれる

instance Functor [] where
    fmap = map

instance Functor Maybe where
    fmap _ Nothing  = Nothing
    fmap f (Just x) = Just (f x)

-- Applicative（Functor の拡張）
class Functor f => Applicative f where
    pure  :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b

-- Monad（Applicative の拡張）
class Applicative m => Monad m where
    return :: a -> m a
    (>>=)  :: m a -> (a -> m b) -> m b

-- これにより、List, Maybe, IO, Either など
-- 異なる型コンストラクタに対して統一的なインターフェースを提供
```

```rust
// Rust: 高カインド型の擬似的な表現（GAT: Generic Associated Types）
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

## 9. 実務パターン集

### Builder パターン（ジェネリクス活用）

```rust
// Rust: 型状態パターン（Typestate Pattern）で安全な Builder
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

// build は name と email が両方設定された場合のみ呼び出し可能
impl UserBuilder<HasName, HasEmail> {
    fn build(self) -> User {
        User {
            name: self.name.0,
            email: self.email.0,
            age: self.age,
        }
    }
}

// 使用例
let user = UserBuilder::new()
    .name("Gaku".into())
    .email("gaku@example.com".into())
    .age(30)
    .build();

// これはコンパイルエラー（email が未設定）
// let invalid = UserBuilder::new().name("Gaku".into()).build();
```

### Strategy パターン

```typescript
// TypeScript: ジェネリクスを活用した Strategy パターン
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

// 使い方
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

### Result 型による型安全なエラーハンドリング

```typescript
// TypeScript: Result モナド的パターン
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

// パイプライン的に使う
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

## 10. アンチパターンと注意点

```
1. 過度なジェネリクス化
   → 読みやすさを犠牲にしない。具体的な型で十分な場合はジェネリクスを使わない
   → 「3回以上似たコードを書いたら」ジェネリクスを検討する

2. 型パラメータの乱用
   → 型パラメータが4つ以上になったら設計を見直す
   → 関連型（Associated Types）で減らせないか検討する

3. Java の型消去に起因する問題
   → 実行時にジェネリック型を検査できない
   → instanceof List<String> は不可能
   → 回避策: Class<T> トークンを引数で渡す

4. Go のインターフェース制約の限界
   → メソッドとユニオン型制約を混在できない制約がある
   → 複雑な制約は型アサーションで補完する必要がある場合がある

5. 共変配列の罠（Java, TypeScript）
   → Java: String[] は Object[] のサブタイプ → ArrayStoreException の可能性
   → TypeScript: 配列は共変 → 型安全性の穴

6. Rust のオブジェクト安全性
   → Self を返すメソッドや型パラメータを持つメソッドは dyn Trait に使えない
   → 回避策: 関連型やジェネリックラッパーを使う
```

```rust
// Rust: オブジェクト安全でないトレイト
trait Cloneable {
    fn clone(&self) -> Self;  // Self を返す → dyn Cloneable にできない
}

// 回避策: 型を消去するラッパー
trait CloneBox {
    fn clone_box(&self) -> Box<dyn CloneBox>;
}

impl<T: Clone + 'static> CloneBox for T {
    fn clone_box(&self) -> Box<dyn CloneBox> {
        Box::new(self.clone())
    }
}

// これなら dyn CloneBox として使える
let items: Vec<Box<dyn CloneBox>> = vec![
    Box::new(42),
    Box::new(String::from("hello")),
];
```

---

## 実践演習

### 演習1: [基礎] -- ジェネリックなコレクション
TypeScript または Rust でジェネリックな双方向キュー（Deque）を実装する。push_front, push_back, pop_front, pop_back, peek_front, peek_back, size, is_empty メソッドを持つ。

### 演習2: [応用] -- 型安全なイベントシステム
TypeScript でイベント名と引数の型を型パラメータで厳密に管理するイベントエミッターを実装する。on, off, emit, once メソッドを持ち、イベント名に対応しない引数型ではコンパイルエラーになること。

### 演習3: [応用] -- トレイトオブジェクトとジェネリクスの使い分け
Rust でプラグインシステムを設計する。静的ディスパッチと動的ディスパッチの両方のバージョンを実装し、ベンチマークで性能差を比較する。

### 演習4: [発展] -- 型レベルプログラミング
TypeScript の条件型とマップ型を使って、JSON Schema に相当する型定義から TypeScript の型を自動生成する型ユーティリティを実装する。

### 演習5: [発展] -- 高カインド型のシミュレーション
Rust の GAT（Generic Associated Types）を使って、Functor と Monad に相当するトレイトを定義し、Option と Result に対して実装する。

---

## まとめ

| 多態性の種類 | 仕組み | 例 |
|------------|--------|------|
| パラメトリック | 型をパラメータ化 | `List<T>`, `Vec<T>` |
| サブタイプ | 継承・インターフェース | `impl Trait`, `extends` |
| アドホック | 型ごとに異なる実装 | オーバーロード、型クラス |
| 型制約 | 型パラメータに条件を付ける | `T extends X`, `T: Trait` |
| 行多態性 | 特定フィールドを持つ型 | TypeScript 構造的部分型 |
| カインド多態性 | 型コンストラクタの抽象化 | `Functor f`, `Monad m` |

| 実装方式 | 特徴 | 代表言語 |
|---------|------|---------|
| 単相化 | コンパイル時展開、ゼロコスト | Rust, C++ |
| 型消去 | 実行時型情報なし | Java, TypeScript |
| 具体化 | 実行時型情報あり | C# |
| vtable | 動的ディスパッチ | Rust dyn, Java仮想メソッド |
| 辞書渡し | 型クラスメソッドテーブル | Haskell |

| 変性 | 意味 | 例 |
|------|------|-----|
| 共変 | サブタイプ関係が保存 | `&T`, `? extends T` |
| 反変 | サブタイプ関係が反転 | `fn(T)`, `? super T` |
| 不変 | サブタイプ関係なし | `&mut T`, `List<T>` |

---

## 次に読むべきガイド
→ [[03-algebraic-data-types.md]] -- 代数的データ型

---

## 参考文献
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
