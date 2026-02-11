# ジェネリクスと多態性（Polymorphism）

> 多態性は「同じコードで異なる型を扱う」能力。コードの再利用性と型安全性を両立する鍵。

## この章で学ぶこと

- [ ] 多態性の3つの種類を理解する
- [ ] ジェネリクスの実装と使い方を習得する
- [ ] 型制約（Bounded Polymorphism）を活用できる

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
```

---

## 3. サブタイプ多態性

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
```

```rust
// Rust: トレイトによる多態性
trait Drawable {
    fn draw(&self);
}

struct Circle { radius: f64 }
struct Rectangle { width: f64, height: f64 }

impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle with radius {}", self.radius);
    }
}

impl Drawable for Rectangle {
    fn draw(&self) {
        println!("Drawing {}x{} rectangle", self.width, self.height);
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
    Box::new(Circle { radius: 5.0 }),
    Box::new(Rectangle { width: 3.0, height: 4.0 }),
];
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
}
// 同じメソッド名で引数の型が異なる
```

### 型クラス（Haskell / Rust のトレイト）

```haskell
-- Haskell: 型クラス（アドホック多態性の最も洗練された形）
class Eq a where
    (==) :: a -> a -> Bool

instance Eq Int where
    x == y = eqInt x y

instance Eq String where
    x == y = eqString x y

-- 型ごとに異なる == の実装を持つ
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
```

---

## 5. 型制約（Bounded Polymorphism）

```typescript
// TypeScript: extends による型制約
function getLength<T extends { length: number }>(item: T): number {
    return item.length;
}

getLength("hello");     // ✅ string は length を持つ
getLength([1, 2, 3]);   // ✅ array は length を持つ
getLength(42);           // ❌ number は length を持たない

// keyof 制約
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
    return obj[key];
}

const user = { name: "Gaku", age: 30 };
getProperty(user, "name");  // ✅ → string
getProperty(user, "foo");   // ❌ "foo" は keyof User にない
```

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
```

---

## 6. 実装方式の違い

```
単相化（Monomorphization）— Rust, C++
  コンパイル時に型ごとにコードを生成
  Vec<i32> と Vec<String> は別のコードになる
  利点: ゼロコスト抽象化（実行時オーバーヘッドなし）
  欠点: バイナリサイズが増加

型消去（Type Erasure）— Java, TypeScript
  コンパイル後にジェネリクスの型情報を削除
  List<Integer> と List<String> は実行時に同じ List
  利点: バイナリサイズが小さい、後方互換性
  欠点: 実行時に型情報が失われる

ボックス化 + vtable — Rust の dyn Trait
  実行時に仮想関数テーブルで解決
  利点: 異なる型を同一コレクションに格納可能
  欠点: 間接参照のオーバーヘッド
```

---

## まとめ

| 多態性の種類 | 仕組み | 例 |
|------------|--------|------|
| パラメトリック | 型をパラメータ化 | `List<T>`, `Vec<T>` |
| サブタイプ | 継承・インターフェース | `impl Trait`, `extends` |
| アドホック | 型ごとに異なる実装 | オーバーロード、型クラス |
| 型制約 | 型パラメータに条件を付ける | `T extends X`, `T: Trait` |

---

## 次に読むべきガイド
→ [[03-algebraic-data-types.md]] — 代数的データ型

---

## 参考文献
1. Pierce, B. "Types and Programming Languages." Ch.23-26, MIT Press, 2002.
2. Wadler, P. & Blott, S. "How to make ad-hoc polymorphism less ad hoc." 1989.
