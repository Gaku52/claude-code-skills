# ポリモーフィズム

> ポリモーフィズム（多態性）は「同じインターフェースで異なる実装を呼び出せる」仕組み。OOPの最も強力な概念であり、柔軟で拡張性の高い設計の基盤。

## この章で学ぶこと

- [ ] 3種類のポリモーフィズムを理解する
- [ ] 動的ディスパッチの仕組み（vtable）を把握する
- [ ] ポリモーフィズムの実践的な活用パターンを学ぶ

---

## 1. 3種類のポリモーフィズム

```
1. サブタイプポリモーフィズム（Subtype / Inclusion）
   → 親型の変数にサブクラスのオブジェクトを代入
   → OOPの「ポリモーフィズム」はこれを指すことが多い
   → 実行時に実際の型のメソッドが呼ばれる

2. パラメトリックポリモーフィズム（Parametric）
   → ジェネリクス。型パラメータで汎用的なコードを書く
   → List<T>, Map<K,V> など

3. アドホックポリモーフィズム（Ad-hoc）
   → メソッドオーバーロード。同名で引数の型が異なるメソッド
   → 演算子オーバーロードも含む
```

---

## 2. サブタイプポリモーフィズム

```
  Shape（インターフェース/抽象クラス）
    area(): number
    draw(): void
       ↑
  ┌────┼────┬───────────┐
  ▼    ▼    ▼           ▼
Circle  Rect  Triangle  Polygon
 area() area()  area()   area()
 各々が独自の実装を持つ

  shapes: Shape[] = [Circle, Rect, Triangle, ...]
  for (shape of shapes) {
    shape.area()  ← 実行時に正しい実装が呼ばれる
  }
```

```typescript
// TypeScript: サブタイプポリモーフィズム
interface Shape {
  area(): number;
  perimeter(): number;
  describe(): string;
}

class Circle implements Shape {
  constructor(private radius: number) {}

  area(): number {
    return Math.PI * this.radius ** 2;
  }

  perimeter(): number {
    return 2 * Math.PI * this.radius;
  }

  describe(): string {
    return `円（半径: ${this.radius}）`;
  }
}

class Rectangle implements Shape {
  constructor(private width: number, private height: number) {}

  area(): number {
    return this.width * this.height;
  }

  perimeter(): number {
    return 2 * (this.width + this.height);
  }

  describe(): string {
    return `長方形（${this.width} × ${this.height}）`;
  }
}

// ポリモーフィズム: Shape型で統一的に扱う
function printShapeInfo(shape: Shape): void {
  console.log(`${shape.describe()}: 面積=${shape.area().toFixed(2)}`);
}

const shapes: Shape[] = [
  new Circle(5),
  new Rectangle(3, 4),
  new Circle(10),
];

shapes.forEach(printShapeInfo);
// 円（半径: 5）: 面積=78.54
// 長方形（3 × 4）: 面積=12.00
// 円（半径: 10）: 面積=314.16
```

---

## 3. 動的ディスパッチの仕組み（vtable）

```
仮想関数テーブル（vtable / Virtual Method Table）:
  → C++, Java, C# 等で使われる実装メカニズム
  → 各クラスが持つメソッドポインタの配列
  → 実行時にオブジェクトの実際の型に基づいてメソッドを選択

  メモリレイアウト:

  Circle オブジェクト            Circle の vtable
  ┌──────────────┐            ┌──────────────────┐
  │ vptr ────────┼───────────→│ area() → Circle実装│
  │ radius: 5.0  │            │ perimeter() → ...  │
  └──────────────┘            │ describe() → ...   │
                              └──────────────────┘

  Rectangle オブジェクト         Rectangle の vtable
  ┌──────────────┐            ┌──────────────────┐
  │ vptr ────────┼───────────→│ area() → Rect実装  │
  │ width: 3.0   │            │ perimeter() → ...  │
  │ height: 4.0  │            │ describe() → ...   │
  └──────────────┘            └──────────────────┘

  shape.area() の呼び出し:
    1. shape の vptr を取得
    2. vtable から area() のアドレスを取得
    3. そのアドレスのメソッドを呼ぶ

  コスト: ポインタ間接参照1回分（ほぼゼロコスト）
```

---

## 4. 実践的な活用パターン

### Strategy パターン

```typescript
// 支払い方法のポリモーフィズム
interface PaymentStrategy {
  pay(amount: number): Promise<PaymentResult>;
  validate(): boolean;
}

class CreditCardPayment implements PaymentStrategy {
  constructor(private cardNumber: string, private cvv: string) {}

  async pay(amount: number): Promise<PaymentResult> {
    // クレジットカード決済処理
    return { success: true, transactionId: "cc-123" };
  }

  validate(): boolean {
    return this.cardNumber.length === 16 && this.cvv.length === 3;
  }
}

class PayPayPayment implements PaymentStrategy {
  constructor(private userId: string) {}

  async pay(amount: number): Promise<PaymentResult> {
    // PayPay決済処理
    return { success: true, transactionId: "pp-456" };
  }

  validate(): boolean {
    return this.userId.length > 0;
  }
}

// 利用側: PaymentStrategy のみに依存
class Checkout {
  async process(strategy: PaymentStrategy, amount: number): Promise<void> {
    if (!strategy.validate()) {
      throw new Error("決済情報が無効です");
    }
    const result = await strategy.pay(amount);
    // 新しい決済方法が追加されても、このコードは変更不要
  }
}
```

### プラグインシステム

```python
# Python: プラグインシステム
from abc import ABC, abstractmethod

class FileExporter(ABC):
    @abstractmethod
    def export(self, data: list[dict]) -> bytes:
        ...

    @abstractmethod
    def file_extension(self) -> str:
        ...

class CsvExporter(FileExporter):
    def export(self, data: list[dict]) -> bytes:
        if not data:
            return b""
        headers = ",".join(data[0].keys())
        rows = [",".join(str(v) for v in row.values()) for row in data]
        return (headers + "\n" + "\n".join(rows)).encode()

    def file_extension(self) -> str:
        return ".csv"

class JsonExporter(FileExporter):
    def export(self, data: list[dict]) -> bytes:
        import json
        return json.dumps(data, ensure_ascii=False).encode()

    def file_extension(self) -> str:
        return ".json"

# 利用側: FileExporter のみに依存
def save_report(exporter: FileExporter, data: list[dict], filename: str):
    content = exporter.export(data)
    path = f"{filename}{exporter.file_extension()}"
    with open(path, "wb") as f:
        f.write(content)
```

---

## 5. アドホックポリモーフィズム

```java
// Java: メソッドオーバーロード
public class Calculator {
    // 同名メソッドで引数の型が異なる
    public int add(int a, int b) { return a + b; }
    public double add(double a, double b) { return a + b; }
    public String add(String a, String b) { return a + b; }
}

// コンパイル時に呼ぶメソッドが決定（静的ディスパッチ）
```

```python
# Python: 演算子オーバーロード
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> "Vector":
        return Vector(self.x * scalar, self.y * scalar)

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)      # Vector(4, 6)
print(v1 * 3)       # Vector(3, 6)
```

---

## 6. 静的 vs 動的ディスパッチ

```
静的ディスパッチ（コンパイル時に決定）:
  → メソッドオーバーロード
  → Rust のジェネリクス（単相化）
  → C++ のテンプレート
  → 高速（インライン化可能）

動的ディスパッチ（実行時に決定）:
  → サブタイプポリモーフィズム
  → Java/C# の仮想メソッド
  → Rust の dyn Trait
  → 柔軟（実行時にオブジェクトの型で分岐）
  → vtable のオーバーヘッドあり（ほぼゼロだが）
```

```rust
// Rust: 静的 vs 動的ディスパッチの選択
trait Drawable {
    fn draw(&self);
}

struct Circle { radius: f64 }
struct Rect { width: f64, height: f64 }

impl Drawable for Circle {
    fn draw(&self) { println!("Drawing circle r={}", self.radius); }
}
impl Drawable for Rect {
    fn draw(&self) { println!("Drawing rect {}x{}", self.width, self.height); }
}

// 静的ディスパッチ（ジェネリクス）: コンパイル時に型が確定
fn draw_static<T: Drawable>(item: &T) {
    item.draw(); // インライン化可能、高速
}

// 動的ディスパッチ（トレイトオブジェクト）: 実行時に型が確定
fn draw_dynamic(item: &dyn Drawable) {
    item.draw(); // vtable 経由、柔軟
}

// 動的ディスパッチが必要な場面: 異なる型のコレクション
let shapes: Vec<Box<dyn Drawable>> = vec![
    Box::new(Circle { radius: 5.0 }),
    Box::new(Rect { width: 3.0, height: 4.0 }),
];
```

---

## まとめ

| 種類 | 決定時期 | 手段 | 典型例 |
|------|---------|------|--------|
| サブタイプ | 実行時 | 継承/インターフェース | Strategy, Plugin |
| パラメトリック | コンパイル時 | ジェネリクス | List<T>, Map<K,V> |
| アドホック | コンパイル時 | オーバーロード | add(int), add(double) |

---

## 次に読むべきガイド
→ [[03-abstraction.md]] — 抽象化

---

## 参考文献
1. Cardelli, L. "On Understanding Types, Data Abstraction, and Polymorphism." 1985.
2. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
