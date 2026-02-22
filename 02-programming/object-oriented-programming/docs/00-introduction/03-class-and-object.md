# クラスとオブジェクト

> クラスは「設計図」、オブジェクトは「実体」。この関係を深く理解し、メモリ上の配置、コンストラクタの設計、静的メンバの適切な使い方をマスターする。

## この章で学ぶこと

- [ ] クラスとオブジェクトの関係をメモリレベルで理解する
- [ ] コンストラクタの設計パターンを把握する
- [ ] 静的メンバとインスタンスメンバの使い分けを学ぶ
- [ ] オブジェクトのライフサイクル（生成・利用・破棄）を理解する
- [ ] 値型と参照型の違いとコピーセマンティクスを把握する
- [ ] クラス設計のベストプラクティスを実践できるようになる

---

## 1. クラスとオブジェクトの関係

```
クラス（Class）= 設計図・型
  → メモリ上に1つだけ存在（メタデータ）
  → フィールドの定義、メソッドの実装を保持

オブジェクト（Object）= インスタンス・実体
  → メモリ上に複数存在可能
  → 各オブジェクトが独自の状態（フィールド値）を持つ

  ┌─────────── Class: User ───────────┐
  │ 設計図（メタデータ）               │
  │   fields: name, age, email        │
  │   methods: greet(), isAdult()     │
  └─────────────────────────────────────┘
           │ new User(...)
     ┌─────┴─────┬─────────────┐
     ▼           ▼             ▼
  ┌────────┐ ┌────────┐ ┌────────┐
  │ obj_1  │ │ obj_2  │ │ obj_3  │
  │ 田中   │ │ 山田   │ │ 佐藤   │
  │ 25歳   │ │ 30歳   │ │ 17歳   │
  └────────┘ └────────┘ └────────┘
  0x1000      0x2000      0x3000
  各オブジェクトは独自のメモリ領域を持つ
```

### 1.1 クラスの3つの役割

クラスは単なるデータ構造の定義ではなく、3つの重要な役割を担う。

```
1. 型（Type）としての役割:
   → 変数の型を定義する
   → コンパイル時の型チェックを可能にする
   → インターフェースの契約を表現する

2. テンプレート（Template）としての役割:
   → オブジェクト生成の設計図
   → フィールドのレイアウトを定義
   → メソッドの実装を保持

3. モジュール（Module）としての役割:
   → 関連する機能をカプセル化
   → 名前空間を提供
   → アクセス制御の境界を設定
```

```typescript
// TypeScript: クラスの3つの役割を示す例

// 1. 型として: 変数の型注釈に使える
class Product {
  constructor(
    public readonly id: string,
    public readonly name: string,
    public readonly price: number,
    public readonly category: string,
  ) {}

  // 2. テンプレートとして: フィールドとメソッドを定義
  getDisplayPrice(): string {
    return `${this.price.toLocaleString()}円`;
  }

  isInCategory(category: string): boolean {
    return this.category === category;
  }

  // 3. モジュールとして: 関連機能をまとめる
  applyDiscount(rate: number): Product {
    if (rate < 0 || rate > 1) {
      throw new Error("割引率は0〜1の範囲で指定してください");
    }
    return new Product(
      this.id,
      this.name,
      Math.floor(this.price * (1 - rate)),
      this.category,
    );
  }
}

// 型として利用
function findExpensiveProducts(products: Product[], threshold: number): Product[] {
  return products.filter(p => p.price >= threshold);
}

// テンプレートとして利用（インスタンス生成）
const laptop = new Product("P001", "MacBook Pro", 298000, "Electronics");
const phone = new Product("P002", "iPhone 15", 149800, "Electronics");

// モジュールとして利用（オブジェクトに指示）
const discountedLaptop = laptop.applyDiscount(0.1);
console.log(discountedLaptop.getDisplayPrice()); // "268,200円"
```

### 1.2 オブジェクトの3つの特性

すべてのオブジェクトは「状態」「振る舞い」「同一性」の3つの特性を持つ。

```python
# Python: オブジェクトの3つの特性

class Employee:
    """従業員クラス: 状態・振る舞い・同一性を持つ"""

    _next_id = 1

    def __init__(self, name: str, department: str, salary: int):
        # 同一性（Identity）: 各オブジェクトを一意に識別
        self._id = Employee._next_id
        Employee._next_id += 1

        # 状態（State）: オブジェクト固有のデータ
        self._name = name
        self._department = department
        self._salary = salary
        self._is_active = True

    # 振る舞い（Behavior）: オブジェクトが実行できる操作

    def promote(self, raise_amount: int) -> None:
        """昇進: 給与を増額する"""
        if not self._is_active:
            raise RuntimeError(f"{self._name}は退職済みです")
        if raise_amount <= 0:
            raise ValueError("昇給額は正の数である必要があります")
        self._salary += raise_amount

    def transfer(self, new_department: str) -> None:
        """部署異動"""
        if not self._is_active:
            raise RuntimeError(f"{self._name}は退職済みです")
        old = self._department
        self._department = new_department
        print(f"{self._name}: {old} → {new_department}")

    def resign(self) -> None:
        """退職処理"""
        self._is_active = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def employee_id(self) -> int:
        return self._id

    def __repr__(self) -> str:
        status = "在職" if self._is_active else "退職"
        return (
            f"Employee(id={self._id}, name='{self._name}', "
            f"dept='{self._department}', salary={self._salary}, status={status})"
        )

    def __eq__(self, other: object) -> bool:
        """同一性の比較: IDが同じなら同一人物"""
        if not isinstance(other, Employee):
            return NotImplemented
        return self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)


# 使用例
emp1 = Employee("田中太郎", "開発部", 500000)
emp2 = Employee("山田花子", "営業部", 450000)

# 状態の確認
print(emp1)  # Employee(id=1, name='田中太郎', dept='開発部', salary=500000, status=在職)

# 振る舞いの実行
emp1.promote(50000)
emp1.transfer("技術部")  # 田中太郎: 開発部 → 技術部

# 同一性の確認
emp1_copy = emp1  # 参照のコピー
print(emp1 is emp1_copy)   # True（同一オブジェクト）
print(emp1 == emp1_copy)   # True（IDが同じ）
print(emp1 == emp2)        # False（IDが異なる）
```

### 1.3 クラスとオブジェクトの関係を各言語で

```java
// Java: クラスとオブジェクトの基本的な関係

public class Book {
    // フィールド（状態の定義）
    private final String isbn;
    private final String title;
    private final String author;
    private int stock;
    private boolean isAvailable;

    // コンストラクタ（初期化）
    public Book(String isbn, String title, String author, int stock) {
        this.isbn = isbn;
        this.title = title;
        this.author = author;
        this.stock = stock;
        this.isAvailable = stock > 0;
    }

    // メソッド（振る舞いの定義）
    public boolean borrow() {
        if (stock <= 0) {
            return false;
        }
        stock--;
        isAvailable = stock > 0;
        return true;
    }

    public void returnBook() {
        stock++;
        isAvailable = true;
    }

    public String getInfo() {
        return String.format(
            "『%s』(%s) by %s [在庫: %d]",
            title, isbn, author, stock
        );
    }

    // 同一性: ISBNで判定
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Book other)) return false;
        return isbn.equals(other.isbn);
    }

    @Override
    public int hashCode() {
        return isbn.hashCode();
    }
}

// クラスからオブジェクトを生成
Book book1 = new Book("978-4-xxx", "デザインパターン", "GoF", 3);
Book book2 = new Book("978-4-yyy", "リファクタリング", "Fowler", 2);

// 各オブジェクトは独立した状態を持つ
book1.borrow();  // book1の在庫のみ減少
System.out.println(book1.getInfo()); // 在庫: 2
System.out.println(book2.getInfo()); // 在庫: 2（影響なし）
```

```kotlin
// Kotlin: より簡潔なクラス定義

class BankAccount(
    val accountNumber: String,
    val owner: String,
    initialBalance: Long = 0
) {
    var balance: Long = initialBalance
        private set  // setterはprivate

    private val transactions = mutableListOf<String>()

    fun deposit(amount: Long): BankAccount {
        require(amount > 0) { "入金額は正の数である必要があります" }
        balance += amount
        transactions.add("入金: +${amount}円")
        return this
    }

    fun withdraw(amount: Long): BankAccount {
        require(amount > 0) { "出金額は正の数である必要があります" }
        require(balance >= amount) { "残高不足です（残高: ${balance}円、出金: ${amount}円）" }
        balance -= amount
        transactions.add("出金: -${amount}円")
        return this
    }

    fun getStatement(): String {
        val header = "=== 口座明細 ===\n口座番号: $accountNumber\n名義: $owner\n"
        val body = transactions.joinToString("\n")
        val footer = "\n残高: ${balance}円"
        return header + body + footer
    }
}

// オブジェクト生成と利用
val account = BankAccount("1234-5678", "田中太郎", 100000)
account.deposit(50000).withdraw(30000)
println(account.getStatement())
// === 口座明細 ===
// 口座番号: 1234-5678
// 名義: 田中太郎
// 入金: +50000円
// 出金: -30000円
// 残高: 120000円
```

```rust
// Rust: 構造体とimplブロックによるクラス相当の表現

use std::fmt;

struct Rectangle {
    width: f64,
    height: f64,
}

impl Rectangle {
    // 関連関数（コンストラクタ相当）
    fn new(width: f64, height: f64) -> Self {
        assert!(width > 0.0, "幅は正の数である必要があります");
        assert!(height > 0.0, "高さは正の数である必要があります");
        Rectangle { width, height }
    }

    fn square(size: f64) -> Self {
        Rectangle::new(size, size)
    }

    // メソッド（&self で不変借用）
    fn area(&self) -> f64 {
        self.width * self.height
    }

    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }

    fn is_square(&self) -> bool {
        (self.width - self.height).abs() < f64::EPSILON
    }

    // &mut self で可変借用
    fn scale(&mut self, factor: f64) {
        assert!(factor > 0.0, "拡大率は正の数である必要があります");
        self.width *= factor;
        self.height *= factor;
    }

    // self を消費して新しい値を返す
    fn rotate(self) -> Rectangle {
        Rectangle::new(self.height, self.width)
    }
}

impl fmt::Display for Rectangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rectangle({}x{}, 面積={})", self.width, self.height, self.area())
    }
}

fn main() {
    let mut rect = Rectangle::new(10.0, 5.0);
    println!("{}", rect);  // Rectangle(10x5, 面積=50)

    rect.scale(2.0);
    println!("{}", rect);  // Rectangle(20x10, 面積=200)

    let rotated = rect.rotate(); // rect は move される
    println!("{}", rotated); // Rectangle(10x20, 面積=200)
    // println!("{}", rect);  // コンパイルエラー: rect は move 済み
}
```

---

## 2. メモリ上の配置

```
Java/C# のメモリモデル:

  スタック                    ヒープ
  ┌────────────┐            ┌─────────────────────┐
  │ user1 ref ─┼───────────→│ User object          │
  │  (8 bytes) │            │ ┌───────────────────┐│
  ├────────────┤            │ │ header (16 bytes) ││
  │ user2 ref ─┼──────┐    │ │ クラスポインタ    ││
  │  (8 bytes) │      │    │ │ ハッシュコード    ││
  └────────────┘      │    │ │ ロック情報        ││
                      │    │ ├───────────────────┤│
                      │    │ │ name: "田中"  ref ││→ String object
                      │    │ │ age: 25           ││
                      │    │ │ email: ref        ││→ String object
                      │    │ └───────────────────┘│
                      │    ├─────────────────────┤
                      └───→│ User object          │
                           │ name: "山田", age:30 │
                           └─────────────────────┘

C++ のメモリモデル:
  → スタック上にもヒープ上にも配置可能
  User user1("田中", 25);      // スタック上
  User* user2 = new User("山田", 30); // ヒープ上

Python のメモリモデル:
  → 全てがヒープ上のオブジェクト
  → 変数は全て参照（ポインタ）
```

### 2.1 メモリレイアウトの詳細

各言語のランタイムによって、オブジェクトのメモリレイアウトは大きく異なる。

```
Java のオブジェクトヘッダー（64bit JVM, Compressed Oops有効時）:

  ┌─────────────────────────────────────────┐
  │ Mark Word (8 bytes)                     │
  │   - ハッシュコード (31 bits)              │
  │   - GC年齢 (4 bits)                      │
  │   - ロック情報 (2 bits)                   │
  │   - バイアスロック情報 (1 bit)             │
  ├─────────────────────────────────────────┤
  │ Class Pointer (4 bytes, compressed)     │
  │   → メソッドテーブル（vtable）へのポインタ │
  ├─────────────────────────────────────────┤
  │ Padding (4 bytes)                       │
  │   → 8バイトアラインメントのため            │
  ├─────────────────────────────────────────┤
  │ Instance Fields                         │
  │   → フィールドの値（プリミティブ or 参照）  │
  └─────────────────────────────────────────┘

  合計: ヘッダー 16 bytes + フィールド

  例: class Point { int x; int y; }
  → 16 (header) + 4 (x) + 4 (y) = 24 bytes

  例: class User { String name; int age; String email; }
  → 16 (header) + 4 (name ref) + 4 (age) + 4 (email ref) + 4 (padding) = 32 bytes

フィールドの並び替え:
  JVMはメモリ効率のためにフィールドの順序を並び替える:
  1. double / long  (8 bytes)
  2. int / float    (4 bytes)
  3. short / char   (2 bytes)
  4. byte / boolean (1 byte)
  5. 参照型         (4 bytes, Compressed Oops)
```

```java
// Java: オブジェクトサイズの推測と最適化

// 悪い例: 無駄なメモリ使用
public class WastefulObject {
    private boolean flag1;    // 1 byte → 8 bytes (padding)
    private long value;       // 8 bytes
    private boolean flag2;    // 1 byte → 8 bytes (padding)
    private long timestamp;   // 8 bytes
    // ヘッダー16 + 32 = 48 bytes

    // JVMが並び替えて最適化:
    // long value;      → 8 bytes
    // long timestamp;  → 8 bytes
    // boolean flag1;   → 1 byte
    // boolean flag2;   → 1 byte + 6 bytes padding
    // 実際: ヘッダー16 + 24 = 40 bytes
}

// メモリ効率を意識した設計
public class EfficientObject {
    private long value;       // 8 bytes
    private long timestamp;   // 8 bytes
    private int count;        // 4 bytes
    private short type;       // 2 bytes
    private boolean flag1;    // 1 byte
    private boolean flag2;    // 1 byte
    // ヘッダー16 + 24 = 40 bytes（パディング最小）
}
```

### 2.2 メソッドテーブル（vtable）

```
メソッドの格納場所:

  メソッドはオブジェクトごとにコピーされない。
  クラスのメタデータ領域に1つだけ存在し、全インスタンスが共有する。

  ┌─────── Class Metadata: Animal ─────────┐
  │ vtable (Virtual Method Table):          │
  │   [0] speak()  → 0x4000 (Animal.speak)│
  │   [1] move()   → 0x4100 (Animal.move) │
  │   [2] eat()    → 0x4200 (Animal.eat)  │
  └─────────────────────────────────────────┘

  ┌─────── Class Metadata: Dog ────────────┐
  │ vtable (Virtual Method Table):          │
  │   [0] speak()  → 0x5000 (Dog.speak)   │  ← オーバーライド
  │   [1] move()   → 0x4100 (Animal.move) │  ← 継承
  │   [2] eat()    → 0x5200 (Dog.eat)     │  ← オーバーライド
  │   [3] fetch()  → 0x5300 (Dog.fetch)   │  ← 新規追加
  └─────────────────────────────────────────┘

  Animal animal = new Dog();
  animal.speak();
  → animal のクラスポインタ → Dog のメタデータ → vtable[0] → Dog.speak()
  → これが「動的ディスパッチ」の仕組み
```

```typescript
// TypeScript: プロトタイプチェーンによるメソッド共有

class Animal {
  constructor(public name: string) {}

  speak(): string {
    return `${this.name}は音を出します`;
  }

  move(distance: number): string {
    return `${this.name}は${distance}m移動しました`;
  }
}

class Dog extends Animal {
  constructor(name: string, public breed: string) {
    super(name);
  }

  speak(): string {
    return `${this.name}（${this.breed}）: ワンワン！`;
  }

  fetch(item: string): string {
    return `${this.name}は${item}を取ってきました`;
  }
}

// JavaScriptのメモリモデル:
// dog1.__proto__ → Dog.prototype → Animal.prototype → Object.prototype
// メソッドはプロトタイプチェーン上に存在し、
// 全インスタンスが共有する

const dog1 = new Dog("ポチ", "柴犬");
const dog2 = new Dog("ハチ", "秋田犬");

// dog1.speak と dog2.speak は同じ関数オブジェクトを参照
console.log(dog1.speak === dog2.speak); // true

// プロトタイプチェーンの確認
console.log(dog1 instanceof Dog);    // true
console.log(dog1 instanceof Animal); // true
```

### 2.3 ガベージコレクションとオブジェクトのライフサイクル

```
オブジェクトのライフサイクル:

  1. 割り当て（Allocation）
     → new でヒープ上にメモリ確保
     → コンストラクタで初期化

  2. 利用（Usage）
     → メソッド呼び出し、フィールドアクセス
     → 参照を通じてオブジェクトにアクセス

  3. 到達不能（Unreachable）
     → すべての参照がなくなる
     → GCの回収対象になる

  4. 回収（Collection）
     → GCがメモリを解放
     → ファイナライザ/デストラクタが呼ばれる（言語による）

  ┌──────────┐     ┌──────────┐     ┌────────────┐     ┌──────────┐
  │ 割り当て │ ──→ │   利用   │ ──→ │ 到達不能   │ ──→ │   回収   │
  │ (new)    │     │ (使用中) │     │ (参照なし) │     │ (GC)    │
  └──────────┘     └──────────┘     └────────────┘     └──────────┘
```

```python
# Python: オブジェクトのライフサイクルとガベージコレクション

import gc
import weakref

class Resource:
    """リソース管理を示すクラス"""

    _instance_count = 0

    def __init__(self, name: str):
        self.name = name
        Resource._instance_count += 1
        print(f"[生成] {self.name} (総数: {Resource._instance_count})")

    def __del__(self):
        Resource._instance_count -= 1
        print(f"[破棄] {self.name} (残: {Resource._instance_count})")

    def process(self) -> str:
        return f"{self.name}: 処理実行"


def demonstrate_lifecycle():
    # 1. 割り当て
    r1 = Resource("リソースA")
    r2 = Resource("リソースB")
    r3 = Resource("リソースC")

    # 2. 利用
    print(r1.process())
    print(r2.process())

    # 3. 参照の解放
    r2 = None  # リソースBへの参照がなくなる → GC対象

    # 弱参照: オブジェクトの生存確認（参照カウントに影響しない）
    weak_r3 = weakref.ref(r3)
    print(f"r3は生存中: {weak_r3() is not None}")  # True

    r3 = None  # リソースCへの参照がなくなる
    print(f"r3は生存中: {weak_r3() is not None}")  # False

    # 4. 明示的GC実行
    gc.collect()

    print(f"残存オブジェクト: {Resource._instance_count}")
    # r1 はまだスコープ内なので残存


# コンテキストマネージャによる確実なリソース解放
class DatabaseConnection:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._connection = None

    def __enter__(self):
        print(f"接続開始: {self.host}:{self.port}")
        self._connection = f"Connection({self.host}:{self.port})"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"接続終了: {self.host}:{self.port}")
        self._connection = None
        return False  # 例外を再送出

    def execute(self, query: str) -> str:
        if self._connection is None:
            raise RuntimeError("接続されていません")
        return f"実行: {query}"


# with文でライフサイクルを明示的に管理
with DatabaseConnection("localhost", 5432) as db:
    result = db.execute("SELECT * FROM users")
    print(result)
# ← ここで自動的に __exit__ が呼ばれる
```

```cpp
// C++: 手動メモリ管理とスマートポインタ

#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Sensor {
public:
    Sensor(const std::string& name, double threshold)
        : name_(name), threshold_(threshold), reading_(0.0) {
        std::cout << "[生成] " << name_ << std::endl;
    }

    ~Sensor() {
        std::cout << "[破棄] " << name_ << std::endl;
    }

    // コピー禁止（リソースの重複を防ぐ）
    Sensor(const Sensor&) = delete;
    Sensor& operator=(const Sensor&) = delete;

    // ムーブは許可
    Sensor(Sensor&& other) noexcept
        : name_(std::move(other.name_)),
          threshold_(other.threshold_),
          reading_(other.reading_) {
        std::cout << "[ムーブ] " << name_ << std::endl;
    }

    void update(double value) {
        reading_ = value;
    }

    bool isAlarm() const {
        return reading_ > threshold_;
    }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    double threshold_;
    double reading_;
};

int main() {
    // 1. スタック上のオブジェクト（自動管理）
    {
        Sensor temp("温度センサー", 40.0);
        temp.update(42.0);
        std::cout << temp.name() << " アラーム: "
                  << (temp.isAlarm() ? "Yes" : "No") << std::endl;
    } // ← スコープを抜けると自動的にデストラクタが呼ばれる

    // 2. unique_ptr（排他的所有権）
    auto humidity = std::make_unique<Sensor>("湿度センサー", 80.0);
    humidity->update(75.0);

    // 所有権の移動
    auto transferred = std::move(humidity);
    // humidity は nullptr になる

    // 3. shared_ptr（共有所有権）
    auto pressure = std::make_shared<Sensor>("気圧センサー", 1050.0);
    {
        auto shared_ref = pressure;  // 参照カウント: 2
        std::cout << "参照カウント: " << pressure.use_count() << std::endl;
    } // shared_ref のスコープ終了 → 参照カウント: 1

    std::cout << "参照カウント: " << pressure.use_count() << std::endl;

    return 0;
}
// 全スマートポインタのスコープ終了 → 自動的にデストラクタ呼び出し
```

### 2.4 参照カウントとサイクル検出

```
参照カウント方式（Python, Swift, Rust の Arc）:

  各オブジェクトが「何箇所から参照されているか」をカウント。
  カウントが0になった時点で即座に解放。

  問題: 循環参照
    A → B → A  （互いに参照し合う）
    → どちらの参照カウントも0にならない
    → メモリリーク

  解決策:
    1. 弱参照（weak reference）を使う
    2. サイクル検出GC（Python の gc モジュール）
    3. 所有権モデル（Rust）で循環参照を構造的に防止
```

```python
# Python: 循環参照の問題と対策

import gc
import weakref

class Parent:
    def __init__(self, name: str):
        self.name = name
        self.children: list["Child"] = []

    def add_child(self, child: "Child") -> None:
        self.children.append(child)

    def __repr__(self) -> str:
        return f"Parent({self.name})"

    def __del__(self):
        print(f"[GC] Parent({self.name}) destroyed")

class Child:
    def __init__(self, name: str, parent: Parent):
        self.name = name
        # 問題: 強参照 → 循環参照が発生
        # self.parent = parent

        # 解決: 弱参照を使う
        self._parent_ref = weakref.ref(parent)

    @property
    def parent(self) -> Parent | None:
        return self._parent_ref()

    def __repr__(self) -> str:
        return f"Child({self.name})"

    def __del__(self):
        print(f"[GC] Child({self.name}) destroyed")

def demo_weak_ref():
    p = Parent("太郎")
    c1 = Child("一郎", p)
    c2 = Child("次郎", p)
    p.add_child(c1)
    p.add_child(c2)

    print(c1.parent)  # Parent(太郎)

    del p  # Parent が解放される
    print(c1.parent)  # None（弱参照が無効になった）

demo_weak_ref()
gc.collect()
```

---

## 3. コンストラクタ

```
コンストラクタの役割:
  1. フィールドの初期化
  2. 不変条件（invariant）の確立
  3. 依存オブジェクトの注入
```

### 3.1 基本的なコンストラクタパターン

```typescript
// TypeScript: コンストラクタの設計パターン

// 基本: 必須パラメータのみ
class User {
  constructor(
    public readonly name: string,
    public readonly email: string,
  ) {}
}

// オプショナルパラメータ
class HttpClient {
  private baseUrl: string;
  private timeout: number;
  private retries: number;

  constructor(baseUrl: string, options?: { timeout?: number; retries?: number }) {
    this.baseUrl = baseUrl;
    this.timeout = options?.timeout ?? 5000;
    this.retries = options?.retries ?? 3;
  }
}

// ファクトリメソッド（コンストラクタの代替）
class Temperature {
  private constructor(private readonly kelvin: number) {}

  static fromCelsius(c: number): Temperature {
    return new Temperature(c + 273.15);
  }

  static fromFahrenheit(f: number): Temperature {
    return new Temperature((f - 32) * 5 / 9 + 273.15);
  }

  toCelsius(): number {
    return this.kelvin - 273.15;
  }
}

// 使い方が明確
const temp = Temperature.fromCelsius(100);
// new Temperature(373.15) はプライベートなので不可
```

### 3.2 コンストラクタでの不変条件の確立

```typescript
// TypeScript: 不変条件をコンストラクタで保証する

class EmailAddress {
  private readonly value: string;

  constructor(email: string) {
    const trimmed = email.trim().toLowerCase();

    // 不変条件1: 空文字列でないこと
    if (trimmed.length === 0) {
      throw new Error("メールアドレスは空にできません");
    }

    // 不変条件2: @を含むこと
    if (!trimmed.includes("@")) {
      throw new Error("メールアドレスには@が必要です");
    }

    // 不変条件3: ドメイン部分があること
    const [local, domain] = trimmed.split("@");
    if (!local || !domain || !domain.includes(".")) {
      throw new Error("メールアドレスの形式が不正です");
    }

    // 不変条件4: 長さ制限
    if (trimmed.length > 254) {
      throw new Error("メールアドレスが長すぎます（最大254文字）");
    }

    this.value = trimmed;
  }

  toString(): string {
    return this.value;
  }

  equals(other: EmailAddress): boolean {
    return this.value === other.value;
  }

  getDomain(): string {
    return this.value.split("@")[1];
  }
}

class Age {
  private readonly value: number;

  constructor(value: number) {
    if (!Number.isInteger(value)) {
      throw new Error("年齢は整数である必要があります");
    }
    if (value < 0 || value > 150) {
      throw new Error("年齢は0〜150の範囲である必要があります");
    }
    this.value = value;
  }

  toNumber(): number {
    return this.value;
  }

  isAdult(): boolean {
    return this.value >= 18;
  }

  isElderly(): boolean {
    return this.value >= 65;
  }
}

class UserProfile {
  constructor(
    public readonly name: string,
    public readonly email: EmailAddress,
    public readonly age: Age,
  ) {
    // 名前の不変条件
    if (name.trim().length === 0) {
      throw new Error("名前は空にできません");
    }
    if (name.length > 100) {
      throw new Error("名前は100文字以内である必要があります");
    }
  }

  getInfo(): string {
    return `${this.name} (${this.age.toNumber()}歳) - ${this.email}`;
  }
}

// 利用例: 不正なデータではオブジェクトが生成されない
try {
  const email = new EmailAddress("invalid-email");
} catch (e) {
  console.error(e); // "メールアドレスには@が必要です"
}

// 正常なデータのみでオブジェクト生成可能
const profile = new UserProfile(
  "田中太郎",
  new EmailAddress("tanaka@example.com"),
  new Age(30),
);
console.log(profile.getInfo()); // "田中太郎 (30歳) - tanaka@example.com"
```

### 3.3 Builder パターン

```java
// Java: テレスコーピングコンストラクタ問題 → Builder パターン
public class Pizza {
    private final int size;          // 必須
    private final boolean cheese;    // オプション
    private final boolean pepperoni; // オプション
    private final boolean mushroom;  // オプション

    // コンストラクタが爆発する
    // Pizza(int)
    // Pizza(int, boolean)
    // Pizza(int, boolean, boolean)
    // Pizza(int, boolean, boolean, boolean)

    // → Builder パターンで解決
    private Pizza(Builder builder) {
        this.size = builder.size;
        this.cheese = builder.cheese;
        this.pepperoni = builder.pepperoni;
        this.mushroom = builder.mushroom;
    }

    public static class Builder {
        private final int size;
        private boolean cheese = false;
        private boolean pepperoni = false;
        private boolean mushroom = false;

        public Builder(int size) { this.size = size; }
        public Builder cheese() { this.cheese = true; return this; }
        public Builder pepperoni() { this.pepperoni = true; return this; }
        public Builder mushroom() { this.mushroom = true; return this; }
        public Pizza build() { return new Pizza(this); }
    }
}

// 可読性が高い
Pizza pizza = new Pizza.Builder(12)
    .cheese()
    .pepperoni()
    .build();
```

### 3.4 複雑なオブジェクトの構築パターン

```typescript
// TypeScript: 段階的な Builder パターン（型安全版）

// 型レベルで構築段階を制御する
interface NeedsHost {
  host(host: string): NeedsPort;
}

interface NeedsPort {
  port(port: number): OptionalConfig;
}

interface OptionalConfig {
  ssl(enabled: boolean): OptionalConfig;
  timeout(ms: number): OptionalConfig;
  maxRetries(count: number): OptionalConfig;
  build(): DatabaseConfig;
}

class DatabaseConfig {
  private constructor(
    public readonly host: string,
    public readonly port: number,
    public readonly ssl: boolean,
    public readonly timeout: number,
    public readonly maxRetries: number,
  ) {}

  static builder(): NeedsHost {
    return new DatabaseConfigBuilder();
  }

  getConnectionString(): string {
    const protocol = this.ssl ? "ssl" : "tcp";
    return `${protocol}://${this.host}:${this.port}?timeout=${this.timeout}&retries=${this.maxRetries}`;
  }
}

class DatabaseConfigBuilder implements NeedsHost, NeedsPort, OptionalConfig {
  private _host = "";
  private _port = 0;
  private _ssl = false;
  private _timeout = 5000;
  private _maxRetries = 3;

  host(host: string): NeedsPort {
    this._host = host;
    return this;
  }

  port(port: number): OptionalConfig {
    this._port = port;
    return this;
  }

  ssl(enabled: boolean): OptionalConfig {
    this._ssl = enabled;
    return this;
  }

  timeout(ms: number): OptionalConfig {
    this._timeout = ms;
    return this;
  }

  maxRetries(count: number): OptionalConfig {
    this._maxRetries = count;
    return this;
  }

  build(): DatabaseConfig {
    return new (DatabaseConfig as any)(
      this._host,
      this._port,
      this._ssl,
      this._timeout,
      this._maxRetries,
    );
  }
}

// 使用例: 必須パラメータを順番に指定しないとコンパイルエラー
const config = DatabaseConfig.builder()
  .host("db.example.com")   // NeedsHost → NeedsPort
  .port(5432)               // NeedsPort → OptionalConfig
  .ssl(true)                // OptionalConfig → OptionalConfig
  .timeout(10000)           // OptionalConfig → OptionalConfig
  .build();                 // OptionalConfig → DatabaseConfig

console.log(config.getConnectionString());
// "ssl://db.example.com:5432?timeout=10000&retries=3"

// コンパイルエラー: host() の前に port() は呼べない
// DatabaseConfig.builder().port(5432);  // エラー
```

```python
# Python: dataclass + ファクトリメソッドによる構築パターン

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass(frozen=True)  # frozen=True で不変オブジェクト
class Task:
    title: str
    description: str
    priority: Priority
    assignee: str
    due_date: datetime
    tags: tuple[str, ...] = ()  # tupleで不変性を保証
    created_at: datetime = field(default_factory=datetime.now)
    task_id: str = field(default_factory=lambda: f"TASK-{id(object()):08x}")

    def __post_init__(self):
        """不変条件の検証"""
        if not self.title.strip():
            raise ValueError("タスクタイトルは空にできません")
        if self.due_date < self.created_at:
            raise ValueError("期限は作成日以降である必要があります")

    # ファクトリメソッド
    @classmethod
    def create_bug(cls, title: str, assignee: str,
                   severity: str = "medium") -> "Task":
        """バグ報告タスクを作成"""
        priority = {
            "low": Priority.LOW,
            "medium": Priority.MEDIUM,
            "high": Priority.HIGH,
            "critical": Priority.CRITICAL,
        }.get(severity, Priority.MEDIUM)

        return cls(
            title=f"[Bug] {title}",
            description=f"バグ報告: {title}",
            priority=priority,
            assignee=assignee,
            due_date=datetime.now() + timedelta(days=7),
            tags=("bug", severity),
        )

    @classmethod
    def create_feature(cls, title: str, assignee: str,
                       sprint_days: int = 14) -> "Task":
        """機能開発タスクを作成"""
        return cls(
            title=f"[Feature] {title}",
            description=f"機能開発: {title}",
            priority=Priority.MEDIUM,
            assignee=assignee,
            due_date=datetime.now() + timedelta(days=sprint_days),
            tags=("feature", "development"),
        )

    def is_overdue(self) -> bool:
        return datetime.now() > self.due_date

    def days_until_due(self) -> int:
        delta = self.due_date - datetime.now()
        return max(0, delta.days)


# 使用例
bug = Task.create_bug("ログイン画面がクラッシュする", "田中", "critical")
feature = Task.create_feature("ダッシュボード機能", "山田")

print(f"{bug.title} - 期限まで{bug.days_until_due()}日")
print(f"{feature.title} - 優先度: {feature.priority.value}")
```

### 3.5 コピーコンストラクタとclone

```java
// Java: コピーコンストラクタ vs clone

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Playlist {
    private final String name;
    private final String owner;
    private final List<String> songs;

    public Playlist(String name, String owner) {
        this.name = name;
        this.owner = owner;
        this.songs = new ArrayList<>();
    }

    // コピーコンストラクタ（clone より推奨）
    public Playlist(Playlist other) {
        this.name = other.name + " (コピー)";
        this.owner = other.owner;
        this.songs = new ArrayList<>(other.songs); // 防衛的コピー
    }

    // 特定の属性を変更したコピー
    public Playlist withName(String newName) {
        Playlist copy = new Playlist(this);
        // 直接フィールドにアクセス（同一クラス内）
        // nameがfinalの場合はリフレクションか、別のコンストラクタが必要
        return copy;
    }

    public void addSong(String song) {
        songs.add(song);
    }

    public List<String> getSongs() {
        return Collections.unmodifiableList(songs); // 防衛的コピー
    }

    public String getName() { return name; }
    public String getOwner() { return owner; }

    @Override
    public String toString() {
        return String.format("Playlist('%s' by %s, %d曲)", name, owner, songs.size());
    }
}

// 使用例
Playlist original = new Playlist("お気に入り", "田中");
original.addSong("Song A");
original.addSong("Song B");

Playlist copy = new Playlist(original);
copy.addSong("Song C");

System.out.println(original); // Playlist('お気に入り' by 田中, 2曲)
System.out.println(copy);     // Playlist('お気に入り (コピー)' by 田中, 3曲)
// → 独立したオブジェクト
```

---

## 4. 静的メンバ vs インスタンスメンバ

```
┌──────────────────┬────────────────────┬────────────────────┐
│                  │ 静的（static）      │ インスタンス        │
├──────────────────┼────────────────────┼────────────────────┤
│ 所属             │ クラスに属する      │ オブジェクトに属する│
├──────────────────┼────────────────────┼────────────────────┤
│ メモリ           │ クラス領域に1つ     │ 各オブジェクトに1つ│
├──────────────────┼────────────────────┼────────────────────┤
│ アクセス         │ クラス名.メンバ     │ インスタンス.メンバ │
├──────────────────┼────────────────────┼────────────────────┤
│ this参照         │ なし               │ あり               │
├──────────────────┼────────────────────┼────────────────────┤
│ 用途             │ ユーティリティ      │ オブジェクト固有    │
│                  │ ファクトリメソッド  │ の状態と振る舞い    │
│                  │ 定数               │                    │
└──────────────────┴────────────────────┴────────────────────┘
```

### 4.1 Python のクラスメソッド・静的メソッド・インスタンスメソッド

```python
# Python: クラスメソッド vs インスタンスメソッド
class Counter:
    _instance_count = 0  # クラス変数（全インスタンスで共有）

    def __init__(self, name: str):
        self.name = name         # インスタンス変数
        self.count = 0           # インスタンス変数
        Counter._instance_count += 1

    def increment(self):          # インスタンスメソッド
        self.count += 1

    @classmethod
    def get_instance_count(cls):  # クラスメソッド
        return cls._instance_count

    @staticmethod
    def is_valid_name(name: str) -> bool:  # 静的メソッド
        return len(name) > 0 and name.isalpha()

c1 = Counter("alpha")
c2 = Counter("beta")
print(Counter.get_instance_count())  # 2
print(Counter.is_valid_name("test")) # True
```

### 4.2 静的メンバの適切な使い方

```typescript
// TypeScript: 静的メンバの活用パターン

// パターン1: 定数の定義
class HttpStatus {
  static readonly OK = 200;
  static readonly NOT_FOUND = 404;
  static readonly INTERNAL_SERVER_ERROR = 500;

  static isSuccess(code: number): boolean {
    return code >= 200 && code < 300;
  }

  static isClientError(code: number): boolean {
    return code >= 400 && code < 500;
  }

  static isServerError(code: number): boolean {
    return code >= 500 && code < 600;
  }
}

// パターン2: ファクトリメソッド
class Color {
  private constructor(
    public readonly r: number,
    public readonly g: number,
    public readonly b: number,
    public readonly a: number = 1.0,
  ) {}

  // 名前付きファクトリメソッド
  static fromRGB(r: number, g: number, b: number): Color {
    return new Color(
      Math.max(0, Math.min(255, r)),
      Math.max(0, Math.min(255, g)),
      Math.max(0, Math.min(255, b)),
    );
  }

  static fromHex(hex: string): Color {
    const match = hex.replace("#", "").match(/.{2}/g);
    if (!match || match.length < 3) {
      throw new Error(`無効なHexカラー: ${hex}`);
    }
    return new Color(
      parseInt(match[0], 16),
      parseInt(match[1], 16),
      parseInt(match[2], 16),
    );
  }

  static fromHSL(h: number, s: number, l: number): Color {
    // HSL → RGB 変換ロジック
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;

    let r = 0, g = 0, b = 0;
    if (h < 60)       { r = c; g = x; }
    else if (h < 120) { r = x; g = c; }
    else if (h < 180) { g = c; b = x; }
    else if (h < 240) { g = x; b = c; }
    else if (h < 300) { r = x; b = c; }
    else              { r = c; b = x; }

    return new Color(
      Math.round((r + m) * 255),
      Math.round((g + m) * 255),
      Math.round((b + m) * 255),
    );
  }

  // よく使う色の定数
  static readonly RED = Color.fromRGB(255, 0, 0);
  static readonly GREEN = Color.fromRGB(0, 255, 0);
  static readonly BLUE = Color.fromRGB(0, 0, 255);
  static readonly WHITE = Color.fromRGB(255, 255, 255);
  static readonly BLACK = Color.fromRGB(0, 0, 0);

  toHex(): string {
    const hex = (n: number) => n.toString(16).padStart(2, "0");
    return `#${hex(this.r)}${hex(this.g)}${hex(this.b)}`;
  }

  mix(other: Color, ratio: number = 0.5): Color {
    return Color.fromRGB(
      Math.round(this.r * (1 - ratio) + other.r * ratio),
      Math.round(this.g * (1 - ratio) + other.g * ratio),
      Math.round(this.b * (1 - ratio) + other.b * ratio),
    );
  }
}

// パターン3: シングルトン（唯一のインスタンス）
class AppConfig {
  private static instance: AppConfig | null = null;

  private constructor(
    public readonly appName: string,
    public readonly version: string,
    public readonly debug: boolean,
  ) {}

  static getInstance(): AppConfig {
    if (AppConfig.instance === null) {
      AppConfig.instance = new AppConfig(
        "MyApp",
        "1.0.0",
        process.env.NODE_ENV !== "production",
      );
    }
    return AppConfig.instance;
  }

  // テスト用: インスタンスをリセット
  static resetForTesting(): void {
    AppConfig.instance = null;
  }
}

// パターン4: レジストリ（オブジェクトの管理）
class EventBus {
  private static handlers = new Map<string, Set<Function>>();

  static on(event: string, handler: Function): void {
    if (!EventBus.handlers.has(event)) {
      EventBus.handlers.set(event, new Set());
    }
    EventBus.handlers.get(event)!.add(handler);
  }

  static off(event: string, handler: Function): void {
    EventBus.handlers.get(event)?.delete(handler);
  }

  static emit(event: string, ...args: unknown[]): void {
    const handlers = EventBus.handlers.get(event);
    if (handlers) {
      for (const handler of handlers) {
        handler(...args);
      }
    }
  }

  static clear(): void {
    EventBus.handlers.clear();
  }
}
```

### 4.3 静的メンバの注意点とアンチパターン

```
静的メンバのリスク:

1. テストが困難になる:
   → 静的メソッドはモック化しにくい
   → 依存性注入ができない
   → テスト間で状態が共有される

2. 並行性の問題:
   → 静的フィールドは全スレッドで共有
   → 適切な同期が必要

3. 過度な使用:
   → 「全部staticでいいじゃん」→ 手続き型プログラミングに逆戻り
   → OOPの利点（ポリモーフィズム等）が失われる

判断基準:
  ✅ static が適切:
    → ファクトリメソッド
    → ユーティリティ関数（Math.max, Collections.sort）
    → 定数
    → オブジェクトの状態に依存しない純粋関数

  ❌ static が不適切:
    → ビジネスロジック（テスト性が下がる）
    → 状態を持つ処理（並行性の問題）
    → インターフェースの実装が必要な処理
```

```java
// Java: 静的メソッドの過剰使用を避ける

// ❌ 悪い例: 静的メソッドの過剰使用
public class UserService {
    public static User findById(long id) {
        // データベースに直接アクセス → テスト困難
        return Database.query("SELECT * FROM users WHERE id = ?", id);
    }

    public static void updateEmail(long userId, String email) {
        // 静的メソッド同士の呼び出し → モック不可
        User user = findById(userId);
        Database.execute("UPDATE users SET email = ? WHERE id = ?", email, userId);
    }
}

// ✅ 良い例: インスタンスメソッド + 依存性注入
public class UserService {
    private final UserRepository repository;
    private final EmailValidator validator;

    public UserService(UserRepository repository, EmailValidator validator) {
        this.repository = repository;
        this.validator = validator;
    }

    public User findById(long id) {
        return repository.findById(id)
            .orElseThrow(() -> new UserNotFoundException(id));
    }

    public void updateEmail(long userId, String email) {
        if (!validator.isValid(email)) {
            throw new InvalidEmailException(email);
        }
        User user = findById(userId);
        user.setEmail(email);
        repository.save(user);
    }
}

// テスト時: モックを注入できる
// UserService service = new UserService(mockRepo, mockValidator);
```

---

## 5. 値型 vs 参照型

```
参照型（Reference Type）:
  → 変数はオブジェクトへの参照（ポインタ）を保持
  → 代入するとポインタがコピーされる（浅いコピー）
  → Java のクラス、Python の全オブジェクト

  user1 = User("田中")
  user2 = user1        ← user1 と user2 は同じオブジェクトを参照
  user2.name = "山田"  ← user1.name も "山田" に変わる！

値型（Value Type）:
  → 変数が値そのものを保持
  → 代入すると値がコピーされる（深いコピー）
  → C# の struct、Swift の struct、Rust の非参照型

  var point1 = Point(x: 1, y: 2)
  var point2 = point1  ← point2 は point1 のコピー
  point2.x = 10        ← point1.x は 1 のまま
```

### 5.1 Swift: 値型と参照型の明確な区別

```swift
// Swift: 値型 vs 参照型の明確な区別
struct Point {          // 値型
    var x: Double
    var y: Double
}

class Circle {          // 参照型
    var center: Point
    var radius: Double

    init(center: Point, radius: Double) {
        self.center = center
        self.radius = radius
    }
}

var p1 = Point(x: 0, y: 0)
var p2 = p1             // コピー
p2.x = 10
print(p1.x)            // 0（変わらない）

let c1 = Circle(center: Point(x: 0, y: 0), radius: 5)
let c2 = c1             // 参照共有
c2.radius = 10
print(c1.radius)        // 10（変わる！）
```

### 5.2 各言語の値型・参照型の扱い

```
┌──────────┬──────────────────────┬──────────────────────┐
│ 言語     │ 値型                 │ 参照型               │
├──────────┼──────────────────────┼──────────────────────┤
│ Java     │ プリミティブ型       │ クラス（全て）        │
│          │ (int, double, etc.)  │                      │
├──────────┼──────────────────────┼──────────────────────┤
│ C#       │ struct, enum,        │ class, interface,    │
│          │ プリミティブ型       │ delegate, array      │
├──────────┼──────────────────────┼──────────────────────┤
│ Swift    │ struct, enum, tuple  │ class, closure       │
├──────────┼──────────────────────┼──────────────────────┤
│ Kotlin   │ inline class         │ class（全て）         │
│          │ (JVM上は最適化)      │                      │
├──────────┼──────────────────────┼──────────────────────┤
│ Rust     │ デフォルトで全て      │ Box, Rc, Arc,       │
│          │ (Copy trait実装時)   │ 参照(&T)             │
├──────────┼──────────────────────┼──────────────────────┤
│ Python   │ なし（全て参照型）    │ 全オブジェクト       │
│          │ ※ int, str は不変    │                      │
├──────────┼──────────────────────┼──────────────────────┤
│ Go       │ struct, 基本型       │ ポインタ, slice,     │
│          │                      │ map, channel         │
└──────────┴──────────────────────┴──────────────────────┘
```

```csharp
// C#: struct（値型）と class（参照型）の使い分け

// 値型: 小さくて不変なデータに適する
public readonly struct Vector2D
{
    public double X { get; }
    public double Y { get; }

    public Vector2D(double x, double y)
    {
        X = x;
        Y = y;
    }

    public double Magnitude => Math.Sqrt(X * X + Y * Y);

    public Vector2D Normalize()
    {
        var mag = Magnitude;
        return mag > 0 ? new Vector2D(X / mag, Y / mag) : this;
    }

    public static Vector2D operator +(Vector2D a, Vector2D b)
        => new Vector2D(a.X + b.X, a.Y + b.Y);

    public static Vector2D operator -(Vector2D a, Vector2D b)
        => new Vector2D(a.X - b.X, a.Y - b.Y);

    public static Vector2D operator *(Vector2D v, double scalar)
        => new Vector2D(v.X * scalar, v.Y * scalar);

    public static double Dot(Vector2D a, Vector2D b)
        => a.X * b.X + a.Y * b.Y;

    public override string ToString() => $"({X:F2}, {Y:F2})";
}

// 参照型: 状態を持つ複雑なオブジェクトに適する
public class Particle
{
    public Vector2D Position { get; private set; }
    public Vector2D Velocity { get; private set; }
    public double Mass { get; }
    public bool IsActive { get; private set; }

    public Particle(Vector2D position, Vector2D velocity, double mass)
    {
        Position = position;
        Velocity = velocity;
        Mass = mass;
        IsActive = true;
    }

    public void Update(double deltaTime)
    {
        if (!IsActive) return;
        Position = Position + Velocity * deltaTime;
    }

    public void ApplyForce(Vector2D force)
    {
        if (!IsActive) return;
        // F = ma → a = F/m → v += a * dt
        Velocity = Velocity + force * (1.0 / Mass);
    }

    public void Deactivate() => IsActive = false;
}

// 使用例
var v1 = new Vector2D(3, 4);
var v2 = v1; // コピー（値型）
// v2 を変更しても v1 に影響なし

var p1 = new Particle(new Vector2D(0, 0), new Vector2D(1, 0), 1.0);
var p2 = p1; // 参照のコピー（参照型）
p2.Update(1.0); // p1 にも影響する（同一オブジェクト）
```

### 5.3 Copy-on-Write（COW）最適化

```
Copy-on-Write:
  → コピー時は参照を共有し、変更時に初めて実際のコピーを作成
  → メモリ効率と値型のセマンティクスを両立

  Swift の Array, String, Dictionary は内部的にCOWを使用:

  var array1 = [1, 2, 3]
  var array2 = array1   ← この時点ではメモリを共有
  array2.append(4)      ← この時点で初めてコピーが発生

  ┌──────┐
  │array1├─→ [1, 2, 3]     (共有状態)
  │array2├─┘
  └──────┘

  array2.append(4) の後:

  ┌──────┐
  │array1├─→ [1, 2, 3]     (元のデータ)
  │array2├─→ [1, 2, 3, 4]  (コピーされた新しいデータ)
  └──────┘
```

```swift
// Swift: Copy-on-Write の実装例

final class Storage<T> {
    var value: T

    init(_ value: T) {
        self.value = value
    }

    func copy() -> Storage<T> {
        return Storage(value)
    }
}

struct COWWrapper<T> {
    private var storage: Storage<T>

    init(_ value: T) {
        self.storage = Storage(value)
    }

    var value: T {
        get { storage.value }
        set {
            // 他に参照者がいる場合のみコピー
            if !isKnownUniquelyReferenced(&storage) {
                storage = storage.copy()
            }
            storage.value = newValue
        }
    }
}

// 使用例
var a = COWWrapper([1, 2, 3])
var b = a  // 参照を共有（コピーは発生しない）

// b を変更した時点で初めてコピーが発生
b.value = [1, 2, 3, 4]
print(a.value)  // [1, 2, 3]（影響なし）
print(b.value)  // [1, 2, 3, 4]
```

---

## 6. 等価性と同一性

```
同一性（Identity）:
  → 2つの変数が「同じオブジェクト」を指しているか
  → メモリアドレスの比較
  → Java: ==, Python: is, JavaScript: ===（オブジェクト同士）

等価性（Equality）:
  → 2つのオブジェクトが「同じ値」を持っているか
  → 論理的な等しさ
  → Java: .equals(), Python: ==, JavaScript: カスタム実装

  ┌──────┐      ┌──────────┐
  │ a ───┼─────→│ "Hello"  │  a と b は同一（同じオブジェクト）
  │ b ───┼─────→│          │  a と b は等価（同じ値）
  └──────┘      └──────────┘

  ┌──────┐      ┌──────────┐
  │ a ───┼─────→│ "Hello"  │  a と b は同一でない（別オブジェクト）
  │ b ───┼──┐   └──────────┘  a と b は等価（同じ値）
  └──────┘  │   ┌──────────┐
            └──→│ "Hello"  │
                └──────────┘
```

```java
// Java: equals と hashCode の正しい実装

import java.util.Objects;

public class Money implements Comparable<Money> {
    private final long amount;  // 最小単位（銭）で保持
    private final String currency;

    public Money(long amount, String currency) {
        this.amount = amount;
        this.currency = Objects.requireNonNull(currency);
    }

    // 等価性の定義: 金額と通貨が同じなら等価
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;  // 同一性チェック（最適化）
        if (!(obj instanceof Money other)) return false;
        return amount == other.amount
            && currency.equals(other.currency);
    }

    // equals をオーバーライドしたら hashCode も必ずオーバーライド
    // 等価なオブジェクトは同じ hashCode を返す必要がある
    @Override
    public int hashCode() {
        return Objects.hash(amount, currency);
    }

    @Override
    public int compareTo(Money other) {
        if (!currency.equals(other.currency)) {
            throw new IllegalArgumentException("通貨が異なります");
        }
        return Long.compare(amount, other.amount);
    }

    @Override
    public String toString() {
        return String.format("%d.%02d %s", amount / 100, amount % 100, currency);
    }
}

// 使用例
Money m1 = new Money(10000, "JPY");
Money m2 = new Money(10000, "JPY");
Money m3 = m1;

System.out.println(m1 == m2);      // false（別オブジェクト）
System.out.println(m1 == m3);      // true（同一オブジェクト）
System.out.println(m1.equals(m2)); // true（等価）
System.out.println(m1.equals(m3)); // true（等価）

// HashSet / HashMap で正しく動作する
Set<Money> set = new HashSet<>();
set.add(m1);
set.add(m2);
System.out.println(set.size()); // 1（等価なので重複排除）
```

```python
# Python: __eq__ と __hash__ の実装

from dataclasses import dataclass

@dataclass(frozen=True)  # frozen=True で __eq__ と __hash__ が自動生成
class Coordinate:
    latitude: float
    longitude: float

    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"緯度は-90〜90の範囲: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"経度は-180〜180の範囲: {self.longitude}")

    def distance_to(self, other: "Coordinate") -> float:
        """2点間の距離をHaversine公式で計算（km）"""
        import math
        R = 6371  # 地球の半径（km）
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

# 等価性の確認
tokyo = Coordinate(35.6762, 139.6503)
tokyo2 = Coordinate(35.6762, 139.6503)
osaka = Coordinate(34.6937, 135.5023)

print(tokyo == tokyo2)          # True（等価）
print(tokyo is tokyo2)          # False（同一ではない）
print(tokyo == osaka)           # False（等価でない）

# 辞書のキーとして使用可能
cities = {
    tokyo: "東京",
    osaka: "大阪",
}
print(cities[tokyo2])  # "東京"（等価なキーでアクセス）

# 距離計算
print(f"東京-大阪: {tokyo.distance_to(osaka):.1f} km")
```

---

## 7. オブジェクトの合成（Composition）

```
合成（Composition）:
  → オブジェクトが他のオブジェクトを「持っている（has-a）」関係
  → 継承（is-a）よりも柔軟で推奨される
  → 実行時に構成を変更可能

  ┌────────────┐     ┌────────────┐
  │   Car      │     │   Engine   │
  │            │────→│            │  Carは Engineを「持っている」
  │ engine     │     │ start()    │
  │ start()    │     │ stop()     │
  └────────────┘     └────────────┘
        │
        ├────→┌────────────┐
        │     │ Wheels[]   │
        │     │ rotate()   │
        │     └────────────┘
        │
        └────→┌────────────┐
              │ GPS        │
              │ navigate() │
              └────────────┘
```

```typescript
// TypeScript: 合成による柔軟な設計

// 個別の責務を持つクラス
class Logger {
  constructor(private readonly prefix: string) {}

  info(message: string): void {
    console.log(`[INFO][${this.prefix}] ${message}`);
  }

  error(message: string): void {
    console.error(`[ERROR][${this.prefix}] ${message}`);
  }

  warn(message: string): void {
    console.warn(`[WARN][${this.prefix}] ${message}`);
  }
}

class Validator {
  private rules = new Map<string, (value: unknown) => string | null>();

  addRule(field: string, validate: (value: unknown) => string | null): this {
    this.rules.set(field, validate);
    return this;
  }

  validate(data: Record<string, unknown>): string[] {
    const errors: string[] = [];
    for (const [field, rule] of this.rules) {
      const error = rule(data[field]);
      if (error) {
        errors.push(`${field}: ${error}`);
      }
    }
    return errors;
  }
}

class EventEmitter<T extends Record<string, unknown[]>> {
  private handlers = new Map<string, Set<Function>>();

  on<K extends keyof T>(event: K, handler: (...args: T[K]) => void): void {
    if (!this.handlers.has(event as string)) {
      this.handlers.set(event as string, new Set());
    }
    this.handlers.get(event as string)!.add(handler);
  }

  emit<K extends keyof T>(event: K, ...args: T[K]): void {
    const handlers = this.handlers.get(event as string);
    if (handlers) {
      for (const handler of handlers) {
        handler(...args);
      }
    }
  }
}

// 合成: 個別のクラスを組み合わせて複雑な機能を構築
type UserEvents = {
  "user:created": [user: { name: string; email: string }];
  "user:updated": [userId: string, changes: Record<string, unknown>];
  "user:deleted": [userId: string];
};

class UserService {
  private readonly logger: Logger;
  private readonly validator: Validator;
  private readonly events: EventEmitter<UserEvents>;
  private readonly users = new Map<string, { name: string; email: string }>();

  constructor() {
    // 合成: 各コンポーネントを内部に保持
    this.logger = new Logger("UserService");

    this.validator = new Validator()
      .addRule("name", (v) =>
        typeof v === "string" && v.length > 0 ? null : "名前は必須です"
      )
      .addRule("email", (v) =>
        typeof v === "string" && v.includes("@") ? null : "有効なメールアドレスが必要です"
      );

    this.events = new EventEmitter<UserEvents>();
  }

  onUserCreated(handler: (user: { name: string; email: string }) => void): void {
    this.events.on("user:created", handler);
  }

  createUser(name: string, email: string): string {
    // バリデーション（Validatorに委譲）
    const errors = this.validator.validate({ name, email });
    if (errors.length > 0) {
      this.logger.error(`バリデーションエラー: ${errors.join(", ")}`);
      throw new Error(`バリデーションエラー: ${errors.join(", ")}`);
    }

    // ユーザー作成
    const id = crypto.randomUUID();
    const user = { name, email };
    this.users.set(id, user);

    // ログ出力（Loggerに委譲）
    this.logger.info(`ユーザー作成: ${name} (${email})`);

    // イベント発行（EventEmitterに委譲）
    this.events.emit("user:created", user);

    return id;
  }

  getUser(id: string): { name: string; email: string } | undefined {
    return this.users.get(id);
  }
}

// 使用例
const service = new UserService();
service.onUserCreated((user) => {
  console.log(`新規ユーザー通知: ${user.name}`);
});

const userId = service.createUser("田中太郎", "tanaka@example.com");
```

```python
# Python: 合成パターンの実践例（通知システム）

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

# 通知の送信先（Strategyパターン）
class NotificationSender(Protocol):
    def send(self, recipient: str, subject: str, body: str) -> bool: ...

class EmailSender:
    def __init__(self, smtp_host: str, smtp_port: int):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def send(self, recipient: str, subject: str, body: str) -> bool:
        print(f"[Email] To: {recipient}, Subject: {subject}")
        return True

class SlackSender:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, recipient: str, subject: str, body: str) -> bool:
        print(f"[Slack] Channel: {recipient}, Message: {subject}")
        return True

class SmsSender:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def send(self, recipient: str, subject: str, body: str) -> bool:
        print(f"[SMS] To: {recipient}, Message: {body[:100]}")
        return True

# テンプレートエンジン
@dataclass
class MessageTemplate:
    subject_template: str
    body_template: str

    def render(self, **kwargs: str) -> tuple[str, str]:
        subject = self.subject_template.format(**kwargs)
        body = self.body_template.format(**kwargs)
        return subject, body

# 通知ログ
@dataclass
class NotificationLog:
    entries: list[dict] = field(default_factory=list)

    def record(self, channel: str, recipient: str,
               subject: str, success: bool) -> None:
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "channel": channel,
            "recipient": recipient,
            "subject": subject,
            "success": success,
        })

    def get_failures(self) -> list[dict]:
        return [e for e in self.entries if not e["success"]]

# 合成: 通知サービス
class NotificationService:
    """合成パターン: 各コンポーネントを組み合わせて通知機能を提供"""

    def __init__(self):
        self._senders: dict[str, NotificationSender] = {}
        self._templates: dict[str, MessageTemplate] = {}
        self._log = NotificationLog()

    def register_sender(self, name: str, sender: NotificationSender) -> None:
        self._senders[name] = sender

    def register_template(self, name: str, template: MessageTemplate) -> None:
        self._templates[name] = template

    def send(self, channel: str, recipient: str,
             template_name: str, **kwargs: str) -> bool:
        sender = self._senders.get(channel)
        if sender is None:
            raise ValueError(f"未登録のチャンネル: {channel}")

        template = self._templates.get(template_name)
        if template is None:
            raise ValueError(f"未登録のテンプレート: {template_name}")

        subject, body = template.render(**kwargs)
        success = sender.send(recipient, subject, body)
        self._log.record(channel, recipient, subject, success)
        return success

    def broadcast(self, template_name: str,
                  recipients: dict[str, str], **kwargs: str) -> dict[str, bool]:
        results = {}
        for channel, recipient in recipients.items():
            results[channel] = self.send(channel, recipient, template_name, **kwargs)
        return results

    @property
    def log(self) -> NotificationLog:
        return self._log


# 使用例
service = NotificationService()

# コンポーネントの登録
service.register_sender("email", EmailSender("smtp.example.com", 587))
service.register_sender("slack", SlackSender("https://hooks.slack.com/xxx"))
service.register_sender("sms", SmsSender("api-key-123"))

service.register_template("welcome", MessageTemplate(
    subject_template="ようこそ、{name}さん！",
    body_template="{name}さん、アカウントの作成が完了しました。",
))

# 複数チャンネルへの一斉送信
results = service.broadcast(
    "welcome",
    recipients={
        "email": "tanaka@example.com",
        "slack": "#new-users",
        "sms": "090-1234-5678",
    },
    name="田中太郎",
)

print(f"送信結果: {results}")
print(f"失敗数: {len(service.log.get_failures())}")
```

---

## 8. クラス設計のベストプラクティス

### 8.1 単一責任の原則（SRP）

```
1つのクラスは「1つの理由」でのみ変更されるべき:

  ❌ 悪い例: UserManager が全責任を持つ
    class UserManager {
      createUser()
      deleteUser()
      sendEmail()        ← メール送信は別の責任
      generateReport()   ← レポート生成は別の責任
      validateInput()    ← バリデーションは別の責任
    }

  ✅ 良い例: 責任を分離
    class UserRepository { create() / delete() / find() }
    class EmailService { send() }
    class ReportGenerator { generate() }
    class UserValidator { validate() }
    class UserService { // これらを合成して使う }
```

### 8.2 凝集度を高める

```typescript
// TypeScript: 凝集度の高いクラス設計

// ❌ 低凝集: 関連性の薄いメソッドが混在
class Utils {
  static formatDate(date: Date): string { /* ... */ return ""; }
  static parseJSON(json: string): unknown { /* ... */ return {}; }
  static calculateTax(amount: number): number { /* ... */ return 0; }
  static sendEmail(to: string, body: string): void { /* ... */ }
  static resizeImage(path: string, width: number): void { /* ... */ }
}

// ✅ 高凝集: 関連するデータとメソッドが1つのクラスに
class DateRange {
  constructor(
    public readonly start: Date,
    public readonly end: Date,
  ) {
    if (start > end) {
      throw new Error("開始日は終了日より前である必要があります");
    }
  }

  contains(date: Date): boolean {
    return date >= this.start && date <= this.end;
  }

  overlaps(other: DateRange): boolean {
    return this.start <= other.end && other.start <= this.end;
  }

  getDurationDays(): number {
    const diff = this.end.getTime() - this.start.getTime();
    return Math.ceil(diff / (1000 * 60 * 60 * 24));
  }

  intersection(other: DateRange): DateRange | null {
    const start = new Date(Math.max(this.start.getTime(), other.start.getTime()));
    const end = new Date(Math.min(this.end.getTime(), other.end.getTime()));
    if (start > end) return null;
    return new DateRange(start, end);
  }

  extend(days: number): DateRange {
    const newEnd = new Date(this.end);
    newEnd.setDate(newEnd.getDate() + days);
    return new DateRange(this.start, newEnd);
  }

  format(locale: string = "ja-JP"): string {
    const fmt = (d: Date) => d.toLocaleDateString(locale);
    return `${fmt(this.start)} 〜 ${fmt(this.end)}（${this.getDurationDays()}日間）`;
  }
}

// 使用例
const vacation = new DateRange(
  new Date("2025-08-10"),
  new Date("2025-08-20"),
);
const holiday = new DateRange(
  new Date("2025-08-15"),
  new Date("2025-08-16"),
);

console.log(vacation.format());           // "2025/8/10 〜 2025/8/20（10日間）"
console.log(vacation.overlaps(holiday));   // true
console.log(vacation.contains(new Date("2025-08-12"))); // true

const overlap = vacation.intersection(holiday);
console.log(overlap?.format());           // "2025/8/15 〜 2025/8/16（1日間）"
```

### 8.3 実践例: ECサイトの商品管理

```typescript
// TypeScript: 実践的なクラス設計（ECサイト）

// 値オブジェクト
class Money {
  constructor(
    public readonly amount: number,
    public readonly currency: string,
  ) {
    if (amount < 0) throw new Error("金額は0以上である必要があります");
  }

  add(other: Money): Money {
    this.assertSameCurrency(other);
    return new Money(this.amount + other.amount, this.currency);
  }

  subtract(other: Money): Money {
    this.assertSameCurrency(other);
    if (this.amount < other.amount) {
      throw new Error("結果が負になります");
    }
    return new Money(this.amount - other.amount, this.currency);
  }

  multiply(factor: number): Money {
    return new Money(Math.floor(this.amount * factor), this.currency);
  }

  private assertSameCurrency(other: Money): void {
    if (this.currency !== other.currency) {
      throw new Error(`通貨が異なります: ${this.currency} vs ${other.currency}`);
    }
  }

  format(): string {
    return `${this.amount.toLocaleString()} ${this.currency}`;
  }

  equals(other: Money): boolean {
    return this.amount === other.amount && this.currency === other.currency;
  }
}

class Quantity {
  constructor(public readonly value: number) {
    if (!Number.isInteger(value) || value < 0) {
      throw new Error("数量は0以上の整数である必要があります");
    }
  }

  add(n: number): Quantity {
    return new Quantity(this.value + n);
  }

  subtract(n: number): Quantity {
    return new Quantity(this.value - n);
  }

  isZero(): boolean {
    return this.value === 0;
  }
}

// エンティティ
class Product {
  private _stock: Quantity;

  constructor(
    public readonly id: string,
    public readonly name: string,
    public readonly price: Money,
    stock: number,
    public readonly category: string,
  ) {
    this._stock = new Quantity(stock);
  }

  get stock(): Quantity {
    return this._stock;
  }

  isAvailable(): boolean {
    return !this._stock.isZero();
  }

  reserve(quantity: number): void {
    if (this._stock.value < quantity) {
      throw new Error(`在庫不足: ${this.name}（在庫: ${this._stock.value}, 要求: ${quantity}）`);
    }
    this._stock = this._stock.subtract(quantity);
  }

  restock(quantity: number): void {
    this._stock = this._stock.add(quantity);
  }
}

// カートアイテム
class CartItem {
  constructor(
    public readonly product: Product,
    private _quantity: Quantity,
  ) {}

  get quantity(): number {
    return this._quantity.value;
  }

  getSubtotal(): Money {
    return this.product.price.multiply(this._quantity.value);
  }

  updateQuantity(newQuantity: number): CartItem {
    return new CartItem(this.product, new Quantity(newQuantity));
  }
}

// ショッピングカート
class ShoppingCart {
  private items = new Map<string, CartItem>();

  addItem(product: Product, quantity: number = 1): void {
    if (!product.isAvailable()) {
      throw new Error(`${product.name}は在庫切れです`);
    }
    const existing = this.items.get(product.id);
    if (existing) {
      const newQty = existing.quantity + quantity;
      this.items.set(product.id, existing.updateQuantity(newQty));
    } else {
      this.items.set(product.id, new CartItem(product, new Quantity(quantity)));
    }
  }

  removeItem(productId: string): void {
    this.items.delete(productId);
  }

  getTotal(): Money {
    let total = new Money(0, "JPY");
    for (const item of this.items.values()) {
      total = total.add(item.getSubtotal());
    }
    return total;
  }

  getItemCount(): number {
    let count = 0;
    for (const item of this.items.values()) {
      count += item.quantity;
    }
    return count;
  }

  getSummary(): string {
    const lines: string[] = ["=== ショッピングカート ==="];
    for (const item of this.items.values()) {
      lines.push(
        `  ${item.product.name} x ${item.quantity} = ${item.getSubtotal().format()}`
      );
    }
    lines.push(`  合計: ${this.getTotal().format()} (${this.getItemCount()}点)`);
    return lines.join("\n");
  }

  isEmpty(): boolean {
    return this.items.size === 0;
  }

  clear(): void {
    this.items.clear();
  }
}

// 使用例
const laptop = new Product("P001", "MacBook Pro", new Money(298000, "JPY"), 5, "Electronics");
const mouse = new Product("P002", "Magic Mouse", new Money(13800, "JPY"), 20, "Accessories");
const keyboard = new Product("P003", "HHKB", new Money(35200, "JPY"), 3, "Accessories");

const cart = new ShoppingCart();
cart.addItem(laptop, 1);
cart.addItem(mouse, 2);
cart.addItem(keyboard, 1);

console.log(cart.getSummary());
// === ショッピングカート ===
//   MacBook Pro x 1 = 298,000 JPY
//   Magic Mouse x 2 = 27,600 JPY
//   HHKB x 1 = 35,200 JPY
//   合計: 360,800 JPY (4点)
```

---

## 9. メタクラスとリフレクション

```
メタクラス（Metaclass）:
  → 「クラスのクラス」
  → クラスそのものの振る舞いを定義
  → Python では type がデフォルトのメタクラス

リフレクション（Reflection）:
  → 実行時にクラスやオブジェクトの構造を検査・操作
  → メタプログラミングの基盤
```

```python
# Python: メタクラスによるクラスの振る舞い制御

class SingletonMeta(type):
    """シングルトンパターンをメタクラスで実装"""
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class AppConfig(metaclass=SingletonMeta):
    def __init__(self):
        self.settings: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self.settings[key] = value

    def get(self, key: str, default: str = "") -> str:
        return self.settings.get(key, default)

# 何度インスタンス化しても同一オブジェクト
config1 = AppConfig()
config2 = AppConfig()
config1.set("debug", "true")
print(config1 is config2)         # True
print(config2.get("debug"))       # "true"


# リフレクションの例
class ValidationMeta(type):
    """バリデーション付きフィールドを自動検出するメタクラス"""

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        # _validate_ プレフィックスのメソッドを自動収集
        validators = {}
        for attr_name, attr_value in namespace.items():
            if attr_name.startswith("_validate_") and callable(attr_value):
                field_name = attr_name[len("_validate_"):]
                validators[field_name] = attr_value

        cls._validators = validators
        return cls

class ValidatedModel(metaclass=ValidationMeta):
    def __setattr__(self, name: str, value):
        validator = self.__class__._validators.get(name)
        if validator:
            validator(self, value)
        super().__setattr__(name, value)

class User(ValidatedModel):
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def _validate_name(self, value: str):
        if not isinstance(value, str) or len(value) == 0:
            raise ValueError("名前は空文字列にできません")

    def _validate_age(self, value: int):
        if not isinstance(value, int) or value < 0 or value > 150:
            raise ValueError(f"年齢が範囲外です: {value}")

    def _validate_email(self, value: str):
        if not isinstance(value, str) or "@" not in value:
            raise ValueError(f"無効なメールアドレス: {value}")

# 使用例
user = User("田中", 30, "tanaka@example.com")
try:
    user.age = -5  # ValueError: 年齢が範囲外です: -5
except ValueError as e:
    print(e)
```

```java
// Java: リフレクションの活用例

import java.lang.reflect.*;
import java.lang.annotation.*;

// カスタムアノテーション
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
@interface JsonField {
    String name() default "";
}

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
@interface Required {}

public class SimpleJsonSerializer {

    public static String toJson(Object obj) throws IllegalAccessException {
        StringBuilder sb = new StringBuilder("{");
        Field[] fields = obj.getClass().getDeclaredFields();
        boolean first = true;

        for (Field field : fields) {
            field.setAccessible(true);

            JsonField jsonField = field.getAnnotation(JsonField.class);
            if (jsonField == null) continue;

            String name = jsonField.name().isEmpty()
                ? field.getName()
                : jsonField.name();

            Object value = field.get(obj);

            if (!first) sb.append(",");
            first = false;

            sb.append("\"").append(name).append("\":");
            if (value instanceof String) {
                sb.append("\"").append(value).append("\"");
            } else {
                sb.append(value);
            }
        }

        sb.append("}");
        return sb.toString();
    }

    public static void validate(Object obj) throws Exception {
        for (Field field : obj.getClass().getDeclaredFields()) {
            field.setAccessible(true);
            if (field.isAnnotationPresent(Required.class)) {
                Object value = field.get(obj);
                if (value == null || (value instanceof String && ((String) value).isEmpty())) {
                    throw new IllegalStateException(
                        field.getName() + " は必須フィールドです"
                    );
                }
            }
        }
    }
}

// 使用例
class UserDto {
    @JsonField(name = "user_name")
    @Required
    String name;

    @JsonField
    @Required
    String email;

    @JsonField
    int age;

    UserDto(String name, String email, int age) {
        this.name = name;
        this.email = email;
        this.age = age;
    }
}

UserDto user = new UserDto("田中", "tanaka@example.com", 30);
SimpleJsonSerializer.validate(user);  // OK
String json = SimpleJsonSerializer.toJson(user);
// {"user_name":"田中","email":"tanaka@example.com","age":30}
```

---

## 10. クラスの種類と特殊なクラス

```
┌──────────────────┬──────────────────────────────────────┐
│ 種類             │ 説明                                  │
├──────────────────┼──────────────────────────────────────┤
│ 具象クラス       │ 通常のクラス。インスタンス化可能       │
├──────────────────┼──────────────────────────────────────┤
│ 抽象クラス       │ 直接インスタンス化不可。                │
│                  │ サブクラスに共通機能を提供             │
├──────────────────┼──────────────────────────────────────┤
│ インターフェース │ メソッドのシグネチャのみ定義           │
│                  │ 実装は持たない（Java 8+ はdefault可） │
├──────────────────┼──────────────────────────────────────┤
│ sealed クラス    │ サブクラスを限定的に許可               │
│                  │ パターンマッチングと好相性              │
├──────────────────┼──────────────────────────────────────┤
│ data クラス      │ データ保持専用。equals/hashCode自動   │
│                  │ Kotlin data class, Java record       │
├──────────────────┼──────────────────────────────────────┤
│ enum クラス      │ 有限個の定数インスタンスを定義         │
├──────────────────┼──────────────────────────────────────┤
│ 内部クラス       │ 他のクラス内に定義されたクラス         │
├──────────────────┼──────────────────────────────────────┤
│ 無名クラス       │ 名前のないクラス。一度だけの利用       │
└──────────────────┴──────────────────────────────────────┘
```

```kotlin
// Kotlin: 各種クラスの活用

// sealed class: サブクラスが限定される
sealed class Shape {
    abstract fun area(): Double
    abstract fun perimeter(): Double

    data class Circle(val radius: Double) : Shape() {
        override fun area() = Math.PI * radius * radius
        override fun perimeter() = 2 * Math.PI * radius
    }

    data class Rectangle(val width: Double, val height: Double) : Shape() {
        override fun area() = width * height
        override fun perimeter() = 2 * (width + height)
    }

    data class Triangle(val a: Double, val b: Double, val c: Double) : Shape() {
        override fun area(): Double {
            val s = (a + b + c) / 2
            return Math.sqrt(s * (s - a) * (s - b) * (s - c))
        }
        override fun perimeter() = a + b + c
    }
}

// sealed class + when式 = 網羅的パターンマッチング
fun describeShape(shape: Shape): String = when (shape) {
    is Shape.Circle -> "半径${shape.radius}の円（面積: ${shape.area():.2f}）"
    is Shape.Rectangle -> "${shape.width}x${shape.height}の長方形"
    is Shape.Triangle -> "三角形（周囲長: ${shape.perimeter():.2f}）"
    // 全サブクラスを網羅しているので else 不要
}

// data class: 値オブジェクト
data class Address(
    val postalCode: String,
    val prefecture: String,
    val city: String,
    val street: String,
    val building: String? = null,
) {
    fun toSingleLine(): String {
        val parts = listOfNotNull(
            "〒$postalCode",
            prefecture,
            city,
            street,
            building,
        )
        return parts.joinToString(" ")
    }
}

// enum class: 列挙型
enum class OrderStatus(val label: String, val isFinal: Boolean) {
    PENDING("注文受付", false),
    CONFIRMED("確認済み", false),
    SHIPPING("配送中", false),
    DELIVERED("配達済み", true),
    CANCELLED("キャンセル", true);

    fun canTransitionTo(next: OrderStatus): Boolean = when (this) {
        PENDING -> next == CONFIRMED || next == CANCELLED
        CONFIRMED -> next == SHIPPING || next == CANCELLED
        SHIPPING -> next == DELIVERED
        DELIVERED, CANCELLED -> false
    }
}

// object: シングルトン
object IdGenerator {
    private var counter = 0L

    @Synchronized
    fun nextId(): String {
        counter++
        return "ID-${counter.toString().padStart(8, '0')}"
    }
}

// companion object: 静的メンバ相当
class User private constructor(
    val id: String,
    val name: String,
    val email: String,
) {
    companion object {
        fun create(name: String, email: String): User {
            require(name.isNotBlank()) { "名前は空にできません" }
            require("@" in email) { "有効なメールアドレスが必要です" }
            return User(IdGenerator.nextId(), name, email)
        }
    }
}

// 使用例
val shapes = listOf(
    Shape.Circle(5.0),
    Shape.Rectangle(3.0, 4.0),
    Shape.Triangle(3.0, 4.0, 5.0),
)
shapes.forEach { println(describeShape(it)) }

val addr = Address("100-0001", "東京都", "千代田区", "千代田1-1")
println(addr.toSingleLine())

val user = User.create("田中太郎", "tanaka@example.com")
println("${user.id}: ${user.name}")
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| クラス | 設計図。メタデータとしてメモリに1つ |
| オブジェクト | 実体。ヒープ上に複数存在 |
| 3つの特性 | 状態・振る舞い・同一性 |
| メモリ配置 | vtable、ヘッダー、フィールドレイアウト |
| コンストラクタ | 初期化 + 不変条件の確立 |
| Builder | 複雑な構築を段階的に行う |
| 静的メンバ | クラスに属する。ユーティリティ/ファクトリ |
| 値型 vs 参照型 | コピーセマンティクスの違い |
| 等価性 vs 同一性 | equals/hashCode の正しい実装 |
| 合成 | has-a 関係。継承より柔軟 |
| メタクラス | クラスの振る舞いを制御する |
| クラスの種類 | 具象、抽象、sealed、data、enum |

---

## 次に読むべきガイド
→ [[../01-four-pillars/00-encapsulation.md]] — カプセル化

---

## 参考文献
1. Bloch, J. "Effective Java." 3rd Ed, Addison-Wesley, 2018.
2. Eckel, B. "Thinking in Java." 4th Ed, Prentice Hall, 2006.
3. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
4. Meyer, B. "Object-Oriented Software Construction." 2nd Ed, Prentice Hall, 1997.
5. Evans, E. "Domain-Driven Design." Addison-Wesley, 2003.
6. Fowler, M. "Patterns of Enterprise Application Architecture." Addison-Wesley, 2002.
