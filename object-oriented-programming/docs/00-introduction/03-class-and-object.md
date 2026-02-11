# クラスとオブジェクト

> クラスは「設計図」、オブジェクトは「実体」。この関係を深く理解し、メモリ上の配置、コンストラクタの設計、静的メンバの適切な使い方をマスターする。

## この章で学ぶこと

- [ ] クラスとオブジェクトの関係をメモリレベルで理解する
- [ ] コンストラクタの設計パターンを把握する
- [ ] 静的メンバとインスタンスメンバの使い分けを学ぶ

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

---

## 3. コンストラクタ

```
コンストラクタの役割:
  1. フィールドの初期化
  2. 不変条件（invariant）の確立
  3. 依存オブジェクトの注入
```

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

---

## まとめ

| 概念 | ポイント |
|------|---------|
| クラス | 設計図。メタデータとしてメモリに1つ |
| オブジェクト | 実体。ヒープ上に複数存在 |
| コンストラクタ | 初期化 + 不変条件の確立 |
| 静的メンバ | クラスに属する。ユーティリティ/ファクトリ |
| 値型 vs 参照型 | コピーセマンティクスの違い |

---

## 次に読むべきガイド
→ [[../01-four-pillars/00-encapsulation.md]] — カプセル化

---

## 参考文献
1. Bloch, J. "Effective Java." 3rd Ed, Addison-Wesley, 2018.
2. Eckel, B. "Thinking in Java." 4th Ed, Prentice Hall, 2006.
