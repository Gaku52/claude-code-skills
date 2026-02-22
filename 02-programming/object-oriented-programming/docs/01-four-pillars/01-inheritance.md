# 継承

> 継承は「既存クラスの機能を引き継いで新しいクラスを作る」仕組み。強力だが誤用しやすく、モダンOOPでは「継承よりコンポジション」が原則となっている。

## この章で学ぶこと

- [ ] 継承の仕組みとメモリ上の表現を理解する
- [ ] 継承の適切な使い方と落とし穴を把握する
- [ ] 抽象クラスとインターフェースの違いを学ぶ
- [ ] 多重継承の問題と各言語での解決策を理解する
- [ ] コンポジションとの使い分け基準を実践的に把握する
- [ ] テンプレートメソッドパターンの活用方法を学ぶ

---

## 1. 継承の基本

```
継承（Inheritance）:
  → 親クラス（スーパークラス）のフィールドとメソッドを
    子クラス（サブクラス）が引き継ぐ仕組み

  Animal（親クラス）
  ├── name: string
  ├── sound(): string
  └── move(): void
       ↑ 継承
  ┌─────┴──────┐
  Dog          Cat
  ├── breed    ├── indoor
  └── fetch()  └── purr()

  Dog は Animal の name, sound(), move() を自動的に持つ
  + 独自の breed, fetch() を追加
```

### 1.1 継承の基本構文（各言語比較）

```typescript
// TypeScript: 基本的な継承
class Animal {
  constructor(
    protected name: string,
    protected age: number,
  ) {}

  speak(): string {
    return `${this.name}が鳴いています`;
  }

  toString(): string {
    return `${this.name} (${this.age}歳)`;
  }
}

class Dog extends Animal {
  constructor(name: string, age: number, private breed: string) {
    super(name, age); // 親のコンストラクタ呼び出し
  }

  // オーバーライド（親のメソッドを上書き）
  speak(): string {
    return `${this.name}「ワン！」`;
  }

  fetch(): string {
    return `${this.name}がボールを取ってきた`;
  }
}

class Cat extends Animal {
  speak(): string {
    return `${this.name}「ニャー」`;
  }
}

const dog = new Dog("ポチ", 3, "柴犬");
const cat = new Cat("タマ", 5);
console.log(dog.speak()); // ポチ「ワン！」
console.log(cat.speak()); // タマ「ニャー」
```

```java
// Java: 基本的な継承
public abstract class Vehicle {
    protected String name;
    protected int year;
    protected double fuelLevel;

    public Vehicle(String name, int year) {
        this.name = name;
        this.year = year;
        this.fuelLevel = 100.0;
    }

    // 共通メソッド
    public String getInfo() {
        return String.format("%s (%d年式) 燃料: %.1f%%", name, year, fuelLevel);
    }

    // 抽象メソッド（サブクラスで実装必須）
    public abstract double getFuelEfficiency();

    // テンプレートメソッド
    public final void startEngine() {
        if (fuelLevel <= 0) {
            System.out.println("燃料がありません");
            return;
        }
        performPreCheck();
        ignite();
        System.out.println(name + " のエンジンが始動しました");
    }

    protected void performPreCheck() {
        System.out.println("基本チェック実行中...");
    }

    protected abstract void ignite();
}

public class Car extends Vehicle {
    private int doorCount;

    public Car(String name, int year, int doorCount) {
        super(name, year);
        this.doorCount = doorCount;
    }

    @Override
    public double getFuelEfficiency() {
        return 15.0; // km/L
    }

    @Override
    protected void ignite() {
        System.out.println("セルモーター始動");
    }

    @Override
    protected void performPreCheck() {
        super.performPreCheck(); // 親の処理も実行
        System.out.println("ドアロック確認: " + doorCount + "ドア");
    }
}

public class Motorcycle extends Vehicle {
    private boolean hasSidecar;

    public Motorcycle(String name, int year, boolean hasSidecar) {
        super(name, year);
        this.hasSidecar = hasSidecar;
    }

    @Override
    public double getFuelEfficiency() {
        return hasSidecar ? 20.0 : 30.0;
    }

    @Override
    protected void ignite() {
        System.out.println("キックスタート");
    }
}

// 使用例
Vehicle car = new Car("トヨタ カローラ", 2024, 4);
Vehicle bike = new Motorcycle("ホンダ CB400", 2023, false);

car.startEngine();
// 基本チェック実行中...
// ドアロック確認: 4ドア
// セルモーター始動
// トヨタ カローラ のエンジンが始動しました

bike.startEngine();
// 基本チェック実行中...
// キックスタート
// ホンダ CB400 のエンジンが始動しました
```

```python
# Python: 基本的な継承
class Employee:
    """従業員の基底クラス"""

    def __init__(self, name: str, employee_id: str, base_salary: float):
        self.name = name
        self.employee_id = employee_id
        self.base_salary = base_salary
        self._benefits: list[str] = ["健康保険", "厚生年金"]

    def calculate_pay(self) -> float:
        """月額給与を計算"""
        return self.base_salary

    def get_benefits(self) -> list[str]:
        """福利厚生一覧を取得"""
        return self._benefits.copy()

    def __str__(self) -> str:
        return f"{self.name} (ID: {self.employee_id})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}', id='{self.employee_id}')"


class FullTimeEmployee(Employee):
    """正社員"""

    def __init__(self, name: str, employee_id: str, base_salary: float,
                 bonus_rate: float = 0.2):
        super().__init__(name, employee_id, base_salary)
        self.bonus_rate = bonus_rate
        self._benefits.extend(["退職金", "住宅手当"])

    def calculate_pay(self) -> float:
        """基本給 + ボーナス分"""
        return self.base_salary * (1 + self.bonus_rate)

    def calculate_annual_bonus(self) -> float:
        """年間ボーナス"""
        return self.base_salary * self.bonus_rate * 2  # 夏冬


class PartTimeEmployee(Employee):
    """パートタイム従業員"""

    def __init__(self, name: str, employee_id: str,
                 hourly_rate: float, hours_per_month: float):
        # base_salary は時給 × 時間で計算
        super().__init__(name, employee_id, hourly_rate * hours_per_month)
        self.hourly_rate = hourly_rate
        self.hours_per_month = hours_per_month
        # パートタイムは福利厚生が限定的
        self._benefits = ["健康保険"]

    def calculate_pay(self) -> float:
        """時給 × 時間"""
        return self.hourly_rate * self.hours_per_month


class Manager(FullTimeEmployee):
    """管理職（正社員からさらに継承）"""

    def __init__(self, name: str, employee_id: str, base_salary: float,
                 bonus_rate: float = 0.3, team_size: int = 0):
        super().__init__(name, employee_id, base_salary, bonus_rate)
        self.team_size = team_size
        self._benefits.append("管理職手当")

    def calculate_pay(self) -> float:
        """基本給 + ボーナス + 管理手当"""
        base_pay = super().calculate_pay()
        management_allowance = 50000 * (self.team_size // 5)  # 5人ごとに5万円
        return base_pay + management_allowance


# 使用例: ポリモーフィズムとの連携
employees: list[Employee] = [
    FullTimeEmployee("田中太郎", "FT001", 350000),
    PartTimeEmployee("鈴木花子", "PT001", 1200, 80),
    Manager("佐藤部長", "MG001", 500000, team_size=12),
]

for emp in employees:
    print(f"{emp}: ¥{emp.calculate_pay():,.0f}")
# 田中太郎 (ID: FT001): ¥420,000
# 鈴木花子 (ID: PT001): ¥96,000
# 佐藤部長 (ID: MG001): ¥750,000
```

### 1.2 継承のメモリレイアウト

```
メモリ上での継承オブジェクトの表現:

  Manager オブジェクトのメモリレイアウト:
  ┌──────────────────────────────────────┐
  │ vptr → Manager の vtable             │ ← 仮想関数テーブルポインタ
  ├──────────────────────────────────────┤
  │ name: "佐藤部長"                     │ ← Employee のフィールド
  │ employee_id: "MG001"                 │
  │ base_salary: 500000                  │
  │ _benefits: [...]                     │
  ├──────────────────────────────────────┤
  │ bonus_rate: 0.3                      │ ← FullTimeEmployee のフィールド
  ├──────────────────────────────────────┤
  │ team_size: 12                        │ ← Manager のフィールド
  └──────────────────────────────────────┘

  継承チェーン:
  Employee → FullTimeEmployee → Manager

  各レベルのフィールドが連続して配置される
  → 継承が深いほどオブジェクトサイズが大きくなる
```

---

## 2. メソッドオーバーライドと super

```
オーバーライド（Override）:
  → 親クラスのメソッドを子クラスで再定義
  → 動的ディスパッチ: 実行時に実際の型のメソッドが呼ばれる

super の役割:
  → 親クラスのコンストラクタ/メソッドを明示的に呼ぶ
  → 「親の処理 + 追加の処理」パターン

オーバーライドの3つのパターン:
  1. 完全置換: 親のメソッドを完全に新しい実装に置き換え
  2. 拡張: super() を呼んだ上で追加処理
  3. 条件分岐: 条件に応じて親の実装を使うか独自実装を使うか切り替え
```

```python
# Python: super() の使い方と3つのオーバーライドパターン
class Shape:
    def __init__(self, color: str = "black"):
        self.color = color

    def area(self) -> float:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.color}の{type(self).__name__}"

    def validate(self) -> bool:
        """形状が有効かどうかを検証"""
        return True

class Circle(Shape):
    def __init__(self, radius: float, color: str = "black"):
        super().__init__(color)  # 親の初期化
        self.radius = radius

    # パターン1: 完全置換
    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    # パターン2: 拡張（super() + 追加処理）
    def describe(self) -> str:
        return f"{super().describe()}, 半径{self.radius}"

    # パターン3: 条件分岐
    def validate(self) -> bool:
        if self.radius <= 0:
            return False
        return super().validate()

class Rectangle(Shape):
    def __init__(self, width: float, height: float, color: str = "black"):
        super().__init__(color)
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def describe(self) -> str:
        return f"{super().describe()}, {self.width}×{self.height}"

    def validate(self) -> bool:
        if self.width <= 0 or self.height <= 0:
            return False
        return super().validate()

class Square(Rectangle):
    """正方形は長方形の特殊ケース（ただし注意が必要）"""
    def __init__(self, side: float, color: str = "black"):
        super().__init__(side, side, color)

    def describe(self) -> str:
        # 親の describe を完全に置き換え
        return f"{self.color}の正方形, 一辺{self.width}"
```

### 2.1 super() の高度な使い方

```python
# Python: 協調的多重継承での super()
class Loggable:
    """ログ機能を提供するMixin"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log: list[str] = []

    def log(self, message: str) -> None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.append(f"[{timestamp}] {message}")

    def get_log(self) -> list[str]:
        return self._log.copy()


class Validatable:
    """バリデーション機能を提供するMixin"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._errors: list[str] = []

    def add_error(self, error: str) -> None:
        self._errors.append(error)

    def is_valid(self) -> bool:
        self._errors.clear()
        self.validate()
        return len(self._errors) == 0

    def validate(self) -> None:
        """サブクラスでオーバーライドしてバリデーションルールを追加"""
        pass


class Product(Loggable, Validatable):
    """商品クラス（複数のMixinを継承）"""
    def __init__(self, name: str, price: float):
        super().__init__()  # MRO に従って全ての __init__ が呼ばれる
        self.name = name
        self.price = price
        self.log(f"商品作成: {name}")

    def validate(self) -> None:
        super().validate()  # Validatable.validate() を呼ぶ
        if not self.name:
            self.add_error("商品名は必須です")
        if self.price < 0:
            self.add_error("価格は0以上である必要があります")
        if self.price > 10_000_000:
            self.add_error("価格が上限を超えています")

    def update_price(self, new_price: float) -> None:
        old_price = self.price
        self.price = new_price
        self.log(f"価格変更: {old_price} → {new_price}")


# 使用例
product = Product("ノートPC", 150000)
print(product.is_valid())    # True
print(product.get_log())     # ['[2024-...] 商品作成: ノートPC']

product.update_price(-100)
print(product.is_valid())    # False（価格が負）

# MRO の確認
print(Product.__mro__)
# (Product, Loggable, Validatable, object)
```

```typescript
// TypeScript: super の高度な活用パターン

abstract class Component {
  protected children: Component[] = [];
  protected parent: Component | null = null;

  constructor(protected id: string) {}

  addChild(child: Component): void {
    child.parent = this;
    this.children.push(child);
  }

  // テンプレートメソッド
  render(): string {
    const self = this.renderSelf();
    const children = this.children.map(c => c.render()).join("\n");
    return children ? `${self}\n${children}` : self;
  }

  protected abstract renderSelf(): string;

  // ライフサイクルフック
  mount(): void {
    this.onBeforeMount();
    this.children.forEach(c => c.mount());
    this.onMounted();
  }

  protected onBeforeMount(): void {}
  protected onMounted(): void {}
}

class Panel extends Component {
  constructor(id: string, private title: string) {
    super(id);
  }

  protected renderSelf(): string {
    return `<panel id="${this.id}" title="${this.title}">`;
  }

  protected onMounted(): void {
    console.log(`Panel "${this.title}" mounted`);
  }
}

class Button extends Component {
  constructor(id: string, private label: string, private onClick: () => void) {
    super(id);
  }

  protected renderSelf(): string {
    return `<button id="${this.id}">${this.label}</button>`;
  }

  // super.mount() を呼んだ上でイベントハンドラを追加
  mount(): void {
    super.mount();
    console.log(`Button "${this.label}" にクリックハンドラを登録`);
  }
}

class Form extends Panel {
  private fields: Map<string, string> = new Map();

  constructor(id: string, title: string) {
    super(id, title);
  }

  addField(name: string, defaultValue: string = ""): void {
    this.fields.set(name, defaultValue);
  }

  // 親の renderSelf を拡張
  protected renderSelf(): string {
    const base = super.renderSelf();
    const fieldsHtml = Array.from(this.fields.entries())
      .map(([name, value]) => `  <input name="${name}" value="${value}" />`)
      .join("\n");
    return `${base}\n${fieldsHtml}`;
  }

  protected onMounted(): void {
    super.onMounted(); // Panel.onMounted() を呼ぶ
    console.log(`Form fields: ${this.fields.size}個`);
  }
}

// 使用例
const form = new Form("login-form", "ログイン");
form.addField("username");
form.addField("password");
form.addChild(new Button("submit-btn", "ログイン", () => {}));
form.mount();
// Panel "ログイン" mounted
// Form fields: 2個
// Button "ログイン" にクリックハンドラを登録
```

---

## 3. 継承の種類

```
単一継承（Single Inheritance）:
  → 1つの親クラスのみ継承可能
  → Java, C#, Swift, Kotlin, Ruby
  → シンプルだが表現力に制限

多重継承（Multiple Inheritance）:
  → 複数の親クラスを継承可能
  → C++, Python
  → 強力だがダイヤモンド問題が発生

  ┌─────────┐
  │ Animal  │ ← ダイヤモンド問題
  └────┬────┘
  ┌────┴────┐
  ▼         ▼
┌─────┐  ┌──────┐
│ Fly │  │ Swim │
└──┬──┘  └──┬───┘
   └────┬───┘
        ▼
  ┌──────────┐
  │ FlyFish  │ ← Animal のメソッドをどちらから継承？
  └──────────┘

インターフェースによる多重実装:
  → Java, C#, TypeScript, Kotlin, Swift
  → 実装を持たない（Java 8+ の default メソッドは例外）
  → ダイヤモンド問題を部分的に回避

Mixin / Trait:
  → Ruby (module), Scala (trait), Rust (trait), Kotlin (interface + default)
  → 実装を含むが、状態（フィールド）は制限的
  → 多重継承の利点を安全に提供
```

### 3.1 Python の MRO（Method Resolution Order）

```python
# Python: MRO（Method Resolution Order）でダイヤモンド問題を解決
class Animal:
    def move(self):
        return "移動"

    def breathe(self):
        return "呼吸する"

class Flyer(Animal):
    def move(self):
        return "飛ぶ"

    def take_off(self):
        return "離陸"

class Swimmer(Animal):
    def move(self):
        return "泳ぐ"

    def dive(self):
        return "潜水"

class FlyingFish(Flyer, Swimmer):
    pass

fish = FlyingFish()
print(fish.move())      # "飛ぶ"（MRO: FlyingFish → Flyer → Swimmer → Animal）
print(fish.breathe())   # "呼吸する"（Animal から継承）
print(fish.take_off())  # "離陸"（Flyer から継承）
print(fish.dive())      # "潜水"（Swimmer から継承）

# MROの確認
print(FlyingFish.__mro__)
# (FlyingFish, Flyer, Swimmer, Animal, object)
# → C3線形化アルゴリズムで順序を決定
```

### 3.2 C3線形化アルゴリズムの詳細

```
C3線形化（C3 Linearization）:
  Python が MRO を決定するために使用するアルゴリズム

ルール:
  1. 子クラスは常に親クラスより先
  2. 複数の親がある場合、定義順序を維持
  3. 矛盾する順序は許可しない（TypeError が発生）

例: class D(B, C) で B(A), C(A) の場合
  L[D] = D + merge(L[B], L[C], [B, C])
  L[B] = B, A, object
  L[C] = C, A, object
  merge([B, A, object], [C, A, object], [B, C])
  = B + merge([A, object], [C, A, object], [C])
  = B, C + merge([A, object], [A, object])
  = B, C, A + merge([object], [object])
  = B, C, A, object
  → D の MRO = [D, B, C, A, object]
```

```python
# 矛盾するMROの例（TypeError が発生）
class A:
    pass

class B(A):
    pass

class C(A, B):  # A が B より先だが、B は A を継承している
    pass
# TypeError: Cannot create a consistent method resolution order (MRO)
# → A と B の順序が矛盾するため
```

### 3.3 各言語の多重継承への対処

```java
// Java: インターフェースの default メソッドによる多重実装
interface Flyable {
    default String move() {
        return "飛ぶ";
    }

    String altitude();
}

interface Swimmable {
    default String move() {
        return "泳ぐ";
    }

    String depth();
}

// 両方のインターフェースを実装する場合、
// 同名の default メソッドは明示的にオーバーライドが必要
class Duck implements Flyable, Swimmable {
    @Override
    public String move() {
        // どちらかを選ぶ、または独自実装
        return Flyable.super.move() + "ことも" + Swimmable.super.move() + "こともできる";
    }

    @Override
    public String altitude() {
        return "100m";
    }

    @Override
    public String depth() {
        return "5m";
    }
}

Duck duck = new Duck();
System.out.println(duck.move()); // "飛ぶことも泳ぐこともできる"
```

```kotlin
// Kotlin: インターフェースの default 実装
interface Logger {
    fun log(message: String) {
        println("[LOG] $message")
    }
}

interface Auditable {
    fun audit(action: String) {
        println("[AUDIT] $action")
    }

    fun log(message: String) {
        println("[AUDIT-LOG] $message")
    }
}

class UserService : Logger, Auditable {
    // 同名メソッドが衝突するため、明示的にオーバーライド
    override fun log(message: String) {
        super<Logger>.log(message)      // Logger の実装を呼ぶ
        super<Auditable>.audit(message) // Auditable の audit も呼ぶ
    }

    fun createUser(name: String) {
        log("ユーザー作成: $name")
    }
}
```

```ruby
# Ruby: Module（Mixin）による多重継承の代替
module Serializable
  def serialize
    instance_variables.each_with_object({}) do |var, hash|
      hash[var.to_s.delete('@')] = instance_variable_get(var)
    end
  end

  def to_json
    require 'json'
    JSON.generate(serialize)
  end
end

module Cacheable
  def cache_key
    "#{self.class.name}:#{object_id}"
  end

  def cached?
    @_cached ||= false
  end

  def mark_cached!
    @_cached = true
  end
end

module Auditable
  def audit_trail
    @_audit_trail ||= []
  end

  def record_change(field, old_value, new_value)
    audit_trail << {
      field: field,
      old: old_value,
      new: new_value,
      at: Time.now
    }
  end
end

class User
  include Serializable
  include Cacheable
  include Auditable

  attr_reader :name, :email

  def initialize(name, email)
    @name = name
    @email = email
  end

  def update_email(new_email)
    record_change(:email, @email, new_email)
    @email = new_email
  end
end

user = User.new("田中", "tanaka@example.com")
puts user.to_json        # {"name":"田中","email":"tanaka@example.com"}
puts user.cache_key      # "User:12345"
user.update_email("new@example.com")
puts user.audit_trail    # [{field: :email, old: "tanaka@...", new: "new@...", ...}]

# 継承チェーンの確認
puts User.ancestors
# [User, Auditable, Cacheable, Serializable, Object, Kernel, BasicObject]
```

---

## 4. 継承の落とし穴

```
問題1: 脆い基底クラス問題（Fragile Base Class）
  → 親クラスの変更が子クラスを壊す

問題2: 不適切な is-a 関係
  → 正方形 is-a 長方形？（リスコフの置換原則に違反）

問題3: 深い継承階層
  → 3段階以上の継承は理解困難
  → Entity → LivingEntity → Animal → Mammal → Dog → GuideDog
  → 各レイヤーの変更が下位全てに影響

問題4: 継承によるカプセル化の破壊
  → 子クラスが親の実装詳細に依存
  → protected フィールドへの直接アクセス

問題5: ゴリラ・バナナ問題
  「バナナが欲しいだけなのに、バナナを持ったゴリラと
   ジャングル全体がついてきた」
  → 継承すると不要な機能も全てついてくる
```

### 4.1 脆い基底クラス問題

```java
// 脆い基底クラス問題の例（Effective Java Item 18より）
public class HashSet<E> {
    private int addCount = 0;

    public boolean add(E e) {
        addCount++;
        // ... 実際の追加処理
        return true;
    }

    public boolean addAll(Collection<E> c) {
        // 内部で add() を呼ぶ実装
        for (E e : c) add(e);
        return true;
    }

    public int getAddCount() { return addCount; }
}

// 問題のあるサブクラス
public class InstrumentedHashSet<E> extends HashSet<E> {
    private int addCount = 0;

    @Override
    public boolean add(E e) {
        addCount++;
        return super.add(e);
    }

    @Override
    public boolean addAll(Collection<E> c) {
        addCount += c.size();
        return super.addAll(c); // super.addAll() が add() を呼ぶ！
    }
    // addAll({a, b, c}) → addCount = 6（期待は3）
    // → super.addAll() が内部で add() を呼び、二重カウント
}
```

```java
// 解決策: コンポジションを使う（Effective Java推奨）
public class InstrumentedSet<E> {
    private final Set<E> set;  // コンポジション
    private int addCount = 0;

    public InstrumentedSet(Set<E> set) {
        this.set = set;
    }

    public boolean add(E e) {
        addCount++;
        return set.add(e);  // 委譲（delegation）
    }

    public boolean addAll(Collection<E> c) {
        addCount += c.size();
        return set.addAll(c);  // set 内部の実装に依存しない
    }

    public int getAddCount() { return addCount; }

    // 必要な Set のメソッドを委譲
    public boolean contains(Object o) { return set.contains(o); }
    public int size() { return set.size(); }
    public Iterator<E> iterator() { return set.iterator(); }
}

// 使用例: どの Set 実装でも使える
InstrumentedSet<String> s1 = new InstrumentedSet<>(new HashSet<>());
InstrumentedSet<String> s2 = new InstrumentedSet<>(new TreeSet<>());
InstrumentedSet<String> s3 = new InstrumentedSet<>(new LinkedHashSet<>());
```

### 4.2 不適切な is-a 関係（正方形-長方形問題）

```typescript
// ❌ 正方形 extends 長方形: LSP 違反の典型例
class Rectangle {
  constructor(protected width: number, protected height: number) {}

  setWidth(w: number): void {
    this.width = w;
  }

  setHeight(h: number): void {
    this.height = h;
  }

  area(): number {
    return this.width * this.height;
  }
}

class Square extends Rectangle {
  constructor(side: number) {
    super(side, side);
  }

  // 正方形は幅を変えたら高さも変わる（親と異なる振る舞い）
  setWidth(w: number): void {
    this.width = w;
    this.height = w; // ← 親クラスにない副作用
  }

  setHeight(h: number): void {
    this.width = h;
    this.height = h; // ← 親クラスにない副作用
  }
}

// 問題: Rectangle として使うと期待通りに動かない
function doubleWidth(rect: Rectangle): void {
  const originalHeight = rect.area() / rect.area(); // 元の高さを保持
  rect.setWidth(rect.area() / 10); // 幅だけ変えたつもり
  // Square だと高さも変わってしまう！
}

// ✅ 解決策: 共通インターフェースで抽象化
interface Shape {
  area(): number;
  perimeter(): number;
}

class ImmutableRectangle implements Shape {
  constructor(readonly width: number, readonly height: number) {}
  area(): number { return this.width * this.height; }
  perimeter(): number { return 2 * (this.width + this.height); }
}

class ImmutableSquare implements Shape {
  constructor(readonly side: number) {}
  area(): number { return this.side ** 2; }
  perimeter(): number { return 4 * this.side; }
}
```

### 4.3 深い継承階層の問題

```
深い継承階層のリスク:

  Level 0: Entity
  Level 1: └── LivingEntity
  Level 2:     └── Animal
  Level 3:         └── Mammal
  Level 4:             └── Canine
  Level 5:                 └── Dog
  Level 6:                     └── GuideDog

  問題:
  1. Level 2 (Animal) を変更 → Level 3-6 全てに影響
  2. GuideDog のバグ原因が Level 1 にある可能性
  3. 新しい種類の犬を追加するとき、どのレベルに追加すべきか不明
  4. テスト時に全レベルのセットアップが必要

  推奨: 最大2-3レベルまで
  → それ以上はコンポジションに切り替える
```

```python
# 深い継承階層をコンポジションで改善する例

# ❌ 深い継承階層
class Entity:
    def __init__(self, id: str):
        self.id = id

class LivingEntity(Entity):
    def __init__(self, id: str, health: float):
        super().__init__(id)
        self.health = health

class Animal(LivingEntity):
    def __init__(self, id: str, health: float, species: str):
        super().__init__(id, health)
        self.species = species

class Pet(Animal):
    def __init__(self, id: str, health: float, species: str, owner: str):
        super().__init__(id, health, species)
        self.owner = owner

class Dog(Pet):
    def __init__(self, id: str, health: float, owner: str, breed: str):
        super().__init__(id, health, "犬", owner)
        self.breed = breed

class GuideDog(Dog):
    def __init__(self, id: str, health: float, owner: str, breed: str,
                 handler: str, certification: str):
        super().__init__(id, health, owner, breed)
        self.handler = handler
        self.certification = certification


# ✅ コンポジションで改善（浅い継承 + 機能を部品として組み立て）
from dataclasses import dataclass
from typing import Optional

@dataclass
class Identity:
    id: str
    created_at: Optional[str] = None

@dataclass
class HealthStatus:
    current_hp: float
    max_hp: float

    @property
    def is_alive(self) -> bool:
        return self.current_hp > 0

    def take_damage(self, amount: float) -> None:
        self.current_hp = max(0, self.current_hp - amount)

@dataclass
class OwnerInfo:
    owner_name: str
    owner_contact: str

@dataclass
class GuideDogCertification:
    handler_name: str
    certification_id: str
    expires_at: str

class AnimalV2:
    """浅い構造: コンポジションで機能を組み立て"""
    def __init__(self, identity: Identity, species: str, breed: str):
        self.identity = identity
        self.species = species
        self.breed = breed
        self.health: Optional[HealthStatus] = None
        self.owner: Optional[OwnerInfo] = None
        self.guide_cert: Optional[GuideDogCertification] = None

    def is_guide_dog(self) -> bool:
        return self.guide_cert is not None

    def is_pet(self) -> bool:
        return self.owner is not None


# 使用例
guide_dog = AnimalV2(
    identity=Identity(id="GD-001"),
    species="犬",
    breed="ラブラドール",
)
guide_dog.health = HealthStatus(current_hp=100, max_hp=100)
guide_dog.owner = OwnerInfo(owner_name="佐藤", owner_contact="090-XXXX")
guide_dog.guide_cert = GuideDogCertification(
    handler_name="田中",
    certification_id="CERT-2024-001",
    expires_at="2026-12-31",
)
```

---

## 5. 抽象クラス

```
抽象クラス:
  → インスタンス化できないクラス
  → 共通の実装 + サブクラスへの実装義務を定義
  → 「テンプレートメソッドパターン」の基盤

用途:
  - 共通のフィールドと一部のメソッド実装を提供
  - サブクラスが実装すべきメソッドを強制
  - is-a 関係が明確な場合

抽象クラス vs インターフェース:
  抽象クラス: 「何であるか」+ 共通実装
  インターフェース: 「何ができるか」のみ
```

### 5.1 抽象クラスの基本

```typescript
// TypeScript: 抽象クラス
abstract class DatabaseConnection {
  protected connected: boolean = false;
  protected queryCount: number = 0;

  // 共通実装
  async query(sql: string): Promise<any[]> {
    if (!this.connected) {
      await this.connect();
    }
    this.queryCount++;
    const startTime = Date.now();
    const result = await this.executeQuery(sql);
    const duration = Date.now() - startTime;
    console.log(`Query #${this.queryCount} took ${duration}ms`);
    return result;
  }

  // トランザクション管理（テンプレートメソッド）
  async withTransaction<T>(fn: () => Promise<T>): Promise<T> {
    await this.beginTransaction();
    try {
      const result = await fn();
      await this.commitTransaction();
      return result;
    } catch (error) {
      await this.rollbackTransaction();
      throw error;
    }
  }

  // サブクラスが実装すべき抽象メソッド
  abstract connect(): Promise<void>;
  abstract disconnect(): Promise<void>;
  protected abstract executeQuery(sql: string): Promise<any[]>;
  protected abstract beginTransaction(): Promise<void>;
  protected abstract commitTransaction(): Promise<void>;
  protected abstract rollbackTransaction(): Promise<void>;
}

class PostgresConnection extends DatabaseConnection {
  private client: any; // pg.Client

  async connect(): Promise<void> {
    // PostgreSQL固有の接続処理
    this.client = {}; // new pg.Client(connectionString)
    // await this.client.connect();
    this.connected = true;
    console.log("PostgreSQL に接続しました");
  }

  async disconnect(): Promise<void> {
    // await this.client.end();
    this.connected = false;
    console.log("PostgreSQL から切断しました");
  }

  protected async executeQuery(sql: string): Promise<any[]> {
    // const result = await this.client.query(sql);
    // return result.rows;
    console.log(`PostgreSQL 実行: ${sql}`);
    return [];
  }

  protected async beginTransaction(): Promise<void> {
    await this.executeQuery("BEGIN");
  }

  protected async commitTransaction(): Promise<void> {
    await this.executeQuery("COMMIT");
  }

  protected async rollbackTransaction(): Promise<void> {
    await this.executeQuery("ROLLBACK");
  }
}

class SQLiteConnection extends DatabaseConnection {
  private db: any;

  async connect(): Promise<void> {
    // SQLite固有の接続処理
    this.db = {}; // new sqlite3.Database(path)
    this.connected = true;
    console.log("SQLite に接続しました");
  }

  async disconnect(): Promise<void> {
    // this.db.close();
    this.connected = false;
  }

  protected async executeQuery(sql: string): Promise<any[]> {
    console.log(`SQLite 実行: ${sql}`);
    return [];
  }

  protected async beginTransaction(): Promise<void> {
    await this.executeQuery("BEGIN TRANSACTION");
  }

  protected async commitTransaction(): Promise<void> {
    await this.executeQuery("COMMIT TRANSACTION");
  }

  protected async rollbackTransaction(): Promise<void> {
    await this.executeQuery("ROLLBACK TRANSACTION");
  }
}
```

### 5.2 テンプレートメソッドパターン

```python
# Python: テンプレートメソッドパターンの実践例
from abc import ABC, abstractmethod
from typing import Any
import time


class DataPipeline(ABC):
    """データ処理パイプラインの抽象基底クラス"""

    def run(self, source: str) -> dict[str, Any]:
        """テンプレートメソッド: 処理の流れを定義"""
        start_time = time.time()

        # 1. データ取得
        raw_data = self.extract(source)
        print(f"  取得: {len(raw_data)} 件")

        # 2. バリデーション
        valid_data = self.validate(raw_data)
        print(f"  有効: {len(valid_data)} 件")

        # 3. データ変換
        transformed = self.transform(valid_data)
        print(f"  変換完了")

        # 4. データ保存
        result = self.load(transformed)
        print(f"  保存完了")

        # 5. 後処理（オプショナル: フック）
        self.on_complete(result)

        elapsed = time.time() - start_time
        return {
            "source": source,
            "input_count": len(raw_data),
            "output_count": len(valid_data),
            "elapsed_seconds": round(elapsed, 2),
            "result": result,
        }

    # 必須: サブクラスが実装する
    @abstractmethod
    def extract(self, source: str) -> list[dict]:
        """データソースからデータを取得"""
        ...

    @abstractmethod
    def transform(self, data: list[dict]) -> list[dict]:
        """データを変換"""
        ...

    @abstractmethod
    def load(self, data: list[dict]) -> Any:
        """データを保存先に書き込む"""
        ...

    # オプショナル: デフォルト実装あり（オーバーライド可能）
    def validate(self, data: list[dict]) -> list[dict]:
        """デフォルトのバリデーション（None/空を除外）"""
        return [d for d in data if d]

    def on_complete(self, result: Any) -> None:
        """処理完了時のフック（デフォルトは何もしない）"""
        pass


class CsvToJsonPipeline(DataPipeline):
    """CSVファイルを読み込んでJSONに変換するパイプライン"""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def extract(self, source: str) -> list[dict]:
        import csv
        with open(source, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def validate(self, data: list[dict]) -> list[dict]:
        # 親のバリデーション + 追加ルール
        valid = super().validate(data)
        return [d for d in valid if all(v.strip() for v in d.values())]

    def transform(self, data: list[dict]) -> list[dict]:
        # 文字列の数値フィールドを変換
        for row in data:
            for key, value in row.items():
                try:
                    row[key] = int(value)
                except (ValueError, TypeError):
                    try:
                        row[key] = float(value)
                    except (ValueError, TypeError):
                        pass
        return data

    def load(self, data: list[dict]) -> str:
        import json
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return self.output_path

    def on_complete(self, result: Any) -> None:
        print(f"  → {result} に保存しました")


class ApiToDbPipeline(DataPipeline):
    """REST APIからデータを取得してDBに保存するパイプライン"""

    def __init__(self, db_connection: Any):
        self.db = db_connection

    def extract(self, source: str) -> list[dict]:
        import urllib.request
        import json
        with urllib.request.urlopen(source) as response:
            return json.loads(response.read())

    def transform(self, data: list[dict]) -> list[dict]:
        # APIレスポンスをDBスキーマにマッピング
        return [
            {
                "external_id": item.get("id"),
                "name": item.get("name", "").strip(),
                "email": item.get("email", "").lower(),
                "created_at": item.get("created_at"),
            }
            for item in data
        ]

    def load(self, data: list[dict]) -> int:
        # DBに一括保存
        count = 0
        for row in data:
            # self.db.insert("users", row)
            count += 1
        return count

    def on_complete(self, result: Any) -> None:
        print(f"  → {result} 件をDBに保存しました")
```

---

## 6. 継承とコンポジションの比較

```
継承を使うべき場面:
  ✓ 明確な is-a 関係がある
  ✓ サブクラスが親のインターフェースを完全に満たす（LSP）
  ✓ フレームワークが継承を要求する（Androidの Activity等）
  ✓ テンプレートメソッドパターン

継承を避けるべき場面:
  ✗ コードの再利用だけが目的
  ✗ has-a 関係（コンポジションを使う）
  ✗ 3段階以上の継承が必要
  ✗ 親クラスの一部のメソッドだけ使いたい

判断基準:
  「このサブクラスは、親クラスが使える全ての場面で
   代替として使えるか？」
  → Yes → 継承が適切
  → No  → コンポジションを使う
```

### 6.1 継承 vs コンポジション 実践比較

```typescript
// ❌ 継承の誤用: コードの再利用のための継承
class ArrayList<T> {
  protected items: T[] = [];

  add(item: T): void {
    this.items.push(item);
  }

  get(index: number): T {
    return this.items[index];
  }

  size(): number {
    return this.items.length;
  }

  remove(index: number): T {
    return this.items.splice(index, 1)[0];
  }
}

// Stack は ArrayList ではない（is-a 関係がない）
// Stack は「先頭への追加/削除」のみ、ランダムアクセスは不要
class Stack<T> extends ArrayList<T> {
  push(item: T): void {
    this.add(item);
  }

  pop(): T | undefined {
    if (this.size() === 0) return undefined;
    return this.remove(this.size() - 1);
  }

  peek(): T | undefined {
    if (this.size() === 0) return undefined;
    return this.get(this.size() - 1);
  }

  // 問題: get(), remove() 等が公開されてしまう
  // stack.get(0) や stack.remove(3) が可能 = Stack の契約を破る
}


// ✅ コンポジションで実装
class StackV2<T> {
  private items: T[] = []; // 内部実装を隠蔽

  push(item: T): void {
    this.items.push(item);
  }

  pop(): T | undefined {
    return this.items.pop();
  }

  peek(): T | undefined {
    return this.items[this.items.length - 1];
  }

  size(): number {
    return this.items.length;
  }

  isEmpty(): boolean {
    return this.items.length === 0;
  }

  // get(), remove() は公開しない = Stack の契約を守る
}
```

```python
# 継承 vs コンポジション: ゲームキャラクターの設計

# ❌ 継承: 組み合わせが爆発する
class Character:
    pass

class Warrior(Character):
    def attack(self):
        return "剣で攻撃"

class Mage(Character):
    def cast_spell(self):
        return "魔法を唱える"

class Archer(Character):
    def shoot(self):
        return "弓で攻撃"

# 魔法戦士が欲しい → WarriorMage？ どう継承する？
# 弓を使う魔法使い → MageArcher？
# 全部できるキャラ → WarriorMageArcher？？
# → 組み合わせが 2^n で爆発する


# ✅ コンポジション: 能力を部品として組み立て
from abc import ABC, abstractmethod
from typing import Optional


class Ability(ABC):
    """能力の抽象基底クラス"""
    @abstractmethod
    def use(self, user_name: str) -> str:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...


class SwordSkill(Ability):
    def __init__(self, damage: int = 10):
        self.damage = damage

    def use(self, user_name: str) -> str:
        return f"{user_name}が剣で攻撃！ {self.damage}ダメージ"

    def get_name(self) -> str:
        return "剣術"


class MagicSkill(Ability):
    def __init__(self, mana_cost: int = 5):
        self.mana_cost = mana_cost

    def use(self, user_name: str) -> str:
        return f"{user_name}が魔法を唱えた！ MP-{self.mana_cost}"

    def get_name(self) -> str:
        return "魔法"


class ArcherySkill(Ability):
    def __init__(self, range_bonus: int = 20):
        self.range_bonus = range_bonus

    def use(self, user_name: str) -> str:
        return f"{user_name}が弓で攻撃！ 射程+{self.range_bonus}"

    def get_name(self) -> str:
        return "弓術"


class HealingSkill(Ability):
    def __init__(self, heal_amount: int = 15):
        self.heal_amount = heal_amount

    def use(self, user_name: str) -> str:
        return f"{user_name}が回復！ HP+{self.heal_amount}"

    def get_name(self) -> str:
        return "回復"


class GameCharacter:
    """コンポジションベースのキャラクター"""
    def __init__(self, name: str, hp: int = 100, mp: int = 50):
        self.name = name
        self.hp = hp
        self.mp = mp
        self._abilities: list[Ability] = []

    def add_ability(self, ability: Ability) -> "GameCharacter":
        """メソッドチェーン対応"""
        self._abilities.append(ability)
        return self

    def use_ability(self, index: int) -> str:
        if 0 <= index < len(self._abilities):
            return self._abilities[index].use(self.name)
        return f"能力 {index} は存在しません"

    def list_abilities(self) -> list[str]:
        return [a.get_name() for a in self._abilities]

    def __str__(self) -> str:
        abilities = ", ".join(self.list_abilities())
        return f"{self.name} (HP:{self.hp} MP:{self.mp}) [{abilities}]"


# 自由に組み合わせ可能
warrior = GameCharacter("戦士", hp=150).add_ability(SwordSkill(damage=15))
mage = GameCharacter("魔法使い", mp=100).add_ability(MagicSkill()).add_ability(HealingSkill())
magic_warrior = (GameCharacter("魔法戦士", hp=120, mp=70)
    .add_ability(SwordSkill(damage=12))
    .add_ability(MagicSkill(mana_cost=8)))
all_rounder = (GameCharacter("万能者", hp=100, mp=80)
    .add_ability(SwordSkill())
    .add_ability(MagicSkill())
    .add_ability(ArcherySkill())
    .add_ability(HealingSkill()))

print(all_rounder)
# 万能者 (HP:100 MP:80) [剣術, 魔法, 弓術, 回復]
print(all_rounder.use_ability(1))
# 万能者が魔法を唱えた！ MP-5
```

### 6.2 Delegation（委譲）パターン

```kotlin
// Kotlin: by キーワードによる委譲
interface Printer {
    fun print(content: String)
}

interface Scanner {
    fun scan(): String
}

class LaserPrinter : Printer {
    override fun print(content: String) {
        println("レーザー印刷: $content")
    }
}

class FlatbedScanner : Scanner {
    override fun scan(): String {
        return "スキャンデータ"
    }
}

// Kotlin の by キーワードで委譲を簡潔に記述
class MultiFunctionDevice(
    printer: Printer,
    scanner: Scanner
) : Printer by printer, Scanner by scanner {
    // print() と scan() は自動的に委譲される
    // 必要に応じてオーバーライドも可能

    fun copyDocument() {
        val data = scan()
        print(data)
    }
}

val device = MultiFunctionDevice(LaserPrinter(), FlatbedScanner())
device.print("Hello")    // レーザー印刷: Hello
device.scan()             // スキャンデータ
device.copyDocument()     // スキャン → 印刷
```

---

## 7. 継承のベストプラクティス

### 7.1 継承を使う際のチェックリスト

```
継承を導入する前のチェックリスト:

□ is-a 関係が自然に成立するか？
  「Dog is-a Animal」→ ✓
  「Stack is-a ArrayList」→ ✗

□ リスコフの置換原則を満たすか？
  「親クラスが使われる全ての場所で、サブクラスに置き換えても正しく動くか？」
  → Square extends Rectangle は ✗（setWidth の副作用が異なる）

□ 親クラスの全てのメソッドがサブクラスで意味を持つか？
  → 「ゴリラ・バナナ問題」の回避

□ 継承階層は3レベル以内に収まるか？
  → 3レベル以上 → コンポジションを検討

□ 親クラスは安定しているか？
  → 頻繁に変更される親クラス → 脆い基底クラス問題のリスク

□ テストは容易か？
  → 親クラスの巨大なセットアップが必要 → コンポジション推奨
```

### 7.2 sealed / final による継承の制御

```typescript
// TypeScript: 継承を制限するパターン

// 意図的に継承を禁止する（TypeScript には final がないため慣習的に）
class Configuration {
  // @final のようなデコレータは存在しないが、コメントで意図を示す
  /** @final このクラスは継承しないでください */
  private constructor(private readonly settings: Map<string, string>) {}

  static create(settings: Record<string, string>): Configuration {
    return new Configuration(new Map(Object.entries(settings)));
  }

  get(key: string): string | undefined {
    return this.settings.get(key);
  }
}
```

```java
// Java: final と sealed による継承制御

// final: 継承を完全に禁止
public final class ImmutablePoint {
    private final double x;
    private final double y;

    public ImmutablePoint(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() { return x; }
    public double getY() { return y; }

    public double distanceTo(ImmutablePoint other) {
        return Math.sqrt(
            Math.pow(this.x - other.x, 2) +
            Math.pow(this.y - other.y, 2)
        );
    }
}
// class ExtendedPoint extends ImmutablePoint {} // コンパイルエラー！

// sealed（Java 17+）: 許可されたクラスのみ継承可能
public sealed class Shape permits Circle, Rectangle, Triangle {
    public abstract double area();
}

public final class Circle extends Shape {
    private final double radius;
    public Circle(double radius) { this.radius = radius; }
    public double area() { return Math.PI * radius * radius; }
}

public final class Rectangle extends Shape {
    private final double width, height;
    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }
    public double area() { return width * height; }
}

public final class Triangle extends Shape {
    private final double base, height;
    public Triangle(double base, double height) {
        this.base = base;
        this.height = height;
    }
    public double area() { return 0.5 * base * height; }
}

// パターンマッチング（Java 21+）で安全に分岐
public String describeShape(Shape shape) {
    return switch (shape) {
        case Circle c    -> "円（半径: " + c.getRadius() + "）";
        case Rectangle r -> "長方形（" + r.getWidth() + " × " + r.getHeight() + "）";
        case Triangle t  -> "三角形（底辺: " + t.getBase() + "）";
        // sealed なので全パターンを網羅 → default 不要
    };
}
```

```kotlin
// Kotlin: sealed class（ADT: 代数的データ型の実現）
sealed class Result<out T> {
    data class Success<T>(val value: T) : Result<T>()
    data class Failure(val error: Throwable) : Result<Nothing>()
    data object Loading : Result<Nothing>()
}

fun <T> handleResult(result: Result<T>) {
    when (result) {
        is Result.Success -> println("成功: ${result.value}")
        is Result.Failure -> println("失敗: ${result.error.message}")
        is Result.Loading -> println("読み込み中...")
        // sealed なので全パターン網羅 → else 不要
    }
}

// 使用例
val result: Result<String> = Result.Success("データ取得完了")
handleResult(result) // 成功: データ取得完了
```

---

## 8. 実践ケーススタディ: Webフレームワークでの継承

```python
# Django のクラスベースビュー: フレームワークが求める継承
from django.views import View
from django.views.generic import ListView, CreateView, DetailView
from django.http import JsonResponse


# パターン1: 基本的な View の継承
class HealthCheckView(View):
    """ヘルスチェックAPI"""

    def get(self, request):
        return JsonResponse({
            "status": "healthy",
            "version": "1.0.0",
        })


# パターン2: ジェネリックビューの活用
class ArticleListView(ListView):
    """記事一覧（テンプレートメソッドパターンの活用）"""
    model = Article               # テンプレート変数
    template_name = "articles/list.html"
    paginate_by = 20
    ordering = ["-created_at"]

    def get_queryset(self):
        """フック: クエリセットをカスタマイズ"""
        qs = super().get_queryset()
        category = self.request.GET.get("category")
        if category:
            qs = qs.filter(category=category)
        return qs

    def get_context_data(self, **kwargs):
        """フック: テンプレートコンテキストを追加"""
        context = super().get_context_data(**kwargs)
        context["categories"] = Category.objects.all()
        return context


# パターン3: Mixin の活用
class LoginRequiredMixin:
    """ログイン必須Mixin"""
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "認証が必要です"}, status=401)
        return super().dispatch(request, *args, **kwargs)


class RateLimitMixin:
    """レートリミットMixin"""
    rate_limit = 100  # リクエスト/分

    def dispatch(self, request, *args, **kwargs):
        # レートリミットチェック（簡略化）
        client_ip = request.META.get("REMOTE_ADDR")
        # if is_rate_limited(client_ip, self.rate_limit):
        #     return JsonResponse({"error": "Too many requests"}, status=429)
        return super().dispatch(request, *args, **kwargs)


class ProtectedArticleCreateView(LoginRequiredMixin, RateLimitMixin, CreateView):
    """保護された記事作成ビュー（複数のMixinを組み合わせ）"""
    model = Article
    fields = ["title", "content", "category"]
    template_name = "articles/create.html"
    success_url = "/articles/"

    # MRO: ProtectedArticleCreateView → LoginRequiredMixin
    #   → RateLimitMixin → CreateView → ...
    # dispatch() の呼び出し順:
    # 1. LoginRequiredMixin.dispatch() → ログインチェック
    # 2. RateLimitMixin.dispatch() → レートリミットチェック
    # 3. CreateView.dispatch() → 実際のリクエスト処理
```

```typescript
// React コンポーネント: クラスベース → 関数ベースへの移行

// 古い: クラスベースコンポーネント（継承ベース）
class UserProfile extends React.Component<UserProfileProps, UserProfileState> {
  constructor(props: UserProfileProps) {
    super(props);
    this.state = { user: null, loading: true };
  }

  async componentDidMount() {
    const user = await fetchUser(this.props.userId);
    this.setState({ user, loading: false });
  }

  componentDidUpdate(prevProps: UserProfileProps) {
    if (prevProps.userId !== this.props.userId) {
      this.setState({ loading: true });
      fetchUser(this.props.userId).then(user => {
        this.setState({ user, loading: false });
      });
    }
  }

  render() {
    if (this.state.loading) return <div>Loading...</div>;
    return <div>{this.state.user?.name}</div>;
  }
}

// モダン: 関数コンポーネント（コンポジションベース）
function UserProfileV2({ userId }: { userId: string }) {
  // カスタムフック = コンポジションによるロジックの再利用
  const { data: user, loading, error } = useUser(userId);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  return <div>{user?.name}</div>;
}

// カスタムフック: 継承なしでロジックを再利用
function useUser(userId: string) {
  const [data, setData] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchUser(userId)
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [userId]);

  return { data, loading, error };
}
// → 継承なしで、どのコンポーネントでも useUser() を呼ぶだけで再利用可能
```

---

## 9. 言語別・継承機能の比較表

```
┌─────────────┬──────┬──────┬────────┬────────┬────────┬───────┐
│ 機能         │ Java │ C#   │ Python │ C++    │ Kotlin │ Rust  │
├─────────────┼──────┼──────┼────────┼────────┼────────┼───────┤
│ 単一継承     │  ✓   │  ✓   │  ✓     │  ✓     │  ✓     │  ✗*  │
│ 多重継承     │  ✗   │  ✗   │  ✓     │  ✓     │  ✗     │  ✗   │
│ Interface    │  ✓   │  ✓   │ ABC    │ 純粋仮想│  ✓     │ trait │
│ default実装  │  ✓   │  ✓   │  -     │  -     │  ✓     │  ✓   │
│ Mixin        │  ✗   │  ✗   │  ✓     │  ✗     │  ✗     │  ✗   │
│ abstract     │  ✓   │  ✓   │ ABC    │  ✓     │  ✓     │  -   │
│ final class  │  ✓   │sealed│  -     │  -     │ 既定   │  -   │
│ sealed class │  ✓*  │  ✓   │  -     │  -     │  ✓     │  -   │
│ 委譲(by)     │  ✗   │  ✗   │  ✗     │  ✗     │  ✓     │  ✗   │
│ 仮想デフォルト│  ✗   │ ✓(virtual) │ ✓ │ ✓(virtual)│ ✓(open)│  -   │
└─────────────┴──────┴──────┴────────┴────────┴────────┴───────┘

* Rust は継承を持たず、trait による型合成が基本
* Java の sealed は 17+
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 継承 | 親の機能を引き継いで拡張 |
| オーバーライド | 親メソッドの再定義（3パターン: 完全置換/拡張/条件分岐） |
| 多重継承 | ダイヤモンド問題に注意。MRO/Interface/Mixin で解決 |
| 抽象クラス | 共通実装 + 実装義務。テンプレートメソッドパターンの基盤 |
| コンポジション | has-a 関係。継承より柔軟で安全 |
| 原則 | 「継承よりコンポジション」。is-a 関係が明確な場合のみ継承 |
| sealed/final | 継承を制限して安全性を高める |
| テスト | コンポジションの方がモック差し替えが容易 |

---

## 次に読むべきガイド
→ [[02-polymorphism.md]] -- ポリモーフィズム

---

## 参考文献
1. Bloch, J. "Effective Java." Item 18: Favor composition over inheritance. 2018.
2. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
3. Martin, R. "Clean Architecture." Prentice Hall, 2017.
4. Sandi Metz. "Practical Object-Oriented Design in Ruby." 2nd edition, 2018.
5. Joshua Kerievsky. "Refactoring to Patterns." Addison-Wesley, 2004.
6. Eric Freeman et al. "Head First Design Patterns." O'Reilly, 2nd edition, 2020.
