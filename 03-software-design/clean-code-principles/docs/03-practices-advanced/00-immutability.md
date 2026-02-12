# イミュータビリティ（不変性）の原則

> なぜ不変データが安全なコードを生むのか。言語別の実装パターン、パフォーマンスへの影響、マルチスレッド環境での恩恵まで、イミュータビリティの理論と実践を網羅する。

---

## この章で学ぶこと

1. **イミュータビリティの理論的根拠**を理解し、なぜ不変データが安全で予測可能なコードを生むか説明できる
2. **言語別の不変性実装パターン**（Java、TypeScript、Python、Rust、Kotlin）を習得し、実務に適用できる
3. **パフォーマンスとのトレードオフ**を理解し、構造共有やコピーオンライトなどの最適化手法を使い分けられる
4. **不変データとアーキテクチャの関係**を把握し、イベントソーシング・CQRS・React状態管理に応用できる
5. **段階的導入戦略**を立案し、既存チーム・既存コードベースにイミュータビリティを実践的に導入できる

---

## 前提知識

このガイドを理解するには、以下の知識が必要です。

| 前提知識 | 参照先 |
|---------|-------|
| 変数、参照、値渡し/参照渡しの基礎 | [01-practices/00-naming-conventions.md](../01-practices/00-naming-conventions.md) |
| オブジェクト指向の基本（クラス、インスタンス） | [00-principles/02-solid.md](../00-principles/02-solid.md) |
| 基本的なデータ構造（配列、辞書、ツリー） | `01-cs-fundamentals/data-structures-algorithms` |
| 関数型プログラミングの基礎概念 | [02-functional-principles.md](./02-functional-principles.md) |
| マルチスレッドの基本概念（スレッド、ロック） | `01-cs-fundamentals/operating-systems` |

---

## 1. イミュータビリティとは何か

### 1.1 定義と基本概念

イミュータビリティ（Immutability、不変性）とは、一度生成されたデータが以降変更されないという性質のことである。オブジェクトの状態を変えたい場合は、既存のオブジェクトを変更するのではなく、変更後の値を持つ新しいオブジェクトを生成する。

この概念は数学における変数と対応する。数学において x = 5 と定義したら、x は常に 5 である。プログラミングにおける「変数」は、名前に反して多くの言語で再代入可能であり、これがバグの温床となる。

### 1.2 ミュータブル vs イミュータブル

```
ミュータブル（可変）                 イミュータブル（不変）
─────────────────                ─────────────────

  user.name = "田中"               newUser = user.copy(name="田中")
       │                                │
       v                                v
  ┌──────────┐                    ┌──────────┐  ┌──────────┐
  │ user     │                    │ user     │  │ newUser  │
  │ name:"田中"│  ← 元が変わる     │ name:"鈴木"│  │ name:"田中"│
  │ age: 30  │                    │ age: 30  │  │ age: 30  │
  └──────────┘                    └──────────┘  └──────────┘
                                   元は不変        新しいコピー

  問題: 誰がいつ変えた？           利点: 変更履歴が明確
  共有参照で予期せぬ変更           共有しても安全
```

この違いは些細に見えるが、システムの複雑さが増すにつれて決定的な差をもたらす。ミュータブルなデータを複数のコンポーネントが共有すると、ある箇所での変更が予期せぬ形で他の箇所に波及する。いわゆる「幽霊のようなバグ」が発生し、再現困難なデバッグに膨大な時間を費やすことになる。

### 1.3 不変性がもたらす利点

```
┌───────────────────────────────────────────────────┐
│            イミュータビリティの5つの利点              │
├───────────────────────────────────────────────────┤
│                                                   │
│  1. 予測可能性                                     │
│     値が変わらない → 関数の結果が常に同じ            │
│                                                   │
│  2. スレッド安全性                                  │
│     変更がない → ロック不要 → デッドロックなし       │
│                                                   │
│  3. デバッグ容易性                                  │
│     状態変化がない → 問題の再現が容易               │
│                                                   │
│  4. 変更検知の効率化                                │
│     参照比較だけで変更判定 → O(1)                   │
│                                                   │
│  5. 履歴管理（Undo/Redo）                          │
│     古い状態がそのまま残る → タイムトラベル可能      │
│                                                   │
└───────────────────────────────────────────────────┘
```

### 1.4 不変性のレベル分類

不変性には複数のレベルが存在する。これを理解しないと、「不変にしたつもり」のコードでバグが発生する。

```
レベル1: 変数の不変性 (Variable Immutability)
─────────────────────────────────────────────
  const x = 5;        // xに再代入できない
  const obj = {a: 1}; // objに再代入できない
  obj.a = 2;          // しかしプロパティは変更できる！

レベル2: オブジェクトの浅い不変性 (Shallow Immutability)
──────────────────────────────────────────────────────
  Object.freeze(obj);  // 直下のプロパティを変更不可
  obj.a = 2;           // 静かに無視される（strictモードではエラー）
  obj.nested.b = 3;    // ネストされたオブジェクトは変更可能！

レベル3: オブジェクトの深い不変性 (Deep Immutability)
───────────────────────────────────────────────────
  deepFreeze(obj);     // 全階層のプロパティが変更不可
  // または
  type DeepReadonly<T> // TypeScriptの型レベルで保証

レベル4: 言語レベルの不変性 (Language-level Immutability)
────────────────────────────────────────────────────────
  Rust: let x = 5;    // デフォルトで不変
  Haskell: 全てが不変  // 可変性は型で明示（IORef, STRef）
```

### 1.5 不変性の適用スペクトラム

```
完全ミュータブル ◄──────────────────────────────► 完全イミュータブル

  C言語          Java        TypeScript      Scala         Haskell
  (全て可変)     (finalあり)  (readonlyあり)  (valデフォルト) (全て不変)

                    Rust
                    (letデフォルト不変)

                    Kotlin
                    (val/varで明示)

推奨ゾーン:
  ┌──────────────────────────────┐
  │  デフォルト不変 + 必要な箇所のみ可変  │
  │  (Rust/Kotlinのアプローチ)          │
  └──────────────────────────────┘
```

---

## 2. 言語別イミュータビリティ実装

### 2.1 TypeScript / JavaScript

```typescript
// TypeScript: イミュータブルなデータ操作

// === 1. readonly と as const ===

interface User {
  readonly id: string;
  readonly name: string;
  readonly age: number;
  readonly address: Readonly<Address>;
}

interface Address {
  readonly prefecture: string;
  readonly city: string;
}

// as const で深い不変性
const CONFIG = {
  api: {
    baseUrl: "https://api.example.com",
    timeout: 5000,
  },
  features: ["auth", "logging"] as const,
} as const;

// CONFIG.api.baseUrl = "xxx"; // コンパイルエラー
// CONFIG.features.push("x");  // コンパイルエラー

// === 2. Readonly ユーティリティ型 ===

// 浅いReadonly（1階層のみ）
type ShallowReadonlyUser = Readonly<{
  name: string;
  address: { city: string };
}>;
// address.city は変更可能（浅い）

// 深いReadonly（再帰的に全階層）
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object
    ? T[P] extends Function
      ? T[P]
      : DeepReadonly<T[P]>
    : T[P];
};

type FullyReadonlyUser = DeepReadonly<{
  name: string;
  address: { city: string; tags: string[] };
}>;
// address.city も tags も全て変更不可

// === 3. 不変な更新パターン ===

function updateUserName(user: User, newName: string): User {
  // スプレッド構文で新しいオブジェクトを生成
  return { ...user, name: newName };
}

// ネストされたオブジェクトの更新
function updateCity(user: User, newCity: string): User {
  return {
    ...user,
    address: {
      ...user.address,
      city: newCity,
    },
  };
}

// === 4. 配列の不変操作 ===

function addItem<T>(items: readonly T[], item: T): readonly T[] {
  return [...items, item]; // 新しい配列を返す
}

function removeItem<T>(items: readonly T[], index: number): readonly T[] {
  return [...items.slice(0, index), ...items.slice(index + 1)];
}

function updateItem<T>(
  items: readonly T[],
  index: number,
  updater: (item: T) => T
): readonly T[] {
  return items.map((item, i) => (i === index ? updater(item) : item));
}

// 配列の不変ソート
function sortedBy<T, K>(
  items: readonly T[],
  keyFn: (item: T) => K
): readonly T[] {
  return [...items].sort((a, b) => {
    const ka = keyFn(a);
    const kb = keyFn(b);
    return ka < kb ? -1 : ka > kb ? 1 : 0;
  });
}

// === 5. Map/Set の不変操作 ===

function addToMap<K, V>(
  map: ReadonlyMap<K, V>,
  key: K,
  value: V
): ReadonlyMap<K, V> {
  const newMap = new Map(map);
  newMap.set(key, value);
  return newMap;
}

function removeFromMap<K, V>(
  map: ReadonlyMap<K, V>,
  key: K
): ReadonlyMap<K, V> {
  const newMap = new Map(map);
  newMap.delete(key);
  return newMap;
}

function addToSet<T>(set: ReadonlySet<T>, value: T): ReadonlySet<T> {
  const newSet = new Set(set);
  newSet.add(value);
  return newSet;
}

// === 6. Object.freeze による実行時の不変性保証 ===

function deepFreeze<T extends object>(obj: T): Readonly<T> {
  Object.freeze(obj);
  Object.getOwnPropertyNames(obj).forEach((prop) => {
    const value = (obj as any)[prop];
    if (value && typeof value === "object" && !Object.isFrozen(value)) {
      deepFreeze(value);
    }
  });
  return obj;
}

const frozenConfig = deepFreeze({
  api: { url: "https://example.com" },
  retries: 3,
});
// frozenConfig.api.url = "xxx"; // 実行時にTypeErrorが発生（strictモード）
```

### 2.2 Java

```java
// Java: イミュータブルクラスの設計パターン

// === 1. 基本的なイミュータブルクラス ===
// Effective Java Item 17: Minimize mutability のルール
// (1) セッターを提供しない
// (2) クラスをfinalにする（継承禁止）
// (3) 全フィールドをfinalにする
// (4) 全フィールドをprivateにする
// (5) 可変コンポーネントへの排他的アクセスを保証する

public final class Money {
    private final int amount;
    private final String currency;

    public Money(int amount, String currency) {
        if (amount < 0) {
            throw new IllegalArgumentException("金額は0以上: " + amount);
        }
        this.amount = amount;
        this.currency = Objects.requireNonNull(currency, "通貨は必須");
    }

    public int getAmount() { return amount; }
    public String getCurrency() { return currency; }

    // 変更は新しいインスタンスを返す
    public Money add(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException(
                "通貨が異なります: " + this.currency + " vs " + other.currency
            );
        }
        return new Money(this.amount + other.amount, this.currency);
    }

    public Money multiply(int factor) {
        return new Money(this.amount * factor, this.currency);
    }

    public Money subtract(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException("通貨が異なります");
        }
        return new Money(this.amount - other.amount, this.currency);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Money money)) return false;
        return amount == money.amount && currency.equals(money.currency);
    }

    @Override
    public int hashCode() {
        return Objects.hash(amount, currency);
    }

    @Override
    public String toString() {
        return amount + " " + currency;
    }
}

// === 2. Java 16+ Record（自動的にイミュータブル） ===

public record User(
    String id,
    String name,
    int age,
    Address address
) {
    // コンパクトコンストラクタでバリデーション
    public User {
        Objects.requireNonNull(id, "IDは必須です");
        Objects.requireNonNull(name, "名前は必須です");
        if (age < 0 || age > 200) {
            throw new IllegalArgumentException("年齢は0〜200の範囲: " + age);
        }
    }

    // Wither パターン：フィールドを変更した新しいインスタンスを返す
    public User withName(String newName) {
        return new User(id, newName, age, address);
    }

    public User withAge(int newAge) {
        return new User(id, name, newAge, address);
    }

    public User withAddress(Address newAddress) {
        return new User(id, name, age, newAddress);
    }
}

public record Address(String prefecture, String city, String zipCode) {
    public Address {
        Objects.requireNonNull(prefecture);
        Objects.requireNonNull(city);
    }

    public Address withCity(String newCity) {
        return new Address(prefecture, newCity, zipCode);
    }
}

// === 3. 不変コレクション ===

// Java 9+ ファクトリメソッド
var immutableList = List.of("a", "b", "c");
var immutableMap = Map.of("key1", "value1", "key2", "value2");
var immutableSet = Set.of(1, 2, 3);
// immutableList.add("d"); // UnsupportedOperationException

// Java 10+ コピーファクトリ
var mutableList = new ArrayList<>(List.of("a", "b"));
var snapshot = List.copyOf(mutableList); // 不変のスナップショット
mutableList.add("c"); // 元は変更可能
// snapshot は変わらない

// Collections.unmodifiableList との違い
var original = new ArrayList<>(List.of("a", "b"));
var unmodifiable = Collections.unmodifiableList(original);
original.add("c");
// unmodifiable もサイズが3に！（ビューなので元が変わると影響を受ける）

var copied = List.copyOf(original);
original.add("d");
// copied はサイズ3のまま（コピーなので元の変更に影響されない）

// === 4. 可変コンポーネントの防御的コピー ===

public final class Period {
    private final Date start;
    private final Date end;

    public Period(Date start, Date end) {
        // 防御的コピー: 呼び出し元のDateオブジェクトの変更から守る
        this.start = new Date(start.getTime());
        this.end = new Date(end.getTime());
        if (this.start.compareTo(this.end) > 0) {
            throw new IllegalArgumentException("start > end");
        }
    }

    public Date getStart() {
        return new Date(start.getTime()); // 防御的コピーを返す
    }

    public Date getEnd() {
        return new Date(end.getTime());
    }
}
// 注意: Java 8+ では Date の代わりに不変な LocalDate/LocalDateTime を使うべき
```

### 2.3 Python

```python
# Python: イミュータビリティの実装パターン

# === 1. frozen dataclass ===
from dataclasses import dataclass, replace, field
from typing import Tuple

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def translate(self, dx: float, dy: float) -> "Point":
        """新しいPointを返す（元は変更しない）"""
        return replace(self, x=self.x + dx, y=self.y + dy)

    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def scale(self, factor: float) -> "Point":
        return Point(self.x * factor, self.y * factor)

p1 = Point(1.0, 2.0)
p2 = p1.translate(3.0, 4.0)
print(p1)  # Point(x=1.0, y=2.0)  ← 元は変わらない
print(p2)  # Point(x=4.0, y=6.0)
# p1.x = 5.0  # FrozenInstanceError

# frozen dataclass でコレクションを持つ場合
@dataclass(frozen=True)
class Polygon:
    name: str
    vertices: Tuple[Point, ...]  # tupleは不変

    def add_vertex(self, point: Point) -> "Polygon":
        return replace(self, vertices=self.vertices + (point,))

    def vertex_count(self) -> int:
        return len(self.vertices)

triangle = Polygon("三角形", (Point(0, 0), Point(1, 0), Point(0.5, 1)))
# triangle.vertices = ()  # FrozenInstanceError

# === 2. NamedTuple（軽量なイミュータブル型） ===
from typing import NamedTuple

class Color(NamedTuple):
    r: int
    g: int
    b: int
    a: float = 1.0

    def with_alpha(self, alpha: float) -> "Color":
        return self._replace(a=alpha)

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def blend(self, other: "Color", ratio: float = 0.5) -> "Color":
        """2色を混合して新しい色を返す"""
        return Color(
            r=int(self.r * (1 - ratio) + other.r * ratio),
            g=int(self.g * (1 - ratio) + other.g * ratio),
            b=int(self.b * (1 - ratio) + other.b * ratio),
            a=self.a * (1 - ratio) + other.a * ratio,
        )

red = Color(255, 0, 0)
semi_transparent = red.with_alpha(0.5)
print(red)                # Color(r=255, g=0, b=0, a=1.0)
print(semi_transparent)   # Color(r=255, g=0, b=0, a=0.5)

# NamedTuple はハッシュ可能（dictのキーやsetの要素に使える）
color_names = {Color(255, 0, 0): "赤", Color(0, 255, 0): "緑"}

# === 3. 不変辞書パターン ===
from types import MappingProxyType

def create_config(overrides: dict = None) -> MappingProxyType:
    """変更不可な設定辞書を作成"""
    defaults = {
        "debug": False,
        "log_level": "INFO",
        "max_retries": 3,
        "timeout_seconds": 30,
    }
    if overrides:
        defaults.update(overrides)
    return MappingProxyType(defaults)

config = create_config({"debug": True})
print(config["debug"])    # True
# config["debug"] = False  # TypeError: 'mappingproxy' object does not support item assignment

# 不変な設定を更新するには新しいMappingProxyTypeを作る
def update_config(
    config: MappingProxyType, **updates
) -> MappingProxyType:
    new_dict = dict(config)
    new_dict.update(updates)
    return MappingProxyType(new_dict)

new_config = update_config(config, debug=False, log_level="DEBUG")

# === 4. Pydantic v2のイミュータブルモデル ===
from pydantic import BaseModel, ConfigDict, field_validator

class UserModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    email: str
    tags: tuple[str, ...] = ()  # listではなくtupleで不変性を保証

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("有効なメールアドレスを指定してください")
        return v

    def update_name(self, new_name: str) -> "UserModel":
        return self.model_copy(update={"name": new_name})

    def add_tag(self, tag: str) -> "UserModel":
        return self.model_copy(update={"tags": self.tags + (tag,)})

user = UserModel(id="1", name="田中", email="tanaka@example.com")
updated = user.update_name("鈴木")
print(user.name)     # "田中"  ← 元は変わらない
print(updated.name)  # "鈴木"

# === 5. __slots__ + frozen による最適化 ===

@dataclass(frozen=True, slots=True)
class OptimizedPoint:
    """slots=True でメモリ効率が向上（__dict__を持たない）"""
    x: float
    y: float
    z: float = 0.0

# slots=True により:
# - メモリ使用量が約40%削減
# - 属性アクセスが約10%高速化
# - __dict__ がないため動的な属性追加が不可（不変性をさらに強化）

# === 6. イミュータブルなEnum ===
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

    def can_transition_to(self, target: "OrderStatus") -> bool:
        """状態遷移の妥当性を検証"""
        valid_transitions = {
            OrderStatus.PENDING: {OrderStatus.CONFIRMED, OrderStatus.CANCELLED},
            OrderStatus.CONFIRMED: {OrderStatus.SHIPPED, OrderStatus.CANCELLED},
            OrderStatus.SHIPPED: {OrderStatus.DELIVERED},
            OrderStatus.DELIVERED: set(),
            OrderStatus.CANCELLED: set(),
        }
        return target in valid_transitions.get(self, set())
```

### 2.4 Rust（言語レベルでの不変性）

```rust
// Rust: デフォルトがイミュータブル
// Rustは「不変性がデフォルト」という設計哲学を持つ唯一の主流言語

// === 1. 変数はデフォルトで不変 ===
fn basic_immutability() {
    let x = 5;
    // x = 6;  // コンパイルエラー: cannot assign twice to immutable variable

    let mut y = 5;  // mut を明示的に付ける
    y = 6;          // OK

    // シャドーイング: 再定義（再代入ではない）
    let x = x + 1;  // 新しいxが古いxを隠す
    let x = x * 2;  // さらに新しいx
    // 型の変更もシャドーイングなら可能
    let x = x.to_string(); // i32 → String
}

// === 2. 構造体の不変性 ===
#[derive(Clone, Debug, PartialEq)]
struct User {
    name: String,
    age: u32,
    email: String,
}

impl User {
    fn new(name: String, age: u32, email: String) -> Self {
        Self { name, age, email }
    }

    // Builder パターンで不変更新（所有権を消費して新しいインスタンスを返す）
    fn with_name(self, name: String) -> Self {
        Self { name, ..self }
    }

    fn with_age(self, age: u32) -> Self {
        Self { age, ..self }
    }

    fn with_email(self, email: String) -> Self {
        Self { email, ..self }
    }
}

fn usage() {
    let user = User::new("tanaka".into(), 30, "tanaka@example.com".into());
    let updated = user.with_name("suzuki".into()); // userの所有権が移動
    // println!("{:?}", user); // コンパイルエラー: value moved
    println!("{:?}", updated); // OK
}

// === 3. 所有権と借用による不変性の保証 ===
fn print_user(user: &User) {       // 共有参照（読み取りのみ）
    println!("{:?}", user);
    // user.age += 1; // コンパイルエラー: cannot assign to immutable borrow
}

fn update_age(user: &mut User) {   // 排他参照（変更可能）
    user.age += 1;
}

// 借用のルール:
// - 共有参照(&T)は同時に何個でも存在できる
// - 排他参照(&mut T)は同時に1個だけ
// - 共有参照と排他参照は同時に存在できない
// → コンパイル時にデータ競合を完全に防止

// === 4. 不変コレクション ===
fn immutable_collections() {
    let numbers = vec![1, 2, 3, 4, 5]; // 不変（mutなし）
    // numbers.push(6); // コンパイルエラー

    // 新しいコレクションを生成する関数型操作
    let doubled: Vec<i32> = numbers.iter().map(|x| x * 2).collect();
    let even: Vec<&i32> = numbers.iter().filter(|x| *x % 2 == 0).collect();

    // numbers はそのまま使える（iterは借用のみ）
    println!("{:?}", numbers);  // [1, 2, 3, 4, 5]
    println!("{:?}", doubled);  // [2, 4, 6, 8, 10]
}

// === 5. Arc<T> によるスレッド安全な不変データ共有 ===
use std::sync::Arc;
use std::thread;

fn shared_immutable_data() {
    let data = Arc::new(vec![1, 2, 3, 4, 5]); // 不変データを共有

    let handles: Vec<_> = (0..4).map(|i| {
        let data = Arc::clone(&data); // 参照カウントを増やすだけ
        thread::spawn(move || {
            let sum: i32 = data.iter().sum();
            println!("Thread {}: sum = {}", i, sum);
        })
    }).collect();

    for handle in handles {
        handle.join().unwrap();
    }
    // ロック不要！不変データなのでデータ競合が起きない
}
```

### 2.5 Kotlin

```kotlin
// Kotlin: val/var で不変性を明示

// === 1. 基本的な不変性 ===
val x = 5        // 不変（再代入不可）
// x = 6         // コンパイルエラー
var y = 5        // 可変（再代入可能）
y = 6            // OK

// === 2. data class（不変設計推奨） ===
data class User(
    val id: String,          // valで不変
    val name: String,
    val age: Int,
    val email: String
) {
    init {
        require(age in 0..200) { "年齢は0〜200の範囲: $age" }
        require(email.contains("@")) { "メールアドレスが不正: $email" }
    }
}

// copy メソッドで部分更新（自動生成）
val user = User("1", "田中", 30, "tanaka@example.com")
val updated = user.copy(name = "鈴木")
println(user)    // User(id=1, name=田中, age=30, email=tanaka@example.com)
println(updated) // User(id=1, name=鈴木, age=30, email=tanaka@example.com)

// === 3. 不変コレクション ===
val immutableList = listOf(1, 2, 3)     // List<Int> （不変）
val mutableList = mutableListOf(1, 2, 3) // MutableList<Int>（可変）

// immutableList.add(4)  // コンパイルエラー: Listにはaddがない
mutableList.add(4)        // OK

// 不変リストの「更新」: 新しいリストを返す
val newList = immutableList + 4         // [1, 2, 3, 4]
val filtered = immutableList.filter { it > 1 } // [2, 3]
val mapped = immutableList.map { it * 2 }      // [2, 4, 6]

// 不変Map
val config = mapOf(
    "debug" to false,
    "timeout" to 30,
    "retries" to 3
)
// config["debug"] = true  // コンパイルエラー
val newConfig = config + ("debug" to true) // 新しいMapを生成

// === 4. sealed class + data class で不変なドメインモデル ===
sealed class PaymentResult {
    data class Success(
        val transactionId: String,
        val amount: Int,
        val timestamp: Long
    ) : PaymentResult()

    data class Failure(
        val errorCode: String,
        val message: String
    ) : PaymentResult()

    data class Pending(
        val estimatedCompletion: Long
    ) : PaymentResult()
}

// パターンマッチ（when式）で安全に処理
fun handlePayment(result: PaymentResult): String = when (result) {
    is PaymentResult.Success -> "決済成功: ${result.transactionId}"
    is PaymentResult.Failure -> "決済失敗: ${result.message}"
    is PaymentResult.Pending -> "処理中..."
    // sealed class なので全ケースを網羅しないとコンパイル警告
}

// === 5. 不変な値オブジェクト ===
@JvmInline
value class Email(val value: String) {
    init {
        require(value.contains("@")) { "無効なメールアドレス: $value" }
    }
}

@JvmInline
value class UserId(val value: String) {
    init {
        require(value.isNotBlank()) { "IDは空にできません" }
    }
}

// value class はラッパーのオーバーヘッドがゼロ（コンパイル時に展開）
val email = Email("user@example.com")
val userId = UserId("user-123")
```

---

## 3. パフォーマンスと最適化

### 3.1 構造共有（Structural Sharing）

```
構造共有: 変更されていない部分を共有する

  元のツリー          nameを変更後のツリー
  ──────────          ─────────────────

      root                  newRoot
     /    \                /    \
    A      B            newA     B  ← 共有（コピーなし）
   / \    / \           / \    / \
  a1  a2 b1  b2      a1* a2 b1  b2  ← B以下は全て共有
                       ↑
                   変更された部分のみ新規作成

  メモリ効率: O(log n) の新規ノードで済む
  変更検知: ルートの参照が異なれば変更あり → O(1)
```

構造共有は永続データ構造（Persistent Data Structure）の核心技術である。名前に反して「永続化（ディスク保存）」とは無関係で、「変更前後の全バージョンが保持される」という意味である。Clojure、Scala、Haskellなどの関数型言語で広く使われている。

### 3.2 永続データ構造の内部実装

```
Hash Array Mapped Trie (HAMT):
Clojure/ScalaのPersistentVectorやPersistentHashMapの内部構造

  32分岐のトライ木
  ─────────────

              root
       /    |    |    \
     [0-31] [32-63] ... [992-1023]
      / | \
    [0] [1] ... [31]

  配列サイズ: 1024要素
  木の深さ: log32(1024) = 2段
  1要素の変更に必要な新規ノード: 2個（ルート + 1リーフ）

  操作の計算量:
  ┌──────────────┬──────────────────────────────┐
  │ 操作          │ 計算量                        │
  ├──────────────┼──────────────────────────────┤
  │ 参照（get）   │ O(log32 n) ≈ 実質O(1)         │
  │ 更新（set）   │ O(log32 n) ≈ 実質O(1)         │
  │ 追加（append） │ 償却 O(1)                     │
  │ 変更検知      │ O(1) （参照比較）              │
  └──────────────┴──────────────────────────────┘

  log32(100万) ≈ 4 → 100万要素でも4回のポインタ追跡で到達
```

### 3.3 コピーオンライトとの比較

| 戦略 | メモリ効率 | CPU効率 | 実装複雑度 | 適用場面 |
|------|-----------|---------|-----------|---------|
| 毎回フルコピー | 低 | 低 | 簡単 | 小さなデータ（<100要素） |
| 構造共有 | 高 | 高 | 複雑 | 永続データ構造（Clojure等） |
| コピーオンライト | 高 | 中 | 中 | OS/ランタイムレベル（Swift Array） |
| Immer (JS) | 中 | 中 | 簡単 | Reactアプリの状態管理 |
| 永続データ構造 | 高 | 高 | 非常に複雑 | 関数型言語のコア |
| Freeze + 新規生成 | 低 | 低 | 簡単 | 小規模な設定データ |

### 3.4 パフォーマンスベンチマーク指標

```
ベンチマーク: 10,000要素の配列操作（Node.js v20）

操作              | ミュータブル | スプレッド  | Immer    | Immutable.js
─────────────────|───────────|──────────|─────────|───────────
1要素の追加        | 0.001ms   | 0.15ms   | 0.03ms  | 0.005ms
1要素の更新        | 0.001ms   | 0.12ms   | 0.02ms  | 0.008ms
深いネストの更新    | 0.001ms   | 0.30ms   | 0.05ms  | 0.010ms
変更検知          | O(n)深い比較| O(1)参照  | O(1)参照 | O(1)参照
メモリ使用量       | 1x        | ~2x      | ~1.2x   | ~1.1x

結論:
- 個々の操作はミュータブルが最速だが、差は微小
- 変更検知まで含めるとイミュータブルが有利
- 100要素以下ではスプレッドで十分
- 1000要素以上ではImmerやImmutable.jsを検討
```

### 3.5 Immer.jsによる効率的な不変更新

```typescript
// Immer: ミュータブルな記法でイミュータブルな更新
import { produce, Draft } from "immer";

interface AppState {
  users: User[];
  selectedId: string | null;
  filters: {
    status: string;
    search: string;
  };
}

// produce で「ドラフト」に直接変更を加える
// → Immerが自動的に不変な新しい状態を生成
const nextState = produce(currentState, (draft: Draft<AppState>) => {
  const user = draft.users.find((u) => u.id === "123");
  if (user) {
    user.name = "新しい名前"; // 直接変更OK（ドラフト上）
  }
  draft.filters.search = "検索ワード";
});

// currentState は変更されていない（不変）
// nextState は新しいオブジェクト
console.log(currentState === nextState); // false
console.log(currentState.users[1] === nextState.users[1]); // true（変更なし→共有）

// === Immerの内部メカニズム ===
// 1. Proxyオブジェクトでドラフトをラップ
// 2. プロパティアクセスをトラップして変更を記録
// 3. 変更されたパスのみ新しいオブジェクトを生成
// 4. 変更されていない部分は元のオブジェクトを共有（構造共有）

// === React useReducer との組み合わせ ===
import { useImmerReducer } from "use-immer";

type Action =
  | { type: "UPDATE_USER"; id: string; name: string }
  | { type: "ADD_USER"; user: User }
  | { type: "DELETE_USER"; id: string }
  | { type: "SET_FILTER"; key: string; value: string };

function reducer(draft: Draft<AppState>, action: Action): void {
  switch (action.type) {
    case "UPDATE_USER": {
      const user = draft.users.find((u) => u.id === action.id);
      if (user) user.name = action.name;
      break;
    }
    case "ADD_USER":
      draft.users.push(action.user);
      break;
    case "DELETE_USER":
      draft.users = draft.users.filter((u) => u.id !== action.id);
      break;
    case "SET_FILTER":
      (draft.filters as any)[action.key] = action.value;
      break;
  }
}

// コンポーネントでの使用
function UserList() {
  const [state, dispatch] = useImmerReducer(reducer, initialState);

  const handleRename = (id: string, name: string) => {
    dispatch({ type: "UPDATE_USER", id, name });
  };

  return (
    <ul>
      {state.users.map((user) => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

---

## 4. マルチスレッド環境での恩恵

### 4.1 可変データの並行処理の危険

```
スレッド1              共有可変データ           スレッド2
─────────             ──────────────          ─────────

read(balance)          balance = 1000         read(balance)
  → 1000                                       → 1000

balance -= 500         balance = ???           balance -= 300
  → 500                                        → 700

write(balance)                                write(balance)
  → balance = 500       ← レースコンディション   → balance = 700

期待値: 200 (1000 - 500 - 300)
実際値: 500 or 700（後にwriteした方が勝つ）
```

このバグは非決定的で、特定の実行タイミングでのみ再現する。テストで見つけるのが非常に困難であり、本番環境で突然顕在化する。金融システムでは致命的な問題になりうる。

### 4.2 不変データならロック不要

```
スレッド1              不変データ              スレッド2
─────────             ──────────             ─────────

read(account)          account{balance:1000}  read(account)
  → {balance:1000}     (変更されない)           → {balance:1000}

new1 = withdraw(500)   account{balance:1000}  new2 = withdraw(300)
  → {balance:500}      (元は不変)              → {balance:700}

CAS(account, new1)     ← Compare-And-Swap     CAS(account, new2)
  → 成功                                       → 失敗→リトライ

                                              new3 = withdraw(300)
                                                from account(500)
                                              CAS → 成功

最終結果: {balance: 200} ← 正しい！
```

### 4.3 各言語のスレッド安全な不変データパターン

```
┌────────────────────────────────────────────────────────────┐
│              スレッド安全な不変データパターン                    │
├─────────┬──────────────────────────────────────────────────┤
│ Java    │ final フィールド + 不変クラス                       │
│         │ ConcurrentHashMap + compute で原子的更新            │
│         │ AtomicReference<ImmutableState> + CAS              │
├─────────┼──────────────────────────────────────────────────┤
│ Rust    │ Arc<T> で不変データを共有                           │
│         │ Arc<Mutex<T>> で可変データを排他制御                 │
│         │ コンパイラがデータ競合を完全に防止                    │
├─────────┼──────────────────────────────────────────────────┤
│ Kotlin  │ kotlinx.coroutines.flow で不変データの流れ          │
│         │ StateFlow<ImmutableState> で状態共有                │
├─────────┼──────────────────────────────────────────────────┤
│ TS/JS   │ シングルスレッド（Worker除く）なので問題なし          │
│         │ SharedArrayBuffer を使う場合は Atomics API          │
├─────────┼──────────────────────────────────────────────────┤
│ Python  │ GIL があるが、multiprocessing では不変データ推奨     │
│         │ frozen dataclass + copy で安全に更新                │
└─────────┴──────────────────────────────────────────────────┘
```

```java
// Java: AtomicReference + CAS による楽観的並行制御
import java.util.concurrent.atomic.AtomicReference;

public class ImmutableAccountService {
    private final AtomicReference<Account> accountRef;

    public ImmutableAccountService(Account initial) {
        this.accountRef = new AtomicReference<>(initial);
    }

    public Account withdraw(int amount) {
        while (true) {
            Account current = accountRef.get();
            Account updated = current.withdraw(amount); // 新しい不変オブジェクト
            if (accountRef.compareAndSet(current, updated)) {
                return updated; // CAS成功
            }
            // CAS失敗 → 他のスレッドが先に変更した → リトライ
        }
    }
}

// Account は不変（record）
public record Account(String id, int balance) {
    public Account withdraw(int amount) {
        if (balance < amount) throw new IllegalStateException("残高不足");
        return new Account(id, balance - amount);
    }
}
```

---

## 5. 実践パターン

### 5.1 React状態管理とイミュータビリティ

```typescript
// Reactにおけるイミュータビリティの重要性
// React は参照比較（===）で再レンダリングの要否を判定する

// === NG: ミュータブルな状態更新 ===
function TodoListBad() {
  const [todos, setTodos] = useState<Todo[]>([]);

  const addTodo = (text: string) => {
    // NG: 同じ配列を変更しても React は変更を検知しない
    todos.push({ id: Date.now().toString(), text, done: false });
    setTodos(todos); // 参照が同じなので再レンダリングされない！
  };

  const toggleTodo = (id: string) => {
    // NG: 要素を直接変更
    const todo = todos.find((t) => t.id === id);
    if (todo) todo.done = !todo.done;
    setTodos(todos); // 再レンダリングされない！
  };
}

// === OK: イミュータブルな状態更新 ===
function TodoListGood() {
  const [todos, setTodos] = useState<Todo[]>([]);

  const addTodo = (text: string) => {
    setTodos((prev) => [
      ...prev,
      { id: Date.now().toString(), text, done: false },
    ]);
  };

  const toggleTodo = (id: string) => {
    setTodos((prev) =>
      prev.map((todo) =>
        todo.id === id ? { ...todo, done: !todo.done } : todo
      )
    );
  };

  const removeTodo = (id: string) => {
    setTodos((prev) => prev.filter((todo) => todo.id !== id));
  };

  // React.memo + 不変データ → 不要な再レンダリングを防止
  return (
    <ul>
      {todos.map((todo) => (
        <MemoizedTodoItem
          key={todo.id}
          todo={todo}
          onToggle={toggleTodo}
          onRemove={removeTodo}
        />
      ))}
    </ul>
  );
}

const MemoizedTodoItem = React.memo(function TodoItem({
  todo,
  onToggle,
  onRemove,
}: {
  todo: Todo;
  onToggle: (id: string) => void;
  onRemove: (id: string) => void;
}) {
  // todoオブジェクトの参照が変わった時だけ再レンダリング
  return (
    <li>
      <span
        style={{ textDecoration: todo.done ? "line-through" : "none" }}
        onClick={() => onToggle(todo.id)}
      >
        {todo.text}
      </span>
      <button onClick={() => onRemove(todo.id)}>削除</button>
    </li>
  );
});
```

### 5.2 イベントソーシングとの親和性

```python
# イベントソーシング: 不変イベントの蓄積で状態を管理
from dataclasses import dataclass, field
from typing import Union
from datetime import datetime
from functools import reduce

# === 不変イベント定義 ===

@dataclass(frozen=True)
class AccountCreated:
    account_id: str
    owner: str
    timestamp: datetime

@dataclass(frozen=True)
class MoneyDeposited:
    account_id: str
    amount: int
    timestamp: datetime
    description: str = ""

@dataclass(frozen=True)
class MoneyWithdrawn:
    account_id: str
    amount: int
    timestamp: datetime
    description: str = ""

@dataclass(frozen=True)
class AccountClosed:
    account_id: str
    timestamp: datetime
    reason: str = ""

Event = Union[AccountCreated, MoneyDeposited, MoneyWithdrawn, AccountClosed]

# === 不変な口座状態 ===

@dataclass(frozen=True)
class AccountState:
    """不変な口座状態"""
    account_id: str
    owner: str
    balance: int
    is_active: bool = True
    transaction_count: int = 0

    @staticmethod
    def apply(state: "AccountState | None", event: Event) -> "AccountState":
        """イベントを適用して新しい状態を返す（純粋関数）"""
        match event:
            case AccountCreated(account_id, owner, _):
                return AccountState(
                    account_id=account_id,
                    owner=owner,
                    balance=0,
                    is_active=True,
                    transaction_count=0,
                )
            case MoneyDeposited(_, amount, _, _):
                return AccountState(
                    account_id=state.account_id,
                    owner=state.owner,
                    balance=state.balance + amount,
                    is_active=state.is_active,
                    transaction_count=state.transaction_count + 1,
                )
            case MoneyWithdrawn(_, amount, _, _):
                if state.balance < amount:
                    raise ValueError(f"残高不足: 残高{state.balance}, 出金{amount}")
                return AccountState(
                    account_id=state.account_id,
                    owner=state.owner,
                    balance=state.balance - amount,
                    is_active=state.is_active,
                    transaction_count=state.transaction_count + 1,
                )
            case AccountClosed(_, _, _):
                return AccountState(
                    account_id=state.account_id,
                    owner=state.owner,
                    balance=state.balance,
                    is_active=False,
                    transaction_count=state.transaction_count,
                )

# === イベントストア ===

class EventStore:
    """不変イベントの保管庫"""
    def __init__(self):
        self._events: tuple[Event, ...] = ()  # タプルで不変性を保証

    def append(self, event: Event) -> "EventStore":
        """新しいイベントを追加（元のストアは変更しない）"""
        new_store = EventStore()
        new_store._events = self._events + (event,)
        return new_store

    def replay(self) -> AccountState | None:
        """全イベントを再生して現在の状態を復元"""
        return reduce(AccountState.apply, self._events, None)

    def replay_at(self, timestamp: datetime) -> AccountState | None:
        """指定時点までのイベントを再生（タイムトラベル）"""
        events_until = tuple(
            e for e in self._events
            if e.timestamp <= timestamp
        )
        return reduce(AccountState.apply, events_until, None)

# === 使用例 ===

now = datetime.now()
store = EventStore()
store = store.append(AccountCreated("acc-1", "田中", now))
store = store.append(MoneyDeposited("acc-1", 10000, now, "初回入金"))
store = store.append(MoneyWithdrawn("acc-1", 3000, now, "食費"))
store = store.append(MoneyDeposited("acc-1", 5000, now, "給与"))

state = store.replay()
print(state)
# AccountState(account_id='acc-1', owner='田中', balance=12000,
#              is_active=True, transaction_count=3)
```

### 5.3 値オブジェクト（Value Object）パターン

```python
# DDD の値オブジェクト: イミュータビリティの代表的な活用先
from dataclasses import dataclass
from typing import Self

@dataclass(frozen=True, slots=True)
class Money:
    """金額を表す値オブジェクト"""
    amount: int  # 最小単位（円）
    currency: str

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError(f"金額は0以上: {self.amount}")
        if self.currency not in ("JPY", "USD", "EUR"):
            raise ValueError(f"未対応の通貨: {self.currency}")

    def add(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def subtract(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        if self.amount < other.amount:
            raise ValueError("残高不足")
        return Money(self.amount - other.amount, self.currency)

    def multiply(self, factor: int) -> "Money":
        return Money(self.amount * factor, self.currency)

    def _assert_same_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise ValueError(
                f"通貨不一致: {self.currency} vs {other.currency}"
            )

    def __str__(self) -> str:
        if self.currency == "JPY":
            return f"¥{self.amount:,}"
        return f"{self.amount / 100:.2f} {self.currency}"


@dataclass(frozen=True, slots=True)
class DateRange:
    """日付範囲を表す値オブジェクト"""
    start: date
    end: date

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(f"開始日が終了日より後: {self.start} > {self.end}")

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end

    def overlaps(self, other: "DateRange") -> bool:
        return self.start <= other.end and other.start <= self.end

    def duration_days(self) -> int:
        return (self.end - self.start).days

    def extend(self, days: int) -> "DateRange":
        from datetime import timedelta
        return DateRange(self.start, self.end + timedelta(days=days))


# 値オブジェクトの利点:
# 1. 等値性: 同じ値なら同じオブジェクト（__eq__が自動生成）
# 2. ハッシュ可能: dictのキーやsetの要素にできる（frozen=True）
# 3. スレッド安全: 変更されないのでロック不要
# 4. バリデーション: 不正な状態のオブジェクトが存在できない

price_a = Money(1000, "JPY")
price_b = Money(1000, "JPY")
print(price_a == price_b)  # True（値の等価性）
print(price_a is price_b)  # False（別のインスタンス）

prices = {price_a: "商品A"}  # ハッシュ可能なのでdictキーに使える
```

### 5.4 Undo/Redo の実装

```typescript
// 不変データによるUndo/Redo（タイムトラベルデバッグの原理）

interface HistoryState<T> {
  readonly past: readonly T[];
  readonly present: T;
  readonly future: readonly T[];
}

function createHistory<T>(initial: T): HistoryState<T> {
  return {
    past: [],
    present: initial,
    future: [],
  };
}

function pushState<T>(
  history: HistoryState<T>,
  newPresent: T
): HistoryState<T> {
  return {
    past: [...history.past, history.present],
    present: newPresent,
    future: [], // 新しい変更後はfutureをクリア
  };
}

function undo<T>(history: HistoryState<T>): HistoryState<T> {
  if (history.past.length === 0) return history; // undoできない

  const previous = history.past[history.past.length - 1];
  const newPast = history.past.slice(0, -1);

  return {
    past: newPast,
    present: previous,
    future: [history.present, ...history.future],
  };
}

function redo<T>(history: HistoryState<T>): HistoryState<T> {
  if (history.future.length === 0) return history; // redoできない

  const next = history.future[0];
  const newFuture = history.future.slice(1);

  return {
    past: [...history.past, history.present],
    present: next,
    future: newFuture,
  };
}

// 使用例: テキストエディタ
let editorHistory = createHistory("Hello");
editorHistory = pushState(editorHistory, "Hello World");
editorHistory = pushState(editorHistory, "Hello World!");

console.log(editorHistory.present); // "Hello World!"

editorHistory = undo(editorHistory);
console.log(editorHistory.present); // "Hello World"

editorHistory = undo(editorHistory);
console.log(editorHistory.present); // "Hello"

editorHistory = redo(editorHistory);
console.log(editorHistory.present); // "Hello World"
```

---

## 6. 不変性の導入戦略

### 6.1 段階的導入ロードマップ

```
Phase 1: 値オブジェクトから（1〜2週間）
──────────────────────────────────────
  対象: Money, Email, DateRange, UserId 等
  方法: frozen dataclass / record / value class
  効果: バリデーション集約、ハッシュ可能

Phase 2: ドメインモデルの不変化（2〜4週間）
──────────────────────────────────────────
  対象: User, Order, Product 等のエンティティ
  方法: Wither パターン、copy メソッド
  効果: 状態変更の明示化、テスト容易性向上

Phase 3: コレクション操作の不変化（1〜2週間）
──────────────────────────────────────────
  対象: リスト操作、マップ操作
  方法: map/filter/reduce、スプレッド構文
  効果: 副作用の除去、宣言的なコード

Phase 4: 状態管理の不変化（2〜4週間）
────────────────────────────────────
  対象: アプリケーション状態、Redux/Zustand
  方法: Immer, 永続データ構造
  効果: タイムトラベルデバッグ、変更追跡

Phase 5: Lint/型システムで強制（1週間）
────────────────────────────────────────
  対象: プロジェクト全体
  方法: ESLint no-param-reassign, TypeScript readonly
  効果: チーム全体での不変性文化の定着
```

### 6.2 Lintルール設定

```json
// .eslintrc.json: 不変性を強制するルール
{
  "rules": {
    "no-param-reassign": ["error", {
      "props": true,
      "ignorePropertyModificationsFor": ["draft"]
    }],
    "prefer-const": "error",
    "no-var": "error",
    "no-let": "warn",
    "immutable/no-let": "error",
    "immutable/no-mutation": "warn",
    "functional/no-let": "error",
    "functional/immutable-data": "error",
    "functional/no-method-signature": "warn"
  }
}
```

```yaml
# .pre-commit-config.yaml: Pythonの不変性チェック
repos:
  - repo: local
    hooks:
      - id: check-mutable-defaults
        name: Check mutable default arguments
        language: pygrep
        entry: 'def\s+\w+\(.*=\s*(\[\]|\{\}|set\(\))'
        types: [python]
        # NG: def func(items=[]), def func(data={})
```

---

## 7. アンチパターン

### 7.1 アンチパターン：浅いコピーの罠

```python
# NG: 浅いコピーでネストされたオブジェクトが共有される
original = {"user": {"name": "田中", "scores": [90, 85]}}
copied = original.copy()  # 浅いコピー

copied["user"]["name"] = "鈴木"
print(original["user"]["name"])  # "鈴木" ← 元も変わってしまう！

copied["user"]["scores"].append(95)
print(original["user"]["scores"])  # [90, 85, 95] ← 元も変わる！

# OK: 深いコピーまたは不変データ構造を使用
import copy
deep_copied = copy.deepcopy(original)
deep_copied["user"]["name"] = "鈴木"
print(original["user"]["name"])  # "田中" ← 元は変わらない

# より良い: frozen dataclass で根本的に防止
@dataclass(frozen=True)
class User:
    name: str
    scores: tuple[int, ...]  # tupleは不変（listではない）

    def add_score(self, score: int) -> "User":
        return replace(self, scores=self.scores + (score,))
```

**問題点**: 浅いコピーはネストされた参照を共有するため、意図しない変更が伝播する。JavaScript のスプレッド構文 `{...obj}` も浅いコピーであり、同じ問題が発生する。深いコピーか不変データ構造で対処する。

### 7.2 アンチパターン：全てをイミュータブルにする

```python
# NG: パフォーマンスクリティカルな処理でも不変性を強制
def process_large_dataset_bad(data: tuple) -> tuple:
    result = data
    for i in range(len(data)):
        # 毎回タプル全体をコピー → O(n^2) の計算量
        result = result[:i] + (transform(result[i]),) + result[i+1:]
    return result

# 10,000要素で約50倍遅くなる

# OK: 内部処理は可変、外部インターフェースは不変
def process_large_dataset_good(data: tuple) -> tuple:
    # 内部ではリスト（可変）で効率的に処理
    work_list = list(data)
    for i in range(len(work_list)):
        work_list[i] = transform(work_list[i])
    # 結果はタプル（不変）で返す
    return tuple(work_list)
```

**問題点**: 全てを不変にするとパフォーマンスが劣化する場合がある。「外部API（公開インターフェース）は不変、内部実装は可変でもよい」という境界を明確にする。

### 7.3 アンチパターン：不変データの過度なネスト更新

```typescript
// NG: 深いネストの手動スプレッド更新
const nextState = {
  ...state,
  users: {
    ...state.users,
    [userId]: {
      ...state.users[userId],
      address: {
        ...state.users[userId].address,
        city: newCity,
      },
    },
  },
};
// 読みにくく、バグが入りやすい

// OK: Immer を使って読みやすく
const nextState = produce(state, (draft) => {
  draft.users[userId].address.city = newCity;
});

// OK: レンズ（lens）パターンで型安全に
import { pipe } from "fp-ts/function";
import * as L from "monocle-ts/Lens";

const cityLens = pipe(
  L.id<State>(),
  L.prop("users"),
  L.key(userId),
  L.prop("address"),
  L.prop("city")
);

const nextState = pipe(state, cityLens.set(newCity));
```

**問題点**: スプレッド構文のネストは可読性が著しく低下する。Immerやレンズライブラリで宣言的に記述すべき。

### 7.4 アンチパターン：freeze の乱用

```javascript
// NG: パフォーマンスホットパスでObject.freezeを使う
function processItems(items) {
  return items.map((item) => {
    const result = Object.freeze({
      // 毎回freezeするとGCの負荷が増大
      ...item,
      processed: true,
    });
    return result;
  });
}

// OK: 型システム（TypeScript）で不変性を保証し、
// Object.freezeは開発時の検証用に限定する
function processItems(items: readonly Item[]): readonly ProcessedItem[] {
  return items.map((item) => ({
    ...item,
    processed: true as const,
  }));
}

// Object.freeze は開発時の検証に使う
if (process.env.NODE_ENV === "development") {
  deepFreeze(config);
}
```

**問題点**: `Object.freeze` は実行時コストがかかり、GCにも影響する。型システムでコンパイル時に不変性を保証する方が効率的である。

---

## 8. 実践演習

### 演習1（基礎）: 不変なユーザー管理

**課題**: 以下の要件を満たす不変なユーザー管理モジュールを Python で実装してください。

```python
# 要件:
# 1. User は frozen dataclass とする（id, name, email, role）
# 2. UserRepository は不変なユーザーリスト（タプル）を管理する
# 3. add_user, remove_user, update_user_email は全て新しいリポジトリを返す
# 4. find_by_id, find_by_role を実装する

# ヒント: replaceとタプル操作を使う
```

**期待される実装**:

```python
from dataclasses import dataclass, replace
from typing import Optional

@dataclass(frozen=True, slots=True)
class User:
    id: str
    name: str
    email: str
    role: str = "member"

    def with_email(self, new_email: str) -> "User":
        if "@" not in new_email:
            raise ValueError(f"無効なメールアドレス: {new_email}")
        return replace(self, email=new_email)

    def with_role(self, new_role: str) -> "User":
        valid_roles = {"member", "admin", "moderator"}
        if new_role not in valid_roles:
            raise ValueError(f"無効なロール: {new_role}")
        return replace(self, role=new_role)


@dataclass(frozen=True)
class UserRepository:
    users: tuple[User, ...] = ()

    def add(self, user: User) -> "UserRepository":
        if self.find_by_id(user.id) is not None:
            raise ValueError(f"ユーザーIDが重複: {user.id}")
        return replace(self, users=self.users + (user,))

    def remove(self, user_id: str) -> "UserRepository":
        new_users = tuple(u for u in self.users if u.id != user_id)
        if len(new_users) == len(self.users):
            raise ValueError(f"ユーザーが見つかりません: {user_id}")
        return replace(self, users=new_users)

    def update(self, user_id: str, updater) -> "UserRepository":
        new_users = tuple(
            updater(u) if u.id == user_id else u
            for u in self.users
        )
        return replace(self, users=new_users)

    def find_by_id(self, user_id: str) -> Optional[User]:
        return next((u for u in self.users if u.id == user_id), None)

    def find_by_role(self, role: str) -> tuple[User, ...]:
        return tuple(u for u in self.users if u.role == role)

    def count(self) -> int:
        return len(self.users)


# テスト
repo = UserRepository()
repo = repo.add(User("1", "田中", "tanaka@example.com"))
repo = repo.add(User("2", "鈴木", "suzuki@example.com", "admin"))
repo = repo.add(User("3", "佐藤", "sato@example.com"))

assert repo.count() == 3
assert repo.find_by_id("1").name == "田中"
assert len(repo.find_by_role("member")) == 2

# メール更新
repo = repo.update("1", lambda u: u.with_email("tanaka_new@example.com"))
assert repo.find_by_id("1").email == "tanaka_new@example.com"

# 削除
repo = repo.remove("3")
assert repo.count() == 2
print("全テスト通過！")
```

**期待される出力**:
```
全テスト通過！
```

---

### 演習2（応用）: 不変なショッピングカート

**課題**: TypeScript で不変なショッピングカートを実装してください。以下の操作を全てイミュータブルに行うこと。

```typescript
// 要件:
// 1. CartItem, Cart は readonly プロパティのみ
// 2. addItem: 既存商品なら数量を加算、新規なら追加
// 3. removeItem: 数量を1減らし、0になったら削除
// 4. applyCoupon: 割引率を適用
// 5. calculateTotal: 合計金額を計算（税込み）
// 6. toSummary: カートの概要をオブジェクトで返す
```

**期待される実装**:

```typescript
interface CartItem {
  readonly productId: string;
  readonly name: string;
  readonly price: number;
  readonly quantity: number;
}

interface Cart {
  readonly items: readonly CartItem[];
  readonly couponRate: number; // 0.0 ~ 1.0
}

// === カート操作関数（全て純粋関数） ===

function createCart(): Cart {
  return { items: [], couponRate: 0 };
}

function addItem(cart: Cart, product: Omit<CartItem, "quantity">): Cart {
  const existingIndex = cart.items.findIndex(
    (item) => item.productId === product.productId
  );

  if (existingIndex >= 0) {
    // 既存商品: 数量を加算
    const updatedItems = cart.items.map((item, i) =>
      i === existingIndex
        ? { ...item, quantity: item.quantity + 1 }
        : item
    );
    return { ...cart, items: updatedItems };
  }

  // 新規商品: 追加
  return {
    ...cart,
    items: [...cart.items, { ...product, quantity: 1 }],
  };
}

function removeItem(cart: Cart, productId: string): Cart {
  const existingIndex = cart.items.findIndex(
    (item) => item.productId === productId
  );
  if (existingIndex < 0) return cart;

  const item = cart.items[existingIndex];

  if (item.quantity <= 1) {
    // 数量が1以下なら削除
    return {
      ...cart,
      items: cart.items.filter((_, i) => i !== existingIndex),
    };
  }

  // 数量を1減らす
  const updatedItems = cart.items.map((item, i) =>
    i === existingIndex
      ? { ...item, quantity: item.quantity - 1 }
      : item
  );
  return { ...cart, items: updatedItems };
}

function applyCoupon(cart: Cart, rate: number): Cart {
  if (rate < 0 || rate > 1) {
    throw new Error(`無効な割引率: ${rate}`);
  }
  return { ...cart, couponRate: rate };
}

function calculateTotal(
  cart: Cart,
  taxRate: number = 0.1
): { subtotal: number; discount: number; tax: number; total: number } {
  const subtotal = cart.items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );
  const discount = Math.floor(subtotal * cart.couponRate);
  const afterDiscount = subtotal - discount;
  const tax = Math.floor(afterDiscount * taxRate);
  const total = afterDiscount + tax;

  return { subtotal, discount, tax, total };
}

function toSummary(cart: Cart) {
  const totals = calculateTotal(cart);
  return {
    itemCount: cart.items.reduce((sum, item) => sum + item.quantity, 0),
    uniqueProducts: cart.items.length,
    ...totals,
  };
}

// === テスト ===
let cart = createCart();
cart = addItem(cart, { productId: "p1", name: "りんご", price: 200 });
cart = addItem(cart, { productId: "p1", name: "りんご", price: 200 }); // 数量2に
cart = addItem(cart, { productId: "p2", name: "みかん", price: 150 });
cart = applyCoupon(cart, 0.1); // 10%割引

const summary = toSummary(cart);
console.log(summary);
// {
//   itemCount: 3,
//   uniqueProducts: 2,
//   subtotal: 550,  // 200*2 + 150*1
//   discount: 55,   // 550 * 0.1
//   tax: 49,        // (550-55) * 0.1 = 49.5 → 49
//   total: 544      // 495 + 49
// }
```

**期待される出力**:
```
{ itemCount: 3, uniqueProducts: 2, subtotal: 550, discount: 55, tax: 49, total: 544 }
```

---

### 演習3（発展）: 不変データによるタイムトラベルデバッグ

**課題**: Pythonで不変データを活用したタイムトラベルデバッグ機能を持つステートマシンを実装してください。

```python
# 要件:
# 1. State は frozen dataclass
# 2. StateMachine は全ての状態履歴を不変タプルとして保持
# 3. dispatch(action) で新しい状態を生成
# 4. undo() / redo() でタイムトラベル
# 5. get_history() で全履歴を返す
# 6. goto(index) で任意の時点にジャンプ
```

**期待される実装**:

```python
from dataclasses import dataclass, replace, field
from typing import Callable, Any, TypeVar
from datetime import datetime

S = TypeVar("S")

@dataclass(frozen=True)
class AppState:
    """アプリケーション状態（不変）"""
    counter: int = 0
    message: str = ""
    items: tuple[str, ...] = ()
    last_action: str = "INIT"

# アクション定義
@dataclass(frozen=True)
class Action:
    type: str
    payload: Any = None
    timestamp: datetime = field(default_factory=datetime.now)

# 純粋なリデューサ
def reducer(state: AppState, action: Action) -> AppState:
    """状態遷移関数（純粋）"""
    match action.type:
        case "INCREMENT":
            return replace(state, counter=state.counter + 1, last_action="INCREMENT")
        case "DECREMENT":
            return replace(state, counter=state.counter - 1, last_action="DECREMENT")
        case "SET_MESSAGE":
            return replace(state, message=action.payload, last_action="SET_MESSAGE")
        case "ADD_ITEM":
            return replace(
                state,
                items=state.items + (action.payload,),
                last_action="ADD_ITEM"
            )
        case "REMOVE_ITEM":
            return replace(
                state,
                items=tuple(i for i in state.items if i != action.payload),
                last_action="REMOVE_ITEM"
            )
        case "RESET":
            return AppState(last_action="RESET")
        case _:
            return state

@dataclass(frozen=True)
class TimeTravelMachine:
    """タイムトラベル可能なステートマシン"""
    past: tuple[AppState, ...] = ()
    present: AppState = field(default_factory=AppState)
    future: tuple[AppState, ...] = ()
    action_log: tuple[Action, ...] = ()

    def dispatch(self, action: Action) -> "TimeTravelMachine":
        """アクションをディスパッチして新しい状態を返す"""
        new_state = reducer(self.present, action)
        return TimeTravelMachine(
            past=self.past + (self.present,),
            present=new_state,
            future=(),  # 新しいアクション後はfutureクリア
            action_log=self.action_log + (action,),
        )

    def undo(self) -> "TimeTravelMachine":
        """1つ前の状態に戻る"""
        if not self.past:
            return self  # undoできない
        previous = self.past[-1]
        return TimeTravelMachine(
            past=self.past[:-1],
            present=previous,
            future=(self.present,) + self.future,
            action_log=self.action_log,
        )

    def redo(self) -> "TimeTravelMachine":
        """1つ先の状態に進む"""
        if not self.future:
            return self  # redoできない
        next_state = self.future[0]
        return TimeTravelMachine(
            past=self.past + (self.present,),
            present=next_state,
            future=self.future[1:],
            action_log=self.action_log,
        )

    def goto(self, index: int) -> "TimeTravelMachine":
        """任意の時点にジャンプ"""
        all_states = self.past + (self.present,) + self.future
        if index < 0 or index >= len(all_states):
            raise IndexError(f"インデックス範囲外: {index}")
        return TimeTravelMachine(
            past=all_states[:index],
            present=all_states[index],
            future=all_states[index + 1:],
            action_log=self.action_log,
        )

    def get_history(self) -> list[dict]:
        """全履歴を取得"""
        all_states = self.past + (self.present,) + self.future
        current_index = len(self.past)
        return [
            {
                "index": i,
                "state": state,
                "is_current": i == current_index,
            }
            for i, state in enumerate(all_states)
        ]

    @property
    def can_undo(self) -> bool:
        return len(self.past) > 0

    @property
    def can_redo(self) -> bool:
        return len(self.future) > 0


# === テスト ===

machine = TimeTravelMachine()

# アクションをディスパッチ
machine = machine.dispatch(Action("INCREMENT"))
machine = machine.dispatch(Action("INCREMENT"))
machine = machine.dispatch(Action("SET_MESSAGE", "Hello"))
machine = machine.dispatch(Action("ADD_ITEM", "りんご"))
machine = machine.dispatch(Action("ADD_ITEM", "みかん"))

assert machine.present.counter == 2
assert machine.present.message == "Hello"
assert machine.present.items == ("りんご", "みかん")

# Undo
machine = machine.undo()
assert machine.present.items == ("りんご",)  # みかん追加前に戻る

machine = machine.undo()
assert machine.present.message == "Hello"
assert machine.present.items == ()  # りんご追加前に戻る

# Redo
machine = machine.redo()
assert machine.present.items == ("りんご",)  # りんご追加後に進む

# Goto
machine = machine.goto(0)
assert machine.present.counter == 0  # 初期状態にジャンプ

machine = machine.goto(2)
assert machine.present.counter == 2  # 2回INCREMENT後にジャンプ

# 履歴確認
history = machine.get_history()
assert len(history) == 6  # 初期 + 5アクション
assert history[2]["is_current"] == True

print("全テスト通過！")
```

**期待される出力**:
```
全テスト通過！
```

---

## 9. FAQ

### Q1: イミュータビリティはパフォーマンスに悪影響か？

**A**: 小〜中規模データでは影響は無視できる。大規模データでは構造共有（Persistent Data Structures）やImmer.jsのようなライブラリで効率的に処理できる。むしろ、変更検知がO(1)になるため、React等のUIフレームワークではパフォーマンス向上に寄与する。ボトルネックが確認された場合のみ、局所的に可変データを使う。実測なしに「パフォーマンスが悪い」と判断するのは早計であり、プロファイラで測定してから最適化すべきである。

### Q2: データベースとの連携でイミュータビリティは維持できるか？

**A**: アプリケーション層でイミュータブルに扱い、永続化層（Repository/DAO）で変換するのが一般的。具体的には、DB から取得したデータを不変なドメインモデルに変換し、ビジネスロジックは不変データのみで処理し、永続化時に再度DBのフォーマットに変換する。イベントソーシングやCQRSパターンを採用すれば、データベース層でも不変性を活かせる。ORMの遅延ロードやダーティチェックとの相性は要注意で、ORMが期待する可変性と不変モデルの間で変換層が必要になる場合がある。

### Q3: チームにイミュータビリティを導入するにはどうすればよいか？

**A**: (1) まず値オブジェクト（Money、Date等）から始める、(2) 新規コードに `readonly`/`final`/`frozen` を適用、(3) Lintルールで可変操作を警告（`no-param-reassign` 等）、(4) コードレビューで不変パターンを推奨。段階的に広げることで抵抗なく導入できる。「既存コードを全部書き換える」のではなく、新規コードから適用し、修正のたびに周辺コードを不変化していく漸進的アプローチが効果的。

### Q4: 不変データは本当にスレッド安全か？初期化中のオブジェクトは危険では？

**A**: 正確に言うと、「完全に構築されたあとの不変オブジェクト」がスレッド安全である。Java では `final` フィールドの初期化は Java Memory Model で安全性が保証されている。ただし、コンストラクタ内で `this` を外部に公開すると、構築途中のオブジェクトが他のスレッドに見える危険がある。Rust では所有権システムがこの問題を完全に防ぐ。他の言語では「コンストラクタ内で this をリークしない」というルールを守る必要がある。

### Q5: ORMやフレームワークが可変オブジェクトを要求する場合はどうするか？

**A**: アダプタパターンを使って境界を明確にする。(1) ドメイン層: 不変データモデル（frozen dataclass/record）、(2) インフラ層: ORMが要求する可変モデル（通常のクラス）、(3) 変換層: 不変モデルとORMモデルの間の変換関数。Spring Data JPA の場合は `@Immutable` アノテーション、Django の場合は `model_to_frozen_dataclass` のような変換ユーティリティを作る。変換のコストはあるが、ドメインロジックの安全性と引き換えに十分な価値がある。

### Q6: JavaScript の `const` とイミュータビリティの違いは？

**A**: `const` は変数の再代入を禁止するだけであり、オブジェクトのプロパティ変更は許容する。`const obj = {a: 1}; obj.a = 2;` は有効である。真のイミュータビリティを実現するには、`Object.freeze`（実行時）、TypeScript の `readonly`/`as const`（型レベル）、Immer（ライブラリ）のいずれかを使う必要がある。`const` は「変数束縛の不変性」であり、「値の不変性」とは異なるレベルの概念である。

### Q7: イミュータブルなデータ構造はGCに負荷をかけないか？

**A**: 短命なオブジェクトを大量に生成するため、GCの負荷は増加する。ただし、世代別GC（JVM、V8）は短命オブジェクトの回収が非常に効率的であり、実測で問題にならないことが多い。構造共有を使えば新規生成するオブジェクト数を大幅に削減できる。Rust の場合はGCがないため、所有権システムによる確定的なメモリ解放でこの問題は存在しない。GCの負荷が問題になるのは、非常に高頻度（秒間数十万回）のオブジェクト生成が発生するゲームエンジンやリアルタイムシステムなどの特殊なケースに限られる。

---

## 10. まとめ

| カテゴリ | ポイント |
|---------|---------|
| 原則 | デフォルトを不変に、可変は明示的に |
| スレッド安全 | 不変データはロック不要で並行処理が安全 |
| 予測可能性 | 値が変わらない → デバッグ・テストが容易 |
| 変更検知 | 参照比較O(1)でUI更新の効率化 |
| パフォーマンス | 構造共有/Immerで大規模データも効率的 |
| 言語選択 | Rustはデフォルト不変、他言語はライブラリ/規約で対応 |
| 導入戦略 | 値オブジェクトから段階的に、Lint支援で定着 |
| アーキテクチャ | イベントソーシング・CQRS・Reduxと好相性 |
| トレードオフ | 内部は可変OK、公開APIは不変がベストプラクティス |

| 言語 | 不変性サポート | 推奨パターン |
|------|-------------|------------|
| TypeScript | readonly, as const, DeepReadonly | スプレッド構文 + Immer |
| Java | final, Record, List.of() | Record + Wither パターン |
| Python | frozen dataclass, NamedTuple | frozen dataclass + replace |
| Rust | デフォルト不変、所有権システム | 言語機能そのまま |
| Kotlin | val, data class, listOf() | data class + copy |
| Scala | val, case class, 永続コレクション | 言語機能そのまま |

---

## 次に読むべきガイド

- [01-composition-over-inheritance.md](./01-composition-over-inheritance.md) -- 継承より合成の原則
- [02-functional-principles.md](./02-functional-principles.md) -- 関数型プログラミングの原則（純粋関数と不変性の深い関係）
- [03-api-design.md](./03-api-design.md) -- API設計（不変なリクエスト/レスポンスモデル）
- [00-principles/02-solid.md](../00-principles/02-solid.md) -- SOLID原則（特にOCPと不変性の関係）
- [02-refactoring/00-code-smells.md](../02-refactoring/00-code-smells.md) -- コードの臭い（可変状態の乱用パターン）
- `design-patterns-guide/docs/03-functional/` -- 関数型デザインパターン
- `system-design-guide/docs/02-architecture/` -- アーキテクチャパターン（イベントソーシング・CQRS）

---

## 参考文献

1. Joshua Bloch, "Effective Java" 第3版 -- Item 17: Minimize mutability
2. Michael Feathers, "Working Effectively with Legacy Code" -- Immutability as a tool for safety
3. Immer.js ドキュメント -- https://immerjs.github.io/immer/
4. Rust Book, "Understanding Ownership" -- https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html
5. Eric Evans, "Domain-Driven Design" -- Value Objects
6. Rich Hickey, "The Value of Values" -- https://www.infoq.com/presentations/Value-Values/ (Clojure作者による不変性の講演)
7. Chris Okasaki, "Purely Functional Data Structures" -- 永続データ構造の理論と実装
8. Gary Bernhardt, "Boundaries" -- https://www.destroyallsoftware.com/talks/boundaries (Functional Core / Imperative Shell)
9. Martin Fowler, "ValueObject" -- https://martinfowler.com/bliki/ValueObject.html
10. Kotlin Documentation, "Properties and Fields" -- https://kotlinlang.org/docs/properties.html
