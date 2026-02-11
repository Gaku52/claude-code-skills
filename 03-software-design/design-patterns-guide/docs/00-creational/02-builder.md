# Builder パターン

> 複雑なオブジェクトの構築プロセスを **段階的に** 分離し、同じ構築手順で異なる表現を作成できるようにする生成パターン。

---

## この章で学ぶこと

1. Builder パターンの構造と Fluent API による直感的なオブジェクト構築
2. Director の役割と Builder 単独での利用の使い分け
3. Telescoping Constructor 問題の解決と不変オブジェクトの構築

---

## 1. Builder の構造

```
+------------+       +-----------------+
|  Director  |------>|  Builder        |
+------------+       |  (interface)    |
| + construct|       +-----------------+
+------------+       | + setPartA()    |
                     | + setPartB()    |
                     | + build(): Product
                     +-----------------+
                             ^
                             |
                     +-----------------+
                     | ConcreteBuilder |
                     +-----------------+
                     | - product       |
                     | + setPartA()    |
                     | + setPartB()    |
                     | + build()       |
                     +-----------------+
```

---

## 2. コード例

### コード例 1: Fluent Builder（TypeScript）

```typescript
class HttpRequest {
  readonly method: string;
  readonly url: string;
  readonly headers: Record<string, string>;
  readonly body?: string;
  readonly timeout: number;

  private constructor(builder: HttpRequestBuilder) {
    this.method  = builder.method;
    this.url     = builder.url;
    this.headers = { ...builder.headers };
    this.body    = builder.body;
    this.timeout = builder.timeout;
  }

  static builder(method: string, url: string): HttpRequestBuilder {
    return new HttpRequestBuilder(method, url);
  }
}

class HttpRequestBuilder {
  method: string;
  url: string;
  headers: Record<string, string> = {};
  body?: string;
  timeout: number = 30_000;

  constructor(method: string, url: string) {
    this.method = method;
    this.url = url;
  }

  setHeader(key: string, value: string): this {
    this.headers[key] = value;
    return this;
  }

  setBody(body: string): this {
    this.body = body;
    return this;
  }

  setTimeout(ms: number): this {
    this.timeout = ms;
    return this;
  }

  build(): HttpRequest {
    return new (HttpRequest as any)(this);
  }
}

// 使用
const req = HttpRequest.builder("POST", "/api/users")
  .setHeader("Content-Type", "application/json")
  .setBody(JSON.stringify({ name: "Taro" }))
  .setTimeout(5000)
  .build();
```

### コード例 2: Director パターン

```typescript
class QueryDirector {
  buildPaginatedQuery(builder: QueryBuilder, page: number, size: number) {
    return builder
      .select("*")
      .from("users")
      .where("active = true")
      .orderBy("created_at", "DESC")
      .limit(size)
      .offset((page - 1) * size)
      .build();
  }

  buildCountQuery(builder: QueryBuilder) {
    return builder
      .select("COUNT(*)")
      .from("users")
      .where("active = true")
      .build();
  }
}
```

### コード例 3: 不変オブジェクトの構築

```typescript
interface UserConfig {
  readonly name: string;
  readonly email: string;
  readonly role: "admin" | "user";
  readonly notifications: boolean;
}

class UserConfigBuilder {
  private config: Partial<UserConfig> = {};

  setName(name: string): this {
    this.config.name = name;
    return this;
  }
  setEmail(email: string): this {
    this.config.email = email;
    return this;
  }
  setRole(role: "admin" | "user"): this {
    this.config.role = role;
    return this;
  }
  enableNotifications(flag: boolean): this {
    this.config.notifications = flag;
    return this;
  }

  build(): UserConfig {
    if (!this.config.name || !this.config.email) {
      throw new Error("name and email are required");
    }
    return Object.freeze({
      name: this.config.name,
      email: this.config.email,
      role: this.config.role ?? "user",
      notifications: this.config.notifications ?? true,
    });
  }
}
```

### コード例 4: Python — Builder with dataclass

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class Pizza:
    size: str
    cheese: bool
    pepperoni: bool
    mushrooms: bool
    extra_toppings: list[str]

class PizzaBuilder:
    def __init__(self, size: str):
        self._size = size
        self._cheese = False
        self._pepperoni = False
        self._mushrooms = False
        self._extra: list[str] = []

    def add_cheese(self) -> "PizzaBuilder":
        self._cheese = True
        return self

    def add_pepperoni(self) -> "PizzaBuilder":
        self._pepperoni = True
        return self

    def add_mushrooms(self) -> "PizzaBuilder":
        self._mushrooms = True
        return self

    def add_topping(self, topping: str) -> "PizzaBuilder":
        self._extra.append(topping)
        return self

    def build(self) -> Pizza:
        return Pizza(
            self._size, self._cheese, self._pepperoni,
            self._mushrooms, list(self._extra)
        )

pizza = (PizzaBuilder("Large")
    .add_cheese()
    .add_pepperoni()
    .add_topping("olives")
    .build())
```

### コード例 5: Step Builder（型安全な必須フィールド保証）

```typescript
// Step 1: name が必須
interface NeedsName {
  setName(name: string): NeedsEmail;
}
// Step 2: email が必須
interface NeedsEmail {
  setEmail(email: string): OptionalFields;
}
// Step 3: 任意フィールド
interface OptionalFields {
  setAge(age: number): OptionalFields;
  build(): Person;
}

class PersonBuilder implements NeedsName, NeedsEmail, OptionalFields {
  private name!: string;
  private email!: string;
  private age?: number;

  static create(): NeedsName {
    return new PersonBuilder();
  }

  setName(name: string): NeedsEmail {
    this.name = name;
    return this;
  }
  setEmail(email: string): OptionalFields {
    this.email = email;
    return this;
  }
  setAge(age: number): OptionalFields {
    this.age = age;
    return this;
  }
  build(): Person {
    return { name: this.name, email: this.email, age: this.age };
  }
}

// コンパイル時に順序が強制される
const person = PersonBuilder.create()
  .setName("Taro")        // NeedsEmail が返る
  .setEmail("t@example.com") // OptionalFields が返る
  .setAge(30)
  .build();
```

---

## 3. Telescoping Constructor の問題

```
// BAD: 引数が増えるとどれが何か分からない
//                  name    email      age  role   notify  theme
new User("Taro", "t@x.com", 30, "admin", true, "dark");
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   6番目が何だったか覚えていますか？

// GOOD: Builder なら自己文書化される
User.builder()
  .setName("Taro")
  .setEmail("t@x.com")
  .setAge(30)
  .setRole("admin")
  .enableNotifications(true)
  .setTheme("dark")
  .build();
```

---

## 4. 比較表

### 比較表 1: Builder vs コンストラクタ vs オブジェクトリテラル

| 観点 | コンストラクタ | オブジェクトリテラル | Builder |
|------|:---:|:---:|:---:|
| 必須フィールド強制 | Yes | 要バリデーション | Yes (Step Builder) |
| 可読性（多引数） | 低い | 中 | 高い |
| 不変性保証 | 可能 | 困難 | 容易 |
| 複雑な構築ロジック | 困難 | 困難 | 容易 |
| コード量 | 少ない | 少ない | 多い |

### 比較表 2: Builder vs Factory

| 観点 | Builder | Factory |
|------|---------|---------|
| 目的 | 段階的な構築 | 型の選択 |
| 返すもの | 1つの複雑なオブジェクト | さまざまな型のオブジェクト |
| メソッドチェーン | 一般的 | まれ |
| 適用場面 | 多数のオプション引数 | バリエーションの切り替え |

---

## 5. アンチパターン

### アンチパターン 1: Builder の build() を呼び忘れる

```typescript
// BAD: build() を忘れて Builder オブジェクトを使ってしまう
const config = new ConfigBuilder()
  .setHost("localhost")
  .setPort(8080);
  // .build() を忘れた！ config は ConfigBuilder 型

server.start(config); // 型エラーまたは実行時エラー
```

**改善**: TypeScript の型システムで Builder と Product を明確に区別する。Step Builder を使えばコンパイルエラーにできる。

### アンチパターン 2: Builder にビジネスロジックを詰める

```typescript
// BAD: Builder が構築以外の責任を持つ
class OrderBuilder {
  // ...
  build(): Order {
    const order = new Order(this);
    order.calculateTax();       // ビジネスロジック
    order.applyDiscount();      // ビジネスロジック
    this.sendNotification();    // 副作用
    return order;
  }
}
```

**改善**: Builder は構築のみ。ビジネスロジックはドメインサービスに委譲する。

---

## 6. FAQ

### Q1: Builder はどの程度の複雑さから導入すべきですか？

目安として、コンストラクタの引数が 4 つ以上、またはオプション引数が 2 つ以上ある場合に Builder の導入を検討します。引数が 2〜3 個であればコンストラクタで十分です。

### Q2: Lombok の @Builder のような自動生成は推奨ですか？

はい。Java では Lombok、Kotlin では data class + copy、TypeScript では zod の `.parse()` など、言語やライブラリの機能を活用して定型コードを減らすのが実務的です。

### Q3: Builder と Named Arguments（キーワード引数）の違いは？

Python や Kotlin のキーワード引数は Builder の多くのメリットを提供します。しかし、バリデーション、段階的構築、不変性保証が必要な場合は Builder が優位です。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | 複雑なオブジェクトを段階的に構築 |
| Fluent API | メソッドチェーンで可読性向上 |
| Step Builder | 型安全に必須フィールドを保証 |
| Director | 構築手順を再利用可能にカプセル化 |
| 判断基準 | 引数 4+ またはオプション 2+ で検討 |

---

## 次に読むべきガイド

- [Prototype パターン](./03-prototype.md) — クローンによる生成
- [Decorator パターン](../01-structural/01-decorator.md) — 動的な機能追加
- [関数設計](../../../clean-code-principles/docs/01-practices/01-functions.md) — 引数設計のベストプラクティス

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. Bloch, J. (2018). *Effective Java* (3rd ed.). Addison-Wesley. Item 2: Consider a builder when faced with many constructor parameters.
3. Refactoring.Guru — Builder. https://refactoring.guru/design-patterns/builder
