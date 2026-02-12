# Builder パターン

> 複雑なオブジェクトの構築プロセスを **段階的に** 分離し、同じ構築手順で異なる表現を作成できるようにする生成パターン。

---

## この章で学ぶこと

1. Builder パターンの本質的な目的と、なぜ「段階的な構築」が必要なのかという設計意図（WHY）
2. Fluent API / Step Builder / Director の各バリエーションと使い分け
3. Telescoping Constructor 問題の根本的な解決と不変オブジェクトの安全な構築
4. 各言語（TypeScript / Python / Java / Go / Kotlin）での実装パターンとベストプラクティス
5. Builder パターンの過剰適用を避けるための判断基準とトレードオフ

---

## 前提知識

このガイドを理解するために、以下の知識を事前に習得しておくことを推奨します。

| トピック | 必要レベル | 参照リンク |
|---------|-----------|-----------|
| オブジェクト指向の基礎（クラス、インタフェース、メソッドチェーン） | 必須 | [OOP基礎](../../../02-programming/oop-guide/docs/) |
| Factory パターン | 推奨 | [Factory](./01-factory.md) |
| 不変性（Immutability）の概念 | 推奨 | [不変性](../../../clean-code-principles/docs/03-practices-advanced/00-immutability.md) |
| TypeScript の型システム（ジェネリクス、条件型） | あると望ましい | TypeScript ドキュメント |
| 関数設計（引数の設計） | あると望ましい | [関数設計](../../../clean-code-principles/docs/01-practices/01-functions.md) |

---

## 1. Builder パターンの本質 -- なぜ段階的構築が必要なのか

### 1.1 解決する問題: Telescoping Constructor

ソフトウェア開発でオブジェクトの属性が増えるにつれ、コンストラクタの引数が膨大になる「**Telescoping Constructor（テレスコーピング・コンストラクタ）問題**」が発生する。

```typescript
// 問題: 引数が増えるとどれが何か分からない
//                  name    email      age  role   notify  theme   lang  tz
new User("Taro", "t@x.com", 30, "admin", true, "dark", "ja", "Asia/Tokyo");
//                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                                  6番目以降が何を表すか、読み手には分からない

// さらに問題: オプション引数がある場合
new User("Taro", "t@x.com", 30, undefined, true, undefined, "ja");
//                                ^^^^^^^^         ^^^^^^^^
//                                何を省略したのか不明
```

**WHY -- なぜこれが深刻な問題なのか?**

```
1. 可読性の破壊
   new User("Taro", "t@x.com", 30, "admin", true, "dark")
   → 3番目は年齢？ ID？ スコア？ コードを読むたびにクラス定義を参照する必要がある

2. 引数の順序ミス（サイレントバグ）
   new Config(8080, 3000)  ← port と timeout のどちらが先？
   → 型が同じ（number）なのでコンパイラも検出できない

3. オプション引数の表現力不足
   new User("Taro", "t@x.com", undefined, undefined, true, undefined, "ja")
   → undefined の連鎖は読みにくく、バグの温床

4. 不変条件の維持が困難
   コンストラクタの引数が多いと、バリデーションロジックが複雑化する
```

### 1.2 Builder による解決

```typescript
// Builder なら自己文書化される
const user = User.builder()
  .setName("Taro")
  .setEmail("t@x.com")
  .setAge(30)
  .setRole("admin")
  .enableNotifications(true)
  .setTheme("dark")
  .setLanguage("ja")
  .setTimezone("Asia/Tokyo")
  .build();

// 利点:
// 1. 各値の意味が明確（自己文書化）
// 2. 順序に依存しない
// 3. オプション値は省略するだけでデフォルト値が適用
// 4. build() でバリデーションを集約できる
```

### 1.3 Builder パターンの定義

GoF の定義:

> **複雑なオブジェクトの構築を表現から分離し、同じ構築プロセスで異なる表現を作成できるようにする。**

この定義には2つの意図がある:

1. **構築と表現の分離**: オブジェクトの組み立て手順（HOW）と、最終的な形（WHAT）を分離
2. **同じプロセスで異なる結果**: 同じ手順で HTML / Markdown / PDF など異なる出力を生成

---

## 2. Builder の構造

### 2.1 UML クラス図

```
+------------+       +-----------------+
|  Director  |------>|  Builder        |
+------------+       |  (interface)    |
| + construct|       +-----------------+
+------------+       | + setPartA()    |
                     | + setPartB()    |
                     | + setPartC()    |
                     | + build(): Product
                     +-----------------+
                             ^
                      +------+------+
                      |             |
              +-----------+  +-----------+
              |ConcreteB1 |  |ConcreteB2 |
              +-----------+  +-----------+
              | - product |  | - product |
              | + build() |  | + build() |
              +-----------+  +-----------+
                    |              |
                    v              v
              +-----------+  +-----------+
              | ProductA  |  | ProductB  |
              +-----------+  +-----------+
```

### 2.2 シーケンス図

```
Client         Director         Builder          Product
  |               |                |                |
  | construct()   |                |                |
  |-------------->|                |                |
  |               | setPartA()    |                |
  |               |--------------->|                |
  |               | setPartB()    |                |
  |               |--------------->|                |
  |               | setPartC()    |                |
  |               |--------------->|                |
  |               |                |                |
  |               | build()       |                |
  |               |--------------->| new Product() |
  |               |                |--------------->|
  |               |                |                |
  |               |<-- product ---|                |
  |<-- product ---|                |                |
```

### 2.3 Fluent Builder の内部動作

```
Fluent Builder の核心:
各 setter が this を返すことで、メソッドチェーンを実現する。

HttpRequest.builder("POST", "/api")  ← Builder を生成
  .setHeader("Content-Type", "json") ← this を返す → 次のメソッド呼出可能
  .setBody('{"name": "Taro"}')       ← this を返す → 次のメソッド呼出可能
  .setTimeout(5000)                  ← this を返す → 次のメソッド呼出可能
  .build()                           ← Product を生成して返す

内部状態の変化:
┌────────────────────────────────────┐
│ Builder 内部                        │
│                                    │
│ Step 1: method = "POST"            │
│         url = "/api"               │
│         headers = {}               │
│         body = undefined           │
│         timeout = 30000 (default)  │
│                                    │
│ Step 2: headers = {"Content-Type": "json"} │
│                                    │
│ Step 3: body = '{"name": "Taro"}' │
│                                    │
│ Step 4: timeout = 5000             │
│                                    │
│ Step 5 (build): → new HttpRequest  │
│   全フィールドを Product にコピー    │
│   バリデーション実行                │
│   Product は不変（readonly）       │
└────────────────────────────────────┘
```

---

## 3. コード例

### コード例 1: Fluent Builder（TypeScript）

最も一般的な Builder 実装。実務で圧倒的に多用される。

```typescript
class HttpRequest {
  readonly method: string;
  readonly url: string;
  readonly headers: Readonly<Record<string, string>>;
  readonly body?: string;
  readonly timeout: number;
  readonly retries: number;

  private constructor(builder: HttpRequestBuilder) {
    this.method  = builder.method;
    this.url     = builder.url;
    this.headers = Object.freeze({ ...builder.headers });
    this.body    = builder.body;
    this.timeout = builder.timeout;
    this.retries = builder.retries;
  }

  static builder(method: string, url: string): HttpRequestBuilder {
    return new HttpRequestBuilder(method, url);
  }

  toString(): string {
    return `${this.method} ${this.url} (timeout=${this.timeout}ms, retries=${this.retries})`;
  }
}

class HttpRequestBuilder {
  method: string;
  url: string;
  headers: Record<string, string> = {};
  body?: string;
  timeout: number = 30_000;
  retries: number = 3;

  constructor(method: string, url: string) {
    this.method = method;
    this.url = url;
  }

  setHeader(key: string, value: string): this {
    this.headers[key] = value;
    return this;  // this を返すことでメソッドチェーンを実現
  }

  setBody(body: string): this {
    this.body = body;
    return this;
  }

  setTimeout(ms: number): this {
    if (ms <= 0) throw new Error("Timeout must be positive");
    this.timeout = ms;
    return this;
  }

  setRetries(count: number): this {
    if (count < 0) throw new Error("Retries must be non-negative");
    this.retries = count;
    return this;
  }

  build(): HttpRequest {
    // バリデーション
    if (!this.method) throw new Error("Method is required");
    if (!this.url) throw new Error("URL is required");

    return new (HttpRequest as any)(this);
  }
}

// 使用
const req = HttpRequest.builder("POST", "/api/users")
  .setHeader("Content-Type", "application/json")
  .setHeader("Authorization", "Bearer token123")
  .setBody(JSON.stringify({ name: "Taro", age: 30 }))
  .setTimeout(5000)
  .setRetries(2)
  .build();

console.log(req.toString());
// POST /api/users (timeout=5000ms, retries=2)
console.log(req.headers);
// { "Content-Type": "application/json", "Authorization": "Bearer token123" }
```

### コード例 2: Director パターン

Director は構築手順をカプセル化し、再利用可能にする。

```typescript
interface QueryBuilder {
  select(columns: string): this;
  from(table: string): this;
  where(condition: string): this;
  orderBy(column: string, direction: "ASC" | "DESC"): this;
  limit(count: number): this;
  offset(count: number): this;
  build(): string;
}

class SQLQueryBuilder implements QueryBuilder {
  private parts = {
    select: "",
    from: "",
    where: [] as string[],
    orderBy: "",
    limit: 0,
    offset: 0,
  };

  select(columns: string): this {
    this.parts.select = columns;
    return this;
  }
  from(table: string): this {
    this.parts.from = table;
    return this;
  }
  where(condition: string): this {
    this.parts.where.push(condition);
    return this;
  }
  orderBy(column: string, direction: "ASC" | "DESC"): this {
    this.parts.orderBy = `${column} ${direction}`;
    return this;
  }
  limit(count: number): this {
    this.parts.limit = count;
    return this;
  }
  offset(count: number): this {
    this.parts.offset = count;
    return this;
  }

  build(): string {
    let sql = `SELECT ${this.parts.select} FROM ${this.parts.from}`;
    if (this.parts.where.length > 0) {
      sql += ` WHERE ${this.parts.where.join(" AND ")}`;
    }
    if (this.parts.orderBy) sql += ` ORDER BY ${this.parts.orderBy}`;
    if (this.parts.limit) sql += ` LIMIT ${this.parts.limit}`;
    if (this.parts.offset) sql += ` OFFSET ${this.parts.offset}`;
    return sql;
  }
}

// Director: 構築手順を再利用可能にカプセル化
class QueryDirector {
  buildPaginatedQuery(builder: QueryBuilder, table: string, page: number, size: number): string {
    return builder
      .select("*")
      .from(table)
      .where("active = true")
      .orderBy("created_at", "DESC")
      .limit(size)
      .offset((page - 1) * size)
      .build();
  }

  buildCountQuery(builder: QueryBuilder, table: string): string {
    return builder
      .select("COUNT(*) as total")
      .from(table)
      .where("active = true")
      .build();
  }

  buildSearchQuery(builder: QueryBuilder, table: string, keyword: string): string {
    return builder
      .select("id, name, description")
      .from(table)
      .where(`name LIKE '%${keyword}%'`)
      .where("active = true")
      .orderBy("name", "ASC")
      .limit(50)
      .build();
  }
}

// 使用
const director = new QueryDirector();

const listQuery = director.buildPaginatedQuery(new SQLQueryBuilder(), "users", 2, 20);
console.log(listQuery);
// SELECT * FROM users WHERE active = true ORDER BY created_at DESC LIMIT 20 OFFSET 20

const countQuery = director.buildCountQuery(new SQLQueryBuilder(), "users");
console.log(countQuery);
// SELECT COUNT(*) as total FROM users WHERE active = true
```

### コード例 3: 不変オブジェクトの構築

```typescript
interface UserConfig {
  readonly name: string;
  readonly email: string;
  readonly role: "admin" | "editor" | "viewer";
  readonly notifications: boolean;
  readonly language: string;
  readonly theme: "light" | "dark" | "system";
}

class UserConfigBuilder {
  private config: Partial<UserConfig> = {};

  setName(name: string): this {
    if (name.trim().length === 0) throw new Error("Name cannot be empty");
    this.config.name = name.trim();
    return this;
  }

  setEmail(email: string): this {
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      throw new Error(`Invalid email: ${email}`);
    }
    this.config.email = email.toLowerCase();
    return this;
  }

  setRole(role: "admin" | "editor" | "viewer"): this {
    this.config.role = role;
    return this;
  }

  enableNotifications(flag: boolean): this {
    this.config.notifications = flag;
    return this;
  }

  setLanguage(lang: string): this {
    this.config.language = lang;
    return this;
  }

  setTheme(theme: "light" | "dark" | "system"): this {
    this.config.theme = theme;
    return this;
  }

  build(): UserConfig {
    // 必須フィールドのバリデーション
    if (!this.config.name) throw new Error("name is required");
    if (!this.config.email) throw new Error("email is required");

    // デフォルト値の適用 + Object.freeze で不変性を保証
    return Object.freeze({
      name: this.config.name,
      email: this.config.email,
      role: this.config.role ?? "viewer",
      notifications: this.config.notifications ?? true,
      language: this.config.language ?? "ja",
      theme: this.config.theme ?? "system",
    });
  }
}

// 使用
const config = new UserConfigBuilder()
  .setName("太郎")
  .setEmail("taro@example.com")
  .setRole("admin")
  .setTheme("dark")
  .build();

console.log(config);
// { name: "太郎", email: "taro@example.com", role: "admin",
//   notifications: true, language: "ja", theme: "dark" }

// config.name = "次郎"; // Error: Cannot assign to read only property
```

### コード例 4: Step Builder（型安全な必須フィールド保証）

コンパイル時に必須フィールドの設定順序を強制する高度なパターン。

```typescript
// Step 1: name が必須
interface NeedsName {
  setName(name: string): NeedsEmail;
}
// Step 2: email が必須
interface NeedsEmail {
  setEmail(email: string): OptionalFields;
}
// Step 3: 任意フィールド + build
interface OptionalFields {
  setAge(age: number): OptionalFields;
  setRole(role: string): OptionalFields;
  setLanguage(lang: string): OptionalFields;
  build(): Person;
}

interface Person {
  name: string;
  email: string;
  age?: number;
  role?: string;
  language?: string;
}

class PersonBuilder implements NeedsName, NeedsEmail, OptionalFields {
  private name!: string;
  private email!: string;
  private age?: number;
  private role?: string;
  private language?: string;

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

  setRole(role: string): OptionalFields {
    this.role = role;
    return this;
  }

  setLanguage(lang: string): OptionalFields {
    this.language = lang;
    return this;
  }

  build(): Person {
    return {
      name: this.name,
      email: this.email,
      age: this.age,
      role: this.role,
      language: this.language,
    };
  }
}

// コンパイル時に順序が強制される
const person = PersonBuilder.create()
  .setName("Taro")           // NeedsEmail が返る → setEmail 以外呼べない
  .setEmail("t@example.com") // OptionalFields が返る → 任意フィールドと build が呼べる
  .setAge(30)
  .setRole("engineer")
  .build();

// コンパイルエラーの例:
// PersonBuilder.create().setAge(30)           // Error: setAge は NeedsName に存在しない
// PersonBuilder.create().setName("A").build() // Error: build は NeedsEmail に存在しない
```

**WHY -- Step Builder が必要な場面:**

```
通常の Fluent Builder の問題:
  new UserBuilder()
    .setAge(30)          // name を設定し忘れ
    .setRole("admin")    // email を設定し忘れ
    .build()             // 実行時エラー（build内のバリデーションで検出）

Step Builder なら:
  PersonBuilder.create()
    .setAge(30)          // コンパイルエラー！NeedsName に setAge は存在しない
    → 実行前に問題を検出できる
```

### コード例 5: Python -- Builder with dataclass

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class Pizza:
    """不変の Pizza オブジェクト"""
    size: str
    crust: str
    cheese: bool
    pepperoni: bool
    mushrooms: bool
    extra_toppings: tuple[str, ...]  # tuple で不変性を保証

    def description(self) -> str:
        toppings = []
        if self.cheese: toppings.append("チーズ")
        if self.pepperoni: toppings.append("ペパロニ")
        if self.mushrooms: toppings.append("マッシュルーム")
        toppings.extend(self.extra_toppings)
        return f"{self.size} {self.crust} ピザ: {', '.join(toppings)}"

class PizzaBuilder:
    """Pizza の段階的構築"""
    def __init__(self, size: str):
        self._size = size
        self._crust = "レギュラー"
        self._cheese = False
        self._pepperoni = False
        self._mushrooms = False
        self._extra: list[str] = []

    def crust(self, crust: str) -> "PizzaBuilder":
        self._crust = crust
        return self

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
            size=self._size,
            crust=self._crust,
            cheese=self._cheese,
            pepperoni=self._pepperoni,
            mushrooms=self._mushrooms,
            extra_toppings=tuple(self._extra),  # list → tuple で不変化
        )

# 使用
pizza = (PizzaBuilder("Large")
    .crust("シンクラスト")
    .add_cheese()
    .add_pepperoni()
    .add_topping("オリーブ")
    .add_topping("バジル")
    .build())

print(pizza.description())
# Large シンクラスト ピザ: チーズ, ペパロニ, オリーブ, バジル
```

### コード例 6: Java -- Builder（Effective Java スタイル）

Joshua Bloch が推奨するスタイル。Builder をネストクラスとして定義する。

```java
public class NutritionFacts {
    private final int servingSize;   // 必須
    private final int servings;      // 必須
    private final int calories;      // 任意
    private final int fat;           // 任意
    private final int sodium;        // 任意
    private final int carbohydrate;  // 任意

    public static class Builder {
        // 必須パラメータ
        private final int servingSize;
        private final int servings;

        // 任意パラメータ（デフォルト値で初期化）
        private int calories = 0;
        private int fat = 0;
        private int sodium = 0;
        private int carbohydrate = 0;

        public Builder(int servingSize, int servings) {
            this.servingSize = servingSize;
            this.servings = servings;
        }

        public Builder calories(int val) {
            if (val < 0) throw new IllegalArgumentException("Calories must be non-negative");
            calories = val;
            return this;
        }

        public Builder fat(int val) { fat = val; return this; }
        public Builder sodium(int val) { sodium = val; return this; }
        public Builder carbohydrate(int val) { carbohydrate = val; return this; }

        public NutritionFacts build() {
            return new NutritionFacts(this);
        }
    }

    private NutritionFacts(Builder builder) {
        servingSize  = builder.servingSize;
        servings     = builder.servings;
        calories     = builder.calories;
        fat          = builder.fat;
        sodium       = builder.sodium;
        carbohydrate = builder.carbohydrate;
    }
}

// 使用
NutritionFacts cocaCola = new NutritionFacts.Builder(240, 8)
    .calories(100)
    .sodium(35)
    .carbohydrate(27)
    .build();
```

### コード例 7: Go -- Functional Options Pattern

Go 言語では Builder パターンの代わりに Functional Options パターンが推奨される。

```go
package main

import (
    "fmt"
    "time"
)

type Server struct {
    host         string
    port         int
    timeout      time.Duration
    maxConn      int
    enableTLS    bool
    certFile     string
}

// Option は Server の設定を変更する関数型
type Option func(*Server)

// 各オプションを関数として定義
func WithPort(port int) Option {
    return func(s *Server) {
        s.port = port
    }
}

func WithTimeout(d time.Duration) Option {
    return func(s *Server) {
        s.timeout = d
    }
}

func WithMaxConnections(max int) Option {
    return func(s *Server) {
        s.maxConn = max
    }
}

func WithTLS(certFile string) Option {
    return func(s *Server) {
        s.enableTLS = true
        s.certFile = certFile
    }
}

// コンストラクタ: 必須引数 + 可変長オプション
func NewServer(host string, opts ...Option) *Server {
    // デフォルト値
    s := &Server{
        host:    host,
        port:    8080,
        timeout: 30 * time.Second,
        maxConn: 100,
    }

    // オプションを適用
    for _, opt := range opts {
        opt(s)
    }

    return s
}

func main() {
    // デフォルト設定
    s1 := NewServer("localhost")
    fmt.Printf("Server: %s:%d\n", s1.host, s1.port)
    // Server: localhost:8080

    // カスタム設定
    s2 := NewServer("api.example.com",
        WithPort(443),
        WithTimeout(10*time.Second),
        WithMaxConnections(1000),
        WithTLS("/etc/certs/server.pem"),
    )
    fmt.Printf("Server: %s:%d (TLS=%v)\n", s2.host, s2.port, s2.enableTLS)
    // Server: api.example.com:443 (TLS=true)
}
```

**WHY -- Go で Functional Options が好まれる理由:**

```
1. Go にはメソッドチェーンの文化がない（error を返す慣習）
2. ゼロ値が有用なデフォルトとして機能する
3. 関数はファーストクラスオブジェクトなので自然
4. テストで特定のオプションだけ変更しやすい
5. 後方互換性を保ちながらオプションを追加できる
```

### コード例 8: Kotlin -- data class + copy

Kotlin では data class の copy メソッドが Builder の多くのユースケースをカバーする。

```kotlin
data class ServerConfig(
    val host: String,
    val port: Int = 8080,
    val timeout: Long = 30_000,
    val maxConnections: Int = 100,
    val enableTLS: Boolean = false,
    val certFile: String? = null,
)

// Kotlin のデフォルト引数 + 名前付き引数 = Builder 不要
val config1 = ServerConfig(
    host = "localhost",
    port = 3000,
    enableTLS = true,
    certFile = "/etc/certs/server.pem",
)

// copy で一部だけ変更（Prototype + Builder の合成）
val config2 = config1.copy(
    host = "api.example.com",
    port = 443,
)

println(config2)
// ServerConfig(host=api.example.com, port=443, timeout=30000,
//   maxConnections=100, enableTLS=true, certFile=/etc/certs/server.pem)
```

### コード例 9: TypeScript -- Generic Builder

型安全にフィールドの設定状態を追跡する高度な Builder。

```typescript
// 設定されたフィールドを型レベルで追跡
type BuilderState = {
  name: boolean;
  email: boolean;
};

class TypedBuilder<State extends BuilderState = { name: false; email: false }> {
  private data: Record<string, any> = {};

  setName(name: string): TypedBuilder<State & { name: true }> {
    this.data.name = name;
    return this as any;
  }

  setEmail(email: string): TypedBuilder<State & { email: true }> {
    this.data.email = email;
    return this as any;
  }

  setAge(age: number): TypedBuilder<State> {
    this.data.age = age;
    return this as any;
  }

  // build は name と email が両方 true の場合のみ呼べる
  build(this: TypedBuilder<{ name: true; email: true }>): Person {
    return this.data as Person;
  }
}

// OK: name と email が設定済み
const p1 = new TypedBuilder()
  .setName("Taro")
  .setEmail("t@example.com")
  .setAge(30)
  .build(); // OK

// Error: build は name: false の状態では呼べない
// new TypedBuilder().setEmail("t@example.com").build();
```

### コード例 10: 複雑なドメインオブジェクトの構築

```typescript
// 実務的な例: 電子メール構築
interface Email {
  readonly from: string;
  readonly to: string[];
  readonly cc: string[];
  readonly bcc: string[];
  readonly subject: string;
  readonly body: string;
  readonly isHtml: boolean;
  readonly attachments: ReadonlyArray<{ name: string; content: Buffer }>;
  readonly replyTo?: string;
  readonly priority: "low" | "normal" | "high";
}

class EmailBuilder {
  private from = "";
  private to: string[] = [];
  private cc: string[] = [];
  private bcc: string[] = [];
  private subject = "";
  private body = "";
  private isHtml = false;
  private attachments: { name: string; content: Buffer }[] = [];
  private replyTo?: string;
  private priority: "low" | "normal" | "high" = "normal";

  setFrom(from: string): this {
    this.from = from;
    return this;
  }

  addTo(...recipients: string[]): this {
    this.to.push(...recipients);
    return this;
  }

  addCc(...recipients: string[]): this {
    this.cc.push(...recipients);
    return this;
  }

  addBcc(...recipients: string[]): this {
    this.bcc.push(...recipients);
    return this;
  }

  setSubject(subject: string): this {
    this.subject = subject;
    return this;
  }

  setTextBody(body: string): this {
    this.body = body;
    this.isHtml = false;
    return this;
  }

  setHtmlBody(html: string): this {
    this.body = html;
    this.isHtml = true;
    return this;
  }

  addAttachment(name: string, content: Buffer): this {
    this.attachments.push({ name, content });
    return this;
  }

  setReplyTo(email: string): this {
    this.replyTo = email;
    return this;
  }

  setPriority(priority: "low" | "normal" | "high"): this {
    this.priority = priority;
    return this;
  }

  build(): Email {
    if (!this.from) throw new Error("From is required");
    if (this.to.length === 0) throw new Error("At least one recipient is required");
    if (!this.subject) throw new Error("Subject is required");
    if (!this.body) throw new Error("Body is required");

    return Object.freeze({
      from: this.from,
      to: [...this.to],
      cc: [...this.cc],
      bcc: [...this.bcc],
      subject: this.subject,
      body: this.body,
      isHtml: this.isHtml,
      attachments: Object.freeze([...this.attachments]),
      replyTo: this.replyTo,
      priority: this.priority,
    });
  }
}

// 使用
const email = new EmailBuilder()
  .setFrom("noreply@example.com")
  .addTo("user1@example.com", "user2@example.com")
  .addCc("manager@example.com")
  .setSubject("Weekly Report")
  .setHtmlBody("<h1>Weekly Report</h1><p>...</p>")
  .setPriority("high")
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
| バリデーション集約 | コンストラクタ内 | 外部 | build() 内 |
| IDEサポート（補完） | 良好 | 良好 | 優秀 |
| 導入の判断基準 | 引数 1-3個 | 全て任意 | 引数 4個以上 |

### 比較表 2: Builder vs Factory

| 観点 | Builder | Factory |
|------|---------|---------|
| 目的 | 段階的な構築 | 型の選択 |
| 返すもの | 1つの複雑なオブジェクト | さまざまな型のオブジェクト |
| メソッドチェーン | 一般的 | まれ |
| 適用場面 | 多数のオプション引数 | バリエーションの切り替え |
| 構築の柔軟性 | 高い（段階的にカスタマイズ） | 低い（事前定義された型から選択） |
| 組み合わせ | Factory が Builder を返すこともある | Builder が Factory を使うこともある |

### 比較表 3: 言語別 Builder 代替手法

| 言語 | Builder 代替 | 使い分け |
|------|-------------|---------|
| TypeScript | オブジェクトリテラル + Partial<T> | 単純なケース |
| Python | キーワード引数 + dataclass | Builder 不要なことが多い |
| Kotlin | 名前付き引数 + copy() | Builder 不要なことが多い |
| Java | Effective Java Builder | 標準的なアプローチ |
| Go | Functional Options | Go のイディオム |
| Rust | Builder derive macro | ボイラープレート削減 |
| C# | Object Initializer | Builder 不要なことが多い |

---

## 5. エッジケースと注意点

### 5.1 build() の呼び忘れ

```typescript
// BAD: build() を忘れて Builder オブジェクトを使ってしまう
const config = new ConfigBuilder()
  .setHost("localhost")
  .setPort(8080);
  // .build() を忘れた！ config は ConfigBuilder 型

server.start(config); // 型エラーまたは実行時エラー
```

**対策:**

```typescript
// 1. TypeScript の型システムで Builder と Product を明確に区別
//    → server.start() は ServerConfig 型のみ受け付ける

// 2. Step Builder で build() を最終ステップに強制

// 3. ESLint ルールで Builder 型の変数への代入を警告
```

### 5.2 Builder の再利用問題

```typescript
// BAD: 同じ Builder インスタンスを再利用
const builder = new UserBuilder().setName("Taro").setEmail("t@example.com");

const user1 = builder.setRole("admin").build();
const user2 = builder.setRole("viewer").build();

// user2 は admin? viewer? → Builder の内部状態に依存する
// builder.setRole("admin") が builder を変更しているので、
// user2 も admin になる可能性がある
```

**対策:**

```typescript
// Builder の build() 後にリセットするか、
// 毎回新しい Builder を生成する

// 方法 1: build() でリセット
build(): User {
  const user = new User(this);
  this.reset(); // 内部状態をリセット
  return user;
}

// 方法 2: 毎回新しい Builder（推奨）
const user1 = User.builder().setName("Taro").setRole("admin").build();
const user2 = User.builder().setName("Taro").setRole("viewer").build();
```

### 5.3 スレッドセーフティ

```
マルチスレッド環境で Builder を共有すると問題が発生する:

Thread A: builder.setName("Taro")
Thread B: builder.setName("Jiro")   ← 競合！
Thread A: builder.build()            ← Jiro が返る可能性

対策:
1. Builder は共有せず、各スレッドで新規生成する（推奨）
2. Builder のメソッドを synchronized にする（非推奨: パフォーマンス低下）
3. Immutable Builder: 各メソッドが新しい Builder を返す
```

---

## 6. アンチパターン

### アンチパターン 1: Builder の build() を呼び忘れる

```typescript
// NG: build() を忘れて Builder オブジェクトを使ってしまう
const config = new ConfigBuilder()
  .setHost("localhost")
  .setPort(8080);
  // .build() を忘れた！

server.start(config); // 型エラーまたは実行時エラー
```

**改善**: TypeScript の型システムで Builder と Product を明確に区別する。Step Builder を使えばコンパイルエラーにできる。

### アンチパターン 2: Builder にビジネスロジックを詰める

```typescript
// NG: Builder が構築以外の責任を持つ
class OrderBuilder {
  private items: CartItem[] = [];
  private coupon?: string;

  addItem(item: CartItem): this {
    this.items.push(item);
    return this;
  }

  setCoupon(code: string): this {
    this.coupon = code;
    return this;
  }

  build(): Order {
    const order = new Order(this.items);

    // NG: ビジネスロジック
    if (this.coupon) {
      const discount = validateCoupon(this.coupon); // クーポン検証
      order.applyDiscount(discount);                 // 割引適用
    }
    order.calculateTax();       // 税計算
    order.calculateShipping();  // 送料計算
    sendNotification(order);    // 通知送信（副作用）

    return order;
  }
}
```

```typescript
// OK: Builder は構築のみ。ビジネスロジックはドメインサービスに委譲
class OrderBuilder {
  private items: CartItem[] = [];

  addItem(item: CartItem): this {
    this.items.push(item);
    return this;
  }

  build(): Order {
    if (this.items.length === 0) throw new Error("Order must have items");
    return new Order([...this.items]); // 構築のみ
  }
}

// ビジネスロジックはサービスに委譲
class OrderService {
  async processOrder(order: Order, couponCode?: string): Promise<Order> {
    if (couponCode) {
      const discount = await this.couponService.validate(couponCode);
      order.applyDiscount(discount);
    }
    order.tax = this.taxService.calculate(order);
    await this.notificationService.send(order);
    return order;
  }
}
```

### アンチパターン 3: 単純なオブジェクトに Builder を使う

```typescript
// NG: フィールドが2つしかないのに Builder を使う
class PointBuilder {
  private x = 0;
  private y = 0;

  setX(x: number): this { this.x = x; return this; }
  setY(y: number): this { this.y = y; return this; }
  build(): Point { return new Point(this.x, this.y); }
}

// 20行のコードが、やりたいことに対して過剰
```

```typescript
// OK: コンストラクタで十分
class Point {
  constructor(public readonly x: number, public readonly y: number) {}
}

const p = new Point(10, 20);
```

**判断基準**: 引数が4つ未満、オプション引数が2つ未満なら、Builder は過剰設計。

---

## 7. トレードオフ分析

```
利点                              欠点
+------------------------------+  +------------------------------+
| 自己文書化された構築コード    |  | ボイラープレートコードが多い  |
| 不変オブジェクトの安全な構築  |  | クラス数の増加                |
| バリデーションの集約          |  | 単純なケースには過剰設計      |
| 順序非依存のフィールド設定    |  | build() の呼び忘れリスク      |
| IDEの優れたオートコンプリート  |  | Builder と Product の同期維持  |
| テストでの柔軟な構築          |  | 言語によっては不要（Kotlin等） |
+------------------------------+  +------------------------------+
```

---

## 8. 実践演習

### 演習 1: 基礎 -- HTTP レスポンス Builder

**課題**: HTTP レスポンスオブジェクトを構築する Builder を実装してください。

**要件**:
- ステータスコード（必須）、ヘッダー、ボディ、Content-Type を設定可能
- Fluent API（メソッドチェーン）
- build() でバリデーション（ステータスコードが 100-599 の範囲内か）
- 生成されたレスポンスは不変

```typescript
// === あなたの実装をここに書いてください ===
```

**期待される出力**:

```
const res = new HttpResponseBuilder(200)
  .setHeader("Content-Type", "application/json")
  .setHeader("Cache-Control", "no-cache")
  .setBody(JSON.stringify({ message: "OK" }))
  .build();

console.log(res.statusCode);  // 200
console.log(res.headers);     // { "Content-Type": "application/json", "Cache-Control": "no-cache" }
console.log(res.body);        // '{"message":"OK"}'
```

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
interface HttpResponse {
  readonly statusCode: number;
  readonly headers: Readonly<Record<string, string>>;
  readonly body?: string;
}

class HttpResponseBuilder {
  private headers: Record<string, string> = {};
  private body?: string;

  constructor(private statusCode: number) {}

  setHeader(key: string, value: string): this {
    this.headers[key] = value;
    return this;
  }

  setBody(body: string): this {
    this.body = body;
    return this;
  }

  setJsonBody(data: unknown): this {
    this.headers["Content-Type"] = "application/json";
    this.body = JSON.stringify(data);
    return this;
  }

  build(): HttpResponse {
    if (this.statusCode < 100 || this.statusCode > 599) {
      throw new Error(`Invalid status code: ${this.statusCode}`);
    }

    return Object.freeze({
      statusCode: this.statusCode,
      headers: Object.freeze({ ...this.headers }),
      body: this.body,
    });
  }
}
```

</details>

### 演習 2: 応用 -- SQL Query Builder with Director

**課題**: SQL クエリを構築する Builder と、よく使うクエリパターンを Director として実装してください。

**要件**:
- SELECT / FROM / WHERE / JOIN / ORDER BY / LIMIT / OFFSET をサポート
- WHERE 条件は複数指定可能（AND 結合）
- Director に「ページネーション付き一覧取得」「件数取得」「検索」の3つの構築パターン
- パラメータバインディング（SQLインジェクション対策）

```typescript
// === あなたの実装をここに書いてください ===
```

**期待される出力**:

```
const director = new QueryDirector();

const listQuery = director.buildPaginatedList(new SQLBuilder(), "users", 2, 20);
console.log(listQuery.sql);
// SELECT * FROM users WHERE active = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3
console.log(listQuery.params);
// [true, 20, 20]
```

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
interface Query {
  sql: string;
  params: any[];
}

class SQLBuilder {
  private selectClause = "*";
  private fromClause = "";
  private whereConditions: { condition: string; param?: any }[] = [];
  private joinClauses: string[] = [];
  private orderByClause = "";
  private limitValue?: number;
  private offsetValue?: number;
  private paramIndex = 1;

  select(columns: string): this {
    this.selectClause = columns;
    return this;
  }

  from(table: string): this {
    this.fromClause = table;
    return this;
  }

  where(condition: string, param?: any): this {
    const paramPlaceholder = param !== undefined ? `$${this.paramIndex++}` : "";
    this.whereConditions.push({
      condition: condition.replace("?", paramPlaceholder),
      param,
    });
    return this;
  }

  join(joinClause: string): this {
    this.joinClauses.push(joinClause);
    return this;
  }

  orderBy(column: string, direction: "ASC" | "DESC" = "ASC"): this {
    this.orderByClause = `${column} ${direction}`;
    return this;
  }

  limit(count: number): this {
    this.limitValue = count;
    return this;
  }

  offset(count: number): this {
    this.offsetValue = count;
    return this;
  }

  build(): Query {
    if (!this.fromClause) throw new Error("FROM clause is required");

    const params: any[] = [];
    let sql = `SELECT ${this.selectClause} FROM ${this.fromClause}`;

    for (const join of this.joinClauses) {
      sql += ` ${join}`;
    }

    if (this.whereConditions.length > 0) {
      const conditions = this.whereConditions.map(w => {
        if (w.param !== undefined) params.push(w.param);
        return w.condition;
      });
      sql += ` WHERE ${conditions.join(" AND ")}`;
    }

    if (this.orderByClause) sql += ` ORDER BY ${this.orderByClause}`;

    if (this.limitValue !== undefined) {
      sql += ` LIMIT $${this.paramIndex++}`;
      params.push(this.limitValue);
    }
    if (this.offsetValue !== undefined) {
      sql += ` OFFSET $${this.paramIndex++}`;
      params.push(this.offsetValue);
    }

    return { sql, params };
  }
}

class QueryDirector {
  buildPaginatedList(builder: SQLBuilder, table: string, page: number, size: number): Query {
    return builder
      .select("*")
      .from(table)
      .where("active = ?", true)
      .orderBy("created_at", "DESC")
      .limit(size)
      .offset((page - 1) * size)
      .build();
  }

  buildCount(builder: SQLBuilder, table: string): Query {
    return builder
      .select("COUNT(*) as total")
      .from(table)
      .where("active = ?", true)
      .build();
  }

  buildSearch(builder: SQLBuilder, table: string, keyword: string): Query {
    return builder
      .select("id, name, description")
      .from(table)
      .where("name ILIKE ?", `%${keyword}%`)
      .where("active = ?", true)
      .orderBy("name", "ASC")
      .limit(50)
      .build();
  }
}
```

</details>

### 演習 3: 発展 -- Step Builder with 条件分岐

**課題**: ユーザー種別（個人/法人）によって必須フィールドが異なる Step Builder を設計してください。

**要件**:
- 個人ユーザー: name, email が必須。age はオプション
- 法人ユーザー: companyName, email, registrationNumber が必須
- 種別選択後、該当する必須フィールドのみが要求される（型安全）
- build() は全ての必須フィールドが設定された場合のみ呼び出し可能

```typescript
// === あなたの実装をここに書いてください ===
```

**期待される出力**:

```
// 個人ユーザー
const individual = UserBuilder.create()
  .asIndividual()            // → NeedsIndividualName が返る
  .setName("太郎")           // → NeedsIndividualEmail が返る
  .setEmail("taro@test.com") // → IndividualOptional が返る
  .setAge(30)
  .build();

// 法人ユーザー
const corporate = UserBuilder.create()
  .asCorporate()                       // → NeedsCorporateName が返る
  .setCompanyName("株式会社Example")    // → NeedsCorporateEmail が返る
  .setEmail("info@example.co.jp")      // → NeedsCorporateRegNum が返る
  .setRegistrationNumber("1234567890") // → CorporateOptional が返る
  .build();
```

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
// 種別選択
interface SelectType {
  asIndividual(): NeedsIndividualName;
  asCorporate(): NeedsCorporateName;
}

// 個人ユーザーのステップ
interface NeedsIndividualName {
  setName(name: string): NeedsIndividualEmail;
}
interface NeedsIndividualEmail {
  setEmail(email: string): IndividualOptional;
}
interface IndividualOptional {
  setAge(age: number): IndividualOptional;
  setPhone(phone: string): IndividualOptional;
  build(): User;
}

// 法人ユーザーのステップ
interface NeedsCorporateName {
  setCompanyName(name: string): NeedsCorporateEmail;
}
interface NeedsCorporateEmail {
  setEmail(email: string): NeedsCorporateRegNum;
}
interface NeedsCorporateRegNum {
  setRegistrationNumber(num: string): CorporateOptional;
}
interface CorporateOptional {
  setRepresentative(name: string): CorporateOptional;
  build(): User;
}

type UserType = "individual" | "corporate";

interface User {
  type: UserType;
  name?: string;
  companyName?: string;
  email: string;
  age?: number;
  phone?: string;
  registrationNumber?: string;
  representative?: string;
}

class UserBuilder implements
  SelectType,
  NeedsIndividualName, NeedsIndividualEmail, IndividualOptional,
  NeedsCorporateName, NeedsCorporateEmail, NeedsCorporateRegNum, CorporateOptional
{
  private data: Partial<User> = {};

  static create(): SelectType {
    return new UserBuilder();
  }

  asIndividual(): NeedsIndividualName {
    this.data.type = "individual";
    return this;
  }

  asCorporate(): NeedsCorporateName {
    this.data.type = "corporate";
    return this;
  }

  setName(name: string): NeedsIndividualEmail {
    this.data.name = name;
    return this;
  }

  setCompanyName(name: string): NeedsCorporateEmail {
    this.data.companyName = name;
    return this;
  }

  setEmail(email: string): any {
    this.data.email = email;
    return this;
  }

  setRegistrationNumber(num: string): CorporateOptional {
    this.data.registrationNumber = num;
    return this;
  }

  setAge(age: number): IndividualOptional {
    this.data.age = age;
    return this;
  }

  setPhone(phone: string): IndividualOptional {
    this.data.phone = phone;
    return this;
  }

  setRepresentative(name: string): CorporateOptional {
    this.data.representative = name;
    return this;
  }

  build(): User {
    return Object.freeze(this.data) as User;
  }
}
```

</details>

---

## 9. FAQ

### Q1: Builder はどの程度の複雑さから導入すべきですか？

目安として、**コンストラクタの引数が 4 つ以上**、または**オプション引数が 2 つ以上**ある場合に Builder の導入を検討します。引数が 2-3 個であればコンストラクタで十分です。

ただし、以下の場合は引数が少なくても Builder を検討する価値があります:
- 同じ型の引数が複数ある（順序ミスのリスク）
- 構築に複数ステップが必要
- 構築ロジックの再利用が必要（Director）

### Q2: Lombok の @Builder のような自動生成は推奨ですか？

はい。ボイラープレートコードの削減は生産性に直結します。

| 言語/ツール | 自動生成方法 |
|------------|-------------|
| Java | Lombok @Builder |
| Kotlin | data class + copy |
| TypeScript | クラス + ジェネリクスで半自動 |
| Python | dataclasses + Pydantic |
| Rust | derive_builder crate |

### Q3: Builder と Named Arguments（キーワード引数）の違いは？

Python や Kotlin のキーワード引数は Builder の多くのメリットを提供します。しかし、以下の場合は Builder が優位です:

| 観点 | キーワード引数 | Builder |
|------|:---:|:---:|
| 段階的構築 | No | Yes |
| バリデーション集約 | No | Yes (build()内) |
| 構築ロジックの再利用 | No | Yes (Director) |
| 不変性の保証 | 言語依存 | 明示的に制御可能 |
| コード量 | 少ない | 多い |

### Q4: Director は必要ですか？

多くの実務では Director なしの Fluent Builder で十分です。Director が有効なのは:

1. 同じ構築パターンが複数箇所で使われる場合
2. 構築手順自体がドメイン知識を表現している場合
3. テストで定型的なオブジェクトを繰り返し構築する場合

### Q5: Builder と Prototype パターンはどう使い分けますか？

```
Builder: ゼロからオブジェクトを構築する
  → 構築パラメータが多い場合に有効

Prototype: 既存オブジェクトをコピーして一部を変更する
  → ベースとなるオブジェクトが存在する場合に有効

Kotlin の copy() は両方を組み合わせている:
  val base = Config(host = "localhost", port = 8080, ...)
  val prod = base.copy(host = "api.example.com", port = 443)
```

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | 複雑なオブジェクトを段階的に構築 |
| 本質的な問題 | Telescoping Constructor の解決と不変オブジェクトの安全な構築 |
| Fluent API | メソッドチェーンで可読性向上。最も一般的な実装 |
| Step Builder | 型安全に必須フィールドを保証。コンパイル時に検証 |
| Director | 構築手順を再利用可能にカプセル化 |
| 判断基準 | 引数 4+ またはオプション 2+ で検討 |
| 言語別推奨 | Java: Effective Java スタイル / Go: Functional Options / Kotlin: data class |
| 最大の注意点 | Builder にビジネスロジックを入れない。build() の呼び忘れに注意 |

---

## 次に読むべきガイド

- [Prototype パターン](./03-prototype.md) -- クローンによる生成。Builder が「ゼロから構築」、Prototype が「既存からコピー」
- [Factory パターン](./01-factory.md) -- Factory が Builder を返すパターン
- [Decorator パターン](../01-structural/01-decorator.md) -- 動的な機能追加。Builder の構築結果を装飾
- [関数設計](../../../clean-code-principles/docs/01-practices/01-functions.md) -- 引数設計のベストプラクティス
- [不変性](../../../clean-code-principles/docs/03-practices-advanced/00-immutability.md) -- イミュータブルデータ構造の設計

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley. -- Builder パターンの原典
2. Bloch, J. (2018). *Effective Java* (3rd ed.). Addison-Wesley. Item 2: Consider a builder when faced with many constructor parameters.
3. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media.
4. Refactoring.Guru -- Builder. https://refactoring.guru/design-patterns/builder
5. Dave Cheney (2014). Functional options for friendly APIs. https://dave.cheney.net/2014/10/17/functional-options-for-friendly-apis
6. Martin, R.C. (2008). *Clean Code*. Prentice Hall. Chapter 3: Functions -- 引数の数を最小化する
