# TypeScript ビルダーパターン

> 型安全なビルダーパターンと Fluent API で、複雑なオブジェクト構築をコンパイル時に検証する

## この章で学ぶこと

1. **クラシックビルダー** -- 段階的にオブジェクトを構築し、不完全な状態でのビルドをコンパイルエラーにする技法
2. **Phantom Type ビルダー** -- ジェネリクスのフラグ型で「設定済み/未設定」を追跡する型レベルステートマシン
3. **Fluent API 設計** -- メソッドチェーンの型推論を最大限活用し、IDE 補完と型安全性を両立する方法

---

## 1. クラシックビルダーパターン

### 1-1. 基本構造

```
+-----------+     .setName()     +-------------+
|  Builder  | -----------------> |   Builder   |
| (空の状態) |                    | (name設定済) |
+-----------+                    +-------------+
                                       |
                                  .setEmail()
                                       |
                                       v
                                 +-------------+
                                 |   Builder   |
                                 | (両方設定済) |
                                 +-------------+
                                       |
                                   .build()
                                       |
                                       v
                                 +-------------+
                                 |   User      |
                                 | (完成品)     |
                                 +-------------+
```

```typescript
// 構築対象
interface HttpRequest {
  readonly method: "GET" | "POST" | "PUT" | "DELETE";
  readonly url: string;
  readonly headers: Record<string, string>;
  readonly body?: string;
  readonly timeout: number;
}

// ビルダー
class HttpRequestBuilder {
  private method: HttpRequest["method"] = "GET";
  private url = "";
  private headers: Record<string, string> = {};
  private body?: string;
  private timeout = 30000;

  setMethod(method: HttpRequest["method"]): this {
    this.method = method;
    return this;
  }

  setUrl(url: string): this {
    this.url = url;
    return this;
  }

  addHeader(key: string, value: string): this {
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
    if (!this.url) throw new Error("URL is required");
    return {
      method: this.method,
      url: this.url,
      headers: { ...this.headers },
      body: this.body,
      timeout: this.timeout,
    };
  }
}

// 使用例
const request = new HttpRequestBuilder()
  .setMethod("POST")
  .setUrl("https://api.example.com/users")
  .addHeader("Content-Type", "application/json")
  .setBody(JSON.stringify({ name: "Alice" }))
  .setTimeout(5000)
  .build();
```

### 1-2. 問題点 -- ランタイムエラー

```typescript
// URL を設定し忘れてもコンパイルは通る
const bad = new HttpRequestBuilder()
  .setMethod("POST")
  .build(); // ランタイムエラー: "URL is required"
```

---

## 2. Phantom Type ビルダー（型安全版）

### 2-1. フラグ型による状態追跡

```
型パラメータ: Builder<HasUrl, HasMethod>

  Builder<false, false>  -- 初期状態
       |
   .url("...")
       |
       v
  Builder<true, false>   -- URL 設定済み
       |
   .method("POST")
       |
       v
  Builder<true, true>    -- 全て設定済み
       |
   .build()  <-- この型の時だけ呼べる
       |
       v
    HttpRequest
```

```typescript
// フラグ型
type True = true;
type False = false;

// Phantom Type ビルダー
class RequestBuilder<
  HasUrl extends boolean = false,
  HasMethod extends boolean = false
> {
  private _url = "";
  private _method: "GET" | "POST" | "PUT" | "DELETE" = "GET";
  private _headers: Record<string, string> = {};
  private _body?: string;

  url(url: string): RequestBuilder<True, HasMethod> {
    this._url = url;
    return this as unknown as RequestBuilder<True, HasMethod>;
  }

  method(
    method: "GET" | "POST" | "PUT" | "DELETE"
  ): RequestBuilder<HasUrl, True> {
    this._method = method;
    return this as unknown as RequestBuilder<HasUrl, True>;
  }

  header(key: string, value: string): this {
    this._headers[key] = value;
    return this;
  }

  body(body: string): this {
    this._body = body;
    return this;
  }

  // build() は HasUrl=true かつ HasMethod=true の時のみ呼び出し可能
  build(this: RequestBuilder<True, True>): HttpRequest {
    return {
      method: this._method,
      url: this._url,
      headers: { ...this._headers },
      body: this._body,
      timeout: 30000,
    };
  }
}

// OK: 両方設定済み
const req = new RequestBuilder()
  .url("https://api.example.com")
  .method("POST")
  .header("Authorization", "Bearer token")
  .build(); // コンパイル OK

// NG: URL 未設定 -- コンパイルエラー
const bad = new RequestBuilder()
  .method("POST")
  .build();
// Error: The 'this' context of type 'RequestBuilder<false, true>'
//        is not assignable to method's 'this' of type 'RequestBuilder<true, true>'
```

### 2-2. Step Builder（順序強制パターン）

```typescript
// 各ステップを別々のインターフェースで定義
interface NeedsUrl {
  url(url: string): NeedsMethod;
}

interface NeedsMethod {
  method(method: "GET" | "POST" | "PUT" | "DELETE"): OptionalConfig;
}

interface OptionalConfig {
  header(key: string, value: string): OptionalConfig;
  body(body: string): OptionalConfig;
  timeout(ms: number): OptionalConfig;
  build(): HttpRequest;
}

function createRequest(): NeedsUrl {
  const config: Partial<HttpRequest> = { headers: {}, timeout: 30000 };

  const optionalConfig: OptionalConfig = {
    header(key, value) {
      (config.headers as Record<string, string>)[key] = value;
      return optionalConfig;
    },
    body(body) {
      config.body = body;
      return optionalConfig;
    },
    timeout(ms) {
      config.timeout = ms;
      return optionalConfig;
    },
    build() {
      return config as HttpRequest;
    },
  };

  return {
    url(url) {
      config.url = url;
      return {
        method(method) {
          config.method = method;
          return optionalConfig;
        },
      };
    },
  };
}

// 順序が強制される
const req = createRequest()
  .url("https://api.example.com")    // 1. まず URL
  .method("POST")                     // 2. 次に method
  .header("Content-Type", "json")     // 3. 以降は自由
  .build();
```

---

## 3. Fluent API の型推論テクニック

### 3-1. Mapped Types で動的にメソッドを生成

```typescript
type QueryBuilder<T extends Record<string, unknown>> = {
  [K in keyof T & string as `where${Capitalize<K>}`]: (
    value: T[K]
  ) => QueryBuilder<T>;
} & {
  orderBy(field: keyof T, direction?: "asc" | "desc"): QueryBuilder<T>;
  limit(n: number): QueryBuilder<T>;
  execute(): Promise<T[]>;
};

// 使用例の型
interface User {
  id: number;
  name: string;
  email: string;
  age: number;
}

// QueryBuilder<User> は自動的に以下のメソッドを持つ:
// - whereId(value: number)
// - whereName(value: string)
// - whereEmail(value: string)
// - whereAge(value: number)
// - orderBy(field: keyof User, direction?)
// - limit(n: number)
// - execute(): Promise<User[]>
```

### 3-2. Template Literal Types による SQL ビルダー

```typescript
// 型レベルで SELECT 文を構築
type SelectFields<T, Fields extends (keyof T)[]> = Pick<T, Fields[number]>;

class TypedQueryBuilder<
  T extends Record<string, unknown>,
  Selected extends (keyof T)[] = []
> {
  private fields: string[] = [];
  private conditions: string[] = [];

  select<F extends keyof T>(
    ...fields: F[]
  ): TypedQueryBuilder<T, [...Selected, ...F[]]> {
    this.fields.push(...(fields as string[]));
    return this as unknown as TypedQueryBuilder<T, [...Selected, ...F[]]>;
  }

  where<K extends keyof T>(
    field: K,
    op: "=" | ">" | "<" | "!=",
    value: T[K]
  ): this {
    this.conditions.push(`${String(field)} ${op} ${JSON.stringify(value)}`);
    return this;
  }

  async execute(): Promise<SelectFields<T, Selected>[]> {
    const sql = `SELECT ${this.fields.join(", ")} FROM ...`;
    console.log(sql);
    return [] as SelectFields<T, Selected>[];
  }
}

// 使用例
const users = await new TypedQueryBuilder<User>()
  .select("name", "email")  // Selected = ["name", "email"]
  .where("age", ">", 18)
  .execute();
// users の型: Pick<User, "name" | "email">[]
```

---

## 比較表

### ビルダーパターンのバリエーション比較

| パターン | 型安全性 | 実装コスト | 柔軟性 | 順序強制 |
|---------|---------|-----------|--------|---------|
| クラシックビルダー | 低 | 低 | 高 | なし |
| Phantom Type | 高 | 中 | 中 | なし |
| Step Builder | 最高 | 高 | 低 | あり |
| 関数合成ビルダー | 高 | 中 | 高 | なし |

### ビルダー vs 他の生成パターン

| 比較軸 | ビルダー | ファクトリ | コンストラクタ | Object.assign |
|--------|---------|-----------|-------------|---------------|
| 引数の多さ | 多い場合に最適 | 少〜中 | 少〜中 | 中 |
| 段階的構築 | 可能 | 不可 | 不可 | 不可 |
| バリデーション | build() 時 | 生成時 | 即座 | なし |
| IDE 補完 | 優秀 | 良好 | 普通 | 限定的 |
| 不変性 | 保証可能 | 保証可能 | 設計次第 | 困難 |

---

## アンチパターン

### AP-1: any を使った型逃げ

```typescript
// NG: as any で型安全性を破壊
class BadBuilder {
  private config: any = {};

  set(key: string, value: any): this {
    this.config[key] = value;
    return this;
  }

  build(): HttpRequest {
    return this.config as HttpRequest; // 検証なし
  }
}

// OK: ジェネリクスで型を追跡
class GoodBuilder<T extends Partial<HttpRequest> = {}> {
  constructor(private config: T) {}

  set<K extends keyof HttpRequest>(
    key: K,
    value: HttpRequest[K]
  ): GoodBuilder<T & Pick<HttpRequest, K>> {
    return new GoodBuilder({ ...this.config, [key]: value } as T & Pick<HttpRequest, K>);
  }

  build(this: GoodBuilder<HttpRequest>): HttpRequest {
    return { ...this.config };
  }
}
```

### AP-2: 可変状態の露出

```typescript
// NG: ビルダーの内部状態が外部から変更可能
class LeakyBuilder {
  headers: Record<string, string> = {}; // public!

  build(): HttpRequest {
    return { /* ... */ headers: this.headers, /* ... */ } as HttpRequest;
    // build 後に headers を変更すると、構築済みオブジェクトも変わる
  }
}

// OK: private + コピー
class SafeBuilder {
  private headers: Record<string, string> = {};

  addHeader(key: string, value: string): this {
    this.headers[key] = value;
    return this;
  }

  build(): HttpRequest {
    return {
      /* ... */
      headers: { ...this.headers }, // スプレッドでコピー
      /* ... */
    } as HttpRequest;
  }
}
```

---

## 実践的な応用例

### テストデータビルダー

```
テストデータ生成フロー:

  UserBuilder.create()
       |
  .withDefaults()     ← 合理的なデフォルト値
       |
  .withName("Alice")  ← テストに必要な値だけ上書き
       |
  .withPosts(3)       ← リレーション生成
       |
  .build()
       |
       v
  { id: "uuid-1", name: "Alice", email: "alice@test.com",
    posts: [Post, Post, Post] }
```

```typescript
// テストデータ用ビルダー
class UserBuilder {
  private data: User = {
    id: crypto.randomUUID(),
    name: "Test User",
    email: "test@example.com",
    age: 25,
    role: "user",
    createdAt: new Date(),
  };

  static create(): UserBuilder {
    return new UserBuilder();
  }

  withName(name: string): this {
    this.data.name = name;
    return this;
  }

  withEmail(email: string): this {
    this.data.email = email;
    return this;
  }

  withAge(age: number): this {
    this.data.age = age;
    return this;
  }

  withRole(role: "user" | "admin"): this {
    this.data.role = role;
    return this;
  }

  build(): User {
    return { ...this.data };
  }

  async persist(db: Database): Promise<User> {
    const user = this.build();
    await db.users.insert(user);
    return user;
  }
}

// テストでの使用
describe("UserService", () => {
  it("should allow admin to delete users", async () => {
    const admin = UserBuilder.create().withRole("admin").build();
    const target = UserBuilder.create().withName("Bob").build();

    const result = await userService.deleteUser(admin.id, target.id);
    expect(result).toBeOk();
  });
});
```

---

## FAQ

### Q1: ビルダーパターンとファクトリパターンの使い分けは？

引数が 4 つ以上ある、または省略可能な引数が多い場合はビルダーが適しています。引数が少なく固定されている場合はファクトリで十分です。テストデータ生成はビルダーの最も効果的な適用場面です。

### Q2: Phantom Type ビルダーはパフォーマンスに影響しますか？

型パラメータはコンパイル時にのみ存在し、JavaScript 出力には一切影響しません。`as unknown as` のキャストもランタイムコストゼロです。パフォーマンスの心配は不要です。

### Q3: Immutable ビルダーと Mutable ビルダーのどちらが良いですか？

Immutable ビルダー（各メソッドで新しいインスタンスを返す）は安全ですがメモリ割り当てが増えます。Mutable ビルダー（this を返す）は効率的ですが、途中の状態を保存して分岐させる使い方ができません。一般的には Mutable で十分ですが、ビルダーを変数に保存して分岐させたい場合は Immutable を選択してください。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| クラシックビルダー | 段階的構築だが型安全性は不十分 |
| Phantom Type | ジェネリクスのフラグで設定状態を追跡 |
| Step Builder | インターフェース分割で順序を強制 |
| Fluent API | メソッドチェーンで直感的な DSL を構築 |
| テストデータビルダー | `.create().withX().build()` でテストデータを生成 |
| 不変コピー | `build()` ではスプレッドでコピーを返す |

---

## 次に読むべきガイド

- [ブランド型](./03-branded-types.md) -- ビルダーで生成する値にブランドを付与する
- [DI パターン](./04-dependency-injection.md) -- ビルダーと DI を組み合わせたファクトリ設計
- [判別共用体](./02-discriminated-unions.md) -- Step Builder の型安全性を支える判別共用体

---

## 参考文献

1. **Design Patterns: Elements of Reusable Object-Oriented Software** -- Gamma et al. (GoF)
   Builder パターンの原典

2. **TypeScript Deep Dive - Phantom Types**
   https://basarat.gitbook.io/typescript/

3. **Fluent Interface** -- Martin Fowler
   https://martinfowler.com/bliki/FluentInterface.html
