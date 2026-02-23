# TypeScript ビルダーパターン

> 型安全なビルダーパターンと Fluent API で、複雑なオブジェクト構築をコンパイル時に検証する

## この章で学ぶこと

1. **クラシックビルダー** -- 段階的にオブジェクトを構築し、不完全な状態でのビルドをコンパイルエラーにする技法
2. **Phantom Type ビルダー** -- ジェネリクスのフラグ型で「設定済み/未設定」を追跡する型レベルステートマシン
3. **Fluent API 設計** -- メソッドチェーンの型推論を最大限活用し、IDE 補完と型安全性を両立する方法
4. **Immutable ビルダー** -- 各ステップで新しいインスタンスを返し、構築途中の状態を安全に分岐させる技法
5. **テストデータビルダー** -- テストコードの可読性を劇的に向上させるファクトリビルダーの実装
6. **実務での応用** -- クエリビルダー、フォームビルダー、設定オブジェクトビルダーなどの実践的パターン

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

クラシックビルダーの最大の問題は、必須フィールドの設定忘れがコンパイル時に検出できないことです。これを型システムで解決するのが次のセクションで紹介する Phantom Type ビルダーです。

### 1-3. バリデーション付きクラシックビルダー

ランタイムでの安全性を高めるため、Result 型と組み合わせたビルダーも実装できます。

```typescript
import { Result, Ok, Err } from "./result";

class ValidatedHttpRequestBuilder {
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

  // build() が Result を返す
  build(): Result<HttpRequest, BuildError[]> {
    const errors: BuildError[] = [];

    if (!this.url) {
      errors.push({ field: "url", message: "URL is required" });
    }

    if (this.url && !this.url.startsWith("http")) {
      errors.push({ field: "url", message: "URL must start with http:// or https://" });
    }

    if (this.timeout < 0) {
      errors.push({ field: "timeout", message: "Timeout must be non-negative" });
    }

    if ((this.method === "POST" || this.method === "PUT") && !this.body) {
      errors.push({ field: "body", message: `Body is required for ${this.method} requests` });
    }

    if (errors.length > 0) {
      return Err(errors);
    }

    return Ok({
      method: this.method,
      url: this.url,
      headers: { ...this.headers },
      body: this.body,
      timeout: this.timeout,
    });
  }
}

interface BuildError {
  field: string;
  message: string;
}

// 使用例
const result = new ValidatedHttpRequestBuilder()
  .setMethod("POST")
  .setUrl("https://api.example.com")
  .build();

if (isErr(result)) {
  console.error("Build errors:", result.error);
  // [{ field: "body", message: "Body is required for POST requests" }]
} else {
  console.log("Request:", result.value);
}
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

### 2-2. 多数の必須フィールドを追跡する Phantom Type

```typescript
// フィールドごとの設定状態を追跡
interface BuilderState {
  hasUrl: boolean;
  hasMethod: boolean;
  hasAuth: boolean;
}

// デフォルト状態
type EmptyState = {
  hasUrl: false;
  hasMethod: false;
  hasAuth: false;
};

// 状態を更新するユーティリティ型
type SetField<S extends BuilderState, K extends keyof BuilderState> = {
  [P in keyof BuilderState]: P extends K ? true : S[P];
};

// すべてが true であることを確認する型
type AllSet<S extends BuilderState> = S extends {
  hasUrl: true;
  hasMethod: true;
  hasAuth: true;
}
  ? true
  : false;

class AdvancedRequestBuilder<S extends BuilderState = EmptyState> {
  private _url = "";
  private _method: "GET" | "POST" | "PUT" | "DELETE" = "GET";
  private _auth = "";
  private _headers: Record<string, string> = {};
  private _body?: string;
  private _timeout = 30000;

  url(url: string): AdvancedRequestBuilder<SetField<S, "hasUrl">> {
    this._url = url;
    return this as unknown as AdvancedRequestBuilder<SetField<S, "hasUrl">>;
  }

  method(
    method: "GET" | "POST" | "PUT" | "DELETE"
  ): AdvancedRequestBuilder<SetField<S, "hasMethod">> {
    this._method = method;
    return this as unknown as AdvancedRequestBuilder<SetField<S, "hasMethod">>;
  }

  auth(token: string): AdvancedRequestBuilder<SetField<S, "hasAuth">> {
    this._auth = token;
    return this as unknown as AdvancedRequestBuilder<SetField<S, "hasAuth">>;
  }

  header(key: string, value: string): this {
    this._headers[key] = value;
    return this;
  }

  body(body: string): this {
    this._body = body;
    return this;
  }

  timeout(ms: number): this {
    this._timeout = ms;
    return this;
  }

  // AllSet<S> が true の場合のみ build() を呼べる
  build(
    this: AdvancedRequestBuilder<{
      hasUrl: true;
      hasMethod: true;
      hasAuth: true;
    }>
  ): HttpRequest {
    return {
      method: this._method,
      url: this._url,
      headers: {
        ...this._headers,
        Authorization: `Bearer ${this._auth}`,
      },
      body: this._body,
      timeout: this._timeout,
    };
  }
}

// OK: 全フィールド設定済み
const req1 = new AdvancedRequestBuilder()
  .url("https://api.example.com")
  .method("POST")
  .auth("my-token")
  .header("Content-Type", "application/json")
  .body(JSON.stringify({ data: "value" }))
  .build(); // コンパイルOK

// NG: auth 未設定
const bad1 = new AdvancedRequestBuilder()
  .url("https://api.example.com")
  .method("POST")
  .build(); // コンパイルエラー
```

### 2-3. Step Builder（順序強制パターン）

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

### 2-4. 条件付きメソッドの型制約

```typescript
// HTTP メソッドに応じて body の有無を型で制御する
interface GetBuilder {
  header(key: string, value: string): GetBuilder;
  query(params: Record<string, string>): GetBuilder;
  build(): HttpRequest;
  // body() は存在しない -- GET リクエストにはボディなし
}

interface PostBuilder {
  header(key: string, value: string): PostBuilder;
  body(body: string): PostBuilder;
  json<T>(data: T): PostBuilder;
  build(): HttpRequest;
}

interface MethodSelector {
  get(): GetBuilder;
  post(): PostBuilder;
  put(): PostBuilder;
  delete(): GetBuilder;
}

function request(url: string): MethodSelector {
  const config: Partial<HttpRequest> & { query?: Record<string, string> } = {
    url,
    headers: {},
    timeout: 30000,
  };

  const getBuilder: GetBuilder = {
    header(key, value) {
      (config.headers as Record<string, string>)[key] = value;
      return getBuilder;
    },
    query(params) {
      config.query = params;
      return getBuilder;
    },
    build() {
      let url = config.url!;
      if (config.query) {
        const qs = new URLSearchParams(config.query).toString();
        url += `?${qs}`;
      }
      return { ...config, url, method: config.method! } as HttpRequest;
    },
  };

  const postBuilder: PostBuilder = {
    header(key, value) {
      (config.headers as Record<string, string>)[key] = value;
      return postBuilder;
    },
    body(body) {
      config.body = body;
      return postBuilder;
    },
    json<T>(data: T) {
      (config.headers as Record<string, string>)["Content-Type"] = "application/json";
      config.body = JSON.stringify(data);
      return postBuilder;
    },
    build() {
      return { ...config, method: config.method! } as HttpRequest;
    },
  };

  return {
    get() {
      config.method = "GET";
      return getBuilder;
    },
    post() {
      config.method = "POST";
      return postBuilder;
    },
    put() {
      config.method = "PUT";
      return postBuilder;
    },
    delete() {
      config.method = "DELETE";
      return getBuilder;
    },
  };
}

// GET リクエスト: body() は呼べない
const getReq = request("https://api.example.com/users")
  .get()
  .header("Accept", "application/json")
  .query({ page: "1", limit: "20" })
  .build();

// POST リクエスト: json() が使える
const postReq = request("https://api.example.com/users")
  .post()
  .json({ name: "Alice", email: "alice@example.com" })
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

### 3-3. 条件付きメソッド表示（Conditional Types）

```typescript
// 設定状態に応じてメソッドの可視性を制御
type ConditionalBuilder<
  T,
  State extends Record<string, boolean>
> = {
  // 常に利用可能なメソッド
  reset(): ConditionalBuilder<T, Record<string, false>>;
} & (State extends { hasSelect: true }
  ? {
      // select 後のみ利用可能
      where(condition: string): ConditionalBuilder<T, State & { hasWhere: true }>;
      orderBy(field: keyof T): ConditionalBuilder<T, State>;
    }
  : {
      // select 前のみ利用可能
      select(...fields: (keyof T)[]): ConditionalBuilder<T, State & { hasSelect: true }>;
    }) &
  (State extends { hasSelect: true }
    ? {
        execute(): Promise<T[]>;
      }
    : {});

// このパターンにより:
// 1. select() を呼ぶ前は where() が表示されない
// 2. select() を呼んだ後は再度 select() が表示されない
// 3. execute() は select() を呼んだ後のみ表示される
```

### 3-4. Fluent API によるバリデーションルールビルダー

```typescript
// バリデーションルールを型安全に構築する Fluent API
type Validator<T> = {
  validate(value: unknown): Result<T, ValidationError[]>;
};

class StringValidatorBuilder {
  private rules: Array<{
    check: (value: string) => boolean;
    message: string;
  }> = [];
  private _optional = false;

  min(length: number): this {
    this.rules.push({
      check: (v) => v.length >= length,
      message: `${length}文字以上で入力してください`,
    });
    return this;
  }

  max(length: number): this {
    this.rules.push({
      check: (v) => v.length <= length,
      message: `${length}文字以内で入力してください`,
    });
    return this;
  }

  pattern(regex: RegExp, message: string): this {
    this.rules.push({
      check: (v) => regex.test(v),
      message,
    });
    return this;
  }

  email(): this {
    return this.pattern(
      /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
      "有効なメールアドレスを入力してください"
    );
  }

  url(): this {
    return this.pattern(
      /^https?:\/\/[^\s]+$/,
      "有効なURLを入力してください"
    );
  }

  optional(): this {
    this._optional = true;
    return this;
  }

  custom(check: (value: string) => boolean, message: string): this {
    this.rules.push({ check, message });
    return this;
  }

  build(): Validator<string> {
    const rules = [...this.rules];
    const optional = this._optional;

    return {
      validate(value: unknown): Result<string, ValidationError[]> {
        if (value === undefined || value === null || value === "") {
          if (optional) return Ok("");
          return Err([{ field: "", message: "この項目は必須です" }]);
        }

        if (typeof value !== "string") {
          return Err([{ field: "", message: "文字列を入力してください" }]);
        }

        const errors: ValidationError[] = [];
        for (const rule of rules) {
          if (!rule.check(value)) {
            errors.push({ field: "", message: rule.message });
          }
        }

        return errors.length > 0 ? Err(errors) : Ok(value);
      },
    };
  }
}

// 使用例
const emailValidator = new StringValidatorBuilder()
  .min(1)
  .max(255)
  .email()
  .build();

const result = emailValidator.validate("test@example.com");
```

### 3-5. メソッドチェーンの型推論と IDE 体験

```typescript
// 設定オブジェクトの型安全なビルダー
type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

interface AppConfig {
  server: {
    host: string;
    port: number;
    cors: {
      origins: string[];
      credentials: boolean;
    };
  };
  database: {
    host: string;
    port: number;
    name: string;
    pool: {
      min: number;
      max: number;
    };
  };
  logging: {
    level: "debug" | "info" | "warn" | "error";
    format: "json" | "text";
  };
}

class ConfigBuilder {
  private config: DeepPartial<AppConfig> = {};

  server(fn: (builder: ServerConfigBuilder) => ServerConfigBuilder): this {
    const serverBuilder = new ServerConfigBuilder();
    fn(serverBuilder);
    this.config.server = serverBuilder.getConfig();
    return this;
  }

  database(fn: (builder: DatabaseConfigBuilder) => DatabaseConfigBuilder): this {
    const dbBuilder = new DatabaseConfigBuilder();
    fn(dbBuilder);
    this.config.database = dbBuilder.getConfig();
    return this;
  }

  logging(level: AppConfig["logging"]["level"], format?: AppConfig["logging"]["format"]): this {
    this.config.logging = { level, format: format ?? "json" };
    return this;
  }

  build(): AppConfig {
    // 必須フィールドの検証
    if (!this.config.server?.host) throw new Error("server.host is required");
    if (!this.config.server?.port) throw new Error("server.port is required");
    if (!this.config.database?.host) throw new Error("database.host is required");
    if (!this.config.database?.name) throw new Error("database.name is required");

    return this.config as AppConfig;
  }
}

class ServerConfigBuilder {
  private config: DeepPartial<AppConfig["server"]> = {};

  host(host: string): this {
    this.config.host = host;
    return this;
  }

  port(port: number): this {
    this.config.port = port;
    return this;
  }

  cors(origins: string[], credentials = false): this {
    this.config.cors = { origins, credentials };
    return this;
  }

  getConfig(): DeepPartial<AppConfig["server"]> {
    return this.config;
  }
}

class DatabaseConfigBuilder {
  private config: DeepPartial<AppConfig["database"]> = {};

  host(host: string): this {
    this.config.host = host;
    return this;
  }

  port(port: number): this {
    this.config.port = port;
    return this;
  }

  name(name: string): this {
    this.config.name = name;
    return this;
  }

  pool(min: number, max: number): this {
    this.config.pool = { min, max };
    return this;
  }

  getConfig(): DeepPartial<AppConfig["database"]> {
    return this.config;
  }
}

// 使用例: ネストしたビルダー
const config = new ConfigBuilder()
  .server((s) =>
    s
      .host("0.0.0.0")
      .port(3000)
      .cors(["https://example.com"], true)
  )
  .database((db) =>
    db
      .host("localhost")
      .port(5432)
      .name("myapp")
      .pool(5, 20)
  )
  .logging("info", "json")
  .build();
```

---

## 4. Immutable ビルダー

### 4-1. 不変ビルダーの基本

Mutable ビルダー（`this` を返す）は効率的ですが、途中の状態を保存して分岐させると予期しない結果になります。

```typescript
// Mutable ビルダーの問題
const baseBuilder = new HttpRequestBuilder()
  .setUrl("https://api.example.com");

// 2つのリクエストを分岐して作成したいが...
const getRequest = baseBuilder.setMethod("GET").build();
const postRequest = baseBuilder.setMethod("POST").build(); // GET が POST に上書きされる

// 解決: Immutable ビルダー
class ImmutableRequestBuilder<
  HasUrl extends boolean = false,
  HasMethod extends boolean = false
> {
  private constructor(
    private readonly _url: string,
    private readonly _method: "GET" | "POST" | "PUT" | "DELETE",
    private readonly _headers: Readonly<Record<string, string>>,
    private readonly _body: string | undefined,
    private readonly _timeout: number
  ) {}

  static create(): ImmutableRequestBuilder<false, false> {
    return new ImmutableRequestBuilder("", "GET", {}, undefined, 30000);
  }

  url(url: string): ImmutableRequestBuilder<true, HasMethod> {
    return new ImmutableRequestBuilder(
      url,
      this._method,
      this._headers,
      this._body,
      this._timeout
    ) as ImmutableRequestBuilder<true, HasMethod>;
  }

  method(
    method: "GET" | "POST" | "PUT" | "DELETE"
  ): ImmutableRequestBuilder<HasUrl, true> {
    return new ImmutableRequestBuilder(
      this._url,
      method,
      this._headers,
      this._body,
      this._timeout
    ) as ImmutableRequestBuilder<HasUrl, true>;
  }

  header(key: string, value: string): ImmutableRequestBuilder<HasUrl, HasMethod> {
    return new ImmutableRequestBuilder(
      this._url,
      this._method,
      { ...this._headers, [key]: value },
      this._body,
      this._timeout
    ) as ImmutableRequestBuilder<HasUrl, HasMethod>;
  }

  body(body: string): ImmutableRequestBuilder<HasUrl, HasMethod> {
    return new ImmutableRequestBuilder(
      this._url,
      this._method,
      this._headers,
      body,
      this._timeout
    ) as ImmutableRequestBuilder<HasUrl, HasMethod>;
  }

  timeout(ms: number): ImmutableRequestBuilder<HasUrl, HasMethod> {
    return new ImmutableRequestBuilder(
      this._url,
      this._method,
      this._headers,
      this._body,
      ms
    ) as ImmutableRequestBuilder<HasUrl, HasMethod>;
  }

  build(this: ImmutableRequestBuilder<true, true>): HttpRequest {
    return {
      method: this._method,
      url: this._url,
      headers: { ...this._headers },
      body: this._body,
      timeout: this._timeout,
    };
  }
}

// 安全に分岐できる
const base = ImmutableRequestBuilder.create()
  .url("https://api.example.com")
  .header("Accept", "application/json");

const getReq = base.method("GET").build();       // GET リクエスト
const postReq = base.method("POST")              // POST リクエスト
  .body(JSON.stringify({ name: "Alice" }))
  .build();
// base は変更されていない
```

### 4-2. Record-based Immutable ビルダー

```typescript
// よりシンプルな不変ビルダーの実装
type RequiredKeys = "url" | "method";
type BuilderConfig = {
  url?: string;
  method?: "GET" | "POST" | "PUT" | "DELETE";
  headers?: Record<string, string>;
  body?: string;
  timeout?: number;
};

type HasRequired<
  Config extends BuilderConfig,
  Keys extends string
> = Keys extends keyof Config
  ? Config[Keys] extends undefined
    ? false
    : true
  : false;

function createBuilder(config: BuilderConfig = {}) {
  const builder = {
    url: (url: string) => createBuilder({ ...config, url }),
    method: (method: "GET" | "POST" | "PUT" | "DELETE") =>
      createBuilder({ ...config, method }),
    header: (key: string, value: string) =>
      createBuilder({
        ...config,
        headers: { ...(config.headers ?? {}), [key]: value },
      }),
    body: (body: string) => createBuilder({ ...config, body }),
    timeout: (ms: number) => createBuilder({ ...config, timeout: ms }),
    build: (): HttpRequest => {
      if (!config.url) throw new Error("url is required");
      if (!config.method) throw new Error("method is required");
      return {
        url: config.url,
        method: config.method,
        headers: config.headers ?? {},
        body: config.body,
        timeout: config.timeout ?? 30000,
      };
    },
  };

  return builder;
}

// 使用例
const req = createBuilder()
  .url("https://api.example.com")
  .method("POST")
  .header("Content-Type", "application/json")
  .body(JSON.stringify({ name: "Alice" }))
  .build();
```

### 4-3. ジェネリクスを使った汎用 Immutable ビルダー

```typescript
// 汎用的な型安全 Immutable ビルダー
type Builder<
  Target,
  Required extends keyof Target,
  Set extends keyof Target = never
> = {
  [K in keyof Target]-?: (
    value: Target[K]
  ) => Builder<Target, Required, Set | K>;
} & (Required extends Set
  ? { build(): Readonly<Target> }
  : {});

function createTypedBuilder<
  Target extends Record<string, unknown>,
  Required extends keyof Target = never
>(
  defaults?: Partial<Target>,
  requiredKeys?: Required[]
): Builder<Target, Required> {
  const config: Partial<Target> = { ...(defaults ?? {}) };

  const handler: ProxyHandler<any> = {
    get(_, prop: string) {
      if (prop === "build") {
        return () => ({ ...config }) as Target;
      }
      return (value: unknown) => {
        const newConfig = { ...config, [prop]: value };
        return new Proxy({}, {
          get(_, prop: string) {
            if (prop === "build") {
              return () => ({ ...newConfig }) as Target;
            }
            return (value: unknown) => {
              return createTypedBuilder<Target, Required>({
                ...newConfig,
                [prop]: value,
              } as Partial<Target>);
            };
          },
        });
      };
    },
  };

  return new Proxy({}, handler) as Builder<Target, Required>;
}

// 使用例
interface EmailMessage {
  to: string;
  from: string;
  subject: string;
  body: string;
  cc?: string[];
  bcc?: string[];
  replyTo?: string;
}

const emailBuilder = createTypedBuilder<EmailMessage, "to" | "from" | "subject" | "body">();
```

---

## 5. テストデータビルダー

### 5-1. 基本的なテストデータビルダー

```
テストデータ生成フロー:

  UserBuilder.create()
       |
  .withDefaults()     <- 合理的なデフォルト値
       |
  .withName("Alice")  <- テストに必要な値だけ上書き
       |
  .withPosts(3)       <- リレーション生成
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

### 5-2. 関連エンティティ付きテストデータビルダー

```typescript
// ─── 各エンティティのビルダー ───

interface Post {
  id: string;
  title: string;
  content: string;
  authorId: string;
  status: "draft" | "published" | "archived";
  tags: string[];
  createdAt: Date;
  updatedAt: Date;
}

interface Comment {
  id: string;
  postId: string;
  authorId: string;
  content: string;
  createdAt: Date;
}

class PostBuilder {
  private data: Post = {
    id: crypto.randomUUID(),
    title: "Test Post",
    content: "This is a test post content.",
    authorId: "",
    status: "draft",
    tags: [],
    createdAt: new Date(),
    updatedAt: new Date(),
  };

  static create(): PostBuilder {
    return new PostBuilder();
  }

  withTitle(title: string): this {
    this.data.title = title;
    return this;
  }

  withContent(content: string): this {
    this.data.content = content;
    return this;
  }

  withAuthor(authorId: string): this {
    this.data.authorId = authorId;
    return this;
  }

  withStatus(status: Post["status"]): this {
    this.data.status = status;
    return this;
  }

  withTags(...tags: string[]): this {
    this.data.tags = tags;
    return this;
  }

  published(): this {
    this.data.status = "published";
    return this;
  }

  archived(): this {
    this.data.status = "archived";
    return this;
  }

  build(): Post {
    return { ...this.data };
  }
}

class CommentBuilder {
  private data: Comment = {
    id: crypto.randomUUID(),
    postId: "",
    authorId: "",
    content: "Test comment",
    createdAt: new Date(),
  };

  static create(): CommentBuilder {
    return new CommentBuilder();
  }

  forPost(postId: string): this {
    this.data.postId = postId;
    return this;
  }

  byAuthor(authorId: string): this {
    this.data.authorId = authorId;
    return this;
  }

  withContent(content: string): this {
    this.data.content = content;
    return this;
  }

  build(): Comment {
    return { ...this.data };
  }
}

// ─── シナリオビルダー: 複数のエンティティを一括生成 ───

interface BlogScenario {
  users: User[];
  posts: Post[];
  comments: Comment[];
}

class BlogScenarioBuilder {
  private users: User[] = [];
  private posts: Post[] = [];
  private comments: Comment[] = [];

  static create(): BlogScenarioBuilder {
    return new BlogScenarioBuilder();
  }

  withUser(
    configurator?: (builder: UserBuilder) => UserBuilder
  ): this {
    const builder = UserBuilder.create();
    const user = configurator ? configurator(builder).build() : builder.build();
    this.users.push(user);
    return this;
  }

  withPost(
    authorIndex: number,
    configurator?: (builder: PostBuilder) => PostBuilder
  ): this {
    const author = this.users[authorIndex];
    if (!author) throw new Error(`User at index ${authorIndex} not found`);

    const builder = PostBuilder.create().withAuthor(author.id);
    const post = configurator ? configurator(builder).build() : builder.build();
    this.posts.push(post);
    return this;
  }

  withComment(
    postIndex: number,
    authorIndex: number,
    content?: string
  ): this {
    const post = this.posts[postIndex];
    const author = this.users[authorIndex];
    if (!post) throw new Error(`Post at index ${postIndex} not found`);
    if (!author) throw new Error(`User at index ${authorIndex} not found`);

    const comment = CommentBuilder.create()
      .forPost(post.id)
      .byAuthor(author.id)
      .withContent(content ?? "Test comment")
      .build();
    this.comments.push(comment);
    return this;
  }

  build(): BlogScenario {
    return {
      users: [...this.users],
      posts: [...this.posts],
      comments: [...this.comments],
    };
  }

  async persist(db: Database): Promise<BlogScenario> {
    const scenario = this.build();
    await db.users.insertMany(scenario.users);
    await db.posts.insertMany(scenario.posts);
    await db.comments.insertMany(scenario.comments);
    return scenario;
  }
}

// テストでの使用
describe("Blog API", () => {
  it("should return published posts with comments", async () => {
    const scenario = BlogScenarioBuilder.create()
      .withUser((u) => u.withName("Alice").withRole("admin"))
      .withUser((u) => u.withName("Bob"))
      .withPost(0, (p) => p.withTitle("Hello World").published().withTags("typescript", "testing"))
      .withPost(0, (p) => p.withTitle("Draft Post"))  // draft
      .withComment(0, 1, "Great post!")
      .withComment(0, 0, "Thanks!")
      .build();

    await persistScenario(scenario);

    const response = await api.get("/posts?status=published");
    expect(response.body).toHaveLength(1);
    expect(response.body[0].title).toBe("Hello World");
    expect(response.body[0].comments).toHaveLength(2);
  });
});
```

### 5-3. ファクトリ関数パターン（軽量テストデータ生成）

```typescript
// ビルダークラスを使わず、関数だけでテストデータを生成する軽量パターン

// カウンタ付きファクトリ
let userCounter = 0;
function createTestUser(overrides: Partial<User> = {}): User {
  userCounter++;
  return {
    id: `user-${userCounter}`,
    name: `User ${userCounter}`,
    email: `user${userCounter}@test.com`,
    age: 25,
    role: "user",
    createdAt: new Date(),
    ...overrides,
  };
}

let postCounter = 0;
function createTestPost(overrides: Partial<Post> = {}): Post {
  postCounter++;
  return {
    id: `post-${postCounter}`,
    title: `Post ${postCounter}`,
    content: `Content of post ${postCounter}`,
    authorId: `user-1`,
    status: "draft",
    tags: [],
    createdAt: new Date(),
    updatedAt: new Date(),
    ...overrides,
  };
}

// 使用例
describe("PostService", () => {
  it("should publish a draft post", async () => {
    const author = createTestUser({ role: "admin" });
    const post = createTestPost({ authorId: author.id, status: "draft" });

    const result = await postService.publish(author.id, post.id);
    expect(result).toBeOk();
  });
});

// ─── faker.js との統合 ───
import { faker } from "@faker-js/faker";

function createRealisticUser(overrides: Partial<User> = {}): User {
  return {
    id: faker.string.uuid(),
    name: faker.person.fullName(),
    email: faker.internet.email(),
    age: faker.number.int({ min: 18, max: 80 }),
    role: faker.helpers.arrayElement(["user", "admin"]),
    createdAt: faker.date.past(),
    ...overrides,
  };
}

function createRealisticPost(overrides: Partial<Post> = {}): Post {
  return {
    id: faker.string.uuid(),
    title: faker.lorem.sentence(),
    content: faker.lorem.paragraphs(3),
    authorId: faker.string.uuid(),
    status: faker.helpers.arrayElement(["draft", "published", "archived"]),
    tags: faker.helpers.arrayElements(
      ["typescript", "javascript", "react", "nodejs", "testing"],
      { min: 1, max: 3 }
    ),
    createdAt: faker.date.past(),
    updatedAt: faker.date.recent(),
    ...overrides,
  };
}
```

### 5-4. 型安全なテストデータビルダージェネリクス

```typescript
// 任意のインターフェースに対応する汎用テストデータビルダー
type WithMethods<T> = {
  [K in keyof T as `with${Capitalize<string & K>}`]: (
    value: T[K]
  ) => WithMethods<T> & { build(): T };
} & {
  build(): T;
};

function createTestDataBuilder<T extends Record<string, unknown>>(
  defaults: T
): WithMethods<T> {
  const data = { ...defaults };

  const handler: ProxyHandler<any> = {
    get(_, prop: string) {
      if (prop === "build") {
        return () => ({ ...data });
      }
      if (prop.startsWith("with")) {
        const fieldName =
          prop.charAt(4).toLowerCase() + prop.slice(5);
        return (value: unknown) => {
          (data as any)[fieldName] = value;
          return new Proxy({}, handler);
        };
      }
      return undefined;
    },
  };

  return new Proxy({}, handler) as WithMethods<T>;
}

// 使用例
const userBuilder = createTestDataBuilder<User>({
  id: "test-id",
  name: "Default User",
  email: "default@test.com",
  age: 25,
  role: "user",
  createdAt: new Date(),
});

const user = userBuilder
  .withName("Custom Name")
  .withAge(30)
  .withRole("admin")
  .build();
// => { id: "test-id", name: "Custom Name", email: "default@test.com", age: 30, role: "admin", ... }
```

---

## 6. 実務での応用例

### 6-1. ORM スタイルのクエリビルダー

```typescript
// Prisma 風の型安全クエリビルダー
interface WhereClause<T> {
  equals?: T;
  not?: T;
  in?: T[];
  notIn?: T[];
  gt?: T;
  gte?: T;
  lt?: T;
  lte?: T;
  contains?: T extends string ? string : never;
  startsWith?: T extends string ? string : never;
  endsWith?: T extends string ? string : never;
}

type WhereInput<T> = {
  [K in keyof T]?: T[K] | WhereClause<T[K]>;
} & {
  AND?: WhereInput<T>[];
  OR?: WhereInput<T>[];
  NOT?: WhereInput<T>;
};

type OrderByInput<T> = {
  [K in keyof T]?: "asc" | "desc";
};

interface FindManyArgs<T> {
  where?: WhereInput<T>;
  orderBy?: OrderByInput<T> | OrderByInput<T>[];
  skip?: number;
  take?: number;
  select?: { [K in keyof T]?: boolean };
  include?: Record<string, boolean>;
}

class TypeSafeQueryBuilder<T extends Record<string, unknown>> {
  private args: FindManyArgs<T> = {};

  where(condition: WhereInput<T>): this {
    this.args.where = { ...this.args.where, ...condition };
    return this;
  }

  orderBy(field: keyof T, direction: "asc" | "desc" = "asc"): this {
    this.args.orderBy = { [field]: direction } as OrderByInput<T>;
    return this;
  }

  skip(count: number): this {
    this.args.skip = count;
    return this;
  }

  take(count: number): this {
    this.args.take = count;
    return this;
  }

  select<K extends keyof T>(...fields: K[]): this {
    const select = {} as { [P in keyof T]?: boolean };
    for (const field of fields) {
      select[field] = true;
    }
    this.args.select = select;
    return this;
  }

  getArgs(): FindManyArgs<T> {
    return { ...this.args };
  }
}

// 使用例
interface Product {
  id: string;
  name: string;
  price: number;
  category: string;
  stock: number;
  createdAt: Date;
}

const query = new TypeSafeQueryBuilder<Product>()
  .where({
    category: "electronics",
    price: { lte: 10000 },
    stock: { gt: 0 },
  })
  .orderBy("price", "asc")
  .skip(0)
  .take(20)
  .select("id", "name", "price")
  .getArgs();
```

### 6-2. フォームビルダー（React 統合）

```typescript
// React フォーム用の型安全ビルダー
interface FormFieldConfig<T> {
  name: keyof T;
  label: string;
  type: "text" | "number" | "email" | "password" | "select" | "textarea" | "checkbox";
  placeholder?: string;
  required?: boolean;
  options?: Array<{ value: string; label: string }>;
  validation?: (value: unknown) => string | undefined;
  defaultValue?: T[keyof T];
}

class FormBuilder<T extends Record<string, unknown>> {
  private fields: FormFieldConfig<T>[] = [];
  private _onSubmit?: (data: T) => void | Promise<void>;

  field<K extends keyof T & string>(
    name: K,
    config: Omit<FormFieldConfig<T>, "name"> & { type: FormFieldConfig<T>["type"] }
  ): this {
    this.fields.push({ ...config, name } as FormFieldConfig<T>);
    return this;
  }

  text<K extends keyof T & string>(
    name: K,
    label: string,
    options?: Partial<FormFieldConfig<T>>
  ): this {
    return this.field(name, { label, type: "text", ...options });
  }

  email<K extends keyof T & string>(name: K, label: string): this {
    return this.field(name, {
      label,
      type: "email",
      validation: (v) => {
        if (typeof v === "string" && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v)) {
          return "有効なメールアドレスを入力してください";
        }
        return undefined;
      },
    });
  }

  password<K extends keyof T & string>(name: K, label: string): this {
    return this.field(name, { label, type: "password" });
  }

  number<K extends keyof T & string>(
    name: K,
    label: string,
    options?: { min?: number; max?: number }
  ): this {
    return this.field(name, {
      label,
      type: "number",
      validation: (v) => {
        const num = Number(v);
        if (isNaN(num)) return "数値を入力してください";
        if (options?.min !== undefined && num < options.min) {
          return `${options.min}以上の値を入力してください`;
        }
        if (options?.max !== undefined && num > options.max) {
          return `${options.max}以下の値を入力してください`;
        }
        return undefined;
      },
    });
  }

  select<K extends keyof T & string>(
    name: K,
    label: string,
    options: Array<{ value: string; label: string }>
  ): this {
    return this.field(name, { label, type: "select", options });
  }

  onSubmit(handler: (data: T) => void | Promise<void>): this {
    this._onSubmit = handler;
    return this;
  }

  build() {
    return {
      fields: [...this.fields],
      onSubmit: this._onSubmit,
    };
  }
}

// 使用例
interface UserForm {
  name: string;
  email: string;
  age: number;
  role: string;
}

const formConfig = new FormBuilder<UserForm>()
  .text("name", "名前", { required: true, placeholder: "山田太郎" })
  .email("email", "メールアドレス")
  .number("age", "年齢", { min: 0, max: 150 })
  .select("role", "役割", [
    { value: "user", label: "一般ユーザー" },
    { value: "admin", label: "管理者" },
  ])
  .onSubmit(async (data) => {
    await api.post("/users", data);
  })
  .build();
```

### 6-3. メール送信ビルダー

```typescript
interface EmailConfig {
  to: string[];
  cc?: string[];
  bcc?: string[];
  from: string;
  replyTo?: string;
  subject: string;
  text?: string;
  html?: string;
  attachments?: Array<{
    filename: string;
    content: Buffer | string;
    contentType?: string;
  }>;
  priority?: "high" | "normal" | "low";
  headers?: Record<string, string>;
}

class EmailBuilder<
  HasTo extends boolean = false,
  HasFrom extends boolean = false,
  HasSubject extends boolean = false,
  HasBody extends boolean = false
> {
  private config: Partial<EmailConfig> = {
    to: [],
    cc: [],
    bcc: [],
    attachments: [],
    priority: "normal",
  };

  to(...addresses: string[]): EmailBuilder<true, HasFrom, HasSubject, HasBody> {
    this.config.to = addresses;
    return this as any;
  }

  cc(...addresses: string[]): this {
    this.config.cc = addresses;
    return this;
  }

  bcc(...addresses: string[]): this {
    this.config.bcc = addresses;
    return this;
  }

  from(address: string): EmailBuilder<HasTo, true, HasSubject, HasBody> {
    this.config.from = address;
    return this as any;
  }

  replyTo(address: string): this {
    this.config.replyTo = address;
    return this;
  }

  subject(subject: string): EmailBuilder<HasTo, HasFrom, true, HasBody> {
    this.config.subject = subject;
    return this as any;
  }

  text(content: string): EmailBuilder<HasTo, HasFrom, HasSubject, true> {
    this.config.text = content;
    return this as any;
  }

  html(content: string): EmailBuilder<HasTo, HasFrom, HasSubject, true> {
    this.config.html = content;
    return this as any;
  }

  attach(
    filename: string,
    content: Buffer | string,
    contentType?: string
  ): this {
    this.config.attachments!.push({ filename, content, contentType });
    return this;
  }

  priority(level: "high" | "normal" | "low"): this {
    this.config.priority = level;
    return this;
  }

  // 全必須フィールドが設定済みの場合のみ build 可能
  build(
    this: EmailBuilder<true, true, true, true>
  ): EmailConfig {
    return { ...this.config } as EmailConfig;
  }

  // 送信まで一気に行う
  async send(
    this: EmailBuilder<true, true, true, true>,
    transporter: EmailTransporter
  ): Promise<Result<void, EmailError>> {
    const config = this.build();
    return transporter.send(config);
  }
}

// 使用例
const email = new EmailBuilder()
  .to("alice@example.com", "bob@example.com")
  .from("noreply@myapp.com")
  .subject("注文確認")
  .html("<h1>ご注文ありがとうございます</h1><p>注文番号: #12345</p>")
  .attach("invoice.pdf", invoiceBuffer, "application/pdf")
  .priority("high")
  .build();

// 必須フィールドが不足: コンパイルエラー
const incomplete = new EmailBuilder()
  .to("alice@example.com")
  .subject("テスト")
  .build(); // Error: from と body が未設定
```

### 6-4. CLI コマンドビルダー

```typescript
// CLI コマンドの型安全な構築
interface CommandConfig {
  name: string;
  description: string;
  args: Array<{
    name: string;
    description: string;
    required: boolean;
    type: "string" | "number" | "boolean";
    default?: unknown;
  }>;
  options: Array<{
    long: string;
    short?: string;
    description: string;
    type: "string" | "number" | "boolean";
    default?: unknown;
    required?: boolean;
  }>;
  handler: (args: Record<string, unknown>, options: Record<string, unknown>) => void | Promise<void>;
}

class CommandBuilder {
  private config: Partial<CommandConfig> = {
    args: [],
    options: [],
  };

  name(name: string): this {
    this.config.name = name;
    return this;
  }

  description(desc: string): this {
    this.config.description = desc;
    return this;
  }

  argument(
    name: string,
    description: string,
    options?: { required?: boolean; type?: "string" | "number" | "boolean"; default?: unknown }
  ): this {
    this.config.args!.push({
      name,
      description,
      required: options?.required ?? true,
      type: options?.type ?? "string",
      default: options?.default,
    });
    return this;
  }

  option(
    long: string,
    description: string,
    options?: {
      short?: string;
      type?: "string" | "number" | "boolean";
      default?: unknown;
      required?: boolean;
    }
  ): this {
    this.config.options!.push({
      long,
      description,
      short: options?.short,
      type: options?.type ?? "string",
      default: options?.default,
      required: options?.required,
    });
    return this;
  }

  handler(
    fn: (args: Record<string, unknown>, options: Record<string, unknown>) => void | Promise<void>
  ): this {
    this.config.handler = fn;
    return this;
  }

  build(): CommandConfig {
    if (!this.config.name) throw new Error("Command name is required");
    if (!this.config.handler) throw new Error("Command handler is required");
    return this.config as CommandConfig;
  }
}

// 使用例
const deployCommand = new CommandBuilder()
  .name("deploy")
  .description("アプリケーションをデプロイする")
  .argument("environment", "デプロイ先の環境", { type: "string" })
  .option("--force", "強制デプロイ", { short: "-f", type: "boolean", default: false })
  .option("--tag", "デプロイするタグ", { short: "-t", type: "string" })
  .option("--timeout", "タイムアウト（秒）", { type: "number", default: 300 })
  .handler(async (args, options) => {
    console.log(`Deploying to ${args.environment}...`);
    if (options.force) console.log("Force mode enabled");
  })
  .build();
```

### 6-5. パイプラインビルダー

```typescript
// データ変換パイプラインの型安全な構築
type TransformFn<In, Out> = (input: In) => Out | Promise<Out>;

class PipelineBuilder<TInput, TCurrent = TInput> {
  private steps: Array<TransformFn<any, any>> = [];

  static from<T>(): PipelineBuilder<T, T> {
    return new PipelineBuilder();
  }

  pipe<TNext>(
    transform: TransformFn<TCurrent, TNext>
  ): PipelineBuilder<TInput, TNext> {
    this.steps.push(transform);
    return this as unknown as PipelineBuilder<TInput, TNext>;
  }

  tap(fn: (value: TCurrent) => void): PipelineBuilder<TInput, TCurrent> {
    this.steps.push((value: TCurrent) => {
      fn(value);
      return value;
    });
    return this;
  }

  filter(
    predicate: (value: TCurrent) => boolean,
    errorMessage?: string
  ): PipelineBuilder<TInput, TCurrent> {
    this.steps.push((value: TCurrent) => {
      if (!predicate(value)) {
        throw new Error(errorMessage ?? "Filter condition not met");
      }
      return value;
    });
    return this;
  }

  build(): (input: TInput) => Promise<TCurrent> {
    const steps = [...this.steps];
    return async (input: TInput) => {
      let result: unknown = input;
      for (const step of steps) {
        result = await step(result);
      }
      return result as TCurrent;
    };
  }
}

// 使用例
interface RawOrder {
  items: Array<{ productId: string; quantity: string; price: string }>;
  customerId: string;
  note?: string;
}

interface ProcessedOrder {
  items: Array<{ productId: string; quantity: number; price: number; total: number }>;
  customerId: string;
  subtotal: number;
  tax: number;
  total: number;
  note?: string;
}

const processOrder = PipelineBuilder.from<RawOrder>()
  .pipe((raw) => ({
    ...raw,
    items: raw.items.map((item) => ({
      productId: item.productId,
      quantity: parseInt(item.quantity, 10),
      price: parseFloat(item.price),
      total: parseInt(item.quantity, 10) * parseFloat(item.price),
    })),
  }))
  .filter(
    (order) => order.items.length > 0,
    "注文には1つ以上の商品が必要です"
  )
  .filter(
    (order) => order.items.every((i) => i.quantity > 0),
    "数量は1以上である必要があります"
  )
  .pipe((order) => {
    const subtotal = order.items.reduce((sum, item) => sum + item.total, 0);
    const tax = Math.round(subtotal * 0.1);
    return {
      ...order,
      subtotal,
      tax,
      total: subtotal + tax,
    } as ProcessedOrder;
  })
  .tap((order) => console.log(`Order total: ${order.total}`))
  .build();

// 実行
const result = await processOrder({
  items: [
    { productId: "p1", quantity: "2", price: "1000" },
    { productId: "p2", quantity: "1", price: "2500" },
  ],
  customerId: "c1",
});
```

---

## 比較表

### ビルダーパターンのバリエーション比較

| パターン | 型安全性 | 実装コスト | 柔軟性 | 順序強制 | 分岐 |
|---------|---------|-----------|--------|---------|------|
| クラシックビルダー | 低 | 低 | 高 | なし | 不可 |
| Phantom Type | 高 | 中 | 中 | なし | 不可 |
| Step Builder | 最高 | 高 | 低 | あり | 不可 |
| Immutable Builder | 高 | 中 | 高 | なし | 可能 |
| 関数合成ビルダー | 高 | 中 | 高 | なし | 可能 |
| Proxy ビルダー | 中 | 低 | 高 | なし | 可能 |

### ビルダー vs 他の生成パターン

| 比較軸 | ビルダー | ファクトリ | コンストラクタ | Object.assign |
|--------|---------|-----------|-------------|---------------|
| 引数の多さ | 多い場合に最適 | 少〜中 | 少〜中 | 中 |
| 段階的構築 | 可能 | 不可 | 不可 | 不可 |
| バリデーション | build() 時 | 生成時 | 即座 | なし |
| IDE 補完 | 優秀 | 良好 | 普通 | 限定的 |
| 不変性 | 保証可能 | 保証可能 | 設計次第 | 困難 |
| テストデータ | 最適 | 良好 | 不向き | 可能 |
| 可読性 | 高 | 中 | 低（引数多数時） | 中 |

### 適用場面のガイドライン

| 場面 | 推奨パターン | 理由 |
|------|------------|------|
| HTTPリクエスト構築 | Step Builder | メソッドとURLは必須、順序も自然 |
| テストデータ生成 | クラシック or ファクトリ関数 | デフォルト値が重要、型安全性は低くてもOK |
| 設定オブジェクト | Phantom Type | 必須設定の検証が重要 |
| クエリ構築 | Fluent API | メソッドチェーンが自然な DSL になる |
| メール送信 | Phantom Type | to/from/subject は必須 |
| CLI コマンド | クラシック | 柔軟性重視、必須項目は少ない |
| パイプライン | 関数合成 | 型の変換を追跡する必要がある |

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

### AP-3: ビルダーの再利用による状態汚染

```typescript
// NG: Mutable ビルダーの再利用
const builder = new HttpRequestBuilder()
  .setUrl("https://api.example.com");

const req1 = builder.setMethod("GET").build();
const req2 = builder.setMethod("POST").build();
// req1.method が "POST" になっている可能性がある!

// OK: build 後にリセットする
class ResettableBuilder {
  // ... フィールド省略 ...

  build(): HttpRequest {
    const result = { /* ... */ } as HttpRequest;
    this.reset(); // build 後に状態をリセット
    return result;
  }

  private reset(): void {
    this.method = "GET";
    this.url = "";
    this.headers = {};
    this.body = undefined;
    this.timeout = 30000;
  }
}

// より良い: Immutable ビルダーを使う（セクション4参照）
```

### AP-4: 過度に複雑なビルダー

```typescript
// NG: 単純なオブジェクトにビルダーは不要
class PointBuilder {
  private x = 0;
  private y = 0;

  setX(x: number): this { this.x = x; return this; }
  setY(y: number): this { this.y = y; return this; }

  build(): { x: number; y: number } {
    return { x: this.x, y: this.y };
  }
}

// OK: 単純なオブジェクトは直接生成
const point = { x: 10, y: 20 };

// ビルダーを使うべき目安:
// - フィールドが5つ以上
// - 必須/任意の組み合わせが複雑
// - 構築にバリデーションが必要
// - テストデータの生成で頻繁に使う
```

### AP-5: build() 後のメソッド呼び出し

```typescript
// NG: build 後にメソッドを呼んでも意味がない
const builder = new HttpRequestBuilder();
const request = builder.setUrl("https://example.com").build();
builder.setMethod("POST"); // request には反映されない

// OK: ビルダーの使い捨てを推奨するAPI設計
// 方法1: build 後にビルダーを無効化
class OneTimeBuilder {
  private built = false;

  // ... setter メソッド ...

  build(): HttpRequest {
    if (this.built) {
      throw new Error("Builder already used. Create a new instance.");
    }
    this.built = true;
    return { /* ... */ } as HttpRequest;
  }
}

// 方法2: static メソッドでファクトリスタイル
const request2 = HttpRequestBuilder.create()
  .setUrl("https://example.com")
  .setMethod("POST")
  .build();
```

---

## パフォーマンス考慮事項

### Immutable vs Mutable ビルダーのパフォーマンス

```typescript
// ─── ベンチマーク比較 ───

// Mutable ビルダー: ~50ns/op
// - オブジェクト生成は1回のみ
// - 各メソッドはプロパティ代入のみ

// Immutable ビルダー: ~200ns/op
// - 各メソッドで新しいオブジェクトを生成
// - スプレッド演算子によるコピーコスト

// ─── 最適化のヒント ───

// 1. ビルダーの使用頻度が高い場合は Mutable を選択
// 2. テストデータ生成では Immutable が安全
// 3. ホットパスでは直接オブジェクトリテラルを使用
// 4. Proxy ベースのビルダーは最も遅い（~1000ns/op）

// ─── メモリ効率 ───

// 大量のビルダーインスタンスを生成する場合:
// - Mutable: 1インスタンスで複数のオブジェクトを生成可能
// - Immutable: メソッドチェーンの長さ分のインスタンスが生成される
//   （ただしGCで回収される）

// 実測値の参考:
// 10,000回のビルド:
//   Mutable Builder: ~0.5ms
//   Immutable Builder: ~2ms
//   Proxy Builder: ~10ms
//   Direct object literal: ~0.1ms
```

---

## FAQ

### Q1: ビルダーパターンとファクトリパターンの使い分けは？

引数が 4 つ以上ある、または省略可能な引数が多い場合はビルダーが適しています。引数が少なく固定されている場合はファクトリで十分です。テストデータ生成はビルダーの最も効果的な適用場面です。

### Q2: Phantom Type ビルダーはパフォーマンスに影響しますか？

型パラメータはコンパイル時にのみ存在し、JavaScript 出力には一切影響しません。`as unknown as` のキャストもランタイムコストゼロです。パフォーマンスの心配は不要です。

### Q3: Immutable ビルダーと Mutable ビルダーのどちらが良いですか？

Immutable ビルダー（各メソッドで新しいインスタンスを返す）は安全ですがメモリ割り当てが増えます。Mutable ビルダー（this を返す）は効率的ですが、途中の状態を保存して分岐させる使い方ができません。一般的には Mutable で十分ですが、ビルダーを変数に保存して分岐させたい場合は Immutable を選択してください。

### Q4: ビルダーパターンは関数型プログラミングと矛盾しませんか？

Immutable ビルダーは各メソッドが新しいインスタンスを返すため、関数型プログラミングの原則と完全に一致します。実際、関数合成でパイプラインを構築するパターンはビルダーの関数型版といえます。ただし Mutable ビルダーは内部状態を変更するため、純粋関数型のスタイルには適しません。

### Q5: テストデータビルダーで faker.js を使うべきですか？

ランダムデータはテストの再現性を下げるため、基本的には固定値のデフォルトを使うことを推奨します。ただし、プロパティベーステスト（fast-check）やストレステストでは faker.js が有用です。通常のユニットテストでは `UserBuilder.create().withName("Alice")` のように意図が明確な固定値を使いましょう。

### Q6: Step Builder のインターフェースが増えすぎる問題は？

必須フィールドが N 個ある場合、最大で N+1 個のインターフェースが必要です。これが多すぎる場合は、Phantom Type ビルダーを使って型パラメータで状態を追跡するか、必須フィールドをコンストラクタで受け取り、任意フィールドだけビルダーで設定する折衷案を検討してください。

### Q7: ネストしたオブジェクトのビルダーはどう設計すべきですか？

親ビルダーにコールバック関数を受け取るメソッドを用意し、子ビルダーを引数として渡すパターンが効果的です。`.server((s) => s.host("localhost").port(3000))` のようなAPIになります。セクション3-5の ConfigBuilder を参照してください。

### Q8: ビルダーパターンをどのようにテストすべきですか？

ビルダー自体のテストでは、(1) 全フィールド設定時の正常な build、(2) 必須フィールド未設定時のエラー、(3) デフォルト値の確認、(4) メソッドチェーンの戻り値の型をテストします。Phantom Type ビルダーの場合はコンパイルエラーのテスト（`// @ts-expect-error` アノテーション）も重要です。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| クラシックビルダー | 段階的構築だが型安全性は不十分 |
| Phantom Type | ジェネリクスのフラグで設定状態を追跡 |
| Step Builder | インターフェース分割で順序を強制 |
| Fluent API | メソッドチェーンで直感的な DSL を構築 |
| Immutable ビルダー | 各メソッドで新しいインスタンスを返し、分岐が安全 |
| テストデータビルダー | `.create().withX().build()` でテストデータを生成 |
| シナリオビルダー | 複数の関連エンティティを一括生成 |
| パイプラインビルダー | データ変換の型を連鎖的に追跡 |
| 不変コピー | `build()` ではスプレッドでコピーを返す |
| Proxy ビルダー | 動的にメソッドを生成する汎用ビルダー |

---

## 次に読むべきガイド

- [ブランド型](./03-branded-types.md) -- ビルダーで生成する値にブランドを付与する
- [DI パターン](./04-dependency-injection.md) -- ビルダーと DI を組み合わせたファクトリ設計
- [判別共用体](./02-discriminated-unions.md) -- Step Builder の型安全性を支える判別共用体
- [エラーハンドリング](./00-error-handling.md) -- Result 型との統合パターン

---

## 参考文献

1. **Design Patterns: Elements of Reusable Object-Oriented Software** -- Gamma et al. (GoF)
   Builder パターンの原典

2. **TypeScript Deep Dive - Phantom Types**
   https://basarat.gitbook.io/typescript/

3. **Fluent Interface** -- Martin Fowler
   https://martinfowler.com/bliki/FluentInterface.html

4. **Effective TypeScript** -- Dan Vanderkam
   型安全なビルダーパターンの解説

5. **The Builder Pattern in TypeScript** -- Marius Schulz
   https://mariusschulz.com/blog/tagged/typescript

6. **Phantom Types in TypeScript** -- GitHub Advanced Security
   https://github.blog/engineering/

7. **Test Data Builders** -- Nat Pryce
   http://natpryce.com/articles/000714.html

8. **faker.js** -- Fake data generator
   https://fakerjs.dev/
