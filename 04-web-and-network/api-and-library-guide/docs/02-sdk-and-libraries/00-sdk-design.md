# SDK設計

> SDKはAPIの利用体験を決定づけるフロントライン。型安全なクライアント設計、Builderパターン、エラーハンドリング、リトライ戦略、認証の抽象化まで、開発者に愛されるSDK設計のベストプラクティスを習得する。

## この章で学ぶこと

- [ ] SDK設計の原則とDX（開発者体験）の定量的評価方法を理解する
- [ ] 型安全なクライアント実装パターンと設計上のトレードオフを把握する
- [ ] リトライ、認証、ページネーションの抽象化レイヤーを設計・実装できる
- [ ] エラー階層設計とユーザーフレンドリーな障害復旧フローを構築できる
- [ ] バージョニング戦略とブレイキングチェンジの管理手法を習得する
- [ ] テスタビリティとモック戦略を通じたSDK品質保証を実践できる

---

## 1. SDK設計の全体像

### 1.1 SDKとは何か

SDK（Software Development Kit）は、特定のAPIやプラットフォームを利用するための開発ツールキットである。単なるHTTPラッパーではなく、認証管理、エラーハンドリング、型安全性、ページネーション、リトライ、ロギングなど多層的な機能を統合した開発基盤として設計される。

```
+------------------------------------------------------------------+
|                        SDK アーキテクチャ全体図                      |
+------------------------------------------------------------------+
|                                                                    |
|  +--------------------+    +--------------------+                  |
|  |   開発者コード       |    |   SDK Public API   |                  |
|  |                    | -> |  (型安全インターフェース) |                |
|  +--------------------+    +--------+-----------+                  |
|                                     |                              |
|                          +----------v-----------+                  |
|                          |  Resource Layer       |                  |
|                          |  users / orders / ...  |                 |
|                          +----------+-----------+                  |
|                                     |                              |
|                          +----------v-----------+                  |
|                          |  Middleware Pipeline  |                  |
|                          |  Auth -> Retry ->     |                  |
|                          |  RateLimit -> Log     |                  |
|                          +----------+-----------+                  |
|                                     |                              |
|                          +----------v-----------+                  |
|                          |  HTTP Transport       |                  |
|                          |  fetch / axios / node  |                 |
|                          +----------+-----------+                  |
|                                     |                              |
|                          +----------v-----------+                  |
|                          |  Serialization        |                  |
|                          |  JSON / Protobuf      |                  |
|                          +----------+-----------+                  |
|                                     |                              |
|                          +----------v-----------+                  |
|                          |  外部 API サーバー      |                 |
|                          +----------------------+                  |
+------------------------------------------------------------------+
```

### 1.2 SDK設計の5原則

優れたSDKは以下の5原則に従って設計される。これらは Stripe、Twilio、AWS など世界的に評価の高いSDKから抽出された共通パターンである。

#### 原則1: Principle of Least Surprise（最小驚き原則）

SDKの振る舞いは開発者の直感に沿うべきである。メソッド名、引数の順序、戻り値の型すべてが「予測可能」であることが求められる。

```typescript
// 良い例: 直感的なメソッド名と引数
const user = await client.users.get("user_123");
const users = await client.users.list({ limit: 20 });

// 悪い例: 何をするのか予測できない
const user = await client.fetch("users", "user_123", true, null);
const users = await client.query({ type: "user", max: 20, mode: 1 });
```

#### 原則2: Progressive Disclosure（段階的開示）

基本操作は最小限のコードで実現でき、高度な機能は必要になったときに発見・利用できる。

```typescript
// レベル1: 最小限の設定で利用開始
const client = new ExampleClient({ apiKey: "sk_live_abc" });

// レベル2: 必要に応じてオプションを追加
const client = new ExampleClient({
  apiKey: "sk_live_abc",
  timeout: 60000,
  maxRetries: 5,
});

// レベル3: 高度なカスタマイズ
const client = new ExampleClient({
  apiKey: "sk_live_abc",
  timeout: 60000,
  maxRetries: 5,
  httpAgent: new https.Agent({ keepAlive: true }),
  middleware: [loggingMiddleware, metricsMiddleware],
  baseUrl: "https://api-staging.example.com/v2",
});
```

#### 原則3: Fail Fast, Fail Clearly（早期・明確な失敗）

不正な入力やコンフィグレーションは、APIコールの前に検出して即座にわかりやすいエラーを投げる。

```typescript
class ExampleClient {
  constructor(config: ClientConfig) {
    // 初期化時にバリデーション
    if (!config.apiKey) {
      throw new ConfigurationError(
        "API key is required. Get your key at https://dashboard.example.com/api-keys"
      );
    }
    if (config.apiKey.startsWith("sk_test_") && config.baseUrl?.includes("production")) {
      throw new ConfigurationError(
        "Test API key cannot be used with production endpoint. " +
        "Use a live key (sk_live_*) or switch to the sandbox endpoint."
      );
    }
    if (config.timeout !== undefined && config.timeout < 0) {
      throw new ConfigurationError(
        `Invalid timeout value: ${config.timeout}. Timeout must be a positive number in milliseconds.`
      );
    }
  }
}
```

#### 原則4: Idiomatic Design（言語慣用句に従う設計）

各プログラミング言語の慣用句やエコシステムの慣行に従う。TypeScript SDK は Promise を返し、Go SDK はエラー値を返し、Python SDK はジェネレータを活用する。

```typescript
// TypeScript: async/await + Promise
const user = await client.users.get("123");

// Go では同じ操作が以下のようになる想定:
// user, err := client.Users.Get(ctx, "123")
// if err != nil { ... }

// Python では以下のようになる想定:
// user = client.users.get("123")
// for user in client.users.list():  # ジェネレータ
```

#### 原則5: Backward Compatibility（後方互換性）

マイナーバージョンアップでは既存コードが壊れてはならない。ブレイキングチェンジはメジャーバージョンに集約し、マイグレーションガイドを提供する。

### 1.3 DX（開発者体験）の定量指標

```
+---------------------------------------------------------------+
|              DX 評価マトリクス                                    |
+------------------+-------------+-------------+----------------+
| 指標              | 目標値       | 測定方法      | 改善手段        |
+------------------+-------------+-------------+----------------+
| Time to First    | < 5分        | チュートリアル  | Quick Start    |
| API Call (TTFAC) |             | 完了時間       | ガイド整備      |
+------------------+-------------+-------------+----------------+
| Lines of Code    | < 5行        | 基本操作に     | デフォルト値     |
| (LOC)            |             | 必要な行数     | の最適化        |
+------------------+-------------+-------------+----------------+
| Error Recovery   | < 30秒       | エラーメッセージ | actionable     |
| Time (ERT)       |             | からの復帰     | error messages |
+------------------+-------------+-------------+----------------+
| Feature          | > 90%       | IDE補完で      | 型定義の        |
| Discoverability  |             | 発見可能な     | 充実            |
|                  |             | 機能の割合     |                |
+------------------+-------------+-------------+----------------+
| Dependency       | < 3個        | package.json  | バンドル最小化   |
| Count            |             | の依存数       |                |
+------------------+-------------+-------------+----------------+
| Bundle Size      | < 50KB      | minified +    | Tree shaking   |
|                  | (gzip)      | gzip         | 対応            |
+------------------+-------------+-------------+----------------+
```

### 1.4 SDK設計のスコープ決定

SDKを設計する際、最初に決定すべきはスコープである。すべてのAPIエンドポイントを網羅するフルカバレッジSDKなのか、主要ユースケースに絞ったライトウェイトSDKなのかで、設計判断が大きく変わる。

| スコープ | 特徴 | 適するケース |
|---------|------|------------|
| フルカバレッジ | 全エンドポイントを型安全にラップ | エンタープライズ向け、APIが安定している |
| コアのみ | 主要操作（CRUD）のみ提供 | スタートアップ、API変更が頻繁 |
| コード生成 | OpenAPI仕様から自動生成 | 大規模API、多言語対応 |
| ハイブリッド | コア手書き + 拡張は自動生成 | バランス重視 |

---

## 2. クライアント設計パターン

### 2.1 主要パターンの比較

SDKクライアントの設計パターンは大きく3種類に分類される。それぞれの特性を理解し、プロジェクトに最適なパターンを選択する。

| パターン | 型安全性 | 学習コスト | 拡張性 | 代表的SDK |
|---------|---------|----------|-------|----------|
| Resource-based | 高 | 低 | 高 | Stripe, Twilio |
| Fluent API | 中 | 中 | 中 | Elasticsearch |
| Function-based | 高 | 低 | 中 | AWS SDK v3 |
| Builder | 高 | 高 | 高 | Google Cloud |
| Proxy-based | 高 | 低 | 高 | tRPC |

### 2.2 Resource-based パターン（推奨）

Resource-basedパターンは、APIリソースをオブジェクトとして表現し、そのオブジェクトにCRUDメソッドを持たせる設計である。REST APIとの親和性が高く、最も広く採用されている。

```typescript
// --- 利用者側コード ---

// 基本的なCRUD操作
const user = await client.users.get("user_123");
const users = await client.users.list({ role: "admin", limit: 20 });
const newUser = await client.users.create({
  name: "Taro Yamada",
  email: "taro@example.com",
});
const updated = await client.users.update("user_123", { name: "Updated Name" });
await client.users.delete("user_123");

// ネストされたリソース
const orders = await client.users.orders.list("user_123", { status: "active" });
const address = await client.users.addresses.get("user_123", "addr_456");
```

```typescript
// --- SDK内部実装 ---

// クライアント本体
class ExampleClient {
  private config: ResolvedClientConfig;
  readonly users: UsersResource;
  readonly orders: OrdersResource;
  readonly products: ProductsResource;

  constructor(config: ClientConfig) {
    this.config = this.resolveConfig(config);
    this.validateConfig(this.config);

    const httpClient = new HttpClient(this.config);
    this.users = new UsersResource(httpClient);
    this.orders = new OrdersResource(httpClient);
    this.products = new ProductsResource(httpClient);
  }

  private resolveConfig(config: ClientConfig): ResolvedClientConfig {
    return {
      apiKey: config.apiKey,
      baseUrl: config.baseUrl ?? "https://api.example.com/v1",
      timeout: config.timeout ?? 30000,
      maxRetries: config.maxRetries ?? 3,
      retryDelay: config.retryDelay ?? 1000,
      userAgent: `example-sdk-ts/${SDK_VERSION}`,
    };
  }

  private validateConfig(config: ResolvedClientConfig): void {
    if (!config.apiKey) {
      throw new ConfigurationError(
        "API key is required. " +
        "Obtain your API key from https://dashboard.example.com/api-keys"
      );
    }
  }
}

// 型定義
interface ClientConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
}

interface ResolvedClientConfig {
  apiKey: string;
  baseUrl: string;
  timeout: number;
  maxRetries: number;
  retryDelay: number;
  userAgent: string;
}
```

### 2.3 Fluent API / Method Chaining パターン

クエリビルダーのような用途で力を発揮するパターン。複雑なフィルタリングや検索条件を直感的に構築できる。

```typescript
// 利用例
const users = await client.users
  .list()
  .filter({ role: "admin", status: "active" })
  .sort("-createdAt")
  .fields("id", "name", "email")
  .limit(20)
  .execute();

// 実装
class QueryBuilder<T> {
  private params: Record<string, any> = {};

  constructor(
    private httpClient: HttpClient,
    private path: string
  ) {}

  filter(conditions: Record<string, any>): this {
    this.params.filter = { ...this.params.filter, ...conditions };
    return this;
  }

  sort(field: string): this {
    this.params.sort = field;
    return this;
  }

  fields(...fields: string[]): this {
    this.params.fields = fields.join(",");
    return this;
  }

  limit(n: number): this {
    this.params.limit = n;
    return this;
  }

  async execute(): Promise<PaginatedResponse<T>> {
    return this.httpClient.request<PaginatedResponse<T>>(
      "GET",
      this.path,
      { params: this.params }
    );
  }
}
```

### 2.4 Function-based パターン

AWS SDK v3で採用されているパターン。Tree shakingとの相性がよく、バンドルサイズの最適化に優れる。

```typescript
// 利用例（AWS SDK v3 スタイル）
import { ExampleClient, GetUserCommand, ListUsersCommand } from "example-sdk";

const client = new ExampleClient({ apiKey: "sk_live_abc" });

const user = await client.send(new GetUserCommand({ userId: "user_123" }));
const users = await client.send(new ListUsersCommand({ limit: 20 }));

// Command クラスの実装
class GetUserCommand {
  readonly input: { userId: string };

  constructor(input: { userId: string }) {
    this.input = input;
  }

  resolveEndpoint(): string {
    return `/users/${this.input.userId}`;
  }

  resolveMethod(): string {
    return "GET";
  }
}

class ListUsersCommand {
  readonly input: { limit?: number; cursor?: string; role?: string };

  constructor(input: { limit?: number; cursor?: string; role?: string }) {
    this.input = input;
  }

  resolveEndpoint(): string {
    return "/users";
  }

  resolveMethod(): string {
    return "GET";
  }
}
```

### 2.5 Builder パターン

複雑な設定を持つオブジェクトの構築に適したパターン。Google Cloud SDKなどで採用されている。

```typescript
// リクエストのBuilder
const request = new SearchRequestBuilder()
  .query("typescript sdk")
  .filter("language", "ja")
  .dateRange(new Date("2024-01-01"), new Date("2024-12-31"))
  .pageSize(50)
  .includeMetadata(true)
  .build();

const results = await client.search.execute(request);

// Builder実装
class SearchRequestBuilder {
  private request: Partial<SearchRequest> = {};

  query(q: string): this {
    this.request.query = q;
    return this;
  }

  filter(field: string, value: string): this {
    if (!this.request.filters) this.request.filters = {};
    this.request.filters[field] = value;
    return this;
  }

  dateRange(from: Date, to: Date): this {
    this.request.dateFrom = from.toISOString();
    this.request.dateTo = to.toISOString();
    return this;
  }

  pageSize(size: number): this {
    if (size < 1 || size > 100) {
      throw new ValidationError("Page size must be between 1 and 100");
    }
    this.request.pageSize = size;
    return this;
  }

  includeMetadata(include: boolean): this {
    this.request.includeMetadata = include;
    return this;
  }

  build(): SearchRequest {
    if (!this.request.query) {
      throw new ValidationError("Query is required");
    }
    return this.request as SearchRequest;
  }
}
```

---

## 3. HTTP通信基盤

### 3.1 HTTPクライアントの抽象化

SDK内部のHTTP通信層は、外部ライブラリに依存しない抽象化が望ましい。テスタビリティと環境移植性を確保するためである。

```typescript
// HTTPクライアントインターフェース
interface HttpTransport {
  request<T>(options: HttpRequestOptions): Promise<HttpResponse<T>>;
}

interface HttpRequestOptions {
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  url: string;
  headers?: Record<string, string>;
  body?: unknown;
  timeout?: number;
  signal?: AbortSignal;
}

interface HttpResponse<T> {
  status: number;
  headers: Record<string, string>;
  data: T;
  requestId?: string;
}

// fetch ベースの実装
class FetchTransport implements HttpTransport {
  async request<T>(options: HttpRequestOptions): Promise<HttpResponse<T>> {
    const response = await fetch(options.url, {
      method: options.method,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: options.signal ?? AbortSignal.timeout(options.timeout ?? 30000),
    });

    const data = response.status === 204
      ? (undefined as T)
      : await response.json() as T;

    return {
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
      data,
      requestId: response.headers.get("x-request-id") ?? undefined,
    };
  }
}

// Node.js 環境用の実装（keep-alive対応）
class NodeTransport implements HttpTransport {
  private agent: https.Agent;

  constructor() {
    this.agent = new https.Agent({
      keepAlive: true,
      maxSockets: 50,
      maxFreeSockets: 10,
      timeout: 60000,
    });
  }

  async request<T>(options: HttpRequestOptions): Promise<HttpResponse<T>> {
    // node:https を使った実装
    // ...省略
  }
}
```

### 3.2 BaseResource: HTTP通信の共通基盤

```typescript
class BaseResource {
  constructor(private httpClient: HttpClient) {}

  protected async request<T>(
    method: string,
    path: string,
    options?: { params?: Record<string, any>; body?: any }
  ): Promise<T> {
    const url = new URL(path, this.httpClient.baseUrl);

    // クエリパラメータの構築
    if (options?.params) {
      for (const [key, value] of Object.entries(options.params)) {
        if (value !== undefined && value !== null) {
          if (Array.isArray(value)) {
            // 配列パラメータ: ?role=admin&role=user
            value.forEach(v => url.searchParams.append(key, String(v)));
          } else {
            url.searchParams.set(key, String(value));
          }
        }
      }
    }

    // ミドルウェアパイプラインを通じたリクエスト実行
    return this.httpClient.executeWithMiddleware<T>({
      method,
      url: url.toString(),
      body: options?.body,
    });
  }
}
```

### 3.3 ミドルウェアパイプライン

SDK内部の横断的関心事をミドルウェアとして分離する設計。認証、リトライ、ロギング、メトリクスなどを独立したモジュールとして管理できる。

```
+------------------------------------------------------------------+
|                  ミドルウェアパイプライン                             |
+------------------------------------------------------------------+
|                                                                    |
|  Request -->  [Auth]  -->  [Retry]  -->  [RateLimit]              |
|                                              |                     |
|                                         [Logging]                  |
|                                              |                     |
|                                         [Metrics]                  |
|                                              |                     |
|                                         [Transport]                |
|                                              |                     |
|  Response <-- [Transform] <-- [Validate] <---+                    |
|                                                                    |
+------------------------------------------------------------------+
```

```typescript
// ミドルウェア型定義
type Middleware = (
  request: HttpRequestOptions,
  next: (request: HttpRequestOptions) => Promise<HttpResponse<unknown>>
) => Promise<HttpResponse<unknown>>;

// 認証ミドルウェア
const authMiddleware = (authManager: AuthManager): Middleware => {
  return async (request, next) => {
    const token = await authManager.getToken();
    request.headers = {
      ...request.headers,
      Authorization: `Bearer ${token}`,
    };
    return next(request);
  };
};

// ロギングミドルウェア
const loggingMiddleware = (logger: Logger): Middleware => {
  return async (request, next) => {
    const startTime = Date.now();
    logger.debug(`[SDK] ${request.method} ${request.url}`);

    try {
      const response = await next(request);
      const duration = Date.now() - startTime;
      logger.debug(
        `[SDK] ${request.method} ${request.url} -> ${response.status} (${duration}ms)`
      );
      return response;
    } catch (error) {
      const duration = Date.now() - startTime;
      logger.error(
        `[SDK] ${request.method} ${request.url} -> ERROR (${duration}ms)`,
        error
      );
      throw error;
    }
  };
};

// メトリクスミドルウェア
const metricsMiddleware = (metrics: MetricsCollector): Middleware => {
  return async (request, next) => {
    const startTime = performance.now();
    try {
      const response = await next(request);
      metrics.recordLatency(request.method, request.url, performance.now() - startTime);
      metrics.incrementCounter(`sdk.request.${response.status}`);
      return response;
    } catch (error) {
      metrics.incrementCounter("sdk.request.error");
      throw error;
    }
  };
};

// HTTPクライアント（ミドルウェア統合）
class HttpClient {
  readonly baseUrl: string;
  private transport: HttpTransport;
  private middlewares: Middleware[];

  constructor(config: ResolvedClientConfig) {
    this.baseUrl = config.baseUrl;
    this.transport = new FetchTransport();
    this.middlewares = [];
  }

  use(middleware: Middleware): this {
    this.middlewares.push(middleware);
    return this;
  }

  async executeWithMiddleware<T>(
    options: HttpRequestOptions
  ): Promise<T> {
    // ミドルウェアチェーンの構築
    const execute = this.middlewares.reduceRight(
      (next, middleware) => (req: HttpRequestOptions) => middleware(req, next),
      (req: HttpRequestOptions) => this.transport.request<T>(req)
    );

    const response = await execute(options);
    if (response.status >= 400) {
      throw this.createError(response);
    }
    return response.data as T;
  }

  private createError(response: HttpResponse<unknown>): ExampleError {
    const body = response.data as any;
    return new ExampleError({
      status: response.status,
      code: body?.code ?? "UNKNOWN_ERROR",
      message: body?.message ?? body?.detail ?? `HTTP ${response.status}`,
      retryable: response.status >= 500 || response.status === 429,
      headers: response.headers,
      requestId: response.requestId,
    });
  }
}
```

---

## 4. リトライ戦略

### 4.1 エクスポネンシャルバックオフ

リトライは一時的なエラー（ネットワーク障害、サーバー過負荷、レートリミット）に対処するための重要な機構である。ただし、無秩序なリトライはサーバーへの負荷を増大させるため、エクスポネンシャルバックオフとジッターを組み合わせる。

```typescript
// リトライミドルウェア
const retryMiddleware = (config: RetryConfig): Middleware => {
  return async (request, next) => {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
      try {
        return await next(request);
      } catch (error) {
        lastError = error as Error;

        // リトライ可能かどうかの判定
        if (!shouldRetry(error, attempt, config)) {
          throw error;
        }

        // 待機時間の計算
        const delay = calculateDelay(attempt, error, config);
        await sleep(delay);
      }
    }

    throw lastError;
  };
};

function shouldRetry(
  error: unknown,
  attempt: number,
  config: RetryConfig
): boolean {
  if (attempt >= config.maxRetries) return false;

  if (error instanceof ExampleError) {
    // 明示的にリトライ不可のエラー
    if (!error.retryable) return false;

    // 429 Too Many Requests: Retry-After ヘッダーに従う
    if (error.status === 429) return true;

    // 5xx: サーバーエラーはリトライ
    if (error.status >= 500) return true;

    // 408 Request Timeout
    if (error.status === 408) return true;
  }

  // ネットワークエラー
  if (error instanceof TypeError && error.message.includes("fetch")) {
    return true;
  }

  return false;
}

function calculateDelay(
  attempt: number,
  error: unknown,
  config: RetryConfig
): number {
  // Retry-After ヘッダーがあればそれに従う
  if (error instanceof RateLimitError && error.retryAfter > 0) {
    return error.retryAfter * 1000;
  }

  // エクスポネンシャルバックオフ: baseDelay * 2^attempt
  const exponentialDelay = config.baseDelay * Math.pow(2, attempt);

  // 最大待機時間の制限
  const cappedDelay = Math.min(exponentialDelay, config.maxDelay);

  // フルジッター: [0, cappedDelay] の範囲でランダム化
  const jitter = Math.random() * cappedDelay;

  return jitter;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

interface RetryConfig {
  maxRetries: number;    // デフォルト: 3
  baseDelay: number;     // デフォルト: 1000ms
  maxDelay: number;      // デフォルト: 30000ms
}
```

### 4.2 ジッター戦略の比較

```
待機時間
  ^
  |                                          * フルジッター
  |                                     *         (推奨)
  |                                *
  |                           *    ..... 等間隔ジッター
  |                      *  ..
  |                 *  ..
  |            * ..          _____ 固定バックオフ
  |         *..         ____/      (ジッターなし)
  |      *.        ____/
  |   ..*     ____/
  |  .*  ____/
  | * __/
  |_/________________________________________________> リトライ回数
  0    1    2    3    4    5    6    7    8
```

| ジッター戦略 | 計算式 | 特徴 |
|-------------|-------|------|
| ジッターなし | `base * 2^attempt` | サーバーに集中負荷を与える |
| フルジッター | `random(0, base * 2^attempt)` | 負荷を最も均等に分散（推奨） |
| 等間隔ジッター | `base * 2^attempt / 2 + random(0, base * 2^attempt / 2)` | フルジッターより予測しやすい |
| 装飾的ジッター | `min(cap, random(base, prev * 3))` | AWS推奨、相関を持つランダム化 |

### 4.3 冪等性とリトライの安全性

リトライはすべてのHTTPメソッドに対して安全に実行できるわけではない。冪等でないリクエスト（POST）のリトライには冪等性キーが必要である。

```typescript
// 冪等性キーの自動付与
class IdempotencyMiddleware implements Middleware {
  async handle(
    request: HttpRequestOptions,
    next: (req: HttpRequestOptions) => Promise<HttpResponse<unknown>>
  ): Promise<HttpResponse<unknown>> {
    // POST リクエストに冪等性キーを自動付与
    if (request.method === "POST" && !request.headers?.["Idempotency-Key"]) {
      request.headers = {
        ...request.headers,
        "Idempotency-Key": crypto.randomUUID(),
      };
    }
    return next(request);
  }
}
```

---

## 5. エラー設計

### 5.1 エラー階層の設計

SDKのエラーは階層的に設計し、利用者が適切な粒度でエラーハンドリングできるようにする。基底クラスでキャッチすれば全エラーを処理でき、個別のサブクラスでキャッチすれば特定のエラーのみを処理できる。

```
+------------------------------------------------------------------+
|                     エラークラス階層                                 |
+------------------------------------------------------------------+
|                                                                    |
|  Error (JavaScript組み込み)                                        |
|    |                                                               |
|    +-- ExampleError (SDK基底エラー)                                 |
|          |                                                         |
|          +-- AuthenticationError (401)                              |
|          |     +-- InvalidApiKeyError                               |
|          |     +-- ExpiredTokenError                                |
|          |                                                         |
|          +-- AuthorizationError (403)                               |
|          |     +-- InsufficientPermissionError                      |
|          |                                                         |
|          +-- NotFoundError (404)                                    |
|          |                                                         |
|          +-- ConflictError (409)                                    |
|          |                                                         |
|          +-- ValidationError (422)                                  |
|          |     +-- InvalidParameterError                            |
|          |     +-- MissingRequiredFieldError                        |
|          |                                                         |
|          +-- RateLimitError (429)                                   |
|          |                                                         |
|          +-- InternalServerError (500)                              |
|          |                                                         |
|          +-- ServiceUnavailableError (503)                          |
|          |                                                         |
|          +-- NetworkError (接続系)                                  |
|          |     +-- TimeoutError                                     |
|          |     +-- ConnectionRefusedError                           |
|          |                                                         |
|          +-- ConfigurationError (SDK設定エラー)                     |
|                                                                    |
+------------------------------------------------------------------+
```

### 5.2 基底エラークラスの実装

```typescript
class ExampleError extends Error {
  /** HTTP ステータスコード（ネットワークエラーの場合は 0） */
  readonly status: number;

  /** APIから返されるエラーコード */
  readonly code: string;

  /** 自動リトライが安全かどうか */
  readonly retryable: boolean;

  /** レスポンスヘッダー */
  readonly headers: Record<string, string>;

  /** サーバー側のリクエストID（問い合わせ用） */
  readonly requestId?: string;

  /** エラー発生時のタイムスタンプ */
  readonly timestamp: Date;

  constructor(params: {
    status: number;
    code: string;
    message: string;
    retryable: boolean;
    headers: Record<string, string>;
    requestId?: string;
  }) {
    super(params.message);
    this.name = "ExampleError";
    this.status = params.status;
    this.code = params.code;
    this.retryable = params.retryable;
    this.headers = params.headers;
    this.requestId = params.requestId;
    this.timestamp = new Date();

    // V8 のスタックトレースを正しく保持
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /** 人間が読みやすい形式のエラー情報 */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      message: this.message,
      status: this.status,
      code: this.code,
      retryable: this.retryable,
      requestId: this.requestId,
      timestamp: this.timestamp.toISOString(),
    };
  }

  /** サポートへの問い合わせ用メッセージ生成 */
  toSupportMessage(): string {
    return [
      `Error: ${this.message}`,
      `Code: ${this.code}`,
      `Status: ${this.status}`,
      `Request ID: ${this.requestId ?? "N/A"}`,
      `Timestamp: ${this.timestamp.toISOString()}`,
    ].join("\n");
  }
}
```

### 5.3 サブクラスの実装

```typescript
// 認証エラー
class AuthenticationError extends ExampleError {
  constructor(message: string, requestId?: string) {
    super({
      status: 401,
      code: "AUTHENTICATION_ERROR",
      message: message || "Authentication failed. Please check your API key.",
      retryable: false,
      headers: {},
      requestId,
    });
    this.name = "AuthenticationError";
  }
}

// レートリミットエラー
class RateLimitError extends ExampleError {
  /** リトライまでの待機秒数 */
  readonly retryAfter: number;

  /** レート制限のリセット時刻（Unix timestamp） */
  readonly resetAt?: number;

  /** 残りリクエスト数 */
  readonly remaining?: number;

  constructor(params: {
    message: string;
    retryAfter: number;
    resetAt?: number;
    remaining?: number;
    requestId?: string;
  }) {
    super({
      status: 429,
      code: "RATE_LIMIT_EXCEEDED",
      message: params.message ||
        `Rate limit exceeded. Retry after ${params.retryAfter} seconds.`,
      retryable: true,
      headers: {},
      requestId: params.requestId,
    });
    this.name = "RateLimitError";
    this.retryAfter = params.retryAfter;
    this.resetAt = params.resetAt;
    this.remaining = params.remaining;
  }
}

// バリデーションエラー
class ValidationError extends ExampleError {
  /** フィールドごとのエラー詳細 */
  readonly fieldErrors: Array<{
    field: string;
    message: string;
    code: string;
    expected?: string;
    received?: string;
  }>;

  constructor(
    message: string,
    fieldErrors: Array<{
      field: string;
      message: string;
      code: string;
      expected?: string;
      received?: string;
    }>,
    requestId?: string
  ) {
    super({
      status: 422,
      code: "VALIDATION_ERROR",
      message,
      retryable: false,
      headers: {},
      requestId,
    });
    this.name = "ValidationError";
    this.fieldErrors = fieldErrors;
  }

  /** 特定フィールドのエラーを取得 */
  getFieldError(fieldName: string): string | undefined {
    return this.fieldErrors.find(e => e.field === fieldName)?.message;
  }

  /** 全フィールドエラーを文字列で表示 */
  formatErrors(): string {
    return this.fieldErrors
      .map(e => `  - ${e.field}: ${e.message}`)
      .join("\n");
  }
}

// ネットワークエラー
class NetworkError extends ExampleError {
  /** 元のネットワークエラー */
  readonly cause: Error;

  constructor(cause: Error) {
    super({
      status: 0,
      code: "NETWORK_ERROR",
      message: `Network error: ${cause.message}. Please check your internet connection.`,
      retryable: true,
      headers: {},
    });
    this.name = "NetworkError";
    this.cause = cause;
  }
}

// タイムアウトエラー
class TimeoutError extends NetworkError {
  readonly timeoutMs: number;

  constructor(timeoutMs: number) {
    super(new Error(`Request timed out after ${timeoutMs}ms`));
    this.name = "TimeoutError";
    this.timeoutMs = timeoutMs;
  }
}
```

### 5.4 エラーハンドリングのパターン

```typescript
// パターン1: 型に基づく分岐（推奨）
try {
  const user = await client.users.create({
    name: "",
    email: "invalid-email",
  });
} catch (error) {
  if (error instanceof ValidationError) {
    // フィールドごとのエラーを表示
    console.log("Validation failed:");
    console.log(error.formatErrors());
    // 例:
    //   - name: Name must not be empty
    //   - email: Invalid email format
  } else if (error instanceof RateLimitError) {
    console.log(`Rate limited. Retry after ${error.retryAfter}s`);
    // SDK の自動リトライを超えた場合のみここに到達
  } else if (error instanceof AuthenticationError) {
    console.log("Invalid API key. Please check your configuration.");
    // 設定の見直しを促す
  } else if (error instanceof NetworkError) {
    console.log("Network issue. Please check your connection.");
  } else if (error instanceof ExampleError) {
    // その他のAPIエラー
    console.log(`API error [${error.code}]: ${error.message}`);
    console.log(`Request ID: ${error.requestId}`);
  } else {
    // 予期しないエラー
    throw error;
  }
}

// パターン2: エラーコードに基づく分岐
try {
  await client.users.get("nonexistent");
} catch (error) {
  if (error instanceof ExampleError) {
    switch (error.code) {
      case "NOT_FOUND":
        console.log("User not found");
        break;
      case "RATE_LIMIT_EXCEEDED":
        console.log("Please slow down");
        break;
      default:
        console.log(`Unexpected error: ${error.code}`);
    }
  }
}

// パターン3: Result型パターン（エラーをthrowしない）
type Result<T, E = ExampleError> =
  | { success: true; data: T }
  | { success: false; error: E };

async function safeGetUser(
  client: ExampleClient,
  id: string
): Promise<Result<User>> {
  try {
    const data = await client.users.get(id);
    return { success: true, data };
  } catch (error) {
    if (error instanceof ExampleError) {
      return { success: false, error };
    }
    throw error; // 予期しないエラーは再throw
  }
}

// 使用例
const result = await safeGetUser(client, "123");
if (result.success) {
  console.log(result.data.name);
} else {
  console.log(result.error.message);
}
```

---

## 6. 認証パターン

### 6.1 認証方式の比較

| 認証方式 | セキュリティ | 実装難易度 | ユースケース |
|---------|-----------|----------|------------|
| API Key | 中 | 低 | サーバー間通信、個人利用 |
| Bearer Token | 高 | 中 | モバイルアプリ、SPA |
| OAuth 2.0 PKCE | 高 | 高 | パブリッククライアント |
| mTLS | 非常に高 | 高 | 金融、医療 |
| HMAC Signature | 高 | 中 | Webhook、S2S |

### 6.2 認証マネージャーの実装

```typescript
// 認証戦略のインターフェース
interface AuthStrategy {
  /** リクエストに認証情報を付与 */
  authenticate(headers: Record<string, string>): Promise<Record<string, string>>;
  /** トークンの有効期限切れをチェック */
  isExpired(): boolean;
  /** トークンのリフレッシュ（必要な場合） */
  refresh?(): Promise<void>;
}

// API Key 認証
class ApiKeyAuth implements AuthStrategy {
  constructor(
    private apiKey: string,
    private headerName: string = "Authorization",
    private prefix: string = "Bearer"
  ) {}

  async authenticate(
    headers: Record<string, string>
  ): Promise<Record<string, string>> {
    return {
      ...headers,
      [this.headerName]: `${this.prefix} ${this.apiKey}`,
    };
  }

  isExpired(): boolean {
    return false; // API Key は期限切れにならない
  }
}

// OAuth 2.0 トークン自動リフレッシュ認証
class OAuth2Auth implements AuthStrategy {
  private accessToken: string | null = null;
  private expiresAt: number = 0;
  private refreshPromise: Promise<void> | null = null;

  constructor(
    private clientId: string,
    private clientSecret: string,
    private refreshToken: string,
    private tokenEndpoint: string = "https://auth.example.com/oauth/token"
  ) {}

  async authenticate(
    headers: Record<string, string>
  ): Promise<Record<string, string>> {
    if (this.isExpired()) {
      await this.refresh();
    }
    return {
      ...headers,
      Authorization: `Bearer ${this.accessToken}`,
    };
  }

  isExpired(): boolean {
    // 有効期限の60秒前に更新（バッファ）
    return !this.accessToken || Date.now() >= this.expiresAt - 60000;
  }

  async refresh(): Promise<void> {
    // 同時リフレッシュの防止（デデュプリケーション）
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = this.doRefresh();
    try {
      await this.refreshPromise;
    } finally {
      this.refreshPromise = null;
    }
  }

  private async doRefresh(): Promise<void> {
    const response = await fetch(this.tokenEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "refresh_token",
        client_id: this.clientId,
        client_secret: this.clientSecret,
        refresh_token: this.refreshToken,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new AuthenticationError(
        `Token refresh failed: ${error.error_description ?? response.statusText}`
      );
    }

    const data = await response.json();
    this.accessToken = data.access_token;
    this.expiresAt = Date.now() + data.expires_in * 1000;

    // refresh_token が更新された場合
    if (data.refresh_token) {
      this.refreshToken = data.refresh_token;
    }
  }
}

// HMAC署名認証（Webhook検証用）
class HmacAuth implements AuthStrategy {
  constructor(
    private secretKey: string,
    private algorithm: string = "sha256"
  ) {}

  async authenticate(
    headers: Record<string, string>
  ): Promise<Record<string, string>> {
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const payload = `${timestamp}.${headers["x-request-body"] ?? ""}`;

    const signature = await this.computeHmac(payload);

    return {
      ...headers,
      "X-Signature": signature,
      "X-Timestamp": timestamp,
    };
  }

  isExpired(): boolean {
    return false;
  }

  private async computeHmac(payload: string): Promise<string> {
    const encoder = new TextEncoder();
    const key = await crypto.subtle.importKey(
      "raw",
      encoder.encode(this.secretKey),
      { name: "HMAC", hash: `SHA-256` },
      false,
      ["sign"]
    );
    const signature = await crypto.subtle.sign(
      "HMAC",
      key,
      encoder.encode(payload)
    );
    return Array.from(new Uint8Array(signature))
      .map(b => b.toString(16).padStart(2, "0"))
      .join("");
  }
}
```

### 6.3 クライアント初期化パターン

```typescript
// ファクトリーパターンによる柔軟な初期化

// パターン1: API Key（最もシンプル）
const client = ExampleClient.withApiKey("sk_live_abc123");

// パターン2: OAuth 2.0 Bearer Token
const client = ExampleClient.withAccessToken("eyJhbG...");

// パターン3: OAuth 2.0 with auto-refresh
const client = ExampleClient.withOAuth({
  clientId: "client_123",
  clientSecret: "secret_456",
  refreshToken: "rt_789",
});

// パターン4: 環境変数から自動検出
const client = ExampleClient.fromEnvironment();
// EXAMPLE_API_KEY, EXAMPLE_CLIENT_ID 等を自動検出

// ファクトリーメソッドの実装
class ExampleClient {
  private constructor(private config: ResolvedClientConfig) {
    // ...初期化処理
  }

  static withApiKey(apiKey: string, options?: Partial<ClientConfig>): ExampleClient {
    return new ExampleClient({
      ...DEFAULT_CONFIG,
      ...options,
      auth: new ApiKeyAuth(apiKey),
    });
  }

  static withAccessToken(token: string, options?: Partial<ClientConfig>): ExampleClient {
    return new ExampleClient({
      ...DEFAULT_CONFIG,
      ...options,
      auth: new ApiKeyAuth(token, "Authorization", "Bearer"),
    });
  }

  static withOAuth(
    oauthConfig: OAuthConfig,
    options?: Partial<ClientConfig>
  ): ExampleClient {
    return new ExampleClient({
      ...DEFAULT_CONFIG,
      ...options,
      auth: new OAuth2Auth(
        oauthConfig.clientId,
        oauthConfig.clientSecret,
        oauthConfig.refreshToken,
        oauthConfig.tokenEndpoint
      ),
    });
  }

  static fromEnvironment(options?: Partial<ClientConfig>): ExampleClient {
    const apiKey = process.env.EXAMPLE_API_KEY;
    if (apiKey) {
      return ExampleClient.withApiKey(apiKey, options);
    }

    const clientId = process.env.EXAMPLE_CLIENT_ID;
    const clientSecret = process.env.EXAMPLE_CLIENT_SECRET;
    const refreshToken = process.env.EXAMPLE_REFRESH_TOKEN;

    if (clientId && clientSecret && refreshToken) {
      return ExampleClient.withOAuth(
        { clientId, clientSecret, refreshToken },
        options
      );
    }

    throw new ConfigurationError(
      "No authentication credentials found. " +
      "Set EXAMPLE_API_KEY or EXAMPLE_CLIENT_ID/EXAMPLE_CLIENT_SECRET/EXAMPLE_REFRESH_TOKEN " +
      "environment variables."
    );
  }
}
```

---

## 7. ページネーション

### 7.1 ページネーション戦略

API のページネーション方式に応じたSDK側の抽象化パターンを示す。

| 方式 | 仕組み | メリット | デメリット |
|-----|-------|---------|----------|
| カーソルベース | `cursor` パラメータで次ページを指定 | リアルタイムデータに強い | 任意ページジャンプ不可 |
| オフセットベース | `offset` + `limit` で位置指定 | 任意ページにジャンプ可能 | データ変動時にずれる |
| キーセットベース | `after_id` で最後のIDの次から取得 | 高パフォーマンス | ソート順が制限される |

### 7.2 自動ページネーションイテレータ

```typescript
// ページネーション抽象化

// 型定義
interface PaginatedResponse<T> {
  data: T[];
  hasNextPage: boolean;
  nextCursor: string | null;
  totalCount?: number;
}

interface PaginationParams {
  limit?: number;
  cursor?: string;
}

// 自動イテレータの実装
class AutoPaginator<T> implements AsyncIterable<T> {
  constructor(
    private fetchPage: (params: PaginationParams) => Promise<PaginatedResponse<T>>,
    private params: Omit<PaginationParams, "cursor"> = {}
  ) {}

  async *[Symbol.asyncIterator](): AsyncIterator<T> {
    let cursor: string | undefined;
    do {
      const response = await this.fetchPage({
        ...this.params,
        cursor,
      });
      for (const item of response.data) {
        yield item;
      }
      cursor = response.nextCursor ?? undefined;
    } while (cursor);
  }

  /** 全データを配列として取得 */
  async toArray(): Promise<T[]> {
    const items: T[] = [];
    for await (const item of this) {
      items.push(item);
    }
    return items;
  }

  /** 最初のN件を取得 */
  async take(n: number): Promise<T[]> {
    const items: T[] = [];
    for await (const item of this) {
      items.push(item);
      if (items.length >= n) break;
    }
    return items;
  }

  /** 条件に合う最初の要素を取得 */
  async find(predicate: (item: T) => boolean): Promise<T | undefined> {
    for await (const item of this) {
      if (predicate(item)) return item;
    }
    return undefined;
  }

  /** 全要素に対してコールバックを実行 */
  async forEach(callback: (item: T, index: number) => void | Promise<void>): Promise<void> {
    let index = 0;
    for await (const item of this) {
      await callback(item, index++);
    }
  }

  /** 全要素を変換して配列で返す */
  async map<U>(fn: (item: T) => U): Promise<U[]> {
    const results: U[] = [];
    for await (const item of this) {
      results.push(fn(item));
    }
    return results;
  }

  /** 条件に合う要素のみをフィルタして配列で返す */
  async filter(predicate: (item: T) => boolean): Promise<T[]> {
    const results: T[] = [];
    for await (const item of this) {
      if (predicate(item)) results.push(item);
    }
    return results;
  }
}

// リソースクラスでの使用
class UsersResource extends BaseResource {
  async get(id: string): Promise<User> {
    return this.request<User>("GET", `/users/${id}`);
  }

  async list(params?: ListUsersParams): Promise<PaginatedResponse<User>> {
    return this.request<PaginatedResponse<User>>("GET", "/users", { params });
  }

  async create(data: CreateUserParams): Promise<User> {
    return this.request<User>("POST", "/users", { body: data });
  }

  async update(id: string, data: Partial<CreateUserParams>): Promise<User> {
    return this.request<User>("PATCH", `/users/${id}`, { body: data });
  }

  async delete(id: string): Promise<void> {
    return this.request<void>("DELETE", `/users/${id}`);
  }

  /** 自動ページネーションイテレータを返す */
  listAll(params?: Omit<ListUsersParams, "cursor">): AutoPaginator<User> {
    return new AutoPaginator(
      (paginationParams) => this.list({ ...params, ...paginationParams }),
      params
    );
  }
}

// 利用例
// 全ユーザーを反復処理
for await (const user of client.users.listAll({ role: "admin" })) {
  console.log(user.name);
}

// 最初の100件を配列で取得
const first100 = await client.users.listAll().take(100);

// 条件に合うユーザーを検索
const targetUser = await client.users
  .listAll({ role: "admin" })
  .find(user => user.email === "admin@example.com");
```

---

## 8. バージョニング戦略

### 8.1 セマンティックバージョニング

SDKのバージョニングにはセマンティックバージョニング（SemVer）を厳格に適用する。

```
MAJOR.MINOR.PATCH
  |     |     |
  |     |     +-- バグ修正（後方互換性あり）
  |     +-------- 機能追加（後方互換性あり）
  +-------------- ブレイキングチェンジ（後方互換性なし）

例:
  1.0.0 → 1.0.1  パッチ: バグ修正
  1.0.1 → 1.1.0  マイナー: 新メソッド追加
  1.1.0 → 2.0.0  メジャー: メソッドシグネチャ変更
```

### 8.2 ブレイキングチェンジの定義

何が「ブレイキングチェンジ」に該当するかを明確に定義することが重要である。

| 変更の種類 | ブレイキング？ | 理由 |
|-----------|-------------|------|
| メソッドの削除 | はい | 既存コードがコンパイルエラーになる |
| 必須パラメータの追加 | はい | 既存の呼び出しが失敗する |
| 戻り値の型変更 | はい | 型チェックが壊れる |
| オプショナルパラメータの追加 | いいえ | 既存コードは影響を受けない |
| 新メソッドの追加 | いいえ | 既存コードは影響を受けない |
| エラーメッセージの変更 | いいえ（※） | ※ メッセージ文字列で分岐している場合は問題 |
| 新しいエラーサブクラスの追加 | いいえ | 既存の catch ブロックで補足される |
| デフォルト値の変更 | 場合による | 振る舞いが変わる可能性がある |

### 8.3 APIバージョンとSDKバージョンの関係

```typescript
// SDKバージョンとAPIバージョンは独立して管理する

// SDK v2.3.1 は API v1 と API v2 の両方をサポート
const clientV1 = new ExampleClient({
  apiKey: "sk_live_abc",
  apiVersion: "2024-01-01", // API バージョンの日付指定（Stripe方式）
});

const clientV2 = new ExampleClient({
  apiKey: "sk_live_abc",
  apiVersion: "2024-06-15",
});

// API バージョンヘッダーの自動付与
class ApiVersionMiddleware implements Middleware {
  constructor(private apiVersion: string) {}

  async handle(
    request: HttpRequestOptions,
    next: (req: HttpRequestOptions) => Promise<HttpResponse<unknown>>
  ): Promise<HttpResponse<unknown>> {
    request.headers = {
      ...request.headers,
      "Example-Version": this.apiVersion,
    };
    return next(request);
  }
}
```

### 8.4 非推奨（Deprecation）の管理

```typescript
// 非推奨メソッドの警告
class UsersResource extends BaseResource {
  /**
   * @deprecated v2.0.0 で削除予定。代わりに `list()` を使用してください。
   */
  async getAll(params?: ListUsersParams): Promise<User[]> {
    if (typeof process !== "undefined" && process.emitWarning) {
      process.emitWarning(
        "users.getAll() is deprecated and will be removed in v2.0.0. " +
        "Use users.list() instead.",
        "DeprecationWarning"
      );
    }
    const response = await this.list(params);
    return response.data;
  }

  async list(params?: ListUsersParams): Promise<PaginatedResponse<User>> {
    return this.request<PaginatedResponse<User>>("GET", "/users", { params });
  }
}

// TypeScript の @deprecated JSDoc タグ
// IDE がメソッドに取り消し線を表示し、利用者に視覚的に通知
```

---

## 9. テスタビリティ

### 9.1 テスト戦略の概要

SDKのテストは3層で構成する。

| テストレイヤー | 対象 | ツール | 実行頻度 |
|-------------|------|-------|---------|
| ユニットテスト | 個別メソッド、バリデーション | Jest/Vitest | CI毎回 |
| 統合テスト | HTTPクライアント、認証フロー | MSW | CI毎回 |
| E2Eテスト | 実API接続 | 本番sandbox | リリース前 |

### 9.2 インターフェースベースのモック

```typescript
// SDKの各リソースにインターフェースを定義
interface IUsersResource {
  get(id: string): Promise<User>;
  list(params?: ListUsersParams): Promise<PaginatedResponse<User>>;
  create(data: CreateUserParams): Promise<User>;
  update(id: string, data: Partial<CreateUserParams>): Promise<User>;
  delete(id: string): Promise<void>;
}

interface IExampleClient {
  readonly users: IUsersResource;
  readonly orders: IOrdersResource;
}

// テスト用モッククライアント
class MockExampleClient implements IExampleClient {
  readonly users: MockUsersResource;
  readonly orders: MockOrdersResource;

  constructor() {
    this.users = new MockUsersResource();
    this.orders = new MockOrdersResource();
  }
}

class MockUsersResource implements IUsersResource {
  private store: Map<string, User> = new Map();
  private callLog: Array<{ method: string; args: any[] }> = [];

  // テスト用のデータセットアップ
  seed(users: User[]): void {
    for (const user of users) {
      this.store.set(user.id, user);
    }
  }

  // 呼び出し履歴の確認
  getCalls(method: string): any[][] {
    return this.callLog
      .filter(c => c.method === method)
      .map(c => c.args);
  }

  async get(id: string): Promise<User> {
    this.callLog.push({ method: "get", args: [id] });
    const user = this.store.get(id);
    if (!user) {
      throw new NotFoundError(`User ${id} not found`);
    }
    return user;
  }

  async list(params?: ListUsersParams): Promise<PaginatedResponse<User>> {
    this.callLog.push({ method: "list", args: [params] });
    let users = Array.from(this.store.values());
    if (params?.role) {
      users = users.filter(u => u.role === params.role);
    }
    return {
      data: users.slice(0, params?.limit ?? 20),
      hasNextPage: false,
      nextCursor: null,
    };
  }

  async create(data: CreateUserParams): Promise<User> {
    this.callLog.push({ method: "create", args: [data] });
    const user: User = {
      id: `user_${Date.now()}`,
      name: data.name,
      email: data.email,
      role: data.role ?? "user",
      createdAt: new Date().toISOString(),
    };
    this.store.set(user.id, user);
    return user;
  }

  async update(id: string, data: Partial<CreateUserParams>): Promise<User> {
    this.callLog.push({ method: "update", args: [id, data] });
    const existing = this.store.get(id);
    if (!existing) throw new NotFoundError(`User ${id} not found`);
    const updated = { ...existing, ...data };
    this.store.set(id, updated);
    return updated;
  }

  async delete(id: string): Promise<void> {
    this.callLog.push({ method: "delete", args: [id] });
    if (!this.store.has(id)) throw new NotFoundError(`User ${id} not found`);
    this.store.delete(id);
  }
}

// テストコード例
describe("UserService", () => {
  let client: MockExampleClient;

  beforeEach(() => {
    client = new MockExampleClient();
    client.users.seed([
      {
        id: "user_1",
        name: "Alice",
        email: "alice@example.com",
        role: "admin",
        createdAt: "2024-01-01T00:00:00Z",
      },
      {
        id: "user_2",
        name: "Bob",
        email: "bob@example.com",
        role: "user",
        createdAt: "2024-01-02T00:00:00Z",
      },
    ]);
  });

  test("get user by ID", async () => {
    const user = await client.users.get("user_1");
    expect(user.name).toBe("Alice");
  });

  test("list admin users", async () => {
    const result = await client.users.list({ role: "admin" });
    expect(result.data).toHaveLength(1);
    expect(result.data[0].name).toBe("Alice");
  });

  test("throw NotFoundError for unknown user", async () => {
    await expect(client.users.get("unknown"))
      .rejects.toThrow(NotFoundError);
  });
});
```

### 9.3 MSW（Mock Service Worker）によるHTTPレベルテスト

```typescript
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

const handlers = [
  http.get("https://api.example.com/v1/users/:id", ({ params }) => {
    if (params.id === "nonexistent") {
      return HttpResponse.json(
        { code: "NOT_FOUND", message: "User not found" },
        { status: 404 }
      );
    }
    return HttpResponse.json({
      id: params.id,
      name: "Test User",
      email: "test@example.com",
      role: "user",
      createdAt: "2024-01-01T00:00:00Z",
    });
  }),

  http.post("https://api.example.com/v1/users", async ({ request }) => {
    const body = await request.json() as any;
    if (!body.name || !body.email) {
      return HttpResponse.json(
        {
          code: "VALIDATION_ERROR",
          message: "Validation failed",
          errors: [
            ...(!body.name ? [{ field: "name", message: "Name is required" }] : []),
            ...(!body.email ? [{ field: "email", message: "Email is required" }] : []),
          ],
        },
        { status: 422 }
      );
    }
    return HttpResponse.json(
      { id: "new_user", ...body, role: body.role ?? "user", createdAt: new Date().toISOString() },
      { status: 201 }
    );
  }),

  // レートリミットのシミュレーション
  http.get("https://api.example.com/v1/rate-limited", () => {
    return HttpResponse.json(
      { code: "RATE_LIMIT_EXCEEDED", message: "Too many requests" },
      {
        status: 429,
        headers: { "Retry-After": "5" },
      }
    );
  }),
];

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe("ExampleClient with MSW", () => {
  const client = new ExampleClient({ apiKey: "test_key" });

  test("get user returns user data", async () => {
    const user = await client.users.get("123");
    expect(user.name).toBe("Test User");
    expect(user.email).toBe("test@example.com");
  });

  test("get nonexistent user throws NotFoundError", async () => {
    await expect(client.users.get("nonexistent"))
      .rejects.toThrow(NotFoundError);
  });

  test("create user with missing fields throws ValidationError", async () => {
    try {
      await client.users.create({ name: "", email: "" } as any);
      fail("Should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(ValidationError);
      if (error instanceof ValidationError) {
        expect(error.fieldErrors).toHaveLength(2);
      }
    }
  });

  test("handles server errors with retry", async () => {
    let callCount = 0;
    server.use(
      http.get("https://api.example.com/v1/users/retry-test", () => {
        callCount++;
        if (callCount < 3) {
          return HttpResponse.json(
            { message: "Internal error" },
            { status: 500 }
          );
        }
        return HttpResponse.json({
          id: "retry-test",
          name: "Success",
          email: "ok@example.com",
          role: "user",
          createdAt: "2024-01-01T00:00:00Z",
        });
      })
    );

    const user = await client.users.get("retry-test");
    expect(user.name).toBe("Success");
    expect(callCount).toBe(3);
  });
});
```

