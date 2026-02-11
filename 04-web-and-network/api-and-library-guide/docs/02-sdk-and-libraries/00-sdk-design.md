# SDK設計

> SDKはAPIの利用体験を決定づけるフロントライン。型安全なクライアント設計、Builderパターン、エラーハンドリング、リトライ戦略、認証の抽象化まで、開発者に愛されるSDK設計のベストプラクティスを習得する。

## この章で学ぶこと

- [ ] SDK設計の原則とDX（開発者体験）を理解する
- [ ] 型安全なクライアント実装パターンを把握する
- [ ] リトライ、認証、ページネーションの抽象化を学ぶ

---

## 1. SDK設計の原則

```
SDKの役割:
  → APIの複雑さを隠蔽し、開発者にシンプルなインターフェースを提供

  良いSDK:
  ✓ 5分で使い始められる（Quick Start）
  ✓ 型補完が効く（TypeScript / JSDoc）
  ✓ エラーメッセージが親切
  ✓ 認証の設定が簡単
  ✓ ページネーションが自動化
  ✓ リトライが組み込み
  ✓ テストしやすい（モック可能）

  悪いSDK:
  ✗ ドキュメントを読まないと使えない
  ✗ 型情報がない
  ✗ エラーが不透明
  ✗ 認証のたびにトークンを手動管理
  ✗ 依存が多すぎる

DX（Developer Experience）の指標:
  → Time to First API Call: 最初のAPI呼び出しまでの時間
  → Lines of Code: 基本操作に必要なコード行数
  → Error Recovery: エラーから復帰するまでの時間
  → Discovery: 機能を発見する容易さ（補完、ドキュメント）
```

---

## 2. クライアント設計パターン

```typescript
// ① Fluent API / Method Chaining パターン
const users = await client.users
  .list()
  .filter({ role: 'admin' })
  .sort('-createdAt')
  .limit(20)
  .execute();

// ② Resource-based パターン（推奨）
const user = await client.users.get('123');
const users = await client.users.list({ role: 'admin', limit: 20 });
const newUser = await client.users.create({ name: 'Taro', email: 'taro@example.com' });
const updated = await client.users.update('123', { name: 'Updated' });
await client.users.delete('123');

// ③ Function-based パターン
const user = await getUser(client, '123');
const users = await listUsers(client, { role: 'admin' });
```

```typescript
// Resource-based パターンの実装

// --- クライアント本体 ---
class ExampleClient {
  private config: ClientConfig;
  readonly users: UsersResource;
  readonly orders: OrdersResource;

  constructor(config: { apiKey: string; baseUrl?: string }) {
    this.config = {
      apiKey: config.apiKey,
      baseUrl: config.baseUrl ?? 'https://api.example.com/v1',
      timeout: 30000,
      maxRetries: 3,
    };
    this.users = new UsersResource(this.config);
    this.orders = new OrdersResource(this.config);
  }
}

// --- 型定義 ---
interface User {
  id: string;
  name: string;
  email: string;
  role: 'user' | 'admin';
  createdAt: string;
}

interface CreateUserParams {
  name: string;
  email: string;
  role?: 'user' | 'admin';
}

interface ListUsersParams {
  role?: 'user' | 'admin';
  limit?: number;
  cursor?: string;
  sort?: string;
}

interface PaginatedResponse<T> {
  data: T[];
  hasNextPage: boolean;
  nextCursor: string | null;
}

// --- リソースクラス ---
class UsersResource extends BaseResource {
  async get(id: string): Promise<User> {
    return this.request<User>('GET', `/users/${id}`);
  }

  async list(params?: ListUsersParams): Promise<PaginatedResponse<User>> {
    return this.request<PaginatedResponse<User>>('GET', '/users', { params });
  }

  async create(data: CreateUserParams): Promise<User> {
    return this.request<User>('POST', '/users', { body: data });
  }

  async update(id: string, data: Partial<CreateUserParams>): Promise<User> {
    return this.request<User>('PATCH', `/users/${id}`, { body: data });
  }

  async delete(id: string): Promise<void> {
    return this.request<void>('DELETE', `/users/${id}`);
  }

  // イテレーター: 全ページを自動取得
  async *listAll(params?: Omit<ListUsersParams, 'cursor'>): AsyncGenerator<User> {
    let cursor: string | undefined;
    do {
      const response = await this.list({ ...params, cursor });
      for (const item of response.data) {
        yield item;
      }
      cursor = response.nextCursor ?? undefined;
    } while (cursor);
  }
}
```

---

## 3. HTTP基盤

```typescript
// BaseResource: HTTP通信の共通基盤

class BaseResource {
  constructor(private config: ClientConfig) {}

  protected async request<T>(
    method: string,
    path: string,
    options?: { params?: Record<string, any>; body?: any }
  ): Promise<T> {
    const url = new URL(path, this.config.baseUrl);

    // クエリパラメータ
    if (options?.params) {
      for (const [key, value] of Object.entries(options.params)) {
        if (value !== undefined) url.searchParams.set(key, String(value));
      }
    }

    // リトライ付きリクエスト
    return this.requestWithRetry<T>(async () => {
      const response = await fetch(url.toString(), {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`,
          'User-Agent': 'example-sdk/1.0.0',
        },
        body: options?.body ? JSON.stringify(options.body) : undefined,
        signal: AbortSignal.timeout(this.config.timeout),
      });

      if (!response.ok) {
        throw await this.handleError(response);
      }

      if (response.status === 204) return undefined as T;
      return response.json() as Promise<T>;
    });
  }

  // エクスポネンシャルバックオフ付きリトライ
  private async requestWithRetry<T>(
    fn: () => Promise<T>,
    attempt = 0
  ): Promise<T> {
    try {
      return await fn();
    } catch (error) {
      if (
        attempt < this.config.maxRetries &&
        error instanceof ExampleError &&
        error.retryable
      ) {
        const delay = Math.min(1000 * 2 ** attempt, 30000);
        const jitter = Math.random() * 1000;
        await new Promise(r => setTimeout(r, delay + jitter));
        return this.requestWithRetry(fn, attempt + 1);
      }
      throw error;
    }
  }

  // エラーハンドリング
  private async handleError(response: Response): Promise<ExampleError> {
    const body = await response.json().catch(() => ({}));
    return new ExampleError({
      status: response.status,
      code: body.code ?? 'UNKNOWN_ERROR',
      message: body.detail ?? response.statusText,
      retryable: response.status >= 500 || response.status === 429,
      headers: Object.fromEntries(response.headers.entries()),
    });
  }
}
```

---

## 4. エラー設計

```typescript
// カスタムエラークラス
class ExampleError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;
  readonly headers: Record<string, string>;

  constructor(params: {
    status: number;
    code: string;
    message: string;
    retryable: boolean;
    headers: Record<string, string>;
  }) {
    super(params.message);
    this.name = 'ExampleError';
    this.status = params.status;
    this.code = params.code;
    this.retryable = params.retryable;
    this.headers = params.headers;
  }
}

// 型付きエラー（サブクラス）
class AuthenticationError extends ExampleError {
  constructor(message: string) {
    super({ status: 401, code: 'AUTHENTICATION_ERROR', message, retryable: false, headers: {} });
  }
}

class RateLimitError extends ExampleError {
  readonly retryAfter: number;
  constructor(message: string, retryAfter: number) {
    super({ status: 429, code: 'RATE_LIMIT_EXCEEDED', message, retryable: true, headers: {} });
    this.retryAfter = retryAfter;
  }
}

class ValidationError extends ExampleError {
  readonly errors: Array<{ field: string; message: string }>;
  constructor(message: string, errors: Array<{ field: string; message: string }>) {
    super({ status: 422, code: 'VALIDATION_ERROR', message, retryable: false, headers: {} });
    this.errors = errors;
  }
}

// 使用例
try {
  await client.users.create({ name: '', email: 'invalid' });
} catch (error) {
  if (error instanceof ValidationError) {
    error.errors.forEach(e => console.log(`${e.field}: ${e.message}`));
  } else if (error instanceof RateLimitError) {
    console.log(`Retry after ${error.retryAfter} seconds`);
  } else if (error instanceof AuthenticationError) {
    console.log('Please check your API key');
  }
}
```

---

## 5. 認証パターン

```typescript
// 複数の認証方式をサポート

// ① API Key（最もシンプル）
const client = new ExampleClient({
  apiKey: 'sk_live_abc123',
});

// ② OAuth 2.0 Bearer Token
const client = new ExampleClient({
  accessToken: 'eyJhbG...',
});

// ③ OAuth 2.0 with auto-refresh
const client = new ExampleClient({
  clientId: 'client_123',
  clientSecret: 'secret_456',
  refreshToken: 'rt_789',
  // SDKが自動的にトークンを更新
});

// 内部実装
class AuthManager {
  private accessToken: string | null = null;
  private expiresAt: number = 0;

  async getToken(): Promise<string> {
    if (this.accessToken && Date.now() < this.expiresAt - 60000) {
      return this.accessToken;
    }
    return this.refreshAccessToken();
  }

  private async refreshAccessToken(): Promise<string> {
    const response = await fetch('https://auth.example.com/oauth/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        grant_type: 'refresh_token',
        client_id: this.clientId,
        client_secret: this.clientSecret,
        refresh_token: this.refreshToken,
      }),
    });
    const data = await response.json();
    this.accessToken = data.access_token;
    this.expiresAt = Date.now() + data.expires_in * 1000;
    return this.accessToken;
  }
}
```

---

## 6. テスタビリティ

```typescript
// SDKのモック方法

// ① インターフェースベース
interface UserService {
  get(id: string): Promise<User>;
  list(params?: ListUsersParams): Promise<PaginatedResponse<User>>;
  create(data: CreateUserParams): Promise<User>;
}

// テストではモック実装を注入
class MockUserService implements UserService {
  async get(id: string): Promise<User> {
    return { id, name: 'Mock User', email: 'mock@example.com', role: 'user', createdAt: '' };
  }
  async list(): Promise<PaginatedResponse<User>> {
    return { data: [], hasNextPage: false, nextCursor: null };
  }
  async create(data: CreateUserParams): Promise<User> {
    return { id: 'new', ...data, role: data.role ?? 'user', createdAt: '' };
  }
}

// ② MSW（Mock Service Worker）
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

const server = setupServer(
  http.get('https://api.example.com/v1/users/:id', ({ params }) => {
    return HttpResponse.json({
      id: params.id,
      name: 'Test User',
      email: 'test@example.com',
    });
  }),
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

test('get user', async () => {
  const client = new ExampleClient({ apiKey: 'test' });
  const user = await client.users.get('123');
  expect(user.name).toBe('Test User');
});
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 設計原則 | 5分でQuick Start、型安全、親切なエラー |
| クライアント | Resource-basedパターンが最も直感的 |
| リトライ | エクスポネンシャルバックオフ + ジッター |
| エラー | 型付きエラークラスで種別判定 |
| 認証 | トークンの自動更新を組み込み |
| テスト | インターフェース分離 + MSW |

---

## 次に読むべきガイド
→ [[01-npm-package-development.md]] — npmパッケージ開発

---

## 参考文献
1. Stripe. "Stripe API Reference." stripe.com/docs, 2024.
2. Twilio. "SDK Design Principles." twilio.com/docs, 2024.
3. AWS. "SDK Design Guide." aws.amazon.com, 2024.
