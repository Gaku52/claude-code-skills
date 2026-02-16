# 非同期テスト

> 非同期コードのテストには固有の課題がある。タイマーのモック、非同期関数のテスト、フレイキーテストの回避など、実践的な手法を解説する。

## この章で学ぶこと

- [ ] 非同期テストの基本パターンを理解する
- [ ] タイマーとI/Oのモック手法を把握する
- [ ] フレイキーテストの原因と対策を学ぶ
- [ ] テストフレームワーク間の差異を理解する
- [ ] E2Eテストにおける非同期待機戦略を習得する
- [ ] プロパティベーステストで非同期コードを検証する手法を学ぶ

---

## 1. 非同期テストの基本

### 1.1 async/await パターン（Jest / Vitest）

```typescript
// Jest / Vitest: 非同期テストの基本パターン

// async/await — 最も推奨されるパターン
test('ユーザーを取得できる', async () => {
  const user = await getUser('user-123');
  expect(user.name).toBe('田中太郎');
});

// Promise を return するパターン（async/await が使えない場合）
test('注文を作成できる', () => {
  return createOrder(orderData).then(order => {
    expect(order.status).toBe('pending');
  });
});

// エラーのテスト — rejects マッチャー
test('存在しないユーザーでエラー', async () => {
  await expect(getUser('invalid')).rejects.toThrow('User not found');
});

// エラーの型を検証
test('認証エラーの詳細を検証', async () => {
  await expect(authenticate('wrong-token')).rejects.toMatchObject({
    code: 'AUTH_INVALID_TOKEN',
    statusCode: 401,
  });
});

// タイムアウト設定（テスト単位）
test('遅いテスト', async () => {
  const result = await slowOperation();
  expect(result).toBeDefined();
}, 10000); // 10秒タイムアウト
```

### 1.2 コールバックパターン（レガシーコード対応）

```typescript
// done コールバック — レガシーな非同期テスト
// 注意: async 関数と done を混ぜてはいけない

test('コールバックベースの非同期テスト', (done) => {
  fetchDataWithCallback('user-123', (error, data) => {
    try {
      expect(error).toBeNull();
      expect(data.name).toBe('田中太郎');
      done();
    } catch (e) {
      done(e); // エラーを done に渡す
    }
  });
});

// コールバックを Promise にラップする方がよい
function fetchDataPromise(id: string): Promise<User> {
  return new Promise((resolve, reject) => {
    fetchDataWithCallback(id, (error, data) => {
      if (error) reject(error);
      else resolve(data);
    });
  });
}

test('ラップされた非同期テスト', async () => {
  const data = await fetchDataPromise('user-123');
  expect(data.name).toBe('田中太郎');
});
```

### 1.3 並行テストと直列テスト

```typescript
// Jest はデフォルトでファイル内のテストを直列実行、ファイル間は並行
// describe.concurrent でテストを並行実行

describe.concurrent('並行実行テスト', () => {
  test('テスト1', async () => {
    const result = await fetchUser('user-1');
    expect(result).toBeDefined();
  });

  test('テスト2', async () => {
    const result = await fetchUser('user-2');
    expect(result).toBeDefined();
  });

  test('テスト3', async () => {
    const result = await fetchUser('user-3');
    expect(result).toBeDefined();
  });
});

// Vitest では test.concurrent が使える
// it.concurrent('並行テスト', async () => { ... });
```

### 1.4 Vitest 固有の機能

```typescript
// Vitest: vi オブジェクトを使用
import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest';

test('Vitest の非同期テスト', async () => {
  const mockFn = vi.fn().mockResolvedValue({ id: 1, name: 'テスト' });
  const result = await mockFn();
  expect(result.name).toBe('テスト');
});

// Vitest: スナップショットテスト（非同期）
test('APIレスポンスのスナップショット', async () => {
  const response = await fetchUserProfile('user-123');
  expect(response).toMatchSnapshot();
});

// Vitest: インラインスナップショット
test('エラーメッセージのインラインスナップショット', async () => {
  await expect(fetchUser('invalid')).rejects.toThrowErrorMatchingInlineSnapshot(
    `"User not found: invalid"`
  );
});

// Vitest: テストのタイムアウトをグローバル設定
// vitest.config.ts
// export default defineConfig({
//   test: { testTimeout: 10000 }
// });
```

---

## 2. タイマーのモック

### 2.1 Jest フェイクタイマー

```typescript
// Jest: フェイクタイマーの基本
describe('debounce', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('300ms後に実行される', () => {
    const fn = jest.fn();
    const debounced = debounce(fn, 300);

    debounced();
    expect(fn).not.toHaveBeenCalled();

    jest.advanceTimersByTime(200);
    expect(fn).not.toHaveBeenCalled();

    jest.advanceTimersByTime(100);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('リトライの指数バックオフ', async () => {
    const mockFn = jest.fn()
      .mockRejectedValueOnce(new Error('fail'))
      .mockRejectedValueOnce(new Error('fail'))
      .mockResolvedValue('success');

    const promise = retryWithBackoff(mockFn, { maxRetries: 3 });

    // 1回目のリトライ待ち（1000ms）
    jest.advanceTimersByTime(1000);
    await Promise.resolve(); // マイクロタスクを処理

    // 2回目のリトライ待ち（2000ms）
    jest.advanceTimersByTime(2000);
    await Promise.resolve();

    const result = await promise;
    expect(result).toBe('success');
    expect(mockFn).toHaveBeenCalledTimes(3);
  });
});
```

### 2.2 高度なタイマーモック

```typescript
// setInterval のテスト
describe('PollingService', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('5秒ごとにポーリングする', () => {
    const fetchStatus = jest.fn().mockResolvedValue({ status: 'running' });
    const poller = new PollingService(fetchStatus, 5000);
    poller.start();

    // 初回呼び出し
    expect(fetchStatus).toHaveBeenCalledTimes(1);

    // 5秒後
    jest.advanceTimersByTime(5000);
    expect(fetchStatus).toHaveBeenCalledTimes(2);

    // さらに5秒後
    jest.advanceTimersByTime(5000);
    expect(fetchStatus).toHaveBeenCalledTimes(3);

    poller.stop();
  });

  test('ポーリング停止後は呼ばれない', () => {
    const fetchStatus = jest.fn().mockResolvedValue({ status: 'done' });
    const poller = new PollingService(fetchStatus, 5000);
    poller.start();
    poller.stop();

    jest.advanceTimersByTime(15000);
    expect(fetchStatus).toHaveBeenCalledTimes(1); // 初回のみ
  });
});

// setTimeout + Promise の組み合わせ
describe('delayedRetry', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('タイマーとPromiseの正しいインターリーブ', async () => {
    const operation = jest.fn()
      .mockRejectedValueOnce(new Error('transient'))
      .mockResolvedValueOnce('ok');

    // retryWithDelay は内部で setTimeout を使う
    const resultPromise = retryWithDelay(operation, {
      retries: 3,
      delay: 1000,
    });

    // マイクロタスクを処理（最初の呼び出しの reject を処理）
    await jest.advanceTimersByTimeAsync(1000);

    const result = await resultPromise;
    expect(result).toBe('ok');
    expect(operation).toHaveBeenCalledTimes(2);
  });
});

// requestAnimationFrame のモック
describe('アニメーション', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  test('requestAnimationFrame が正しく実行される', () => {
    const callback = jest.fn();
    requestAnimationFrame(callback);

    jest.advanceTimersByTime(16); // 約 60fps の1フレーム
    expect(callback).toHaveBeenCalled();
  });
});
```

### 2.3 advanceTimersByTimeAsync の活用

```typescript
// Jest 29.5+ / Vitest: advanceTimersByTimeAsync
// Promise と タイマーを正しくインターリーブする

describe('非同期タイマーの高度なテスト', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('advanceTimersByTimeAsync で Promise チェーンを処理', async () => {
    const log: string[] = [];

    async function workflow() {
      log.push('start');
      await delay(100);      // setTimeout(resolve, 100)
      log.push('after-100ms');
      await delay(200);      // setTimeout(resolve, 200)
      log.push('after-300ms');
      return 'done';
    }

    const promise = workflow();

    // advanceTimersByTimeAsync は Promise のマイクロタスクも処理する
    await jest.advanceTimersByTimeAsync(100);
    expect(log).toEqual(['start', 'after-100ms']);

    await jest.advanceTimersByTimeAsync(200);
    expect(log).toEqual(['start', 'after-100ms', 'after-300ms']);

    const result = await promise;
    expect(result).toBe('done');
  });

  test('runAllTimersAsync で全タイマーを一括処理', async () => {
    const fn1 = jest.fn();
    const fn2 = jest.fn();

    setTimeout(fn1, 1000);
    setTimeout(fn2, 5000);

    await jest.runAllTimersAsync();

    expect(fn1).toHaveBeenCalled();
    expect(fn2).toHaveBeenCalled();
  });

  // 注意: runAllTimersAsync は無限ループの setInterval には使えない
  test('runOnlyPendingTimersAsync でペンディングのみ処理', async () => {
    const fn = jest.fn();
    setInterval(fn, 1000);

    // 現時点でペンディングのタイマーのみ実行（新しく作られるものは実行しない）
    await jest.runOnlyPendingTimersAsync();
    expect(fn).toHaveBeenCalledTimes(1);

    await jest.runOnlyPendingTimersAsync();
    expect(fn).toHaveBeenCalledTimes(2);
  });
});
```

### 2.4 Vitest のフェイクタイマー

```typescript
import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest';

describe('Vitest フェイクタイマー', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test('debounce のテスト', () => {
    const fn = vi.fn();
    const debounced = debounce(fn, 300);

    debounced();
    vi.advanceTimersByTime(300);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('setSystemTime で日時を固定', () => {
    vi.setSystemTime(new Date('2025-06-15T10:00:00Z'));

    const now = new Date();
    expect(now.getFullYear()).toBe(2025);
    expect(now.getMonth()).toBe(5); // 0-indexed
    expect(now.getDate()).toBe(15);
  });

  test('特定のタイマーAPIのみモック', () => {
    vi.useFakeTimers({
      toFake: ['setTimeout', 'Date'], // setInterval は本物を使う
    });

    const fn = vi.fn();
    setTimeout(fn, 1000);
    vi.advanceTimersByTime(1000);
    expect(fn).toHaveBeenCalled();
  });
});
```

---

## 3. APIモック

### 3.1 MSW（Mock Service Worker）v2

```typescript
// msw v2: HTTP ハンドラーベースのモック
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

// ハンドラーの定義
const handlers = [
  // GET リクエスト
  http.get('/api/users/:id', ({ params }) => {
    const { id } = params;
    if (id === 'not-found') {
      return HttpResponse.json(
        { error: 'Not found' },
        { status: 404 },
      );
    }
    return HttpResponse.json({
      id,
      name: '田中太郎',
      email: 'tanaka@example.com',
    });
  }),

  // POST リクエスト
  http.post('/api/orders', async ({ request }) => {
    const body = await request.json() as Record<string, unknown>;
    return HttpResponse.json(
      { id: 'order-1', ...body, status: 'created' },
      { status: 201 },
    );
  }),

  // PATCH リクエスト
  http.patch('/api/users/:id', async ({ params, request }) => {
    const { id } = params;
    const updates = await request.json() as Record<string, unknown>;
    return HttpResponse.json({ id, ...updates, updatedAt: new Date().toISOString() });
  }),

  // DELETE リクエスト
  http.delete('/api/users/:id', ({ params }) => {
    return new HttpResponse(null, { status: 204 });
  }),
];

// サーバーのセットアップ
const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// テスト
test('ユーザーAPIのテスト', async () => {
  const user = await fetchUser('user-123');
  expect(user.name).toBe('田中太郎');
});

test('404エラーのテスト', async () => {
  await expect(fetchUser('not-found')).rejects.toThrow('User not found');
});

test('注文作成のテスト', async () => {
  const order = await createOrder({ productId: 'prod-1', quantity: 2 });
  expect(order.status).toBe('created');
  expect(order.productId).toBe('prod-1');
});
```

### 3.2 テスト固有のハンドラーオーバーライド

```typescript
// 特定テストでハンドラーを上書き
test('サーバーエラーのハンドリング', async () => {
  server.use(
    http.get('/api/users/:id', () => {
      return HttpResponse.json(
        { error: 'Internal Server Error' },
        { status: 500 },
      );
    }),
  );

  await expect(fetchUser('user-123')).rejects.toThrow('Server error');
});

test('ネットワークエラーのシミュレーション', async () => {
  server.use(
    http.get('/api/users/:id', () => {
      return HttpResponse.error(); // ネットワークエラー
    }),
  );

  await expect(fetchUser('user-123')).rejects.toThrow('Network error');
});

test('遅延レスポンスのシミュレーション', async () => {
  server.use(
    http.get('/api/users/:id', async () => {
      await delay(5000); // 5秒遅延
      return HttpResponse.json({ id: 'user-123', name: '田中太郎' });
    }),
  );

  // タイムアウトのテスト
  await expect(
    fetchUserWithTimeout('user-123', { timeout: 1000 }),
  ).rejects.toThrow('Request timeout');
});

// レスポンスヘッダーのテスト
test('Rate Limit ヘッダーの処理', async () => {
  server.use(
    http.get('/api/users/:id', () => {
      return HttpResponse.json(
        { error: 'Too Many Requests' },
        {
          status: 429,
          headers: {
            'Retry-After': '30',
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': '1700000000',
          },
        },
      );
    }),
  );

  const error = await fetchUser('user-123').catch(e => e);
  expect(error.retryAfter).toBe(30);
});
```

### 3.3 GraphQL モック

```typescript
import { graphql, HttpResponse } from 'msw';

const graphqlHandlers = [
  // Query のモック
  graphql.query('GetUser', ({ variables }) => {
    const { id } = variables;
    return HttpResponse.json({
      data: {
        user: {
          id,
          name: '田中太郎',
          email: 'tanaka@example.com',
          posts: [
            { id: 'post-1', title: '最初の投稿' },
            { id: 'post-2', title: '二番目の投稿' },
          ],
        },
      },
    });
  }),

  // Mutation のモック
  graphql.mutation('CreatePost', ({ variables }) => {
    return HttpResponse.json({
      data: {
        createPost: {
          id: 'post-new',
          title: variables.title,
          createdAt: new Date().toISOString(),
        },
      },
    });
  }),

  // エラーレスポンス
  graphql.query('GetPrivateData', () => {
    return HttpResponse.json({
      errors: [
        {
          message: 'Not authorized',
          extensions: { code: 'UNAUTHORIZED' },
        },
      ],
    });
  }),
];

const server = setupServer(...graphqlHandlers);

test('GraphQL クエリのテスト', async () => {
  const { data } = await graphqlClient.query({
    query: GET_USER,
    variables: { id: 'user-123' },
  });
  expect(data.user.name).toBe('田中太郎');
  expect(data.user.posts).toHaveLength(2);
});
```

### 3.4 fetch / axios のモック（msw を使わない場合）

```typescript
// jest.spyOn で fetch をモック
describe('fetch モック', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('fetch の直接モック', async () => {
    const mockResponse = {
      ok: true,
      status: 200,
      json: async () => ({ id: 'user-123', name: '田中太郎' }),
      headers: new Headers({ 'content-type': 'application/json' }),
    };

    jest.spyOn(globalThis, 'fetch').mockResolvedValue(mockResponse as Response);

    const user = await fetchUser('user-123');
    expect(user.name).toBe('田中太郎');
    expect(fetch).toHaveBeenCalledWith(
      '/api/users/user-123',
      expect.objectContaining({ method: 'GET' }),
    );
  });
});

// axios のモック
import axios from 'axios';
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

test('axios モックのテスト', async () => {
  mockedAxios.get.mockResolvedValue({
    data: { id: 'user-123', name: '田中太郎' },
    status: 200,
  });

  const user = await fetchUserWithAxios('user-123');
  expect(user.name).toBe('田中太郎');
  expect(mockedAxios.get).toHaveBeenCalledWith('/api/users/user-123');
});

// axios インターセプターのテスト
test('リトライインターセプターのテスト', async () => {
  let callCount = 0;
  mockedAxios.get.mockImplementation(async () => {
    callCount++;
    if (callCount < 3) {
      throw { response: { status: 503 }, isAxiosError: true };
    }
    return { data: { status: 'ok' }, status: 200 };
  });

  const result = await apiClientWithRetry.get('/api/health');
  expect(result.data.status).toBe('ok');
  expect(callCount).toBe(3);
});
```

---

## 4. フレイキーテストの回避

### 4.1 フレイキーテストの原因と対策

```
フレイキーテスト（不安定なテスト）の原因:
  1. タイミング依存（setTimeout, setInterval）
  2. 実行順序の仮定（並行テスト）
  3. 外部サービス依存（実APIを呼ぶ）
  4. 共有状態（テスト間でデータが残る）
  5. 非決定的な値（Math.random, Date.now）
  6. ネットワークの不安定性（DNS解決、タイムアウト）
  7. ファイルシステムの競合（一時ファイル、ロック）
  8. テスト間の暗黙的な依存（実行順序に依存するテスト）

対策:
  → タイマー → フェイクタイマー
  → 外部API → モック（msw）
  → 共有状態 → beforeEach でリセット
  → ランダム → シード付きランダム or 固定値
  → Date.now → jest.setSystemTime()
  → ネットワーク → msw / nock でモック
  → ファイル → 一時ディレクトリの確実なクリーンアップ
  → 順序依存 → 各テストの独立性を保証
```

### 4.2 非決定的な値のモック

```typescript
// 日付のモック
describe('日付依存のテスト', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2025-01-15T10:00:00Z'));
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('請求書の支払期限が30日後', () => {
    const invoice = createInvoice();
    expect(invoice.dueDate).toEqual(new Date('2025-02-14T10:00:00Z'));
  });

  test('深夜0時をまたぐ処理', () => {
    jest.setSystemTime(new Date('2025-01-15T23:59:59Z'));
    const report1 = createDailyReport();

    jest.setSystemTime(new Date('2025-01-16T00:00:01Z'));
    const report2 = createDailyReport();

    expect(report1.date).not.toBe(report2.date);
  });

  test('タイムゾーン依存の処理', () => {
    // UTC+9（JST）の場合
    jest.setSystemTime(new Date('2025-01-15T15:00:00Z')); // JST 2025-01-16 00:00
    const jstDate = formatDateJST(new Date());
    expect(jstDate).toBe('2025-01-16');
  });
});

// Math.random のモック
describe('ランダム値のテスト', () => {
  test('固定シードでの乱数', () => {
    // シード付き疑似乱数生成器
    const rng = seedrandom('test-seed-123');
    const values = Array.from({ length: 5 }, () => rng());

    // 同じシードなら常に同じ結果
    const rng2 = seedrandom('test-seed-123');
    const values2 = Array.from({ length: 5 }, () => rng2());

    expect(values).toEqual(values2);
  });

  test('Math.random のスパイ', () => {
    const mockRandom = jest.spyOn(Math, 'random');
    mockRandom.mockReturnValue(0.5);

    const result = generateRandomId();
    expect(result).toBe('expected-id-for-0.5');

    mockRandom.mockRestore();
  });
});

// UUID のモック
describe('UUID のテスト', () => {
  test('crypto.randomUUID のモック', () => {
    const mockUUID = jest.spyOn(crypto, 'randomUUID');
    mockUUID.mockReturnValue('550e8400-e29b-41d4-a716-446655440000');

    const order = createOrder({ productId: 'prod-1' });
    expect(order.id).toBe('550e8400-e29b-41d4-a716-446655440000');

    mockUUID.mockRestore();
  });
});
```

### 4.3 テスト間の分離

```typescript
// 共有状態の適切なリセット
describe('データベース操作', () => {
  let testDb: TestDatabase;

  beforeAll(async () => {
    // テストスイート全体で1回: DB接続
    testDb = await TestDatabase.connect();
  });

  beforeEach(async () => {
    // 各テスト前: データをクリーン
    await testDb.truncateAll();
    await testDb.seed(defaultTestData);
  });

  afterAll(async () => {
    // テストスイート終了: DB切断
    await testDb.disconnect();
  });

  test('ユーザー作成', async () => {
    const user = await userService.create({ name: 'テスト' });
    expect(user.id).toBeDefined();
  });

  test('ユーザー数のカウント', async () => {
    // 前のテストの影響を受けない
    const count = await userService.count();
    expect(count).toBe(defaultTestData.users.length);
  });
});

// シングルトンのリセット
describe('キャッシュサービス', () => {
  beforeEach(() => {
    // シングルトンの内部状態をリセット
    CacheService.getInstance().clear();
  });

  test('キャッシュミス', async () => {
    const result = await CacheService.getInstance().get('key-1');
    expect(result).toBeNull();
  });

  test('キャッシュヒット', async () => {
    await CacheService.getInstance().set('key-1', 'value-1');
    const result = await CacheService.getInstance().get('key-1');
    expect(result).toBe('value-1');
  });
});

// 環境変数のリセット
describe('環境変数依存のテスト', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    // 環境変数のコピーを作成
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    // 元に戻す
    process.env = originalEnv;
  });

  test('本番環境の設定', () => {
    process.env.NODE_ENV = 'production';
    process.env.API_URL = 'https://api.example.com';

    const config = loadConfig();
    expect(config.apiUrl).toBe('https://api.example.com');
    expect(config.debug).toBe(false);
  });

  test('開発環境の設定', () => {
    process.env.NODE_ENV = 'development';
    process.env.API_URL = 'http://localhost:3000';

    const config = loadConfig();
    expect(config.apiUrl).toBe('http://localhost:3000');
    expect(config.debug).toBe(true);
  });
});
```

### 4.4 waitFor パターン（非同期アサーション）

```typescript
// Testing Library: waitFor
import { render, screen, waitFor } from '@testing-library/react';

test('データ読み込み後に表示される', async () => {
  render(<UserProfile userId="user-123" />);

  // ローディング表示
  expect(screen.getByText('読み込み中...')).toBeInTheDocument();

  // データ取得完了を待つ
  await waitFor(() => {
    expect(screen.getByText('田中太郎')).toBeInTheDocument();
  });

  // ローディングが消えている
  expect(screen.queryByText('読み込み中...')).not.toBeInTheDocument();
});

// waitFor のオプション設定
test('カスタムタイムアウトとインターバル', async () => {
  render(<SlowComponent />);

  await waitFor(
    () => {
      expect(screen.getByTestId('result')).toHaveTextContent('完了');
    },
    {
      timeout: 5000,   // 最大待機時間
      interval: 100,   // ポーリング間隔
    },
  );
});

// findBy クエリ（waitFor + getBy のショートカット）
test('findBy で非同期要素を取得', async () => {
  render(<UserList />);

  // findByText は内部的に waitFor を使う
  const userElement = await screen.findByText('田中太郎');
  expect(userElement).toBeInTheDocument();
});

// waitForElementToBeRemoved
test('要素の消失を待つ', async () => {
  render(<DeletableItem id="item-1" />);

  const deleteButton = screen.getByRole('button', { name: '削除' });
  fireEvent.click(deleteButton);

  // 要素が消えるのを待つ
  await waitForElementToBeRemoved(() =>
    screen.queryByTestId('item-1'),
  );
});
```

---

## 5. Python での非同期テスト

### 5.1 pytest-asyncio

```python
# pytest-asyncio: Python の非同期テスト
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# pytest.mark.asyncio で非同期テスト関数を宣言
@pytest.mark.asyncio
async def test_fetch_user():
    """非同期関数の基本テスト"""
    user = await fetch_user("user-123")
    assert user["name"] == "田中太郎"

@pytest.mark.asyncio
async def test_fetch_user_not_found():
    """非同期例外のテスト"""
    with pytest.raises(UserNotFoundError, match="User not found"):
        await fetch_user("invalid-id")

@pytest.mark.asyncio
async def test_concurrent_requests():
    """並行リクエストのテスト"""
    users = await asyncio.gather(
        fetch_user("user-1"),
        fetch_user("user-2"),
        fetch_user("user-3"),
    )
    assert len(users) == 3
    assert all(u["id"] is not None for u in users)


# pytest-asyncio のモード設定
# pyproject.toml:
# [tool.pytest.ini_options]
# asyncio_mode = "auto"  # @pytest.mark.asyncio を省略可能にする


# フィクスチャ
@pytest.fixture
async def db_connection():
    """非同期フィクスチャ"""
    conn = await create_db_connection("test_db")
    yield conn
    await conn.close()

@pytest.fixture
async def test_user(db_connection):
    """テスト用ユーザーを作成するフィクスチャ"""
    user = await db_connection.execute(
        "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *",
        "テストユーザー", "test@example.com"
    )
    yield user
    await db_connection.execute("DELETE FROM users WHERE id = $1", user["id"])

@pytest.mark.asyncio
async def test_update_user(db_connection, test_user):
    """フィクスチャを使ったテスト"""
    updated = await update_user(db_connection, test_user["id"], name="更新済み")
    assert updated["name"] == "更新済み"
```

### 5.2 AsyncMock

```python
from unittest.mock import AsyncMock, patch, MagicMock

# AsyncMock の基本
@pytest.mark.asyncio
async def test_with_async_mock():
    """AsyncMock でモック"""
    mock_repo = AsyncMock()
    mock_repo.find_by_id.return_value = {"id": "user-123", "name": "田中太郎"}

    service = UserService(repository=mock_repo)
    user = await service.get_user("user-123")

    assert user["name"] == "田中太郎"
    mock_repo.find_by_id.assert_called_once_with("user-123")

# AsyncMock で例外を発生
@pytest.mark.asyncio
async def test_async_mock_exception():
    mock_repo = AsyncMock()
    mock_repo.find_by_id.side_effect = DatabaseError("Connection failed")

    service = UserService(repository=mock_repo)
    with pytest.raises(ServiceError, match="Failed to fetch user"):
        await service.get_user("user-123")

# patch デコレーターとの組み合わせ
@pytest.mark.asyncio
@patch("myapp.services.user_service.send_email", new_callable=AsyncMock)
@patch("myapp.services.user_service.UserRepository", new_callable=AsyncMock)
async def test_create_user_sends_email(mock_repo, mock_send_email):
    mock_repo.return_value.save.return_value = {
        "id": "user-new",
        "name": "新規ユーザー",
        "email": "new@example.com",
    }

    service = UserService(repository=mock_repo.return_value)
    user = await service.create_user(name="新規ユーザー", email="new@example.com")

    mock_send_email.assert_called_once_with(
        to="new@example.com",
        subject="ようこそ",
    )

# side_effect で呼び出し回数に応じた動作
@pytest.mark.asyncio
async def test_retry_behavior():
    mock_fn = AsyncMock(side_effect=[
        ConnectionError("Timeout"),
        ConnectionError("Timeout"),
        {"status": "ok"},
    ])

    result = await retry_with_backoff(mock_fn, max_retries=3)
    assert result == {"status": "ok"}
    assert mock_fn.call_count == 3
```

### 5.3 aiohttp テスト

```python
import aiohttp
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from aiohttp import web
import pytest

# aiohttp のテストサーバー
@pytest.fixture
async def app():
    """テスト用 aiohttp アプリケーション"""
    app = web.Application()
    app.router.add_get("/api/users/{id}", handle_get_user)
    app.router.add_post("/api/users", handle_create_user)
    return app

@pytest.fixture
async def client(app, aiohttp_client):
    """テストクライアント"""
    return await aiohttp_client(app)

@pytest.mark.asyncio
async def test_get_user(client):
    resp = await client.get("/api/users/user-123")
    assert resp.status == 200
    data = await resp.json()
    assert data["name"] == "田中太郎"

@pytest.mark.asyncio
async def test_create_user(client):
    resp = await client.post("/api/users", json={
        "name": "新規ユーザー",
        "email": "new@example.com",
    })
    assert resp.status == 201
    data = await resp.json()
    assert data["id"] is not None


# aioresponses で外部 API をモック
from aioresponses import aioresponses

@pytest.mark.asyncio
async def test_external_api_call():
    with aioresponses() as mocked:
        mocked.get(
            "https://api.external.com/data",
            payload={"key": "value"},
            status=200,
        )

        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.external.com/data") as resp:
                data = await resp.json()
                assert data["key"] == "value"

@pytest.mark.asyncio
async def test_external_api_timeout():
    with aioresponses() as mocked:
        mocked.get(
            "https://api.external.com/data",
            exception=asyncio.TimeoutError(),
        )

        with pytest.raises(asyncio.TimeoutError):
            async with aiohttp.ClientSession() as session:
                await session.get("https://api.external.com/data")
```

### 5.4 FastAPI テスト

```python
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from unittest.mock import AsyncMock, patch

app = FastAPI()

# FastAPI のテスト（httpx の AsyncClient を使用）
@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_get_users(async_client):
    response = await async_client.get("/api/users")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

@pytest.mark.asyncio
async def test_create_user(async_client):
    response = await async_client.post("/api/users", json={
        "name": "テストユーザー",
        "email": "test@example.com",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "テストユーザー"

# 依存性のオーバーライド
@pytest.mark.asyncio
async def test_with_mock_dependency(async_client):
    mock_user_repo = AsyncMock()
    mock_user_repo.find_all.return_value = [
        {"id": "1", "name": "ユーザー1"},
        {"id": "2", "name": "ユーザー2"},
    ]

    app.dependency_overrides[get_user_repository] = lambda: mock_user_repo

    response = await async_client.get("/api/users")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    # クリーンアップ
    app.dependency_overrides.clear()
```

---

## 6. Go での非同期テスト

### 6.1 goroutine のテスト

```go
package async_test

import (
    "context"
    "sync"
    "testing"
    "time"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

// goroutine の基本テスト
func TestConcurrentProcessor(t *testing.T) {
    processor := NewConcurrentProcessor(5) // 並行数5

    items := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
    results, err := processor.Process(context.Background(), items)

    require.NoError(t, err)
    assert.Len(t, results, len(items))
}

// コンテキストキャンセルのテスト
func TestCancellation(t *testing.T) {
    ctx, cancel := context.WithCancel(context.Background())

    var started sync.WaitGroup
    started.Add(1)

    errCh := make(chan error, 1)
    go func() {
        started.Done()
        errCh <- longRunningOperation(ctx)
    }()

    started.Wait()
    cancel() // キャンセル

    err := <-errCh
    assert.ErrorIs(t, err, context.Canceled)
}

// タイムアウトのテスト
func TestTimeout(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
    defer cancel()

    err := slowOperation(ctx) // 内部で1秒かかる処理
    assert.ErrorIs(t, err, context.DeadlineExceeded)
}

// データ競合の検出（go test -race）
func TestNoDataRace(t *testing.T) {
    counter := NewAtomicCounter()

    var wg sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter.Increment()
        }()
    }

    wg.Wait()
    assert.Equal(t, int64(1000), counter.Value())
}
```

### 6.2 チャネルのテスト

```go
// チャネルベースのテスト
func TestWorkerPool(t *testing.T) {
    jobs := make(chan Job, 10)
    results := make(chan Result, 10)

    // ワーカーを起動
    pool := NewWorkerPool(3, jobs, results)
    pool.Start()

    // ジョブを送信
    for i := 0; i < 5; i++ {
        jobs <- Job{ID: i, Data: fmt.Sprintf("task-%d", i)}
    }
    close(jobs)

    // 結果を収集
    var collected []Result
    for r := range results {
        collected = append(collected, r)
    }

    assert.Len(t, collected, 5)
    for _, r := range collected {
        assert.NoError(t, r.Error)
    }
}

// select でタイムアウト付きチャネル待機
func TestChannelWithTimeout(t *testing.T) {
    ch := make(chan string, 1)

    // 非同期で値を送信
    go func() {
        time.Sleep(50 * time.Millisecond)
        ch <- "result"
    }()

    select {
    case result := <-ch:
        assert.Equal(t, "result", result)
    case <-time.After(1 * time.Second):
        t.Fatal("タイムアウト: 1秒以内に結果が来なかった")
    }
}

// testify の Eventually（ポーリングベースのアサーション）
func TestEventualConsistency(t *testing.T) {
    service := NewEventualService()
    service.TriggerUpdate("key-1", "new-value")

    // 最終的に値が更新されることを確認
    assert.Eventually(t, func() bool {
        val, err := service.Get("key-1")
        return err == nil && val == "new-value"
    }, 5*time.Second, 100*time.Millisecond)
}

// testify の Never（特定条件が発生しないことを確認）
func TestNeverHappens(t *testing.T) {
    service := NewStableService()

    assert.Never(t, func() bool {
        return service.HasError()
    }, 1*time.Second, 100*time.Millisecond)
}
```

### 6.3 HTTP テスト

```go
import (
    "net/http"
    "net/http/httptest"
    "testing"
)

// httptest.Server でモックサーバー
func TestExternalAPIClient(t *testing.T) {
    // テスト用サーバー
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        switch r.URL.Path {
        case "/api/users/user-123":
            w.Header().Set("Content-Type", "application/json")
            w.WriteHeader(http.StatusOK)
            w.Write([]byte(`{"id": "user-123", "name": "田中太郎"}`))
        case "/api/users/not-found":
            w.WriteHeader(http.StatusNotFound)
            w.Write([]byte(`{"error": "Not found"}`))
        default:
            w.WriteHeader(http.StatusNotFound)
        }
    }))
    defer server.Close()

    // テスト用サーバーのURLを使ってクライアントを作成
    client := NewAPIClient(server.URL)

    t.Run("正常系", func(t *testing.T) {
        user, err := client.GetUser(context.Background(), "user-123")
        require.NoError(t, err)
        assert.Equal(t, "田中太郎", user.Name)
    })

    t.Run("404エラー", func(t *testing.T) {
        _, err := client.GetUser(context.Background(), "not-found")
        assert.ErrorIs(t, err, ErrUserNotFound)
    })
}

// httptest.NewRecorder でハンドラーのテスト
func TestUserHandler(t *testing.T) {
    handler := NewUserHandler(mockUserService)

    req := httptest.NewRequest("GET", "/api/users/user-123", nil)
    rec := httptest.NewRecorder()

    handler.ServeHTTP(rec, req)

    assert.Equal(t, http.StatusOK, rec.Code)

    var user User
    err := json.NewDecoder(rec.Body).Decode(&user)
    require.NoError(t, err)
    assert.Equal(t, "田中太郎", user.Name)
}
```

---

## 7. E2Eテストの非同期待機戦略

### 7.1 Playwright（TypeScript）

```typescript
import { test, expect } from '@playwright/test';

test.describe('ユーザー管理画面', () => {
  test('ユーザー一覧が表示される', async ({ page }) => {
    await page.goto('/users');

    // ネットワークリクエストの完了を待つ
    await page.waitForResponse(
      response => response.url().includes('/api/users') && response.status() === 200,
    );

    // 要素が表示されるのを待つ
    await expect(page.getByText('田中太郎')).toBeVisible();
    await expect(page.getByText('鈴木花子')).toBeVisible();
  });

  test('ユーザー作成フロー', async ({ page }) => {
    await page.goto('/users/new');

    // フォーム入力
    await page.getByLabel('名前').fill('新規ユーザー');
    await page.getByLabel('メールアドレス').fill('new@example.com');

    // API レスポンスを待ちながらフォーム送信
    const responsePromise = page.waitForResponse('/api/users');
    await page.getByRole('button', { name: '作成' }).click();
    const response = await responsePromise;

    expect(response.status()).toBe(201);

    // リダイレクトを待つ
    await page.waitForURL('/users/*');

    // 成功メッセージの表示を確認
    await expect(page.getByText('ユーザーが作成されました')).toBeVisible();
  });

  test('エラー表示のテスト', async ({ page }) => {
    // API をモック（Playwright のルーティング）
    await page.route('/api/users', route =>
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal Server Error' }),
      }),
    );

    await page.goto('/users');

    // エラーメッセージの表示を待つ
    await expect(page.getByText('データの取得に失敗しました')).toBeVisible();

    // リトライボタンをクリック
    await page.route('/api/users', route =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([{ id: '1', name: '田中太郎' }]),
      }),
    );

    await page.getByRole('button', { name: 'リトライ' }).click();
    await expect(page.getByText('田中太郎')).toBeVisible();
  });
});

// ネットワーク状態のテスト
test('オフライン時の動作', async ({ page, context }) => {
  await page.goto('/dashboard');
  await expect(page.getByTestId('status')).toHaveText('オンライン');

  // オフラインにする
  await context.setOffline(true);
  await expect(page.getByTestId('status')).toHaveText('オフライン');

  // オンラインに戻す
  await context.setOffline(false);
  await expect(page.getByTestId('status')).toHaveText('オンライン');
});
```

### 7.2 Cypress

```typescript
// Cypress: 非同期テストの待機戦略

describe('ユーザー管理', () => {
  beforeEach(() => {
    // APIモックのセットアップ
    cy.intercept('GET', '/api/users', {
      fixture: 'users.json',
    }).as('getUsers');

    cy.intercept('POST', '/api/users', {
      statusCode: 201,
      body: { id: 'user-new', name: '新規ユーザー' },
    }).as('createUser');
  });

  it('ユーザー一覧を表示する', () => {
    cy.visit('/users');

    // API レスポンスを待つ
    cy.wait('@getUsers');

    // 要素の表示を確認
    cy.findByText('田中太郎').should('be.visible');
    cy.findByText('鈴木花子').should('be.visible');
  });

  it('ユーザーを作成する', () => {
    cy.visit('/users/new');

    cy.findByLabelText('名前').type('新規ユーザー');
    cy.findByLabelText('メールアドレス').type('new@example.com');
    cy.findByRole('button', { name: '作成' }).click();

    // API レスポンスを待ち、リクエストの内容も検証
    cy.wait('@createUser').then((interception) => {
      expect(interception.request.body).to.deep.equal({
        name: '新規ユーザー',
        email: 'new@example.com',
      });
    });

    // リダイレクトの確認
    cy.url().should('match', /\/users\/.+/);
    cy.findByText('ユーザーが作成されました').should('be.visible');
  });

  it('ネットワークエラーのハンドリング', () => {
    cy.intercept('GET', '/api/users', {
      forceNetworkError: true,
    }).as('getUsersFailed');

    cy.visit('/users');
    cy.wait('@getUsersFailed');

    cy.findByText('ネットワークエラーが発生しました').should('be.visible');
  });

  it('遅延レスポンスのテスト', () => {
    cy.intercept('GET', '/api/users', {
      fixture: 'users.json',
      delay: 3000, // 3秒遅延
    }).as('getUsers');

    cy.visit('/users');

    // ローディング表示の確認
    cy.findByTestId('loading-spinner').should('be.visible');

    // データ取得完了後
    cy.wait('@getUsers');
    cy.findByTestId('loading-spinner').should('not.exist');
    cy.findByText('田中太郎').should('be.visible');
  });
});

// Cypress: カスタムコマンドで非同期操作をラップ
Cypress.Commands.add('waitForApiAndAssert', (alias: string, assertion: Function) => {
  cy.wait(alias).then((interception) => {
    assertion(interception);
  });
});
```

### 7.3 WebSocket E2Eテスト

```typescript
// Playwright: WebSocket テスト
test('WebSocket リアルタイム通信', async ({ page }) => {
  // WebSocket メッセージを監視
  const messages: string[] = [];
  page.on('websocket', ws => {
    ws.on('framereceived', frame => {
      messages.push(frame.payload as string);
    });
  });

  await page.goto('/chat');

  // メッセージ送信
  await page.getByPlaceholder('メッセージを入力').fill('こんにちは');
  await page.getByRole('button', { name: '送信' }).click();

  // 送信されたメッセージの表示を確認
  await expect(page.getByText('こんにちは')).toBeVisible();

  // WebSocket を通じてメッセージが送信されたことを確認
  expect(messages.some(m => m.includes('こんにちは'))).toBe(true);
});

// WebSocket のモック
test('WebSocket モック', async ({ page }) => {
  // WebSocket ルートのモック
  await page.routeWebSocket('/ws', ws => {
    ws.onMessage(message => {
      // エコーバック
      const data = JSON.parse(message as string);
      ws.send(JSON.stringify({
        type: 'echo',
        data: data.message,
        timestamp: Date.now(),
      }));
    });
  });

  await page.goto('/chat');
  await page.getByPlaceholder('メッセージを入力').fill('テスト');
  await page.getByRole('button', { name: '送信' }).click();

  await expect(page.getByText('テスト')).toBeVisible();
});
```

---

## 8. 非同期テストのデザインパターン

### 8.1 テストヘルパーの設計

```typescript
// 再利用可能な非同期テストヘルパー

/**
 * 非同期操作が指定時間内に完了することを検証
 */
async function expectToCompleteWithin<T>(
  operation: () => Promise<T>,
  timeoutMs: number,
  message?: string,
): Promise<T> {
  const start = Date.now();
  const result = await Promise.race([
    operation(),
    new Promise<never>((_, reject) =>
      setTimeout(
        () => reject(new Error(message || `Operation timed out after ${timeoutMs}ms`)),
        timeoutMs,
      ),
    ),
  ]);
  const elapsed = Date.now() - start;
  console.log(`Operation completed in ${elapsed}ms`);
  return result;
}

/**
 * 非同期操作が最終的に成功することを検証（ポーリング）
 */
async function waitUntil(
  predicate: () => Promise<boolean> | boolean,
  options: { timeout?: number; interval?: number; message?: string } = {},
): Promise<void> {
  const { timeout = 5000, interval = 100, message = 'Condition not met' } = options;
  const start = Date.now();

  while (Date.now() - start < timeout) {
    if (await predicate()) return;
    await new Promise(resolve => setTimeout(resolve, interval));
  }

  throw new Error(`${message} (waited ${timeout}ms)`);
}

/**
 * 非同期操作を指定回数リトライしてテスト
 */
async function retryTest(
  testFn: () => Promise<void>,
  maxRetries: number = 3,
): Promise<void> {
  let lastError: Error | undefined;

  for (let i = 0; i < maxRetries; i++) {
    try {
      await testFn();
      return;
    } catch (error) {
      lastError = error as Error;
      console.warn(`Test attempt ${i + 1} failed: ${lastError.message}`);
    }
  }

  throw lastError;
}

// 使用例
test('APIが1秒以内に応答する', async () => {
  const result = await expectToCompleteWithin(
    () => fetchUser('user-123'),
    1000,
    'API response too slow',
  );
  expect(result.name).toBe('田中太郎');
});

test('キャッシュが更新される', async () => {
  cache.invalidate('user-123');
  triggerCacheRefresh();

  await waitUntil(
    async () => {
      const cached = await cache.get('user-123');
      return cached !== null;
    },
    { timeout: 3000, message: 'Cache was not refreshed' },
  );
});
```

### 8.2 テストダブルパターン

```typescript
// 非同期テストダブルの分類と実装

// 1. Stub: 固定値を返す
class StubUserRepository {
  async findById(id: string): Promise<User | null> {
    const users: Record<string, User> = {
      'user-1': { id: 'user-1', name: '田中太郎', email: 'tanaka@example.com' },
      'user-2': { id: 'user-2', name: '鈴木花子', email: 'suzuki@example.com' },
    };
    return users[id] ?? null;
  }

  async save(user: User): Promise<User> {
    return { ...user, id: user.id || 'generated-id' };
  }
}

// 2. Spy: 呼び出しを記録する
class SpyEmailService implements EmailService {
  readonly sentEmails: Array<{ to: string; subject: string; body: string }> = [];

  async send(to: string, subject: string, body: string): Promise<void> {
    this.sentEmails.push({ to, subject, body });
  }

  getCallCount(): number {
    return this.sentEmails.length;
  }

  wasCalledWith(to: string): boolean {
    return this.sentEmails.some(email => email.to === to);
  }
}

// 3. Fake: 簡略化した実装
class FakeCache implements CacheService {
  private store = new Map<string, { value: string; expiresAt: number }>();

  async get(key: string): Promise<string | null> {
    const entry = this.store.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.store.delete(key);
      return null;
    }
    return entry.value;
  }

  async set(key: string, value: string, ttlMs: number): Promise<void> {
    this.store.set(key, { value, expiresAt: Date.now() + ttlMs });
  }

  async delete(key: string): Promise<void> {
    this.store.delete(key);
  }

  // テスト用ヘルパー
  clear(): void {
    this.store.clear();
  }

  size(): number {
    return this.store.size;
  }
}

// 4. Mock: 期待値を設定して検証
class MockPaymentGateway implements PaymentGateway {
  private expectations: Array<{
    method: string;
    args: any[];
    result: any;
    called: boolean;
  }> = [];

  expectCharge(amount: number, currency: string): MockPaymentGateway {
    this.expectations.push({
      method: 'charge',
      args: [amount, currency],
      result: { transactionId: 'txn-mock', status: 'success' },
      called: false,
    });
    return this;
  }

  async charge(amount: number, currency: string): Promise<PaymentResult> {
    const expectation = this.expectations.find(
      e => e.method === 'charge' && !e.called,
    );
    if (!expectation) {
      throw new Error(`Unexpected call: charge(${amount}, ${currency})`);
    }
    expect([amount, currency]).toEqual(expectation.args);
    expectation.called = true;
    return expectation.result;
  }

  verify(): void {
    const uncalled = this.expectations.filter(e => !e.called);
    if (uncalled.length > 0) {
      throw new Error(
        `Expected calls not made: ${uncalled.map(e => e.method).join(', ')}`,
      );
    }
  }
}

// テストでの使用
test('注文処理でメール送信と決済が行われる', async () => {
  const emailSpy = new SpyEmailService();
  const paymentMock = new MockPaymentGateway();
  paymentMock.expectCharge(1000, 'JPY');

  const orderService = new OrderService({
    email: emailSpy,
    payment: paymentMock,
    repository: new StubUserRepository(),
    cache: new FakeCache(),
  });

  await orderService.placeOrder({
    userId: 'user-1',
    productId: 'prod-1',
    amount: 1000,
  });

  // Spy の検証
  expect(emailSpy.getCallCount()).toBe(1);
  expect(emailSpy.wasCalledWith('tanaka@example.com')).toBe(true);

  // Mock の検証
  paymentMock.verify();
});
```

### 8.3 イベント駆動テスト

```typescript
// EventEmitter ベースの非同期テスト

import { EventEmitter } from 'events';

// once でイベントを待つ
test('イベントが発火される', async () => {
  const emitter = new EventEmitter();

  const eventPromise = new Promise<{ type: string; data: any }>((resolve) => {
    emitter.once('user:created', (data) => resolve({ type: 'user:created', data }));
  });

  // 非同期でイベントを発火
  setTimeout(() => {
    emitter.emit('user:created', { id: 'user-1', name: '田中太郎' });
  }, 100);

  const event = await eventPromise;
  expect(event.type).toBe('user:created');
  expect(event.data.name).toBe('田中太郎');
});

// Node.js の events.once を使う
import { once } from 'events';

test('events.once で待つ', async () => {
  const emitter = new EventEmitter();

  setTimeout(() => {
    emitter.emit('data', { value: 42 });
  }, 50);

  const [data] = await once(emitter, 'data');
  expect(data.value).toBe(42);
});

// 複数イベントの順序テスト
test('イベントの順序を検証', async () => {
  const events: string[] = [];
  const processor = new OrderProcessor();

  processor.on('started', () => events.push('started'));
  processor.on('validated', () => events.push('validated'));
  processor.on('charged', () => events.push('charged'));
  processor.on('completed', () => events.push('completed'));

  await processor.process({ productId: 'prod-1', amount: 1000 });

  expect(events).toEqual(['started', 'validated', 'charged', 'completed']);
});

// エラーイベントのテスト
test('エラーイベントが発火される', async () => {
  const processor = new OrderProcessor();

  const errorPromise = new Promise<Error>((resolve) => {
    processor.on('error', resolve);
  });

  // 不正な注文でエラーを発生させる
  processor.process({ productId: '', amount: -100 }).catch(() => {});

  const error = await errorPromise;
  expect(error.message).toContain('Invalid order');
});
```

---

## 9. プロパティベーステスト

### 9.1 fast-check で非同期プロパティをテスト

```typescript
import fc from 'fast-check';

// 非同期プロパティベーステスト
test('エンコード→デコードで元に戻る', async () => {
  await fc.assert(
    fc.asyncProperty(fc.string(), async (input) => {
      const encoded = await encode(input);
      const decoded = await decode(encoded);
      expect(decoded).toBe(input);
    }),
  );
});

// 並行処理のプロパティテスト
test('並行アクセスでもカウンターは正確', async () => {
  await fc.assert(
    fc.asyncProperty(
      fc.integer({ min: 1, max: 100 }),
      fc.integer({ min: 1, max: 50 }),
      async (incrementCount, concurrency) => {
        const counter = new AtomicCounter();

        const tasks = Array.from({ length: incrementCount }, () =>
          counter.increment(),
        );

        // 並行数を制限して実行
        await promisePool(tasks.map(t => () => t), concurrency);

        expect(counter.value).toBe(incrementCount);
      },
    ),
    { numRuns: 50 },
  );
});

// リトライのプロパティテスト
test('リトライは最終的に成功するか最大回数で停止する', async () => {
  await fc.assert(
    fc.asyncProperty(
      fc.integer({ min: 0, max: 10 }), // 失敗回数
      fc.integer({ min: 1, max: 5 }),   // 最大リトライ
      async (failCount, maxRetries) => {
        let callCount = 0;
        const fn = async () => {
          callCount++;
          if (callCount <= failCount) {
            throw new Error('transient');
          }
          return 'success';
        };

        try {
          const result = await retryWithBackoff(fn, {
            maxRetries,
            initialDelay: 1,
          });

          // 成功した場合: 失敗回数 < 最大リトライ
          expect(result).toBe('success');
          expect(failCount).toBeLessThan(maxRetries);
        } catch {
          // 失敗した場合: 失敗回数 >= 最大リトライ
          expect(failCount).toBeGreaterThanOrEqual(maxRetries);
        }

        // 呼び出し回数の検証
        expect(callCount).toBeLessThanOrEqual(maxRetries + 1);
      },
    ),
    { numRuns: 100 },
  );
});

// データベース操作のプロパティテスト
test('CRUD操作の一貫性', async () => {
  await fc.assert(
    fc.asyncProperty(
      fc.record({
        name: fc.string({ minLength: 1, maxLength: 100 }),
        email: fc.emailAddress(),
        age: fc.integer({ min: 0, max: 150 }),
      }),
      async (userData) => {
        // Create
        const created = await userRepo.create(userData);
        expect(created.id).toBeDefined();

        // Read
        const fetched = await userRepo.findById(created.id);
        expect(fetched).toMatchObject(userData);

        // Update
        const updated = await userRepo.update(created.id, { name: 'Updated' });
        expect(updated.name).toBe('Updated');

        // Delete
        await userRepo.delete(created.id);
        const deleted = await userRepo.findById(created.id);
        expect(deleted).toBeNull();
      },
    ),
    { numRuns: 20 },
  );
});
```

### 9.2 Hypothesis（Python）

```python
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
import asyncio

# hypothesis の非同期テスト
@pytest.mark.asyncio
@given(st.text(min_size=1, max_size=100))
@settings(max_examples=50)
async def test_encode_decode_roundtrip(text):
    """エンコード→デコードの往復テスト"""
    encoded = await encode(text)
    decoded = await decode(encoded)
    assert decoded == text

@pytest.mark.asyncio
@given(
    items=st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=50),
    batch_size=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=30)
async def test_batch_processing_preserves_all_items(items, batch_size):
    """バッチ処理で全アイテムが処理される"""
    processed = []

    async def processor(item):
        processed.append(item)
        return item * 2

    results = await process_in_batches(items, processor, batch_size=batch_size)

    assert len(results) == len(items)
    assert sorted(processed) == sorted(items)
    assert all(r == i * 2 for r, i in zip(sorted(results), sorted(items)))
```

---

## 10. テストのパフォーマンスとCI最適化

### 10.1 テスト実行の高速化

```typescript
// テストの並行実行設定
// jest.config.ts
export default {
  // ワーカー数の最適化
  maxWorkers: '50%', // CPU の50%を使用
  // maxWorkers: 4, // 固定値も指定可能

  // テストの並行実行
  // ファイル間は並行、ファイル内は直列（デフォルト）

  // 遅いテストのタイムアウト
  testTimeout: 10000,

  // グローバルセットアップ（テストスイート全体で1回）
  globalSetup: './test/global-setup.ts',
  globalTeardown: './test/global-teardown.ts',

  // プロジェクト設定で異なるテスト環境を分離
  projects: [
    {
      displayName: 'unit',
      testMatch: ['<rootDir>/src/**/*.test.ts'],
      testTimeout: 5000,
    },
    {
      displayName: 'integration',
      testMatch: ['<rootDir>/test/integration/**/*.test.ts'],
      testTimeout: 30000,
    },
  ],
};
```

### 10.2 テストのグループ化と選択実行

```typescript
// タグベースのテスト選択

// テストにタグを付ける
test('遅いテスト #slow', async () => {
  // ...
});

test('高速なテスト #fast', async () => {
  // ...
});

// 実行時にフィルタ
// jest --testNamePattern="#fast"
// jest --testNamePattern="^(?!.*#slow)"  // #slow を除外

// Vitest のタグ機能
// vitest run --reporter=verbose --bail 1

// describe.skip / test.skip で一時的に無効化
describe.skip('WIP: 新機能のテスト', () => {
  test('未完成のテスト', async () => {
    // ...
  });
});

// describe.only / test.only でフォーカス（CIでは禁止）
// eslint-plugin-jest: no-focused-tests ルールで防止
```

### 10.3 CI環境での非同期テスト

```yaml
# GitHub Actions: 非同期テストの設定
name: Test

on: [push, pull_request]

jobs:
  unit-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [18, 20, 22]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      - run: npm ci
      - run: npm run test:unit -- --ci --coverage
        timeout-minutes: 10
        env:
          # CI環境でのタイムアウトを長めに設定
          JEST_TIMEOUT: 15000

  integration-test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'
      - run: npm ci
      - run: npm run test:integration -- --ci
        timeout-minutes: 15
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/test
          REDIS_URL: redis://localhost:6379

  e2e-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'
      - run: npm ci
      - run: npx playwright install --with-deps
      - run: npm run test:e2e -- --ci
        timeout-minutes: 20
      - uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: playwright-report
          path: playwright-report/
          retention-days: 7
```

### 10.4 テストのリトライ戦略（CI向け）

```typescript
// Playwright: テストリトライ設定
// playwright.config.ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  retries: process.env.CI ? 2 : 0, // CIでは2回リトライ

  use: {
    // 失敗時のスクリーンショットとトレース
    screenshot: 'only-on-failure',
    trace: 'on-first-retry',
    video: 'on-first-retry',
  },

  // プロジェクト別設定
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
      retries: 2,
    },
    {
      name: 'firefox',
      use: { browserName: 'firefox' },
      retries: 3, // Firefox はさらにリトライ
    },
  ],
});

// Jest: テストリトライ（jest-circus）
// jest.config.ts
export default {
  // jest-circus のリトライ機能（実験的）
  // テスト単位のリトライ
};

// カスタムリトライラッパー
function testWithRetry(
  name: string,
  fn: () => Promise<void>,
  retries: number = 3,
): void {
  test(name, async () => {
    let lastError: Error | undefined;
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        await fn();
        return; // 成功
      } catch (error) {
        lastError = error as Error;
        if (attempt < retries) {
          console.warn(`Attempt ${attempt} failed, retrying...`);
        }
      }
    }
    throw lastError;
  });
}

// 使用例
testWithRetry('不安定な外部API連携テスト', async () => {
  const result = await externalApiCall();
  expect(result.status).toBe('ok');
}, 3);
```

---

## 11. テストカバレッジと品質指標

### 11.1 非同期コードのカバレッジ

```typescript
// 非同期コードのカバレッジで注意すべきポイント

// 問題: catch ブランチがテストされていない
async function fetchUserSafe(id: string): Promise<User | null> {
  try {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`); // ← このブランチ
    }
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch user:', error); // ← このブランチ
    return null;
  }
}

// 正常系のテスト
test('正常にユーザーを取得', async () => {
  const user = await fetchUserSafe('user-123');
  expect(user).not.toBeNull();
  expect(user!.name).toBe('田中太郎');
});

// 異常系のテスト（カバレッジ向上）
test('HTTP エラーで null を返す', async () => {
  server.use(
    http.get('/api/users/:id', () => {
      return HttpResponse.json({ error: 'Not found' }, { status: 404 });
    }),
  );

  const user = await fetchUserSafe('invalid');
  expect(user).toBeNull();
});

test('ネットワークエラーで null を返す', async () => {
  server.use(
    http.get('/api/users/:id', () => {
      return HttpResponse.error();
    }),
  );

  const user = await fetchUserSafe('user-123');
  expect(user).toBeNull();
});

// Promise.allSettled のカバレッジ
async function fetchMultipleUsers(ids: string[]): Promise<{
  users: User[];
  errors: string[];
}> {
  const results = await Promise.allSettled(
    ids.map(id => fetchUser(id)),
  );

  const users: User[] = [];
  const errors: string[] = [];

  for (const result of results) {
    if (result.status === 'fulfilled') {
      users.push(result.value);
    } else {
      errors.push(result.reason.message);
    }
  }

  return { users, errors };
}

test('一部失敗時の結果を検証', async () => {
  server.use(
    http.get('/api/users/bad', () => {
      return HttpResponse.json({ error: 'Not found' }, { status: 404 });
    }),
  );

  const { users, errors } = await fetchMultipleUsers(['user-1', 'bad', 'user-2']);
  expect(users).toHaveLength(2);
  expect(errors).toHaveLength(1);
});
```

### 11.2 ミューテーションテスト

```typescript
// Stryker: ミューテーションテストで非同期コードの品質を検証
// stryker.conf.json
{
  "mutate": ["src/**/*.ts", "!src/**/*.test.ts"],
  "testRunner": "jest",
  "reporters": ["html", "clear-text", "progress"],
  "coverageAnalysis": "perTest",
  "timeoutMS": 60000,

  // 非同期コード用の追加ミュータント
  "mutator": {
    "excludedMutations": [
      // 不要なミュータントを除外
    ]
  }
}

// ミュータントの例:
// 元のコード:
// if (retries < maxRetries) { ... }
// ミュータント:
// if (retries <= maxRetries) { ... }  ← 境界値ミューテーション
// if (retries > maxRetries) { ... }   ← 条件反転
// if (true) { ... }                   ← 条件削除

// これらのミュータントを殺すテスト
test('最大リトライ回数で正確に停止', async () => {
  const fn = jest.fn().mockRejectedValue(new Error('fail'));

  await expect(
    retryWithBackoff(fn, { maxRetries: 3, initialDelay: 1 }),
  ).rejects.toThrow('fail');

  // 初回 + 3回リトライ = 4回
  expect(fn).toHaveBeenCalledTimes(4);
});

test('最大リトライ回数 - 1 ではまだリトライする', async () => {
  let callCount = 0;
  const fn = jest.fn().mockImplementation(async () => {
    callCount++;
    if (callCount <= 2) throw new Error('fail'); // 2回失敗
    return 'success';
  });

  const result = await retryWithBackoff(fn, { maxRetries: 3, initialDelay: 1 });
  expect(result).toBe('success');
  expect(callCount).toBe(3); // 初回 + 2回リトライ
});
```

---

## まとめ

| 手法 | 目的 | ツール |
|------|------|--------|
| async/await | 非同期テスト | Jest, Vitest |
| フェイクタイマー | タイマーのモック | jest.useFakeTimers, vi.useFakeTimers |
| advanceTimersByTimeAsync | Promise + タイマー | Jest 29.5+, Vitest |
| msw v2 | HTTP APIモック | Mock Service Worker |
| GraphQL モック | GraphQL APIモック | msw graphql ハンドラー |
| フェイク日時 | 日付のモック | jest.setSystemTime, vi.setSystemTime |
| pytest-asyncio | Python 非同期テスト | pytest + asyncio |
| AsyncMock | Python モック | unittest.mock |
| aioresponses | Python HTTP モック | aiohttp テスト用 |
| httptest | Go HTTP テスト | net/http/httptest |
| testify | Go アサーション | github.com/stretchr/testify |
| Playwright | E2E テスト | @playwright/test |
| Cypress | E2E テスト | cypress |
| fast-check | プロパティベーステスト | fast-check |
| Hypothesis | Python プロパティテスト | hypothesis |
| waitFor | 非同期DOM待機 | @testing-library |
| Stryker | ミューテーションテスト | @stryker-mutator |

### テストフレームワーク比較

| 機能 | Jest | Vitest | pytest |
|------|------|--------|--------|
| フェイクタイマー | jest.useFakeTimers() | vi.useFakeTimers() | freezegun |
| 非同期タイマー | advanceTimersByTimeAsync | advanceTimersByTimeAsync | - |
| HTTP モック | msw, nock | msw, nock | aioresponses, httpx-mock |
| スナップショット | toMatchSnapshot | toMatchSnapshot | syrupy |
| 並行実行 | --maxWorkers | --pool threads | pytest-xdist |
| カバレッジ | --coverage (istanbul/v8) | --coverage (v8/istanbul) | pytest-cov |
| ウォッチモード | --watch | --watch (HMR対応) | pytest-watch |

### 非同期テストのベストプラクティス

```
1. テストの独立性を保証する
   - 各テストが他のテストに依存しない
   - beforeEach で状態をリセット
   - 共有リソースは適切にクリーンアップ

2. 決定論的なテストを書く
   - 日時、乱数、UUIDはモックする
   - 外部APIはmsw等でモックする
   - タイマーはフェイクタイマーを使う

3. 適切な待機戦略を選ぶ
   - 固定 sleep() は避ける（フレイキーの原因）
   - waitFor / waitUntil でポーリング待機
   - イベントベースの待機を優先

4. テストの粒度を意識する
   - 単体テスト: 個々の非同期関数
   - 統合テスト: 複数コンポーネントの連携
   - E2E テスト: ユーザーシナリオ全体

5. CIでの安定性を確保する
   - リトライ戦略を設定
   - タイムアウトを適切に設定
   - 失敗時のアーティファクト収集
```

---

## 次に読むべきガイド
→ [[03-real-world-patterns.md]] — 実践パターン集

---

## 参考文献
1. Jest Documentation. "Timer Mocks." https://jestjs.io/docs/timer-mocks
2. MSW Documentation. "Getting Started." https://mswjs.io/docs/getting-started
3. Playwright Documentation. "Test Assertions." https://playwright.dev/docs/test-assertions
4. pytest-asyncio Documentation. https://pytest-asyncio.readthedocs.io/
5. Cypress Documentation. "Network Requests." https://docs.cypress.io/guides/guides/network-requests
6. fast-check Documentation. "Async Properties." https://fast-check.dev/docs/core-blocks/arbitraries/
7. Testing Library Documentation. "Async Methods." https://testing-library.com/docs/dom-testing-library/api-async
8. Stryker Mutator Documentation. https://stryker-mutator.io/docs/
9. Go Testing Documentation. "httptest." https://pkg.go.dev/net/http/httptest
10. Vitest Documentation. "Mocking." https://vitest.dev/guide/mocking.html
