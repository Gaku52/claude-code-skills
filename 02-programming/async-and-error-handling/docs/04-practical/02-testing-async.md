# 非同期テスト

> 非同期コードのテストには固有の課題がある。タイマーのモック、非同期関数のテスト、フレイキーテストの回避など、実践的な手法を解説する。

## この章で学ぶこと

- [ ] 非同期テストの基本パターンを理解する
- [ ] タイマーとI/Oのモック手法を把握する
- [ ] フレイキーテストの原因と対策を学ぶ

---

## 1. 非同期テストの基本

```typescript
// Jest: 非同期テスト

// async/await
test('ユーザーを取得できる', async () => {
  const user = await getUser('user-123');
  expect(user.name).toBe('田中太郎');
});

// Promise
test('注文を作成できる', () => {
  return createOrder(orderData).then(order => {
    expect(order.status).toBe('pending');
  });
});

// エラーのテスト
test('存在しないユーザーでエラー', async () => {
  await expect(getUser('invalid')).rejects.toThrow('User not found');
});

// タイムアウト設定
test('遅いテスト', async () => {
  const result = await slowOperation();
  expect(result).toBeDefined();
}, 10000); // 10秒タイムアウト
```

---

## 2. タイマーのモック

```typescript
// Jest: フェイクタイマー
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

    // 1回目のリトライ待ち
    jest.advanceTimersByTime(1000);
    await Promise.resolve(); // マイクロタスクを処理

    // 2回目のリトライ待ち
    jest.advanceTimersByTime(2000);
    await Promise.resolve();

    const result = await promise;
    expect(result).toBe('success');
    expect(mockFn).toHaveBeenCalledTimes(3);
  });
});
```

---

## 3. APIモック

```typescript
// msw（Mock Service Worker）で API をモック
import { rest } from 'msw';
import { setupServer } from 'msw/node';

const server = setupServer(
  rest.get('/api/users/:id', (req, res, ctx) => {
    const { id } = req.params;
    if (id === 'not-found') {
      return res(ctx.status(404), ctx.json({ error: 'Not found' }));
    }
    return res(ctx.json({ id, name: '田中太郎' }));
  }),

  rest.post('/api/orders', async (req, res, ctx) => {
    const body = await req.json();
    return res(
      ctx.status(201),
      ctx.json({ id: 'order-1', ...body }),
    );
  }),
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

test('ユーザーAPIのテスト', async () => {
  const user = await fetchUser('user-123');
  expect(user.name).toBe('田中太郎');
});

test('404エラーのテスト', async () => {
  await expect(fetchUser('not-found')).rejects.toThrow('User not found');
});
```

---

## 4. フレイキーテストの回避

```
フレイキーテスト（不安定なテスト）の原因:
  1. タイミング依存（setTimeout, setInterval）
  2. 実行順序の仮定（並行テスト）
  3. 外部サービス依存（実APIを呼ぶ）
  4. 共有状態（テスト間でデータが残る）
  5. 非決定的な値（Math.random, Date.now）

対策:
  → タイマー → フェイクタイマー
  → 外部API → モック（msw）
  → 共有状態 → beforeEach でリセット
  → ランダム → シード付きランダム or 固定値
  → Date.now → jest.setSystemTime()
```

```typescript
// 非決定的な値のモック
test('日付依存のテスト', () => {
  jest.setSystemTime(new Date('2025-01-15T10:00:00Z'));

  const invoice = createInvoice();
  expect(invoice.dueDate).toEqual(new Date('2025-02-14T10:00:00Z'));
});
```

---

## まとめ

| 手法 | 目的 | ツール |
|------|------|--------|
| async/await | 非同期テスト | Jest, Vitest |
| フェイクタイマー | タイマーのモック | jest.useFakeTimers |
| msw | APIモック | Mock Service Worker |
| フェイク日時 | 日付のモック | jest.setSystemTime |

---

## 次に読むべきガイド
→ [[03-real-world-patterns.md]] — 実践パターン集

---

## 参考文献
1. Jest Documentation. "Timer Mocks."
2. MSW Documentation. mswjs.io.
