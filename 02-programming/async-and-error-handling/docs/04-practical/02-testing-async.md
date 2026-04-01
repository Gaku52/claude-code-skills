# Asynchronous Testing

> Testing asynchronous code presents unique challenges. This guide covers practical techniques including timer mocking, testing async functions, and avoiding flaky tests.

## What You Will Learn in This Chapter

- [ ] Understand fundamental patterns for asynchronous testing
- [ ] Master timer and I/O mocking techniques
- [ ] Learn the causes and countermeasures for flaky tests
- [ ] Understand the differences between testing frameworks
- [ ] Acquire async waiting strategies for E2E testing
- [ ] Learn how to verify asynchronous code with property-based testing


## Prerequisites

Before reading this guide, the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content in [Logging and Monitoring](./01-logging-and-monitoring.md)

---

## 1. Asynchronous Testing Basics

### 1.1 async/await Pattern (Jest / Vitest)

```typescript
// Jest / Vitest: Fundamental patterns for asynchronous testing

// async/await -- the most recommended pattern
test('can retrieve a user', async () => {
  const user = await getUser('user-123');
  expect(user.name).toBe('Taro Tanaka');
});

// Pattern that returns a Promise (when async/await is not available)
test('can create an order', () => {
  return createOrder(orderData).then(order => {
    expect(order.status).toBe('pending');
  });
});

// Testing errors -- rejects matcher
test('throws error for non-existent user', async () => {
  await expect(getUser('invalid')).rejects.toThrow('User not found');
});

// Validating error types
test('validate authentication error details', async () => {
  await expect(authenticate('wrong-token')).rejects.toMatchObject({
    code: 'AUTH_INVALID_TOKEN',
    statusCode: 401,
  });
});

// Timeout setting (per test)
test('slow test', async () => {
  const result = await slowOperation();
  expect(result).toBeDefined();
}, 10000); // 10-second timeout
```

### 1.2 Callback Pattern (Legacy Code Support)

```typescript
// done callback -- legacy asynchronous testing
// Note: do not mix async functions with done

test('callback-based asynchronous test', (done) => {
  fetchDataWithCallback('user-123', (error, data) => {
    try {
      expect(error).toBeNull();
      expect(data.name).toBe('Taro Tanaka');
      done();
    } catch (e) {
      done(e); // Pass the error to done
    }
  });
});

// It is better to wrap callbacks in a Promise
function fetchDataPromise(id: string): Promise<User> {
  return new Promise((resolve, reject) => {
    fetchDataWithCallback(id, (error, data) => {
      if (error) reject(error);
      else resolve(data);
    });
  });
}

test('wrapped asynchronous test', async () => {
  const data = await fetchDataPromise('user-123');
  expect(data.name).toBe('Taro Tanaka');
});
```

### 1.3 Concurrent and Sequential Tests

```typescript
// Jest runs tests sequentially within a file and concurrently across files by default
// Use describe.concurrent to run tests concurrently

describe.concurrent('concurrent execution tests', () => {
  test('test 1', async () => {
    const result = await fetchUser('user-1');
    expect(result).toBeDefined();
  });

  test('test 2', async () => {
    const result = await fetchUser('user-2');
    expect(result).toBeDefined();
  });

  test('test 3', async () => {
    const result = await fetchUser('user-3');
    expect(result).toBeDefined();
  });
});

// In Vitest, you can use test.concurrent
// it.concurrent('concurrent test', async () => { ... });
```

### 1.4 Vitest-Specific Features

```typescript
// Vitest: uses the vi object
import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest';

test('Vitest asynchronous test', async () => {
  const mockFn = vi.fn().mockResolvedValue({ id: 1, name: 'Test' });
  const result = await mockFn();
  expect(result.name).toBe('Test');
});

// Vitest: snapshot testing (async)
test('API response snapshot', async () => {
  const response = await fetchUserProfile('user-123');
  expect(response).toMatchSnapshot();
});

// Vitest: inline snapshot
test('error message inline snapshot', async () => {
  await expect(fetchUser('invalid')).rejects.toThrowErrorMatchingInlineSnapshot(
    `"User not found: invalid"`
  );
});

// Vitest: global test timeout setting
// vitest.config.ts
// export default defineConfig({
//   test: { testTimeout: 10000 }
// });
```

---

## 2. Timer Mocking

### 2.1 Jest Fake Timers

```typescript
// Jest: fake timer basics
describe('debounce', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('executes after 300ms', () => {
    const fn = jest.fn();
    const debounced = debounce(fn, 300);

    debounced();
    expect(fn).not.toHaveBeenCalled();

    jest.advanceTimersByTime(200);
    expect(fn).not.toHaveBeenCalled();

    jest.advanceTimersByTime(100);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('exponential backoff for retry', async () => {
    const mockFn = jest.fn()
      .mockRejectedValueOnce(new Error('fail'))
      .mockRejectedValueOnce(new Error('fail'))
      .mockResolvedValue('success');

    const promise = retryWithBackoff(mockFn, { maxRetries: 3 });

    // Wait for first retry (1000ms)
    jest.advanceTimersByTime(1000);
    await Promise.resolve(); // Process microtasks

    // Wait for second retry (2000ms)
    jest.advanceTimersByTime(2000);
    await Promise.resolve();

    const result = await promise;
    expect(result).toBe('success');
    expect(mockFn).toHaveBeenCalledTimes(3);
  });
});
```

### 2.2 Advanced Timer Mocking

```typescript
// Testing setInterval
describe('PollingService', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('polls every 5 seconds', () => {
    const fetchStatus = jest.fn().mockResolvedValue({ status: 'running' });
    const poller = new PollingService(fetchStatus, 5000);
    poller.start();

    // Initial call
    expect(fetchStatus).toHaveBeenCalledTimes(1);

    // After 5 seconds
    jest.advanceTimersByTime(5000);
    expect(fetchStatus).toHaveBeenCalledTimes(2);

    // After another 5 seconds
    jest.advanceTimersByTime(5000);
    expect(fetchStatus).toHaveBeenCalledTimes(3);

    poller.stop();
  });

  test('not called after polling stops', () => {
    const fetchStatus = jest.fn().mockResolvedValue({ status: 'done' });
    const poller = new PollingService(fetchStatus, 5000);
    poller.start();
    poller.stop();

    jest.advanceTimersByTime(15000);
    expect(fetchStatus).toHaveBeenCalledTimes(1); // Initial call only
  });
});

// Combining setTimeout + Promise
describe('delayedRetry', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('correct interleaving of timers and Promises', async () => {
    const operation = jest.fn()
      .mockRejectedValueOnce(new Error('transient'))
      .mockResolvedValueOnce('ok');

    // retryWithDelay uses setTimeout internally
    const resultPromise = retryWithDelay(operation, {
      retries: 3,
      delay: 1000,
    });

    // Process microtasks (process the first call's rejection)
    await jest.advanceTimersByTimeAsync(1000);

    const result = await resultPromise;
    expect(result).toBe('ok');
    expect(operation).toHaveBeenCalledTimes(2);
  });
});

// Mocking requestAnimationFrame
describe('animation', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  test('requestAnimationFrame executes correctly', () => {
    const callback = jest.fn();
    requestAnimationFrame(callback);

    jest.advanceTimersByTime(16); // Approximately one frame at 60fps
    expect(callback).toHaveBeenCalled();
  });
});
```

### 2.3 Utilizing advanceTimersByTimeAsync

```typescript
// Jest 29.5+ / Vitest: advanceTimersByTimeAsync
// Correctly interleaves Promises and timers

describe('advanced asynchronous timer tests', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('process Promise chains with advanceTimersByTimeAsync', async () => {
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

    // advanceTimersByTimeAsync also processes Promise microtasks
    await jest.advanceTimersByTimeAsync(100);
    expect(log).toEqual(['start', 'after-100ms']);

    await jest.advanceTimersByTimeAsync(200);
    expect(log).toEqual(['start', 'after-100ms', 'after-300ms']);

    const result = await promise;
    expect(result).toBe('done');
  });

  test('process all timers at once with runAllTimersAsync', async () => {
    const fn1 = jest.fn();
    const fn2 = jest.fn();

    setTimeout(fn1, 1000);
    setTimeout(fn2, 5000);

    await jest.runAllTimersAsync();

    expect(fn1).toHaveBeenCalled();
    expect(fn2).toHaveBeenCalled();
  });

  // Note: runAllTimersAsync cannot be used with infinite setInterval loops
  test('process only pending timers with runOnlyPendingTimersAsync', async () => {
    const fn = jest.fn();
    setInterval(fn, 1000);

    // Only execute currently pending timers (do not execute newly created ones)
    await jest.runOnlyPendingTimersAsync();
    expect(fn).toHaveBeenCalledTimes(1);

    await jest.runOnlyPendingTimersAsync();
    expect(fn).toHaveBeenCalledTimes(2);
  });
});
```

### 2.4 Vitest Fake Timers

```typescript
import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest';

describe('Vitest fake timers', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test('debounce test', () => {
    const fn = vi.fn();
    const debounced = debounce(fn, 300);

    debounced();
    vi.advanceTimersByTime(300);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('fix date/time with setSystemTime', () => {
    vi.setSystemTime(new Date('2025-06-15T10:00:00Z'));

    const now = new Date();
    expect(now.getFullYear()).toBe(2025);
    expect(now.getMonth()).toBe(5); // 0-indexed
    expect(now.getDate()).toBe(15);
  });

  test('mock only specific timer APIs', () => {
    vi.useFakeTimers({
      toFake: ['setTimeout', 'Date'], // Use real setInterval
    });

    const fn = vi.fn();
    setTimeout(fn, 1000);
    vi.advanceTimersByTime(1000);
    expect(fn).toHaveBeenCalled();
  });
});
```

---

## 3. API Mocking

### 3.1 MSW (Mock Service Worker) v2

```typescript
// msw v2: HTTP handler-based mocking
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

// Handler definitions
const handlers = [
  // GET request
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
      name: 'Taro Tanaka',
      email: 'tanaka@example.com',
    });
  }),

  // POST request
  http.post('/api/orders', async ({ request }) => {
    const body = await request.json() as Record<string, unknown>;
    return HttpResponse.json(
      { id: 'order-1', ...body, status: 'created' },
      { status: 201 },
    );
  }),

  // PATCH request
  http.patch('/api/users/:id', async ({ params, request }) => {
    const { id } = params;
    const updates = await request.json() as Record<string, unknown>;
    return HttpResponse.json({ id, ...updates, updatedAt: new Date().toISOString() });
  }),

  // DELETE request
  http.delete('/api/users/:id', ({ params }) => {
    return new HttpResponse(null, { status: 204 });
  }),
];

// Server setup
const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// Tests
test('user API test', async () => {
  const user = await fetchUser('user-123');
  expect(user.name).toBe('Taro Tanaka');
});

test('404 error test', async () => {
  await expect(fetchUser('not-found')).rejects.toThrow('User not found');
});

test('order creation test', async () => {
  const order = await createOrder({ productId: 'prod-1', quantity: 2 });
  expect(order.status).toBe('created');
  expect(order.productId).toBe('prod-1');
});
```

### 3.2 Test-Specific Handler Overrides

```typescript
// Override handlers for specific tests
test('server error handling', async () => {
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

test('network error simulation', async () => {
  server.use(
    http.get('/api/users/:id', () => {
      return HttpResponse.error(); // Network error
    }),
  );

  await expect(fetchUser('user-123')).rejects.toThrow('Network error');
});

test('delayed response simulation', async () => {
  server.use(
    http.get('/api/users/:id', async () => {
      await delay(5000); // 5-second delay
      return HttpResponse.json({ id: 'user-123', name: 'Taro Tanaka' });
    }),
  );

  // Timeout test
  await expect(
    fetchUserWithTimeout('user-123', { timeout: 1000 }),
  ).rejects.toThrow('Request timeout');
});

// Testing response headers
test('Rate Limit header processing', async () => {
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

### 3.3 GraphQL Mocking

```typescript
import { graphql, HttpResponse } from 'msw';

const graphqlHandlers = [
  // Query mock
  graphql.query('GetUser', ({ variables }) => {
    const { id } = variables;
    return HttpResponse.json({
      data: {
        user: {
          id,
          name: 'Taro Tanaka',
          email: 'tanaka@example.com',
          posts: [
            { id: 'post-1', title: 'First Post' },
            { id: 'post-2', title: 'Second Post' },
          ],
        },
      },
    });
  }),

  // Mutation mock
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

  // Error response
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

test('GraphQL query test', async () => {
  const { data } = await graphqlClient.query({
    query: GET_USER,
    variables: { id: 'user-123' },
  });
  expect(data.user.name).toBe('Taro Tanaka');
  expect(data.user.posts).toHaveLength(2);
});
```

### 3.4 Mocking fetch / axios (Without msw)

```typescript
// Mocking fetch with jest.spyOn
describe('fetch mock', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('direct fetch mock', async () => {
    const mockResponse = {
      ok: true,
      status: 200,
      json: async () => ({ id: 'user-123', name: 'Taro Tanaka' }),
      headers: new Headers({ 'content-type': 'application/json' }),
    };

    jest.spyOn(globalThis, 'fetch').mockResolvedValue(mockResponse as Response);

    const user = await fetchUser('user-123');
    expect(user.name).toBe('Taro Tanaka');
    expect(fetch).toHaveBeenCalledWith(
      '/api/users/user-123',
      expect.objectContaining({ method: 'GET' }),
    );
  });
});

// Mocking axios
import axios from 'axios';
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

test('axios mock test', async () => {
  mockedAxios.get.mockResolvedValue({
    data: { id: 'user-123', name: 'Taro Tanaka' },
    status: 200,
  });

  const user = await fetchUserWithAxios('user-123');
  expect(user.name).toBe('Taro Tanaka');
  expect(mockedAxios.get).toHaveBeenCalledWith('/api/users/user-123');
});

// Testing axios interceptors
test('retry interceptor test', async () => {
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

## 4. Avoiding Flaky Tests

### 4.1 Causes and Countermeasures for Flaky Tests

```
Causes of flaky tests (unstable tests):
  1. Timing dependency (setTimeout, setInterval)
  2. Assumptions about execution order (concurrent tests)
  3. External service dependency (calling real APIs)
  4. Shared state (data persisting between tests)
  5. Non-deterministic values (Math.random, Date.now)
  6. Network instability (DNS resolution, timeouts)
  7. File system contention (temporary files, locks)
  8. Implicit dependencies between tests (tests dependent on execution order)

Countermeasures:
  -> Timers -> Fake timers
  -> External APIs -> Mocks (msw)
  -> Shared state -> Reset in beforeEach
  -> Random values -> Seeded random or fixed values
  -> Date.now -> jest.setSystemTime()
  -> Network -> Mock with msw / nock
  -> Files -> Ensure reliable cleanup of temporary directories
  -> Order dependency -> Guarantee independence of each test
```

### 4.2 Mocking Non-Deterministic Values

```typescript
// Mocking dates
describe('date-dependent tests', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2025-01-15T10:00:00Z'));
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('invoice due date is 30 days later', () => {
    const invoice = createInvoice();
    expect(invoice.dueDate).toEqual(new Date('2025-02-14T10:00:00Z'));
  });

  test('processing across midnight', () => {
    jest.setSystemTime(new Date('2025-01-15T23:59:59Z'));
    const report1 = createDailyReport();

    jest.setSystemTime(new Date('2025-01-16T00:00:01Z'));
    const report2 = createDailyReport();

    expect(report1.date).not.toBe(report2.date);
  });

  test('timezone-dependent processing', () => {
    // For UTC+9 (JST)
    jest.setSystemTime(new Date('2025-01-15T15:00:00Z')); // JST 2025-01-16 00:00
    const jstDate = formatDateJST(new Date());
    expect(jstDate).toBe('2025-01-16');
  });
});

// Mocking Math.random
describe('random value tests', () => {
  test('random numbers with fixed seed', () => {
    // Seeded pseudo-random number generator
    const rng = seedrandom('test-seed-123');
    const values = Array.from({ length: 5 }, () => rng());

    // Same seed always produces the same results
    const rng2 = seedrandom('test-seed-123');
    const values2 = Array.from({ length: 5 }, () => rng2());

    expect(values).toEqual(values2);
  });

  test('spying on Math.random', () => {
    const mockRandom = jest.spyOn(Math, 'random');
    mockRandom.mockReturnValue(0.5);

    const result = generateRandomId();
    expect(result).toBe('expected-id-for-0.5');

    mockRandom.mockRestore();
  });
});

// Mocking UUIDs
describe('UUID tests', () => {
  test('mocking crypto.randomUUID', () => {
    const mockUUID = jest.spyOn(crypto, 'randomUUID');
    mockUUID.mockReturnValue('550e8400-e29b-41d4-a716-446655440000');

    const order = createOrder({ productId: 'prod-1' });
    expect(order.id).toBe('550e8400-e29b-41d4-a716-446655440000');

    mockUUID.mockRestore();
  });
});
```

### 4.3 Test Isolation

```typescript
// Proper shared state reset
describe('database operations', () => {
  let testDb: TestDatabase;

  beforeAll(async () => {
    // Once for the entire test suite: DB connection
    testDb = await TestDatabase.connect();
  });

  beforeEach(async () => {
    // Before each test: clean data
    await testDb.truncateAll();
    await testDb.seed(defaultTestData);
  });

  afterAll(async () => {
    // End of test suite: DB disconnect
    await testDb.disconnect();
  });

  test('user creation', async () => {
    const user = await userService.create({ name: 'Test' });
    expect(user.id).toBeDefined();
  });

  test('user count', async () => {
    // Not affected by the previous test
    const count = await userService.count();
    expect(count).toBe(defaultTestData.users.length);
  });
});

// Resetting singletons
describe('cache service', () => {
  beforeEach(() => {
    // Reset the singleton's internal state
    CacheService.getInstance().clear();
  });

  test('cache miss', async () => {
    const result = await CacheService.getInstance().get('key-1');
    expect(result).toBeNull();
  });

  test('cache hit', async () => {
    await CacheService.getInstance().set('key-1', 'value-1');
    const result = await CacheService.getInstance().get('key-1');
    expect(result).toBe('value-1');
  });
});

// Resetting environment variables
describe('environment variable dependent tests', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    // Create a copy of environment variables
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    // Restore original
    process.env = originalEnv;
  });

  test('production environment configuration', () => {
    process.env.NODE_ENV = 'production';
    process.env.API_URL = 'https://api.example.com';

    const config = loadConfig();
    expect(config.apiUrl).toBe('https://api.example.com');
    expect(config.debug).toBe(false);
  });

  test('development environment configuration', () => {
    process.env.NODE_ENV = 'development';
    process.env.API_URL = 'http://localhost:3000';

    const config = loadConfig();
    expect(config.apiUrl).toBe('http://localhost:3000');
    expect(config.debug).toBe(true);
  });
});
```

### 4.4 waitFor Pattern (Asynchronous Assertions)

```typescript
// Testing Library: waitFor
import { render, screen, waitFor } from '@testing-library/react';

test('displayed after data loading', async () => {
  render(<UserProfile userId="user-123" />);

  // Loading display
  expect(screen.getByText('Loading...')).toBeInTheDocument();

  // Wait for data fetch to complete
  await waitFor(() => {
    expect(screen.getByText('Taro Tanaka')).toBeInTheDocument();
  });

  // Loading indicator has disappeared
  expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
});

// waitFor option configuration
test('custom timeout and interval', async () => {
  render(<SlowComponent />);

  await waitFor(
    () => {
      expect(screen.getByTestId('result')).toHaveTextContent('Complete');
    },
    {
      timeout: 5000,   // Maximum wait time
      interval: 100,   // Polling interval
    },
  );
});

// findBy queries (shortcut for waitFor + getBy)
test('get async element with findBy', async () => {
  render(<UserList />);

  // findByText internally uses waitFor
  const userElement = await screen.findByText('Taro Tanaka');
  expect(userElement).toBeInTheDocument();
});

// waitForElementToBeRemoved
test('wait for element removal', async () => {
  render(<DeletableItem id="item-1" />);

  const deleteButton = screen.getByRole('button', { name: 'Delete' });
  fireEvent.click(deleteButton);

  // Wait for element to be removed
  await waitForElementToBeRemoved(() =>
    screen.queryByTestId('item-1'),
  );
});
```

---

## 5. Asynchronous Testing in Python

### 5.1 pytest-asyncio

```python
# pytest-asyncio: asynchronous testing in Python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# Declare an async test function with pytest.mark.asyncio
@pytest.mark.asyncio
async def test_fetch_user():
    """Basic test for an async function"""
    user = await fetch_user("user-123")
    assert user["name"] == "Taro Tanaka"

@pytest.mark.asyncio
async def test_fetch_user_not_found():
    """Test for async exceptions"""
    with pytest.raises(UserNotFoundError, match="User not found"):
        await fetch_user("invalid-id")

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test for concurrent requests"""
    users = await asyncio.gather(
        fetch_user("user-1"),
        fetch_user("user-2"),
        fetch_user("user-3"),
    )
    assert len(users) == 3
    assert all(u["id"] is not None for u in users)


# pytest-asyncio mode configuration
# pyproject.toml:
# [tool.pytest.ini_options]
# asyncio_mode = "auto"  # Allows omitting @pytest.mark.asyncio


# Fixtures
@pytest.fixture
async def db_connection():
    """Async fixture"""
    conn = await create_db_connection("test_db")
    yield conn
    await conn.close()

@pytest.fixture
async def test_user(db_connection):
    """Fixture that creates a test user"""
    user = await db_connection.execute(
        "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *",
        "Test User", "test@example.com"
    )
    yield user
    await db_connection.execute("DELETE FROM users WHERE id = $1", user["id"])

@pytest.mark.asyncio
async def test_update_user(db_connection, test_user):
    """Test using fixtures"""
    updated = await update_user(db_connection, test_user["id"], name="Updated")
    assert updated["name"] == "Updated"
```

### 5.2 AsyncMock

```python
from unittest.mock import AsyncMock, patch, MagicMock

# AsyncMock basics
@pytest.mark.asyncio
async def test_with_async_mock():
    """Mocking with AsyncMock"""
    mock_repo = AsyncMock()
    mock_repo.find_by_id.return_value = {"id": "user-123", "name": "Taro Tanaka"}

    service = UserService(repository=mock_repo)
    user = await service.get_user("user-123")

    assert user["name"] == "Taro Tanaka"
    mock_repo.find_by_id.assert_called_once_with("user-123")

# Raising exceptions with AsyncMock
@pytest.mark.asyncio
async def test_async_mock_exception():
    mock_repo = AsyncMock()
    mock_repo.find_by_id.side_effect = DatabaseError("Connection failed")

    service = UserService(repository=mock_repo)
    with pytest.raises(ServiceError, match="Failed to fetch user"):
        await service.get_user("user-123")

# Combining with patch decorator
@pytest.mark.asyncio
@patch("myapp.services.user_service.send_email", new_callable=AsyncMock)
@patch("myapp.services.user_service.UserRepository", new_callable=AsyncMock)
async def test_create_user_sends_email(mock_repo, mock_send_email):
    mock_repo.return_value.save.return_value = {
        "id": "user-new",
        "name": "New User",
        "email": "new@example.com",
    }

    service = UserService(repository=mock_repo.return_value)
    user = await service.create_user(name="New User", email="new@example.com")

    mock_send_email.assert_called_once_with(
        to="new@example.com",
        subject="Welcome",
    )

# Behavior based on call count with side_effect
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

### 5.3 aiohttp Testing

```python
import aiohttp
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from aiohttp import web
import pytest

# aiohttp test server
@pytest.fixture
async def app():
    """Test aiohttp application"""
    app = web.Application()
    app.router.add_get("/api/users/{id}", handle_get_user)
    app.router.add_post("/api/users", handle_create_user)
    return app

@pytest.fixture
async def client(app, aiohttp_client):
    """Test client"""
    return await aiohttp_client(app)

@pytest.mark.asyncio
async def test_get_user(client):
    resp = await client.get("/api/users/user-123")
    assert resp.status == 200
    data = await resp.json()
    assert data["name"] == "Taro Tanaka"

@pytest.mark.asyncio
async def test_create_user(client):
    resp = await client.post("/api/users", json={
        "name": "New User",
        "email": "new@example.com",
    })
    assert resp.status == 201
    data = await resp.json()
    assert data["id"] is not None


# Mocking external APIs with aioresponses
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

### 5.4 FastAPI Testing

```python
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from unittest.mock import AsyncMock, patch

app = FastAPI()

# FastAPI testing (using httpx AsyncClient)
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
        "name": "Test User",
        "email": "test@example.com",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test User"

# Dependency override
@pytest.mark.asyncio
async def test_with_mock_dependency(async_client):
    mock_user_repo = AsyncMock()
    mock_user_repo.find_all.return_value = [
        {"id": "1", "name": "User 1"},
        {"id": "2", "name": "User 2"},
    ]

    app.dependency_overrides[get_user_repository] = lambda: mock_user_repo

    response = await async_client.get("/api/users")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    # Cleanup
    app.dependency_overrides.clear()
```

---

## 6. Asynchronous Testing in Go

### 6.1 Testing Goroutines

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

// Basic goroutine test
func TestConcurrentProcessor(t *testing.T) {
    processor := NewConcurrentProcessor(5) // concurrency of 5

    items := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
    results, err := processor.Process(context.Background(), items)

    require.NoError(t, err)
    assert.Len(t, results, len(items))
}

// Testing context cancellation
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
    cancel() // Cancel

    err := <-errCh
    assert.ErrorIs(t, err, context.Canceled)
}

// Testing timeouts
func TestTimeout(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
    defer cancel()

    err := slowOperation(ctx) // Internally takes 1 second
    assert.ErrorIs(t, err, context.DeadlineExceeded)
}

// Data race detection (go test -race)
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

### 6.2 Testing Channels

```go
// Channel-based testing
func TestWorkerPool(t *testing.T) {
    jobs := make(chan Job, 10)
    results := make(chan Result, 10)

    // Start workers
    pool := NewWorkerPool(3, jobs, results)
    pool.Start()

    // Submit jobs
    for i := 0; i < 5; i++ {
        jobs <- Job{ID: i, Data: fmt.Sprintf("task-%d", i)}
    }
    close(jobs)

    // Collect results
    var collected []Result
    for r := range results {
        collected = append(collected, r)
    }

    assert.Len(t, collected, 5)
    for _, r := range collected {
        assert.NoError(t, r.Error)
    }
}

// Channel waiting with timeout using select
func TestChannelWithTimeout(t *testing.T) {
    ch := make(chan string, 1)

    // Send value asynchronously
    go func() {
        time.Sleep(50 * time.Millisecond)
        ch <- "result"
    }()

    select {
    case result := <-ch:
        assert.Equal(t, "result", result)
    case <-time.After(1 * time.Second):
        t.Fatal("Timeout: result not received within 1 second")
    }
}

// testify's Eventually (polling-based assertion)
func TestEventualConsistency(t *testing.T) {
    service := NewEventualService()
    service.TriggerUpdate("key-1", "new-value")

    // Verify that the value is eventually updated
    assert.Eventually(t, func() bool {
        val, err := service.Get("key-1")
        return err == nil && val == "new-value"
    }, 5*time.Second, 100*time.Millisecond)
}

// testify's Never (verify a condition never occurs)
func TestNeverHappens(t *testing.T) {
    service := NewStableService()

    assert.Never(t, func() bool {
        return service.HasError()
    }, 1*time.Second, 100*time.Millisecond)
}
```

### 6.3 HTTP Testing

```go
import (
    "net/http"
    "net/http/httptest"
    "testing"
)

// Mock server with httptest.Server
func TestExternalAPIClient(t *testing.T) {
    // Test server
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        switch r.URL.Path {
        case "/api/users/user-123":
            w.Header().Set("Content-Type", "application/json")
            w.WriteHeader(http.StatusOK)
            w.Write([]byte(`{"id": "user-123", "name": "Taro Tanaka"}`))
        case "/api/users/not-found":
            w.WriteHeader(http.StatusNotFound)
            w.Write([]byte(`{"error": "Not found"}`))
        default:
            w.WriteHeader(http.StatusNotFound)
        }
    }))
    defer server.Close()

    // Create client using test server URL
    client := NewAPIClient(server.URL)

    t.Run("success case", func(t *testing.T) {
        user, err := client.GetUser(context.Background(), "user-123")
        require.NoError(t, err)
        assert.Equal(t, "Taro Tanaka", user.Name)
    })

    t.Run("404 error", func(t *testing.T) {
        _, err := client.GetUser(context.Background(), "not-found")
        assert.ErrorIs(t, err, ErrUserNotFound)
    })
}

// Handler testing with httptest.NewRecorder
func TestUserHandler(t *testing.T) {
    handler := NewUserHandler(mockUserService)

    req := httptest.NewRequest("GET", "/api/users/user-123", nil)
    rec := httptest.NewRecorder()

    handler.ServeHTTP(rec, req)

    assert.Equal(t, http.StatusOK, rec.Code)

    var user User
    err := json.NewDecoder(rec.Body).Decode(&user)
    require.NoError(t, err)
    assert.Equal(t, "Taro Tanaka", user.Name)
}
```

---

## 7. Async Waiting Strategies for E2E Testing

### 7.1 Playwright (TypeScript)

```typescript
import { test, expect } from '@playwright/test';

test.describe('User management screen', () => {
  test('user list is displayed', async ({ page }) => {
    await page.goto('/users');

    // Wait for network request to complete
    await page.waitForResponse(
      response => response.url().includes('/api/users') && response.status() === 200,
    );

    // Wait for elements to appear
    await expect(page.getByText('Taro Tanaka')).toBeVisible();
    await expect(page.getByText('Hanako Suzuki')).toBeVisible();
  });

  test('user creation flow', async ({ page }) => {
    await page.goto('/users/new');

    // Form input
    await page.getByLabel('Name').fill('New User');
    await page.getByLabel('Email').fill('new@example.com');

    // Submit form while waiting for API response
    const responsePromise = page.waitForResponse('/api/users');
    await page.getByRole('button', { name: 'Create' }).click();
    const response = await responsePromise;

    expect(response.status()).toBe(201);

    // Wait for redirect
    await page.waitForURL('/users/*');

    // Verify success message is displayed
    await expect(page.getByText('User has been created')).toBeVisible();
  });

  test('error display test', async ({ page }) => {
    // Mock API (Playwright routing)
    await page.route('/api/users', route =>
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal Server Error' }),
      }),
    );

    await page.goto('/users');

    // Wait for error message to appear
    await expect(page.getByText('Failed to fetch data')).toBeVisible();

    // Click retry button
    await page.route('/api/users', route =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([{ id: '1', name: 'Taro Tanaka' }]),
      }),
    );

    await page.getByRole('button', { name: 'Retry' }).click();
    await expect(page.getByText('Taro Tanaka')).toBeVisible();
  });
});

// Testing network state
test('behavior when offline', async ({ page, context }) => {
  await page.goto('/dashboard');
  await expect(page.getByTestId('status')).toHaveText('Online');

  // Go offline
  await context.setOffline(true);
  await expect(page.getByTestId('status')).toHaveText('Offline');

  // Go back online
  await context.setOffline(false);
  await expect(page.getByTestId('status')).toHaveText('Online');
});
```

### 7.2 Cypress

```typescript
// Cypress: waiting strategies for asynchronous tests

describe('User management', () => {
  beforeEach(() => {
    // API mock setup
    cy.intercept('GET', '/api/users', {
      fixture: 'users.json',
    }).as('getUsers');

    cy.intercept('POST', '/api/users', {
      statusCode: 201,
      body: { id: 'user-new', name: 'New User' },
    }).as('createUser');
  });

  it('displays user list', () => {
    cy.visit('/users');

    // Wait for API response
    cy.wait('@getUsers');

    // Verify element display
    cy.findByText('Taro Tanaka').should('be.visible');
    cy.findByText('Hanako Suzuki').should('be.visible');
  });

  it('creates a user', () => {
    cy.visit('/users/new');

    cy.findByLabelText('Name').type('New User');
    cy.findByLabelText('Email').type('new@example.com');
    cy.findByRole('button', { name: 'Create' }).click();

    // Wait for API response and also validate request content
    cy.wait('@createUser').then((interception) => {
      expect(interception.request.body).to.deep.equal({
        name: 'New User',
        email: 'new@example.com',
      });
    });

    // Verify redirect
    cy.url().should('match', /\/users\/.+/);
    cy.findByText('User has been created').should('be.visible');
  });

  it('handles network errors', () => {
    cy.intercept('GET', '/api/users', {
      forceNetworkError: true,
    }).as('getUsersFailed');

    cy.visit('/users');
    cy.wait('@getUsersFailed');

    cy.findByText('A network error occurred').should('be.visible');
  });

  it('tests delayed responses', () => {
    cy.intercept('GET', '/api/users', {
      fixture: 'users.json',
      delay: 3000, // 3-second delay
    }).as('getUsers');

    cy.visit('/users');

    // Verify loading display
    cy.findByTestId('loading-spinner').should('be.visible');

    // After data fetch completes
    cy.wait('@getUsers');
    cy.findByTestId('loading-spinner').should('not.exist');
    cy.findByText('Taro Tanaka').should('be.visible');
  });
});

// Cypress: wrapping async operations with custom commands
Cypress.Commands.add('waitForApiAndAssert', (alias: string, assertion: Function) => {
  cy.wait(alias).then((interception) => {
    assertion(interception);
  });
});
```

### 7.3 WebSocket E2E Testing

```typescript
// Playwright: WebSocket testing
test('WebSocket real-time communication', async ({ page }) => {
  // Monitor WebSocket messages
  const messages: string[] = [];
  page.on('websocket', ws => {
    ws.on('framereceived', frame => {
      messages.push(frame.payload as string);
    });
  });

  await page.goto('/chat');

  // Send message
  await page.getByPlaceholder('Enter message').fill('Hello');
  await page.getByRole('button', { name: 'Send' }).click();

  // Verify the sent message is displayed
  await expect(page.getByText('Hello')).toBeVisible();

  // Verify the message was sent via WebSocket
  expect(messages.some(m => m.includes('Hello'))).toBe(true);
});

// WebSocket mocking
test('WebSocket mock', async ({ page }) => {
  // Mock WebSocket route
  await page.routeWebSocket('/ws', ws => {
    ws.onMessage(message => {
      // Echo back
      const data = JSON.parse(message as string);
      ws.send(JSON.stringify({
        type: 'echo',
        data: data.message,
        timestamp: Date.now(),
      }));
    });
  });

  await page.goto('/chat');
  await page.getByPlaceholder('Enter message').fill('Test');
  await page.getByRole('button', { name: 'Send' }).click();

  await expect(page.getByText('Test')).toBeVisible();
});
```

---

## 8. Design Patterns for Asynchronous Testing

### 8.1 Test Helper Design

```typescript
// Reusable asynchronous test helpers

/**
 * Verify that an async operation completes within a specified time
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
 * Verify that an async operation eventually succeeds (polling)
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
 * Retry an async operation a specified number of times for testing
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

// Usage examples
test('API responds within 1 second', async () => {
  const result = await expectToCompleteWithin(
    () => fetchUser('user-123'),
    1000,
    'API response too slow',
  );
  expect(result.name).toBe('Taro Tanaka');
});

test('cache is refreshed', async () => {
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

### 8.2 Test Double Patterns

```typescript
// Classification and implementation of asynchronous test doubles

// 1. Stub: returns fixed values
class StubUserRepository {
  async findById(id: string): Promise<User | null> {
    const users: Record<string, User> = {
      'user-1': { id: 'user-1', name: 'Taro Tanaka', email: 'tanaka@example.com' },
      'user-2': { id: 'user-2', name: 'Hanako Suzuki', email: 'suzuki@example.com' },
    };
    return users[id] ?? null;
  }

  async save(user: User): Promise<User> {
    return { ...user, id: user.id || 'generated-id' };
  }
}

// 2. Spy: records invocations
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

// 3. Fake: simplified implementation
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

  // Test helpers
  clear(): void {
    this.store.clear();
  }

  size(): number {
    return this.store.size;
  }
}

// 4. Mock: sets expectations and verifies
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

// Usage in tests
test('order processing triggers email and payment', async () => {
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

  // Spy verification
  expect(emailSpy.getCallCount()).toBe(1);
  expect(emailSpy.wasCalledWith('tanaka@example.com')).toBe(true);

  // Mock verification
  paymentMock.verify();
});
```

### 8.3 Event-Driven Testing

```typescript
// EventEmitter-based asynchronous testing

import { EventEmitter } from 'events';

// Wait for an event with once
test('event is emitted', async () => {
  const emitter = new EventEmitter();

  const eventPromise = new Promise<{ type: string; data: any }>((resolve) => {
    emitter.once('user:created', (data) => resolve({ type: 'user:created', data }));
  });

  // Emit event asynchronously
  setTimeout(() => {
    emitter.emit('user:created', { id: 'user-1', name: 'Taro Tanaka' });
  }, 100);

  const event = await eventPromise;
  expect(event.type).toBe('user:created');
  expect(event.data.name).toBe('Taro Tanaka');
});

// Using Node.js events.once
import { once } from 'events';

test('wait with events.once', async () => {
  const emitter = new EventEmitter();

  setTimeout(() => {
    emitter.emit('data', { value: 42 });
  }, 50);

  const [data] = await once(emitter, 'data');
  expect(data.value).toBe(42);
});

// Testing event ordering
test('verify event order', async () => {
  const events: string[] = [];
  const processor = new OrderProcessor();

  processor.on('started', () => events.push('started'));
  processor.on('validated', () => events.push('validated'));
  processor.on('charged', () => events.push('charged'));
  processor.on('completed', () => events.push('completed'));

  await processor.process({ productId: 'prod-1', amount: 1000 });

  expect(events).toEqual(['started', 'validated', 'charged', 'completed']);
});

// Testing error events
test('error event is emitted', async () => {
  const processor = new OrderProcessor();

  const errorPromise = new Promise<Error>((resolve) => {
    processor.on('error', resolve);
  });

  // Trigger error with invalid order
  processor.process({ productId: '', amount: -100 }).catch(() => {});

  const error = await errorPromise;
  expect(error.message).toContain('Invalid order');
});
```

---

## 9. Property-Based Testing

### 9.1 Testing Async Properties with fast-check

```typescript
import fc from 'fast-check';

// Asynchronous property-based testing
test('encode then decode returns original', async () => {
  await fc.assert(
    fc.asyncProperty(fc.string(), async (input) => {
      const encoded = await encode(input);
      const decoded = await decode(encoded);
      expect(decoded).toBe(input);
    }),
  );
});

// Property testing for concurrent processing
test('counter remains accurate under concurrent access', async () => {
  await fc.assert(
    fc.asyncProperty(
      fc.integer({ min: 1, max: 100 }),
      fc.integer({ min: 1, max: 50 }),
      async (incrementCount, concurrency) => {
        const counter = new AtomicCounter();

        const tasks = Array.from({ length: incrementCount }, () =>
          counter.increment(),
        );

        // Execute with concurrency limit
        await promisePool(tasks.map(t => () => t), concurrency);

        expect(counter.value).toBe(incrementCount);
      },
    ),
    { numRuns: 50 },
  );
});

// Property testing for retries
test('retry either eventually succeeds or stops at max attempts', async () => {
  await fc.assert(
    fc.asyncProperty(
      fc.integer({ min: 0, max: 10 }), // Number of failures
      fc.integer({ min: 1, max: 5 }),   // Max retries
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

          // If successful: failure count < max retries
          expect(result).toBe('success');
          expect(failCount).toBeLessThan(maxRetries);
        } catch {
          // If failed: failure count >= max retries
          expect(failCount).toBeGreaterThanOrEqual(maxRetries);
        }

        // Verify call count
        expect(callCount).toBeLessThanOrEqual(maxRetries + 1);
      },
    ),
    { numRuns: 100 },
  );
});

// Property testing for database operations
test('CRUD operation consistency', async () => {
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

### 9.2 Hypothesis (Python)

```python
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
import asyncio

# Async testing with hypothesis
@pytest.mark.asyncio
@given(st.text(min_size=1, max_size=100))
@settings(max_examples=50)
async def test_encode_decode_roundtrip(text):
    """Encode-decode roundtrip test"""
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
    """All items are processed in batch processing"""
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

## 10. Test Performance and CI Optimization

### 10.1 Speeding Up Test Execution

```typescript
// Parallel test execution configuration
// jest.config.ts
export default {
  // Worker count optimization
  maxWorkers: '50%', // Use 50% of CPU
  // maxWorkers: 4, // Fixed value is also possible

  // Parallel test execution
  // Between files: parallel, within files: sequential (default)

  // Slow test timeout
  testTimeout: 10000,

  // Global setup (once for the entire test suite)
  globalSetup: './test/global-setup.ts',
  globalTeardown: './test/global-teardown.ts',

  // Project configuration to isolate different test environments
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

### 10.2 Test Grouping and Selective Execution

```typescript
// Tag-based test selection

// Tagging tests
test('slow test #slow', async () => {
  // ...
});

test('fast test #fast', async () => {
  // ...
});

// Filtering at runtime
// jest --testNamePattern="#fast"
// jest --testNamePattern="^(?!.*#slow)"  // Exclude #slow

// Vitest tag feature
// vitest run --reporter=verbose --bail 1

// Temporarily disabling with describe.skip / test.skip
describe.skip('WIP: new feature tests', () => {
  test('incomplete test', async () => {
    // ...
  });
});

// Focusing with describe.only / test.only (prohibited in CI)
// eslint-plugin-jest: no-focused-tests rule for prevention
```

### 10.3 Async Tests in CI Environments

```yaml
# GitHub Actions: async test configuration
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
          # Set longer timeout for CI environment
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

### 10.4 Retry Strategies for CI

```typescript
// Playwright: test retry configuration
// playwright.config.ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  retries: process.env.CI ? 2 : 0, // Retry twice in CI

  use: {
    // Screenshots and traces on failure
    screenshot: 'only-on-failure',
    trace: 'on-first-retry',
    video: 'on-first-retry',
  },

  // Per-project configuration
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
      retries: 2,
    },
    {
      name: 'firefox',
      use: { browserName: 'firefox' },
      retries: 3, // More retries for Firefox
    },
  ],
});

// Jest: test retry (jest-circus)
// jest.config.ts
export default {
  // jest-circus retry feature (experimental)
  // Per-test retry
};

// Custom retry wrapper
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
        return; // Success
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

// Usage example
testWithRetry('unstable external API integration test', async () => {
  const result = await externalApiCall();
  expect(result.status).toBe('ok');
}, 3);
```

---

## 11. Test Coverage and Quality Metrics

### 11.1 Coverage for Asynchronous Code

```typescript
// Key points to watch for async code coverage

// Problem: catch branch is not tested
async function fetchUserSafe(id: string): Promise<User | null> {
  try {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`); // <- This branch
    }
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch user:', error); // <- This branch
    return null;
  }
}

// Happy path test
test('successfully retrieve user', async () => {
  const user = await fetchUserSafe('user-123');
  expect(user).not.toBeNull();
  expect(user!.name).toBe('Taro Tanaka');
});

// Error case tests (for coverage improvement)
test('returns null on HTTP error', async () => {
  server.use(
    http.get('/api/users/:id', () => {
      return HttpResponse.json({ error: 'Not found' }, { status: 404 });
    }),
  );

  const user = await fetchUserSafe('invalid');
  expect(user).toBeNull();
});

test('returns null on network error', async () => {
  server.use(
    http.get('/api/users/:id', () => {
      return HttpResponse.error();
    }),
  );

  const user = await fetchUserSafe('user-123');
  expect(user).toBeNull();
});

// Coverage for Promise.allSettled
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

test('verify results when some requests fail', async () => {
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

### 11.2 Mutation Testing

```typescript
// Stryker: verify async code quality with mutation testing
// stryker.conf.json
{
  "mutate": ["src/**/*.ts", "!src/**/*.test.ts"],
  "testRunner": "jest",
  "reporters": ["html", "clear-text", "progress"],
  "coverageAnalysis": "perTest",
  "timeoutMS": 60000,

  // Additional mutants for async code
  "mutator": {
    "excludedMutations": [
      // Exclude unnecessary mutants
    ]
  }
}

// Mutant examples:
// Original code:
// if (retries < maxRetries) { ... }
// Mutants:
// if (retries <= maxRetries) { ... }  <- Boundary value mutation
// if (retries > maxRetries) { ... }   <- Condition inversion
// if (true) { ... }                   <- Condition removal

// Tests that kill these mutants
test('stops precisely at max retry count', async () => {
  const fn = jest.fn().mockRejectedValue(new Error('fail'));

  await expect(
    retryWithBackoff(fn, { maxRetries: 3, initialDelay: 1 }),
  ).rejects.toThrow('fail');

  // Initial + 3 retries = 4 times
  expect(fn).toHaveBeenCalledTimes(4);
});

test('still retries at max retry count - 1', async () => {
  let callCount = 0;
  const fn = jest.fn().mockImplementation(async () => {
    callCount++;
    if (callCount <= 2) throw new Error('fail'); // Fail 2 times
    return 'success';
  });

  const result = await retryWithBackoff(fn, { maxRetries: 3, initialDelay: 1 });
  expect(result).toBe('success');
  expect(callCount).toBe(3); // Initial + 2 retries
});
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this applied in professional settings?

The knowledge from this topic is frequently applied in daily development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Technique | Purpose | Tools |
|-----------|---------|-------|
| async/await | Async testing | Jest, Vitest |
| Fake timers | Timer mocking | jest.useFakeTimers, vi.useFakeTimers |
| advanceTimersByTimeAsync | Promise + timers | Jest 29.5+, Vitest |
| msw v2 | HTTP API mocking | Mock Service Worker |
| GraphQL mocking | GraphQL API mocking | msw graphql handlers |
| Fake date/time | Date mocking | jest.setSystemTime, vi.setSystemTime |
| pytest-asyncio | Python async testing | pytest + asyncio |
| AsyncMock | Python mocking | unittest.mock |
| aioresponses | Python HTTP mocking | For aiohttp testing |
| httptest | Go HTTP testing | net/http/httptest |
| testify | Go assertions | github.com/stretchr/testify |
| Playwright | E2E testing | @playwright/test |
| Cypress | E2E testing | cypress |
| fast-check | Property-based testing | fast-check |
| Hypothesis | Python property testing | hypothesis |
| waitFor | Async DOM waiting | @testing-library |
| Stryker | Mutation testing | @stryker-mutator |

### Testing Framework Comparison

| Feature | Jest | Vitest | pytest |
|---------|------|--------|--------|
| Fake timers | jest.useFakeTimers() | vi.useFakeTimers() | freezegun |
| Async timers | advanceTimersByTimeAsync | advanceTimersByTimeAsync | - |
| HTTP mocking | msw, nock | msw, nock | aioresponses, httpx-mock |
| Snapshots | toMatchSnapshot | toMatchSnapshot | syrupy |
| Parallel execution | --maxWorkers | --pool threads | pytest-xdist |
| Coverage | --coverage (istanbul/v8) | --coverage (v8/istanbul) | pytest-cov |
| Watch mode | --watch | --watch (HMR support) | pytest-watch |

### Best Practices for Asynchronous Testing

```
1. Guarantee test independence
   - Each test must not depend on other tests
   - Reset state in beforeEach
   - Properly clean up shared resources

2. Write deterministic tests
   - Mock date/time, random numbers, and UUIDs
   - Mock external APIs with msw, etc.
   - Use fake timers for timers

3. Choose appropriate waiting strategies
   - Avoid fixed sleep() (causes flakiness)
   - Use waitFor / waitUntil for polling waits
   - Prefer event-based waiting

4. Be mindful of test granularity
   - Unit tests: individual async functions
   - Integration tests: multiple component interactions
   - E2E tests: entire user scenarios

5. Ensure stability in CI
   - Configure retry strategies
   - Set appropriate timeouts
   - Collect artifacts on failure
```

---

## Recommended Next Guides

---

## References
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
