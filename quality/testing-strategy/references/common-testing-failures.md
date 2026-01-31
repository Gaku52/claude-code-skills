# よくあるテストの失敗パターン集

**目的**: 開発現場でよく見られるテストの失敗パターンとその解決策をまとめたリファレンス

---

## 目次

1. [テスト設計の失敗](#1-テスト設計の失敗)
2. [テストの独立性の問題](#2-テストの独立性の問題)
3. [モック・スタブの誤用](#3-モックスタブの誤用)
4. [非同期処理の問題](#4-非同期処理の問題)
5. [フレイキーテスト](#5-フレイキーテスト)
6. [パフォーマンスの問題](#6-パフォーマンスの問題)
7. [アサーションの問題](#7-アサーションの問題)
8. [テストデータの問題](#8-テストデータの問題)
9. [CI/CDの問題](#9-cicdの問題)
10. [メンテナンス性の問題](#10-メンテナンス性の問題)

---

## 1. テスト設計の失敗

### ❌ 失敗例 #1: テストが実装の詳細に依存

**問題**:
```typescript
// ❌ 悪い例: 内部実装をテスト
it('should call _calculateInternal method', () => {
  const spy = jest.spyOn(calculator, '_calculateInternal');
  calculator.calculate(5, 3);
  expect(spy).toHaveBeenCalled();
});
```

**なぜダメ**:
- リファクタリングでテストが壊れる
- 内部実装の変更がテストに影響
- 公開APIではなく実装をテスト

**✅ 解決策**:
```typescript
// ✅ 良い例: 公開APIの振る舞いをテスト
it('should return correct calculation result', () => {
  const result = calculator.calculate(5, 3);
  expect(result).toBe(15); // 結果のみを検証
});
```

---

### ❌ 失敗例 #2: 一度に複数の振る舞いをテスト

**問題**:
```typescript
// ❌ 悪い例: 1つのテストで複数をチェック
it('should handle user operations', () => {
  const user = createUser();
  expect(user.name).toBe('Test');

  user.updateEmail('new@example.com');
  expect(user.email).toBe('new@example.com');

  user.delete();
  expect(user.isDeleted).toBe(true);
  // どこで失敗したか分かりにくい
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: 1テスト = 1振る舞い
describe('User', () => {
  it('should create user with correct name', () => {
    const user = createUser();
    expect(user.name).toBe('Test');
  });

  it('should update email', () => {
    const user = createUser();
    user.updateEmail('new@example.com');
    expect(user.email).toBe('new@example.com');
  });

  it('should mark as deleted when deleted', () => {
    const user = createUser();
    user.delete();
    expect(user.isDeleted).toBe(true);
  });
});
```

---

### ❌ 失敗例 #3: 異常系テストの不足

**問題**:
```typescript
// ❌ 悪い例: 正常系のみ
describe('divide', () => {
  it('should divide numbers', () => {
    expect(divide(10, 2)).toBe(5);
  });
  // ゼロ除算のテストがない！
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: 正常系 + 異常系
describe('divide', () => {
  it('should divide numbers correctly', () => {
    expect(divide(10, 2)).toBe(5);
  });

  it('should throw error for division by zero', () => {
    expect(() => divide(10, 0)).toThrow('Division by zero');
  });

  it('should handle negative numbers', () => {
    expect(divide(-10, 2)).toBe(-5);
  });

  it('should handle decimals', () => {
    expect(divide(5, 2)).toBe(2.5);
  });
});
```

---

## 2. テストの独立性の問題

### ❌ 失敗例 #4: テスト間で状態を共有

**問題**:
```typescript
// ❌ 悪い例: グローバル変数を共有
let sharedUser;

it('test 1', () => {
  sharedUser = { name: 'Test' };
  expect(sharedUser.name).toBe('Test');
});

it('test 2', () => {
  sharedUser.name = 'Modified';
  expect(sharedUser.name).toBe('Modified');
  // test 1 の状態に依存
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: 各テストで独立したデータ
describe('User', () => {
  let user;

  beforeEach(() => {
    user = { name: 'Test' }; // 毎回新しいデータ
  });

  it('test 1', () => {
    expect(user.name).toBe('Test');
  });

  it('test 2', () => {
    user.name = 'Modified';
    expect(user.name).toBe('Modified');
    // 他のテストに影響しない
  });
});
```

---

### ❌ 失敗例 #5: テストの実行順序に依存

**問題**:
```typescript
// ❌ 悪い例: 順序依存
describe('Counter', () => {
  const counter = new Counter();

  it('should start at 0', () => {
    expect(counter.value).toBe(0);
  });

  it('should increment', () => {
    counter.increment();
    expect(counter.value).toBe(1); // 前のテストに依存
  });

  it('should decrement', () => {
    counter.decrement();
    expect(counter.value).toBe(0); // さらに依存
  });
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: 各テストで初期化
describe('Counter', () => {
  let counter;

  beforeEach(() => {
    counter = new Counter(); // 毎回リセット
  });

  it('should start at 0', () => {
    expect(counter.value).toBe(0);
  });

  it('should increment from 0 to 1', () => {
    counter.increment();
    expect(counter.value).toBe(1);
  });

  it('should decrement from 0 to -1', () => {
    counter.decrement();
    expect(counter.value).toBe(-1);
  });
});
```

---

## 3. モック・スタブの誤用

### ❌ 失敗例 #6: 過度なモック化

**問題**:
```typescript
// ❌ 悪い例: 全てモック
it('should create user', async () => {
  const mockDb = jest.fn();
  const mockValidator = jest.fn().mockReturnValue(true);
  const mockHasher = jest.fn().mockReturnValue('hashed');
  const mockMailer = jest.fn();

  const service = new UserService(mockDb, mockValidator, mockHasher, mockMailer);
  await service.createUser({ /* ... */ });

  // 実際のロジックが全くテストされていない
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: 必要最小限のモック
it('should create user with hashed password', async () => {
  const mockHasher = jest.fn().mockReturnValue('hashed-password');
  const mockMailer = jest.fn(); // 外部サービスのみモック

  const service = new UserService(realDb, realValidator, mockHasher, mockMailer);
  const user = await service.createUser({
    email: 'test@example.com',
    password: 'plain',
  });

  expect(user.password).toBe('hashed-password');
  expect(mockHasher).toHaveBeenCalledWith('plain');
});
```

---

### ❌ 失敗例 #7: モックの振る舞いが現実と乖離

**問題**:
```typescript
// ❌ 悪い例: 現実的でないモック
it('should fetch user', async () => {
  const mockApi = jest.fn().mockResolvedValue({
    // 実際のAPIとは異なる構造
    user: { name: 'Test' },
  });

  const result = await service.getUser(mockApi, '123');
  expect(result.name).toBe('Test');
  // 本番では失敗する可能性
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: 実際のAPIレスポンスと同じ構造
it('should fetch user', async () => {
  const mockApi = jest.fn().mockResolvedValue({
    // 実際のAPIレスポンスと同じ
    data: {
      user: {
        id: '123',
        name: 'Test',
        email: 'test@example.com',
      },
    },
    status: 200,
  });

  const result = await service.getUser(mockApi, '123');
  expect(result.name).toBe('Test');
});
```

---

## 4. 非同期処理の問題

### ❌ 失敗例 #8: async/await の忘れ

**問題**:
```typescript
// ❌ 悪い例: awaitを忘れる
it('should fetch data', () => {
  const result = apiService.fetchData(); // Promiseを返す
  expect(result).toBe('data'); // 失敗: Promiseオブジェクトと比較
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: async/await を使用
it('should fetch data', async () => {
  const result = await apiService.fetchData();
  expect(result).toBe('data');
});
```

---

### ❌ 失敗例 #9: タイマー処理の誤り

**問題**:
```typescript
// ❌ 悪い例: 実際に待機
it('should call callback after delay', (done) => {
  const callback = jest.fn();
  setTimeout(callback, 5000); // 5秒待機

  setTimeout(() => {
    expect(callback).toHaveBeenCalled();
    done();
  }, 5100); // テストが遅い
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: フェイクタイマー使用
it('should call callback after delay', () => {
  jest.useFakeTimers();
  const callback = jest.fn();

  setTimeout(callback, 5000);

  jest.advanceTimersByTime(5000); // 即座に5秒進める
  expect(callback).toHaveBeenCalled();

  jest.useRealTimers();
});
```

---

## 5. フレイキーテスト

### ❌ 失敗例 #10: ランダム値の使用

**問題**:
```typescript
// ❌ 悪い例: ランダム値でテストが不安定
it('should generate unique ID', () => {
  const id = generateId(); // Math.random() を使用
  expect(id).toBe('abc123'); // 時々失敗
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: パターンを検証
it('should generate unique ID', () => {
  const id = generateId();
  expect(id).toMatch(/^[a-z0-9]{6}$/); // 形式のみ検証
  expect(id.length).toBe(6);
});

// または、Math.random をモック
it('should generate deterministic ID', () => {
  jest.spyOn(Math, 'random').mockReturnValue(0.5);
  const id = generateId();
  expect(id).toBe('predictable-id');
});
```

---

### ❌ 失敗例 #11: 固定待機時間

**問題**:
```typescript
// ❌ 悪い例: 固定待機
it('should render after loading', async () => {
  render(<AsyncComponent />);
  await new Promise(r => setTimeout(r, 1000)); // 遅い環境では失敗
  expect(screen.getByText('Loaded')).toBeInTheDocument();
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: waitFor を使用
it('should render after loading', async () => {
  render(<AsyncComponent />);
  await waitFor(() => {
    expect(screen.getByText('Loaded')).toBeInTheDocument();
  });
});
```

---

## 6. パフォーマンスの問題

### ❌ 失敗例 #12: 不要なセットアップ

**問題**:
```typescript
// ❌ 悪い例: 全テストで重い処理
describe('Calculator', () => {
  beforeEach(async () => {
    await setupComplexDatabase(); // 毎回実行
    await loadLargeConfig();
  });

  it('should add', () => {
    expect(add(2, 3)).toBe(5); // DBもConfigも不要
  });
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: 必要なテストのみセットアップ
describe('Calculator', () => {
  describe('simple operations', () => {
    it('should add', () => {
      expect(add(2, 3)).toBe(5);
    });
  });

  describe('database operations', () => {
    beforeEach(async () => {
      await setupComplexDatabase(); // 必要な時だけ
    });

    it('should save result', async () => {
      await saveToDb(5);
      // ...
    });
  });
});
```

---

## 7. アサーションの問題

### ❌ 失敗例 #13: アサーションなし

**問題**:
```typescript
// ❌ 悪い例: アサーションがない
it('should process data', () => {
  processData({ value: 10 });
  // 何も検証していない
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: 結果を検証
it('should process data and return result', () => {
  const result = processData({ value: 10 });
  expect(result).toBeDefined();
  expect(result.processed).toBe(true);
  expect(result.value).toBe(10);
});
```

---

### ❌ 失敗例 #14: 曖昧なアサーション

**問題**:
```typescript
// ❌ 悪い例: 曖昧
it('should return data', async () => {
  const result = await fetchData();
  expect(result).toBeTruthy(); // 何でもOK
  expect(result.length).toBeGreaterThan(0); // 具体性なし
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: 具体的
it('should return array of users', async () => {
  const result = await fetchUsers();
  expect(Array.isArray(result)).toBe(true);
  expect(result).toHaveLength(5);
  expect(result[0]).toMatchObject({
    id: expect.any(String),
    name: expect.any(String),
    email: expect.stringMatching(/^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/),
  });
});
```

---

## 8. テストデータの問題

### ❌ 失敗例 #15: ハードコードされたデータ

**問題**:
```typescript
// ❌ 悪い例: ハードコード
it('test 1', () => {
  const user = { id: '123', name: 'Test', email: 'test@example.com' };
  // ...
});

it('test 2', () => {
  const user = { id: '123', name: 'Test', email: 'test@example.com' }; // 重複
  // ...
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: ファクトリー関数
function createTestUser(overrides = {}) {
  return {
    id: '123',
    name: 'Test User',
    email: 'test@example.com',
    ...overrides,
  };
}

it('test 1', () => {
  const user = createTestUser();
  // ...
});

it('test 2', () => {
  const user = createTestUser({ name: 'Custom Name' });
  // ...
});
```

---

## 9. CI/CDの問題

### ❌ 失敗例 #16: 環境依存のテスト

**問題**:
```typescript
// ❌ 悪い例: ローカルパスに依存
it('should read file', () => {
  const data = readFileSync('/Users/myname/project/data.json');
  expect(data).toBeDefined(); // CI環境で失敗
});
```

**✅ 解決策**:
```typescript
// ✅ 良い例: 相対パスや環境変数
it('should read file', () => {
  const dataPath = path.join(__dirname, '../fixtures/data.json');
  const data = readFileSync(dataPath);
  expect(data).toBeDefined();
});
```

---

## 10. メンテナンス性の問題

### ❌ 失敗例 #17: テスト名が不明瞭

**問題**:
```typescript
// ❌ 悪い例: 何をテストしているか不明
it('works', () => { /* ... */ });
it('test 1', () => { /* ... */ });
it('should return true', () => { /* ... */ });
```

**✅ 解決策**:
```typescript
// ✅ 良い例: 明確な名前
it('should return true when user is authenticated', () => { /* ... */ });
it('should throw ValidationError when email is invalid', () => { /* ... */ });
it('should update user profile and send confirmation email', () => { /* ... */ });
```

---

## まとめ

### 失敗パターンの分類

| カテゴリ | 主な原因 | 対策 |
|---------|---------|------|
| テスト設計 | 実装の詳細への依存 | 公開APIをテスト |
| 独立性 | 状態の共有 | beforeEach で初期化 |
| モック | 過度なモック化 | 必要最小限に |
| 非同期 | await忘れ | async/await 徹底 |
| フレイキー | ランダム値・タイムアウト | 決定的なテスト |
| パフォーマンス | 不要なセットアップ | 最適化 |
| アサーション | 曖昧な検証 | 具体的に |
| データ | 重複・ハードコード | ファクトリー |
| CI/CD | 環境依存 | 可搬性確保 |
| メンテナンス | 不明瞭な名前 | 明確な命名 |

### 推奨チェックリスト

- [ ] テストは公開APIのみを検証している
- [ ] 各テストが独立して実行可能
- [ ] モックは必要最小限
- [ ] 非同期処理を適切に扱っている
- [ ] テストが決定的（毎回同じ結果）
- [ ] テストの実行時間が適切（<1s for unit）
- [ ] アサーションが具体的
- [ ] テストデータが再利用可能
- [ ] 環境に依存しない
- [ ] テスト名が明確

---

**最終更新**: 2026-01-02
**関連ドキュメント**:
- [Troubleshooting Guide](./troubleshooting-guide.md)
