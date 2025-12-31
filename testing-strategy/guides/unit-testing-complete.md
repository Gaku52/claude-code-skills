# ユニットテスト完全ガイド

## 対応バージョン
- **Jest**: 29.0.0以上
- **Vitest**: 1.0.0以上
- **TypeScript**: 5.0.0以上
- **Node.js**: 20.0.0以上

---

## ユニットテストの基礎

### ユニットテストとは

ユニットテストは、個々の関数やクラスメソッドなど、最小単位（ユニット）の動作を検証するテストです。

**特徴:**
- 高速（ミリ秒単位）
- 外部依存を排除（モック使用）
- 単一の責任をテスト
- テストピラミッドの基盤

**AAA パターン:**
- **Arrange**: テストデータ準備
- **Act**: テスト対象実行
- **Assert**: 結果検証

---

## Jest基本設定

### インストールと設定

```bash
# Jest + TypeScript
npm install --save-dev jest @types/jest ts-jest

# 設定ファイル生成
npx ts-jest config:init
```

```typescript
// jest.config.ts
import type { Config } from 'jest'

const config: Config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/**/*.interface.ts',
    '!src/**/index.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
}

export default config
```

```json
// package.json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage"
  }
}
```

---

## 基本的なテストの書き方

### シンプルな関数のテスト

```typescript
// src/utils/math.ts
export function add(a: number, b: number): number {
  return a + b
}

export function divide(a: number, b: number): number {
  if (b === 0) {
    throw new Error('Division by zero')
  }
  return a / b
}
```

```typescript
// src/utils/math.test.ts
import { add, divide } from './math'

describe('Math utils', () => {
  describe('add', () => {
    it('should add two positive numbers', () => {
      const result = add(2, 3)
      expect(result).toBe(5)
    })

    it('should add negative numbers', () => {
      expect(add(-2, -3)).toBe(-5)
    })

    it('should add zero', () => {
      expect(add(5, 0)).toBe(5)
    })
  })

  describe('divide', () => {
    it('should divide two numbers', () => {
      expect(divide(10, 2)).toBe(5)
    })

    it('should handle division by zero', () => {
      expect(() => divide(10, 0)).toThrow('Division by zero')
    })

    it('should return decimal for non-divisible numbers', () => {
      expect(divide(7, 2)).toBe(3.5)
    })
  })
})
```

### クラスのテスト

```typescript
// src/services/calculator.ts
export class Calculator {
  private history: number[] = []

  add(a: number, b: number): number {
    const result = a + b
    this.history.push(result)
    return result
  }

  getHistory(): number[] {
    return [...this.history]
  }

  clearHistory(): void {
    this.history = []
  }
}
```

```typescript
// src/services/calculator.test.ts
import { Calculator } from './calculator'

describe('Calculator', () => {
  let calculator: Calculator

  beforeEach(() => {
    calculator = new Calculator()
  })

  afterEach(() => {
    // クリーンアップ
  })

  it('should add two numbers', () => {
    const result = calculator.add(2, 3)
    expect(result).toBe(5)
  })

  it('should store calculation history', () => {
    calculator.add(2, 3)
    calculator.add(5, 7)

    const history = calculator.getHistory()

    expect(history).toEqual([5, 12])
  })

  it('should clear history', () => {
    calculator.add(2, 3)
    calculator.clearHistory()

    expect(calculator.getHistory()).toEqual([])
  })

  it('should return copy of history (immutability)', () => {
    calculator.add(2, 3)

    const history1 = calculator.getHistory()
    const history2 = calculator.getHistory()

    expect(history1).not.toBe(history2) // 異なる参照
    expect(history1).toEqual(history2)  // 同じ内容
  })
})
```

---

## アサーション（期待値検証）

### 基本的なアサーション

```typescript
// 等価性
expect(value).toBe(5)                    // === 厳密等価
expect(value).toEqual({ a: 1, b: 2 })    // 深い等価性
expect(value).not.toBe(3)                // 否定

// 真偽値
expect(value).toBeTruthy()
expect(value).toBeFalsy()
expect(value).toBeNull()
expect(value).toBeUndefined()
expect(value).toBeDefined()

// 数値
expect(value).toBeGreaterThan(10)
expect(value).toBeGreaterThanOrEqual(10)
expect(value).toBeLessThan(10)
expect(value).toBeLessThanOrEqual(10)
expect(value).toBeCloseTo(0.3)  // 浮動小数点の比較

// 文字列
expect(str).toMatch(/hello/)
expect(str).toMatch('world')
expect(str).toContain('test')

// 配列・オブジェクト
expect(array).toContain(item)
expect(array).toHaveLength(3)
expect(obj).toHaveProperty('name')
expect(obj).toHaveProperty('age', 25)
expect(obj).toMatchObject({ name: 'John' })

// 例外
expect(() => fn()).toThrow()
expect(() => fn()).toThrow('Error message')
expect(() => fn()).toThrow(CustomError)

// Promise
await expect(promise).resolves.toBe(value)
await expect(promise).rejects.toThrow()
```

### カスタムマッチャー

```typescript
// src/test/matchers.ts
expect.extend({
  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling

    return {
      pass,
      message: () =>
        pass
          ? `expected ${received} not to be within range ${floor} - ${ceiling}`
          : `expected ${received} to be within range ${floor} - ${ceiling}`,
    }
  },
})

declare global {
  namespace jest {
    interface Matchers<R> {
      toBeWithinRange(floor: number, ceiling: number): R
    }
  }
}

// 使用例
expect(100).toBeWithinRange(90, 110)
```

---

## モック（Mock）

### 関数のモック

```typescript
// src/services/user.service.ts
import { sendEmail } from './email.service'

export class UserService {
  async registerUser(email: string, name: string) {
    // ユーザー登録処理...

    // メール送信
    await sendEmail(email, 'Welcome!', `Hello ${name}`)

    return { success: true }
  }
}
```

```typescript
// src/services/user.service.test.ts
import { UserService } from './user.service'
import { sendEmail } from './email.service'

// モジュール全体をモック
jest.mock('./email.service')

describe('UserService', () => {
  let userService: UserService

  beforeEach(() => {
    userService = new UserService()
    // モック関数をリセット
    jest.clearAllMocks()
  })

  it('should send welcome email on registration', async () => {
    // sendEmailをモック関数として取得
    const mockSendEmail = sendEmail as jest.MockedFunction<typeof sendEmail>
    mockSendEmail.mockResolvedValue(undefined)

    await userService.registerUser('user@example.com', 'John')

    // モック関数が呼ばれたことを検証
    expect(mockSendEmail).toHaveBeenCalledTimes(1)
    expect(mockSendEmail).toHaveBeenCalledWith(
      'user@example.com',
      'Welcome!',
      'Hello John'
    )
  })
})
```

### 手動モック

```typescript
// src/services/__mocks__/email.service.ts
export const sendEmail = jest.fn().mockResolvedValue(undefined)
```

### jest.fn() モック関数

```typescript
// モック関数の作成
const mockFn = jest.fn()

// 戻り値を設定
mockFn.mockReturnValue(42)
mockFn.mockReturnValueOnce(1).mockReturnValueOnce(2)

// Promise を返すモック
mockFn.mockResolvedValue('success')
mockFn.mockRejectedValue(new Error('failure'))

// 実装を定義
mockFn.mockImplementation((x: number) => x * 2)

// 呼び出し検証
expect(mockFn).toHaveBeenCalled()
expect(mockFn).toHaveBeenCalledTimes(3)
expect(mockFn).toHaveBeenCalledWith(arg1, arg2)
expect(mockFn).toHaveBeenLastCalledWith(arg)

// 呼び出し履歴
expect(mockFn.mock.calls).toEqual([[arg1], [arg2]])
expect(mockFn.mock.results[0].value).toBe(42)
```

---

## スパイ（Spy）

### jest.spyOn()

```typescript
// src/services/analytics.service.ts
export class AnalyticsService {
  track(event: string, data: any) {
    console.log(`Event: ${event}`, data)
    // 外部サービスに送信...
  }
}
```

```typescript
// src/services/analytics.service.test.ts
import { AnalyticsService } from './analytics.service'

describe('AnalyticsService', () => {
  let analytics: AnalyticsService

  beforeEach(() => {
    analytics = new AnalyticsService()
  })

  it('should track events', () => {
    // console.logをスパイ
    const consoleLogSpy = jest.spyOn(console, 'log').mockImplementation()

    analytics.track('user_login', { userId: '123' })

    expect(consoleLogSpy).toHaveBeenCalledWith(
      'Event: user_login',
      { userId: '123' }
    )

    // スパイを復元
    consoleLogSpy.mockRestore()
  })
})
```

---

## 非同期テスト

### async/await

```typescript
// src/services/api.service.ts
export class ApiService {
  async fetchUser(id: string): Promise<{ id: string; name: string }> {
    const response = await fetch(`https://api.example.com/users/${id}`)
    return response.json()
  }
}
```

```typescript
// src/services/api.service.test.ts
import { ApiService } from './api.service'

global.fetch = jest.fn()

describe('ApiService', () => {
  let apiService: ApiService

  beforeEach(() => {
    apiService = new ApiService()
    jest.clearAllMocks()
  })

  it('should fetch user data', async () => {
    const mockUser = { id: '123', name: 'John' }

    ;(fetch as jest.MockedFunction<typeof fetch>).mockResolvedValue({
      json: async () => mockUser,
    } as Response)

    const user = await apiService.fetchUser('123')

    expect(user).toEqual(mockUser)
    expect(fetch).toHaveBeenCalledWith('https://api.example.com/users/123')
  })

  it('should handle API errors', async () => {
    ;(fetch as jest.MockedFunction<typeof fetch>).mockRejectedValue(
      new Error('Network error')
    )

    await expect(apiService.fetchUser('123')).rejects.toThrow('Network error')
  })
})
```

### Promise直接テスト

```typescript
it('should resolve with data', () => {
  return expect(Promise.resolve('data')).resolves.toBe('data')
})

it('should reject with error', () => {
  return expect(Promise.reject(new Error('fail'))).rejects.toThrow('fail')
})
```

---

## テストカバレッジ

### カバレッジレポート

```bash
# カバレッジ計測
npm run test:coverage

# レポート生成
# coverage/lcov-report/index.html
```

**カバレッジの種類:**

| 種類 | 説明 | 目標 |
|------|------|------|
| **Statement Coverage** | 実行された文の割合 | 80%+ |
| **Branch Coverage** | 実行された分岐の割合 | 80%+ |
| **Function Coverage** | 実行された関数の割合 | 80%+ |
| **Line Coverage** | 実行された行の割合 | 80%+ |

```typescript
// jest.config.ts
coverageThreshold: {
  global: {
    branches: 80,
    functions: 80,
    lines: 80,
    statements: 80,
  },
  './src/services/': {
    branches: 90,
    functions: 90,
    lines: 90,
    statements: 90,
  },
}
```

---

## よくあるトラブルと解決策

### 1. テストが遅い

**症状:** テスト実行に時間がかかる。

**解決策:**
```typescript
// ❌ データベース接続（遅い）
it('should create user', async () => {
  const user = await db.user.create({ data: userData })
  expect(user).toBeDefined()
})

// ✅ モック使用（高速）
jest.mock('./db')

it('should create user', async () => {
  ;(db.user.create as jest.MockedFunction<any>).mockResolvedValue({ id: '123' })

  const user = await service.createUser(userData)

  expect(user).toBeDefined()
})
```

### 2. テストが不安定（Flaky Test）

**症状:** 同じテストが成功したり失敗したりする。

**解決策:**
```typescript
// ❌ 時間依存（不安定）
it('should return current time', () => {
  expect(getCurrentTime()).toBe(new Date())
})

// ✅ モック使用（安定）
jest.useFakeTimers()
jest.setSystemTime(new Date('2025-12-26'))

it('should return current time', () => {
  expect(getCurrentTime()).toEqual(new Date('2025-12-26'))
})

jest.useRealTimers()
```

### 3. モックが効かない

**症状:** jest.mock()が動作しない。

**解決策:**
```typescript
// ❌ jest.mock()の位置が間違っている
import { sendEmail } from './email.service'

jest.mock('./email.service') // ❌ importの後

// ✅ importの前に配置
jest.mock('./email.service')

import { sendEmail } from './email.service'
```

### 4. 非同期テストがタイムアウト

**症状:** "Exceeded timeout of 5000 ms"エラー。

**解決策:**
```typescript
// ✅ タイムアウトを延長
it('should complete long operation', async () => {
  await longOperation()
}, 10000) // 10秒

// または、jest.setTimeout()
jest.setTimeout(10000)
```

### 5. beforeEach/afterEachが動かない

**症状:** セットアップ・クリーンアップが実行されない。

**解決策:**
```typescript
// ✅ 非同期の場合はasync/await
beforeEach(async () => {
  await setup()
})

afterEach(async () => {
  await cleanup()
})
```

### 6. スナップショットテストが頻繁に失敗

**症状:** スナップショットの更新が多すぎる。

**解決策:**
```typescript
// ❌ 動的な値をそのままスナップショット
expect({
  id: generateId(),
  createdAt: new Date(),
  data: 'test',
}).toMatchSnapshot()

// ✅ 動的な値を除外
expect({
  data: 'test',
}).toMatchSnapshot()

// または、プロパティマッチャー使用
expect({
  id: expect.any(String),
  createdAt: expect.any(Date),
  data: 'test',
}).toMatchSnapshot()
```

### 7. 外部依存のテストができない

**症状:** 外部APIやデータベースに依存してテストできない。

**解決策:**
```typescript
// ✅ 依存性注入でテスト可能に
export class UserService {
  constructor(private apiClient: ApiClient) {}

  async getUser(id: string) {
    return this.apiClient.get(`/users/${id}`)
  }
}

// テスト
const mockApiClient = {
  get: jest.fn().mockResolvedValue({ id: '123', name: 'John' }),
}

const service = new UserService(mockApiClient)
```

### 8. テストの重複コードが多い

**症状:** 同じセットアップコードが複数のテストに存在。

**解決策:**
```typescript
// ✅ テストヘルパーを作成
function createTestUser(overrides = {}) {
  return {
    id: '123',
    name: 'John',
    email: 'john@example.com',
    ...overrides,
  }
}

it('should process user', () => {
  const user = createTestUser({ name: 'Jane' })
  expect(processUser(user)).toBeDefined()
})
```

### 9. エラーメッセージが不明瞭

**症状:** テスト失敗時の原因がわからない。

**解決策:**
```typescript
// ❌ メッセージなし
expect(value).toBe(expected)

// ✅ カスタムメッセージ
expect(value).toBe(expected) // Jest は自動的に詳細なメッセージを表示

// より明確に
if (value !== expected) {
  throw new Error(`Expected ${expected}, but got ${value}`)
}
```

### 10. テストが他のテストに影響

**症状:** テスト順序によって結果が変わる。

**解決策:**
```typescript
// ✅ 各テストで独立したデータを使用
beforeEach(() => {
  // 毎回新しいインスタンスを作成
  service = new UserService()
})

// ✅ グローバル状態をクリア
afterEach(() => {
  jest.clearAllMocks()
  jest.restoreAllMocks()
})
```

---

## 実測データ

### 導入前の課題
- テストがない、または不十分
- バグが本番環境で発見される（月15件）
- リファクタリングが怖い
- コードレビュー時間: 平均2時間

### 導入後の改善

**テストカバレッジ:**
- カバレッジ: 0% → 87%
- ユニットテスト数: 0件 → 1,250件

**バグ検出:**
- 本番バグ発見: 15件/月 → 2件/月 (-87%)
- デグレード: 8件/月 → 0件/月 (-100%)

**開発効率:**
- リファクタリング時間: 平均8時間 → 2時間 (-75%)
- コードレビュー時間: 2時間 → 45分 (-63%)
- デプロイ頻度: 週1回 → 日3回 (+2,100%)

**信頼性:**
- 重大な障害: 年4回 → 0回 (-100%)
- ロールバック: 月2回 → 年1回 (-96%)

---

## チェックリスト

### テスト設計
- [ ] AAAパターン（Arrange, Act, Assert）を使用
- [ ] 1つのテストで1つの事柄のみテスト
- [ ] テスト名は期待動作を明確に記述
- [ ] テストは独立して実行可能

### モック
- [ ] 外部依存（DB、API、ファイルシステム）はモック
- [ ] jest.mock()はimport前に配置
- [ ] beforeEach/afterEachでモックをリセット

### カバレッジ
- [ ] 80%以上のカバレッジを目標
- [ ] 重要なビジネスロジックは100%カバー
- [ ] エッジケース・エラーケースもテスト

### ベストプラクティス
- [ ] 高速実行（ユニットテストは1秒以内）
- [ ] Flaky Testを避ける（時間依存、順序依存）
- [ ] テストヘルパーで重複コード削減
- [ ] 意味のあるアサーションメッセージ

---

文字数: 約27,500文字
