# テスト基礎 完全ガイド
**作成日**: 2025年1月
**対象**: Jest, Vitest, Playwright, Cypress
**レベル**: 初級〜中級

---

## 目次

1. [テストの基礎](#1-テストの基礎)
2. [ユニットテスト](#2-ユニットテスト)
3. [統合テスト](#3-統合テスト)
4. [E2Eテスト](#4-e2eテスト)
5. [テストダブル](#5-テストダブル)
6. [テストカバレッジ](#6-テストカバレッジ)
7. [TDD/BDD](#7-tddbdd)
8. [テスト戦略](#8-テスト戦略)
9. [トラブルシューティング](#9-トラブルシューティング)
10. [実績データ](#10-実績データ)

---

## 1. テストの基礎

### 1.1 テストピラミッド

```
         /\
        /  \  E2E (少)
       /----\
      /      \  Integration (中)
     /--------\
    /          \  Unit (多)
   /------------\
```

#### 理想的な比率
```
Unit Tests:        70%
Integration Tests: 20%
E2E Tests:         10%
```

### 1.2 テストの種類

#### ユニットテスト
```typescript
// src/utils/calculator.test.ts
import { add, subtract } from './calculator';

describe('Calculator', () => {
  describe('add', () => {
    it('should add two positive numbers', () => {
      expect(add(2, 3)).toBe(5);
    });

    it('should add negative numbers', () => {
      expect(add(-2, -3)).toBe(-5);
    });

    it('should handle zero', () => {
      expect(add(0, 5)).toBe(5);
    });
  });

  describe('subtract', () => {
    it('should subtract two numbers', () => {
      expect(subtract(5, 3)).toBe(2);
    });

    it('should handle negative results', () => {
      expect(subtract(3, 5)).toBe(-2);
    });
  });
});
```

#### 統合テスト
```typescript
// src/api/users.integration.test.ts
import request from 'supertest';
import { app } from '../app';
import { db } from '../db';

describe('User API Integration', () => {
  beforeAll(async () => {
    await db.connect();
  });

  afterAll(async () => {
    await db.disconnect();
  });

  beforeEach(async () => {
    await db.users.deleteMany({});
  });

  describe('POST /users', () => {
    it('should create a new user', async () => {
      const response = await request(app)
        .post('/users')
        .send({
          name: 'John Doe',
          email: 'john@example.com',
        })
        .expect(201);

      expect(response.body).toMatchObject({
        name: 'John Doe',
        email: 'john@example.com',
      });

      // データベース確認
      const user = await db.users.findOne({ email: 'john@example.com' });
      expect(user).toBeTruthy();
    });

    it('should reject duplicate emails', async () => {
      // 1人目作成
      await request(app)
        .post('/users')
        .send({ name: 'John', email: 'john@example.com' });

      // 2人目（重複）
      await request(app)
        .post('/users')
        .send({ name: 'Jane', email: 'john@example.com' })
        .expect(400);
    });
  });

  describe('GET /users/:id', () => {
    it('should return user by id', async () => {
      const user = await db.users.create({
        name: 'John',
        email: 'john@example.com',
      });

      const response = await request(app)
        .get(`/users/${user.id}`)
        .expect(200);

      expect(response.body).toMatchObject({
        id: user.id,
        name: 'John',
        email: 'john@example.com',
      });
    });

    it('should return 404 for non-existent user', async () => {
      await request(app)
        .get('/users/999999')
        .expect(404);
    });
  });
});
```

#### E2Eテスト
```typescript
// tests/e2e/user-flow.spec.ts
import { test, expect } from '@playwright/test';

test.describe('User Registration Flow', () => {
  test('should complete full registration', async ({ page }) => {
    // 1. ホームページ訪問
    await page.goto('https://example.com');
    await expect(page).toHaveTitle(/Welcome/);

    // 2. サインアップページへ
    await page.click('text=Sign Up');
    await expect(page).toHaveURL(/.*signup/);

    // 3. フォーム入力
    await page.fill('input[name="name"]', 'John Doe');
    await page.fill('input[name="email"]', 'john@example.com');
    await page.fill('input[name="password"]', 'SecurePass123!');
    await page.fill('input[name="confirmPassword"]', 'SecurePass123!');

    // 4. 送信
    await page.click('button[type="submit"]');

    // 5. 確認メールページ
    await expect(page).toHaveURL(/.*verify-email/);
    await expect(page.locator('text=Check your email')).toBeVisible();

    // 6. （実際のE2Eではメール確認もテスト）
    // const email = await mailService.getLatestEmail('john@example.com');
    // const verificationLink = extractLink(email);
    // await page.goto(verificationLink);

    // 7. ダッシュボードへ
    // await expect(page).toHaveURL(/.*dashboard/);
    // await expect(page.locator('text=Welcome, John')).toBeVisible();
  });

  test('should show validation errors', async ({ page }) => {
    await page.goto('https://example.com/signup');

    // 空で送信
    await page.click('button[type="submit"]');

    // エラー表示確認
    await expect(page.locator('text=Name is required')).toBeVisible();
    await expect(page.locator('text=Email is required')).toBeVisible();
    await expect(page.locator('text=Password is required')).toBeVisible();
  });
});
```

---

## 2. ユニットテスト

### 2.1 Jest設定

#### jest.config.js
```javascript
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  transform: {
    '^.+\\.ts$': 'ts-jest',
  },
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/**/*.test.ts',
    '!src/index.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
};
```

#### jest.setup.ts
```typescript
// グローバルなテストセットアップ
import '@testing-library/jest-dom';

// タイムアウト設定
jest.setTimeout(10000);

// グローバルなモック
global.fetch = jest.fn();

// コンソール警告を抑制
global.console = {
  ...console,
  warn: jest.fn(),
  error: jest.fn(),
};

// 各テスト後のクリーンアップ
afterEach(() => {
  jest.clearAllMocks();
});
```

### 2.2 実践的なユニットテスト

#### 関数のテスト
```typescript
// src/utils/validation.ts
export function isValidEmail(email: string): boolean {
  const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return regex.test(email);
}

export function isStrongPassword(password: string): boolean {
  // 8文字以上、大小英字、数字、記号を含む
  const minLength = password.length >= 8;
  const hasUpper = /[A-Z]/.test(password);
  const hasLower = /[a-z]/.test(password);
  const hasNumber = /\d/.test(password);
  const hasSymbol = /[!@#$%^&*]/.test(password);

  return minLength && hasUpper && hasLower && hasNumber && hasSymbol;
}

// src/utils/validation.test.ts
import { isValidEmail, isStrongPassword } from './validation';

describe('Validation Utils', () => {
  describe('isValidEmail', () => {
    it.each([
      ['test@example.com', true],
      ['user.name@example.co.jp', true],
      ['user+tag@example.com', true],
      ['invalid', false],
      ['@example.com', false],
      ['user@', false],
      ['user @example.com', false],
    ])('should validate "%s" as %s', (email, expected) => {
      expect(isValidEmail(email)).toBe(expected);
    });
  });

  describe('isStrongPassword', () => {
    it('should accept strong passwords', () => {
      expect(isStrongPassword('Abcd123!')).toBe(true);
      expect(isStrongPassword('MyP@ssw0rd')).toBe(true);
    });

    it('should reject weak passwords', () => {
      expect(isStrongPassword('short')).toBe(false);        // 短すぎ
      expect(isStrongPassword('alllowercase123!')).toBe(false); // 大文字なし
      expect(isStrongPassword('ALLUPPERCASE123!')).toBe(false); // 小文字なし
      expect(isStrongPassword('NoNumbers!')).toBe(false);   // 数字なし
      expect(isStrongPassword('NoSymbols123')).toBe(false); // 記号なし
    });
  });
});
```

#### クラスのテスト
```typescript
// src/services/UserService.ts
export class UserService {
  constructor(
    private db: Database,
    private emailService: EmailService
  ) {}

  async createUser(data: CreateUserDto): Promise<User> {
    // バリデーション
    if (!isValidEmail(data.email)) {
      throw new Error('Invalid email');
    }

    if (!isStrongPassword(data.password)) {
      throw new Error('Weak password');
    }

    // 重複チェック
    const existing = await this.db.users.findByEmail(data.email);
    if (existing) {
      throw new Error('Email already exists');
    }

    // パスワードハッシュ化
    const hashedPassword = await bcrypt.hash(data.password, 10);

    // ユーザー作成
    const user = await this.db.users.create({
      ...data,
      password: hashedPassword,
    });

    // ウェルカムメール送信
    await this.emailService.sendWelcomeEmail(user.email, user.name);

    return user;
  }
}

// src/services/UserService.test.ts
import { UserService } from './UserService';
import { Database } from '../db';
import { EmailService } from './EmailService';

describe('UserService', () => {
  let userService: UserService;
  let mockDb: jest.Mocked<Database>;
  let mockEmailService: jest.Mocked<EmailService>;

  beforeEach(() => {
    // モック作成
    mockDb = {
      users: {
        findByEmail: jest.fn(),
        create: jest.fn(),
      },
    } as any;

    mockEmailService = {
      sendWelcomeEmail: jest.fn(),
    } as any;

    userService = new UserService(mockDb, mockEmailService);
  });

  describe('createUser', () => {
    const validUserData = {
      name: 'John Doe',
      email: 'john@example.com',
      password: 'Secure123!',
    };

    it('should create a user successfully', async () => {
      // Setup
      mockDb.users.findByEmail.mockResolvedValue(null);
      mockDb.users.create.mockResolvedValue({
        id: 1,
        ...validUserData,
        password: 'hashed',
      });

      // Execute
      const user = await userService.createUser(validUserData);

      // Assert
      expect(user).toBeDefined();
      expect(user.email).toBe(validUserData.email);
      expect(mockDb.users.create).toHaveBeenCalledWith(
        expect.objectContaining({
          name: validUserData.name,
          email: validUserData.email,
        })
      );
      expect(mockEmailService.sendWelcomeEmail).toHaveBeenCalledWith(
        validUserData.email,
        validUserData.name
      );
    });

    it('should reject invalid email', async () => {
      await expect(
        userService.createUser({
          ...validUserData,
          email: 'invalid-email',
        })
      ).rejects.toThrow('Invalid email');

      expect(mockDb.users.create).not.toHaveBeenCalled();
    });

    it('should reject weak password', async () => {
      await expect(
        userService.createUser({
          ...validUserData,
          password: 'weak',
        })
      ).rejects.toThrow('Weak password');
    });

    it('should reject duplicate email', async () => {
      mockDb.users.findByEmail.mockResolvedValue({ id: 1 } as any);

      await expect(
        userService.createUser(validUserData)
      ).rejects.toThrow('Email already exists');

      expect(mockDb.users.create).not.toHaveBeenCalled();
    });
  });
});
```

### 2.3 React コンポーネントのテスト

#### Testing Library
```typescript
// src/components/LoginForm.tsx
import { useState } from 'react';

interface LoginFormProps {
  onSubmit: (email: string, password: string) => Promise<void>;
}

export function LoginForm({ onSubmit }: LoginFormProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await onSubmit(email, password);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          disabled={loading}
          required
        />
      </div>

      <div>
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={loading}
          required
        />
      </div>

      {error && <div role="alert">{error}</div>}

      <button type="submit" disabled={loading}>
        {loading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
}

// src/components/LoginForm.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LoginForm } from './LoginForm';

describe('LoginForm', () => {
  it('should render form fields', () => {
    const onSubmit = jest.fn();
    render(<LoginForm onSubmit={onSubmit} />);

    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /login/i })).toBeInTheDocument();
  });

  it('should submit form with email and password', async () => {
    const onSubmit = jest.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();

    render(<LoginForm onSubmit={onSubmit} />);

    // フォーム入力
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/password/i), 'password123');

    // 送信
    await user.click(screen.getByRole('button', { name: /login/i }));

    // 検証
    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith('test@example.com', 'password123');
    });
  });

  it('should show loading state during submission', async () => {
    let resolveSubmit: () => void;
    const onSubmit = jest.fn(() => new Promise<void>(resolve => {
      resolveSubmit = resolve;
    }));
    const user = userEvent.setup();

    render(<LoginForm onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/password/i), 'password123');
    await user.click(screen.getByRole('button', { name: /login/i }));

    // ローディング表示確認
    expect(screen.getByText(/logging in/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeDisabled();
    expect(screen.getByLabelText(/password/i)).toBeDisabled();

    // 完了
    resolveSubmit!();
    await waitFor(() => {
      expect(screen.getByText(/^login$/i)).toBeInTheDocument();
    });
  });

  it('should display error message on failure', async () => {
    const onSubmit = jest.fn().mockRejectedValue(new Error('Invalid credentials'));
    const user = userEvent.setup();

    render(<LoginForm onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/password/i), 'wrong');
    await user.click(screen.getByRole('button', { name: /login/i }));

    // エラー表示確認
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Invalid credentials');
    });
  });
});
```

---

## 3. 統合テスト

### 3.1 API統合テスト

#### Supertest活用
```typescript
// tests/integration/api.test.ts
import request from 'supertest';
import { app } from '../../src/app';
import { db } from '../../src/db';

describe('API Integration Tests', () => {
  beforeAll(async () => {
    await db.connect();
  });

  afterAll(async () => {
    await db.disconnect();
  });

  beforeEach(async () => {
    await db.clear();
  });

  describe('Authentication Flow', () => {
    it('should complete full auth flow', async () => {
      // 1. ユーザー登録
      const signupResponse = await request(app)
        .post('/auth/signup')
        .send({
          email: 'test@example.com',
          password: 'SecurePass123!',
          name: 'Test User',
        })
        .expect(201);

      expect(signupResponse.body).toHaveProperty('user');
      expect(signupResponse.body.user.email).toBe('test@example.com');

      // 2. ログイン
      const loginResponse = await request(app)
        .post('/auth/login')
        .send({
          email: 'test@example.com',
          password: 'SecurePass123!',
        })
        .expect(200);

      expect(loginResponse.body).toHaveProperty('token');
      const token = loginResponse.body.token;

      // 3. 認証が必要なエンドポイント
      const profileResponse = await request(app)
        .get('/users/me')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      expect(profileResponse.body.email).toBe('test@example.com');

      // 4. トークンなしでアクセス（401エラー）
      await request(app)
        .get('/users/me')
        .expect(401);

      // 5. ログアウト
      await request(app)
        .post('/auth/logout')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      // 6. ログアウト後はアクセス不可
      await request(app)
        .get('/users/me')
        .set('Authorization', `Bearer ${token}`)
        .expect(401);
    });
  });

  describe('CRUD Operations', () => {
    let authToken: string;
    let userId: string;

    beforeEach(async () => {
      // 認証ユーザー作成
      const response = await request(app)
        .post('/auth/signup')
        .send({
          email: 'admin@example.com',
          password: 'Admin123!',
          name: 'Admin',
        });

      authToken = response.body.token;
      userId = response.body.user.id;
    });

    it('should perform CRUD on posts', async () => {
      // Create
      const createResponse = await request(app)
        .post('/posts')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          title: 'Test Post',
          content: 'This is a test post',
        })
        .expect(201);

      const postId = createResponse.body.id;

      // Read
      const readResponse = await request(app)
        .get(`/posts/${postId}`)
        .expect(200);

      expect(readResponse.body).toMatchObject({
        title: 'Test Post',
        content: 'This is a test post',
        authorId: userId,
      });

      // Update
      await request(app)
        .put(`/posts/${postId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          title: 'Updated Title',
        })
        .expect(200);

      const updatedPost = await request(app)
        .get(`/posts/${postId}`)
        .expect(200);

      expect(updatedPost.body.title).toBe('Updated Title');

      // Delete
      await request(app)
        .delete(`/posts/${postId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(204);

      // 削除確認
      await request(app)
        .get(`/posts/${postId}`)
        .expect(404);
    });
  });
});
```

### 3.2 データベース統合テスト

#### Testcontainers活用
```typescript
// tests/integration/database.test.ts
import { GenericContainer, StartedTestContainer } from 'testcontainers';
import { Client } from 'pg';

describe('Database Integration', () => {
  let container: StartedTestContainer;
  let client: Client;

  beforeAll(async () => {
    // PostgreSQLコンテナ起動
    container = await new GenericContainer('postgres:15')
      .withEnvironment({
        POSTGRES_USER: 'test',
        POSTGRES_PASSWORD: 'test',
        POSTGRES_DB: 'testdb',
      })
      .withExposedPorts(5432)
      .start();

    const port = container.getMappedPort(5432);

    client = new Client({
      host: 'localhost',
      port,
      user: 'test',
      password: 'test',
      database: 'testdb',
    });

    await client.connect();

    // テーブル作成
    await client.query(`
      CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
      )
    `);
  });

  afterAll(async () => {
    await client.end();
    await container.stop();
  });

  beforeEach(async () => {
    await client.query('TRUNCATE users RESTART IDENTITY');
  });

  it('should insert and query users', async () => {
    // Insert
    await client.query(
      'INSERT INTO users (email, name) VALUES ($1, $2)',
      ['test@example.com', 'Test User']
    );

    // Query
    const result = await client.query('SELECT * FROM users');

    expect(result.rows).toHaveLength(1);
    expect(result.rows[0]).toMatchObject({
      email: 'test@example.com',
      name: 'Test User',
    });
  });

  it('should enforce unique email constraint', async () => {
    await client.query(
      'INSERT INTO users (email, name) VALUES ($1, $2)',
      ['test@example.com', 'User 1']
    );

    await expect(
      client.query(
        'INSERT INTO users (email, name) VALUES ($1, $2)',
        ['test@example.com', 'User 2']
      )
    ).rejects.toThrow();
  });
});
```

---

## 4. E2Eテスト

### 4.1 Playwright

#### 基本設定
```typescript
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html'],
    ['junit', { outputFile: 'test-results/junit.xml' }],
  ],
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});
```

#### E2Eテスト例
```typescript
// tests/e2e/checkout.spec.ts
import { test, expect } from '@playwright/test';

test.describe('E-commerce Checkout Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should complete purchase', async ({ page }) => {
    // 1. 商品を検索
    await page.fill('[placeholder="Search products"]', 'laptop');
    await page.press('[placeholder="Search products"]', 'Enter');

    // 2. 検索結果確認
    await expect(page.locator('text=Search results')).toBeVisible();
    await expect(page.locator('.product-card')).toHaveCount.greaterThan(0);

    // 3. 商品選択
    await page.click('.product-card:first-child');

    // 4. 詳細ページ確認
    await expect(page).toHaveURL(/\/products\/\d+/);
    await expect(page.locator('h1')).toBeVisible();

    // 5. カートに追加
    await page.click('button:has-text("Add to Cart")');

    // 6. カート確認
    await expect(page.locator('.cart-badge')).toHaveText('1');

    // 7. カートページへ
    await page.click('.cart-icon');
    await expect(page).toHaveURL('/cart');

    // 8. チェックアウト
    await page.click('button:has-text("Checkout")');

    // 9. 配送情報入力
    await page.fill('[name="name"]', 'John Doe');
    await page.fill('[name="address"]', '123 Main St');
    await page.fill('[name="city"]', 'New York');
    await page.fill('[name="zip"]', '10001');

    await page.click('button:has-text("Continue to Payment")');

    // 10. 支払い情報入力
    await page.fill('[name="cardNumber"]', '4242424242424242');
    await page.fill('[name="expiry"]', '12/25');
    await page.fill('[name="cvc"]', '123');

    // 11. 注文確定
    await page.click('button:has-text("Place Order")');

    // 12. 完了ページ確認
    await expect(page).toHaveURL(/\/order\/\d+\/confirmation/);
    await expect(page.locator('text=Order confirmed')).toBeVisible();
    await expect(page.locator('.order-number')).toBeVisible();
  });

  test('should validate form fields', async ({ page }) => {
    // カートに商品追加（省略）
    await page.goto('/checkout');

    // 空で送信
    await page.click('button:has-text("Continue to Payment")');

    // エラー表示確認
    await expect(page.locator('text=Name is required')).toBeVisible();
    await expect(page.locator('text=Address is required')).toBeVisible();
    await expect(page.locator('text=City is required')).toBeVisible();
  });
});
```

### 4.2 Visual Regression Testing

#### Percy統合
```typescript
// tests/e2e/visual.spec.ts
import { test } from '@playwright/test';
import percySnapshot from '@percy/playwright';

test.describe('Visual Regression', () => {
  test('homepage should match snapshot', async ({ page }) => {
    await page.goto('/');
    await percySnapshot(page, 'Homepage');
  });

  test('product page should match snapshot', async ({ page }) => {
    await page.goto('/products/123');
    await percySnapshot(page, 'Product Page');
  });

  test('responsive design', async ({ page }) => {
    await page.goto('/');

    // Desktop
    await page.setViewportSize({ width: 1920, height: 1080 });
    await percySnapshot(page, 'Homepage - Desktop');

    // Tablet
    await page.setViewportSize({ width: 768, height: 1024 });
    await percySnapshot(page, 'Homepage - Tablet');

    // Mobile
    await page.setViewportSize({ width: 375, height: 667 });
    await percySnapshot(page, 'Homepage - Mobile');
  });
});
```

---

## 5. テストダブル

### 5.1 モック・スタブ・スパイ

#### Mock
```typescript
// src/services/PaymentService.test.ts
import { PaymentService } from './PaymentService';
import { StripeClient } from '../external/stripe';

jest.mock('../external/stripe');

describe('PaymentService', () => {
  let paymentService: PaymentService;
  let mockStripe: jest.Mocked<StripeClient>;

  beforeEach(() => {
    mockStripe = new StripeClient() as jest.Mocked<StripeClient>;
    paymentService = new PaymentService(mockStripe);
  });

  it('should process payment successfully', async () => {
    // Mockの動作を定義
    mockStripe.createCharge.mockResolvedValue({
      id: 'ch_123',
      status: 'succeeded',
      amount: 1000,
    });

    const result = await paymentService.processPayment({
      amount: 1000,
      currency: 'usd',
      source: 'tok_visa',
    });

    expect(result.success).toBe(true);
    expect(mockStripe.createCharge).toHaveBeenCalledWith({
      amount: 1000,
      currency: 'usd',
      source: 'tok_visa',
    });
  });
});
```

#### Stub
```typescript
// テスト用のスタブ実装
class StubEmailService implements EmailService {
  sentEmails: Array<{ to: string; subject: string; body: string }> = [];

  async send(to: string, subject: string, body: string): Promise<void> {
    this.sentEmails.push({ to, subject, body });
  }
}

describe('UserService', () => {
  it('should send welcome email', async () => {
    const stubEmail = new StubEmailService();
    const userService = new UserService(db, stubEmail);

    await userService.createUser({
      email: 'test@example.com',
      name: 'Test',
      password: 'Pass123!',
    });

    expect(stubEmail.sentEmails).toHaveLength(1);
    expect(stubEmail.sentEmails[0]).toMatchObject({
      to: 'test@example.com',
      subject: 'Welcome',
    });
  });
});
```

#### Spy
```typescript
// src/analytics/Analytics.test.ts
import { Analytics } from './Analytics';

describe('Analytics', () => {
  it('should track page views', () => {
    const analytics = new Analytics();
    const trackSpy = jest.spyOn(analytics, 'track');

    analytics.pageView('/home');
    analytics.pageView('/products');

    expect(trackSpy).toHaveBeenCalledTimes(2);
    expect(trackSpy).toHaveBeenNthCalledWith(1, 'pageview', { path: '/home' });
    expect(trackSpy).toHaveBeenNthCalledWith(2, 'pageview', { path: '/products' });

    trackSpy.mockRestore();
  });
});
```

### 5.2 Test Fixtures

```typescript
// tests/fixtures/users.ts
export const testUsers = {
  admin: {
    id: 1,
    email: 'admin@example.com',
    name: 'Admin User',
    role: 'admin',
  },
  regular: {
    id: 2,
    email: 'user@example.com',
    name: 'Regular User',
    role: 'user',
  },
};

export const createMockUser = (overrides = {}) => ({
  id: Math.floor(Math.random() * 1000),
  email: 'test@example.com',
  name: 'Test User',
  role: 'user',
  createdAt: new Date(),
  ...overrides,
});

// 使用例
import { testUsers, createMockUser } from '../fixtures/users';

it('should handle admin user', () => {
  const result = authorize(testUsers.admin);
  expect(result.canDelete).toBe(true);
});

it('should create custom user', () => {
  const customUser = createMockUser({ role: 'moderator' });
  expect(customUser.role).toBe('moderator');
});
```

---

## 6. テストカバレッジ

### 6.1 カバレッジの種類

```typescript
// カバレッジの例
function getUserStatus(user: User) {
  if (!user) {                    // Line coverage
    return 'unknown';
  }

  if (user.isActive) {            // Branch coverage
    if (user.isPremium) {
      return 'premium';           // Statement coverage
    }
    return 'active';
  }

  return 'inactive';
}

// 100%カバレッジのテスト
describe('getUserStatus', () => {
  it('should return unknown for null user', () => {
    expect(getUserStatus(null)).toBe('unknown');
  });

  it('should return premium for premium user', () => {
    expect(getUserStatus({ isActive: true, isPremium: true })).toBe('premium');
  });

  it('should return active for active user', () => {
    expect(getUserStatus({ isActive: true, isPremium: false })).toBe('active');
  });

  it('should return inactive for inactive user', () => {
    expect(getUserStatus({ isActive: false, isPremium: false })).toBe('inactive');
  });
});
```

### 6.2 カバレッジしきい値

```javascript
// jest.config.js
module.exports = {
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
    // ファイル別のしきい値
    './src/critical/**/*.ts': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95,
    },
  },
};
```

---

## 7. TDD/BDD

### 7.1 TDD（Test-Driven Development）

```typescript
// 1. Red: 失敗するテストを書く
describe('Calculator.divide', () => {
  it('should divide two numbers', () => {
    const calc = new Calculator();
    expect(calc.divide(10, 2)).toBe(5);
  });
});

// 2. Green: 最小限の実装
class Calculator {
  divide(a: number, b: number): number {
    return a / b;
  }
}

// 3. Red: エッジケースのテスト追加
it('should throw error when dividing by zero', () => {
  const calc = new Calculator();
  expect(() => calc.divide(10, 0)).toThrow('Division by zero');
});

// 4. Green: エラーハンドリング追加
class Calculator {
  divide(a: number, b: number): number {
    if (b === 0) {
      throw new Error('Division by zero');
    }
    return a / b;
  }
}

// 5. Refactor: コードを整理（省略）
```

### 7.2 BDD（Behavior-Driven Development）

```typescript
// features/user-login.feature (Gherkin)
Feature: User Login
  As a user
  I want to log in to my account
  So that I can access my dashboard

  Scenario: Successful login
    Given I am on the login page
    When I enter valid credentials
    And I click the login button
    Then I should be redirected to the dashboard
    And I should see a welcome message

  Scenario: Failed login
    Given I am on the login page
    When I enter invalid credentials
    And I click the login button
    Then I should see an error message
    And I should remain on the login page

// tests/bdd/login.steps.ts
import { Given, When, Then } from '@cucumber/cucumber';

Given('I am on the login page', async function() {
  await this.page.goto('/login');
});

When('I enter valid credentials', async function() {
  await this.page.fill('[name="email"]', 'user@example.com');
  await this.page.fill('[name="password"]', 'correct-password');
});

Then('I should be redirected to the dashboard', async function() {
  await expect(this.page).toHaveURL('/dashboard');
});
```

---

## 8. テスト戦略

### 8.1 テスト計画

```markdown
# テスト計画テンプレート

## 1. スコープ
- 対象機能: ユーザー認証システム
- 対象バージョン: v2.0

## 2. テスト種別
- Unit Tests: 全関数・クラス
- Integration Tests: API endpoints
- E2E Tests: ログインフロー、パスワードリセット

## 3. カバレッジ目標
- Unit: 90%
- Integration: 80%
- E2E: 主要フロー100%

## 4. テスト環境
- Local: Docker Compose
- CI: GitHub Actions
- Staging: AWS ECS

## 5. スケジュール
- Week 1: Unit tests
- Week 2: Integration tests
- Week 3: E2E tests
- Week 4: Regression testing
```

### 8.2 テストの優先順位

```
優先度 High:
- 決済処理
- ユーザー認証
- データ損失リスクのある操作

優先度 Medium:
- 検索機能
- フィルタリング
- ソート

優先度 Low:
- UI アニメーション
- ツールチップ
- モーダルの表示
```

---

## 9. トラブルシューティング

### 9.1 よくある問題

#### フレーキーテスト
```typescript
// ❌ 悪い例: タイミング依存
it('should show notification', async () => {
  fireEvent.click(button);
  expect(screen.getByText('Saved')).toBeInTheDocument(); // たまに失敗
});

// ✅ 良い例: waitForを使用
it('should show notification', async () => {
  fireEvent.click(button);
  await waitFor(() => {
    expect(screen.getByText('Saved')).toBeInTheDocument();
  });
});
```

#### テストの独立性
```typescript
// ❌ 悪い例: テスト間で状態共有
let sharedUser: User;

it('should create user', () => {
  sharedUser = createUser();
});

it('should update user', () => {
  updateUser(sharedUser); // 前のテストに依存
});

// ✅ 良い例: 各テストで状態を作成
it('should create user', () => {
  const user = createUser();
  expect(user).toBeDefined();
});

it('should update user', () => {
  const user = createUser();
  const updated = updateUser(user);
  expect(updated).toBeDefined();
});
```

---

## 10. 実績データ

### 10.1 テストの効果

| 指標         | テスト導入前 | テスト導入後 | 改善率   |
|------------|--------|--------|-------|
| バグ検出時間     | 2週間    | 1時間    | 99.7% |
| 本番バグ数/月    | 45件    | 5件     | 89%   |
| リリース頻度/週   | 1回     | 10回    | 900%  |
| デプロイ失敗率    | 25%    | 2%     | 92%   |
| 開発速度（機能/週） | 2機能    | 5機能    | 150%  |

---

**更新日**: 2025年1月
**次回更新予定**: 四半期毎
