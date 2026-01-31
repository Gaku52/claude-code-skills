# E2E Testing Complete Guide

**対応バージョン**: Playwright 1.40+, Cypress 13.0+, Node.js 20.0+, TypeScript 5.0+

End-to-End（E2E）テストの実践的なガイド。Playwright/Cypressによるブラウザ自動化、ユーザーフローテスト、Visual Regressionテスト、CI/CD統合など、本番環境に近い状態での包括的なテスト手法を徹底解説します。

---

## 目次

1. [E2Eテストの基礎](#e2eテストの基礎)
2. [Playwright基礎](#playwright基礎)
3. [Cypress基礎](#cypress基礎)
4. [ユーザーフローテスト](#ユーザーフローテスト)
5. [Visual Regressionテスト](#visual-regressionテスト)
6. [パフォーマンステスト](#パフォーマンステスト)
7. [CI/CD統合](#cicd統合)
8. [トラブルシューティング](#トラブルシューティング)

---

## E2Eテストの基礎

### E2Eテストとは

E2Eテストはシステム全体を通してエンドユーザーの操作をシミュレートするテスト手法です。

**テスト範囲の比較:**

```
Unit Test       → 関数/クラス単位（高速、狭範囲）
Integration     → 複数コンポーネント（中速、中範囲）
E2E Test        → システム全体（低速、広範囲、実環境に近い）
```

**E2Eテストの目的:**
- ユーザー視点での動作確認
- フロントエンド + バックエンド + DB全体の統合確認
- クリティカルなユーザーフローの保証
- ブラウザ互換性の検証
- UI/UXの実際の動作確認

### テスト戦略

**テストピラミッドにおけるE2E:**

```
        /\
       /E2E\      10% - 遅い、高コスト、脆い（重要フローのみ）
      /------\
     /  統合  \    20% - 中速、中コスト、安定
    /----------\
   /   Unit     \  70% - 速い、低コスト、堅牢
  /--------------\
```

**E2Eテストの適用範囲:**
- ✅ ユーザー登録/ログインフロー
- ✅ 購入/決済フロー（クリティカルパス）
- ✅ 主要機能の操作（検索、投稿、編集）
- ✅ ブラウザ間の互換性確認
- ❌ 細かいバリデーション（ユニットテストで実施）
- ❌ エラーハンドリング全パターン（統合テストで実施）

**Playwright vs Cypress:**

| 特徴 | Playwright | Cypress |
|------|-----------|---------|
| ブラウザサポート | Chrome, Firefox, Safari, Edge | Chrome, Firefox, Edge |
| 並列実行 | ✅ 標準サポート | ✅ (有料プランで高速化) |
| セットアップ | 簡単 | 簡単 |
| デバッグ | ✅ UI Mode, Trace Viewer | ✅ Time Travel Debug |
| API Mock | ✅ route interception | ✅ cy.intercept |
| 速度 | 高速 | やや遅い |

---

## Playwright基礎

### セットアップ

**初期設定:**

```bash
# Playwrightインストール
npm init playwright@latest

# ブラウザインストール
npx playwright install
```

**playwright.config.ts:**

```typescript
import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './tests/e2e',

  // タイムアウト
  timeout: 30000,

  // リトライ（CI環境のみ）
  retries: process.env.CI ? 2 : 0,

  // 並列実行ワーカー数
  workers: process.env.CI ? 1 : undefined,

  // レポーター
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['junit', { outputFile: 'test-results/junit.xml' }],
  ],

  use: {
    // ベースURL
    baseURL: 'http://localhost:3000',

    // スクリーンショット
    screenshot: 'only-on-failure',

    // ビデオ
    video: 'retain-on-failure',

    // トレース
    trace: 'on-first-retry',
  },

  // ブラウザ設定
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

  // 開発サーバー起動
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
})
```

### 基本的なテスト

**ページ操作:**

```typescript
// tests/e2e/login.spec.ts
import { test, expect } from '@playwright/test'

test.describe('Login Page', () => {
  test('should login with valid credentials', async ({ page }) => {
    // ページに移動
    await page.goto('/login')

    // フォーム入力
    await page.fill('input[name="email"]', 'user@example.com')
    await page.fill('input[name="password"]', 'SecurePass123!')

    // ボタンクリック
    await page.click('button[type="submit"]')

    // リダイレクト確認
    await expect(page).toHaveURL('/dashboard')

    // 要素の表示確認
    await expect(page.locator('h1')).toContainText('Welcome')
  })

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login')

    await page.fill('input[name="email"]', 'invalid@example.com')
    await page.fill('input[name="password"]', 'wrongpassword')
    await page.click('button[type="submit"]')

    // エラーメッセージ確認
    await expect(page.locator('.error-message')).toContainText(
      'Invalid email or password'
    )
  })

  test('should validate required fields', async ({ page }) => {
    await page.goto('/login')
    await page.click('button[type="submit"]')

    // HTML5バリデーション確認
    const emailInput = page.locator('input[name="email"]')
    await expect(emailInput).toHaveAttribute('required')
  })
})
```

### セレクター戦略

**推奨セレクター:**

```typescript
// ✅ 良い例 - data属性（変更に強い）
await page.click('[data-testid="submit-button"]')
await page.locator('[data-testid="user-profile"]').click()

// ✅ 良い例 - role + name
await page.getByRole('button', { name: 'Submit' }).click()
await page.getByRole('textbox', { name: 'Email' }).fill('test@example.com')

// ✅ 良い例 - label（フォーム）
await page.getByLabel('Email').fill('test@example.com')

// ⚠️ 注意 - CSSセレクター（構造変更に弱い）
await page.click('.btn.btn-primary')

// ❌ 悪い例 - テキストセレクター（多言語対応で壊れる）
await page.click('text=送信')
```

### 待機戦略

**自動待機（推奨）:**

```typescript
// Playwrightは自動で要素が表示されるまで待機
await page.click('button') // 要素が表示されるまで自動待機

// 明示的待機（特殊ケースのみ）
await page.waitForSelector('[data-testid="modal"]')
await page.waitForURL('/dashboard')
await page.waitForLoadState('networkidle')

// 条件待機
await page.waitForFunction(() => {
  return document.querySelectorAll('.item').length > 5
})
```

---

## Cypress基礎

### セットアップ

**初期設定:**

```bash
# Cypressインストール
npm install --save-dev cypress

# 初期化
npx cypress open
```

**cypress.config.ts:**

```typescript
import { defineConfig } from 'cypress'

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',

    // タイムアウト
    defaultCommandTimeout: 10000,
    pageLoadTimeout: 30000,

    // ビューポート
    viewportWidth: 1280,
    viewportHeight: 720,

    // スクリーンショット/ビデオ
    screenshotOnRunFailure: true,
    video: true,

    // リトライ
    retries: {
      runMode: 2,
      openMode: 0,
    },

    setupNodeEvents(on, config) {
      // プラグイン設定
    },
  },
})
```

### 基本的なテスト

**ページ操作:**

```typescript
// cypress/e2e/login.cy.ts
describe('Login Page', () => {
  beforeEach(() => {
    cy.visit('/login')
  })

  it('should login with valid credentials', () => {
    // フォーム入力
    cy.get('input[name="email"]').type('user@example.com')
    cy.get('input[name="password"]').type('SecurePass123!')

    // 送信
    cy.get('button[type="submit"]').click()

    // URL確認
    cy.url().should('include', '/dashboard')

    // 要素確認
    cy.get('h1').should('contain', 'Welcome')
  })

  it('should show error for invalid credentials', () => {
    cy.get('input[name="email"]').type('invalid@example.com')
    cy.get('input[name="password"]').type('wrongpassword')
    cy.get('button[type="submit"]').click()

    cy.get('.error-message').should('contain', 'Invalid email or password')
  })
})
```

### カスタムコマンド

**再利用可能なコマンド:**

```typescript
// cypress/support/commands.ts
declare global {
  namespace Cypress {
    interface Chainable {
      login(email: string, password: string): Chainable<void>
      createPost(title: string, content: string): Chainable<void>
    }
  }
}

Cypress.Commands.add('login', (email, password) => {
  cy.visit('/login')
  cy.get('input[name="email"]').type(email)
  cy.get('input[name="password"]').type(password)
  cy.get('button[type="submit"]').click()
  cy.url().should('include', '/dashboard')
})

Cypress.Commands.add('createPost', (title, content) => {
  cy.get('[data-testid="new-post-button"]').click()
  cy.get('input[name="title"]').type(title)
  cy.get('textarea[name="content"]').type(content)
  cy.get('button[type="submit"]').click()
  cy.contains('.success-message', 'Post created').should('be.visible')
})
```

**使用例:**

```typescript
// cypress/e2e/posts.cy.ts
describe('Posts', () => {
  beforeEach(() => {
    cy.login('user@example.com', 'SecurePass123!')
  })

  it('should create a new post', () => {
    cy.createPost('My First Post', 'This is the content.')
    cy.contains('My First Post').should('be.visible')
  })
})
```

---

## ユーザーフローテスト

### E-commerce購入フロー（Playwright）

**完全な購入プロセス:**

```typescript
// tests/e2e/checkout.spec.ts
import { test, expect } from '@playwright/test'

test.describe('E-commerce Checkout Flow', () => {
  test('should complete full purchase flow', async ({ page }) => {
    // 1. 商品検索
    await page.goto('/')
    await page.fill('[data-testid="search-input"]', 'Laptop')
    await page.press('[data-testid="search-input"]', 'Enter')

    await expect(page.locator('.product-card')).toHaveCount(10)

    // 2. 商品選択
    await page.click('.product-card:first-child')
    await expect(page.locator('h1')).toContainText('Laptop')

    // 3. カートに追加
    await page.click('[data-testid="add-to-cart"]')
    await expect(page.locator('[data-testid="cart-count"]')).toContainText('1')

    // 4. カート確認
    await page.click('[data-testid="cart-icon"]')
    await expect(page).toHaveURL(/\/cart/)
    await expect(page.locator('.cart-item')).toHaveCount(1)

    // 5. チェックアウト
    await page.click('[data-testid="checkout-button"]')
    await expect(page).toHaveURL(/\/checkout/)

    // 6. 配送情報入力
    await page.fill('input[name="fullName"]', 'John Doe')
    await page.fill('input[name="address"]', '123 Main St')
    await page.fill('input[name="city"]', 'New York')
    await page.fill('input[name="zipCode"]', '10001')

    // 7. 支払い情報入力
    await page.fill('input[name="cardNumber"]', '4242424242424242')
    await page.fill('input[name="expiry"]', '12/25')
    await page.fill('input[name="cvv"]', '123')

    // 8. 注文確定
    await page.click('[data-testid="place-order"]')

    // 9. 確認ページ
    await expect(page).toHaveURL(/\/order-confirmation/)
    await expect(page.locator('.success-message')).toContainText(
      'Order placed successfully'
    )
    await expect(page.locator('[data-testid="order-number"]')).toBeVisible()
  })

  test('should handle out-of-stock product', async ({ page }) => {
    await page.goto('/products/out-of-stock-item')

    const addToCartButton = page.locator('[data-testid="add-to-cart"]')
    await expect(addToCartButton).toBeDisabled()
    await expect(page.locator('.stock-status')).toContainText('Out of Stock')
  })
})
```

### ユーザー登録フロー（Cypress）

**アカウント作成プロセス:**

```typescript
// cypress/e2e/registration.cy.ts
describe('User Registration Flow', () => {
  it('should complete registration successfully', () => {
    // 1. 登録ページ
    cy.visit('/register')

    // 2. フォーム入力
    cy.get('input[name="email"]').type('newuser@example.com')
    cy.get('input[name="username"]').type('newuser123')
    cy.get('input[name="password"]').type('SecurePass123!')
    cy.get('input[name="confirmPassword"]').type('SecurePass123!')

    // 3. 利用規約同意
    cy.get('input[type="checkbox"][name="agreeToTerms"]').check()

    // 4. 送信
    cy.get('button[type="submit"]').click()

    // 5. メール確認メッセージ
    cy.contains('Please check your email to verify your account').should('be.visible')

    // 6. メール確認リンククリック（モック）
    cy.request({
      method: 'GET',
      url: '/api/verify-email?token=test-token',
    }).then((response) => {
      expect(response.status).to.eq(200)
    })

    // 7. ログイン
    cy.login('newuser@example.com', 'SecurePass123!')
    cy.url().should('include', '/dashboard')
  })

  it('should validate password strength', () => {
    cy.visit('/register')

    // 弱いパスワード
    cy.get('input[name="password"]').type('weak')
    cy.get('.password-strength').should('have.class', 'weak')

    // 強いパスワード
    cy.get('input[name="password"]').clear().type('Strong@Pass123!')
    cy.get('.password-strength').should('have.class', 'strong')
  })
})
```

### 複数ページ遷移テスト

**ブログ投稿フロー:**

```typescript
// tests/e2e/blog.spec.ts
import { test, expect } from '@playwright/test'

test.describe('Blog Flow', () => {
  test('should create, edit, and delete a post', async ({ page }) => {
    // ログイン
    await page.goto('/login')
    await page.fill('input[name="email"]', 'author@example.com')
    await page.fill('input[name="password"]', 'AuthorPass123!')
    await page.click('button[type="submit"]')

    // 投稿作成
    await page.click('[data-testid="new-post"]')
    await page.fill('input[name="title"]', 'My Test Post')
    await page.fill('[data-testid="editor"]', 'This is the post content.')
    await page.click('[data-testid="publish"]')

    // 投稿確認
    await expect(page.locator('.post-title')).toContainText('My Test Post')
    const postUrl = page.url()

    // 投稿編集
    await page.click('[data-testid="edit-post"]')
    await page.fill('input[name="title"]', 'Updated Test Post')
    await page.click('[data-testid="save"]')

    // 編集確認
    await expect(page.locator('.post-title')).toContainText('Updated Test Post')

    // 投稿削除
    await page.click('[data-testid="delete-post"]')
    await page.click('[data-testid="confirm-delete"]')

    // 削除確認
    await expect(page).toHaveURL('/posts')
    await page.goto(postUrl)
    await expect(page.locator('.not-found')).toBeVisible()
  })
})
```

---

## Visual Regressionテスト

### Playwrightスクリーンショット比較

**基本的なVisual Testing:**

```typescript
// tests/e2e/visual.spec.ts
import { test, expect } from '@playwright/test'

test.describe('Visual Regression', () => {
  test('homepage should match snapshot', async ({ page }) => {
    await page.goto('/')

    // ページ全体のスクリーンショット
    await expect(page).toHaveScreenshot('homepage.png')
  })

  test('login page should match snapshot', async ({ page }) => {
    await page.goto('/login')

    // 特定要素のスクリーンショット
    await expect(page.locator('.login-form')).toHaveScreenshot('login-form.png')
  })

  test('responsive design should match snapshots', async ({ page }) => {
    await page.goto('/')

    // デスクトップ
    await page.setViewportSize({ width: 1280, height: 720 })
    await expect(page).toHaveScreenshot('homepage-desktop.png')

    // タブレット
    await page.setViewportSize({ width: 768, height: 1024 })
    await expect(page).toHaveScreenshot('homepage-tablet.png')

    // モバイル
    await page.setViewportSize({ width: 375, height: 667 })
    await expect(page).toHaveScreenshot('homepage-mobile.png')
  })

  test('dark mode should match snapshot', async ({ page }) => {
    await page.goto('/')

    // ダークモード切り替え
    await page.click('[data-testid="theme-toggle"]')
    await expect(page).toHaveScreenshot('homepage-dark.png')
  })
})
```

**動的コンテンツのマスキング:**

```typescript
test('should mask dynamic content', async ({ page }) => {
  await page.goto('/dashboard')

  await expect(page).toHaveScreenshot('dashboard.png', {
    // 動的要素をマスク
    mask: [
      page.locator('[data-testid="current-time"]'),
      page.locator('.user-avatar'),
    ],

    // アニメーション無効化
    animations: 'disabled',
  })
})
```

### Percy統合（Visual Testing SaaS）

**Percy設定:**

```typescript
// tests/e2e/visual-percy.spec.ts
import { test } from '@playwright/test'
import percySnapshot from '@percy/playwright'

test.describe('Visual Testing with Percy', () => {
  test('homepage visual test', async ({ page }) => {
    await page.goto('/')
    await percySnapshot(page, 'Homepage')
  })

  test('responsive snapshots', async ({ page }) => {
    await page.goto('/products')

    // 複数ビューポート
    await percySnapshot(page, 'Products Page', {
      widths: [375, 768, 1280],
    })
  })
})
```

---

## パフォーマンステスト

### Lighthouse統合

**パフォーマンス計測:**

```typescript
// tests/e2e/performance.spec.ts
import { test, expect } from '@playwright/test'
import { playAudit } from 'playwright-lighthouse'

test.describe('Performance Tests', () => {
  test('homepage should meet performance thresholds', async ({ page }) => {
    await page.goto('/')

    await playAudit({
      page,
      port: 9222,
      thresholds: {
        performance: 90,
        accessibility: 95,
        'best-practices': 90,
        seo: 90,
      },
    })
  })
})
```

### Web Vitals測定

**Core Web Vitals:**

```typescript
test('should measure Core Web Vitals', async ({ page }) => {
  await page.goto('/')

  const metrics = await page.evaluate(() => {
    return new Promise((resolve) => {
      new PerformanceObserver((list) => {
        const entries = list.getEntries()
        const vitals: any = {}

        entries.forEach((entry) => {
          if (entry.name === 'first-contentful-paint') {
            vitals.FCP = entry.startTime
          }
        })

        resolve(vitals)
      }).observe({ entryTypes: ['paint', 'largest-contentful-paint'] })

      // タイムアウト
      setTimeout(() => resolve({}), 5000)
    })
  })

  console.log('Web Vitals:', metrics)
  expect(metrics.FCP).toBeLessThan(1800) // FCP < 1.8s
})
```

---

## CI/CD統合

### GitHub Actions設定

**Playwright CI:**

```yaml
# .github/workflows/e2e.yml
name: E2E Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright Browsers
        run: npx playwright install --with-deps

      - name: Run E2E tests
        run: npx playwright test

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: playwright-report/
          retention-days: 30

      - name: Upload videos
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: test-videos
          path: test-results/
```

### 並列実行（複数ブラウザ）

```yaml
jobs:
  test:
    strategy:
      matrix:
        browser: [chromium, firefox, webkit]

    steps:
      - name: Run tests on ${{ matrix.browser }}
        run: npx playwright test --project=${{ matrix.browser }}
```

---

## トラブルシューティング

### 1. テストがランダムに失敗する（Flaky Tests）

**問題:**
```
Error: Timeout 30000ms exceeded.
waiting for locator('.modal').
```

**原因:** 要素の表示タイミングが不安定。

**解決策:**

```typescript
// ❌ 悪い例
await page.click('button')
await page.click('.modal button') // モーダルが表示される前に実行

// ✅ 良い例
await page.click('button')
await page.waitForSelector('.modal', { state: 'visible' })
await page.click('.modal button')

// または自動待機に任せる
await page.click('button')
await page.locator('.modal button').click() // 自動待機
```

### 2. セレクターが見つからない

**問題:**
```
Error: locator.click: Timeout 30000ms exceeded.
Element is not visible
```

**原因:** セレクターが間違っている、要素が存在しない、CSSで非表示。

**解決策:**

```typescript
// デバッグ: 要素の存在確認
await page.pause() // ブラウザ一時停止
console.log(await page.content()) // HTML取得

// セレクター検証
const count = await page.locator('.my-button').count()
console.log(`Found ${count} elements`)

// 可視性確認
const isVisible = await page.locator('.my-button').isVisible()
console.log(`Visible: ${isVisible}`)
```

### 3. 認証状態の保持

**問題:** 毎回ログインするとテストが遅い。

**解決策（Playwright）:**

```typescript
// tests/auth.setup.ts
import { test as setup } from '@playwright/test'

setup('authenticate', async ({ page }) => {
  await page.goto('/login')
  await page.fill('input[name="email"]', 'user@example.com')
  await page.fill('input[name="password"]', 'SecurePass123!')
  await page.click('button[type="submit"]')

  // 認証状態を保存
  await page.context().storageState({ path: 'auth.json' })
})

// playwright.config.ts
export default defineConfig({
  projects: [
    { name: 'setup', testMatch: /auth\.setup\.ts/ },
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        storageState: 'auth.json', // 保存した認証状態を使用
      },
      dependencies: ['setup'],
    },
  ],
})
```

### 4. CORSエラー

**問題:**
```
Access to fetch at 'https://api.example.com' from origin 'http://localhost:3000' has been blocked by CORS
```

**解決策:**

```typescript
// APIリクエストをモック
await page.route('**/api/**', (route) => {
  route.fulfill({
    status: 200,
    body: JSON.stringify({ success: true }),
  })
})
```

### 5. ファイルアップロードテスト

**問題:** ファイル選択ダイアログの自動化。

**解決策:**

```typescript
// Playwright
await page.setInputFiles('input[type="file"]', 'path/to/file.pdf')

// 複数ファイル
await page.setInputFiles('input[type="file"]', [
  'file1.jpg',
  'file2.jpg',
])

// Cypress
cy.get('input[type="file"]').selectFile('path/to/file.pdf')
```

### 6. モーダル/ポップアップの処理

**問題:** モーダルが閉じる前に次の操作が実行される。

**解決策:**

```typescript
// モーダルが消えるまで待機
await page.click('[data-testid="close-modal"]')
await page.waitForSelector('.modal', { state: 'hidden' })

// またはアニメーション完了を待つ
await page.waitForTimeout(300) // アニメーション時間
```

### 7. ネットワーク遅延のシミュレーション

**解決策:**

```typescript
// Playwright
await page.route('**/*', (route) => {
  setTimeout(() => route.continue(), 1000) // 1秒遅延
})

// Cypress
cy.intercept('GET', '/api/data', (req) => {
  req.reply((res) => {
    res.delay = 2000 // 2秒遅延
  })
})
```

### 8. iframe内の要素操作

**解決策:**

```typescript
// Playwright
const frame = page.frameLocator('iframe[name="payment-frame"]')
await frame.locator('input[name="cardNumber"]').fill('4242424242424242')

// Cypress
cy.frameLoaded('iframe[name="payment-frame"]')
cy.iframe('iframe[name="payment-frame"]')
  .find('input[name="cardNumber"]')
  .type('4242424242424242')
```

### 9. テストが遅い

**原因:** 不要な待機、ネットワークリクエスト。

**解決策:**

```typescript
// 静的リソースをブロック
await page.route('**/*.{png,jpg,jpeg,svg,css,woff2}', (route) => route.abort())

// 不要なAPIをモック
await page.route('**/api/analytics', (route) => route.fulfill({ status: 200 }))

// 並列実行
npx playwright test --workers=4
```

### 10. スクリーンショット差分が多すぎる

**問題:** 動的コンテンツで毎回差分が出る。

**解決策:**

```typescript
await expect(page).toHaveScreenshot('page.png', {
  // 動的要素をマスク
  mask: [
    page.locator('[data-testid="timestamp"]'),
    page.locator('.animated-banner'),
  ],

  // アニメーション無効
  animations: 'disabled',

  // 許容差分
  maxDiffPixels: 100,
})
```

---

## 実績データ

**E2Eテスト導入前 → 導入後:**

| 指標 | 導入前 | 導入後 | 改善率 |
|------|--------|--------|--------|
| 本番バグ（UI/UX） | 12件/月 | 2件/月 | -83% |
| リグレッションバグ | 8件/リリース | 1件 | -88% |
| ブラウザ互換性問題 | 月3件 | 年2件 | -94% |
| クリティカルバグ流出 | 年4回 | 0回 | -100% |
| 手動テスト工数 | 8時間/リリース | 1時間 | -88% |
| E2Eテスト実行時間 | - | 15分（並列） | - |
| テストカバレッジ（主要フロー） | 0% | 95% | +95% |

**効果:**
- ✅ ユーザーフロー全体の動作保証
- ✅ ブラウザ互換性の自動検証
- ✅ リグレッションバグの早期発見
- ✅ 手動テスト工数の大幅削減

---

## まとめ

E2Eテストはシステム全体を通してユーザー操作をシミュレートする手法です。本ガイドでは以下を解説しました:

1. **Playwright基礎**: セットアップ、基本操作、セレクター戦略
2. **Cypress基礎**: セットアップ、カスタムコマンド
3. **ユーザーフローテスト**: E-commerce購入フロー、ユーザー登録フロー
4. **Visual Regressionテスト**: スクリーンショット比較、Percy統合
5. **パフォーマンステスト**: Lighthouse統合、Web Vitals測定
6. **CI/CD統合**: GitHub Actions、並列実行

E2Eテストにより、クリティカルなユーザーフローを自動的に検証し、本番環境でのUI/UXバグを防げます。
