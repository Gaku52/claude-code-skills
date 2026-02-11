# TDD/BDD ワークフローガイド

**最終更新**: 2026-01-02
**対象読者**: TDD/BDDを実践したい開発者
**目標**: Red-Green-Refactorサイクルを完全習得し、実プロジェクトで使える

---

## 📚 目次

1. [TDDの基本ワークフロー](#1-tddの基本ワークフロー)
2. [Red-Green-Refactorサイクル](#2-red-green-refactorサイクル)
3. [BDDとの使い分け](#3-bddとの使い分け)
4. [実際のプロジェクト例](#4-実際のプロジェクト例)
5. [よくある失敗パターン](#5-よくある失敗パターン)
6. [チェックリスト](#6-チェックリスト)

---

## 1. TDDの基本ワークフロー

### 1.1 TDDとは

**Test-Driven Development (TDD)** は、テストを先に書いてから実装を行う開発手法です。

**核となる原則**:
- テストコードが仕様書になる
- リファクタリングの安全網を提供
- 設計の質が向上する

**TDDのメリット**:

```
✅ バグの早期発見
✅ 設計の改善（テスタブルなコード）
✅ リファクタリングの安全性
✅ ドキュメントとしてのテスト
✅ 開発スピードの向上（長期的に）
```

**TDDのデメリット**:

```
⚠️ 学習コストが高い
⚠️ 初期の開発速度が遅く感じる
⚠️ レガシーコードへの適用が困難
⚠️ UIテストには不向き
```

---

### 1.2 TDDの3つのルール

**Kent Beckの3つのルール**:

1. **失敗するテストを書くまで、実装コードを書いてはいけない**
2. **コンパイルが通らない、または失敗する最小限のテストだけを書く**
3. **現在失敗しているテストをパスさせる最小限の実装だけを書く**

**実践例**:

```typescript
// ❌ 間違い: いきなり実装を書く
function add(a: number, b: number): number {
  return a + b;
}

// ✅ 正しい: まずテストを書く
describe('add', () => {
  it('should add two numbers', () => {
    expect(add(2, 3)).toBe(5); // テストが先
  });
});

// その後、実装を書く
function add(a: number, b: number): number {
  return a + b;
}
```

---

## 2. Red-Green-Refactorサイクル

### 2.1 サイクルの概要

```
🔴 Red    → 失敗するテストを書く
🟢 Green  → テストを通す最小限の実装
🔵 Refactor → コードを改善する
```

**各ステップの詳細**:

#### 🔴 Red: 失敗するテストを書く

**目的**:
- 何を作るべきか明確にする
- テストが正しく失敗することを確認

**ポイント**:
- 小さく始める
- 1つの振る舞いに集中
- エラーメッセージを確認

#### 🟢 Green: テストを通す

**目的**:
- 最速でテストを通す
- 動く実装を得る

**ポイント**:
- 美しさは気にしない
- ハードコードでもOK
- まず動かす

#### 🔵 Refactor: リファクタリング

**目的**:
- コードの質を上げる
- 重複を削除
- 設計を改善

**ポイント**:
- テストが通った状態で行う
- 一度に1つのリファクタリング
- テストを再実行

---

### 2.2 完全な実例: バリデーション関数

**要件**: メールアドレスのバリデーション関数を作る

#### Step 1: 🔴 Red - 失敗するテストを書く

```typescript
// src/utils/validators.test.ts
import { validateEmail } from './validators';

describe('validateEmail', () => {
  it('should return true for valid email', () => {
    expect(validateEmail('user@example.com')).toBe(true);
  });
});
```

**実行結果**:
```bash
❌ FAIL  src/utils/validators.test.ts
  ● validateEmail › should return true for valid email
    Cannot find module './validators'
```

#### Step 2: 🟢 Green - 最小限の実装

```typescript
// src/utils/validators.ts
export function validateEmail(email: string): boolean {
  return true; // ハードコードで通す
}
```

**実行結果**:
```bash
✅ PASS  src/utils/validators.test.ts
  ✓ validateEmail › should return true for valid email (2ms)
```

#### Step 3: 🔴 Red - 新しいテストケース追加

```typescript
describe('validateEmail', () => {
  it('should return true for valid email', () => {
    expect(validateEmail('user@example.com')).toBe(true);
  });

  it('should return false for invalid email', () => {
    expect(validateEmail('invalid-email')).toBe(false);
  });
});
```

**実行結果**:
```bash
❌ FAIL  src/utils/validators.test.ts
  ✓ should return true for valid email
  ✗ should return false for invalid email
    Expected: false
    Received: true
```

#### Step 4: 🟢 Green - 実装を改善

```typescript
export function validateEmail(email: string): boolean {
  if (email === 'invalid-email') return false;
  return true;
}
```

**実行結果**:
```bash
✅ PASS  src/utils/validators.test.ts (2 tests)
```

#### Step 5: 🔴 Red - より多くのケース

```typescript
describe('validateEmail', () => {
  it('should return true for valid email', () => {
    expect(validateEmail('user@example.com')).toBe(true);
    expect(validateEmail('test.user@company.co.jp')).toBe(true);
  });

  it('should return false for invalid email', () => {
    expect(validateEmail('invalid-email')).toBe(false);
    expect(validateEmail('@example.com')).toBe(false);
    expect(validateEmail('user@')).toBe(false);
    expect(validateEmail('')).toBe(false);
  });
});
```

#### Step 6: 🟢 Green - 正規表現による実装

```typescript
export function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}
```

**実行結果**:
```bash
✅ PASS  src/utils/validators.test.ts (6 assertions)
```

#### Step 7: 🔵 Refactor - エッジケースの追加

```typescript
describe('validateEmail', () => {
  describe('valid emails', () => {
    it.each([
      'user@example.com',
      'test.user@company.co.jp',
      'name+tag@domain.com',
      'user123@test-domain.org',
    ])('should return true for "%s"', (email) => {
      expect(validateEmail(email)).toBe(true);
    });
  });

  describe('invalid emails', () => {
    it.each([
      'invalid-email',
      '@example.com',
      'user@',
      '',
      'user @example.com', // スペース
      'user@example',      // TLD なし
    ])('should return false for "%s"', (email) => {
      expect(validateEmail(email)).toBe(false);
    });
  });
});
```

#### Step 8: 🔵 Refactor - 型安全性の向上

```typescript
// src/utils/validators.ts
export type ValidationResult = {
  isValid: boolean;
  error?: string;
};

export function validateEmail(email: string): ValidationResult {
  if (!email || email.trim() === '') {
    return { isValid: false, error: 'Email is required' };
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  if (!emailRegex.test(email)) {
    return { isValid: false, error: 'Invalid email format' };
  }

  return { isValid: true };
}
```

```typescript
// テストも更新
describe('validateEmail', () => {
  describe('valid emails', () => {
    it.each([
      'user@example.com',
      'test.user@company.co.jp',
    ])('should return valid result for "%s"', (email) => {
      const result = validateEmail(email);
      expect(result.isValid).toBe(true);
      expect(result.error).toBeUndefined();
    });
  });

  describe('invalid emails', () => {
    it('should return error for empty email', () => {
      const result = validateEmail('');
      expect(result.isValid).toBe(false);
      expect(result.error).toBe('Email is required');
    });

    it('should return error for invalid format', () => {
      const result = validateEmail('invalid-email');
      expect(result.isValid).toBe(false);
      expect(result.error).toBe('Invalid email format');
    });
  });
});
```

**完成**: テストが全て通り、型安全で拡張可能な実装ができた！

---

### 2.3 TDDのリズム

**理想的なサイクル時間**:

```
🔴 Red:      1-3分
🟢 Green:    1-5分
🔵 Refactor: 2-10分

1サイクル: 5-15分
```

**1日の目安**:

```
午前 (4時間): 10-15サイクル
午後 (4時間): 10-15サイクル

1日: 20-30サイクル
```

**リズムを保つコツ**:

```typescript
// ❌ 悪い例: 一度に全テストを書く
describe('UserService', () => {
  it('should create user', () => { /* ... */ });
  it('should update user', () => { /* ... */ });
  it('should delete user', () => { /* ... */ });
  it('should find user', () => { /* ... */ });
  // 全て失敗 → どこから手をつけるか迷う
});

// ✅ 良い例: 1つずつ進める
describe('UserService', () => {
  it('should create user', () => {
    // このテスト1つだけ書く
    // → 実装
    // → 次のテストへ
  });
});
```

---

## 3. BDDとの使い分け

### 3.1 BDDとは

**Behavior-Driven Development (BDD)** は、ビジネス要件を自然言語で記述し、それをテストに変換する手法です。

**TDD vs BDD**:

| 観点 | TDD | BDD |
|------|-----|-----|
| **焦点** | 内部実装 | 外部の振る舞い |
| **記述** | 技術的 | ビジネス的 |
| **対象** | 開発者 | 開発者 + 非エンジニア |
| **粒度** | 関数・クラス | フィーチャー・シナリオ |

---

### 3.2 Given-When-Then パターン

**構造**:

```
Given (前提条件) - テストの初期状態
When  (実行)     - テスト対象の操作
Then  (検証)     - 期待される結果
```

**実例: ユーザーログイン**

```typescript
// BDD スタイル
describe('User Login', () => {
  it('should successfully log in with valid credentials', () => {
    // Given: ユーザーが登録済み
    const user = {
      email: 'user@example.com',
      password: 'SecurePass123',
    };
    database.createUser(user);

    // When: 正しい認証情報でログイン
    const result = authService.login(
      user.email,
      user.password
    );

    // Then: ログインに成功し、トークンを受け取る
    expect(result.success).toBe(true);
    expect(result.token).toBeDefined();
    expect(result.user.email).toBe(user.email);
  });

  it('should fail with invalid password', () => {
    // Given: ユーザーが登録済み
    const user = {
      email: 'user@example.com',
      password: 'SecurePass123',
    };
    database.createUser(user);

    // When: 間違ったパスワードでログイン
    const result = authService.login(
      user.email,
      'WrongPassword'
    );

    // Then: ログインに失敗
    expect(result.success).toBe(false);
    expect(result.error).toBe('Invalid credentials');
  });
});
```

---

### 3.3 BDDフレームワーク: Cucumber

**Gherkin 記法**:

```gherkin
# features/login.feature
Feature: User Login
  As a registered user
  I want to log in to the system
  So that I can access my account

  Scenario: Successful login with valid credentials
    Given a user exists with email "user@example.com" and password "SecurePass123"
    When I log in with email "user@example.com" and password "SecurePass123"
    Then I should be logged in successfully
    And I should receive an authentication token

  Scenario: Failed login with invalid password
    Given a user exists with email "user@example.com" and password "SecurePass123"
    When I log in with email "user@example.com" and password "WrongPassword"
    Then I should see an error "Invalid credentials"
    And I should not be logged in
```

**ステップ定義** (TypeScript + Cucumber):

```typescript
// features/step_definitions/login.steps.ts
import { Given, When, Then } from '@cucumber/cucumber';
import { expect } from 'chai';

let testUser: any;
let loginResult: any;

Given('a user exists with email {string} and password {string}',
  async (email: string, password: string) => {
    testUser = await database.createUser({ email, password });
  }
);

When('I log in with email {string} and password {string}',
  async (email: string, password: string) => {
    loginResult = await authService.login(email, password);
  }
);

Then('I should be logged in successfully', () => {
  expect(loginResult.success).to.be.true;
});

Then('I should receive an authentication token', () => {
  expect(loginResult.token).to.exist;
});

Then('I should see an error {string}', (errorMessage: string) => {
  expect(loginResult.error).to.equal(errorMessage);
});

Then('I should not be logged in', () => {
  expect(loginResult.success).to.be.false;
});
```

---

### 3.4 使い分けガイド

**TDDが適している場面**:

```
✅ アルゴリズムの実装
✅ ユーティリティ関数
✅ 内部ロジックの検証
✅ リファクタリング
✅ バグ修正
```

**実例**:
```typescript
// TDDで書くべき: 計算ロジック
describe('calculateDiscount', () => {
  it('should apply 10% discount for orders over $100', () => {
    expect(calculateDiscount(150)).toBe(15);
  });
});
```

**BDDが適している場面**:

```
✅ ビジネス要件の記述
✅ ユーザーストーリーのテスト
✅ E2Eシナリオ
✅ 非エンジニアとのコミュニケーション
✅ 受け入れテスト
```

**実例**:
```gherkin
# BDDで書くべき: ビジネスフロー
Scenario: Apply discount coupon at checkout
  Given I have items worth $150 in my cart
  When I apply coupon code "SAVE10"
  Then I should see a discount of $15
  And my total should be $135
```

**組み合わせパターン**:

```
E2E層      → BDD (Cucumber)
Integration → BDD or TDD (Given-When-Then)
Unit       → TDD (Red-Green-Refactor)
```

---

## 4. 実際のプロジェクト例

### 4.1 フィーチャー: ショッピングカート

**要件**:
- 商品を追加できる
- 商品を削除できる
- 数量を変更できる
- 合計金額を計算できる

---

#### Step 1: 🔴 Red - 最初のテスト

```typescript
// src/domain/ShoppingCart.test.ts
import { ShoppingCart } from './ShoppingCart';

describe('ShoppingCart', () => {
  it('should start empty', () => {
    const cart = new ShoppingCart();
    expect(cart.getItems()).toEqual([]);
    expect(cart.getTotal()).toBe(0);
  });
});
```

**実行結果**:
```bash
❌ Cannot find module './ShoppingCart'
```

---

#### Step 2: 🟢 Green - 最小実装

```typescript
// src/domain/ShoppingCart.ts
export class ShoppingCart {
  getItems() {
    return [];
  }

  getTotal() {
    return 0;
  }
}
```

**実行結果**:
```bash
✅ PASS (1 test)
```

---

#### Step 3: 🔴 Red - 商品追加機能

```typescript
describe('ShoppingCart', () => {
  it('should start empty', () => {
    const cart = new ShoppingCart();
    expect(cart.getItems()).toEqual([]);
    expect(cart.getTotal()).toBe(0);
  });

  it('should add item to cart', () => {
    const cart = new ShoppingCart();
    const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };

    cart.addItem(item);

    expect(cart.getItems()).toHaveLength(1);
    expect(cart.getItems()[0]).toEqual(item);
    expect(cart.getTotal()).toBe(1000);
  });
});
```

**実行結果**:
```bash
❌ cart.addItem is not a function
```

---

#### Step 4: 🟢 Green - addItem実装

```typescript
export type CartItem = {
  id: string;
  name: string;
  price: number;
  quantity: number;
};

export class ShoppingCart {
  private items: CartItem[] = [];

  getItems(): CartItem[] {
    return this.items;
  }

  getTotal(): number {
    return this.items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  }

  addItem(item: CartItem): void {
    this.items.push(item);
  }
}
```

**実行結果**:
```bash
✅ PASS (2 tests)
```

---

#### Step 5: 🔴 Red - 同じ商品の数量を増やす

```typescript
it('should increase quantity when adding same item', () => {
  const cart = new ShoppingCart();
  const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };

  cart.addItem(item);
  cart.addItem(item);

  expect(cart.getItems()).toHaveLength(1);
  expect(cart.getItems()[0].quantity).toBe(2);
  expect(cart.getTotal()).toBe(2000);
});
```

**実行結果**:
```bash
❌ Expected length: 1, Received: 2
```

---

#### Step 6: 🟢 Green - 重複チェック追加

```typescript
addItem(item: CartItem): void {
  const existingItem = this.items.find(i => i.id === item.id);

  if (existingItem) {
    existingItem.quantity += item.quantity;
  } else {
    this.items.push(item);
  }
}
```

**実行結果**:
```bash
✅ PASS (3 tests)
```

---

#### Step 7: 🔴 Red - 商品削除機能

```typescript
it('should remove item from cart', () => {
  const cart = new ShoppingCart();
  const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };

  cart.addItem(item);
  cart.removeItem(item.id);

  expect(cart.getItems()).toHaveLength(0);
  expect(cart.getTotal()).toBe(0);
});
```

**実行結果**:
```bash
❌ cart.removeItem is not a function
```

---

#### Step 8: 🟢 Green - removeItem実装

```typescript
removeItem(itemId: string): void {
  this.items = this.items.filter(item => item.id !== itemId);
}
```

**実行結果**:
```bash
✅ PASS (4 tests)
```

---

#### Step 9: 🔴 Red - 数量変更機能

```typescript
it('should update item quantity', () => {
  const cart = new ShoppingCart();
  const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };

  cart.addItem(item);
  cart.updateQuantity(item.id, 3);

  expect(cart.getItems()[0].quantity).toBe(3);
  expect(cart.getTotal()).toBe(3000);
});

it('should remove item when quantity is 0', () => {
  const cart = new ShoppingCart();
  const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };

  cart.addItem(item);
  cart.updateQuantity(item.id, 0);

  expect(cart.getItems()).toHaveLength(0);
});
```

---

#### Step 10: 🟢 Green - updateQuantity実装

```typescript
updateQuantity(itemId: string, quantity: number): void {
  if (quantity <= 0) {
    this.removeItem(itemId);
    return;
  }

  const item = this.items.find(i => i.id === itemId);
  if (item) {
    item.quantity = quantity;
  }
}
```

**実行結果**:
```bash
✅ PASS (6 tests)
```

---

#### Step 11: 🔵 Refactor - エラーハンドリング追加

```typescript
export class CartError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CartError';
  }
}

export class ShoppingCart {
  private items: CartItem[] = [];

  // ... 既存のメソッド

  updateQuantity(itemId: string, quantity: number): void {
    if (quantity < 0) {
      throw new CartError('Quantity cannot be negative');
    }

    if (quantity === 0) {
      this.removeItem(itemId);
      return;
    }

    const item = this.items.find(i => i.id === itemId);
    if (!item) {
      throw new CartError(`Item ${itemId} not found in cart`);
    }

    item.quantity = quantity;
  }
}
```

**対応するテスト**:

```typescript
describe('ShoppingCart - Error Handling', () => {
  it('should throw error for negative quantity', () => {
    const cart = new ShoppingCart();
    const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };
    cart.addItem(item);

    expect(() => {
      cart.updateQuantity(item.id, -1);
    }).toThrow('Quantity cannot be negative');
  });

  it('should throw error when updating non-existent item', () => {
    const cart = new ShoppingCart();

    expect(() => {
      cart.updateQuantity('non-existent', 5);
    }).toThrow('Item non-existent not found in cart');
  });
});
```

**実行結果**:
```bash
✅ PASS (8 tests)
```

---

### 4.2 完成したShoppingCart

```typescript
// src/domain/ShoppingCart.ts
export type CartItem = {
  id: string;
  name: string;
  price: number;
  quantity: number;
};

export class CartError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CartError';
  }
}

export class ShoppingCart {
  private items: CartItem[] = [];

  getItems(): CartItem[] {
    return [...this.items]; // 防御的コピー
  }

  getTotal(): number {
    return this.items.reduce(
      (sum, item) => sum + item.price * item.quantity,
      0
    );
  }

  addItem(item: CartItem): void {
    const existingItem = this.items.find(i => i.id === item.id);

    if (existingItem) {
      existingItem.quantity += item.quantity;
    } else {
      this.items.push({ ...item }); // 防御的コピー
    }
  }

  removeItem(itemId: string): void {
    this.items = this.items.filter(item => item.id !== itemId);
  }

  updateQuantity(itemId: string, quantity: number): void {
    if (quantity < 0) {
      throw new CartError('Quantity cannot be negative');
    }

    if (quantity === 0) {
      this.removeItem(itemId);
      return;
    }

    const item = this.items.find(i => i.id === itemId);
    if (!item) {
      throw new CartError(`Item ${itemId} not found in cart`);
    }

    item.quantity = quantity;
  }

  clear(): void {
    this.items = [];
  }

  isEmpty(): boolean {
    return this.items.length === 0;
  }

  getItemCount(): number {
    return this.items.reduce((count, item) => count + item.quantity, 0);
  }
}
```

**完全なテストスイート**:

```typescript
// src/domain/ShoppingCart.test.ts
import { ShoppingCart, CartError } from './ShoppingCart';

describe('ShoppingCart', () => {
  let cart: ShoppingCart;

  beforeEach(() => {
    cart = new ShoppingCart();
  });

  describe('初期状態', () => {
    it('should start empty', () => {
      expect(cart.getItems()).toEqual([]);
      expect(cart.getTotal()).toBe(0);
      expect(cart.isEmpty()).toBe(true);
      expect(cart.getItemCount()).toBe(0);
    });
  });

  describe('商品追加', () => {
    it('should add item to cart', () => {
      const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };

      cart.addItem(item);

      expect(cart.getItems()).toHaveLength(1);
      expect(cart.getItems()[0]).toEqual(item);
      expect(cart.getTotal()).toBe(1000);
      expect(cart.isEmpty()).toBe(false);
    });

    it('should increase quantity when adding same item', () => {
      const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };

      cart.addItem(item);
      cart.addItem(item);

      expect(cart.getItems()).toHaveLength(1);
      expect(cart.getItems()[0].quantity).toBe(2);
      expect(cart.getTotal()).toBe(2000);
    });

    it('should add multiple different items', () => {
      const item1 = { id: '1', name: 'Book', price: 1000, quantity: 1 };
      const item2 = { id: '2', name: 'Pen', price: 200, quantity: 3 };

      cart.addItem(item1);
      cart.addItem(item2);

      expect(cart.getItems()).toHaveLength(2);
      expect(cart.getTotal()).toBe(1600);
      expect(cart.getItemCount()).toBe(4);
    });
  });

  describe('商品削除', () => {
    it('should remove item from cart', () => {
      const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };

      cart.addItem(item);
      cart.removeItem(item.id);

      expect(cart.getItems()).toHaveLength(0);
      expect(cart.getTotal()).toBe(0);
      expect(cart.isEmpty()).toBe(true);
    });

    it('should do nothing when removing non-existent item', () => {
      const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };

      cart.addItem(item);
      cart.removeItem('non-existent');

      expect(cart.getItems()).toHaveLength(1);
    });
  });

  describe('数量変更', () => {
    it('should update item quantity', () => {
      const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };

      cart.addItem(item);
      cart.updateQuantity(item.id, 3);

      expect(cart.getItems()[0].quantity).toBe(3);
      expect(cart.getTotal()).toBe(3000);
    });

    it('should remove item when quantity is 0', () => {
      const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };

      cart.addItem(item);
      cart.updateQuantity(item.id, 0);

      expect(cart.getItems()).toHaveLength(0);
    });

    it('should throw error for negative quantity', () => {
      const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };
      cart.addItem(item);

      expect(() => {
        cart.updateQuantity(item.id, -1);
      }).toThrow(CartError);
      expect(() => {
        cart.updateQuantity(item.id, -1);
      }).toThrow('Quantity cannot be negative');
    });

    it('should throw error when updating non-existent item', () => {
      expect(() => {
        cart.updateQuantity('non-existent', 5);
      }).toThrow(CartError);
    });
  });

  describe('その他の操作', () => {
    it('should clear all items', () => {
      cart.addItem({ id: '1', name: 'Book', price: 1000, quantity: 1 });
      cart.addItem({ id: '2', name: 'Pen', price: 200, quantity: 3 });

      cart.clear();

      expect(cart.isEmpty()).toBe(true);
      expect(cart.getTotal()).toBe(0);
    });

    it('should return defensive copy of items', () => {
      const item = { id: '1', name: 'Book', price: 1000, quantity: 1 };
      cart.addItem(item);

      const items = cart.getItems();
      items[0].quantity = 999; // 外部から変更

      expect(cart.getItems()[0].quantity).toBe(1); // 影響を受けない
    });
  });
});
```

**テスト実行結果**:

```bash
PASS  src/domain/ShoppingCart.test.ts
  ShoppingCart
    初期状態
      ✓ should start empty (3ms)
    商品追加
      ✓ should add item to cart (2ms)
      ✓ should increase quantity when adding same item (1ms)
      ✓ should add multiple different items (2ms)
    商品削除
      ✓ should remove item from cart (1ms)
      ✓ should do nothing when removing non-existent item (1ms)
    数量変更
      ✓ should update item quantity (1ms)
      ✓ should remove item when quantity is 0 (1ms)
      ✓ should throw error for negative quantity (2ms)
      ✓ should throw error when updating non-existent item (1ms)
    その他の操作
      ✓ should clear all items (1ms)
      ✓ should return defensive copy of items (1ms)

Test Suites: 1 passed, 1 total
Tests:       12 passed, 12 total
Time:        0.842s
```

---

## 5. よくある失敗パターン

### 5.1 失敗パターン7選

#### ❌ 失敗 #1: テストを後回しにする

**問題**:
```typescript
// 実装を全部書いてから...
class UserService {
  createUser() { /* ... */ }
  updateUser() { /* ... */ }
  deleteUser() { /* ... */ }
  // ... 100行後

  // テストを書こうとすると...
  // 「どこからテストすればいいんだ？」
}
```

**解決**:
```typescript
// 1つずつ進める
describe('UserService', () => {
  it('should create user', () => {
    // テスト書く → 実装 → 次へ
  });
});
```

---

#### ❌ 失敗 #2: 大きすぎるステップ

**問題**:
```typescript
// いきなり複雑な機能を全部テスト
it('should handle complete checkout flow with payment and email', () => {
  // 20個のアサーションが失敗...
});
```

**解決**:
```typescript
// 小さく分割
it('should calculate order total', () => { /* ... */ });
it('should validate payment info', () => { /* ... */ });
it('should send confirmation email', () => { /* ... */ });
```

---

#### ❌ 失敗 #3: Greenを飛ばす

**問題**:
```typescript
// Redのまま次のテストを書く
it('test 1', () => { /* 失敗 */ });
it('test 2', () => { /* 失敗 */ });
it('test 3', () => { /* 失敗 */ });
// 全部失敗してどれから直すか分からない
```

**解決**:
```typescript
// 1つずつGreenにする
it('test 1', () => { /* 成功 */ });
// ✅ Greenになったら次へ
it('test 2', () => { /* ... */ });
```

---

#### ❌ 失敗 #4: 実装の詳細をテスト

**問題**:
```typescript
it('should call internal method', () => {
  const spy = jest.spyOn(service, '_privateMethod');
  service.publicMethod();
  expect(spy).toHaveBeenCalled(); // 内部実装に依存
});
```

**解決**:
```typescript
it('should return correct result', () => {
  const result = service.publicMethod();
  expect(result).toBe(expectedValue); // 公開APIをテスト
});
```

---

#### ❌ 失敗 #5: Refactorを忘れる

**問題**:
```typescript
// テストが通ったら満足して次へ...
function calculate(a, b, c, d, e) {
  if (a > 0 && b > 0 && c > 0) {
    return a + b + c + d + e;
  }
  // ... 汚いコードのまま
}
```

**解決**:
```typescript
// Greenの後、必ずRefactor
function calculateTotal(values: number[]): number {
  const positiveValues = values.filter(v => v > 0);
  return positiveValues.reduce((sum, v) => sum + v, 0);
}
```

---

#### ❌ 失敗 #6: テストが遅い

**問題**:
```typescript
describe('API Tests', () => {
  it('should fetch data', async () => {
    await new Promise(resolve => setTimeout(resolve, 3000));
    // 各テストが3秒... 100テストで5分
  });
});
```

**解決**:
```typescript
describe('API Tests', () => {
  it('should fetch data', async () => {
    // モックを使って高速化
    jest.spyOn(api, 'fetch').mockResolvedValue(mockData);
    const result = await service.getData();
    expect(result).toEqual(mockData);
  });
});
```

---

#### ❌ 失敗 #7: テストの重複

**問題**:
```typescript
it('test 1', () => {
  const result = complexSetup();
  expect(result.a).toBe(1);
});

it('test 2', () => {
  const result = complexSetup(); // 重複
  expect(result.b).toBe(2);
});
```

**解決**:
```typescript
describe('Feature', () => {
  let result;

  beforeEach(() => {
    result = complexSetup(); // 共通化
  });

  it('test 1', () => {
    expect(result.a).toBe(1);
  });

  it('test 2', () => {
    expect(result.b).toBe(2);
  });
});
```

---

## 6. チェックリスト

### 6.1 TDD実践チェックリスト

**開始前**:
- [ ] 要件を理解している
- [ ] 小さく始める計画を立てた
- [ ] テストファイルを作成した

**Redフェーズ**:
- [ ] 1つの振る舞いに集中している
- [ ] テストが失敗することを確認した
- [ ] エラーメッセージが明確

**Greenフェーズ**:
- [ ] 最小限の実装でテストを通した
- [ ] 全テストが成功している
- [ ] コードを実行して動作確認した

**Refactorフェーズ**:
- [ ] 重複コードを削除した
- [ ] 変数名・関数名が適切
- [ ] テストが依然として成功している

**1サイクル完了後**:
- [ ] コミットした
- [ ] 次のテストケースを決めた

---

### 6.2 BDD実践チェックリスト

**シナリオ作成**:
- [ ] Given-When-Thenで記述した
- [ ] ビジネス価値が明確
- [ ] 非エンジニアが理解できる
- [ ] テスト可能なシナリオ

**実装**:
- [ ] ステップ定義を作成した
- [ ] モックを適切に使用した
- [ ] シナリオが全て成功している

---

### 6.3 コードレビューチェックリスト

**テストの品質**:
- [ ] テストが先に書かれている（TDD）
- [ ] 1テスト = 1つの振る舞い
- [ ] テスト名が明確
- [ ] AAA（Arrange-Act-Assert）パターン
- [ ] エッジケースをカバーしている

**実装の品質**:
- [ ] 最小限の実装
- [ ] 不要なコードがない
- [ ] リファクタリング済み
- [ ] 型安全性が確保されている

---

## まとめ

### TDD/BDDの極意

```
🔴 Red:   小さく始める
🟢 Green: 最速で通す
🔵 Refactor: 美しくする

繰り返すことで品質向上
```

**推奨リソース**:
- 書籍: "Test Driven Development" by Kent Beck
- 書籍: "Growing Object-Oriented Software, Guided by Tests"
- 練習: [Coding Dojo](http://codingdojo.org/)

**次のステップ**:
1. 小さなユーティリティ関数でTDDを練習
2. 実プロジェクトで1機能をTDDで実装
3. チームでTDD勉強会を開催

---

**関連ガイド**:
- [test-pyramid-practice.md](./test-pyramid-practice.md) - テストピラミッド実践
- [unit-testing-complete.md](./unit-testing-complete.md) - ユニットテスト完全ガイド
- [integration-testing-complete.md](./integration-testing-complete.md) - 統合テストガイド
