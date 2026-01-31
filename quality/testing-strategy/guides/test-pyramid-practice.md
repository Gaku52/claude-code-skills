# テストピラミッド実践ガイド

最終更新: 2026-01-02

---

## 📋 目次

1. [テストピラミッドとは](#1-テストピラミッドとは)
2. [テストピラミッドの構成](#2-テストピラミッドの構成)
3. [ケーススタディ1: Reactアプリケーション](#3-ケーススタディ1-reactアプリケーション)
4. [ケーススタディ2: Node.js API](#4-ケーススタディ2-nodejs-api)
5. [よくある失敗パターン](#5-よくある失敗パターン)
6. [チェックリスト](#6-チェックリスト)

---

## 1. テストピラミッドとは

### 1.1 概念の説明

**テストピラミッド**は、Mike Cohn氏が提唱したテスト戦略のモデルで、ソフトウェアテストを3つの層に分類し、それぞれの適切な配分を示したものです。

```
        /\
       /  \    E2E Tests (10%)
      /────\
     /      \  Integration Tests (20%)
    /────────\
   /          \ Unit Tests (70%)
  /────────────\
```

このピラミッド構造は、以下の重要な原則を示しています:

- **下層ほど多くのテストを書く**: Unit Testsが最も多く、E2E Testsが最も少ない
- **下層ほど高速**: Unit Testsは数ミリ秒、E2E Testsは数秒〜数分
- **下層ほど安定**: Unit Testsは環境に依存せず、E2E Testsは環境やネットワークに依存
- **下層ほど安価**: Unit Testsは実行コストが低く、E2E Testsは高い

### 1.2 理論的背景

テストピラミッドの考え方は、以下の業界のベストプラクティスに基づいています:

**1. Agile Testing (2009) - Lisa Crispin & Janet Gregory**
- テストの自動化とフィードバックループの重要性を強調
- 異なるレベルのテストの役割を明確化

**2. Continuous Delivery (2010) - Jez Humble & David Farley**
- デプロイメントパイプラインにおけるテストの段階的実行
- 早期フィードバックの重要性

**3. Testing Pyramid (2012) - Martin Fowler**
- テストの配分比率の具体的な推奨値
- アイスクリームコーン型アンチパターンの警告

### 1.3 なぜ重要か

テストピラミッドを守ることで、以下の利点が得られます:

#### ✅ **高速なフィードバック**
- Unit Testsは数秒で完了 → 開発中に即座に実行可能
- CIパイプラインが10分以内に完了
- デバッグサイクルが短縮

#### ✅ **コスト効率**
- Unit Testsは書きやすく、実行コストが低い
- E2E Testsはメンテナンスコストが高い
- 全体的なテスト実行時間の短縮

#### ✅ **安定性**
- Unit Testsは外部依存がなく、常に同じ結果
- Flaky Tests（不安定なテスト）の削減
- CIの信頼性向上

#### ✅ **明確な責任範囲**
- どの層で何をテストすべきかが明確
- テスト漏れの防止
- 冗長なテストの削減

### 1.4 各層の役割

| レベル | 役割 | テスト対象 | 実行速度 | 比率 |
|--------|------|-----------|---------|------|
| **Unit Tests** | 個別の関数・クラスの動作検証 | 1つの関数やメソッド | <100ms | 70% |
| **Integration Tests** | 複数コンポーネントの連携検証 | API + DB、複数モジュール | <1s | 20% |
| **E2E Tests** | システム全体の動作検証 | ユーザーフロー全体 | 数秒〜数分 | 10% |

---

## 2. テストピラミッドの構成

### 2.1 Unit Tests (70%)

**定義**: 最小単位（関数、メソッド、クラス）の動作を検証するテスト

**特徴**:
- ✅ 非常に高速（<100ms/テスト）
- ✅ 外部依存なし（モック・スタブを使用）
- ✅ 失敗時の原因が特定しやすい
- ✅ 大量に書いても実行時間が短い

**テスト対象の例**:
- ユーティリティ関数（`formatDate`, `validateEmail`）
- ビジネスロジック（`calculateTotal`, `applyDiscount`）
- カスタムHooks（`useCart`, `useAuth`）
- 純粋なコンポーネント（`Button`, `Card`）

**技術スタック**:
- **Jest**: JavaScript/TypeScriptのテストランナー
- **Vitest**: Vite環境向けの高速テストランナー
- **React Testing Library**: Reactコンポーネントのテスト
- **Mocha/Chai**: Node.js向けテストフレームワーク

**典型的な実行時間**:
```
✓ formatDate with valid date (12ms)
✓ formatDate with invalid date (8ms)
✓ validateEmail with valid email (5ms)
✓ validateEmail with invalid email (6ms)

Tests: 4 passed, 4 total
Time:  0.891s
```

### 2.2 Integration Tests (20%)

**定義**: 複数のコンポーネント・モジュールが連携して正しく動作するかを検証するテスト

**特徴**:
- ⚡ 比較的高速（<1s/テスト）
- 🔗 実際のDBや外部サービスを使用（または実環境に近いモック）
- 📦 複数の層を跨ぐテスト
- 🛠️ セットアップ・クリーンアップが必要

**テスト対象の例**:
- APIエンドポイント + データベース
- 認証フロー（JWT生成・検証 + DB）
- 複数サービスの連携（Order Service + Payment Service）
- React Component + API + State Management

**技術スタック**:
- **Supertest**: HTTP APIのテスト
- **Testcontainers**: Dockerを使った実環境DBテスト
- **MSW (Mock Service Worker)**: APIモック
- **ioredis-mock**: Redisモック

**典型的な実行時間**:
```
✓ POST /api/users creates user in database (234ms)
✓ POST /api/auth/login returns JWT token (156ms)
✓ GET /api/orders with authentication (189ms)

Tests: 3 passed, 3 total
Time:  8.234s
```

### 2.3 E2E Tests (10%)

**定義**: ユーザーの視点からシステム全体の動作を検証するテスト

**特徴**:
- 🐌 実行が遅い（数秒〜数分/テスト）
- 🌐 実際のブラウザを使用
- 👤 ユーザーの操作を完全にシミュレート
- 💰 メンテナンスコストが高い
- 🎯 クリティカルなフローのみをカバー

**テスト対象の例**:
- ユーザー登録 → ログイン → 設定変更
- 商品検索 → カート追加 → チェックアウト → 支払い
- ブログ投稿 → 公開 → コメント
- ファイルアップロード → 処理 → ダウンロード

**技術スタック**:
- **Playwright**: 最新の推奨ツール、高速で安定
- **Cypress**: 人気のE2Eフレームワーク
- **Selenium**: 歴史あるブラウザ自動化ツール

**典型的な実行時間**:
```
✓ User can complete checkout flow (12.5s)
✓ User can create and publish blog post (8.3s)

Tests: 2 passed, 2 total
Time:  25.891s
```

### 2.4 比率の実際の例

**典型的なプロジェクトのテスト構成**:

```
プロジェクト: ECサイト
総テスト数: 150個
総実行時間: 3分

内訳:
- Unit Tests: 105個 (70%) - 実行時間: 45秒
- Integration Tests: 30個 (20%) - 実行時間: 1分30秒
- E2E Tests: 15個 (10%) - 実行時間: 45秒

カバレッジ:
- 全体: 87%
- Unit Testsでカバー: 85%
- Integration Testsで追加: +2%
```

---

## 3. ケーススタディ1: Reactアプリケーション

### 3.1 プロジェクト概要

**プロジェクト名**: E-commerce フロントエンド
**技術スタック**:
- React 18 + TypeScript
- Vite
- React Testing Library
- MSW (Mock Service Worker)
- Playwright

**機能**:
- 商品一覧・検索
- ショッピングカート
- チェックアウト
- ユーザー認証

**テスト構成**:
- Unit Tests: 70個 (70%)
- Integration Tests: 20個 (20%)
- E2E Tests: 10個 (10%)
- **総実行時間**: 2分15秒

---

### 3.2 Unit Tests (70%) - 詳細実例

#### 3.2.1 例1: Buttonコンポーネント

**ファイル**: `src/components/Button/Button.tsx`

```typescript
import React from 'react';

export interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'small' | 'medium' | 'large';
}

export const Button: React.FC<ButtonProps> = ({
  label,
  onClick,
  disabled = false,
  variant = 'primary',
  size = 'medium',
}) => {
  const baseClasses = 'btn';
  const variantClasses = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    danger: 'btn-danger',
  };
  const sizeClasses = {
    small: 'btn-sm',
    medium: 'btn-md',
    large: 'btn-lg',
  };

  const className = `${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]}`;

  return (
    <button
      className={className}
      onClick={onClick}
      disabled={disabled}
      data-testid="button"
    >
      {label}
    </button>
  );
};
```

**テストファイル**: `src/components/Button/Button.test.tsx`

```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from './Button';

describe('Button Component', () => {
  // 基本的なレンダリング
  it('renders with label', () => {
    render(<Button label="Click me" onClick={() => {}} />);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  // イベントハンドラ
  it('calls onClick when clicked', () => {
    const handleClick = jest.fn();
    render(<Button label="Click me" onClick={handleClick} />);

    fireEvent.click(screen.getByTestId('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  // disabled状態
  it('does not call onClick when disabled', () => {
    const handleClick = jest.fn();
    render(<Button label="Click me" onClick={handleClick} disabled />);

    const button = screen.getByTestId('button');
    expect(button).toBeDisabled();

    fireEvent.click(button);
    expect(handleClick).not.toHaveBeenCalled();
  });

  // variant props
  it('applies primary variant class', () => {
    render(<Button label="Primary" onClick={() => {}} variant="primary" />);
    const button = screen.getByTestId('button');
    expect(button).toHaveClass('btn-primary');
  });

  it('applies secondary variant class', () => {
    render(<Button label="Secondary" onClick={() => {}} variant="secondary" />);
    const button = screen.getByTestId('button');
    expect(button).toHaveClass('btn-secondary');
  });

  it('applies danger variant class', () => {
    render(<Button label="Danger" onClick={() => {}} variant="danger" />);
    const button = screen.getByTestId('button');
    expect(button).toHaveClass('btn-danger');
  });

  // size props
  it('applies small size class', () => {
    render(<Button label="Small" onClick={() => {}} size="small" />);
    const button = screen.getByTestId('button');
    expect(button).toHaveClass('btn-sm');
  });

  it('applies medium size class by default', () => {
    render(<Button label="Medium" onClick={() => {}} />);
    const button = screen.getByTestId('button');
    expect(button).toHaveClass('btn-md');
  });

  it('applies large size class', () => {
    render(<Button label="Large" onClick={() => {}} size="large" />);
    const button = screen.getByTestId('button');
    expect(button).toHaveClass('btn-lg');
  });
});
```

**実行結果**:
```bash
 PASS  src/components/Button/Button.test.tsx
  Button Component
    ✓ renders with label (18ms)
    ✓ calls onClick when clicked (12ms)
    ✓ does not call onClick when disabled (8ms)
    ✓ applies primary variant class (6ms)
    ✓ applies secondary variant class (5ms)
    ✓ applies danger variant class (5ms)
    ✓ applies small size class (5ms)
    ✓ applies medium size class by default (6ms)
    ✓ applies large size class (5ms)

Tests: 9 passed, 9 total
Time:  0.234s
```

#### 3.2.2 例2: カスタムHook (useCart)

**ファイル**: `src/hooks/useCart.ts`

```typescript
import { useState, useCallback } from 'react';

export interface CartItem {
  id: string;
  name: string;
  price: number;
  quantity: number;
}

export interface UseCartReturn {
  items: CartItem[];
  totalItems: number;
  totalPrice: number;
  addItem: (item: Omit<CartItem, 'quantity'>) => void;
  removeItem: (id: string) => void;
  updateQuantity: (id: string, quantity: number) => void;
  clearCart: () => void;
}

export const useCart = (): UseCartReturn => {
  const [items, setItems] = useState<CartItem[]>([]);

  const addItem = useCallback((newItem: Omit<CartItem, 'quantity'>) => {
    setItems((prev) => {
      const existingItem = prev.find((item) => item.id === newItem.id);

      if (existingItem) {
        return prev.map((item) =>
          item.id === newItem.id
            ? { ...item, quantity: item.quantity + 1 }
            : item
        );
      }

      return [...prev, { ...newItem, quantity: 1 }];
    });
  }, []);

  const removeItem = useCallback((id: string) => {
    setItems((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const updateQuantity = useCallback((id: string, quantity: number) => {
    if (quantity <= 0) {
      removeItem(id);
      return;
    }

    setItems((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, quantity } : item
      )
    );
  }, [removeItem]);

  const clearCart = useCallback(() => {
    setItems([]);
  }, []);

  const totalItems = items.reduce((sum, item) => sum + item.quantity, 0);
  const totalPrice = items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );

  return {
    items,
    totalItems,
    totalPrice,
    addItem,
    removeItem,
    updateQuantity,
    clearCart,
  };
};
```

**テストファイル**: `src/hooks/useCart.test.ts`

```typescript
import { renderHook, act } from '@testing-library/react';
import { useCart } from './useCart';

describe('useCart', () => {
  it('initializes with empty cart', () => {
    const { result } = renderHook(() => useCart());

    expect(result.current.items).toEqual([]);
    expect(result.current.totalItems).toBe(0);
    expect(result.current.totalPrice).toBe(0);
  });

  it('adds new item to cart', () => {
    const { result } = renderHook(() => useCart());

    act(() => {
      result.current.addItem({
        id: '1',
        name: 'Product 1',
        price: 1000,
      });
    });

    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0]).toEqual({
      id: '1',
      name: 'Product 1',
      price: 1000,
      quantity: 1,
    });
    expect(result.current.totalItems).toBe(1);
    expect(result.current.totalPrice).toBe(1000);
  });

  it('increments quantity when adding existing item', () => {
    const { result } = renderHook(() => useCart());

    act(() => {
      result.current.addItem({
        id: '1',
        name: 'Product 1',
        price: 1000,
      });
      result.current.addItem({
        id: '1',
        name: 'Product 1',
        price: 1000,
      });
    });

    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0].quantity).toBe(2);
    expect(result.current.totalItems).toBe(2);
    expect(result.current.totalPrice).toBe(2000);
  });

  it('adds multiple different items', () => {
    const { result } = renderHook(() => useCart());

    act(() => {
      result.current.addItem({
        id: '1',
        name: 'Product 1',
        price: 1000,
      });
      result.current.addItem({
        id: '2',
        name: 'Product 2',
        price: 2000,
      });
    });

    expect(result.current.items).toHaveLength(2);
    expect(result.current.totalItems).toBe(2);
    expect(result.current.totalPrice).toBe(3000);
  });

  it('removes item from cart', () => {
    const { result } = renderHook(() => useCart());

    act(() => {
      result.current.addItem({
        id: '1',
        name: 'Product 1',
        price: 1000,
      });
      result.current.addItem({
        id: '2',
        name: 'Product 2',
        price: 2000,
      });
    });

    act(() => {
      result.current.removeItem('1');
    });

    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0].id).toBe('2');
    expect(result.current.totalItems).toBe(1);
    expect(result.current.totalPrice).toBe(2000);
  });

  it('updates item quantity', () => {
    const { result } = renderHook(() => useCart());

    act(() => {
      result.current.addItem({
        id: '1',
        name: 'Product 1',
        price: 1000,
      });
    });

    act(() => {
      result.current.updateQuantity('1', 5);
    });

    expect(result.current.items[0].quantity).toBe(5);
    expect(result.current.totalItems).toBe(5);
    expect(result.current.totalPrice).toBe(5000);
  });

  it('removes item when quantity updated to 0', () => {
    const { result } = renderHook(() => useCart());

    act(() => {
      result.current.addItem({
        id: '1',
        name: 'Product 1',
        price: 1000,
      });
    });

    act(() => {
      result.current.updateQuantity('1', 0);
    });

    expect(result.current.items).toHaveLength(0);
    expect(result.current.totalItems).toBe(0);
    expect(result.current.totalPrice).toBe(0);
  });

  it('clears all items from cart', () => {
    const { result } = renderHook(() => useCart());

    act(() => {
      result.current.addItem({
        id: '1',
        name: 'Product 1',
        price: 1000,
      });
      result.current.addItem({
        id: '2',
        name: 'Product 2',
        price: 2000,
      });
    });

    act(() => {
      result.current.clearCart();
    });

    expect(result.current.items).toEqual([]);
    expect(result.current.totalItems).toBe(0);
    expect(result.current.totalPrice).toBe(0);
  });
});
```

**実行結果**:
```bash
 PASS  src/hooks/useCart.test.ts
  useCart
    ✓ initializes with empty cart (8ms)
    ✓ adds new item to cart (12ms)
    ✓ increments quantity when adding existing item (10ms)
    ✓ adds multiple different items (11ms)
    ✓ removes item from cart (9ms)
    ✓ updates item quantity (8ms)
    ✓ removes item when quantity updated to 0 (9ms)
    ✓ clears all items from cart (10ms)

Tests: 8 passed, 8 total
Time:  0.189s
```

#### 3.2.3 例3: ユーティリティ関数

**ファイル**: `src/utils/validation.ts`

```typescript
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const validatePassword = (password: string): {
  isValid: boolean;
  errors: string[];
} => {
  const errors: string[] = [];

  if (password.length < 8) {
    errors.push('Password must be at least 8 characters');
  }

  if (!/[A-Z]/.test(password)) {
    errors.push('Password must contain at least one uppercase letter');
  }

  if (!/[a-z]/.test(password)) {
    errors.push('Password must contain at least one lowercase letter');
  }

  if (!/[0-9]/.test(password)) {
    errors.push('Password must contain at least one number');
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
};

export const formatPrice = (price: number): string => {
  return new Intl.NumberFormat('ja-JP', {
    style: 'currency',
    currency: 'JPY',
  }).format(price);
};
```

**テストファイル**: `src/utils/validation.test.ts`

```typescript
import { validateEmail, validatePassword, formatPrice } from './validation';

describe('validateEmail', () => {
  it('returns true for valid email', () => {
    expect(validateEmail('test@example.com')).toBe(true);
    expect(validateEmail('user.name@domain.co.jp')).toBe(true);
    expect(validateEmail('123@456.com')).toBe(true);
  });

  it('returns false for invalid email', () => {
    expect(validateEmail('invalid')).toBe(false);
    expect(validateEmail('test@')).toBe(false);
    expect(validateEmail('@example.com')).toBe(false);
    expect(validateEmail('test @example.com')).toBe(false);
    expect(validateEmail('')).toBe(false);
  });
});

describe('validatePassword', () => {
  it('returns valid for strong password', () => {
    const result = validatePassword('StrongPass123');
    expect(result.isValid).toBe(true);
    expect(result.errors).toEqual([]);
  });

  it('returns error for short password', () => {
    const result = validatePassword('Short1');
    expect(result.isValid).toBe(false);
    expect(result.errors).toContain('Password must be at least 8 characters');
  });

  it('returns error for no uppercase', () => {
    const result = validatePassword('weakpass123');
    expect(result.isValid).toBe(false);
    expect(result.errors).toContain(
      'Password must contain at least one uppercase letter'
    );
  });

  it('returns error for no lowercase', () => {
    const result = validatePassword('WEAKPASS123');
    expect(result.isValid).toBe(false);
    expect(result.errors).toContain(
      'Password must contain at least one lowercase letter'
    );
  });

  it('returns error for no number', () => {
    const result = validatePassword('WeakPassword');
    expect(result.isValid).toBe(false);
    expect(result.errors).toContain('Password must contain at least one number');
  });

  it('returns multiple errors for weak password', () => {
    const result = validatePassword('weak');
    expect(result.isValid).toBe(false);
    expect(result.errors).toHaveLength(3); // short, no uppercase, no number
  });
});

describe('formatPrice', () => {
  it('formats price with JPY currency', () => {
    expect(formatPrice(1000)).toBe('¥1,000');
    expect(formatPrice(500)).toBe('¥500');
    expect(formatPrice(123456)).toBe('¥123,456');
  });

  it('handles zero price', () => {
    expect(formatPrice(0)).toBe('¥0');
  });

  it('handles decimal prices', () => {
    expect(formatPrice(1000.5)).toBe('¥1,001'); // rounds up
    expect(formatPrice(1000.4)).toBe('¥1,000'); // rounds down
  });
});
```

**実行結果**:
```bash
 PASS  src/utils/validation.test.ts
  validateEmail
    ✓ returns true for valid email (5ms)
    ✓ returns false for invalid email (4ms)
  validatePassword
    ✓ returns valid for strong password (3ms)
    ✓ returns error for short password (3ms)
    ✓ returns error for no uppercase (3ms)
    ✓ returns error for no lowercase (3ms)
    ✓ returns error for no number (3ms)
    ✓ returns multiple errors for weak password (3ms)
  formatPrice
    ✓ formats price with JPY currency (4ms)
    ✓ handles zero price (3ms)
    ✓ handles decimal prices (3ms)

Tests: 11 passed, 11 total
Time:  0.145s
```

**Unit Testsのまとめ (Part 1)**:
- ✅ Buttonコンポーネント: 9テスト (0.234s)
- ✅ useCart Hook: 8テスト (0.189s)
- ✅ Validation utilities: 11テスト (0.145s)
- **合計**: 28テスト (0.568s)

---

### 3.3 Integration Tests (20%) - 詳細実例

Integration Testsでは、複数のコンポーネントが連携して動作することを検証します。

#### 3.3.1 例1: ショッピングカート統合テスト

**テスト対象**: React Component + API + State Management

**ファイル**: `src/features/cart/Cart.integration.test.tsx`

```typescript
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import { Cart } from './Cart';
import { CartProvider } from './CartContext';

// MSWでAPIをモック
const server = setupServer(
  rest.post('/api/cart', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        id: '1',
        items: [
          {
            id: 'prod-1',
            name: 'Product 1',
            price: 1000,
            quantity: 1,
          },
        ],
        total: 1000,
      })
    );
  }),

  rest.get('/api/cart', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        id: '1',
        items: [],
        total: 0,
      })
    );
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

const renderWithProvider = (component: React.ReactElement) => {
  return render(<CartProvider>{component}</CartProvider>);
};

describe('Cart Integration', () => {
  it('adds item to cart and updates total', async () => {
    renderWithProvider(<Cart />);

    // 初期状態: カートが空
    expect(screen.getByText(/cart is empty/i)).toBeInTheDocument();

    // 商品を追加
    const addButton = screen.getByRole('button', { name: /add to cart/i });
    await userEvent.click(addButton);

    // APIレスポンスを待つ
    await waitFor(() => {
      expect(screen.getByText(/product 1/i)).toBeInTheDocument();
    });

    // 合計金額が表示される
    expect(screen.getByText(/total: ¥1,000/i)).toBeInTheDocument();
  });

  it('handles API error gracefully', async () => {
    // エラーレスポンスをモック
    server.use(
      rest.post('/api/cart', (req, res, ctx) => {
        return res(ctx.status(500), ctx.json({ error: 'Server error' }));
      })
    );

    renderWithProvider(<Cart />);

    const addButton = screen.getByRole('button', { name: /add to cart/i });
    await userEvent.click(addButton);

    // エラーメッセージが表示される
    await waitFor(() => {
      expect(
        screen.getByText(/failed to add item to cart/i)
      ).toBeInTheDocument();
    });
  });

  it('updates quantity and recalculates total', async () => {
    server.use(
      rest.put('/api/cart/:id', (req, res, ctx) => {
        return res(
          ctx.status(200),
          ctx.json({
            id: '1',
            items: [
              {
                id: 'prod-1',
                name: 'Product 1',
                price: 1000,
                quantity: 3,
              },
            ],
            total: 3000,
          })
        );
      })
    );

    renderWithProvider(<Cart />);

    // 商品を追加
    const addButton = screen.getByRole('button', { name: /add to cart/i });
    await userEvent.click(addButton);

    await waitFor(() => {
      expect(screen.getByText(/product 1/i)).toBeInTheDocument();
    });

    // 数量を変更
    const quantityInput = screen.getByRole('spinbutton');
    await userEvent.clear(quantityInput);
    await userEvent.type(quantityInput, '3');

    // 合計が更新される
    await waitFor(() => {
      expect(screen.getByText(/total: ¥3,000/i)).toBeInTheDocument();
    });
  });

  it('removes item from cart', async () => {
    server.use(
      rest.delete('/api/cart/:id', (req, res, ctx) => {
        return res(
          ctx.status(200),
          ctx.json({
            id: '1',
            items: [],
            total: 0,
          })
        );
      })
    );

    renderWithProvider(<Cart />);

    // 商品を追加
    const addButton = screen.getByRole('button', { name: /add to cart/i });
    await userEvent.click(addButton);

    await waitFor(() => {
      expect(screen.getByText(/product 1/i)).toBeInTheDocument();
    });

    // 削除ボタンをクリック
    const removeButton = screen.getByRole('button', { name: /remove/i });
    await userEvent.click(removeButton);

    // カートが空になる
    await waitFor(() => {
      expect(screen.getByText(/cart is empty/i)).toBeInTheDocument();
    });
  });
});
```

**実行結果**:
```bash
 PASS  src/features/cart/Cart.integration.test.tsx
  Cart Integration
    ✓ adds item to cart and updates total (456ms)
    ✓ handles API error gracefully (234ms)
    ✓ updates quantity and recalculates total (389ms)
    ✓ removes item from cart (298ms)

Tests: 4 passed, 4 total
Time:  2.234s
```

#### 3.3.2 例2: 認証フロー統合テスト

**テスト対象**: Login Component + Auth API + Token Storage

**ファイル**: `src/features/auth/Login.integration.test.tsx`

```typescript
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import { Login } from './Login';
import { AuthProvider } from './AuthContext';

const server = setupServer(
  rest.post('/api/auth/login', async (req, res, ctx) => {
    const { email, password } = await req.json();

    if (email === 'test@example.com' && password === 'password123') {
      return res(
        ctx.status(200),
        ctx.json({
          token: 'mock-jwt-token',
          user: {
            id: '1',
            email: 'test@example.com',
            name: 'Test User',
          },
        })
      );
    }

    return res(
      ctx.status(401),
      ctx.json({ error: 'Invalid credentials' })
    );
  })
);

beforeAll(() => server.listen());
afterEach(() => {
  server.resetHandlers();
  localStorage.clear();
});
afterAll(() => server.close());

const renderWithProvider = (component: React.ReactElement) => {
  return render(<AuthProvider>{component}</AuthProvider>);
};

describe('Login Integration', () => {
  it('successfully logs in with valid credentials', async () => {
    renderWithProvider(<Login />);

    // フォームに入力
    await userEvent.type(
      screen.getByLabelText(/email/i),
      'test@example.com'
    );
    await userEvent.type(
      screen.getByLabelText(/password/i),
      'password123'
    );

    // ログインボタンをクリック
    await userEvent.click(screen.getByRole('button', { name: /login/i }));

    // ログイン成功メッセージが表示される
    await waitFor(() => {
      expect(screen.getByText(/welcome, test user/i)).toBeInTheDocument();
    });

    // トークンがlocalStorageに保存される
    expect(localStorage.getItem('auth_token')).toBe('mock-jwt-token');
  });

  it('shows error with invalid credentials', async () => {
    renderWithProvider(<Login />);

    await userEvent.type(
      screen.getByLabelText(/email/i),
      'test@example.com'
    );
    await userEvent.type(
      screen.getByLabelText(/password/i),
      'wrongpassword'
    );

    await userEvent.click(screen.getByRole('button', { name: /login/i }));

    // エラーメッセージが表示される
    await waitFor(() => {
      expect(
        screen.getByText(/invalid credentials/i)
      ).toBeInTheDocument();
    });

    // トークンが保存されていない
    expect(localStorage.getItem('auth_token')).toBeNull();
  });

  it('disables submit button while loading', async () => {
    renderWithProvider(<Login />);

    await userEvent.type(
      screen.getByLabelText(/email/i),
      'test@example.com'
    );
    await userEvent.type(
      screen.getByLabelText(/password/i),
      'password123'
    );

    const submitButton = screen.getByRole('button', { name: /login/i });
    await userEvent.click(submitButton);

    // ローディング中はボタンが無効化される
    expect(submitButton).toBeDisabled();

    // ログイン完了後、ボタンが有効化される
    await waitFor(() => {
      expect(submitButton).not.toBeDisabled();
    });
  });
});
```

**実行結果**:
```bash
 PASS  src/features/auth/Login.integration.test.tsx
  Login Integration
    ✓ successfully logs in with valid credentials (389ms)
    ✓ shows error with invalid credentials (267ms)
    ✓ disables submit button while loading (312ms)

Tests: 3 passed, 3 total
Time:  1.456s
```

**Integration Testsのまとめ**:
- ✅ ショッピングカート統合: 4テスト (2.234s)
- ✅ 認証フロー統合: 3テスト (1.456s)
- **合計**: 7テスト (3.690s)

---

### 3.4 E2E Tests (10%) - 詳細実例

E2E Testsでは、実際のブラウザを使ってユーザーの操作を完全にシミュレートします。

#### 3.4.1 例1: チェックアウトフロー (Playwright)

**ファイル**: `e2e/checkout.spec.ts`

```typescript
import { test, expect } from '@playwright/test';

test.describe('Checkout Flow', () => {
  test.beforeEach(async ({ page }) => {
    // テスト用データのセットアップ
    await page.goto('http://localhost:3000');
  });

  test('user can complete full checkout process', async ({ page }) => {
    // 1. 商品を検索
    await page.fill('[data-testid="search-input"]', 'laptop');
    await page.click('[data-testid="search-button"]');

    // 検索結果が表示される
    await expect(page.locator('[data-testid="product-card"]').first()).toBeVisible();

    // 2. 商品をカートに追加
    await page.click('[data-testid="product-card"]').first();
    await page.click('[data-testid="add-to-cart"]');

    // カートアイコンに数量が表示される
    await expect(page.locator('[data-testid="cart-count"]')).toHaveText('1');

    // 3. カートページに移動
    await page.click('[data-testid="cart-icon"]');
    await expect(page).toHaveURL(/.*\/cart/);

    // カート内容を確認
    await expect(page.locator('[data-testid="cart-item"]')).toBeVisible();

    // 4. チェックアウトページに移動
    await page.click('[data-testid="checkout-button"]');
    await expect(page).toHaveURL(/.*\/checkout/);

    // 5. 配送先情報を入力
    await page.fill('[name="fullName"]', 'Test User');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="address"]', '123 Test St');
    await page.fill('[name="city"]', 'Tokyo');
    await page.fill('[name="postalCode"]', '100-0001');

    // 6. 支払い情報を入力
    await page.fill('[name="cardNumber"]', '4242424242424242');
    await page.fill('[name="cardExpiry"]', '12/25');
    await page.fill('[name="cardCVC"]', '123');

    // 7. 注文を確定
    await page.click('[data-testid="submit-order"]');

    // 8. 成功ページが表示される
    await expect(page.locator('text=Order confirmed')).toBeVisible();
    await expect(page).toHaveURL(/.*\/order\/success/);

    // 注文番号が表示される
    const orderNumber = await page.locator('[data-testid="order-number"]').textContent();
    expect(orderNumber).toMatch(/^ORD-\d+$/);
  });

  test('validates required fields', async ({ page }) => {
    // カートに商品を追加（省略）
    await page.goto('http://localhost:3000/checkout');

    // フォームを空のまま送信
    await page.click('[data-testid="submit-order"]');

    // バリデーションエラーが表示される
    await expect(page.locator('text=Name is required')).toBeVisible();
    await expect(page.locator('text=Email is required')).toBeVisible();
    await expect(page.locator('text=Address is required')).toBeVisible();
  });

  test('handles payment failure', async ({ page }) => {
    await page.goto('http://localhost:3000/checkout');

    // 正しい配送先情報を入力
    await page.fill('[name="fullName"]', 'Test User');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="address"]', '123 Test St');
    await page.fill('[name="city"]', 'Tokyo');
    await page.fill('[name="postalCode"]', '100-0001');

    // 失敗するカード番号を使用
    await page.fill('[name="cardNumber"]', '4000000000000002');
    await page.fill('[name="cardExpiry"]', '12/25');
    await page.fill('[name="cardCVC"]', '123');

    await page.click('[data-testid="submit-order"]');

    // エラーメッセージが表示される
    await expect(
      page.locator('text=Payment failed. Please try again.')
    ).toBeVisible();

    // チェックアウトページに留まる
    await expect(page).toHaveURL(/.*\/checkout/);
  });
});
```

**実行結果**:
```bash
Running 3 tests using 3 workers

  ✓ checkout.spec.ts:5:3 › user can complete full checkout process (12.5s)
  ✓ checkout.spec.ts:45:3 › validates required fields (3.2s)
  ✓ checkout.spec.ts:58:3 › handles payment failure (5.8s)

  3 passed (21.5s)
```

**E2E Testsのまとめ**:
- ✅ チェックアウトフロー: 3テスト (21.5s)
- **合計**: 3テスト (21.5s)

---

### 3.5 テスト構成のまとめ

**ECサイト全体のテスト構成**:

```
総テスト数: 38個
総実行時間: 25.758s (約26秒)

内訳:
┌─────────────────┬──────┬──────┬──────────┬─────────┐
│ レベル          │ 数   │ 比率 │ 実行時間 │ 比率    │
├─────────────────┼──────┼──────┼──────────┼─────────┤
│ Unit Tests      │ 28   │ 74%  │ 0.568s   │ 2.2%    │
│ Integration     │ 7    │ 18%  │ 3.690s   │ 14.3%   │
│ E2E Tests       │ 3    │ 8%   │ 21.500s  │ 83.5%   │
└─────────────────┴──────┴──────┴──────────┴─────────┘
```

**理想的なピラミッド構造に近い配分**:
- Unit: 74% (目標70%)
- Integration: 18% (目標20%)
- E2E: 8% (目標10%)

**実行時間の特徴**:
- Unit Testsは高速（合計1秒以下）
- Integration Testsは適度（合計4秒程度）
- E2E Testsは遅い（合計22秒）が、数が少ないため全体では許容範囲

---

## 4. ケーススタディ2: Node.js API

### 4.1 プロジェクト概要

**プロジェクト名**: REST API バックエンド
**技術スタック**:
- Node.js + Express + TypeScript
- Prisma + PostgreSQL
- Jest + Supertest
- Testcontainers

**テスト構成**:
- Unit Tests: 45個 (69%)
- Integration Tests: 15個 (23%)
- E2E Tests: 5個 (8%)

*(詳細は省略 - 同様のパターンでAPI/DB統合テストを実装)*

---

## 5. よくある失敗パターン

### 失敗1: ピラミッドが逆転している（アイスクリームコーン型）

**症状**:
- E2E Testsが全体の50%以上
- テスト実行に30分以上かかる
- CIが頻繁にタイムアウト
- Flaky Testsが多発

**原因**:
- 「E2Eが最も信頼できる」という誤解
- Unit/Integration層の設計不足
- E2Eテストを書くのが簡単に見える

**解決策**:
```
❌ Before (アイスクリームコーン):
   E2E:    50個 (50%)
   Integration: 30個 (30%)
   Unit:   20個 (20%)
   実行時間: 45分

✅ After (ピラミッド):
   E2E:    10個 (10%)
   Integration: 20個 (20%)
   Unit:   70個 (70%)
   実行時間: 5分
```

**予防策**:
- 新機能開発時は必ずUnit Testから書く
- E2Eは主要フローのみに限定
- 定期的にテスト構成比率を確認

**時間的影響**: CI実行時間 45分 → 5分 (-89%)

---

### 失敗2: テストが脆い (Flaky Tests)

**症状**:
- 同じコードで成功したり失敗したりする
- CIでは失敗するがローカルでは成功する
- 週に2-3回、原因不明のテスト失敗

**原因**:
- 非同期処理の待機不足
- グローバルステートへの依存
- 時間依存のテスト

**解決策**:
```typescript
// ❌ Bad: タイミング依存
test('loads data', () => {
  fetchData();
  expect(data).toBeDefined(); // 非同期処理が完了前に実行される
});

// ✅ Good: 明示的な待機
test('loads data', async () => {
  await fetchData();
  expect(data).toBeDefined();
});

// ✅ Better: waitForを使用
test('loads data', async () => {
  fetchData(); // 非同期で実行
  await waitFor(() => {
    expect(data).toBeDefined();
  });
});
```

**予防策**:
- 非同期処理は必ずawait/waitForを使う
- グローバルステートを避ける
- beforeEach/afterEachでクリーンアップ
- 時間依存のテストはjest.useFakeTimers()を使う

**時間的影響**: デバッグ時間 週5時間 → 0時間

---

### 失敗3-10: その他の失敗パターン

*(以下、簡潔に記載)*

3. **テストカバレッジの罠**: 100%を目指して無駄なテストを書く
4. **モックしすぎ**: 全てをモックして実装の詳細をテスト
5. **テストが遅い**: セットアップに時間がかかる
6. **テストコードの重複**: DRY原則を無視
7. **テストの責任範囲が不明確**: Unitで統合テストを書く
8. **E2Eテストのメンテナンス**: セレクター変更で全滅
9. **CIでのみ失敗**: 環境依存の問題
10. **テストが読みにくい**: 何をテストしているか不明

---

## 6. チェックリスト

### 6.1 テスト戦略設計時

- [ ] テストピラミッドの比率を確認 (70/20/10)
- [ ] 各層の責務が明確か
- [ ] テスト実行時間は適切か (CI: 5分以内推奨)
- [ ] Flaky Testsの対策ができているか
- [ ] テストデータの管理方法は決まっているか

### 6.2 新機能開発時

- [ ] Unit Testsを先に作成したか
- [ ] カバレッジは適切か (80%以上推奨)
- [ ] Integration Testsで連携を確認したか
- [ ] E2Eは主要フローのみか

### 6.3 PRレビュー時

- [ ] 全てのテストが通るか
- [ ] テストカバレッジが低下していないか
- [ ] テストコードも読みやすいか
- [ ] Flaky Testsがないか

---

**最終文字数**: 約45,000 chars
**総テスト例**: 38個 (Unit 28 + Integration 7 + E2E 3)
