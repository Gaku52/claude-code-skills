/**
 * Jest セットアップファイル
 * 全てのテスト実行前に1度だけ実行されます
 */

// jest-dom のカスタムマッチャーをインポート
import '@testing-library/jest-dom';

// グローバルなモック設定
global.console = {
  ...console,
  // テスト中のconsole.errorを抑制（必要に応じて）
  // error: jest.fn(),
  // warn: jest.fn(),
};

// LocalStorageのモック
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock as any;

// SessionStorageのモック
const sessionStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.sessionStorage = sessionStorageMock as any;

// IntersectionObserver のモック
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  takeRecords() {
    return [];
  }
  unobserve() {}
} as any;

// ResizeObserver のモック
global.ResizeObserver = class ResizeObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
} as any;

// matchMedia のモック
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// fetch のグローバルモック（必要に応じてコメントアウト解除）
// global.fetch = jest.fn();

// タイムゾーンを固定（テストの一貫性のため）
process.env.TZ = 'UTC';

// テスト毎のクリーンアップ
beforeEach(() => {
  // LocalStorage/SessionStorage をクリア
  localStorageMock.clear();
  sessionStorageMock.clear();

  // fetchモックのクリア（使用している場合）
  // (global.fetch as jest.Mock).mockClear();
});

// 各テスト後のクリーンアップ
afterEach(() => {
  // すべてのモックをクリア
  jest.clearAllMocks();
});

// カスタムマッチャーの追加例
expect.extend({
  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () =>
          `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () =>
          `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
});

// TypeScript用の型定義拡張
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeWithinRange(floor: number, ceiling: number): R;
    }
  }
}
