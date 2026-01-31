/**
 * テストユーティリティ関数
 * 共通のヘルパー関数やカスタムレンダー関数を定義
 */

import { render, RenderOptions } from '@testing-library/react';
import { ReactElement, ReactNode } from 'react';

/**
 * テスト用のコンテキストプロバイダー
 * プロジェクトに応じてカスタマイズしてください
 */
interface AllProvidersProps {
  children: ReactNode;
}

function AllProviders({ children }: AllProvidersProps) {
  return (
    <>
      {/* 必要なプロバイダーを追加 */}
      {/* <ThemeProvider theme={theme}> */}
      {/*   <AuthProvider> */}
      {/*     <QueryClientProvider client={queryClient}> */}
      {children}
      {/*     </QueryClientProvider> */}
      {/*   </AuthProvider> */}
      {/* </ThemeProvider> */}
    </>
  );
}

/**
 * カスタムレンダー関数
 * 全てのプロバイダーでラップされたコンポーネントをレンダリング
 *
 * @example
 * const { getByText } = renderWithProviders(<MyComponent />);
 */
export function renderWithProviders(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) {
  return render(ui, {
    wrapper: AllProviders,
    ...options,
  });
}

/**
 * 非同期処理の待機ヘルパー
 * @param ms - 待機時間（ミリ秒）
 */
export const wait = (ms: number) =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * ランダムな文字列を生成
 * @param length - 文字列の長さ
 */
export function randomString(length: number = 10): string {
  return Math.random()
    .toString(36)
    .substring(2, 2 + length);
}

/**
 * ランダムなメールアドレスを生成
 */
export function randomEmail(): string {
  return `${randomString()}@example.com`;
}

/**
 * テスト用のユーザーデータを生成
 */
export function createMockUser(overrides?: Partial<User>) {
  return {
    id: randomString(8),
    name: 'Test User',
    email: randomEmail(),
    createdAt: new Date().toISOString(),
    ...overrides,
  };
}

/**
 * テスト用のAPIレスポンスを生成
 */
export function createMockApiResponse<T>(data: T, overrides?: any) {
  return {
    data,
    status: 200,
    statusText: 'OK',
    headers: {},
    config: {},
    ...overrides,
  };
}

/**
 * テスト用のAPIエラーを生成
 */
export function createMockApiError(
  message: string = 'API Error',
  status: number = 500
) {
  return {
    response: {
      data: { message },
      status,
      statusText: 'Internal Server Error',
      headers: {},
      config: {},
    },
    message,
    config: {},
    isAxiosError: true,
  };
}

/**
 * LocalStorageのヘルパー
 */
export const localStorageHelper = {
  get(key: string): any {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : null;
  },
  set(key: string, value: any): void {
    localStorage.setItem(key, JSON.stringify(value));
  },
  remove(key: string): void {
    localStorage.removeItem(key);
  },
  clear(): void {
    localStorage.clear();
  },
};

/**
 * コンソールのモック化
 * テスト中のconsole出力を抑制
 */
export function suppressConsole() {
  const originalError = console.error;
  const originalWarn = console.warn;

  beforeAll(() => {
    console.error = jest.fn();
    console.warn = jest.fn();
  });

  afterAll(() => {
    console.error = originalError;
    console.warn = originalWarn;
  });
}

/**
 * フォーム入力のヘルパー
 */
export async function fillForm(
  getByLabelText: any,
  fields: Record<string, string>
) {
  const { userEvent } = await import('@testing-library/user-event');
  const user = userEvent.setup();

  for (const [label, value] of Object.entries(fields)) {
    const input = getByLabelText(label);
    await user.clear(input);
    await user.type(input, value);
  }
}

/**
 * ファイルアップロードのヘルパー
 */
export function createMockFile(
  name: string = 'test.png',
  size: number = 1024,
  type: string = 'image/png'
): File {
  const file = new File(['dummy content'], name, { type });
  Object.defineProperty(file, 'size', { value: size });
  return file;
}

/**
 * Date.now() をモック化
 */
export function mockDateNow(timestamp: number) {
  const original = Date.now;

  beforeAll(() => {
    Date.now = jest.fn(() => timestamp);
  });

  afterAll(() => {
    Date.now = original;
  });
}

/**
 * タイマーのヘルパー
 */
export const timerHelper = {
  useFake() {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.runOnlyPendingTimers();
      jest.useRealTimers();
    });
  },
};

/**
 * デバッグヘルパー
 * テストの途中でDOMの状態を確認
 */
export function debugElement(element: any) {
  console.log('=== DEBUG ===');
  console.log('HTML:', element.innerHTML);
  console.log('Text:', element.textContent);
  console.log('============');
}

// 型定義例（プロジェクトに応じて調整）
export interface User {
  id: string;
  name: string;
  email: string;
  createdAt: string;
}

// 全てエクスポート
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';
