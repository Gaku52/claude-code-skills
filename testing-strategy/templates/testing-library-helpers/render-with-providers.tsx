/**
 * カスタムレンダー関数
 * 全てのプロバイダーでラップされたコンポーネントをレンダリング
 */

import { render, RenderOptions } from '@testing-library/react';
import { ReactElement, ReactNode } from 'react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

/**
 * テスト用の QueryClient 設定
 * リトライを無効化して高速化
 */
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        cacheTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });

/**
 * プロバイダーのプロップス
 */
interface AllProvidersProps {
  children: ReactNode;
}

/**
 * 全プロバイダーを統合
 * プロジェクトに応じてカスタマイズ
 */
function AllProviders({ children }: AllProvidersProps) {
  const queryClient = createTestQueryClient();

  return (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        {/* 他のプロバイダーを追加 */}
        {/* <ThemeProvider theme={theme}> */}
        {/*   <AuthProvider> */}
        {children}
        {/*   </AuthProvider> */}
        {/* </ThemeProvider> */}
      </QueryClientProvider>
    </BrowserRouter>
  );
}

/**
 * カスタムレンダー関数のオプション
 */
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  // 初期ルート（React Router使用時）
  initialRoute?: string;
  // カスタム QueryClient
  queryClient?: QueryClient;
}

/**
 * カスタムレンダー関数
 *
 * @example
 * // 基本的な使用
 * const { getByText } = renderWithProviders(<MyComponent />);
 *
 * @example
 * // 初期ルートを指定
 * const { getByText } = renderWithProviders(<MyComponent />, {
 *   initialRoute: '/dashboard',
 * });
 */
export function renderWithProviders(
  ui: ReactElement,
  {
    initialRoute = '/',
    queryClient,
    ...renderOptions
  }: CustomRenderOptions = {}
) {
  // 初期ルートを設定
  if (initialRoute !== '/') {
    window.history.pushState({}, 'Test page', initialRoute);
  }

  // カスタム QueryClient があればそれを使用
  const testQueryClient = queryClient || createTestQueryClient();

  const Wrapper = ({ children }: { children: ReactNode }) => (
    <BrowserRouter>
      <QueryClientProvider client={testQueryClient}>
        {children}
      </QueryClientProvider>
    </BrowserRouter>
  );

  return render(ui, {
    wrapper: Wrapper,
    ...renderOptions,
  });
}

/**
 * Reduxストア付きレンダー
 * Redux を使用している場合
 */
// import { Provider } from 'react-redux';
// import { configureStore } from '@reduxjs/toolkit';

// export function renderWithRedux(
//   ui: ReactElement,
//   {
//     preloadedState = {},
//     store = configureStore({
//       reducer: rootReducer,
//       preloadedState,
//     }),
//     ...renderOptions
//   }: any = {}
// ) {
//   const Wrapper = ({ children }: { children: ReactNode }) => (
//     <Provider store={store}>{children}</Provider>
//   );

//   return render(ui, { wrapper: Wrapper, ...renderOptions });
// }

/**
 * Next.js 用のカスタムレンダー
 */
// import { RouterContext } from 'next/dist/shared/lib/router-context';
// import { NextRouter } from 'next/router';

// export function renderWithNextRouter(
//   ui: ReactElement,
//   {
//     router = {},
//     ...renderOptions
//   }: { router?: Partial<NextRouter> } & RenderOptions = {}
// ) {
//   const mockRouter: NextRouter = {
//     basePath: '',
//     pathname: '/',
//     route: '/',
//     asPath: '/',
//     query: {},
//     push: jest.fn(),
//     replace: jest.fn(),
//     reload: jest.fn(),
//     back: jest.fn(),
//     prefetch: jest.fn(),
//     beforePopState: jest.fn(),
//     events: {
//       on: jest.fn(),
//       off: jest.fn(),
//       emit: jest.fn(),
//     },
//     isFallback: false,
//     isLocaleDomain: false,
//     isReady: true,
//     isPreview: false,
//     ...router,
//   };

//   return render(
//     <RouterContext.Provider value={mockRouter}>
//       {ui}
//     </RouterContext.Provider>,
//     renderOptions
//   );
// }

// 全てエクスポート
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';
