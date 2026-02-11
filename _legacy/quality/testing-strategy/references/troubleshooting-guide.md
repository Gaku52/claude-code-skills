# テストトラブルシューティングガイド

**目的**: テスト実行中によく発生する問題とその解決方法をまとめたリファレンス

---

## 目次

1. [Jest/Vitest 実行エラー](#1-jestvitest-実行エラー)
2. [React Testing Library エラー](#2-react-testing-library-エラー)
3. [非同期テストの問題](#3-非同期テストの問題)
4. [モックの問題](#4-モックの問題)
5. [環境設定の問題](#5-環境設定の問題)
6. [カバレッジの問題](#6-カバレッジの問題)
7. [CI/CD環境での問題](#7-cicd環境での問題)
8. [パフォーマンスの問題](#8-パフォーマンスの問題)

---

## 1. Jest/Vitest 実行エラー

### 問題: `Cannot find module` エラー

**エラーメッセージ**:
```
Cannot find module '@/utils/helper' from 'src/components/Button.test.ts'
```

**原因**:
- モジュールパスのエイリアスが設定されていない
- tsconfig.json の paths 設定が jest.config.js に反映されていない

**解決策**:

```javascript
// jest.config.js
module.exports = {
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
};
```

または、`ts-jest` を使用:

```javascript
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
};
```

---

### 問題: `SyntaxError: Unexpected token 'export'`

**エラーメッセージ**:
```
SyntaxError: Unexpected token 'export'
```

**原因**:
- ES Modules がトランスパイルされていない
- node_modules 内の ESM パッケージが変換されていない

**解決策**:

```javascript
// jest.config.js
module.exports = {
  transformIgnorePatterns: [
    'node_modules/(?!(module-to-transform)/)', // 特定モジュールのみ変換
  ],
};
```

または Vitest を使用（ESM ネイティブサポート）:

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // ESM サポート
  },
});
```

---

### 問題: `ReferenceError: regeneratorRuntime is not defined`

**原因**:
- async/await がトランスパイルされていない

**解決策**:

```bash
npm install --save-dev @babel/preset-env
```

```javascript
// babel.config.js
module.exports = {
  presets: [
    ['@babel/preset-env', { targets: { node: 'current' } }],
    '@babel/preset-typescript',
  ],
};
```

---

## 2. React Testing Library エラー

### 問題: `Unable to find an element with the text`

**エラーメッセージ**:
```
TestingLibraryElementError: Unable to find an element with the text: /hello/i
```

**原因**:
- 要素が存在しない
- 要素がまだレンダリングされていない（非同期）
- セレクタが間違っている

**解決策**:

```typescript
// ❌ 悪い例
const element = getByText('Hello');

// ✅ 良い例1: queryByText で存在チェック
const element = queryByText('Hello');
expect(element).toBeNull(); // 存在しないことを確認

// ✅ 良い例2: 非同期で待機
const element = await findByText('Hello'); // 自動的に待機

// ✅ 良い例3: waitFor を使用
await waitFor(() => {
  expect(getByText('Hello')).toBeInTheDocument();
});

// ✅ 良い例4: デバッグ
screen.debug(); // 現在のDOMを確認
```

---

### 問題: `Not wrapped in act(...)`

**エラーメッセージ**:
```
Warning: An update to Component inside a test was not wrapped in act(...)
```

**原因**:
- 非同期の状態更新が完了する前にテストが終了
- useEffect などの副作用が処理されていない

**解決策**:

```typescript
// ❌ 悪い例
it('updates state', () => {
  const { getByText } = render(<Counter />);
  fireEvent.click(getByText('Increment'));
  // 状態更新が完了していない
});

// ✅ 良い例: waitFor を使用
it('updates state', async () => {
  const { getByText } = render(<Counter />);
  fireEvent.click(getByText('Increment'));
  await waitFor(() => {
    expect(getByText('Count: 1')).toBeInTheDocument();
  });
});

// ✅ 良い例: userEvent を使用（自動的にact）
it('updates state', async () => {
  const user = userEvent.setup();
  const { getByText } = render(<Counter />);
  await user.click(getByText('Increment'));
  expect(getByText('Count: 1')).toBeInTheDocument();
});
```

---

### 問題: `Cannot read property 'click' of null`

**原因**:
- 要素が存在しない
- セレクタが間違っている

**解決策**:

```typescript
// ❌ 悪い例
fireEvent.click(getByText('Submit')); // 存在しないとエラー

// ✅ 良い例1: まず存在を確認
const button = getByText('Submit');
expect(button).toBeInTheDocument();
fireEvent.click(button);

// ✅ 良い例2: data-testid を使用
<button data-testid="submit-button">Submit</button>

const button = getByTestId('submit-button');
fireEvent.click(button);

// ✅ 良い例3: デバッグ
screen.debug(); // DOMを確認
```

---

## 3. 非同期テストの問題

### 問題: `Timeout - Async callback was not invoked`

**エラーメッセージ**:
```
Timeout - Async callback was not invoked within the 5000 ms timeout
```

**原因**:
- Promise が resolve/reject されない
- done() が呼ばれない
- テストが長時間かかる

**解決策**:

```typescript
// ❌ 悪い例: done() を忘れる
it('async test', (done) => {
  fetchData().then((data) => {
    expect(data).toBe('result');
    // done() を忘れている
  });
});

// ✅ 良い例1: async/await を使用
it('async test', async () => {
  const data = await fetchData();
  expect(data).toBe('result');
});

// ✅ 良い例2: タイムアウトを延長
it('slow async test', async () => {
  const data = await slowFetchData();
  expect(data).toBe('result');
}, 10000); // 10秒に延長
```

---

### 問題: Promise が resolve しない

**原因**:
- モックが正しく設定されていない
- API呼び出しが失敗している

**解決策**:

```typescript
// ❌ 悪い例: モックが不完全
const mockFetch = jest.fn();
// mockReturnValue を忘れている

// ✅ 良い例: 正しくモック
const mockFetch = jest.fn().mockResolvedValue({
  data: 'result',
});

// または
const mockFetch = jest.fn().mockImplementation(() =>
  Promise.resolve({ data: 'result' })
);

// エラーをモック
const mockFetch = jest.fn().mockRejectedValue(
  new Error('API Error')
);
```

---

## 4. モックの問題

### 問題: `jest.mock()` が効かない

**原因**:
- モックの定義場所が間違っている
- モジュールのインポート順序が間違っている

**解決策**:

```typescript
// ❌ 悪い例: インポート後にモック
import { apiService } from './api';
jest.mock('./api'); // 遅すぎる

// ✅ 良い例: インポート前にモック
jest.mock('./api');
import { apiService } from './api';

describe('Service', () => {
  it('should call API', () => {
    apiService.fetch();
    expect(apiService.fetch).toHaveBeenCalled();
  });
});
```

---

### 問題: `mockResolvedValue` が効かない

**原因**:
- モックの設定が不完全
- モック対象が間違っている

**解決策**:

```typescript
// ❌ 悪い例: モジュール全体をモック
jest.mock('./api');

// ✅ 良い例: 特定の関数をモック
jest.mock('./api', () => ({
  apiService: {
    fetch: jest.fn().mockResolvedValue({ data: 'test' }),
  },
}));

// または spyOn を使用
import * as api from './api';
jest.spyOn(api, 'fetchData').mockResolvedValue('result');
```

---

### 問題: モックがリセットされない

**原因**:
- `clearMocks`/`resetMocks` が設定されていない

**解決策**:

```javascript
// jest.config.js
module.exports = {
  clearMocks: true, // 各テスト後にモックをクリア
  resetMocks: false,
  restoreMocks: true,
};
```

または手動でリセット:

```typescript
describe('Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('test 1', () => { /* ... */ });
  it('test 2', () => { /* ... */ });
});
```

---

## 5. 環境設定の問題

### 問題: `localStorage is not defined`

**原因**:
- テスト環境が Node.js で、DOM API が存在しない

**解決策**:

```javascript
// jest.config.js
module.exports = {
  testEnvironment: 'jsdom', // ブラウザ環境をエミュレート
};
```

または手動でモック:

```typescript
// setupTests.ts
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock as any;
```

---

### 問題: CSS/画像ファイルのインポートエラー

**エラーメッセージ**:
```
SyntaxError: Unexpected token
  import './styles.css';
```

**解決策**:

```javascript
// jest.config.js
module.exports = {
  moduleNameMapper: {
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.(jpg|jpeg|png|gif|svg)$': '<rootDir>/__mocks__/fileMock.js',
  },
};
```

```javascript
// __mocks__/fileMock.js
module.exports = 'test-file-stub';
```

---

## 6. カバレッジの問題

### 問題: カバレッジが正しく表示されない

**原因**:
- `collectCoverageFrom` の設定が間違っている
- 除外パターンが広すぎる

**解決策**:

```javascript
// jest.config.js
module.exports = {
  collectCoverage: true,
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/*.stories.tsx',
    '!src/**/__tests__/**',
    '!src/main.tsx',
  ],
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
  ],
};
```

---

### 問題: カバレッジが100%なのにファイルが未テスト

**原因**:
- ファイルが `collectCoverageFrom` に含まれていない

**解決策**:

```bash
# カバレッジ対象を確認
npm test -- --coverage --collectCoverageFrom='src/**/*.ts'
```

---

## 7. CI/CD環境での問題

### 問題: ローカルでは成功、CIでは失敗

**原因**:
- 環境変数が設定されていない
- タイムゾーンの違い
- ファイルパスの違い（Windows vs Unix）

**解決策**:

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      TZ: UTC # タイムゾーン固定
      NODE_ENV: test
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: npm test
```

```typescript
// テスト内でタイムゾーン固定
process.env.TZ = 'UTC';
```

---

### 問題: CI環境でテストがタイムアウト

**原因**:
- CI環境が遅い
- 並列実行されていない

**解決策**:

```javascript
// jest.config.js
module.exports = {
  testTimeout: 10000, // CI用に延長
  maxWorkers: '50%', // 並列実行
};
```

---

## 8. パフォーマンスの問題

### 問題: テストが遅い

**原因**:
- 不要なセットアップ
- モックされていない外部呼び出し
- ファイルの再読み込み

**解決策**:

```typescript
// ❌ 悪い例: 毎回DB初期化
beforeEach(async () => {
  await setupDatabase(); // 遅い
});

// ✅ 良い例: 一度だけ初期化
beforeAll(async () => {
  await setupDatabase();
});

beforeEach(async () => {
  await cleanupDatabase(); // クリーンアップのみ
});
```

---

### 問題: メモリリーク

**原因**:
- テスト後のクリーンアップ不足
- グローバル変数の蓄積

**解決策**:

```typescript
describe('Service', () => {
  let service;

  beforeEach(() => {
    service = new Service();
  });

  afterEach(() => {
    service.cleanup(); // リソース解放
    service = null;
  });
});
```

---

## クイックリファレンス

### よくあるエラーと対処法

| エラー | 原因 | 解決策 |
|-------|------|--------|
| `Cannot find module` | パスエイリアス未設定 | `moduleNameMapper` 設定 |
| `Unexpected token 'export'` | ESM未変換 | `transformIgnorePatterns` 設定 |
| `Unable to find element` | 要素が存在しない | `findBy*` / `waitFor` 使用 |
| `Not wrapped in act` | 非同期更新未完了 | `waitFor` / `userEvent` 使用 |
| `Timeout` | Promise未解決 | `async/await` / タイムアウト延長 |
| `mock()` が効かない | モック順序間違い | インポート前にモック |
| `localStorage undefined` | DOM API不足 | `testEnvironment: 'jsdom'` |

---

**最終更新**: 2026-01-02
**関連ドキュメント**:
- [Common Testing Failures](./common-testing-failures.md)
