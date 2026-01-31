# Testing Library ヘルパー集

React Testing Library を使用する際の便利なヘルパー関数とカスタムマッチャーのコレクションです。

## 📦 インストール

```bash
npm install --save-dev @testing-library/react @testing-library/jest-dom
npm install --save-dev @testing-library/user-event
```

## 🚀 使い方

### 1. ファイルをプロジェクトにコピー

```bash
cp render-with-providers.tsx your-project/src/test/
cp custom-matchers.ts your-project/src/test/
```

### 2. テストでインポート

```typescript
import { renderWithProviders } from '@/test/render-with-providers';
import '@/test/custom-matchers';

describe('MyComponent', () => {
  it('should render', () => {
    const { getByText } = renderWithProviders(<MyComponent />);
    expect(getByText('Hello')).toBeInTheDocument();
  });
});
```

## 📚 含まれる機能

### render-with-providers.tsx
- ✅ プロバイダーでラップしたカスタムレンダー
- ✅ React Router 統合
- ✅ Redux / Context API サポート
- ✅ React Query 統合

### custom-matchers.ts
- ✅ カスタムマッチャー
- ✅ アクセシビリティチェック
- ✅ フォーム検証ヘルパー

## 🔧 カスタマイズ

プロジェクトに応じて、プロバイダーを追加・変更してください。

### 例: Themeプロバイダー追加

```typescript
// render-with-providers.tsx
import { ThemeProvider } from '@mui/material/styles';

function AllProviders({ children }: AllProvidersProps) {
  return (
    <ThemeProvider theme={theme}>
      {children}
    </ThemeProvider>
  );
}
```

## 📖 関連ドキュメント

- [React Testing Library](https://testing-library.com/react)
- [jest-dom](https://github.com/testing-library/jest-dom)
- [user-event](https://testing-library.com/docs/user-event/intro)
