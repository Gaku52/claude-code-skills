# Jest セットアップテンプレート

このテンプレートは、Jest を使用したテスト環境を素早くセットアップするためのものです。

## 📦 インストール

```bash
npm install --save-dev jest @types/jest ts-jest
npm install --save-dev @testing-library/react @testing-library/jest-dom
npm install --save-dev @testing-library/user-event
```

## 🚀 セットアップ手順

### 1. 設定ファイルをコピー

```bash
cp jest.config.js your-project/
cp setupTests.ts your-project/src/
cp testUtils.ts your-project/src/
```

### 2. package.json にスクリプト追加

```json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "test:coverage:check": "jest --coverage --coverageThreshold='{\"global\":{\"branches\":80,\"functions\":80,\"lines\":80,\"statements\":80}}'"
  }
}
```

### 3. tsconfig.json の設定

```json
{
  "compilerOptions": {
    "types": ["jest", "@testing-library/jest-dom"]
  }
}
```

## 📁 ファイル構成

```
your-project/
├── jest.config.js          # Jest設定
├── src/
│   ├── setupTests.ts       # テストセットアップ
│   ├── testUtils.ts        # ヘルパー関数
│   └── __tests__/          # テストファイル
│       └── example.test.ts
```

## ✅ 動作確認

```bash
npm test
```

成功すれば、Jest が正しくセットアップされています！

## 📚 含まれる機能

- ✅ TypeScript サポート
- ✅ React Testing Library 統合
- ✅ カバレッジレポート
- ✅ カスタムマッチャー
- ✅ ヘルパー関数（renderWithProviders など）

## 🔧 カスタマイズ

### カバレッジ除外パターンの追加

`jest.config.js` の `coveragePathIgnorePatterns` を編集:

```javascript
coveragePathIgnorePatterns: [
  '/node_modules/',
  '/src/__tests__/',
  '/src/**/*.stories.tsx',
  // 追加のパターン
],
```

### テストタイムアウトの変更

```javascript
// jest.config.js
module.exports = {
  testTimeout: 10000, // 10秒
};
```

## 📖 関連ドキュメント

- [Jest 公式ドキュメント](https://jestjs.io/)
- [React Testing Library](https://testing-library.com/react)
- [jest-dom](https://github.com/testing-library/jest-dom)
