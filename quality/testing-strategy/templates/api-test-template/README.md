# API テストテンプレート

REST API のテストを素早く開始するためのテンプレートです。

## 📦 インストール

```bash
npm install --save-dev jest supertest
npm install --save-dev @types/jest @types/supertest
npm install --save-dev ts-jest
```

## 🚀 セットアップ手順

### 1. ファイルをコピー

```bash
cp setup.ts your-project/tests/
cp example.test.ts your-project/tests/
```

### 2. package.json にスクリプト追加

```json
{
  "scripts": {
    "test:api": "jest --testPathPattern=tests/.*\\.test\\.ts",
    "test:api:watch": "jest --watch --testPathPattern=tests/.*\\.test\\.ts"
  }
}
```

## 📁 ファイル構成

```
your-project/
├── tests/
│   ├── setup.ts           # テストセットアップ
│   ├── example.test.ts    # テスト例
│   └── helpers/           # ヘルパー関数
│       ├── auth.ts
│       └── db.ts
└── src/
    └── app.ts             # Expressアプリ
```

## ✅ 使用例

### 基本的なGETリクエスト

```typescript
import request from 'supertest';
import app from '../src/app';

describe('GET /api/users', () => {
  it('should return users list', async () => {
    const response = await request(app)
      .get('/api/users')
      .expect(200);

    expect(response.body).toHaveProperty('users');
    expect(Array.isArray(response.body.users)).toBe(true);
  });
});
```

### POSTリクエスト + 認証

```typescript
describe('POST /api/posts', () => {
  it('should create a new post', async () => {
    const token = await getAuthToken();

    const response = await request(app)
      .post('/api/posts')
      .set('Authorization', `Bearer ${token}`)
      .send({
        title: 'Test Post',
        content: 'This is a test post',
      })
      .expect(201);

    expect(response.body).toHaveProperty('id');
    expect(response.body.title).toBe('Test Post');
  });
});
```

## 📚 含まれる機能

- ✅ Supertest による API テスト
- ✅ データベースのセットアップ/クリーンアップ
- ✅ 認証ヘルパー
- ✅ テストデータファクトリー
- ✅ エラーハンドリングテスト

## 🔧 カスタマイズ

### データベース接続の変更

```typescript
// setup.ts
beforeAll(async () => {
  await mongoose.connect(process.env.MONGO_TEST_URL);
});

afterAll(async () => {
  await mongoose.connection.close();
});
```

### 認証方式の変更

```typescript
// helpers/auth.ts
export async function getAuthToken() {
  // JWT, OAuth, etc.
  return 'your-auth-token';
}
```

## 📖 関連ドキュメント

- [Supertest](https://github.com/visionmedia/supertest)
- [Jest](https://jestjs.io/)
- [Express Testing Best Practices](https://github.com/goldbergyoni/nodebestpractices#-62-component-testing)
