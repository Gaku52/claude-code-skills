# APIテスト

> APIテストは品質の最後の砦。統合テスト、コントラクトテスト、負荷テスト、E2Eテストまで、APIの正確性・信頼性・性能を保証するテスト戦略の全体像と実践パターンを習得する。

## この章で学ぶこと

- [ ] APIテストの種類と戦略を理解する
- [ ] supertest / Pact によるテスト実装を把握する
- [ ] 負荷テストとパフォーマンステストの方法を学ぶ

---

## 1. APIテスト戦略

```
テストピラミッド（API版）:

         /\
        /  \     E2E テスト（少数）
       /    \    → 本番に近い環境で全体フロー
      /──────\
     /        \  統合テスト（中程度）
    /          \ → API エンドポイント単位
   /────────────\
  /              \ ユニットテスト（多数）
 /                \ → リゾルバー、バリデーション、ビジネスロジック

テストの種類:
  ① ユニットテスト:
     → バリデーションロジック
     → ビジネスルール
     → データ変換

  ② 統合テスト:
     → エンドポイント単位のリクエスト/レスポンス
     → DB + API の結合テスト
     → 認証・認可のテスト

  ③ コントラクトテスト:
     → API の仕様（契約）通りにレスポンスが返るか
     → Provider（API）と Consumer（クライアント）の合意
     → Pact, Dredd

  ④ E2E テスト:
     → ユーザーシナリオの通しテスト
     → 登録 → ログイン → 注文 → 支払い

  ⑤ 負荷テスト:
     → 性能要件の検証
     → k6, Artillery, Locust
```

---

## 2. 統合テスト（supertest）

```javascript
// __tests__/api/users.test.js
import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import supertest from 'supertest';
import { app } from '../../src/app';
import { db } from '../../src/db';

const request = supertest(app);

describe('Users API', () => {
  let authToken;

  beforeAll(async () => {
    await db.migrate.latest();
  });

  beforeEach(async () => {
    await db('users').truncate();
    // テスト用ユーザーとトークンを作成
    const user = await db('users').insert({
      id: 'user_test', name: 'Test', email: 'test@example.com', role: 'admin',
    }).returning('*');
    authToken = generateToken(user[0]);
  });

  afterAll(async () => {
    await db.destroy();
  });

  // --- GET /users ---
  describe('GET /api/v1/users', () => {
    it('should return paginated users', async () => {
      // データ準備
      await db('users').insert([
        { id: 'u1', name: 'Alice', email: 'alice@example.com', role: 'user' },
        { id: 'u2', name: 'Bob', email: 'bob@example.com', role: 'admin' },
      ]);

      const res = await request
        .get('/api/v1/users?limit=10')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(res.body.data).toHaveLength(3); // test user + 2
      expect(res.body.meta).toHaveProperty('hasNextPage');
    });

    it('should filter by role', async () => {
      await db('users').insert([
        { id: 'u1', name: 'Alice', email: 'alice@example.com', role: 'user' },
        { id: 'u2', name: 'Bob', email: 'bob@example.com', role: 'admin' },
      ]);

      const res = await request
        .get('/api/v1/users?filter[role]=admin')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(res.body.data.every(u => u.role === 'admin')).toBe(true);
    });

    it('should return 401 without auth', async () => {
      await request
        .get('/api/v1/users')
        .expect(401);
    });
  });

  // --- POST /users ---
  describe('POST /api/v1/users', () => {
    it('should create a user', async () => {
      const res = await request
        .post('/api/v1/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ name: 'New User', email: 'new@example.com' })
        .expect(201);

      expect(res.body.data).toMatchObject({
        name: 'New User',
        email: 'new@example.com',
        role: 'user',
      });
      expect(res.body.data.id).toBeDefined();
      expect(res.headers.location).toMatch(/\/api\/v1\/users\//);
    });

    it('should return 422 for invalid email', async () => {
      const res = await request
        .post('/api/v1/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ name: 'Test', email: 'invalid' })
        .expect(422);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'email' })
      );
    });

    it('should return 409 for duplicate email', async () => {
      await request
        .post('/api/v1/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ name: 'Test', email: 'dup@example.com' })
        .expect(201);

      await request
        .post('/api/v1/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ name: 'Test2', email: 'dup@example.com' })
        .expect(409);
    });
  });
});
```

---

## 3. コントラクトテスト（Pact）

```javascript
// Consumer側（クライアント）のテスト
import { PactV3 } from '@pact-foundation/pact';

const provider = new PactV3({
  consumer: 'FrontendApp',
  provider: 'UserAPI',
});

describe('User API Contract', () => {
  it('should get a user by ID', async () => {
    // 期待するインタラクションを定義
    provider
      .given('a user with ID 123 exists')
      .uponReceiving('a request to get user 123')
      .withRequest({
        method: 'GET',
        path: '/api/v1/users/123',
        headers: { Authorization: 'Bearer valid-token' },
      })
      .willRespondWith({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: {
          data: {
            id: '123',
            name: like('Taro'),         // 型のみ検証
            email: like('taro@example.com'),
            role: term({ matcher: 'user|admin', generate: 'user' }),
          },
        },
      });

    // テスト実行
    await provider.executeTest(async (mockServer) => {
      const client = new UserClient({ baseUrl: mockServer.url });
      const user = await client.getUser('123');
      expect(user.id).toBe('123');
      expect(user.name).toBeDefined();
    });
  });
});

// Provider側（API）の検証
// → Pact Broker からコントラクトを取得して検証
```

---

## 4. 負荷テスト（k6）

```javascript
// load-test.js（k6）
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 50 },   // ランプアップ
    { duration: '1m', target: 50 },    // 定常状態
    { duration: '30s', target: 100 },  // ピーク
    { duration: '1m', target: 100 },   // ピーク維持
    { duration: '30s', target: 0 },    // ランプダウン
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'], // 95%ile < 500ms
    http_req_failed: ['rate<0.01'],                  // エラー率 < 1%
    http_reqs: ['rate>100'],                         // 100 req/s以上
  },
};

const BASE_URL = 'https://api.example.com/v1';
const TOKEN = __ENV.API_TOKEN;

export default function () {
  // ユーザー一覧取得
  const listRes = http.get(`${BASE_URL}/users?limit=20`, {
    headers: { Authorization: `Bearer ${TOKEN}` },
  });

  check(listRes, {
    'list status is 200': (r) => r.status === 200,
    'list has data': (r) => JSON.parse(r.body).data.length > 0,
    'list response time < 500ms': (r) => r.timings.duration < 500,
  });

  // ユーザー作成
  const createRes = http.post(`${BASE_URL}/users`, JSON.stringify({
    name: `User_${Date.now()}`,
    email: `user_${Date.now()}@example.com`,
  }), {
    headers: {
      Authorization: `Bearer ${TOKEN}`,
      'Content-Type': 'application/json',
    },
  });

  check(createRes, {
    'create status is 201': (r) => r.status === 201,
  });

  sleep(1); // シンクタイム
}

// 実行: k6 run --env API_TOKEN=xxx load-test.js
```

---

## 5. OpenAPI仕様のテスト

```javascript
// Schemathesis: OpenAPI仕様ベースのファジングテスト

// CLI:
// schemathesis run https://api.example.com/openapi.yaml \
//   --auth "Bearer sk_test_xxx" \
//   --stateful=links \
//   --hypothesis-seed=42

// Python API:
// import schemathesis
// schema = schemathesis.from_url("https://api.example.com/openapi.yaml")
// @schema.parametrize()
// def test_api(case):
//     case.call_and_validate()

// OpenAPIスキーマとの整合性テスト（手動）
import { describe, it, expect } from 'vitest';
import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import yaml from 'js-yaml';
import { readFileSync } from 'fs';

const spec = yaml.load(readFileSync('./openapi.yaml', 'utf-8'));
const ajv = new Ajv({ allErrors: true });
addFormats(ajv);

describe('API Response Schema Validation', () => {
  it('GET /users response matches schema', async () => {
    const res = await request.get('/api/v1/users')
      .set('Authorization', `Bearer ${token}`);

    const schema = spec.paths['/users'].get.responses['200']
      .content['application/json'].schema;

    const validate = ajv.compile(resolveRefs(schema, spec));
    const valid = validate(res.body);
    expect(valid).toBe(true);
  });
});
```

---

## 6. テスト環境と戦略

```
テスト環境:
  ✓ テスト用DB（SQLite in-memory / Docker PostgreSQL）
  ✓ テスト用の認証トークン
  ✓ 外部サービスのモック（MSW, WireMock）
  ✓ テストデータのシード/クリーンアップ

テスト戦略チェックリスト:
  □ 正常系: 期待通りのリクエスト → 期待通りのレスポンス
  □ 異常系: 不正な入力 → 適切なエラー
  □ 認証: トークンなし → 401
  □ 認可: 権限なし → 403
  □ ページネーション: cursor/limit の動作確認
  □ フィルタリング: 各フィルタの動作確認
  □ 冪等性: 同じリクエスト2回 → 同じ結果
  □ 競合: 同時更新 → 409 Conflict
  □ レート制限: 制限超過 → 429
  □ 大量データ: 10万件でもパフォーマンス維持
```

---

## まとめ

| テスト種類 | ツール | 目的 |
|-----------|--------|------|
| ユニット | Vitest, Jest | ロジックの検証 |
| 統合 | supertest | エンドポイントの検証 |
| コントラクト | Pact | API仕様の合意 |
| 負荷 | k6, Artillery | パフォーマンスの検証 |
| ファジング | Schemathesis | エッジケースの発見 |

---

## 次に読むべきガイド
→ [[01-monitoring-and-logging.md]] — 監視とロギング

---

## 参考文献
1. k6. "Load Testing for Engineering Teams." k6.io, 2024.
2. Pact. "Contract Testing." pact.io, 2024.
3. Schemathesis. "API Testing with OpenAPI." github.com/schemathesis, 2024.
