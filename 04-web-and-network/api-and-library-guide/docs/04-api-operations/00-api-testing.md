# APIテスト

> APIテストは品質の最後の砦。単体テスト、統合テスト、コントラクトテスト、負荷テスト、E2Eテストまで、APIの正確性・信頼性・性能を保証するテスト戦略の全体像と実践パターンを習得する。

## この章で学ぶこと

- [ ] APIテストの種類と戦略（テストピラミッド）を理解する
- [ ] supertest / Jest / Vitest による単体テスト・統合テストを実装できる
- [ ] Pact によるコントラクトテストの原理と実装を把握する
- [ ] k6 / Artillery による負荷テスト・パフォーマンステストを設計・実行できる
- [ ] Postman / Newman を用いたテスト自動化の手法を身につける
- [ ] E2Eテストと統合テストの境界を理解し、適切な粒度でテストを書ける
- [ ] CI/CDパイプラインへのテスト組み込み手法を学ぶ

---

## 1. APIテストの全体像

### 1.1 テストピラミッドとAPIテストの位置づけ

ソフトウェアテストにおいて、テストピラミッドは各レベルのテストの理想的な割合を示す概念モデルである。APIテストにおいても同様のピラミッド構造が適用され、下層ほど実行速度が速く、数が多く、上層ほど実行コストが高いが現実に近い検証が可能となる。

```
テストピラミッド（API版）

                  /\
                 /  \        E2E テスト（少数: 5-10%）
                /    \       ・本番同等環境で全体フローを検証
               /      \     ・ユーザーシナリオ単位（登録→購入→確認）
              /--------\    ・実行時間: 数分〜数十分
             /          \
            /  統合テスト  \   統合テスト（中程度: 20-30%）
           /   (API層)    \  ・エンドポイント単位のリクエスト/レスポンス
          /                \ ・DB + API + 認証の結合検証
         /------------------\・実行時間: 数秒〜数十秒
        /                    \
       / コントラクトテスト    \  コントラクトテスト（中程度: 10-15%）
      /                        \ ・API仕様の合意検証
     /--------------------------\・Consumer-Provider間の契約
    /                            \
   /    ユニットテスト（多数）     \  ユニットテスト（最多: 50-60%）
  /    バリデーション/ビジネス      \ ・バリデーション、変換、計算ロジック
 /    ロジック/データ変換            \・モック/スタブ活用、DB不要
/------------------------------------\・実行時間: ミリ秒単位
```

### 1.2 テスト種別の詳細分類

APIテストは目的と粒度によって以下の6種類に大別される。

```
APIテスト種別マップ

+------------------------------------------------------------------+
|                    APIテストの種類                                  |
+------------------------------------------------------------------+
|                                                                    |
|  [1] ユニットテスト        [2] 統合テスト                           |
|  +-----------------------+ +--------------------------+           |
|  | ・バリデーション関数   | | ・HTTP リクエスト/レスポンス|          |
|  | ・ビジネスルール計算   | | ・DB 読み書き含む検証     |           |
|  | ・データ変換・整形     | | ・認証/認可フロー         |           |
|  | ・エラーハンドリング   | | ・ミドルウェア連携         |           |
|  +-----------------------+ +--------------------------+           |
|                                                                    |
|  [3] コントラクトテスト    [4] E2Eテスト                            |
|  +-----------------------+ +--------------------------+           |
|  | ・スキーマ整合性       | | ・複数API横断シナリオ     |           |
|  | ・Consumer-Provider   | | ・外部サービス連携        |           |
|  | ・バージョン互換性     | | ・データ一貫性の検証      |           |
|  +-----------------------+ +--------------------------+           |
|                                                                    |
|  [5] 負荷テスト            [6] セキュリティテスト                   |
|  +-----------------------+ +--------------------------+           |
|  | ・スループット測定     | | ・認証バイパス検証        |           |
|  | ・レイテンシ分析       | | ・インジェクション検証    |           |
|  | ・スケーラビリティ検証 | | ・レート制限検証          |           |
|  | ・障害耐性テスト       | | ・入力バリデーション      |           |
|  +-----------------------+ +--------------------------+           |
+------------------------------------------------------------------+
```

### 1.3 テスト戦略の設計原則

APIテスト戦略を設計する際の基本原則は以下の通りである。

**原則1: テストの独立性**
各テストケースは他のテストに依存してはならない。テスト実行順序が変わっても結果が変わらないことが求められる。

**原則2: テストデータの管理**
テストごとにデータをセットアップし、終了時にクリーンアップする。共有状態を避けることで、テストの信頼性を確保する。

**原則3: 適切な粒度の選択**
テストピラミッドに従い、高速に実行できるユニットテストを最も多く、実行コストの高いE2Eテストを最小限にする。

**原則4: 決定論的なテスト**
日時やランダム値に依存するテストは、固定値を注入できる設計にする。flaky test（不安定なテスト）を生まないことが重要である。

**原則5: 境界値とエッジケースの網羅**
正常系だけでなく、空文字列、null値、巨大データ、特殊文字、同時アクセスなどのエッジケースを意識的にテストする。

---

## 2. ユニットテストの実践

### 2.1 バリデーションロジックのテスト

ユニットテストはAPIの最も基礎的なテスト層であり、DBやネットワークに依存しない純粋な関数やクラスのロジックを検証する。

```javascript
// src/validators/userValidator.js
export class UserValidator {
  static validateEmail(email) {
    if (!email || typeof email !== 'string') {
      return { valid: false, error: 'メールアドレスは必須です' };
    }
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    if (!emailRegex.test(email)) {
      return { valid: false, error: 'メールアドレスの形式が不正です' };
    }
    if (email.length > 254) {
      return { valid: false, error: 'メールアドレスが長すぎます（最大254文字）' };
    }
    return { valid: true, error: null };
  }

  static validateAge(age) {
    if (age === undefined || age === null) {
      return { valid: false, error: '年齢は必須です' };
    }
    if (!Number.isInteger(age)) {
      return { valid: false, error: '年齢は整数で指定してください' };
    }
    if (age < 0 || age > 150) {
      return { valid: false, error: '年齢は0〜150の範囲で指定してください' };
    }
    return { valid: true, error: null };
  }

  static validateCreateUserInput(input) {
    const errors = [];
    const emailResult = this.validateEmail(input.email);
    if (!emailResult.valid) errors.push({ field: 'email', message: emailResult.error });

    const ageResult = this.validateAge(input.age);
    if (!ageResult.valid) errors.push({ field: 'age', message: ageResult.error });

    if (!input.name || input.name.trim().length === 0) {
      errors.push({ field: 'name', message: '名前は必須です' });
    } else if (input.name.length > 100) {
      errors.push({ field: 'name', message: '名前は100文字以内で指定してください' });
    }

    return { valid: errors.length === 0, errors };
  }
}
```

```javascript
// __tests__/unit/userValidator.test.js
import { describe, it, expect } from 'vitest';
import { UserValidator } from '../../src/validators/userValidator';

describe('UserValidator', () => {
  // === メールアドレスバリデーション ===
  describe('validateEmail', () => {
    // 正常系
    it('有効なメールアドレスを受け入れる', () => {
      const testCases = [
        'user@example.com',
        'user.name@example.co.jp',
        'user+tag@example.com',
        'user123@sub.domain.example.com',
      ];
      testCases.forEach(email => {
        const result = UserValidator.validateEmail(email);
        expect(result.valid).toBe(true);
        expect(result.error).toBeNull();
      });
    });

    // 異常系
    it('不正なメールアドレスを拒否する', () => {
      const testCases = [
        { input: '', expected: 'メールアドレスは必須です' },
        { input: null, expected: 'メールアドレスは必須です' },
        { input: undefined, expected: 'メールアドレスは必須です' },
        { input: 'invalid', expected: 'メールアドレスの形式が不正です' },
        { input: '@example.com', expected: 'メールアドレスの形式が不正です' },
        { input: 'user@', expected: 'メールアドレスの形式が不正です' },
        { input: 'user@.com', expected: 'メールアドレスの形式が不正です' },
      ];
      testCases.forEach(({ input, expected }) => {
        const result = UserValidator.validateEmail(input);
        expect(result.valid).toBe(false);
        expect(result.error).toBe(expected);
      });
    });

    // 境界値
    it('254文字を超えるメールアドレスを拒否する', () => {
      const longEmail = 'a'.repeat(243) + '@example.com'; // 255文字
      const result = UserValidator.validateEmail(longEmail);
      expect(result.valid).toBe(false);
      expect(result.error).toBe('メールアドレスが長すぎます（最大254文字）');
    });

    it('254文字ちょうどのメールアドレスを受け入れる', () => {
      const email = 'a'.repeat(242) + '@example.com'; // 254文字
      const result = UserValidator.validateEmail(email);
      expect(result.valid).toBe(true);
    });
  });

  // === 年齢バリデーション ===
  describe('validateAge', () => {
    it('有効な年齢を受け入れる', () => {
      [0, 1, 25, 100, 150].forEach(age => {
        expect(UserValidator.validateAge(age).valid).toBe(true);
      });
    });

    it('境界外の年齢を拒否する', () => {
      expect(UserValidator.validateAge(-1).valid).toBe(false);
      expect(UserValidator.validateAge(151).valid).toBe(false);
    });

    it('非整数を拒否する', () => {
      expect(UserValidator.validateAge(25.5).valid).toBe(false);
      expect(UserValidator.validateAge('25').valid).toBe(false);
    });
  });

  // === 複合バリデーション ===
  describe('validateCreateUserInput', () => {
    it('全フィールドが有効な場合にtrueを返す', () => {
      const result = UserValidator.validateCreateUserInput({
        name: 'Taro Yamada',
        email: 'taro@example.com',
        age: 30,
      });
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('複数フィールドのエラーを同時に返す', () => {
      const result = UserValidator.validateCreateUserInput({
        name: '',
        email: 'invalid',
        age: -5,
      });
      expect(result.valid).toBe(false);
      expect(result.errors).toHaveLength(3);
      expect(result.errors.map(e => e.field)).toEqual(
        expect.arrayContaining(['name', 'email', 'age'])
      );
    });
  });
});
```

### 2.2 ビジネスロジックのテスト

```javascript
// src/services/pricingService.js
export class PricingService {
  /**
   * 商品の最終価格を計算する
   * @param {number} basePrice - 基本価格
   * @param {string} membershipTier - 会員ランク ('bronze'|'silver'|'gold'|'platinum')
   * @param {number} quantity - 数量
   * @param {string|null} couponCode - クーポンコード
   * @returns {{ finalPrice: number, discount: number, breakdown: object }}
   */
  static calculatePrice(basePrice, membershipTier, quantity, couponCode = null) {
    if (basePrice < 0) throw new Error('基本価格は0以上である必要があります');
    if (quantity < 1) throw new Error('数量は1以上である必要があります');

    // 会員割引率
    const tierDiscounts = {
      bronze: 0,
      silver: 0.05,
      gold: 0.10,
      platinum: 0.15,
    };

    // 数量割引率
    let quantityDiscount = 0;
    if (quantity >= 100) quantityDiscount = 0.10;
    else if (quantity >= 50) quantityDiscount = 0.07;
    else if (quantity >= 10) quantityDiscount = 0.05;

    // クーポン割引
    const couponDiscounts = {
      'SUMMER2024': 0.20,
      'WELCOME10': 0.10,
      'VIP30': 0.30,
    };
    const couponDiscount = couponCode ? (couponDiscounts[couponCode] || 0) : 0;

    // 割引は加算ではなく、最大の割引を適用
    const tierRate = tierDiscounts[membershipTier] || 0;
    const maxDiscount = Math.max(tierRate, quantityDiscount, couponDiscount);

    const subtotal = basePrice * quantity;
    const discountAmount = Math.round(subtotal * maxDiscount);
    const finalPrice = subtotal - discountAmount;

    return {
      finalPrice,
      discount: discountAmount,
      breakdown: {
        basePrice,
        quantity,
        subtotal,
        tierDiscount: tierRate,
        quantityDiscount,
        couponDiscount,
        appliedDiscount: maxDiscount,
      },
    };
  }
}
```

```javascript
// __tests__/unit/pricingService.test.js
import { describe, it, expect } from 'vitest';
import { PricingService } from '../../src/services/pricingService';

describe('PricingService.calculatePrice', () => {
  it('基本的な価格計算（割引なし）', () => {
    const result = PricingService.calculatePrice(1000, 'bronze', 1);
    expect(result.finalPrice).toBe(1000);
    expect(result.discount).toBe(0);
  });

  it('会員割引の適用', () => {
    const result = PricingService.calculatePrice(1000, 'gold', 1);
    // gold: 10%割引 → 1000 - 100 = 900
    expect(result.finalPrice).toBe(900);
    expect(result.discount).toBe(100);
    expect(result.breakdown.appliedDiscount).toBe(0.10);
  });

  it('数量割引の適用（10個以上）', () => {
    const result = PricingService.calculatePrice(100, 'bronze', 10);
    // 100 * 10 = 1000, 5%割引 → 1000 - 50 = 950
    expect(result.finalPrice).toBe(950);
  });

  it('クーポン割引が会員割引より大きい場合はクーポンを適用', () => {
    const result = PricingService.calculatePrice(1000, 'silver', 1, 'SUMMER2024');
    // silver: 5%, クーポン: 20% → 最大の20%を適用
    expect(result.finalPrice).toBe(800);
    expect(result.breakdown.appliedDiscount).toBe(0.20);
  });

  it('無効なクーポンコードは無視される', () => {
    const result = PricingService.calculatePrice(1000, 'bronze', 1, 'INVALID');
    expect(result.finalPrice).toBe(1000);
    expect(result.breakdown.couponDiscount).toBe(0);
  });

  it('負の価格でエラーを投げる', () => {
    expect(() => PricingService.calculatePrice(-100, 'bronze', 1))
      .toThrow('基本価格は0以上である必要があります');
  });

  it('数量0でエラーを投げる', () => {
    expect(() => PricingService.calculatePrice(1000, 'bronze', 0))
      .toThrow('数量は1以上である必要があります');
  });
});
```

---

## 3. 統合テストの実践（supertest）

### 3.1 テスト環境のセットアップ

統合テストではHTTPリクエストを実際に送信し、エンドポイントの動作を検証する。supertestはNode.jsのHTTPサーバーに対してリクエストを送信するためのライブラリであり、Express / Koa / Fastify などのフレームワークと組み合わせて使用する。

```javascript
// __tests__/setup/testServer.js
import { beforeAll, afterAll, beforeEach } from 'vitest';
import { app } from '../../src/app';
import { db } from '../../src/db';
import { createTestUser, generateToken } from './helpers';

// テスト用のサーバーとDB接続を管理
export function setupTestServer() {
  let server;
  let authToken;
  let adminToken;
  let testUser;
  let adminUser;

  beforeAll(async () => {
    // テスト用DBのマイグレーション実行
    await db.migrate.latest();
    // テスト用シードデータ投入
    await db.seed.run();
  });

  beforeEach(async () => {
    // 各テスト前にテーブルをクリーンアップ
    await db.raw('TRUNCATE TABLE users, orders, products CASCADE');

    // テスト用ユーザーとトークンを作成
    testUser = await createTestUser(db, {
      name: 'Test User',
      email: 'test@example.com',
      role: 'user',
    });
    adminUser = await createTestUser(db, {
      name: 'Admin User',
      email: 'admin@example.com',
      role: 'admin',
    });

    authToken = generateToken(testUser);
    adminToken = generateToken(adminUser);
  });

  afterAll(async () => {
    await db.destroy();
  });

  return {
    getApp: () => app,
    getAuthToken: () => authToken,
    getAdminToken: () => adminToken,
    getTestUser: () => testUser,
    getAdminUser: () => adminUser,
    getDb: () => db,
  };
}
```

### 3.2 CRUDエンドポイントの統合テスト

```javascript
// __tests__/integration/users.test.js
import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import supertest from 'supertest';
import { app } from '../../src/app';
import { db } from '../../src/db';

const request = supertest(app);

describe('Users API - 統合テスト', () => {
  let authToken;

  beforeAll(async () => {
    await db.migrate.latest();
  });

  beforeEach(async () => {
    await db('users').truncate();
    const user = await db('users').insert({
      id: 'user_test',
      name: 'Test Admin',
      email: 'test@example.com',
      role: 'admin',
    }).returning('*');
    authToken = generateToken(user[0]);
  });

  afterAll(async () => {
    await db.destroy();
  });

  // ============================================
  // GET /api/v1/users - ユーザー一覧取得
  // ============================================
  describe('GET /api/v1/users', () => {
    it('ページネーション付きのユーザー一覧を返す', async () => {
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
      expect(res.body.meta).toHaveProperty('total');
      expect(res.body.meta.total).toBe(3);
    });

    it('ロールでフィルタリングできる', async () => {
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

    it('認証なしで401を返す', async () => {
      await request
        .get('/api/v1/users')
        .expect(401);
    });

    it('無効なトークンで401を返す', async () => {
      await request
        .get('/api/v1/users')
        .set('Authorization', 'Bearer invalid-token-here')
        .expect(401);
    });

    it('期限切れトークンで401を返す', async () => {
      const expiredToken = generateToken(
        { id: 'user_test', role: 'admin' },
        { expiresIn: '-1h' }
      );

      await request
        .get('/api/v1/users')
        .set('Authorization', `Bearer ${expiredToken}`)
        .expect(401);
    });
  });

  // ============================================
  // POST /api/v1/users - ユーザー作成
  // ============================================
  describe('POST /api/v1/users', () => {
    it('ユーザーを作成して201を返す', async () => {
      const res = await request
        .post('/api/v1/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ name: 'New User', email: 'new@example.com', age: 25 })
        .expect(201);

      expect(res.body.data).toMatchObject({
        name: 'New User',
        email: 'new@example.com',
        role: 'user', // デフォルトロール
      });
      expect(res.body.data.id).toBeDefined();
      expect(res.headers.location).toMatch(/\/api\/v1\/users\//);
    });

    it('不正なメールアドレスで422を返す', async () => {
      const res = await request
        .post('/api/v1/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ name: 'Test', email: 'invalid-email' })
        .expect(422);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'email' })
      );
    });

    it('重複メールアドレスで409を返す', async () => {
      await request
        .post('/api/v1/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ name: 'First', email: 'dup@example.com' })
        .expect(201);

      const res = await request
        .post('/api/v1/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ name: 'Second', email: 'dup@example.com' })
        .expect(409);

      expect(res.body.error.code).toBe('DUPLICATE_RESOURCE');
    });

    it('必須フィールド欠落で422を返す', async () => {
      const res = await request
        .post('/api/v1/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send({}) // 空ボディ
        .expect(422);

      expect(res.body.errors.length).toBeGreaterThanOrEqual(2);
    });

    it('一般ユーザーがadminロールで作成しようとすると403を返す', async () => {
      const userToken = generateToken({ id: 'u_normal', role: 'user' });

      await request
        .post('/api/v1/users')
        .set('Authorization', `Bearer ${userToken}`)
        .send({ name: 'Hacker', email: 'hack@example.com', role: 'admin' })
        .expect(403);
    });
  });

  // ============================================
  // PUT /api/v1/users/:id - ユーザー更新
  // ============================================
  describe('PUT /api/v1/users/:id', () => {
    it('ユーザー情報を更新する', async () => {
      const res = await request
        .put('/api/v1/users/user_test')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ name: 'Updated Name' })
        .expect(200);

      expect(res.body.data.name).toBe('Updated Name');
    });

    it('存在しないユーザーIDで404を返す', async () => {
      await request
        .put('/api/v1/users/nonexistent_id')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ name: 'Ghost' })
        .expect(404);
    });

    it('楽観的ロックによる競合検出', async () => {
      // バージョン1で取得
      const getRes = await request
        .get('/api/v1/users/user_test')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      const version = getRes.body.data.version;

      // 1回目の更新（成功）
      await request
        .put('/api/v1/users/user_test')
        .set('Authorization', `Bearer ${authToken}`)
        .set('If-Match', `"${version}"`)
        .send({ name: 'Update 1' })
        .expect(200);

      // 2回目の更新（古いバージョンで競合）
      await request
        .put('/api/v1/users/user_test')
        .set('Authorization', `Bearer ${authToken}`)
        .set('If-Match', `"${version}"`)
        .send({ name: 'Update 2' })
        .expect(409);
    });
  });

  // ============================================
  // DELETE /api/v1/users/:id - ユーザー削除
  // ============================================
  describe('DELETE /api/v1/users/:id', () => {
    it('ユーザーを削除して204を返す', async () => {
      const created = await db('users').insert({
        id: 'u_delete', name: 'To Delete', email: 'delete@example.com', role: 'user',
      }).returning('*');

      await request
        .delete(`/api/v1/users/${created[0].id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(204);

      // 削除後に取得すると404
      await request
        .get(`/api/v1/users/${created[0].id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);
    });

    it('冪等性: 同じリソースを2回削除しても安全', async () => {
      await db('users').insert({
        id: 'u_idem', name: 'Idempotent', email: 'idem@example.com', role: 'user',
      });

      await request
        .delete('/api/v1/users/u_idem')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(204);

      // 2回目の削除は404（既に存在しない）
      await request
        .delete('/api/v1/users/u_idem')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);
    });
  });
});
```

### 3.3 テストヘルパーとファクトリー

テストコードの重複を避けるため、テストヘルパーとファクトリーパターンを活用する。

```javascript
// __tests__/helpers/factories.js
import { faker } from '@faker-js/faker';
import { db } from '../../src/db';

export class UserFactory {
  static defaults() {
    return {
      id: faker.string.uuid(),
      name: faker.person.fullName(),
      email: faker.internet.email(),
      role: 'user',
      age: faker.number.int({ min: 18, max: 80 }),
      createdAt: new Date(),
      updatedAt: new Date(),
    };
  }

  static async create(overrides = {}) {
    const data = { ...this.defaults(), ...overrides };
    const [user] = await db('users').insert(data).returning('*');
    return user;
  }

  static async createMany(count, overrides = {}) {
    const users = Array.from({ length: count }, (_, i) => ({
      ...this.defaults(),
      email: `user${i}@example.com`,
      ...overrides,
    }));
    return db('users').insert(users).returning('*');
  }
}

export class OrderFactory {
  static defaults(userId) {
    return {
      id: faker.string.uuid(),
      userId,
      status: 'pending',
      totalAmount: faker.number.int({ min: 100, max: 100000 }),
      items: JSON.stringify([
        { productId: faker.string.uuid(), quantity: 1, price: 1000 },
      ]),
      createdAt: new Date(),
    };
  }

  static async create(userId, overrides = {}) {
    const data = { ...this.defaults(userId), ...overrides };
    const [order] = await db('orders').insert(data).returning('*');
    return order;
  }
}
```

---

## 4. Postman / Newman によるAPIテスト自動化

### 4.1 Postman コレクションの構造化

Postmanは手動テストだけでなく、コレクションとして定義されたテストを自動実行する機能を提供する。CI/CDパイプラインではNewman（Postmanのコマンドラインランナー）を使用する。

```
Postman コレクション構成例

Collection: "User Management API"
  |
  +-- Folder: "Authentication"
  |     +-- POST /auth/login
  |     +-- POST /auth/register
  |     +-- POST /auth/refresh
  |     +-- POST /auth/logout
  |
  +-- Folder: "Users (CRUD)"
  |     +-- GET  /api/v1/users       (一覧取得)
  |     +-- POST /api/v1/users       (作成)
  |     +-- GET  /api/v1/users/:id   (個別取得)
  |     +-- PUT  /api/v1/users/:id   (更新)
  |     +-- DELETE /api/v1/users/:id (削除)
  |
  +-- Folder: "Error Cases"
  |     +-- 認証エラー (401)
  |     +-- 認可エラー (403)
  |     +-- バリデーションエラー (422)
  |     +-- リソース競合 (409)
  |
  +-- Folder: "Edge Cases"
        +-- 空リクエスト
        +-- 巨大ペイロード
        +-- 特殊文字入力
        +-- 同時リクエスト
```

### 4.2 Postman テストスクリプトの記述

```javascript
// Postman の Tests タブに記述するスクリプト例

// === POST /auth/login のテストスクリプト ===
// ステータスコードの検証
pm.test("ステータスコード200を返す", function () {
    pm.response.to.have.status(200);
});

// レスポンスボディの検証
pm.test("アクセストークンを含むレスポンスを返す", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('accessToken');
    pm.expect(jsonData).to.have.property('refreshToken');
    pm.expect(jsonData).to.have.property('expiresIn');
    pm.expect(jsonData.expiresIn).to.be.a('number');
});

// トークンを環境変数に保存（後続リクエストで使用）
pm.test("トークンを環境変数に保存する", function () {
    const jsonData = pm.response.json();
    pm.environment.set("accessToken", jsonData.accessToken);
    pm.environment.set("refreshToken", jsonData.refreshToken);
});

// レスポンス時間の検証
pm.test("レスポンスが500ms以内に返る", function () {
    pm.expect(pm.response.responseTime).to.be.below(500);
});

// ヘッダーの検証
pm.test("Content-Typeがapplication/jsonである", function () {
    pm.response.to.have.header("Content-Type", "application/json; charset=utf-8");
});

// スキーマバリデーション
const schema = {
    type: "object",
    required: ["accessToken", "refreshToken", "expiresIn", "user"],
    properties: {
        accessToken: { type: "string" },
        refreshToken: { type: "string" },
        expiresIn: { type: "number" },
        user: {
            type: "object",
            required: ["id", "name", "email", "role"],
            properties: {
                id: { type: "string" },
                name: { type: "string" },
                email: { type: "string", format: "email" },
                role: { type: "string", enum: ["user", "admin"] },
            }
        }
    }
};

pm.test("レスポンスがスキーマに準拠している", function () {
    pm.response.to.have.jsonSchema(schema);
});
```

### 4.3 Newman によるCI/CD統合

```bash
# Newman のインストール
npm install -g newman newman-reporter-htmlextra

# コレクションの実行
newman run ./postman/collection.json \
  --environment ./postman/env-staging.json \
  --reporters cli,htmlextra \
  --reporter-htmlextra-export ./reports/api-test-report.html \
  --iteration-count 3 \
  --delay-request 100 \
  --timeout-request 10000

# GitHub Actions での実行例
# .github/workflows/api-tests.yml
# name: API Tests
# on:
#   push:
#     branches: [main, develop]
#   pull_request:
#     branches: [main]
#
# jobs:
#   api-tests:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#       - uses: actions/setup-node@v4
#         with:
#           node-version: '20'
#       - run: npm ci
#       - run: npm run start:test &
#       - run: npx newman run ./postman/collection.json \
#              --environment ./postman/env-test.json \
#              --reporters cli,junit \
#              --reporter-junit-export ./reports/junit.xml
#       - uses: actions/upload-artifact@v4
#         with:
#           name: test-reports
#           path: ./reports/
```

---

## 5. コントラクトテスト（Pact）

### 5.1 コントラクトテストの概念

コントラクトテストは、サービス間のAPI仕様（契約）が両者間で合意されていることを検証するテスト手法である。マイクロサービスアーキテクチャにおいて、Consumer（APIを呼び出す側）とProvider（APIを提供する側）が互いの期待に沿って動作していることを保証する。

```
コントラクトテスト フロー図

  Consumer側                  Pact Broker                Provider側
  (フロントエンド)             (契約管理)                 (APIサーバー)
  +------------------+      +------------------+      +------------------+
  |                  |      |                  |      |                  |
  | 1. テスト実行    |      |                  |      |                  |
  |    (Mockサーバー |----->| 2. コントラクト  |      |                  |
  |     に対して)    |      |    (Pact JSON)を |      |                  |
  |                  |      |    アップロード  |      |                  |
  +------------------+      |                  |      |                  |
                            |                  |----->| 3. コントラクトを|
                            |                  |      |    ダウンロードし|
                            |                  |      |    Providerに対し|
                            |                  |<-----| 4. 検証結果を    |
                            |                  |      |    報告          |
                            +------------------+      +------------------+
                                     |
                                     v
                            +------------------+
                            | 5. CI/CDで       |
                            | can-i-deploy を  |
                            | チェックし、     |
                            | デプロイ可否判定 |
                            +------------------+
```

### 5.2 Consumer側テストの実装

```javascript
// __tests__/contract/userApiConsumer.pact.test.js
import { PactV3, MatchersV3 } from '@pact-foundation/pact';
import path from 'path';
import { UserApiClient } from '../../src/clients/userApiClient';

const { like, eachLike, regex, string, integer, boolean } = MatchersV3;

const provider = new PactV3({
  consumer: 'FrontendApp',
  provider: 'UserAPI',
  dir: path.resolve(process.cwd(), 'pacts'),
  logLevel: 'warn',
});

describe('User API Contract - Consumer側', () => {
  // ユーザー一覧取得のコントラクト
  describe('GET /api/v1/users', () => {
    it('ページネーション付きのユーザー一覧を返すこと', async () => {
      provider
        .given('ユーザーが複数存在する')
        .uponReceiving('ユーザー一覧取得リクエスト')
        .withRequest({
          method: 'GET',
          path: '/api/v1/users',
          query: { limit: '10', offset: '0' },
          headers: {
            Authorization: regex(/^Bearer .+$/, 'Bearer valid-token'),
            Accept: 'application/json',
          },
        })
        .willRespondWith({
          status: 200,
          headers: {
            'Content-Type': 'application/json; charset=utf-8',
          },
          body: {
            data: eachLike({
              id: string('user_123'),
              name: string('Taro Yamada'),
              email: string('taro@example.com'),
              role: regex(/^(user|admin)$/, 'user'),
              createdAt: regex(
                /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/,
                '2024-01-15T09:00:00Z'
              ),
            }),
            meta: {
              total: integer(100),
              limit: integer(10),
              offset: integer(0),
              hasNextPage: boolean(true),
            },
          },
        });

      await provider.executeTest(async (mockServer) => {
        const client = new UserApiClient({
          baseUrl: mockServer.url,
          token: 'valid-token',
        });

        const result = await client.listUsers({ limit: 10, offset: 0 });

        expect(result.data).toBeDefined();
        expect(result.data.length).toBeGreaterThan(0);
        expect(result.meta.total).toBeGreaterThanOrEqual(0);
        expect(result.meta.hasNextPage).toBeDefined();
      });
    });
  });

  // ユーザー個別取得のコントラクト
  describe('GET /api/v1/users/:id', () => {
    it('指定IDのユーザーを返すこと', async () => {
      provider
        .given('ID 123 のユーザーが存在する')
        .uponReceiving('ユーザー個別取得リクエスト')
        .withRequest({
          method: 'GET',
          path: '/api/v1/users/123',
          headers: {
            Authorization: regex(/^Bearer .+$/, 'Bearer valid-token'),
          },
        })
        .willRespondWith({
          status: 200,
          headers: { 'Content-Type': 'application/json; charset=utf-8' },
          body: {
            data: {
              id: string('123'),
              name: like('Taro Yamada'),
              email: like('taro@example.com'),
              role: regex(/^(user|admin)$/, 'user'),
              profile: {
                bio: like('ソフトウェアエンジニア'),
                avatarUrl: like('https://example.com/avatar.png'),
              },
            },
          },
        });

      await provider.executeTest(async (mockServer) => {
        const client = new UserApiClient({
          baseUrl: mockServer.url,
          token: 'valid-token',
        });

        const user = await client.getUser('123');
        expect(user.id).toBe('123');
        expect(user.name).toBeDefined();
        expect(user.email).toBeDefined();
        expect(user.profile).toBeDefined();
      });
    });

    it('存在しないIDで404を返すこと', async () => {
      provider
        .given('ID 999 のユーザーは存在しない')
        .uponReceiving('存在しないユーザーの取得リクエスト')
        .withRequest({
          method: 'GET',
          path: '/api/v1/users/999',
          headers: {
            Authorization: regex(/^Bearer .+$/, 'Bearer valid-token'),
          },
        })
        .willRespondWith({
          status: 404,
          headers: { 'Content-Type': 'application/json; charset=utf-8' },
          body: {
            error: {
              code: string('NOT_FOUND'),
              message: like('指定されたユーザーは見つかりませんでした'),
            },
          },
        });

      await provider.executeTest(async (mockServer) => {
        const client = new UserApiClient({
          baseUrl: mockServer.url,
          token: 'valid-token',
        });

        await expect(client.getUser('999')).rejects.toThrow(/not found/i);
      });
    });
  });

  // ユーザー作成のコントラクト
  describe('POST /api/v1/users', () => {
    it('新規ユーザーを作成して201を返すこと', async () => {
      const newUser = {
        name: 'Hanako Suzuki',
        email: 'hanako@example.com',
        age: 28,
      };

      provider
        .given('ユーザー登録が可能な状態')
        .uponReceiving('ユーザー作成リクエスト')
        .withRequest({
          method: 'POST',
          path: '/api/v1/users',
          headers: {
            Authorization: regex(/^Bearer .+$/, 'Bearer valid-token'),
            'Content-Type': 'application/json',
          },
          body: newUser,
        })
        .willRespondWith({
          status: 201,
          headers: {
            'Content-Type': 'application/json; charset=utf-8',
            Location: regex(/^\/api\/v1\/users\//, '/api/v1/users/new_id'),
          },
          body: {
            data: {
              id: string('new_id'),
              name: string('Hanako Suzuki'),
              email: string('hanako@example.com'),
              role: string('user'),
              age: integer(28),
            },
          },
        });

      await provider.executeTest(async (mockServer) => {
        const client = new UserApiClient({
          baseUrl: mockServer.url,
          token: 'valid-token',
        });

        const created = await client.createUser(newUser);
        expect(created.name).toBe('Hanako Suzuki');
        expect(created.email).toBe('hanako@example.com');
        expect(created.id).toBeDefined();
      });
    });
  });
});
```

### 5.3 Provider側の検証

```javascript
// __tests__/contract/userApiProvider.pact.test.js
import { Verifier } from '@pact-foundation/pact';
import { app } from '../../src/app';
import { db } from '../../src/db';

describe('User API Contract - Provider検証', () => {
  let server;
  const port = 4567;

  beforeAll(async () => {
    await db.migrate.latest();
    server = app.listen(port);
  });

  afterAll(async () => {
    server.close();
    await db.destroy();
  });

  it('Consumer のコントラクトを満たすこと', async () => {
    const opts = {
      provider: 'UserAPI',
      providerBaseUrl: `http://localhost:${port}`,

      // Pact Brokerから取得する場合
      pactBrokerUrl: process.env.PACT_BROKER_URL,
      pactBrokerToken: process.env.PACT_BROKER_TOKEN,

      // ローカルファイルから取得する場合
      // pactUrls: ['./pacts/FrontendApp-UserAPI.json'],

      publishVerificationResult: process.env.CI === 'true',
      providerVersion: process.env.GIT_COMMIT_SHA,
      providerVersionBranch: process.env.GIT_BRANCH,

      // Provider Stateのハンドラー
      stateHandlers: {
        'ユーザーが複数存在する': async () => {
          await db('users').truncate();
          await db('users').insert([
            { id: 'user_1', name: 'Taro', email: 'taro@example.com', role: 'user' },
            { id: 'user_2', name: 'Hanako', email: 'hanako@example.com', role: 'admin' },
          ]);
        },
        'ID 123 のユーザーが存在する': async () => {
          await db('users').truncate();
          await db('users').insert({
            id: '123',
            name: 'Taro Yamada',
            email: 'taro@example.com',
            role: 'user',
            profile: JSON.stringify({
              bio: 'ソフトウェアエンジニア',
              avatarUrl: 'https://example.com/avatar.png',
            }),
          });
        },
        'ID 999 のユーザーは存在しない': async () => {
          await db('users').where({ id: '999' }).delete();
        },
        'ユーザー登録が可能な状態': async () => {
          await db('users').where({ email: 'hanako@example.com' }).delete();
        },
      },

      // リクエストフィルター（認証トークンの注入など）
      requestFilter: (req, res, next) => {
        req.headers['authorization'] = 'Bearer test-provider-token';
        next();
      },
    };

    await new Verifier(opts).verifyProvider();
  });
});
```

### 5.4 Pact Broker とデプロイ安全性

```bash
# Pact Broker での can-i-deploy チェック
# Consumer のデプロイ前に実行
pact-broker can-i-deploy \
  --pacticipant FrontendApp \
  --version $(git rev-parse HEAD) \
  --to-environment production

# Provider のデプロイ前に実行
pact-broker can-i-deploy \
  --pacticipant UserAPI \
  --version $(git rev-parse HEAD) \
  --to-environment production

# デプロイ成功の記録
pact-broker record-deployment \
  --pacticipant UserAPI \
  --version $(git rev-parse HEAD) \
  --environment production
```

---

## 6. 負荷テスト

### 6.1 負荷テストの種類と目的

負荷テストは、APIが一定の負荷条件下で正常に動作するかを検証するテストである。目的に応じて複数の種類が存在する。

| テスト種類 | 目的 | VU数 | 期間 | 特徴 |
|------------|------|------|------|------|
| スモークテスト | 基本動作確認 | 1-5 | 1分 | デプロイ後の簡易確認 |
| ロードテスト | 通常負荷検証 | 50-200 | 5-30分 | 平常時のパフォーマンス検証 |
| ストレステスト | 限界点の特定 | 200-1000+ | 10-60分 | システムの破綻点を発見 |
| スパイクテスト | 急激な負荷変動 | 0→500→0 | 5-10分 | 瞬間的な負荷への耐性 |
| ソークテスト | 長時間安定性 | 50-100 | 1-24時間 | メモリリーク等の検出 |
| ブレイクポイントテスト | 破綻点の特定 | 段階的に増加 | 可変 | 最大許容量の測定 |

### 6.2 k6 による負荷テスト

```javascript
// load-tests/scenarios/user-api-load.js（k6）
import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// カスタムメトリクス定義
const errorRate = new Rate('errors');
const userCreated = new Counter('users_created');
const listDuration = new Trend('list_duration', true);
const createDuration = new Trend('create_duration', true);

// テストシナリオ設定
export const options = {
  scenarios: {
    // シナリオ1: 読み取り中心の通常負荷
    read_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 30 },    // ウォームアップ
        { duration: '2m',  target: 30 },    // 定常負荷
        { duration: '30s', target: 0 },     // クールダウン
      ],
      gracefulRampDown: '10s',
      exec: 'readScenario',
      tags: { scenario: 'read' },
    },
    // シナリオ2: 書き込み中心の高負荷
    write_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 },
        { duration: '2m',  target: 10 },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '10s',
      exec: 'writeScenario',
      startTime: '10s',  // 10秒遅れて開始
      tags: { scenario: 'write' },
    },
    // シナリオ3: スパイクテスト
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 5 },     // ベースライン
        { duration: '5s',  target: 100 },   // 急激なスパイク
        { duration: '30s', target: 100 },   // スパイク維持
        { duration: '5s',  target: 5 },     // 急激な減少
        { duration: '30s', target: 5 },     // 回復確認
        { duration: '10s', target: 0 },     // 終了
      ],
      exec: 'readScenario',
      startTime: '4m',  // 他のシナリオ後に実行
      tags: { scenario: 'spike' },
    },
  },
  thresholds: {
    // グローバル閾値
    http_req_duration: ['p(50)<200', 'p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.05'],

    // シナリオ別の閾値
    'http_req_duration{scenario:read}': ['p(95)<300'],
    'http_req_duration{scenario:write}': ['p(95)<800'],
    'http_req_duration{scenario:spike}': ['p(95)<2000'],

    // カスタムメトリクスの閾値
    list_duration: ['p(95)<400'],
    create_duration: ['p(95)<700'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'https://api-staging.example.com/v1';
const TOKEN = __ENV.API_TOKEN;

const headers = {
  Authorization: `Bearer ${TOKEN}`,
  'Content-Type': 'application/json',
};

// 読み取りシナリオ
export function readScenario() {
  group('ユーザー一覧取得', () => {
    const startTime = Date.now();
    const res = http.get(`${BASE_URL}/users?limit=20`, { headers });
    listDuration.add(Date.now() - startTime);

    const success = check(res, {
      'ステータスが200': (r) => r.status === 200,
      'レスポンスにdataが存在': (r) => {
        try { return JSON.parse(r.body).data !== undefined; }
        catch { return false; }
      },
      'レスポンスタイム < 500ms': (r) => r.timings.duration < 500,
    });

    errorRate.add(!success);
  });

  group('ユーザー個別取得', () => {
    const userId = `user_${Math.floor(Math.random() * 100) + 1}`;
    const res = http.get(`${BASE_URL}/users/${userId}`, { headers });

    const success = check(res, {
      'ステータスが200または404': (r) => [200, 404].includes(r.status),
      'レスポンスタイム < 300ms': (r) => r.timings.duration < 300,
    });

    errorRate.add(!success);
  });

  sleep(Math.random() * 2 + 1); // 1〜3秒のランダムなシンクタイム
}

// 書き込みシナリオ
export function writeScenario() {
  group('ユーザー作成', () => {
    const startTime = Date.now();
    const uniqueId = `${Date.now()}_${__VU}_${__ITER}`;
    const payload = JSON.stringify({
      name: `LoadTest User ${uniqueId}`,
      email: `loadtest_${uniqueId}@example.com`,
      age: Math.floor(Math.random() * 60) + 18,
    });

    const res = http.post(`${BASE_URL}/users`, payload, { headers });
    createDuration.add(Date.now() - startTime);

    const success = check(res, {
      'ステータスが201': (r) => r.status === 201,
      '作成されたユーザーIDが返る': (r) => {
        try { return JSON.parse(r.body).data.id !== undefined; }
        catch { return false; }
      },
      'レスポンスタイム < 1000ms': (r) => r.timings.duration < 1000,
    });

    if (success) userCreated.add(1);
    errorRate.add(!success);
  });

  sleep(Math.random() * 3 + 2); // 2〜5秒のシンクタイム
}

// テスト結果のサマリー出力
export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    './reports/load-test-summary.json': JSON.stringify(data, null, 2),
  };
}
```

```bash
# k6 実行コマンド
k6 run load-tests/scenarios/user-api-load.js \
  --env BASE_URL=https://api-staging.example.com/v1 \
  --env API_TOKEN=sk_test_xxx \
  --out json=./reports/k6-results.json

# Grafana + InfluxDB への出力
k6 run load-tests/scenarios/user-api-load.js \
  --out influxdb=http://localhost:8086/k6
```

### 6.3 Artillery による負荷テスト

```yaml
# load-tests/artillery/user-api.yml
config:
  target: "https://api-staging.example.com"
  phases:
    - name: "ウォームアップ"
      duration: 30
      arrivalRate: 5
    - name: "通常負荷"
      duration: 120
      arrivalRate: 20
    - name: "ピーク負荷"
      duration: 60
      arrivalRate: 50
    - name: "クールダウン"
      duration: 30
      arrivalRate: 5
  defaults:
    headers:
      Authorization: "Bearer {{ $processEnvironment.API_TOKEN }}"
      Content-Type: "application/json"
  plugins:
    expect: {}
  ensure:
    p95: 500
    p99: 1000
    maxErrorRate: 1

scenarios:
  - name: "ユーザー一覧取得"
    weight: 60
    flow:
      - get:
          url: "/api/v1/users?limit=20"
          expect:
            - statusCode: 200
            - hasProperty: "data"
            - contentType: "application/json"

  - name: "ユーザー作成→取得→更新"
    weight: 30
    flow:
      - post:
          url: "/api/v1/users"
          json:
            name: "Artillery User {{ $randomString() }}"
            email: "artillery_{{ $timestamp() }}@example.com"
            age: "{{ $randomNumber(18, 80) }}"
          capture:
            - json: "$.data.id"
              as: "userId"
          expect:
            - statusCode: 201
      - think: 1
      - get:
          url: "/api/v1/users/{{ userId }}"
          expect:
            - statusCode: 200
      - think: 1
      - put:
          url: "/api/v1/users/{{ userId }}"
          json:
            name: "Updated User {{ $randomString() }}"
          expect:
            - statusCode: 200

  - name: "検索シナリオ"
    weight: 10
    flow:
      - get:
          url: "/api/v1/users?filter[role]=admin&sort=-createdAt&limit=5"
          expect:
            - statusCode: 200
```

```bash
# Artillery 実行コマンド
artillery run load-tests/artillery/user-api.yml \
  --output ./reports/artillery-report.json

# HTMLレポート生成
artillery report ./reports/artillery-report.json \
  --output ./reports/artillery-report.html
```

---

## 7. E2Eテスト

### 7.1 E2Eテストの設計

E2E（End-to-End）テストは、ユーザーの実際の利用シナリオを模擬し、複数のAPIを横断して全体的なフローが正しく動作することを検証する。テストピラミッドの最上位に位置し、数は少ないが高い信頼性を提供する。

```
E2Eテスト シナリオ例: ECサイト購入フロー

  [1] 会員登録                [2] ログイン
  POST /auth/register   -->  POST /auth/login
  201 Created                 200 OK (token)
       |                           |
       v                           v
  [3] 商品一覧取得            [4] 商品をカートに追加
  GET /products          -->  POST /cart/items
  200 OK                      201 Created
       |                           |
       v                           v
  [5] カート確認              [6] 注文作成
  GET /cart              -->  POST /orders
  200 OK                      201 Created
       |                           |
       v                           v
  [7] 決済実行                [8] 注文確認
  POST /payments         -->  GET /orders/:id
  200 OK                      200 OK (status: paid)
       |
       v
  [9] メール送信確認（非同期）
  → 注文確認メールが送信されたことをキューで検証
```

### 7.2 E2Eテストの実装

```javascript
// __tests__/e2e/purchaseFlow.test.js
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import supertest from 'supertest';
import { app } from '../../src/app';
import { db } from '../../src/db';
import { seedProducts } from '../helpers/seedData';

const request = supertest(app);

describe('E2E: 商品購入フロー', () => {
  let accessToken;
  let userId;
  let productId;
  let cartId;
  let orderId;

  beforeAll(async () => {
    await db.migrate.latest();
    await db.raw('TRUNCATE TABLE users, products, carts, orders, payments CASCADE');
    // テスト用商品データを投入
    const products = await seedProducts(db);
    productId = products[0].id;
  });

  afterAll(async () => {
    await db.destroy();
  });

  it('ステップ1: 会員登録', async () => {
    const res = await request
      .post('/auth/register')
      .send({
        name: 'E2E Test User',
        email: 'e2e@example.com',
        password: 'SecurePass123!',
      })
      .expect(201);

    expect(res.body.data.id).toBeDefined();
    userId = res.body.data.id;
  });

  it('ステップ2: ログイン', async () => {
    const res = await request
      .post('/auth/login')
      .send({
        email: 'e2e@example.com',
        password: 'SecurePass123!',
      })
      .expect(200);

    expect(res.body.accessToken).toBeDefined();
    expect(res.body.refreshToken).toBeDefined();
    accessToken = res.body.accessToken;
  });

  it('ステップ3: 商品一覧取得', async () => {
    const res = await request
      .get('/api/v1/products?limit=10')
      .set('Authorization', `Bearer ${accessToken}`)
      .expect(200);

    expect(res.body.data.length).toBeGreaterThan(0);
    expect(res.body.data[0]).toHaveProperty('id');
    expect(res.body.data[0]).toHaveProperty('price');
  });

  it('ステップ4: 商品をカートに追加', async () => {
    const res = await request
      .post('/api/v1/cart/items')
      .set('Authorization', `Bearer ${accessToken}`)
      .send({
        productId,
        quantity: 2,
      })
      .expect(201);

    expect(res.body.data.items).toHaveLength(1);
    expect(res.body.data.items[0].productId).toBe(productId);
    cartId = res.body.data.id;
  });

  it('ステップ5: カート確認', async () => {
    const res = await request
      .get('/api/v1/cart')
      .set('Authorization', `Bearer ${accessToken}`)
      .expect(200);

    expect(res.body.data.items).toHaveLength(1);
    expect(res.body.data.totalAmount).toBeGreaterThan(0);
  });

  it('ステップ6: 注文作成', async () => {
    const res = await request
      .post('/api/v1/orders')
      .set('Authorization', `Bearer ${accessToken}`)
      .send({
        cartId,
        shippingAddress: {
          postalCode: '100-0001',
          prefecture: '東京都',
          city: '千代田区',
          line1: '千代田1-1',
        },
      })
      .expect(201);

    expect(res.body.data.status).toBe('pending');
    expect(res.body.data.totalAmount).toBeGreaterThan(0);
    orderId = res.body.data.id;
  });

  it('ステップ7: 決済実行', async () => {
    const res = await request
      .post('/api/v1/payments')
      .set('Authorization', `Bearer ${accessToken}`)
      .send({
        orderId,
        method: 'credit_card',
        cardToken: 'tok_test_visa',
      })
      .expect(200);

    expect(res.body.data.status).toBe('succeeded');
    expect(res.body.data.orderId).toBe(orderId);
  });

  it('ステップ8: 注文ステータス確認', async () => {
    const res = await request
      .get(`/api/v1/orders/${orderId}`)
      .set('Authorization', `Bearer ${accessToken}`)
      .expect(200);

    expect(res.body.data.status).toBe('paid');
    expect(res.body.data.payment).toBeDefined();
    expect(res.body.data.payment.status).toBe('succeeded');
  });
});
```

---

## 8. OpenAPI仕様ベースのテスト

### 8.1 スキーマ検証テスト

OpenAPI（旧Swagger）仕様書を基にしたテストは、APIレスポンスが定義されたスキーマに準拠していることを自動的に検証する。手動でテストケースを書く手間を削減し、仕様と実装の乖離を防ぐ。

```javascript
// __tests__/schema/openapi-validation.test.js
import { describe, it, expect, beforeAll } from 'vitest';
import supertest from 'supertest';
import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import yaml from 'js-yaml';
import { readFileSync } from 'fs';
import { app } from '../../src/app';
import { resolveRefs } from '../helpers/schemaResolver';

const request = supertest(app);

describe('OpenAPI スキーマ検証', () => {
  let spec;
  let ajv;
  let token;

  beforeAll(async () => {
    // OpenAPI仕様を読み込み
    spec = yaml.load(readFileSync('./openapi.yaml', 'utf-8'));

    // JSONスキーマバリデーター設定
    ajv = new Ajv({
      allErrors: true,
      strict: false,
      validateFormats: true,
    });
    addFormats(ajv);

    // テスト用トークン取得
    const loginRes = await request
      .post('/auth/login')
      .send({ email: 'test@example.com', password: 'test123' });
    token = loginRes.body.accessToken;
  });

  // OpenAPIの各パスに対して自動テスト生成
  const endpoints = [
    { method: 'get', path: '/api/v1/users', status: 200 },
    { method: 'get', path: '/api/v1/users/test_id', status: 200 },
    { method: 'get', path: '/api/v1/products', status: 200 },
  ];

  endpoints.forEach(({ method, path, status }) => {
    it(`${method.toUpperCase()} ${path} のレスポンスがスキーマに準拠する`, async () => {
      const res = await request[method](path)
        .set('Authorization', `Bearer ${token}`);

      // OpenAPIスキーマを取得
      const specPath = path.replace(/\/test_id/, '/{id}')
                           .replace('/api/v1', '');
      const responseSchema = spec.paths[specPath]?.[method]
        ?.responses?.[String(status)]
        ?.content?.['application/json']?.schema;

      if (!responseSchema) {
        throw new Error(`スキーマが見つかりません: ${method.toUpperCase()} ${specPath} ${status}`);
      }

      // $ref を解決してバリデーション実行
      const resolvedSchema = resolveRefs(responseSchema, spec);
      const validate = ajv.compile(resolvedSchema);
      const valid = validate(res.body);

      if (!valid) {
        console.error('バリデーションエラー:', JSON.stringify(validate.errors, null, 2));
      }

      expect(valid).toBe(true);
    });
  });
});
```

### 8.2 Schemathesisによるファジングテスト

```python
# fuzz-tests/test_api_fuzz.py
# Schemathesis: OpenAPI仕様ベースの自動ファジングテスト

import schemathesis
import pytest

# OpenAPI仕様を読み込み
schema = schemathesis.from_url(
    "https://api-staging.example.com/openapi.yaml",
    base_url="https://api-staging.example.com",
)

@schema.parametrize()
def test_api_conformance(case):
    """
    OpenAPI仕様に基づく自動テスト
    - 全エンドポイントに対してランダムなリクエストを生成
    - レスポンスがスキーマに準拠しているかを検証
    - 5xx エラーが返らないことを確認
    """
    response = case.call_and_validate()

    # 5xxエラーは許容しない
    assert response.status_code < 500, \
        f"サーバーエラー: {response.status_code} - {response.text}"

# 状態遷移を考慮したテスト
@schema.parametrize(method="POST")
def test_create_operations(case):
    """POST操作の検証: 作成後に取得できることを確認"""
    response = case.call_and_validate()

    if response.status_code == 201:
        # Locationヘッダーからリソースを取得
        location = response.headers.get("Location")
        if location:
            get_response = case.session.get(location)
            assert get_response.status_code == 200

# CLIでの実行例:
# schemathesis run https://api-staging.example.com/openapi.yaml \
#   --auth "Bearer sk_test_xxx" \
#   --stateful=links \
#   --hypothesis-seed=42 \
#   --hypothesis-max-examples=100 \
#   --checks all
```

---

## 9. テスト環境とモック

### 9.1 外部サービスのモック（MSW）

統合テストやE2Eテストにおいて、外部のサードパーティAPIに実際のリクエストを送ることは避けるべきである。MSW（Mock Service Worker）を使用することで、ネットワークレベルでリクエストをインターセプトし、モックレスポンスを返すことができる。

```javascript
// __tests__/mocks/handlers.js
import { http, HttpResponse } from 'msw';

export const handlers = [
  // Stripe 決済API のモック
  http.post('https://api.stripe.com/v1/charges', async ({ request }) => {
    const body = await request.formData();
    const amount = body.get('amount');

    if (Number(amount) > 999999) {
      return HttpResponse.json(
        {
          error: {
            type: 'card_error',
            code: 'amount_too_large',
            message: 'Amount must be no more than ¥999,999',
          },
        },
        { status: 400 }
      );
    }

    return HttpResponse.json({
      id: `ch_test_${Date.now()}`,
      object: 'charge',
      amount: Number(amount),
      currency: body.get('currency') || 'jpy',
      status: 'succeeded',
      created: Math.floor(Date.now() / 1000),
    });
  }),

  // SendGrid メール送信API のモック
  http.post('https://api.sendgrid.com/v3/mail/send', async ({ request }) => {
    const body = await request.json();

    // モックの中でも最低限のバリデーション
    if (!body.personalizations?.[0]?.to?.[0]?.email) {
      return HttpResponse.json(
        { errors: [{ message: 'The to array is required' }] },
        { status: 400 }
      );
    }

    return new HttpResponse(null, { status: 202 });
  }),

  // 地理情報API のモック
  http.get('https://api.geocoding.example.com/v1/search', ({ request }) => {
    const url = new URL(request.url);
    const query = url.searchParams.get('q');

    const mockResults = {
      '東京都千代田区': {
        lat: 35.6812,
        lng: 139.7671,
        formattedAddress: '日本、〒100-0001 東京都千代田区',
      },
      '大阪府大阪市': {
        lat: 34.6937,
        lng: 135.5023,
        formattedAddress: '日本、〒530-0001 大阪府大阪市北区',
      },
    };

    const result = mockResults[query];
    if (!result) {
      return HttpResponse.json({ results: [] });
    }

    return HttpResponse.json({ results: [result] });
  }),
];

// __tests__/mocks/server.js
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

export const mockServer = setupServer(...handlers);
```

```javascript
// __tests__/setup.js （Vitest のグローバルセットアップ）
import { beforeAll, afterAll, afterEach } from 'vitest';
import { mockServer } from './mocks/server';

beforeAll(() => {
  mockServer.listen({
    onUnhandledRequest: 'warn', // モック外のリクエストを警告
  });
});

afterEach(() => {
  mockServer.resetHandlers(); // テスト間でハンドラーをリセット
});

afterAll(() => {
  mockServer.close();
});
```

### 9.2 テスト用データベース戦略

```
テスト用データベース戦略の比較

+-------------------+----------------+----------------+------------------+
| 戦略              | 速度           | 分離性         | 本番との近さ     |
+-------------------+----------------+----------------+------------------+
| SQLite in-memory  | 非常に高速     | 完全分離       | 低い             |
|                   | (ms単位)       | (プロセスごと) | (SQL方言の差)    |
+-------------------+----------------+----------------+------------------+
| Docker PostgreSQL | 中程度         | 完全分離       | 高い             |
|                   | (秒単位)       | (コンテナごと) | (同一エンジン)   |
+-------------------+----------------+----------------+------------------+
| テスト用スキーマ  | 高速           | スキーマ分離   | 高い             |
|                   | (ms〜秒)       | (同一DB内)     | (同一エンジン)   |
+-------------------+----------------+----------------+------------------+
| トランザクション  | 非常に高速     | テスト単位     | 高い             |
| ロールバック       | (ms単位)       | (ロールバック) | (同一エンジン)   |
+-------------------+----------------+----------------+------------------+
```

```javascript
// __tests__/setup/testDb.js
// トランザクションロールバック戦略の実装例

import { beforeEach, afterEach } from 'vitest';
import { db } from '../../src/db';

let transaction;

export function useTransactionalTests() {
  beforeEach(async () => {
    // 各テストをトランザクション内で実行
    transaction = await db.transaction();
    // アプリケーションのDBインスタンスをトランザクションに差し替え
    db._originalKnex = db.client;
    db.client = transaction;
  });

  afterEach(async () => {
    // テスト終了後にロールバック（データを元に戻す）
    await transaction.rollback();
    db.client = db._originalKnex;
  });
}
```

---

## 10. CI/CDパイプラインへの統合

### 10.1 テスト実行の自動化フロー

```
CI/CDパイプライン上のテスト実行フロー

  コード変更をプッシュ
         |
         v
  +----------------------------------------------+
  |  Stage 1: 静的解析 (並列実行, 約1分)          |
  |  +----------+ +----------+ +----------+      |
  |  | ESLint   | | TypeCheck| | Prettier |      |
  |  +----------+ +----------+ +----------+      |
  +----------------------------------------------+
         |
         v
  +----------------------------------------------+
  |  Stage 2: ユニットテスト (並列実行, 約2分)    |
  |  +------------------+ +-------------------+  |
  |  | バリデーション   | | ビジネスロジック  |  |
  |  | テスト (500+)    | | テスト (300+)     |  |
  |  +------------------+ +-------------------+  |
  +----------------------------------------------+
         |
         v
  +----------------------------------------------+
  |  Stage 3: 統合テスト (Docker環境, 約5分)      |
  |  +------------------+ +-------------------+  |
  |  | APIエンドポイント| | DB統合テスト      |  |
  |  | テスト (200+)    | | (100+)            |  |
  |  +------------------+ +-------------------+  |
  +----------------------------------------------+
         |
         v
  +----------------------------------------------+
  |  Stage 4: コントラクトテスト (約3分)          |
  |  +------------------+ +-------------------+  |
  |  | Consumer検証     | | can-i-deploy      |  |
  |  +------------------+ | チェック           |  |
  |                       +-------------------+  |
  +----------------------------------------------+
         |
         v
  +----------------------------------------------+
  |  Stage 5: E2Eテスト (ステージング環境, 約10分)|
  |  +---------------------+                     |
  |  | シナリオテスト (20+) |                     |
  |  +---------------------+                     |
  +----------------------------------------------+
         |
         v
  +----------------------------------------------+
  |  Stage 6: 負荷テスト (夜間バッチ/手動, 約30分)|
  |  +---------------------+                     |
  |  | k6 / Artillery      |                     |
  |  +---------------------+                     |
  +----------------------------------------------+
         |
         v
  デプロイ (全テストパス時のみ)
```

### 10.2 GitHub Actions 設定例

```yaml
# .github/workflows/api-tests.yml
name: API Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # 毎日深夜2時に負荷テスト実行

jobs:
  # ===== ユニットテスト =====
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run test:unit -- --coverage --reporter=junit
        env:
          JUNIT_OUTPUT: ./reports/unit-tests.xml
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: unit-test-results
          path: ./reports/

  # ===== 統合テスト =====
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run db:migrate:test
        env:
          DATABASE_URL: postgres://test:test@localhost:5432/testdb
      - run: npm run test:integration
        env:
          DATABASE_URL: postgres://test:test@localhost:5432/testdb
          REDIS_URL: redis://localhost:6379

  # ===== コントラクトテスト =====
  contract-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run test:contract
        env:
          PACT_BROKER_URL: ${{ secrets.PACT_BROKER_URL }}
          PACT_BROKER_TOKEN: ${{ secrets.PACT_BROKER_TOKEN }}
          GIT_COMMIT_SHA: ${{ github.sha }}
          GIT_BRANCH: ${{ github.ref_name }}

  # ===== 負荷テスト（スケジュール実行時のみ）=====
  load-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    needs: integration-tests
    steps:
      - uses: actions/checkout@v4
      - uses: grafana/k6-action@v0.3.1
        with:
          filename: load-tests/scenarios/user-api-load.js
        env:
          BASE_URL: ${{ secrets.STAGING_API_URL }}
          API_TOKEN: ${{ secrets.STAGING_API_TOKEN }}
```

---

## 11. アンチパターンと対策

### 11.1 アンチパターン1: テスト間の暗黙的な依存関係

テストが他のテストの副作用に依存している場合、テスト実行順序が変わると予期しない失敗が起きる。これは最も多いflakyテストの原因の一つである。

**問題のあるコード:**

```javascript
// アンチパターン: テストAで作成したデータにテストBが依存
describe('Orders API', () => {
  // テストA: ユーザーを作成（副作用がDBに残る）
  it('should create a user', async () => {
    await request.post('/api/v1/users')
      .send({ name: 'SharedUser', email: 'shared@example.com' })
      .expect(201);
  });

  // テストB: テストAで作成されたユーザーに依存（危険）
  it('should create an order for the user', async () => {
    const users = await request.get('/api/v1/users?filter[email]=shared@example.com');
    const userId = users.body.data[0].id; // テストAが先に実行されていないとundefined

    await request.post('/api/v1/orders')
      .send({ userId, items: [{ productId: 'p1', quantity: 1 }] })
      .expect(201);
  });
});
```

**改善されたコード:**

```javascript
// 正しいパターン: 各テストが独立してデータを準備
describe('Orders API', () => {
  let testUser;

  beforeEach(async () => {
    // 各テスト前にクリーンな状態を構築
    await db.raw('TRUNCATE TABLE users, orders CASCADE');
    testUser = await UserFactory.create({
      name: 'Test User',
      email: 'test@example.com',
    });
  });

  it('should create an order for the user', async () => {
    const res = await request.post('/api/v1/orders')
      .set('Authorization', `Bearer ${authToken}`)
      .send({
        userId: testUser.id,
        items: [{ productId: 'p1', quantity: 1 }],
      })
      .expect(201);

    expect(res.body.data.userId).toBe(testUser.id);
  });
});
```

### 11.2 アンチパターン2: タイムアウトに依存したテスト

非同期処理のテストで `setTimeout` や固定待機時間に依存すると、環境によってテストが不安定になる。

**問題のあるコード:**

```javascript
// アンチパターン: 固定のsleepで非同期処理の完了を待つ
it('should send a notification after order creation', async () => {
  await request.post('/api/v1/orders')
    .send({ userId: 'u1', items: [{ productId: 'p1', quantity: 1 }] })
    .expect(201);

  // 2秒待てば通知が送られるだろう... という希望的観測
  await new Promise(resolve => setTimeout(resolve, 2000));

  const notifications = await db('notifications').where({ userId: 'u1' });
  expect(notifications).toHaveLength(1); // CI環境では失敗する可能性が高い
});
```

**改善されたコード:**

```javascript
// 正しいパターン: ポーリングまたはイベント駆動で完了を待つ
import { waitFor } from '../helpers/async';

it('should send a notification after order creation', async () => {
  await request.post('/api/v1/orders')
    .send({ userId: 'u1', items: [{ productId: 'p1', quantity: 1 }] })
    .expect(201);

  // ポーリングで条件が満たされるまで待機（最大5秒、100msごとにチェック）
  const notifications = await waitFor(
    async () => {
      const rows = await db('notifications').where({ userId: 'u1' });
      if (rows.length === 0) throw new Error('通知がまだ作成されていない');
      return rows;
    },
    { timeout: 5000, interval: 100 }
  );

  expect(notifications).toHaveLength(1);
  expect(notifications[0].type).toBe('order_confirmation');
});

// __tests__/helpers/async.js
export async function waitFor(fn, { timeout = 5000, interval = 100 } = {}) {
  const startTime = Date.now();
  while (Date.now() - startTime < timeout) {
    try {
      return await fn();
    } catch {
      await new Promise(resolve => setTimeout(resolve, interval));
    }
  }
  throw new Error(`waitFor: ${timeout}ms 以内に条件が満たされませんでした`);
}
```

### 11.3 アンチパターン3: 本番データを使ったテスト

テスト環境で本番データのコピーを使用することは、プライバシーリスクとテストの再現性の両面で問題がある。テスト用のシードデータを明示的に管理することが推奨される。

---

## 12. エッジケース分析

### 12.1 同時リクエストによる競合状態

複数のクライアントが同時に同じリソースを操作する場合の振る舞いをテストする。

```javascript
// __tests__/edge-cases/concurrency.test.js
describe('同時リクエストの競合処理', () => {
  it('同じ商品への同時在庫引当で一方が失敗すること', async () => {
    // 在庫1個の商品を準備
    await db('products').insert({
      id: 'prod_limited',
      name: 'Limited Item',
      stock: 1,
      price: 5000,
    });

    // 2つのリクエストを同時に送信
    const [res1, res2] = await Promise.all([
      request.post('/api/v1/orders')
        .set('Authorization', `Bearer ${token1}`)
        .send({ items: [{ productId: 'prod_limited', quantity: 1 }] }),
      request.post('/api/v1/orders')
        .set('Authorization', `Bearer ${token2}`)
        .send({ items: [{ productId: 'prod_limited', quantity: 1 }] }),
    ]);

    // 一方が201、もう一方が409（在庫不足）であることを検証
    const statuses = [res1.status, res2.status].sort();
    expect(statuses).toEqual([201, 409]);

    // 在庫が負にならないことを検証
    const product = await db('products').where({ id: 'prod_limited' }).first();
    expect(product.stock).toBe(0);
  });

  it('楽観的ロック違反で適切なエラーが返ること', async () => {
    const user = await UserFactory.create({ name: 'Original' });

    // 同時に2つの更新リクエスト
    const [res1, res2] = await Promise.all([
      request.put(`/api/v1/users/${user.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .set('If-Match', `"${user.version}"`)
        .send({ name: 'Update A' }),
      request.put(`/api/v1/users/${user.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .set('If-Match', `"${user.version}"`)
        .send({ name: 'Update B' }),
    ]);

    const statuses = [res1.status, res2.status].sort();
    expect(statuses).toEqual([200, 409]);
  });
});
```

### 12.2 巨大ペイロードとレート制限

```javascript
// __tests__/edge-cases/limits.test.js
describe('ペイロードサイズとレート制限', () => {
  it('1MBを超えるリクエストボディで413を返すこと', async () => {
    const largePayload = {
      name: 'Test User',
      email: 'test@example.com',
      bio: 'x'.repeat(1024 * 1024 + 1), // 1MB超
    };

    const res = await request
      .post('/api/v1/users')
      .set('Authorization', `Bearer ${authToken}`)
      .send(largePayload);

    expect(res.status).toBe(413);
    expect(res.body.error.code).toBe('PAYLOAD_TOO_LARGE');
  });

  it('レート制限を超過すると429を返すこと', async () => {
    // レート制限が100req/分と仮定
    const requests = Array.from({ length: 110 }, () =>
      request.get('/api/v1/users')
        .set('Authorization', `Bearer ${authToken}`)
    );

    const responses = await Promise.all(requests);

    const tooManyRequests = responses.filter(r => r.status === 429);
    expect(tooManyRequests.length).toBeGreaterThan(0);

    // 429レスポンスにRetry-Afterヘッダーが含まれること
    const rateLimitedRes = tooManyRequests[0];
    expect(rateLimitedRes.headers['retry-after']).toBeDefined();
    expect(rateLimitedRes.body.error.code).toBe('RATE_LIMIT_EXCEEDED');
  });

  it('Unicode特殊文字を含むリクエストが正しく処理されること', async () => {
    const unicodePayload = {
      name: '日本語テスト emoji混在',
      email: 'unicode@example.com',
      bio: '改行\nタブ\t特殊文字<script>alert("xss")</script>',
    };

    const res = await request
      .post('/api/v1/users')
      .set('Authorization', `Bearer ${authToken}`)
      .send(unicodePayload)
      .expect(201);

    expect(res.body.data.name).toBe('日本語テスト emoji混在');
    // XSSスクリプトがエスケープまたは除去されていること
    expect(res.body.data.bio).not.toContain('<script>');
  });

  it('空配列やnull値のフィールドが正しく処理されること', async () => {
    const edgeCasePayload = {
      name: 'Edge Case User',
      email: 'edge@example.com',
      tags: [],
      metadata: null,
      preferences: {},
    };

    const res = await request
      .post('/api/v1/users')
      .set('Authorization', `Bearer ${authToken}`)
      .send(edgeCasePayload)
      .expect(201);

    expect(res.body.data.tags).toEqual([]);
    expect(res.body.data.metadata).toBeNull();
  });
});
```

---

## 13. テストツール比較表

### 13.1 テストフレームワーク比較

| 特性 | Vitest | Jest | Mocha + Chai | Playwright Test |
|------|--------|------|-------------|----------------|
| 実行速度 | 非常に高速 | 高速 | 中程度 | 中程度 |
| TypeScript対応 | ネイティブ | 変換必要 | 変換必要 | ネイティブ |
| ESM対応 | ネイティブ | 実験的 | 対応 | ネイティブ |
| ウォッチモード | HMR統合 | 標準搭載 | 別途導入 | 標準搭載 |
| スナップショット | 対応 | 対応 | 別途導入 | 対応 |
| カバレッジ | v8/istanbul | istanbul | 別途導入 | 標準搭載 |
| 並列実行 | スレッドベース | ワーカーベース | 逐次 | ワーカーベース |
| モック | vi.mock | jest.mock | sinon | 標準搭載 |
| 設定の簡潔さ | 非常に簡潔 | 中程度 | 柔軟だが冗長 | 簡潔 |
| Vite統合 | 完全統合 | なし | なし | なし |
| コミュニティ | 急成長中 | 最大規模 | 成熟 | 急成長中 |

### 13.2 負荷テストツール比較

| 特性 | k6 | Artillery | Locust | Gatling | JMeter |
|------|-----|-----------|--------|---------|--------|
| 記述言語 | JavaScript | YAML/JS | Python | Scala/Java | GUI/XML |
| 学習曲線 | 低い | 非常に低い | 低い | 中程度 | 高い |
| リソース効率 | 非常に高い | 中程度 | 中程度 | 高い | 低い |
| 分散実行 | k6 Cloud | Artillery Cloud | 標準対応 | 標準対応 | 要設定 |
| CI/CD統合 | 容易 | 容易 | 中程度 | 中程度 | 困難 |
| プロトコル | HTTP/WS/gRPC | HTTP/WS/Socket.io | HTTP/WS | HTTP/WS | 多数 |
| リアルタイム監視 | Grafana連携 | 標準ダッシュボード | Web UI | 標準レポート | リスナー |
| スクリプト柔軟性 | 高い | 中程度 | 高い | 高い | 低い |
| OSS/商用 | OSS (Cloud有料) | OSS (Cloud有料) | 完全OSS | OSS (Enterprise有) | 完全OSS |
| 推奨用途 | 汎用 | 小〜中規模 | 大規模分散 | 大規模 | レガシー |

---

## 14. 演習問題

### 14.1 演習1: 基礎レベル（ユニットテスト）

**課題:** 以下の `OrderService` クラスに対するユニットテストを作成せよ。正常系3ケース、異常系3ケース、境界値2ケースを含むこと。

```javascript
// src/services/orderService.js
export class OrderService {
  static calculateShippingCost(totalAmount, prefecture, isExpress) {
    // 基本送料
    let baseCost = 600;

    // 離島加算
    const remoteAreas = ['沖縄県', '北海道'];
    if (remoteAreas.includes(prefecture)) {
      baseCost += 500;
    }

    // 速達加算
    if (isExpress) {
      baseCost += 400;
    }

    // 一定金額以上で送料無料
    if (totalAmount >= 5000) {
      return { cost: 0, freeShipping: true, reason: '5,000円以上で送料無料' };
    }

    return { cost: baseCost, freeShipping: false, reason: null };
  }
}
```

**期待する解答の方向性:**

- 通常の都道府県でisExpress=falseの基本料金（600円）
- 北海道の加算（1,100円）
- 速達の加算（1,000円）
- 5,000円以上の送料無料
- 4,999円（境界値ぎりぎり送料あり）と5,000円（送料無料）
- 離島 + 速達の組み合わせ（1,500円）
- 負の金額やundefinedの入力に対する堅牢性

### 14.2 演習2: 中級レベル（統合テスト + モック）

**課題:** 以下の条件を満たす統合テストスイートを作成せよ。

1. `POST /api/v1/orders` エンドポイントのテスト
2. 注文作成時にStripe APIが呼ばれることをMSWでモック
3. 在庫不足の場合のエラーハンドリング
4. トランザクションのロールバック（決済失敗時に注文が保存されないこと）

**ヒント:**

```javascript
// テスト構成の骨子
describe('POST /api/v1/orders - 統合テスト', () => {
  // MSWで Stripe API をモック
  // beforeEach で商品・ユーザーデータを準備
  // afterEach でデータをクリーンアップ

  it('正常な注文フロー: 作成 -> 決済 -> 在庫更新', async () => {
    // 1. 注文作成リクエスト送信
    // 2. レスポンスの検証（201, 注文ID, ステータス）
    // 3. DBの注文レコード検証
    // 4. 在庫が減少していることを検証
  });

  it('決済失敗時: 注文がロールバックされること', async () => {
    // 1. Stripeモックをエラーレスポンスに変更
    // 2. 注文作成リクエスト送信
    // 3. レスポンスの検証（402 Payment Required）
    // 4. DBに注文レコードが存在しないこと
    // 5. 在庫が変わっていないこと
  });
});
```

### 14.3 演習3: 上級レベル（負荷テスト + パフォーマンス分析）

**課題:** k6を使って以下の要件を満たす負荷テストシナリオを設計・実装せよ。

1. **ロードテスト**: 同時50ユーザーで5分間、p95 < 500msを検証
2. **スパイクテスト**: 10->200->10ユーザーの急激な変動、エラー率 < 5%を検証
3. **ソークテスト**: 同時20ユーザーで1時間、メモリ使用量の増加傾向を観察
4. カスタムメトリクスでエンドポイントごとのレイテンシを計測
5. テスト結果をJSONで出力し、しきい値違反を検出

**評価基準:**

- シナリオ設計の適切さ（段階的なVU変化）
- 閾値の設定（p50, p95, p99, エラー率）
- カスタムメトリクスの活用
- テスト結果の可視化とレポート

---

## 15. テスト戦略チェックリスト

```
APIテスト品質チェックリスト

[ユニットテスト]
  [ ] バリデーションロジックの全パスがテストされている
  [ ] ビジネスルールの境界値が網羅されている
  [ ] エラーケースが適切にテストされている
  [ ] 依存関係がモック/スタブで分離されている
  [ ] テストカバレッジが80%以上である

[統合テスト]
  [ ] 全CRUDエンドポイントがテストされている
  [ ] 認証・認可のフローが検証されている
  [ ] ページネーション・フィルタリングがテストされている
  [ ] エラーレスポンスの形式が仕様に準拠している
  [ ] テストデータが各テストで独立に管理されている
  [ ] 冪等性（同じリクエスト2回で同じ結果）が検証されている

[コントラクトテスト]
  [ ] Consumer-Provider間の契約が定義されている
  [ ] Pact Brokerで契約が管理されている
  [ ] can-i-deployでデプロイ前チェックが実施されている
  [ ] Provider Stateが適切に設定されている

[E2Eテスト]
  [ ] 主要ユーザーシナリオ（3-5個）がテストされている
  [ ] テスト環境が本番と同等に構成されている
  [ ] 外部サービスが適切にモックされている

[負荷テスト]
  [ ] 性能目標（SLA/SLO）が明確に定義されている
  [ ] ロードテストの閾値が設定されている
  [ ] スパイクテストで障害耐性が検証されている
  [ ] 定期的な負荷テスト実行がCIに組み込まれている

[テスト運用]
  [ ] flakyテストの検出と修正プロセスがある
  [ ] テスト実行時間が許容範囲内である
  [ ] テストレポートが自動生成されている
  [ ] テストカバレッジの推移が追跡されている
```

---

## 16. よくある質問（FAQ）

### Q1: 統合テストとE2Eテストの境界はどこにあるのか

**A:** 統合テストは単一のAPIエンドポイント（または密接に関連する少数のエンドポイント）の動作を検証し、外部サービスはモックする。E2Eテストは複数のAPI/サービスを横断するユーザーシナリオを検証し、可能な限り実環境に近い構成で実行する。

具体的には、`POST /users` 単体の入力バリデーションやDB保存は統合テストであり、「ユーザー登録 -> ログイン -> プロフィール更新 -> メール確認」のような一連のフローはE2Eテストに分類される。判断に迷った場合は、テストが失敗したときに「どのコンポーネントが壊れたか」を特定できるかどうかが基準となる。特定できるならば統合テスト、特定が困難ならばE2Eテストである。

### Q2: テストカバレッジは何%を目指すべきか

**A:** 一律に数値目標を設定するのは危険だが、一般的なガイドラインとして以下の目標が参考になる。

- **ユニットテスト**: ビジネスロジック層のステートメントカバレッジ80%以上
- **統合テスト**: 全エンドポイントの正常系 + 主要な異常系（認証エラー、バリデーションエラー）がカバーされていること
- **コントラクトテスト**: Consumer-Provider間の全インタラクションがカバーされていること

カバレッジ数値よりも重要なのは「テストが実際にバグを検出できるか」という観点である。分岐カバレッジ（branch coverage）を重視し、特にエッジケースや境界値のテストが不足していないかを定期的にレビューすることが望ましい。なお、生成されたコードやボイラープレート（設定ファイルなど）にまでカバレッジを求める必要はない。

### Q3: flakyテスト（不安定なテスト）をどのように管理すべきか

**A:** flakyテストはCI/CDの信頼性を著しく低下させるため、発見次第すぐに対処することが重要である。管理手法は以下の通り。

1. **検出**: テスト実行結果を記録し、同一テストの成功/失敗のばらつきを可視化する。多くのCIツールにはflaky test検出機能が備わっている。
2. **隔離**: 発見されたflakyテストには `@flaky` タグを付与し、メインのテストスイートから一時的に隔離する。隔離中のテストは別のジョブとして実行し、メインパイプラインをブロックしないようにする。
3. **根本原因分析**: 主な原因は (a) テスト間のデータ共有、(b) 時刻依存、(c) ネットワーク遅延、(d) 非同期処理のタイミング問題である。
4. **修正**: 根本原因を特定したら速やかに修正する。1週間以上flakyのまま放置されたテストは削除を検討する。
5. **予防**: コードレビューで新しいテストにflaky要素がないか確認する。固定シードの使用、時刻のモック、ポーリングによる非同期待機などの手法を徹底する。

### Q4: マイクロサービス間のテストで最も効果的なアプローチは何か

**A:** マイクロサービス環境では、コントラクトテスト（Pact等）が最も費用対効果の高いアプローチである。各サービスのE2Eテストを全サービスの結合状態で実行しようとすると、環境構築・メンテナンスコストが爆発的に増加する。

推奨される戦略は次の通りである。

1. 各サービス内のユニットテスト・統合テストを充実させる
2. サービス間のインターフェースをコントラクトテストで保護する
3. E2Eテストは主要なビジネスフロー（3-5シナリオ）に限定する
4. Pact BrokerのWebhookを活用し、コントラクト変更時にProviderの検証を自動起動する

### Q5: APIテストにおけるテストデータ戦略のベストプラクティスは何か

**A:** テストデータ戦略は以下の3層で構成するのが望ましい。

1. **ファクトリーパターン**: テスト内で必要なデータを動的に生成する。faker等のライブラリを使いつつ、テスト目的に応じたデフォルト値をファクトリーで管理する。
2. **フィクスチャー**: 共通的なマスターデータ（商品カテゴリ、都道府県リストなど）は固定のシードファイルとして管理する。
3. **スナップショット**: 特定のテストシナリオに必要な複雑なデータセットは、スナップショットとしてJSON/SQLファイルで保持する。

原則として、各テストは自身が必要とするデータを自身のセットアップで作成し、終了後にクリーンアップすることが求められる。テスト間でデータを共有する場合は、読み取り専用のマスターデータに限定する。

---

## 17. まとめ

### テスト種別と推奨ツール

| テスト種類 | 推奨ツール | 目的 | 実行頻度 |
|-----------|-----------|------|---------|
| ユニット | Vitest, Jest | ロジックの正確性検証 | コミットごと |
| 統合 | supertest + Vitest | エンドポイントの動作検証 | コミットごと |
| コントラクト | Pact | サービス間の仕様合意 | PR作成時 |
| E2E | supertest / Playwright | シナリオ全体の検証 | デプロイ前 |
| 負荷 | k6, Artillery | パフォーマンスの検証 | 定期/リリース前 |
| ファジング | Schemathesis | エッジケースの自動発見 | 週次 |
| セキュリティ | OWASP ZAP | 脆弱性の検出 | リリース前 |

### テスト戦略設計の指針

1. テストピラミッドに従い、ユニットテストを最も多く、E2Eテストを最小限にする
2. 各テストは独立して実行可能であること（テスト間の依存関係を排除）
3. CI/CDパイプラインに組み込み、自動実行されること
4. テスト結果をレポートとして可視化し、継続的に品質を追跡すること
5. flakyテストは即座に対処し、テストスイートの信頼性を維持すること

---

## 次に読むべきガイド
→ [[01-monitoring-and-logging.md]] -- 監視とロギング

---

## 参考文献

1. k6. "Load Testing for Engineering Teams." k6.io, 2024. https://k6.io/docs/
2. Pact Foundation. "Consumer-Driven Contract Testing." pact.io, 2024. https://docs.pact.io/
3. Schemathesis. "Property-Based API Testing with OpenAPI." github.com/schemathesis, 2024. https://github.com/schemathesis/schemathesis
4. Martin Fowler. "Testing Strategies in a Microservice Architecture." martinfowler.com, 2014. https://martinfowler.com/articles/microservice-testing/
5. MSW (Mock Service Worker). "API mocking of the next generation." mswjs.io, 2024. https://mswjs.io/docs/
6. Postman. "API Test Automation." postman.com, 2024. https://learning.postman.com/docs/
7. Artillery. "Cloud-Scale Load Testing." artillery.io, 2024. https://www.artillery.io/docs

