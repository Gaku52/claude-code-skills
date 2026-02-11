# Zod バリデーション完全ガイド

> TypeScript ファーストのスキーマ定義ライブラリ Zod で、ランタイムバリデーションと型推論を統合する

## この章で学ぶこと

1. **スキーマ定義の基本** -- プリミティブ型からオブジェクト、配列、ユニオンまでの定義パターン
2. **高度なバリデーション** -- transform, refine, pipe, discriminatedUnion による複雑なスキーマ設計
3. **実践的な統合** -- フォームバリデーション、API リクエスト/レスポンス、環境変数検証への適用

---

## 1. スキーマ定義の基本

### 1-1. プリミティブ型

```typescript
import { z } from "zod";

// プリミティブ
const stringSchema = z.string();
const numberSchema = z.number();
const boolSchema = z.boolean();
const dateSchema = z.date();
const bigintSchema = z.bigint();

// リテラル
const literalSchema = z.literal("active");
const numLiteral = z.literal(42);

// enum
const statusSchema = z.enum(["active", "inactive", "pending"]);
type Status = z.infer<typeof statusSchema>; // "active" | "inactive" | "pending"

// native enum
enum Direction {
  Up = "UP",
  Down = "DOWN",
}
const directionSchema = z.nativeEnum(Direction);

// parse と safeParse
const result = stringSchema.parse("hello");      // "hello" (失敗時は throw)
const safe = stringSchema.safeParse(123);         // { success: false, error: ZodError }
if (safe.success) {
  console.log(safe.data); // 型: string
}
```

### 1-2. 文字列バリデーション

```typescript
const emailSchema = z.string()
  .email("有効なメールアドレスを入力してください")
  .min(5, "5文字以上で入力してください")
  .max(255, "255文字以内で入力してください");

const urlSchema = z.string().url();
const uuidSchema = z.string().uuid();
const emojiSchema = z.string().emoji();
const datetimeSchema = z.string().datetime(); // ISO 8601
const ipSchema = z.string().ip();
const regexSchema = z.string().regex(/^[A-Z]{3}-\d{4}$/);

// trim + toLowerCase をバリデーション前に適用
const normalizedEmail = z.string()
  .trim()
  .toLowerCase()
  .email();
```

### 1-3. 数値バリデーション

```typescript
const ageSchema = z.number()
  .int("整数を入力してください")
  .min(0, "0以上の値を入力してください")
  .max(150, "150以下の値を入力してください");

const priceSchema = z.number()
  .positive("正の値を入力してください")
  .multipleOf(0.01); // 小数第2位まで

const percentSchema = z.number().min(0).max(100);
```

---

## 2. オブジェクトと配列

### 2-1. オブジェクトスキーマ

```
Zod オブジェクトスキーマと型推論:

  z.object({                     type User = {
    name: z.string(),    ------>   name: string;
    age: z.number(),     ------>   age: number;
    email: z.string()    ------>   email: string;
      .email(),                      // (検証ルールは型に影響しない)
  })                             }

  z.infer<typeof schema> で自動推論
```

```typescript
// オブジェクトスキーマ
const UserSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  age: z.number().int().min(0).optional(),
  role: z.enum(["user", "admin"]).default("user"),
  tags: z.array(z.string()).default([]),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

type User = z.infer<typeof UserSchema>;
// {
//   name: string;
//   email: string;
//   age?: number | undefined;
//   role: "user" | "admin";     // default があるので optional ではない
//   tags: string[];
//   metadata?: Record<string, unknown> | undefined;
// }

// 入力型と出力型が異なるスキーマ
type UserInput = z.input<typeof UserSchema>;
// age? は number | undefined
// role? は "user" | "admin" | undefined (default 適用前)
// tags? は string[] | undefined

type UserOutput = z.output<typeof UserSchema>;
// role は "user" | "admin" (default 適用後)
// tags は string[]
```

### 2-2. オブジェクトの操作

```typescript
// pick / omit
const UserCreateSchema = UserSchema.pick({
  name: true,
  email: true,
  age: true,
});

const UserPublicSchema = UserSchema.omit({
  metadata: true,
});

// partial / required
const UserUpdateSchema = UserSchema.partial(); // 全フィールド optional
const UserStrictSchema = UserSchema.required(); // 全フィールド required

// merge / extend
const UserWithIdSchema = UserSchema.extend({
  id: z.string().uuid(),
  createdAt: z.date(),
  updatedAt: z.date(),
});

// passthrough / strict / strip
const strictSchema = UserSchema.strict(); // 余分なフィールドでエラー
const passthroughSchema = UserSchema.passthrough(); // 余分なフィールドを保持
// デフォルト (strip): 余分なフィールドを除去
```

### 2-3. 配列とタプル

```typescript
// 配列
const tagsSchema = z.array(z.string()).min(1).max(10);
const uniqueTags = z.array(z.string()).refine(
  (items) => new Set(items).size === items.length,
  { message: "タグは重複できません" }
);

// タプル
const coordinateSchema = z.tuple([z.number(), z.number()]);
type Coordinate = z.infer<typeof coordinateSchema>; // [number, number]

// 可変長タプル
const argsSchema = z.tuple([z.string(), z.number()]).rest(z.boolean());
type Args = z.infer<typeof argsSchema>; // [string, number, ...boolean[]]
```

---

## 3. 高度なパターン

### 3-1. discriminatedUnion

```typescript
// 判別共用体スキーマ
const PaymentSchema = z.discriminatedUnion("method", [
  z.object({
    method: z.literal("credit_card"),
    cardNumber: z.string().regex(/^\d{16}$/),
    expiry: z.string().regex(/^\d{2}\/\d{2}$/),
    cvv: z.string().regex(/^\d{3,4}$/),
  }),
  z.object({
    method: z.literal("bank_transfer"),
    bankCode: z.string().length(4),
    accountNumber: z.string(),
  }),
  z.object({
    method: z.literal("wallet"),
    walletId: z.string().uuid(),
  }),
]);

type Payment = z.infer<typeof PaymentSchema>;
```

### 3-2. transform と pipe

```
transform のフロー:

  入力値  -->  バリデーション  -->  変換  -->  出力値
  "123"       z.string()          Number()     123
              (string チェック)    (string→number)
```

```typescript
// transform: バリデーション後に値を変換
const StringToNumberSchema = z.string()
  .transform((val) => Number(val))
  .pipe(z.number().positive()); // 変換後の値をさらにバリデーション

const result = StringToNumberSchema.parse("42"); // 42 (number)

// 日付文字列をDateに変換
const DateStringSchema = z.string()
  .datetime()
  .transform((val) => new Date(val));

// coerce: 暗黙的な型変換
const CoerceNumberSchema = z.coerce.number(); // Number(input)
const CoerceDateSchema = z.coerce.date();     // new Date(input)
const CoerceBoolSchema = z.coerce.boolean();  // Boolean(input)
```

### 3-3. refine と superRefine

```typescript
// refine: カスタムバリデーション
const PasswordSchema = z.string()
  .min(8, "8文字以上")
  .refine((val) => /[A-Z]/.test(val), "大文字を含めてください")
  .refine((val) => /[0-9]/.test(val), "数字を含めてください")
  .refine((val) => /[!@#$%]/.test(val), "特殊文字を含めてください");

// superRefine: 複数フィールドにまたがるバリデーション
const RegisterSchema = z.object({
  password: z.string().min(8),
  confirmPassword: z.string(),
}).superRefine((data, ctx) => {
  if (data.password !== data.confirmPassword) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "パスワードが一致しません",
      path: ["confirmPassword"],
    });
  }
});
```

### 3-4. 再帰型スキーマ

```typescript
// 再帰的なツリー構造
type Category = {
  name: string;
  children: Category[];
};

const CategorySchema: z.ZodType<Category> = z.lazy(() =>
  z.object({
    name: z.string(),
    children: z.array(CategorySchema),
  })
);
```

---

## 4. 実践的な統合

### 4-1. 環境変数バリデーション

```typescript
// env.ts
const EnvSchema = z.object({
  NODE_ENV: z.enum(["development", "production", "test"]),
  PORT: z.coerce.number().default(3000),
  DATABASE_URL: z.string().url(),
  REDIS_URL: z.string().url().optional(),
  API_KEY: z.string().min(32),
  LOG_LEVEL: z.enum(["debug", "info", "warn", "error"]).default("info"),
});

// アプリ起動時に検証
export const env = EnvSchema.parse(process.env);
// 型: { NODE_ENV: "development" | ..., PORT: number, ... }
```

### 4-2. API レスポンスバリデーション

```typescript
// API レスポンスのスキーマ
const ApiResponseSchema = <T extends z.ZodTypeAny>(dataSchema: T) =>
  z.object({
    success: z.literal(true),
    data: dataSchema,
    meta: z.object({
      page: z.number(),
      total: z.number(),
    }).optional(),
  });

const UserListResponseSchema = ApiResponseSchema(z.array(UserSchema));

// フェッチ関数
async function fetchUsers(): Promise<z.infer<typeof UserListResponseSchema>> {
  const response = await fetch("/api/users");
  const json = await response.json();
  return UserListResponseSchema.parse(json);
}
```

---

## 比較表

### バリデーションライブラリ比較

| ライブラリ | サイズ | 型推論 | パフォーマンス | API スタイル |
|-----------|--------|--------|-------------|-------------|
| zod | ~14KB | 最高 | 良好 | メソッドチェーン |
| yup | ~15KB | 中 | 良好 | メソッドチェーン |
| joi | ~30KB | 低(@types) | 良好 | メソッドチェーン |
| superstruct | ~3KB | 高 | 良好 | 関数合成 |
| valibot | ~1KB | 高 | 最高 | 関数合成 |
| typia | 0KB(生成) | 最高 | 最高 | デコレータ |
| arktype | ~5KB | 最高 | 最高 | 文字列DSL |

### parse vs safeParse

| メソッド | 失敗時 | 戻り値型 | 用途 |
|---------|--------|---------|------|
| `.parse()` | ZodError throw | `T` | 信頼できる内部データ |
| `.safeParse()` | `{ success: false }` | `SafeParseResult<T>` | ユーザー入力、API |
| `.parseAsync()` | ZodError throw | `Promise<T>` | async refine 使用時 |
| `.safeParseAsync()` | `{ success: false }` | `Promise<SafeParseResult<T>>` | async 安全版 |

---

## アンチパターン

### AP-1: スキーマと型を二重定義する

```typescript
// NG: 型とスキーマを別々に定義（同期が崩れるリスク）
interface User {
  name: string;
  email: string;
  age: number;
}

const UserSchema = z.object({
  name: z.string(),
  email: z.string().email(),
  age: z.number(), // interface と齟齬が生じやすい
});

// OK: スキーマから型を推論
const UserSchema = z.object({
  name: z.string(),
  email: z.string().email(),
  age: z.number().int().min(0),
});
type User = z.infer<typeof UserSchema>;
// 単一の情報源（Single Source of Truth）
```

### AP-2: parse を catch なしで使う

```typescript
// NG: parse の例外を処理しない
app.post("/users", (req, res) => {
  const data = UserSchema.parse(req.body); // ZodError が throw される可能性
  // ...
});

// OK: safeParse で安全に処理
app.post("/users", (req, res) => {
  const result = UserSchema.safeParse(req.body);
  if (!result.success) {
    return res.status(400).json({
      errors: result.error.flatten().fieldErrors,
    });
  }
  const data = result.data; // 検証済み
});
```

---

## FAQ

### Q1: zod と valibot のどちらを選ぶべきですか？

zod はエコシステムが最も充実しており、tRPC, React Hook Form, Prisma などとの連携プラグインが豊富です。valibot はバンドルサイズが圧倒的に小さく（Tree-shakable）、パフォーマンスも優れています。新規の小〜中規模プロジェクトでは valibot、エコシステムとの統合が重要な場合は zod を選択してください。

### Q2: zod はサーバーサイドとクライアントサイドの両方で使えますか？

はい。zod は環境非依存で、Node.js, ブラウザ, Edge Runtime 全てで動作します。同じスキーマ定義をサーバーのリクエスト検証とクライアントのフォームバリデーションの両方で共有できるのが大きな利点です。

### Q3: 大量のデータを検証する場合のパフォーマンスは？

数千件程度の配列は問題ありません。数万件以上の場合は、事前にサイズチェックを入れるか、ストリーム処理を検討してください。コンパイル時にバリデーションコードを生成する typia を使えば、ランタイムのパフォーマンスが最大限になります。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| z.infer | スキーマから TypeScript 型を自動推論 |
| safeParse | 例外を投げずに検証結果を返す |
| transform | バリデーション後に値を変換 |
| discriminatedUnion | 判別子フィールドで型を分岐 |
| refine / superRefine | カスタムバリデーションロジック |
| brand | ブランド型の付与 |

---

## 次に読むべきガイド

- [エラーハンドリング](../02-patterns/00-error-handling.md) -- zod と Result 型の統合
- [tRPC](./02-trpc.md) -- zod をスキーマとして使う型安全 API
- [ブランド型](../02-patterns/03-branded-types.md) -- zod の `.brand()` 活用

---

## 参考文献

1. **Zod Documentation**
   https://zod.dev/

2. **Zod GitHub Repository**
   https://github.com/colinhacks/zod

3. **Total TypeScript - Zod Tutorial**
   https://www.totaltypescript.com/tutorials/zod
