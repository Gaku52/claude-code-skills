# Zod バリデーション完全ガイド

> TypeScript ファーストのスキーマ定義ライブラリ Zod で、ランタイムバリデーションと型推論を統合する

## この章で学ぶこと

1. **スキーマ定義の基本** -- プリミティブ型からオブジェクト、配列、ユニオンまでの定義パターン
2. **高度なバリデーション** -- transform, refine, pipe, discriminatedUnion による複雑なスキーマ設計
3. **実践的な統合** -- フォームバリデーション、API リクエスト/レスポンス、環境変数検証への適用
4. **エラーハンドリング** -- ZodError の解析、カスタムエラーメッセージ、国際化対応
5. **パフォーマンスとベストプラクティス** -- スキーマ設計の指針、テスト、エコシステム連携

---

## 1. スキーマ定義の基本

### Zodとは何か

Zod は TypeScript ファーストのスキーマ宣言・バリデーションライブラリである。最大の特徴は、スキーマ定義から TypeScript の型を自動推論できること。これにより「型とバリデーションの二重定義」問題を解決し、Single Source of Truth（単一の情報源）を実現する。

```
Zod の核心的な価値:

  従来のアプローチ（二重定義の問題）:
  ┌─────────────────────┐     ┌─────────────────────┐
  │ TypeScript 型定義    │     │ バリデーション       │
  │ interface User {     │     │ function validate(x) │
  │   name: string;      │ ←→ │   if (!x.name) ...   │
  │   age: number;       │     │   if (!x.age) ...    │
  │ }                    │     │ }                    │
  └─────────────────────┘     └─────────────────────┘
     手動同期が必要 → 乖離リスク

  Zod のアプローチ（Single Source of Truth）:
  ┌─────────────────────────────┐
  │ const UserSchema = z.object({│
  │   name: z.string(),          │
  │   age: z.number(),           │
  │ })                           │
  │                              │
  │ type User = z.infer<...>     │ ← 型は自動推論
  │ schema.parse(data)           │ ← バリデーション機能も内蔵
  └─────────────────────────────┘
```

### 1-1. プリミティブ型

```typescript
import { z } from "zod";

// プリミティブ
const stringSchema = z.string();
const numberSchema = z.number();
const boolSchema = z.boolean();
const dateSchema = z.date();
const bigintSchema = z.bigint();
const undefinedSchema = z.undefined();
const nullSchema = z.null();
const voidSchema = z.void();
const anySchema = z.any();
const unknownSchema = z.unknown();
const neverSchema = z.never();

// リテラル
const literalSchema = z.literal("active");
const numLiteral = z.literal(42);
const boolLiteral = z.literal(true);

// enum
const statusSchema = z.enum(["active", "inactive", "pending"]);
type Status = z.infer<typeof statusSchema>; // "active" | "inactive" | "pending"

// enum の値一覧を取得
statusSchema.options; // ["active", "inactive", "pending"]

// enum にない値を検証
statusSchema.safeParse("unknown"); // { success: false, ... }

// native enum
enum Direction {
  Up = "UP",
  Down = "DOWN",
  Left = "LEFT",
  Right = "RIGHT",
}
const directionSchema = z.nativeEnum(Direction);
type Dir = z.infer<typeof directionSchema>; // Direction

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
const cuuidSchema = z.string().cuid();
const cuid2Schema = z.string().cuid2();
const ulidSchema = z.string().ulid();
const emojiSchema = z.string().emoji();
const datetimeSchema = z.string().datetime(); // ISO 8601
const ipSchema = z.string().ip(); // IPv4 or IPv6
const ipv4Schema = z.string().ip({ version: "v4" });
const ipv6Schema = z.string().ip({ version: "v6" });
const regexSchema = z.string().regex(/^[A-Z]{3}-\d{4}$/);

// trim + toLowerCase をバリデーション前に適用
const normalizedEmail = z.string()
  .trim()
  .toLowerCase()
  .email();

// 文字列バリデーションの全メソッド
const fullStringValidation = z.string()
  .min(1, "必須項目です")           // 最小文字数
  .max(100, "100文字以内")          // 最大文字数
  .length(10, "10文字ちょうど")     // 固定長
  .startsWith("https://")           // 前方一致
  .endsWith(".com")                 // 後方一致
  .includes("example")              // 部分一致
  .trim()                           // 前後の空白を除去
  .toLowerCase()                    // 小文字変換
  .toUpperCase();                   // 大文字変換

// 日本語対応の文字列バリデーション
const japaneseNameSchema = z.string()
  .min(1, "氏名を入力してください")
  .max(50, "50文字以内で入力してください")
  .regex(/^[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}ー\s]+$/u, "日本語で入力してください");

const phoneSchema = z.string()
  .regex(/^0\d{1,4}-?\d{1,4}-?\d{4}$/, "有効な電話番号を入力してください");

const postalCodeSchema = z.string()
  .regex(/^\d{3}-?\d{4}$/, "有効な郵便番号を入力してください")
  .transform((val) => val.replace("-", "").replace(/(\d{3})(\d{4})/, "$1-$2"));
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

// 数値バリデーションの全メソッド
const fullNumberValidation = z.number()
  .int()              // 整数
  .positive()         // 正の数 (> 0)
  .nonnegative()      // 非負 (>= 0)
  .negative()         // 負の数 (< 0)
  .nonpositive()      // 非正 (<= 0)
  .multipleOf(5)      // 5の倍数
  .min(0)             // 最小値
  .max(100)           // 最大値
  .gt(0)              // より大きい (greater than)
  .gte(0)             // 以上 (greater than or equal)
  .lt(100)            // より小さい (less than)
  .lte(100)           // 以下 (less than or equal)
  .finite()           // 有限数（Infinity を除外）
  .safe();            // Number.MIN_SAFE_INTEGER 〜 MAX_SAFE_INTEGER

// NaN のハンドリング
const safeNumber = z.number().refine((n) => !Number.isNaN(n), "数値を入力してください");
```

### 1-4. 日付バリデーション

```typescript
const dateSchema = z.date();

// 日付の範囲チェック
const futureDate = z.date().min(new Date(), "未来の日付を指定してください");
const pastDate = z.date().max(new Date(), "過去の日付を指定してください");

// 文字列からDateに変換するスキーマ
const dateStringSchema = z.string()
  .datetime()
  .transform((val) => new Date(val));

// coerce で自動変換
const coerceDateSchema = z.coerce.date();
coerceDateSchema.parse("2024-01-15"); // Date オブジェクト
coerceDateSchema.parse(1705276800000); // Date オブジェクト（timestamp）
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

### z.input vs z.output vs z.infer の違い

```
  z.input<typeof Schema>     変換前の入力型（transform, default 適用前）
  z.output<typeof Schema>    変換後の出力型（transform, default 適用後）
  z.infer<typeof Schema>     z.output と同じ（エイリアス）

  例: z.string().default("hello")
    z.input  → string | undefined
    z.output → string
    z.infer  → string

  例: z.string().transform(Number)
    z.input  → string
    z.output → number
    z.infer  → number
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

// deepPartial（ネストされたオブジェクトも全て optional）
const DeepPartialUser = UserSchema.deepPartial();

// merge / extend
const UserWithIdSchema = UserSchema.extend({
  id: z.string().uuid(),
  createdAt: z.date(),
  updatedAt: z.date(),
});

// 2つのスキーマを merge
const PersonSchema = z.object({ name: z.string(), age: z.number() });
const ContactSchema = z.object({ email: z.string(), phone: z.string() });
const PersonContactSchema = PersonSchema.merge(ContactSchema);

// passthrough / strict / strip
const strictSchema = UserSchema.strict(); // 余分なフィールドでエラー
const passthroughSchema = UserSchema.passthrough(); // 余分なフィールドを保持
// デフォルト (strip): 余分なフィールドを除去

// catchall: 未定義のキーのバリデーション
const configSchema = z.object({
  host: z.string(),
  port: z.number(),
}).catchall(z.string());
// { host: string; port: number; [key: string]: string }
```

### 2-3. 配列とタプル

```typescript
// 配列
const tagsSchema = z.array(z.string()).min(1).max(10);
const uniqueTags = z.array(z.string()).refine(
  (items) => new Set(items).size === items.length,
  { message: "タグは重複できません" }
);

// nonempty: 少なくとも1要素ある配列
const nonEmptyArray = z.array(z.number()).nonempty();
type NonEmptyNumbers = z.infer<typeof nonEmptyArray>;
// [number, ...number[]]

// タプル
const coordinateSchema = z.tuple([z.number(), z.number()]);
type Coordinate = z.infer<typeof coordinateSchema>; // [number, number]

// 可変長タプル
const argsSchema = z.tuple([z.string(), z.number()]).rest(z.boolean());
type Args = z.infer<typeof argsSchema>; // [string, number, ...boolean[]]

// record: 動的キーのオブジェクト
const scoresSchema = z.record(z.string(), z.number());
type Scores = z.infer<typeof scoresSchema>; // Record<string, number>

// キーにもバリデーションを適用
const envSchema = z.record(
  z.string().regex(/^[A-Z_]+$/), // キーは大文字+アンダースコアのみ
  z.string(),
);

// Map と Set
const mapSchema = z.map(z.string(), z.number());
const setSchema = z.set(z.string());
type MyMap = z.infer<typeof mapSchema>; // Map<string, number>
type MySet = z.infer<typeof setSchema>; // Set<string>
```

### 2-4. Union と Intersection

```typescript
// union
const stringOrNumber = z.union([z.string(), z.number()]);
// 省略記法
const stringOrNumber2 = z.string().or(z.number());

// discriminatedUnion
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

// intersection
const hasId = z.object({ id: z.string().uuid() });
const hasTimestamps = z.object({
  createdAt: z.date(),
  updatedAt: z.date(),
});
const entitySchema = z.intersection(hasId, hasTimestamps);
// 省略記法
const entitySchema2 = hasId.and(hasTimestamps);

// nullable / optional / nullish
const nullableString = z.string().nullable();     // string | null
const optionalString = z.string().optional();     // string | undefined
const nullishString = z.string().nullish();       // string | null | undefined
```

---

## 3. 高度なパターン

### 3-1. discriminatedUnion の詳細

```typescript
// discriminatedUnion vs union の比較
// discriminatedUnion は判別子で高速にバリデーション
// union は各メンバーを順番に試行（遅い）

const ShapeSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("circle"),
    radius: z.number().positive(),
  }),
  z.object({
    type: z.literal("rectangle"),
    width: z.number().positive(),
    height: z.number().positive(),
  }),
  z.object({
    type: z.literal("triangle"),
    base: z.number().positive(),
    height: z.number().positive(),
  }),
]);

// エラーメッセージが的確
ShapeSchema.safeParse({ type: "circle", radius: -1 });
// → "radius must be positive" (circle スキーマ内でエラー)

// union だと全メンバーのエラーが列挙されて分かりにくい
```

### 3-2. transform と pipe

```
transform のフロー:

  入力値  -->  バリデーション  -->  変換  -->  出力値
  "123"       z.string()          Number()     123
              (string チェック)    (string→number)

pipe のフロー:

  入力値  -->  前段スキーマ  -->  変換  -->  後段スキーマ  -->  出力値
  "123"       z.string()         Number()     z.number()        123
              (string チェック)   (変換)       .positive()
                                              (number チェック)
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
const CoerceStringSchema = z.coerce.string(); // String(input)
const CoerceBigintSchema = z.coerce.bigint(); // BigInt(input)

// 実践的なtransform例
const MoneySchema = z.object({
  amount: z.string()
    .regex(/^\d+(\.\d{1,2})?$/, "金額の形式が不正です")
    .transform((val) => Math.round(parseFloat(val) * 100)), // セント変換
  currency: z.enum(["USD", "EUR", "JPY"]),
});

type Money = z.infer<typeof MoneySchema>;
// { amount: number; currency: "USD" | "EUR" | "JPY" }

MoneySchema.parse({ amount: "19.99", currency: "USD" });
// { amount: 1999, currency: "USD" }

// CSV行をオブジェクトに変換
const CsvRowSchema = z.string()
  .transform((row) => row.split(","))
  .pipe(z.tuple([z.string(), z.string(), z.coerce.number()]))
  .transform(([name, email, age]) => ({ name, email, age }));

CsvRowSchema.parse("Alice,alice@test.com,30");
// { name: "Alice", email: "alice@test.com", age: 30 }
```

### 3-3. refine と superRefine

```typescript
// refine: カスタムバリデーション
const PasswordSchema = z.string()
  .min(8, "8文字以上")
  .refine((val) => /[A-Z]/.test(val), "大文字を含めてください")
  .refine((val) => /[a-z]/.test(val), "小文字を含めてください")
  .refine((val) => /[0-9]/.test(val), "数字を含めてください")
  .refine((val) => /[!@#$%^&*]/.test(val), "特殊文字を含めてください");

// refine に path を指定
const DateRangeSchema = z.object({
  startDate: z.date(),
  endDate: z.date(),
}).refine(
  (data) => data.endDate > data.startDate,
  {
    message: "終了日は開始日より後にしてください",
    path: ["endDate"], // エラーを endDate フィールドに紐付け
  }
);

// superRefine: 複数フィールドにまたがるバリデーション
const RegisterSchema = z.object({
  password: z.string().min(8),
  confirmPassword: z.string(),
  email: z.string().email(),
  acceptTerms: z.boolean(),
}).superRefine((data, ctx) => {
  if (data.password !== data.confirmPassword) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "パスワードが一致しません",
      path: ["confirmPassword"],
    });
  }

  if (!data.acceptTerms) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "利用規約に同意してください",
      path: ["acceptTerms"],
    });
  }
});

// superRefine で非同期バリデーション
const UniqueEmailSchema = z.object({
  email: z.string().email(),
}).superRefine(async (data, ctx) => {
  const exists = await checkEmailExists(data.email);
  if (exists) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "このメールアドレスは既に使用されています",
      path: ["email"],
    });
  }
});

// 非同期バリデーションは parseAsync / safeParseAsync で使用
const result = await UniqueEmailSchema.safeParseAsync({
  email: "test@example.com",
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

// 再帰的なJSON型
type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(JsonValueSchema),
    z.record(JsonValueSchema),
  ])
);

// ネストの深さ制限付き再帰
function createNestedSchema(maxDepth: number): z.ZodTypeAny {
  if (maxDepth <= 0) {
    return z.object({ name: z.string() });
  }
  return z.object({
    name: z.string(),
    children: z.array(createNestedSchema(maxDepth - 1)).optional(),
  });
}

const shallowTree = createNestedSchema(3); // 最大3階層
```

### 3-5. preprocess と preprocessor パターン

```typescript
// preprocess: バリデーション前にデータを前処理
const NumberFromString = z.preprocess(
  (val) => (typeof val === "string" ? Number(val) : val),
  z.number(),
);

NumberFromString.parse("42"); // 42
NumberFromString.parse(42);   // 42

// フォームデータの前処理（空文字を undefined に変換）
const FormFieldSchema = z.preprocess(
  (val) => (val === "" ? undefined : val),
  z.string().optional(),
);

// チェックボックスの値を boolean に変換
const CheckboxSchema = z.preprocess(
  (val) => val === "on" || val === "true" || val === true,
  z.boolean(),
);
```

### 3-6. ブランド型（Branded Types）

```typescript
// brand でブランド型を付与
const UserIdSchema = z.string().uuid().brand<"UserId">();
type UserId = z.infer<typeof UserIdSchema>;
// string & { __brand: "UserId" }

const OrderIdSchema = z.string().uuid().brand<"OrderId">();
type OrderId = z.infer<typeof OrderIdSchema>;

function getUserById(id: UserId): Promise<User> {
  // UserId型のみ受け入れる
  return fetch(`/api/users/${id}`).then((r) => r.json());
}

const userId = UserIdSchema.parse("550e8400-e29b-41d4-a716-446655440000");
const orderId = OrderIdSchema.parse("550e8400-e29b-41d4-a716-446655440001");

getUserById(userId);  // OK
// getUserById(orderId); // エラー: OrderId は UserId に代入できない
// getUserById("raw-string"); // エラー: string は UserId に代入できない
```

---

## 4. 実践的な統合

### 4-1. 環境変数バリデーション

```typescript
// env.ts
const EnvSchema = z.object({
  // サーバー設定
  NODE_ENV: z.enum(["development", "production", "test"]),
  PORT: z.coerce.number().default(3000),
  HOST: z.string().default("0.0.0.0"),

  // データベース
  DATABASE_URL: z.string().url(),
  DATABASE_POOL_SIZE: z.coerce.number().int().min(1).max(50).default(10),

  // Redis
  REDIS_URL: z.string().url().optional(),

  // 認証
  JWT_SECRET: z.string().min(32),
  JWT_EXPIRES_IN: z.string().default("7d"),

  // 外部API
  API_KEY: z.string().min(32),
  API_BASE_URL: z.string().url(),

  // ログ
  LOG_LEVEL: z.enum(["debug", "info", "warn", "error"]).default("info"),

  // メール
  SMTP_HOST: z.string().optional(),
  SMTP_PORT: z.coerce.number().optional(),
  SMTP_USER: z.string().optional(),
  SMTP_PASS: z.string().optional(),
}).superRefine((env, ctx) => {
  // SMTP 設定は全部指定するか全部省略するか
  const smtpFields = [env.SMTP_HOST, env.SMTP_PORT, env.SMTP_USER, env.SMTP_PASS];
  const defined = smtpFields.filter((f) => f !== undefined).length;
  if (defined > 0 && defined < 4) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "SMTP設定は全て指定するか、全て省略してください",
      path: ["SMTP_HOST"],
    });
  }
});

// アプリ起動時に検証
function loadEnv() {
  const result = EnvSchema.safeParse(process.env);
  if (!result.success) {
    console.error("環境変数の検証に失敗しました:");
    for (const issue of result.error.issues) {
      console.error(`  ${issue.path.join(".")}: ${issue.message}`);
    }
    process.exit(1);
  }
  return result.data;
}

export const env = loadEnv();
// 型: { NODE_ENV: "development" | ..., PORT: number, ... }
```

### 4-2. API レスポンスバリデーション

```typescript
// 汎用的なAPIレスポンススキーマ
const ApiSuccessSchema = <T extends z.ZodTypeAny>(dataSchema: T) =>
  z.object({
    success: z.literal(true),
    data: dataSchema,
    meta: z.object({
      page: z.number().int(),
      pageSize: z.number().int(),
      total: z.number().int(),
      hasNext: z.boolean(),
    }).optional(),
  });

const ApiErrorSchema = z.object({
  success: z.literal(false),
  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.array(z.object({
      field: z.string(),
      message: z.string(),
    })).optional(),
  }),
});

const ApiResponseSchema = <T extends z.ZodTypeAny>(dataSchema: T) =>
  z.discriminatedUnion("success", [
    ApiSuccessSchema(dataSchema),
    ApiErrorSchema,
  ]);

const UserListResponseSchema = ApiResponseSchema(z.array(UserSchema));

// 型安全なフェッチ関数
async function fetchApi<T extends z.ZodTypeAny>(
  url: string,
  schema: T,
): Promise<z.infer<T>> {
  const response = await fetch(url);
  const json = await response.json();
  return schema.parse(json);
}

// 使用例
const usersResponse = await fetchApi(
  "/api/users",
  ApiResponseSchema(z.array(UserSchema)),
);

if (usersResponse.success) {
  // usersResponse.data の型は User[]
  console.log(usersResponse.data);
} else {
  // usersResponse.error の型
  console.error(usersResponse.error.message);
}
```

### 4-3. フォームバリデーション（React Hook Form + Zod）

```typescript
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

const ContactFormSchema = z.object({
  name: z.string()
    .min(1, "氏名を入力してください")
    .max(100, "100文字以内で入力してください"),
  email: z.string()
    .min(1, "メールアドレスを入力してください")
    .email("有効なメールアドレスを入力してください"),
  category: z.enum(["inquiry", "support", "feedback"], {
    errorMap: () => ({ message: "カテゴリを選択してください" }),
  }),
  message: z.string()
    .min(10, "10文字以上で入力してください")
    .max(1000, "1000文字以内で入力してください"),
  attachments: z.array(z.instanceof(File))
    .max(3, "ファイルは最大3つまでです")
    .refine(
      (files) => files.every((f) => f.size <= 5 * 1024 * 1024),
      "各ファイルは5MB以下にしてください",
    )
    .optional(),
});

type ContactForm = z.infer<typeof ContactFormSchema>;

function ContactFormComponent() {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<ContactForm>({
    resolver: zodResolver(ContactFormSchema),
    defaultValues: {
      category: "inquiry",
    },
  });

  const onSubmit = async (data: ContactForm) => {
    // data は検証済みの ContactForm 型
    await submitForm(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register("name")} />
      {errors.name && <span>{errors.name.message}</span>}

      <input {...register("email")} />
      {errors.email && <span>{errors.email.message}</span>}

      <select {...register("category")}>
        <option value="inquiry">お問い合わせ</option>
        <option value="support">サポート</option>
        <option value="feedback">フィードバック</option>
      </select>

      <textarea {...register("message")} />
      {errors.message && <span>{errors.message.message}</span>}

      <button type="submit" disabled={isSubmitting}>送信</button>
    </form>
  );
}
```

### 4-4. Express / Hono ミドルウェア

```typescript
import { z } from "zod";
import type { Request, Response, NextFunction } from "express";

// 汎用バリデーションミドルウェア
function validate<T extends z.ZodTypeAny>(schema: T) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse({
      body: req.body,
      query: req.query,
      params: req.params,
    });

    if (!result.success) {
      return res.status(400).json({
        success: false,
        errors: result.error.flatten().fieldErrors,
      });
    }

    // 検証済みデータを req に格納
    req.body = result.data.body;
    req.query = result.data.query;
    req.params = result.data.params;
    next();
  };
}

// ルート定義
const CreateUserSchema = z.object({
  body: z.object({
    name: z.string().min(1),
    email: z.string().email(),
  }),
  query: z.object({}),
  params: z.object({}),
});

app.post("/users", validate(CreateUserSchema), (req, res) => {
  // req.body は { name: string; email: string } として型安全
  const user = createUser(req.body);
  res.json(user);
});
```

---

## 5. エラーハンドリング

### 5-1. ZodError の構造

```typescript
const schema = z.object({
  name: z.string().min(1),
  email: z.string().email(),
  age: z.number().int().min(0),
});

const result = schema.safeParse({ name: "", email: "invalid", age: -1 });

if (!result.success) {
  // result.error は ZodError インスタンス

  // issues: 個別のエラー配列
  console.log(result.error.issues);
  // [
  //   { code: "too_small", path: ["name"], message: "..." },
  //   { code: "invalid_string", path: ["email"], message: "..." },
  //   { code: "too_small", path: ["age"], message: "..." },
  // ]

  // flatten: フィールドごとにエラーメッセージをまとめる
  console.log(result.error.flatten());
  // {
  //   formErrors: [],
  //   fieldErrors: {
  //     name: ["String must contain at least 1 character(s)"],
  //     email: ["Invalid email"],
  //     age: ["Number must be greater than or equal to 0"],
  //   },
  // }

  // format: ネストされた構造で取得
  console.log(result.error.format());
  // {
  //   _errors: [],
  //   name: { _errors: ["..."] },
  //   email: { _errors: ["..."] },
  //   age: { _errors: ["..."] },
  // }
}
```

### 5-2. カスタムエラーメッセージ

```typescript
// 各バリデーションにメッセージを指定
const schema = z.string({
  required_error: "必須項目です",
  invalid_type_error: "文字列を入力してください",
}).min(1, { message: "1文字以上入力してください" });

// errorMap でグローバルにカスタマイズ
const customErrorMap: z.ZodErrorMap = (issue, ctx) => {
  if (issue.code === z.ZodIssueCode.invalid_type) {
    if (issue.expected === "string") {
      return { message: "文字列を入力してください" };
    }
    if (issue.expected === "number") {
      return { message: "数値を入力してください" };
    }
  }
  if (issue.code === z.ZodIssueCode.too_small) {
    if (issue.type === "string") {
      return { message: `${issue.minimum}文字以上で入力してください` };
    }
  }
  return { message: ctx.defaultError };
};

z.setErrorMap(customErrorMap);

// i18n対応のエラーマップ（zod-i18n-map）
import { zodI18nMap } from "zod-i18n-map";
import translation from "zod-i18n-map/locales/ja/zod.json";
import i18next from "i18next";

i18next.init({
  lng: "ja",
  resources: { ja: { zod: translation } },
});

z.setErrorMap(zodI18nMap);
// → エラーメッセージが自動的に日本語になる
```

---

## 比較表

### バリデーションライブラリ比較

| ライブラリ | サイズ | 型推論 | パフォーマンス | API スタイル | エコシステム |
|-----------|--------|--------|-------------|-------------|------------|
| zod | ~14KB | 最高 | 良好 | メソッドチェーン | 最大 |
| yup | ~15KB | 中 | 良好 | メソッドチェーン | 大 |
| joi | ~30KB | 低(@types) | 良好 | メソッドチェーン | 大(Node) |
| superstruct | ~3KB | 高 | 良好 | 関数合成 | 小 |
| valibot | ~1KB | 高 | 最高 | 関数合成 | 成長中 |
| typia | 0KB(生成) | 最高 | 最高 | デコレータ | 小 |
| arktype | ~5KB | 最高 | 最高 | 文字列DSL | 小 |

### parse vs safeParse

| メソッド | 失敗時 | 戻り値型 | 用途 |
|---------|--------|---------|------|
| `.parse()` | ZodError throw | `T` | 信頼できる内部データ |
| `.safeParse()` | `{ success: false }` | `SafeParseResult<T>` | ユーザー入力、API |
| `.parseAsync()` | ZodError throw | `Promise<T>` | async refine 使用時 |
| `.safeParseAsync()` | `{ success: false }` | `Promise<SafeParseResult<T>>` | async 安全版 |

### Zod メソッドチートシート

| カテゴリ | メソッド | 説明 |
|---------|---------|------|
| 変換 | `.transform()` | バリデーション後に値を変換 |
| 変換 | `.pipe()` | 別のスキーマにパイプ |
| 変換 | `.preprocess()` | バリデーション前に前処理 |
| 変換 | `.coerce` | 暗黙的型変換 |
| バリデーション | `.refine()` | カスタム検証 |
| バリデーション | `.superRefine()` | 高度なカスタム検証 |
| オプション | `.optional()` | `T \| undefined` |
| オプション | `.nullable()` | `T \| null` |
| オプション | `.nullish()` | `T \| null \| undefined` |
| オプション | `.default()` | デフォルト値 |
| オプション | `.catch()` | パース失敗時のフォールバック |
| 型変換 | `.brand()` | ブランド型の付与 |
| 型変換 | `.readonly()` | Readonly化 |
| 取得 | `z.infer<>` | 出力型の取得 |
| 取得 | `z.input<>` | 入力型の取得 |

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

### AP-3: バリデーションを集約しない

```typescript
// NG: 各所でバラバラにバリデーション
function createUser(name: string, email: string) {
  if (!name) throw new Error("Name required");
  if (!email.includes("@")) throw new Error("Invalid email");
  // ...
}

// OK: スキーマでバリデーションを集約
const CreateUserSchema = z.object({
  name: z.string().min(1),
  email: z.string().email(),
});

function createUser(input: unknown) {
  const data = CreateUserSchema.parse(input);
  // data は検証済み
}
```

### AP-4: coerce の濫用

```typescript
// NG: coerce を安易に使い暗黙変換に依存
const schema = z.object({
  count: z.coerce.number(),  // null → 0, undefined → NaN, "abc" → NaN
  active: z.coerce.boolean(), // 0 → false, "" → false, "false" → true!
});

// OK: 明示的に transform で変換
const schema = z.object({
  count: z.string()
    .regex(/^\d+$/, "数値を入力してください")
    .transform(Number),
  active: z.enum(["true", "false"])
    .transform((val) => val === "true"),
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

### Q4: Prisma のスキーマから Zod スキーマを自動生成できますか？

はい。`zod-prisma-types` や `prisma-zod-generator` などのジェネレーターを使えば、Prisma スキーマから Zod スキーマを自動生成できます。

```prisma
// prisma/schema.prisma
generator zod {
  provider = "zod-prisma-types"
}
```

### Q5: z.infer と z.input の違いは何ですか？

`z.infer`（= `z.output`）はスキーマの出力型（transform/default 適用後）、`z.input` は入力型（transform/default 適用前）です。フォームの型定義には `z.input`、API レスポンスの型定義には `z.infer` を使うのが一般的です。

### Q6: エラーメッセージを国際化（i18n）するには？

`zod-i18n-map` ライブラリを使用すると、i18next と連携して自動的にエラーメッセージを翻訳できます。日本語を含む多言語がサポートされています。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| z.infer | スキーマから TypeScript 型を自動推論 |
| safeParse | 例外を投げずに検証結果を返す |
| transform | バリデーション後に値を変換 |
| pipe | 変換後のスキーマで再バリデーション |
| discriminatedUnion | 判別子フィールドで型を分岐 |
| refine / superRefine | カスタムバリデーションロジック |
| brand | ブランド型の付与 |
| coerce | 暗黙的な型変換 |
| preprocess | バリデーション前の前処理 |
| z.lazy | 再帰型スキーマの定義 |

---

## 演習問題

### 問題1: ユーザー登録フォームのスキーマ

以下の要件を満たすユーザー登録フォームのスキーマを定義してください。

- 名前: 必須、1〜50文字
- メール: 必須、有効なメールアドレス形式
- パスワード: 8文字以上、大文字・小文字・数字・特殊文字を含む
- パスワード確認: パスワードと一致
- 年齢: オプション、0〜150の整数
- 利用規約への同意: true でなければならない

### 問題2: APIレスポンスの汎用スキーマ

以下の構造を持つ汎用的なAPIレスポンススキーマを定義してください。

- 成功時: `{ success: true, data: T, meta?: { page, total } }`
- 失敗時: `{ success: false, error: { code, message } }`
- discriminatedUnion を使うこと

### 問題3: 環境変数バリデーション

実際のプロジェクトを想定して、以下の環境変数を検証するスキーマを定義してください。

- NODE_ENV: development / production / test
- PORT: 数値（デフォルト 3000）
- DATABASE_URL: URL形式
- REDIS_URL: オプション、URL形式
- JWT_SECRET: 32文字以上
- ログレベル: debug / info / warn / error（デフォルト info）

### 問題4: ネストされたフォームのバリデーション

住所情報を含むネストされたオブジェクトのスキーマを定義してください。都道府県は47都道府県の enum とし、郵便番号は `xxx-xxxx` 形式を検証すること。

### 問題5: transform を使ったCSVパーサー

CSV文字列を受け取り、バリデーション後にオブジェクトの配列に変換するスキーマを定義してください。各行は `name,email,age` の形式とします。

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

4. **React Hook Form + Zod**
   https://react-hook-form.com/get-started#SchemaValidation

5. **zod-i18n-map**
   https://github.com/aiji42/zod-i18n
