# バリデーションパターン

> バリデーションはフォームの品質を決定づける。Zod統合、リアルタイム検証、非同期バリデーション、クライアント/サーバー二重検証まで、堅牢で使いやすいバリデーション設計の全パターンを習得する。

## この章で学ぶこと

- [ ] Zodスキーマの設計原則と高度なパターンを理解する
- [ ] React Hook Formとの統合パターンを実装できるようになる
- [ ] バリデーションのタイミング戦略を使い分けられるようになる
- [ ] リアルタイム・非同期バリデーションの実装を把握する
- [ ] クライアント/サーバー二重検証の設計を学ぶ
- [ ] パスワード強度インジケーターを実装できるようになる
- [ ] エラーメッセージの国際化とアクセシビリティ対応を理解する
- [ ] バリデーションのテスト戦略を習得する

---

## 前提知識

この章を最大限活用するために、以下の知識を事前に習得しておくことを推奨する:

- **フォーム設計**: `./00-form-design.md` で学ぶ、React Hook Formの基本パターンと制御/非制御コンポーネントの概念を理解していること
- **TypeScriptの型システム**: ジェネリクス、ユニオン型、インターセクション型、型推論といったTypeScriptの中級レベルの型システムを把握していること
- **Zodスキーマの基礎**: Zodの基本的なスキーマ定義（`z.string()`, `z.number()`, `z.object()` など）と `.parse()` / `.safeParse()` の使い方を理解していること

---

## 1. バリデーションの基本概念と設計思想

### 1.1 なぜバリデーションが重要なのか

Webアプリケーションにおけるバリデーションは、単なる入力チェックにとどまらない。データの整合性を保証し、セキュリティを担保し、ユーザーに適切なフィードバックを提供するための根幹的な仕組みである。

バリデーションが不十分な場合に起こりうる問題:

1. **セキュリティリスク**: SQLインジェクション、XSS攻撃、不正なデータの注入
2. **データ不整合**: データベースに不正な値が保存され、後続処理で障害が発生
3. **UXの低下**: ユーザーが何を修正すべきか分からず、フォーム離脱率が上昇
4. **ビジネスロジックの破綻**: 想定外のデータが業務処理に流れ込み、計算ミスや不正な状態遷移が発生

```
バリデーションの守備範囲:

  ┌──────────────────────────────────────┐
  │           クライアントサイド           │
  │  ┌──────────────────────────────┐    │
  │  │  HTML5ネイティブバリデーション  │    │
  │  │  (required, pattern, min等)   │    │
  │  └──────────────────────────────┘    │
  │  ┌──────────────────────────────┐    │
  │  │  JavaScript バリデーション      │    │
  │  │  (Zod, Yup, カスタムロジック)  │    │
  │  └──────────────────────────────┘    │
  │  → 即座のフィードバック、UX向上       │
  └──────────────────────────────────────┘
              ↓ HTTP Request
  ┌──────────────────────────────────────┐
  │           サーバーサイド               │
  │  ┌──────────────────────────────┐    │
  │  │  アプリケーション層バリデーション│    │
  │  │  (Zod, class-validator等)     │    │
  │  └──────────────────────────────┘    │
  │  ┌──────────────────────────────┐    │
  │  │  ビジネスロジック層            │    │
  │  │  (重複チェック、権限チェック等) │    │
  │  └──────────────────────────────┘    │
  │  ┌──────────────────────────────┐    │
  │  │  データベース層制約            │    │
  │  │  (UNIQUE, CHECK, NOT NULL等)  │    │
  │  └──────────────────────────────┘    │
  │  → セキュリティ担保、データ整合性      │
  └──────────────────────────────────────┘
```

### 1.2 バリデーションライブラリの比較

現在のTypeScript/JavaScript エコシステムで利用される主要なバリデーションライブラリを比較する。

| ライブラリ | 型推論 | バンドルサイズ | パフォーマンス | エコシステム | 学習コスト |
|-----------|--------|-------------|--------------|------------|-----------|
| **Zod** | 優秀 | 13KB (gzip) | 良好 | React Hook Form, tRPC, Next.js | 低 |
| **Yup** | 良好 | 12KB (gzip) | 良好 | Formik, React Hook Form | 低 |
| **Valibot** | 優秀 | 1KB〜 (tree-shake) | 非常に良好 | React Hook Form | 中 |
| **Joi** | 限定的 | 大きい | 良好 | Express/Hapi | 中 |
| **class-validator** | デコレータ | 中程度 | 良好 | NestJS | 中 |
| **ArkType** | 非常に優秀 | 小さい | 非常に良好 | 限定的 | 高 |
| **TypeBox** | 優秀 | 小さい | 非常に良好 | Fastify | 中 |

```typescript
// 各ライブラリでの同等スキーマ定義の比較

// === Zod ===
import { z } from 'zod';
const zodSchema = z.object({
  name: z.string().min(1).max(100),
  age: z.number().int().min(0).max(150),
  email: z.string().email(),
});
type ZodUser = z.infer<typeof zodSchema>;

// === Yup ===
import * as yup from 'yup';
const yupSchema = yup.object({
  name: yup.string().required().min(1).max(100),
  age: yup.number().integer().min(0).max(150).required(),
  email: yup.string().email().required(),
});
type YupUser = yup.InferType<typeof yupSchema>;

// === Valibot ===
import * as v from 'valibot';
const valibotSchema = v.object({
  name: v.pipe(v.string(), v.minLength(1), v.maxLength(100)),
  age: v.pipe(v.number(), v.integer(), v.minValue(0), v.maxValue(150)),
  email: v.pipe(v.string(), v.email()),
});
type ValibotUser = v.InferOutput<typeof valibotSchema>;

// === ArkType ===
import { type } from 'arktype';
const arkSchema = type({
  name: 'string>=1<=100',
  age: 'integer>=0<=150',
  email: 'string.email',
});
type ArkUser = typeof arkSchema.infer;
```

### 1.3 なぜZodを選ぶのか

本ガイドでは主にZodを使った実装パターンを解説する。Zodを選択する理由は以下の通りである:

1. **TypeScriptファーストの設計**: スキーマ定義から自動的に型が推論される
2. **豊富なエコシステム**: React Hook Form, tRPC, Next.js Server Actionsとの公式連携
3. **直感的なAPI**: メソッドチェーンによる宣言的なスキーマ定義
4. **ゼロ依存**: 外部ライブラリに依存しない
5. **コミュニティの活発さ**: 大規模なユーザーベースと豊富なドキュメント

```typescript
// Zodの最大の利点: スキーマから型を自動生成
const userSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1),
  email: z.string().email(),
  age: z.number().int().positive(),
  role: z.enum(['admin', 'user', 'moderator']),
  createdAt: z.date(),
});

// スキーマから型を自動推論 - 手動でinterface定義する必要がない
type User = z.infer<typeof userSchema>;
// => {
//   id: string;
//   name: string;
//   email: string;
//   age: number;
//   role: 'admin' | 'user' | 'moderator';
//   createdAt: Date;
// }

// 部分型も簡単に生成
type UserUpdate = z.infer<typeof userSchema.partial()>;
type UserCreate = z.infer<typeof userSchema.omit({ id: true, createdAt: true })>;
```

---

## 2. Zodスキーマ設計

### 2.1 基本的なスキーマパターン

#### プリミティブ型のバリデーション

```typescript
import { z } from 'zod';

// 文字列のバリデーション
const stringSchemas = {
  // 基本
  required: z.string().min(1, '必須項目です'),

  // 長さ制約
  username: z.string()
    .min(3, '3文字以上で入力してください')
    .max(20, '20文字以下で入力してください'),

  // 正規表現パターン
  alphanumeric: z.string()
    .regex(/^[a-zA-Z0-9]+$/, '英数字のみ使用可能です'),

  // 組み込みバリデーション
  email: z.string().email('有効なメールアドレスを入力してください'),
  url: z.string().url('有効なURLを入力してください'),
  uuid: z.string().uuid('有効なUUID形式で入力してください'),
  cuid: z.string().cuid(),
  datetime: z.string().datetime(),
  ip: z.string().ip(),

  // トリム + 変換
  trimmed: z.string().trim().min(1, '空白のみは不可です'),
  lowercase: z.string().toLowerCase(),
  uppercase: z.string().toUpperCase(),

  // 日本語対応: 全角文字のバリデーション
  japaneseName: z.string()
    .min(1, '名前を入力してください')
    .max(50, '50文字以下で入力してください')
    .regex(/^[ぁ-んァ-ヶー一-龠々\s]+$/, '日本語で入力してください'),

  // 電話番号（日本）
  phoneJP: z.string()
    .regex(/^0\d{1,4}-?\d{1,4}-?\d{3,4}$/, '有効な電話番号を入力してください'),

  // 郵便番号（日本）
  postalCodeJP: z.string()
    .regex(/^\d{3}-?\d{4}$/, '有効な郵便番号を入力してください'),
};

// 数値のバリデーション
const numberSchemas = {
  // 基本
  positive: z.number().positive('正の数を入力してください'),
  nonNegative: z.number().nonnegative('0以上の数を入力してください'),
  integer: z.number().int('整数を入力してください'),

  // 範囲
  age: z.number()
    .int('整数を入力してください')
    .min(0, '0以上で入力してください')
    .max(150, '150以下で入力してください'),

  // 小数桁数
  price: z.number()
    .nonnegative('0以上で入力してください')
    .multipleOf(0.01, '小数点以下2桁まで入力可能です'),

  // 文字列からの変換（フォーム入力値対応）
  fromString: z.coerce.number()
    .int('整数を入力してください')
    .min(1, '1以上で入力してください'),
};

// 日付のバリデーション
const dateSchemas = {
  // 基本
  date: z.coerce.date(),

  // 範囲
  pastDate: z.coerce.date()
    .max(new Date(), '未来の日付は指定できません'),

  futureDate: z.coerce.date()
    .min(new Date(), '過去の日付は指定できません'),

  // カスタム範囲
  dateRange: z.coerce.date()
    .min(new Date('2020-01-01'), '2020年1月1日以降の日付を指定してください')
    .max(new Date('2030-12-31'), '2030年12月31日以前の日付を指定してください'),
};

// 真偽値のバリデーション
const booleanSchemas = {
  // 利用規約同意（trueのみ許可）
  agreeToTerms: z.literal(true, {
    errorMap: () => ({ message: '利用規約に同意してください' }),
  }),

  // チェックボックス（フォーム値の変換）
  checkbox: z.coerce.boolean(),
};

// 列挙型のバリデーション
const enumSchemas = {
  // Zodのenum
  role: z.enum(['admin', 'user', 'moderator'], {
    errorMap: () => ({ message: '有効な権限を選択してください' }),
  }),

  // TypeScriptのenumとの連携
  // enum Status { Active = 'active', Inactive = 'inactive' }
  status: z.nativeEnum({ Active: 'active', Inactive: 'inactive' } as const),
};
```

#### ユーザー登録フォームの完全なスキーマ

```typescript
import { z } from 'zod';

// ユーザー登録フォームスキーマ
const registerSchema = z.object({
  // ユーザー名
  username: z.string()
    .min(3, '3文字以上で入力してください')
    .max(20, '20文字以下で入力してください')
    .regex(/^[a-zA-Z0-9_]+$/, '英数字とアンダースコアのみ使用可能')
    .refine(
      (val) => !['admin', 'root', 'system', 'null', 'undefined'].includes(val.toLowerCase()),
      '予約語は使用できません'
    ),

  // メールアドレス
  email: z.string()
    .email('有効なメールアドレスを入力してください')
    .max(254, 'メールアドレスが長すぎます'),

  // パスワード
  password: z.string()
    .min(8, '8文字以上で入力してください')
    .max(100, '100文字以下で入力してください')
    .regex(/[A-Z]/, '大文字を1文字以上含めてください')
    .regex(/[a-z]/, '小文字を1文字以上含めてください')
    .regex(/[0-9]/, '数字を1文字以上含めてください')
    .regex(/[^A-Za-z0-9]/, '記号を1文字以上含めてください'),

  // パスワード確認
  confirmPassword: z.string(),

  // 生年月日
  birthDate: z.coerce.date()
    .max(new Date(), '未来の日付は指定できません')
    .refine(
      (date) => {
        const age = Math.floor(
          (Date.now() - date.getTime()) / (365.25 * 24 * 60 * 60 * 1000)
        );
        return age >= 13;
      },
      '13歳以上である必要があります'
    ),

  // Webサイト（オプショナル）
  website: z.string()
    .url('有効なURLを入力してください')
    .optional()
    .or(z.literal('')),

  // 自己紹介（オプショナル）
  bio: z.string()
    .max(500, '500文字以下で入力してください')
    .optional()
    .or(z.literal('')),

  // 利用規約同意
  agreeToTerms: z.literal(true, {
    errorMap: () => ({ message: '利用規約に同意してください' }),
  }),

  // プライバシーポリシー同意
  agreeToPrivacy: z.literal(true, {
    errorMap: () => ({ message: 'プライバシーポリシーに同意してください' }),
  }),

}).refine(
  (data) => data.password === data.confirmPassword,
  {
    message: 'パスワードが一致しません',
    path: ['confirmPassword'],
  }
).refine(
  (data) => {
    // パスワードにユーザー名が含まれていないかチェック
    if (data.username && data.password) {
      return !data.password.toLowerCase().includes(data.username.toLowerCase());
    }
    return true;
  },
  {
    message: 'パスワードにユーザー名を含めることはできません',
    path: ['password'],
  }
);

// 型の自動推論
type RegisterFormData = z.infer<typeof registerSchema>;
```

### 2.2 高度なスキーマパターン

#### 条件付きバリデーション（Discriminated Union）

```typescript
// 支払い方法に応じて異なるフィールドを要求するパターン
const creditCardSchema = z.object({
  paymentMethod: z.literal('credit_card'),
  cardNumber: z.string()
    .regex(/^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$/, 'カード番号の形式が不正です')
    .transform((val) => val.replace(/[\s-]/g, '')),
  expiryMonth: z.number().int().min(1).max(12),
  expiryYear: z.number().int().min(new Date().getFullYear()),
  cvv: z.string().regex(/^\d{3,4}$/, 'CVVは3〜4桁の数字です'),
  cardholderName: z.string().min(1, 'カード名義人を入力してください'),
});

const bankTransferSchema = z.object({
  paymentMethod: z.literal('bank_transfer'),
  bankName: z.string().min(1, '銀行名を入力してください'),
  branchName: z.string().min(1, '支店名を入力してください'),
  accountType: z.enum(['普通', '当座']),
  accountNumber: z.string()
    .regex(/^\d{7}$/, '口座番号は7桁の数字です'),
  accountHolder: z.string().min(1, '口座名義人を入力してください'),
});

const digitalWalletSchema = z.object({
  paymentMethod: z.literal('digital_wallet'),
  walletType: z.enum(['paypay', 'linepay', 'merpay']),
  walletId: z.string().min(1, 'ウォレットIDを入力してください'),
});

// Discriminated Unionで統合
const paymentSchema = z.discriminatedUnion('paymentMethod', [
  creditCardSchema,
  bankTransferSchema,
  digitalWalletSchema,
]);

type PaymentData = z.infer<typeof paymentSchema>;

// 使用例: フォームでの切り替え
function PaymentForm() {
  const [paymentMethod, setPaymentMethod] = useState<
    'credit_card' | 'bank_transfer' | 'digital_wallet'
  >('credit_card');

  const form = useForm<PaymentData>({
    resolver: zodResolver(paymentSchema),
    defaultValues: {
      paymentMethod: 'credit_card',
    },
  });

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <select
        value={paymentMethod}
        onChange={(e) => {
          const method = e.target.value as PaymentData['paymentMethod'];
          setPaymentMethod(method);
          form.setValue('paymentMethod', method);
          // 支払い方法が変更されたらフォームをリセット
          form.clearErrors();
        }}
      >
        <option value="credit_card">クレジットカード</option>
        <option value="bank_transfer">銀行振込</option>
        <option value="digital_wallet">電子マネー</option>
      </select>

      {paymentMethod === 'credit_card' && <CreditCardFields form={form} />}
      {paymentMethod === 'bank_transfer' && <BankTransferFields form={form} />}
      {paymentMethod === 'digital_wallet' && <DigitalWalletFields form={form} />}
    </form>
  );
}
```

#### superRefineによる複雑なバリデーション

```typescript
// superRefineを使った複雑なクロスフィールドバリデーション
const eventSchema = z.object({
  title: z.string().min(1, 'タイトルを入力してください'),
  startDate: z.coerce.date(),
  endDate: z.coerce.date(),
  startTime: z.string().regex(/^\d{2}:\d{2}$/, '時刻の形式が不正です'),
  endTime: z.string().regex(/^\d{2}:\d{2}$/, '時刻の形式が不正です'),
  isAllDay: z.boolean(),
  maxParticipants: z.number().int().positive().optional(),
  currentParticipants: z.number().int().nonnegative().default(0),
  isOnline: z.boolean(),
  venue: z.string().optional(),
  meetingUrl: z.string().url().optional(),
}).superRefine((data, ctx) => {
  // 開始日と終了日の整合性チェック
  if (data.endDate < data.startDate) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: '終了日は開始日以降にしてください',
      path: ['endDate'],
    });
  }

  // 同日の場合、終了時刻が開始時刻より後であることを確認
  if (
    data.startDate.toDateString() === data.endDate.toDateString() &&
    !data.isAllDay
  ) {
    if (data.endTime <= data.startTime) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: '終了時刻は開始時刻より後にしてください',
        path: ['endTime'],
      });
    }
  }

  // 参加人数の整合性
  if (data.maxParticipants !== undefined && data.currentParticipants > data.maxParticipants) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: '現在の参加者数が定員を超えています',
      path: ['maxParticipants'],
    });
  }

  // オンライン/オフラインに応じた必須フィールド
  if (data.isOnline && !data.meetingUrl) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'オンラインイベントの場合はミーティングURLが必要です',
      path: ['meetingUrl'],
    });
  }

  if (!data.isOnline && !data.venue) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'オフラインイベントの場合は会場を入力してください',
      path: ['venue'],
    });
  }
});
```

#### スキーマの再利用と合成

```typescript
// 基本スキーマの定義
const baseUserSchema = z.object({
  name: z.string().min(1, '名前を入力してください').max(100),
  email: z.string().email('有効なメールアドレスを入力してください'),
});

const addressSchema = z.object({
  postalCode: z.string().regex(/^\d{3}-?\d{4}$/, '有効な郵便番号を入力してください'),
  prefecture: z.string().min(1, '都道府県を選択してください'),
  city: z.string().min(1, '市区町村を入力してください'),
  street: z.string().min(1, '番地を入力してください'),
  building: z.string().optional(),
});

const phoneSchema = z.object({
  phoneNumber: z.string()
    .regex(/^0\d{1,4}-?\d{1,4}-?\d{3,4}$/, '有効な電話番号を入力してください'),
  phoneType: z.enum(['mobile', 'home', 'work']),
});

// extend: フィールドを追加
const createUserSchema = baseUserSchema.extend({
  password: z.string().min(8, '8文字以上で入力してください'),
  role: z.enum(['admin', 'user']).default('user'),
});

// merge: 複数のスキーマを統合
const fullUserSchema = baseUserSchema
  .merge(addressSchema)
  .merge(phoneSchema);

// pick: 特定のフィールドだけ取得
const loginSchema = baseUserSchema.pick({
  email: true,
}).extend({
  password: z.string().min(1, 'パスワードを入力してください'),
});

// omit: 特定のフィールドを除外
const publicUserSchema = fullUserSchema.omit({
  phoneNumber: true,
  phoneType: true,
});

// partial: 全フィールドをオプショナルに（更新APIに最適）
const updateUserSchema = baseUserSchema.partial();

// deepPartial: ネストされたオブジェクトも含めて全てオプショナルに
const deepUpdateSchema = fullUserSchema.deepPartial();

// required: オプショナルフィールドを必須に
const strictSchema = updateUserSchema.required();

// passthrough: 未知のプロパティを保持
const flexibleSchema = baseUserSchema.passthrough();

// strict: 未知のプロパティでエラー
const strictUserSchema = baseUserSchema.strict();

// 配列スキーマ
const usersSchema = z.array(baseUserSchema)
  .min(1, '最低1人のユーザーが必要です')
  .max(100, 'ユーザー数は100人までです');

// レコードスキーマ
const settingsSchema = z.record(
  z.string(),
  z.union([z.string(), z.number(), z.boolean()])
);
```

### 2.3 transformとpreprocessによるデータ変換

```typescript
// transform: バリデーション後にデータを変換
const formSchema = z.object({
  // 文字列をトリムして正規化
  name: z.string()
    .transform((val) => val.trim())
    .pipe(z.string().min(1, '名前を入力してください')),

  // カンマ区切りの文字列を配列に変換
  tags: z.string()
    .transform((val) => val.split(',').map((s) => s.trim()).filter(Boolean))
    .pipe(z.array(z.string()).min(1, 'タグを1つ以上入力してください')),

  // 電話番号のハイフンを除去
  phone: z.string()
    .transform((val) => val.replace(/-/g, ''))
    .pipe(z.string().regex(/^0\d{9,10}$/, '有効な電話番号を入力してください')),

  // 郵便番号の正規化
  postalCode: z.string()
    .transform((val) => {
      const digits = val.replace(/[^\d]/g, '');
      return digits.length === 7 ? `${digits.slice(0, 3)}-${digits.slice(3)}` : val;
    })
    .pipe(z.string().regex(/^\d{3}-\d{4}$/, '有効な郵便番号を入力してください')),

  // 金額の変換（カンマ除去 → 数値化）
  amount: z.string()
    .transform((val) => Number(val.replace(/,/g, '')))
    .pipe(z.number().positive('正の金額を入力してください')),
});

// preprocess: バリデーション前にデータを前処理
const preprocessedSchema = z.object({
  // 空文字列をundefinedに変換（オプショナルフィールド対応）
  website: z.preprocess(
    (val) => (val === '' ? undefined : val),
    z.string().url('有効なURLを入力してください').optional()
  ),

  // チェックボックスの値を真偽値に変換
  isActive: z.preprocess(
    (val) => val === 'on' || val === 'true' || val === true,
    z.boolean()
  ),

  // 文字列の数値を数値型に変換
  quantity: z.preprocess(
    (val) => (typeof val === 'string' ? Number(val) : val),
    z.number().int().positive()
  ),
});

// coerceの活用（FormDataからの自動変換）
const formDataSchema = z.object({
  name: z.string().min(1),
  age: z.coerce.number().int().min(0),     // string → number
  isAdmin: z.coerce.boolean(),              // string → boolean
  createdAt: z.coerce.date(),               // string → Date
  score: z.coerce.bigint(),                 // string → Bigint
});
```

### 2.4 カスタムエラーメッセージとエラーマップ

```typescript
// フィールドレベルのカスタムエラーメッセージ
const detailedSchema = z.object({
  username: z.string({
    required_error: 'ユーザー名は必須です',
    invalid_type_error: 'ユーザー名は文字列で入力してください',
  })
    .min(3, { message: '3文字以上で入力してください' })
    .max(20, { message: '20文字以下で入力してください' }),

  age: z.number({
    required_error: '年齢は必須です',
    invalid_type_error: '年齢は数値で入力してください',
  })
    .int({ message: '整数で入力してください' })
    .min(0, { message: '0以上で入力してください' }),
});

// グローバルエラーマップ
const customErrorMap: z.ZodErrorMap = (issue, ctx) => {
  // デフォルトのエラーメッセージをカスタマイズ
  switch (issue.code) {
    case z.ZodIssueCode.invalid_type:
      if (issue.expected === 'string') {
        return { message: 'テキストを入力してください' };
      }
      if (issue.expected === 'number') {
        return { message: '数値を入力してください' };
      }
      break;
    case z.ZodIssueCode.too_small:
      if (issue.type === 'string') {
        return { message: `${issue.minimum}文字以上で入力してください` };
      }
      if (issue.type === 'number') {
        return { message: `${issue.minimum}以上の値を入力してください` };
      }
      break;
    case z.ZodIssueCode.too_big:
      if (issue.type === 'string') {
        return { message: `${issue.maximum}文字以下で入力してください` };
      }
      break;
    case z.ZodIssueCode.invalid_string:
      if (issue.validation === 'email') {
        return { message: '有効なメールアドレスを入力してください' };
      }
      if (issue.validation === 'url') {
        return { message: '有効なURLを入力してください' };
      }
      break;
  }
  return { message: ctx.defaultError };
};

// グローバルに設定
z.setErrorMap(customErrorMap);

// 特定のスキーマにのみ適用
const schemaWithCustomErrors = z.string().min(1).describe('ユーザー名');
```

---

## 3. React Hook Formとの統合

### 3.1 基本セットアップ

```typescript
import { useForm, FormProvider, useFormContext } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

// スキーマ定義
const contactSchema = z.object({
  firstName: z.string().min(1, '名を入力してください'),
  lastName: z.string().min(1, '姓を入力してください'),
  email: z.string().email('有効なメールアドレスを入力してください'),
  phone: z.string()
    .regex(/^0\d{1,4}-?\d{1,4}-?\d{3,4}$/, '有効な電話番号を入力してください')
    .optional()
    .or(z.literal('')),
  subject: z.enum(['inquiry', 'support', 'feedback', 'other'], {
    errorMap: () => ({ message: 'お問い合わせ種別を選択してください' }),
  }),
  message: z.string()
    .min(10, '10文字以上で入力してください')
    .max(2000, '2000文字以下で入力してください'),
  agreeToTerms: z.literal(true, {
    errorMap: () => ({ message: '利用規約に同意してください' }),
  }),
});

type ContactFormData = z.infer<typeof contactSchema>;

// フォームコンポーネント
function ContactForm() {
  const form = useForm<ContactFormData>({
    resolver: zodResolver(contactSchema),
    mode: 'onSubmit',
    reValidateMode: 'onChange',
    defaultValues: {
      firstName: '',
      lastName: '',
      email: '',
      phone: '',
      subject: undefined,
      message: '',
      agreeToTerms: false as unknown as true,
    },
  });

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting, isValid, isDirty, touchedFields },
    watch,
    reset,
    setError,
    clearErrors,
  } = form;

  const onSubmit = async (data: ContactFormData) => {
    try {
      const response = await fetch('/api/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errorData = await response.json();
        // サーバーエラーをフォームにマッピング
        if (errorData.fieldErrors) {
          Object.entries(errorData.fieldErrors).forEach(([field, messages]) => {
            setError(field as keyof ContactFormData, {
              type: 'server',
              message: (messages as string[])[0],
            });
          });
          return;
        }
        throw new Error(errorData.message || '送信に失敗しました');
      }

      // 成功時の処理
      reset();
      alert('お問い合わせを送信しました');
    } catch (error) {
      setError('root', {
        type: 'server',
        message: error instanceof Error ? error.message : '送信に失敗しました',
      });
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} noValidate>
      {/* ルートエラー表示 */}
      {errors.root && (
        <div role="alert" className="bg-red-50 border border-red-200 p-4 rounded">
          <p className="text-red-700">{errors.root.message}</p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        <FormField
          label="姓"
          error={errors.lastName?.message}
          required
        >
          <input
            {...register('lastName')}
            aria-invalid={!!errors.lastName}
            aria-describedby={errors.lastName ? 'lastName-error' : undefined}
            className={errors.lastName ? 'border-red-500' : 'border-gray-300'}
          />
        </FormField>

        <FormField
          label="名"
          error={errors.firstName?.message}
          required
        >
          <input
            {...register('firstName')}
            aria-invalid={!!errors.firstName}
            className={errors.firstName ? 'border-red-500' : 'border-gray-300'}
          />
        </FormField>
      </div>

      <FormField label="メールアドレス" error={errors.email?.message} required>
        <input
          type="email"
          {...register('email')}
          aria-invalid={!!errors.email}
        />
      </FormField>

      <FormField label="電話番号" error={errors.phone?.message}>
        <input
          type="tel"
          {...register('phone')}
          aria-invalid={!!errors.phone}
        />
      </FormField>

      <FormField label="お問い合わせ種別" error={errors.subject?.message} required>
        <select {...register('subject')}>
          <option value="">選択してください</option>
          <option value="inquiry">お問い合わせ</option>
          <option value="support">サポート</option>
          <option value="feedback">フィードバック</option>
          <option value="other">その他</option>
        </select>
      </FormField>

      <FormField label="メッセージ" error={errors.message?.message} required>
        <textarea
          {...register('message')}
          rows={5}
          aria-invalid={!!errors.message}
        />
        <p className="text-sm text-gray-500">
          {watch('message')?.length || 0}/2000文字
        </p>
      </FormField>

      <FormField error={errors.agreeToTerms?.message}>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            {...register('agreeToTerms')}
          />
          <span>利用規約に同意する</span>
        </label>
      </FormField>

      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full bg-blue-600 text-white py-2 px-4 rounded disabled:opacity-50"
      >
        {isSubmitting ? '送信中...' : '送信する'}
      </button>
    </form>
  );
}

// 再利用可能なフォームフィールドコンポーネント
function FormField({
  label,
  error,
  required,
  children,
}: {
  label?: string;
  error?: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="mb-4">
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-1">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      {children}
      {error && (
        <p className="text-red-500 text-sm mt-1" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
```

### 3.2 FormProviderを使ったネストフォーム

```typescript
// 大規模フォームをコンポーネントに分割するパターン
const orderSchema = z.object({
  // 個人情報
  personal: z.object({
    name: z.string().min(1, '名前を入力してください'),
    email: z.string().email('有効なメールアドレスを入力してください'),
    phone: z.string().regex(/^0\d{1,4}-?\d{1,4}-?\d{3,4}$/),
  }),
  // 配送先住所
  shipping: z.object({
    postalCode: z.string().regex(/^\d{3}-?\d{4}$/),
    prefecture: z.string().min(1),
    city: z.string().min(1),
    street: z.string().min(1),
    building: z.string().optional(),
  }),
  // 注文内容
  items: z.array(z.object({
    productId: z.string(),
    quantity: z.number().int().min(1).max(99),
  })).min(1, '商品を1つ以上選択してください'),
  // 備考
  notes: z.string().max(500).optional(),
});

type OrderFormData = z.infer<typeof orderSchema>;

// 親コンポーネント
function OrderForm() {
  const methods = useForm<OrderFormData>({
    resolver: zodResolver(orderSchema),
    defaultValues: {
      personal: { name: '', email: '', phone: '' },
      shipping: { postalCode: '', prefecture: '', city: '', street: '' },
      items: [],
      notes: '',
    },
  });

  return (
    <FormProvider {...methods}>
      <form onSubmit={methods.handleSubmit(onSubmit)}>
        <PersonalInfoSection />
        <ShippingAddressSection />
        <OrderItemsSection />
        <NotesSection />
        <SubmitButton />
      </form>
    </FormProvider>
  );
}

// 子コンポーネント（useFormContextで親のフォーム状態にアクセス）
function PersonalInfoSection() {
  const {
    register,
    formState: { errors },
  } = useFormContext<OrderFormData>();

  return (
    <fieldset>
      <legend className="text-lg font-bold">お客様情報</legend>
      <input
        {...register('personal.name')}
        placeholder="お名前"
      />
      {errors.personal?.name && (
        <span className="text-red-500">{errors.personal.name.message}</span>
      )}
      <input
        type="email"
        {...register('personal.email')}
        placeholder="メールアドレス"
      />
      {errors.personal?.email && (
        <span className="text-red-500">{errors.personal.email.message}</span>
      )}
      <input
        type="tel"
        {...register('personal.phone')}
        placeholder="電話番号"
      />
      {errors.personal?.phone && (
        <span className="text-red-500">{errors.personal.phone.message}</span>
      )}
    </fieldset>
  );
}

// 配送先住所セクション
function ShippingAddressSection() {
  const {
    register,
    setValue,
    formState: { errors },
  } = useFormContext<OrderFormData>();

  // 郵便番号から住所を自動入力
  const handlePostalCodeChange = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const postalCode = e.target.value.replace(/-/g, '');
    if (postalCode.length === 7) {
      try {
        const res = await fetch(
          `https://zipcloud.ibsnet.co.jp/api/search?zipcode=${postalCode}`
        );
        const data = await res.json();
        if (data.results?.[0]) {
          const result = data.results[0];
          setValue('shipping.prefecture', result.address1);
          setValue('shipping.city', result.address2 + result.address3);
        }
      } catch {
        // 住所検索APIのエラーは無視（ユーザーが手動入力可能）
      }
    }
  };

  return (
    <fieldset>
      <legend className="text-lg font-bold">配送先</legend>
      <input
        {...register('shipping.postalCode')}
        onChange={(e) => {
          register('shipping.postalCode').onChange(e);
          handlePostalCodeChange(e);
        }}
        placeholder="郵便番号（例: 100-0001）"
      />
      {errors.shipping?.postalCode && (
        <span className="text-red-500">{errors.shipping.postalCode.message}</span>
      )}
      {/* 他のフィールドも同様 */}
    </fieldset>
  );
}
```

### 3.3 useFieldArrayを使った動的フォーム

```typescript
import { useForm, useFieldArray } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';

// 複数の連絡先を管理するスキーマ
const contactListSchema = z.object({
  contacts: z.array(
    z.object({
      name: z.string().min(1, '名前を入力してください'),
      email: z.string().email('有効なメールアドレスを入力してください'),
      relationship: z.enum(['family', 'friend', 'colleague', 'other']),
      isPrimary: z.boolean().default(false),
    })
  )
    .min(1, '連絡先を1件以上追加してください')
    .max(10, '連絡先は10件までです')
    .refine(
      (contacts) => contacts.filter((c) => c.isPrimary).length <= 1,
      '主要連絡先は1件のみ設定可能です'
    ),
});

type ContactListData = z.infer<typeof contactListSchema>;

function ContactListForm() {
  const form = useForm<ContactListData>({
    resolver: zodResolver(contactListSchema),
    defaultValues: {
      contacts: [{ name: '', email: '', relationship: 'friend', isPrimary: false }],
    },
  });

  const { fields, append, remove, move, swap, insert } = useFieldArray({
    control: form.control,
    name: 'contacts',
  });

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      {fields.map((field, index) => (
        <div key={field.id} className="border p-4 mb-4 rounded">
          <div className="flex justify-between items-center mb-2">
            <h3>連絡先 #{index + 1}</h3>
            <div className="flex gap-2">
              {index > 0 && (
                <button type="button" onClick={() => move(index, index - 1)}>
                  上に移動
                </button>
              )}
              {index < fields.length - 1 && (
                <button type="button" onClick={() => move(index, index + 1)}>
                  下に移動
                </button>
              )}
              {fields.length > 1 && (
                <button
                  type="button"
                  onClick={() => remove(index)}
                  className="text-red-500"
                >
                  削除
                </button>
              )}
            </div>
          </div>

          <input
            {...form.register(`contacts.${index}.name`)}
            placeholder="名前"
          />
          {form.formState.errors.contacts?.[index]?.name && (
            <span className="text-red-500">
              {form.formState.errors.contacts[index]?.name?.message}
            </span>
          )}

          <input
            type="email"
            {...form.register(`contacts.${index}.email`)}
            placeholder="メールアドレス"
          />
          {form.formState.errors.contacts?.[index]?.email && (
            <span className="text-red-500">
              {form.formState.errors.contacts[index]?.email?.message}
            </span>
          )}

          <select {...form.register(`contacts.${index}.relationship`)}>
            <option value="family">家族</option>
            <option value="friend">友人</option>
            <option value="colleague">同僚</option>
            <option value="other">その他</option>
          </select>

          <label className="flex items-center gap-2 mt-2">
            <input
              type="checkbox"
              {...form.register(`contacts.${index}.isPrimary`)}
            />
            主要連絡先に設定
          </label>
        </div>
      ))}

      {/* 配列レベルのエラー */}
      {form.formState.errors.contacts?.root && (
        <p className="text-red-500">{form.formState.errors.contacts.root.message}</p>
      )}

      <button
        type="button"
        onClick={() =>
          append({ name: '', email: '', relationship: 'friend', isPrimary: false })
        }
        disabled={fields.length >= 10}
        className="mb-4"
      >
        連絡先を追加
      </button>

      <button type="submit">保存</button>
    </form>
  );
}
```

---

## 4. バリデーションのタイミング戦略

### 4.1 バリデーションモードの詳細比較

```
バリデーション戦略の比較:

  ① mode: 'onSubmit'（デフォルト）
     ┌──────────────────────────────────────────────┐
     │ 入力中   →  何も表示しない                      │
     │ Blur時   →  何も表示しない                      │
     │ Submit時 →  全フィールドをバリデーション          │
     │ 修正中   →  reValidateModeに依存                │
     └──────────────────────────────────────────────┘
     → メリット: 入力中にエラーが表示されない
     → デメリット: Submit後まで問題に気付かない
     → 適用場面: シンプルなフォーム、短いフォーム

  ② mode: 'onSubmit' + reValidateMode: 'onChange'（推奨）
     ┌──────────────────────────────────────────────┐
     │ 初回入力 →  何も表示しない                      │
     │ Submit時 →  全フィールドをバリデーション          │
     │ エラー後 →  入力するたびにリアルタイムで再検証    │
     └──────────────────────────────────────────────┘
     → メリット: 初回入力は邪魔せず、エラー発生後は素早くフィードバック
     → デメリット: 初回Submitまでエラーが表示されない
     → 適用場面: ほとんどのフォーム（最もバランスが良い）

  ③ mode: 'onBlur'
     ┌──────────────────────────────────────────────┐
     │ 入力中   →  何も表示しない                      │
     │ Blur時   →  そのフィールドをバリデーション       │
     │ Submit時 →  全フィールドをバリデーション          │
     └──────────────────────────────────────────────┘
     → メリット: フィールドを離れた時に即座にフィードバック
     → デメリット: 入力中はフィードバックがない
     → 適用場面: 中〜大規模なフォーム

  ④ mode: 'onChange'
     ┌──────────────────────────────────────────────┐
     │ 入力中   →  即座にバリデーション                 │
     │ Blur時   →  即座にバリデーション                 │
     │ Submit時 →  全フィールドをバリデーション          │
     └──────────────────────────────────────────────┘
     → メリット: 最も即座のフィードバック
     → デメリット: パフォーマンスへの影響、入力初期にエラーが出すぎる
     → 適用場面: パスワード強度表示、リアルタイム検索

  ⑤ mode: 'onTouched'
     ┌──────────────────────────────────────────────┐
     │ 初回Touch前 → 何も表示しない                    │
     │ 初回Blur後  → onChange + onBlurで検証           │
     │ Submit時    → 全フィールドをバリデーション        │
     └──────────────────────────────────────────────┘
     → メリット: 初回タッチ後はリアルタイムフィードバック
     → デメリット: onBlurに似ているが微妙に挙動が異なる
     → 適用場面: onBlurよりやや積極的にフィードバックしたい場合

  ⑥ mode: 'all'
     ┌──────────────────────────────────────────────┐
     │ onChange + onBlur の両方でバリデーション          │
     └──────────────────────────────────────────────┘
     → メリット: 最も積極的なバリデーション
     → デメリット: パフォーマンスへの影響が最も大きい
     → 適用場面: 特殊な要件がある場合のみ
```

### 4.2 推奨設定パターン

```typescript
// パターン1: 標準的なフォーム（最も推奨）
const standardForm = useForm<FormData>({
  resolver: zodResolver(schema),
  mode: 'onSubmit',           // 初回はSubmit時
  reValidateMode: 'onChange', // 再検証はリアルタイム
  defaultValues: {
    // 全フィールドにデフォルト値を設定
  },
});

// パターン2: 段階的フィードバック（中〜大規模フォーム）
const progressiveForm = useForm<FormData>({
  resolver: zodResolver(schema),
  mode: 'onBlur',             // フォーカスアウト時にチェック
  reValidateMode: 'onChange', // 修正中はリアルタイムチェック
  defaultValues: {},
});

// パターン3: リアルタイム検索/フィルター
const realtimeForm = useForm<FilterData>({
  resolver: zodResolver(filterSchema),
  mode: 'onChange',           // 即座にチェック
  defaultValues: {},
});

// パターン4: ウィザード形式の場合（ステップごとに検証）
const wizardForm = useForm<WizardData>({
  resolver: zodResolver(currentStepSchema),
  mode: 'onSubmit',           // 「次へ」ボタン押下時に検証
  reValidateMode: 'onChange',
  defaultValues: {},
});
```

### 4.3 フィールド単位のバリデーション制御

```typescript
// triggerを使った手動バリデーション
function StepForm() {
  const form = useForm<FormData>({
    resolver: zodResolver(schema),
    mode: 'onSubmit',
  });

  const [step, setStep] = useState(1);

  // 特定のフィールドだけバリデーション
  const handleNextStep = async () => {
    let isValid = false;

    switch (step) {
      case 1:
        isValid = await form.trigger(['name', 'email']); // Step1のフィールドのみ
        break;
      case 2:
        isValid = await form.trigger(['address', 'phone']); // Step2のフィールドのみ
        break;
      case 3:
        isValid = await form.trigger(); // 全フィールド
        break;
    }

    if (isValid) {
      if (step < 3) {
        setStep(step + 1);
      } else {
        form.handleSubmit(onSubmit)();
      }
    }
  };

  return (
    <form>
      {step === 1 && <Step1 form={form} />}
      {step === 2 && <Step2 form={form} />}
      {step === 3 && <Step3 form={form} />}

      <div className="flex justify-between mt-4">
        {step > 1 && (
          <button type="button" onClick={() => setStep(step - 1)}>
            前へ
          </button>
        )}
        <button type="button" onClick={handleNextStep}>
          {step < 3 ? '次へ' : '送信'}
        </button>
      </div>

      {/* ステップインジケーター */}
      <div className="flex justify-center gap-2 mt-4">
        {[1, 2, 3].map((s) => (
          <div
            key={s}
            className={`w-8 h-8 rounded-full flex items-center justify-center ${
              s === step
                ? 'bg-blue-600 text-white'
                : s < step
                ? 'bg-green-500 text-white'
                : 'bg-gray-200 text-gray-500'
            }`}
          >
            {s < step ? '✓' : s}
          </div>
        ))}
      </div>
    </form>
  );
}
```

---

## 5. 非同期バリデーション

### 5.1 基本的な非同期バリデーション

```typescript
// メールアドレスの重複チェック（非同期）
const schema = z.object({
  email: z.string().email('有効なメールアドレスを入力してください'),
  username: z.string().min(3, '3文字以上で入力してください'),
});

function RegisterForm() {
  const form = useForm({
    resolver: zodResolver(schema),
  });

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <input
        {...form.register('email', {
          validate: async (value) => {
            if (!value) return true;
            // 基本的なフォーマットチェックを先に行う
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(value)) return true; // Zodに任せる

            try {
              const response = await fetch(
                `/api/check-email?email=${encodeURIComponent(value)}`
              );
              const data = await response.json();
              return data.available
                ? true
                : 'このメールアドレスは既に使用されています';
            } catch {
              // ネットワークエラー時はバリデーションをスキップ（サーバーで検証）
              return true;
            }
          },
        })}
      />
      {form.formState.errors.email && (
        <span className="text-red-500">
          {form.formState.errors.email.message}
        </span>
      )}

      <input
        {...form.register('username', {
          validate: async (value) => {
            if (!value || value.length < 3) return true;

            try {
              const response = await fetch(
                `/api/check-username?username=${encodeURIComponent(value)}`
              );
              const data = await response.json();
              return data.available
                ? true
                : 'このユーザー名は既に使用されています';
            } catch {
              return true;
            }
          },
        })}
      />
    </form>
  );
}
```

### 5.2 デバウンス付き非同期バリデーション

```typescript
import { useState, useCallback, useRef, useEffect } from 'react';

// カスタムフック: デバウンス付き非同期バリデーション
function useAsyncValidation(
  validateFn: (value: string) => Promise<string | true>,
  delay = 500
) {
  const [error, setError] = useState<string | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const lastValueRef = useRef<string>('');

  const validate = useCallback(
    (value: string) => {
      lastValueRef.current = value;

      // 前のタイマーをクリア
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      // 前のリクエストをキャンセル
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // 空の値はバリデーションスキップ
      if (!value) {
        setError(null);
        setIsValidating(false);
        return;
      }

      setIsValidating(true);

      timeoutRef.current = setTimeout(async () => {
        const controller = new AbortController();
        abortControllerRef.current = controller;

        try {
          const result = await validateFn(value);

          // 最新の値と一致する場合のみ結果を反映
          if (lastValueRef.current === value && !controller.signal.aborted) {
            setError(result === true ? null : result);
          }
        } catch (err) {
          if (err instanceof DOMException && err.name === 'AbortError') {
            // キャンセルされたリクエストは無視
            return;
          }
          // その他のエラーはバリデーションをスキップ
          setError(null);
        } finally {
          if (lastValueRef.current === value) {
            setIsValidating(false);
          }
        }
      }, delay);
    },
    [validateFn, delay]
  );

  // クリーンアップ
  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (abortControllerRef.current) abortControllerRef.current.abort();
    };
  }, []);

  const reset = useCallback(() => {
    setError(null);
    setIsValidating(false);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    if (abortControllerRef.current) abortControllerRef.current.abort();
  }, []);

  return { error, isValidating, validate, reset };
}

// 使用例
function UsernameField() {
  const form = useFormContext();

  const checkUsername = useCallback(async (value: string): Promise<string | true> => {
    const response = await fetch(
      `/api/check-username?username=${encodeURIComponent(value)}`
    );
    const data = await response.json();
    return data.available ? true : 'このユーザー名は既に使用されています';
  }, []);

  const { error: asyncError, isValidating, validate } = useAsyncValidation(
    checkUsername,
    500
  );

  const formError = form.formState.errors.username?.message;
  const displayError = formError || asyncError;

  return (
    <div className="relative">
      <input
        {...form.register('username')}
        onChange={(e) => {
          form.register('username').onChange(e); // React Hook FormのonChange
          validate(e.target.value);               // 非同期バリデーション
        }}
        className={displayError ? 'border-red-500' : 'border-gray-300'}
      />

      {/* ローディングインジケーター */}
      {isValidating && (
        <div className="absolute right-3 top-1/2 -translate-y-1/2">
          <svg className="animate-spin h-4 w-4 text-gray-400" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12" cy="12" r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
        </div>
      )}

      {/* 成功インジケーター */}
      {!isValidating && !displayError && form.getValues('username') && (
        <div className="absolute right-3 top-1/2 -translate-y-1/2 text-green-500">
          ✓
        </div>
      )}

      {displayError && (
        <p className="text-red-500 text-sm mt-1">{displayError as string}</p>
      )}
    </div>
  );
}
```

### 5.3 Zodでの非同期バリデーション（refine）

```typescript
// Zodスキーマ内での非同期バリデーション
const asyncRegisterSchema = z.object({
  username: z.string()
    .min(3, '3文字以上で入力してください')
    .max(20, '20文字以下で入力してください')
    .regex(/^[a-zA-Z0-9_]+$/, '英数字とアンダースコアのみ使用可能'),
  email: z.string().email('有効なメールアドレスを入力してください'),
  password: z.string().min(8, '8文字以上で入力してください'),
}).superRefine(async (data, ctx) => {
  // 注意: superRefineでの非同期バリデーションは
  // Submit時にのみ実行される（onChange/onBlurでは実行されない）

  // ユーザー名の重複チェック
  const usernameExists = await checkUsernameExists(data.username);
  if (usernameExists) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'このユーザー名は既に使用されています',
      path: ['username'],
    });
  }

  // メールアドレスの重複チェック
  const emailExists = await checkEmailExists(data.email);
  if (emailExists) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'このメールアドレスは既に使用されています',
      path: ['email'],
    });
  }
});

// 非同期バリデーションのAPIエンドポイント例
// app/api/check-username/route.ts
import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const username = searchParams.get('username');

  if (!username) {
    return NextResponse.json({ available: false }, { status: 400 });
  }

  const existingUser = await prisma.user.findUnique({
    where: { username },
    select: { id: true },
  });

  return NextResponse.json({
    available: !existingUser,
  });
}
```

---

## 6. クライアント/サーバー二重検証

### 6.1 なぜ二重検証が必要なのか

クライアントサイドバリデーションだけでは、以下の攻撃・問題を防げない:

1. **JavaScriptの無効化**: ブラウザ設定やアドオンでJSを無効にしたユーザー
2. **開発者ツールによる改ざん**: フォームの`required`属性やバリデーションロジックを書き換え
3. **直接のHTTPリクエスト**: cURL、Postmanなどでフォームを経由せずにリクエスト
4. **Botによる不正送信**: 自動化ツールがフォームバリデーションをバイパス
5. **ビジネスロジックの整合性**: DB状態に依存するバリデーション（重複チェック等）はサーバーでしか実行できない

```
二重検証のアーキテクチャ:

  ブラウザ                      サーバー
  ┌─────────────────┐         ┌─────────────────────────┐
  │                 │         │                         │
  │  フォーム入力    │         │  同じZodスキーマで検証    │
  │       ↓         │         │         ↓                │
  │  Zodスキーマ     │  HTTP   │  ビジネスロジック検証     │
  │  (クライアント)  │ ──────→ │  (重複チェック、権限等)   │
  │       ↓         │         │         ↓                │
  │  即座のFB表示    │  ←────  │  DB保存 or エラー返却    │
  │                 │  Error   │                         │
  └─────────────────┘  JSON   └─────────────────────────┘

  ポイント:
  - 同じZodスキーマを共有（shared/schemas/に配置）
  - クライアントはUX向上のため
  - サーバーはセキュリティ担保のため
  - サーバーにしかできない検証（DB依存）もある
```

### 6.2 共有スキーマの設計

```typescript
// ===================================================================
// shared/schemas/user.ts
// クライアントとサーバーで共有するスキーマ定義
// ===================================================================
import { z } from 'zod';

// 基本的なフィールドスキーマ（再利用可能なパーツ）
export const emailSchema = z.string()
  .email('有効なメールアドレスを入力してください')
  .max(254, 'メールアドレスが長すぎます')
  .toLowerCase();

export const passwordSchema = z.string()
  .min(8, '8文字以上で入力してください')
  .max(100, '100文字以下で入力してください')
  .regex(/[A-Z]/, '大文字を1文字以上含めてください')
  .regex(/[a-z]/, '小文字を1文字以上含めてください')
  .regex(/[0-9]/, '数字を1文字以上含めてください');

export const usernameSchema = z.string()
  .min(3, '3文字以上で入力してください')
  .max(20, '20文字以下で入力してください')
  .regex(/^[a-zA-Z0-9_]+$/, '英数字とアンダースコアのみ使用可能');

// ユーザー作成スキーマ
export const createUserSchema = z.object({
  username: usernameSchema,
  email: emailSchema,
  password: passwordSchema,
  name: z.string().min(1, '名前を入力してください').max(100),
  role: z.enum(['user', 'admin']).default('user'),
}).refine(
  (data) => !data.password.toLowerCase().includes(data.username.toLowerCase()),
  {
    message: 'パスワードにユーザー名を含めることはできません',
    path: ['password'],
  }
);

export type CreateUserInput = z.infer<typeof createUserSchema>;

// ユーザー更新スキーマ（部分更新対応）
export const updateUserSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  email: emailSchema.optional(),
  bio: z.string().max(500).optional(),
  website: z.string().url().optional().or(z.literal('')),
});

export type UpdateUserInput = z.infer<typeof updateUserSchema>;

// ログインスキーマ
export const loginSchema = z.object({
  email: z.string().email('有効なメールアドレスを入力してください'),
  password: z.string().min(1, 'パスワードを入力してください'),
  rememberMe: z.boolean().default(false),
});

export type LoginInput = z.infer<typeof loginSchema>;

// パスワードリセットスキーマ
export const passwordResetSchema = z.object({
  token: z.string().min(1),
  password: passwordSchema,
  confirmPassword: z.string(),
}).refine(
  (data) => data.password === data.confirmPassword,
  {
    message: 'パスワードが一致しません',
    path: ['confirmPassword'],
  }
);

export type PasswordResetInput = z.infer<typeof passwordResetSchema>;
```

### 6.3 クライアント側の実装

```typescript
// ===================================================================
// app/(auth)/register/page.tsx
// クライアント側: React Hook Form + 共有Zodスキーマ
// ===================================================================
'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { createUserSchema, type CreateUserInput } from '@shared/schemas/user';
import { registerAction } from './actions';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

export default function RegisterPage() {
  const router = useRouter();
  const [serverError, setServerError] = useState<string | null>(null);

  const form = useForm<CreateUserInput>({
    resolver: zodResolver(createUserSchema),
    mode: 'onSubmit',
    reValidateMode: 'onChange',
    defaultValues: {
      username: '',
      email: '',
      password: '',
      name: '',
      role: 'user',
    },
  });

  const onSubmit = async (data: CreateUserInput) => {
    setServerError(null);

    try {
      const result = await registerAction(data);

      if (result.errors) {
        // サーバーからのフィールドエラーをフォームにマッピング
        Object.entries(result.errors).forEach(([field, messages]) => {
          if (field === '_form') {
            // フォーム全体のエラー
            setServerError((messages as string[])[0]);
          } else {
            form.setError(field as keyof CreateUserInput, {
              type: 'server',
              message: (messages as string[])[0],
            });
          }
        });
        return;
      }

      // 成功時
      router.push('/login?registered=true');
    } catch (error) {
      setServerError('予期しないエラーが発生しました。もう一度お試しください。');
    }
  };

  return (
    <div className="max-w-md mx-auto mt-8">
      <h1 className="text-2xl font-bold mb-6">アカウント作成</h1>

      {serverError && (
        <div role="alert" className="bg-red-50 border border-red-200 p-4 rounded mb-4">
          <p className="text-red-700">{serverError}</p>
        </div>
      )}

      <form onSubmit={form.handleSubmit(onSubmit)} noValidate>
        {/* フォームフィールド */}
        <div className="space-y-4">
          <FormField label="ユーザー名" error={form.formState.errors.username?.message} required>
            <input {...form.register('username')} autoComplete="username" />
          </FormField>

          <FormField label="名前" error={form.formState.errors.name?.message} required>
            <input {...form.register('name')} autoComplete="name" />
          </FormField>

          <FormField label="メールアドレス" error={form.formState.errors.email?.message} required>
            <input type="email" {...form.register('email')} autoComplete="email" />
          </FormField>

          <FormField label="パスワード" error={form.formState.errors.password?.message} required>
            <input type="password" {...form.register('password')} autoComplete="new-password" />
          </FormField>

          <button
            type="submit"
            disabled={form.formState.isSubmitting}
            className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {form.formState.isSubmitting ? '登録中...' : 'アカウント作成'}
          </button>
        </div>
      </form>
    </div>
  );
}
```

### 6.4 サーバー側の実装（Server Actions）

```typescript
// ===================================================================
// app/(auth)/register/actions.ts
// サーバー側: Server Action + 同じZodスキーマ
// ===================================================================
'use server';

import { createUserSchema, type CreateUserInput } from '@shared/schemas/user';
import { prisma } from '@/lib/prisma';
import { hash } from 'bcryptjs';
import { revalidatePath } from 'next/cache';

// サーバー固有のバリデーション（DB依存の検証）
async function validateServerConstraints(data: CreateUserInput) {
  const errors: Record<string, string[]> = {};

  // ユーザー名の重複チェック
  const existingUsername = await prisma.user.findUnique({
    where: { username: data.username },
    select: { id: true },
  });
  if (existingUsername) {
    errors.username = ['このユーザー名は既に使用されています'];
  }

  // メールアドレスの重複チェック
  const existingEmail = await prisma.user.findUnique({
    where: { email: data.email },
    select: { id: true },
  });
  if (existingEmail) {
    errors.email = ['このメールアドレスは既に使用されています'];
  }

  // ブロックリストのチェック
  const isBlocked = await prisma.blockedEmail.findUnique({
    where: { email: data.email },
  });
  if (isBlocked) {
    errors.email = ['このメールアドレスは使用できません'];
  }

  return Object.keys(errors).length > 0 ? errors : null;
}

export async function registerAction(input: unknown) {
  // Step 1: 共有スキーマでバリデーション
  const parsed = createUserSchema.safeParse(input);

  if (!parsed.success) {
    return {
      errors: parsed.error.flatten().fieldErrors,
    };
  }

  // Step 2: サーバー固有のバリデーション
  const serverErrors = await validateServerConstraints(parsed.data);
  if (serverErrors) {
    return { errors: serverErrors };
  }

  // Step 3: データベースへの保存
  try {
    const hashedPassword = await hash(parsed.data.password, 12);

    await prisma.user.create({
      data: {
        username: parsed.data.username,
        email: parsed.data.email,
        name: parsed.data.name,
        password: hashedPassword,
        role: parsed.data.role,
      },
    });

    revalidatePath('/users');
    return { success: true };
  } catch (error) {
    console.error('User registration failed:', error);
    return {
      errors: {
        _form: ['ユーザー登録に失敗しました。もう一度お試しください。'],
      },
    };
  }
}
```

### 6.5 APIルートでの二重検証

```typescript
// ===================================================================
// app/api/users/route.ts
// REST APIでの二重検証パターン
// ===================================================================
import { NextRequest, NextResponse } from 'next/server';
import { createUserSchema } from '@shared/schemas/user';
import { prisma } from '@/lib/prisma';
import { hash } from 'bcryptjs';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

// バリデーションエラーレスポンスの型定義
type ValidationErrorResponse = {
  success: false;
  errors: {
    fieldErrors: Record<string, string[]>;
    formErrors: string[];
  };
};

type SuccessResponse<T> = {
  success: true;
  data: T;
};

type ApiResponse<T> = ValidationErrorResponse | SuccessResponse<T>;

// バリデーションヘルパー関数
function createValidationError(
  fieldErrors: Record<string, string[]>,
  formErrors: string[] = []
): NextResponse<ValidationErrorResponse> {
  return NextResponse.json(
    {
      success: false,
      errors: { fieldErrors, formErrors },
    },
    { status: 422 }
  );
}

export async function POST(request: NextRequest) {
  // 認証チェック
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json(
      { success: false, message: '認証が必要です' },
      { status: 401 }
    );
  }

  // 権限チェック
  if (session.user.role !== 'admin') {
    return NextResponse.json(
      { success: false, message: '権限がありません' },
      { status: 403 }
    );
  }

  // リクエストボディのパース
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { success: false, message: 'リクエストボディが不正です' },
      { status: 400 }
    );
  }

  // Step 1: Zodスキーマでバリデーション
  const parsed = createUserSchema.safeParse(body);
  if (!parsed.success) {
    const flattened = parsed.error.flatten();
    return createValidationError(
      flattened.fieldErrors as Record<string, string[]>,
      flattened.formErrors
    );
  }

  // Step 2: ビジネスロジックバリデーション
  const existingUser = await prisma.user.findFirst({
    where: {
      OR: [
        { email: parsed.data.email },
        { username: parsed.data.username },
      ],
    },
    select: { email: true, username: true },
  });

  if (existingUser) {
    const fieldErrors: Record<string, string[]> = {};
    if (existingUser.email === parsed.data.email) {
      fieldErrors.email = ['このメールアドレスは既に使用されています'];
    }
    if (existingUser.username === parsed.data.username) {
      fieldErrors.username = ['このユーザー名は既に使用されています'];
    }
    return createValidationError(fieldErrors);
  }

  // Step 3: データの保存
  try {
    const hashedPassword = await hash(parsed.data.password, 12);
    const user = await prisma.user.create({
      data: {
        ...parsed.data,
        password: hashedPassword,
      },
      select: {
        id: true,
        username: true,
        email: true,
        name: true,
        role: true,
        createdAt: true,
      },
    });

    return NextResponse.json(
      { success: true, data: user },
      { status: 201 }
    );
  } catch (error) {
    console.error('Failed to create user:', error);
    return NextResponse.json(
      { success: false, message: 'ユーザーの作成に失敗しました' },
      { status: 500 }
    );
  }
}
```

### 6.6 tRPCでの二重検証

```typescript
// ===================================================================
// server/routers/user.ts
// tRPCでのスキーマ共有パターン
// ===================================================================
import { router, protectedProcedure, publicProcedure } from '../trpc';
import { createUserSchema, updateUserSchema, loginSchema } from '@shared/schemas/user';
import { hash, compare } from 'bcryptjs';
import { TRPCError } from '@trpc/server';

export const userRouter = router({
  // ユーザー作成
  create: publicProcedure
    .input(createUserSchema) // Zodスキーマをそのまま入力バリデーションに使用
    .mutation(async ({ input, ctx }) => {
      // tRPCは自動的にinputをZodスキーマでバリデーションする
      // バリデーションエラーは自動的にTRPCErrorとして返される

      // ビジネスロジックバリデーション
      const existing = await ctx.prisma.user.findFirst({
        where: {
          OR: [{ email: input.email }, { username: input.username }],
        },
      });

      if (existing) {
        throw new TRPCError({
          code: 'CONFLICT',
          message: existing.email === input.email
            ? 'このメールアドレスは既に使用されています'
            : 'このユーザー名は既に使用されています',
        });
      }

      const hashedPassword = await hash(input.password, 12);
      return ctx.prisma.user.create({
        data: { ...input, password: hashedPassword },
        select: { id: true, username: true, email: true, name: true },
      });
    }),

  // ユーザー更新
  update: protectedProcedure
    .input(updateUserSchema) // 部分更新スキーマ
    .mutation(async ({ input, ctx }) => {
      return ctx.prisma.user.update({
        where: { id: ctx.session.user.id },
        data: input,
      });
    }),

  // ログイン
  login: publicProcedure
    .input(loginSchema)
    .mutation(async ({ input, ctx }) => {
      const user = await ctx.prisma.user.findUnique({
        where: { email: input.email },
      });

      if (!user || !(await compare(input.password, user.password))) {
        throw new TRPCError({
          code: 'UNAUTHORIZED',
          message: 'メールアドレスまたはパスワードが正しくありません',
        });
      }

      // セッション作成ロジック...
      return { user: { id: user.id, email: user.email, name: user.name } };
    }),
});
```

---

## 7. パスワード強度インジケーター

### 7.1 パスワード強度の計算ロジック

```typescript
// ===================================================================
// lib/password-strength.ts
// パスワード強度を多角的に評価する関数群
// ===================================================================

// パスワード強度の評価結果
export type PasswordStrength = {
  score: number;          // 0〜4のスコア
  label: string;          // 強度ラベル
  color: string;          // Tailwind CSSカラークラス
  textColor: string;      // テキストカラークラス
  percentage: number;     // パーセンテージ（0〜100）
  feedback: string[];     // 改善のためのフィードバック
  requirements: {         // 各要件の達成状況
    minLength: boolean;
    hasUppercase: boolean;
    hasLowercase: boolean;
    hasNumber: boolean;
    hasSpecial: boolean;
    noCommonPattern: boolean;
  };
};

// よく使われるパスワードのリスト（上位100件）
const COMMON_PASSWORDS = new Set([
  'password', '123456', '123456789', 'qwerty', 'abc123',
  'monkey', '1234567', 'letmein', 'trustno1', 'dragon',
  'baseball', 'iloveyou', 'master', 'sunshine', 'ashley',
  'michael', 'shadow', '123123', '654321', 'superman',
  'qazwsx', 'football', 'password1', 'password123',
  // ... 省略
]);

// よく使われるパターンの検出
const COMMON_PATTERNS = [
  /^(.)\1{2,}$/,                    // 同一文字の繰り返し（aaa, 1111）
  /^(012|123|234|345|456|567|678|789|890)+$/, // 連番
  /^(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)+$/i, // アルファベット連続
  /^(qwerty|asdf|zxcv|wasd)/i,     // キーボード配列
  /^(19|20)\d{2}/,                   // 年号で始まる
];

// エントロピーの計算
function calculateEntropy(password: string): number {
  const charsetSize = getCharsetSize(password);
  return password.length * Math.log2(charsetSize);
}

function getCharsetSize(password: string): number {
  let size = 0;
  if (/[a-z]/.test(password)) size += 26;
  if (/[A-Z]/.test(password)) size += 26;
  if (/[0-9]/.test(password)) size += 10;
  if (/[^a-zA-Z0-9]/.test(password)) size += 32;
  return size || 1;
}

// メインの評価関数
export function getPasswordStrength(password: string): PasswordStrength {
  if (!password) {
    return {
      score: 0,
      label: '',
      color: 'bg-gray-200',
      textColor: 'text-gray-400',
      percentage: 0,
      feedback: [],
      requirements: {
        minLength: false,
        hasUppercase: false,
        hasLowercase: false,
        hasNumber: false,
        hasSpecial: false,
        noCommonPattern: true,
      },
    };
  }

  // 要件チェック
  const requirements = {
    minLength: password.length >= 8,
    hasUppercase: /[A-Z]/.test(password),
    hasLowercase: /[a-z]/.test(password),
    hasNumber: /[0-9]/.test(password),
    hasSpecial: /[^A-Za-z0-9]/.test(password),
    noCommonPattern: !COMMON_PASSWORDS.has(password.toLowerCase()) &&
      !COMMON_PATTERNS.some((pattern) => pattern.test(password)),
  };

  // スコア計算
  let score = 0;

  // 基本要件のスコア
  const metRequirements = Object.values(requirements).filter(Boolean).length;
  score += metRequirements * 0.5;

  // 長さボーナス
  if (password.length >= 12) score += 0.5;
  if (password.length >= 16) score += 0.5;

  // エントロピーボーナス
  const entropy = calculateEntropy(password);
  if (entropy >= 40) score += 0.5;
  if (entropy >= 60) score += 0.5;

  // よくあるパスワード・パターンはスコアを大幅に下げる
  if (!requirements.noCommonPattern) {
    score = Math.min(score, 1);
  }

  // 0〜4に正規化
  score = Math.min(Math.round(score), 4);

  // フィードバック生成
  const feedback: string[] = [];
  if (!requirements.minLength) feedback.push('8文字以上にしてください');
  if (!requirements.hasUppercase) feedback.push('大文字を追加してください');
  if (!requirements.hasLowercase) feedback.push('小文字を追加してください');
  if (!requirements.hasNumber) feedback.push('数字を追加してください');
  if (!requirements.hasSpecial) feedback.push('記号（!@#$%等）を追加してください');
  if (!requirements.noCommonPattern) feedback.push('よく使われるパスワード・パターンは避けてください');
  if (password.length < 12) feedback.push('12文字以上にするとより安全です');

  const labels = ['非常に弱い', '弱い', '普通', '強い', '非常に強い'];
  const colors = ['bg-red-500', 'bg-orange-500', 'bg-yellow-500', 'bg-green-400', 'bg-green-600'];
  const textColors = ['text-red-600', 'text-orange-600', 'text-yellow-600', 'text-green-500', 'text-green-700'];

  return {
    score,
    label: labels[score],
    color: colors[score],
    textColor: textColors[score],
    percentage: (score / 4) * 100,
    feedback,
    requirements,
  };
}
```

### 7.2 パスワード強度インジケーターコンポーネント

```typescript
// ===================================================================
// components/PasswordStrengthIndicator.tsx
// パスワード強度を視覚的に表示するコンポーネント
// ===================================================================
'use client';

import { useMemo } from 'react';
import { getPasswordStrength, type PasswordStrength } from '@/lib/password-strength';

type Props = {
  password: string;
  showRequirements?: boolean;
  showFeedback?: boolean;
};

export function PasswordStrengthIndicator({
  password,
  showRequirements = true,
  showFeedback = true,
}: Props) {
  const strength = useMemo(() => getPasswordStrength(password), [password]);

  if (!password) return null;

  return (
    <div className="mt-2 space-y-2">
      {/* 強度バー */}
      <div className="space-y-1">
        <div className="flex gap-1 h-1.5">
          {[0, 1, 2, 3].map((i) => (
            <div
              key={i}
              className={`flex-1 rounded-full transition-colors duration-300 ${
                i <= strength.score - 1 ? strength.color : 'bg-gray-200'
              }`}
              role="presentation"
            />
          ))}
        </div>
        <p className={`text-xs font-medium ${strength.textColor}`}>
          パスワード強度: {strength.label}
        </p>
      </div>

      {/* 要件チェックリスト */}
      {showRequirements && (
        <ul className="text-xs space-y-1" aria-label="パスワード要件">
          <RequirementItem met={strength.requirements.minLength} label="8文字以上" />
          <RequirementItem met={strength.requirements.hasUppercase} label="大文字を含む" />
          <RequirementItem met={strength.requirements.hasLowercase} label="小文字を含む" />
          <RequirementItem met={strength.requirements.hasNumber} label="数字を含む" />
          <RequirementItem met={strength.requirements.hasSpecial} label="記号を含む" />
        </ul>
      )}

      {/* 改善のフィードバック */}
      {showFeedback && strength.feedback.length > 0 && strength.score < 3 && (
        <div className="text-xs text-gray-500">
          <p className="font-medium">改善のヒント:</p>
          <ul className="list-disc list-inside">
            {strength.feedback.slice(0, 3).map((fb, i) => (
              <li key={i}>{fb}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function RequirementItem({ met, label }: { met: boolean; label: string }) {
  return (
    <li className={`flex items-center gap-1.5 ${met ? 'text-green-600' : 'text-gray-400'}`}>
      {met ? (
        <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
            clipRule="evenodd"
          />
        </svg>
      ) : (
        <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
            clipRule="evenodd"
          />
        </svg>
      )}
      <span>{label}</span>
    </li>
  );
}
```

### 7.3 パスワードフィールドの完全な実装

```typescript
// ===================================================================
// components/PasswordField.tsx
// パスワード表示/非表示切替 + 強度表示を備えた完全なパスワードフィールド
// ===================================================================
'use client';

import { useState, useCallback } from 'react';
import { useFormContext } from 'react-hook-form';
import { PasswordStrengthIndicator } from './PasswordStrengthIndicator';

type Props = {
  name: string;
  label: string;
  showStrength?: boolean;
  autoComplete?: string;
  placeholder?: string;
};

export function PasswordField({
  name,
  label,
  showStrength = false,
  autoComplete = 'current-password',
  placeholder,
}: Props) {
  const [showPassword, setShowPassword] = useState(false);
  const {
    register,
    watch,
    formState: { errors },
  } = useFormContext();

  const password = watch(name) || '';
  const error = errors[name]?.message as string | undefined;

  const toggleVisibility = useCallback(() => {
    setShowPassword((prev) => !prev);
  }, []);

  return (
    <div className="space-y-1">
      <label htmlFor={name} className="block text-sm font-medium text-gray-700">
        {label}
        <span className="text-red-500 ml-1">*</span>
      </label>

      <div className="relative">
        <input
          id={name}
          type={showPassword ? 'text' : 'password'}
          {...register(name)}
          autoComplete={autoComplete}
          placeholder={placeholder}
          aria-invalid={!!error}
          aria-describedby={error ? `${name}-error` : undefined}
          className={`w-full pr-10 rounded-md border ${
            error ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 focus:ring-blue-500'
          } focus:outline-none focus:ring-2 px-3 py-2`}
        />

        <button
          type="button"
          onClick={toggleVisibility}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
          aria-label={showPassword ? 'パスワードを隠す' : 'パスワードを表示'}
          tabIndex={-1}
        >
          {showPassword ? (
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"
              />
            </svg>
          ) : (
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
              />
            </svg>
          )}
        </button>
      </div>

      {error && (
        <p id={`${name}-error`} className="text-red-500 text-sm" role="alert">
          {error}
        </p>
      )}

      {showStrength && <PasswordStrengthIndicator password={password} />}
    </div>
  );
}
```

---

## 8. エラーメッセージとアクセシビリティ

### 8.1 アクセシブルなエラー表示パターン

フォームのバリデーションエラーをアクセシブルに表示するためには、WAI-ARIA仕様に準拠した実装が必要である。スクリーンリーダーユーザーがエラーの存在と内容を把握できるように設計する。

```typescript
// ===================================================================
// components/AccessibleFormField.tsx
// WAI-ARIA準拠のアクセシブルなフォームフィールド
// ===================================================================
import { forwardRef, useId } from 'react';

type Props = {
  label: string;
  error?: string;
  hint?: string;
  required?: boolean;
  children: (props: {
    id: string;
    'aria-invalid': boolean;
    'aria-describedby': string | undefined;
    'aria-required': boolean;
  }) => React.ReactNode;
};

export function AccessibleFormField({
  label,
  error,
  hint,
  required = false,
  children,
}: Props) {
  const id = useId();
  const errorId = `${id}-error`;
  const hintId = `${id}-hint`;

  // aria-describedbyに設定するIDのリスト
  const describedByIds = [
    hint ? hintId : null,
    error ? errorId : null,
  ].filter(Boolean).join(' ') || undefined;

  return (
    <div className="mb-4">
      <label htmlFor={id} className="block text-sm font-medium text-gray-700 mb-1">
        {label}
        {required && (
          <span className="text-red-500 ml-1" aria-hidden="true">*</span>
        )}
        {required && <span className="sr-only">（必須）</span>}
      </label>

      {/* ヒントテキスト */}
      {hint && (
        <p id={hintId} className="text-sm text-gray-500 mb-1">
          {hint}
        </p>
      )}

      {/* フォーム要素（render prop パターン） */}
      {children({
        id,
        'aria-invalid': !!error,
        'aria-describedby': describedByIds,
        'aria-required': required,
      })}

      {/* エラーメッセージ（aria-liveで動的に通知） */}
      {error && (
        <p
          id={errorId}
          className="text-red-500 text-sm mt-1"
          role="alert"
          aria-live="polite"
        >
          {error}
        </p>
      )}
    </div>
  );
}

// 使用例
function ExampleForm() {
  const form = useForm();

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <AccessibleFormField
        label="メールアドレス"
        error={form.formState.errors.email?.message as string}
        hint="ログインに使用するメールアドレスを入力してください"
        required
      >
        {(ariaProps) => (
          <input
            type="email"
            {...form.register('email')}
            {...ariaProps}
            className={`w-full border rounded-md px-3 py-2 ${
              ariaProps['aria-invalid'] ? 'border-red-500' : 'border-gray-300'
            }`}
          />
        )}
      </AccessibleFormField>
    </form>
  );
}
```

### 8.2 エラーサマリー表示

```typescript
// ===================================================================
// components/ErrorSummary.tsx
// フォーム送信後にエラーのサマリーを表示するコンポーネント
// ===================================================================
import { useEffect, useRef } from 'react';
import { type FieldErrors } from 'react-hook-form';

type Props = {
  errors: FieldErrors;
  fieldLabels: Record<string, string>;
};

export function ErrorSummary({ errors, fieldLabels }: Props) {
  const summaryRef = useRef<HTMLDivElement>(null);
  const errorEntries = Object.entries(errors).filter(
    ([key]) => key !== 'root'
  );

  // エラーが表示されたら自動的にフォーカスを移動
  useEffect(() => {
    if (errorEntries.length > 0 && summaryRef.current) {
      summaryRef.current.focus();
    }
  }, [errorEntries.length]);

  if (errorEntries.length === 0) return null;

  return (
    <div
      ref={summaryRef}
      role="alert"
      aria-labelledby="error-summary-title"
      tabIndex={-1}
      className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6"
    >
      <h2
        id="error-summary-title"
        className="text-red-800 font-semibold text-sm mb-2"
      >
        {errorEntries.length}件のエラーがあります
      </h2>
      <ul className="list-disc list-inside space-y-1">
        {errorEntries.map(([fieldName, error]) => (
          <li key={fieldName} className="text-red-700 text-sm">
            <a
              href={`#${fieldName}`}
              className="underline hover:no-underline"
              onClick={(e) => {
                e.preventDefault();
                // エラーのあるフィールドにフォーカスを移動
                const field = document.getElementById(fieldName);
                field?.focus();
                field?.scrollIntoView({ behavior: 'smooth', block: 'center' });
              }}
            >
              {fieldLabels[fieldName] || fieldName}
            </a>
            : {(error as { message?: string })?.message}
          </li>
        ))}
      </ul>
    </div>
  );
}

// 使用例
function FormWithErrorSummary() {
  const form = useForm();
  const [showSummary, setShowSummary] = useState(false);

  const fieldLabels: Record<string, string> = {
    name: '名前',
    email: 'メールアドレス',
    password: 'パスワード',
    phone: '電話番号',
  };

  const onInvalid = () => {
    setShowSummary(true);
  };

  return (
    <form onSubmit={form.handleSubmit(onSubmit, onInvalid)}>
      {showSummary && (
        <ErrorSummary
          errors={form.formState.errors}
          fieldLabels={fieldLabels}
        />
      )}
      {/* フォームフィールド */}
    </form>
  );
}
```

### 8.3 国際化（i18n）対応のエラーメッセージ

```typescript
// ===================================================================
// lib/validation-messages.ts
// 多言語対応のバリデーションメッセージ管理
// ===================================================================
type Locale = 'ja' | 'en' | 'zh' | 'ko';

type MessageKey =
  | 'required'
  | 'email'
  | 'url'
  | 'min_length'
  | 'max_length'
  | 'min_value'
  | 'max_value'
  | 'pattern'
  | 'password_mismatch'
  | 'password_too_weak'
  | 'email_taken'
  | 'username_taken'
  | 'agree_to_terms';

// メッセージテンプレート（パラメータ対応）
const messages: Record<Locale, Record<MessageKey, string>> = {
  ja: {
    required: 'この項目は必須です',
    email: '有効なメールアドレスを入力してください',
    url: '有効なURLを入力してください',
    min_length: '{min}文字以上で入力してください',
    max_length: '{max}文字以下で入力してください',
    min_value: '{min}以上の値を入力してください',
    max_value: '{max}以下の値を入力してください',
    pattern: '入力形式が正しくありません',
    password_mismatch: 'パスワードが一致しません',
    password_too_weak: 'パスワードが弱すぎます',
    email_taken: 'このメールアドレスは既に使用されています',
    username_taken: 'このユーザー名は既に使用されています',
    agree_to_terms: '利用規約に同意してください',
  },
  en: {
    required: 'This field is required',
    email: 'Please enter a valid email address',
    url: 'Please enter a valid URL',
    min_length: 'Must be at least {min} characters',
    max_length: 'Must be at most {max} characters',
    min_value: 'Must be at least {min}',
    max_value: 'Must be at most {max}',
    pattern: 'Invalid format',
    password_mismatch: 'Passwords do not match',
    password_too_weak: 'Password is too weak',
    email_taken: 'This email is already in use',
    username_taken: 'This username is already taken',
    agree_to_terms: 'You must agree to the terms',
  },
  zh: {
    required: '此项为必填项',
    email: '请输入有效的电子邮件地址',
    url: '请输入有效的URL',
    min_length: '请输入至少{min}个字符',
    max_length: '请输入不超过{max}个字符',
    min_value: '请输入不小于{min}的值',
    max_value: '请输入不大于{max}的值',
    pattern: '输入格式不正确',
    password_mismatch: '两次输入的密码不一致',
    password_too_weak: '密码强度不够',
    email_taken: '此电子邮件已被使用',
    username_taken: '此用户名已被使用',
    agree_to_terms: '请同意使用条款',
  },
  ko: {
    required: '이 항목은 필수입니다',
    email: '유효한 이메일 주소를 입력해 주세요',
    url: '유효한 URL을 입력해 주세요',
    min_length: '{min}자 이상 입력해 주세요',
    max_length: '{max}자 이하로 입력해 주세요',
    min_value: '{min} 이상의 값을 입력해 주세요',
    max_value: '{max} 이하의 값을 입력해 주세요',
    pattern: '입력 형식이 올바르지 않습니다',
    password_mismatch: '비밀번호가 일치하지 않습니다',
    password_too_weak: '비밀번호가 너무 약합니다',
    email_taken: '이미 사용중인 이메일 주소입니다',
    username_taken: '이미 사용중인 사용자 이름입니다',
    agree_to_terms: '이용 약관에 동의해 주세요',
  },
};

// メッセージ取得関数
export function getMessage(
  key: MessageKey,
  locale: Locale = 'ja',
  params?: Record<string, string | number>
): string {
  let message = messages[locale]?.[key] || messages.ja[key] || key;

  if (params) {
    Object.entries(params).forEach(([paramKey, value]) => {
      message = message.replace(`{${paramKey}}`, String(value));
    });
  }

  return message;
}

// Zodスキーマで使用する場合
export function createLocalizedSchema(locale: Locale = 'ja') {
  const t = (key: MessageKey, params?: Record<string, string | number>) =>
    getMessage(key, locale, params);

  return z.object({
    name: z.string().min(1, t('required')),
    email: z.string().email(t('email')),
    password: z.string()
      .min(8, t('min_length', { min: 8 }))
      .max(100, t('max_length', { max: 100 })),
    confirmPassword: z.string(),
    agreeToTerms: z.literal(true, {
      errorMap: () => ({ message: t('agree_to_terms') }),
    }),
  }).refine(
    (data) => data.password === data.confirmPassword,
    {
      message: t('password_mismatch'),
      path: ['confirmPassword'],
    }
  );
}

// Reactコンテキストでの使用
import { createContext, useContext, type ReactNode } from 'react';

const LocaleContext = createContext<Locale>('ja');

export function LocaleProvider({
  locale,
  children,
}: {
  locale: Locale;
  children: ReactNode;
}) {
  return (
    <LocaleContext.Provider value={locale}>{children}</LocaleContext.Provider>
  );
}

export function useLocale() {
  return useContext(LocaleContext);
}

export function useValidationMessage() {
  const locale = useLocale();
  return (key: MessageKey, params?: Record<string, string | number>) =>
    getMessage(key, locale, params);
}
```

### 8.4 アクセシビリティチェックリスト

フォームバリデーションにおけるアクセシビリティ対応のチェックリスト:

| 項目 | 対応方法 | 重要度 |
|------|---------|--------|
| エラーメッセージとフィールドの関連付け | `aria-describedby`でエラーメッセージのIDを参照 | 必須 |
| エラー状態の明示 | `aria-invalid="true"`を設定 | 必須 |
| 必須フィールドの明示 | `aria-required="true"` + 視覚的なマーカー | 必須 |
| エラーの動的通知 | `role="alert"` または `aria-live="polite"` | 必須 |
| 色だけに依存しない | アイコンやテキストでもエラーを伝達 | 必須 |
| エラーサマリーへのフォーカス | Submit失敗時にエラーサマリーにフォーカス移動 | 推奨 |
| エラーフィールドへのスクロール | エラーリンクからフィールドにジャンプ | 推奨 |
| 入力ヒントの提供 | `aria-describedby`でヒントテキストを関連付け | 推奨 |
| フォーカス管理 | Tab順序が論理的であること | 必須 |
| キーボード操作 | Enterキーで送信できること | 必須 |

```typescript
// アクセシビリティ対応の完全なフォーム例
function AccessibleRegistrationForm() {
  const form = useForm<FormData>({
    resolver: zodResolver(schema),
    mode: 'onSubmit',
    reValidateMode: 'onChange',
  });

  const firstErrorRef = useRef<HTMLInputElement>(null);

  const onInvalid = (errors: FieldErrors) => {
    // 最初のエラーフィールドにフォーカスを移動
    const firstErrorField = Object.keys(errors)[0];
    if (firstErrorField) {
      const element = document.getElementById(firstErrorField);
      element?.focus();
      element?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  return (
    <form
      onSubmit={form.handleSubmit(onSubmit, onInvalid)}
      noValidate  // ブラウザネイティブバリデーションを無効化
      aria-label="ユーザー登録フォーム"
    >
      {/* Submit失敗時のエラーサマリー */}
      {form.formState.isSubmitted && !form.formState.isValid && (
        <div
          role="alert"
          tabIndex={-1}
          className="bg-red-50 border-l-4 border-red-500 p-4 mb-6"
        >
          <p className="font-bold text-red-800">入力内容にエラーがあります</p>
          <p className="text-red-700 text-sm">
            以下のフィールドを確認してください。
          </p>
        </div>
      )}

      {/* 各フィールド */}
      <div className="space-y-4">
        <div>
          <label htmlFor="email" className="block font-medium">
            メールアドレス
            <span className="text-red-500" aria-hidden="true">*</span>
            <span className="sr-only">（必須）</span>
          </label>
          <input
            id="email"
            type="email"
            {...form.register('email')}
            aria-invalid={!!form.formState.errors.email}
            aria-describedby={
              [
                'email-hint',
                form.formState.errors.email ? 'email-error' : null,
              ].filter(Boolean).join(' ')
            }
            aria-required="true"
            autoComplete="email"
          />
          <p id="email-hint" className="text-sm text-gray-500 mt-1">
            ログインに使用するメールアドレスを入力してください
          </p>
          {form.formState.errors.email && (
            <p id="email-error" role="alert" className="text-red-500 text-sm mt-1">
              {form.formState.errors.email.message}
            </p>
          )}
        </div>
      </div>

      <button
        type="submit"
        disabled={form.formState.isSubmitting}
        aria-busy={form.formState.isSubmitting}
      >
        {form.formState.isSubmitting ? (
          <>
            <span className="sr-only">送信中</span>
            <span aria-hidden="true">送信中...</span>
          </>
        ) : (
          '登録する'
        )}
      </button>
    </form>
  );
}
```

---

## 9. バリデーションのテスト戦略

### 9.1 Zodスキーマのユニットテスト

Zodスキーマは純粋な関数として動作するため、ユニットテストが書きやすい。すべてのエッジケースを網羅的にテストすることで、バリデーションロジックの信頼性を確保できる。

```typescript
// ===================================================================
// __tests__/schemas/user.test.ts
// Zodスキーマのユニットテスト
// ===================================================================
import { describe, it, expect } from 'vitest';
import { createUserSchema, loginSchema, passwordResetSchema } from '@shared/schemas/user';

describe('createUserSchema', () => {
  // 正常系テスト
  describe('valid inputs', () => {
    it('全ての必須フィールドが正しい場合にパースが成功する', () => {
      const validData = {
        username: 'testuser',
        email: 'test@example.com',
        password: 'Password1!',
        name: 'テストユーザー',
        role: 'user',
      };

      const result = createUserSchema.safeParse(validData);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.username).toBe('testuser');
        expect(result.data.email).toBe('test@example.com');
      }
    });

    it('roleのデフォルト値が適用される', () => {
      const data = {
        username: 'testuser',
        email: 'test@example.com',
        password: 'Password1!',
        name: 'テストユーザー',
      };

      const result = createUserSchema.safeParse(data);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.role).toBe('user');
      }
    });
  });

  // 異常系テスト: username
  describe('username validation', () => {
    const baseData = {
      email: 'test@example.com',
      password: 'Password1!',
      name: 'テストユーザー',
      role: 'user' as const,
    };

    it('3文字未満のユーザー名を拒否する', () => {
      const result = createUserSchema.safeParse({
        ...baseData,
        username: 'ab',
      });
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.flatten().fieldErrors.username).toContain(
          '3文字以上で入力してください'
        );
      }
    });

    it('20文字超のユーザー名を拒否する', () => {
      const result = createUserSchema.safeParse({
        ...baseData,
        username: 'a'.repeat(21),
      });
      expect(result.success).toBe(false);
    });

    it('特殊文字を含むユーザー名を拒否する', () => {
      const invalidUsernames = ['user name', 'user@name', 'user.name', 'ユーザー'];
      invalidUsernames.forEach((username) => {
        const result = createUserSchema.safeParse({ ...baseData, username });
        expect(result.success).toBe(false);
      });
    });

    it('アンダースコアを含むユーザー名を許可する', () => {
      const result = createUserSchema.safeParse({
        ...baseData,
        username: 'test_user_123',
      });
      expect(result.success).toBe(true);
    });
  });

  // 異常系テスト: email
  describe('email validation', () => {
    const baseData = {
      username: 'testuser',
      password: 'Password1!',
      name: 'テストユーザー',
      role: 'user' as const,
    };

    it.each([
      ['missing @', 'testexample.com'],
      ['missing domain', 'test@'],
      ['missing local part', '@example.com'],
      ['spaces', 'test @example.com'],
      ['double dots', 'test@example..com'],
    ])('%s: "%s" を拒否する', (_, email) => {
      const result = createUserSchema.safeParse({ ...baseData, email });
      expect(result.success).toBe(false);
    });
  });

  // 異常系テスト: password
  describe('password validation', () => {
    const baseData = {
      username: 'testuser',
      email: 'test@example.com',
      name: 'テストユーザー',
      role: 'user' as const,
    };

    it('8文字未満のパスワードを拒否する', () => {
      const result = createUserSchema.safeParse({
        ...baseData,
        password: 'Pass1!',
      });
      expect(result.success).toBe(false);
    });

    it('大文字を含まないパスワードを拒否する', () => {
      const result = createUserSchema.safeParse({
        ...baseData,
        password: 'password1!',
      });
      expect(result.success).toBe(false);
    });

    it('小文字を含まないパスワードを拒否する', () => {
      const result = createUserSchema.safeParse({
        ...baseData,
        password: 'PASSWORD1!',
      });
      expect(result.success).toBe(false);
    });

    it('数字を含まないパスワードを拒否する', () => {
      const result = createUserSchema.safeParse({
        ...baseData,
        password: 'Password!@',
      });
      expect(result.success).toBe(false);
    });

    it('ユーザー名を含むパスワードを拒否する', () => {
      const result = createUserSchema.safeParse({
        ...baseData,
        username: 'testuser',
        password: 'Testuser1!',
      });
      expect(result.success).toBe(false);
      if (!result.success) {
        const errors = result.error.flatten();
        expect(errors.fieldErrors.password).toBeDefined();
      }
    });
  });

  // クロスフィールドバリデーション
  describe('cross-field validation', () => {
    it('パスワードにユーザー名が含まれている場合に拒否する', () => {
      const result = createUserSchema.safeParse({
        username: 'johndoe',
        email: 'john@example.com',
        password: 'Johndoe123!',
        name: 'John Doe',
      });
      expect(result.success).toBe(false);
    });
  });
});

// ===================================================================
// パスワード強度のテスト
// ===================================================================
import { getPasswordStrength } from '@/lib/password-strength';

describe('getPasswordStrength', () => {
  it('空文字列のスコアは0', () => {
    const result = getPasswordStrength('');
    expect(result.score).toBe(0);
  });

  it('短く単純なパスワードは低スコア', () => {
    const result = getPasswordStrength('abc');
    expect(result.score).toBeLessThanOrEqual(1);
  });

  it('よく使われるパスワードは低スコア', () => {
    const result = getPasswordStrength('password');
    expect(result.score).toBeLessThanOrEqual(1);
    expect(result.requirements.noCommonPattern).toBe(false);
  });

  it('十分に複雑なパスワードは高スコア', () => {
    const result = getPasswordStrength('MyStr0ng!P@ssw0rd2024');
    expect(result.score).toBeGreaterThanOrEqual(3);
  });

  it('全ての要件を満たす場合にrequirementsが全てtrue', () => {
    const result = getPasswordStrength('MyP@ssw0rd!');
    expect(result.requirements.minLength).toBe(true);
    expect(result.requirements.hasUppercase).toBe(true);
    expect(result.requirements.hasLowercase).toBe(true);
    expect(result.requirements.hasNumber).toBe(true);
    expect(result.requirements.hasSpecial).toBe(true);
  });
});
```

### 9.2 フォームコンポーネントの統合テスト

```typescript
// ===================================================================
// __tests__/components/ContactForm.test.tsx
// React Testing Libraryを使ったフォームテスト
// ===================================================================
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ContactForm } from '@/components/ContactForm';

// API モック
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('ContactForm', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    mockFetch.mockReset();
  });

  // ヘルパー: フォームを入力する
  async function fillForm(overrides: Partial<Record<string, string>> = {}) {
    const defaults = {
      lastName: '山田',
      firstName: '太郎',
      email: 'taro@example.com',
      message: 'これはテストメッセージです。10文字以上のメッセージ。',
      ...overrides,
    };

    if (defaults.lastName) {
      await user.type(screen.getByLabelText('姓'), defaults.lastName);
    }
    if (defaults.firstName) {
      await user.type(screen.getByLabelText('名'), defaults.firstName);
    }
    if (defaults.email) {
      await user.type(screen.getByLabelText('メールアドレス'), defaults.email);
    }
    if (defaults.message) {
      await user.type(screen.getByLabelText('メッセージ'), defaults.message);
    }

    // お問い合わせ種別を選択
    await user.selectOptions(screen.getByLabelText('お問い合わせ種別'), 'inquiry');

    // 利用規約に同意
    await user.click(screen.getByLabelText('利用規約に同意する'));
  }

  it('必須フィールドが空の場合にエラーメッセージを表示する', async () => {
    render(<ContactForm />);

    // 空のままSubmit
    await user.click(screen.getByRole('button', { name: '送信する' }));

    await waitFor(() => {
      expect(screen.getByText('姓を入力してください')).toBeInTheDocument();
      expect(screen.getByText('名を入力してください')).toBeInTheDocument();
      expect(screen.getByText('有効なメールアドレスを入力してください')).toBeInTheDocument();
    });
  });

  it('メールアドレスの形式が不正な場合にエラーを表示する', async () => {
    render(<ContactForm />);

    await user.type(screen.getByLabelText('メールアドレス'), 'invalid-email');
    await user.click(screen.getByRole('button', { name: '送信する' }));

    await waitFor(() => {
      expect(
        screen.getByText('有効なメールアドレスを入力してください')
      ).toBeInTheDocument();
    });
  });

  it('メッセージが10文字未満の場合にエラーを表示する', async () => {
    render(<ContactForm />);

    await user.type(screen.getByLabelText('メッセージ'), '短い');
    await user.click(screen.getByRole('button', { name: '送信する' }));

    await waitFor(() => {
      expect(
        screen.getByText('10文字以上で入力してください')
      ).toBeInTheDocument();
    });
  });

  it('正しい入力でフォームを送信できる', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true }),
    });

    render(<ContactForm />);
    await fillForm();
    await user.click(screen.getByRole('button', { name: '送信する' }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/contact', expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }));
    });
  });

  it('サーバーエラーをフォームに表示する', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      json: async () => ({
        fieldErrors: {
          email: ['このメールアドレスは既に使用されています'],
        },
      }),
    });

    render(<ContactForm />);
    await fillForm();
    await user.click(screen.getByRole('button', { name: '送信する' }));

    await waitFor(() => {
      expect(
        screen.getByText('このメールアドレスは既に使用されています')
      ).toBeInTheDocument();
    });
  });

  it('送信中はボタンが無効化される', async () => {
    mockFetch.mockImplementation(
      () => new Promise((resolve) => setTimeout(resolve, 1000))
    );

    render(<ContactForm />);
    await fillForm();
    await user.click(screen.getByRole('button', { name: '送信する' }));

    expect(screen.getByRole('button', { name: '送信中...' })).toBeDisabled();
  });

  it('エラー修正後にリアルタイムでエラーが消える', async () => {
    render(<ContactForm />);

    // まずSubmitしてエラーを表示
    await user.click(screen.getByRole('button', { name: '送信する' }));

    await waitFor(() => {
      expect(screen.getByText('姓を入力してください')).toBeInTheDocument();
    });

    // エラーのあるフィールドに入力
    await user.type(screen.getByLabelText('姓'), '山田');

    // エラーが消えることを確認（reValidateMode: 'onChange'）
    await waitFor(() => {
      expect(screen.queryByText('姓を入力してください')).not.toBeInTheDocument();
    });
  });

  // アクセシビリティテスト
  it('エラーフィールドにaria-invalid属性が設定される', async () => {
    render(<ContactForm />);
    await user.click(screen.getByRole('button', { name: '送信する' }));

    await waitFor(() => {
      expect(screen.getByLabelText('姓')).toHaveAttribute('aria-invalid', 'true');
    });
  });

  it('エラーメッセージにrole="alert"が設定されている', async () => {
    render(<ContactForm />);
    await user.click(screen.getByRole('button', { name: '送信する' }));

    await waitFor(() => {
      const alerts = screen.getAllByRole('alert');
      expect(alerts.length).toBeGreaterThan(0);
    });
  });
});
```

### 9.3 E2Eテスト（Playwright）

```typescript
// ===================================================================
// e2e/registration.spec.ts
// PlaywrightによるE2Eテスト
// ===================================================================
import { test, expect } from '@playwright/test';

test.describe('ユーザー登録フォーム', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/register');
  });

  test('正常な入力で登録が成功する', async ({ page }) => {
    // フォーム入力
    await page.fill('[name="username"]', 'newuser123');
    await page.fill('[name="name"]', 'テストユーザー');
    await page.fill('[name="email"]', `test-${Date.now()}@example.com`);
    await page.fill('[name="password"]', 'StrongP@ss1');

    // 送信
    await page.click('button[type="submit"]');

    // リダイレクトを確認
    await expect(page).toHaveURL('/login?registered=true');
  });

  test('バリデーションエラーが表示される', async ({ page }) => {
    // 空のままSubmit
    await page.click('button[type="submit"]');

    // エラーメッセージの表示を確認
    await expect(page.getByText('3文字以上で入力してください')).toBeVisible();
    await expect(page.getByText('名前を入力してください')).toBeVisible();
    await expect(page.getByText('有効なメールアドレスを入力してください')).toBeVisible();
  });

  test('パスワード強度インジケーターが動的に更新される', async ({ page }) => {
    const passwordField = page.locator('[name="password"]');

    // 弱いパスワード
    await passwordField.fill('abc');
    await expect(page.getByText('非常に弱い')).toBeVisible();

    // 中程度のパスワード
    await passwordField.fill('Password1');
    await expect(page.getByText(/普通|強い/)).toBeVisible();

    // 強いパスワード
    await passwordField.fill('MyStr0ng!P@ss');
    await expect(page.getByText(/強い|非常に強い/)).toBeVisible();
  });

  test('メールアドレスの重複チェックが機能する', async ({ page }) => {
    // 既存のメールアドレスを入力
    await page.fill('[name="email"]', 'existing@example.com');
    await page.locator('[name="email"]').blur();

    // 重複エラーの表示を確認（非同期バリデーション）
    await expect(
      page.getByText('このメールアドレスは既に使用されています')
    ).toBeVisible({ timeout: 5000 });
  });

  test('キーボードのみで操作できる', async ({ page }) => {
    // Tabキーで全フィールドにフォーカスが移動できることを確認
    await page.keyboard.press('Tab'); // username
    await expect(page.locator('[name="username"]')).toBeFocused();

    await page.keyboard.press('Tab'); // name
    await expect(page.locator('[name="name"]')).toBeFocused();

    await page.keyboard.press('Tab'); // email
    await expect(page.locator('[name="email"]')).toBeFocused();

    await page.keyboard.press('Tab'); // password
    await expect(page.locator('[name="password"]')).toBeFocused();
  });
});
```

---

## 10. アンチパターンとベストプラクティス

### 10.1 よくあるアンチパターン

#### アンチパターン1: クライアントのみのバリデーション

```typescript
// BAD: クライアントサイドのみでバリデーション
function BadForm() {
  const onSubmit = async (data: FormData) => {
    // サーバーに送信（バリデーションなし）
    await fetch('/api/users', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  };
}

// GOOD: サーバーサイドでも同じスキーマで検証
// server
export async function POST(request: Request) {
  const body = await request.json();
  const parsed = createUserSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ errors: parsed.error.flatten() }, { status: 422 });
  }
  // ... データベース操作
}
```

#### アンチパターン2: バリデーションロジックの重複

```typescript
// BAD: クライアントとサーバーで別々にバリデーションルールを定義
// client
const clientSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});

// server（別ファイルに同じルールを重複定義）
function validateOnServer(data: unknown) {
  if (typeof data.email !== 'string' || !data.email.includes('@')) {
    throw new Error('Invalid email');
  }
  if (typeof data.password !== 'string' || data.password.length < 8) {
    throw new Error('Invalid password');
  }
}

// GOOD: スキーマを共有ディレクトリに配置して再利用
// shared/schemas/auth.ts
export const authSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});

// client & server で同じスキーマを import
import { authSchema } from '@shared/schemas/auth';
```

#### アンチパターン3: エラーメッセージのハードコーディング

```typescript
// BAD: エラーメッセージを直接ハードコーディング
const schema = z.object({
  name: z.string().min(1, '名前は必須です'),  // 日本語固定
});

// GOOD: メッセージを外部化して国際化対応
const schema = z.object({
  name: z.string().min(1, getMessage('required', locale)),
});
```

#### アンチパターン4: onChangeモードの安易な使用

```typescript
// BAD: 全てのフォームでonChangeモード（パフォーマンス問題）
const form = useForm({
  resolver: zodResolver(schema),
  mode: 'onChange', // 毎回のキーストロークでバリデーション実行
});

// GOOD: onSubmit + onChangeの組み合わせ
const form = useForm({
  resolver: zodResolver(schema),
  mode: 'onSubmit',           // 初回はSubmit時
  reValidateMode: 'onChange', // エラー後はリアルタイム
});
```

#### アンチパターン5: 型安全性の欠如

```typescript
// BAD: any型の使用、型推論を活用しない
const onSubmit = async (data: any) => {
  await fetch('/api/users', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

// GOOD: Zodスキーマから型を推論
type FormData = z.infer<typeof userSchema>;

const onSubmit = async (data: FormData) => {
  // dataは自動的に型安全
  await createUser(data);
};
```

#### アンチパターン6: デバウンスなしの非同期バリデーション

```typescript
// BAD: キーストロークごとにAPIリクエスト
<input
  {...register('username', {
    validate: async (value) => {
      // 毎回APIを呼ぶ → サーバーに大量リクエスト
      const res = await fetch(`/api/check?username=${value}`);
      const data = await res.json();
      return data.available || 'このユーザー名は使用されています';
    },
  })}
/>

// GOOD: デバウンス + AbortControllerで制御
const { validate, isValidating } = useAsyncValidation(
  async (value) => {
    const res = await fetch(`/api/check?username=${value}`);
    const data = await res.json();
    return data.available || 'このユーザー名は使用されています';
  },
  500 // 500msのデバウンス
);
```

### 10.2 ベストプラクティスまとめ

| カテゴリ | ベストプラクティス | 理由 |
|---------|-------------------|------|
| **スキーマ設計** | 共有ディレクトリにスキーマを配置 | クライアント/サーバーでの再利用 |
| **スキーマ設計** | z.inferで型を自動推論 | 型の二重定義を防止 |
| **スキーマ設計** | baseSchema + extend/pick/omitで派生 | DRY原則の遵守 |
| **バリデーション** | mode: 'onSubmit' + reValidateMode: 'onChange' | UXとパフォーマンスのバランス |
| **バリデーション** | 非同期バリデーションにはデバウンスを適用 | サーバー負荷の軽減 |
| **バリデーション** | サーバーでも必ずバリデーションを実行 | セキュリティの担保 |
| **エラー表示** | aria-invalid, aria-describedbyの使用 | アクセシビリティ対応 |
| **エラー表示** | Submit失敗時にエラーフィールドへフォーカス | UX向上 |
| **テスト** | スキーマの境界値テストを網羅 | バリデーションロジックの信頼性 |
| **テスト** | 統合テストでフォームの挙動を検証 | ユーザー体験の品質保証 |
| **国際化** | エラーメッセージを外部化 | 多言語対応の容易さ |
| **パフォーマンス** | 大規模フォームはFormProviderで分割 | レンダリング最適化 |

---

## 11. トラブルシューティング

### 11.1 よくある問題と解決策

#### 問題1: zodResolverでバリデーションが機能しない

```typescript
// 症状: フォームを送信してもバリデーションエラーが表示されない

// 原因1: resolverの設定漏れ
// BAD
const form = useForm<FormData>({
  // resolverを設定していない
});

// GOOD
const form = useForm<FormData>({
  resolver: zodResolver(schema), // 必須
});

// 原因2: スキーマとデフォルト値の型不一致
// BAD: undefinedとstringの不整合
const schema = z.object({
  name: z.string().min(1),
});
const form = useForm({
  resolver: zodResolver(schema),
  defaultValues: {
    name: undefined, // stringが期待されているのにundefined
  },
});

// GOOD
const form = useForm({
  resolver: zodResolver(schema),
  defaultValues: {
    name: '', // 空文字列で初期化
  },
});
```

#### 問題2: refineのエラーが表示されない

```typescript
// 症状: refineで追加したバリデーションエラーが画面に表示されない

// 原因: pathの指定漏れ
// BAD
const schema = z.object({
  password: z.string(),
  confirmPassword: z.string(),
}).refine(
  (data) => data.password === data.confirmPassword,
  'パスワードが一致しません'  // pathが指定されていない → rootエラー扱い
);

// GOOD
const schema = z.object({
  password: z.string(),
  confirmPassword: z.string(),
}).refine(
  (data) => data.password === data.confirmPassword,
  {
    message: 'パスワードが一致しません',
    path: ['confirmPassword'], // エラーを関連付けるフィールドを指定
  }
);

// rootエラーを表示する場合
{errors.root && (
  <div role="alert">{errors.root.message}</div>
)}
```

#### 問題3: selectやcheckboxのバリデーションが動作しない

```typescript
// 症状: select要素やcheckboxでバリデーションエラーにならない

// 原因1: select - 空のoption valueの扱い
// BAD: value=""はZodでは空文字列(string)として扱われる
const schema = z.object({
  category: z.string().min(1, 'カテゴリを選択してください'),
});

// GOOD: enumを使用する
const schema = z.object({
  category: z.enum(['tech', 'design', 'business'], {
    errorMap: () => ({ message: 'カテゴリを選択してください' }),
  }),
});

// 原因2: checkbox - boolean vs literal
// BAD: z.boolean()はfalseも許可してしまう
const schema = z.object({
  agree: z.boolean(), // falseでも通る
});

// GOOD: z.literal(true)でtrueのみ許可
const schema = z.object({
  agree: z.literal(true, {
    errorMap: () => ({ message: '同意が必要です' }),
  }),
});

// defaultValuesの注意点
const form = useForm({
  resolver: zodResolver(schema),
  defaultValues: {
    agree: false as unknown as true, // 型アサーション必要
  },
});
```

#### 問題4: useFieldArrayでのバリデーション

```typescript
// 症状: 動的フィールドの配列バリデーションエラーが正しく表示されない

// 原因: エラーアクセスのパスが間違っている

// GOOD: 正しいアクセス方法
// 配列全体のバリデーションエラー（min, max, refine等）
{errors.contacts?.root?.message}

// 個別要素のバリデーションエラー
{errors.contacts?.[index]?.name?.message}

// 配列スキーマの定義例
const schema = z.object({
  contacts: z.array(
    z.object({
      name: z.string().min(1),
      email: z.string().email(),
    })
  )
  .min(1, '最低1件必要です')     // → errors.contacts?.root?.message
  .max(10, '最大10件までです'),   // → errors.contacts?.root?.message
});
```

#### 問題5: フォーム送信後にフォームがリセットされない

```typescript
// 症状: 送信成功後もフォームの値やエラー状態が残る

// GOOD: reset()を正しく使用する
const onSubmit = async (data: FormData) => {
  const result = await submitForm(data);

  if (result.success) {
    // フォーム全体をリセット
    form.reset();

    // または特定のフィールドだけリセット
    form.reset({
      name: '',
      email: data.email, // メールアドレスは保持
    });

    // デフォルト値に戻す場合
    form.reset(undefined, {
      keepDirtyValues: false,
      keepErrors: false,
    });
  }
};
```

#### 問題6: coerceの型変換が意図通りに動作しない

```typescript
// 症状: z.coerce.number()が意図しない値を返す

// 原因: 空文字列がNaN/0に変換される
const schema = z.object({
  age: z.coerce.number(), // '' → 0, 'abc' → NaN
});

// GOOD: preprocessで空文字列を処理してからcoerce
const schema = z.object({
  age: z.preprocess(
    (val) => {
      if (val === '' || val === null || val === undefined) return undefined;
      return val;
    },
    z.coerce.number().int().min(0).optional()
  ),
});

// または、transformでパース後に検証
const schema = z.object({
  age: z.string()
    .transform((val) => {
      if (val === '') return undefined;
      const num = Number(val);
      if (isNaN(num)) return undefined;
      return num;
    })
    .pipe(z.number().int().min(0).optional()),
});
```

### 11.2 パフォーマンス最適化

```typescript
// ===================================================================
// パフォーマンスに関する注意点と最適化手法
// ===================================================================

// 1. 大規模フォームのレンダリング最適化
// BAD: 全フィールドがwatchで再レンダリング
function BadForm() {
  const form = useForm<FormData>();
  const allValues = form.watch(); // 全フィールドの変更で再レンダリング

  return (
    <form>
      {/* 100個のフィールド... 全て再レンダリング */}
    </form>
  );
}

// GOOD: 必要なフィールドだけwatch
function GoodForm() {
  const form = useForm<FormData>();
  const password = form.watch('password'); // passwordのみ監視

  return (
    <form>
      <input {...form.register('password')} />
      <PasswordStrengthIndicator password={password} />
      {/* 他のフィールドは再レンダリングされない */}
    </form>
  );
}

// 2. useWatchを使った子コンポーネントの分離
function OptimizedPasswordField() {
  // このコンポーネントだけがpasswordの変更で再レンダリング
  const password = useWatch({ name: 'password' });

  return (
    <div>
      <input {...register('password')} />
      <PasswordStrengthIndicator password={password || ''} />
    </div>
  );
}

// 3. メモ化の活用
const MemoizedField = React.memo(function Field({
  name,
  label,
  error,
}: {
  name: string;
  label: string;
  error?: string;
}) {
  const { register } = useFormContext();

  return (
    <div>
      <label>{label}</label>
      <input {...register(name)} />
      {error && <span className="text-red-500">{error}</span>}
    </div>
  );
});

// 4. Zodスキーマのメモ化（動的スキーマの場合）
const useDynamicSchema = (locale: string) => {
  return useMemo(() => {
    return z.object({
      name: z.string().min(1, getMessage('required', locale)),
      email: z.string().email(getMessage('email', locale)),
    });
  }, [locale]); // localeが変わった時だけ再作成
};

// 5. 非同期バリデーションのキャッシュ
const validationCache = new Map<string, boolean>();

async function checkUsernameWithCache(username: string): Promise<string | true> {
  // キャッシュを確認
  if (validationCache.has(username)) {
    return validationCache.get(username) ? true : 'このユーザー名は使用されています';
  }

  const res = await fetch(`/api/check-username?username=${encodeURIComponent(username)}`);
  const data = await res.json();

  // 結果をキャッシュ（最大100件）
  if (validationCache.size > 100) {
    const firstKey = validationCache.keys().next().value;
    validationCache.delete(firstKey);
  }
  validationCache.set(username, data.available);

  return data.available ? true : 'このユーザー名は使用されています';
}
```

---

## 12. 実践的なフォームパターン集

### 12.1 マルチステップフォーム（ウィザード形式）

```typescript
// ===================================================================
// components/MultiStepForm.tsx
// ステップごとにバリデーションする大規模フォーム
// ===================================================================
import { useState } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

// 各ステップのスキーマを定義
const step1Schema = z.object({
  name: z.string().min(1, '名前を入力してください'),
  email: z.string().email('有効なメールアドレスを入力してください'),
  phone: z.string().regex(/^0\d{1,4}-?\d{1,4}-?\d{3,4}$/, '有効な電話番号を入力してください'),
});

const step2Schema = z.object({
  postalCode: z.string().regex(/^\d{3}-?\d{4}$/, '有効な郵便番号を入力してください'),
  prefecture: z.string().min(1, '都道府県を選択してください'),
  city: z.string().min(1, '市区町村を入力してください'),
  street: z.string().min(1, '番地を入力してください'),
  building: z.string().optional(),
});

const step3Schema = z.object({
  paymentMethod: z.enum(['credit_card', 'bank_transfer', 'convenience']),
  agreeToTerms: z.literal(true, {
    errorMap: () => ({ message: '利用規約に同意してください' }),
  }),
});

// 全体のスキーマ（最終送信時に使用）
const fullSchema = step1Schema.merge(step2Schema).merge(step3Schema);

type FullFormData = z.infer<typeof fullSchema>;

// ステップ定義
const steps = [
  { title: 'お客様情報', schema: step1Schema },
  { title: '配送先住所', schema: step2Schema },
  { title: 'お支払い・確認', schema: step3Schema },
] as const;

function MultiStepForm() {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());

  const methods = useForm<FullFormData>({
    resolver: zodResolver(fullSchema),
    mode: 'onSubmit',
    reValidateMode: 'onChange',
    defaultValues: {
      name: '',
      email: '',
      phone: '',
      postalCode: '',
      prefecture: '',
      city: '',
      street: '',
      building: '',
      paymentMethod: 'credit_card',
      agreeToTerms: false as unknown as true,
    },
  });

  // 現在のステップのフィールドだけバリデーション
  const validateCurrentStep = async (): Promise<boolean> => {
    const currentSchema = steps[currentStep].schema;
    const currentFields = Object.keys(currentSchema.shape) as Array<keyof FullFormData>;

    const isValid = await methods.trigger(currentFields);
    return isValid;
  };

  const handleNext = async () => {
    const isValid = await validateCurrentStep();
    if (isValid) {
      setCompletedSteps((prev) => new Set([...prev, currentStep]));
      setCurrentStep((prev) => Math.min(prev + 1, steps.length - 1));
    }
  };

  const handleBack = () => {
    setCurrentStep((prev) => Math.max(prev - 1, 0));
  };

  const handleStepClick = async (stepIndex: number) => {
    // 現在のステップより前のステップには自由に戻れる
    if (stepIndex < currentStep) {
      setCurrentStep(stepIndex);
      return;
    }

    // 現在のステップを検証してから先に進む
    if (stepIndex === currentStep + 1) {
      await handleNext();
    }
  };

  const onSubmit = async (data: FullFormData) => {
    console.log('Submitting:', data);
    // API送信処理
  };

  return (
    <FormProvider {...methods}>
      <div className="max-w-2xl mx-auto">
        {/* ステップインジケーター */}
        <nav aria-label="進捗" className="mb-8">
          <ol className="flex justify-between">
            {steps.map((step, index) => (
              <li key={index} className="flex items-center">
                <button
                  type="button"
                  onClick={() => handleStepClick(index)}
                  className={`flex items-center gap-2 ${
                    index === currentStep
                      ? 'text-blue-600 font-bold'
                      : completedSteps.has(index)
                      ? 'text-green-600'
                      : 'text-gray-400'
                  }`}
                  aria-current={index === currentStep ? 'step' : undefined}
                >
                  <span className={`w-8 h-8 rounded-full flex items-center justify-center border-2 ${
                    index === currentStep
                      ? 'border-blue-600 bg-blue-600 text-white'
                      : completedSteps.has(index)
                      ? 'border-green-500 bg-green-500 text-white'
                      : 'border-gray-300 text-gray-400'
                  }`}>
                    {completedSteps.has(index) ? '✓' : index + 1}
                  </span>
                  <span className="hidden sm:inline">{step.title}</span>
                </button>
              </li>
            ))}
          </ol>
        </nav>

        {/* ステップコンテンツ */}
        <form onSubmit={methods.handleSubmit(onSubmit)}>
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold mb-4">
              {steps[currentStep].title}
            </h2>

            {currentStep === 0 && <Step1Fields />}
            {currentStep === 1 && <Step2Fields />}
            {currentStep === 2 && <Step3Fields />}
          </div>

          {/* ナビゲーションボタン */}
          <div className="flex justify-between mt-6">
            <button
              type="button"
              onClick={handleBack}
              disabled={currentStep === 0}
              className="px-6 py-2 border rounded disabled:opacity-50"
            >
              戻る
            </button>

            {currentStep < steps.length - 1 ? (
              <button
                type="button"
                onClick={handleNext}
                className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                次へ
              </button>
            ) : (
              <button
                type="submit"
                disabled={methods.formState.isSubmitting}
                className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
              >
                {methods.formState.isSubmitting ? '送信中...' : '注文を確定する'}
              </button>
            )}
          </div>
        </form>
      </div>
    </FormProvider>
  );
}
```

### 12.2 インラインエディットパターン

```typescript
// ===================================================================
// components/InlineEdit.tsx
// テーブル内でフィールドをクリックして編集するパターン
// ===================================================================
import { useState, useRef, useEffect } from 'react';
import { z } from 'zod';

type InlineEditProps<T extends z.ZodType> = {
  value: string;
  schema: T;
  onSave: (newValue: z.infer<T>) => Promise<void>;
  displayComponent?: React.ReactNode;
  placeholder?: string;
};

function InlineEdit<T extends z.ZodType>({
  value,
  schema,
  onSave,
  displayComponent,
  placeholder = 'クリックして編集',
}: InlineEditProps<T>) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(value);
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleSave = async () => {
    // Zodでバリデーション
    const result = schema.safeParse(editValue);

    if (!result.success) {
      setError(result.error.errors[0]?.message || 'バリデーションエラー');
      return;
    }

    setIsSaving(true);
    setError(null);

    try {
      await onSave(result.data);
      setIsEditing(false);
    } catch (err) {
      setError('保存に失敗しました');
    } finally {
      setIsSaving(false);
    }
  };

  const handleCancel = () => {
    setEditValue(value);
    setError(null);
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      handleCancel();
    }
  };

  if (!isEditing) {
    return (
      <button
        onClick={() => setIsEditing(true)}
        className="group flex items-center gap-1 hover:bg-gray-50 px-2 py-1 rounded cursor-pointer"
        aria-label={`${value || placeholder}を編集`}
      >
        {displayComponent || (
          <span className={value ? 'text-gray-900' : 'text-gray-400'}>
            {value || placeholder}
          </span>
        )}
        <svg
          className="w-3.5 h-3.5 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity"
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
          />
        </svg>
      </button>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1">
        <input
          ref={inputRef}
          type="text"
          value={editValue}
          onChange={(e) => {
            setEditValue(e.target.value);
            setError(null);
          }}
          onKeyDown={handleKeyDown}
          className={`w-full px-2 py-1 border rounded text-sm ${
            error ? 'border-red-500' : 'border-blue-500'
          }`}
          disabled={isSaving}
          aria-invalid={!!error}
        />
        {error && (
          <p className="text-red-500 text-xs mt-0.5" role="alert">
            {error}
          </p>
        )}
      </div>
      <button
        onClick={handleSave}
        disabled={isSaving}
        className="text-green-600 hover:text-green-800"
        aria-label="保存"
      >
        {isSaving ? '...' : '✓'}
      </button>
      <button
        onClick={handleCancel}
        disabled={isSaving}
        className="text-red-600 hover:text-red-800"
        aria-label="キャンセル"
      >
        x
      </button>
    </div>
  );
}

// 使用例
function UserTable() {
  return (
    <table>
      <tbody>
        <tr>
          <td>
            <InlineEdit
              value={user.name}
              schema={z.string().min(1, '名前を入力してください').max(100)}
              onSave={async (newName) => {
                await updateUser({ name: newName });
              }}
            />
          </td>
          <td>
            <InlineEdit
              value={user.email}
              schema={z.string().email('有効なメールアドレスを入力してください')}
              onSave={async (newEmail) => {
                await updateUser({ email: newEmail });
              }}
            />
          </td>
        </tr>
      </tbody>
    </table>
  );
}
```

### 12.3 条件付きフィールドの表示・非表示パターン

```typescript
// ===================================================================
// 条件に応じてフィールドが増減するフォーム
// ===================================================================
const shippingSchema = z.discriminatedUnion('shippingType', [
  z.object({
    shippingType: z.literal('standard'),
    // 標準配送は追加フィールドなし
  }),
  z.object({
    shippingType: z.literal('express'),
    expressOption: z.enum(['next_day', 'same_day']),
    expressNote: z.string().max(200).optional(),
  }),
  z.object({
    shippingType: z.literal('pickup'),
    pickupLocation: z.string().min(1, '受取場所を選択してください'),
    pickupDate: z.coerce.date().min(new Date(), '過去の日付は指定できません'),
    pickupTime: z.string().regex(/^\d{2}:\d{2}$/, '受取時刻を入力してください'),
  }),
]);

type ShippingData = z.infer<typeof shippingSchema>;

function ShippingForm() {
  const form = useForm<ShippingData>({
    resolver: zodResolver(shippingSchema),
    defaultValues: {
      shippingType: 'standard',
    },
  });

  const shippingType = form.watch('shippingType');

  // 配送方法が変更されたらフォームをリセット
  useEffect(() => {
    form.clearErrors();
    // 現在のshippingType以外のフィールドをクリア
    if (shippingType === 'standard') {
      form.unregister(['expressOption', 'expressNote', 'pickupLocation', 'pickupDate', 'pickupTime']);
    } else if (shippingType === 'express') {
      form.unregister(['pickupLocation', 'pickupDate', 'pickupTime']);
    } else if (shippingType === 'pickup') {
      form.unregister(['expressOption', 'expressNote']);
    }
  }, [shippingType]);

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <fieldset>
        <legend>配送方法</legend>
        <label>
          <input type="radio" {...form.register('shippingType')} value="standard" />
          標準配送（3-5営業日）
        </label>
        <label>
          <input type="radio" {...form.register('shippingType')} value="express" />
          速達配送
        </label>
        <label>
          <input type="radio" {...form.register('shippingType')} value="pickup" />
          店舗受取
        </label>
      </fieldset>

      {/* 条件付きフィールド */}
      {shippingType === 'express' && (
        <div className="mt-4 p-4 bg-yellow-50 rounded">
          <h3>速達オプション</h3>
          <select {...form.register('expressOption')}>
            <option value="next_day">翌日配送</option>
            <option value="same_day">当日配送</option>
          </select>
          <textarea
            {...form.register('expressNote')}
            placeholder="備考（任意）"
            maxLength={200}
          />
        </div>
      )}

      {shippingType === 'pickup' && (
        <div className="mt-4 p-4 bg-blue-50 rounded">
          <h3>店舗受取情報</h3>
          <select {...form.register('pickupLocation')}>
            <option value="">受取場所を選択</option>
            <option value="tokyo">東京店</option>
            <option value="osaka">大阪店</option>
            <option value="nagoya">名古屋店</option>
          </select>
          {form.formState.errors.pickupLocation && (
            <span className="text-red-500">
              {form.formState.errors.pickupLocation.message}
            </span>
          )}
          <input type="date" {...form.register('pickupDate')} />
          <input type="time" {...form.register('pickupTime')} />
        </div>
      )}

      <button type="submit">確定</button>
    </form>
  );
}
```

---

## まとめ

### バリデーションパターンの全体像

| パターン | 用途 | 推奨度 |
|---------|------|-------|
| Zod + React Hook Form | 型安全なフォームバリデーション | 最も推奨 |
| 共有スキーマによる二重検証 | クライアント + サーバーで同じスキーマ | 必須 |
| Discriminated Union | 条件付きフィールドのバリデーション | 状況に応じて |
| 非同期バリデーション + デバウンス | メール重複チェック等 | 推奨 |
| mode: 'onSubmit' + reValidateMode: 'onChange' | 最適なUXバランス | 最も推奨 |
| useFieldArray | 動的なフィールド配列の管理 | 状況に応じて |
| FormProvider + useFormContext | 大規模フォームのコンポーネント分割 | 推奨 |
| パスワード強度インジケーター | パスワードの品質フィードバック | 推奨 |
| エラーサマリー | Submit後のエラー一覧表示 | 推奨 |
| マルチステップフォーム | 大量フィールドの段階的入力 | 状況に応じて |
| インラインエディット | テーブル内の直接編集 | 状況に応じて |
| 条件付きフィールド表示 | Discriminated Unionによる動的UI | 状況に応じて |
| 国際化対応エラーメッセージ | 多言語サポート | 状況に応じて |

### 設計指針

1. **スキーマファースト**: 先にZodスキーマを定義し、そこから型とバリデーションルールを導出する
2. **シングルソースオブトゥルース**: バリデーションルールは一か所で定義し、クライアント・サーバーで共有する
3. **段階的なフィードバック**: 初回は控えめに、エラー発生後は積極的にフィードバックする
4. **アクセシビリティファースト**: WAI-ARIA属性を正しく使い、キーボード操作とスクリーンリーダーに対応する
5. **防御的プログラミング**: クライアントバリデーションはUXのため、サーバーバリデーションはセキュリティのため
6. **テスト駆動**: スキーマの境界値テスト、フォームの統合テスト、E2Eテストを段階的に整備する

---

## よくある質問（FAQ）

### Q1. クライアント側とサーバー側のバリデーション、どう使い分けるべきですか？

**A:** **両方必須** というのが大原則である。それぞれの役割は以下の通り:

**クライアント側バリデーション（JavaScript/Zod）:**

- **役割**: UX向上のため
- **目的**: ユーザーに即座にフィードバックを提供し、無駄なサーバーリクエストを防ぐ
- **信頼性**: **信頼してはならない**（ブラウザの開発者ツールで簡単にバイパス可能）

実装例:

```typescript
// クライアント側（React Hook Form + Zod）
const schema = z.object({
  email: z.string().email('有効なメールアドレスを入力してください'),
  age: z.number().min(18, '18歳以上である必要があります'),
});

const form = useForm({ resolver: zodResolver(schema) });
```

**サーバー側バリデーション（API/バックエンド）:**

- **役割**: セキュリティとデータ整合性の保証
- **目的**: 悪意のあるリクエストや不正なデータからシステムを守る
- **信頼性**: **唯一信頼できるバリデーション**

実装例:

```typescript
// サーバー側（Next.js Server Action）
'use server';

export async function createUser(formData: FormData) {
  // サーバー側で必ず再バリデーション
  const result = schema.safeParse({
    email: formData.get('email'),
    age: Number(formData.get('age')),
  });

  if (!result.success) {
    return { errors: result.error.flatten() };
  }

  // データベースに保存前に追加のビジネスルール検証
  const emailExists = await db.user.findUnique({ where: { email: result.data.email } });
  if (emailExists) {
    return { errors: { email: 'このメールアドレスは既に使用されています' } };
  }

  // 保存処理
  await db.user.create({ data: result.data });
}
```

**ベストプラクティス:**

1. **スキーマを共有**: クライアントとサーバーで同じZodスキーマを使う（モノレポやパッケージ共有）
2. **段階的なバリデーション**:
   - クライアント: フォーマット検証（メール形式、必須フィールド等）
   - サーバー: フォーマット再検証 + ビジネスルール（重複チェック、権限チェック等）
3. **エラーメッセージの統一**: クライアントとサーバーで同じメッセージを返す

### Q2. Zod と yup、どちらを選ぶべきですか？

**A:** 現在のプロジェクトでは **Zod を強く推奨** する:

| 項目 | Zod | yup |
|------|-----|-----|
| TypeScript対応 | TypeScript-first、型推論が完璧 | JavaScriptベース、型定義は後付け |
| バンドルサイズ | 8KB（gzip） | 13KB（gzip） |
| パフォーマンス | 高速 | やや遅い |
| エラーメッセージ | カスタマイズ容易 | カスタマイズやや複雑 |
| エコシステム | Next.js、tRPC、Prismaなど最新ツールと統合 | Formikとの統合が強い |
| メンテナンス | 活発 | 活発だが成長鈍化 |

**Zodが優れている点:**

```typescript
// 型推論が完璧
const userSchema = z.object({
  name: z.string(),
  age: z.number(),
});

type User = z.infer<typeof userSchema>;
// → { name: string; age: number } が自動推論される
```

**yupが優れている点:**

- Formikとの統合が歴史的に強い（Formik公式ドキュメントでyupを推奨）
- 学習リソースが豊富（歴史が長い）

**移行は簡単か？**

Zod と yup は API が似ているため、移行は比較的容易:

```typescript
// yup
const schema = yup.object({
  email: yup.string().email().required(),
});

// Zod
const schema = z.object({
  email: z.string().email(),
});
```

### Q3. 非同期バリデーション（重複チェック等）はどう実装すべきですか？

**A:** 非同期バリデーションは **API呼び出しが必要な検証**（メールアドレス重複チェック、ユーザー名使用可否チェック等）で使用する。

**パターン1: Zod の `.refine()` で非同期チェック**

```typescript
const emailSchema = z.string().email().refine(
  async (email) => {
    // APIで重複チェック
    const response = await fetch(`/api/check-email?email=${email}`);
    const { available } = await response.json();
    return available;
  },
  { message: 'このメールアドレスは既に使用されています' }
);
```

**パターン2: React Hook Form の `validate` オプション**

```typescript
const form = useForm();

<input
  {...form.register('email', {
    validate: async (value) => {
      const response = await fetch(`/api/check-email?email=${value}`);
      const { available } = await response.json();
      return available || 'このメールアドレスは既に使用されています';
    }
  })}
/>
```

**パターン3: カスタムフックで防御的に実装（推奨）**

```typescript
function useEmailValidation() {
  const [isChecking, setIsChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timeoutRef = useRef<NodeJS.Timeout>();

  const validateEmail = useCallback(async (email: string) => {
    // デバウンス: 連続入力中はAPIを叩かない
    clearTimeout(timeoutRef.current);

    return new Promise<boolean>((resolve) => {
      timeoutRef.current = setTimeout(async () => {
        setIsChecking(true);
        try {
          const response = await fetch(`/api/check-email?email=${email}`);
          const { available } = await response.json();

          if (!available) {
            setError('このメールアドレスは既に使用されています');
            resolve(false);
          } else {
            setError(null);
            resolve(true);
          }
        } catch (err) {
          setError('確認中にエラーが発生しました');
          resolve(false);
        } finally {
          setIsChecking(false);
        }
      }, 500); // 500msデバウンス
    });
  }, []);

  return { validateEmail, isChecking, error };
}

// 使用例
function EmailField() {
  const form = useForm();
  const { validateEmail, isChecking, error } = useEmailValidation();

  return (
    <div>
      <input
        {...form.register('email', {
          validate: validateEmail,
        })}
      />
      {isChecking && <span>確認中...</span>}
      {error && <span>{error}</span>}
    </div>
  );
}
```

**ベストプラクティス:**

1. **デバウンスを必ず実装**: 連続入力中は不要なAPIリクエストを避ける（500ms推奨）
2. **ローディング状態を表示**: `isChecking` フラグでユーザーに処理中であることを伝える
3. **エラーハンドリング**: ネットワークエラーやタイムアウトに対処
4. **キャッシュを活用**: 同じ入力値を何度もチェックしない（React QueryやSWRの活用）
5. **サーバー側で最終検証**: クライアント側の非同期チェックはUXのため、サーバー側で必ず再検証

**注意点:**

- onBlur（フォーカスアウト時）にトリガーするか、onChange（入力中）にトリガーするかを慎重に選ぶ
- onChangeで実装する場合、デバウンスは必須（でないとAPIが大量に叩かれる）
- フォーム送信時にサーバー側で再度検証することを忘れずに

---

## 次に読むべきガイド
→ [[02-file-upload.md]] --- ファイルアップロード

---

## 参考文献
1. React Hook Form. "Resolvers." react-hook-form.com, 2024.
2. Zod. "Documentation." zod.dev, 2024.
3. WAI-ARIA. "Forms Pattern." w3.org/WAI/ARIA/apg/patterns/forms, 2024.
4. MDN Web Docs. "Client-side form validation." developer.mozilla.org, 2024.
5. Colinhacks. "Zod: TypeScript-first schema validation." GitHub, 2024.
6. React Hook Form. "Advanced Usage - Wizard Form." react-hook-form.com, 2024.
7. Valibot. "Documentation." valibot.dev, 2024.
8. Web Content Accessibility Guidelines (WCAG) 2.2. "Understanding Success Criterion 3.3.1: Error Identification." w3.org, 2023.
