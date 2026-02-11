# 入力バリデーション

> 入力バリデーションはAPIセキュリティの最前線。Zodによる型安全なバリデーション、JSON Schema、サニタイゼーション、一般的な攻撃パターンへの防御まで、信頼境界での入力検証の全技法を習得する。

## この章で学ぶこと

- [ ] バリデーションの設計原則と信頼境界を理解する
- [ ] Zodによる型安全なバリデーション実装を把握する
- [ ] サニタイゼーションとインジェクション防止を学ぶ

---

## 1. バリデーションの原則

```
信頼境界（Trust Boundary）:
  → 外部入力は全て信頼しない
  → バリデーションは信頼境界で行う

  信頼できない入力:
  ✗ リクエストボディ
  ✗ クエリパラメータ
  ✗ パスパラメータ
  ✗ ヘッダー（Authorizationを含む）
  ✗ Cookie
  ✗ ファイルアップロード
  ✗ 外部APIのレスポンス

  バリデーションの種類:
  ① 型チェック: string, number, boolean, array, object
  ② 形式チェック: email, URL, UUID, 日時
  ③ 範囲チェック: min, max, minLength, maxLength
  ④ パターンチェック: 正規表現
  ⑤ ビジネスルール: 在庫数 > 0, 金額 > 0
  ⑥ 関連チェック: 開始日 < 終了日

  原則:
  ✓ Fail Fast（早期に失敗）
  ✓ 全エラーをまとめて返す（1つずつではなく）
  ✓ エラーメッセージは具体的に
  ✓ 内部エラーは外部に露出しない
```

---

## 2. Zod によるバリデーション

```typescript
// Zod: TypeScript-first のバリデーションライブラリ
import { z } from 'zod';

// --- スキーマ定義 ---
const CreateUserSchema = z.object({
  name: z.string()
    .min(1, 'Name is required')
    .max(100, 'Name must be 100 characters or less')
    .trim(),

  email: z.string()
    .email('Invalid email format')
    .toLowerCase(),

  age: z.number()
    .int('Age must be an integer')
    .min(0, 'Age must be non-negative')
    .max(150, 'Age must be 150 or less')
    .optional(),

  role: z.enum(['user', 'admin', 'editor'])
    .default('user'),

  tags: z.array(z.string().max(50))
    .max(10, 'Maximum 10 tags')
    .default([]),

  address: z.object({
    street: z.string().min(1),
    city: z.string().min(1),
    postalCode: z.string().regex(/^\d{3}-?\d{4}$/, 'Invalid postal code'),
  }).optional(),
});

// 型の自動推論
type CreateUserInput = z.infer<typeof CreateUserSchema>;
// {
//   name: string;
//   email: string;
//   age?: number;
//   role: 'user' | 'admin' | 'editor';
//   tags: string[];
//   address?: { street: string; city: string; postalCode: string };
// }

// --- バリデーション実行 ---
function validateInput(schema, data) {
  const result = schema.safeParse(data);

  if (!result.success) {
    const errors = result.error.issues.map(issue => ({
      field: issue.path.join('.'),
      message: issue.message,
      code: 'VALIDATION_ERROR',
    }));
    return { success: false, errors };
  }

  return { success: true, data: result.data };
}

// --- Express ミドルウェア ---
function validate(schema) {
  return (req, res, next) => {
    const result = schema.safeParse(req.body);

    if (!result.success) {
      return res.status(422).json({
        type: 'https://api.example.com/errors/validation',
        title: 'Validation Error',
        status: 422,
        detail: 'The request body contains invalid fields.',
        errors: result.error.issues.map(issue => ({
          field: issue.path.join('.'),
          code: issue.code,
          message: issue.message,
        })),
      });
    }

    req.validatedBody = result.data; // バリデーション済みデータ
    next();
  };
}

// 使用例
app.post('/api/v1/users', validate(CreateUserSchema), async (req, res) => {
  const user = await createUser(req.validatedBody); // 型安全
  res.status(201).json({ data: user });
});
```

---

## 3. 高度なバリデーション

```typescript
// カスタムバリデーション
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
  );

// 相互依存バリデーション
const DateRangeSchema = z.object({
  startDate: z.string().datetime(),
  endDate: z.string().datetime(),
}).refine(
  (data) => new Date(data.startDate) < new Date(data.endDate),
  { message: 'End date must be after start date', path: ['endDate'] }
);

// discriminatedUnion（条件分岐）
const NotificationSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('email'),
    email: z.string().email(),
    subject: z.string().min(1),
  }),
  z.object({
    type: z.literal('sms'),
    phone: z.string().regex(/^\+?\d{10,15}$/),
  }),
  z.object({
    type: z.literal('push'),
    deviceToken: z.string().min(1),
  }),
]);

// パスパラメータのバリデーション
const UserIdSchema = z.string().uuid('Invalid user ID format');
const PaginationSchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  perPage: z.coerce.number().int().min(1).max(100).default(20),
  sort: z.enum(['createdAt', 'name', 'email']).default('createdAt'),
  order: z.enum(['asc', 'desc']).default('desc'),
});
```

---

## 4. サニタイゼーション

```typescript
// 入力のサニタイゼーション（無害化）

// HTML エスケープ
function escapeHtml(str: string): string {
  const map: Record<string, string> = {
    '&': '&amp;', '<': '&lt;', '>': '&gt;',
    '"': '&quot;', "'": '&#x27;',
  };
  return str.replace(/[&<>"']/g, (c) => map[c]);
}

// Zod の transform でサニタイズ
const CommentSchema = z.object({
  content: z.string()
    .min(1)
    .max(10000)
    .transform(escapeHtml),    // HTML タグを無効化

  // SQL インジェクション対策はバリデーションではなく
  // パラメタライズドクエリで行う
});

// ファイルアップロードのバリデーション
const FileUploadSchema = z.object({
  filename: z.string()
    .max(255)
    .regex(/^[a-zA-Z0-9._-]+$/, 'Invalid filename characters'),
  mimeType: z.enum([
    'image/jpeg', 'image/png', 'image/webp', 'application/pdf',
  ]),
  size: z.number()
    .max(10 * 1024 * 1024, 'File must be under 10MB'),
});
```

---

## 5. 一般的な攻撃への防御

```
① SQL インジェクション:
  攻撃: email = "'; DROP TABLE users; --"

  防御:
  ✓ パラメタライズドクエリ（プリペアドステートメント）
  ✗ 文字列結合でSQLを構築しない

  // NG
  db.query(`SELECT * FROM users WHERE email = '${email}'`);

  // OK
  db.query('SELECT * FROM users WHERE email = $1', [email]);

② XSS（クロスサイトスクリプティング）:
  攻撃: name = "<script>alert('xss')</script>"

  防御:
  ✓ 出力時のエスケープ（HTMLエンコード）
  ✓ Content-Type: application/json（HTMLとして解釈しない）
  ✓ CSP ヘッダー

③ パストラバーサル:
  攻撃: filename = "../../etc/passwd"

  防御:
  ✓ ファイル名のバリデーション（英数字とドットのみ）
  ✓ パスの正規化後にベースディレクトリ内か確認

④ Mass Assignment:
  攻撃: { "name": "Taro", "role": "admin", "isVerified": true }

  防御:
  ✓ 許可するフィールドをホワイトリストで定義
  ✓ Zod スキーマに定義されたフィールドのみ受け入れ
  → .strict() で未定義フィールドを拒否

⑤ ReDoS（正規表現DoS）:
  攻撃: 悪意のある入力でバックトラッキング爆発

  防御:
  ✓ 入力の長さを先に制限
  ✓ 正規表現の複雑度を制限
  ✓ タイムアウトの設定

⑥ JSONペイロード攻撃:
  攻撃: 巨大なJSONオブジェクト、深いネスト

  防御:
  ✓ ボディサイズ制限: app.use(express.json({ limit: '1mb' }))
  ✓ ネスト深度の制限
  ✓ 配列サイズの制限
```

---

## 6. バリデーションエラーレスポンス

```
一貫したエラーレスポンス設計:

  422 Unprocessable Entity
  {
    "type": "https://api.example.com/errors/validation",
    "title": "Validation Error",
    "status": 422,
    "detail": "The request body contains 3 invalid fields.",
    "errors": [
      {
        "field": "email",
        "code": "invalid_string",
        "message": "Invalid email format"
      },
      {
        "field": "age",
        "code": "too_small",
        "message": "Age must be non-negative"
      },
      {
        "field": "address.postalCode",
        "code": "invalid_string",
        "message": "Invalid postal code"
      }
    ],
    "requestId": "req_abc123"
  }

ポイント:
  ✓ 全エラーをまとめて返す（1つずつではなく）
  ✓ field でどのフィールドか明示
  ✓ code で機械的に判定可能
  ✓ message でユーザーに表示可能
  ✓ ネストしたフィールドはドット区切り
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 信頼境界 | 外部入力は全て信頼しない |
| Zod | TypeScript-first、型推論、safeParse |
| サニタイゼーション | HTML エスケープ、ファイル名検証 |
| SQLi防御 | パラメタライズドクエリ |
| Mass Assignment | ホワイトリスト + .strict() |
| エラー | 全エラーをまとめて422で返す |

---

## 次に読むべきガイド
→ [[00-api-testing.md]] — APIテスト

---

## 参考文献
1. Zod. "TypeScript-first schema validation." github.com/colinhacks/zod, 2024.
2. OWASP. "Input Validation Cheat Sheet." cheatsheetseries.owasp.org, 2024.
3. OWASP. "API Security Top 10." owasp.org, 2023.
