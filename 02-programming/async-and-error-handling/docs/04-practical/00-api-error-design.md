# APIエラー設計

> APIのエラーレスポンスは、クライアント開発者の体験を左右する。HTTPステータスの正しい使い方、RFC 7807 Problem Details、エラーレスポンス設計のベストプラクティスを解説。

## この章で学ぶこと

- [ ] HTTPステータスコードの適切な使い分けを理解する
- [ ] エラーレスポンスの標準フォーマットを把握する
- [ ] 実践的なAPIエラー設計を学ぶ
- [ ] バリデーションエラーの設計パターンを習得する
- [ ] エラーの国際化（i18n）対応を理解する
- [ ] GraphQL/gRPC のエラー設計との比較を把握する

---

## 1. HTTPステータスコード

### 1.1 ステータスコードの分類

```
2xx 成功:
  200 OK               — 汎用的な成功
  201 Created           — リソース作成成功
  202 Accepted          — 非同期処理の受付完了
  204 No Content        — 成功（レスポンスボディなし）
  206 Partial Content   — 部分的なコンテンツ（Range指定）

3xx リダイレクト:
  301 Moved Permanently  — 恒久的なリダイレクト
  302 Found              — 一時的なリダイレクト
  304 Not Modified       — キャッシュ有効

4xx クライアントエラー:
  400 Bad Request       — リクエストが不正（構文エラー等）
  401 Unauthorized      — 認証が必要（未認証）
  403 Forbidden         — 認可されていない（権限不足）
  404 Not Found         — リソースが存在しない
  405 Method Not Allowed — HTTPメソッドが不正
  406 Not Acceptable    — Accept ヘッダーに対応不可
  408 Request Timeout   — リクエストタイムアウト
  409 Conflict          — 競合（重複登録、楽観的ロック失敗等）
  410 Gone              — リソースが恒久的に削除済み
  413 Payload Too Large — ペイロードサイズ超過
  415 Unsupported Media Type — Content-Type 未対応
  422 Unprocessable Entity — バリデーションエラー
  429 Too Many Requests — レート制限超過

5xx サーバーエラー:
  500 Internal Server Error — サーバー内部エラー
  501 Not Implemented       — 未実装のエンドポイント
  502 Bad Gateway           — 上流サーバーエラー
  503 Service Unavailable   — サービス一時停止
  504 Gateway Timeout       — 上流サーバータイムアウト

判断基準:
  クライアントのミス → 4xx
  サーバーの問題 → 5xx
  リトライで解決する可能性 → 429, 503, 504
```

### 1.2 よくある間違い

```
間違い1: 全てのエラーに 200 を返す
  ✗ Bad:
    HTTP 200 OK
    { "success": false, "error": "User not found" }

  ✓ Good:
    HTTP 404 Not Found
    { "type": "...", "title": "Not Found", "status": 404, "detail": "..." }

  理由: HTTPクライアント、CDN、プロキシ、モニタリングツールは
        ステータスコードに基づいて動作する

間違い2: 401 と 403 の混同
  401 Unauthorized = 認証されていない（ログインしていない）
    → WWW-Authenticate ヘッダーを返す
    → クライアントは認証情報を送り直す

  403 Forbidden = 認可されていない（権限がない）
    → 再認証しても結果は変わらない
    → 管理者に権限を依頼する

間違い3: 400 と 422 の混用
  400 Bad Request = リクエストの構文が不正
    → JSONが壊れている、必須パラメータがない
    → パースできないレベルの問題

  422 Unprocessable Entity = 構文は正しいが意味的に不正
    → メールアドレスのフォーマットが違う
    → 数値が範囲外
    → ビジネスルール違反

間違い4: 500 の乱用
  → 500 は「予期しないエラー」のみに使う
  → バリデーションエラーを 500 で返すのは誤り
  → 適切な 4xx コードを選ぶ

間違い5: 404 のセキュリティリスク
  → リソースの存在を確認できてしまう
  → 場合によっては 403 を返す（リソースの存在を隠す）
  → 例: /api/admin/users → 権限がなければ 403（404 ではなく）
```

### 1.3 ステータスコード選択フローチャート

```
リクエスト受信
  ├─ JSON パースできない? → 400 Bad Request
  ├─ 認証トークンがない/無効? → 401 Unauthorized
  ├─ 権限が不足? → 403 Forbidden
  ├─ リソースが見つからない? → 404 Not Found
  ├─ HTTPメソッドが不正? → 405 Method Not Allowed
  ├─ レート制限超過? → 429 Too Many Requests
  ├─ バリデーションエラー?
  │   ├─ 必須パラメータ欠如 → 400 Bad Request
  │   └─ 値の意味的不正 → 422 Unprocessable Entity
  ├─ 競合（重複、楽観的ロック失敗）? → 409 Conflict
  ├─ 処理成功?
  │   ├─ リソース作成 → 201 Created
  │   ├─ 非同期受付 → 202 Accepted
  │   ├─ レスポンスボディなし → 204 No Content
  │   └─ その他 → 200 OK
  └─ サーバー内部エラー → 500 Internal Server Error
```

---

## 2. エラーレスポンスフォーマット

### 2.1 RFC 7807 Problem Details

```json
// RFC 7807 Problem Details（推奨）
{
  "type": "https://api.example.com/errors/validation",
  "title": "Validation Error",
  "status": 422,
  "detail": "入力値に問題があります",
  "instance": "/api/users",
  "errors": [
    {
      "field": "email",
      "message": "有効なメールアドレスを入力してください"
    },
    {
      "field": "password",
      "message": "8文字以上で入力してください"
    }
  ],
  "traceId": "abc-123-def"
}
```

```
RFC 7807 の各フィールド:

  type（必須）:
    → エラーの種類を識別するURI
    → ドキュメントページのURLにすると便利
    → 例: "https://api.example.com/errors/validation"
    → デフォルト: "about:blank"

  title（必須）:
    → 人間可読なエラータイトル
    → type に対応する短い説明
    → 例: "Validation Error"

  status（推奨）:
    → HTTPステータスコード
    → レスポンスヘッダーと一致させる
    → 例: 422

  detail（推奨）:
    → エラーの詳細な説明
    → このリクエスト固有の情報
    → 例: "入力値に問題があります"

  instance（オプション）:
    → エラーが発生したリクエストのパス
    → デバッグに有用
    → 例: "/api/users"

  拡張フィールド（オプション）:
    → RFC 7807 は拡張可能
    → errors, traceId, timestamp 等を追加可能
```

### 2.2 TypeScript 型定義と実装

```typescript
// エラーレスポンスの型定義
interface ApiError {
  type: string;          // エラーの種類（URL or コード）
  title: string;         // 人間可読なタイトル
  status: number;        // HTTPステータス
  detail: string;        // 詳細メッセージ
  instance?: string;     // リクエストパス
  traceId?: string;      // トレーシングID
  timestamp?: string;    // 発生時刻
  errors?: FieldError[]; // フィールドレベルのエラー
}

interface FieldError {
  field: string;
  message: string;
  code?: string;
  rejectedValue?: unknown;
}

// アプリケーションエラーの基底クラス
class AppError extends Error {
  constructor(
    public readonly code: string,
    public readonly statusCode: number,
    message: string,
    public readonly details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = this.constructor.name;
    Error.captureStackTrace(this, this.constructor);
  }
}

// 具体的なエラークラス
class NotFoundError extends AppError {
  constructor(resource: string, id: string) {
    super('NOT_FOUND', 404, `${resource} with id '${id}' was not found`, {
      resource,
      id,
    });
  }
}

class ValidationError extends AppError {
  constructor(
    public readonly fields: FieldError[],
    message: string = '入力値に問題があります',
  ) {
    super('VALIDATION_ERROR', 422, message);
  }
}

class ConflictError extends AppError {
  constructor(resource: string, conflict: string) {
    super('CONFLICT', 409, `${resource}: ${conflict}`, {
      resource,
      conflict,
    });
  }
}

class UnauthorizedError extends AppError {
  constructor(message: string = '認証が必要です') {
    super('UNAUTHORIZED', 401, message);
  }
}

class ForbiddenError extends AppError {
  constructor(message: string = 'この操作を行う権限がありません') {
    super('FORBIDDEN', 403, message);
  }
}

class RateLimitError extends AppError {
  constructor(
    public readonly retryAfterSeconds: number,
    message: string = 'リクエスト制限を超過しました',
  ) {
    super('RATE_LIMIT_EXCEEDED', 429, message, { retryAfterSeconds });
  }
}

class InternalError extends AppError {
  constructor(
    message: string = 'サーバーエラーが発生しました',
    public readonly cause?: Error,
  ) {
    super('INTERNAL_ERROR', 500, message);
  }
}
```

### 2.3 Express エラーミドルウェア

```typescript
// Express ミドルウェア
function errorHandler(
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction,
): void {
  const traceId = req.headers['x-trace-id'] as string
    ?? req.headers['x-request-id'] as string
    ?? crypto.randomUUID();

  if (err instanceof AppError) {
    const response: ApiError = {
      type: `https://api.example.com/errors/${err.code.toLowerCase()}`,
      title: err.code.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      status: err.statusCode,
      detail: err.message,
      instance: req.originalUrl,
      traceId,
      timestamp: new Date().toISOString(),
    };

    // バリデーションエラーの場合はフィールド情報を追加
    if (err instanceof ValidationError) {
      response.errors = err.fields;
    }

    // レート制限の場合は Retry-After ヘッダーを追加
    if (err instanceof RateLimitError) {
      res.setHeader('Retry-After', String(err.retryAfterSeconds));
    }

    // ログ出力
    if (err.statusCode >= 500) {
      logger.error({ err, traceId, path: req.originalUrl }, 'Server error');
    } else if (err.statusCode >= 400) {
      logger.warn({ err, traceId, path: req.originalUrl }, 'Client error');
    }

    res.status(err.statusCode).json(response);
  } else {
    // 予期しないエラー（内部詳細を隠す）
    logger.error(
      { err, traceId, path: req.originalUrl, stack: err.stack },
      'Unexpected error',
    );

    res.status(500).json({
      type: 'https://api.example.com/errors/internal',
      title: 'Internal Server Error',
      status: 500,
      detail: 'サーバーエラーが発生しました',
      instance: req.originalUrl,
      traceId,
      timestamp: new Date().toISOString(),
    });
  }
}

// ミドルウェアの登録
app.use(errorHandler);

// 404 ハンドラー
app.use((req: Request, res: Response) => {
  res.status(404).json({
    type: 'https://api.example.com/errors/not_found',
    title: 'Not Found',
    status: 404,
    detail: `${req.method} ${req.originalUrl} は存在しません`,
    instance: req.originalUrl,
    timestamp: new Date().toISOString(),
  });
});
```

### 2.4 NestJS でのエラー設計

```typescript
// NestJS: Exception Filter
import {
  ExceptionFilter,
  Catch,
  ArgumentsHost,
  HttpException,
  HttpStatus,
} from '@nestjs/common';

@Catch()
export class GlobalExceptionFilter implements ExceptionFilter {
  constructor(private readonly logger: Logger) {}

  catch(exception: unknown, host: ArgumentsHost): void {
    const ctx = host.switchToHttp();
    const request = ctx.getRequest<Request>();
    const response = ctx.getResponse<Response>();

    const traceId = request.headers['x-trace-id'] as string
      ?? crypto.randomUUID();

    let status: number;
    let errorResponse: ApiError;

    if (exception instanceof AppError) {
      status = exception.statusCode;
      errorResponse = {
        type: `https://api.example.com/errors/${exception.code.toLowerCase()}`,
        title: exception.code,
        status,
        detail: exception.message,
        instance: request.url,
        traceId,
        timestamp: new Date().toISOString(),
      };

      if (exception instanceof ValidationError) {
        errorResponse.errors = exception.fields;
      }
    } else if (exception instanceof HttpException) {
      status = exception.getStatus();
      const exceptionResponse = exception.getResponse();
      errorResponse = {
        type: 'https://api.example.com/errors/http',
        title: HttpStatus[status] ?? 'Error',
        status,
        detail: typeof exceptionResponse === 'string'
          ? exceptionResponse
          : (exceptionResponse as any).message ?? 'エラーが発生しました',
        instance: request.url,
        traceId,
        timestamp: new Date().toISOString(),
      };
    } else {
      status = 500;
      this.logger.error(
        'Unexpected error',
        exception instanceof Error ? exception.stack : String(exception),
      );
      errorResponse = {
        type: 'https://api.example.com/errors/internal',
        title: 'Internal Server Error',
        status: 500,
        detail: 'サーバーエラーが発生しました',
        instance: request.url,
        traceId,
        timestamp: new Date().toISOString(),
      };
    }

    response.status(status).json(errorResponse);
  }
}
```

---

## 3. バリデーションエラーの設計

### 3.1 フィールドレベルのエラー

```typescript
// バリデーションエラーの詳細設計
interface DetailedFieldError {
  field: string;         // フィールドのパス（ネストもドット記法で）
  code: string;          // エラーコード（機械可読）
  message: string;       // 人間可読メッセージ
  rejectedValue?: unknown; // 拒否された値（セキュリティに注意）
  constraints?: Record<string, unknown>; // 制約条件
}

// 例: ユーザー登録のバリデーションエラー
const validationErrorExample: ApiError = {
  type: 'https://api.example.com/errors/validation',
  title: 'Validation Error',
  status: 422,
  detail: '3件のバリデーションエラーがあります',
  instance: '/api/users',
  traceId: 'trace-abc-123',
  timestamp: '2025-01-15T10:30:00Z',
  errors: [
    {
      field: 'email',
      code: 'INVALID_FORMAT',
      message: '有効なメールアドレスを入力してください',
      rejectedValue: 'invalid-email',
      constraints: { pattern: '^[^@]+@[^@]+\\.[^@]+$' },
    },
    {
      field: 'password',
      code: 'TOO_SHORT',
      message: '8文字以上で入力してください',
      constraints: { minLength: 8 },
    },
    {
      field: 'profile.age',
      code: 'OUT_OF_RANGE',
      message: '0以上130以下の値を入力してください',
      rejectedValue: -1,
      constraints: { min: 0, max: 130 },
    },
  ],
};
```

### 3.2 バリデーションライブラリとの統合

```typescript
// Zod との統合
import { z } from 'zod';

const CreateUserSchema = z.object({
  email: z.string().email('有効なメールアドレスを入力してください'),
  password: z.string().min(8, '8文字以上で入力してください'),
  name: z.string().min(1, '名前を入力してください').max(100),
  profile: z.object({
    age: z.number().int().min(0).max(130).optional(),
    bio: z.string().max(500).optional(),
  }).optional(),
});

// Zod エラーを API エラーに変換
function zodToFieldErrors(error: z.ZodError): FieldError[] {
  return error.errors.map((issue) => ({
    field: issue.path.join('.'),
    code: issue.code.toUpperCase(),
    message: issue.message,
  }));
}

// バリデーションミドルウェア
function validate<T>(schema: z.ZodSchema<T>) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req.body);
    if (!result.success) {
      throw new ValidationError(zodToFieldErrors(result.error));
    }
    req.body = result.data;
    next();
  };
}

// 使用例
app.post('/api/users', validate(CreateUserSchema), async (req, res) => {
  const user = await userService.create(req.body);
  res.status(201).json(user);
});
```

```python
# Python: Pydantic との統合
from pydantic import BaseModel, EmailStr, Field, validator
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)
    name: str = Field(min_length=1, max_length=100)
    age: int | None = Field(None, ge=0, le=130)

    @validator('password')
    def password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('大文字を1文字以上含めてください')
        if not any(c.isdigit() for c in v):
            raise ValueError('数字を1文字以上含めてください')
        return v


# Pydantic バリデーションエラーを RFC 7807 に変換
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = []
    for error in exc.errors():
        field_path = '.'.join(str(loc) for loc in error['loc'] if loc != 'body')
        errors.append({
            'field': field_path,
            'code': error['type'].upper(),
            'message': error['msg'],
        })

    return JSONResponse(
        status_code=422,
        content={
            'type': 'https://api.example.com/errors/validation',
            'title': 'Validation Error',
            'status': 422,
            'detail': f'{len(errors)}件のバリデーションエラーがあります',
            'instance': str(request.url.path),
            'errors': errors,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        },
    )


@app.post('/api/users', status_code=201)
async def create_user(user: CreateUserRequest):
    return await user_service.create(user)
```

### 3.3 ビジネスルールバリデーション

```typescript
// ビジネスルールのバリデーション
class OrderValidator {
  async validate(order: CreateOrderInput): Promise<FieldError[]> {
    const errors: FieldError[] = [];

    // 在庫チェック
    for (const item of order.items) {
      const stock = await this.stockService.getAvailable(item.productId);
      if (stock < item.quantity) {
        errors.push({
          field: `items[${item.productId}].quantity`,
          code: 'INSUFFICIENT_STOCK',
          message: `在庫が不足しています（残り${stock}個）`,
          rejectedValue: item.quantity,
          constraints: { available: stock },
        });
      }
    }

    // 注文金額チェック
    const total = order.items.reduce(
      (sum, item) => sum + item.price * item.quantity, 0,
    );
    if (total > 1_000_000) {
      errors.push({
        field: 'total',
        code: 'AMOUNT_EXCEEDS_LIMIT',
        message: '1回の注文は100万円以下にしてください',
        rejectedValue: total,
        constraints: { maxAmount: 1_000_000 },
      });
    }

    // 配送先チェック
    if (order.shippingAddress) {
      const isDeliverable = await this.shippingService.isDeliverable(
        order.shippingAddress.zipCode,
      );
      if (!isDeliverable) {
        errors.push({
          field: 'shippingAddress.zipCode',
          code: 'UNDELIVERABLE_AREA',
          message: 'この郵便番号への配送は対応していません',
          rejectedValue: order.shippingAddress.zipCode,
        });
      }
    }

    return errors;
  }
}

// コントローラーでの使用
app.post('/api/orders', async (req, res) => {
  // 構文バリデーション（Zod）
  const input = CreateOrderSchema.parse(req.body);

  // ビジネスルールバリデーション
  const validator = new OrderValidator();
  const errors = await validator.validate(input);

  if (errors.length > 0) {
    throw new ValidationError(errors);
  }

  const order = await orderService.create(input);
  res.status(201).json(order);
});
```

---

## 4. エラー設計のベストプラクティス

### 4.1 設計原則

```
1. 一貫性
   → 全エンドポイントで同じエラーフォーマット
   → ステータスコードの使い方を統一
   → Content-Type: application/problem+json（RFC 7807）

2. セキュリティ
   → 500エラーで内部情報を漏らさない
   → スタックトレースは本番では非表示
   → 「ユーザーが存在しない」vs「パスワードが違う」を区別しない
   → SQLエラーの詳細を返さない
   → 内部のクラス名やファイルパスを返さない

3. 機械可読性
   → エラーコードは文字列（enum対応）
   → HTTPステータスとエラーコードの組み合わせ
   → type フィールドでドキュメントへリンク

4. 人間可読性
   → detail フィールドで具体的なメッセージ
   → フィールドレベルのバリデーションエラー
   → エンドユーザーに表示可能なメッセージ

5. リトライ可能性の明示
   → 429: Retry-After ヘッダー
   → 503: Retry-After ヘッダー
   → エラーコードでリトライ判定可能に

6. デバッグ容易性
   → traceId でリクエストを追跡
   → timestamp で時系列を追跡
   → instance でエンドポイントを特定
```

### 4.2 エラーコード体系

```typescript
// エラーコードの体系的な設計
const ERROR_CODES = {
  // 認証・認可
  AUTH_TOKEN_EXPIRED: { status: 401, title: 'Token Expired' },
  AUTH_TOKEN_INVALID: { status: 401, title: 'Invalid Token' },
  AUTH_INSUFFICIENT_PERMISSIONS: { status: 403, title: 'Insufficient Permissions' },

  // バリデーション
  VALIDATION_FAILED: { status: 422, title: 'Validation Failed' },
  VALIDATION_REQUIRED_FIELD: { status: 422, title: 'Required Field Missing' },
  VALIDATION_INVALID_FORMAT: { status: 422, title: 'Invalid Format' },

  // リソース
  RESOURCE_NOT_FOUND: { status: 404, title: 'Resource Not Found' },
  RESOURCE_ALREADY_EXISTS: { status: 409, title: 'Resource Already Exists' },
  RESOURCE_CONFLICT: { status: 409, title: 'Resource Conflict' },
  RESOURCE_GONE: { status: 410, title: 'Resource Gone' },

  // レート制限
  RATE_LIMIT_EXCEEDED: { status: 429, title: 'Rate Limit Exceeded' },

  // ビジネスロジック
  BUSINESS_INSUFFICIENT_BALANCE: { status: 422, title: 'Insufficient Balance' },
  BUSINESS_ORDER_LIMIT_EXCEEDED: { status: 422, title: 'Order Limit Exceeded' },
  BUSINESS_ACCOUNT_SUSPENDED: { status: 403, title: 'Account Suspended' },

  // サーバーエラー
  INTERNAL_ERROR: { status: 500, title: 'Internal Server Error' },
  SERVICE_UNAVAILABLE: { status: 503, title: 'Service Unavailable' },
  UPSTREAM_ERROR: { status: 502, title: 'Upstream Service Error' },
} as const;

type ErrorCode = keyof typeof ERROR_CODES;

// エラーコードから ApiError を構築
function createApiError(
  code: ErrorCode,
  detail: string,
  extras?: Partial<ApiError>,
): ApiError {
  const { status, title } = ERROR_CODES[code];
  return {
    type: `https://api.example.com/errors/${code.toLowerCase()}`,
    title,
    status,
    detail,
    ...extras,
  };
}
```

### 4.3 セキュリティ考慮事項

```typescript
// セキュリティを考慮したエラーレスポンス

// 認証エラー: ユーザーの存在を漏らさない
app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;

  const user = await userService.findByEmail(email);

  // ✗ Bad: ユーザーの存在を漏らす
  // if (!user) throw new NotFoundError('User', email);
  // if (!bcrypt.compareSync(password, user.password)) throw new Error('Wrong password');

  // ✓ Good: 同じメッセージを返す
  if (!user || !await bcrypt.compare(password, user.passwordHash)) {
    throw new UnauthorizedError('メールアドレスまたはパスワードが正しくありません');
  }

  // タイミング攻撃への対策
  // ユーザーが見つからない場合もハッシュ比較を行う
  const dummyHash = '$2b$10$dummyhashfortimingattackprevention';
  if (!user) {
    await bcrypt.compare(password, dummyHash); // 処理時間を均一化
    throw new UnauthorizedError('メールアドレスまたはパスワードが正しくありません');
  }
});

// 500エラー: 内部情報を隠す
function sanitizeError(err: Error, isProduction: boolean): ApiError {
  if (isProduction) {
    return {
      type: 'https://api.example.com/errors/internal',
      title: 'Internal Server Error',
      status: 500,
      detail: 'サーバーエラーが発生しました。しばらくしてから再度お試しください。',
      // スタックトレース、SQLクエリ、ファイルパス等は含めない
    };
  }

  // 開発環境では詳細情報を返す
  return {
    type: 'https://api.example.com/errors/internal',
    title: 'Internal Server Error',
    status: 500,
    detail: err.message,
    // 開発環境でのみ追加情報を含める
    ...(isProduction ? {} : {
      stack: err.stack,
      cause: err.cause ? String(err.cause) : undefined,
    }),
  };
}

// レート制限エラーの情報開示
// ✗ Bad: レート制限の詳細を公開
// { "detail": "100 requests per minute exceeded. Current: 105" }

// ✓ Good: 必要最小限の情報
// { "detail": "リクエスト制限を超過しました", "retryAfter": 30 }
```

---

## 5. エラーの国際化（i18n）

### 5.1 多言語対応の設計

```typescript
// エラーメッセージの国際化

// メッセージカタログ
const errorMessages: Record<string, Record<string, string>> = {
  en: {
    'VALIDATION_FAILED': 'Validation failed',
    'VALIDATION_REQUIRED': '{field} is required',
    'VALIDATION_TOO_SHORT': '{field} must be at least {min} characters',
    'VALIDATION_TOO_LONG': '{field} must be at most {max} characters',
    'VALIDATION_INVALID_EMAIL': 'Please enter a valid email address',
    'NOT_FOUND': '{resource} not found',
    'UNAUTHORIZED': 'Authentication required',
    'FORBIDDEN': 'You do not have permission to perform this action',
    'RATE_LIMIT': 'Too many requests. Please try again later.',
    'INTERNAL_ERROR': 'An internal error occurred. Please try again later.',
  },
  ja: {
    'VALIDATION_FAILED': '入力値に問題があります',
    'VALIDATION_REQUIRED': '{field}は必須です',
    'VALIDATION_TOO_SHORT': '{field}は{min}文字以上で入力してください',
    'VALIDATION_TOO_LONG': '{field}は{max}文字以下で入力してください',
    'VALIDATION_INVALID_EMAIL': '有効なメールアドレスを入力してください',
    'NOT_FOUND': '{resource}が見つかりません',
    'UNAUTHORIZED': '認証が必要です',
    'FORBIDDEN': 'この操作を行う権限がありません',
    'RATE_LIMIT': 'リクエスト制限を超過しました。しばらくしてから再試行してください。',
    'INTERNAL_ERROR': 'サーバーエラーが発生しました。しばらくしてから再試行してください。',
  },
};

// フィールド名の翻訳
const fieldNames: Record<string, Record<string, string>> = {
  en: {
    'email': 'Email',
    'password': 'Password',
    'name': 'Name',
    'age': 'Age',
  },
  ja: {
    'email': 'メールアドレス',
    'password': 'パスワード',
    'name': '名前',
    'age': '年齢',
  },
};

// メッセージの解決
function resolveMessage(
  code: string,
  locale: string,
  params: Record<string, string | number> = {},
): string {
  const messages = errorMessages[locale] ?? errorMessages['en'];
  let template = messages[code] ?? messages['INTERNAL_ERROR'];

  // プレースホルダーを置換
  for (const [key, value] of Object.entries(params)) {
    template = template.replace(`{${key}}`, String(value));
  }

  return template;
}

// Accept-Language ヘッダーからロケールを決定
function getLocale(req: Request): string {
  const acceptLanguage = req.headers['accept-language'];
  if (!acceptLanguage) return 'en';

  // 簡易的なパース
  const preferred = acceptLanguage.split(',')[0].split(';')[0].trim().substring(0, 2);
  return errorMessages[preferred] ? preferred : 'en';
}

// i18n対応のエラーミドルウェア
function i18nErrorHandler(
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction,
): void {
  const locale = getLocale(req);

  if (err instanceof AppError) {
    const detail = resolveMessage(err.code, locale, err.details as any);

    const response: ApiError = {
      type: `https://api.example.com/errors/${err.code.toLowerCase()}`,
      title: err.code,
      status: err.statusCode,
      detail,
      instance: req.originalUrl,
    };

    if (err instanceof ValidationError) {
      response.errors = err.fields.map(field => ({
        ...field,
        message: resolveMessage(
          field.code ?? 'VALIDATION_FAILED',
          locale,
          {
            field: fieldNames[locale]?.[field.field] ?? field.field,
            ...field.constraints,
          } as any,
        ),
      }));
    }

    res.status(err.statusCode).json(response);
  } else {
    res.status(500).json({
      type: 'https://api.example.com/errors/internal',
      title: 'Internal Server Error',
      status: 500,
      detail: resolveMessage('INTERNAL_ERROR', locale),
    });
  }
}
```

---

## 6. クライアント側のエラーハンドリング

### 6.1 TypeScript HTTP クライアント

```typescript
// APIクライアントのエラーハンドリング
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async request<T>(
    path: string,
    options?: RequestInit,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Accept-Language': navigator.language,
          ...options?.headers,
        },
      });

      if (!response.ok) {
        const errorBody: ApiError = await response.json().catch(() => ({
          type: 'https://api.example.com/errors/unknown',
          title: 'Unknown Error',
          status: response.status,
          detail: response.statusText,
        }));

        throw new ApiRequestError(response.status, errorBody);
      }

      // 204 No Content の場合
      if (response.status === 204) {
        return undefined as T;
      }

      return response.json();
    } catch (error) {
      if (error instanceof ApiRequestError) {
        throw error;
      }

      // ネットワークエラー
      throw new NetworkError(
        'ネットワーク接続に問題があります',
        error as Error,
      );
    }
  }
}

// APIリクエストエラー
class ApiRequestError extends Error {
  constructor(
    public readonly statusCode: number,
    public readonly apiError: ApiError,
  ) {
    super(apiError.detail);
    this.name = 'ApiRequestError';
  }

  get isValidationError(): boolean {
    return this.statusCode === 422;
  }

  get isAuthError(): boolean {
    return this.statusCode === 401;
  }

  get isNotFound(): boolean {
    return this.statusCode === 404;
  }

  get isServerError(): boolean {
    return this.statusCode >= 500;
  }

  get isRetryable(): boolean {
    return [408, 429, 500, 502, 503, 504].includes(this.statusCode);
  }

  get fieldErrors(): FieldError[] {
    return this.apiError.errors ?? [];
  }
}

// React コンポーネントでの使用例
function UserRegistrationForm() {
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [globalError, setGlobalError] = useState<string | null>(null);

  async function handleSubmit(data: FormData) {
    try {
      setFieldErrors({});
      setGlobalError(null);

      await api.request('/api/users', {
        method: 'POST',
        body: JSON.stringify(data),
      });

      navigate('/registration-complete');
    } catch (error) {
      if (error instanceof ApiRequestError) {
        if (error.isValidationError) {
          // フィールドレベルのエラーをフォームに表示
          const errors: Record<string, string> = {};
          for (const fieldError of error.fieldErrors) {
            errors[fieldError.field] = fieldError.message;
          }
          setFieldErrors(errors);
        } else if (error.isAuthError) {
          navigate('/login');
        } else {
          setGlobalError(error.apiError.detail);
        }
      } else if (error instanceof NetworkError) {
        setGlobalError('ネットワーク接続を確認してください');
      } else {
        setGlobalError('予期しないエラーが発生しました');
      }
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <input name="email" />
      {fieldErrors.email && <span className="error">{fieldErrors.email}</span>}

      <input name="password" type="password" />
      {fieldErrors.password && <span className="error">{fieldErrors.password}</span>}

      {globalError && <div className="alert alert-error">{globalError}</div>}

      <button type="submit">登録</button>
    </form>
  );
}
```

---

## 7. GraphQL のエラー設計

### 7.1 GraphQL エラーの特性

```
GraphQL のエラーは REST とは異なる:

  REST:
    → HTTPステータスコードでエラーの種類を示す
    → ボディにエラー詳細を含める
    → 1リクエスト = 1レスポンス

  GraphQL:
    → 常に HTTP 200 を返す（クエリ自体は成功）
    → errors フィールドでエラーを返す
    → 部分的な成功が可能（data と errors が共存）
    → 1リクエストに複数のクエリを含められる
```

```typescript
// GraphQL エラーレスポンスの例
const graphqlErrorResponse = {
  data: {
    user: { id: '1', name: '田中太郎' },
    orders: null, // エラーで取得できなかった
  },
  errors: [
    {
      message: '注文情報の取得に失敗しました',
      locations: [{ line: 3, column: 5 }],
      path: ['orders'],
      extensions: {
        code: 'SERVICE_UNAVAILABLE',
        classification: 'ExecutionError',
        retryable: true,
      },
    },
  ],
};

// Apollo Server でのエラー定義
import { GraphQLError } from 'graphql';

class NotFoundGraphQLError extends GraphQLError {
  constructor(resource: string, id: string) {
    super(`${resource} with id '${id}' not found`, {
      extensions: {
        code: 'NOT_FOUND',
        resource,
        id,
        http: { status: 404 },
      },
    });
  }
}

class ValidationGraphQLError extends GraphQLError {
  constructor(errors: FieldError[]) {
    super('Validation failed', {
      extensions: {
        code: 'VALIDATION_ERROR',
        errors,
        http: { status: 422 },
      },
    });
  }
}

// Resolver でのエラー使用
const resolvers = {
  Query: {
    user: async (_, { id }) => {
      const user = await userService.findById(id);
      if (!user) {
        throw new NotFoundGraphQLError('User', id);
      }
      return user;
    },
  },
  Mutation: {
    createUser: async (_, { input }) => {
      const errors = await validator.validate(input);
      if (errors.length > 0) {
        throw new ValidationGraphQLError(errors);
      }
      return userService.create(input);
    },
  },
};
```

---

## 8. gRPC のエラー設計

### 8.1 gRPC ステータスコード

```
gRPC ステータスコード:
  OK (0)              — 成功
  CANCELLED (1)       — クライアントがキャンセル
  UNKNOWN (2)         — 不明なエラー
  INVALID_ARGUMENT (3) — 不正な引数
  DEADLINE_EXCEEDED (4) — デッドライン超過
  NOT_FOUND (5)       — リソースが存在しない
  ALREADY_EXISTS (6)   — リソースが既に存在
  PERMISSION_DENIED (7) — 権限なし
  RESOURCE_EXHAUSTED (8) — リソース枯渇
  FAILED_PRECONDITION (9) — 前提条件の不一致
  ABORTED (10)        — 操作が中断（トランザクション競合等）
  OUT_OF_RANGE (11)    — 範囲外
  UNIMPLEMENTED (12)   — 未実装
  INTERNAL (13)        — 内部エラー
  UNAVAILABLE (14)     — サービス利用不可
  DATA_LOSS (15)       — データ損失
  UNAUTHENTICATED (16) — 認証なし

HTTP ステータスとの対応:
  INVALID_ARGUMENT   ↔ 400 Bad Request
  UNAUTHENTICATED    ↔ 401 Unauthorized
  PERMISSION_DENIED  ↔ 403 Forbidden
  NOT_FOUND          ↔ 404 Not Found
  ALREADY_EXISTS     ↔ 409 Conflict
  RESOURCE_EXHAUSTED ↔ 429 Too Many Requests
  INTERNAL           ↔ 500 Internal Server Error
  UNAVAILABLE        ↔ 503 Service Unavailable
  DEADLINE_EXCEEDED  ↔ 504 Gateway Timeout
```

```protobuf
// gRPC エラー詳細（google.rpc.Status）
syntax = "proto3";

import "google/rpc/status.proto";
import "google/rpc/error_details.proto";

// エラー詳細を含むレスポンス
message ErrorResponse {
  google.rpc.Status status = 1;
}

// バリデーションエラーの詳細
// google.rpc.BadRequest を使用
message BadRequest {
  repeated FieldViolation field_violations = 1;

  message FieldViolation {
    string field = 1;
    string description = 2;
  }
}
```

```go
// Go: gRPC エラーの送信
import (
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
)

func (s *UserService) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
	user, err := s.repo.FindByID(ctx, req.Id)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to fetch user: %v", err)
	}
	if user == nil {
		return nil, status.Errorf(codes.NotFound, "user %s not found", req.Id)
	}
	return user, nil
}

func (s *UserService) CreateUser(ctx context.Context, req *pb.CreateUserRequest) (*pb.User, error) {
	// バリデーションエラーの詳細
	violations := validateCreateUser(req)
	if len(violations) > 0 {
		st := status.New(codes.InvalidArgument, "validation failed")
		br := &errdetails.BadRequest{
			FieldViolations: violations,
		}
		st, _ = st.WithDetails(br)
		return nil, st.Err()
	}

	return s.repo.Create(ctx, req)
}
```

---

## 9. エラードキュメントの自動生成

### 9.1 OpenAPI でのエラー定義

```yaml
# OpenAPI 3.0: エラーレスポンスの定義
openapi: "3.0.0"
info:
  title: Example API
  version: "1.0.0"

paths:
  /api/users:
    post:
      summary: ユーザー作成
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: ユーザー作成成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          $ref: '#/components/responses/BadRequest'
        '409':
          $ref: '#/components/responses/Conflict'
        '422':
          $ref: '#/components/responses/ValidationError'
        '500':
          $ref: '#/components/responses/InternalError'

components:
  schemas:
    ProblemDetail:
      type: object
      required: [type, title, status, detail]
      properties:
        type:
          type: string
          format: uri
          description: エラーの種類を識別するURI
          example: "https://api.example.com/errors/validation"
        title:
          type: string
          description: エラータイトル
          example: "Validation Error"
        status:
          type: integer
          description: HTTPステータスコード
          example: 422
        detail:
          type: string
          description: エラーの詳細
          example: "入力値に問題があります"
        instance:
          type: string
          description: リクエストパス
          example: "/api/users"
        traceId:
          type: string
          description: トレーシングID
          example: "abc-123-def"
        timestamp:
          type: string
          format: date-time
          description: エラー発生時刻
        errors:
          type: array
          items:
            $ref: '#/components/schemas/FieldError'

    FieldError:
      type: object
      required: [field, message]
      properties:
        field:
          type: string
          description: エラーのあるフィールド
          example: "email"
        code:
          type: string
          description: エラーコード
          example: "INVALID_FORMAT"
        message:
          type: string
          description: エラーメッセージ
          example: "有効なメールアドレスを入力してください"

  responses:
    BadRequest:
      description: リクエストが不正
      content:
        application/problem+json:
          schema:
            $ref: '#/components/schemas/ProblemDetail'
    ValidationError:
      description: バリデーションエラー
      content:
        application/problem+json:
          schema:
            $ref: '#/components/schemas/ProblemDetail'
    Conflict:
      description: リソース競合
      content:
        application/problem+json:
          schema:
            $ref: '#/components/schemas/ProblemDetail'
    InternalError:
      description: サーバーエラー
      content:
        application/problem+json:
          schema:
            $ref: '#/components/schemas/ProblemDetail'
```

---

## まとめ

| 原則 | ポイント |
|------|---------|
| ステータスコード | 正しいコードを選ぶ（4xx vs 5xx） |
| フォーマット | RFC 7807 Problem Details 準拠 |
| セキュリティ | 内部情報を漏らさない |
| 一貫性 | 全エンドポイントで統一 |
| バリデーション | フィールドレベルの詳細エラー |
| リトライ | Retry-After ヘッダー |
| 国際化 | Accept-Language 対応 |
| ドキュメント | OpenAPI でエラー定義 |
| エラーコード | 体系的なコード設計 |
| クライアントDX | 使いやすいエラーレスポンス |

---

## 次に読むべきガイド
→ [[01-logging-and-monitoring.md]] — ログとモニタリング

---

## 参考文献
1. RFC 7807. "Problem Details for HTTP APIs." IETF, 2016.
2. RFC 9457. "Problem Details for HTTP APIs." IETF, 2023.（RFC 7807 の後継）
3. Fielding, R. "REST APIs must be hypertext-driven." 2008.
4. Google Cloud API Design Guide. "Errors." cloud.google.com.
5. Microsoft REST API Guidelines. "Error Handling." github.com/microsoft.
6. GraphQL Specification. "Errors." spec.graphql.org.
7. gRPC Error Handling. "Status codes and their use." grpc.io.
8. Zalando RESTful API Guidelines. "Error Handling." opensource.zalando.com.
9. Stripe API Reference. "Errors." stripe.com/docs/api/errors.
10. Twitter API Documentation. "Error Handling." developer.twitter.com.
