# カスタムエラー設計

> エラーを適切にモデリングすることは、ソフトウェアの信頼性と保守性の基盤。エラーコード体系、ドメインエラー、エラーのシリアライズ手法を解説する。

## この章で学ぶこと

- [ ] カスタムエラーの設計原則を理解する
- [ ] エラーコード体系の作り方を把握する
- [ ] ドメイン駆動のエラー設計を学ぶ
- [ ] 各言語でのカスタムエラー実装パターンを習得する
- [ ] エラーのシリアライズとAPI設計を学ぶ
- [ ] エラーの国際化（i18n）対応を理解する

---

## 1. エラーの分類

### 1.1 操作エラー vs プログラマエラー

```
操作エラー（Operational Error）:
  → 予期される実行時エラー
  → 例: ネットワーク切断、DB接続失敗、バリデーション失敗
  → 対処: リトライ、フォールバック、ユーザーへ通知

プログラマエラー（Programmer Error）:
  → バグ。コードの修正が必要
  → 例: null参照、型エラー、配列の範囲外アクセス
  → 対処: クラッシュ → 修正 → デプロイ

この区別が重要:
  → 操作エラー: ハンドリングする（回復可能）
  → プログラマエラー: クラッシュさせる（回復不能）
```

### 1.2 エラーの詳細な分類体系

```
エラーの分類体系:

  1. クライアントエラー（4xx 系）
     → バリデーションエラー（400）
     → 認証エラー（401）
     → 認可エラー（403）
     → リソース未発見（404）
     → 競合エラー（409）
     → リクエスト制限（429）

  2. サーバーエラー（5xx 系）
     → 内部エラー（500）
     → 外部サービスエラー（502）
     → サービス利用不可（503）
     → タイムアウト（504）

  3. ビジネスロジックエラー
     → 残高不足
     → 注文キャンセル済み
     → 在庫切れ
     → 有効期限切れ
     → ポリシー違反

  4. インフラエラー
     → データベース接続エラー
     → メッセージキュー接続エラー
     → ファイルシステムエラー
     → メモリ不足

  それぞれの特性:
  ┌─────────────────┬────────┬──────────┬────────────┐
  │ 分類            │ 回復性 │ 通知先   │ ログレベル │
  ├─────────────────┼────────┼──────────┼────────────┤
  │ クライアント    │ 可能   │ ユーザー │ warn       │
  │ サーバー        │ 不可能 │ 開発者   │ error      │
  │ ビジネスロジック│ 可能   │ ユーザー │ warn       │
  │ インフラ        │ リトライ│ 運用     │ error      │
  └─────────────────┴────────┴──────────┴────────────┘
```

---

## 2. TypeScript でのカスタムエラー

### 2.1 基底エラークラス

```typescript
// 基底エラークラス
abstract class AppError extends Error {
  abstract readonly code: string;
  abstract readonly statusCode: number;
  readonly timestamp: Date;
  readonly isOperational: boolean;

  constructor(message: string, isOperational = true) {
    super(message);
    this.name = this.constructor.name;
    this.timestamp = new Date();
    this.isOperational = isOperational;
    Error.captureStackTrace(this, this.constructor);
  }

  toJSON() {
    return {
      error: {
        code: this.code,
        message: this.message,
        timestamp: this.timestamp.toISOString(),
      },
    };
  }
}
```

### 2.2 ドメインエラーの実装

```typescript
// ドメインエラー
class UserNotFoundError extends AppError {
  readonly code = "USER_NOT_FOUND";
  readonly statusCode = 404;

  constructor(public readonly userId: string) {
    super(`User not found: ${userId}`);
  }
}

class EmailAlreadyExistsError extends AppError {
  readonly code = "EMAIL_ALREADY_EXISTS";
  readonly statusCode = 409;

  constructor(public readonly email: string) {
    super(`Email already registered: ${email}`);
  }
}

class InsufficientBalanceError extends AppError {
  readonly code = "INSUFFICIENT_BALANCE";
  readonly statusCode = 400;

  constructor(
    public readonly required: number,
    public readonly available: number,
  ) {
    super(`Insufficient balance: required ${required}, available ${available}`);
  }
}

// バリデーションエラー（複数フィールド）
class ValidationError extends AppError {
  readonly code = "VALIDATION_ERROR";
  readonly statusCode = 400;

  constructor(
    public readonly errors: { field: string; message: string }[],
  ) {
    super(`Validation failed: ${errors.map(e => e.field).join(", ")}`);
  }

  toJSON() {
    return {
      error: {
        code: this.code,
        message: this.message,
        details: this.errors,
        timestamp: this.timestamp.toISOString(),
      },
    };
  }
}
```

### 2.3 完全なエラー階層の設計例

```typescript
// ========== 実務的な完全なエラー階層 ==========

// 基底クラス
abstract class AppError extends Error {
    abstract readonly code: string;
    abstract readonly httpStatus: number;
    readonly timestamp: string;
    readonly correlationId?: string;

    constructor(
        message: string,
        public readonly isOperational: boolean = true,
        options?: { cause?: Error; correlationId?: string }
    ) {
        super(message, { cause: options?.cause });
        this.name = this.constructor.name;
        this.timestamp = new Date().toISOString();
        this.correlationId = options?.correlationId;
        Error.captureStackTrace(this, this.constructor);
    }

    // APIレスポンス用のシリアライズ
    toResponse(): ErrorResponse {
        return {
            error: {
                code: this.code,
                message: this.message,
                timestamp: this.timestamp,
                ...(this.correlationId && { correlationId: this.correlationId }),
            }
        };
    }

    // ログ用のシリアライズ（内部情報を含む）
    toLog(): Record<string, unknown> {
        return {
            type: this.name,
            code: this.code,
            message: this.message,
            httpStatus: this.httpStatus,
            isOperational: this.isOperational,
            timestamp: this.timestamp,
            correlationId: this.correlationId,
            stack: this.stack,
            cause: this.cause instanceof Error ? {
                type: this.cause.name,
                message: this.cause.message,
            } : undefined,
        };
    }
}

// ---------- 認証系 ----------
class AuthenticationError extends AppError {
    readonly code = "AUTHENTICATION_REQUIRED";
    readonly httpStatus = 401;
    constructor(message = "認証が必要です", options?: { cause?: Error }) {
        super(message, true, options);
    }
}

class TokenExpiredError extends AppError {
    readonly code = "TOKEN_EXPIRED";
    readonly httpStatus = 401;
    constructor(public readonly expiredAt: Date) {
        super(`トークンが期限切れです（${expiredAt.toISOString()}）`);
    }
}

class InvalidTokenError extends AppError {
    readonly code = "INVALID_TOKEN";
    readonly httpStatus = 401;
    constructor(public readonly reason: string) {
        super(`無効なトークンです: ${reason}`);
    }
}

class AuthorizationError extends AppError {
    readonly code = "FORBIDDEN";
    readonly httpStatus = 403;
    constructor(
        public readonly requiredPermission: string,
        public readonly actualPermissions: string[] = [],
    ) {
        super(`権限が不足しています: ${requiredPermission} が必要です`);
    }
}

// ---------- リソース系 ----------
class NotFoundError extends AppError {
    readonly code = "NOT_FOUND";
    readonly httpStatus = 404;
    constructor(
        public readonly resourceType: string,
        public readonly resourceId: string,
    ) {
        super(`${resourceType} が見つかりません: ${resourceId}`);
    }
}

class ConflictError extends AppError {
    readonly code = "CONFLICT";
    readonly httpStatus = 409;
    constructor(
        public readonly resourceType: string,
        public readonly conflictField: string,
        public readonly conflictValue: string,
    ) {
        super(`${resourceType} の ${conflictField} が重複しています: ${conflictValue}`);
    }
}

class GoneError extends AppError {
    readonly code = "GONE";
    readonly httpStatus = 410;
    constructor(
        public readonly resourceType: string,
        public readonly resourceId: string,
        public readonly deletedAt: Date,
    ) {
        super(`${resourceType} ${resourceId} は削除されました（${deletedAt.toISOString()}）`);
    }
}

// ---------- バリデーション系 ----------
interface FieldError {
    field: string;
    message: string;
    code: string;
    value?: unknown;
    constraints?: Record<string, unknown>;
}

class ValidationError extends AppError {
    readonly code = "VALIDATION_ERROR";
    readonly httpStatus = 400;
    constructor(public readonly fieldErrors: FieldError[]) {
        super(`入力値が不正です: ${fieldErrors.map(e => e.field).join(", ")}`);
    }

    toResponse(): ErrorResponse {
        return {
            error: {
                code: this.code,
                message: this.message,
                timestamp: this.timestamp,
                details: this.fieldErrors.map(e => ({
                    field: e.field,
                    message: e.message,
                    code: e.code,
                })),
            }
        };
    }

    // フィールド別のエラー取得
    getFieldError(field: string): FieldError | undefined {
        return this.fieldErrors.find(e => e.field === field);
    }

    // エラーの追加
    static builder(): ValidationErrorBuilder {
        return new ValidationErrorBuilder();
    }
}

// バリデーションエラービルダー
class ValidationErrorBuilder {
    private errors: FieldError[] = [];

    addError(field: string, message: string, code: string, value?: unknown): this {
        this.errors.push({ field, message, code, value });
        return this;
    }

    required(field: string): this {
        return this.addError(field, `${field} は必須です`, "REQUIRED");
    }

    invalidFormat(field: string, expectedFormat: string): this {
        return this.addError(field, `${field} の形式が不正です（期待: ${expectedFormat}）`, "INVALID_FORMAT");
    }

    tooLong(field: string, maxLength: number): this {
        return this.addError(field, `${field} は ${maxLength} 文字以内で入力してください`, "TOO_LONG");
    }

    tooShort(field: string, minLength: number): this {
        return this.addError(field, `${field} は ${minLength} 文字以上で入力してください`, "TOO_SHORT");
    }

    outOfRange(field: string, min: number, max: number): this {
        return this.addError(field, `${field} は ${min} から ${max} の範囲で入力してください`, "OUT_OF_RANGE");
    }

    hasErrors(): boolean {
        return this.errors.length > 0;
    }

    build(): ValidationError {
        if (this.errors.length === 0) {
            throw new Error("ValidationError requires at least one field error");
        }
        return new ValidationError(this.errors);
    }

    buildIfErrors(): ValidationError | null {
        return this.errors.length > 0 ? new ValidationError(this.errors) : null;
    }
}

// 使用例
function validateCreateUser(data: unknown): ValidationError | null {
    const builder = ValidationError.builder();

    if (!data || typeof data !== "object") {
        builder.addError("body", "リクエストボディが不正です", "INVALID_BODY");
        return builder.build();
    }

    const { name, email, password, age } = data as any;

    if (!name) builder.required("name");
    else if (name.length > 100) builder.tooLong("name", 100);

    if (!email) builder.required("email");
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        builder.invalidFormat("email", "user@example.com");
    }

    if (!password) builder.required("password");
    else if (password.length < 8) builder.tooShort("password", 8);

    if (age !== undefined && (age < 0 || age > 150)) {
        builder.outOfRange("age", 0, 150);
    }

    return builder.buildIfErrors();
}

// ---------- ビジネスロジック系 ----------
class InsufficientBalanceError extends AppError {
    readonly code = "INSUFFICIENT_BALANCE";
    readonly httpStatus = 400;
    constructor(
        public readonly required: number,
        public readonly available: number,
        public readonly currency: string = "JPY",
    ) {
        super(`残高不足: ${required.toLocaleString()} ${currency} 必要、${available.toLocaleString()} ${currency} 利用可能`);
    }
}

class OrderAlreadyCancelledError extends AppError {
    readonly code = "ORDER_ALREADY_CANCELLED";
    readonly httpStatus = 400;
    constructor(
        public readonly orderId: string,
        public readonly cancelledAt: Date,
    ) {
        super(`注文 ${orderId} は既にキャンセルされています（${cancelledAt.toISOString()}）`);
    }
}

class StockNotAvailableError extends AppError {
    readonly code = "STOCK_NOT_AVAILABLE";
    readonly httpStatus = 400;
    constructor(
        public readonly productId: string,
        public readonly requested: number,
        public readonly available: number,
    ) {
        super(`在庫不足: 商品 ${productId}（要求: ${requested}, 在庫: ${available}）`);
    }
}

class RateLimitExceededError extends AppError {
    readonly code = "RATE_LIMIT_EXCEEDED";
    readonly httpStatus = 429;
    constructor(
        public readonly limit: number,
        public readonly windowMs: number,
        public readonly retryAfterMs: number,
    ) {
        super(`レート制限を超過しました（${limit} リクエスト / ${windowMs / 1000}秒）`);
    }

    toResponse(): ErrorResponse {
        return {
            error: {
                code: this.code,
                message: this.message,
                timestamp: this.timestamp,
                retryAfter: Math.ceil(this.retryAfterMs / 1000),
            }
        };
    }
}

// ---------- 外部サービス系 ----------
class ExternalServiceError extends AppError {
    readonly code = "EXTERNAL_SERVICE_ERROR";
    readonly httpStatus = 502;
    constructor(
        public readonly serviceName: string,
        public readonly serviceStatus?: number,
        options?: { cause?: Error }
    ) {
        super(
            `外部サービス ${serviceName} でエラーが発生しました${serviceStatus ? `（HTTP ${serviceStatus}）` : ''}`,
            true,
            options
        );
    }
}

class ServiceTimeoutError extends AppError {
    readonly code = "SERVICE_TIMEOUT";
    readonly httpStatus = 504;
    constructor(
        public readonly serviceName: string,
        public readonly timeoutMs: number,
    ) {
        super(`${serviceName} への接続がタイムアウトしました（${timeoutMs}ms）`);
    }
}

// ---------- 内部エラー ----------
class InternalError extends AppError {
    readonly code = "INTERNAL_ERROR";
    readonly httpStatus = 500;
    constructor(message: string, options?: { cause?: Error }) {
        super(message, false, options);  // isOperational = false
    }
}
```

---

## 3. エラーコード体系

### 3.1 命名規則

```
エラーコード設計:
  → 一意の文字列識別子
  → 機械可読（プログラムで判定可能）
  → 人間可読（見て意味が分かる）

命名規則:
  {DOMAIN}_{ENTITY}_{ACTION}

  AUTH_TOKEN_EXPIRED         — 認証トークン期限切れ
  AUTH_CREDENTIALS_INVALID   — 認証情報不正
  USER_NOT_FOUND            — ユーザー未発見
  USER_EMAIL_DUPLICATE      — メール重複
  ORDER_PAYMENT_FAILED      — 注文の支払い失敗
  ORDER_ALREADY_CANCELLED   — 注文は既にキャンセル済み
  RATE_LIMIT_EXCEEDED       — レート制限超過
  INTERNAL_SERVER_ERROR     — サーバー内部エラー
```

### 3.2 エラーコードレジストリ

```typescript
// エラーコードをenumで管理
const ErrorCodes = {
  // 認証
  AUTH_TOKEN_EXPIRED: { status: 401, message: "トークンが期限切れです" },
  AUTH_CREDENTIALS_INVALID: { status: 401, message: "認証情報が不正です" },
  AUTH_FORBIDDEN: { status: 403, message: "アクセスが禁止されています" },

  // ユーザー
  USER_NOT_FOUND: { status: 404, message: "ユーザーが見つかりません" },
  USER_EMAIL_DUPLICATE: { status: 409, message: "メールアドレスは既に使用されています" },

  // バリデーション
  VALIDATION_ERROR: { status: 400, message: "入力値が不正です" },

  // サーバー
  INTERNAL_ERROR: { status: 500, message: "サーバーエラーが発生しました" },
} as const;

type ErrorCode = keyof typeof ErrorCodes;
```

### 3.3 エラーコードの階層的管理

```typescript
// エラーコードの階層的管理
const ERROR_REGISTRY = {
    // ========== 認証・認可 ==========
    AUTH: {
        UNAUTHENTICATED: {
            httpStatus: 401,
            message: "認証が必要です",
            retryable: false,
            userMessage: "ログインしてください",
        },
        TOKEN_EXPIRED: {
            httpStatus: 401,
            message: "トークンが期限切れです",
            retryable: true,
            userMessage: "セッションの有効期限が切れました。再度ログインしてください",
        },
        INVALID_TOKEN: {
            httpStatus: 401,
            message: "無効なトークンです",
            retryable: false,
            userMessage: "認証に失敗しました。再度ログインしてください",
        },
        FORBIDDEN: {
            httpStatus: 403,
            message: "権限がありません",
            retryable: false,
            userMessage: "この操作を実行する権限がありません",
        },
    },

    // ========== ユーザー ==========
    USER: {
        NOT_FOUND: {
            httpStatus: 404,
            message: "ユーザーが見つかりません",
            retryable: false,
            userMessage: "指定されたユーザーは存在しません",
        },
        EMAIL_DUPLICATE: {
            httpStatus: 409,
            message: "メールアドレスが重複しています",
            retryable: false,
            userMessage: "このメールアドレスは既に登録されています",
        },
        PROFILE_INCOMPLETE: {
            httpStatus: 400,
            message: "プロフィールが不完全です",
            retryable: false,
            userMessage: "必要な情報を入力してください",
        },
    },

    // ========== 注文 ==========
    ORDER: {
        NOT_FOUND: {
            httpStatus: 404,
            message: "注文が見つかりません",
            retryable: false,
            userMessage: "指定された注文は存在しません",
        },
        PAYMENT_FAILED: {
            httpStatus: 400,
            message: "支払いに失敗しました",
            retryable: true,
            userMessage: "お支払いを処理できませんでした。別の支払い方法をお試しください",
        },
        ALREADY_CANCELLED: {
            httpStatus: 400,
            message: "注文は既にキャンセルされています",
            retryable: false,
            userMessage: "この注文は既にキャンセルされています",
        },
        STOCK_UNAVAILABLE: {
            httpStatus: 400,
            message: "在庫がありません",
            retryable: false,
            userMessage: "申し訳ありませんが、この商品は現在在庫切れです",
        },
    },

    // ========== システム ==========
    SYSTEM: {
        INTERNAL_ERROR: {
            httpStatus: 500,
            message: "内部エラーが発生しました",
            retryable: true,
            userMessage: "サーバーエラーが発生しました。しばらく待ってから再試行してください",
        },
        SERVICE_UNAVAILABLE: {
            httpStatus: 503,
            message: "サービスが一時的に利用できません",
            retryable: true,
            userMessage: "現在メンテナンス中です。しばらくお待ちください",
        },
        RATE_LIMITED: {
            httpStatus: 429,
            message: "リクエストが多すぎます",
            retryable: true,
            userMessage: "リクエストの頻度が高すぎます。しばらく待ってから再試行してください",
        },
    },

    // ========== バリデーション ==========
    VALIDATION: {
        INVALID_INPUT: {
            httpStatus: 400,
            message: "入力値が不正です",
            retryable: false,
            userMessage: "入力内容に誤りがあります。確認して再入力してください",
        },
        MISSING_FIELD: {
            httpStatus: 400,
            message: "必須フィールドが欠落しています",
            retryable: false,
            userMessage: "必須項目を入力してください",
        },
        INVALID_FORMAT: {
            httpStatus: 400,
            message: "フォーマットが不正です",
            retryable: false,
            userMessage: "正しい形式で入力してください",
        },
    },
} as const;

// 型安全なエラーコードの取得
type ErrorDomain = keyof typeof ERROR_REGISTRY;
type ErrorCodeOf<D extends ErrorDomain> = keyof typeof ERROR_REGISTRY[D];

function getErrorInfo<D extends ErrorDomain>(
    domain: D,
    code: ErrorCodeOf<D>
): typeof ERROR_REGISTRY[D][ErrorCodeOf<D>] {
    return ERROR_REGISTRY[domain][code];
}

// 使用例
const info = getErrorInfo('AUTH', 'TOKEN_EXPIRED');
// info.httpStatus === 401
// info.message === "トークンが期限切れです"
// info.retryable === true
```

---

## 4. Rust でのエラー設計

### 4.1 thiserror を使ったカスタムエラー

```rust
// Rust: thiserror クレートでカスタムエラー
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("User not found: {user_id}")]
    UserNotFound { user_id: String },

    #[error("Email already exists: {email}")]
    EmailAlreadyExists { email: String },

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Database error")]
    Database(#[from] sqlx::Error),

    #[error("External API error")]
    ExternalApi(#[from] reqwest::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl AppError {
    pub fn status_code(&self) -> u16 {
        match self {
            Self::UserNotFound { .. } => 404,
            Self::EmailAlreadyExists { .. } => 409,
            Self::Validation(_) => 400,
            Self::Database(_) => 500,
            Self::ExternalApi(_) => 502,
            Self::Internal(_) => 500,
        }
    }
}
```

### 4.2 エラーの階層化

```rust
// ドメインごとにエラー型を分ける
use thiserror::Error;

// ユーザードメインのエラー
#[derive(Error, Debug)]
pub enum UserError {
    #[error("User not found: {0}")]
    NotFound(String),

    #[error("Email already exists: {0}")]
    EmailDuplicate(String),

    #[error("Invalid user data: {0}")]
    InvalidData(String),

    #[error("User is deactivated: {0}")]
    Deactivated(String),
}

// 注文ドメインのエラー
#[derive(Error, Debug)]
pub enum OrderError {
    #[error("Order not found: {0}")]
    NotFound(String),

    #[error("Insufficient balance: required {required}, available {available}")]
    InsufficientBalance { required: f64, available: f64 },

    #[error("Order already cancelled: {0}")]
    AlreadyCancelled(String),

    #[error("Stock not available: product {product_id}, requested {requested}, available {available}")]
    StockNotAvailable {
        product_id: String,
        requested: u32,
        available: u32,
    },
}

// アプリケーション全体のエラー（各ドメインエラーを集約）
#[derive(Error, Debug)]
pub enum AppError {
    #[error(transparent)]
    User(#[from] UserError),

    #[error(transparent)]
    Order(#[from] OrderError),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Database error")]
    Database(#[from] sqlx::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

// Actix-web でのレスポンス変換
impl actix_web::ResponseError for AppError {
    fn error_response(&self) -> actix_web::HttpResponse {
        let status = self.status_code();
        let body = serde_json::json!({
            "error": {
                "code": self.error_code(),
                "message": self.to_string(),
            }
        });
        actix_web::HttpResponse::build(status).json(body)
    }

    fn status_code(&self) -> actix_web::http::StatusCode {
        use actix_web::http::StatusCode;
        match self {
            AppError::User(UserError::NotFound(_)) => StatusCode::NOT_FOUND,
            AppError::User(UserError::EmailDuplicate(_)) => StatusCode::CONFLICT,
            AppError::User(UserError::InvalidData(_)) => StatusCode::BAD_REQUEST,
            AppError::Order(OrderError::NotFound(_)) => StatusCode::NOT_FOUND,
            AppError::Order(OrderError::InsufficientBalance { .. }) => StatusCode::BAD_REQUEST,
            AppError::Auth(_) => StatusCode::UNAUTHORIZED,
            AppError::Database(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl AppError {
    fn error_code(&self) -> &str {
        match self {
            AppError::User(UserError::NotFound(_)) => "USER_NOT_FOUND",
            AppError::User(UserError::EmailDuplicate(_)) => "USER_EMAIL_DUPLICATE",
            AppError::User(UserError::InvalidData(_)) => "USER_INVALID_DATA",
            AppError::Order(OrderError::NotFound(_)) => "ORDER_NOT_FOUND",
            AppError::Order(OrderError::InsufficientBalance { .. }) => "INSUFFICIENT_BALANCE",
            AppError::Order(OrderError::AlreadyCancelled(_)) => "ORDER_ALREADY_CANCELLED",
            AppError::Auth(_) => "AUTHENTICATION_ERROR",
            AppError::Database(_) => "DATABASE_ERROR",
            AppError::Internal(_) => "INTERNAL_ERROR",
            _ => "UNKNOWN_ERROR",
        }
    }
}
```

---

## 5. Python でのカスタムエラー

### 5.1 エラー階層の設計

```python
# Python: カスタム例外階層
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import traceback

class AppError(Exception):
    """アプリケーションエラーの基底クラス"""
    code: str = "INTERNAL_ERROR"
    http_status: int = 500
    is_operational: bool = True

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        http_status: Optional[int] = None,
        cause: Optional[Exception] = None,
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        if code:
            self.code = code
        if http_status:
            self.http_status = http_status
        self.timestamp = datetime.utcnow().isoformat()
        self.context = context or {}
        if cause:
            self.__cause__ = cause

    def to_dict(self) -> dict:
        """APIレスポンス用のdict変換"""
        result = {
            "error": {
                "code": self.code,
                "message": str(self),
                "timestamp": self.timestamp,
            }
        }
        return result

    def to_log(self) -> dict:
        """ログ用のdict変換（内部情報を含む）"""
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": str(self),
            "http_status": self.http_status,
            "is_operational": self.is_operational,
            "context": self.context,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exc() if self.__traceback__ else None,
            "cause": str(self.__cause__) if self.__cause__ else None,
        }


# ========== 認証系 ==========
class AuthenticationError(AppError):
    code = "AUTHENTICATION_REQUIRED"
    http_status = 401

    def __init__(self, message: str = "認証が必要です", **kwargs):
        super().__init__(message, **kwargs)


class TokenExpiredError(AuthenticationError):
    code = "TOKEN_EXPIRED"

    def __init__(self, expired_at: datetime, **kwargs):
        self.expired_at = expired_at
        super().__init__(f"トークンが期限切れです（{expired_at.isoformat()}）", **kwargs)


class AuthorizationError(AppError):
    code = "FORBIDDEN"
    http_status = 403

    def __init__(
        self,
        required_permission: str,
        actual_permissions: list[str] | None = None,
        **kwargs,
    ):
        self.required_permission = required_permission
        self.actual_permissions = actual_permissions or []
        super().__init__(
            f"権限が不足しています: {required_permission} が必要です",
            **kwargs,
        )


# ========== リソース系 ==========
class NotFoundError(AppError):
    code = "NOT_FOUND"
    http_status = 404

    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(
            f"{resource_type} が見つかりません: {resource_id}",
            **kwargs,
        )


class ConflictError(AppError):
    code = "CONFLICT"
    http_status = 409

    def __init__(self, resource_type: str, conflict_field: str, conflict_value: str, **kwargs):
        self.resource_type = resource_type
        self.conflict_field = conflict_field
        self.conflict_value = conflict_value
        super().__init__(
            f"{resource_type} の {conflict_field} が重複しています: {conflict_value}",
            **kwargs,
        )


# ========== バリデーション系 ==========
@dataclass
class FieldError:
    field: str
    message: str
    code: str = "INVALID"
    value: Any = None


class ValidationError(AppError):
    code = "VALIDATION_ERROR"
    http_status = 400

    def __init__(self, field_errors: list[FieldError], **kwargs):
        self.field_errors = field_errors
        fields = ", ".join(e.field for e in field_errors)
        super().__init__(f"入力値が不正です: {fields}", **kwargs)

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["error"]["details"] = [
            {"field": e.field, "message": e.message, "code": e.code}
            for e in self.field_errors
        ]
        return result


# ========== ビジネスロジック系 ==========
class InsufficientBalanceError(AppError):
    code = "INSUFFICIENT_BALANCE"
    http_status = 400

    def __init__(self, required: float, available: float, currency: str = "JPY", **kwargs):
        self.required = required
        self.available = available
        self.currency = currency
        super().__init__(
            f"残高不足: {required:,.0f} {currency} 必要、{available:,.0f} {currency} 利用可能",
            **kwargs,
        )


# ========== 使用例 ==========
def get_user(user_id: str) -> User:
    user = user_repository.find_by_id(user_id)
    if user is None:
        raise NotFoundError("User", user_id)
    return user


def create_user(data: dict) -> User:
    # バリデーション
    errors = []
    if not data.get("name"):
        errors.append(FieldError(field="name", message="名前は必須です", code="REQUIRED"))
    if not data.get("email"):
        errors.append(FieldError(field="email", message="メールは必須です", code="REQUIRED"))
    elif not is_valid_email(data["email"]):
        errors.append(FieldError(field="email", message="メールの形式が不正です", code="INVALID_FORMAT"))

    if errors:
        raise ValidationError(errors)

    # 重複チェック
    existing = user_repository.find_by_email(data["email"])
    if existing:
        raise ConflictError("User", "email", data["email"])

    return user_repository.create(data)
```

---

## 6. Go でのカスタムエラー

### 6.1 構造化エラー

```go
// Go: 構造化カスタムエラー
package apperror

import (
    "fmt"
    "net/http"
    "time"
)

// AppError: アプリケーションエラーの基底
type AppError struct {
    Code       string         `json:"code"`
    Message    string         `json:"message"`
    HTTPStatus int            `json:"-"`
    Timestamp  time.Time      `json:"timestamp"`
    Details    map[string]any `json:"details,omitempty"`
    Err        error          `json:"-"`  // 内部エラー（外部に公開しない）
}

func (e *AppError) Error() string {
    if e.Err != nil {
        return fmt.Sprintf("[%s] %s: %v", e.Code, e.Message, e.Err)
    }
    return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

func (e *AppError) Unwrap() error {
    return e.Err
}

// ファクトリ関数
func NewNotFound(resourceType, resourceID string) *AppError {
    return &AppError{
        Code:       "NOT_FOUND",
        Message:    fmt.Sprintf("%s not found: %s", resourceType, resourceID),
        HTTPStatus: http.StatusNotFound,
        Timestamp:  time.Now(),
        Details: map[string]any{
            "resource_type": resourceType,
            "resource_id":   resourceID,
        },
    }
}

func NewValidationError(fields map[string]string) *AppError {
    return &AppError{
        Code:       "VALIDATION_ERROR",
        Message:    "入力値が不正です",
        HTTPStatus: http.StatusBadRequest,
        Timestamp:  time.Now(),
        Details: map[string]any{
            "fields": fields,
        },
    }
}

func NewConflict(resourceType, field, value string) *AppError {
    return &AppError{
        Code:       "CONFLICT",
        Message:    fmt.Sprintf("%s の %s が重複しています: %s", resourceType, field, value),
        HTTPStatus: http.StatusConflict,
        Timestamp:  time.Now(),
    }
}

func NewInternalError(message string, cause error) *AppError {
    return &AppError{
        Code:       "INTERNAL_ERROR",
        Message:    message,
        HTTPStatus: http.StatusInternalServerError,
        Timestamp:  time.Now(),
        Err:        cause,
    }
}

func NewUnauthorized(message string) *AppError {
    if message == "" {
        message = "認証が必要です"
    }
    return &AppError{
        Code:       "UNAUTHORIZED",
        Message:    message,
        HTTPStatus: http.StatusUnauthorized,
        Timestamp:  time.Now(),
    }
}

func NewForbidden(requiredPermission string) *AppError {
    return &AppError{
        Code:       "FORBIDDEN",
        Message:    fmt.Sprintf("権限が不足しています: %s が必要です", requiredPermission),
        HTTPStatus: http.StatusForbidden,
        Timestamp:  time.Now(),
        Details: map[string]any{
            "required_permission": requiredPermission,
        },
    }
}

// エラーの判定
func IsNotFound(err error) bool {
    var appErr *AppError
    if errors.As(err, &appErr) {
        return appErr.Code == "NOT_FOUND"
    }
    return false
}

func IsValidationError(err error) bool {
    var appErr *AppError
    if errors.As(err, &appErr) {
        return appErr.Code == "VALIDATION_ERROR"
    }
    return false
}

// HTTPレスポンスへの変換
func (e *AppError) ToResponse() map[string]any {
    response := map[string]any{
        "error": map[string]any{
            "code":      e.Code,
            "message":   e.Message,
            "timestamp": e.Timestamp.Format(time.RFC3339),
        },
    }
    if len(e.Details) > 0 {
        response["error"].(map[string]any)["details"] = e.Details
    }
    return response
}
```

---

## 7. エラー設計の原則

### 7.1 基本原則

```
1. エラーは具体的に
   ❌ throw new Error("Error occurred")
   ✅ throw new UserNotFoundError(userId)

2. エラーにコンテキストを含める
   ❌ "Not found"
   ✅ "User not found: user-123"
   ✅ { code: "USER_NOT_FOUND", userId: "user-123" }

3. ユーザー向けと開発者向けを分離
   ユーザー: "ログインに失敗しました"
   開発者: "Auth0 returned 429: rate limit exceeded for IP 192.168.1.1"

4. エラーの原因チェーン
   → 根本原因を追跡可能にする
   → Error.cause (ES2022), Rust の source(), Go の %w

5. エラーは不変（Immutable）
   → 作成後に状態を変更しない
   → 安全に渡せる、ログに記録できる
```

### 7.2 エラーメッセージの設計

```
開発者向けメッセージのガイドライン:

  1. What（何が起きたか）
     → "Database connection refused"
     → "JSON parse error at position 42"

  2. Where（どこで起きたか）
     → "in UserService.createUser"
     → "while processing order ORD-123"

  3. Why（なぜ起きたか、推定される原因）
     → "connection pool exhausted (max: 10, active: 10)"
     → "unexpected field 'naem' (did you mean 'name'?)"

  4. How（どう対処すべきか）
     → "retry after 5 seconds"
     → "check database connection settings"
     → "contact support with error code ERR-123"

ユーザー向けメッセージのガイドライン:

  1. 何が起きたかを簡潔に
     → "ログインに失敗しました"
     → "注文を処理できませんでした"

  2. 何をすべきかを具体的に
     → "パスワードを確認して再試行してください"
     → "別の支払い方法をお試しください"

  3. 技術用語を避ける
     ❌ "NullPointerException が発生しました"
     ✅ "予期しないエラーが発生しました"

  4. ユーザーを責めない
     ❌ "不正なメールアドレスです"
     ✅ "メールアドレスの形式を確認してください"
```

### 7.3 エラーの原因チェーン

```typescript
// ES2022: Error.cause
class ServiceError extends Error {
    constructor(message: string, options?: { cause?: Error }) {
        super(message, options);
    }
}

// 原因チェーンの構築
async function processOrder(orderId: string): Promise<Order> {
    try {
        const order = await orderRepository.findById(orderId);
        if (!order) throw new NotFoundError("Order", orderId);

        const payment = await paymentService.charge(order);
        return { ...order, paymentId: payment.id };
    } catch (error) {
        if (error instanceof NotFoundError) throw error;

        // 原因チェーンを構築
        throw new ServiceError(
            `注文 ${orderId} の処理に失敗しました`,
            { cause: error as Error }
        );
    }
}

// 原因チェーンの走査
function getAllCauses(error: Error): Error[] {
    const causes: Error[] = [error];
    let current: unknown = error.cause;
    while (current instanceof Error) {
        causes.push(current);
        current = current.cause;
    }
    return causes;
}

// ログ出力時に全原因を出力
function logErrorChain(error: Error): void {
    const causes = getAllCauses(error);
    logger.error({
        message: error.message,
        chain: causes.map((e, i) => ({
            depth: i,
            type: e.name,
            message: e.message,
        })),
    });
}
```

---

## 8. エラーのシリアライズと API 設計

### 8.1 APIエラーレスポンスの標準化

```typescript
// RFC 7807 準拠のエラーレスポンス
interface ApiErrorResponse {
    type: string;        // エラータイプのURI
    title: string;       // 人間可読な概要
    status: number;      // HTTPステータスコード
    detail?: string;     // 詳細な説明
    instance?: string;   // エラーが発生したリソースのURI
    errors?: FieldError[];  // バリデーションエラー詳細
    retryAfter?: number;    // リトライまでの秒数
    requestId?: string;     // リクエストID
}

// エラーレスポンスの生成
function createApiErrorResponse(
    error: AppError,
    requestId: string,
    requestPath: string
): ApiErrorResponse {
    const base: ApiErrorResponse = {
        type: `https://api.example.com/errors/${error.code.toLowerCase()}`,
        title: error.message,
        status: error.httpStatus,
        instance: requestPath,
        requestId,
    };

    if (error instanceof ValidationError) {
        return {
            ...base,
            errors: error.fieldErrors,
        };
    }

    if (error instanceof RateLimitExceededError) {
        return {
            ...base,
            retryAfter: Math.ceil(error.retryAfterMs / 1000),
        };
    }

    return base;
}
```

### 8.2 GraphQL でのエラー設計

```typescript
// GraphQL: エラーの設計
// GraphQL では HTTP ステータスコードに頼らず、レスポンスボディでエラーを表現

// スキーマ定義
const typeDefs = `
  type Query {
    user(id: ID!): UserResult!
  }

  union UserResult = User | UserError

  type User {
    id: ID!
    name: String!
    email: String!
  }

  type UserError {
    code: String!
    message: String!
  }

  # または extensions を使ったアプローチ
`;

// リゾルバ
const resolvers = {
    Query: {
        user: async (_: any, { id }: { id: string }) => {
            try {
                const user = await userService.getUser(id);
                return { __typename: 'User', ...user };
            } catch (error) {
                if (error instanceof NotFoundError) {
                    return {
                        __typename: 'UserError',
                        code: error.code,
                        message: error.message,
                    };
                }
                throw error;  // 予期しないエラーは GraphQL のエラーとして伝播
            }
        },
    },
};

// formatError: GraphQL エラーのフォーマット
const server = new ApolloServer({
    typeDefs,
    resolvers,
    formatError: (formattedError, error) => {
        // 内部エラーの情報を隠す
        if (error instanceof GraphQLError) {
            const originalError = error.extensions?.originalError;
            if (originalError instanceof AppError) {
                return {
                    message: originalError.message,
                    extensions: {
                        code: originalError.code,
                    },
                };
            }
        }

        // 予期しないエラー
        return {
            message: 'Internal server error',
            extensions: {
                code: 'INTERNAL_ERROR',
            },
        };
    },
});
```

---

## 9. エラーの国際化（i18n）

### 9.1 多言語対応のエラーメッセージ

```typescript
// エラーメッセージの国際化
const ERROR_MESSAGES: Record<string, Record<string, string>> = {
    "USER_NOT_FOUND": {
        en: "User not found: {userId}",
        ja: "ユーザーが見つかりません: {userId}",
        zh: "未找到用户: {userId}",
    },
    "EMAIL_ALREADY_EXISTS": {
        en: "Email already registered: {email}",
        ja: "メールアドレスは既に登録されています: {email}",
        zh: "邮箱已注册: {email}",
    },
    "VALIDATION_REQUIRED": {
        en: "{field} is required",
        ja: "{field} は必須です",
        zh: "{field} 是必填项",
    },
    "VALIDATION_TOO_LONG": {
        en: "{field} must be {maxLength} characters or less",
        ja: "{field} は {maxLength} 文字以内で入力してください",
        zh: "{field} 不能超过 {maxLength} 个字符",
    },
};

function getLocalizedMessage(
    code: string,
    locale: string,
    params: Record<string, string | number> = {}
): string {
    const templates = ERROR_MESSAGES[code];
    if (!templates) return code;

    const template = templates[locale] || templates["en"] || code;

    return template.replace(/\{(\w+)\}/g, (_, key) => {
        return String(params[key] ?? `{${key}}`);
    });
}

// 使用例
const message = getLocalizedMessage(
    "USER_NOT_FOUND",
    "ja",
    { userId: "user-123" }
);
// "ユーザーが見つかりません: user-123"

// API ミドルウェアでの locale 取得
function getLocale(req: Request): string {
    // 1. クエリパラメータ
    if (req.query.locale) return req.query.locale as string;

    // 2. Accept-Language ヘッダー
    const acceptLanguage = req.headers['accept-language'];
    if (acceptLanguage) {
        const preferred = acceptLanguage.split(',')[0].split('-')[0].trim();
        if (['en', 'ja', 'zh'].includes(preferred)) return preferred;
    }

    // 3. デフォルト
    return 'en';
}

// エラーレスポンスへの統合
function createLocalizedErrorResponse(
    error: AppError,
    locale: string
): ErrorResponse {
    return {
        error: {
            code: error.code,
            message: getLocalizedMessage(error.code, locale, error.params),
            timestamp: error.timestamp,
        }
    };
}
```

---

## 10. テストパターン

### 10.1 カスタムエラーのテスト

```typescript
// カスタムエラーのユニットテスト
describe("NotFoundError", () => {
    it("正しいプロパティが設定される", () => {
        const error = new NotFoundError("User", "user-123");

        expect(error).toBeInstanceOf(AppError);
        expect(error).toBeInstanceOf(NotFoundError);
        expect(error.code).toBe("NOT_FOUND");
        expect(error.httpStatus).toBe(404);
        expect(error.message).toBe("User が見つかりません: user-123");
        expect(error.resourceType).toBe("User");
        expect(error.resourceId).toBe("user-123");
        expect(error.isOperational).toBe(true);
        expect(error.timestamp).toBeDefined();
    });

    it("toResponse() が正しいフォーマットを返す", () => {
        const error = new NotFoundError("User", "user-123");
        const response = error.toResponse();

        expect(response).toEqual({
            error: {
                code: "NOT_FOUND",
                message: expect.stringContaining("user-123"),
                timestamp: expect.any(String),
            }
        });
    });

    it("スタックトレースが含まれる", () => {
        const error = new NotFoundError("User", "user-123");
        expect(error.stack).toBeDefined();
        expect(error.stack).toContain("NotFoundError");
    });
});

describe("ValidationError", () => {
    it("ビルダーパターンで構築できる", () => {
        const error = ValidationError.builder()
            .required("name")
            .invalidFormat("email", "user@example.com")
            .tooShort("password", 8)
            .build();

        expect(error.fieldErrors).toHaveLength(3);
        expect(error.fieldErrors[0].field).toBe("name");
        expect(error.fieldErrors[1].field).toBe("email");
        expect(error.fieldErrors[2].field).toBe("password");
    });

    it("エラーがない場合 buildIfErrors() は null を返す", () => {
        const error = ValidationError.builder().buildIfErrors();
        expect(error).toBeNull();
    });
});
```

---

## まとめ

| 原則 | ポイント |
|------|---------|
| 分類 | 操作エラー vs プログラマエラー |
| コード | DOMAIN_ENTITY_ACTION 命名 |
| コンテキスト | エラーに十分な情報を含める |
| 分離 | ユーザー向け / 開発者向け |
| チェーン | 根本原因を追跡可能に |
| 国際化 | エラーコード + メッセージテンプレート |
| テスト | プロパティ、シリアライズ、ビルダーのテスト |

---

## 次に読むべきガイド
→ [[../03-advanced/00-event-loop.md]] — イベントループ

---

## 参考文献
1. Goldberg, J. "Error Handling in Node.js." joyent.com, 2014.
2. RFC 7807. "Problem Details for HTTP APIs."
3. thiserror crate. Rust Documentation.
4. NestJS Documentation. "Exception Filters."
5. Python Documentation. "Built-in Exceptions."
6. Go Blog. "Working with Errors in Go 1.13."
