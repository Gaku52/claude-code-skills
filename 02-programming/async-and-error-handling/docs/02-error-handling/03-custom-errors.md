# カスタムエラー設計

> エラーを適切にモデリングすることは、ソフトウェアの信頼性と保守性の基盤。エラーコード体系、ドメインエラー、エラーのシリアライズ手法を解説する。

## この章で学ぶこと

- [ ] カスタムエラーの設計原則を理解する
- [ ] エラーコード体系の作り方を把握する
- [ ] ドメイン駆動のエラー設計を学ぶ

---

## 1. エラーの分類

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

---

## 2. TypeScript でのカスタムエラー

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

---

## 3. エラーコード体系

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

---

## 4. Rust でのエラー設計

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

---

## 5. エラー設計の原則

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

---

## 次に読むべきガイド
→ [[../03-advanced/00-event-loop.md]] — イベントループ

---

## 参考文献
1. Goldberg, J. "Error Handling in Node.js." joyent.com, 2014.
2. RFC 7807. "Problem Details for HTTP APIs."
