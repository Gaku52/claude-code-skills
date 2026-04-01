# Custom Error Design

> Properly modeling errors is the foundation of software reliability and maintainability. This guide covers error code systems, domain errors, and error serialization techniques.

## Learning Objectives

- [ ] Understand design principles for custom errors
- [ ] Learn how to build an error code system
- [ ] Study domain-driven error design
- [ ] Master custom error implementation patterns in various languages
- [ ] Learn error serialization and API design
- [ ] Understand error internationalization (i18n)


## Prerequisites

Understanding the following will help you get the most out of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with [Error Boundaries](./02-error-boundaries.md)

---

## 1. Error Classification

### 1.1 Operational Errors vs Programmer Errors

```
Operational Error:
  -> Expected runtime errors
  -> Examples: Network disconnection, DB connection failure, validation failure
  -> Response: Retry, fallback, notify user

Programmer Error:
  -> Bugs. Code needs to be fixed
  -> Examples: Null reference, type error, array out-of-bounds access
  -> Response: Crash -> Fix -> Deploy

This distinction is important:
  -> Operational errors: Handle them (recoverable)
  -> Programmer errors: Let them crash (unrecoverable)
```

### 1.2 Detailed Error Classification System

```
Error Classification System:

  1. Client Errors (4xx)
     -> Validation Error (400)
     -> Authentication Error (401)
     -> Authorization Error (403)
     -> Resource Not Found (404)
     -> Conflict Error (409)
     -> Rate Limit (429)

  2. Server Errors (5xx)
     -> Internal Error (500)
     -> External Service Error (502)
     -> Service Unavailable (503)
     -> Timeout (504)

  3. Business Logic Errors
     -> Insufficient Balance
     -> Order Already Cancelled
     -> Out of Stock
     -> Expired
     -> Policy Violation

  4. Infrastructure Errors
     -> Database Connection Error
     -> Message Queue Connection Error
     -> File System Error
     -> Out of Memory

  Characteristics of each:
  +-------------------+------------+------------+-----------+
  | Category          | Recoverabi | Notify     | Log Level |
  |                   | lity       |            |           |
  +-------------------+------------+------------+-----------+
  | Client            | Possible   | User       | warn      |
  | Server            | Impossible | Developer  | error     |
  | Business Logic    | Possible   | User       | warn      |
  | Infrastructure    | Retryable  | Operations | error     |
  +-------------------+------------+------------+-----------+
```

---

## 2. Custom Errors in TypeScript

### 2.1 Base Error Class

```typescript
// Base error class
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

### 2.2 Domain Error Implementation

```typescript
// Domain errors
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

// Validation error (multiple fields)
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

### 2.3 Complete Error Hierarchy Design Example

```typescript
// ========== Practical Complete Error Hierarchy ==========

// Base class
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

    // Serialization for API responses
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

    // Serialization for logging (includes internal information)
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

// ---------- Authentication ----------
class AuthenticationError extends AppError {
    readonly code = "AUTHENTICATION_REQUIRED";
    readonly httpStatus = 401;
    constructor(message = "Authentication required", options?: { cause?: Error }) {
        super(message, true, options);
    }
}

class TokenExpiredError extends AppError {
    readonly code = "TOKEN_EXPIRED";
    readonly httpStatus = 401;
    constructor(public readonly expiredAt: Date) {
        super(`Token expired (${expiredAt.toISOString()})`);
    }
}

class InvalidTokenError extends AppError {
    readonly code = "INVALID_TOKEN";
    readonly httpStatus = 401;
    constructor(public readonly reason: string) {
        super(`Invalid token: ${reason}`);
    }
}

class AuthorizationError extends AppError {
    readonly code = "FORBIDDEN";
    readonly httpStatus = 403;
    constructor(
        public readonly requiredPermission: string,
        public readonly actualPermissions: string[] = [],
    ) {
        super(`Insufficient permissions: ${requiredPermission} required`);
    }
}

// ---------- Resources ----------
class NotFoundError extends AppError {
    readonly code = "NOT_FOUND";
    readonly httpStatus = 404;
    constructor(
        public readonly resourceType: string,
        public readonly resourceId: string,
    ) {
        super(`${resourceType} not found: ${resourceId}`);
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
        super(`${resourceType} ${conflictField} already exists: ${conflictValue}`);
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
        super(`${resourceType} ${resourceId} has been deleted (${deletedAt.toISOString()})`);
    }
}

// ---------- Validation ----------
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
        super(`Invalid input: ${fieldErrors.map(e => e.field).join(", ")}`);
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

    // Get error for a specific field
    getFieldError(field: string): FieldError | undefined {
        return this.fieldErrors.find(e => e.field === field);
    }

    // Add errors
    static builder(): ValidationErrorBuilder {
        return new ValidationErrorBuilder();
    }
}

// Validation error builder
class ValidationErrorBuilder {
    private errors: FieldError[] = [];

    addError(field: string, message: string, code: string, value?: unknown): this {
        this.errors.push({ field, message, code, value });
        return this;
    }

    required(field: string): this {
        return this.addError(field, `${field} is required`, "REQUIRED");
    }

    invalidFormat(field: string, expectedFormat: string): this {
        return this.addError(field, `${field} has an invalid format (expected: ${expectedFormat})`, "INVALID_FORMAT");
    }

    tooLong(field: string, maxLength: number): this {
        return this.addError(field, `${field} must be ${maxLength} characters or fewer`, "TOO_LONG");
    }

    tooShort(field: string, minLength: number): this {
        return this.addError(field, `${field} must be at least ${minLength} characters`, "TOO_SHORT");
    }

    outOfRange(field: string, min: number, max: number): this {
        return this.addError(field, `${field} must be between ${min} and ${max}`, "OUT_OF_RANGE");
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

// Usage example
function validateCreateUser(data: unknown): ValidationError | null {
    const builder = ValidationError.builder();

    if (!data || typeof data !== "object") {
        builder.addError("body", "Invalid request body", "INVALID_BODY");
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

// ---------- Business Logic ----------
class InsufficientBalanceError extends AppError {
    readonly code = "INSUFFICIENT_BALANCE";
    readonly httpStatus = 400;
    constructor(
        public readonly required: number,
        public readonly available: number,
        public readonly currency: string = "JPY",
    ) {
        super(`Insufficient balance: ${required.toLocaleString()} ${currency} required, ${available.toLocaleString()} ${currency} available`);
    }
}

class OrderAlreadyCancelledError extends AppError {
    readonly code = "ORDER_ALREADY_CANCELLED";
    readonly httpStatus = 400;
    constructor(
        public readonly orderId: string,
        public readonly cancelledAt: Date,
    ) {
        super(`Order ${orderId} has already been cancelled (${cancelledAt.toISOString()})`);
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
        super(`Out of stock: product ${productId} (requested: ${requested}, available: ${available})`);
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
        super(`Rate limit exceeded (${limit} requests / ${windowMs / 1000}s)`);
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

// ---------- External Services ----------
class ExternalServiceError extends AppError {
    readonly code = "EXTERNAL_SERVICE_ERROR";
    readonly httpStatus = 502;
    constructor(
        public readonly serviceName: string,
        public readonly serviceStatus?: number,
        options?: { cause?: Error }
    ) {
        super(
            `Error occurred in external service ${serviceName}${serviceStatus ? ` (HTTP ${serviceStatus})` : ''}`,
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
        super(`Connection to ${serviceName} timed out (${timeoutMs}ms)`);
    }
}

// ---------- Internal Errors ----------
class InternalError extends AppError {
    readonly code = "INTERNAL_ERROR";
    readonly httpStatus = 500;
    constructor(message: string, options?: { cause?: Error }) {
        super(message, false, options);  // isOperational = false
    }
}
```

---

## 3. Error Code System

### 3.1 Naming Conventions

```
Error Code Design:
  -> Unique string identifier
  -> Machine-readable (can be evaluated programmatically)
  -> Human-readable (understandable at a glance)

Naming Convention:
  {DOMAIN}_{ENTITY}_{ACTION}

  AUTH_TOKEN_EXPIRED         - Authentication token expired
  AUTH_CREDENTIALS_INVALID   - Invalid credentials
  USER_NOT_FOUND            - User not found
  USER_EMAIL_DUPLICATE      - Duplicate email
  ORDER_PAYMENT_FAILED      - Order payment failed
  ORDER_ALREADY_CANCELLED   - Order already cancelled
  RATE_LIMIT_EXCEEDED       - Rate limit exceeded
  INTERNAL_SERVER_ERROR     - Internal server error
```

### 3.2 Error Code Registry

```typescript
// Managing error codes as an enum
const ErrorCodes = {
  // Authentication
  AUTH_TOKEN_EXPIRED: { status: 401, message: "Token has expired" },
  AUTH_CREDENTIALS_INVALID: { status: 401, message: "Invalid credentials" },
  AUTH_FORBIDDEN: { status: 403, message: "Access forbidden" },

  // User
  USER_NOT_FOUND: { status: 404, message: "User not found" },
  USER_EMAIL_DUPLICATE: { status: 409, message: "Email address is already in use" },

  // Validation
  VALIDATION_ERROR: { status: 400, message: "Invalid input" },

  // Server
  INTERNAL_ERROR: { status: 500, message: "Server error occurred" },
} as const;

type ErrorCode = keyof typeof ErrorCodes;
```

### 3.3 Hierarchical Error Code Management

```typescript
// Hierarchical error code management
const ERROR_REGISTRY = {
    // ========== Authentication & Authorization ==========
    AUTH: {
        UNAUTHENTICATED: {
            httpStatus: 401,
            message: "Authentication required",
            retryable: false,
            userMessage: "Please log in",
        },
        TOKEN_EXPIRED: {
            httpStatus: 401,
            message: "Token has expired",
            retryable: true,
            userMessage: "Your session has expired. Please log in again",
        },
        INVALID_TOKEN: {
            httpStatus: 401,
            message: "Invalid token",
            retryable: false,
            userMessage: "Authentication failed. Please log in again",
        },
        FORBIDDEN: {
            httpStatus: 403,
            message: "Permission denied",
            retryable: false,
            userMessage: "You do not have permission to perform this action",
        },
    },

    // ========== User ==========
    USER: {
        NOT_FOUND: {
            httpStatus: 404,
            message: "User not found",
            retryable: false,
            userMessage: "The specified user does not exist",
        },
        EMAIL_DUPLICATE: {
            httpStatus: 409,
            message: "Duplicate email address",
            retryable: false,
            userMessage: "This email address is already registered",
        },
        PROFILE_INCOMPLETE: {
            httpStatus: 400,
            message: "Incomplete profile",
            retryable: false,
            userMessage: "Please fill in the required information",
        },
    },

    // ========== Order ==========
    ORDER: {
        NOT_FOUND: {
            httpStatus: 404,
            message: "Order not found",
            retryable: false,
            userMessage: "The specified order does not exist",
        },
        PAYMENT_FAILED: {
            httpStatus: 400,
            message: "Payment failed",
            retryable: true,
            userMessage: "We could not process your payment. Please try a different payment method",
        },
        ALREADY_CANCELLED: {
            httpStatus: 400,
            message: "Order already cancelled",
            retryable: false,
            userMessage: "This order has already been cancelled",
        },
        STOCK_UNAVAILABLE: {
            httpStatus: 400,
            message: "Out of stock",
            retryable: false,
            userMessage: "Sorry, this item is currently out of stock",
        },
    },

    // ========== System ==========
    SYSTEM: {
        INTERNAL_ERROR: {
            httpStatus: 500,
            message: "Internal error occurred",
            retryable: true,
            userMessage: "A server error occurred. Please wait and try again",
        },
        SERVICE_UNAVAILABLE: {
            httpStatus: 503,
            message: "Service temporarily unavailable",
            retryable: true,
            userMessage: "Currently under maintenance. Please try again later",
        },
        RATE_LIMITED: {
            httpStatus: 429,
            message: "Too many requests",
            retryable: true,
            userMessage: "Request frequency is too high. Please wait and try again",
        },
    },

    // ========== Validation ==========
    VALIDATION: {
        INVALID_INPUT: {
            httpStatus: 400,
            message: "Invalid input",
            retryable: false,
            userMessage: "There are errors in your input. Please check and re-enter",
        },
        MISSING_FIELD: {
            httpStatus: 400,
            message: "Required field is missing",
            retryable: false,
            userMessage: "Please fill in the required fields",
        },
        INVALID_FORMAT: {
            httpStatus: 400,
            message: "Invalid format",
            retryable: false,
            userMessage: "Please enter in the correct format",
        },
    },
} as const;

// Type-safe error code retrieval
type ErrorDomain = keyof typeof ERROR_REGISTRY;
type ErrorCodeOf<D extends ErrorDomain> = keyof typeof ERROR_REGISTRY[D];

function getErrorInfo<D extends ErrorDomain>(
    domain: D,
    code: ErrorCodeOf<D>
): typeof ERROR_REGISTRY[D][ErrorCodeOf<D>] {
    return ERROR_REGISTRY[domain][code];
}

// Usage example
const info = getErrorInfo('AUTH', 'TOKEN_EXPIRED');
// info.httpStatus === 401
// info.message === "Token has expired"
// info.retryable === true
```

---

## 4. Error Design in Rust

### 4.1 Custom Errors with thiserror

```rust
// Rust: Custom errors with the thiserror crate
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

### 4.2 Error Hierarchy

```rust
// Separate error types per domain
use thiserror::Error;

// User domain errors
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

// Order domain errors
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

// Application-wide error (aggregating domain errors)
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

// Response conversion for Actix-web
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

## 5. Custom Errors in Python

### 5.1 Error Hierarchy Design

```python
# Python: Custom exception hierarchy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import traceback

class AppError(Exception):
    """Base class for application errors"""
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
        """Convert to dict for API response"""
        result = {
            "error": {
                "code": self.code,
                "message": str(self),
                "timestamp": self.timestamp,
            }
        }
        return result

    def to_log(self) -> dict:
        """Convert to dict for logging (includes internal information)"""
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


# ========== Authentication ==========
class AuthenticationError(AppError):
    code = "AUTHENTICATION_REQUIRED"
    http_status = 401

    def __init__(self, message: str = "Authentication required", **kwargs):
        super().__init__(message, **kwargs)


class TokenExpiredError(AuthenticationError):
    code = "TOKEN_EXPIRED"

    def __init__(self, expired_at: datetime, **kwargs):
        self.expired_at = expired_at
        super().__init__(f"Token expired ({expired_at.isoformat()})", **kwargs)


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
            f"Insufficient permissions: {required_permission} required",
            **kwargs,
        )


# ========== Resources ==========
class NotFoundError(AppError):
    code = "NOT_FOUND"
    http_status = 404

    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(
            f"{resource_type} not found: {resource_id}",
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
            f"{resource_type} {conflict_field} already exists: {conflict_value}",
            **kwargs,
        )


# ========== Validation ==========
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
        super().__init__(f"Invalid input: {fields}", **kwargs)

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["error"]["details"] = [
            {"field": e.field, "message": e.message, "code": e.code}
            for e in self.field_errors
        ]
        return result


# ========== Business Logic ==========
class InsufficientBalanceError(AppError):
    code = "INSUFFICIENT_BALANCE"
    http_status = 400

    def __init__(self, required: float, available: float, currency: str = "JPY", **kwargs):
        self.required = required
        self.available = available
        self.currency = currency
        super().__init__(
            f"Insufficient balance: {required:,.0f} {currency} required, {available:,.0f} {currency} available",
            **kwargs,
        )


# ========== Usage Examples ==========
def get_user(user_id: str) -> User:
    user = user_repository.find_by_id(user_id)
    if user is None:
        raise NotFoundError("User", user_id)
    return user


def create_user(data: dict) -> User:
    # Validation
    errors = []
    if not data.get("name"):
        errors.append(FieldError(field="name", message="Name is required", code="REQUIRED"))
    if not data.get("email"):
        errors.append(FieldError(field="email", message="Email is required", code="REQUIRED"))
    elif not is_valid_email(data["email"]):
        errors.append(FieldError(field="email", message="Invalid email format", code="INVALID_FORMAT"))

    if errors:
        raise ValidationError(errors)

    # Duplicate check
    existing = user_repository.find_by_email(data["email"])
    if existing:
        raise ConflictError("User", "email", data["email"])

    return user_repository.create(data)
```

---

## 6. Custom Errors in Go

### 6.1 Structured Errors

```go
// Go: Structured custom errors
package apperror

import (
    "fmt"
    "net/http"
    "time"
)

// AppError: Base application error
type AppError struct {
    Code       string         `json:"code"`
    Message    string         `json:"message"`
    HTTPStatus int            `json:"-"`
    Timestamp  time.Time      `json:"timestamp"`
    Details    map[string]any `json:"details,omitempty"`
    Err        error          `json:"-"`  // Internal error (not exposed externally)
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

// Factory functions
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
        Message:    "Invalid input",
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
        Message:    fmt.Sprintf("%s %s already exists: %s", resourceType, field, value),
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
        message = "Authentication required"
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
        Message:    fmt.Sprintf("Insufficient permissions: %s required", requiredPermission),
        HTTPStatus: http.StatusForbidden,
        Timestamp:  time.Now(),
        Details: map[string]any{
            "required_permission": requiredPermission,
        },
    }
}

// Error checking
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

// Convert to HTTP response
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

## 7. Error Design Principles

### 7.1 Fundamental Principles

```
1. Be specific with errors
   x throw new Error("Error occurred")
   o throw new UserNotFoundError(userId)

2. Include context in errors
   x "Not found"
   o "User not found: user-123"
   o { code: "USER_NOT_FOUND", userId: "user-123" }

3. Separate user-facing and developer-facing messages
   User: "Login failed"
   Developer: "Auth0 returned 429: rate limit exceeded for IP 192.168.1.1"

4. Error cause chains
   -> Make root causes traceable
   -> Error.cause (ES2022), Rust's source(), Go's %w

5. Errors should be immutable
   -> Do not modify state after creation
   -> Safe to pass around, safe to log
```

### 7.2 Error Message Design

```
Guidelines for developer-facing messages:

  1. What (what happened)
     -> "Database connection refused"
     -> "JSON parse error at position 42"

  2. Where (where it happened)
     -> "in UserService.createUser"
     -> "while processing order ORD-123"

  3. Why (why it happened, suspected cause)
     -> "connection pool exhausted (max: 10, active: 10)"
     -> "unexpected field 'naem' (did you mean 'name'?)"

  4. How (how to resolve it)
     -> "retry after 5 seconds"
     -> "check database connection settings"
     -> "contact support with error code ERR-123"

Guidelines for user-facing messages:

  1. State what happened concisely
     -> "Login failed"
     -> "Could not process your order"

  2. Be specific about what to do
     -> "Please check your password and try again"
     -> "Please try a different payment method"

  3. Avoid technical jargon
     x "A NullPointerException occurred"
     o "An unexpected error occurred"

  4. Do not blame the user
     x "Invalid email address"
     o "Please check the email address format"
```

### 7.3 Error Cause Chains

```typescript
// ES2022: Error.cause
class ServiceError extends Error {
    constructor(message: string, options?: { cause?: Error }) {
        super(message, options);
    }
}

// Building a cause chain
async function processOrder(orderId: string): Promise<Order> {
    try {
        const order = await orderRepository.findById(orderId);
        if (!order) throw new NotFoundError("Order", orderId);

        const payment = await paymentService.charge(order);
        return { ...order, paymentId: payment.id };
    } catch (error) {
        if (error instanceof NotFoundError) throw error;

        // Build the cause chain
        throw new ServiceError(
            `Failed to process order ${orderId}`,
            { cause: error as Error }
        );
    }
}

// Traversing the cause chain
function getAllCauses(error: Error): Error[] {
    const causes: Error[] = [error];
    let current: unknown = error.cause;
    while (current instanceof Error) {
        causes.push(current);
        current = current.cause;
    }
    return causes;
}

// Output all causes when logging
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

## 8. Error Serialization and API Design

### 8.1 Standardizing API Error Responses

```typescript
// RFC 7807 compliant error response
interface ApiErrorResponse {
    type: string;        // Error type URI
    title: string;       // Human-readable summary
    status: number;      // HTTP status code
    detail?: string;     // Detailed description
    instance?: string;   // URI of the resource where the error occurred
    errors?: FieldError[];  // Validation error details
    retryAfter?: number;    // Seconds until retry
    requestId?: string;     // Request ID
}

// Generating error responses
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

### 8.2 Error Design in GraphQL

```typescript
// GraphQL: Error design
// In GraphQL, errors are expressed in the response body rather than relying on HTTP status codes

// Schema definition
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

  # Or an approach using extensions
`;

// Resolvers
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
                throw error;  // Unexpected errors propagate as GraphQL errors
            }
        },
    },
};

// formatError: Formatting GraphQL errors
const server = new ApolloServer({
    typeDefs,
    resolvers,
    formatError: (formattedError, error) => {
        // Hide internal error information
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

        // Unexpected errors
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

## 9. Error Internationalization (i18n)

### 9.1 Multilingual Error Messages

```typescript
// Error message internationalization
const ERROR_MESSAGES: Record<string, Record<string, string>> = {
    "USER_NOT_FOUND": {
        en: "User not found: {userId}",
        ja: "User not found: {userId}",
        zh: "User not found: {userId}",
    },
    "EMAIL_ALREADY_EXISTS": {
        en: "Email already registered: {email}",
        ja: "Email already registered: {email}",
        zh: "Email already registered: {email}",
    },
    "VALIDATION_REQUIRED": {
        en: "{field} is required",
        ja: "{field} is required",
        zh: "{field} is required",
    },
    "VALIDATION_TOO_LONG": {
        en: "{field} must be {maxLength} characters or less",
        ja: "{field} must be {maxLength} characters or less",
        zh: "{field} must be {maxLength} characters or less",
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

// Usage example
const message = getLocalizedMessage(
    "USER_NOT_FOUND",
    "ja",
    { userId: "user-123" }
);
// "User not found: user-123"

// Locale retrieval in API middleware
function getLocale(req: Request): string {
    // 1. Query parameter
    if (req.query.locale) return req.query.locale as string;

    // 2. Accept-Language header
    const acceptLanguage = req.headers['accept-language'];
    if (acceptLanguage) {
        const preferred = acceptLanguage.split(',')[0].split('-')[0].trim();
        if (['en', 'ja', 'zh'].includes(preferred)) return preferred;
    }

    // 3. Default
    return 'en';
}

// Integration into error responses
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

## 10. Testing Patterns

### 10.1 Testing Custom Errors

```typescript
// Unit tests for custom errors
describe("NotFoundError", () => {
    it("sets the correct properties", () => {
        const error = new NotFoundError("User", "user-123");

        expect(error).toBeInstanceOf(AppError);
        expect(error).toBeInstanceOf(NotFoundError);
        expect(error.code).toBe("NOT_FOUND");
        expect(error.httpStatus).toBe(404);
        expect(error.message).toBe("User not found: user-123");
        expect(error.resourceType).toBe("User");
        expect(error.resourceId).toBe("user-123");
        expect(error.isOperational).toBe(true);
        expect(error.timestamp).toBeDefined();
    });

    it("toResponse() returns the correct format", () => {
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

    it("includes a stack trace", () => {
        const error = new NotFoundError("User", "user-123");
        expect(error.stack).toBeDefined();
        expect(error.stack).toContain("NotFoundError");
    });
});

describe("ValidationError", () => {
    it("can be built using the builder pattern", () => {
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

    it("buildIfErrors() returns null when there are no errors", () => {
        const error = ValidationError.builder().buildIfErrors();
        expect(error).toBeNull();
    });
});
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in professional practice?

Knowledge of this topic is frequently used in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Principle | Key Point |
|-----------|-----------|
| Classification | Operational errors vs programmer errors |
| Codes | DOMAIN_ENTITY_ACTION naming convention |
| Context | Include sufficient information in errors |
| Separation | User-facing vs developer-facing messages |
| Chaining | Make root causes traceable |
| Internationalization | Error codes + message templates |
| Testing | Test properties, serialization, and builders |

---

## Recommended Next Guides

---

## References
1. Goldberg, J. "Error Handling in Node.js." joyent.com, 2014.
2. RFC 7807. "Problem Details for HTTP APIs."
3. thiserror crate. Rust Documentation.
4. NestJS Documentation. "Exception Filters."
5. Python Documentation. "Built-in Exceptions."
6. Go Blog. "Working with Errors in Go 1.13."
