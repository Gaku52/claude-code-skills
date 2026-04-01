# API Error Design

> API error responses directly affect the experience of client developers. This guide covers the proper use of HTTP status codes, RFC 7807 Problem Details, and best practices for error response design.

## What You Will Learn

- [ ] Understand the proper use of HTTP status codes
- [ ] Learn the standard format for error responses
- [ ] Learn practical API error design
- [ ] Master validation error design patterns
- [ ] Understand error internationalization (i18n)
- [ ] Understand error design differences in GraphQL/gRPC

## Prerequisites

Before reading this guide, having the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. HTTP Status Codes

### 1.1 Status Code Categories

```
2xx Success:
  200 OK               - General success
  201 Created           - Resource created successfully
  202 Accepted          - Asynchronous processing accepted
  204 No Content        - Success (no response body)
  206 Partial Content   - Partial content (Range specified)

3xx Redirect:
  301 Moved Permanently  - Permanent redirect
  302 Found              - Temporary redirect
  304 Not Modified       - Cache is valid

4xx Client Error:
  400 Bad Request       - Invalid request (syntax error, etc.)
  401 Unauthorized      - Authentication required (not authenticated)
  403 Forbidden         - Not authorized (insufficient permissions)
  404 Not Found         - Resource does not exist
  405 Method Not Allowed - Invalid HTTP method
  406 Not Acceptable    - Cannot satisfy Accept header
  408 Request Timeout   - Request timeout
  409 Conflict          - Conflict (duplicate registration, optimistic lock failure, etc.)
  410 Gone              - Resource permanently deleted
  413 Payload Too Large - Payload size exceeded
  415 Unsupported Media Type - Content-Type not supported
  422 Unprocessable Entity - Validation error
  429 Too Many Requests - Rate limit exceeded

5xx Server Error:
  500 Internal Server Error - Internal server error
  501 Not Implemented       - Unimplemented endpoint
  502 Bad Gateway           - Upstream server error
  503 Service Unavailable   - Service temporarily unavailable
  504 Gateway Timeout       - Upstream server timeout

Decision Criteria:
  Client's mistake -> 4xx
  Server's problem -> 5xx
  Potentially resolved by retry -> 429, 503, 504
```

### 1.2 Common Mistakes

```
Mistake 1: Returning 200 for all errors
  X Bad:
    HTTP 200 OK
    { "success": false, "error": "User not found" }

  O Good:
    HTTP 404 Not Found
    { "type": "...", "title": "Not Found", "status": 404, "detail": "..." }

  Reason: HTTP clients, CDNs, proxies, and monitoring tools
          operate based on status codes

Mistake 2: Confusing 401 and 403
  401 Unauthorized = Not authenticated (not logged in)
    -> Return WWW-Authenticate header
    -> Client resends authentication credentials

  403 Forbidden = Not authorized (no permission)
    -> Re-authentication will not change the result
    -> Request permissions from an administrator

Mistake 3: Misusing 400 and 422
  400 Bad Request = Request syntax is invalid
    -> JSON is broken, required parameters are missing
    -> Problems at the parsing level

  422 Unprocessable Entity = Syntax is correct but semantically invalid
    -> Email address format is incorrect
    -> Number is out of range
    -> Business rule violation

Mistake 4: Overusing 500
  -> Use 500 only for "unexpected errors"
  -> Returning validation errors as 500 is incorrect
  -> Choose an appropriate 4xx code

Mistake 5: Security risk of 404
  -> Can reveal the existence of a resource
  -> In some cases, return 403 (to hide resource existence)
  -> Example: /api/admin/users -> return 403 if no permission (not 404)
```

### 1.3 Status Code Selection Flowchart

```
Request received
  |-- Cannot parse JSON? -> 400 Bad Request
  |-- Auth token missing/invalid? -> 401 Unauthorized
  |-- Insufficient permissions? -> 403 Forbidden
  |-- Resource not found? -> 404 Not Found
  |-- Invalid HTTP method? -> 405 Method Not Allowed
  |-- Rate limit exceeded? -> 429 Too Many Requests
  |-- Validation error?
  |   |-- Missing required parameter -> 400 Bad Request
  |   +-- Semantically invalid value -> 422 Unprocessable Entity
  |-- Conflict (duplicate, optimistic lock failure)? -> 409 Conflict
  |-- Processing succeeded?
  |   |-- Resource created -> 201 Created
  |   |-- Async accepted -> 202 Accepted
  |   |-- No response body -> 204 No Content
  |   +-- Other -> 200 OK
  +-- Internal server error -> 500 Internal Server Error
```

---

## 2. Error Response Format

### 2.1 RFC 7807 Problem Details

```json
// RFC 7807 Problem Details (recommended)
{
  "type": "https://api.example.com/errors/validation",
  "title": "Validation Error",
  "status": 422,
  "detail": "There are issues with the input values",
  "instance": "/api/users",
  "errors": [
    {
      "field": "email",
      "message": "Please enter a valid email address"
    },
    {
      "field": "password",
      "message": "Must be at least 8 characters"
    }
  ],
  "traceId": "abc-123-def"
}
```

```
RFC 7807 Fields:

  type (required):
    -> URI that identifies the error type
    -> Useful to make it the URL of a documentation page
    -> Example: "https://api.example.com/errors/validation"
    -> Default: "about:blank"

  title (required):
    -> Human-readable error title
    -> Short description corresponding to the type
    -> Example: "Validation Error"

  status (recommended):
    -> HTTP status code
    -> Should match the response header
    -> Example: 422

  detail (recommended):
    -> Detailed description of the error
    -> Information specific to this request
    -> Example: "There are issues with the input values"

  instance (optional):
    -> Path of the request where the error occurred
    -> Useful for debugging
    -> Example: "/api/users"

  Extension fields (optional):
    -> RFC 7807 is extensible
    -> Additional fields like errors, traceId, timestamp can be added
```

### 2.2 TypeScript Type Definitions and Implementation

```typescript
// Error response type definition
interface ApiError {
  type: string;          // Error type (URL or code)
  title: string;         // Human-readable title
  status: number;        // HTTP status
  detail: string;        // Detailed message
  instance?: string;     // Request path
  traceId?: string;      // Tracing ID
  timestamp?: string;    // Occurrence time
  errors?: FieldError[]; // Field-level errors
}

interface FieldError {
  field: string;
  message: string;
  code?: string;
  rejectedValue?: unknown;
}

// Base class for application errors
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

// Specific error classes
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
    message: string = 'There are issues with the input values',
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
  constructor(message: string = 'Authentication required') {
    super('UNAUTHORIZED', 401, message);
  }
}

class ForbiddenError extends AppError {
  constructor(message: string = 'You do not have permission to perform this action') {
    super('FORBIDDEN', 403, message);
  }
}

class RateLimitError extends AppError {
  constructor(
    public readonly retryAfterSeconds: number,
    message: string = 'Request rate limit exceeded',
  ) {
    super('RATE_LIMIT_EXCEEDED', 429, message, { retryAfterSeconds });
  }
}

class InternalError extends AppError {
  constructor(
    message: string = 'A server error occurred',
    public readonly cause?: Error,
  ) {
    super('INTERNAL_ERROR', 500, message);
  }
}
```

### 2.3 Express Error Middleware

```typescript
// Express middleware
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

    // Add field information for validation errors
    if (err instanceof ValidationError) {
      response.errors = err.fields;
    }

    // Add Retry-After header for rate limit errors
    if (err instanceof RateLimitError) {
      res.setHeader('Retry-After', String(err.retryAfterSeconds));
    }

    // Log output
    if (err.statusCode >= 500) {
      logger.error({ err, traceId, path: req.originalUrl }, 'Server error');
    } else if (err.statusCode >= 400) {
      logger.warn({ err, traceId, path: req.originalUrl }, 'Client error');
    }

    res.status(err.statusCode).json(response);
  } else {
    // Unexpected error (hide internal details)
    logger.error(
      { err, traceId, path: req.originalUrl, stack: err.stack },
      'Unexpected error',
    );

    res.status(500).json({
      type: 'https://api.example.com/errors/internal',
      title: 'Internal Server Error',
      status: 500,
      detail: 'A server error occurred',
      instance: req.originalUrl,
      traceId,
      timestamp: new Date().toISOString(),
    });
  }
}

// Register middleware
app.use(errorHandler);

// 404 handler
app.use((req: Request, res: Response) => {
  res.status(404).json({
    type: 'https://api.example.com/errors/not_found',
    title: 'Not Found',
    status: 404,
    detail: `${req.method} ${req.originalUrl} does not exist`,
    instance: req.originalUrl,
    timestamp: new Date().toISOString(),
  });
});
```

### 2.4 Error Design in NestJS

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
          : (exceptionResponse as any).message ?? 'An error occurred',
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
        detail: 'A server error occurred',
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

## 3. Validation Error Design

### 3.1 Field-Level Errors

```typescript
// Detailed design for validation errors
interface DetailedFieldError {
  field: string;         // Field path (dot notation for nested fields)
  code: string;          // Error code (machine-readable)
  message: string;       // Human-readable message
  rejectedValue?: unknown; // Rejected value (be mindful of security)
  constraints?: Record<string, unknown>; // Constraint conditions
}

// Example: Validation error for user registration
const validationErrorExample: ApiError = {
  type: 'https://api.example.com/errors/validation',
  title: 'Validation Error',
  status: 422,
  detail: 'There are 3 validation errors',
  instance: '/api/users',
  traceId: 'trace-abc-123',
  timestamp: '2025-01-15T10:30:00Z',
  errors: [
    {
      field: 'email',
      code: 'INVALID_FORMAT',
      message: 'Please enter a valid email address',
      rejectedValue: 'invalid-email',
      constraints: { pattern: '^[^@]+@[^@]+\\.[^@]+$' },
    },
    {
      field: 'password',
      code: 'TOO_SHORT',
      message: 'Must be at least 8 characters',
      constraints: { minLength: 8 },
    },
    {
      field: 'profile.age',
      code: 'OUT_OF_RANGE',
      message: 'Please enter a value between 0 and 130',
      rejectedValue: -1,
      constraints: { min: 0, max: 130 },
    },
  ],
};
```

### 3.2 Integration with Validation Libraries

```typescript
// Integration with Zod
import { z } from 'zod';

const CreateUserSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(8, 'Must be at least 8 characters'),
  name: z.string().min(1, 'Please enter a name').max(100),
  profile: z.object({
    age: z.number().int().min(0).max(130).optional(),
    bio: z.string().max(500).optional(),
  }).optional(),
});

// Convert Zod errors to API errors
function zodToFieldErrors(error: z.ZodError): FieldError[] {
  return error.errors.map((issue) => ({
    field: issue.path.join('.'),
    code: issue.code.toUpperCase(),
    message: issue.message,
  }));
}

// Validation middleware
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

// Usage example
app.post('/api/users', validate(CreateUserSchema), async (req, res) => {
  const user = await userService.create(req.body);
  res.status(201).json(user);
});
```

```python
# Python: Integration with Pydantic
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
            raise ValueError('Must contain at least one uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Must contain at least one digit')
        return v


# Convert Pydantic validation errors to RFC 7807
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
            'detail': f'{len(errors)} validation error(s) found',
            'instance': str(request.url.path),
            'errors': errors,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        },
    )


@app.post('/api/users', status_code=201)
async def create_user(user: CreateUserRequest):
    return await user_service.create(user)
```

### 3.3 Business Rule Validation

```typescript
// Business rule validation
class OrderValidator {
  async validate(order: CreateOrderInput): Promise<FieldError[]> {
    const errors: FieldError[] = [];

    // Stock check
    for (const item of order.items) {
      const stock = await this.stockService.getAvailable(item.productId);
      if (stock < item.quantity) {
        errors.push({
          field: `items[${item.productId}].quantity`,
          code: 'INSUFFICIENT_STOCK',
          message: `Insufficient stock (${stock} remaining)`,
          rejectedValue: item.quantity,
          constraints: { available: stock },
        });
      }
    }

    // Order amount check
    const total = order.items.reduce(
      (sum, item) => sum + item.price * item.quantity, 0,
    );
    if (total > 1_000_000) {
      errors.push({
        field: 'total',
        code: 'AMOUNT_EXCEEDS_LIMIT',
        message: 'A single order must be 1,000,000 yen or less',
        rejectedValue: total,
        constraints: { maxAmount: 1_000_000 },
      });
    }

    // Shipping address check
    if (order.shippingAddress) {
      const isDeliverable = await this.shippingService.isDeliverable(
        order.shippingAddress.zipCode,
      );
      if (!isDeliverable) {
        errors.push({
          field: 'shippingAddress.zipCode',
          code: 'UNDELIVERABLE_AREA',
          message: 'Delivery to this postal code is not supported',
          rejectedValue: order.shippingAddress.zipCode,
        });
      }
    }

    return errors;
  }
}

// Usage in controller
app.post('/api/orders', async (req, res) => {
  // Syntax validation (Zod)
  const input = CreateOrderSchema.parse(req.body);

  // Business rule validation
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

## 4. Error Design Best Practices

### 4.1 Design Principles

```
1. Consistency
   -> Same error format across all endpoints
   -> Unified use of status codes
   -> Content-Type: application/problem+json (RFC 7807)

2. Security
   -> Do not leak internal information in 500 errors
   -> Stack traces hidden in production
   -> Do not differentiate between "user does not exist" and "wrong password"
   -> Do not return SQL error details
   -> Do not return internal class names or file paths

3. Machine Readability
   -> Error codes as strings (enum-compatible)
   -> Combination of HTTP status and error codes
   -> type field links to documentation

4. Human Readability
   -> Specific messages in the detail field
   -> Field-level validation errors
   -> Messages displayable to end users

5. Retryability Indication
   -> 429: Retry-After header
   -> 503: Retry-After header
   -> Enable retry decisions based on error codes

6. Debugging Ease
   -> Track requests with traceId
   -> Track timeline with timestamp
   -> Identify endpoints with instance
```

### 4.2 Error Code System

```typescript
// Systematic error code design
const ERROR_CODES = {
  // Authentication & Authorization
  AUTH_TOKEN_EXPIRED: { status: 401, title: 'Token Expired' },
  AUTH_TOKEN_INVALID: { status: 401, title: 'Invalid Token' },
  AUTH_INSUFFICIENT_PERMISSIONS: { status: 403, title: 'Insufficient Permissions' },

  // Validation
  VALIDATION_FAILED: { status: 422, title: 'Validation Failed' },
  VALIDATION_REQUIRED_FIELD: { status: 422, title: 'Required Field Missing' },
  VALIDATION_INVALID_FORMAT: { status: 422, title: 'Invalid Format' },

  // Resources
  RESOURCE_NOT_FOUND: { status: 404, title: 'Resource Not Found' },
  RESOURCE_ALREADY_EXISTS: { status: 409, title: 'Resource Already Exists' },
  RESOURCE_CONFLICT: { status: 409, title: 'Resource Conflict' },
  RESOURCE_GONE: { status: 410, title: 'Resource Gone' },

  // Rate Limiting
  RATE_LIMIT_EXCEEDED: { status: 429, title: 'Rate Limit Exceeded' },

  // Business Logic
  BUSINESS_INSUFFICIENT_BALANCE: { status: 422, title: 'Insufficient Balance' },
  BUSINESS_ORDER_LIMIT_EXCEEDED: { status: 422, title: 'Order Limit Exceeded' },
  BUSINESS_ACCOUNT_SUSPENDED: { status: 403, title: 'Account Suspended' },

  // Server Errors
  INTERNAL_ERROR: { status: 500, title: 'Internal Server Error' },
  SERVICE_UNAVAILABLE: { status: 503, title: 'Service Unavailable' },
  UPSTREAM_ERROR: { status: 502, title: 'Upstream Service Error' },
} as const;

type ErrorCode = keyof typeof ERROR_CODES;

// Build ApiError from error code
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

### 4.3 Security Considerations

```typescript
// Security-conscious error responses

// Authentication error: do not reveal user existence
app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;

  const user = await userService.findByEmail(email);

  // X Bad: reveals user existence
  // if (!user) throw new NotFoundError('User', email);
  // if (!bcrypt.compareSync(password, user.password)) throw new Error('Wrong password');

  // O Good: return the same message
  if (!user || !await bcrypt.compare(password, user.passwordHash)) {
    throw new UnauthorizedError('Email address or password is incorrect');
  }

  // Timing attack mitigation
  // Perform hash comparison even when user is not found
  const dummyHash = '$2b$10$dummyhashfortimingattackprevention';
  if (!user) {
    await bcrypt.compare(password, dummyHash); // Equalize processing time
    throw new UnauthorizedError('Email address or password is incorrect');
  }
});

// 500 error: hide internal information
function sanitizeError(err: Error, isProduction: boolean): ApiError {
  if (isProduction) {
    return {
      type: 'https://api.example.com/errors/internal',
      title: 'Internal Server Error',
      status: 500,
      detail: 'A server error occurred. Please try again later.',
      // Do not include stack traces, SQL queries, file paths, etc.
    };
  }

  // Return detailed information in development environments
  return {
    type: 'https://api.example.com/errors/internal',
    title: 'Internal Server Error',
    status: 500,
    detail: err.message,
    // Include additional information only in development environments
    ...(isProduction ? {} : {
      stack: err.stack,
      cause: err.cause ? String(err.cause) : undefined,
    }),
  };
}

// Rate limit error information disclosure
// X Bad: exposes rate limit details
// { "detail": "100 requests per minute exceeded. Current: 105" }

// O Good: minimum necessary information
// { "detail": "Request rate limit exceeded", "retryAfter": 30 }
```

---

## 5. Error Internationalization (i18n)

### 5.1 Multi-Language Support Design

```typescript
// Error message internationalization

// Message catalog
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
    'VALIDATION_FAILED': 'There are issues with the input values',
    'VALIDATION_REQUIRED': '{field} is required',
    'VALIDATION_TOO_SHORT': '{field} must be at least {min} characters',
    'VALIDATION_TOO_LONG': '{field} must be at most {max} characters',
    'VALIDATION_INVALID_EMAIL': 'Please enter a valid email address',
    'NOT_FOUND': '{resource} not found',
    'UNAUTHORIZED': 'Authentication required',
    'FORBIDDEN': 'You do not have permission to perform this action',
    'RATE_LIMIT': 'Request rate limit exceeded. Please try again later.',
    'INTERNAL_ERROR': 'A server error occurred. Please try again later.',
  },
};

// Field name translations
const fieldNames: Record<string, Record<string, string>> = {
  en: {
    'email': 'Email',
    'password': 'Password',
    'name': 'Name',
    'age': 'Age',
  },
  ja: {
    'email': 'Email address',
    'password': 'Password',
    'name': 'Name',
    'age': 'Age',
  },
};

// Message resolution
function resolveMessage(
  code: string,
  locale: string,
  params: Record<string, string | number> = {},
): string {
  const messages = errorMessages[locale] ?? errorMessages['en'];
  let template = messages[code] ?? messages['INTERNAL_ERROR'];

  // Replace placeholders
  for (const [key, value] of Object.entries(params)) {
    template = template.replace(`{${key}}`, String(value));
  }

  return template;
}

// Determine locale from Accept-Language header
function getLocale(req: Request): string {
  const acceptLanguage = req.headers['accept-language'];
  if (!acceptLanguage) return 'en';

  // Simple parsing
  const preferred = acceptLanguage.split(',')[0].split(';')[0].trim().substring(0, 2);
  return errorMessages[preferred] ? preferred : 'en';
}

// i18n-enabled error middleware
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

## 6. Client-Side Error Handling

### 6.1 TypeScript HTTP Client

```typescript
// API client error handling
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

      // Handle 204 No Content
      if (response.status === 204) {
        return undefined as T;
      }

      return response.json();
    } catch (error) {
      if (error instanceof ApiRequestError) {
        throw error;
      }

      // Network error
      throw new NetworkError(
        'There is a problem with the network connection',
        error as Error,
      );
    }
  }
}

// API request error
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

// Usage example in a React component
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
          // Display field-level errors on the form
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
        setGlobalError('Please check your network connection');
      } else {
        setGlobalError('An unexpected error occurred');
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

      <button type="submit">Register</button>
    </form>
  );
}
```

---

## 7. GraphQL Error Design

### 7.1 Characteristics of GraphQL Errors

```
GraphQL errors differ from REST:

  REST:
    -> Indicates error type via HTTP status codes
    -> Includes error details in the body
    -> 1 request = 1 response

  GraphQL:
    -> Always returns HTTP 200 (the query itself succeeded)
    -> Returns errors in the errors field
    -> Partial success is possible (data and errors coexist)
    -> A single request can contain multiple queries
```

```typescript
// GraphQL error response example
const graphqlErrorResponse = {
  data: {
    user: { id: '1', name: 'Taro Tanaka' },
    orders: null, // Failed to retrieve due to error
  },
  errors: [
    {
      message: 'Failed to retrieve order information',
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

// Error definitions in Apollo Server
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

// Error usage in resolvers
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

## 8. gRPC Error Design

### 8.1 gRPC Status Codes

```
gRPC Status Codes:
  OK (0)              - Success
  CANCELLED (1)       - Cancelled by client
  UNKNOWN (2)         - Unknown error
  INVALID_ARGUMENT (3) - Invalid argument
  DEADLINE_EXCEEDED (4) - Deadline exceeded
  NOT_FOUND (5)       - Resource does not exist
  ALREADY_EXISTS (6)   - Resource already exists
  PERMISSION_DENIED (7) - No permission
  RESOURCE_EXHAUSTED (8) - Resource exhausted
  FAILED_PRECONDITION (9) - Precondition mismatch
  ABORTED (10)        - Operation aborted (transaction conflict, etc.)
  OUT_OF_RANGE (11)    - Out of range
  UNIMPLEMENTED (12)   - Not implemented
  INTERNAL (13)        - Internal error
  UNAVAILABLE (14)     - Service unavailable
  DATA_LOSS (15)       - Data loss
  UNAUTHENTICATED (16) - Not authenticated

Mapping to HTTP Status:
  INVALID_ARGUMENT   <-> 400 Bad Request
  UNAUTHENTICATED    <-> 401 Unauthorized
  PERMISSION_DENIED  <-> 403 Forbidden
  NOT_FOUND          <-> 404 Not Found
  ALREADY_EXISTS     <-> 409 Conflict
  RESOURCE_EXHAUSTED <-> 429 Too Many Requests
  INTERNAL           <-> 500 Internal Server Error
  UNAVAILABLE        <-> 503 Service Unavailable
  DEADLINE_EXCEEDED  <-> 504 Gateway Timeout
```

```protobuf
// gRPC error details (google.rpc.Status)
syntax = "proto3";

import "google/rpc/status.proto";
import "google/rpc/error_details.proto";

// Response containing error details
message ErrorResponse {
  google.rpc.Status status = 1;
}

// Validation error details
// Using google.rpc.BadRequest
message BadRequest {
  repeated FieldViolation field_violations = 1;

  message FieldViolation {
    string field = 1;
    string description = 2;
  }
}
```

```go
// Go: Sending gRPC errors
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
	// Validation error details
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

## 9. Automatic Error Documentation Generation

### 9.1 Error Definition in OpenAPI

```yaml
# OpenAPI 3.0: Error response definition
openapi: "3.0.0"
info:
  title: Example API
  version: "1.0.0"

paths:
  /api/users:
    post:
      summary: Create user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: User created successfully
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
          description: URI identifying the error type
          example: "https://api.example.com/errors/validation"
        title:
          type: string
          description: Error title
          example: "Validation Error"
        status:
          type: integer
          description: HTTP status code
          example: 422
        detail:
          type: string
          description: Error details
          example: "There are issues with the input values"
        instance:
          type: string
          description: Request path
          example: "/api/users"
        traceId:
          type: string
          description: Tracing ID
          example: "abc-123-def"
        timestamp:
          type: string
          format: date-time
          description: Error occurrence time
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
          description: Field with the error
          example: "email"
        code:
          type: string
          description: Error code
          example: "INVALID_FORMAT"
        message:
          type: string
          description: Error message
          example: "Please enter a valid email address"

  responses:
    BadRequest:
      description: Invalid request
      content:
        application/problem+json:
          schema:
            $ref: '#/components/schemas/ProblemDetail'
    ValidationError:
      description: Validation error
      content:
        application/problem+json:
          schema:
            $ref: '#/components/schemas/ProblemDetail'
    Conflict:
      description: Resource conflict
      content:
        application/problem+json:
          schema:
            $ref: '#/components/schemas/ProblemDetail'
    InternalError:
      description: Server error
      content:
        application/problem+json:
          schema:
            $ref: '#/components/schemas/ProblemDetail'
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Create test code as well

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main logic for data processing"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient: {slow_time:.4f}s")
    print(f"Efficient:   {fast_time:.6f}s")
    print(f"Speedup:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be conscious of algorithm complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify the path and format of configuration files |
| Timeout | Network latency/resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Increased data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access permissions | Verify user permissions, review configuration |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Steps

1. **Check error messages**: Read the stack trace and identify the location of occurrence
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Form hypotheses**: List possible causes
4. **Verify step by step**: Use log output or debuggers to verify hypotheses
5. **Fix and regression test**: After fixing, run tests on related areas as well

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function input and output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Steps for diagnosing performance issues:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Check the status of disk and network I/O
4. **Check concurrent connections**: Check the status of the connection pool

| Problem Type | Diagnostic Tool | Countermeasure |
|-------------|-----------------|----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |
---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Principle | Key Point |
|-----------|-----------|
| Status Codes | Choose the correct code (4xx vs 5xx) |
| Format | Comply with RFC 7807 Problem Details |
| Security | Do not leak internal information |
| Consistency | Unified across all endpoints |
| Validation | Detailed field-level errors |
| Retry | Retry-After header |
| Internationalization | Accept-Language support |
| Documentation | Define errors in OpenAPI |
| Error Codes | Systematic code design |
| Client DX | User-friendly error responses |

---

## Recommended Next Guides

---

## References
1. RFC 7807. "Problem Details for HTTP APIs." IETF, 2016.
2. RFC 9457. "Problem Details for HTTP APIs." IETF, 2023. (Successor to RFC 7807)
3. Fielding, R. "REST APIs must be hypertext-driven." 2008.
4. Google Cloud API Design Guide. "Errors." cloud.google.com.
5. Microsoft REST API Guidelines. "Error Handling." github.com/microsoft.
6. GraphQL Specification. "Errors." spec.graphql.org.
7. gRPC Error Handling. "Status codes and their use." grpc.io.
8. Zalando RESTful API Guidelines. "Error Handling." opensource.zalando.com.
9. Stripe API Reference. "Errors." stripe.com/docs/api/errors.
10. Twitter API Documentation. "Error Handling." developer.twitter.com.
