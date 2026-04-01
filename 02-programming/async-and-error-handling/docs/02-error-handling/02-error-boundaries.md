# Error Boundaries

> Error boundaries are a mechanism for "localizing the impact of errors." Understand React Error Boundaries, global error handlers, and process-level error handling.

## What You Will Learn in This Chapter

- [ ] Understand the concept of error boundaries and layered design
- [ ] Grasp how to implement React Error Boundaries
- [ ] Learn how to design global error handlers
- [ ] Understand error boundaries in microservices
- [ ] Learn how to integrate error reporting and monitoring
- [ ] Design graduated fallback strategies


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of the content in [Result Types](./01-result-type.md)

---

## 1. Error Boundary Layers

### 1.1 Layer Model Overview

```
The principle is to handle errors as close to the source as possible.
However, if they cannot be handled, they are caught at a higher layer.

  Layer 4: Process/Application Level
    -> Catching uncaught exceptions
    -> Error reporting (Sentry)
    -> Graceful shutdown

  Layer 3: Middleware/Framework
    -> Unified HTTP error responses
    -> Log output

  Layer 2: Service/Use Case
    -> Business logic error handling
    -> Retry, fallback

  Layer 1: Function/Method
    -> Input validation
    -> Individual try/catch
```

### 1.2 Responsibilities of Each Layer

```
Layer 1: Function/Method Level
  Responsibilities:
    -> Input value validation
    -> Catching exceptions from individual operations
    -> Converting to appropriate error types
  Do NOT:
    -> Swallow all exceptions silently
    -> Handle concerns belonging to higher layers
    -> Just log and ignore

Layer 2: Service/Use Case Level
  Responsibilities:
    -> Business logic error determination
    -> Retry logic
    -> Fallback strategies
    -> Transaction management
  Do NOT:
    -> Directly construct HTTP responses
    -> Control UI display
    -> Return infrastructure-specific errors directly

Layer 3: Middleware/Framework Level
  Responsibilities:
    -> Unified error response format
    -> Determining HTTP status codes
    -> Attaching request IDs
    -> Outputting access logs
  Do NOT:
    -> Make business logic decisions
    -> Branch in detail for individual error cases

Layer 4: Process/Application Level
  Responsibilities:
    -> Final catch for uncaught exceptions
    -> Sending to error reporting services
    -> Graceful shutdown
    -> Responding to health checks
  Do NOT:
    -> Attempt to recover individual errors
    -> Execute business logic
```

### 1.3 Error Propagation Flow

```
Error Propagation Flow (Practical Example):

  1. DB query timeout occurs
     Layer 1: repository.findById()
     -> Throws DatabaseTimeoutError

  2. Caught at service layer
     Layer 2: userService.getUser()
     -> Retry #1: Timeout again
     -> Retry #2: Timeout again
     -> Wraps in ServiceUnavailableError and re-throws

  3. Caught at middleware
     Layer 3: errorMiddleware()
     -> HTTP 503 Service Unavailable response
     -> Attaches Retry-After header
     -> Structured log output

  4. If middleware also fails to catch
     Layer 4: process.on('uncaughtException')
     -> Reports to Sentry
     -> Begins graceful shutdown
```

---

## 2. React Error Boundary

### 2.1 Basic Error Boundary

```tsx
// React: Error Boundary (requires a class component)
import React, { Component, ReactNode } from 'react';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    // Send to error reporting service
    console.error('Error Boundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div>
          <h2>An error occurred</h2>
          <button onClick={() => this.setState({ hasError: false, error: null })}>
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// Usage: Localizing the impact of errors
function App() {
  return (
    <div>
      <Header /> {/* Header always displays */}
      <ErrorBoundary fallback={<p>Failed to load sidebar</p>}>
        <Sidebar /> {/* Sidebar errors do not affect others */}
      </ErrorBoundary>
      <ErrorBoundary fallback={<p>Failed to load main content</p>}>
        <MainContent /> {/* Main content errors are also localized */}
      </ErrorBoundary>
    </div>
  );
}
```

### 2.2 Advanced Error Boundary

```tsx
// More practical Error Boundary implementation
import React, { Component, ReactNode, ErrorInfo } from 'react';
import * as Sentry from '@sentry/react';

interface ErrorBoundaryProps {
    children: ReactNode;
    fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
    onError?: (error: Error, errorInfo: ErrorInfo) => void;
    onReset?: () => void;
    resetKeys?: unknown[];  // Reset when these values change
    level?: 'page' | 'section' | 'component';
}

interface ErrorBoundaryState {
    hasError: boolean;
    error: Error | null;
    errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
    state: ErrorBoundaryState = {
        hasError: false,
        error: null,
        errorInfo: null,
    };

    static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
        this.setState({ errorInfo });

        // Custom error handler
        this.props.onError?.(error, errorInfo);

        // Report to Sentry
        Sentry.captureException(error, {
            contexts: {
                react: {
                    componentStack: errorInfo.componentStack,
                },
            },
            tags: {
                errorBoundaryLevel: this.props.level ?? 'component',
            },
        });
    }

    componentDidUpdate(prevProps: ErrorBoundaryProps): void {
        // Reset error state when resetKeys change
        if (
            this.state.hasError &&
            this.props.resetKeys &&
            prevProps.resetKeys &&
            !arraysEqual(this.props.resetKeys, prevProps.resetKeys)
        ) {
            this.resetErrorBoundary();
        }
    }

    resetErrorBoundary = (): void => {
        this.props.onReset?.();
        this.setState({ hasError: false, error: null, errorInfo: null });
    };

    render(): ReactNode {
        if (this.state.hasError && this.state.error) {
            // Function-type fallback (passes error info and reset function)
            if (typeof this.props.fallback === 'function') {
                return this.props.fallback(this.state.error, this.resetErrorBoundary);
            }

            // ReactNode-type fallback
            if (this.props.fallback) {
                return this.props.fallback;
            }

            // Default fallback
            return (
                <div role="alert" className="error-boundary-fallback">
                    <h2>An unexpected error occurred</h2>
                    <p>Please reload the page or try again after a moment.</p>
                    <details>
                        <summary>Error details (for development)</summary>
                        <pre>{this.state.error.message}</pre>
                        <pre>{this.state.error.stack}</pre>
                    </details>
                    <button onClick={this.resetErrorBoundary}>
                        Retry
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}

function arraysEqual(a: unknown[], b: unknown[]): boolean {
    if (a.length !== b.length) return false;
    return a.every((val, idx) => Object.is(val, b[idx]));
}

// ========== Usage Examples ==========

// Error Boundary placement by level
function App() {
    return (
        // App-wide Error Boundary (last resort)
        <ErrorBoundary
            level="page"
            fallback={(error, reset) => (
                <FullPageError error={error} onRetry={reset} />
            )}
        >
            <Layout>
                {/* Section-level Error Boundary */}
                <ErrorBoundary
                    level="section"
                    fallback={<SectionErrorFallback />}
                >
                    <DashboardWidgets />
                </ErrorBoundary>

                {/* Component-level Error Boundary */}
                <ErrorBoundary
                    level="component"
                    fallback={<p>Failed to load notifications</p>}
                >
                    <NotificationPanel />
                </ErrorBoundary>
            </Layout>
        </ErrorBoundary>
    );
}
```

### 2.3 react-error-boundary Library

```tsx
// react-error-boundary: Officially recommended library
import { ErrorBoundary, useErrorBoundary } from 'react-error-boundary';

// Basic usage
function App() {
    return (
        <ErrorBoundary
            FallbackComponent={ErrorFallback}
            onError={(error, info) => {
                // Error reporting
                reportError(error, info);
            }}
            onReset={(details) => {
                // Processing on reset (e.g., refetch data)
                queryClient.invalidateQueries();
            }}
            resetKeys={[userId]}  // Reset when userId changes
        >
            <UserProfile userId={userId} />
        </ErrorBoundary>
    );
}

// FallbackComponent
function ErrorFallback({
    error,
    resetErrorBoundary,
}: {
    error: Error;
    resetErrorBoundary: () => void;
}) {
    return (
        <div role="alert">
            <h2>Something went wrong</h2>
            <p>{error.message}</p>
            <button onClick={resetErrorBoundary}>Retry</button>
        </div>
    );
}

// useErrorBoundary hook: Explicitly throw errors from child components
function UserProfile({ userId }: { userId: string }) {
    const { showBoundary } = useErrorBoundary();

    const handleClick = async () => {
        try {
            await deleteUser(userId);
        } catch (error) {
            // Errors in event handlers are NOT caught by Error Boundaries
            // Use showBoundary to explicitly pass the error to the Error Boundary
            showBoundary(error);
        }
    };

    return <button onClick={handleClick}>Delete User</button>;
}

// withErrorBoundary HOC
const SafeComponent = withErrorBoundary(DangerousComponent, {
    FallbackComponent: ErrorFallback,
    onError: reportError,
});
```

### 2.4 Error Boundary Caveats

```
Errors that Error Boundaries do NOT catch:

  1. Event handlers
     -> Errors in onClick, onChange, etc. are not caught
     -> Solution: Use the useErrorBoundary() hook

  2. Asynchronous code
     -> Errors in setTimeout, Promise are not caught
     -> Solution: Use the useErrorBoundary() hook

  3. Server-side rendering (SSR)
     -> Error Boundaries are client-side only
     -> Solution: Handle errors separately on the server side

  4. Errors in the Error Boundary itself
     -> When the Error Boundary's own rendering throws an error
     -> Solution: A parent Error Boundary catches it

Error Boundary placement strategy:

  Choosing granularity:
  -> Too coarse: One for the entire app -> A trivial error causes full-page fallback
  -> Too fine: Wrapping every component -> Bloated code, fragmented UX

  Recommended:
  -> App level: One (last resort)
  -> Route/page level: One per page
  -> Section level: One per independent data source
  -> Component level: Parts that can fail without affecting others
```

### 2.5 Integration with Suspense

```tsx
// Combining Error Boundary and Suspense
import { Suspense } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

// React 18+: Suspense + Error Boundary pattern
function UserDashboard({ userId }: { userId: string }) {
    return (
        <ErrorBoundary
            FallbackComponent={ErrorFallback}
            resetKeys={[userId]}
        >
            <Suspense fallback={<DashboardSkeleton />}>
                <UserStats userId={userId} />
            </Suspense>
            <Suspense fallback={<OrdersSkeleton />}>
                <RecentOrders userId={userId} />
            </Suspense>
        </ErrorBoundary>
    );
}

// Integration with TanStack Query (React Query)
import { QueryErrorResetBoundary } from '@tanstack/react-query';

function DataSection() {
    return (
        <QueryErrorResetBoundary>
            {({ reset }) => (
                <ErrorBoundary
                    onReset={reset}
                    FallbackComponent={ErrorFallback}
                >
                    <Suspense fallback={<Loading />}>
                        <DataComponent />
                    </Suspense>
                </ErrorBoundary>
            )}
        </QueryErrorResetBoundary>
    );
}

// useQuery with suspense + throwOnError
function DataComponent() {
    const { data } = useQuery({
        queryKey: ['data'],
        queryFn: fetchData,
        throwOnError: true,  // Propagate errors to Error Boundary
    });

    return <div>{data}</div>;
}
```

---

## 3. Server-Side Error Boundaries

### 3.1 Express Error Middleware

```typescript
// Express: Global error middleware
import express, { Request, Response, NextFunction } from 'express';

const app = express();

// Route handler
app.get('/api/users/:id', async (req, res, next) => {
  try {
    const user = await userService.getUser(req.params.id);
    res.json(user);
  } catch (error) {
    next(error); // Delegate to error middleware
  }
});

// Error boundary: Global error handler
app.use((error: Error, req: Request, res: Response, next: NextFunction) => {
  // Branch response based on error type
  if (error instanceof ValidationError) {
    res.status(400).json({
      type: "validation_error",
      message: error.message,
      fields: error.fields,
    });
  } else if (error instanceof NotFoundError) {
    res.status(404).json({
      type: "not_found",
      message: error.message,
    });
  } else if (error instanceof AuthError) {
    res.status(401).json({
      type: "unauthorized",
      message: "Authentication required",
    });
  } else {
    // Unexpected error
    console.error("Unexpected error:", error);
    // Send to Sentry
    res.status(500).json({
      type: "internal_error",
      message: "A server error occurred",
    });
  }
});
```

### 3.2 Fastify Error Handling

```typescript
// Fastify: Schema-based error handling
import Fastify from 'fastify';

const fastify = Fastify({ logger: true });

// Register error handler
fastify.setErrorHandler(async (error, request, reply) => {
    const requestId = request.id;

    // Fastify validation error
    if (error.validation) {
        return reply.status(400).send({
            error: {
                code: 'VALIDATION_ERROR',
                message: 'Request validation failed',
                details: error.validation.map(v => ({
                    field: v.params?.missingProperty || v.instancePath,
                    message: v.message,
                })),
                requestId,
            }
        });
    }

    // Custom error
    if (error instanceof AppError) {
        request.log.warn({
            code: error.code,
            message: error.message,
        });

        return reply.status(error.httpStatus).send({
            error: {
                code: error.code,
                message: error.message,
                requestId,
            }
        });
    }

    // Unexpected error
    request.log.error({
        err: error,
        message: 'Unhandled error',
    });

    Sentry.captureException(error, {
        tags: { requestId },
    });

    return reply.status(500).send({
        error: {
            code: 'INTERNAL_ERROR',
            message: 'A server error occurred',
            requestId,
        }
    });
});

// 404 handler
fastify.setNotFoundHandler(async (request, reply) => {
    return reply.status(404).send({
        error: {
            code: 'NOT_FOUND',
            message: `${request.method} ${request.url} was not found`,
        }
    });
});
```

### 3.3 NestJS Error Filters

```typescript
// NestJS: Exception Filter
import {
    ExceptionFilter,
    Catch,
    ArgumentsHost,
    HttpException,
    HttpStatus,
} from '@nestjs/common';

// Filter that catches all exceptions
@Catch()
export class GlobalExceptionFilter implements ExceptionFilter {
    constructor(
        private readonly logger: LoggerService,
        private readonly sentry: SentryService,
    ) {}

    catch(exception: unknown, host: ArgumentsHost): void {
        const ctx = host.switchToHttp();
        const response = ctx.getResponse();
        const request = ctx.getRequest();
        const requestId = request.headers['x-request-id'] || generateId();

        if (exception instanceof HttpException) {
            // Standard NestJS HTTP exception
            const status = exception.getStatus();
            const exceptionResponse = exception.getResponse();

            this.logger.warn({
                status,
                message: exception.message,
                requestId,
                path: request.url,
            });

            response.status(status).json({
                error: {
                    code: this.getErrorCode(status),
                    message: typeof exceptionResponse === 'string'
                        ? exceptionResponse
                        : (exceptionResponse as any).message,
                    requestId,
                    timestamp: new Date().toISOString(),
                }
            });
        } else if (exception instanceof AppError) {
            // Custom application error
            this.logger.warn({
                code: exception.code,
                message: exception.message,
                requestId,
            });

            response.status(exception.httpStatus).json({
                error: {
                    code: exception.code,
                    message: exception.message,
                    requestId,
                    timestamp: new Date().toISOString(),
                }
            });
        } else {
            // Unexpected error
            this.logger.error({
                error: exception,
                requestId,
                path: request.url,
            });

            this.sentry.captureException(exception);

            response.status(HttpStatus.INTERNAL_SERVER_ERROR).json({
                error: {
                    code: 'INTERNAL_ERROR',
                    message: 'A server error occurred',
                    requestId,
                    timestamp: new Date().toISOString(),
                }
            });
        }
    }

    private getErrorCode(status: number): string {
        const codeMap: Record<number, string> = {
            400: 'BAD_REQUEST',
            401: 'UNAUTHORIZED',
            403: 'FORBIDDEN',
            404: 'NOT_FOUND',
            409: 'CONFLICT',
            422: 'UNPROCESSABLE_ENTITY',
            429: 'TOO_MANY_REQUESTS',
        };
        return codeMap[status] || 'HTTP_ERROR';
    }
}

// Register in main.ts
async function bootstrap() {
    const app = await NestFactory.create(AppModule);
    app.useGlobalFilters(new GlobalExceptionFilter(logger, sentry));
    await app.listen(3000);
}
```

### 3.4 Python (FastAPI) Error Handling

```python
# FastAPI: Exception handlers
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

app = FastAPI()

# Custom errors
class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 500):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class NotFoundError(AppError):
    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            code="NOT_FOUND",
            message=f"{resource} not found: {resource_id}",
            status_code=404,
        )

# AppError handler
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": request.state.request_id,
            }
        },
    )

# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid input values",
                "details": [
                    {
                        "field": ".".join(str(loc) for loc in err["loc"]),
                        "message": err["msg"],
                        "type": err["type"],
                    }
                    for err in exc.errors()
                ],
            }
        },
    )

# Generic exception handler (last resort)
@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "A server error occurred",
            }
        },
    )

# Middleware to attach request ID
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    import uuid
    request.state.request_id = request.headers.get(
        "x-request-id", str(uuid.uuid4())
    )
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    return response
```

### 3.5 Go Error Middleware

```go
// Go (Gin): Error middleware
package middleware

import (
    "net/http"
    "github.com/gin-gonic/gin"
    "github.com/getsentry/sentry-go"
)

// Error response struct
type ErrorResponse struct {
    Error struct {
        Code      string `json:"code"`
        Message   string `json:"message"`
        RequestID string `json:"request_id,omitempty"`
    } `json:"error"`
}

// Global error handler
func ErrorHandler() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Next()

        // Check for errors after handler execution
        if len(c.Errors) > 0 {
            err := c.Errors.Last().Err
            requestID := c.GetString("request_id")

            switch e := err.(type) {
            case *NotFoundError:
                c.JSON(http.StatusNotFound, ErrorResponse{
                    Error: struct {
                        Code      string `json:"code"`
                        Message   string `json:"message"`
                        RequestID string `json:"request_id,omitempty"`
                    }{
                        Code:      "NOT_FOUND",
                        Message:   e.Error(),
                        RequestID: requestID,
                    },
                })
            case *ValidationError:
                c.JSON(http.StatusBadRequest, gin.H{
                    "error": gin.H{
                        "code":       "VALIDATION_ERROR",
                        "message":    e.Error(),
                        "details":    e.Fields,
                        "request_id": requestID,
                    },
                })
            default:
                // Unexpected error
                sentry.CaptureException(err)
                c.JSON(http.StatusInternalServerError, ErrorResponse{
                    Error: struct {
                        Code      string `json:"code"`
                        Message   string `json:"message"`
                        RequestID string `json:"request_id,omitempty"`
                    }{
                        Code:      "INTERNAL_ERROR",
                        Message:   "A server error occurred",
                        RequestID: requestID,
                    },
                })
            }
        }
    }
}

// Panic recovery middleware
func RecoveryHandler() gin.HandlerFunc {
    return func(c *gin.Context) {
        defer func() {
            if r := recover(); r != nil {
                sentry.CurrentHub().Recover(r)
                c.JSON(http.StatusInternalServerError, ErrorResponse{
                    Error: struct {
                        Code      string `json:"code"`
                        Message   string `json:"message"`
                        RequestID string `json:"request_id,omitempty"`
                    }{
                        Code:    "PANIC_RECOVERED",
                        Message: "A server error occurred",
                    },
                })
                c.Abort()
            }
        }()
        c.Next()
    }
}
```

---

## 4. Process-Level Error Handling

### 4.1 Node.js Process Error Handling

```typescript
// Node.js: Uncaught exceptions and unhandled rejections
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  // Error reporting
  // Graceful shutdown
  process.exit(1); // Must exit
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection:', reason);
  // In Node.js 15+, terminates similarly to uncaughtException
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received. Graceful shutdown...');
  await server.close();
  await db.disconnect();
  process.exit(0);
});
```

### 4.2 Complete Graceful Shutdown Implementation

```typescript
// Production-quality graceful shutdown
class GracefulShutdown {
    private isShuttingDown = false;
    private shutdownTimeout = 30_000;  // 30 seconds
    private cleanupTasks: Array<() => Promise<void>> = [];

    constructor(private readonly server: Server) {
        this.setupHandlers();
    }

    register(task: () => Promise<void>): void {
        this.cleanupTasks.push(task);
    }

    private setupHandlers(): void {
        // SIGTERM (stop signal from Docker, Kubernetes)
        process.on('SIGTERM', () => this.shutdown('SIGTERM'));

        // SIGINT (Ctrl+C)
        process.on('SIGINT', () => this.shutdown('SIGINT'));

        // Uncaught exceptions
        process.on('uncaughtException', (error) => {
            logger.fatal('Uncaught Exception:', error);
            Sentry.captureException(error);
            this.shutdown('uncaughtException', 1);
        });

        // Unhandled Promise rejections
        process.on('unhandledRejection', (reason) => {
            logger.fatal('Unhandled Rejection:', reason);
            Sentry.captureException(reason);
            this.shutdown('unhandledRejection', 1);
        });
    }

    private async shutdown(signal: string, exitCode: number = 0): Promise<void> {
        if (this.isShuttingDown) {
            logger.warn(`Already shutting down. Ignoring ${signal}`);
            return;
        }
        this.isShuttingDown = true;
        logger.info(`${signal} received. Starting graceful shutdown...`);

        // Forced exit timer
        const forceExitTimer = setTimeout(() => {
            logger.error('Forced shutdown: cleanup timed out');
            process.exit(1);
        }, this.shutdownTimeout);
        forceExitTimer.unref();  // Prevent timer from blocking process exit

        try {
            // 1. Stop accepting new requests
            logger.info('Stopping HTTP server...');
            await new Promise<void>((resolve, reject) => {
                this.server.close((err) => {
                    if (err) reject(err);
                    else resolve();
                });
            });
            logger.info('HTTP server stopped');

            // 2. Wait for in-flight requests to complete
            // (server.close() waits for in-flight requests to finish)

            // 3. Run cleanup tasks
            logger.info('Running cleanup tasks...');
            for (const task of this.cleanupTasks) {
                try {
                    await task();
                } catch (error) {
                    logger.error('Cleanup task failed:', error);
                }
            }
            logger.info('Cleanup completed');

            // 4. Flush Sentry (send buffered events)
            await Sentry.flush(5000);

        } catch (error) {
            logger.error('Error during shutdown:', error);
            exitCode = 1;
        } finally {
            clearTimeout(forceExitTimer);
            logger.info(`Exiting with code ${exitCode}`);
            process.exit(exitCode);
        }
    }
}

// Usage example
const server = app.listen(3000);
const shutdown = new GracefulShutdown(server);

// Register cleanup tasks
shutdown.register(async () => {
    logger.info('Closing database connections...');
    await db.disconnect();
});

shutdown.register(async () => {
    logger.info('Closing Redis connections...');
    await redis.quit();
});

shutdown.register(async () => {
    logger.info('Closing message queue connections...');
    await messageQueue.close();
});
```

### 4.3 Integration with Kubernetes Health Checks

```typescript
// Kubernetes: Health check endpoints
class HealthCheck {
    private isReady = false;
    private isLive = true;
    private checks: Map<string, () => Promise<boolean>> = new Map();

    registerCheck(name: string, check: () => Promise<boolean>): void {
        this.checks.set(name, check);
    }

    setReady(ready: boolean): void {
        this.isReady = ready;
    }

    setLive(live: boolean): void {
        this.isLive = live;
    }

    setupRoutes(app: Express): void {
        // Liveness Probe: Is the process running normally?
        app.get('/healthz', (req, res) => {
            if (this.isLive) {
                res.status(200).json({ status: 'ok' });
            } else {
                res.status(503).json({ status: 'not healthy' });
            }
        });

        // Readiness Probe: Is it ready to accept requests?
        app.get('/readyz', async (req, res) => {
            if (!this.isReady) {
                return res.status(503).json({ status: 'not ready' });
            }

            // Check each dependent service
            const results: Record<string, boolean> = {};
            for (const [name, check] of this.checks) {
                try {
                    results[name] = await check();
                } catch {
                    results[name] = false;
                }
            }

            const allHealthy = Object.values(results).every(Boolean);
            res.status(allHealthy ? 200 : 503).json({
                status: allHealthy ? 'ready' : 'not ready',
                checks: results,
            });
        });
    }
}

// Usage example
const health = new HealthCheck();

health.registerCheck('database', async () => {
    try {
        await db.query('SELECT 1');
        return true;
    } catch {
        return false;
    }
});

health.registerCheck('redis', async () => {
    try {
        await redis.ping();
        return true;
    } catch {
        return false;
    }
});

health.setupRoutes(app);

// After application startup is complete
health.setReady(true);

// When shutdown begins
// health.setReady(false);
// -> Kubernetes routes traffic to other Pods
```

---

## 5. Error Boundary Design Principles

### 5.1 Graduated Fallback

```
1. Graduated fallback
   -> Component level -> Page level -> App level

2. Informing the user
   -> Tell them what happened and what they can do
   -> Hide technical details (security)

3. Error logging and reporting
   -> Output logs at every layer
   -> Send to Sentry etc. in production

4. Providing recovery options
   -> Retry button
   -> Guide to alternative actions
   -> Last resort: Page reload
```

### 5.2 Fallback Strategy Patterns

```typescript
// Pattern 1: Cache fallback
async function getUserWithFallback(id: string): Promise<User> {
    try {
        // Primary: Fetch from API
        const user = await apiClient.getUser(id);
        await cache.set(`user:${id}`, user, { ttl: 300 });
        return user;
    } catch (error) {
        // Fallback 1: Fetch from cache
        const cached = await cache.get(`user:${id}`);
        if (cached) {
            logger.warn(`Using cached data for user ${id}`);
            return cached;
        }

        // Fallback 2: Default value
        logger.error(`No data available for user ${id}`);
        return {
            id,
            name: 'Unknown User',
            isStale: true,
        };
    }
}

// Pattern 2: Graduated degradation
async function loadDashboard(userId: string): Promise<DashboardData> {
    const results = await Promise.allSettled([
        fetchUserStats(userId),
        fetchRecentOrders(userId),
        fetchNotifications(userId),
        fetchRecommendations(userId),
    ]);

    return {
        // Required data: Error if failed
        stats: unwrapOrThrow(results[0], 'Failed to load stats'),

        // Important data: Default if failed
        orders: unwrapOrDefault(results[1], []),

        // Supplementary data: Hide if failed
        notifications: unwrapOrDefault(results[2], null),
        recommendations: unwrapOrDefault(results[3], null),
    };
}

function unwrapOrThrow<T>(
    result: PromiseSettledResult<T>,
    message: string
): T {
    if (result.status === 'fulfilled') return result.value;
    throw new Error(message, { cause: result.reason });
}

function unwrapOrDefault<T>(
    result: PromiseSettledResult<T>,
    defaultValue: T
): T {
    if (result.status === 'fulfilled') return result.value;
    logger.warn('Degraded mode:', result.reason);
    return defaultValue;
}

// Pattern 3: Circuit breaker
class CircuitBreaker {
    private failures = 0;
    private lastFailure: number = 0;
    private state: 'closed' | 'open' | 'half-open' = 'closed';

    constructor(
        private readonly threshold: number = 5,
        private readonly resetTimeout: number = 60_000,
    ) {}

    async execute<T>(fn: () => Promise<T>): Promise<T> {
        if (this.state === 'open') {
            if (Date.now() - this.lastFailure > this.resetTimeout) {
                this.state = 'half-open';
            } else {
                throw new CircuitOpenError('Service unavailable');
            }
        }

        try {
            const result = await fn();
            this.onSuccess();
            return result;
        } catch (error) {
            this.onFailure();
            throw error;
        }
    }

    private onSuccess(): void {
        this.failures = 0;
        this.state = 'closed';
    }

    private onFailure(): void {
        this.failures++;
        this.lastFailure = Date.now();
        if (this.failures >= this.threshold) {
            this.state = 'open';
            logger.warn('Circuit breaker opened');
        }
    }
}

// Usage example
const paymentCircuit = new CircuitBreaker(5, 30_000);

async function processPayment(order: Order): Promise<PaymentResult> {
    try {
        return await paymentCircuit.execute(() =>
            paymentGateway.charge(order.total, order.paymentMethod)
        );
    } catch (error) {
        if (error instanceof CircuitOpenError) {
            // Payment service is down
            await queueForLaterProcessing(order);
            return { status: 'queued', message: 'Will be processed later' };
        }
        throw error;
    }
}
```

---

## 6. Error Boundaries in Microservices

### 6.1 Error Propagation Between Services

```
Error boundaries in microservices:

  Client -> API Gateway -> Service A -> Service B -> Database
                                    |
                              Service C -> External API

  Error propagation rules:
  1. Do not leak internal errors externally
     -> Service B's DB error should not reach the Client directly
     -> Each service converts errors

  2. Maintain appropriate error granularity
     -> Aggregate downstream service errors
     -> Return minimal necessary information upstream

  3. Timeouts and retries
     -> Set timeouts for inter-service communication
     -> Only retry idempotent operations

  4. Circuit breakers
     -> Block calls to failing services
     -> Execute fallback processing
```

### 6.2 API Gateway Error Aggregation

```typescript
// API Gateway: Error conversion and aggregation
class ApiGateway {
    private circuits = new Map<string, CircuitBreaker>();

    async handleRequest(req: GatewayRequest): Promise<GatewayResponse> {
        const startTime = Date.now();

        try {
            // Routing
            const service = this.resolveService(req.path);
            const circuit = this.getCircuit(service.name);

            // Request via circuit breaker
            const response = await circuit.execute(async () => {
                return await this.forwardRequest(service, req, {
                    timeout: 5000,
                    retries: service.isIdempotent ? 2 : 0,
                });
            });

            return response;

        } catch (error) {
            // Error conversion
            return this.convertToGatewayError(error, {
                path: req.path,
                method: req.method,
                duration: Date.now() - startTime,
            });
        }
    }

    private convertToGatewayError(
        error: unknown,
        context: RequestContext
    ): GatewayResponse {
        if (error instanceof CircuitOpenError) {
            return {
                status: 503,
                body: {
                    error: {
                        code: 'SERVICE_UNAVAILABLE',
                        message: 'Service temporarily unavailable',
                        retryAfter: 30,
                    }
                }
            };
        }

        if (error instanceof TimeoutError) {
            return {
                status: 504,
                body: {
                    error: {
                        code: 'GATEWAY_TIMEOUT',
                        message: 'Request timed out',
                    }
                }
            };
        }

        // Pass through downstream service error response
        if (error instanceof UpstreamError) {
            // However, strip internal information
            return {
                status: error.status,
                body: {
                    error: {
                        code: error.code,
                        message: error.publicMessage,
                        // error.internalDetails is NOT included
                    }
                }
            };
        }

        // Unexpected error
        logger.error('Gateway error:', error, context);
        return {
            status: 500,
            body: {
                error: {
                    code: 'INTERNAL_ERROR',
                    message: 'A server error occurred',
                }
            }
        };
    }
}
```

### 6.3 Distributed Tracing and Error Tracking

```typescript
// Integration with OpenTelemetry
import { trace, SpanStatusCode } from '@opentelemetry/api';

const tracer = trace.getTracer('user-service');

async function getUser(id: string): Promise<User> {
    return tracer.startActiveSpan('getUser', async (span) => {
        try {
            span.setAttribute('user.id', id);
            const user = await userRepository.findById(id);

            if (!user) {
                span.setStatus({
                    code: SpanStatusCode.ERROR,
                    message: 'User not found',
                });
                throw new NotFoundError('User', id);
            }

            span.setStatus({ code: SpanStatusCode.OK });
            return user;

        } catch (error) {
            span.recordException(error as Error);
            span.setStatus({
                code: SpanStatusCode.ERROR,
                message: (error as Error).message,
            });
            throw error;
        } finally {
            span.end();
        }
    });
}

// Error Correlation ID
// -> Propagate the same ID throughout the entire request
// -> Link logs, traces, and error reports
class CorrelationContext {
    private static storage = new AsyncLocalStorage<{
        correlationId: string;
        requestId: string;
        userId?: string;
    }>();

    static run<T>(context: { correlationId: string; requestId: string }, fn: () => T): T {
        return this.storage.run(context, fn);
    }

    static get(): { correlationId: string; requestId: string } | undefined {
        return this.storage.getStore();
    }
}

// Set in middleware
app.use((req, res, next) => {
    const correlationId = req.headers['x-correlation-id'] as string || generateId();
    const requestId = generateId();

    CorrelationContext.run({ correlationId, requestId }, () => {
        res.setHeader('x-correlation-id', correlationId);
        res.setHeader('x-request-id', requestId);
        next();
    });
});
```

---

## 7. Error Reporting

### 7.1 Sentry Integration

```typescript
// Production-grade Sentry configuration
import * as Sentry from '@sentry/node';
import { nodeProfilingIntegration } from '@sentry/profiling-node';

Sentry.init({
    dsn: process.env.SENTRY_DSN,
    environment: process.env.NODE_ENV,
    release: process.env.APP_VERSION,

    // Sampling rate
    tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
    profilesSampleRate: 0.1,

    // Error filtering
    beforeSend(event, hint) {
        const error = hint.originalException;

        // Do not report operational errors (AppError with isOperational = true)
        if (error instanceof AppError && error.isOperational) {
            return null;
        }

        // Remove sensitive information
        if (event.request?.cookies) {
            delete event.request.cookies;
        }

        return event;
    },

    // Breadcrumbs (history of actions leading to the error)
    beforeBreadcrumb(breadcrumb) {
        // Remove SQL query content
        if (breadcrumb.category === 'query') {
            breadcrumb.data = { ...breadcrumb.data, query: '[REDACTED]' };
        }
        return breadcrumb;
    },

    integrations: [
        nodeProfilingIntegration(),
    ],
});

// Attaching custom context
function reportError(error: Error, context: Record<string, unknown> = {}): void {
    Sentry.withScope((scope) => {
        // User information
        if (context.userId) {
            scope.setUser({ id: context.userId as string });
        }

        // Tags (searchable metadata)
        scope.setTag('error_code', (error as any).code || 'UNKNOWN');
        scope.setTag('service', 'user-service');

        // Context (detailed information)
        scope.setContext('request', {
            path: context.path,
            method: context.method,
            requestId: context.requestId,
        });

        // Fingerprint (custom grouping)
        if (error instanceof AppError) {
            scope.setFingerprint([error.code]);
        }

        Sentry.captureException(error);
    });
}
```

### 7.2 Integration with Structured Logging

```typescript
// Integration with structured logging (pino)
import pino from 'pino';

const logger = pino({
    level: process.env.LOG_LEVEL || 'info',
    formatters: {
        level(label) {
            return { level: label };
        },
    },
    mixin() {
        const context = CorrelationContext.get();
        return context ? {
            correlationId: context.correlationId,
            requestId: context.requestId,
        } : {};
    },
    serializers: {
        err: pino.stdSerializers.err,
        // Custom error serializer
        error(error: unknown) {
            if (error instanceof AppError) {
                return {
                    type: error.name,
                    code: error.code,
                    message: error.message,
                    httpStatus: error.httpStatus,
                    isOperational: error.isOperational,
                    stack: error.stack,
                };
            }
            if (error instanceof Error) {
                return {
                    type: error.name,
                    message: error.message,
                    stack: error.stack,
                };
            }
            return { message: String(error) };
        },
    },
});

// Log level usage guide
// fatal: Fatal error that terminates the process
// error: Unexpected error (also sent to Sentry)
// warn: Expected error (operational error)
// info: Record of normal processing
// debug: Debug information
// trace: Detailed trace information
```

---

## 8. RFC 7807: Problem Details for HTTP APIs

### 8.1 Standard Error Format

```typescript
// RFC 7807 compliant error response
interface ProblemDetails {
    type: string;        // URI indicating the type of error
    title: string;       // Human-readable summary of the error
    status: number;      // HTTP status code
    detail?: string;     // Human-readable detailed explanation
    instance?: string;   // URI of the request where this error occurred
    [key: string]: unknown;  // Extension fields
}

// Implementation example
function createProblemDetails(
    error: AppError,
    req: Request
): ProblemDetails {
    const base: ProblemDetails = {
        type: `https://api.example.com/errors/${error.code.toLowerCase()}`,
        title: error.message,
        status: error.httpStatus,
        instance: req.originalUrl,
    };

    // Error-specific extension fields
    if (error instanceof ValidationError) {
        return {
            ...base,
            errors: error.fieldErrors.map(e => ({
                field: e.field,
                message: e.message,
            })),
        };
    }

    if (error instanceof RateLimitError) {
        return {
            ...base,
            retryAfter: error.retryAfterMs / 1000,
        };
    }

    return base;
}

// Express middleware
app.use((error: Error, req: Request, res: Response, next: NextFunction) => {
    if (error instanceof AppError) {
        const problem = createProblemDetails(error, req);
        res.status(problem.status)
            .contentType('application/problem+json')
            .json(problem);
    } else {
        res.status(500)
            .contentType('application/problem+json')
            .json({
                type: 'https://api.example.com/errors/internal-error',
                title: 'Internal Server Error',
                status: 500,
                instance: req.originalUrl,
            });
    }
});
```

---

## 9. Frontend Error Boundaries

### 9.1 Global Error Handlers (Browser)

```typescript
// Browser global error handlers

// JavaScript runtime errors
window.onerror = (message, source, lineno, colno, error) => {
    reportError({
        type: 'runtime_error',
        message: String(message),
        source,
        lineno,
        colno,
        stack: error?.stack,
    });
    // Returning true suppresses default error handling
    return false;
};

// Unhandled Promise rejections
window.addEventListener('unhandledrejection', (event) => {
    reportError({
        type: 'unhandled_rejection',
        reason: event.reason,
        promise: event.promise,
    });
});

// Resource loading errors (images, scripts, etc.)
window.addEventListener('error', (event) => {
    if (event.target instanceof HTMLElement) {
        reportError({
            type: 'resource_error',
            tagName: event.target.tagName,
            src: (event.target as any).src || (event.target as any).href,
        });
    }
}, true);  // Capture in the capture phase

// Network error monitoring
window.addEventListener('offline', () => {
    showNotification('Network connection lost');
});

window.addEventListener('online', () => {
    showNotification('Network connection restored');
});
```

### 9.2 Vue Error Handling

```typescript
// Vue 3: Global error handler
import { createApp } from 'vue';

const app = createApp(App);

// Uncaught errors from all components
app.config.errorHandler = (error, instance, info) => {
    console.error('Vue Error:', error);
    console.error('Component:', instance);
    console.error('Info:', info);

    Sentry.captureException(error, {
        extra: {
            componentName: instance?.$options?.name,
            lifecycleHook: info,
        },
    });
};

// Warning handler (recommended for development only)
app.config.warnHandler = (msg, instance, trace) => {
    console.warn('Vue Warning:', msg);
    console.warn('Trace:', trace);
};

// Component-level error handling
// onErrorCaptured: Functions like an Error Boundary
import { onErrorCaptured, ref } from 'vue';

export default {
    setup() {
        const error = ref<Error | null>(null);

        onErrorCaptured((err, instance, info) => {
            error.value = err;
            // Returning false stops error propagation
            return false;
        });

        return { error };
    },
};
```

### 9.3 Angular Error Handling

```typescript
// Angular: ErrorHandler
import { ErrorHandler, Injectable, NgModule } from '@angular/core';

@Injectable()
class GlobalErrorHandler implements ErrorHandler {
    constructor(
        private logger: LoggingService,
        private notification: NotificationService,
    ) {}

    handleError(error: unknown): void {
        // Handle HttpErrorResponse
        if (error instanceof HttpErrorResponse) {
            this.handleHttpError(error);
            return;
        }

        // Client-side error
        const appError = error instanceof Error ? error : new Error(String(error));

        this.logger.error('Unhandled error', {
            message: appError.message,
            stack: appError.stack,
        });

        Sentry.captureException(appError);

        this.notification.showError(
            'An unexpected error occurred. Please reload the page.'
        );
    }

    private handleHttpError(error: HttpErrorResponse): void {
        switch (error.status) {
            case 0:
                this.notification.showError('Please check your network connection');
                break;
            case 401:
                this.notification.showError('Your session has expired');
                // Redirect
                break;
            case 403:
                this.notification.showError('You do not have permission to access this resource');
                break;
            case 404:
                this.notification.showError('The requested resource was not found');
                break;
            case 429:
                this.notification.showError('Too many requests. Please wait a moment');
                break;
            default:
                this.notification.showError('A server error occurred');
        }
    }
}

@NgModule({
    providers: [
        { provide: ErrorHandler, useClass: GlobalErrorHandler },
    ],
})
export class AppModule {}

// HTTP Interceptor for error handling
@Injectable()
class ErrorInterceptor implements HttpInterceptor {
    intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
        return next.handle(req).pipe(
            retry({ count: 2, delay: 1000 }),  // Retry twice
            catchError((error: HttpErrorResponse) => {
                if (error.status === 401) {
                    // Attempt token refresh
                    return this.authService.refreshToken().pipe(
                        switchMap(() => next.handle(req)),
                    );
                }
                return throwError(() => error);
            }),
        );
    }
}
```

---

## 10. Testing Strategies

### 10.1 Testing Error Boundaries

```tsx
// Testing React Error Boundaries
import { render, screen, fireEvent } from '@testing-library/react';

// Component that intentionally throws an error
function ThrowError({ shouldThrow }: { shouldThrow: boolean }) {
    if (shouldThrow) {
        throw new Error('Test error');
    }
    return <div>Normal content</div>;
}

describe('ErrorBoundary', () => {
    // Suppress console.error output
    const originalError = console.error;
    beforeAll(() => { console.error = jest.fn(); });
    afterAll(() => { console.error = originalError; });

    it('catches errors from child components', () => {
        render(
            <ErrorBoundary fallback={<p>An error occurred</p>}>
                <ThrowError shouldThrow={true} />
            </ErrorBoundary>
        );

        expect(screen.getByText('An error occurred')).toBeInTheDocument();
        expect(screen.queryByText('Normal content')).not.toBeInTheDocument();
    });

    it('displays child components when there is no error', () => {
        render(
            <ErrorBoundary fallback={<p>An error occurred</p>}>
                <ThrowError shouldThrow={false} />
            </ErrorBoundary>
        );

        expect(screen.getByText('Normal content')).toBeInTheDocument();
        expect(screen.queryByText('An error occurred')).not.toBeInTheDocument();
    });

    it('can reset with the retry button', () => {
        const { rerender } = render(
            <ErrorBoundary
                fallback={(error, reset) => (
                    <div>
                        <p>Error: {error.message}</p>
                        <button onClick={reset}>Retry</button>
                    </div>
                )}
            >
                <ThrowError shouldThrow={true} />
            </ErrorBoundary>
        );

        expect(screen.getByText('Error: Test error')).toBeInTheDocument();

        // Change shouldThrow to false and retry
        fireEvent.click(screen.getByText('Retry'));

        // If no error occurs during re-render after reset, normal content displays
    });

    it('calls the onError callback', () => {
        const onError = jest.fn();

        render(
            <ErrorBoundary
                onError={onError}
                fallback={<p>Error</p>}
            >
                <ThrowError shouldThrow={true} />
            </ErrorBoundary>
        );

        expect(onError).toHaveBeenCalledWith(
            expect.any(Error),
            expect.objectContaining({
                componentStack: expect.any(String),
            })
        );
    });
});
```

### 10.2 Testing Error Middleware

```typescript
// Testing Express error middleware
import request from 'supertest';

describe('Error Middleware', () => {
    it('returns 400 for ValidationError', async () => {
        // Throw ValidationError in route handler
        app.get('/test-validation', (req, res, next) => {
            next(new ValidationError([
                { field: 'email', message: 'Required' },
            ]));
        });

        const response = await request(app)
            .get('/test-validation')
            .expect(400);

        expect(response.body.error.code).toBe('VALIDATION_ERROR');
        expect(response.body.error.details).toHaveLength(1);
    });

    it('returns 404 for NotFoundError', async () => {
        app.get('/test-not-found', (req, res, next) => {
            next(new NotFoundError('User', 'user-123'));
        });

        const response = await request(app)
            .get('/test-not-found')
            .expect(404);

        expect(response.body.error.code).toBe('NOT_FOUND');
    });

    it('returns 500 for unexpected errors', async () => {
        app.get('/test-internal', (req, res, next) => {
            next(new Error('Unexpected'));
        });

        const response = await request(app)
            .get('/test-internal')
            .expect(500);

        expect(response.body.error.code).toBe('INTERNAL_ERROR');
        // Verify internal information is not leaked
        expect(response.body.error.message).not.toContain('Unexpected');
    });
});
```

---


## FAQ

### Q1: What is the most important point to focus on when learning this topic?

Gaining hands-on experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts covered in this guide before moving on to the next step.

### Q3: How is this applied in real-world development?

Knowledge of this topic is frequently used in everyday development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Layer | Technique | Purpose |
|-------|-----------|---------|
| Component | Error Boundary | Partial UI errors |
| Middleware | Error handler | Unified HTTP responses |
| Process | uncaughtException | Last resort |
| External service | Sentry | Monitoring |
| Microservices | Circuit breaker | Preventing failure propagation |
| Frontend | Global handler | Collecting uncaught errors |
| API | RFC 7807 | Unified error format |

---

## Recommended Next Guides

---

## References
1. React Documentation. "Error Boundaries."
2. Express.js Documentation. "Error Handling."
3. NestJS Documentation. "Exception Filters."
4. RFC 7807. "Problem Details for HTTP APIs."
5. Sentry Documentation. "JavaScript SDK."
6. Nygard, M. "Release It!" 2nd Edition, 2018.
7. Newman, S. "Building Microservices." 2nd Edition, 2021.
8. react-error-boundary. GitHub.
