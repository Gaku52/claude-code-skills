# エラー境界

> エラー境界は「エラーの影響を局所化する」仕組み。React Error Boundary、グローバルエラーハンドラ、プロセスレベルのエラー処理を理解する。

## この章で学ぶこと

- [ ] エラー境界の概念とレイヤー設計を理解する
- [ ] React Error Boundary の実装を把握する
- [ ] グローバルエラーハンドラの設計を学ぶ
- [ ] マイクロサービスにおけるエラー境界を理解する
- [ ] エラーレポーティングとモニタリングの統合を学ぶ
- [ ] 段階的フォールバック戦略を設計する

---

## 1. エラー境界のレイヤー

### 1.1 レイヤーモデルの概要

```
エラーは発生源に近い場所で処理するのが原則。
ただし、処理できない場合は上位のレイヤーで捕捉する。

  Layer 4: プロセス/アプリレベル
    → 未捕捉例外のキャッチ
    → エラーレポーティング（Sentry）
    → グレースフルシャットダウン

  Layer 3: ミドルウェア/フレームワーク
    → HTTPエラーレスポンスの統一
    → ログ出力

  Layer 2: サービス/ユースケース
    → ビジネスロジックのエラー処理
    → リトライ、フォールバック

  Layer 1: 関数/メソッド
    → 入力バリデーション
    → 個別のtry/catch
```

### 1.2 各レイヤーの責務

```
Layer 1: 関数/メソッドレベル
  責務:
    → 入力値の検証
    → 個別操作の例外キャッチ
    → 適切なエラー型への変換
  やってはいけないこと:
    → 全ての例外を握りつぶす
    → 上位レイヤーの関心事を処理する
    → ログだけ出して無視する

Layer 2: サービス/ユースケースレベル
  責務:
    → ビジネスロジックのエラー判定
    → リトライロジック
    → フォールバック戦略
    → トランザクション管理
  やってはいけないこと:
    → HTTPレスポンスを直接構築する
    → UIの表示を制御する
    → インフラ固有のエラーを直接返す

Layer 3: ミドルウェア/フレームワークレベル
  責務:
    → エラーレスポンスの統一フォーマット
    → HTTPステータスコードの決定
    → リクエストIDの付与
    → アクセスログの出力
  やってはいけないこと:
    → ビジネスロジックの判定をする
    → 個別のエラーケースを詳細に分岐する

Layer 4: プロセス/アプリレベル
  責務:
    → 未捕捉例外の最終キャッチ
    → エラーレポーティングサービスへの送信
    → グレースフルシャットダウン
    → ヘルスチェックの応答
  やってはいけないこと:
    → 個別のエラーを回復しようとする
    → ビジネスロジックを実行する
```

### 1.3 エラーの伝播フロー

```
エラーの伝播フロー（実例）:

  1. DBクエリでタイムアウト発生
     Layer 1: repository.findById()
     → DatabaseTimeoutError をスロー

  2. サービス層でキャッチ
     Layer 2: userService.getUser()
     → リトライ1回目: 再度タイムアウト
     → リトライ2回目: 再度タイムアウト
     → ServiceUnavailableError にラップして再スロー

  3. ミドルウェアでキャッチ
     Layer 3: errorMiddleware()
     → HTTP 503 Service Unavailable レスポンス
     → Retry-After ヘッダー付与
     → 構造化ログ出力

  4. もしミドルウェアでもキャッチできなかった場合
     Layer 4: process.on('uncaughtException')
     → Sentry にレポート
     → グレースフルシャットダウン開始
```

---

## 2. React Error Boundary

### 2.1 基本的な Error Boundary

```tsx
// React: Error Boundary（クラスコンポーネントが必要）
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
    // エラーレポーティングサービスに送信
    console.error('Error Boundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div>
          <h2>エラーが発生しました</h2>
          <button onClick={() => this.setState({ hasError: false, error: null })}>
            再試行
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// 使い方: エラーの影響を局所化
function App() {
  return (
    <div>
      <Header /> {/* ヘッダーは常に表示 */}
      <ErrorBoundary fallback={<p>サイドバーの読み込みに失敗</p>}>
        <Sidebar /> {/* サイドバーのエラーは他に影響しない */}
      </ErrorBoundary>
      <ErrorBoundary fallback={<p>メインコンテンツの読み込みに失敗</p>}>
        <MainContent /> {/* メインのエラーも局所化 */}
      </ErrorBoundary>
    </div>
  );
}
```

### 2.2 高機能な Error Boundary

```tsx
// より実用的な Error Boundary の実装
import React, { Component, ReactNode, ErrorInfo } from 'react';
import * as Sentry from '@sentry/react';

interface ErrorBoundaryProps {
    children: ReactNode;
    fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
    onError?: (error: Error, errorInfo: ErrorInfo) => void;
    onReset?: () => void;
    resetKeys?: unknown[];  // これらの値が変わったらリセット
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

        // カスタムエラーハンドラ
        this.props.onError?.(error, errorInfo);

        // Sentry にレポート
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
        // resetKeys が変わったらエラー状態をリセット
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
            // 関数型 fallback（エラー情報とリセット関数を渡す）
            if (typeof this.props.fallback === 'function') {
                return this.props.fallback(this.state.error, this.resetErrorBoundary);
            }

            // ReactNode 型 fallback
            if (this.props.fallback) {
                return this.props.fallback;
            }

            // デフォルトのフォールバック
            return (
                <div role="alert" className="error-boundary-fallback">
                    <h2>予期しないエラーが発生しました</h2>
                    <p>ページを再読み込みするか、しばらく待ってからお試しください。</p>
                    <details>
                        <summary>エラー詳細（開発用）</summary>
                        <pre>{this.state.error.message}</pre>
                        <pre>{this.state.error.stack}</pre>
                    </details>
                    <button onClick={this.resetErrorBoundary}>
                        再試行
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

// ========== 使用例 ==========

// レベル別の Error Boundary 配置
function App() {
    return (
        // アプリ全体の Error Boundary（最後の砦）
        <ErrorBoundary
            level="page"
            fallback={(error, reset) => (
                <FullPageError error={error} onRetry={reset} />
            )}
        >
            <Layout>
                {/* セクション単位の Error Boundary */}
                <ErrorBoundary
                    level="section"
                    fallback={<SectionErrorFallback />}
                >
                    <DashboardWidgets />
                </ErrorBoundary>

                {/* コンポーネント単位の Error Boundary */}
                <ErrorBoundary
                    level="component"
                    fallback={<p>通知の読み込みに失敗</p>}
                >
                    <NotificationPanel />
                </ErrorBoundary>
            </Layout>
        </ErrorBoundary>
    );
}
```

### 2.3 react-error-boundary ライブラリ

```tsx
// react-error-boundary: 公式推奨のライブラリ
import { ErrorBoundary, useErrorBoundary } from 'react-error-boundary';

// 基本的な使い方
function App() {
    return (
        <ErrorBoundary
            FallbackComponent={ErrorFallback}
            onError={(error, info) => {
                // エラーレポーティング
                reportError(error, info);
            }}
            onReset={(details) => {
                // リセット時の処理（データの再取得など）
                queryClient.invalidateQueries();
            }}
            resetKeys={[userId]}  // userId が変わったらリセット
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
            <h2>問題が発生しました</h2>
            <p>{error.message}</p>
            <button onClick={resetErrorBoundary}>再試行</button>
        </div>
    );
}

// useErrorBoundary フック: 子コンポーネントから明示的にエラーを投げる
function UserProfile({ userId }: { userId: string }) {
    const { showBoundary } = useErrorBoundary();

    const handleClick = async () => {
        try {
            await deleteUser(userId);
        } catch (error) {
            // イベントハンドラのエラーは Error Boundary でキャッチされない
            // showBoundary で明示的に Error Boundary にエラーを伝える
            showBoundary(error);
        }
    };

    return <button onClick={handleClick}>ユーザー削除</button>;
}

// withErrorBoundary HOC
const SafeComponent = withErrorBoundary(DangerousComponent, {
    FallbackComponent: ErrorFallback,
    onError: reportError,
});
```

### 2.4 Error Boundary の注意点

```
Error Boundary がキャッチしないエラー:

  1. イベントハンドラ
     → onClick, onChange などのエラーはキャッチされない
     → 解決: useErrorBoundary() フックを使う

  2. 非同期コード
     → setTimeout, Promise のエラーはキャッチされない
     → 解決: useErrorBoundary() フックを使う

  3. サーバーサイドレンダリング（SSR）
     → Error Boundary はクライアント側のみ
     → 解決: サーバー側で別途エラーハンドリング

  4. Error Boundary 自体のエラー
     → Error Boundary のレンダリングでエラーが起きた場合
     → 解決: 上位の Error Boundary がキャッチ

Error Boundary の配置戦略:

  粒度の選び方:
  → 粗すぎる: アプリ全体が1つ → 些細なエラーで全画面がフォールバック
  → 細かすぎる: 全コンポーネントにラップ → コード膨大、UX が断片的

  推奨:
  → アプリレベル: 1つ（最後の砦）
  → ルート/ページレベル: 各ページに1つ
  → セクションレベル: 独立したデータソースごとに1つ
  → コンポーネントレベル: 失敗しても他に影響しない部分
```

### 2.5 Suspense との統合

```tsx
// Error Boundary と Suspense の組み合わせ
import { Suspense } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

// React 18+: Suspense + Error Boundary パターン
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

// TanStack Query (React Query) との統合
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

// useQuery の suspense + throwOnError
function DataComponent() {
    const { data } = useQuery({
        queryKey: ['data'],
        queryFn: fetchData,
        throwOnError: true,  // エラーを Error Boundary に伝播
    });

    return <div>{data}</div>;
}
```

---

## 3. サーバーサイドのエラー境界

### 3.1 Express のエラーミドルウェア

```typescript
// Express: グローバルエラーミドルウェア
import express, { Request, Response, NextFunction } from 'express';

const app = express();

// ルートハンドラ
app.get('/api/users/:id', async (req, res, next) => {
  try {
    const user = await userService.getUser(req.params.id);
    res.json(user);
  } catch (error) {
    next(error); // エラーミドルウェアに委譲
  }
});

// エラー境界: グローバルエラーハンドラ
app.use((error: Error, req: Request, res: Response, next: NextFunction) => {
  // エラーの種類に応じてレスポンスを分岐
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
      message: "認証が必要です",
    });
  } else {
    // 予期しないエラー
    console.error("Unexpected error:", error);
    // Sentry に送信
    res.status(500).json({
      type: "internal_error",
      message: "サーバーエラーが発生しました",
    });
  }
});
```

### 3.2 Fastify のエラーハンドリング

```typescript
// Fastify: スキーマベースのエラーハンドリング
import Fastify from 'fastify';

const fastify = Fastify({ logger: true });

// エラーハンドラの登録
fastify.setErrorHandler(async (error, request, reply) => {
    const requestId = request.id;

    // Fastify のバリデーションエラー
    if (error.validation) {
        return reply.status(400).send({
            error: {
                code: 'VALIDATION_ERROR',
                message: 'リクエストの検証に失敗しました',
                details: error.validation.map(v => ({
                    field: v.params?.missingProperty || v.instancePath,
                    message: v.message,
                })),
                requestId,
            }
        });
    }

    // カスタムエラー
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

    // 予期しないエラー
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
            message: 'サーバーエラーが発生しました',
            requestId,
        }
    });
});

// 404 ハンドラ
fastify.setNotFoundHandler(async (request, reply) => {
    return reply.status(404).send({
        error: {
            code: 'NOT_FOUND',
            message: `${request.method} ${request.url} は見つかりません`,
        }
    });
});
```

### 3.3 NestJS のエラーフィルター

```typescript
// NestJS: Exception Filter
import {
    ExceptionFilter,
    Catch,
    ArgumentsHost,
    HttpException,
    HttpStatus,
} from '@nestjs/common';

// 全例外をキャッチするフィルター
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
            // NestJS の標準 HTTP 例外
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
            // カスタムアプリケーションエラー
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
            // 予期しないエラー
            this.logger.error({
                error: exception,
                requestId,
                path: request.url,
            });

            this.sentry.captureException(exception);

            response.status(HttpStatus.INTERNAL_SERVER_ERROR).json({
                error: {
                    code: 'INTERNAL_ERROR',
                    message: 'サーバーエラーが発生しました',
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

// main.ts で登録
async function bootstrap() {
    const app = await NestFactory.create(AppModule);
    app.useGlobalFilters(new GlobalExceptionFilter(logger, sentry));
    await app.listen(3000);
}
```

### 3.4 Python (FastAPI) のエラーハンドリング

```python
# FastAPI: 例外ハンドラ
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

app = FastAPI()

# カスタムエラー
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

# AppError ハンドラ
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

# バリデーションエラーハンドラ
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "入力値が不正です",
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

# 汎用例外ハンドラ（最後の砦）
@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "サーバーエラーが発生しました",
            }
        },
    )

# ミドルウェアでリクエストIDを付与
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

### 3.5 Go のエラーミドルウェア

```go
// Go (Gin): エラーミドルウェア
package middleware

import (
    "net/http"
    "github.com/gin-gonic/gin"
    "github.com/getsentry/sentry-go"
)

// エラーレスポンスの構造体
type ErrorResponse struct {
    Error struct {
        Code      string `json:"code"`
        Message   string `json:"message"`
        RequestID string `json:"request_id,omitempty"`
    } `json:"error"`
}

// グローバルエラーハンドラ
func ErrorHandler() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Next()

        // ハンドラ実行後にエラーをチェック
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
                // 予期しないエラー
                sentry.CaptureException(err)
                c.JSON(http.StatusInternalServerError, ErrorResponse{
                    Error: struct {
                        Code      string `json:"code"`
                        Message   string `json:"message"`
                        RequestID string `json:"request_id,omitempty"`
                    }{
                        Code:      "INTERNAL_ERROR",
                        Message:   "サーバーエラーが発生しました",
                        RequestID: requestID,
                    },
                })
            }
        }
    }
}

// パニックリカバリーミドルウェア
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
                        Message: "サーバーエラーが発生しました",
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

## 4. プロセスレベルのエラー処理

### 4.1 Node.js のプロセスエラーハンドリング

```typescript
// Node.js: 未捕捉例外とunhandled rejection
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  // エラーレポーティング
  // グレースフルシャットダウン
  process.exit(1); // 必ず終了
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection:', reason);
  // Node.js 15+ では uncaughtException と同様に終了
});

// グレースフルシャットダウン
process.on('SIGTERM', async () => {
  console.log('SIGTERM received. Graceful shutdown...');
  await server.close();
  await db.disconnect();
  process.exit(0);
});
```

### 4.2 グレースフルシャットダウンの完全実装

```typescript
// プロダクション品質のグレースフルシャットダウン
class GracefulShutdown {
    private isShuttingDown = false;
    private shutdownTimeout = 30_000;  // 30秒
    private cleanupTasks: Array<() => Promise<void>> = [];

    constructor(private readonly server: Server) {
        this.setupHandlers();
    }

    register(task: () => Promise<void>): void {
        this.cleanupTasks.push(task);
    }

    private setupHandlers(): void {
        // SIGTERM（Docker, Kubernetes からの停止シグナル）
        process.on('SIGTERM', () => this.shutdown('SIGTERM'));

        // SIGINT（Ctrl+C）
        process.on('SIGINT', () => this.shutdown('SIGINT'));

        // 未捕捉例外
        process.on('uncaughtException', (error) => {
            logger.fatal('Uncaught Exception:', error);
            Sentry.captureException(error);
            this.shutdown('uncaughtException', 1);
        });

        // 未処理の Promise rejection
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

        // 強制終了タイマー
        const forceExitTimer = setTimeout(() => {
            logger.error('Forced shutdown: cleanup timed out');
            process.exit(1);
        }, this.shutdownTimeout);
        forceExitTimer.unref();  // タイマーがプロセス終了を妨げないように

        try {
            // 1. 新しいリクエストの受付を停止
            logger.info('Stopping HTTP server...');
            await new Promise<void>((resolve, reject) => {
                this.server.close((err) => {
                    if (err) reject(err);
                    else resolve();
                });
            });
            logger.info('HTTP server stopped');

            // 2. 進行中のリクエストの完了を待機
            // （server.close() は進行中のリクエストが完了するのを待つ）

            // 3. クリーンアップタスクの実行
            logger.info('Running cleanup tasks...');
            for (const task of this.cleanupTasks) {
                try {
                    await task();
                } catch (error) {
                    logger.error('Cleanup task failed:', error);
                }
            }
            logger.info('Cleanup completed');

            // 4. Sentry のフラッシュ（バッファされたイベントを送信）
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

// 使用例
const server = app.listen(3000);
const shutdown = new GracefulShutdown(server);

// クリーンアップタスクの登録
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

### 4.3 Kubernetes ヘルスチェックとの統合

```typescript
// Kubernetes: ヘルスチェックエンドポイント
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
        // Liveness Probe: プロセスが正常に動作しているか
        app.get('/healthz', (req, res) => {
            if (this.isLive) {
                res.status(200).json({ status: 'ok' });
            } else {
                res.status(503).json({ status: 'not healthy' });
            }
        });

        // Readiness Probe: リクエストを受け付ける準備ができているか
        app.get('/readyz', async (req, res) => {
            if (!this.isReady) {
                return res.status(503).json({ status: 'not ready' });
            }

            // 各依存サービスのチェック
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

// 使用例
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

// アプリケーション起動完了後
health.setReady(true);

// シャットダウン開始時
// health.setReady(false);
// → Kubernetes がトラフィックを他のPodに振り分ける
```

---

## 5. エラー境界の設計原則

### 5.1 段階的フォールバック

```
1. 段階的なフォールバック
   → コンポーネントレベル → ページレベル → アプリレベル

2. ユーザーへの情報提供
   → 何が起きたか、何ができるかを伝える
   → 技術的な詳細は隠す（セキュリティ）

3. エラーのログとレポーティング
   → 全てのレイヤーでログを出力
   → 本番環境では Sentry 等に送信

4. リカバリー手段の提供
   → 再試行ボタン
   → 別の操作への誘導
   → 最終手段: ページリロード
```

### 5.2 フォールバック戦略パターン

```typescript
// パターン1: キャッシュフォールバック
async function getUserWithFallback(id: string): Promise<User> {
    try {
        // プライマリ: API から取得
        const user = await apiClient.getUser(id);
        await cache.set(`user:${id}`, user, { ttl: 300 });
        return user;
    } catch (error) {
        // フォールバック1: キャッシュから取得
        const cached = await cache.get(`user:${id}`);
        if (cached) {
            logger.warn(`Using cached data for user ${id}`);
            return cached;
        }

        // フォールバック2: デフォルト値
        logger.error(`No data available for user ${id}`);
        return {
            id,
            name: 'Unknown User',
            isStale: true,
        };
    }
}

// パターン2: 段階的デグレード
async function loadDashboard(userId: string): Promise<DashboardData> {
    const results = await Promise.allSettled([
        fetchUserStats(userId),
        fetchRecentOrders(userId),
        fetchNotifications(userId),
        fetchRecommendations(userId),
    ]);

    return {
        // 必須データ: 失敗したらエラー
        stats: unwrapOrThrow(results[0], 'Failed to load stats'),

        // 重要データ: 失敗したらデフォルト
        orders: unwrapOrDefault(results[1], []),

        // 付加データ: 失敗したら非表示
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

// パターン3: サーキットブレーカー
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

// 使用例
const paymentCircuit = new CircuitBreaker(5, 30_000);

async function processPayment(order: Order): Promise<PaymentResult> {
    try {
        return await paymentCircuit.execute(() =>
            paymentGateway.charge(order.total, order.paymentMethod)
        );
    } catch (error) {
        if (error instanceof CircuitOpenError) {
            // 支払いサービスが停止中
            await queueForLaterProcessing(order);
            return { status: 'queued', message: '後ほど処理されます' };
        }
        throw error;
    }
}
```

---

## 6. マイクロサービスにおけるエラー境界

### 6.1 サービス間のエラー伝播

```
マイクロサービスでのエラー境界:

  Client → API Gateway → Service A → Service B → Database
                                  ↓
                            Service C → External API

  エラー伝播のルール:
  1. 内部エラーを外部に漏らさない
     → Service B のDBエラーが Client に直接届かない
     → 各サービスがエラーを変換する

  2. エラーの粒度を適切に保つ
     → 下流サービスのエラーを集約
     → 上流には必要最小限の情報を返す

  3. タイムアウトとリトライ
     → 各サービス間の通信にタイムアウトを設定
     → 冪等な操作にのみリトライ

  4. サーキットブレーカー
     → 障害が発生したサービスへの呼び出しを遮断
     → フォールバック処理を実行
```

### 6.2 API Gateway のエラー集約

```typescript
// API Gateway: エラーの変換と集約
class ApiGateway {
    private circuits = new Map<string, CircuitBreaker>();

    async handleRequest(req: GatewayRequest): Promise<GatewayResponse> {
        const startTime = Date.now();

        try {
            // ルーティング
            const service = this.resolveService(req.path);
            const circuit = this.getCircuit(service.name);

            // サーキットブレーカー経由でリクエスト
            const response = await circuit.execute(async () => {
                return await this.forwardRequest(service, req, {
                    timeout: 5000,
                    retries: service.isIdempotent ? 2 : 0,
                });
            });

            return response;

        } catch (error) {
            // エラーの変換
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
                        message: 'サービスが一時的に利用できません',
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
                        message: 'リクエストがタイムアウトしました',
                    }
                }
            };
        }

        // 下流サービスのエラーレスポンスをそのまま返す
        if (error instanceof UpstreamError) {
            // ただし、内部情報は除去
            return {
                status: error.status,
                body: {
                    error: {
                        code: error.code,
                        message: error.publicMessage,
                        // error.internalDetails は含めない
                    }
                }
            };
        }

        // 予期しないエラー
        logger.error('Gateway error:', error, context);
        return {
            status: 500,
            body: {
                error: {
                    code: 'INTERNAL_ERROR',
                    message: 'サーバーエラーが発生しました',
                }
            }
        };
    }
}
```

### 6.3 分散トレーシングとエラー追跡

```typescript
// OpenTelemetry との統合
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

// エラー相関ID（Correlation ID）
// → リクエスト全体を通して同じIDを伝播
// → ログ、トレース、エラーレポートを紐付ける
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

// ミドルウェアで設定
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

## 7. エラーレポーティング

### 7.1 Sentry の統合

```typescript
// Sentry の本格的な設定
import * as Sentry from '@sentry/node';
import { nodeProfilingIntegration } from '@sentry/profiling-node';

Sentry.init({
    dsn: process.env.SENTRY_DSN,
    environment: process.env.NODE_ENV,
    release: process.env.APP_VERSION,

    // サンプリングレート
    tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
    profilesSampleRate: 0.1,

    // エラーのフィルタリング
    beforeSend(event, hint) {
        const error = hint.originalException;

        // 操作エラー（AppError で isOperational = true）はレポートしない
        if (error instanceof AppError && error.isOperational) {
            return null;
        }

        // 機密情報の除去
        if (event.request?.cookies) {
            delete event.request.cookies;
        }

        return event;
    },

    // ブレッドクラム（エラーに至るまでの操作履歴）
    beforeBreadcrumb(breadcrumb) {
        // SQL クエリの内容を除去
        if (breadcrumb.category === 'query') {
            breadcrumb.data = { ...breadcrumb.data, query: '[REDACTED]' };
        }
        return breadcrumb;
    },

    integrations: [
        nodeProfilingIntegration(),
    ],
});

// カスタムコンテキストの付与
function reportError(error: Error, context: Record<string, unknown> = {}): void {
    Sentry.withScope((scope) => {
        // ユーザー情報
        if (context.userId) {
            scope.setUser({ id: context.userId as string });
        }

        // タグ（検索可能なメタデータ）
        scope.setTag('error_code', (error as any).code || 'UNKNOWN');
        scope.setTag('service', 'user-service');

        // コンテキスト（詳細情報）
        scope.setContext('request', {
            path: context.path,
            method: context.method,
            requestId: context.requestId,
        });

        // フィンガープリント（グルーピングのカスタマイズ）
        if (error instanceof AppError) {
            scope.setFingerprint([error.code]);
        }

        Sentry.captureException(error);
    });
}
```

### 7.2 構造化ログとの統合

```typescript
// 構造化ログ（pino）との統合
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
        // カスタムエラーシリアライザ
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

// ログレベルの使い分け
// fatal: プロセスが終了する致命的エラー
// error: 予期しないエラー（Sentry にも送信）
// warn: 予期されるエラー（操作エラー）
// info: 正常な処理の記録
// debug: デバッグ情報
// trace: 詳細なトレース情報
```

---

## 8. RFC 7807: Problem Details for HTTP APIs

### 8.1 標準エラーフォーマット

```typescript
// RFC 7807 準拠のエラーレスポンス
interface ProblemDetails {
    type: string;        // エラーの種類を示すURI
    title: string;       // 人間可読なエラーの概要
    status: number;      // HTTPステータスコード
    detail?: string;     // 人間可読な詳細説明
    instance?: string;   // このエラーが発生したリクエストのURI
    [key: string]: unknown;  // 拡張フィールド
}

// 実装例
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

    // エラー固有の拡張フィールド
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

// Express ミドルウェア
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

## 9. フロントエンドのエラー境界

### 9.1 グローバルエラーハンドラ（ブラウザ）

```typescript
// ブラウザのグローバルエラーハンドラ

// JavaScript のランタイムエラー
window.onerror = (message, source, lineno, colno, error) => {
    reportError({
        type: 'runtime_error',
        message: String(message),
        source,
        lineno,
        colno,
        stack: error?.stack,
    });
    // true を返すとデフォルトのエラー処理を抑制
    return false;
};

// 未処理の Promise rejection
window.addEventListener('unhandledrejection', (event) => {
    reportError({
        type: 'unhandled_rejection',
        reason: event.reason,
        promise: event.promise,
    });
});

// リソース読み込みエラー（画像、スクリプトなど）
window.addEventListener('error', (event) => {
    if (event.target instanceof HTMLElement) {
        reportError({
            type: 'resource_error',
            tagName: event.target.tagName,
            src: (event.target as any).src || (event.target as any).href,
        });
    }
}, true);  // キャプチャフェーズで捕捉

// ネットワークエラーの監視
window.addEventListener('offline', () => {
    showNotification('ネットワーク接続が切断されました');
});

window.addEventListener('online', () => {
    showNotification('ネットワーク接続が回復しました');
});
```

### 9.2 Vue のエラーハンドリング

```typescript
// Vue 3: グローバルエラーハンドラ
import { createApp } from 'vue';

const app = createApp(App);

// 全コンポーネントの未キャッチエラー
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

// 警告ハンドラ（開発時のみ推奨）
app.config.warnHandler = (msg, instance, trace) => {
    console.warn('Vue Warning:', msg);
    console.warn('Trace:', trace);
};

// コンポーネントレベルのエラーハンドリング
// onErrorCaptured: Error Boundary のように機能
import { onErrorCaptured, ref } from 'vue';

export default {
    setup() {
        const error = ref<Error | null>(null);

        onErrorCaptured((err, instance, info) => {
            error.value = err;
            // false を返すとエラーの伝播を停止
            return false;
        });

        return { error };
    },
};
```

### 9.3 Angular のエラーハンドリング

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
        // HttpErrorResponse の処理
        if (error instanceof HttpErrorResponse) {
            this.handleHttpError(error);
            return;
        }

        // クライアントサイドのエラー
        const appError = error instanceof Error ? error : new Error(String(error));

        this.logger.error('Unhandled error', {
            message: appError.message,
            stack: appError.stack,
        });

        Sentry.captureException(appError);

        this.notification.showError(
            '予期しないエラーが発生しました。ページを再読み込みしてください。'
        );
    }

    private handleHttpError(error: HttpErrorResponse): void {
        switch (error.status) {
            case 0:
                this.notification.showError('ネットワーク接続を確認してください');
                break;
            case 401:
                this.notification.showError('セッションが期限切れです');
                // リダイレクト
                break;
            case 403:
                this.notification.showError('アクセス権がありません');
                break;
            case 404:
                this.notification.showError('要求されたリソースが見つかりません');
                break;
            case 429:
                this.notification.showError('リクエストが多すぎます。しばらくお待ちください');
                break;
            default:
                this.notification.showError('サーバーエラーが発生しました');
        }
    }
}

@NgModule({
    providers: [
        { provide: ErrorHandler, useClass: GlobalErrorHandler },
    ],
})
export class AppModule {}

// HTTP Interceptor でのエラーハンドリング
@Injectable()
class ErrorInterceptor implements HttpInterceptor {
    intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
        return next.handle(req).pipe(
            retry({ count: 2, delay: 1000 }),  // 2回リトライ
            catchError((error: HttpErrorResponse) => {
                if (error.status === 401) {
                    // トークンリフレッシュを試みる
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

## 10. テスト戦略

### 10.1 Error Boundary のテスト

```tsx
// React Error Boundary のテスト
import { render, screen, fireEvent } from '@testing-library/react';

// エラーを意図的に発生させるコンポーネント
function ThrowError({ shouldThrow }: { shouldThrow: boolean }) {
    if (shouldThrow) {
        throw new Error('Test error');
    }
    return <div>正常なコンテンツ</div>;
}

describe('ErrorBoundary', () => {
    // console.error の出力を抑制
    const originalError = console.error;
    beforeAll(() => { console.error = jest.fn(); });
    afterAll(() => { console.error = originalError; });

    it('子コンポーネントのエラーをキャッチする', () => {
        render(
            <ErrorBoundary fallback={<p>エラーが発生しました</p>}>
                <ThrowError shouldThrow={true} />
            </ErrorBoundary>
        );

        expect(screen.getByText('エラーが発生しました')).toBeInTheDocument();
        expect(screen.queryByText('正常なコンテンツ')).not.toBeInTheDocument();
    });

    it('エラーがない場合は子コンポーネントを表示する', () => {
        render(
            <ErrorBoundary fallback={<p>エラーが発生しました</p>}>
                <ThrowError shouldThrow={false} />
            </ErrorBoundary>
        );

        expect(screen.getByText('正常なコンテンツ')).toBeInTheDocument();
        expect(screen.queryByText('エラーが発生しました')).not.toBeInTheDocument();
    });

    it('再試行ボタンでリセットできる', () => {
        const { rerender } = render(
            <ErrorBoundary
                fallback={(error, reset) => (
                    <div>
                        <p>エラー: {error.message}</p>
                        <button onClick={reset}>再試行</button>
                    </div>
                )}
            >
                <ThrowError shouldThrow={true} />
            </ErrorBoundary>
        );

        expect(screen.getByText('エラー: Test error')).toBeInTheDocument();

        // shouldThrow を false に変更して再試行
        fireEvent.click(screen.getByText('再試行'));

        // リセット後のレンダリングでエラーが発生しなければ正常表示
    });

    it('onError コールバックが呼ばれる', () => {
        const onError = jest.fn();

        render(
            <ErrorBoundary
                onError={onError}
                fallback={<p>エラー</p>}
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

### 10.2 エラーミドルウェアのテスト

```typescript
// Express エラーミドルウェアのテスト
import request from 'supertest';

describe('Error Middleware', () => {
    it('ValidationError に対して 400 を返す', async () => {
        // ルートハンドラで ValidationError をスロー
        app.get('/test-validation', (req, res, next) => {
            next(new ValidationError([
                { field: 'email', message: '必須です' },
            ]));
        });

        const response = await request(app)
            .get('/test-validation')
            .expect(400);

        expect(response.body.error.code).toBe('VALIDATION_ERROR');
        expect(response.body.error.details).toHaveLength(1);
    });

    it('NotFoundError に対して 404 を返す', async () => {
        app.get('/test-not-found', (req, res, next) => {
            next(new NotFoundError('User', 'user-123'));
        });

        const response = await request(app)
            .get('/test-not-found')
            .expect(404);

        expect(response.body.error.code).toBe('NOT_FOUND');
    });

    it('予期しないエラーに対して 500 を返す', async () => {
        app.get('/test-internal', (req, res, next) => {
            next(new Error('Unexpected'));
        });

        const response = await request(app)
            .get('/test-internal')
            .expect(500);

        expect(response.body.error.code).toBe('INTERNAL_ERROR');
        // 内部情報が漏洩していないことを確認
        expect(response.body.error.message).not.toContain('Unexpected');
    });
});
```

---

## まとめ

| レイヤー | 手法 | 目的 |
|---------|------|------|
| コンポーネント | Error Boundary | UI の部分的エラー |
| ミドルウェア | エラーハンドラ | HTTPレスポンス統一 |
| プロセス | uncaughtException | 最後の砦 |
| 外部サービス | Sentry | モニタリング |
| マイクロサービス | サーキットブレーカー | 障害の伝播防止 |
| フロントエンド | グローバルハンドラ | 未捕捉エラーの収集 |
| API | RFC 7807 | エラーフォーマット統一 |

---

## 次に読むべきガイド
→ [[03-custom-errors.md]] — カスタムエラー

---

## 参考文献
1. React Documentation. "Error Boundaries."
2. Express.js Documentation. "Error Handling."
3. NestJS Documentation. "Exception Filters."
4. RFC 7807. "Problem Details for HTTP APIs."
5. Sentry Documentation. "JavaScript SDK."
6. Nygard, M. "Release It!" 2nd Edition, 2018.
7. Newman, S. "Building Microservices." 2nd Edition, 2021.
8. react-error-boundary. GitHub.
