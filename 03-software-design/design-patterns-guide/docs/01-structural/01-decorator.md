# Decorator パターン

> オブジェクトに **動的に** 新しい機能を追加するための構造パターン。サブクラス化の代替として合成（コンポジション）を用い、機能の自由な組み合わせを実現する。

---

## 前提知識

| トピック | 必要レベル | 参照先 |
|---------|-----------|--------|
| オブジェクト指向プログラミング | 基礎 | [OOP基礎](../../../02-programming/oop-guide/docs/) |
| インタフェースと抽象クラス | 基礎 | [インタフェース設計](../../../02-programming/oop-guide/docs/) |
| 合成（Composition）と委譲 | 理解 | [合成優先の原則](../../../03-software-design/clean-code-principles/docs/) |
| 開放閉鎖原則（OCP） | 理解 | [SOLID](../../../03-software-design/clean-code-principles/docs/) |
| TypeScript / Python のデコレータ構文 | 基礎 | 各言語ガイド |

---

## この章で学ぶこと

1. Decorator パターンの**目的**と、なぜ継承ではなく合成で機能拡張するのか
2. デコレータの**積み重ね（チェーン）** の仕組みと実行順序
3. GoF の Decorator パターンと言語組み込みのデコレータ構文（TypeScript/Python）の関係
4. HTTP クライアント・ストリーム処理・React HOC など実践的な活用パターン
5. デコレータの過剰な積み重ねやインタフェース外依存などのアンチパターン

---

## なぜ Decorator パターンが必要なのか（WHY）

### 問題: 継承による機能追加のクラス爆発

機能の組み合わせを全て継承で表現すると、クラスの数が爆発的に増加します。

```
[継承で機能追加する場合 — クラス爆発]

                     DataSource
                    /    |     \
            FileDS    EncryptedDS   CompressedDS
           /    \        |
    EncryptedFileDS  CompressedFileDS
          |
  CompressedEncryptedFileDS

3つの機能の全組み合わせ → 2^3 = 8 クラスが必要
N個の機能なら 2^N クラスが必要！

[Decorator で機能追加する場合 — 線形]

  DataSource (interface)
      △
      |
  FileDataSource (concrete)
      △
      |
  DataSourceDecorator (abstract)
      △
      |─── EncryptionDecorator
      |─── CompressionDecorator
      |─── LoggingDecorator

3つの機能 → 3+1 = 4 クラスで全組み合わせをカバー
N個の機能なら N+1 クラスで済む！
```

### Decorator の本質: 入れ子構造による機能合成

```
Client の呼び出し:
  source.write("Hello")

実行時のオブジェクト構造:
┌───────────────────────────┐
│    LoggingDecorator       │  ← ログ出力
│  ┌───────────────────┐    │
│  │ CompressionDeco   │    │  ← 圧縮
│  │  ┌─────────────┐  │    │
│  │  │ EncryptDeco │  │    │  ← 暗号化
│  │  │  ┌───────┐  │  │    │
│  │  │  │FileDS │  │  │    │  ← 実処理
│  │  │  └───────┘  │  │    │
│  │  └─────────────┘  │    │
│  └───────────────────┘    │
└───────────────────────────┘

write("Hello") の実行順序:
  Logging.write()
    → Compression.write()
      → Encryption.write()
        → FileDS.write()  ← 実際のファイル書き込み
```

このパターンにより:
- **開放閉鎖原則（OCP）**: 既存コードを変更せずに機能を追加
- **単一責任原則（SRP）**: 各デコレータが1つの責務だけを担う
- **実行時の柔軟性**: 機能の組み合わせと順序を実行時に変更可能
- **テスト容易性**: 各デコレータを個別にテスト可能

---

## 1. Decorator の構造

### クラス図

```
+------------------+
|    Component     |
|   (interface)    |
+------------------+
| + operation()    |
+------------------+
      △        △
      |        |
+----------+  +--------------------+
| Concrete |  | BaseDecorator      |
| Component|  +--------------------+
+----------+  | - wrapped:         |
              |   Component        |
              | + operation() {    |
              |   wrapped          |
              |   .operation()    }|
              +--------------------+
                       △
               ________|________
              |                 |
       +-------------+  +-------------+
       | DecoratorA  |  | DecoratorB  |
       +-------------+  +-------------+
       | + operation |  | + operation |
       |  {          |  |  {          |
       |   // 前処理 |  |   // 前処理 |
       |   super     |  |   super     |
       |   .operation|  |   .operation|
       |   // 後処理 |  |   // 後処理 |
       |  }          |  |  }          |
       +-------------+  +-------------+
```

### シーケンス図

```
Client    DecoratorA      DecoratorB      ConcreteComponent
  |           |                |                |
  |--op()---->|                |                |
  |           |--前処理A       |                |
  |           |--op()--------->|                |
  |           |                |--前処理B       |
  |           |                |--op()--------->|
  |           |                |                |--実処理
  |           |                |                |
  |           |                |<--result-------|
  |           |                |--後処理B       |
  |           |<--result-------|                |
  |           |--後処理A       |                |
  |<--result--|                |                |
```

### Decorator チェーンの構築

```
// 構築方法1: コンストラクタのネスト
const source = new LoggingDecorator(
  new CompressionDecorator(
    new EncryptionDecorator(
      new FileDataSource("data.txt")
    )
  )
);

// 構築方法2: Builder パターンとの組み合わせ
const source = DataSourceBuilder
  .from(new FileDataSource("data.txt"))
  .withEncryption()
  .withCompression()
  .withLogging()
  .build();

// 構築方法3: 関数パイプライン
const source = pipe(
  new FileDataSource("data.txt"),
  withEncryption,
  withCompression,
  withLogging
);
```

---

## 2. コード例

### コード例 1: データソース Decorator（基本形）

```typescript
// === Component インタフェース ===
interface DataSource {
  read(): string;
  write(data: string): void;
}

// === ConcreteComponent ===
class FileDataSource implements DataSource {
  private content = "";

  constructor(private filename: string) {}

  read(): string {
    console.log(`  [File] Reading from ${this.filename}`);
    return this.content;
  }

  write(data: string): void {
    console.log(`  [File] Writing to ${this.filename}: "${data}"`);
    this.content = data;
  }
}

// === BaseDecorator（オプション: 共通の委譲ロジック）===
abstract class DataSourceDecorator implements DataSource {
  constructor(protected wrapped: DataSource) {}

  read(): string {
    return this.wrapped.read();
  }

  write(data: string): void {
    this.wrapped.write(data);
  }
}

// === ConcreteDecorator 1: 暗号化 ===
class EncryptionDecorator extends DataSourceDecorator {
  read(): string {
    const data = super.read();
    const decrypted = this.decrypt(data);
    console.log(`  [Encrypt] Decrypted: "${data}" → "${decrypted}"`);
    return decrypted;
  }

  write(data: string): void {
    const encrypted = this.encrypt(data);
    console.log(`  [Encrypt] Encrypted: "${data}" → "${encrypted}"`);
    super.write(encrypted);
  }

  private encrypt(data: string): string {
    return Buffer.from(data).toString("base64");
  }

  private decrypt(data: string): string {
    return Buffer.from(data, "base64").toString("utf-8");
  }
}

// === ConcreteDecorator 2: 圧縮 ===
class CompressionDecorator extends DataSourceDecorator {
  read(): string {
    const data = super.read();
    const decompressed = this.decompress(data);
    console.log(`  [Compress] Decompressed`);
    return decompressed;
  }

  write(data: string): void {
    const compressed = this.compress(data);
    console.log(`  [Compress] Compressed: ${data.length} → ${compressed.length} chars`);
    super.write(compressed);
  }

  private compress(data: string): string {
    return `compressed(${data})`;
  }

  private decompress(data: string): string {
    return data.replace(/^compressed\(/, "").replace(/\)$/, "");
  }
}

// === ConcreteDecorator 3: ログ ===
class LoggingDecorator extends DataSourceDecorator {
  read(): string {
    console.log("[LOG] read() called");
    const start = Date.now();
    const result = super.read();
    console.log(`[LOG] read() completed in ${Date.now() - start}ms`);
    return result;
  }

  write(data: string): void {
    console.log(`[LOG] write("${data}") called`);
    const start = Date.now();
    super.write(data);
    console.log(`[LOG] write() completed in ${Date.now() - start}ms`);
  }
}

// === 使用例: デコレータの積み重ね ===
const source: DataSource = new LoggingDecorator(
  new CompressionDecorator(
    new EncryptionDecorator(
      new FileDataSource("secret.txt")
    )
  )
);

source.write("Hello, World!");
// [LOG] write("Hello, World!") called
//   [Compress] Compressed: 13 → 29 chars
//   [Encrypt] Encrypted: "compressed(Hello, World!)" → "Y29tcHJlc3NlZ..."
//   [File] Writing to secret.txt: "Y29tcHJlc3NlZ..."
// [LOG] write() completed in 1ms

const data = source.read();
// [LOG] read() called
//   [File] Reading from secret.txt
//   [Encrypt] Decrypted: ...
//   [Compress] Decompressed
// [LOG] read() completed in 0ms
```

---

### コード例 2: HTTP クライアント Decorator

```typescript
// === Component ===
interface HttpClient {
  request(url: string, options?: RequestInit): Promise<Response>;
}

class FetchClient implements HttpClient {
  async request(url: string, options?: RequestInit): Promise<Response> {
    return fetch(url, options);
  }
}

// === Decorator 1: リトライ ===
class RetryDecorator implements HttpClient {
  constructor(
    private client: HttpClient,
    private maxRetries = 3,
    private baseDelay = 1000
  ) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    let lastError: Error | undefined;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const response = await this.client.request(url, options);
        if (response.ok || response.status < 500) return response;
        throw new Error(`Server error: ${response.status}`);
      } catch (error) {
        lastError = error as Error;
        if (attempt < this.maxRetries) {
          const delay = this.baseDelay * Math.pow(2, attempt);
          console.log(`Retry ${attempt + 1}/${this.maxRetries} after ${delay}ms`);
          await new Promise(r => setTimeout(r, delay));
        }
      }
    }
    throw lastError;
  }
}

// === Decorator 2: タイムアウト ===
class TimeoutDecorator implements HttpClient {
  constructor(private client: HttpClient, private timeoutMs = 5000) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      return await this.client.request(url, {
        ...options,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
  }
}

// === Decorator 3: ロギング ===
class LoggingHttpDecorator implements HttpClient {
  constructor(private client: HttpClient) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    const method = options?.method ?? "GET";
    console.log(`→ ${method} ${url}`);
    const start = Date.now();

    try {
      const response = await this.client.request(url, options);
      console.log(`← ${response.status} (${Date.now() - start}ms)`);
      return response;
    } catch (error) {
      console.log(`✗ ERROR (${Date.now() - start}ms): ${error}`);
      throw error;
    }
  }
}

// === Decorator 4: 認証ヘッダー追加 ===
class AuthDecorator implements HttpClient {
  constructor(
    private client: HttpClient,
    private getToken: () => Promise<string>
  ) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    const token = await this.getToken();
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${token}`);
    return this.client.request(url, { ...options, headers });
  }
}

// === Decorator 5: サーキットブレーカー ===
class CircuitBreakerDecorator implements HttpClient {
  private failures = 0;
  private lastFailure = 0;
  private state: "closed" | "open" | "half-open" = "closed";

  constructor(
    private client: HttpClient,
    private threshold = 5,
    private resetTimeout = 30000
  ) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    if (this.state === "open") {
      if (Date.now() - this.lastFailure > this.resetTimeout) {
        this.state = "half-open";
      } else {
        throw new Error("Circuit breaker is OPEN");
      }
    }

    try {
      const response = await this.client.request(url, options);
      this.onSuccess();
      return response;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failures = 0;
    this.state = "closed";
  }

  private onFailure(): void {
    this.failures++;
    this.lastFailure = Date.now();
    if (this.failures >= this.threshold) {
      this.state = "open";
      console.log("Circuit breaker OPENED");
    }
  }
}

// === 組み立て ===
const httpClient: HttpClient = new LoggingHttpDecorator(
  new CircuitBreakerDecorator(
    new RetryDecorator(
      new TimeoutDecorator(
        new AuthDecorator(
          new FetchClient(),
          async () => "token-xxx"
        ),
        5000
      ),
      3
    )
  )
);

// 実行順: Logging → CircuitBreaker → Retry → Timeout → Auth → Fetch
```

---

### コード例 3: TypeScript TC39 デコレータ構文（Stage 3）

```typescript
// TC39 Stage 3 Decorators (TypeScript 5+)
// GoF の Decorator パターンとは異なるが、動機は共通

// === メソッドデコレータ: ログ出力 ===
function logged(
  target: any,
  context: ClassMethodDecoratorContext
) {
  const methodName = String(context.name);
  return function (this: any, ...args: any[]) {
    console.log(`→ ${methodName}(${args.map(a => JSON.stringify(a)).join(", ")})`);
    const result = target.call(this, ...args);
    console.log(`← ${methodName} = ${JSON.stringify(result)}`);
    return result;
  };
}

// === メソッドデコレータ: パフォーマンス計測 ===
function timed(
  target: any,
  context: ClassMethodDecoratorContext
) {
  const methodName = String(context.name);
  return function (this: any, ...args: any[]) {
    const start = performance.now();
    const result = target.call(this, ...args);
    const elapsed = performance.now() - start;
    console.log(`${methodName}: ${elapsed.toFixed(2)}ms`);
    return result;
  };
}

// === メソッドデコレータ: メモ化 ===
function memoize(
  target: any,
  context: ClassMethodDecoratorContext
) {
  const cache = new Map<string, any>();
  return function (this: any, ...args: any[]) {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const result = target.call(this, ...args);
    cache.set(key, result);
    return result;
  };
}

// === メソッドデコレータ: バリデーション ===
function validate(schema: Record<string, (v: any) => boolean>) {
  return function (
    target: any,
    context: ClassMethodDecoratorContext
  ) {
    return function (this: any, ...args: any[]) {
      // 最初の引数がオブジェクトの場合バリデーション
      const input = args[0];
      if (typeof input === "object" && input !== null) {
        for (const [key, validator] of Object.entries(schema)) {
          if (!validator(input[key])) {
            throw new Error(`Validation failed for field "${key}"`);
          }
        }
      }
      return target.call(this, ...args);
    };
  };
}

// === 使用例 ===
class Calculator {
  @logged
  @timed
  add(a: number, b: number): number {
    return a + b;
  }

  @memoize
  fibonacci(n: number): number {
    if (n <= 1) return n;
    return this.fibonacci(n - 1) + this.fibonacci(n - 2);
  }
}

class UserService {
  @validate({
    name: (v: any) => typeof v === "string" && v.length > 0,
    age: (v: any) => typeof v === "number" && v >= 0,
  })
  createUser(data: { name: string; age: number }): void {
    console.log(`Created user: ${data.name}`);
  }
}
```

---

### コード例 4: Python デコレータ（関数デコレータ + クラスデコレータ）

```python
import functools
import time
import logging
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

# === 関数デコレータ: リトライ ===
def retry(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """指定回数までリトライするデコレータ"""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait = delay * (2 ** attempt)
                        logging.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {wait}s: {e}"
                        )
                        time.sleep(wait)
            raise last_exception  # type: ignore
        return wrapper
    return decorator


# === 関数デコレータ: 実行時間計測 ===
def timed(func: Callable[P, R]) -> Callable[P, R]:
    """実行時間を計測するデコレータ"""
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper


# === 関数デコレータ: キャッシュ（TTL付き）===
def cache_with_ttl(ttl_seconds: float = 60.0):
    """TTL付きキャッシュデコレータ"""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        _cache: dict[str, tuple[R, float]] = {}

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key = str((args, tuple(sorted(kwargs.items()))))
            if key in _cache:
                value, expiry = _cache[key]
                if time.time() < expiry:
                    return value
            result = func(*args, **kwargs)
            _cache[key] = (result, time.time() + ttl_seconds)
            return result
        return wrapper
    return decorator


# === 関数デコレータ: 入力バリデーション ===
def validate_args(**validators: Callable):
    """引数のバリデーションデコレータ"""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for '{param_name}': {value}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# === 使用例 ===
@retry(max_retries=3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
@timed
def fetch_data(url: str) -> dict:
    """外部APIからデータを取得"""
    import urllib.request
    response = urllib.request.urlopen(url)
    return {"status": response.status}


@cache_with_ttl(ttl_seconds=300)
def get_config(key: str) -> str:
    """設定値を取得（キャッシュ付き）"""
    return f"value_for_{key}"


@validate_args(
    name=lambda v: isinstance(v, str) and len(v) > 0,
    age=lambda v: isinstance(v, int) and 0 <= v <= 150,
)
def create_user(name: str, age: int) -> dict:
    return {"name": name, "age": age}


# クラスデコレータ: Singleton
def singleton(cls):
    """シングルトンにするクラスデコレータ"""
    instances: dict = {}

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class DatabaseConnection:
    def __init__(self, url: str):
        self.url = url
        print(f"Connecting to {url}")
```

---

### コード例 5: React Higher-Order Component（HOC）as Decorator

```typescript
import React, { ComponentType, useEffect, useState } from "react";

// === HOC 1: ローディング状態の追加 ===
function withLoading<P extends object>(
  WrappedComponent: ComponentType<P>
): ComponentType<P & { isLoading?: boolean }> {
  return function WithLoadingComponent(props: P & { isLoading?: boolean }) {
    const { isLoading, ...rest } = props;
    if (isLoading) {
      return <div className="spinner">Loading...</div>;
    }
    return <WrappedComponent {...(rest as P)} />;
  };
}

// === HOC 2: 認証ガード ===
function withAuth<P extends object>(
  WrappedComponent: ComponentType<P>
): ComponentType<P> {
  return function WithAuthComponent(props: P) {
    const { user, isAuthenticated } = useAuth();
    if (!isAuthenticated) {
      return <Navigate to="/login" />;
    }
    return <WrappedComponent {...props} user={user} />;
  };
}

// === HOC 3: エラーバウンダリ ===
function withErrorBoundary<P extends object>(
  WrappedComponent: ComponentType<P>,
  FallbackComponent: ComponentType<{ error: Error }>
): ComponentType<P> {
  return class ErrorBoundaryWrapper extends React.Component<P, { error: Error | null }> {
    state = { error: null };

    static getDerivedStateFromError(error: Error) {
      return { error };
    }

    render() {
      if (this.state.error) {
        return <FallbackComponent error={this.state.error} />;
      }
      return <WrappedComponent {...this.props} />;
    }
  };
}

// === HOC 4: パフォーマンストラッキング ===
function withPerformanceTracking<P extends object>(
  WrappedComponent: ComponentType<P>,
  componentName: string
): ComponentType<P> {
  return function WithPerformanceTracking(props: P) {
    useEffect(() => {
      const start = performance.now();
      return () => {
        const elapsed = performance.now() - start;
        console.log(`${componentName} rendered for ${elapsed.toFixed(0)}ms`);
      };
    });
    return <WrappedComponent {...props} />;
  };
}

// === 積み重ね ===
const UserList: React.FC<{ users: User[] }> = ({ users }) => (
  <ul>{users.map(u => <li key={u.id}>{u.name}</li>)}</ul>
);

// デコレータの積み重ね（外側から内側へ適用）
const EnhancedUserList = withErrorBoundary(
  withAuth(
    withLoading(
      withPerformanceTracking(UserList, "UserList")
    )
  ),
  ErrorFallback
);

// 現代の React では Hooks が主流だが、
// HOC は条件付きレンダリングや Provider ラッピングでは依然有効
```

---

### コード例 6: Go — ミドルウェアパターン（Decorator の変形）

```go
package main

import (
    "fmt"
    "log"
    "net/http"
    "time"
)

// === Middleware 型（Decorator の Go イディオム）===
type Middleware func(http.Handler) http.Handler

// === Middleware 1: ロギング ===
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        log.Printf("→ %s %s", r.Method, r.URL.Path)
        next.ServeHTTP(w, r)
        log.Printf("← %s %s (%v)", r.Method, r.URL.Path, time.Since(start))
    })
}

// === Middleware 2: 認証 ===
func AuthMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        if token == "" {
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }
        next.ServeHTTP(w, r)
    })
}

// === Middleware 3: リカバリ ===
func RecoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("Panic recovered: %v", err)
                http.Error(w, "Internal Server Error", http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

// === Middleware チェーン ===
func Chain(handler http.Handler, middlewares ...Middleware) http.Handler {
    // 逆順に適用（最初に指定したミドルウェアが最外側）
    for i := len(middlewares) - 1; i >= 0; i-- {
        handler = middlewares[i](handler)
    }
    return handler
}

// === 使用例 ===
func main() {
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintln(w, "Hello, World!")
    })

    // デコレータの積み重ね
    enhanced := Chain(handler,
        RecoveryMiddleware,
        LoggingMiddleware,
        AuthMiddleware,
    )

    http.Handle("/api/", enhanced)
    http.ListenAndServe(":8080", nil)
}
```

---

### コード例 7: Java — I/O Streams（標準ライブラリの Decorator）

```java
import java.io.*;

// Java の I/O ストリームは Decorator パターンの典型例

public class JavaIODecoratorExample {

    // === 読み込みの Decorator チェーン ===
    public static void readExample() throws IOException {
        // InputStream 階層:
        //   BufferedInputStream(Decorator)
        //     → DataInputStream(Decorator)
        //       → FileInputStream(ConcreteComponent)

        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(
                    new FileInputStream("data.bin")))) {
            int value = dis.readInt();
            String text = dis.readUTF();
            System.out.println(value + " " + text);
        }
    }

    // === 書き込みの Decorator チェーン ===
    public static void writeExample() throws IOException {
        // OutputStream 階層:
        //   BufferedOutputStream(Decorator)
        //     → GZIPOutputStream(Decorator)
        //       → FileOutputStream(ConcreteComponent)

        try (var out = new BufferedOutputStream(
                new java.util.zip.GZIPOutputStream(
                    new FileOutputStream("output.gz")))) {
            out.write("Hello, compressed world!".getBytes());
        }
    }

    // === カスタム Decorator ===
    static class CountingInputStream extends FilterInputStream {
        private long bytesRead = 0;

        protected CountingInputStream(InputStream in) {
            super(in);
        }

        @Override
        public int read() throws IOException {
            int b = super.read();
            if (b != -1) bytesRead++;
            return b;
        }

        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            int count = super.read(b, off, len);
            if (count > 0) bytesRead += count;
            return count;
        }

        public long getBytesRead() {
            return bytesRead;
        }
    }
}
```

---

### コード例 8: Kotlin — 拡張関数とデコレータ

```kotlin
// Kotlin のデリゲーションパターンで Decorator を実装

interface Logger {
    fun log(level: String, message: String)
    fun close()
}

class ConsoleLogger : Logger {
    override fun log(level: String, message: String) {
        println("[$level] $message")
    }
    override fun close() {}
}

// Kotlin の by キーワードによるデリゲーション
class TimestampLogger(private val inner: Logger) : Logger by inner {
    override fun log(level: String, message: String) {
        val timestamp = java.time.LocalDateTime.now()
        inner.log(level, "[$timestamp] $message")
    }
}

class FilterLogger(
    private val inner: Logger,
    private val minLevel: String
) : Logger by inner {
    private val levels = listOf("DEBUG", "INFO", "WARN", "ERROR")

    override fun log(level: String, message: String) {
        if (levels.indexOf(level) >= levels.indexOf(minLevel)) {
            inner.log(level, message)
        }
    }
}

// 積み重ね
fun main() {
    val logger: Logger = FilterLogger(
        TimestampLogger(ConsoleLogger()),
        "INFO"
    )

    logger.log("DEBUG", "This will be filtered")  // 出力なし
    logger.log("INFO", "Application started")      // 出力あり
    logger.log("ERROR", "Something went wrong")    // 出力あり
}
```

---

### コード例 9: 関数合成による軽量 Decorator

```typescript
// クラスを使わず、関数合成でデコレータを実現

type AsyncFn<T> = (...args: any[]) => Promise<T>;

// === Decorator Factory 関数 ===
function withRetry<T>(fn: AsyncFn<T>, maxRetries = 3): AsyncFn<T> {
  return async (...args) => {
    for (let i = 0; i <= maxRetries; i++) {
      try {
        return await fn(...args);
      } catch (e) {
        if (i === maxRetries) throw e;
        await new Promise(r => setTimeout(r, 1000 * 2 ** i));
      }
    }
    throw new Error("Unreachable");
  };
}

function withTimeout<T>(fn: AsyncFn<T>, ms: number): AsyncFn<T> {
  return (...args) =>
    Promise.race([
      fn(...args),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("Timeout")), ms)
      ),
    ]);
}

function withLogging<T>(fn: AsyncFn<T>, name: string): AsyncFn<T> {
  return async (...args) => {
    console.log(`→ ${name}(${args.join(", ")})`);
    try {
      const result = await fn(...args);
      console.log(`← ${name}: success`);
      return result;
    } catch (e) {
      console.log(`✗ ${name}: ${e}`);
      throw e;
    }
  };
}

// === pipe ユーティリティ ===
function pipe<T>(
  fn: AsyncFn<T>,
  ...decorators: Array<(fn: AsyncFn<T>) => AsyncFn<T>>
): AsyncFn<T> {
  return decorators.reduce((acc, decorator) => decorator(acc), fn);
}

// === 使用例 ===
async function fetchUser(id: string): Promise<{ name: string }> {
  const res = await fetch(`/api/users/${id}`);
  return res.json();
}

// 関数合成でデコレータを積み重ね
const enhancedFetchUser = pipe(
  fetchUser,
  fn => withTimeout(fn, 5000),
  fn => withRetry(fn, 3),
  fn => withLogging(fn, "fetchUser"),
);

const user = await enhancedFetchUser("123");
```

---

### コード例 10: Node.js Stream Transform（実践的 Decorator）

```typescript
import { Transform, TransformCallback, Readable, pipeline } from "stream";
import { promisify } from "util";

const pipelineAsync = promisify(pipeline);

// === Transform 1: JSON パース ===
class JsonParseTransform extends Transform {
  constructor() {
    super({ objectMode: true });
  }

  _transform(chunk: Buffer, _encoding: string, callback: TransformCallback): void {
    try {
      const parsed = JSON.parse(chunk.toString());
      this.push(parsed);
      callback();
    } catch (error) {
      callback(error as Error);
    }
  }
}

// === Transform 2: フィルタリング ===
class FilterTransform extends Transform {
  constructor(private predicate: (item: any) => boolean) {
    super({ objectMode: true });
  }

  _transform(chunk: any, _encoding: string, callback: TransformCallback): void {
    if (this.predicate(chunk)) {
      this.push(chunk);
    }
    callback();
  }
}

// === Transform 3: マッピング ===
class MapTransform extends Transform {
  constructor(private mapper: (item: any) => any) {
    super({ objectMode: true });
  }

  _transform(chunk: any, _encoding: string, callback: TransformCallback): void {
    try {
      this.push(this.mapper(chunk));
      callback();
    } catch (error) {
      callback(error as Error);
    }
  }
}

// === Transform 4: バッチ集約 ===
class BatchTransform extends Transform {
  private buffer: any[] = [];

  constructor(private batchSize: number) {
    super({ objectMode: true });
  }

  _transform(chunk: any, _encoding: string, callback: TransformCallback): void {
    this.buffer.push(chunk);
    if (this.buffer.length >= this.batchSize) {
      this.push([...this.buffer]);
      this.buffer = [];
    }
    callback();
  }

  _flush(callback: TransformCallback): void {
    if (this.buffer.length > 0) {
      this.push([...this.buffer]);
    }
    callback();
  }
}

// === パイプラインでデコレータをチェーン ===
async function processLogs(): Promise<void> {
  await pipelineAsync(
    Readable.from(logLines),           // ソース
    new JsonParseTransform(),           // JSON パース（Decorator 1）
    new FilterTransform(                // フィルタ（Decorator 2）
      log => log.level === "error"
    ),
    new MapTransform(                   // 変換（Decorator 3）
      log => ({ message: log.message, timestamp: log.ts })
    ),
    new BatchTransform(100),            // バッチ（Decorator 4）
    createWriteStream("errors.jsonl")   // 出力先
  );
}
```

---

## 3. 比較表

### 比較表 1: Decorator vs 継承

| 観点 | Decorator（合成） | 継承 |
|------|:---:|:---:|
| 機能追加タイミング | **実行時（動的）** | コンパイル時（静的） |
| 組み合わせ | **自由に積み重ね可能** | クラス爆発 |
| 既存コード変更 | **不要** | サブクラス追加 |
| OCP 準拠 | **Yes** | 部分的 |
| SRP 準拠 | **Yes**（1デコレータ=1責務） | 違反しやすい |
| デバッグ | スタックトレースが深い | **直感的** |
| パフォーマンス | 間接呼び出しコスト | **直接呼び出し** |
| 型安全性 | **インタフェースで保証** | 継承階層で保証 |

### 比較表 2: GoF Decorator vs 言語デコレータ構文

| 観点 | GoF Decorator パターン | TypeScript/Python デコレータ |
|------|:---:|:---:|
| **適用対象** | オブジェクト（インスタンス） | クラス/メソッド/プロパティ |
| **適用タイミング** | **実行時**（動的） | **定義時**（静的） |
| **インタフェース維持** | **明示的に保証** | 暗黙的 |
| **積み重ね** | コンストラクタのネスト | `@` 構文の積み重ね |
| **主な用途** | 機能のラッピング | メタプログラミング、AOP |
| **状態管理** | デコレータがフィールドを持てる | クロージャで保持 |
| **テスト** | 個別にテスト可能 | 関数単位でテスト |

### 比較表 3: Decorator vs Proxy vs Adapter

| 観点 | Decorator | Proxy | Adapter |
|------|:---:|:---:|:---:|
| **目的** | 機能**追加** | アクセス**制御** | インタフェース**変換** |
| **インタフェース** | **同じ** | **同じ** | **変換** |
| **積み重ね** | **可能** | 通常1層 | 通常1層 |
| **RealSubject管理** | 外部から受け取る | **自身で管理** | 外部から受け取る |
| **OCP** | **準拠** | 準拠 | 準拠 |

---

## 4. アンチパターン

### アンチパターン 1: デコレータの過剰な積み重ね

```typescript
// NG: 5層以上のデコレータ → デバッグ困難
const client = new MetricsDecorator(
  new CircuitBreakerDecorator(
    new RetryDecorator(
      new TimeoutDecorator(
        new LoggingDecorator(
          new AuthDecorator(
            new CacheDecorator(
              new FetchClient()
            )
          )
        )
      )
    )
  )
);
// 7層のネスト → スタックトレースが非常に深い
```

```typescript
// OK: ミドルウェアパターンやパイプラインで宣言的に構成
const client = createHttpClient({
  middlewares: [
    metrics(),
    circuitBreaker({ threshold: 5 }),
    retry({ maxRetries: 3 }),
    timeout({ ms: 5000 }),
    logging(),
    auth({ tokenProvider }),
    cache({ ttl: 60000 }),
  ],
});
```

---

### アンチパターン 2: デコレータがインタフェース外のメソッドに依存

```typescript
// NG: Component インタフェースにない getFilename() にキャストしてアクセス
class BadCachingDecorator implements DataSource {
  constructor(private wrapped: DataSource) {}

  read(): string {
    // 具象クラスに依存 → Decorator パターンの利点が失われる
    const name = (this.wrapped as FileDataSource).getFilename();
    const cached = this.cache.get(name);
    if (cached) return cached;
    return this.wrapped.read();
  }

  write(data: string): void {
    this.wrapped.write(data);
  }
}
```

```typescript
// OK: Component インタフェースのみに依存
class GoodCachingDecorator implements DataSource {
  private cachedData: string | null = null;

  constructor(private wrapped: DataSource) {}

  read(): string {
    if (this.cachedData !== null) return this.cachedData;
    this.cachedData = this.wrapped.read();
    return this.cachedData;
  }

  write(data: string): void {
    this.cachedData = null; // キャッシュ無効化
    this.wrapped.write(data);
  }
}
```

---

### アンチパターン 3: デコレータ順序の暗黙的な依存

```typescript
// NG: デコレータの順序を間違えると壊れる
// 暗号化してから圧縮すると、暗号化データは圧縮効率が悪い

// 悪い順序（暗号化 → 圧縮: 圧縮効率が悪い）
const bad = new CompressionDecorator(
  new EncryptionDecorator(new FileDataSource("data.txt"))
);

// 良い順序（圧縮 → 暗号化: 圧縮効率が良い）
const good = new EncryptionDecorator(
  new CompressionDecorator(new FileDataSource("data.txt"))
);
```

```typescript
// OK: Builder パターンで順序を制御し、ドキュメント化する
class DataSourceBuilder {
  private decorators: Array<(ds: DataSource) => DataSource> = [];

  constructor(private base: DataSource) {}

  // 圧縮 → 暗号化の順序を Builder が保証
  withCompressionAndEncryption(): this {
    this.decorators.push(ds => new CompressionDecorator(ds));
    this.decorators.push(ds => new EncryptionDecorator(ds));
    return this;
  }

  withLogging(): this {
    this.decorators.push(ds => new LoggingDecorator(ds));
    return this;
  }

  build(): DataSource {
    return this.decorators.reduce(
      (ds, decorator) => decorator(ds),
      this.base
    );
  }
}
```

---

## 5. エッジケースと注意点

### エッジケース 1: デコレータ内での例外処理

```typescript
// デコレータが例外を握りつぶすと、デバッグが困難になる
class SafeDecorator implements DataSource {
  constructor(private wrapped: DataSource) {}

  read(): string {
    try {
      return this.wrapped.read();
    } catch (error) {
      // NG: 例外を握りつぶして空文字を返す
      // return "";

      // OK: ログを出力してから再 throw
      console.error("Read failed:", error);
      throw error;
    }
  }

  write(data: string): void {
    this.wrapped.write(data);
  }
}
```

### エッジケース 2: デコレータの等価性とアイデンティティ

```typescript
const base = new FileDataSource("data.txt");
const decorated = new LoggingDecorator(base);

// デコレータは元のオブジェクトとは別のインスタンス
console.log(decorated === base);           // false
console.log(decorated instanceof FileDataSource); // false

// Set や Map のキーとして使う場合に注意
const set = new Set<DataSource>();
set.add(base);
set.add(decorated);
console.log(set.size); // 2（同じ base を指すが別オブジェクト）
```

### エッジケース 3: 非同期デコレータの順序保証

```typescript
// 非同期デコレータでは、前後処理の順序に注意
class AsyncLoggingDecorator implements AsyncDataSource {
  constructor(private wrapped: AsyncDataSource) {}

  async read(): Promise<string> {
    console.log("Before read");
    const result = await this.wrapped.read();
    console.log("After read"); // await の後なので確実に後処理
    return result;
  }
}
```

---

## 6. トレードオフ分析

```
[使うべき場面] ✅
┌─────────────────────────────────────────────────┐
│ 1. 機能の自由な組み合わせが必要                   │
│    例: ストリーム処理、HTTP ミドルウェア           │
│                                                  │
│ 2. 既存コードを変更せずに機能追加したい           │
│    例: サードパーティライブラリの拡張             │
│                                                  │
│ 3. 横断的関心事の分離                             │
│    例: ログ、キャッシュ、認証、リトライ           │
│                                                  │
│ 4. 実行時に機能の ON/OFF を切り替えたい           │
│    例: フィーチャーフラグ、設定ベースの切替       │
└─────────────────────────────────────────────────┘

[使うべきでない場面] ❌
┌─────────────────────────────────────────────────┐
│ 1. 機能の組み合わせが固定的                       │
│    → 継承やメソッドの直接追加の方がシンプル       │
│                                                  │
│ 2. デコレータの順序が重要で間違えやすい           │
│    → 順序を強制する仕組み（Builder等）が必要     │
│                                                  │
│ 3. パフォーマンスが最重要                         │
│    → 間接呼び出しのオーバーヘッドが問題          │
│                                                  │
│ 4. チーム全員がパターンを理解していない           │
│    → 可読性が低下する                            │
└─────────────────────────────────────────────────┘
```

---

## 7. 演習問題

### 演習 1（基礎）: テキスト変換 Decorator

`TextProcessor` インタフェースに対して3つのデコレータを実装してください。

**要件**:
- `TextProcessor`: `process(text: string): string`
- `UpperCaseDecorator`: 全て大文字に変換
- `TrimDecorator`: 前後の空白を除去
- `CensorDecorator`: 指定した単語を `***` に置換

```typescript
// テスト
const processor: TextProcessor = new CensorDecorator(
  new TrimDecorator(
    new UpperCaseDecorator(
      new PlainTextProcessor()
    )
  ),
  ["bad", "ugly"]
);

console.log(processor.process("  Hello bad World  "));
// "HELLO *** WORLD"
```

**期待される出力**: `HELLO *** WORLD`

<details>
<summary>解答例</summary>

```typescript
interface TextProcessor {
  process(text: string): string;
}

class PlainTextProcessor implements TextProcessor {
  process(text: string): string { return text; }
}

class UpperCaseDecorator implements TextProcessor {
  constructor(private wrapped: TextProcessor) {}
  process(text: string): string {
    return this.wrapped.process(text).toUpperCase();
  }
}

class TrimDecorator implements TextProcessor {
  constructor(private wrapped: TextProcessor) {}
  process(text: string): string {
    return this.wrapped.process(text).trim();
  }
}

class CensorDecorator implements TextProcessor {
  constructor(private wrapped: TextProcessor, private words: string[]) {}
  process(text: string): string {
    let result = this.wrapped.process(text);
    for (const word of this.words) {
      result = result.replace(new RegExp(word, "gi"), "***");
    }
    return result;
  }
}

const processor: TextProcessor = new CensorDecorator(
  new TrimDecorator(
    new UpperCaseDecorator(new PlainTextProcessor())
  ),
  ["BAD", "UGLY"]
);
console.log(processor.process("  Hello bad World  ")); // "HELLO *** WORLD"
```
</details>

---

### 演習 2（応用）: HTTP クライアント Decorator チェーン

以下のデコレータを組み合わせて堅牢な HTTP クライアントを構築してください。

**要件**:
- `HttpClient` インタフェース: `get(url): Promise<Response>`
- `RetryDecorator`: 指数バックオフでリトライ
- `CacheDecorator`: TTL 付きキャッシュ
- `LoggingDecorator`: リクエスト/レスポンスログ
- 適切な順序で積み重ねること

<details>
<summary>解答例</summary>

```typescript
interface HttpClient {
  get(url: string): Promise<{ status: number; body: string }>;
}

class SimpleClient implements HttpClient {
  async get(url: string) {
    return { status: 200, body: `Response from ${url}` };
  }
}

class RetryDecorator implements HttpClient {
  constructor(private client: HttpClient, private maxRetries = 3) {}
  async get(url: string) {
    for (let i = 0; i <= this.maxRetries; i++) {
      try {
        return await this.client.get(url);
      } catch (e) {
        if (i === this.maxRetries) throw e;
        await new Promise(r => setTimeout(r, 1000 * 2 ** i));
      }
    }
    throw new Error("Unreachable");
  }
}

class CacheDecorator implements HttpClient {
  private cache = new Map<string, { data: any; expiry: number }>();
  constructor(private client: HttpClient, private ttl = 60000) {}
  async get(url: string) {
    const cached = this.cache.get(url);
    if (cached && cached.expiry > Date.now()) return cached.data;
    const result = await this.client.get(url);
    this.cache.set(url, { data: result, expiry: Date.now() + this.ttl });
    return result;
  }
}

class LoggingDecorator implements HttpClient {
  constructor(private client: HttpClient) {}
  async get(url: string) {
    console.log(`GET ${url}`);
    const start = Date.now();
    const result = await this.client.get(url);
    console.log(`${result.status} (${Date.now() - start}ms)`);
    return result;
  }
}

// Logging → Cache → Retry → SimpleClient
const client = new LoggingDecorator(
  new CacheDecorator(
    new RetryDecorator(new SimpleClient(), 3),
    60000
  )
);
```
</details>

---

### 演習 3（上級）: 型安全な Decorator Builder

デコレータの積み重ねを型安全に構築できる Builder を実装してください。

**要件**:
- `DecoratorBuilder<T>` クラス
- `wrap(decorator)` メソッドでデコレータを追加
- `build()` で最終的なデコレートされたオブジェクトを返す
- TypeScript の型推論で、build() の戻り値型が正しく推論される

<details>
<summary>解答例</summary>

```typescript
class DecoratorBuilder<T> {
  private decorators: Array<(target: T) => T> = [];

  constructor(private base: T) {}

  static from<T>(base: T): DecoratorBuilder<T> {
    return new DecoratorBuilder(base);
  }

  wrap(decorator: (target: T) => T): this {
    this.decorators.push(decorator);
    return this;
  }

  build(): T {
    return this.decorators.reduce(
      (target, decorator) => decorator(target),
      this.base
    );
  }
}

// 使用例
const source = DecoratorBuilder.from<DataSource>(new FileDataSource("data.txt"))
  .wrap(ds => new EncryptionDecorator(ds))
  .wrap(ds => new CompressionDecorator(ds))
  .wrap(ds => new LoggingDecorator(ds))
  .build();
```
</details>

---

## 8. FAQ

### Q1: Decorator と Proxy の違いは何ですか？

構造は同じ（wrapped オブジェクトに委譲）ですが、**意図が異なります**:
- **Decorator**: 機能を**追加**する（ログ、圧縮、暗号化）
- **Proxy**: アクセスを**制御**する（遅延初期化、権限チェック、キャッシュ）

実用上は区別が曖昧になることもあります。キャッシュは「機能追加」とも「アクセス制御」とも解釈できます。

### Q2: TypeScript のデコレータ構文は GoF の Decorator パターンですか？

厳密には異なります。GoF Decorator はオブジェクトレベルの合成パターンで、実行時に動的に適用します。TypeScript/Python のデコレータ構文はクラス/メソッド定義へのメタプログラミングで、定義時に静的に適用されます。ただし「既存の振る舞いを非侵入的に拡張する」という動機は共通しています。

### Q3: React Hooks が登場して HOC（Decorator）は不要になりましたか？

多くのユースケースで Hooks が HOC を置き換えましたが、以下では HOC が依然有効です:
- **条件付きレンダリング**: 認証ガード（未認証ならリダイレクト）
- **Provider ラッピング**: テーマ、国際化などのコンテキスト提供
- **エラーバウンダリ**: クラスコンポーネントのライフサイクルが必要
- **クロスカッティング**: 複数コンポーネントへの一括適用

### Q4: デコレータの積み重ね順序はどう決めるべきですか？

一般的な原則:
1. **外側**: 横断的関心事（ログ、メトリクス）
2. **中間**: 回復力（リトライ、サーキットブレーカー、タイムアウト）
3. **内側**: ビジネス寄りの処理（認証、バリデーション）
4. **最内側**: ConcreteComponent

### Q5: Decorator パターンとミドルウェアパターンの関係は？

ミドルウェアパターンは Decorator パターンの宣言的な変形です。Express.js、Koa、Go の net/http など、多くの Web フレームワークがミドルウェアパターンを採用しています。本質は同じですが、ミドルウェアは配列ベースの設定が可能で、動的な追加・削除が容易です。

### Q6: Java の I/O ストリームが Decorator パターンの代表例とされるのはなぜですか？

Java の `java.io` パッケージは GoF Decorator パターンの最も有名な実装例です:
- `InputStream`/`OutputStream` = Component
- `FileInputStream`/`FileOutputStream` = ConcreteComponent
- `FilterInputStream`/`FilterOutputStream` = BaseDecorator
- `BufferedInputStream`, `DataInputStream`, `GZIPInputStream` = ConcreteDecorator

機能を自由に組み合わせられる一方で、ネストが深くなるという Decorator パターンのトレードオフも体現しています。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| **目的** | 動的に機能を追加する（継承の代替） |
| **手段** | 合成（has-a）でラッピング、同一インタフェースを維持 |
| **利点** | 柔軟な組み合わせ、OCP/SRP 準拠、実行時の動的構成 |
| **欠点** | 多層化でデバッグ困難、順序依存の可能性 |
| **GoF vs 言語** | GoF = オブジェクトレベル、TS/Python = メソッド/クラスレベル |
| **実装バリエーション** | クラス、関数合成、HOC、ミドルウェア、Stream Transform |
| **注意** | 過剰な積み重ね回避、インタフェース外依存禁止、順序の明文化 |

---

## 次に読むべきガイド

- [Proxy パターン](./03-proxy.md) — アクセス制御（Decorator と構造が同じだが目的が異なる）
- [Adapter パターン](./00-adapter.md) — インタフェース変換
- [Strategy パターン](../02-behavioral/01-strategy.md) — アルゴリズムの交換
- [Composite パターン](./04-composite.md) — ツリー構造
- [合成優先の原則](../../../03-software-design/clean-code-principles/docs/03-practices-advanced/01-composition-over-inheritance.md) — 継承より合成
- [Chain of Responsibility](../02-behavioral/02-chain-of-responsibility.md) — 処理チェーン

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media. — Chapter 3: Decorator Pattern
3. TC39 Decorators Proposal. https://github.com/tc39/proposal-decorators
4. Python Documentation — Decorators. https://docs.python.org/3/glossary.html#term-decorator
5. Refactoring.Guru — Decorator. https://refactoring.guru/design-patterns/decorator
6. Martin, R. C. (2003). *Agile Software Development*. Prentice Hall. — OCP (Open-Closed Principle)
