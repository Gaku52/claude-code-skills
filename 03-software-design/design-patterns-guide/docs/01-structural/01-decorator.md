# Decorator パターン

> オブジェクトに **動的に** 新しい機能を追加するための構造パターン。サブクラス化の代替として合成（コンポジション）を用いる。

---

## この章で学ぶこと

1. Decorator パターンの構造と、機能の積み重ね（チェーン）の仕組み
2. 継承ではなく合成で機能拡張を実現する利点とトレードオフ
3. TypeScript/Python の実デコレータ構文との関係と使い分け

---

## 1. Decorator の構造

```
+----------------+
|   Component    |
|  (interface)   |
+----------------+
| + operation()  |
+----------------+
     ^        ^
     |        |
+---------+  +-----------------+
|Concrete |  | BaseDecorator   |
|Component|  +-----------------+
+---------+  | - wrapped:      |
             |   Component     |
             | + operation()   |
             +-----------------+
                      ^
              ________|________
             |                 |
      +-------------+  +-------------+
      | DecoratorA  |  | DecoratorB  |
      +-------------+  +-------------+
      | + operation |  | + operation |
      +-------------+  +-------------+
```

---

## 2. デコレータの積み重ね

```
呼び出し方向 →

Client
  │
  ▼
┌─────────────┐
│ LoggingDeco  │  ← ログ出力
│┌───────────┐│
││CachingDeco ││  ← キャッシュ確認
││┌─────────┐││
│││ Service  │││  ← 本来の処理
││└─────────┘││
│└───────────┘│
└─────────────┘

実行順: Logging → Caching → Service → Caching → Logging
```

---

## 3. コード例

### コード例 1: データソース Decorator

```typescript
interface DataSource {
  read(): string;
  write(data: string): void;
}

class FileDataSource implements DataSource {
  constructor(private filename: string) {}
  read(): string { return `[content of ${this.filename}]`; }
  write(data: string): void { console.log(`Write to ${this.filename}: ${data}`); }
}

class EncryptionDecorator implements DataSource {
  constructor(private wrapped: DataSource) {}
  read(): string {
    const data = this.wrapped.read();
    return this.decrypt(data);
  }
  write(data: string): void {
    this.wrapped.write(this.encrypt(data));
  }
  private encrypt(data: string): string { return btoa(data); }
  private decrypt(data: string): string { return atob(data); }
}

class CompressionDecorator implements DataSource {
  constructor(private wrapped: DataSource) {}
  read(): string {
    const data = this.wrapped.read();
    return this.decompress(data);
  }
  write(data: string): void {
    this.wrapped.write(this.compress(data));
  }
  private compress(data: string): string { return `compressed(${data})`; }
  private decompress(data: string): string { return data.replace(/compressed\(|\)/g, ""); }
}

// 積み重ね
const source: DataSource = new CompressionDecorator(
  new EncryptionDecorator(
    new FileDataSource("data.txt")
  )
);
source.write("Hello"); // 圧縮 → 暗号化 → ファイル書込
```

### コード例 2: HTTP クライアント Decorator

```typescript
interface HttpClient {
  get(url: string): Promise<Response>;
}

class FetchClient implements HttpClient {
  async get(url: string): Promise<Response> {
    return fetch(url);
  }
}

class RetryDecorator implements HttpClient {
  constructor(private client: HttpClient, private maxRetries = 3) {}

  async get(url: string): Promise<Response> {
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

class LoggingDecorator implements HttpClient {
  constructor(private client: HttpClient) {}

  async get(url: string): Promise<Response> {
    console.log(`GET ${url}`);
    const start = Date.now();
    const res = await this.client.get(url);
    console.log(`${res.status} (${Date.now() - start}ms)`);
    return res;
  }
}

// 組み合わせ
const client: HttpClient = new LoggingDecorator(
  new RetryDecorator(new FetchClient(), 3)
);
```

### コード例 3: TypeScript デコレータ構文

```typescript
// TC39 Stage 3 Decorators (TypeScript 5+)
function logged(
  target: any,
  context: ClassMethodDecoratorContext
) {
  const methodName = String(context.name);
  return function (this: any, ...args: any[]) {
    console.log(`→ ${methodName}(${args.join(", ")})`);
    const result = target.call(this, ...args);
    console.log(`← ${methodName} = ${result}`);
    return result;
  };
}

class Calculator {
  @logged
  add(a: number, b: number): number {
    return a + b;
  }
}
```

### コード例 4: Python デコレータ

```python
import functools
import time

def retry(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    time.sleep(delay * (2 ** attempt))
        return wrapper
    return decorator

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time() - start:.3f}s")
        return result
    return wrapper

@retry(max_retries=3)
@timed
def fetch_data(url: str) -> str:
    # HTTP リクエスト
    ...
```

### コード例 5: React Higher-Order Component (HOC)

```typescript
// Decorator パターンの React 版
function withLoading<P extends object>(
  WrappedComponent: React.ComponentType<P>
) {
  return function WithLoadingComponent(
    props: P & { isLoading: boolean }
  ) {
    const { isLoading, ...rest } = props;
    if (isLoading) return <Spinner />;
    return <WrappedComponent {...(rest as P)} />;
  };
}

function withAuth<P extends object>(
  WrappedComponent: React.ComponentType<P>
) {
  return function WithAuthComponent(props: P) {
    const { user } = useAuth();
    if (!user) return <Navigate to="/login" />;
    return <WrappedComponent {...props} />;
  };
}

// 積み重ね
const ProtectedUserList = withAuth(withLoading(UserList));
```

---

## 4. 比較表

### 比較表 1: Decorator vs 継承

| 観点 | Decorator | 継承 |
|------|:---:|:---:|
| 機能追加 | 実行時（動的） | コンパイル時（静的） |
| 組み合わせ | 自由に積み重ね可 | クラス爆発 |
| 既存コード変更 | 不要 | サブクラス追加 |
| 複雑度 | 多層で増大 | 階層で増大 |
| デバッグ | スタック追跡が深い | 直感的 |

### 比較表 2: GoF Decorator vs 言語デコレータ構文

| 観点 | GoF Decorator | 言語デコレータ (TS/Python) |
|------|:---:|:---:|
| 適用対象 | オブジェクト | クラス/メソッド/プロパティ |
| 実行時変更 | Yes | No（定義時に適用） |
| インタフェース維持 | Yes | 暗黙的 |
| 主な用途 | 機能のラッピング | メタプログラミング |

---

## 5. アンチパターン

### アンチパターン 1: デコレータの過剰な積み重ね

```typescript
// BAD: 5層以上のデコレータ → デバッグ困難
const client = new MetricsDecorator(
  new CircuitBreakerDecorator(
    new RetryDecorator(
      new TimeoutDecorator(
        new LoggingDecorator(
          new AuthDecorator(
            new FetchClient()
          )
        )
      )
    )
  )
);
```

**改善**: ミドルウェアパターンやパイプラインに切り替え、宣言的に構成する。

### アンチパターン 2: デコレータがインタフェース外のメソッドに依存

```typescript
// BAD: ConcreteComponent 固有のメソッドにキャストしてアクセス
class CachingDecorator implements DataSource {
  read(): string {
    // Component インタフェースにない getFilename() を呼ぶ
    const name = (this.wrapped as FileDataSource).getFilename();
    // ...
  }
}
```

**改善**: デコレータは Component インタフェースのみに依存する。

---

## 6. FAQ

### Q1: Decorator と Proxy の違いは何ですか？

構造は同じですが意図が異なります。Decorator は **機能追加** 、Proxy は **アクセス制御**（遅延読み込み、権限チェック等）が目的です。

### Q2: TypeScript のデコレータ構文は GoF の Decorator パターンですか？

厳密には異なります。GoF Decorator はオブジェクトレベルの合成、言語デコレータはクラス/メソッド定義へのメタプログラミングです。ただし動機（機能追加を非侵入的に行う）は共通しています。

### Q3: React Hooks が登場して HOC は不要になりましたか？

多くのユースケースで Hooks が HOC を置き換えましたが、条件付きレンダリング（認証ガード等）や Provider ラッピングでは HOC が依然有効です。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | 動的に機能を追加する |
| 手段 | 合成（has-a）でラッピング |
| 利点 | 柔軟な組み合わせ、OCP 準拠 |
| 欠点 | 多層化でデバッグ困難 |
| 言語サポート | TS/Python のデコレータ構文 |

---

## 次に読むべきガイド

- [Proxy パターン](./03-proxy.md) — アクセス制御
- [Composite パターン](./04-composite.md) — ツリー構造
- [合成優先の原則](../../../clean-code-principles/docs/03-practices-advanced/01-composition-over-inheritance.md)

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. TC39 Decorators Proposal. https://github.com/tc39/proposal-decorators
3. Python Documentation — Decorators. https://docs.python.org/3/glossary.html#term-decorator
