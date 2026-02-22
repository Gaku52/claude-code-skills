# 構造パターン（Structural Patterns）

> クラスやオブジェクトの組み合わせ方に関するパターン。Adapter、Decorator、Facade、Proxy、Composite、Bridge、Flyweight の7つを実践的に解説。

## この章で学ぶこと

- [ ] 各構造パターンの目的と適用場面を理解する
- [ ] 各パターンのコード実装を複数言語で把握する
- [ ] パターンの組み合わせと使い分けを学ぶ
- [ ] 現代のフレームワーク（React、Express、NestJS等）での応用を知る
- [ ] テスタビリティとメンテナンス性を考慮した構造設計ができるようになる

---

## 1. Adapter パターン

### 1.1 概要と目的

```
目的: 互換性のないインターフェースを変換して接続する

  ┌─────────┐     ┌─────────┐     ┌──────────┐
  │ Client  │────→│ Adapter │────→│ Adaptee  │
  │         │     │ (変換)  │     │ (既存)   │
  └─────────┘     └─────────┘     └──────────┘

いつ使うか:
  → 既存のライブラリを自分たちのインターフェースに合わせたい
  → レガシーコードを新しいシステムに統合したい
  → サードパーティAPIの差異を吸収したい
  → テスト時に外部依存を差し替えたい
```

### 1.2 クラスアダプタとオブジェクトアダプタ

```typescript
// 既存のライブラリ（変更不可）
class LegacyPaymentGateway {
  processPayment(cardNumber: string, amount: number, currency: string): boolean {
    console.log(`Legacy: ${amount} ${currency} charged to ${cardNumber}`);
    return true;
  }

  refundPayment(transactionId: string, amount: number): boolean {
    console.log(`Legacy: Refund ${amount} for ${transactionId}`);
    return true;
  }

  getTransactionStatus(transactionId: string): string {
    return "completed";
  }
}

// 新しいインターフェース（自分たちの標準）
interface PaymentProcessor {
  pay(request: PaymentRequest): Promise<PaymentResult>;
  refund(transactionId: string, amount: number): Promise<RefundResult>;
  getStatus(transactionId: string): Promise<TransactionStatus>;
}

interface PaymentRequest {
  amount: number;
  currency: string;
  method: { type: "card"; cardNumber: string } | { type: "bank"; accountId: string };
}

interface PaymentResult {
  success: boolean;
  transactionId: string;
  timestamp: Date;
}

interface RefundResult {
  success: boolean;
  refundId: string;
}

type TransactionStatus = "pending" | "completed" | "failed" | "refunded";

// Adapter: 新旧を橋渡し
class LegacyPaymentAdapter implements PaymentProcessor {
  constructor(private legacy: LegacyPaymentGateway) {}

  async pay(request: PaymentRequest): Promise<PaymentResult> {
    if (request.method.type !== "card") {
      throw new Error("Legacy gateway only supports card payments");
    }

    const success = this.legacy.processPayment(
      request.method.cardNumber,
      request.amount,
      request.currency,
    );

    return {
      success,
      transactionId: crypto.randomUUID(),
      timestamp: new Date(),
    };
  }

  async refund(transactionId: string, amount: number): Promise<RefundResult> {
    const success = this.legacy.refundPayment(transactionId, amount);
    return {
      success,
      refundId: `ref-${crypto.randomUUID()}`,
    };
  }

  async getStatus(transactionId: string): Promise<TransactionStatus> {
    const legacyStatus = this.legacy.getTransactionStatus(transactionId);
    // レガシーのステータスを新しい型にマッピング
    const statusMap: Record<string, TransactionStatus> = {
      "completed": "completed",
      "pending": "pending",
      "error": "failed",
      "refund": "refunded",
    };
    return statusMap[legacyStatus] ?? "pending";
  }
}

// 利用側は PaymentProcessor のみに依存
const processor: PaymentProcessor = new LegacyPaymentAdapter(
  new LegacyPaymentGateway()
);
```

### 1.3 複数のアダプタによるプロバイダー抽象化

```typescript
// 複数の外部サービスを統一インターフェースで扱う
interface StorageProvider {
  upload(key: string, data: Buffer): Promise<string>;
  download(key: string): Promise<Buffer>;
  delete(key: string): Promise<void>;
  exists(key: string): Promise<boolean>;
  list(prefix: string): Promise<string[]>;
  getSignedUrl(key: string, expiresIn: number): Promise<string>;
}

// AWS S3 アダプタ
class S3StorageAdapter implements StorageProvider {
  constructor(private s3Client: S3Client, private bucket: string) {}

  async upload(key: string, data: Buffer): Promise<string> {
    await this.s3Client.send(new PutObjectCommand({
      Bucket: this.bucket,
      Key: key,
      Body: data,
    }));
    return `s3://${this.bucket}/${key}`;
  }

  async download(key: string): Promise<Buffer> {
    const response = await this.s3Client.send(new GetObjectCommand({
      Bucket: this.bucket,
      Key: key,
    }));
    return Buffer.from(await response.Body!.transformToByteArray());
  }

  async delete(key: string): Promise<void> {
    await this.s3Client.send(new DeleteObjectCommand({
      Bucket: this.bucket,
      Key: key,
    }));
  }

  async exists(key: string): Promise<boolean> {
    try {
      await this.s3Client.send(new HeadObjectCommand({
        Bucket: this.bucket,
        Key: key,
      }));
      return true;
    } catch {
      return false;
    }
  }

  async list(prefix: string): Promise<string[]> {
    const response = await this.s3Client.send(new ListObjectsV2Command({
      Bucket: this.bucket,
      Prefix: prefix,
    }));
    return response.Contents?.map(obj => obj.Key!) ?? [];
  }

  async getSignedUrl(key: string, expiresIn: number): Promise<string> {
    return `https://${this.bucket}.s3.amazonaws.com/${key}?expires=${expiresIn}`;
  }
}

// ローカルファイルシステムアダプタ（テスト・開発用）
class LocalStorageAdapter implements StorageProvider {
  constructor(private basePath: string) {}

  async upload(key: string, data: Buffer): Promise<string> {
    const fullPath = path.join(this.basePath, key);
    await fs.mkdir(path.dirname(fullPath), { recursive: true });
    await fs.writeFile(fullPath, data);
    return `file://${fullPath}`;
  }

  async download(key: string): Promise<Buffer> {
    const fullPath = path.join(this.basePath, key);
    return fs.readFile(fullPath);
  }

  async delete(key: string): Promise<void> {
    const fullPath = path.join(this.basePath, key);
    await fs.unlink(fullPath);
  }

  async exists(key: string): Promise<boolean> {
    try {
      await fs.access(path.join(this.basePath, key));
      return true;
    } catch {
      return false;
    }
  }

  async list(prefix: string): Promise<string[]> {
    const dir = path.join(this.basePath, prefix);
    try {
      const entries = await fs.readdir(dir, { recursive: true });
      return entries.map(e => path.join(prefix, e.toString()));
    } catch {
      return [];
    }
  }

  async getSignedUrl(key: string, _expiresIn: number): Promise<string> {
    return `file://${path.join(this.basePath, key)}`;
  }
}

// GCS アダプタ
class GCSStorageAdapter implements StorageProvider {
  constructor(private gcsClient: Storage, private bucket: string) {}

  async upload(key: string, data: Buffer): Promise<string> {
    const file = this.gcsClient.bucket(this.bucket).file(key);
    await file.save(data);
    return `gs://${this.bucket}/${key}`;
  }

  async download(key: string): Promise<Buffer> {
    const file = this.gcsClient.bucket(this.bucket).file(key);
    const [contents] = await file.download();
    return contents;
  }

  async delete(key: string): Promise<void> {
    await this.gcsClient.bucket(this.bucket).file(key).delete();
  }

  async exists(key: string): Promise<boolean> {
    const [exists] = await this.gcsClient.bucket(this.bucket).file(key).exists();
    return exists;
  }

  async list(prefix: string): Promise<string[]> {
    const [files] = await this.gcsClient.bucket(this.bucket).getFiles({ prefix });
    return files.map(f => f.name);
  }

  async getSignedUrl(key: string, expiresIn: number): Promise<string> {
    const file = this.gcsClient.bucket(this.bucket).file(key);
    const [url] = await file.getSignedUrl({
      action: "read",
      expires: Date.now() + expiresIn * 1000,
    });
    return url;
  }
}

// 環境に応じてアダプタを選択
function createStorage(env: string): StorageProvider {
  switch (env) {
    case "production":
      return new S3StorageAdapter(new S3Client({}), "prod-bucket");
    case "staging":
      return new GCSStorageAdapter(new Storage(), "staging-bucket");
    default:
      return new LocalStorageAdapter("./tmp/storage");
  }
}
```

### 1.4 Java での Adapter パターン

```java
// Java: Adapter パターン
// 既存ライブラリのインターフェース
public class ExternalLogService {
    public void writeLog(int severity, String component, String msg, long timestamp) {
        // 外部ログサービスへの書き込み
    }
}

// 自社標準のロガーインターフェース
public interface AppLogger {
    void debug(String message);
    void info(String message);
    void warn(String message);
    void error(String message, Throwable cause);
}

// Adapter
public class ExternalLogAdapter implements AppLogger {
    private final ExternalLogService service;
    private final String componentName;

    public ExternalLogAdapter(ExternalLogService service, String componentName) {
        this.service = service;
        this.componentName = componentName;
    }

    @Override
    public void debug(String message) {
        service.writeLog(0, componentName, message, System.currentTimeMillis());
    }

    @Override
    public void info(String message) {
        service.writeLog(1, componentName, message, System.currentTimeMillis());
    }

    @Override
    public void warn(String message) {
        service.writeLog(2, componentName, message, System.currentTimeMillis());
    }

    @Override
    public void error(String message, Throwable cause) {
        String fullMessage = message + "\n" + cause.toString();
        service.writeLog(3, componentName, fullMessage, System.currentTimeMillis());
    }
}
```

---

## 2. Decorator パターン

### 2.1 概要と目的

```
目的: 既存オブジェクトに動的に機能を追加する

  ┌────────┐     ┌───────────┐     ┌───────────┐
  │ Client │────→│ Decorator │────→│ Component │
  │        │     │ (機能追加)│     │ (元)      │
  └────────┘     └───────────┘     └───────────┘

  複数のDecoratorを重ねられる（入れ子）

いつ使うか:
  → 既存クラスを変更せずに機能を追加したい
  → 機能の組み合わせが多く、継承では爆発する
  → 実行時に動的に機能を着脱したい
  → ミドルウェア・パイプライン処理
```

### 2.2 HTTPクライアントのデコレータ

```typescript
// コンポーネントインターフェース
interface HttpClient {
  request(url: string, options?: RequestInit): Promise<Response>;
}

// 基本実装
class BasicHttpClient implements HttpClient {
  async request(url: string, options?: RequestInit): Promise<Response> {
    return fetch(url, options);
  }
}

// Decorator: ログ追加
class LoggingHttpClient implements HttpClient {
  constructor(private wrapped: HttpClient, private logger?: Console) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    const log = this.logger ?? console;
    const method = options?.method ?? "GET";
    log.log(`[HTTP] → ${method} ${url}`);
    const start = Date.now();
    try {
      const response = await this.wrapped.request(url, options);
      log.log(`[HTTP] ← ${response.status} (${Date.now() - start}ms)`);
      return response;
    } catch (error) {
      log.error(`[HTTP] ✗ ${method} ${url} failed (${Date.now() - start}ms)`, error);
      throw error;
    }
  }
}

// Decorator: リトライ追加
class RetryHttpClient implements HttpClient {
  constructor(
    private wrapped: HttpClient,
    private maxRetries: number = 3,
    private retryableStatuses: number[] = [408, 429, 500, 502, 503, 504],
  ) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    let lastError: Error | undefined;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const response = await this.wrapped.request(url, options);
        if (this.retryableStatuses.includes(response.status) && attempt < this.maxRetries) {
          const backoff = Math.min(1000 * Math.pow(2, attempt), 10000);
          await new Promise(r => setTimeout(r, backoff));
          continue;
        }
        return response;
      } catch (error) {
        lastError = error as Error;
        if (attempt < this.maxRetries) {
          const backoff = Math.min(1000 * Math.pow(2, attempt), 10000);
          await new Promise(r => setTimeout(r, backoff));
        }
      }
    }
    throw lastError ?? new Error("Max retries exceeded");
  }
}

// Decorator: 認証ヘッダー追加
class AuthHttpClient implements HttpClient {
  constructor(
    private wrapped: HttpClient,
    private tokenProvider: () => string | Promise<string>,
  ) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    const token = await this.tokenProvider();
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${token}`);
    return this.wrapped.request(url, { ...options, headers });
  }
}

// Decorator: キャッシュ追加
class CachingHttpClient implements HttpClient {
  private cache = new Map<string, { response: Response; expiry: number }>();

  constructor(
    private wrapped: HttpClient,
    private ttlMs: number = 60000,
  ) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    // GET リクエストのみキャッシュ
    if (options?.method && options.method !== "GET") {
      return this.wrapped.request(url, options);
    }

    const cacheKey = url;
    const cached = this.cache.get(cacheKey);
    if (cached && cached.expiry > Date.now()) {
      return cached.response.clone();
    }

    const response = await this.wrapped.request(url, options);
    if (response.ok) {
      this.cache.set(cacheKey, {
        response: response.clone(),
        expiry: Date.now() + this.ttlMs,
      });
    }
    return response;
  }
}

// Decorator: レートリミット追加
class RateLimitHttpClient implements HttpClient {
  private requestTimes: number[] = [];

  constructor(
    private wrapped: HttpClient,
    private maxRequests: number = 10,
    private windowMs: number = 1000,
  ) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    const now = Date.now();
    this.requestTimes = this.requestTimes.filter(t => now - t < this.windowMs);

    if (this.requestTimes.length >= this.maxRequests) {
      const oldestInWindow = this.requestTimes[0];
      const waitTime = this.windowMs - (now - oldestInWindow);
      await new Promise(r => setTimeout(r, waitTime));
    }

    this.requestTimes.push(Date.now());
    return this.wrapped.request(url, options);
  }
}

// Decorator: タイムアウト追加
class TimeoutHttpClient implements HttpClient {
  constructor(
    private wrapped: HttpClient,
    private timeoutMs: number = 30000,
  ) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      return await this.wrapped.request(url, {
        ...options,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
  }
}

// デコレータを重ねる
const client = new LoggingHttpClient(
  new RateLimitHttpClient(
    new RetryHttpClient(
      new TimeoutHttpClient(
        new CachingHttpClient(
          new AuthHttpClient(
            new BasicHttpClient(),
            () => "my-token"
          ),
          60000 // 1分キャッシュ
        ),
        10000 // 10秒タイムアウト
      ),
      3 // 3回リトライ
    ),
    10, 1000 // 1秒あたり10リクエスト
  )
);

// リクエスト → RateLimit → Retry → Timeout → Cache → Auth → 基本 の順で処理
```

### 2.3 ストリーム処理のデコレータ

```typescript
// データ変換パイプラインのデコレータ
interface DataTransformer<T> {
  transform(data: T[]): T[];
}

class BaseTransformer<T> implements DataTransformer<T> {
  transform(data: T[]): T[] {
    return data;
  }
}

// フィルタリングデコレータ
class FilterTransformer<T> implements DataTransformer<T> {
  constructor(
    private wrapped: DataTransformer<T>,
    private predicate: (item: T) => boolean,
  ) {}

  transform(data: T[]): T[] {
    return this.wrapped.transform(data).filter(this.predicate);
  }
}

// ソートデコレータ
class SortTransformer<T> implements DataTransformer<T> {
  constructor(
    private wrapped: DataTransformer<T>,
    private comparator: (a: T, b: T) => number,
  ) {}

  transform(data: T[]): T[] {
    return [...this.wrapped.transform(data)].sort(this.comparator);
  }
}

// ページネーションデコレータ
class PaginationTransformer<T> implements DataTransformer<T> {
  constructor(
    private wrapped: DataTransformer<T>,
    private page: number,
    private pageSize: number,
  ) {}

  transform(data: T[]): T[] {
    const transformed = this.wrapped.transform(data);
    const start = (this.page - 1) * this.pageSize;
    return transformed.slice(start, start + this.pageSize);
  }
}

// マッピングデコレータ
class MapTransformer<T> implements DataTransformer<T> {
  constructor(
    private wrapped: DataTransformer<T>,
    private mapper: (item: T) => T,
  ) {}

  transform(data: T[]): T[] {
    return this.wrapped.transform(data).map(this.mapper);
  }
}

// 使用例: ユーザーリストの加工パイプライン
interface User {
  id: string;
  name: string;
  age: number;
  active: boolean;
}

const pipeline = new PaginationTransformer(
  new SortTransformer(
    new FilterTransformer(
      new BaseTransformer<User>(),
      user => user.active  // アクティブユーザーのみ
    ),
    (a, b) => a.name.localeCompare(b.name)  // 名前順
  ),
  1, 20  // 1ページ目、20件
);

const result = pipeline.transform(allUsers);
```

### 2.4 Python での Decorator パターン

```python
# Python: デコレータパターン（関数デコレータとクラスデコレータ）
import functools
import time
import logging
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
R = TypeVar('R')

# 関数デコレータ: ログ
def log_calls(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logging.info(f"Calling {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} returned {result}")
        return result
    return wrapper

# 関数デコレータ: 実行時間計測
def timed(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

# 関数デコレータ: リトライ
def retry(max_retries: int = 3, delay: float = 1.0):
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        time.sleep(delay * (2 ** attempt))
            raise last_error
        return wrapper
    return decorator

# 関数デコレータ: キャッシュ（メモ化）
def memoize(func: Callable[P, R]) -> Callable[P, R]:
    cache: dict = {}

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

# デコレータの重ね掛け
@log_calls
@timed
@retry(max_retries=3, delay=0.5)
def fetch_data(url: str) -> dict:
    """外部APIからデータを取得"""
    import requests
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# 呼び出し時: retry → timed → log_calls の順で処理される
result = fetch_data("https://api.example.com/data")
```

---

## 3. Facade パターン

### 3.1 概要と目的

```
目的: 複雑なサブシステムにシンプルなインターフェースを提供する

  ┌────────┐     ┌──────────┐     ┌─────┐ ┌─────┐ ┌─────┐
  │ Client │────→│  Facade  │────→│ SubA│ │ SubB│ │ SubC│
  │        │     │ (窓口)   │     └─────┘ └─────┘ └─────┘
  └────────┘     └──────────┘

いつ使うか:
  → 複雑なサブシステムの操作を簡略化したい
  → レイヤー間の依存を減らしたい
  → ライブラリの使い方を統一したい
  → マイクロサービスのAPIゲートウェイ
```

### 3.2 Eコマースの注文処理

```typescript
// 複雑なサブシステム群
class InventoryService {
  checkAvailability(productId: string, quantity: number): boolean {
    console.log(`Checking inventory for ${productId}`);
    return true;
  }

  reserveStock(productId: string, quantity: number): string {
    console.log(`Reserved ${quantity} of ${productId}`);
    return `reservation-${Date.now()}`;
  }

  releaseStock(reservationId: string): void {
    console.log(`Released reservation ${reservationId}`);
  }
}

class PricingService {
  calculatePrice(productId: string, quantity: number): number {
    return 1000 * quantity;
  }

  applyDiscount(price: number, discountCode?: string): number {
    if (discountCode === "SAVE10") return price * 0.9;
    return price;
  }

  calculateTax(price: number, region: string): number {
    const taxRates: Record<string, number> = {
      "JP": 0.10,
      "US": 0.08,
      "EU": 0.20,
    };
    return price * (taxRates[region] ?? 0.10);
  }

  calculateShipping(weight: number, region: string): number {
    return weight * (region === "JP" ? 100 : 500);
  }
}

class PaymentService {
  async authorize(amount: number, paymentMethod: PaymentMethod): Promise<string> {
    console.log(`Authorizing payment of ¥${amount}`);
    return `auth-${Date.now()}`;
  }

  async capture(authorizationId: string): Promise<string> {
    console.log(`Capturing payment ${authorizationId}`);
    return `payment-${Date.now()}`;
  }

  async void(authorizationId: string): Promise<void> {
    console.log(`Voiding authorization ${authorizationId}`);
  }
}

class ShippingService {
  async createShipment(address: Address, items: OrderItem[]): Promise<string> {
    console.log(`Creating shipment to ${address.city}`);
    return `ship-${Date.now()}`;
  }

  async getTrackingUrl(shipmentId: string): Promise<string> {
    return `https://tracking.example.com/${shipmentId}`;
  }
}

class NotificationService {
  async sendOrderConfirmation(email: string, orderId: string): Promise<void> {
    console.log(`Confirmation email sent to ${email}`);
  }

  async sendShippingNotification(email: string, trackingUrl: string): Promise<void> {
    console.log(`Shipping notification sent to ${email}`);
  }
}

// Facade: 注文処理の全体を1つのシンプルなAPIに
class OrderFacade {
  constructor(
    private inventory: InventoryService,
    private pricing: PricingService,
    private payment: PaymentService,
    private shipping: ShippingService,
    private notification: NotificationService,
  ) {}

  async placeOrder(order: OrderRequest): Promise<OrderResult> {
    // 1. 在庫チェック
    for (const item of order.items) {
      if (!this.inventory.checkAvailability(item.productId, item.quantity)) {
        return { success: false, error: `${item.productId} is out of stock` };
      }
    }

    // 2. 在庫予約
    const reservations: string[] = [];
    for (const item of order.items) {
      reservations.push(
        this.inventory.reserveStock(item.productId, item.quantity)
      );
    }

    try {
      // 3. 価格計算
      let totalPrice = 0;
      for (const item of order.items) {
        const price = this.pricing.calculatePrice(item.productId, item.quantity);
        totalPrice += price;
      }
      totalPrice = this.pricing.applyDiscount(totalPrice, order.discountCode);
      const tax = this.pricing.calculateTax(totalPrice, order.shippingAddress.country);
      const shipping = this.pricing.calculateShipping(
        order.items.reduce((sum, item) => sum + item.weight * item.quantity, 0),
        order.shippingAddress.country,
      );
      const grandTotal = totalPrice + tax + shipping;

      // 4. 決済
      const authId = await this.payment.authorize(grandTotal, order.paymentMethod);
      const paymentId = await this.payment.capture(authId);

      // 5. 配送手配
      const shipmentId = await this.shipping.createShipment(
        order.shippingAddress,
        order.items,
      );
      const trackingUrl = await this.shipping.getTrackingUrl(shipmentId);

      // 6. 通知
      await this.notification.sendOrderConfirmation(order.customerEmail, paymentId);
      await this.notification.sendShippingNotification(order.customerEmail, trackingUrl);

      return {
        success: true,
        orderId: paymentId,
        total: grandTotal,
        trackingUrl,
      };
    } catch (error) {
      // エラー時は在庫を解放
      for (const reservationId of reservations) {
        this.inventory.releaseStock(reservationId);
      }
      return { success: false, error: (error as Error).message };
    }
  }
}

// 利用側はシンプル
const orderFacade = new OrderFacade(
  new InventoryService(),
  new PricingService(),
  new PaymentService(),
  new ShippingService(),
  new NotificationService(),
);

const result = await orderFacade.placeOrder({
  customerEmail: "tanaka@example.com",
  items: [
    { productId: "P-001", quantity: 2, weight: 0.5 },
    { productId: "P-002", quantity: 1, weight: 1.0 },
  ],
  shippingAddress: { city: "東京", country: "JP", zip: "100-0001" },
  paymentMethod: { type: "card", cardNumber: "4111..." },
  discountCode: "SAVE10",
});
```

### 3.3 メディアプレイヤーのFacade

```typescript
// 複雑なサブシステム
class VideoDecoder {
  decode(file: string): Buffer {
    console.log(`Decoding video: ${file}`);
    return Buffer.alloc(0);
  }
  getSupportedFormats(): string[] {
    return ["mp4", "avi", "mkv", "webm"];
  }
}

class AudioDecoder {
  decode(file: string): Buffer {
    console.log(`Decoding audio: ${file}`);
    return Buffer.alloc(0);
  }
  setVolume(level: number): void {
    console.log(`Volume set to ${level}%`);
  }
}

class SubtitleParser {
  parse(file: string): string[] {
    console.log(`Parsing subtitles: ${file}`);
    return [];
  }
  getSupportedFormats(): string[] {
    return ["srt", "vtt", "ass"];
  }
}

class VideoRenderer {
  render(video: Buffer, audio: Buffer, subs: string[]): void {
    console.log("Rendering video with audio and subtitles");
  }
  setResolution(width: number, height: number): void {
    console.log(`Resolution set to ${width}x${height}`);
  }
}

// Facade: シンプルなAPI
class MediaPlayer {
  private videoDecoder = new VideoDecoder();
  private audioDecoder = new AudioDecoder();
  private subtitleParser = new SubtitleParser();
  private renderer = new VideoRenderer();
  private currentVolume = 50;

  play(videoFile: string, subtitleFile?: string): void {
    const video = this.videoDecoder.decode(videoFile);
    const audio = this.audioDecoder.decode(videoFile);
    const subs = subtitleFile ? this.subtitleParser.parse(subtitleFile) : [];
    this.audioDecoder.setVolume(this.currentVolume);
    this.renderer.render(video, audio, subs);
  }

  setVolume(level: number): void {
    this.currentVolume = Math.max(0, Math.min(100, level));
    this.audioDecoder.setVolume(this.currentVolume);
  }

  setResolution(width: number, height: number): void {
    this.renderer.setResolution(width, height);
  }

  getSupportedFormats(): { video: string[]; subtitle: string[] } {
    return {
      video: this.videoDecoder.getSupportedFormats(),
      subtitle: this.subtitleParser.getSupportedFormats(),
    };
  }
}

// 利用側はシンプル
const player = new MediaPlayer();
player.setResolution(1920, 1080);
player.setVolume(75);
player.play("movie.mp4", "movie.srt");
```

---

## 4. Proxy パターン

### 4.1 概要と目的

```
目的: オブジェクトへのアクセスを制御する代理を提供する

種類:
  → 仮想Proxy: 遅延初期化（重いオブジェクトを必要時に生成）
  → 保護Proxy: アクセス制御（権限チェック）
  → キャッシュProxy: 結果のキャッシュ
  → リモートProxy: ネットワーク越しのオブジェクトアクセス
  → ログProxy: 操作の記録
```

### 4.2 キャッシュProxy

```typescript
// キャッシュProxy
interface DataService {
  fetchUser(id: string): Promise<User>;
  fetchUsers(ids: string[]): Promise<User[]>;
  searchUsers(query: string): Promise<User[]>;
}

class RealDataService implements DataService {
  async fetchUser(id: string): Promise<User> {
    console.log(`Fetching user ${id} from database...`);
    // 重いDB/APIアクセスをシミュレート
    await new Promise(r => setTimeout(r, 100));
    return { id, name: "User " + id, email: `user${id}@example.com` };
  }

  async fetchUsers(ids: string[]): Promise<User[]> {
    console.log(`Fetching ${ids.length} users from database...`);
    return Promise.all(ids.map(id => this.fetchUser(id)));
  }

  async searchUsers(query: string): Promise<User[]> {
    console.log(`Searching users with query: ${query}`);
    return [];
  }
}

class CachingProxy implements DataService {
  private cache = new Map<string, { data: any; expiry: number }>();
  private pendingRequests = new Map<string, Promise<any>>();

  constructor(
    private real: DataService,
    private ttlMs: number = 60000,
  ) {}

  async fetchUser(id: string): Promise<User> {
    const cacheKey = `user:${id}`;

    // キャッシュヒット
    const cached = this.cache.get(cacheKey);
    if (cached && cached.expiry > Date.now()) {
      console.log(`Cache hit for user ${id}`);
      return cached.data;
    }

    // リクエストの重複排除（同じIDへの同時リクエストを1つにまとめる）
    if (this.pendingRequests.has(cacheKey)) {
      return this.pendingRequests.get(cacheKey)!;
    }

    const promise = this.real.fetchUser(id).then(data => {
      this.cache.set(cacheKey, { data, expiry: Date.now() + this.ttlMs });
      this.pendingRequests.delete(cacheKey);
      return data;
    });

    this.pendingRequests.set(cacheKey, promise);
    return promise;
  }

  async fetchUsers(ids: string[]): Promise<User[]> {
    return Promise.all(ids.map(id => this.fetchUser(id)));
  }

  async searchUsers(query: string): Promise<User[]> {
    const cacheKey = `search:${query}`;
    const cached = this.cache.get(cacheKey);
    if (cached && cached.expiry > Date.now()) {
      return cached.data;
    }
    const data = await this.real.searchUsers(query);
    this.cache.set(cacheKey, { data, expiry: Date.now() + this.ttlMs });
    return data;
  }

  invalidate(id: string): void {
    this.cache.delete(`user:${id}`);
  }

  invalidateAll(): void {
    this.cache.clear();
  }
}

const service: DataService = new CachingProxy(new RealDataService());
await service.fetchUser("123"); // DB アクセス
await service.fetchUser("123"); // キャッシュから
```

### 4.3 保護Proxy（アクセス制御）

```typescript
// 保護Proxy: 権限チェック
interface AdminService {
  deleteUser(userId: string): Promise<void>;
  resetPassword(userId: string): Promise<string>;
  viewAuditLog(): Promise<AuditEntry[]>;
  changeRole(userId: string, role: string): Promise<void>;
  exportAllData(): Promise<Buffer>;
}

class RealAdminService implements AdminService {
  async deleteUser(userId: string): Promise<void> {
    console.log(`Deleting user ${userId}`);
  }

  async resetPassword(userId: string): Promise<string> {
    return "new-temp-password";
  }

  async viewAuditLog(): Promise<AuditEntry[]> {
    return [];
  }

  async changeRole(userId: string, role: string): Promise<void> {
    console.log(`Changed ${userId} role to ${role}`);
  }

  async exportAllData(): Promise<Buffer> {
    return Buffer.from("all data");
  }
}

class AdminServiceProxy implements AdminService {
  private permissionMap: Record<string, string[]> = {
    deleteUser: ["super_admin"],
    resetPassword: ["admin", "super_admin"],
    viewAuditLog: ["admin", "super_admin", "auditor"],
    changeRole: ["super_admin"],
    exportAllData: ["super_admin"],
  };

  constructor(
    private real: AdminService,
    private currentUser: { id: string; role: string },
    private auditLogger: AuditLogger,
  ) {}

  private checkPermission(operation: string): void {
    const allowedRoles = this.permissionMap[operation] ?? [];
    if (!allowedRoles.includes(this.currentUser.role)) {
      this.auditLogger.logUnauthorizedAccess(
        this.currentUser.id, operation
      );
      throw new Error(
        `Access denied: ${this.currentUser.role} cannot perform ${operation}`
      );
    }
    this.auditLogger.logAccess(this.currentUser.id, operation);
  }

  async deleteUser(userId: string): Promise<void> {
    this.checkPermission("deleteUser");
    return this.real.deleteUser(userId);
  }

  async resetPassword(userId: string): Promise<string> {
    this.checkPermission("resetPassword");
    return this.real.resetPassword(userId);
  }

  async viewAuditLog(): Promise<AuditEntry[]> {
    this.checkPermission("viewAuditLog");
    return this.real.viewAuditLog();
  }

  async changeRole(userId: string, role: string): Promise<void> {
    this.checkPermission("changeRole");
    return this.real.changeRole(userId, role);
  }

  async exportAllData(): Promise<Buffer> {
    this.checkPermission("exportAllData");
    return this.real.exportAllData();
  }
}
```

### 4.4 仮想Proxy（遅延初期化）

```typescript
// 仮想Proxy: 重いオブジェクトの遅延初期化
interface ImageRenderer {
  render(x: number, y: number): void;
  getWidth(): number;
  getHeight(): number;
}

class HighResImage implements ImageRenderer {
  private pixels: Buffer;

  constructor(private path: string) {
    // 重い処理: 高解像度画像の読み込み
    console.log(`Loading high-res image: ${path}`);
    this.pixels = Buffer.alloc(50 * 1024 * 1024); // 50MB
  }

  render(x: number, y: number): void {
    console.log(`Rendering ${this.path} at (${x}, ${y})`);
  }

  getWidth(): number { return 4096; }
  getHeight(): number { return 2160; }
}

class LazyImageProxy implements ImageRenderer {
  private realImage: HighResImage | null = null;
  private thumbnailPath: string;

  constructor(private imagePath: string) {
    this.thumbnailPath = imagePath.replace(/\.\w+$/, "_thumb.jpg");
  }

  private loadImage(): HighResImage {
    if (!this.realImage) {
      this.realImage = new HighResImage(this.imagePath);
    }
    return this.realImage;
  }

  render(x: number, y: number): void {
    // 実際のレンダリング時にのみ読み込む
    this.loadImage().render(x, y);
  }

  getWidth(): number {
    // メタデータはプロキシが持っていれば読み込み不要
    return 4096;
  }

  getHeight(): number {
    return 2160;
  }
}

// 1000枚の画像を持つギャラリー
// 全部ロードすると 50GB になるが、Proxy なら必要なものだけロード
const gallery: ImageRenderer[] = [];
for (let i = 0; i < 1000; i++) {
  gallery.push(new LazyImageProxy(`/images/photo_${i}.raw`));
}
// この時点ではメモリ消費は最小限
// 表示する画像だけが実際にロードされる
gallery[42].render(0, 0); // この時にだけ photo_42.raw がロードされる
```

### 4.5 JavaScript Proxy を使った実装

```typescript
// JavaScript のネイティブ Proxy を使った実装
function createValidationProxy<T extends object>(target: T, rules: ValidationRules<T>): T {
  return new Proxy(target, {
    set(obj: T, prop: string | symbol, value: any): boolean {
      const rule = rules[prop as keyof T];
      if (rule) {
        const error = rule(value);
        if (error) {
          throw new Error(`Validation failed for ${String(prop)}: ${error}`);
        }
      }
      (obj as any)[prop] = value;
      return true;
    },

    get(obj: T, prop: string | symbol): any {
      const value = (obj as any)[prop];
      // メソッドの場合は呼び出しログを追加
      if (typeof value === "function") {
        return function (...args: any[]) {
          console.log(`Called ${String(prop)}(${args.join(", ")})`);
          return value.apply(obj, args);
        };
      }
      return value;
    },
  });
}

// 使用例
const user = createValidationProxy(
  { name: "", age: 0, email: "" },
  {
    name: (v: string) => v.length === 0 ? "名前は必須です" : null,
    age: (v: number) => v < 0 || v > 150 ? "年齢は0-150の範囲" : null,
    email: (v: string) => !v.includes("@") ? "有効なメールアドレスではありません" : null,
  }
);

user.name = "太郎";    // OK
user.age = 30;         // OK
// user.age = -5;      // Error: Validation failed for age: 年齢は0-150の範囲
// user.email = "invalid"; // Error: Validation failed for email
```

---

## 5. Composite パターン

### 5.1 概要と目的

```
目的: 個別オブジェクトとオブジェクトの集合を同一視して扱う

  Component（共通インターフェース）
  ├── Leaf（葉: 個別要素）
  └── Composite（枝: 子要素を含む）

いつ使うか:
  → ツリー構造を表現したい
  → 個と集合を区別せずに扱いたい
  → 再帰的な構造（ファイルシステム、UIコンポーネント、組織図）
```

### 5.2 ファイルシステム

```typescript
// ファイルシステムの例
interface FileSystemEntry {
  name: string;
  size(): number;
  display(indent?: string): string;
  find(predicate: (entry: FileSystemEntry) => boolean): FileSystemEntry[];
  getPath(parentPath?: string): string;
}

class File implements FileSystemEntry {
  constructor(
    public name: string,
    private bytes: number,
    public readonly extension: string,
  ) {}

  size(): number { return this.bytes; }

  display(indent = ""): string {
    return `${indent}${this.name} (${this.formatSize()})`;
  }

  find(predicate: (entry: FileSystemEntry) => boolean): FileSystemEntry[] {
    return predicate(this) ? [this] : [];
  }

  getPath(parentPath = ""): string {
    return parentPath ? `${parentPath}/${this.name}` : this.name;
  }

  private formatSize(): string {
    if (this.bytes < 1024) return `${this.bytes}B`;
    if (this.bytes < 1024 * 1024) return `${(this.bytes / 1024).toFixed(1)}KB`;
    return `${(this.bytes / 1024 / 1024).toFixed(1)}MB`;
  }
}

class Directory implements FileSystemEntry {
  private children: FileSystemEntry[] = [];

  constructor(public name: string) {}

  add(entry: FileSystemEntry): this {
    this.children.push(entry);
    return this;
  }

  remove(name: string): boolean {
    const index = this.children.findIndex(c => c.name === name);
    if (index >= 0) {
      this.children.splice(index, 1);
      return true;
    }
    return false;
  }

  size(): number {
    return this.children.reduce((sum, child) => sum + child.size(), 0);
  }

  display(indent = ""): string {
    const lines = [`${indent}${this.name}/ (${this.formatSize()})`];
    for (const child of this.children) {
      lines.push(child.display(indent + "  "));
    }
    return lines.join("\n");
  }

  find(predicate: (entry: FileSystemEntry) => boolean): FileSystemEntry[] {
    const results: FileSystemEntry[] = [];
    if (predicate(this)) results.push(this);
    for (const child of this.children) {
      results.push(...child.find(predicate));
    }
    return results;
  }

  getPath(parentPath = ""): string {
    return parentPath ? `${parentPath}/${this.name}` : this.name;
  }

  getChildren(): FileSystemEntry[] {
    return [...this.children];
  }

  private formatSize(): string {
    const bytes = this.size();
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
  }
}

// File と Directory を統一的に扱える
const root = new Directory("src");
const components = new Directory("components");
components.add(new File("Button.tsx", 2048, "tsx"));
components.add(new File("Modal.tsx", 4096, "tsx"));
components.add(new File("Button.test.tsx", 1024, "tsx"));

const utils = new Directory("utils");
utils.add(new File("format.ts", 512, "ts"));
utils.add(new File("validate.ts", 768, "ts"));

root.add(components);
root.add(utils);
root.add(new File("index.ts", 256, "ts"));
root.add(new File("App.tsx", 3072, "tsx"));

console.log(root.display());
console.log(`Total: ${root.size()}B`);

// 検索: .test. を含むファイルを探す
const testFiles = root.find(entry => entry.name.includes(".test."));
console.log("Test files:", testFiles.map(f => f.name));
```

### 5.3 UIコンポーネントツリー

```typescript
// UIコンポーネントの Composite
interface UIComponent {
  render(): string;
  getWidth(): number;
  getHeight(): number;
  addEventListener(event: string, handler: () => void): void;
}

class Label implements UIComponent {
  constructor(private text: string, private fontSize: number = 14) {}

  render(): string {
    return `<span style="font-size: ${this.fontSize}px">${this.text}</span>`;
  }
  getWidth(): number { return this.text.length * this.fontSize * 0.6; }
  getHeight(): number { return this.fontSize * 1.5; }
  addEventListener(_event: string, _handler: () => void): void {}
}

class TextInput implements UIComponent {
  constructor(private placeholder: string, private width: number = 200) {}

  render(): string {
    return `<input type="text" placeholder="${this.placeholder}" style="width: ${this.width}px">`;
  }
  getWidth(): number { return this.width; }
  getHeight(): number { return 32; }
  addEventListener(_event: string, _handler: () => void): void {}
}

class Panel implements UIComponent {
  private children: UIComponent[] = [];
  private padding = 16;

  constructor(private direction: "horizontal" | "vertical" = "vertical") {}

  add(component: UIComponent): this {
    this.children.push(component);
    return this;
  }

  render(): string {
    const display = this.direction === "horizontal" ? "flex-direction: row" : "flex-direction: column";
    const childrenHtml = this.children.map(c => c.render()).join("\n");
    return `<div style="display: flex; ${display}; padding: ${this.padding}px; gap: 8px">\n${childrenHtml}\n</div>`;
  }

  getWidth(): number {
    if (this.direction === "horizontal") {
      return this.children.reduce((sum, c) => sum + c.getWidth(), 0) + this.padding * 2;
    }
    return Math.max(...this.children.map(c => c.getWidth()), 0) + this.padding * 2;
  }

  getHeight(): number {
    if (this.direction === "vertical") {
      return this.children.reduce((sum, c) => sum + c.getHeight(), 0) + this.padding * 2;
    }
    return Math.max(...this.children.map(c => c.getHeight()), 0) + this.padding * 2;
  }

  addEventListener(event: string, handler: () => void): void {
    this.children.forEach(c => c.addEventListener(event, handler));
  }
}

// ログインフォームの構築
const loginForm = new Panel("vertical")
  .add(new Label("ログイン", 24))
  .add(new Panel("vertical")
    .add(new Label("メールアドレス"))
    .add(new TextInput("email@example.com", 300))
  )
  .add(new Panel("vertical")
    .add(new Label("パスワード"))
    .add(new TextInput("••••••••", 300))
  );

console.log(loginForm.render());
console.log(`Form size: ${loginForm.getWidth()} x ${loginForm.getHeight()}`);
```

---

## 6. Bridge パターン

### 6.1 概要と目的

```
目的: 抽象化と実装を分離し、それぞれを独立に変更可能にする

  ┌───────────────┐          ┌─────────────────┐
  │  Abstraction  │ ────────→│ Implementation  │
  │  (抽象化)     │          │ (実装)          │
  └───────┬───────┘          └────────┬────────┘
          │                           │
  ┌───────┴───────┐          ┌────────┴────────┐
  │  Refined      │          │  Concrete       │
  │  Abstraction  │          │  Implementation │
  └───────────────┘          └─────────────────┘

いつ使うか:
  → 抽象化と実装の両方が拡張される可能性がある
  → クロスプラットフォーム対応
  → 継承の組み合わせ爆発を避けたい
```

### 6.2 通知システムの Bridge

```typescript
// Implementation: メッセージ送信の具体的な方法
interface MessageSender {
  send(to: string, title: string, body: string): Promise<boolean>;
  getName(): string;
}

class EmailSender implements MessageSender {
  async send(to: string, title: string, body: string): Promise<boolean> {
    console.log(`[Email] To: ${to}, Subject: ${title}, Body: ${body}`);
    return true;
  }
  getName(): string { return "Email"; }
}

class SmsSender implements MessageSender {
  async send(to: string, title: string, body: string): Promise<boolean> {
    console.log(`[SMS] To: ${to}, Message: ${title} - ${body}`);
    return true;
  }
  getName(): string { return "SMS"; }
}

class SlackSender implements MessageSender {
  async send(to: string, title: string, body: string): Promise<boolean> {
    console.log(`[Slack] Channel: ${to}, Title: ${title}, Body: ${body}`);
    return true;
  }
  getName(): string { return "Slack"; }
}

// Abstraction: 通知の種類
abstract class Notification {
  constructor(protected sender: MessageSender) {}

  abstract notify(to: string): Promise<boolean>;

  // 送信方法を動的に切り替え可能
  setSender(sender: MessageSender): void {
    this.sender = sender;
  }
}

class AlertNotification extends Notification {
  constructor(
    sender: MessageSender,
    private alertLevel: "warning" | "critical",
    private message: string,
  ) {
    super(sender);
  }

  async notify(to: string): Promise<boolean> {
    const prefix = this.alertLevel === "critical" ? "[CRITICAL]" : "[WARNING]";
    return this.sender.send(to, `${prefix} Alert`, this.message);
  }
}

class ReminderNotification extends Notification {
  constructor(
    sender: MessageSender,
    private task: string,
    private dueDate: Date,
  ) {
    super(sender);
  }

  async notify(to: string): Promise<boolean> {
    const formatted = this.dueDate.toLocaleDateString("ja-JP");
    return this.sender.send(to, `リマインダー: ${this.task}`, `期限: ${formatted}`);
  }
}

class PromotionNotification extends Notification {
  constructor(
    sender: MessageSender,
    private campaign: string,
    private discount: number,
  ) {
    super(sender);
  }

  async notify(to: string): Promise<boolean> {
    return this.sender.send(
      to,
      `${this.campaign} キャンペーン`,
      `今なら ${this.discount}% OFF!`,
    );
  }
}

// 使用例: 通知の種類 x 送信方法 を自由に組み合わせ
const emailSender = new EmailSender();
const smsSender = new SmsSender();
const slackSender = new SlackSender();

// クリティカルアラートをSlackに
const criticalAlert = new AlertNotification(slackSender, "critical", "サーバーダウン");
await criticalAlert.notify("#ops-alerts");

// 同じアラートをメールでも送信
criticalAlert.setSender(emailSender);
await criticalAlert.notify("ops-team@example.com");

// リマインダーをSMSで
const reminder = new ReminderNotification(smsSender, "レポート提出", new Date("2025-03-01"));
await reminder.notify("090-1234-5678");
```

---

## 7. Flyweight パターン

### 7.1 概要と目的

```
目的: 大量のオブジェクトを効率的にメモリ共有する

いつ使うか:
  → 同じ状態のオブジェクトが大量に必要
  → オブジェクトの内部状態（不変）と外部状態（可変）を分離できる
  → メモリ使用量を削減したい
```

### 7.2 テキストエディタの文字オブジェクト

```typescript
// Flyweight: 文字のフォント情報を共有
interface CharacterStyle {
  font: string;
  size: number;
  color: string;
  bold: boolean;
  italic: boolean;
}

class StyleFlyweight {
  // 不変の内部状態（共有される）
  constructor(
    public readonly font: string,
    public readonly size: number,
    public readonly color: string,
    public readonly bold: boolean,
    public readonly italic: boolean,
  ) {}

  render(char: string, x: number, y: number): string {
    const weight = this.bold ? "bold" : "normal";
    const style = this.italic ? "italic" : "normal";
    return `<span style="font-family:${this.font};font-size:${this.size}px;color:${this.color};font-weight:${weight};font-style:${style}" data-pos="${x},${y}">${char}</span>`;
  }
}

// Flyweight Factory: スタイルオブジェクトの管理
class StyleFactory {
  private styles = new Map<string, StyleFlyweight>();

  getStyle(config: CharacterStyle): StyleFlyweight {
    const key = `${config.font}-${config.size}-${config.color}-${config.bold}-${config.italic}`;

    if (!this.styles.has(key)) {
      this.styles.set(key, new StyleFlyweight(
        config.font, config.size, config.color, config.bold, config.italic,
      ));
    }

    return this.styles.get(key)!;
  }

  get poolSize(): number {
    return this.styles.size;
  }
}

// テキストエディタでの使用
class TextEditor {
  private characters: Array<{ char: string; style: StyleFlyweight; x: number; y: number }> = [];
  private styleFactory = new StyleFactory();

  insert(char: string, style: CharacterStyle, x: number, y: number): void {
    // スタイルは共有される Flyweight を使う
    const flyweight = this.styleFactory.getStyle(style);
    this.characters.push({ char, style: flyweight, x, y });
  }

  render(): string {
    return this.characters.map(c => c.style.render(c.char, c.x, c.y)).join("");
  }

  getMemoryStats(): { characters: number; uniqueStyles: number } {
    return {
      characters: this.characters.length,
      uniqueStyles: this.styleFactory.poolSize,
    };
  }
}

// 使用例: 10万文字のドキュメント
const editor = new TextEditor();
const normalStyle = { font: "Noto Sans JP", size: 14, color: "#333", bold: false, italic: false };
const boldStyle = { font: "Noto Sans JP", size: 14, color: "#333", bold: true, italic: false };
const headingStyle = { font: "Noto Sans JP", size: 24, color: "#000", bold: true, italic: false };

// 10万文字を挿入しても、スタイルオブジェクトは3つだけ共有される
for (let i = 0; i < 100000; i++) {
  const style = i < 50 ? headingStyle : i % 10 === 0 ? boldStyle : normalStyle;
  editor.insert("あ", style, i % 80 * 14, Math.floor(i / 80) * 20);
}

console.log(editor.getMemoryStats());
// { characters: 100000, uniqueStyles: 3 }
// スタイルなしなら100000個のオブジェクトが必要だが、共有で3個に
```

---

## 8. パターンの組み合わせ

### 8.1 Adapter + Facade

```typescript
// 複数の外部サービスを Adapter で統一し、Facade でまとめる
class UnifiedNotificationFacade {
  private adapters: Map<string, NotificationAdapter> = new Map();

  registerAdapter(channel: string, adapter: NotificationAdapter): void {
    this.adapters.set(channel, adapter);
  }

  async sendAll(message: string, recipients: { channel: string; target: string }[]): Promise<{
    sent: number;
    failed: number;
    errors: string[];
  }> {
    let sent = 0;
    let failed = 0;
    const errors: string[] = [];

    for (const recipient of recipients) {
      const adapter = this.adapters.get(recipient.channel);
      if (!adapter) {
        failed++;
        errors.push(`Unknown channel: ${recipient.channel}`);
        continue;
      }
      try {
        await adapter.send(message, recipient.target);
        sent++;
      } catch (error) {
        failed++;
        errors.push(`${recipient.channel}:${recipient.target}: ${error}`);
      }
    }

    return { sent, failed, errors };
  }
}
```

### 8.2 Decorator + Proxy

```typescript
// ログ付きキャッシュプロキシ = Decorator + Proxy の組み合わせ
function createMonitoredService<T extends object>(
  service: T,
  serviceName: string,
): T {
  return new Proxy(service, {
    get(target: T, prop: string) {
      const original = (target as any)[prop];
      if (typeof original !== "function") return original;

      return async function (...args: any[]) {
        const start = Date.now();
        console.log(`[${serviceName}] ${prop} called with`, args);
        try {
          const result = await original.apply(target, args);
          console.log(`[${serviceName}] ${prop} completed in ${Date.now() - start}ms`);
          return result;
        } catch (error) {
          console.error(`[${serviceName}] ${prop} failed in ${Date.now() - start}ms`, error);
          throw error;
        }
      };
    },
  });
}

// 使用例
const monitoredUserService = createMonitoredService(
  new UserService(),
  "UserService"
);
await monitoredUserService.findById("123");
// [UserService] findById called with ["123"]
// [UserService] findById completed in 15ms
```

---

## まとめ

| パターン | 目的 | キーワード |
|---------|------|-----------|
| Adapter | インターフェース変換 | 既存コードの統合、ラッパー |
| Decorator | 動的な機能追加 | 重ね掛け、ミドルウェア |
| Facade | 複雑さの隠蔽 | シンプルなAPI、窓口 |
| Proxy | アクセス制御 | キャッシュ、遅延、権限 |
| Composite | 個と集合の同一視 | ツリー構造、再帰 |
| Bridge | 抽象と実装の分離 | クロスプラットフォーム |
| Flyweight | メモリ共有 | 大量オブジェクトの最適化 |

### パターン選択の指針

```
構造上の問題を解決したい
├── 互換性のないインターフェース → Adapter
├── 既存オブジェクトに機能を追加 → Decorator
├── 複雑なサブシステムを簡略化 → Facade
├── オブジェクトへのアクセス制御 → Proxy
├── ツリー構造の表現 → Composite
├── 抽象化と実装を独立に拡張 → Bridge
└── 大量オブジェクトのメモリ最適化 → Flyweight
```

---

## 次に読むべきガイド
→ [[02-behavioral-patterns.md]] — 振る舞いパターン

---

## 参考文献
1. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
2. Freeman, E. et al. "Head First Design Patterns." O'Reilly, 2020.
3. Martin, R.C. "Agile Software Development, Principles, Patterns, and Practices." Prentice Hall, 2002.
4. Fowler, M. "Patterns of Enterprise Application Architecture." Addison-Wesley, 2002.
