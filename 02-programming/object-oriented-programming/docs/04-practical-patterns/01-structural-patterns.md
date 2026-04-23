# Structural Patterns

> Patterns concerning how classes and objects are composed. A practical walkthrough of the seven structural patterns: Adapter, Decorator, Facade, Proxy, Composite, Bridge, and Flyweight.

## What you will learn in this chapter

- [ ] Understand the purpose and use cases of each structural pattern
- [ ] Grasp the code implementation of each pattern across multiple languages
- [ ] Learn how to combine and choose between patterns
- [ ] Know how these patterns are applied in modern frameworks (React, Express, NestJS, etc.)
- [ ] Be able to design structures with testability and maintainability in mind


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the contents of [Creational Patterns](./00-creational-patterns.md)

---

## 1. The Adapter Pattern

### 1.1 Overview and Purpose

```
Purpose: Convert and connect incompatible interfaces

  ┌─────────┐     ┌─────────┐     ┌──────────┐
  │ Client  │────→│ Adapter │────→│ Adaptee  │
  │         │     │(convert)│     │(existing)│
  └─────────┘     └─────────┘     └──────────┘

When to use:
  → You want to adapt an existing library to your own interface
  → You want to integrate legacy code into a new system
  → You want to absorb differences between third-party APIs
  → You want to swap out external dependencies during testing
```

### 1.2 Class Adapter and Object Adapter

```typescript
// An existing library (cannot be modified)
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

// A new interface (our standard)
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

// Adapter: bridges old and new
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
    // Map the legacy status to the new type
    const statusMap: Record<string, TransactionStatus> = {
      "completed": "completed",
      "pending": "pending",
      "error": "failed",
      "refund": "refunded",
    };
    return statusMap[legacyStatus] ?? "pending";
  }
}

// Callers depend only on PaymentProcessor
const processor: PaymentProcessor = new LegacyPaymentAdapter(
  new LegacyPaymentGateway()
);
```

### 1.3 Abstracting Providers with Multiple Adapters

```typescript
// Handle multiple external services through a unified interface
interface StorageProvider {
  upload(key: string, data: Buffer): Promise<string>;
  download(key: string): Promise<Buffer>;
  delete(key: string): Promise<void>;
  exists(key: string): Promise<boolean>;
  list(prefix: string): Promise<string[]>;
  getSignedUrl(key: string, expiresIn: number): Promise<string>;
}

// AWS S3 adapter
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

// Local filesystem adapter (for testing and development)
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

// GCS adapter
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

// Choose the adapter based on the environment
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

### 1.4 The Adapter Pattern in Java

```java
// Java: the Adapter pattern
// Interface of an existing library
public class ExternalLogService {
    public void writeLog(int severity, String component, String msg, long timestamp) {
        // Write to the external log service
    }
}

// In-house standard logger interface
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

## 2. The Decorator Pattern

### 2.1 Overview and Purpose

```
Purpose: Dynamically add functionality to an existing object

  ┌────────┐     ┌───────────┐     ┌───────────┐
  │ Client │────→│ Decorator │────→│ Component │
  │        │     │   (adds)  │     │ (origin)  │
  └────────┘     └───────────┘     └───────────┘

  Multiple Decorators can be layered (nested)

When to use:
  → You want to add features without modifying the existing class
  → There are many combinations of features and inheritance explodes
  → You want to attach/detach features dynamically at runtime
  → Middleware and pipeline processing
```

### 2.2 An HTTP Client Decorator

```typescript
// The Component interface
interface HttpClient {
  request(url: string, options?: RequestInit): Promise<Response>;
}

// The basic implementation
class BasicHttpClient implements HttpClient {
  async request(url: string, options?: RequestInit): Promise<Response> {
    return fetch(url, options);
  }
}

// Decorator: add logging
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

// Decorator: add retries
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

// Decorator: add an authentication header
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

// Decorator: add caching
class CachingHttpClient implements HttpClient {
  private cache = new Map<string, { response: Response; expiry: number }>();

  constructor(
    private wrapped: HttpClient,
    private ttlMs: number = 60000,
  ) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    // Cache only GET requests
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

// Decorator: add rate limiting
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

// Decorator: add a timeout
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

// Stacking decorators
const client = new LoggingHttpClient(
  new RateLimitHttpClient(
    new RetryHttpClient(
      new TimeoutHttpClient(
        new CachingHttpClient(
          new AuthHttpClient(
            new BasicHttpClient(),
            () => "my-token"
          ),
          60000 // cache for 1 minute
        ),
        10000 // 10-second timeout
      ),
      3 // retry 3 times
    ),
    10, 1000 // 10 requests per second
  )
);

// Request → RateLimit → Retry → Timeout → Cache → Auth → Basic, in that order
```

### 2.3 A Stream Processing Decorator

```typescript
// A decorator for a data transformation pipeline
interface DataTransformer<T> {
  transform(data: T[]): T[];
}

class BaseTransformer<T> implements DataTransformer<T> {
  transform(data: T[]): T[] {
    return data;
  }
}

// Filtering decorator
class FilterTransformer<T> implements DataTransformer<T> {
  constructor(
    private wrapped: DataTransformer<T>,
    private predicate: (item: T) => boolean,
  ) {}

  transform(data: T[]): T[] {
    return this.wrapped.transform(data).filter(this.predicate);
  }
}

// Sorting decorator
class SortTransformer<T> implements DataTransformer<T> {
  constructor(
    private wrapped: DataTransformer<T>,
    private comparator: (a: T, b: T) => number,
  ) {}

  transform(data: T[]): T[] {
    return [...this.wrapped.transform(data)].sort(this.comparator);
  }
}

// Pagination decorator
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

// Mapping decorator
class MapTransformer<T> implements DataTransformer<T> {
  constructor(
    private wrapped: DataTransformer<T>,
    private mapper: (item: T) => T,
  ) {}

  transform(data: T[]): T[] {
    return this.wrapped.transform(data).map(this.mapper);
  }
}

// Example: a user list processing pipeline
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
      user => user.active  // active users only
    ),
    (a, b) => a.name.localeCompare(b.name)  // sort by name
  ),
  1, 20  // page 1, 20 items
);

const result = pipeline.transform(allUsers);
```

### 2.4 The Decorator Pattern in Python

```python
# Python: the decorator pattern (function and class decorators)
import functools
import time
import logging
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
R = TypeVar('R')

# Function decorator: logging
def log_calls(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logging.info(f"Calling {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} returned {result}")
        return result
    return wrapper

# Function decorator: measure execution time
def timed(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

# Function decorator: retry
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

# Function decorator: caching (memoization)
def memoize(func: Callable[P, R]) -> Callable[P, R]:
    cache: dict = {}

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

# Stacking decorators
@log_calls
@timed
@retry(max_retries=3, delay=0.5)
def fetch_data(url: str) -> dict:
    """Fetch data from an external API"""
    import requests
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# On call: processed in the order retry → timed → log_calls
result = fetch_data("https://api.example.com/data")
```

---

## 3. The Facade Pattern

### 3.1 Overview and Purpose

```
Purpose: Provide a simple interface to a complex subsystem

  ┌────────┐     ┌──────────┐     ┌─────┐ ┌─────┐ ┌─────┐
  │ Client │────→│  Facade  │────→│ SubA│ │ SubB│ │ SubC│
  │        │     │ (front)  │     └─────┘ └─────┘ └─────┘
  └────────┘     └──────────┘

When to use:
  → You want to simplify operations on a complex subsystem
  → You want to reduce dependencies between layers
  → You want to unify how libraries are used
  → An API gateway for microservices
```

### 3.2 E-Commerce Order Processing

```typescript
// A group of complex subsystems
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

// Facade: wrap the entire order flow in a single simple API
class OrderFacade {
  constructor(
    private inventory: InventoryService,
    private pricing: PricingService,
    private payment: PaymentService,
    private shipping: ShippingService,
    private notification: NotificationService,
  ) {}

  async placeOrder(order: OrderRequest): Promise<OrderResult> {
    // 1. Inventory check
    for (const item of order.items) {
      if (!this.inventory.checkAvailability(item.productId, item.quantity)) {
        return { success: false, error: `${item.productId} is out of stock` };
      }
    }

    // 2. Reserve stock
    const reservations: string[] = [];
    for (const item of order.items) {
      reservations.push(
        this.inventory.reserveStock(item.productId, item.quantity)
      );
    }

    try {
      // 3. Price calculation
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

      // 4. Payment
      const authId = await this.payment.authorize(grandTotal, order.paymentMethod);
      const paymentId = await this.payment.capture(authId);

      // 5. Arrange shipping
      const shipmentId = await this.shipping.createShipment(
        order.shippingAddress,
        order.items,
      );
      const trackingUrl = await this.shipping.getTrackingUrl(shipmentId);

      // 6. Notifications
      await this.notification.sendOrderConfirmation(order.customerEmail, paymentId);
      await this.notification.sendShippingNotification(order.customerEmail, trackingUrl);

      return {
        success: true,
        orderId: paymentId,
        total: grandTotal,
        trackingUrl,
      };
    } catch (error) {
      // On error, release the reserved stock
      for (const reservationId of reservations) {
        this.inventory.releaseStock(reservationId);
      }
      return { success: false, error: (error as Error).message };
    }
  }
}

// Calling side stays simple
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

### 3.3 A Media Player Facade

```typescript
// Complex subsystems
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

// Facade: a simple API
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

// The calling side stays simple
const player = new MediaPlayer();
player.setResolution(1920, 1080);
player.setVolume(75);
player.play("movie.mp4", "movie.srt");
```

---

## 4. The Proxy Pattern

### 4.1 Overview and Purpose

```
Purpose: Provide a surrogate that controls access to an object

Variants:
  → Virtual Proxy: lazy initialization (create heavy objects on demand)
  → Protection Proxy: access control (permission checks)
  → Cache Proxy: cache the results
  → Remote Proxy: access to objects across the network
  → Log Proxy: record the operations
```

### 4.2 Cache Proxy

```typescript
// Cache Proxy
interface DataService {
  fetchUser(id: string): Promise<User>;
  fetchUsers(ids: string[]): Promise<User[]>;
  searchUsers(query: string): Promise<User[]>;
}

class RealDataService implements DataService {
  async fetchUser(id: string): Promise<User> {
    console.log(`Fetching user ${id} from database...`);
    // Simulate heavy DB/API access
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

    // Cache hit
    const cached = this.cache.get(cacheKey);
    if (cached && cached.expiry > Date.now()) {
      console.log(`Cache hit for user ${id}`);
      return cached.data;
    }

    // Deduplicate requests (collapse simultaneous requests for the same ID into one)
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
await service.fetchUser("123"); // DB access
await service.fetchUser("123"); // from cache
```

### 4.3 Protection Proxy (Access Control)

```typescript
// Protection Proxy: permission checks
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

### 4.4 Virtual Proxy (Lazy Initialization)

```typescript
// Virtual Proxy: lazy initialization of heavy objects
interface ImageRenderer {
  render(x: number, y: number): void;
  getWidth(): number;
  getHeight(): number;
}

class HighResImage implements ImageRenderer {
  private pixels: Buffer;

  constructor(private path: string) {
    // Heavy operation: load a high-resolution image
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
    // Load only when actually rendering
    this.loadImage().render(x, y);
  }

  getWidth(): number {
    // Metadata held by the proxy itself needs no load
    return 4096;
  }

  getHeight(): number {
    return 2160;
  }
}

// A gallery holding 1,000 images
// Loading them all would take 50GB; with the Proxy, only what is needed is loaded
const gallery: ImageRenderer[] = [];
for (let i = 0; i < 1000; i++) {
  gallery.push(new LazyImageProxy(`/images/photo_${i}.raw`));
}
// At this point memory usage is minimal
// Only the images that are displayed are actually loaded
gallery[42].render(0, 0); // only now is photo_42.raw loaded
```

### 4.5 An Implementation Using the JavaScript Proxy

```typescript
// An implementation using the native JavaScript Proxy
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
      // For methods, add a call log
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

// Example
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

## 5. The Composite Pattern

### 5.1 Overview and Purpose

```
Purpose: Treat individual objects and compositions of objects uniformly

  Component (the common interface)
  ├── Leaf (individual element)
  └── Composite (branch: contains child elements)

When to use:
  → You want to represent a tree structure
  → You want to handle individual items and collections uniformly
  → Recursive structures (file systems, UI components, org charts)
```

### 5.2 A File System

```typescript
// A file system example
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

// File and Directory can be handled uniformly
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

// Search: find files whose names contain ".test."
const testFiles = root.find(entry => entry.name.includes(".test."));
console.log("Test files:", testFiles.map(f => f.name));
```

### 5.3 A UI Component Tree

```typescript
// A Composite of UI components
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

// Building a login form
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

## 6. The Bridge Pattern

### 6.1 Overview and Purpose

```
Purpose: Separate an abstraction from its implementation so each can vary independently

  ┌───────────────┐          ┌─────────────────┐
  │  Abstraction  │ ────────→│ Implementation  │
  │ (abstraction) │          │ (implementation)│
  └───────┬───────┘          └────────┬────────┘
          │                           │
  ┌───────┴───────┐          ┌────────┴────────┐
  │  Refined      │          │  Concrete       │
  │  Abstraction  │          │  Implementation │
  └───────────────┘          └─────────────────┘

When to use:
  → Both the abstraction and the implementation may evolve
  → Cross-platform support
  → You want to avoid the combinatorial explosion of inheritance
```

### 6.2 A Bridge for a Notification System

```typescript
// Implementation: the concrete way to send messages
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

// Abstraction: types of notifications
abstract class Notification {
  constructor(protected sender: MessageSender) {}

  abstract notify(to: string): Promise<boolean>;

  // The sending method can be switched dynamically
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

// Example: freely combine notification type x sending method
const emailSender = new EmailSender();
const smsSender = new SmsSender();
const slackSender = new SlackSender();

// Critical alert to Slack
const criticalAlert = new AlertNotification(slackSender, "critical", "サーバーダウン");
await criticalAlert.notify("#ops-alerts");

// Send the same alert via email as well
criticalAlert.setSender(emailSender);
await criticalAlert.notify("ops-team@example.com");

// Reminder via SMS
const reminder = new ReminderNotification(smsSender, "レポート提出", new Date("2025-03-01"));
await reminder.notify("090-1234-5678");
```

---

## 7. The Flyweight Pattern

### 7.1 Overview and Purpose

```
Purpose: Efficiently share memory among a large number of objects

When to use:
  → You need many objects with the same state
  → You can separate an object's intrinsic (immutable) state from its extrinsic (mutable) state
  → You want to reduce memory usage
```

### 7.2 Character Objects in a Text Editor

```typescript
// Flyweight: share font information for characters
interface CharacterStyle {
  font: string;
  size: number;
  color: string;
  bold: boolean;
  italic: boolean;
}

class StyleFlyweight {
  // Immutable intrinsic state (shared)
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

// Flyweight Factory: manages the style objects
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

// Usage inside a text editor
class TextEditor {
  private characters: Array<{ char: string; style: StyleFlyweight; x: number; y: number }> = [];
  private styleFactory = new StyleFactory();

  insert(char: string, style: CharacterStyle, x: number, y: number): void {
    // Use a shared Flyweight for the style
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

// Example: a 100,000-character document
const editor = new TextEditor();
const normalStyle = { font: "Noto Sans JP", size: 14, color: "#333", bold: false, italic: false };
const boldStyle = { font: "Noto Sans JP", size: 14, color: "#333", bold: true, italic: false };
const headingStyle = { font: "Noto Sans JP", size: 24, color: "#000", bold: true, italic: false };

// Even after inserting 100,000 characters, only 3 style objects are shared
for (let i = 0; i < 100000; i++) {
  const style = i < 50 ? headingStyle : i % 10 === 0 ? boldStyle : normalStyle;
  editor.insert("あ", style, i % 80 * 14, Math.floor(i / 80) * 20);
}

console.log(editor.getMemoryStats());
// { characters: 100000, uniqueStyles: 3 }
// Without sharing, 100,000 objects would be needed; sharing reduces it to 3
```

---

## 8. Combining Patterns

### 8.1 Adapter + Facade

```typescript
// Unify multiple external services with Adapters, then wrap them with a Facade
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
// A logging cache proxy = a combination of Decorator and Proxy
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

// Example
const monitoredUserService = createMonitoredService(
  new UserService(),
  "UserService"
);
await monitoredUserService.findById("123");
// [UserService] findById called with ["123"]
// [UserService] findById completed in 15ms
```

---


## FAQ

### Q1: What is the single most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not with theory alone but by actually writing code and verifying how it behaves.

### Q2: What mistakes do beginners commonly make?

Skipping the basics and jumping straight into advanced topics. We recommend firmly understanding the foundational concepts explained in this guide before moving on to the next step.

### Q3: How is this used in real-world work?

The knowledge from this topic is used frequently in everyday development. It becomes especially important during code review and architectural design.

---

## Summary

| Pattern | Purpose | Keywords |
|---------|------|-----------|
| Adapter | Interface conversion | Integrate existing code, wrapper |
| Decorator | Dynamic feature addition | Stacking, middleware |
| Facade | Hide complexity | Simple API, front door |
| Proxy | Control access | Caching, laziness, permissions |
| Composite | Treat individuals and collections alike | Tree structure, recursion |
| Bridge | Decouple abstraction from implementation | Cross-platform |
| Flyweight | Share memory | Optimization for large numbers of objects |

### Guidelines for Choosing a Pattern

```
You want to solve a structural problem
├── Incompatible interfaces → Adapter
├── Add features to an existing object → Decorator
├── Simplify a complex subsystem → Facade
├── Control access to an object → Proxy
├── Represent a tree structure → Composite
├── Extend abstraction and implementation independently → Bridge
└── Optimize memory for a huge number of objects → Flyweight
```

---

## Recommended Next Reads

---

## References
1. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
2. Freeman, E. et al. "Head First Design Patterns." O'Reilly, 2020.
3. Martin, R.C. "Agile Software Development, Principles, Patterns, and Practices." Prentice Hall, 2002.
4. Fowler, M. "Patterns of Enterprise Application Architecture." Addison-Wesley, 2002.
