# Strategy パターン

> アルゴリズムのファミリーを定義し、それぞれを **カプセル化** して交換可能にする振る舞いパターン。実行時にアルゴリズムを切り替えられ、条件分岐の爆発を防止してOpen/Closed Principle を遵守する。

---

## この章で学ぶこと

1. Strategy パターンの構造と、条件分岐の排除によるOCP準拠の設計手法を理解する
2. DI（依存性注入）との関係、関数型アプローチでの実現方法、Registry パターンとの組み合わせを習得する
3. Strategy の過剰適用を避ける判断基準と、Template Method/State パターンとの使い分けを身につける

---

## 前提知識

このガイドを読む前に、以下の概念を理解しておくことを推奨します。

| 前提知識 | 説明 | 参照リンク |
|---------|------|-----------|
| SOLID 原則（特にOCP） | 拡張に開き、修正に閉じる原則 | [SOLID 原則](../../../clean-code-principles/docs/00-principles/01-solid.md) |
| インタフェースとポリモーフィズム | 異なる実装を統一的に扱う概念 | [クリーンコード](../../../clean-code-principles/docs/00-principles/) |
| 依存性注入（DI） | 外部から依存オブジェクトを注入する手法 | [DI/IoC](../../../clean-code-principles/docs/01-practices/) |
| 関数（第一級オブジェクト） | 関数を値として扱い、引数や返り値にする概念 | [関数型パターン](../03-functional/02-fp-patterns.md) |

---

## 1. Strategy パターンとは何か

### 1.1 解決する問題

ソフトウェアでは、同じ種類の処理に複数のアルゴリズムが存在する場面が多い。

- 料金計算: 通常/プレミアム/学生/シニア
- ソート: 名前順/日付順/価格順/関連度順
- 認証: パスワード/OAuth/SAML/APIキー
- 圧縮: gzip/zstd/lz4/brotli

これらを `if/else` や `switch` で分岐すると、アルゴリズムが増えるたびに条件分岐が肥大化し、既存コードの変更が必要になる（OCP 違反）。

```
BEFORE（条件分岐の肥大化）:
function calculate(type, price) {
  if (type === "regular") return price;
  else if (type === "premium") return price * 0.9;
  else if (type === "student") return price * 0.7;
  else if (type === "senior") return price * 0.8;
  else if (type === "vip") return price * 0.6;    // 追加1
  else if (type === "family") return price * 0.75; // 追加2
  // ... 新しい種類のたびにこの関数を修正 -> OCP 違反
}

AFTER（Strategy パターン）:
strategies.get(type).calculate(price);
// 新しい型は register するだけ -> OCP 準拠
```

### 1.2 パターンの意図

GoF の定義:

> Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

日本語訳:

> アルゴリズムのファミリーを定義し、それぞれをカプセル化して交換可能にする。Strategy パターンにより、アルゴリズムをクライアントから独立して変更できる。

### 1.3 WHY: なぜ Strategy パターンが必要なのか

根本的な理由は **アルゴリズムの選択と実行を分離する** ことにある。

1. **Open/Closed Principle**: 新しいアルゴリズムの追加が既存コードの変更なしに行える
2. **Single Responsibility Principle**: 各アルゴリズムが独立したクラス/関数として存在し、個別にテスト可能
3. **実行時の切り替え**: 同じ Context で異なるアルゴリズムを動的に切り替えられる
4. **テスタビリティ**: Strategy をモック/スタブに差し替えることでテストが容易

---

## 2. Strategy の構造

### 2.1 クラス図

```
+-------------+       +-------------------+
|   Context   |------>|   Strategy        |
+-------------+       |   (interface)     |
| - strategy  |       +-------------------+
| + execute() |       | + execute(data)   |
+-------------+       +-------------------+
                              ^
                       _______|_______
                      |               |
               +------------+  +------------+
               | StrategyA  |  | StrategyB  |
               +------------+  +------------+
               | + execute() |  | + execute() |
               +------------+  +------------+
```

### 2.2 構成要素の役割

| 構成要素 | 役割 | 責務 |
|---------|------|------|
| Strategy (Interface) | アルゴリズムの共通契約 | メソッドシグネチャを定義 |
| ConcreteStrategy | 具体的なアルゴリズム実装 | Strategy インタフェースに従って実装 |
| Context | Strategy を使用する側 | Strategy の参照を保持し、委譲する |
| Client | Context と Strategy を組み立て | 具体的な Strategy を Context に注入 |

### 2.3 処理シーケンス

```
Client            Context              Strategy
  |                  |                     |
  |-- new Context(strategyA) -->|          |
  |                  |-- setStrategy(A) -->|
  |                  |                     |
  |-- execute() ---->|                     |
  |                  |-- execute(data) --->|  StrategyA
  |                  |<--- result ---------|
  |<-- result -------|                     |
  |                  |                     |
  |-- setStrategy(B)->|                    |
  |                  |                     |
  |-- execute() ---->|                     |
  |                  |-- execute(data) --->|  StrategyB
  |                  |<--- result ---------|
  |<-- result -------|                     |
```

---

## 3. コード例

### コード例 1: 料金計算 Strategy（TypeScript）

```typescript
// pricing-strategy.ts — Strategy パターンの基本形
interface PricingStrategy {
  readonly name: string;
  calculate(basePrice: number): number;
  getDescription(): string;
}

class RegularPricing implements PricingStrategy {
  readonly name = "regular";
  calculate(basePrice: number): number {
    return basePrice;
  }
  getDescription(): string {
    return "通常価格（割引なし）";
  }
}

class PremiumPricing implements PricingStrategy {
  readonly name = "premium";
  calculate(basePrice: number): number {
    return Math.round(basePrice * 0.9); // 10%割引
  }
  getDescription(): string {
    return "プレミアム会員価格（10%OFF）";
  }
}

class StudentPricing implements PricingStrategy {
  readonly name = "student";
  calculate(basePrice: number): number {
    return Math.round(basePrice * 0.7); // 30%割引
  }
  getDescription(): string {
    return "学生価格（30%OFF）";
  }
}

class SeniorPricing implements PricingStrategy {
  readonly name = "senior";
  calculate(basePrice: number): number {
    return Math.round(basePrice * 0.8); // 20%割引
  }
  getDescription(): string {
    return "シニア価格（20%OFF）";
  }
}

// Context: ショッピングカート
class ShoppingCart {
  private items: { name: string; price: number; quantity: number }[] = [];
  private pricingStrategy: PricingStrategy;

  constructor(pricingStrategy: PricingStrategy = new RegularPricing()) {
    this.pricingStrategy = pricingStrategy;
  }

  setPricingStrategy(strategy: PricingStrategy): void {
    this.pricingStrategy = strategy;
    console.log(`Pricing changed to: ${strategy.getDescription()}`);
  }

  addItem(name: string, price: number, quantity: number = 1): void {
    this.items.push({ name, price, quantity });
  }

  checkout(): { subtotal: number; discount: number; total: number; strategy: string } {
    const subtotal = this.items.reduce((sum, i) => sum + i.price * i.quantity, 0);
    const total = this.pricingStrategy.calculate(subtotal);
    return {
      subtotal,
      discount: subtotal - total,
      total,
      strategy: this.pricingStrategy.name,
    };
  }
}

// --- 使用例: 実行時に戦略を切り替え ---
const cart = new ShoppingCart();
cart.addItem("TypeScript Book", 3000);
cart.addItem("Design Patterns Book", 4000);

console.log(cart.checkout());
// { subtotal: 7000, discount: 0, total: 7000, strategy: "regular" }

cart.setPricingStrategy(new StudentPricing());
// "Pricing changed to: 学生価格（30%OFF）"

console.log(cart.checkout());
// { subtotal: 7000, discount: 2100, total: 4900, strategy: "student" }

cart.setPricingStrategy(new PremiumPricing());
console.log(cart.checkout());
// { subtotal: 7000, discount: 700, total: 6300, strategy: "premium" }
```

### コード例 2: 関数型 Strategy（TypeScript）

```typescript
// functional-strategy.ts — クラスを使わず関数で Strategy を実現
interface User {
  name: string;
  age: number;
  email: string;
  createdAt: Date;
  score: number;
}

// Strategy を関数型で定義
type SortStrategy<T> = (a: T, b: T) => number;

const byName: SortStrategy<User> = (a, b) =>
  a.name.localeCompare(b.name);

const byAge: SortStrategy<User> = (a, b) =>
  a.age - b.age;

const byCreatedDesc: SortStrategy<User> = (a, b) =>
  b.createdAt.getTime() - a.createdAt.getTime();

const byScore: SortStrategy<User> = (a, b) =>
  b.score - a.score;

// 合成: 複数のソート条件を組み合わせる
function composeStrategies<T>(...strategies: SortStrategy<T>[]): SortStrategy<T> {
  return (a, b) => {
    for (const strategy of strategies) {
      const result = strategy(a, b);
      if (result !== 0) return result;
    }
    return 0;
  };
}

// 反転: 降順にする
function reverse<T>(strategy: SortStrategy<T>): SortStrategy<T> {
  return (a, b) => -strategy(a, b);
}

// Context 関数
function sortUsers(users: User[], strategy: SortStrategy<User>): User[] {
  return [...users].sort(strategy);
}

// --- 使用例 ---
const users: User[] = [
  { name: "Charlie", age: 30, email: "c@test.com", createdAt: new Date("2024-01"), score: 85 },
  { name: "Alice", age: 25, email: "a@test.com", createdAt: new Date("2024-03"), score: 92 },
  { name: "Bob", age: 30, email: "b@test.com", createdAt: new Date("2024-02"), score: 88 },
];

// 単一ソート
console.log(sortUsers(users, byName).map(u => u.name));
// ["Alice", "Bob", "Charlie"]

console.log(sortUsers(users, byScore).map(u => u.name));
// ["Alice", "Bob", "Charlie"]

// 合成ソート: 年齢順 -> 名前順（同じ年齢の場合）
const byAgeThenName = composeStrategies(byAge, byName);
console.log(sortUsers(users, byAgeThenName).map(u => u.name));
// ["Alice", "Bob", "Charlie"]

// 反転: 名前の逆順
console.log(sortUsers(users, reverse(byName)).map(u => u.name));
// ["Charlie", "Bob", "Alice"]
```

### コード例 3: バリデーション Strategy（TypeScript）

```typescript
// validation-strategy.ts — フォームバリデーション
interface ValidationResult {
  valid: boolean;
  errors: string[];
}

interface ValidationStrategy {
  validate(value: string): ValidationResult;
  readonly fieldName: string;
}

class EmailValidation implements ValidationStrategy {
  readonly fieldName = "email";

  validate(value: string): ValidationResult {
    const errors: string[] = [];
    if (!value) errors.push("Email is required");
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
      errors.push("Invalid email format");
    }
    return { valid: errors.length === 0, errors };
  }
}

class PasswordValidation implements ValidationStrategy {
  readonly fieldName = "password";

  constructor(
    private options: {
      minLength?: number;
      requireUppercase?: boolean;
      requireDigit?: boolean;
      requireSpecial?: boolean;
    } = {}
  ) {}

  validate(value: string): ValidationResult {
    const errors: string[] = [];
    const { minLength = 8, requireUppercase = true, requireDigit = true, requireSpecial = false } = this.options;

    if (value.length < minLength) errors.push(`Minimum ${minLength} characters`);
    if (requireUppercase && !/[A-Z]/.test(value)) errors.push("Need uppercase letter");
    if (requireDigit && !/[0-9]/.test(value)) errors.push("Need digit");
    if (requireSpecial && !/[!@#$%^&*]/.test(value)) errors.push("Need special character");

    return { valid: errors.length === 0, errors };
  }
}

class PhoneValidation implements ValidationStrategy {
  readonly fieldName = "phone";

  constructor(private country: 'JP' | 'US' = 'JP') {}

  validate(value: string): ValidationResult {
    const errors: string[] = [];
    const patterns = {
      JP: /^0\d{1,4}-?\d{1,4}-?\d{4}$/,
      US: /^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$/,
    };

    if (!value) errors.push("Phone number is required");
    else if (!patterns[this.country].test(value)) {
      errors.push(`Invalid ${this.country} phone number format`);
    }

    return { valid: errors.length === 0, errors };
  }
}

// Context: フォームフィールド
class FormField {
  private strategies: ValidationStrategy[] = [];

  constructor(
    private name: string,
    ...strategies: ValidationStrategy[]
  ) {
    this.strategies = strategies;
  }

  validate(value: string): ValidationResult {
    const allErrors: string[] = [];
    for (const strategy of this.strategies) {
      const result = strategy.validate(value);
      allErrors.push(...result.errors);
    }
    return { valid: allErrors.length === 0, errors: allErrors };
  }
}

// --- 使用例 ---
const emailField = new FormField("email", new EmailValidation());
console.log(emailField.validate("test@example.com"));
// { valid: true, errors: [] }

console.log(emailField.validate("invalid-email"));
// { valid: false, errors: ["Invalid email format"] }

const passwordField = new FormField(
  "password",
  new PasswordValidation({ minLength: 10, requireSpecial: true })
);
console.log(passwordField.validate("short"));
// { valid: false, errors: ["Minimum 10 characters", "Need uppercase letter", "Need digit", "Need special character"] }

console.log(passwordField.validate("MyP@ssw0rd!!"));
// { valid: true, errors: [] }
```

### コード例 4: Python ── Protocol ベースの Strategy

```python
# compression_strategy.py — Python Protocol による Strategy
from typing import Protocol
import gzip
import time


class CompressionStrategy(Protocol):
    """圧縮戦略のプロトコル（インタフェース）"""
    @property
    def name(self) -> str: ...
    def compress(self, data: bytes) -> bytes: ...
    def decompress(self, data: bytes) -> bytes: ...


class GzipCompression:
    name = "gzip"

    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


class NoCompression:
    name = "none"

    def compress(self, data: bytes) -> bytes:
        return data

    def decompress(self, data: bytes) -> bytes:
        return data


class FileProcessor:
    """Context: ファイル処理器"""
    def __init__(self, compression: CompressionStrategy):
        self._compression = compression

    def set_compression(self, compression: CompressionStrategy) -> None:
        self._compression = compression
        print(f"Compression changed to: {compression.name}")

    def save(self, data: bytes, path: str) -> dict:
        start = time.time()
        compressed = self._compression.compress(data)
        elapsed = time.time() - start

        with open(path, "wb") as f:
            f.write(compressed)

        ratio = len(compressed) / len(data) * 100
        return {
            "original_size": len(data),
            "compressed_size": len(compressed),
            "ratio": f"{ratio:.1f}%",
            "time_ms": f"{elapsed * 1000:.2f}",
            "algorithm": self._compression.name,
        }

    def load(self, path: str) -> bytes:
        with open(path, "rb") as f:
            compressed = f.read()
        return self._compression.decompress(compressed)


# --- 使用例 ---
data = b"Hello " * 1000  # 6000 bytes の繰り返しデータ

processor = FileProcessor(GzipCompression())
result = processor.save(data, "/tmp/data.gz")
print(result)
# {"original_size": 6000, "compressed_size": ~40, "ratio": "0.7%", ...}

processor.set_compression(NoCompression())
result = processor.save(data, "/tmp/data.raw")
print(result)
# {"original_size": 6000, "compressed_size": 6000, "ratio": "100.0%", ...}
```

### コード例 5: Strategy の動的選択（Registry パターン）

```typescript
// strategy-registry.ts — Registry + Strategy
class StrategyRegistry<T> {
  private strategies = new Map<string, T>();
  private defaultKey: string | null = null;

  register(name: string, strategy: T, isDefault: boolean = false): this {
    this.strategies.set(name, strategy);
    if (isDefault) this.defaultKey = name;
    return this;
  }

  get(name: string): T {
    const strategy = this.strategies.get(name);
    if (strategy) return strategy;

    // デフォルト戦略があれば返す
    if (this.defaultKey) {
      return this.strategies.get(this.defaultKey)!;
    }

    throw new Error(`Strategy "${name}" not found. Available: ${this.getAvailableNames().join(', ')}`);
  }

  has(name: string): boolean {
    return this.strategies.has(name);
  }

  getAvailableNames(): string[] {
    return [...this.strategies.keys()];
  }
}

// --- 料金計算の Registry ---
const pricingRegistry = new StrategyRegistry<PricingStrategy>();
pricingRegistry
  .register("regular", new RegularPricing(), true) // デフォルト
  .register("premium", new PremiumPricing())
  .register("student", new StudentPricing())
  .register("senior", new SeniorPricing());

// APIリクエストから動的に選択
function handleCheckout(req: { membershipType: string; items: any[] }) {
  const strategy = pricingRegistry.get(req.membershipType);
  const cart = new ShoppingCart(strategy);
  // ...
}

// 利用可能な戦略の一覧
console.log(pricingRegistry.getAvailableNames());
// ["regular", "premium", "student", "senior"]
```

### コード例 6: HTTP リトライ Strategy

```typescript
// retry-strategy.ts — リトライアルゴリズムの Strategy
interface RetryStrategy {
  readonly name: string;
  getDelay(attempt: number, baseDelay: number): number;
  shouldRetry(attempt: number, maxAttempts: number, error: Error): boolean;
}

class LinearRetry implements RetryStrategy {
  readonly name = "linear";
  getDelay(attempt: number, baseDelay: number): number {
    return baseDelay * attempt;
  }
  shouldRetry(attempt: number, maxAttempts: number): boolean {
    return attempt < maxAttempts;
  }
}

class ExponentialBackoff implements RetryStrategy {
  readonly name = "exponential";
  getDelay(attempt: number, baseDelay: number): number {
    return baseDelay * Math.pow(2, attempt - 1);
  }
  shouldRetry(attempt: number, maxAttempts: number): boolean {
    return attempt < maxAttempts;
  }
}

class ExponentialWithJitter implements RetryStrategy {
  readonly name = "exponential-jitter";
  getDelay(attempt: number, baseDelay: number): number {
    const exponentialDelay = baseDelay * Math.pow(2, attempt - 1);
    const jitter = Math.random() * exponentialDelay;
    return Math.floor(jitter);
  }
  shouldRetry(attempt: number, maxAttempts: number, error: Error): boolean {
    // 4xx エラーはリトライしない（クライアントエラー）
    if ('statusCode' in error && (error as any).statusCode >= 400 && (error as any).statusCode < 500) {
      return false;
    }
    return attempt < maxAttempts;
  }
}

// Context: HTTP クライアント
class ResilientHttpClient {
  constructor(
    private retryStrategy: RetryStrategy,
    private maxAttempts: number = 3,
    private baseDelay: number = 1000
  ) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= this.maxAttempts; attempt++) {
      try {
        const response = await fetch(url, options);
        if (response.ok) return response;
        throw Object.assign(new Error(`HTTP ${response.status}`), { statusCode: response.status });
      } catch (error) {
        lastError = error as Error;
        console.log(`[${this.retryStrategy.name}] Attempt ${attempt} failed: ${lastError.message}`);

        if (!this.retryStrategy.shouldRetry(attempt, this.maxAttempts, lastError)) {
          break;
        }

        const delay = this.retryStrategy.getDelay(attempt, this.baseDelay);
        console.log(`Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw lastError;
  }
}

// --- 使用例 ---
// 開発環境: リニアリトライ（予測しやすい）
const devClient = new ResilientHttpClient(new LinearRetry(), 3, 500);

// 本番環境: 指数バックオフ + ジッター（サーバー負荷分散）
const prodClient = new ResilientHttpClient(new ExponentialWithJitter(), 5, 1000);
```

### コード例 7: ロギング Strategy

```typescript
// logging-strategy.ts — ログ出力先の Strategy
interface LogEntry {
  level: 'debug' | 'info' | 'warn' | 'error';
  message: string;
  timestamp: Date;
  context?: Record<string, unknown>;
}

interface LoggingStrategy {
  log(entry: LogEntry): void;
  flush?(): Promise<void>;
}

class ConsoleLogging implements LoggingStrategy {
  log(entry: LogEntry): void {
    const prefix = `[${entry.timestamp.toISOString()}] [${entry.level.toUpperCase()}]`;
    const ctx = entry.context ? ` ${JSON.stringify(entry.context)}` : '';
    console.log(`${prefix} ${entry.message}${ctx}`);
  }
}

class JsonFileLogging implements LoggingStrategy {
  private buffer: string[] = [];

  constructor(private filePath: string, private bufferSize: number = 10) {}

  log(entry: LogEntry): void {
    this.buffer.push(JSON.stringify(entry));
    if (this.buffer.length >= this.bufferSize) {
      this.flush();
    }
  }

  async flush(): Promise<void> {
    if (this.buffer.length === 0) return;
    const data = this.buffer.join('\n') + '\n';
    this.buffer = [];
    console.log(`[JsonFileLogging] Flushed ${data.split('\n').length - 1} entries to ${this.filePath}`);
  }
}

class MultiLogging implements LoggingStrategy {
  constructor(private strategies: LoggingStrategy[]) {}

  log(entry: LogEntry): void {
    this.strategies.forEach(s => s.log(entry));
  }

  async flush(): Promise<void> {
    await Promise.all(
      this.strategies
        .filter(s => s.flush)
        .map(s => s.flush!())
    );
  }
}

// Context: Logger
class Logger {
  constructor(private strategy: LoggingStrategy) {}

  setStrategy(strategy: LoggingStrategy): void {
    this.strategy = strategy;
  }

  private createEntry(level: LogEntry['level'], message: string, context?: Record<string, unknown>): LogEntry {
    return { level, message, timestamp: new Date(), context };
  }

  debug(message: string, context?: Record<string, unknown>): void {
    this.strategy.log(this.createEntry('debug', message, context));
  }

  info(message: string, context?: Record<string, unknown>): void {
    this.strategy.log(this.createEntry('info', message, context));
  }

  warn(message: string, context?: Record<string, unknown>): void {
    this.strategy.log(this.createEntry('warn', message, context));
  }

  error(message: string, context?: Record<string, unknown>): void {
    this.strategy.log(this.createEntry('error', message, context));
  }
}

// --- 使用例 ---
// 開発環境: コンソールのみ
const devLogger = new Logger(new ConsoleLogging());
devLogger.info("Server started", { port: 3000 });

// 本番環境: コンソール + JSON ファイル
const prodLogger = new Logger(new MultiLogging([
  new ConsoleLogging(),
  new JsonFileLogging("/var/log/app.jsonl"),
]));
prodLogger.error("Database connection failed", { host: "db.example.com" });
```

---

## 4. if/else の排除

Strategy パターンの最も実践的な価値は、条件分岐の排除である。

```
BEFORE (条件分岐の肥大化):

  function calculate(type: string, price: number): number {
    if (type === "regular") return price;
    else if (type === "premium") return price * 0.9;
    else if (type === "student") return price * 0.7;
    else if (type === "senior") return price * 0.8;
    // ... 追加のたびにこの関数を変更 -> OCP 違反
  }

  問題:
  1. 関数の肥大化
  2. テストの組み合わせ爆発
  3. 新しい型の追加で既存コードを変更

AFTER (Strategy パターン):

  // 各戦略は独立したクラス/関数
  strategies.get(type).calculate(price);

  利点:
  1. 各戦略が独立してテスト可能
  2. 新しい戦略は register するだけ
  3. 既存のコードは変更不要

判断フロー:
  条件分岐は3つ以上か？ ----No----> if/else で十分
    |
   Yes
    |
  将来の追加が見込まれるか？ --No----> switch + enum で可読性確保
    |
   Yes
    |
  Strategy パターンを導入
```

---

## 5. 比較表

### 比較表 1: Strategy vs State vs Command vs Template Method

| 観点 | Strategy | State | Command | Template Method |
|------|:---:|:---:|:---:|:---:|
| 目的 | アルゴリズム交換 | 状態依存の振る舞い | 操作のカプセル化 | アルゴリズムの骨格定義 |
| 交換タイミング | クライアントが決定 | 内部状態で自動遷移 | キュー/履歴に保存 | コンパイル時 |
| 関係 | has-a（委譲） | has-a（委譲） | has-a（委譲） | is-a（継承） |
| 柔軟性 | 高い（実行時交換） | 中 | 高い | 低い（継承で固定） |
| Undo | なし | なし | あり | なし |
| 典型的な数 | 少数〜中 | 有限個の状態 | 多数のコマンド | 1つの骨格 |
| テスト | 個別にテスト容易 | 状態ごとにテスト | コマンドごとにテスト | サブクラスごとにテスト |

### 比較表 2: クラス Strategy vs 関数 Strategy

| 観点 | クラスベース | 関数ベース |
|------|:---:|:---:|
| 状態保持 | フィールドで可能 | クロージャで可能 |
| 設定パラメータ | コンストラクタで注入 | 高階関数で注入 |
| テスト | インスタンス化して実行 | 直接呼び出し |
| コード量 | 多い（class, implements） | 少ない（関数リテラル） |
| 型安全性 | 高い（インタフェース強制） | 中（型エイリアスに依存） |
| DI フレームワーク | 対応しやすい | 対応にくい場合あり |
| 適用場面 | 複雑な戦略、状態を持つ | 単純な変換、ソート |

### 比較表 3: Strategy の適用判断

| 状況 | 推奨アプローチ | 理由 |
|------|-------------|------|
| バリエーションが2つ | 三項演算子/if-else | Strategy は過剰設計 |
| バリエーションが3〜5 | switch/enum または Strategy | 将来の追加を考慮して判断 |
| バリエーションが6以上 | Strategy + Registry | 条件分岐の管理が困難 |
| 実行時に切り替え必要 | Strategy | 主目的に合致 |
| アルゴリズムが複雑 | クラス Strategy | 状態とロジックのカプセル化 |
| アルゴリズムが単純 | 関数 Strategy | 軽量で十分 |

---

## 6. アンチパターン

### アンチパターン 1: 戦略が2つしかないのに Strategy パターン

```typescript
// NG: 過剰設計（YAGNI 違反）
interface GreetingStrategy {
  greet(name: string): string;
}
class FormalGreeting implements GreetingStrategy {
  greet(name: string) { return `Dear ${name}`; }
}
class CasualGreeting implements GreetingStrategy {
  greet(name: string) { return `Hi ${name}`; }
}

// このためだけにインタフェース + 2クラスは過剰
// 三項演算子で十分:

// OK: シンプルに書く
const greet = (name: string, formal: boolean) =>
  formal ? `Dear ${name}` : `Hi ${name}`;
```

**改善**: バリエーションが3つ以上、または将来の追加が見込まれる場合にのみ Strategy パターンを導入する。YAGNI（You Aren't Gonna Need It）の原則を忘れない。

### アンチパターン 2: Context が Strategy の内部を知っている

```typescript
// NG: Context が Strategy の具象型をチェック
class Context {
  execute(): void {
    if (this.strategy instanceof StrategyA) {
      // StrategyA 固有の前処理
      this.prepareForA();
    }
    if (this.strategy instanceof StrategyB) {
      // StrategyB 固有の前処理
      this.prepareForB();
    }
    this.strategy.execute();
  }
}
// 問題: Strategy を追加するたびに Context も変更が必要 -> OCP 違反

// OK: Context は Strategy インタフェースのみに依存
class Context {
  execute(): void {
    // 前処理は Strategy 内部に閉じ込める
    this.strategy.execute();
  }
}

// Strategy 側で前処理を含める
class StrategyA implements Strategy {
  execute(): void {
    this.prepare(); // 固有の前処理
    this.doWork();  // 本処理
  }
}
```

### アンチパターン 3: Strategy の粒度が不適切

```typescript
// NG: 1つの Strategy に複数の無関係な責務
interface AllInOneStrategy {
  calculatePrice(price: number): number;
  formatOutput(data: any): string;
  validateInput(input: string): boolean;
  sendNotification(message: string): void;
}
// 問題: 料金計算を変えたいだけなのに、全てのメソッドを実装する必要がある

// OK: 責務ごとに Strategy を分割
interface PricingStrategy {
  calculate(price: number): number;
}

interface FormattingStrategy {
  format(data: any): string;
}

interface ValidationStrategy {
  validate(input: string): ValidationResult;
}

// Context は必要な Strategy だけを使う
class OrderService {
  constructor(
    private pricing: PricingStrategy,
    private formatting: FormattingStrategy,
  ) {}
}
```

### アンチパターン 4: Strategy の切り替えがスレッドセーフでない

```typescript
// NG: マルチスレッド環境で Strategy の切り替えが競合する
class PaymentProcessor {
  private strategy: PaymentStrategy;

  setStrategy(strategy: PaymentStrategy): void {
    this.strategy = strategy; // スレッドAが書き換え中にスレッドBが読む可能性
  }

  process(order: Order): PaymentResult {
    return this.strategy.process(order); // どのStrategyが使われるか不定
  }
}

// OK: Strategy を引数で渡すか、イミュータブルなContextを使う
class PaymentProcessor {
  // 方法1: Strategyを引数として受け取る（状態を持たない）
  process(order: Order, strategy: PaymentStrategy): PaymentResult {
    return strategy.process(order);
  }
}

// 方法2: イミュータブルなContext（Strategyの変更時は新インスタンスを生成）
class PaymentProcessor {
  constructor(private readonly strategy: PaymentStrategy) {}

  withStrategy(strategy: PaymentStrategy): PaymentProcessor {
    return new PaymentProcessor(strategy);
  }

  process(order: Order): PaymentResult {
    return this.strategy.process(order);
  }
}
```

**改善**: マルチスレッド環境では、(1) Strategy を引数で渡す、(2) Context をイミュータブルにする、(3) スレッドローカルストレージを使う、のいずれかで安全性を確保する。

---

## 7. 実践演習

### 演習 1: 基礎 ── テキスト変換 Strategy

**課題**: テキスト変換を Strategy パターンで実装せよ。

要件:
1. `TextTransformer` インタフェース: `transform(text: string): string`
2. 具体 Strategy: `UpperCase`, `LowerCase`, `CamelCase`, `SnakeCase`, `KebabCase`
3. `TextProcessor` Context: Strategy を使ってテキストを変換

**テストケース**:

```typescript
const processor = new TextProcessor(new UpperCase());
console.log(processor.process("hello world")); // "HELLO WORLD"

processor.setStrategy(new CamelCase());
console.log(processor.process("hello world")); // "helloWorld"

processor.setStrategy(new SnakeCase());
console.log(processor.process("hello world")); // "hello_world"

processor.setStrategy(new KebabCase());
console.log(processor.process("hello world")); // "hello-world"
```

**期待される出力**: 上記コメントの通り。

---

### 演習 2: 応用 ── 動的 Strategy Registry

**課題**: Strategy Registry を実装し、設定ファイルやAPIパラメータから動的に Strategy を選択できるシステムを構築せよ。

要件:
1. `StrategyRegistry<T>` クラス: Strategy の登録と取得
2. デフォルト Strategy のサポート
3. 利用可能な Strategy の一覧取得
4. 実行時の Strategy 追加（プラグイン対応）
5. Shipping（送料計算）の具体例で実装

**テストケース**:

```typescript
const registry = new StrategyRegistry<ShippingStrategy>();
registry
  .register("standard", new StandardShipping(), true)
  .register("express", new ExpressShipping())
  .register("same-day", new SameDayShipping());

console.log(registry.getAvailableNames());
// ["standard", "express", "same-day"]

const strategy = registry.get("express");
console.log(strategy.calculate(1000, 2.5)); // 送料計算

// 未登録の名前 -> デフォルト戦略
const fallback = registry.get("unknown");
console.log(fallback === registry.get("standard")); // true
```

**期待される出力**: 上記コメントの通り。

---

### 演習 3: 発展 ── 合成可能な Strategy パイプライン

**課題**: 複数の Strategy を組み合わせて、パイプライン的に処理を適用できるフレームワークを構築せよ。

要件:
1. `TransformPipeline<T>` クラス: 複数の変換 Strategy をチェーン
2. `addStep(strategy)`: パイプラインにステップを追加
3. `execute(input)`: パイプラインを順次実行
4. `addConditional(predicate, strategy)`: 条件付き Strategy の適用
5. 画像処理のパイプラインで具体例を実装

**テストケース**:

```typescript
interface ImageData {
  width: number;
  height: number;
  format: string;
  quality: number;
}

const pipeline = new TransformPipeline<ImageData>()
  .addStep(new ResizeStrategy(800, 600))
  .addConditional(
    img => img.format === 'png',
    new ConvertToJpeg()
  )
  .addStep(new CompressStrategy(85));

const result = pipeline.execute({
  width: 1920, height: 1080, format: 'png', quality: 100
});
console.log(result);
// { width: 800, height: 600, format: 'jpg', quality: 85 }
```

**期待される出力**: 上記コメントの通り。

---

## 8. FAQ

### Q1: Strategy と DI は同じですか？

DI（依存性注入）は依存の注入メカニズム、Strategy はアルゴリズム交換のパターンです。DI は Strategy を実現する手段として使えますが、Strategy は DI なしでも実装可能です。DI コンテナ（InversifyJS, tsyringe 等）を使うと、設定ファイルから Strategy を自動的に注入できて便利ですが、必須ではありません。

### Q2: JavaScript では関数を渡すだけで Strategy は実現できますか？

はい。コールバック関数は Strategy パターンの軽量な実装です。`Array.sort(compareFn)` が典型例です。ただし、複雑な状態を持つ戦略やパラメータ設定が必要な戦略にはクラスが適しています。「関数1つで済むならクラスは不要、設定やテストの都合でクラスが必要なら使う」が実用的な判断基準です。

### Q3: Strategy と Template Method の違いは？

Strategy は委譲（has-a）でアルゴリズム全体を交換します。Template Method は継承（is-a）でアルゴリズムの一部をオーバーライドします。Strategy の方が柔軟性が高く、現代のプログラミングでは推奨されます。GoF 自身も「委譲を継承より優先せよ」と述べています。

### Q4: Strategy をいつ導入すべきですか？

以下の条件を満たす場合に導入を検討してください: (1) 同じ処理に3つ以上のバリエーションがある、(2) 将来新しいバリエーションの追加が見込まれる、(3) 実行時にアルゴリズムを切り替える必要がある、(4) アルゴリズムのテストを個別に行いたい。逆に、バリエーションが2つで将来の追加もないなら if-else で十分です。

### Q5: Strategy とポリモーフィズムの関係は？

Strategy パターンはポリモーフィズムの応用です。共通のインタフェースを通じて異なる実装を統一的に扱うという点で、ポリモーフィズムそのものです。OOP ではインタフェース/抽象クラスで、関数型では関数型（type alias）で実現します。

### Q6: Strategy パターンのテストはどう書くべきか？

テストは3つの層に分けて書くのが効果的です。

1. **各 Strategy の単体テスト**: Strategy ごとに入力と期待出力を検証する。Strategy は独立したクラス/関数なのでモック不要でテストしやすい。
2. **Context のテスト**: モック Strategy を注入して、Context が Strategy を正しく呼び出しているかを検証する。ここでは Strategy の実装詳細には踏み込まない。
3. **統合テスト**: 実際の Strategy と Context を組み合わせて、エンドツーエンドの動作を確認する。

```typescript
// 1. Strategy の単体テスト
describe('ExpressShipping', () => {
  it('5kg以下の荷物に速達料金を適用する', () => {
    const strategy = new ExpressShipping();
    expect(strategy.calculate(1000, 3.0)).toBe(1800); // 基本料 + 速達加算
  });
});

// 2. Context のテスト（モック使用）
describe('ShippingCalculator', () => {
  it('設定された Strategy に計算を委譲する', () => {
    const mockStrategy: ShippingStrategy = {
      calculate: jest.fn().mockReturnValue(500),
    };
    const calculator = new ShippingCalculator(mockStrategy);
    const result = calculator.calculateShipping(1000, 2.0);
    expect(mockStrategy.calculate).toHaveBeenCalledWith(1000, 2.0);
    expect(result).toBe(500);
  });
});

// 3. 統合テスト
describe('ShippingCalculator + StandardShipping', () => {
  it('実際の送料計算が正しい', () => {
    const calculator = new ShippingCalculator(new StandardShipping());
    expect(calculator.calculateShipping(1000, 2.0)).toBe(600);
  });
});
```

### Q7: Strategy パターンとデコレータパターンの使い分けは？

Strategy は「アルゴリズムの交換」、デコレータは「機能の追加・装飾」が目的です。Strategy ではある時点で1つの戦略が選択されて実行されますが、デコレータは複数のラッパーを重ねて機能を拡張します。

判断基準:
- 「A **または** B を実行する」→ Strategy（排他的選択）
- 「A **に加えて** B も実行する」→ Decorator（累積的追加）

実務では両者を組み合わせることも多く、例えばログ出力 Strategy をキャッシュ Decorator で包むといった設計が有効です。

---

## 9. まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | アルゴリズムをカプセル化して交換可能にする |
| OCP | 新しい戦略を追加しても既存コード変更不要 |
| 実装方式 | クラスベース（状態あり）/ 関数ベース（軽量） |
| Registry | 動的選択と拡張性を両立する補助パターン |
| 関数型 | 関数を渡すだけでも実現可能（Array.sort等） |
| 判断基準 | 3+バリエーション or 将来の拡張が見込まれる場合に導入 |
| 注意 | Context は Strategy の具象型を知らない設計にする |
| 粒度 | 1つの Strategy に1つの責務（ISP の遵守） |

---

## 次に読むべきガイド

- [Command パターン](./02-command.md) -- 操作のカプセル化と Undo/Redo
- [State パターン](./03-state.md) -- 状態遷移の管理
- [Observer パターン](./00-observer.md) -- イベント駆動設計
- [SOLID 原則](../../../clean-code-principles/docs/00-principles/01-solid.md) -- OCP の詳細
- [関数型パターン](../03-functional/02-fp-patterns.md) -- 関数合成とパイプライン

---

## 参考文献

1. Gamma, E., Helm, R., Johnson, R., Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley. -- Strategy パターンの原典。
2. Freeman, E., Robson, E. (2020). *Head First Design Patterns* (2nd Edition). O'Reilly Media. -- Strategy パターンを最初に扱い、設計原則との関連を丁寧に解説。
3. Refactoring.Guru -- Strategy. https://refactoring.guru/design-patterns/strategy -- 図解と多言語実装例。
4. Martin, R.C. (2003). *Agile Software Development: Principles, Patterns, and Practices*. Prentice Hall. -- OCP と Strategy の関係。
5. Fowler, M. (1999). *Refactoring: Improving the Design of Existing Code*. Addison-Wesley. -- 「条件分岐をポリモーフィズムに置き換える」リファクタリング。
