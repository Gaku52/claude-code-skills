# Adapter パターン

> 互換性のないインタフェースを持つクラスを **ラッパー** で包み、クライアントが期待するインタフェースに変換する構造パターン。

---

## 前提知識

| トピック | 必要レベル | 参照先 |
|---------|-----------|--------|
| オブジェクト指向プログラミング | 基礎 | [OOP基礎](../../../02-programming/oop-guide/docs/) |
| インタフェースと抽象クラス | 基礎 | [インタフェース設計](../../../02-programming/oop-guide/docs/) |
| 委譲（Delegation）と継承 | 理解 | [合成優先の原則](../../../03-software-design/clean-code-principles/docs/) |
| SOLID 原則（特に DIP, ISP） | 基礎 | [SOLID](../../../03-software-design/clean-code-principles/docs/) |
| TypeScript / Python の型システム | 基礎 | 各言語ガイド |

---

## この章で学ぶこと

1. Adapter パターンの**目的**と、なぜインタフェース変換が必要なのか
2. **オブジェクトアダプタ（委譲）** と **クラスアダプタ（継承）** の2つの形態と選択基準
3. 既存ライブラリ・レガシーコード・外部APIとの統合における実践的なアダプタの活用
4. **関数アダプタ（高階関数）** による軽量なインタフェース変換
5. Adapter と Facade・Decorator・Proxy の違い、過剰適用の回避

---

## なぜ Adapter パターンが必要なのか（WHY）

### 問題: インタフェースの不一致

現実のソフトウェア開発では、「使いたいクラスやライブラリがあるが、自分のコードが期待するインタフェースと合わない」という状況が頻繁に発生します。

```
[問題1: 外部ライブラリの統合]
  あなたのアプリは DataParser インタフェースを使っている
  だが、導入したい XML パーサーライブラリは全く別のメソッド名・引数を持つ
  → ライブラリのソースコードは変更できない

[問題2: レガシーコードとの共存]
  新しいシステムは NewOrderService を使う設計
  だが、旧システムの LegacyOrderService はメソッド名も引数も違う
  → 旧システムを全面書き換えする余裕がない

[問題3: サードパーティの切り替え]
  決済処理に Stripe を使っていたが、PayPal も追加したい
  各SDKのインタフェースは全く異なる
  → ビジネスロジックを決済SDKに依存させたくない

[問題4: テストの容易化]
  外部サービスに依存するコードをテストしたい
  モックに差し替えたいが、外部SDKのインタフェースは複雑すぎる
  → テスタブルなインタフェースに変換したい
```

### 解決: Adapter によるインタフェース変換

```
Before（直接依存 — 変更に弱い）:
┌──────────┐         ┌───────────────┐
│  Client  │────────>│ LegacyXmlParser│
│          │  直接依存 │ .parseXml()   │
└──────────┘         └───────────────┘
  ↑ Client が LegacyXmlParser の具象インタフェースに依存
  ↑ ライブラリを変更すると Client も変更が必要

After（Adapter を介在 — 変更に強い）:
┌──────────┐   uses    ┌──────────────┐  delegates  ┌───────────────┐
│  Client  │──────────>│  Adapter     │────────────>│ LegacyXmlParser│
│          │           │ .parse()     │             │ .parseXml()   │
└──────────┘           └──────────────┘             └───────────────┘
       │                      △
       │ depends on           │ implements
       ▼                      │
┌──────────────┐              │
│  DataParser  │──────────────┘
│  (interface) │
│  .parse()    │
└──────────────┘
  ↑ Client は DataParser インタフェースのみに依存
  ↑ ライブラリを変更しても Adapter だけ修正すれば OK
```

このパターンにより:
- **既存コードを変更せず**に互換性のないコンポーネントを統合できる
- クライアントが**具象クラスに依存しない**（依存性逆転の原則: DIP）
- サードパーティライブラリの**差し替えが容易**
- テストでモックに差し替えることが容易

---

## 1. Adapter の構造

### オブジェクトアダプタ（委譲ベース） — 推奨

```
+----------------+
|    Target      |
|  (interface)   |
+----------------+
| + request()    |
+----------------+
        △
        |  implements
+----------------+         delegates        +----------------+
|    Adapter     |─────────────────────────>|    Adaptee     |
+----------------+         has-a            +----------------+
| - adaptee:     |                          | + legacyOp()   |
|   Adaptee      |                          +----------------+
| + request() {  |
|   adaptee      |
|   .legacyOp() }|
+----------------+

Client ──uses──> Target(interface)
                    △
                    |
                 Adapter ──delegates──> Adaptee
```

### クラスアダプタ（継承ベース） — 非推奨

```
+----------------+         +----------------+
|    Target      |         |    Adaptee     |
|  (interface)   |         +----------------+
+----------------+         | + legacyOp()   |
| + request()    |         +----------------+
+----------------+                △
        △                        |  extends
        |  implements             |
        +────────────+────────────+
                     |
              +----------------+
              |    Adapter     |
              +----------------+
              | + request() {  |
              |   legacyOp()  }|  ← 自身の継承メソッドを呼ぶ
              +----------------+

問題: 多重継承が必要（Java/TS では不可）、密結合
```

### シーケンス図

```
Client          Adapter              Adaptee
  |                |                    |
  |--request()---->|                    |
  |                |--legacyOp()------->|
  |                |                    |
  |                |<--result-----------|
  |                |                    |
  |                |  [データ変換]       |
  |                |  convertResult()   |
  |                |                    |
  |<--変換済result--|                    |
  |                |                    |
```

---

## 2. オブジェクトアダプタ vs クラスアダプタ

### 詳細比較

| 観点 | オブジェクトアダプタ | クラスアダプタ |
|------|:---:|:---:|
| 実現方法 | **委譲（has-a）** | 継承（is-a） |
| 複数 Adaptee 対応 | **Yes**（コンストラクタで注入） | No（単一継承） |
| Adaptee のメソッド上書き | No（private のため） | Yes（protected にアクセス可） |
| 言語制約 | **なし** | 多重継承が必要（Java/TS で不可） |
| 結合度 | **低い** | 高い |
| テスト容易性 | **高い**（モック注入可） | 低い |
| 推奨度 | **高い** | 低い |
| SOLID 準拠 | **DIP, ISP 準拠** | OCP 違反リスク |

**結論**: ほぼ全てのケースでオブジェクトアダプタを使うべきです。クラスアダプタは Adaptee の protected メソッドにアクセスする必要がある場合にのみ検討してください。

---

## 3. コード例

### コード例 1: 外部ライブラリの Adapter（基本形）

```typescript
// === Adaptee: 既存の外部ライブラリ（変更不可）===
interface XmlDocument {
  root: string;
  format: string;
}

class LegacyXmlParser {
  parseXml(xmlString: string): XmlDocument {
    // XML をパースして独自形式で返す
    return { root: xmlString, format: "xml" };
  }

  validateXml(xmlString: string): boolean {
    return xmlString.startsWith("<");
  }
}

// === Target: クライアントが期待するインタフェース ===
interface DataParser {
  parse(input: string): Record<string, unknown>;
  validate(input: string): boolean;
}

// === Adapter: インタフェースを変換 ===
class XmlParserAdapter implements DataParser {
  private legacyParser: LegacyXmlParser;

  constructor(legacyParser?: LegacyXmlParser) {
    this.legacyParser = legacyParser ?? new LegacyXmlParser();
  }

  parse(input: string): Record<string, unknown> {
    // Adaptee のメソッドを呼び、結果を変換
    const xmlDoc = this.legacyParser.parseXml(input);
    return this.convertToRecord(xmlDoc);
  }

  validate(input: string): boolean {
    return this.legacyParser.validateXml(input);
  }

  private convertToRecord(doc: XmlDocument): Record<string, unknown> {
    return {
      data: doc.root,
      format: doc.format,
      parsedAt: new Date().toISOString()
    };
  }
}

// === Client: DataParser インタフェースだけを知っている ===
function processData(parser: DataParser, input: string): void {
  if (parser.validate(input)) {
    const result = parser.parse(input);
    console.log("Parsed:", result);
  } else {
    console.log("Invalid input");
  }
}

// 使用: クライアントは Adapter を DataParser として受け取る
const adapter = new XmlParserAdapter();
processData(adapter, "<user>Taro</user>");
// Parsed: { data: "<user>Taro</user>", format: "xml", parsedAt: "..." }
```

**ポイント**: Client は `DataParser` インタフェースだけに依存し、`LegacyXmlParser` の存在を知りません。将来 JSON パーサーに切り替えても Client のコードは変更不要です。

---

### コード例 2: ログライブラリの統一 Adapter

```typescript
// === Target: アプリ内の統一ログインタフェース ===
interface AppLogger {
  debug(message: string, context?: Record<string, unknown>): void;
  info(message: string, context?: Record<string, unknown>): void;
  warn(message: string, context?: Record<string, unknown>): void;
  error(message: string, error?: Error, context?: Record<string, unknown>): void;
}

// === Adaptee 1: Winston ===
interface WinstonLogger {
  log(level: string, message: string, meta?: object): void;
}

class WinstonAdapter implements AppLogger {
  constructor(private winston: WinstonLogger) {}

  debug(message: string, context?: Record<string, unknown>): void {
    this.winston.log("debug", message, context);
  }

  info(message: string, context?: Record<string, unknown>): void {
    this.winston.log("info", message, context);
  }

  warn(message: string, context?: Record<string, unknown>): void {
    this.winston.log("warn", message, context);
  }

  error(message: string, error?: Error, context?: Record<string, unknown>): void {
    this.winston.log("error", message, { ...context, error: error?.stack });
  }
}

// === Adaptee 2: Pino ===
interface PinoLogger {
  debug(msg: string): void;
  info(msg: string): void;
  warn(msg: string): void;
  error(obj: object, msg: string): void;
}

class PinoAdapter implements AppLogger {
  constructor(private pino: PinoLogger) {}

  debug(message: string, _context?: Record<string, unknown>): void {
    this.pino.debug(message);
  }

  info(message: string, _context?: Record<string, unknown>): void {
    this.pino.info(message);
  }

  warn(message: string, _context?: Record<string, unknown>): void {
    this.pino.warn(message);
  }

  error(message: string, error?: Error, context?: Record<string, unknown>): void {
    this.pino.error({ err: error, ...context }, message);
  }
}

// === Adaptee 3: Console（開発用） ===
class ConsoleAdapter implements AppLogger {
  debug(message: string, context?: Record<string, unknown>): void {
    console.debug(`[DEBUG] ${message}`, context ?? "");
  }

  info(message: string, context?: Record<string, unknown>): void {
    console.info(`[INFO] ${message}`, context ?? "");
  }

  warn(message: string, context?: Record<string, unknown>): void {
    console.warn(`[WARN] ${message}`, context ?? "");
  }

  error(message: string, error?: Error, context?: Record<string, unknown>): void {
    console.error(`[ERROR] ${message}`, error, context ?? "");
  }
}

// === Factory で適切な Adapter を選択 ===
function createLogger(env: string): AppLogger {
  switch (env) {
    case "production":
      // return new WinstonAdapter(createWinston());
    case "staging":
      // return new PinoAdapter(createPino());
    default:
      return new ConsoleAdapter();
  }
}

// 使用: アプリケーションコードは AppLogger だけに依存
const logger: AppLogger = createLogger(process.env.NODE_ENV ?? "development");
logger.info("Application started", { port: 3000 });
logger.error("Database connection failed", new Error("ECONNREFUSED"), { host: "localhost" });
```

---

### コード例 3: Python — 決済ゲートウェイ Adapter

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class PaymentStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class PaymentResult:
    """統一された決済結果"""
    status: PaymentStatus
    transaction_id: str
    amount: float
    currency: str


class PaymentGateway(ABC):
    """Target: 統一決済インタフェース"""
    @abstractmethod
    def charge(self, amount: float, currency: str, token: str) -> PaymentResult: ...

    @abstractmethod
    def refund(self, transaction_id: str, amount: float) -> PaymentResult: ...


# === Adaptee 1: Stripe SDK（変更不可）===
class StripeSDK:
    def create_charge(self, amount_cents: int, cur: str, source: str) -> dict:
        return {"id": "ch_123", "status": "succeeded", "amount": amount_cents}

    def create_refund(self, charge_id: str, amount_cents: int) -> dict:
        return {"id": "re_456", "status": "succeeded", "amount": amount_cents}


class StripeAdapter(PaymentGateway):
    """Stripe SDK を統一インタフェースに変換"""

    def __init__(self, sdk: StripeSDK):
        self._sdk = sdk

    def charge(self, amount: float, currency: str, token: str) -> PaymentResult:
        cents = int(amount * 100)  # ドル → セント変換
        result = self._sdk.create_charge(cents, currency, token)
        return PaymentResult(
            status=self._convert_status(result["status"]),
            transaction_id=result["id"],
            amount=amount,
            currency=currency,
        )

    def refund(self, transaction_id: str, amount: float) -> PaymentResult:
        cents = int(amount * 100)
        result = self._sdk.create_refund(transaction_id, cents)
        return PaymentResult(
            status=self._convert_status(result["status"]),
            transaction_id=result["id"],
            amount=amount,
            currency="",
        )

    @staticmethod
    def _convert_status(stripe_status: str) -> PaymentStatus:
        mapping = {
            "succeeded": PaymentStatus.SUCCESS,
            "failed": PaymentStatus.FAILED,
            "pending": PaymentStatus.PENDING,
        }
        return mapping.get(stripe_status, PaymentStatus.FAILED)


# === Adaptee 2: PayPal SDK（変更不可）===
class PayPalSDK:
    def execute_payment(self, payment_data: dict) -> dict:
        return {"payment_id": "PAY-789", "state": "approved"}

    def execute_refund(self, sale_id: str, refund_data: dict) -> dict:
        return {"refund_id": "REF-012", "state": "completed"}


class PayPalAdapter(PaymentGateway):
    """PayPal SDK を統一インタフェースに変換"""

    def __init__(self, sdk: PayPalSDK):
        self._sdk = sdk

    def charge(self, amount: float, currency: str, token: str) -> PaymentResult:
        result = self._sdk.execute_payment({
            "intent": "sale",
            "payer": {"payment_method": token},
            "transactions": [{"amount": {"total": str(amount), "currency": currency}}],
        })
        return PaymentResult(
            status=self._convert_status(result["state"]),
            transaction_id=result["payment_id"],
            amount=amount,
            currency=currency,
        )

    def refund(self, transaction_id: str, amount: float) -> PaymentResult:
        result = self._sdk.execute_refund(transaction_id, {
            "amount": {"total": str(amount)},
        })
        return PaymentResult(
            status=self._convert_status(result["state"]),
            transaction_id=result["refund_id"],
            amount=amount,
            currency="",
        )

    @staticmethod
    def _convert_status(paypal_state: str) -> PaymentStatus:
        mapping = {
            "approved": PaymentStatus.SUCCESS,
            "completed": PaymentStatus.SUCCESS,
            "failed": PaymentStatus.FAILED,
            "pending": PaymentStatus.PENDING,
        }
        return mapping.get(paypal_state, PaymentStatus.FAILED)


# === 使用例 ===
def process_order(gateway: PaymentGateway, amount: float) -> None:
    """ビジネスロジックは PaymentGateway インタフェースだけに依存"""
    result = gateway.charge(amount, "USD", "tok_test")
    if result.status == PaymentStatus.SUCCESS:
        print(f"Payment successful: {result.transaction_id}")
    else:
        print(f"Payment failed: {result.status}")


# Stripe を使う場合
stripe_gateway = StripeAdapter(StripeSDK())
process_order(stripe_gateway, 29.99)

# PayPal に切り替える場合 — ビジネスロジックは変更不要
paypal_gateway = PayPalAdapter(PayPalSDK())
process_order(paypal_gateway, 29.99)
```

---

### コード例 4: DOM イベントと独自イベントシステムの橋渡し

```typescript
// === Target: アプリ内の統一イベントシステム ===
interface AppEventEmitter {
  on<T = unknown>(event: string, handler: (data: T) => void): () => void; // unsubscribe 関数を返す
  emit<T = unknown>(event: string, data: T): void;
  off(event: string, handler: Function): void;
}

// === Adapter 1: DOM イベント → AppEventEmitter ===
class DOMEventAdapter implements AppEventEmitter {
  private handlers = new Map<Function, EventListener>();

  constructor(private element: HTMLElement) {}

  on<T = unknown>(event: string, handler: (data: T) => void): () => void {
    const listener = (e: Event) => {
      handler((e as CustomEvent).detail as T);
    };
    this.handlers.set(handler, listener);
    this.element.addEventListener(event, listener);

    // クリーンアップ関数を返す
    return () => this.off(event, handler);
  }

  emit<T = unknown>(event: string, data: T): void {
    this.element.dispatchEvent(
      new CustomEvent(event, { detail: data, bubbles: true })
    );
  }

  off(event: string, handler: Function): void {
    const listener = this.handlers.get(handler);
    if (listener) {
      this.element.removeEventListener(event, listener);
      this.handlers.delete(handler);
    }
  }
}

// === Adapter 2: Node.js EventEmitter → AppEventEmitter ===
// import { EventEmitter } from "events";

class NodeEventAdapter implements AppEventEmitter {
  constructor(private emitter: any /* EventEmitter */) {}

  on<T = unknown>(event: string, handler: (data: T) => void): () => void {
    this.emitter.on(event, handler);
    return () => this.off(event, handler);
  }

  emit<T = unknown>(event: string, data: T): void {
    this.emitter.emit(event, data);
  }

  off(event: string, handler: Function): void {
    this.emitter.removeListener(event, handler);
  }
}

// === Adapter 3: WebSocket → AppEventEmitter ===
class WebSocketEventAdapter implements AppEventEmitter {
  private handlers = new Map<string, Set<Function>>();

  constructor(private ws: WebSocket) {
    ws.addEventListener("message", (event) => {
      try {
        const { type, data } = JSON.parse(event.data);
        const set = this.handlers.get(type);
        if (set) {
          set.forEach(handler => handler(data));
        }
      } catch (e) {
        console.error("Failed to parse WebSocket message", e);
      }
    });
  }

  on<T = unknown>(event: string, handler: (data: T) => void): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
    return () => this.off(event, handler);
  }

  emit<T = unknown>(event: string, data: T): void {
    this.ws.send(JSON.stringify({ type: event, data }));
  }

  off(event: string, handler: Function): void {
    this.handlers.get(event)?.delete(handler);
  }
}

// 使用: クライアントコードは AppEventEmitter だけに依存
function setupNotifications(emitter: AppEventEmitter): void {
  const unsubscribe = emitter.on<{ message: string }>("notification", (data) => {
    console.log("Notification:", data.message);
  });

  // 後でクリーンアップ
  // unsubscribe();
}
```

---

### コード例 5: 関数アダプタ（高階関数）

```typescript
// === コールバック形式 → Promise 形式のアダプタ ===
type NodeCallback<T> = (err: Error | null, result: T) => void;
type CallbackFn<T> = (callback: NodeCallback<T>) => void;

function promisify<T>(fn: CallbackFn<T>): () => Promise<T>;
function promisify<T, A>(fn: (arg: A, cb: NodeCallback<T>) => void): (arg: A) => Promise<T>;
function promisify(fn: Function): (...args: any[]) => Promise<any> {
  return (...args: any[]) =>
    new Promise((resolve, reject) => {
      fn(...args, (err: Error | null, result: any) => {
        if (err) reject(err);
        else resolve(result);
      });
    });
}

// === イテレータ → 配列のアダプタ ===
function iteratorToArray<T>(iterator: Iterator<T>): T[] {
  const result: T[] = [];
  let next = iterator.next();
  while (!next.done) {
    result.push(next.value);
    next = iterator.next();
  }
  return result;
}

// === Observable → Promise のアダプタ ===
function observableToPromise<T>(observable: { subscribe: Function }): Promise<T> {
  return new Promise((resolve, reject) => {
    let lastValue: T;
    observable.subscribe({
      next: (value: T) => { lastValue = value; },
      error: reject,
      complete: () => resolve(lastValue),
    });
  });
}

// === 引数の順序を変えるアダプタ ===
function flip<A, B, R>(fn: (a: A, b: B) => R): (b: B, a: A) => R {
  return (b, a) => fn(a, b);
}

// === 複数引数 → 単一オブジェクト引数のアダプタ ===
type ParamsOf<F> = F extends (...args: infer P) => any ? P : never;

function objectify<F extends (...args: any[]) => any>(
  fn: F,
  paramNames: string[]
): (params: Record<string, any>) => ReturnType<F> {
  return (params) => {
    const args = paramNames.map(name => params[name]);
    return fn(...args);
  };
}

// 使用例
function createUser(name: string, age: number, email: string): { name: string; age: number; email: string } {
  return { name, age, email };
}

const createUserFromObject = objectify(createUser, ["name", "age", "email"]);
const user = createUserFromObject({ name: "Taro", age: 25, email: "taro@example.com" });
```

**関数アダプタのメリット**:
- クラスを定義する必要がない（軽量）
- 関数型プログラミングと相性が良い
- 単純な変換なら一行で済む

---

### コード例 6: Java — ORM と DTO の Adapter

```java
// === Target: アプリケーション層の DTO ===
public record UserDTO(
    String id,
    String fullName,
    String email,
    LocalDateTime createdAt
) {}

// === Adaptee 1: JPA Entity（データベース層）===
@Entity
public class UserEntity {
    @Id private Long id;
    private String firstName;
    private String lastName;
    private String emailAddress;
    private Timestamp createdTimestamp;

    // getters/setters...
    public Long getId() { return id; }
    public String getFirstName() { return firstName; }
    public String getLastName() { return lastName; }
    public String getEmailAddress() { return emailAddress; }
    public Timestamp getCreatedTimestamp() { return createdTimestamp; }
}

// === Adaptee 2: 外部 API レスポンス ===
public class ExternalUserResponse {
    private String user_id;
    private String display_name;
    private String contact_email;
    private String registered_at; // ISO 8601 string

    // getters...
    public String getUserId() { return user_id; }
    public String getDisplayName() { return display_name; }
    public String getContactEmail() { return contact_email; }
    public String getRegisteredAt() { return registered_at; }
}

// === Adapter インタフェース ===
public interface UserAdapter<T> {
    UserDTO toDTO(T source);
    T fromDTO(UserDTO dto);
}

// === Adapter 1: JPA Entity → DTO ===
public class JpaUserAdapter implements UserAdapter<UserEntity> {
    @Override
    public UserDTO toDTO(UserEntity entity) {
        return new UserDTO(
            String.valueOf(entity.getId()),
            entity.getFirstName() + " " + entity.getLastName(),
            entity.getEmailAddress(),
            entity.getCreatedTimestamp().toLocalDateTime()
        );
    }

    @Override
    public UserEntity fromDTO(UserDTO dto) {
        UserEntity entity = new UserEntity();
        String[] names = dto.fullName().split(" ", 2);
        entity.setFirstName(names[0]);
        entity.setLastName(names.length > 1 ? names[1] : "");
        entity.setEmailAddress(dto.email());
        return entity;
    }
}

// === Adapter 2: 外部 API レスポンス → DTO ===
public class ExternalUserAdapter implements UserAdapter<ExternalUserResponse> {
    @Override
    public UserDTO toDTO(ExternalUserResponse response) {
        return new UserDTO(
            response.getUserId(),
            response.getDisplayName(),
            response.getContactEmail(),
            LocalDateTime.parse(response.getRegisteredAt(), DateTimeFormatter.ISO_DATE_TIME)
        );
    }

    @Override
    public ExternalUserResponse fromDTO(UserDTO dto) {
        // 逆変換は必要に応じて実装
        throw new UnsupportedOperationException("One-way adapter");
    }
}
```

---

### コード例 7: Go — インタフェースベースの Adapter

```go
package main

import (
    "fmt"
    "strings"
)

// === Target: アプリケーションが使うインタフェース ===
type MessageSender interface {
    Send(to string, subject string, body string) error
}

// === Adaptee 1: SMTP ライブラリ（レガシー）===
type LegacySMTP struct{}

func (s *LegacySMTP) SendMail(recipient string, headers map[string]string, content string) error {
    fmt.Printf("SMTP: To=%s, Subject=%s\n", recipient, headers["Subject"])
    return nil
}

// === Adapter 1: LegacySMTP → MessageSender ===
type SMTPAdapter struct {
    smtp *LegacySMTP
}

func NewSMTPAdapter(smtp *LegacySMTP) *SMTPAdapter {
    return &SMTPAdapter{smtp: smtp}
}

func (a *SMTPAdapter) Send(to string, subject string, body string) error {
    headers := map[string]string{
        "Subject":      subject,
        "Content-Type": "text/plain",
    }
    return a.smtp.SendMail(to, headers, body)
}

// === Adaptee 2: Slack Webhook ===
type SlackWebhook struct {
    WebhookURL string
}

func (s *SlackWebhook) PostMessage(channel string, text string) error {
    fmt.Printf("Slack: Channel=%s, Text=%s\n", channel, text)
    return nil
}

// === Adapter 2: SlackWebhook → MessageSender ===
type SlackAdapter struct {
    slack *SlackWebhook
}

func NewSlackAdapter(slack *SlackWebhook) *SlackAdapter {
    return &SlackAdapter{slack: slack}
}

func (a *SlackAdapter) Send(to string, subject string, body string) error {
    text := fmt.Sprintf("*%s*\n%s", subject, body)
    return a.slack.PostMessage(to, text)
}

// === 使用例 ===
func notifyUser(sender MessageSender, to string) error {
    return sender.Send(to, "Welcome", "Hello, welcome to our service!")
}

func main() {
    // SMTP で送信
    smtpSender := NewSMTPAdapter(&LegacySMTP{})
    notifyUser(smtpSender, "user@example.com")

    // Slack で送信（コードの変更不要）
    slackSender := NewSlackAdapter(&SlackWebhook{WebhookURL: "https://hooks.slack.com/xxx"})
    notifyUser(slackSender, "#general")
}
```

---

### コード例 8: Kotlin — 拡張関数による軽量 Adapter

```kotlin
// === Adaptee: サードパーティの天気API ===
data class WeatherApiResponse(
    val temp_c: Double,
    val humidity_pct: Int,
    val wind_kph: Double,
    val condition_code: Int
)

// === Target: アプリケーションのドメインモデル ===
data class WeatherInfo(
    val temperatureCelsius: Double,
    val temperatureFahrenheit: Double,
    val humidityPercent: Int,
    val windSpeedKmh: Double,
    val condition: String
)

// === Adapter: 拡張関数で変換 ===
fun WeatherApiResponse.toWeatherInfo(): WeatherInfo {
    return WeatherInfo(
        temperatureCelsius = this.temp_c,
        temperatureFahrenheit = this.temp_c * 9.0 / 5.0 + 32.0,
        humidityPercent = this.humidity_pct,
        windSpeedKmh = this.wind_kph,
        condition = mapCondition(this.condition_code)
    )
}

private fun mapCondition(code: Int): String = when (code) {
    1 -> "Clear"
    2 -> "Partly Cloudy"
    3 -> "Cloudy"
    4 -> "Rain"
    5 -> "Snow"
    else -> "Unknown"
}

// === 使用例 ===
fun displayWeather(info: WeatherInfo) {
    println("${info.temperatureCelsius}°C (${info.temperatureFahrenheit}°F)")
    println("Humidity: ${info.humidityPercent}%")
    println("Condition: ${info.condition}")
}

fun main() {
    // API レスポンスを取得
    val apiResponse = WeatherApiResponse(
        temp_c = 22.5,
        humidity_pct = 65,
        wind_kph = 15.0,
        condition_code = 2
    )

    // 拡張関数で変換（Adapter）
    val weatherInfo = apiResponse.toWeatherInfo()
    displayWeather(weatherInfo)
}
```

---

### コード例 9: Adapter + Strategy パターンの組み合わせ

```typescript
// 複数の通知チャネルを Adapter で統一し、
// Strategy パターンでチャネルを動的に切り替える

// === Target ===
interface NotificationChannel {
  send(userId: string, message: string): Promise<boolean>;
  getName(): string;
}

// === Adapter 1: Email ===
class EmailAdapter implements NotificationChannel {
  constructor(private smtpClient: any) {}

  async send(userId: string, message: string): Promise<boolean> {
    const email = await this.resolveEmail(userId);
    await this.smtpClient.sendMail({
      to: email,
      subject: "Notification",
      text: message,
    });
    return true;
  }

  getName(): string { return "email"; }

  private async resolveEmail(userId: string): Promise<string> {
    return `${userId}@example.com`; // 実際はDB検索
  }
}

// === Adapter 2: SMS ===
class SMSAdapter implements NotificationChannel {
  constructor(private twilioClient: any) {}

  async send(userId: string, message: string): Promise<boolean> {
    const phone = await this.resolvePhone(userId);
    await this.twilioClient.messages.create({
      to: phone,
      body: message,
    });
    return true;
  }

  getName(): string { return "sms"; }

  private async resolvePhone(userId: string): Promise<string> {
    return "+8190XXXXXXXX"; // 実際はDB検索
  }
}

// === Adapter 3: Push Notification ===
class PushNotificationAdapter implements NotificationChannel {
  constructor(private fcmClient: any) {}

  async send(userId: string, message: string): Promise<boolean> {
    const token = await this.resolveToken(userId);
    await this.fcmClient.send({
      token,
      notification: { title: "Notification", body: message },
    });
    return true;
  }

  getName(): string { return "push"; }

  private async resolveToken(userId: string): Promise<string> {
    return "fcm-token-xxx"; // 実際はDB検索
  }
}

// === Strategy: 通知チャネルを動的に選択 ===
class NotificationService {
  private channels = new Map<string, NotificationChannel>();

  registerChannel(channel: NotificationChannel): void {
    this.channels.set(channel.getName(), channel);
  }

  async notify(
    userId: string,
    message: string,
    channelName: string
  ): Promise<boolean> {
    const channel = this.channels.get(channelName);
    if (!channel) {
      throw new Error(`Unknown channel: ${channelName}`);
    }
    return channel.send(userId, message);
  }

  async notifyAll(userId: string, message: string): Promise<boolean[]> {
    const promises = [...this.channels.values()].map(ch =>
      ch.send(userId, message)
    );
    return Promise.all(promises);
  }
}

// 使用例
const service = new NotificationService();
service.registerChannel(new EmailAdapter(smtpClient));
service.registerChannel(new SMSAdapter(twilioClient));
service.registerChannel(new PushNotificationAdapter(fcmClient));

// ユーザー設定に応じてチャネルを選択
await service.notify("user-123", "Your order has shipped!", "email");
await service.notifyAll("user-456", "System maintenance in 1 hour");
```

---

### コード例 10: Two-Way Adapter（双方向アダプタ）

```typescript
// 2つの異なるシステム間でデータを相互変換する双方向アダプタ

// === System A: REST API 形式 ===
interface RestApiUser {
  id: string;
  first_name: string;
  last_name: string;
  email_address: string;
  created_at: string; // ISO 8601
}

// === System B: GraphQL 形式 ===
interface GraphQLUser {
  userId: string;
  fullName: string;
  contactInfo: {
    email: string;
  };
  metadata: {
    registrationDate: number; // Unix timestamp
  };
}

// === Two-Way Adapter ===
class UserFormatAdapter {
  // REST → GraphQL
  restToGraphQL(rest: RestApiUser): GraphQLUser {
    return {
      userId: rest.id,
      fullName: `${rest.first_name} ${rest.last_name}`,
      contactInfo: {
        email: rest.email_address,
      },
      metadata: {
        registrationDate: new Date(rest.created_at).getTime(),
      },
    };
  }

  // GraphQL → REST
  graphQLToRest(gql: GraphQLUser): RestApiUser {
    const [firstName, ...lastNameParts] = gql.fullName.split(" ");
    return {
      id: gql.userId,
      first_name: firstName,
      last_name: lastNameParts.join(" "),
      email_address: gql.contactInfo.email,
      created_at: new Date(gql.metadata.registrationDate).toISOString(),
    };
  }

  // バッチ変換
  restListToGraphQL(users: RestApiUser[]): GraphQLUser[] {
    return users.map(u => this.restToGraphQL(u));
  }

  graphQLListToRest(users: GraphQLUser[]): RestApiUser[] {
    return users.map(u => this.graphQLToRest(u));
  }
}

// 使用例: マイクロサービス間のデータ同期
const adapter = new UserFormatAdapter();

const restUser: RestApiUser = {
  id: "usr-001",
  first_name: "Taro",
  last_name: "Yamada",
  email_address: "taro@example.com",
  created_at: "2024-01-15T09:00:00Z",
};

const graphqlUser = adapter.restToGraphQL(restUser);
console.log(graphqlUser.fullName);           // "Taro Yamada"
console.log(graphqlUser.contactInfo.email);  // "taro@example.com"

const backToRest = adapter.graphQLToRest(graphqlUser);
console.log(backToRest.first_name);  // "Taro"
console.log(backToRest.last_name);   // "Yamada"
```

---

## 4. 比較表

### 比較表 1: Adapter vs Facade vs Decorator vs Proxy

| 観点 | Adapter | Facade | Decorator | Proxy |
|------|---------|--------|-----------|-------|
| **目的** | インタフェース**変換** | 複雑さの**隠蔽** | 機能の**追加** | アクセスの**制御** |
| **対象** | 1つのクラス/API | 複数のクラス群 | 1つのオブジェクト | 1つのオブジェクト |
| **インタフェース** | **変換**する | **単純化**する | **同じまま** | **同じまま** |
| **既存コード** | 変更不可 | 変更不要 | 変更不要 | 変更不要 |
| **使用場面** | ライブラリ統合 | サブシステム公開 | ログ/キャッシュ追加 | 遅延/権限/キャッシュ |
| **GoF 分類** | 構造 | 構造 | 構造 | 構造 |

```
視覚的な違い:

Adapter:   Client ──> [A→B変換] ──> Adaptee
Facade:    Client ──> [簡易窓口] ──> SubSystem1 + SubSystem2 + SubSystem3
Decorator: Client ──> [追加処理] ──> [追加処理] ──> Original
Proxy:     Client ──> [アクセス制御] ──> RealSubject
```

### 比較表 2: オブジェクトアダプタ vs クラスアダプタ（詳細）

| 観点 | オブジェクトアダプタ | クラスアダプタ |
|------|:---:|:---:|
| 実現方法 | **委譲（has-a）** | 継承（is-a） |
| 複数 Adaptee 対応 | **Yes** | No |
| Adaptee のメソッド上書き | No | Yes |
| 言語制約 | **なし** | 多重継承が必要 |
| 結合度 | **低い** | 高い |
| テスト容易性 | **高い** | 低い |
| DI 対応 | **Yes** | No |
| 推奨度 | **高い** | 低い |
| SOLID 準拠 | **DIP/ISP準拠** | LSP/OCP違反リスク |

### 比較表 3: Adapter の実装アプローチ比較

| アプローチ | 適用場面 | 複雑度 | 型安全性 | 再利用性 |
|-----------|---------|:---:|:---:|:---:|
| クラスアダプタ | 大規模な変換、状態管理あり | 中 | **高** | **高** |
| 関数アダプタ | 単純な変換、状態なし | **低** | 中 | 中 |
| 拡張関数（Kotlin） | データ変換、DTO マッピング | **低** | **高** | 中 |
| ジェネリックアダプタ | 共通パターンの抽象化 | 高 | **高** | **最高** |

---

## 5. アンチパターン

### アンチパターン 1: 薄すぎるアダプタ（不要な間接層）

```typescript
// NG: 単にメソッド名を変えただけ、インタフェースが実質同じ
interface Logger {
  log(message: string): void;
}

class ConsoleLogger {
  log(message: string): void {
    console.log(message);
  }
}

// ← このアダプタは不要！ConsoleLogger が直接 Logger を implements すればよい
class UselessAdapter implements Logger {
  constructor(private logger: ConsoleLogger) {}
  log(message: string): void {
    this.logger.log(message);  // シグネチャが完全に同じ
  }
}
```

```typescript
// OK: ConsoleLogger に直接インタフェースを実装
class ConsoleLogger implements Logger {
  log(message: string): void {
    console.log(message);
  }
}

// または TypeScript では構造的部分型なので、
// ConsoleLogger は Logger と互換性があればそのまま使える
```

**判断基準**: インタフェースが同じなら Adapter は不要です。Adapter は「変換が必要な場合」にのみ使うべきです。

---

### アンチパターン 2: アダプタにビジネスロジックを追加

```typescript
// NG: Adapter が変換以上の責任を持つ
class OrderAdapter implements NewOrderService {
  constructor(private legacyService: LegacyOrderService) {}

  createOrder(data: NewOrderData): Order {
    const legacyData = this.convertData(data);
    const order = this.legacyService.createLegacyOrder(legacyData);

    // ビジネスロジック — Adapter の責務ではない！
    order.applyTax(this.calculateTax(order));
    order.validateInventory();
    this.sendNotification(order);
    this.updateAnalytics(order);

    return this.convertOrder(order);
  }
}
```

```typescript
// OK: Adapter は変換のみ。ビジネスロジックはサービス層に配置
class OrderAdapter implements NewOrderService {
  constructor(private legacyService: LegacyOrderService) {}

  createOrder(data: NewOrderData): LegacyOrder {
    // 変換のみ
    const legacyData = this.convertToLegacyFormat(data);
    const result = this.legacyService.createLegacyOrder(legacyData);
    return this.convertToNewFormat(result);
  }
}

// ビジネスロジックはサービス層
class OrderService {
  constructor(
    private orderAdapter: NewOrderService,
    private taxService: TaxService,
    private notificationService: NotificationService
  ) {}

  async processOrder(data: NewOrderData): Promise<Order> {
    const order = this.orderAdapter.createOrder(data);
    order.tax = this.taxService.calculate(order);
    await this.notificationService.send(order);
    return order;
  }
}
```

---

### アンチパターン 3: God Adapter（万能アダプタ）

```typescript
// NG: 1つのアダプタが複数の異なるシステムを扱う
class UniversalPaymentAdapter {
  constructor(
    private stripe: StripeSDK,
    private paypal: PayPalSDK,
    private square: SquareSDK
  ) {}

  charge(provider: string, amount: number): void {
    switch (provider) {
      case "stripe":
        this.stripe.createCharge(amount * 100, "usd");
        break;
      case "paypal":
        this.paypal.executePayment({ amount });
        break;
      case "square":
        this.square.createPayment({ amount_money: { amount, currency: "USD" } });
        break;
    }
  }
  // OCP 違反: 新しいプロバイダ追加のたびに switch を修正
}
```

```typescript
// OK: プロバイダごとに個別の Adapter を作成
interface PaymentGateway {
  charge(amount: number, currency: string): Promise<PaymentResult>;
}

class StripeAdapter implements PaymentGateway { /* ... */ }
class PayPalAdapter implements PaymentGateway { /* ... */ }
class SquareAdapter implements PaymentGateway { /* ... */ }

// Factory で選択
class PaymentGatewayFactory {
  private adapters = new Map<string, PaymentGateway>();

  register(name: string, adapter: PaymentGateway): void {
    this.adapters.set(name, adapter);
  }

  get(name: string): PaymentGateway {
    const adapter = this.adapters.get(name);
    if (!adapter) throw new Error(`Unknown provider: ${name}`);
    return adapter;
  }
}
```

---

## 6. エッジケースと注意点

### エッジケース 1: 双方向変換でのデータロス

```typescript
// REST → GraphQL 変換時に情報が失われる場合がある
interface DetailedRestUser {
  id: string;
  first_name: string;
  middle_name: string;      // GraphQL 側にはこのフィールドがない
  last_name: string;
  email: string;
  internal_notes: string;   // 変換先に該当フィールドがない
}

// 対策1: 変換時に警告ログを出力
// 対策2: 拡張フィールド（extras: Map）を用意
// 対策3: 双方向変換のテストで roundtrip を検証
```

### エッジケース 2: 非同期アダプタのエラーハンドリング

```typescript
class AsyncAdapter implements DataParser {
  constructor(private asyncParser: AsyncLegacyParser) {}

  async parse(input: string): Promise<Record<string, unknown>> {
    try {
      const result = await this.asyncParser.parseAsync(input);
      return this.convert(result);
    } catch (error) {
      // Adaptee 固有のエラーを統一エラーに変換
      if (error instanceof LegacyParseError) {
        throw new ParseError(error.message, error.line, error.column);
      }
      throw new ParseError(`Unknown error: ${error}`);
    }
  }
}
```

### エッジケース 3: アダプタのライフサイクル管理

```typescript
// Adaptee がリソースを持つ場合、cleanup が必要
class DatabaseAdapter implements DataStore {
  constructor(private connection: LegacyDBConnection) {}

  async get(key: string): Promise<string> { /* ... */ }
  async set(key: string, value: string): Promise<void> { /* ... */ }

  // Adapter が Dispose パターンも実装する必要がある
  async close(): Promise<void> {
    await this.connection.disconnect();
  }
}

// using 文（TC39 Stage 3）で自動クリーンアップ
// await using adapter = new DatabaseAdapter(connection);
```

---

## 7. トレードオフ分析

### Adapter パターンを使うべき場面

```
[使うべき場面] ✅
┌─────────────────────────────────────────────────────────┐
│ 1. 外部ライブラリの統合                                   │
│    変更できないサードパーティコードとの接続                 │
│                                                          │
│ 2. レガシーシステムの段階的移行                            │
│    旧APIと新APIの橋渡し（Strangler Fig パターンと併用）    │
│                                                          │
│ 3. テストの容易化                                         │
│    外部依存を統一インタフェースに変換してモック可能にする    │
│                                                          │
│ 4. 複数プロバイダの統一                                    │
│    決済、通知、ストレージ等の複数ベンダー対応              │
│                                                          │
│ 5. データフォーマットの変換                                │
│    REST/GraphQL/gRPC 間、DTO/Entity 間のマッピング        │
└─────────────────────────────────────────────────────────┘

[使うべきでない場面] ❌
┌─────────────────────────────────────────────────────────┐
│ 1. インタフェースが既に一致している                       │
│    → 不要な間接層はコードの可読性を下げる                 │
│                                                          │
│ 2. Adaptee を直接変更できる場合                           │
│    → 直接インタフェースを修正する方がシンプル             │
│                                                          │
│ 3. 変換だけでなく大量のビジネスロジックが必要な場合       │
│    → Adapter ではなく専用のサービス層を作る               │
│                                                          │
│ 4. パフォーマンスが最優先の場合                           │
│    → 間接層のオーバーヘッドが問題になることがある          │
└─────────────────────────────────────────────────────────┘
```

### コスト分析

| 項目 | Adapter あり | Adapter なし |
|------|:---:|:---:|
| 初期実装コスト | 中（Adapter クラス作成） | **低** |
| ライブラリ変更時のコスト | **低**（Adapter のみ修正） | 高（全呼び出し元を修正） |
| テスト容易性 | **高** | 低 |
| コードの複雑度 | やや増加 | **シンプル** |
| 長期保守コスト | **低** | 高 |

---

## 8. 演習問題

### 演習 1（基礎）: ファイルシステム Adapter

以下のインタフェースと既存クラスに対して Adapter を実装してください。

**要件**:
- `Storage` インタフェース: `read(key)`, `write(key, value)`, `delete(key)`, `exists(key)`
- `LegacyFileSystem` クラス: `loadFile(path)`, `saveFile(path, content)`, `removeFile(path)`, `fileExists(path)`
- メソッド名とパラメータ名の違いを吸収する Adapter を作成

```typescript
// テスト
const adapter: Storage = new FileSystemAdapter(new LegacyFileSystem("/data"));
await adapter.write("config", '{"debug": true}');
console.log(await adapter.exists("config"));   // true
console.log(await adapter.read("config"));     // '{"debug": true}'
await adapter.delete("config");
console.log(await adapter.exists("config"));   // false
```

**期待される出力**:
```
true
{"debug": true}
false
```

<details>
<summary>解答例</summary>

```typescript
interface Storage {
  read(key: string): Promise<string>;
  write(key: string, value: string): Promise<void>;
  delete(key: string): Promise<void>;
  exists(key: string): Promise<boolean>;
}

class LegacyFileSystem {
  private files = new Map<string, string>();

  constructor(private basePath: string) {}

  loadFile(path: string): string {
    const content = this.files.get(path);
    if (!content) throw new Error(`File not found: ${path}`);
    return content;
  }

  saveFile(path: string, content: string): void {
    this.files.set(path, content);
  }

  removeFile(path: string): void {
    this.files.delete(path);
  }

  fileExists(path: string): boolean {
    return this.files.has(path);
  }
}

class FileSystemAdapter implements Storage {
  constructor(private fs: LegacyFileSystem) {}

  private toPath(key: string): string {
    return key; // 必要に応じてパス変換
  }

  async read(key: string): Promise<string> {
    return this.fs.loadFile(this.toPath(key));
  }

  async write(key: string, value: string): Promise<void> {
    this.fs.saveFile(this.toPath(key), value);
  }

  async delete(key: string): Promise<void> {
    this.fs.removeFile(this.toPath(key));
  }

  async exists(key: string): Promise<boolean> {
    return this.fs.fileExists(this.toPath(key));
  }
}

// テスト
const adapter: Storage = new FileSystemAdapter(new LegacyFileSystem("/data"));
await adapter.write("config", '{"debug": true}');
console.log(await adapter.exists("config"));   // true
console.log(await adapter.read("config"));     // '{"debug": true}'
await adapter.delete("config");
console.log(await adapter.exists("config"));   // false
```
</details>

---

### 演習 2（応用）: マルチプロバイダ Adapter + Factory

複数のクラウドストレージプロバイダに対応する Adapter と Factory を実装してください。

**要件**:
- `CloudStorage` インタフェース: `upload(key, data)`, `download(key)`, `delete(key)`, `list(prefix)`
- AWS S3, Google Cloud Storage, Azure Blob Storage の3つの Adapter
- `CloudStorageFactory` でプロバイダ名から Adapter を選択

```typescript
// テスト
const factory = new CloudStorageFactory();
factory.register("s3", new S3Adapter(s3Client));
factory.register("gcs", new GCSAdapter(gcsClient));

const storage = factory.get("s3");
await storage.upload("reports/2024.pdf", pdfData);
const files = await storage.list("reports/");
console.log(files); // ["reports/2024.pdf"]
```

<details>
<summary>解答例</summary>

```typescript
interface CloudStorage {
  upload(key: string, data: Buffer): Promise<string>;
  download(key: string): Promise<Buffer>;
  delete(key: string): Promise<void>;
  list(prefix: string): Promise<string[]>;
}

// === S3 Adapter ===
class S3Adapter implements CloudStorage {
  private storage = new Map<string, Buffer>();

  constructor(private client: any /* S3Client */) {}

  async upload(key: string, data: Buffer): Promise<string> {
    this.storage.set(key, data);
    return `s3://bucket/${key}`;
  }

  async download(key: string): Promise<Buffer> {
    const data = this.storage.get(key);
    if (!data) throw new Error(`Not found: ${key}`);
    return data;
  }

  async delete(key: string): Promise<void> {
    this.storage.delete(key);
  }

  async list(prefix: string): Promise<string[]> {
    return [...this.storage.keys()].filter(k => k.startsWith(prefix));
  }
}

// === GCS Adapter ===
class GCSAdapter implements CloudStorage {
  private storage = new Map<string, Buffer>();

  constructor(private client: any /* GCSClient */) {}

  async upload(key: string, data: Buffer): Promise<string> {
    this.storage.set(key, data);
    return `gs://bucket/${key}`;
  }

  async download(key: string): Promise<Buffer> {
    const data = this.storage.get(key);
    if (!data) throw new Error(`Not found: ${key}`);
    return data;
  }

  async delete(key: string): Promise<void> {
    this.storage.delete(key);
  }

  async list(prefix: string): Promise<string[]> {
    return [...this.storage.keys()].filter(k => k.startsWith(prefix));
  }
}

// === Factory ===
class CloudStorageFactory {
  private adapters = new Map<string, CloudStorage>();

  register(name: string, adapter: CloudStorage): void {
    this.adapters.set(name, adapter);
  }

  get(name: string): CloudStorage {
    const adapter = this.adapters.get(name);
    if (!adapter) throw new Error(`Unknown provider: ${name}`);
    return adapter;
  }

  listProviders(): string[] {
    return [...this.adapters.keys()];
  }
}
```
</details>

---

### 演習 3（上級）: 型安全なジェネリック Adapter フレームワーク

任意の2つのインタフェース間のマッピングを型安全に定義できるジェネリック Adapter フレームワークを実装してください。

**要件**:
- フィールドマッピングを宣言的に定義
- 変換関数をフィールドごとに指定可能
- 双方向変換をサポート
- TypeScript の型推論で変換結果の型が保証される

```typescript
// テスト
const userMapper = createMapper<RestUser, DomainUser>({
  id: (src) => src.user_id,
  name: (src) => `${src.first_name} ${src.last_name}`,
  email: (src) => src.email_address,
  createdAt: (src) => new Date(src.created_at),
});

const restUser = {
  user_id: "123",
  first_name: "Taro",
  last_name: "Yamada",
  email_address: "taro@example.com",
  created_at: "2024-01-15T09:00:00Z",
};

const domainUser = userMapper.map(restUser);
console.log(domainUser.name);      // "Taro Yamada"
console.log(domainUser.email);     // "taro@example.com"
console.log(domainUser.createdAt instanceof Date); // true
```

**期待される出力**:
```
Taro Yamada
taro@example.com
true
```

<details>
<summary>解答例</summary>

```typescript
// マッピング定義の型
type MappingConfig<Source, Target> = {
  [K in keyof Target]: (source: Source) => Target[K];
};

// リバースマッピングの型
type ReverseMappingConfig<Source, Target> = {
  [K in keyof Source]: (target: Target) => Source[K];
};

// Mapper インタフェース
interface Mapper<Source, Target> {
  map(source: Source): Target;
  mapMany(sources: Source[]): Target[];
}

// 双方向 Mapper
interface BiMapper<A, B> {
  mapAtoB(a: A): B;
  mapBtoA(b: B): A;
  mapManyAtoB(as: A[]): B[];
  mapManyBtoA(bs: B[]): A[];
}

// Mapper 作成関数
function createMapper<Source, Target>(
  config: MappingConfig<Source, Target>
): Mapper<Source, Target> {
  return {
    map(source: Source): Target {
      const result = {} as Target;
      for (const key of Object.keys(config) as Array<keyof Target>) {
        result[key] = config[key](source);
      }
      return result;
    },
    mapMany(sources: Source[]): Target[] {
      return sources.map(s => this.map(s));
    },
  };
}

// 双方向 Mapper 作成関数
function createBiMapper<A, B>(
  aToB: MappingConfig<A, B>,
  bToA: MappingConfig<B, A>
): BiMapper<A, B> {
  const forwardMapper = createMapper(aToB);
  const reverseMapper = createMapper(bToA);

  return {
    mapAtoB: (a) => forwardMapper.map(a),
    mapBtoA: (b) => reverseMapper.map(b),
    mapManyAtoB: (as) => forwardMapper.mapMany(as),
    mapManyBtoA: (bs) => reverseMapper.mapMany(bs),
  };
}

// === 使用例 ===

interface RestUser {
  user_id: string;
  first_name: string;
  last_name: string;
  email_address: string;
  created_at: string;
}

interface DomainUser {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

const userMapper = createMapper<RestUser, DomainUser>({
  id: (src) => src.user_id,
  name: (src) => `${src.first_name} ${src.last_name}`,
  email: (src) => src.email_address,
  createdAt: (src) => new Date(src.created_at),
});

const restUser: RestUser = {
  user_id: "123",
  first_name: "Taro",
  last_name: "Yamada",
  email_address: "taro@example.com",
  created_at: "2024-01-15T09:00:00Z",
};

const domainUser = userMapper.map(restUser);
console.log(domainUser.name);      // "Taro Yamada"
console.log(domainUser.email);     // "taro@example.com"
console.log(domainUser.createdAt instanceof Date); // true
```
</details>

---

## 9. FAQ

### Q1: Adapter はレガシーコード以外でも使いますか？

はい。Adapter は以下の場面で頻繁に使われます:
- **外部 API**: REST/GraphQL/gRPC の各APIクライアントの統一
- **サードパーティライブラリ**: ログ、決済、通知、ストレージ等のベンダー統一
- **異なるチーム間のモジュール統合**: 内部API のインタフェース不一致の解消
- **テスト**: 外部依存をモック可能なインタフェースに変換
- **データ変換**: DTO/Entity/ViewModel 間のマッピング

### Q2: TypeScript でアダプタを書くとき、クラスと関数のどちらが良いですか？

| 条件 | 推奨 |
|------|------|
| 状態管理が不要 | **関数**（高階関数、ラッパー） |
| 複数メソッドの変換 | **クラス** |
| ライフサイクル管理が必要 | **クラス** |
| DI コンテナで管理 | **クラス** |
| 単純な型変換 | **関数**（`toXxx()` 関数） |

### Q3: Adapter が多数になった場合の管理方法は？

1. **ディレクトリ構成**: `adapters/` ディレクトリに集約
2. **命名規則**: `XxxAdapter` で統一
3. **Factory パターン**: 適切な Adapter を自動選択
4. **DI コンテナ**: インタフェースに対して Adapter を登録
5. **テスト**: 各 Adapter の変換を単体テストで検証

```
src/
  adapters/
    payment/
      stripe-adapter.ts
      paypal-adapter.ts
      square-adapter.ts
    notification/
      email-adapter.ts
      slack-adapter.ts
      sms-adapter.ts
    storage/
      s3-adapter.ts
      gcs-adapter.ts
```

### Q4: Adapter パターンと依存性逆転の原則（DIP）の関係は？

Adapter パターンは DIP の実践そのものです。

```
DIP なし（高レベルモジュールが低レベルモジュールに依存）:
OrderService ──直接依存──> StripeSDK

DIP あり（両方が抽象に依存）:
OrderService ──依存──> PaymentGateway(interface)
                           △
                           |  implements
                     StripeAdapter ──委譲──> StripeSDK
```

高レベルモジュール（OrderService）は抽象（PaymentGateway）にのみ依存し、具象実装（StripeSDK）の詳細を知りません。

### Q5: Adapter と Bridge パターンの違いは？

| | Adapter | Bridge |
|--|--|--|
| 目的 | 既存のインタフェースを**事後的に**変換 | 抽象と実装を**事前に**分離 |
| タイミング | 既存コードに対して適用 | 設計段階で適用 |
| 変更対象 | Adaptee は変更しない | 実装側を自由に変更 |
| 関係 | 1:1（1つのAdapteeに1つのAdapter） | 1:N（1つの抽象に複数の実装） |

### Q6: マイクロサービス間の通信で Adapter はどう使いますか？

マイクロサービスでは各サービスが独自のデータフォーマットを持つことが多く、Adapter は Anti-Corruption Layer（腐敗防止層）として機能します:

```
Service A                    ACL                    Service B
┌──────────┐    REST     ┌──────────────┐    gRPC   ┌──────────┐
│          │ ──────────> │  Adapter     │ ────────> │          │
│ Order    │             │ (format変換) │           │ Inventory│
│ Service  │ <────────── │ (protocol変換)│ <──────── │ Service  │
└──────────┘             └──────────────┘           └──────────┘
  JSON format              変換レイヤー               Protobuf format
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| **目的** | 互換性のないインタフェースを変換して統合する |
| **オブジェクトアダプタ** | 委譲ベース（has-a）、**推奨** |
| **クラスアダプタ** | 継承ベース（is-a）、非推奨 |
| **適用場面** | 外部ライブラリ、レガシー統合、マルチプロバイダ、テスト |
| **責務** | **変換のみ** — ビジネスロジックは入れない |
| **関連パターン** | Factory（Adapter選択）、Strategy（動的切替）、DIP（依存性逆転） |
| **注意点** | 不要な間接層は避ける、変換は確実にテストする |

---

## 次に読むべきガイド

- [Decorator パターン](./01-decorator.md) — 動的な機能追加（Adapter と構造が似ているが目的が異なる）
- [Facade パターン](./02-facade.md) — 複雑なサブシステムの単純化
- [Proxy パターン](./03-proxy.md) — アクセス制御（Adapter と構造が似ている）
- [Strategy パターン](../02-behavioral/01-strategy.md) — アルゴリズムの交換（Adapter と組み合わせて使う）
- [Factory パターン](../00-creational/01-factory.md) — 適切な Adapter の選択に使う
- [Bridge パターン](./04-bridge.md) — 抽象と実装の分離（Adapter と目的が異なる）

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media.
3. Martin, R. C. (2017). *Clean Architecture*. Prentice Hall. — Anti-Corruption Layer
4. Refactoring.Guru — Adapter. https://refactoring.guru/design-patterns/adapter
5. Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley. — Data Mapper
6. Microsoft — Strangler Fig Pattern. https://learn.microsoft.com/en-us/azure/architecture/patterns/strangler-fig
