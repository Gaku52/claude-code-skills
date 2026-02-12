# Factory Method / Abstract Factory パターン

> オブジェクト生成をサブクラスや専用ファクトリに **委譲** し、生成ロジックを利用側から分離する生成パターン。

---

## この章で学ぶこと

1. Factory Method と Abstract Factory の構造的な違い、および Simple Factory との使い分け（WHY）
2. 生成ロジックの抽象化がもたらす柔軟性・テスト容易性・OCP 準拠の仕組み
3. Registry パターンやDIとの組み合わせなど、実務での適用場面と過剰設計を避けるための判断基準
4. 各言語（TypeScript / Python / Java / Go）での具体的な実装方法
5. Factory パターンに伴うトレードオフとアンチパターンの回避方法

---

## 前提知識

このガイドを理解するために、以下の知識を事前に習得しておくことを推奨します。

| トピック | 必要レベル | 参照リンク |
|---------|-----------|-----------|
| オブジェクト指向の基礎（継承、ポリモーフィズム、インタフェース） | 必須 | [OOP基礎](../../../02-programming/oop-guide/docs/) |
| SOLID原則（特にOCP、DIP） | 推奨 | [SOLID原則](../../../clean-code-principles/docs/00-principles/01-solid.md) |
| Singleton パターン | 推奨 | [Singleton](./00-singleton.md) |
| ジェネリクス / 型パラメータ | あると望ましい | TypeScript / Java のジェネリクス |
| DI（依存性注入）の概念 | あると望ましい | [DIP](../../../clean-code-principles/docs/00-principles/01-solid.md) |

---

## 1. Factory パターンの本質 -- なぜ生成を分離するのか

### 1.1 解決する問題

ソフトウェア開発で最も頻繁に直面する問題の1つが「**どのクラスをインスタンス化するかを、利用側が知りすぎている**」ことである。

```typescript
// 問題のあるコード: 利用側が具象クラスを直接知っている
class OrderService {
  processPayment(order: Order): void {
    let processor;

    // 利用側が「どのクラスを生成するか」を知っている
    if (order.paymentMethod === "credit") {
      processor = new CreditCardProcessor(order.cardNumber, order.cvv);
    } else if (order.paymentMethod === "paypal") {
      processor = new PayPalProcessor(order.email);
    } else if (order.paymentMethod === "bank") {
      processor = new BankTransferProcessor(order.bankAccount);
    }
    // 新しい支払い方法を追加するたびに、ここを変更する必要がある → OCP 違反

    processor.charge(order.amount);
  }
}
```

**WHY -- なぜこれが問題なのか?**

```
1. OCP（開放閉鎖原則）違反
   新しい支払い方法の追加 → OrderService の変更が必要
   → テストの再実行、レビュー、デプロイが必要

2. SRP（単一責任原則）違反
   OrderService は「注文処理」と「支払いプロセッサの選択」の2つの責任を持つ

3. テスト困難
   特定の支払いプロセッサをモックするために、
   OrderService 内の条件分岐を意識する必要がある

4. 重複コード
   別のサービスでも同じ条件分岐が発生する可能性がある
```

Factory パターンはこの問題を「**生成の責任を専用のオブジェクトに委譲する**」ことで解決する。

### 1.2 Factory パターンの3つのバリエーション

```
┌─────────────────────────────────────────────────────────────┐
│                Factory パターンの分類                         │
├─────────────┬───────────────────────────────────────────────┤
│             │                                               │
│  Simple     │  関数やクラスメソッドで条件分岐し生成           │
│  Factory    │  GoF パターンではないが実務で最も多用           │
│             │  例: createNotification("email")               │
│             │                                               │
├─────────────┼───────────────────────────────────────────────┤
│             │                                               │
│  Factory    │  サブクラスが生成するプロダクトの型を決定       │
│  Method     │  GoF パターン。テンプレートメソッドの生成版     │
│             │  例: abstract createNotification(): Notification│
│             │                                               │
├─────────────┼───────────────────────────────────────────────┤
│             │                                               │
│  Abstract   │  関連するプロダクト群をまとめて生成             │
│  Factory    │  GoF パターン。テーマ/プラットフォーム切替      │
│             │  例: createButton() + createInput()             │
│             │                                               │
└─────────────┴───────────────────────────────────────────────┘
```

---

## 2. Factory Method の構造

### 2.1 UML クラス図

```
+-------------------+           +-------------------+
|    Creator        |           |    Product        |
|   (abstract)      |           |   (interface)     |
+-------------------+           +-------------------+
| + factoryMethod() |---------->| + use()           |
|   : Product       |  creates  +-------------------+
| + operation()     |                   ^
+-------------------+                   |
        ^                               |
        |                               |
+-------------------+           +-------------------+
| ConcreteCreatorA  |           | ConcreteProductA  |
+-------------------+           +-------------------+
| + factoryMethod() |---------->| + use()           |
|   : ProductA      |  creates  +-------------------+
+-------------------+
        ^
        |
+-------------------+           +-------------------+
| ConcreteCreatorB  |           | ConcreteProductB  |
+-------------------+           +-------------------+
| + factoryMethod() |---------->| + use()           |
|   : ProductB      |  creates  +-------------------+
+-------------------+
```

### 2.2 シーケンス図

```
Client         ConcreteCreator         Product
  |                  |                     |
  |  operation()     |                     |
  |----------------->|                     |
  |                  |                     |
  |                  | factoryMethod()     |
  |                  |--+                  |
  |                  |  | creates          |
  |                  |  v                  |
  |                  | new ConcreteProduct |
  |                  |-------------------->|
  |                  |                     |
  |                  | product.use()       |
  |                  |-------------------->|
  |                  |                     |
  |  <--- result ----|                     |
  |                  |                     |
```

### 2.3 Factory Method の内部動作

```
Factory Method の核心:
「何を生成するか」は子クラスが決定し、
「いつ・どのように使うか」は親クラスが決定する。

Creator (親クラス):
┌──────────────────────────────────┐
│ operation() {                    │
│   const product = this.factory() │ ← 「何を」は子クラスに委譲
│   product.prepare()              │ ← 「どのように使うか」は自分で定義
│   product.use()                  │
│   return product.result()        │
│ }                                │
│                                  │
│ abstract factoryMethod(): Product│ ← 抽象メソッド
└──────────────────────────────────┘
         ^                ^
         |                |
┌────────────────┐ ┌────────────────┐
│ CreatorA       │ │ CreatorB       │
│ factoryMethod()│ │ factoryMethod()│
│ → ProductA     │ │ → ProductB     │
└────────────────┘ └────────────────┘
```

---

## 3. Abstract Factory の構造

### 3.1 UML クラス図

```
+---------------------------+       +-------------+  +-------------+
| AbstractFactory           |       | ProductA    |  | ProductB    |
|---------------------------|       | (interface) |  | (interface) |
| + createProductA(): ProdA |       +-------------+  +-------------+
| + createProductB(): ProdB |             ^                ^
+---------------------------+             |                |
        ^           ^              +-------------+  +-------------+
        |           |              |ConcreteA1   |  |ConcreteB1   |
+-------------+ +-------------+   +-------------+  +-------------+
| Factory1    | | Factory2    |         ^                ^
|createA→ A1  | |createA→ A2  |         |                |
|createB→ B1  | |createB→ B2  |   +-------------+  +-------------+
+-------------+ +-------------+   |ConcreteA2   |  |ConcreteB2   |
                                  +-------------+  +-------------+
```

### 3.2 Abstract Factory と Factory Method の関係

```
Abstract Factory は複数の Factory Method の集合体と見なせる:

AbstractFactory
├── createProductA()  ← Factory Method 1
├── createProductB()  ← Factory Method 2
└── createProductC()  ← Factory Method 3

各メソッドが「1つのプロダクトの生成」を担い、
ファクトリ全体が「関連するプロダクト群の整合性」を保証する。

例: Material Design ファクトリ
├── createButton()  → MaterialButton
├── createInput()   → MaterialInput
└── createDialog()  → MaterialDialog
  ↑ 全てが Material Design のスタイルで統一される
```

---

## 4. コード例

### コード例 1: Factory Method（TypeScript）

```typescript
// Product インタフェース
interface Notification {
  send(message: string): void;
  getType(): string;
}

// Concrete Products
class EmailNotification implements Notification {
  constructor(private recipient: string) {}

  send(message: string): void {
    console.log(`[Email → ${this.recipient}] ${message}`);
  }

  getType(): string {
    return "email";
  }
}

class SlackNotification implements Notification {
  constructor(private channel: string) {}

  send(message: string): void {
    console.log(`[Slack #${this.channel}] ${message}`);
  }

  getType(): string {
    return "slack";
  }
}

class SmsNotification implements Notification {
  constructor(private phoneNumber: string) {}

  send(message: string): void {
    console.log(`[SMS → ${this.phoneNumber}] ${message}`);
  }

  getType(): string {
    return "sms";
  }
}

// Creator（抽象クラス）
abstract class NotificationService {
  // Factory Method: サブクラスが具体的な Notification を決定
  abstract createNotification(): Notification;

  // Template Method: 通知の送信フロー
  notify(message: string): void {
    const notification = this.createNotification();
    console.log(`Sending via ${notification.getType()}...`);
    notification.send(message);
    console.log("Notification sent successfully.");
  }
}

// Concrete Creators
class EmailService extends NotificationService {
  constructor(private recipient: string) {
    super();
  }

  createNotification(): Notification {
    return new EmailNotification(this.recipient);
  }
}

class SlackService extends NotificationService {
  constructor(private channel: string) {
    super();
  }

  createNotification(): Notification {
    return new SlackNotification(this.channel);
  }
}

// 使用: Creator の型で扱えるため、具体的な通知手段に依存しない
function sendDeployNotification(service: NotificationService): void {
  service.notify("デプロイ完了: v2.1.0");
}

sendDeployNotification(new SlackService("deployments"));
// Sending via slack...
// [Slack #deployments] デプロイ完了: v2.1.0
// Notification sent successfully.

sendDeployNotification(new EmailService("admin@example.com"));
// Sending via email...
// [Email → admin@example.com] デプロイ完了: v2.1.0
// Notification sent successfully.
```

### コード例 2: Simple Factory（関数ベース）

実務では GoF の Factory Method より、Simple Factory の方が圧倒的に多用される。

```typescript
type NotificationType = "email" | "slack" | "sms";

interface NotificationConfig {
  type: NotificationType;
  recipient?: string;
  channel?: string;
  phoneNumber?: string;
}

// Simple Factory 関数
function createNotification(config: NotificationConfig): Notification {
  switch (config.type) {
    case "email":
      if (!config.recipient) throw new Error("Email requires recipient");
      return new EmailNotification(config.recipient);
    case "slack":
      if (!config.channel) throw new Error("Slack requires channel");
      return new SlackNotification(config.channel);
    case "sms":
      if (!config.phoneNumber) throw new Error("SMS requires phoneNumber");
      return new SmsNotification(config.phoneNumber);
    default:
      // TypeScript の exhaustive check
      const _exhaustive: never = config.type;
      throw new Error(`Unknown notification type: ${_exhaustive}`);
  }
}

// 使用
const emailNotif = createNotification({
  type: "email",
  recipient: "user@example.com"
});
emailNotif.send("Hello!");

const slackNotif = createNotification({
  type: "slack",
  channel: "general"
});
slackNotif.send("Hello team!");
```

**WHY -- Simple Factory と Factory Method の使い分け:**

```
Simple Factory を選ぶ場合:
- 条件分岐が単純（型名で分岐するだけ）
- テンプレートメソッドパターンが不要
- 生成ロジックを1箇所に集約したい
- 関数型アプローチが好ましい

Factory Method を選ぶ場合:
- 生成するプロダクトに加えて、使用方法もカスタマイズしたい
- テンプレートメソッドとの組み合わせが有効
- フレームワーク設計で、拡張ポイントを提供したい
- サブクラス化による段階的な機能追加が必要
```

### コード例 3: Abstract Factory（TypeScript）

```typescript
// 抽象プロダクト
interface Button {
  render(): string;
  onClick(handler: () => void): void;
}

interface Input {
  render(): string;
  getValue(): string;
  setValue(value: string): void;
}

interface Dialog {
  render(): string;
  show(): void;
  close(): void;
}

// 抽象ファクトリ
interface UIFactory {
  createButton(label: string): Button;
  createInput(placeholder: string): Input;
  createDialog(title: string): Dialog;
}

// Concrete: Material Design
class MaterialButton implements Button {
  constructor(private label: string) {}
  render() { return `<md-button>${this.label}</md-button>`; }
  onClick(handler: () => void) { /* Material ripple effect + handler */ }
}

class MaterialInput implements Input {
  private value = "";
  constructor(private placeholder: string) {}
  render() { return `<md-input placeholder="${this.placeholder}" />`; }
  getValue() { return this.value; }
  setValue(v: string) { this.value = v; }
}

class MaterialDialog implements Dialog {
  constructor(private title: string) {}
  render() { return `<md-dialog><h2>${this.title}</h2></md-dialog>`; }
  show() { console.log(`Material Dialog opened: ${this.title}`); }
  close() { console.log(`Material Dialog closed: ${this.title}`); }
}

class MaterialFactory implements UIFactory {
  createButton(label: string) { return new MaterialButton(label); }
  createInput(placeholder: string) { return new MaterialInput(placeholder); }
  createDialog(title: string) { return new MaterialDialog(title); }
}

// Concrete: iOS Style
class IOSButton implements Button {
  constructor(private label: string) {}
  render() { return `<ios-button>${this.label}</ios-button>`; }
  onClick(handler: () => void) { /* iOS haptic feedback + handler */ }
}

class IOSInput implements Input {
  private value = "";
  constructor(private placeholder: string) {}
  render() { return `<ios-input placeholder="${this.placeholder}" />`; }
  getValue() { return this.value; }
  setValue(v: string) { this.value = v; }
}

class IOSDialog implements Dialog {
  constructor(private title: string) {}
  render() { return `<ios-dialog><h2>${this.title}</h2></ios-dialog>`; }
  show() { console.log(`iOS Dialog opened: ${this.title}`); }
  close() { console.log(`iOS Dialog closed: ${this.title}`); }
}

class IOSFactory implements UIFactory {
  createButton(label: string) { return new IOSButton(label); }
  createInput(placeholder: string) { return new IOSInput(placeholder); }
  createDialog(title: string) { return new IOSDialog(title); }
}

// 使用: ファクトリを差し替えるだけで UI 全体のテーマが変わる
function buildLoginForm(factory: UIFactory): string {
  const emailInput = factory.createInput("メールアドレス");
  const passwordInput = factory.createInput("パスワード");
  const submitButton = factory.createButton("ログイン");

  return [
    emailInput.render(),
    passwordInput.render(),
    submitButton.render(),
  ].join("\n");
}

console.log("--- Material Design ---");
console.log(buildLoginForm(new MaterialFactory()));
// <md-input placeholder="メールアドレス" />
// <md-input placeholder="パスワード" />
// <md-button>ログイン</md-button>

console.log("--- iOS Style ---");
console.log(buildLoginForm(new IOSFactory()));
// <ios-input placeholder="メールアドレス" />
// <ios-input placeholder="パスワード" />
// <ios-button>ログイン</ios-button>
```

### コード例 4: Registry パターン（拡張可能 Factory）

OCP（開放閉鎖原則）を完全に満たすFactory。新しい型の追加が既存コードの変更なしに行える。

```typescript
type Creator<T> = (...args: any[]) => T;

class NotificationRegistry {
  private static registry = new Map<string, Creator<Notification>>();

  // 型を登録（各モジュールが自分で登録する）
  static register(type: string, creator: Creator<Notification>): void {
    if (this.registry.has(type)) {
      console.warn(`Overwriting existing creator for: ${type}`);
    }
    this.registry.set(type, creator);
  }

  // 型からインスタンスを生成
  static create(type: string, ...args: any[]): Notification {
    const creator = this.registry.get(type);
    if (!creator) {
      const available = Array.from(this.registry.keys()).join(", ");
      throw new Error(
        `Unknown notification type: "${type}". Available: [${available}]`
      );
    }
    return creator(...args);
  }

  // 登録されている型の一覧
  static getRegisteredTypes(): string[] {
    return Array.from(this.registry.keys());
  }
}

// 各モジュールが自分のプロダクトを登録
// email-notification.ts
NotificationRegistry.register("email",
  (recipient: string) => new EmailNotification(recipient)
);

// slack-notification.ts
NotificationRegistry.register("slack",
  (channel: string) => new SlackNotification(channel)
);

// sms-notification.ts
NotificationRegistry.register("sms",
  (phone: string) => new SmsNotification(phone)
);

// 新しい型の追加: 既存コードの変更は一切不要（OCP 準拠）
// teams-notification.ts
class TeamsNotification implements Notification {
  constructor(private webhook: string) {}
  send(message: string) { console.log(`[Teams] ${message}`); }
  getType() { return "teams"; }
}
NotificationRegistry.register("teams",
  (webhook: string) => new TeamsNotification(webhook)
);

// 使用
const notification = NotificationRegistry.create("teams", "https://webhook.url");
notification.send("Teams notification!");
console.log(NotificationRegistry.getRegisteredTypes());
// ["email", "slack", "sms", "teams"]
```

### コード例 5: Python -- Factory Method + ABC

```python
from abc import ABC, abstractmethod
from typing import Any
import json
import xml.etree.ElementTree as ET

class Serializer(ABC):
    """直列化の抽象クラス"""

    @abstractmethod
    def serialize(self, data: dict) -> str:
        """データを文字列に変換する"""
        ...

    @abstractmethod
    def deserialize(self, raw: str) -> dict:
        """文字列からデータを復元する"""
        ...

    @abstractmethod
    def content_type(self) -> str:
        """MIME タイプを返す"""
        ...

class JsonSerializer(Serializer):
    def serialize(self, data: dict) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2)

    def deserialize(self, raw: str) -> dict:
        return json.loads(raw)

    def content_type(self) -> str:
        return "application/json"

class XmlSerializer(Serializer):
    def serialize(self, data: dict) -> str:
        root = ET.Element("data")
        for key, value in data.items():
            child = ET.SubElement(root, key)
            child.text = str(value)
        return ET.tostring(root, encoding="unicode")

    def deserialize(self, raw: str) -> dict:
        root = ET.fromstring(raw)
        return {child.tag: child.text for child in root}

    def content_type(self) -> str:
        return "application/xml"

class CsvSerializer(Serializer):
    def serialize(self, data: dict) -> str:
        headers = ",".join(data.keys())
        values = ",".join(str(v) for v in data.values())
        return f"{headers}\n{values}"

    def deserialize(self, raw: str) -> dict:
        lines = raw.strip().split("\n")
        headers = lines[0].split(",")
        values = lines[1].split(",")
        return dict(zip(headers, values))

    def content_type(self) -> str:
        return "text/csv"

# Simple Factory
def get_serializer(fmt: str) -> Serializer:
    """フォーマット名から適切な Serializer を返す Factory"""
    factories: dict[str, type[Serializer]] = {
        "json": JsonSerializer,
        "xml": XmlSerializer,
        "csv": CsvSerializer,
    }
    cls = factories.get(fmt)
    if cls is None:
        available = ", ".join(factories.keys())
        raise ValueError(f"Unknown format: '{fmt}'. Available: [{available}]")
    return cls()

# 使用
data = {"name": "太郎", "age": "30", "city": "東京"}

for fmt in ["json", "xml", "csv"]:
    s = get_serializer(fmt)
    serialized = s.serialize(data)
    print(f"\n--- {fmt} ({s.content_type()}) ---")
    print(serialized)
    restored = s.deserialize(serialized)
    print(f"Restored: {restored}")
```

### コード例 6: Java -- Parameterized Factory Method

```java
public interface Shape {
    double area();
    String description();
}

public class Circle implements Shape {
    private final double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public double area() {
        return Math.PI * radius * radius;
    }

    @Override
    public String description() {
        return String.format("Circle(radius=%.2f, area=%.2f)", radius, area());
    }
}

public class Rectangle implements Shape {
    private final double width, height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public double area() {
        return width * height;
    }

    @Override
    public String description() {
        return String.format("Rectangle(%.2f x %.2f, area=%.2f)", width, height, area());
    }
}

// Parameterized Factory Method with enum
public enum ShapeType {
    CIRCLE, RECTANGLE, TRIANGLE
}

public class ShapeFactory {
    public static Shape create(ShapeType type, double... params) {
        return switch (type) {
            case CIRCLE -> new Circle(params[0]);
            case RECTANGLE -> new Rectangle(params[0], params[1]);
            case TRIANGLE -> new Triangle(params[0], params[1]);
        };
    }
}

// 使用
Shape circle = ShapeFactory.create(ShapeType.CIRCLE, 5.0);
Shape rect = ShapeFactory.create(ShapeType.RECTANGLE, 3.0, 4.0);
System.out.println(circle.description()); // Circle(radius=5.00, area=78.54)
System.out.println(rect.description());   // Rectangle(3.00 x 4.00, area=12.00)
```

### コード例 7: Go -- インタフェースベースの Factory

```go
package main

import "fmt"

// Product インタフェース
type Logger interface {
    Log(message string)
    Level() string
}

// Concrete Products
type ConsoleLogger struct{}

func (l *ConsoleLogger) Log(message string) {
    fmt.Printf("[CONSOLE] %s\n", message)
}
func (l *ConsoleLogger) Level() string { return "console" }

type FileLogger struct {
    filename string
}

func (l *FileLogger) Log(message string) {
    fmt.Printf("[FILE:%s] %s\n", l.filename, message)
}
func (l *FileLogger) Level() string { return "file" }

type CloudLogger struct {
    endpoint string
}

func (l *CloudLogger) Log(message string) {
    fmt.Printf("[CLOUD:%s] %s\n", l.endpoint, message)
}
func (l *CloudLogger) Level() string { return "cloud" }

// Factory 関数（Go ではファーストクラス関数を活用）
type LoggerFactory func() Logger

// Registry
var loggerFactories = map[string]LoggerFactory{
    "console": func() Logger { return &ConsoleLogger{} },
    "file":    func() Logger { return &FileLogger{filename: "app.log"} },
    "cloud":   func() Logger { return &CloudLogger{endpoint: "https://log.example.com"} },
}

func CreateLogger(loggerType string) (Logger, error) {
    factory, exists := loggerFactories[loggerType]
    if !exists {
        return nil, fmt.Errorf("unknown logger type: %s", loggerType)
    }
    return factory(), nil
}

// 新しい型の登録
func RegisterLogger(name string, factory LoggerFactory) {
    loggerFactories[name] = factory
}

func main() {
    logger, _ := CreateLogger("console")
    logger.Log("Application started")

    logger2, _ := CreateLogger("cloud")
    logger2.Log("Cloud logging active")
}
```

### コード例 8: TypeScript -- 非同期 Factory

実務では、Factory がDBやAPIから設定を読み込む必要がある場合がある。

```typescript
// 非同期 Factory パターン
interface DataSource {
  connect(): Promise<void>;
  query(sql: string): Promise<any[]>;
  disconnect(): Promise<void>;
}

class PostgresDataSource implements DataSource {
  private pool: any;

  constructor(private connectionString: string) {}

  async connect(): Promise<void> {
    console.log(`Connecting to PostgreSQL: ${this.connectionString}`);
    // this.pool = await createPool(this.connectionString);
  }

  async query(sql: string): Promise<any[]> {
    console.log(`[PG] ${sql}`);
    return [];
  }

  async disconnect(): Promise<void> {
    // await this.pool.end();
  }
}

class MySQLDataSource implements DataSource {
  constructor(private config: { host: string; port: number; database: string }) {}

  async connect(): Promise<void> {
    console.log(`Connecting to MySQL: ${this.config.host}:${this.config.port}`);
  }

  async query(sql: string): Promise<any[]> {
    console.log(`[MySQL] ${sql}`);
    return [];
  }

  async disconnect(): Promise<void> {}
}

// 非同期 Factory: 生成 + 初期化を1ステップで行う
async function createDataSource(
  type: "postgres" | "mysql",
  config: Record<string, any>
): Promise<DataSource> {
  let ds: DataSource;

  switch (type) {
    case "postgres":
      ds = new PostgresDataSource(config.connectionString);
      break;
    case "mysql":
      ds = new MySQLDataSource({
        host: config.host,
        port: config.port,
        database: config.database,
      });
      break;
    default:
      throw new Error(`Unknown data source type: ${type}`);
  }

  // Factory が初期化まで保証する
  await ds.connect();
  return ds;
}

// 使用
async function main() {
  const ds = await createDataSource("postgres", {
    connectionString: "postgresql://localhost:5432/mydb"
  });
  const users = await ds.query("SELECT * FROM users");
  await ds.disconnect();
}
```

### コード例 9: Factory + Strategy の組み合わせ

```typescript
// バリデーション戦略を Factory で生成
interface ValidationStrategy {
  validate(value: string): { valid: boolean; error?: string };
}

class EmailValidator implements ValidationStrategy {
  validate(value: string) {
    const valid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
    return { valid, error: valid ? undefined : "Invalid email format" };
  }
}

class PhoneValidator implements ValidationStrategy {
  validate(value: string) {
    const valid = /^\d{10,11}$/.test(value.replace(/-/g, ""));
    return { valid, error: valid ? undefined : "Invalid phone number" };
  }
}

class UrlValidator implements ValidationStrategy {
  validate(value: string) {
    try {
      new URL(value);
      return { valid: true };
    } catch {
      return { valid: false, error: "Invalid URL" };
    }
  }
}

// Factory
const validatorFactory = new Map<string, () => ValidationStrategy>([
  ["email", () => new EmailValidator()],
  ["phone", () => new PhoneValidator()],
  ["url", () => new UrlValidator()],
]);

function createValidator(type: string): ValidationStrategy {
  const factory = validatorFactory.get(type);
  if (!factory) throw new Error(`Unknown validator: ${type}`);
  return factory();
}

// 使用: 設定駆動で動的にバリデータを選択
const formFields = [
  { name: "email", type: "email", value: "user@example.com" },
  { name: "phone", type: "phone", value: "090-1234-5678" },
  { name: "website", type: "url", value: "https://example.com" },
];

for (const field of formFields) {
  const validator = createValidator(field.type);
  const result = validator.validate(field.value);
  console.log(`${field.name}: ${result.valid ? "OK" : result.error}`);
}
// email: OK
// phone: OK
// website: OK
```

### コード例 10: Factory と DI の統合

```typescript
// DI コンテナで Factory を管理する実務的なパターン
interface IPaymentProcessor {
  charge(amount: number): Promise<{ success: boolean; transactionId: string }>;
  refund(transactionId: string): Promise<boolean>;
}

interface IPaymentFactory {
  create(method: string): IPaymentProcessor;
  getSupportedMethods(): string[];
}

class PaymentFactory implements IPaymentFactory {
  private processors = new Map<string, () => IPaymentProcessor>();

  register(method: string, factory: () => IPaymentProcessor): void {
    this.processors.set(method, factory);
  }

  create(method: string): IPaymentProcessor {
    const factory = this.processors.get(method);
    if (!factory) {
      throw new Error(`Unsupported payment method: ${method}`);
    }
    return factory();
  }

  getSupportedMethods(): string[] {
    return Array.from(this.processors.keys());
  }
}

// DI コンテナとの統合
// container.ts
const container = new Container();

// Factory 自体を Singleton スコープで登録
container.bind<IPaymentFactory>(TYPES.PaymentFactory)
  .toDynamicValue(() => {
    const factory = new PaymentFactory();
    factory.register("credit", () => new CreditCardProcessor());
    factory.register("paypal", () => new PayPalProcessor());
    factory.register("bank", () => new BankTransferProcessor());
    return factory;
  })
  .inSingletonScope();

// 使用側: Factory を注入してもらう
class OrderService {
  constructor(
    @inject(TYPES.PaymentFactory) private paymentFactory: IPaymentFactory
  ) {}

  async processOrder(order: Order): Promise<void> {
    const processor = this.paymentFactory.create(order.paymentMethod);
    const result = await processor.charge(order.amount);
    if (!result.success) {
      throw new Error("Payment failed");
    }
  }
}
```

---

## 5. Factory の選択フロー

```
生成ロジックを分離したい？
│
├── No → 直接 new で十分。Factory は過剰設計
│
└── Yes
    │
    ├── 生成するプロダクトは1種類？
    │   │
    │   ├── Yes
    │   │   │
    │   │   ├── 型で分岐するだけ？
    │   │   │   ├── Yes → Simple Factory（関数/静的メソッド）
    │   │   │   └── No  → Factory Method（サブクラスにオーバーライド）
    │   │   │
    │   │   └── 実行時に型を動的に追加する必要がある？
    │   │       ├── Yes → Registry パターン
    │   │       └── No  → Simple Factory で十分
    │   │
    │   └── No（関連するプロダクト群がある）
    │       │
    │       └── プロダクト群の整合性が重要？
    │           ├── Yes → Abstract Factory
    │           └── No  → 個別の Factory Method で十分
    │
    └── テスト時にプロダクトを差し替えたい？
        ├── Yes → DI + Factory インタフェース
        └── No  → Simple Factory で十分
```

---

## 6. 比較表

### 比較表 1: Factory Method vs Abstract Factory vs Simple Factory

| 観点 | Simple Factory | Factory Method | Abstract Factory |
|------|:---:|:---:|:---:|
| GoF パターン | No | Yes | Yes |
| 意図 | 条件分岐で生成を集約 | 生成をサブクラスに委譲 | 関連プロダクト群の生成 |
| クラス数 | 最小 | 中 | 多い |
| 拡張方法 | switch 文に追加 | Creator サブクラス追加 | Factory + Product 群追加 |
| OCP 準拠 | No（switch 変更要） | Yes | Yes |
| 使用場面 | 単純な分岐 | テンプレートメソッドと併用 | テーマ/プラットフォーム切替 |
| 複雑度 | 低 | 中 | 高 |
| テスト容易性 | 中 | 高 | 高 |

### 比較表 2: Factory vs その他の生成パターン

| パターン | 目的 | 生成の自由度 | 複雑度 | 使用頻度 |
|---------|------|:---:|:---:|:---:|
| Simple Factory | 条件分岐で生成 | 低 | 低 | 非常に高 |
| Factory Method | サブクラスに委譲 | 中 | 中 | 高 |
| Abstract Factory | ファミリー生成 | 高 | 高 | 中 |
| Builder | 段階的構築 | 高 | 中 | 高 |
| Prototype | クローンで生成 | 中 | 低 | 低 |
| Singleton | 唯一インスタンス | N/A | 低 | 高 |

### 比較表 3: Registry パターン vs DI コンテナ

| 観点 | Registry パターン | DI コンテナ |
|------|:---:|:---:|
| 登録方法 | 手動 register() | 設定/アノテーション |
| ライフタイム管理 | なし（毎回生成） | Singleton/Transient 等 |
| 依存解決 | 自前 | 自動 |
| テスト容易性 | 中 | 高 |
| 学習コスト | 低 | 中~高 |
| 導入コスト | 低 | 中 |
| 推奨場面 | プラグイン登録 | アプリ全体の依存管理 |

---

## 7. エッジケースと注意点

### 7.1 型安全性の確保

```typescript
// 問題: string ベースの Factory は型安全でない
const notif = createNotification("emal");  // typo! 実行時エラー

// 解決策 1: Union 型で制限
type NotificationType = "email" | "slack" | "sms";
function createNotification(type: NotificationType): Notification { ... }

// 解決策 2: Enum（Java/TypeScript）
enum NotificationType {
  Email = "email",
  Slack = "slack",
  Sms = "sms",
}

// 解決策 3: Discriminated Union（TypeScript）
type NotificationRequest =
  | { type: "email"; recipient: string }
  | { type: "slack"; channel: string }
  | { type: "sms"; phoneNumber: string };

function createNotification(req: NotificationRequest): Notification {
  switch (req.type) {
    case "email":
      return new EmailNotification(req.recipient);  // 型安全にアクセス
    case "slack":
      return new SlackNotification(req.channel);     // 型安全にアクセス
    case "sms":
      return new SmsNotification(req.phoneNumber);   // 型安全にアクセス
  }
}
```

### 7.2 循環依存の回避

```
// 問題: Factory と Product が相互依存
ProductA → Factory（生成に使う）
Factory → ProductA（生成する）

// 解決策: インタフェースを中間層として導入
ProductA → IFactory（インタフェースに依存）
Factory → IProduct（インタフェースに依存）
ConcreteFactory → ConcreteProduct（具象が具象を参照）

DIP（依存性逆転原則）を適用:
  上位モジュール（利用側）→ インタフェース ← 下位モジュール（実装）
```

### 7.3 Factory のメモリリーク

```typescript
// 問題: Registry に登録されたオブジェクトが GC されない
class HeavyRegistry {
  private static cache = new Map<string, LargeObject>();

  static getOrCreate(key: string): LargeObject {
    if (!this.cache.has(key)) {
      this.cache.set(key, new LargeObject(key)); // 蓄積し続ける
    }
    return this.cache.get(key)!;
  }
}

// 解決策: WeakMap または LRU キャッシュを使用
class SafeRegistry {
  private static cache = new LRUCache<string, LargeObject>({ max: 100 });

  static getOrCreate(key: string): LargeObject {
    let obj = this.cache.get(key);
    if (!obj) {
      obj = new LargeObject(key);
      this.cache.set(key, obj);
    }
    return obj;
  }
}
```

---

## 8. アンチパターン

### アンチパターン 1: 万能 Factory（God Factory）

```typescript
// NG: あらゆる型を1つの Factory で処理
class UniversalFactory {
  create(type: string): any {  // any は危険信号
    if (type === "user") return new User();
    if (type === "order") return new Order();
    if (type === "product") return new Product();
    if (type === "payment") return new Payment();
    if (type === "notification") return new Notification();
    if (type === "report") return new Report();
    // ... 50行の if-else
    throw new Error(`Unknown type: ${type}`);
  }
}
```

**問題**:
- OCP 違反: 新しい型追加のたびにこのクラスを変更する必要がある
- SRP 違反: 無関係なドメインのオブジェクト生成が1クラスに集約
- 型安全性がない: 戻り値が any

```typescript
// OK: ドメインごとに Factory を分割
class UserFactory {
  static create(type: UserType): User { ... }
}

class OrderFactory {
  static create(items: CartItem[]): Order { ... }
}

class NotificationFactory {
  static create(type: NotificationType): Notification { ... }
}
```

### アンチパターン 2: 不要な Abstract Factory

```typescript
// NG: プロダクトが1種類なのに Abstract Factory を使う
interface ShapeFactory {
  createShape(): Shape;  // 1メソッドだけ → Abstract Factory は過剰
}

class CircleFactory implements ShapeFactory {
  createShape(): Shape { return new Circle(); }
}

class RectangleFactory implements ShapeFactory {
  createShape(): Shape { return new Rectangle(); }
}
```

**問題**: Factory Method で十分な場面に過剰な抽象化を持ち込んでいる。

```typescript
// OK: Simple Factory または Factory Method を使う
function createShape(type: "circle" | "rectangle"): Shape {
  switch (type) {
    case "circle": return new Circle();
    case "rectangle": return new Rectangle();
  }
}
```

**YAGNI 原則**: 複数プロダクトが実際に必要になるまで Abstract Factory にしない。

### アンチパターン 3: Factory 内でのビジネスロジック

```typescript
// NG: Factory が生成以外の責任を持つ
class OrderFactory {
  static create(items: CartItem[], coupon?: string): Order {
    const order = new Order(items);

    // ビジネスロジック（Factory の責任ではない）
    if (coupon) {
      const discount = this.validateCoupon(coupon);  // クーポン検証
      order.applyDiscount(discount);                 // 割引適用
    }

    order.calculateTax();      // 税計算
    order.calculateShipping(); // 送料計算
    this.sendAnalytics(order); // 分析データ送信

    return order;
  }
}
```

```typescript
// OK: Factory は生成のみ。ビジネスロジックはドメインサービスに委譲
class OrderFactory {
  static create(items: CartItem[]): Order {
    return new Order(items);  // 生成のみ
  }
}

class OrderService {
  constructor(
    private couponService: CouponService,
    private taxService: TaxService,
    private analytics: AnalyticsService,
  ) {}

  async processOrder(items: CartItem[], coupon?: string): Promise<Order> {
    const order = OrderFactory.create(items);

    if (coupon) {
      const discount = await this.couponService.validate(coupon);
      order.applyDiscount(discount);
    }

    order.tax = this.taxService.calculate(order);
    await this.analytics.track("order_created", order);

    return order;
  }
}
```

---

## 9. トレードオフ分析

### 9.1 Factory 導入の利点と欠点

```
利点                              欠点
+------------------------------+  +------------------------------+
| 生成ロジックの集約            |  | クラス数の増加                |
| OCP 準拠（拡張が容易）        |  | 間接層による複雑化            |
| テスト時のモック差替え容易    |  | 過剰設計のリスク              |
| 利用側の具象クラスへの非依存  |  | 単純な場合は YAGNI            |
| コードの再利用性向上          |  | デバッグ時の追跡が煩雑        |
+------------------------------+  +------------------------------+
```

### 9.2 導入判断のガイドライン

```
Factory を導入すべき場面:
- new の呼び出しが3箇所以上に散らばっている
- 生成ロジックに条件分岐がある
- テストでモック差し替えが必要
- プラグインシステムを構築する
- 設定に基づいて動的にオブジェクトを選択する

Factory を導入すべきでない場面:
- new が1-2箇所でしか呼ばれない
- 生成対象が変わる見込みがない
- 生成に条件分岐がない
- 単純なデータオブジェクト（DTO）の生成
```

---

## 10. 実践演習

### 演習 1: 基礎 -- Simple Factory の実装

**課題**: ログフォーマッタの Simple Factory を実装してください。

**要件**:
- `ILogFormatter` インタフェースを定義（`format(level, message, timestamp)` メソッド）
- JSON / Text / CSV の3つの具象フォーマッタを実装
- `createFormatter(type)` Factory 関数を作成
- 不明な型は適切なエラーメッセージで例外を投げる

```typescript
// === あなたの実装をここに書いてください ===
```

**期待される出力**:

```
const jsonFmt = createFormatter("json");
console.log(jsonFmt.format("info", "Server started", new Date()));
// {"level":"info","message":"Server started","timestamp":"2026-01-15T10:30:00.000Z"}

const textFmt = createFormatter("text");
console.log(textFmt.format("error", "Connection failed", new Date()));
// [2026-01-15T10:30:00.000Z] [ERROR] Connection failed

const csvFmt = createFormatter("csv");
console.log(csvFmt.format("warn", "Memory high", new Date()));
// 2026-01-15T10:30:00.000Z,WARN,Memory high
```

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
interface ILogFormatter {
  format(level: string, message: string, timestamp: Date): string;
}

class JsonLogFormatter implements ILogFormatter {
  format(level: string, message: string, timestamp: Date): string {
    return JSON.stringify({
      level,
      message,
      timestamp: timestamp.toISOString(),
    });
  }
}

class TextLogFormatter implements ILogFormatter {
  format(level: string, message: string, timestamp: Date): string {
    return `[${timestamp.toISOString()}] [${level.toUpperCase()}] ${message}`;
  }
}

class CsvLogFormatter implements ILogFormatter {
  format(level: string, message: string, timestamp: Date): string {
    return `${timestamp.toISOString()},${level.toUpperCase()},${message}`;
  }
}

type FormatterType = "json" | "text" | "csv";

function createFormatter(type: FormatterType): ILogFormatter {
  const factories: Record<FormatterType, () => ILogFormatter> = {
    json: () => new JsonLogFormatter(),
    text: () => new TextLogFormatter(),
    csv: () => new CsvLogFormatter(),
  };

  const factory = factories[type];
  if (!factory) {
    throw new Error(`Unknown formatter type: "${type}". Available: ${Object.keys(factories).join(", ")}`);
  }
  return factory();
}
```

</details>

### 演習 2: 応用 -- Registry パターンの拡張

**課題**: プラグインシステムとして機能する Registry パターンの Factory を実装してください。

**要件**:
- `PluginRegistry` クラスを実装
- プラグインの登録（register）、生成（create）、一覧（list）、登録解除（unregister）をサポート
- 重複登録時は警告を出す
- 型安全性を確保（ジェネリクス使用）
- 登録時にバリデーション関数を指定可能

```typescript
// === あなたの実装をここに書いてください ===
```

**期待される出力**:

```
const registry = new PluginRegistry<IPlugin>();
registry.register("analytics", () => new AnalyticsPlugin());
registry.register("auth", () => new AuthPlugin());

const plugin = registry.create("analytics");
plugin.initialize();

console.log(registry.list()); // ["analytics", "auth"]

registry.unregister("auth");
console.log(registry.list()); // ["analytics"]
```

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
interface IPlugin {
  name: string;
  initialize(): void;
  destroy(): void;
}

type PluginFactory<T> = () => T;
type PluginValidator<T> = (plugin: T) => boolean;

class PluginRegistry<T extends IPlugin> {
  private factories = new Map<string, PluginFactory<T>>();
  private validators = new Map<string, PluginValidator<T>>();

  register(
    name: string,
    factory: PluginFactory<T>,
    validator?: PluginValidator<T>
  ): void {
    if (this.factories.has(name)) {
      console.warn(`Warning: Overwriting existing plugin: "${name}"`);
    }
    this.factories.set(name, factory);
    if (validator) {
      this.validators.set(name, validator);
    }
  }

  create(name: string): T {
    const factory = this.factories.get(name);
    if (!factory) {
      const available = this.list().join(", ");
      throw new Error(
        `Plugin "${name}" not found. Available: [${available}]`
      );
    }

    const plugin = factory();

    // バリデーション実行
    const validator = this.validators.get(name);
    if (validator && !validator(plugin)) {
      throw new Error(`Plugin "${name}" failed validation`);
    }

    return plugin;
  }

  list(): string[] {
    return Array.from(this.factories.keys()).sort();
  }

  has(name: string): boolean {
    return this.factories.has(name);
  }

  unregister(name: string): boolean {
    this.validators.delete(name);
    return this.factories.delete(name);
  }

  clear(): void {
    this.factories.clear();
    this.validators.clear();
  }
}
```

</details>

### 演習 3: 発展 -- Abstract Factory によるクロスプラットフォーム UI

**課題**: Web / Mobile / Desktop の3プラットフォームに対応する Abstract Factory を設計してください。

**要件**:
- 各プラットフォームで Button、TextField、Checkbox の3つのUIコンポーネントを生成
- 各コンポーネントは `render(): string` メソッドを持つ
- プラットフォームの切り替えは Factory の差し替えのみで行う
- テスト用の MockFactory も作成
- Factory の生成自体を別の Factory（Factory of Factories）で管理

```typescript
// === あなたの実装をここに書いてください ===
```

**期待される出力**:

```
// Web Platform
const webUI = buildForm(new WebUIFactory());
console.log(webUI);
// <input type="text" class="web-input" placeholder="Name" />
// <input type="checkbox" class="web-checkbox" />
// <button class="web-btn">Submit</button>

// Mobile Platform
const mobileUI = buildForm(new MobileUIFactory());
console.log(mobileUI);
// <TextInput style="mobile" placeholder="Name" />
// <Switch style="mobile" />
// <TouchableOpacity style="mobile">Submit</TouchableOpacity>

// Platform auto-detection
const factory = UIFactoryProvider.getFactory("web");
const autoUI = buildForm(factory);
```

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
// 抽象プロダクト
interface IButton {
  render(): string;
}
interface ITextField {
  render(): string;
}
interface ICheckbox {
  render(): string;
}

// 抽象ファクトリ
interface IUIFactory {
  createButton(label: string): IButton;
  createTextField(placeholder: string): ITextField;
  createCheckbox(label: string): ICheckbox;
}

// Web 実装
class WebButton implements IButton {
  constructor(private label: string) {}
  render() { return `<button class="web-btn">${this.label}</button>`; }
}
class WebTextField implements ITextField {
  constructor(private placeholder: string) {}
  render() { return `<input type="text" class="web-input" placeholder="${this.placeholder}" />`; }
}
class WebCheckbox implements ICheckbox {
  constructor(private label: string) {}
  render() { return `<input type="checkbox" class="web-checkbox" /> ${this.label}`; }
}
class WebUIFactory implements IUIFactory {
  createButton(label: string) { return new WebButton(label); }
  createTextField(ph: string) { return new WebTextField(ph); }
  createCheckbox(label: string) { return new WebCheckbox(label); }
}

// Mobile 実装
class MobileButton implements IButton {
  constructor(private label: string) {}
  render() { return `<TouchableOpacity style="mobile">${this.label}</TouchableOpacity>`; }
}
class MobileTextField implements ITextField {
  constructor(private placeholder: string) {}
  render() { return `<TextInput style="mobile" placeholder="${this.placeholder}" />`; }
}
class MobileCheckbox implements ICheckbox {
  constructor(private label: string) {}
  render() { return `<Switch style="mobile" /> ${this.label}`; }
}
class MobileUIFactory implements IUIFactory {
  createButton(label: string) { return new MobileButton(label); }
  createTextField(ph: string) { return new MobileTextField(ph); }
  createCheckbox(label: string) { return new MobileCheckbox(label); }
}

// Desktop 実装
class DesktopButton implements IButton {
  constructor(private label: string) {}
  render() { return `<QButton text="${this.label}" />`; }
}
class DesktopTextField implements ITextField {
  constructor(private placeholder: string) {}
  render() { return `<QLineEdit placeholder="${this.placeholder}" />`; }
}
class DesktopCheckbox implements ICheckbox {
  constructor(private label: string) {}
  render() { return `<QCheckBox text="${this.label}" />`; }
}
class DesktopUIFactory implements IUIFactory {
  createButton(label: string) { return new DesktopButton(label); }
  createTextField(ph: string) { return new DesktopTextField(ph); }
  createCheckbox(label: string) { return new DesktopCheckbox(label); }
}

// Factory of Factories
type Platform = "web" | "mobile" | "desktop";

class UIFactoryProvider {
  private static factories: Record<Platform, () => IUIFactory> = {
    web: () => new WebUIFactory(),
    mobile: () => new MobileUIFactory(),
    desktop: () => new DesktopUIFactory(),
  };

  static getFactory(platform: Platform): IUIFactory {
    const factory = this.factories[platform];
    if (!factory) throw new Error(`Unknown platform: ${platform}`);
    return factory();
  }
}

// 利用コード（プラットフォーム非依存）
function buildForm(factory: IUIFactory): string {
  const nameField = factory.createTextField("Name");
  const agreeBox = factory.createCheckbox("I agree");
  const submitBtn = factory.createButton("Submit");

  return [nameField.render(), agreeBox.render(), submitBtn.render()].join("\n");
}
```

</details>

---

## 11. FAQ

### Q1: Simple Factory と Factory Method の違いは？

Simple Factory は単なる関数やクラスメソッドで条件分岐し生成します。Factory Method はサブクラス化により生成をオーバーライドする GoF パターンです。多くの実務では Simple Factory で十分です。

| 観点 | Simple Factory | Factory Method |
|------|:---:|:---:|
| パターン分類 | イディオム | GoF パターン |
| 拡張方法 | switch 文の変更 | サブクラスの追加 |
| OCP 準拠 | No | Yes |
| 適用場面 | 単純な分岐 | フレームワーク拡張 |

### Q2: Factory を使うべき判断基準は？

以下の3つの条件のいずれかに該当すれば Factory の導入を検討します:

1. **`new` の呼び出し箇所が3箇所以上に散らばっている**: 生成ロジックの変更が複数箇所に影響する
2. **生成ロジックに条件分岐がある**: どの型をインスタンス化するかの判断が必要
3. **テストでモック差し替えが必要**: 具象クラスへの直接依存を解消する必要がある

逆に、`new` が1-2箇所でしか使われず、条件分岐もなく、モックの必要もなければ、Factory は過剰設計です。

### Q3: DI コンテナがあれば Factory は不要ですか？

DI コンテナは**起動時**に依存を解決しますが、**実行時**に動的に型を切り替える場合は Factory が必要です。両者は補完関係にあります。

```
DI コンテナ: 起動時に決定（設定ベース）
  container.bind<Logger>().to(ConsoleLogger)
  → アプリケーション起動時に1度だけ解決

Factory: 実行時に決定（データベース）
  factory.create(user.preferredNotificationType)
  → リクエストごとに異なる型を生成

ベストプラクティス: DI コンテナで Factory 自体を管理する
  container.bind<INotificationFactory>()
    .to(NotificationFactory)
    .inSingletonScope();
```

### Q4: Factory Method と Strategy パターンはどう使い分けるべきですか？

```
Factory Method:
- 「何を生成するか」が関心事
- サブクラスがプロダクトの型を決定
- 生成されたオブジェクトをテンプレートメソッド内で使用

Strategy:
- 「どう振る舞うか」が関心事
- 実行時にアルゴリズムを切り替え
- コンポジション（委譲）で実現

併用:
- Factory で Strategy オブジェクトを生成するのが一般的
- Strategy が必要な場面では、まず Strategy を設計し、
  その生成に Factory を使う
```

### Q5: Abstract Factory でプロダクトを追加する場合の影響は？

Abstract Factory に新しいプロダクト（createNewProduct()）を追加すると、全ての具象ファクトリに影響します（インタフェースの変更）。これが Abstract Factory の最大の弱点です。

```
解決策:
1. インタフェース分離原則（ISP）を適用し、Factory を分割
2. Default 実装を持つ抽象クラスを使用
3. Generic Factory メソッド: create<T>(type: Class<T>): T
```

---

## 12. まとめ

| 項目 | ポイント |
|------|---------|
| Factory Method | 1プロダクトの生成をサブクラスに委譲。テンプレートメソッドと相性が良い |
| Abstract Factory | 関連プロダクト群をまとめて生成。テーマ/プラットフォーム切替に最適 |
| Simple Factory | 最も軽量。関数1つで実現。実務で最多 |
| Registry | OCP 準拠の拡張可能 Factory。プラグインシステムに最適 |
| 判断基準 | new の散在 / 条件分岐 / モック必要性 のいずれかで導入検討 |
| 過剰設計の回避 | YAGNI 原則: 現在の要件に合った最小の Factory を選択 |
| DI との関係 | DI コンテナと Factory は補完関係。DI で Factory を管理するのが最善 |
| 最大の注意点 | Factory 内にビジネスロジックを入れない |

---

## 次に読むべきガイド

- [Builder パターン](./02-builder.md) -- 複雑なオブジェクトの段階的構築。Factory が「何を」、Builder が「どのように」を担当
- [Prototype パターン](./03-prototype.md) -- クローンによる生成。Factory の代替手法
- [Singleton パターン](./00-singleton.md) -- Factory Registry の Singleton 管理
- [Strategy パターン](../02-behavioral/01-strategy.md) -- アルゴリズムの交換。Factory + Strategy の併用
- [Adapter パターン](../01-structural/00-adapter.md) -- 既存クラスの適合。Factory でAdapter を生成
- [SOLID 原則](../../../clean-code-principles/docs/00-principles/01-solid.md) -- OCP、DIP の詳細

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley. -- Factory Method / Abstract Factory の原典
2. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media. Chapter 4: The Factory Pattern.
3. Martin, R.C. (2017). *Clean Architecture*. Prentice Hall. -- SOLID原則と Factory の関係
4. Refactoring.Guru -- Factory Method. https://refactoring.guru/design-patterns/factory-method
5. Refactoring.Guru -- Abstract Factory. https://refactoring.guru/design-patterns/abstract-factory
6. Fowler, M. -- Plugin Pattern. https://martinfowler.com/eaaCatalog/plugin.html
