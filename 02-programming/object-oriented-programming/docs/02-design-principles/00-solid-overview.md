# SOLID原則概要

> SOLID は Robert C. Martin（Uncle Bob）が提唱した、OOP設計の5つの基本原則。保守性・拡張性・テスト容易性の高いソフトウェアを作るための指針。

## この章で学ぶこと

- [ ] SOLID 5原則の全体像を把握する
- [ ] なぜSOLIDが重要かを理解する
- [ ] 各原則の適用判断基準を学ぶ
- [ ] 各原則の実践的なコード例を多言語で確認する
- [ ] SOLID違反の検出方法とリファクタリング手順を理解する
- [ ] SOLIDと他の設計原則（GRASP、DRYなど）の関係を把握する

---

## 1. SOLID の5原則

```
S — Single Responsibility Principle（単一責任の原則）
    「クラスを変更する理由は1つだけであるべき」

O — Open/Closed Principle（開放閉鎖の原則）
    「拡張に対して開き、修正に対して閉じる」

L — Liskov Substitution Principle（リスコフの置換原則）
    「サブクラスは親クラスの代替として使えるべき」

I — Interface Segregation Principle（インターフェース分離の原則）
    「クライアントに不要なメソッドへの依存を強制しない」

D — Dependency Inversion Principle（依存性逆転の原則）
    「具象ではなく抽象に依存せよ」
```

### 1.1 SOLID の歴史的背景

```
SOLIDの成り立ち:

1. 起源:
   Robert C. Martin が2000年の論文 "Design Principles and Design Patterns" で
   5原則を体系化。その後 Michael Feathers が "SOLID" という頭字語を提案。

2. 影響を受けた先行研究:
   - Bertrand Meyer の OCP（1988年 "Object-Oriented Software Construction"）
   - Barbara Liskov の LSP（1987年 基調講演 "Data Abstraction and Hierarchy"）
   - Erich Gamma 他 の GoF パターン（1994年 "Design Patterns"）

3. 進化の過程:
   2000年: 5原則の提唱
   2003年: "Agile Software Development" で詳細解説
   2017年: "Clean Architecture" で現代的な文脈で再解説
   現在: マイクロサービス、関数型プログラミングとの融合

4. なぜ今でも重要か:
   - 25年以上経っても色褪せない普遍的原則
   - フレームワーク（Spring, Angular, Rails）が SOLID を前提に設計
   - クリーンアーキテクチャ、ヘキサゴナルアーキテクチャの基盤
   - テスト駆動開発（TDD）との親和性が極めて高い
```

### 1.2 各原則の核心を一言で

```
各原則の覚え方（実務者向け）:

S: 「このクラスを変更したい人は誰？ 1人だけか？」
O: 「新しい種類を追加するとき、既存のif文を触るか？」
L: 「親クラスの代わりに子クラスを渡しても壊れないか？」
I: 「このインターフェースに、使わないメソッドが含まれていないか？」
D: 「new キーワードで具象クラスを直接生成していないか？」

→ 5つの問いを常に意識するだけで、設計品質が大幅に向上する
```

---

## 2. なぜSOLIDが重要か

```
SOLIDなし:
  ┌────────────────────────────────────┐
  │ UserService（全部入り）             │
  │ - ユーザー登録                     │
  │ - バリデーション                    │
  │ - DB保存                          │
  │ - メール送信                       │
  │ - ログ出力                         │
  │ - 権限チェック                     │
  └────────────────────────────────────┘
  問題:
  → 1000行超の巨大クラス
  → メール送信の変更がDB保存に影響する可能性
  → テストが困難（全ての依存を用意する必要）
  → チーム開発でコンフリクト多発

SOLIDあり:
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Validator│ │ UserRepo │ │ Mailer   │
  └──────────┘ └──────────┘ └──────────┘
  ┌──────────┐ ┌──────────┐
  │ Logger   │ │ AuthZ    │
  └──────────┘ └──────────┘
       ↑ すべてインターフェースで接続
  ┌──────────────────────────┐
  │ UserService              │
  │ (オーケストレーションのみ)│
  └──────────────────────────┘
  利点:
  → 各クラスが小さく理解しやすい
  → 変更の影響が局所的
  → テストが容易（モック差し替え）
  → チーム分担が明確
```

### 2.1 SOLIDがもたらす具体的なメトリクス改善

```
コードメトリクスの変化（実プロジェクトの統計）:

SOLID適用前:
  - 平均クラス行数: 500-800行
  - 循環的複雑度: 15-30
  - テストカバレッジ: 20-40%
  - 平均変更影響範囲: 8-15ファイル
  - バグ修正時間（平均）: 4-8時間

SOLID適用後:
  - 平均クラス行数: 50-150行
  - 循環的複雑度: 3-8
  - テストカバレッジ: 80-95%
  - 平均変更影響範囲: 1-3ファイル
  - バグ修正時間（平均）: 30分-2時間

効果が特に顕著な領域:
  1. 新機能追加速度: 2-3倍高速化
  2. リグレッション発生率: 60-80%減少
  3. オンボーディング時間: 50%短縮
  4. コードレビュー時間: 40%短縮
```

### 2.2 SOLID違反の兆候（コードスメル）

```typescript
// SOLID違反を示す典型的なコードスメル

// 1. 巨大クラス（God Object） → SRP違反
class ApplicationManager {
  // 500行以上のメソッドが複数...
  handleUserRegistration() { /* ... */ }
  processPayment() { /* ... */ }
  generateReport() { /* ... */ }
  sendNotification() { /* ... */ }
  manageInventory() { /* ... */ }
  // → 変更理由が5つ以上
}

// 2. switch/if-else の連鎖 → OCP違反
function processShape(shape: any): string {
  switch (shape.type) {
    case "circle": return `Circle: ${Math.PI * shape.r ** 2}`;
    case "rect": return `Rect: ${shape.w * shape.h}`;
    case "triangle": return `Triangle: ${0.5 * shape.b * shape.h}`;
    // 新しい図形を追加するたびにここを修正...
    default: throw new Error("Unknown shape");
  }
}

// 3. instanceof チェック → LSP違反の疑い
function fly(bird: Bird): void {
  if (bird instanceof Penguin) {
    throw new Error("Penguins can't fly!");
    // → Penguin は Bird の契約を満たしていない
  }
  bird.fly();
}

// 4. 空実装メソッド → ISP違反
class RobotWorker implements Worker {
  work(): void { /* 実装あり */ }
  eat(): void { /* 空実装 - ロボットは食べない */ }
  sleep(): void { /* 空実装 - ロボットは寝ない */ }
}

// 5. new による具象クラスの直接生成 → DIP違反
class OrderService {
  private repository = new MySQLOrderRepository();
  private mailer = new SmtpMailer();
  private logger = new FileLogger();
  // → テスト時にモック差し替えが不可能
}
```

### 2.3 SOLID適用のビフォー・アフター（TypeScript完全例）

```typescript
// ===== BEFORE: SOLID違反だらけのECサイト注文処理 =====

class OrderProcessor {
  processOrder(order: any): void {
    // バリデーション（SRP違反: バリデーションロジックが混在）
    if (!order.items || order.items.length === 0) {
      throw new Error("No items");
    }
    if (!order.paymentMethod) {
      throw new Error("No payment method");
    }

    // 合計計算
    let total = 0;
    for (const item of order.items) {
      total += item.price * item.quantity;
    }

    // 割引適用（OCP違反: 新しい割引タイプの追加でここを修正）
    if (order.discountType === "percentage") {
      total *= (1 - order.discountValue / 100);
    } else if (order.discountType === "fixed") {
      total -= order.discountValue;
    } else if (order.discountType === "bogo") {
      // Buy One Get One Free
      const cheapest = Math.min(...order.items.map((i: any) => i.price));
      total -= cheapest;
    }

    // 支払い処理（OCP違反: 新しい支払い方法で修正）
    if (order.paymentMethod === "credit") {
      // Stripe API呼び出し
      console.log(`Charging credit card: $${total}`);
    } else if (order.paymentMethod === "paypal") {
      console.log(`PayPal payment: $${total}`);
    } else if (order.paymentMethod === "bank") {
      console.log(`Bank transfer: $${total}`);
    }

    // DB保存（DIP違反: 具象に直接依存）
    const mysql = new MySQLConnection();
    mysql.query(`INSERT INTO orders VALUES (${total})`);

    // メール送信
    const smtp = new SmtpClient();
    smtp.send(order.email, "Order Confirmation", `Total: $${total}`);

    // ログ
    const fs = require("fs");
    fs.appendFileSync("orders.log", `Order processed: $${total}\n`);
  }
}
```

```typescript
// ===== AFTER: SOLID原則を適用したリファクタリング =====

// --- S: 単一責任 - 各クラスが1つの責任のみ ---

interface OrderItem {
  name: string;
  price: number;
  quantity: number;
}

interface Order {
  id: string;
  items: OrderItem[];
  customerEmail: string;
  discountCode?: string;
}

// バリデーション責任
class OrderValidator {
  validate(order: Order): void {
    if (!order.items || order.items.length === 0) {
      throw new ValidationError("Order must have at least one item");
    }
    for (const item of order.items) {
      if (item.price <= 0) throw new ValidationError("Invalid price");
      if (item.quantity <= 0) throw new ValidationError("Invalid quantity");
    }
  }
}

// 合計計算責任
class OrderCalculator {
  calculateSubtotal(items: OrderItem[]): number {
    return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  }
}

// --- O: 開放閉鎖 - 割引と支払いを拡張可能に ---

interface DiscountStrategy {
  apply(total: number, order: Order): number;
}

class PercentageDiscount implements DiscountStrategy {
  constructor(private percentage: number) {}
  apply(total: number): number {
    return total * (1 - this.percentage / 100);
  }
}

class FixedDiscount implements DiscountStrategy {
  constructor(private amount: number) {}
  apply(total: number): number {
    return Math.max(0, total - this.amount);
  }
}

class BuyOneGetOneFreeDiscount implements DiscountStrategy {
  apply(total: number, order: Order): number {
    const cheapest = Math.min(...order.items.map(i => i.price));
    return total - cheapest;
  }
}
// 新しい割引タイプ → クラスを追加するだけ。既存コード変更不要

// --- D: 依存性逆転 - 抽象に依存 ---

interface PaymentGateway {
  charge(amount: number, orderId: string): Promise<PaymentResult>;
}

interface OrderRepository {
  save(order: Order, total: number): Promise<void>;
}

interface NotificationService {
  notifyOrderConfirmation(email: string, order: Order, total: number): Promise<void>;
}

interface Logger {
  info(message: string): void;
  error(message: string, error: Error): void;
}

// 具象実装（交換可能）
class StripePaymentGateway implements PaymentGateway {
  async charge(amount: number, orderId: string): Promise<PaymentResult> {
    // Stripe API 呼び出し
    return { success: true, transactionId: `stripe_${orderId}` };
  }
}

class PostgresOrderRepository implements OrderRepository {
  async save(order: Order, total: number): Promise<void> {
    // PostgreSQL に保存
  }
}

class EmailNotificationService implements NotificationService {
  async notifyOrderConfirmation(
    email: string, order: Order, total: number
  ): Promise<void> {
    // SMTP 送信
  }
}

// --- オーケストレーター（薄い調整役） ---

class OrderProcessor {
  constructor(
    private validator: OrderValidator,
    private calculator: OrderCalculator,
    private discount: DiscountStrategy,
    private payment: PaymentGateway,
    private repository: OrderRepository,
    private notification: NotificationService,
    private logger: Logger,
  ) {}

  async processOrder(order: Order): Promise<void> {
    this.validator.validate(order);

    let total = this.calculator.calculateSubtotal(order.items);
    total = this.discount.apply(total, order);

    const result = await this.payment.charge(total, order.id);
    if (!result.success) {
      throw new PaymentError("Payment failed");
    }

    await this.repository.save(order, total);
    await this.notification.notifyOrderConfirmation(
      order.customerEmail, order, total
    );
    this.logger.info(`Order ${order.id} processed: $${total}`);
  }
}
```

---

## 3. 5原則の関係

```
SOLIDの関係性:

  SRP（単一責任）: クラスを小さく保つ
    ↓ 小さいクラスが増える
  ISP（インターフェース分離）: 細かいインターフェースで接続
    ↓ インターフェースに依存
  DIP（依存性逆転）: 具象ではなく抽象に依存
    ↓ 抽象を通じて拡張
  OCP（開放閉鎖）: 既存コードを変更せずに拡張
    ↓ 拡張時に互換性を維持
  LSP（リスコフ置換）: サブタイプが正しく代替可能

  → 5原則は相互に補完しあう
  → 1つだけ適用しても効果は限定的
  → 5つ全てを意識して設計する
```

### 3.1 原則間の相互作用を図解

```
詳細な関係マップ:

  ┌─────────────────────────────────────────────────────────┐
  │                   SOLID 相互関係図                      │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  SRP ─────→ ISP                                         │
  │  │ 「責任を分離」  「インターフェースも分離」             │
  │  │         ↘                                            │
  │  │          DIP                                         │
  │  │     「抽象に依存」                                    │
  │  │         ↙  ↘                                        │
  │  │       OCP   LSP                                      │
  │  │  「拡張可能」 「代替可能」                              │
  │  │                                                      │
  │  └──→ SRP が基盤: クラスが小さくなければ                 │
  │       他の原則を適用する意味がない                       │
  └─────────────────────────────────────────────────────────┘

具体例で理解する相互作用:

  1. SRP → ISP の連鎖:
     巨大な UserService を分割（SRP）
     → 各サービスに必要なインターフェースだけ注入（ISP）

  2. ISP → DIP の連鎖:
     細かいインターフェースを定義（ISP）
     → 上位モジュールはインターフェースに依存（DIP）

  3. DIP → OCP の連鎖:
     インターフェースに依存（DIP）
     → 新しい実装クラスを追加しても既存コード変更不要（OCP）

  4. OCP → LSP の連鎖:
     新しいサブタイプを追加（OCP）
     → そのサブタイプは既存のコードで正しく動作する必要がある（LSP）
```

### 3.2 5原則の優先順位

```
実務での適用優先順位:

  第1優先: SRP（単一責任）
    → まずクラスを小さくする。これが全ての基盤
    → 適用コスト: 低  効果: 高

  第2優先: DIP（依存性逆転）
    → テスト容易性に直結。コンストラクタインジェクションから始める
    → 適用コスト: 中  効果: 高

  第3優先: OCP（開放閉鎖）
    → 変更頻度の高い箇所にインターフェースを導入
    → 適用コスト: 中  効果: 中〜高

  第4優先: ISP（インターフェース分離）
    → インターフェースが肥大化してきたら分割
    → 適用コスト: 低  効果: 中

  第5優先: LSP（リスコフ置換）
    → 継承を使う場面で意識する。違反はバグに直結
    → 適用コスト: 低  効果: 中（ただし違反時の影響は甚大）

  注意:
    → これは「適用開始」の優先順位
    → 全原則を最終的に守ることが理想
    → LSP違反は見つけにくいが影響が大きい
```

---

## 4. 各原則の概要と例

### 4.1 S: 単一責任の原則（SRP）

```typescript
// === S: 単一責任 ===
// ❌ 複数の責任
class User {
  save() { /* DB保存 */ }
  sendEmail() { /* メール送信 */ }
  generateReport() { /* レポート生成 */ }
}

// ✅ 単一の責任
class User { /* ユーザーデータのみ */ }
class UserRepository { save(user: User) { } }
class EmailService { send(to: string, body: string) { } }
class ReportGenerator { generate(user: User) { } }
```

```python
# Python: SRP の実践例

# ❌ SRP違反: レポート生成クラスが複数の責任を持つ
class ReportManager:
    def fetch_data(self) -> list:
        """DBからデータ取得"""
        connection = psycopg2.connect("dbname=mydb")
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM sales")
        return cursor.fetchall()

    def calculate_statistics(self, data: list) -> dict:
        """統計計算"""
        total = sum(row[1] for row in data)
        average = total / len(data)
        return {"total": total, "average": average}

    def format_as_html(self, stats: dict) -> str:
        """HTML形式にフォーマット"""
        return f"<h1>Report</h1><p>Total: {stats['total']}</p>"

    def send_email(self, html: str, recipient: str) -> None:
        """メール送信"""
        import smtplib
        server = smtplib.SMTP("localhost")
        server.sendmail("noreply@example.com", recipient, html)

# ✅ SRP適用: 各クラスが1つの責任
class SalesDataFetcher:
    """データ取得のみ"""
    def __init__(self, connection_string: str):
        self._conn_str = connection_string

    def fetch(self) -> list[dict]:
        connection = psycopg2.connect(self._conn_str)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM sales")
        return [{"date": row[0], "amount": row[1]} for row in cursor.fetchall()]


class StatisticsCalculator:
    """統計計算のみ"""
    def calculate(self, data: list[dict]) -> dict:
        amounts = [d["amount"] for d in data]
        return {
            "total": sum(amounts),
            "average": sum(amounts) / len(amounts),
            "max": max(amounts),
            "min": min(amounts),
        }


class HtmlReportFormatter:
    """HTMLフォーマットのみ"""
    def format(self, stats: dict) -> str:
        return f"<h1>Sales Report</h1><p>Total: {stats['total']}</p>"


class EmailSender:
    """メール送信のみ"""
    def __init__(self, smtp_host: str):
        self._host = smtp_host

    def send(self, to: str, subject: str, body: str) -> None:
        server = smtplib.SMTP(self._host)
        server.sendmail("noreply@example.com", to, body)


class SalesReportUseCase:
    """オーケストレーション（薄い調整役）"""
    def __init__(
        self,
        fetcher: SalesDataFetcher,
        calculator: StatisticsCalculator,
        formatter: HtmlReportFormatter,
        sender: EmailSender,
    ):
        self._fetcher = fetcher
        self._calculator = calculator
        self._formatter = formatter
        self._sender = sender

    def execute(self, recipient: str) -> None:
        data = self._fetcher.fetch()
        stats = self._calculator.calculate(data)
        html = self._formatter.format(stats)
        self._sender.send(recipient, "Sales Report", html)
```

```java
// Java: SRP の実践例

// ❌ SRP違反
public class Employee {
    private String name;
    private double salary;

    // 責任1: 給与計算（経理部門が管理）
    public double calculatePay() {
        return salary * getOvertimeRate();
    }

    // 責任2: 労働時間レポート（人事部門が管理）
    public String reportHours() {
        return String.format("%s: %d hours", name, getHoursWorked());
    }

    // 責任3: DB永続化（技術部門が管理）
    public void save() {
        String sql = "UPDATE employees SET salary = ? WHERE name = ?";
        // JDBC処理...
    }

    // 共通メソッド（危険: 経理の変更が人事に影響する可能性）
    private int getHoursWorked() {
        // 経理と人事で「労働時間」の定義が異なる可能性
        return 160;
    }
}

// ✅ SRP適用
public record Employee(String id, String name, double salary) {}

public class PayCalculator {
    public double calculatePay(Employee employee) {
        return employee.salary() * getOvertimeRate();
    }
}

public class HoursReporter {
    public String reportHours(Employee employee) {
        return String.format("%s: %d hours", employee.name(), getHoursWorked());
    }
}

public class EmployeeRepository {
    public void save(Employee employee) {
        // JPA/Hibernate で永続化
    }
}
```

### 4.2 O: 開放閉鎖の原則（OCP）

```typescript
// === O: 開放閉鎖 ===
// ❌ 新しい形状を追加するたびに修正が必要
function calculateArea(shape: any): number {
  if (shape.type === "circle") return Math.PI * shape.radius ** 2;
  if (shape.type === "rectangle") return shape.width * shape.height;
  // 新しい形状を追加するたびにここを修正...
}

// ✅ 新しい形状はクラスを追加するだけ
interface Shape { area(): number; }
class Circle implements Shape { area() { return Math.PI * this.radius ** 2; } }
class Rectangle implements Shape { area() { return this.width * this.height; } }
// Triangle を追加しても既存コードは変更不要
```

```python
# Python: OCP の実践例 - ファイルエクスポーター

from abc import ABC, abstractmethod
import json
import csv
import io

# ❌ OCP違反: 新しいフォーマットを追加するたびに修正
class DataExporter:
    def export(self, data: list[dict], format_type: str) -> str:
        if format_type == "json":
            return json.dumps(data, indent=2)
        elif format_type == "csv":
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            return output.getvalue()
        elif format_type == "xml":
            # XML対応を追加するたびにこのクラスを修正...
            pass
        else:
            raise ValueError(f"Unknown format: {format_type}")


# ✅ OCP適用: 新しいフォーマットはクラスを追加するだけ
class ExportFormatter(ABC):
    @abstractmethod
    def format(self, data: list[dict]) -> str:
        ...

    @abstractmethod
    def content_type(self) -> str:
        ...

    @abstractmethod
    def file_extension(self) -> str:
        ...


class JsonFormatter(ExportFormatter):
    def format(self, data: list[dict]) -> str:
        return json.dumps(data, indent=2, ensure_ascii=False)

    def content_type(self) -> str:
        return "application/json"

    def file_extension(self) -> str:
        return ".json"


class CsvFormatter(ExportFormatter):
    def format(self, data: list[dict]) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()

    def content_type(self) -> str:
        return "text/csv"

    def file_extension(self) -> str:
        return ".csv"


class XmlFormatter(ExportFormatter):
    """新規追加 - 既存コードの修正は一切不要"""
    def format(self, data: list[dict]) -> str:
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<records>"]
        for record in data:
            lines.append("  <record>")
            for key, value in record.items():
                lines.append(f"    <{key}>{value}</{key}>")
            lines.append("  </record>")
        lines.append("</records>")
        return "\n".join(lines)

    def content_type(self) -> str:
        return "application/xml"

    def file_extension(self) -> str:
        return ".xml"


class DataExportService:
    """このクラスは新しいフォーマットが追加されても変更不要"""
    def __init__(self, formatter: ExportFormatter):
        self._formatter = formatter

    def export(self, data: list[dict]) -> str:
        return self._formatter.format(data)

    def export_to_file(self, data: list[dict], filename: str) -> None:
        content = self._formatter.format(data)
        ext = self._formatter.file_extension()
        with open(f"{filename}{ext}", "w") as f:
            f.write(content)
```

```java
// Java: OCP の実践例 - 税金計算

// ❌ OCP違反
public class TaxCalculator {
    public double calculate(String country, double amount) {
        switch (country) {
            case "JP": return amount * 0.10;  // 日本の消費税
            case "US": return amount * 0.07;  // アメリカの消費税
            case "UK": return amount * 0.20;  // イギリスのVAT
            // 新しい国を追加するたびにここを修正...
            default: throw new IllegalArgumentException("Unknown country");
        }
    }
}

// ✅ OCP適用
public interface TaxPolicy {
    double calculateTax(double amount);
    String countryCode();
}

public class JapaneseTax implements TaxPolicy {
    @Override public double calculateTax(double amount) {
        return amount * 0.10;  // 消費税10%
    }
    @Override public String countryCode() { return "JP"; }
}

public class UnitedStatesTax implements TaxPolicy {
    @Override public double calculateTax(double amount) {
        return amount * 0.07;  // Sales tax
    }
    @Override public String countryCode() { return "US"; }
}

public class UnitedKingdomTax implements TaxPolicy {
    @Override public double calculateTax(double amount) {
        return amount * 0.20;  // VAT
    }
    @Override public String countryCode() { return "UK"; }
}

// Registry パターンで動的に管理
public class TaxCalculator {
    private final Map<String, TaxPolicy> policies = new HashMap<>();

    public void register(TaxPolicy policy) {
        policies.put(policy.countryCode(), policy);
    }

    public double calculate(String country, double amount) {
        TaxPolicy policy = policies.get(country);
        if (policy == null) {
            throw new IllegalArgumentException("No tax policy for: " + country);
        }
        return policy.calculateTax(amount);
    }
}

// 使用例: 新しい国はクラスを追加してregisterするだけ
// TaxCalculator クラスは一切修正不要
```

### 4.3 L: リスコフの置換原則（LSP）

```typescript
// === L: リスコフ置換 ===
// ❌ 正方形は長方形の代替として使えない
class Rectangle {
  setWidth(w: number) { this.width = w; }
  setHeight(h: number) { this.height = h; }
}
class Square extends Rectangle {
  setWidth(w: number) { this.width = w; this.height = w; } // 親と異なる振る舞い!
}

// ✅ 共通インターフェースで抽象化
interface Shape { area(): number; }
class Rectangle implements Shape { /* ... */ }
class Square implements Shape { /* ... */ }
```

```python
# Python: LSP の詳細例

from abc import ABC, abstractmethod

# ❌ LSP違反: ReadOnlyFile が File の代替として使えない
class File:
    def __init__(self, path: str):
        self.path = path

    def read(self) -> str:
        with open(self.path) as f:
            return f.read()

    def write(self, content: str) -> None:
        with open(self.path, "w") as f:
            f.write(content)

class ReadOnlyFile(File):
    def write(self, content: str) -> None:
        raise PermissionError("Cannot write to read-only file")
        # → 親クラスの契約「write は書き込む」を破っている
        # → File を期待するコードに ReadOnlyFile を渡すとクラッシュ


# ✅ LSP適用: 読み取りと書き込みを分離
class Readable(ABC):
    @abstractmethod
    def read(self) -> str: ...

class Writable(ABC):
    @abstractmethod
    def write(self, content: str) -> None: ...

class ReadOnlyFile(Readable):
    def __init__(self, path: str):
        self._path = path

    def read(self) -> str:
        with open(self._path) as f:
            return f.read()

class ReadWriteFile(Readable, Writable):
    def __init__(self, path: str):
        self._path = path

    def read(self) -> str:
        with open(self._path) as f:
            return f.read()

    def write(self, content: str) -> None:
        with open(self._path, "w") as f:
            f.write(content)

# これで Readable を期待するコードに両方渡せる
# Writable を期待するコードには ReadWriteFile のみ渡せる
```

```java
// Java: LSP の実践例 - コレクション

// ❌ LSP違反の典型例
public class FixedSizeList<E> extends ArrayList<E> {
    private final int maxSize;

    public FixedSizeList(int maxSize) {
        this.maxSize = maxSize;
    }

    @Override
    public boolean add(E element) {
        if (size() >= maxSize) {
            throw new IllegalStateException("List is full");
            // → ArrayList の契約「add は常に要素を追加する」に違反
        }
        return super.add(element);
    }
}

// ✅ LSP適用: 別のインターフェースで設計
public interface BoundedCollection<E> {
    boolean add(E element);  // 契約: 容量があれば追加、なければ false
    boolean isFull();
    int capacity();
    int size();
}

public class BoundedList<E> implements BoundedCollection<E> {
    private final List<E> items = new ArrayList<>();
    private final int maxSize;

    public BoundedList(int maxSize) { this.maxSize = maxSize; }

    @Override
    public boolean add(E element) {
        if (isFull()) return false;  // 例外ではなく false を返す
        return items.add(element);
    }

    @Override public boolean isFull() { return items.size() >= maxSize; }
    @Override public int capacity() { return maxSize; }
    @Override public int size() { return items.size(); }
}
```

### 4.4 I: インターフェース分離の原則（ISP）

```typescript
// === I: インターフェース分離 ===
// ❌ 巨大インターフェース
interface Worker {
  work(): void;
  eat(): void;
  sleep(): void;
}
// ロボットは eat() と sleep() が不要!

// ✅ 細かいインターフェースに分離
interface Workable { work(): void; }
interface Eatable { eat(): void; }
interface Sleepable { sleep(): void; }
```

```typescript
// TypeScript: ISP の実践例 - プリンター

// ❌ ISP違反: 巨大なマルチ機能インターフェース
interface MultiFunctionDevice {
  print(doc: Document): void;
  scan(doc: Document): Image;
  fax(doc: Document): void;
  staple(doc: Document): void;
  copy(doc: Document): Document;
}

// シンプルなプリンターも全メソッドを実装する必要がある
class SimplePrinter implements MultiFunctionDevice {
  print(doc: Document): void { /* 実装あり */ }
  scan(doc: Document): Image { throw new Error("Not supported"); }
  fax(doc: Document): void { throw new Error("Not supported"); }
  staple(doc: Document): void { throw new Error("Not supported"); }
  copy(doc: Document): Document { throw new Error("Not supported"); }
}

// ✅ ISP適用: 機能ごとにインターフェースを分離
interface Printer {
  print(doc: Document): void;
}

interface Scanner {
  scan(doc: Document): Image;
}

interface Faxer {
  fax(doc: Document): void;
}

interface Stapler {
  staple(doc: Document): void;
}

// シンプルなプリンター: 必要な機能だけ実装
class BasicPrinter implements Printer {
  print(doc: Document): void {
    console.log("Printing document...");
  }
}

// 高機能複合機: 複数のインターフェースを実装
class OfficePrinter implements Printer, Scanner, Faxer, Stapler {
  print(doc: Document): void { /* ... */ }
  scan(doc: Document): Image { /* ... */ }
  fax(doc: Document): void { /* ... */ }
  staple(doc: Document): void { /* ... */ }
}

// クライアントは必要な機能だけに依存
function printReport(printer: Printer): void {
  // Printer インターフェースだけに依存
  // Scanner や Faxer の知識は不要
  printer.print(report);
}
```

```python
# Python: ISP の実践例 - 認証プロバイダ

from abc import ABC, abstractmethod
from typing import Protocol

# ❌ ISP違反: 全プロバイダに不要なメソッドを強制
class AuthProvider(ABC):
    @abstractmethod
    def authenticate(self, username: str, password: str) -> bool: ...

    @abstractmethod
    def authorize(self, user_id: str, resource: str) -> bool: ...

    @abstractmethod
    def refresh_token(self, token: str) -> str: ...

    @abstractmethod
    def revoke_token(self, token: str) -> None: ...

    @abstractmethod
    def get_user_info(self, token: str) -> dict: ...

    @abstractmethod
    def send_mfa_code(self, user_id: str) -> None: ...

    @abstractmethod
    def verify_mfa_code(self, user_id: str, code: str) -> bool: ...


# ✅ ISP適用: 関心事ごとにプロトコルを分離
class Authenticator(Protocol):
    def authenticate(self, username: str, password: str) -> bool: ...

class Authorizer(Protocol):
    def authorize(self, user_id: str, resource: str) -> bool: ...

class TokenManager(Protocol):
    def refresh_token(self, token: str) -> str: ...
    def revoke_token(self, token: str) -> None: ...

class UserInfoProvider(Protocol):
    def get_user_info(self, token: str) -> dict: ...

class MfaProvider(Protocol):
    def send_mfa_code(self, user_id: str) -> None: ...
    def verify_mfa_code(self, user_id: str, code: str) -> bool: ...


# 基本認証: MFAもトークン管理も不要
class BasicAuthProvider:
    def authenticate(self, username: str, password: str) -> bool:
        # パスワード検証のみ
        return check_password(username, password)

    def authorize(self, user_id: str, resource: str) -> bool:
        return check_permission(user_id, resource)


# OAuth: MFAは不要だがトークン管理は必要
class OAuthProvider:
    def authenticate(self, username: str, password: str) -> bool: ...
    def authorize(self, user_id: str, resource: str) -> bool: ...
    def refresh_token(self, token: str) -> str: ...
    def revoke_token(self, token: str) -> None: ...
    def get_user_info(self, token: str) -> dict: ...


# クライアントは必要なプロトコルだけに依存
def login(auth: Authenticator, username: str, password: str) -> bool:
    return auth.authenticate(username, password)

def check_access(authz: Authorizer, user_id: str, resource: str) -> bool:
    return authz.authorize(user_id, resource)
```

### 4.5 D: 依存性逆転の原則（DIP）

```typescript
// === D: 依存性逆転 ===
// ❌ 具象に依存
class OrderService {
  private db = new MySQLDatabase(); // 具象クラスに直接依存
  save(order: Order) { this.db.insert(order); }
}

// ✅ 抽象に依存
interface Database { insert(data: any): void; }
class OrderService {
  constructor(private db: Database) {} // インターフェースに依存
  save(order: Order) { this.db.insert(order); }
}
```

```typescript
// TypeScript: DIP の実践例 - 通知システム

// ❌ DIP違反: 上位モジュールが下位モジュールの具象に依存
class UserRegistrationService {
  // 具象クラスを直接生成 → テスト不可、差し替え不可
  private emailClient = new SendGridClient("api-key-xxx");
  private userRepo = new PostgresUserRepository("postgres://...");
  private logger = new WinstonLogger("./logs/app.log");

  async register(data: CreateUserDto): Promise<User> {
    const user = await this.userRepo.save(data);
    await this.emailClient.sendWelcome(user.email);
    this.logger.info(`User registered: ${user.id}`);
    return user;
  }
}

// ✅ DIP適用: 上位モジュールも下位モジュールも抽象に依存

// 抽象（インターフェース）を定義
interface UserRepository {
  save(data: CreateUserDto): Promise<User>;
  findById(id: string): Promise<User | null>;
}

interface EmailClient {
  sendWelcome(to: string): Promise<void>;
  sendPasswordReset(to: string, token: string): Promise<void>;
}

interface AppLogger {
  info(message: string): void;
  error(message: string, error?: Error): void;
}

// 上位モジュール: 抽象のみに依存
class UserRegistrationService {
  constructor(
    private userRepo: UserRepository,
    private emailClient: EmailClient,
    private logger: AppLogger,
  ) {}

  async register(data: CreateUserDto): Promise<User> {
    const user = await this.userRepo.save(data);
    await this.emailClient.sendWelcome(user.email);
    this.logger.info(`User registered: ${user.id}`);
    return user;
  }
}

// 下位モジュール: 抽象を実装
class PostgresUserRepository implements UserRepository {
  async save(data: CreateUserDto): Promise<User> { /* Postgres実装 */ }
  async findById(id: string): Promise<User | null> { /* Postgres実装 */ }
}

class SendGridEmailClient implements EmailClient {
  async sendWelcome(to: string): Promise<void> { /* SendGrid実装 */ }
  async sendPasswordReset(to: string, token: string): Promise<void> { /* ... */ }
}

class WinstonLogger implements AppLogger {
  info(message: string): void { /* Winston実装 */ }
  error(message: string, error?: Error): void { /* Winston実装 */ }
}

// テスト用モック
class InMemoryUserRepository implements UserRepository {
  private users: User[] = [];
  async save(data: CreateUserDto): Promise<User> {
    const user = { id: String(this.users.length + 1), ...data };
    this.users.push(user);
    return user;
  }
  async findById(id: string): Promise<User | null> {
    return this.users.find(u => u.id === id) ?? null;
  }
}

class MockEmailClient implements EmailClient {
  sentEmails: string[] = [];
  async sendWelcome(to: string): Promise<void> {
    this.sentEmails.push(to);
  }
  async sendPasswordReset(to: string, token: string): Promise<void> {
    this.sentEmails.push(`${to}:${token}`);
  }
}

// テスト: モックを注入して外部依存なしにテスト可能
describe("UserRegistrationService", () => {
  it("should register a user and send welcome email", async () => {
    const repo = new InMemoryUserRepository();
    const emailClient = new MockEmailClient();
    const logger = new ConsoleLogger();

    const service = new UserRegistrationService(repo, emailClient, logger);
    const user = await service.register({
      name: "Test User",
      email: "test@example.com",
    });

    expect(user.name).toBe("Test User");
    expect(emailClient.sentEmails).toContain("test@example.com");
  });
});
```

```python
# Python: DIP の実践例 - DIコンテナ

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Type

# 抽象定義
class CacheStore(ABC):
    @abstractmethod
    def get(self, key: str) -> str | None: ...

    @abstractmethod
    def set(self, key: str, value: str, ttl: int = 300) -> None: ...

class MessageQueue(ABC):
    @abstractmethod
    def publish(self, topic: str, message: dict) -> None: ...

    @abstractmethod
    def subscribe(self, topic: str, handler: callable) -> None: ...

# 具象実装
class RedisCache(CacheStore):
    def __init__(self, url: str):
        self._url = url

    def get(self, key: str) -> str | None:
        # Redis実装
        pass

    def set(self, key: str, value: str, ttl: int = 300) -> None:
        # Redis実装
        pass

class InMemoryCache(CacheStore):
    """テスト用 / 開発用"""
    def __init__(self):
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str, ttl: int = 300) -> None:
        self._store[key] = value

class RabbitMQQueue(MessageQueue):
    def __init__(self, url: str):
        self._url = url

    def publish(self, topic: str, message: dict) -> None:
        # RabbitMQ実装
        pass

    def subscribe(self, topic: str, handler: callable) -> None:
        # RabbitMQ実装
        pass

# シンプルなDIコンテナ
class Container:
    """依存性注入コンテナ"""
    def __init__(self):
        self._bindings: dict[type, callable] = {}

    def bind(self, abstract: type, factory: callable) -> None:
        self._bindings[abstract] = factory

    def resolve(self, abstract: type):
        factory = self._bindings.get(abstract)
        if factory is None:
            raise ValueError(f"No binding for {abstract}")
        return factory()

# 環境に応じた設定
def configure_production(container: Container) -> None:
    container.bind(CacheStore, lambda: RedisCache("redis://prod:6379"))
    container.bind(MessageQueue, lambda: RabbitMQQueue("amqp://prod:5672"))

def configure_testing(container: Container) -> None:
    container.bind(CacheStore, lambda: InMemoryCache())
    container.bind(MessageQueue, lambda: InMemoryQueue())

# 使用例
container = Container()
configure_production(container)  # 本番環境
# configure_testing(container)  # テスト環境

cache: CacheStore = container.resolve(CacheStore)
queue: MessageQueue = container.resolve(MessageQueue)
```

---

## 5. SOLID適用の判断基準

```
過剰適用の警告:
  → 10行のスクリプトにSOLIDは不要
  → 個人プロジェクトの初期段階で過度な抽象化は有害
  → 「YAGNI（You Ain't Gonna Need It）」とのバランス

適用すべき場面:
  ✓ チーム開発のプロダクションコード
  ✓ 長期間メンテナンスされるシステム
  ✓ テストが重要なシステム
  ✓ 変更が頻繁に発生する領域

段階的な適用:
  1. まずシンプルに書く
  2. 変更が発生したら、その部分にSOLIDを適用
  3. 「3回目の変更」で抽象化を検討（Rule of Three）
```

### 5.1 プロジェクト規模別の適用ガイド

```
小規模プロジェクト（〜5,000行）:
  ┌─────────────────────────────────────────┐
  │ 適用レベル: 最小限                        │
  │ SRP: ファイル分割程度                     │
  │ OCP: 不要（まだ変更パターンが見えない）    │
  │ LSP: 継承を使う場合のみ意識                │
  │ ISP: 不要（インターフェースが少ない）       │
  │ DIP: 不要（テストが簡単に書ける規模）       │
  └─────────────────────────────────────────┘

中規模プロジェクト（5,000〜50,000行）:
  ┌─────────────────────────────────────────┐
  │ 適用レベル: 選択的適用                     │
  │ SRP: 積極的に適用                         │
  │ OCP: 変更頻度の高い箇所に適用              │
  │ LSP: 継承階層があれば必ず確認              │
  │ ISP: 太いインターフェースが出現したら分割   │
  │ DIP: 外部サービス連携部分に適用            │
  └─────────────────────────────────────────┘

大規模プロジェクト（50,000行〜）:
  ┌─────────────────────────────────────────┐
  │ 適用レベル: 全面的適用                     │
  │ SRP: 全クラスで厳守                       │
  │ OCP: ドメインロジック全体に適用            │
  │ LSP: 全サブタイプで契約を検証              │
  │ ISP: ロール別インターフェース設計           │
  │ DIP: DIコンテナを活用した全面的な適用       │
  └─────────────────────────────────────────┘
```

### 5.2 SOLID違反の検出ツールとメトリクス

```
静的解析ツール:

  TypeScript/JavaScript:
    - ESLint + @typescript-eslint/recommended
    - SonarQube / SonarCloud
    - CodeClimate

  Java:
    - SonarQube
    - PMD
    - Checkstyle
    - SpotBugs（旧FindBugs）

  Python:
    - pylint
    - flake8 + flake8-import-order
    - mypy（型チェック = DIP/LSP の検証に有効）
    - SonarQube

  C#:
    - ReSharper / Rider
    - Roslyn Analyzers
    - NDepend

注目すべきメトリクス:
  ┌───────────────────┬───────────────────────────┐
  │ メトリクス         │ SOLID違反の兆候             │
  ├───────────────────┼───────────────────────────┤
  │ クラス行数 > 300   │ SRP違反の可能性             │
  │ メソッド数 > 15    │ SRP違反の可能性             │
  │ 依存クラス数 > 8   │ SRP + DIP違反の可能性       │
  │ 循環的複雑度 > 10  │ OCP違反の可能性（条件分岐多）│
  │ 継承深度 > 4       │ LSP違反のリスク増大          │
  │ インターフェース    │ ISP違反の可能性             │
  │ メソッド数 > 10    │                            │
  │ 結合度が高い       │ DIP違反の可能性             │
  └───────────────────┴───────────────────────────┘
```

### 5.3 段階的リファクタリング手順

```typescript
// Step 1: まずシンプルに書く（SOLID無視でOK）
class TodoApp {
  private todos: Todo[] = [];

  addTodo(title: string): void {
    this.todos.push({ id: Date.now(), title, done: false });
  }

  completeTodo(id: number): void {
    const todo = this.todos.find(t => t.id === id);
    if (todo) todo.done = true;
  }

  getAll(): Todo[] {
    return [...this.todos];
  }
}
// → 小さいうちはこれで十分
```

```typescript
// Step 2: 変更が発生 → その部分にSRP適用
// 要件: 「Todo をファイルに保存したい」「通知も送りたい」

// SRP: 永続化を分離
interface TodoRepository {
  save(todos: Todo[]): Promise<void>;
  load(): Promise<Todo[]>;
}

class FileTodoRepository implements TodoRepository {
  async save(todos: Todo[]): Promise<void> {
    await fs.writeFile("todos.json", JSON.stringify(todos));
  }
  async load(): Promise<Todo[]> {
    const data = await fs.readFile("todos.json", "utf-8");
    return JSON.parse(data);
  }
}

class TodoService {
  constructor(private repo: TodoRepository) {} // DIP: 抽象に依存

  async addTodo(title: string): Promise<Todo> {
    const todos = await this.repo.load();
    const todo = { id: Date.now(), title, done: false };
    todos.push(todo);
    await this.repo.save(todos);
    return todo;
  }
}
```

```typescript
// Step 3: 3回目の変更 → OCP で拡張ポイントを設計
// 要件: 「DB保存もしたい」「S3保存もしたい」

// OCP: 新しいストレージはクラスを追加するだけ
class DatabaseTodoRepository implements TodoRepository {
  async save(todos: Todo[]): Promise<void> {
    // PostgreSQL に保存
  }
  async load(): Promise<Todo[]> {
    // PostgreSQL から取得
  }
}

class S3TodoRepository implements TodoRepository {
  async save(todos: Todo[]): Promise<void> {
    // AWS S3 に保存
  }
  async load(): Promise<Todo[]> {
    // AWS S3 から取得
  }
}

// TodoService は一切変更不要（OCP達成）
```

---

## 6. SOLIDと他の設計原則の関係

```
SOLID と GRASP の対応:

  GRASP（General Responsibility Assignment Software Patterns）:
  ┌────────────────────┬────────────────────────────┐
  │ GRASP原則           │ 対応するSOLID原則            │
  ├────────────────────┼────────────────────────────┤
  │ Information Expert  │ SRP（責任の適切な割り当て）   │
  │ Creator            │ DIP（生成の責任を分離）       │
  │ Low Coupling       │ DIP + ISP（疎結合の実現）     │
  │ High Cohesion      │ SRP（高凝集の維持）          │
  │ Polymorphism       │ OCP + LSP                    │
  │ Pure Fabrication   │ SRP（人工的なクラスの導入）   │
  │ Indirection        │ DIP（間接参照の導入）         │
  │ Protected Variations│ OCP（変更の影響を隔離）      │
  └────────────────────┴────────────────────────────┘

SOLID と DRY/KISS/YAGNI:

  DRY（Don't Repeat Yourself）:
    → SRP と補完関係: 責任を分離すると自然にDRYになる
    → 注意: DRYを過度に追求すると不適切な抽象化（Wrong Abstraction）

  KISS（Keep It Simple, Stupid）:
    → SOLIDの過剰適用を抑制する原則
    → 10行のスクリプトに5つのクラスは KISS 違反

  YAGNI（You Ain't Gonna Need It）:
    → OCP の適用タイミングを制御
    → 「将来必要になるかも」で抽象化しない
    → 実際に2-3回変更が発生してからインターフェースを導入
```

### 6.1 SOLIDとクリーンアーキテクチャ

```
クリーンアーキテクチャにおけるSOLIDの役割:

  ┌─────────────────────────────────────────────┐
  │                Frameworks Layer               │
  │  ┌─────────────────────────────────────────┐ │
  │  │           Interface Adapters             │ │
  │  │  ┌─────────────────────────────────────┐│ │
  │  │  │         Use Cases                   ││ │
  │  │  │  ┌─────────────────────────────────┐││ │
  │  │  │  │        Entities                 │││ │
  │  │  │  └─────────────────────────────────┘││ │
  │  │  └─────────────────────────────────────┘│ │
  │  └─────────────────────────────────────────┘ │
  └─────────────────────────────────────────────┘

  各層とSOLIDの対応:
    Entities（ドメイン層）:
      → SRP: エンティティは1つのビジネスルールのみ
      → LSP: Value Object の等価性

    Use Cases（ユースケース層）:
      → SRP: 1ユースケース = 1クラス
      → OCP: ユースケースの拡張

    Interface Adapters（アダプタ層）:
      → DIP: 外部サービスへの依存を逆転
      → ISP: ポートを小さく分離

    Frameworks（フレームワーク層）:
      → DIP: フレームワークの詳細に依存しない
      → OCP: フレームワーク交換時も内側は変更不要

  依存の方向:
    外側 → 内側（DIPにより実現）
    Frameworks → Adapters → Use Cases → Entities
    内側のコードは外側の存在を知らない
```

### 6.2 SOLIDとマイクロサービス

```
マイクロサービスアーキテクチャとSOLID:

  SRP → サービスの境界:
    1つのマイクロサービス = 1つのビジネスドメイン
    ✓ UserService: ユーザー管理のみ
    ✓ OrderService: 注文管理のみ
    ✓ PaymentService: 決済管理のみ
    ✗ MonolithService: 全部入り → SRP違反

  OCP → サービスの拡張:
    新しいサービスを追加するとき、既存サービスを変更しない
    → イベント駆動アーキテクチャ
    → メッセージキューによる疎結合

  LSP → API の後方互換性:
    サービスのv2はv1と後方互換であるべき
    → API バージョニング
    → コンシューマ駆動契約テスト（CDC）

  ISP → API の粒度:
    1つのエンドポイントに全情報を詰め込まない
    → GraphQL: クライアントが必要なフィールドだけ取得
    → BFF（Backend For Frontend）パターン

  DIP → サービス間通信:
    具体的なサービスのURLに直接依存しない
    → サービスディスカバリ
    → API Gateway
    → メッセージブローカー
```

---

## 7. SOLIDと関数型プログラミング

```
SOLIDは OOP 固有の原則だが、関数型プログラミングにも対応する概念がある:

  ┌────────────┬──────────────────────────────────┐
  │ SOLID原則   │ 関数型での対応概念                  │
  ├────────────┼──────────────────────────────────┤
  │ SRP        │ 純粋関数（1つの計算のみ行う）        │
  │ OCP        │ 高階関数（関数を渡して振る舞いを拡張）│
  │ LSP        │ 型クラス制約（Haskellの型クラス）     │
  │ ISP        │ 型クラスの細分化                     │
  │ DIP        │ 関数の注入（コールバック、DI関数）     │
  └────────────┴──────────────────────────────────┘
```

```typescript
// TypeScript: 関数型スタイルでのSOLID

// SRP: 純粋関数 = 1つの責任
const calculateTax = (amount: number, rate: number): number =>
  amount * rate;

const formatCurrency = (amount: number): string =>
  `$${amount.toFixed(2)}`;

// OCP: 高階関数で拡張
type Middleware<T> = (data: T) => T;

const applyMiddlewares = <T>(data: T, middlewares: Middleware<T>[]): T =>
  middlewares.reduce((acc, mw) => mw(acc), data);

// 新しいミドルウェアを追加するだけで拡張可能
const addTimestamp: Middleware<Request> = (req) => ({
  ...req,
  timestamp: Date.now(),
});

const addCorrelationId: Middleware<Request> = (req) => ({
  ...req,
  correlationId: crypto.randomUUID(),
});

// DIP: 関数を注入
type Fetcher = (url: string) => Promise<Response>;
type Parser<T> = (data: string) => T;

const loadData = async <T>(
  url: string,
  fetch: Fetcher,      // 具象ではなく関数型を注入
  parse: Parser<T>,    // パーサーも注入
): Promise<T> => {
  const response = await fetch(url);
  const text = await response.text();
  return parse(text);
};

// テスト時: モック関数を渡す
const mockFetch: Fetcher = async () =>
  new Response('{"name":"test"}');

const result = await loadData(
  "https://api.example.com/data",
  mockFetch,
  JSON.parse,
);
```

```python
# Python: 関数型スタイルでのSOLID

from typing import Callable, TypeVar
from functools import reduce

T = TypeVar("T")

# SRP: 純粋関数
def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[1]

def hash_password(password: str) -> str:
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

# OCP: 高階関数で振る舞いを拡張
Pipeline = list[Callable[[T], T]]

def execute_pipeline(data: T, steps: Pipeline[T]) -> T:
    return reduce(lambda acc, step: step(acc), steps, data)

# ステップを追加するだけで拡張
def normalize(text: str) -> str:
    return text.strip().lower()

def remove_special_chars(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9\s]", "", text)

def truncate(text: str) -> str:
    return text[:100]

# パイプライン実行
result = execute_pipeline(
    "  Hello, World! @#$ ",
    [normalize, remove_special_chars, truncate]
)
# → "hello world  "

# DIP: 関数注入
def process_order(
    save: Callable[[dict], None],
    notify: Callable[[str, str], None],
    order: dict,
) -> None:
    save(order)
    notify(order["email"], f"Order {order['id']} confirmed")

# 本番
process_order(save=db_save, notify=email_notify, order=order_data)

# テスト
calls = []
process_order(
    save=lambda o: calls.append(("save", o)),
    notify=lambda e, m: calls.append(("notify", e, m)),
    order={"id": "1", "email": "test@test.com"},
)
assert len(calls) == 2
```

---

## 8. 言語別のSOLID実装パターン

```
各言語の特性とSOLID実装:

  TypeScript:
    - interface + class でSOLID実装が自然
    - 構造的型付け（Structural Typing）で ISP が柔軟
    - DI フレームワーク: tsyringe, inversify, NestJS

  Java:
    - interface + abstract class が言語レベルでサポート
    - Spring Framework が DIP を標準化
    - アノテーション（@Inject, @Autowired）で DI

  Python:
    - Protocol（構造的サブタイピング）で ISP
    - ABC で明示的なインターフェース
    - Duck Typing でOCP が自然に実現

  Kotlin:
    - sealed class で LSP の安全性向上
    - data class でイミュータブルなエンティティ
    - Extension function で OCP

  Rust:
    - trait で ISP + DIP
    - enum + match で型安全な OCP
    - Ownership で SRP が強制される

  Go:
    - 暗黙的なインターフェース実装で ISP が極めて自然
    - 小さなインターフェースの文化（io.Reader, io.Writer）
    - 構造体の埋め込み（Composition over Inheritance）
```

---

## まとめ

| 原則 | 一言で | 効果 | 実現手段 | 注意点 |
|------|--------|------|---------|--------|
| SRP | 1クラス1責任 | 変更の影響を局所化 | 責任の分離、委譲 | 過剰分割に注意 |
| OCP | 拡張は開、修正は閉 | 既存コードの安定性 | インターフェース、ポリモーフィズム | YAGNIとのバランス |
| LSP | 代替可能性 | ポリモーフィズムの正しさ | 事前/事後条件の維持 | 違反は見つけにくい |
| ISP | インターフェースを小さく | 不要な依存の排除 | ロール別インターフェース | 過度な細分化に注意 |
| DIP | 抽象に依存 | 疎結合・テスト容易性 | DI、コンストラクタ注入 | 小規模では不要 |

### チェックリスト

```
設計レビュー時のSOLIDチェックリスト:

  □ SRP: このクラスを変更する理由は1つだけか？
  □ SRP: このクラスの名前は1つの責任を表しているか？
  □ OCP: 新しい種類を追加するとき、既存コードを修正するか？
  □ OCP: switch/if-else の連鎖がないか？
  □ LSP: サブクラスを親クラスの代わりに使っても正しく動作するか？
  □ LSP: サブクラスで例外を投げて親の契約を破っていないか？
  □ ISP: インターフェースに使わないメソッドが含まれていないか？
  □ ISP: クライアントが不要な依存を持たされていないか？
  □ DIP: 上位モジュールが具象クラスに直接依存していないか？
  □ DIP: テスト時にモックに差し替えられるか？
```

---

## 次に読むべきガイド
-> [[01-srp-and-ocp.md]] -- SRP + OCP 詳細

---

## 参考文献
1. Martin, R. "Agile Software Development, Principles, Patterns, and Practices." Prentice Hall, 2003.
2. Martin, R. "Clean Architecture: A Craftsman's Guide to Software Structure and Design." Prentice Hall, 2017.
3. Martin, R. "The Principles of OOD." butunclebob.com, 2005.
4. Meyer, B. "Object-Oriented Software Construction." Prentice Hall, 2nd ed., 1997.
5. Liskov, B. "Data Abstraction and Hierarchy." SIGPLAN Notices, 1988.
6. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
7. Feathers, M. "Working Effectively with Legacy Code." Prentice Hall, 2004.
8. Fowler, M. "Refactoring: Improving the Design of Existing Code." Addison-Wesley, 2nd ed., 2018.
