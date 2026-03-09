# SRP（単一責任の原則）+ OCP（開放閉鎖の原則）

> SRPは「変更する理由を1つに」、OCPは「変更せずに拡張する」。この2つの原則が、保守性の高い設計の土台を作る。

## この章で学ぶこと

- [ ] SRP の「責任」の正しい定義を理解する
- [ ] OCP をポリモーフィズムで実現する方法を把握する
- [ ] 実践的なリファクタリング手法を学ぶ
- [ ] SRP と OCP の違反パターンを検出できるようになる
- [ ] 多言語での SRP/OCP 適用パターンを習得する
- [ ] 現実のプロジェクトでの段階的な適用方法を学ぶ


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [SOLID原則概要](./00-solid-overview.md) の内容を理解していること

---

## 1. SRP: 単一責任の原則

```
定義（Robert C. Martin）:
  「クラスを変更する理由は1つだけであるべき」

より正確な定義:
  「クラスは1つのアクター（利害関係者）に対してのみ責任を持つ」

  例:
    Employee クラスが以下を持つ場合:
    - calculatePay()    → CFO（経理部門）の責任
    - reportHours()     → COO（業務部門）の責任
    - save()            → CTO（技術部門）の責任

    → 3つのアクターに依存 = SRP 違反
    → 経理部門の要求変更が業務部門のコードに影響する可能性
```

### 1.1 SRP の「責任」とは何か

```
「責任」の誤解と正しい理解:

  ❌ 誤解: 「1つのメソッドだけ持つべき」
    → メソッド数で判断するのは間違い
    → 100メソッドでも「1つの責任」ならSRP準拠

  ❌ 誤解: 「1つのことだけする」
    → 抽象度によって「1つのこと」の粒度が変わる
    → 何をもって「1つ」とするかが曖昧

  ✅ 正しい理解: 「変更する理由が1つだけ」
    → 「このクラスを変更したい人（アクター）は誰か？」
    → アクターが1人だけなら SRP 準拠

  ✅ より実践的な理解: 「1つのアクターに対する責任」
    → アクター = ビジネス上の利害関係者
    → 経理部門、人事部門、技術部門など
    → 同じアクターの要求変更は1つのクラスに閉じるべき

  責任の粒度の判断基準:
    1. 「このクラスが変更される場面を3つ挙げてみる」
    2. その3つが同じアクターの要求なら → SRP準拠
    3. 異なるアクターの要求なら → SRP違反の可能性
```

### 1.2 SRP リファクタリング

```typescript
// ❌ SRP違反: 複数の責任を持つクラス
class UserService {
  // 責任1: ユーザーの作成ロジック
  createUser(data: CreateUserDto): User {
    // バリデーション
    if (!data.email.includes("@")) throw new Error("Invalid email");
    if (data.password.length < 8) throw new Error("Password too short");

    // パスワードハッシュ化
    const hashedPassword = bcrypt.hashSync(data.password, 10);

    // DB保存
    const user = db.users.create({ ...data, password: hashedPassword });

    // メール送信
    const html = `<h1>Welcome ${data.name}!</h1>`;
    emailClient.send(data.email, "Welcome", html);

    // ログ
    logger.info(`User created: ${user.id}`);

    return user;
  }
}

// ✅ SRP適用: 各クラスが1つの責任を持つ
class UserValidator {
  validate(data: CreateUserDto): void {
    if (!data.email.includes("@")) throw new ValidationError("Invalid email");
    if (data.password.length < 8) throw new ValidationError("Password too short");
  }
}

class PasswordHasher {
  hash(password: string): string {
    return bcrypt.hashSync(password, 10);
  }
}

class UserRepository {
  create(data: CreateUserDto & { password: string }): User {
    return db.users.create(data);
  }
}

class WelcomeEmailSender {
  send(user: User): void {
    const html = `<h1>Welcome ${user.name}!</h1>`;
    emailClient.send(user.email, "Welcome", html);
  }
}

// オーケストレーター
class UserRegistrationService {
  constructor(
    private validator: UserValidator,
    private hasher: PasswordHasher,
    private repo: UserRepository,
    private emailSender: WelcomeEmailSender,
  ) {}

  async register(data: CreateUserDto): Promise<User> {
    this.validator.validate(data);
    const hashedPassword = this.hasher.hash(data.password);
    const user = await this.repo.create({ ...data, password: hashedPassword });
    this.emailSender.send(user);
    return user;
  }
}
```

### 1.3 SRP の多言語実践例

```python
# Python: SRP の実践例 - EC サイトの注文処理

# ❌ SRP違反: 1つのクラスが注文に関する全てを担当
class OrderManager:
    def __init__(self):
        self.db = psycopg2.connect("dbname=shop")

    def create_order(self, customer_id: int, items: list[dict]) -> dict:
        # バリデーション（責任1）
        if not items:
            raise ValueError("Order must have at least one item")
        for item in items:
            if item["quantity"] <= 0:
                raise ValueError(f"Invalid quantity for {item['name']}")

        # 価格計算（責任2）
        subtotal = sum(i["price"] * i["quantity"] for i in items)
        tax = subtotal * 0.10  # 消費税
        shipping = 500 if subtotal < 5000 else 0
        total = subtotal + tax + shipping

        # 在庫チェック（責任3）
        cursor = self.db.cursor()
        for item in items:
            cursor.execute(
                "SELECT stock FROM products WHERE id = %s",
                (item["product_id"],)
            )
            stock = cursor.fetchone()[0]
            if stock < item["quantity"]:
                raise ValueError(f"Insufficient stock for {item['name']}")

        # DB保存（責任4）
        cursor.execute(
            "INSERT INTO orders (customer_id, total) VALUES (%s, %s) RETURNING id",
            (customer_id, total)
        )
        order_id = cursor.fetchone()[0]
        self.db.commit()

        # メール送信（責任5）
        import smtplib
        server = smtplib.SMTP("smtp.example.com")
        server.sendmail(
            "shop@example.com",
            f"customer_{customer_id}@example.com",
            f"Your order #{order_id} has been placed. Total: ¥{total}"
        )

        return {"order_id": order_id, "total": total}


# ✅ SRP適用: 各クラスが1つの責任のみ持つ

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class OrderItem:
    product_id: int
    name: str
    price: int
    quantity: int

@dataclass
class Order:
    id: int | None
    customer_id: int
    items: list[OrderItem]
    subtotal: int
    tax: int
    shipping: int
    total: int


class OrderValidator:
    """注文データのバリデーションのみ"""
    def validate(self, customer_id: int, items: list[OrderItem]) -> None:
        if not items:
            raise ValueError("Order must have at least one item")
        for item in items:
            if item.quantity <= 0:
                raise ValueError(f"Invalid quantity for {item.name}")
            if item.price <= 0:
                raise ValueError(f"Invalid price for {item.name}")


class PriceCalculator:
    """価格計算のみ"""
    TAX_RATE = 0.10
    FREE_SHIPPING_THRESHOLD = 5000
    SHIPPING_FEE = 500

    def calculate(self, items: list[OrderItem]) -> tuple[int, int, int, int]:
        subtotal = sum(item.price * item.quantity for item in items)
        tax = int(subtotal * self.TAX_RATE)
        shipping = 0 if subtotal >= self.FREE_SHIPPING_THRESHOLD else self.SHIPPING_FEE
        total = subtotal + tax + shipping
        return subtotal, tax, shipping, total


class InventoryChecker:
    """在庫確認のみ"""
    def __init__(self, db_connection):
        self._db = db_connection

    def check_availability(self, items: list[OrderItem]) -> None:
        cursor = self._db.cursor()
        for item in items:
            cursor.execute(
                "SELECT stock FROM products WHERE id = %s",
                (item.product_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Product not found: {item.product_id}")
            if row[0] < item.quantity:
                raise ValueError(f"Insufficient stock for {item.name}")


class OrderRepository:
    """注文のDB永続化のみ"""
    def __init__(self, db_connection):
        self._db = db_connection

    def save(self, order: Order) -> int:
        cursor = self._db.cursor()
        cursor.execute(
            "INSERT INTO orders (customer_id, total) VALUES (%s, %s) RETURNING id",
            (order.customer_id, order.total)
        )
        order_id = cursor.fetchone()[0]
        for item in order.items:
            cursor.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, price) "
                "VALUES (%s, %s, %s, %s)",
                (order_id, item.product_id, item.quantity, item.price)
            )
        self._db.commit()
        return order_id


class OrderConfirmationNotifier:
    """注文確認通知のみ"""
    def __init__(self, email_sender):
        self._sender = email_sender

    def notify(self, order: Order) -> None:
        self._sender.send(
            to=f"customer_{order.customer_id}@example.com",
            subject=f"注文確認 #{order.id}",
            body=f"ご注文ありがとうございます。合計: ¥{order.total}"
        )


class CreateOrderUseCase:
    """オーケストレーション（各責任を組み合わせるだけ）"""
    def __init__(
        self,
        validator: OrderValidator,
        calculator: PriceCalculator,
        inventory: InventoryChecker,
        repository: OrderRepository,
        notifier: OrderConfirmationNotifier,
    ):
        self._validator = validator
        self._calculator = calculator
        self._inventory = inventory
        self._repository = repository
        self._notifier = notifier

    def execute(self, customer_id: int, items: list[OrderItem]) -> Order:
        # 1. バリデーション
        self._validator.validate(customer_id, items)

        # 2. 在庫確認
        self._inventory.check_availability(items)

        # 3. 価格計算
        subtotal, tax, shipping, total = self._calculator.calculate(items)

        # 4. 注文作成・保存
        order = Order(
            id=None,
            customer_id=customer_id,
            items=items,
            subtotal=subtotal,
            tax=tax,
            shipping=shipping,
            total=total,
        )
        order.id = self._repository.save(order)

        # 5. 通知
        self._notifier.notify(order)

        return order
```

```java
// Java: SRP の実践例 - ログ処理

// ❌ SRP違反: ログの取得・整形・出力が1クラスに集約
public class Logger {
    private final String logFile;
    private final String dbUrl;

    public Logger(String logFile, String dbUrl) {
        this.logFile = logFile;
        this.dbUrl = dbUrl;
    }

    public void log(String level, String message) {
        // 責任1: メッセージのフォーマット
        String timestamp = LocalDateTime.now()
            .format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
        String formatted = String.format("[%s] %s: %s", timestamp, level, message);

        // 責任2: ファイルへの出力
        try (FileWriter fw = new FileWriter(logFile, true)) {
            fw.write(formatted + "\n");
        } catch (IOException e) {
            System.err.println("Failed to write log: " + e.getMessage());
        }

        // 責任3: DBへの出力
        try (Connection conn = DriverManager.getConnection(dbUrl)) {
            PreparedStatement ps = conn.prepareStatement(
                "INSERT INTO logs (level, message, created_at) VALUES (?, ?, ?)"
            );
            ps.setString(1, level);
            ps.setString(2, message);
            ps.setTimestamp(3, Timestamp.valueOf(LocalDateTime.now()));
            ps.executeUpdate();
        } catch (SQLException e) {
            System.err.println("Failed to save log to DB: " + e.getMessage());
        }

        // 責任4: アラート送信（ERRORレベルの場合）
        if ("ERROR".equals(level)) {
            // Slack通知
            HttpClient client = HttpClient.newHttpClient();
            // ... Slack API呼び出し
        }
    }
}


// ✅ SRP適用: 各クラスが1つの責任

// フォーマット責任
public interface LogFormatter {
    String format(String level, String message);
}

public class TimestampLogFormatter implements LogFormatter {
    @Override
    public String format(String level, String message) {
        String timestamp = LocalDateTime.now()
            .format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
        return String.format("[%s] %s: %s", timestamp, level, message);
    }
}

public class JsonLogFormatter implements LogFormatter {
    @Override
    public String format(String level, String message) {
        return String.format(
            "{\"timestamp\":\"%s\",\"level\":\"%s\",\"message\":\"%s\"}",
            Instant.now(), level, message
        );
    }
}

// 出力先責任（インターフェースで抽象化 → OCPにもつながる）
public interface LogWriter {
    void write(String formattedMessage);
}

public class FileLogWriter implements LogWriter {
    private final String filePath;

    public FileLogWriter(String filePath) {
        this.filePath = filePath;
    }

    @Override
    public void write(String formattedMessage) {
        try (FileWriter fw = new FileWriter(filePath, true)) {
            fw.write(formattedMessage + "\n");
        } catch (IOException e) {
            System.err.println("File write failed: " + e.getMessage());
        }
    }
}

public class DatabaseLogWriter implements LogWriter {
    private final DataSource dataSource;

    public DatabaseLogWriter(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    @Override
    public void write(String formattedMessage) {
        try (Connection conn = dataSource.getConnection()) {
            PreparedStatement ps = conn.prepareStatement(
                "INSERT INTO logs (message) VALUES (?)"
            );
            ps.setString(1, formattedMessage);
            ps.executeUpdate();
        } catch (SQLException e) {
            System.err.println("DB write failed: " + e.getMessage());
        }
    }
}

// アラート責任
public interface AlertNotifier {
    void notify(String level, String message);
    boolean shouldNotify(String level);
}

public class SlackAlertNotifier implements AlertNotifier {
    @Override
    public boolean shouldNotify(String level) {
        return "ERROR".equals(level) || "FATAL".equals(level);
    }

    @Override
    public void notify(String level, String message) {
        // Slack API 呼び出し
    }
}

// オーケストレーター
public class Logger {
    private final LogFormatter formatter;
    private final List<LogWriter> writers;
    private final List<AlertNotifier> notifiers;

    public Logger(
        LogFormatter formatter,
        List<LogWriter> writers,
        List<AlertNotifier> notifiers
    ) {
        this.formatter = formatter;
        this.writers = writers;
        this.notifiers = notifiers;
    }

    public void log(String level, String message) {
        String formatted = formatter.format(level, message);

        for (LogWriter writer : writers) {
            writer.write(formatted);
        }

        for (AlertNotifier notifier : notifiers) {
            if (notifier.shouldNotify(level)) {
                notifier.notify(level, message);
            }
        }
    }
}
```

```kotlin
// Kotlin: SRP の実践例 - バリデーション

// ❌ SRP違反: バリデーションクラスが全ドメインのルールを知っている
class Validator {
    fun validateUser(user: UserDto): List<String> {
        val errors = mutableListOf<String>()
        if (user.name.isBlank()) errors.add("Name is required")
        if (!user.email.contains("@")) errors.add("Invalid email")
        if (user.age !in 18..120) errors.add("Invalid age")
        return errors
    }

    fun validateProduct(product: ProductDto): List<String> {
        val errors = mutableListOf<String>()
        if (product.name.isBlank()) errors.add("Name is required")
        if (product.price <= 0) errors.add("Price must be positive")
        if (product.stock < 0) errors.add("Stock cannot be negative")
        return errors
    }

    fun validateOrder(order: OrderDto): List<String> {
        val errors = mutableListOf<String>()
        if (order.items.isEmpty()) errors.add("Order must have items")
        if (order.total <= 0) errors.add("Total must be positive")
        return errors
    }
}


// ✅ SRP適用: ドメインごとにバリデーターを分離

// 汎用的なバリデーション結果
sealed class ValidationResult {
    object Valid : ValidationResult()
    data class Invalid(val errors: List<String>) : ValidationResult()
}

// バリデーターインターフェース
interface Validator<T> {
    fun validate(target: T): ValidationResult
}

// ルールベースのバリデーション
interface ValidationRule<T> {
    fun check(target: T): String?  // null = 問題なし、非null = エラーメッセージ
}

// ユーザーバリデーション
class UserNameRule : ValidationRule<UserDto> {
    override fun check(target: UserDto): String? =
        if (target.name.isBlank()) "Name is required" else null
}

class UserEmailRule : ValidationRule<UserDto> {
    override fun check(target: UserDto): String? =
        if (!target.email.contains("@")) "Invalid email format" else null
}

class UserAgeRule : ValidationRule<UserDto> {
    override fun check(target: UserDto): String? =
        if (target.age !in 18..120) "Age must be between 18 and 120" else null
}

class UserValidator(
    private val rules: List<ValidationRule<UserDto>> = listOf(
        UserNameRule(),
        UserEmailRule(),
        UserAgeRule(),
    )
) : Validator<UserDto> {
    override fun validate(target: UserDto): ValidationResult {
        val errors = rules.mapNotNull { it.check(target) }
        return if (errors.isEmpty()) ValidationResult.Valid
               else ValidationResult.Invalid(errors)
    }
}

// 商品バリデーション（独立した責任）
class ProductValidator(
    private val rules: List<ValidationRule<ProductDto>> = listOf(
        ProductNameRule(),
        ProductPriceRule(),
        ProductStockRule(),
    )
) : Validator<ProductDto> {
    override fun validate(target: ProductDto): ValidationResult {
        val errors = rules.mapNotNull { it.check(target) }
        return if (errors.isEmpty()) ValidationResult.Valid
               else ValidationResult.Invalid(errors)
    }
}

// 新しいバリデーションルールは ValidationRule を追加するだけ
// → OCP にもつながる
```

### 1.4 SRP 違反の検出方法

```
SRP違反を検出する5つのヒューリスティック:

  1. クラス名テスト:
     → クラス名に「And」「Or」「Manager」「Handler」が含まれる
     → 例: UserAndOrderManager → SRP違反の疑い
     → 対策: 責任ごとに名前を分ける

  2. 変更理由テスト:
     → 「このクラスを変更する理由を3つ挙げる」
     → 異なるビジネスドメインの理由が混在 → SRP違反
     → 例: "UIの変更" と "DBの変更" が同じクラス

  3. 説明テスト:
     → クラスの目的を1文で説明できない → SRP違反の疑い
     → 「〜と〜と〜をする」→ 責任が3つ
     → 「〜をする」→ 責任が1つ（SRP準拠）

  4. インポートテスト:
     → import文が多様なライブラリを参照 → SRP違反の疑い
     → 例: DB, HTTP, Email, ファイルシステム全てをimport
     → 各ライブラリの変更が影響する = 変更理由が複数

  5. コンストラクタテスト:
     → 依存注入のパラメータが5つ以上 → SRP違反の疑い
     → 多くの依存 = 多くの責任を持っている可能性
     → ただし、オーケストレーターは例外
```

```typescript
// SRP違反検出の具体例

// 🔍 クラス名テスト
class UserRegistrationAndNotificationService { } // ❌ And
class DataProcessingManager { }                   // ❌ Manager（曖昧すぎる）
class UserRegistrationService { }                 // ✅ 1つの責任

// 🔍 インポートテスト
// ❌ 多様すぎるインポート → SRP違反の兆候
import { Database } from './database';
import { SmtpClient } from './email';
import { S3Client } from 'aws-sdk';
import { RedisClient } from 'redis';
import { SlackWebhook } from './slack';
import { PdfGenerator } from './pdf';

class ReportService {
  constructor(
    private db: Database,        // DB依存
    private smtp: SmtpClient,    // メール依存
    private s3: S3Client,        // ストレージ依存
    private redis: RedisClient,  // キャッシュ依存
    private slack: SlackWebhook, // 通知依存
    private pdf: PdfGenerator,   // PDF生成依存
  ) {}
  // → 6つの異なる関心事に依存 = 6つの変更理由
}

// ✅ SRP適用後
class ReportDataFetcher {
  constructor(private db: Database, private redis: RedisClient) {}
}

class ReportGenerator {
  constructor(private pdf: PdfGenerator) {}
}

class ReportStorage {
  constructor(private s3: S3Client) {}
}

class ReportNotifier {
  constructor(private smtp: SmtpClient, private slack: SlackWebhook) {}
}
```

---

## 2. OCP: 開放閉鎖の原則

```
定義:
  「ソフトウェアの構成要素は、拡張に対して開き（Open）、
   修正に対して閉じている（Closed）べき」

つまり:
  → 新しい機能を追加するとき、既存のコードを変更しない
  → ポリモーフィズム（インターフェース + 実装クラス）で実現

なぜ重要か:
  → 既存コードを変更するとリグレッションのリスク
  → テスト済みのコードに触らずに済む
  → チーム開発でのコンフリクト減少
```

### 2.1 OCP の実現方法

```
OCP を実現する4つのパターン:

  1. Strategy パターン（最も基本的）:
     → インターフェースを定義し、実装クラスを追加
     → 利用側は switch/if を使わずインターフェースを呼ぶ

  2. Template Method パターン:
     → 基底クラスでアルゴリズムの骨格を定義
     → サブクラスで詳細をオーバーライド

  3. Decorator パターン:
     → 既存クラスをラップして機能を追加
     → 元のクラスのコードは一切変更しない

  4. Plugin / Registry パターン:
     → 実行時に実装を動的に登録
     → 新しい実装はプラグインとして追加

  適用基準:
    変更が発生していない箇所 → まだOCPは不要（YAGNI）
    同じ種類の変更が2-3回発生 → OCPを適用する時期
```

### 2.2 OCP リファクタリング

```typescript
// ❌ OCP違反: 新しい通知手段を追加するたびに修正が必要
class NotificationService {
  send(type: string, message: string, recipient: string): void {
    if (type === "email") {
      // メール送信処理
      emailClient.send(recipient, message);
    } else if (type === "sms") {
      // SMS送信処理
      smsClient.send(recipient, message);
    } else if (type === "slack") {
      // Slack送信処理（新規追加するたびにここを修正）
      slackClient.post(recipient, message);
    }
    // LINE追加？ Discord追加？ → ここを修正し続ける...
  }
}

// ✅ OCP適用: 新しい通知手段はクラスを追加するだけ
interface NotificationChannel {
  send(message: string, recipient: string): Promise<void>;
}

class EmailChannel implements NotificationChannel {
  async send(message: string, recipient: string): Promise<void> {
    await emailClient.send(recipient, message);
  }
}

class SmsChannel implements NotificationChannel {
  async send(message: string, recipient: string): Promise<void> {
    await smsClient.send(recipient, message);
  }
}

class SlackChannel implements NotificationChannel {
  async send(message: string, recipient: string): Promise<void> {
    await slackClient.post(recipient, message);
  }
}

// LINE追加 → LineChannel クラスを追加するだけ
// NotificationService は一切変更不要

class NotificationService {
  constructor(private channels: NotificationChannel[]) {}

  async sendAll(message: string, recipient: string): Promise<void> {
    await Promise.all(
      this.channels.map(ch => ch.send(message, recipient))
    );
  }
}
```

### 2.3 OCP の多言語実践例

```python
# Python: OCP の実践例 - レポートエンジン

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# ❌ OCP違反: 新しいデータソースやフォーマットで修正が必要
class ReportEngine:
    def generate(
        self, source: str, format_type: str, filters: dict
    ) -> str:
        # データ取得（ソースの種類で分岐）
        if source == "mysql":
            data = self._fetch_from_mysql(filters)
        elif source == "mongodb":
            data = self._fetch_from_mongodb(filters)
        elif source == "api":
            data = self._fetch_from_api(filters)
        else:
            raise ValueError(f"Unknown source: {source}")

        # フォーマット（形式の種類で分岐）
        if format_type == "pdf":
            return self._format_as_pdf(data)
        elif format_type == "excel":
            return self._format_as_excel(data)
        elif format_type == "html":
            return self._format_as_html(data)
        else:
            raise ValueError(f"Unknown format: {format_type}")

    # source ごとに private メソッドが増え続ける...
    def _fetch_from_mysql(self, filters): ...
    def _fetch_from_mongodb(self, filters): ...
    def _fetch_from_api(self, filters): ...
    def _format_as_pdf(self, data): ...
    def _format_as_excel(self, data): ...
    def _format_as_html(self, data): ...


# ✅ OCP適用: データソースとフォーマッターを拡張可能に

@dataclass
class ReportData:
    """レポートデータの共通表現"""
    headers: list[str]
    rows: list[list[Any]]
    metadata: dict[str, Any]


class DataSource(ABC):
    """データソース抽象"""
    @abstractmethod
    def fetch(self, filters: dict) -> ReportData: ...


class MySQLDataSource(DataSource):
    def __init__(self, connection_string: str):
        self._conn_str = connection_string

    def fetch(self, filters: dict) -> ReportData:
        # MySQL からデータ取得
        import mysql.connector
        conn = mysql.connector.connect(self._conn_str)
        cursor = conn.cursor()
        query = self._build_query(filters)
        cursor.execute(query)
        headers = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return ReportData(headers=headers, rows=rows, metadata={"source": "mysql"})

    def _build_query(self, filters: dict) -> str:
        # クエリ構築
        return "SELECT * FROM reports"


class MongoDBDataSource(DataSource):
    def __init__(self, uri: str, database: str):
        self._uri = uri
        self._database = database

    def fetch(self, filters: dict) -> ReportData:
        from pymongo import MongoClient
        client = MongoClient(self._uri)
        db = client[self._database]
        documents = list(db.reports.find(filters))
        if not documents:
            return ReportData(headers=[], rows=[], metadata={})
        headers = list(documents[0].keys())
        rows = [[doc.get(h) for h in headers] for doc in documents]
        return ReportData(headers=headers, rows=rows, metadata={"source": "mongodb"})


class RestApiDataSource(DataSource):
    """REST API からデータ取得 - 新規追加でも既存コード変更なし"""
    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url
        self._api_key = api_key

    def fetch(self, filters: dict) -> ReportData:
        import requests
        response = requests.get(
            f"{self._base_url}/data",
            headers={"Authorization": f"Bearer {self._api_key}"},
            params=filters,
        )
        data = response.json()
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        return ReportData(headers=headers, rows=rows, metadata={"source": "api"})


class ReportFormatter(ABC):
    """フォーマッター抽象"""
    @abstractmethod
    def format(self, data: ReportData) -> bytes: ...

    @abstractmethod
    def content_type(self) -> str: ...

    @abstractmethod
    def file_extension(self) -> str: ...


class PdfFormatter(ReportFormatter):
    def format(self, data: ReportData) -> bytes:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table
        # PDF生成ロジック
        return b"<pdf content>"

    def content_type(self) -> str:
        return "application/pdf"

    def file_extension(self) -> str:
        return ".pdf"


class ExcelFormatter(ReportFormatter):
    def format(self, data: ReportData) -> bytes:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(data.headers)
        for row in data.rows:
            ws.append(row)
        # Excel生成ロジック
        return b"<excel content>"

    def content_type(self) -> str:
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    def file_extension(self) -> str:
        return ".xlsx"


class HtmlFormatter(ReportFormatter):
    def format(self, data: ReportData) -> bytes:
        html = "<table>\n<tr>"
        html += "".join(f"<th>{h}</th>" for h in data.headers)
        html += "</tr>\n"
        for row in data.rows:
            html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>\n"
        html += "</table>"
        return html.encode("utf-8")

    def content_type(self) -> str:
        return "text/html"

    def file_extension(self) -> str:
        return ".html"


class ReportEngine:
    """このクラスは新しいデータソースやフォーマットが追加されても変更不要"""
    def __init__(self, source: DataSource, formatter: ReportFormatter):
        self._source = source
        self._formatter = formatter

    def generate(self, filters: dict | None = None) -> bytes:
        data = self._source.fetch(filters or {})
        return self._formatter.format(data)

    def generate_to_file(self, filename: str, filters: dict | None = None) -> str:
        content = self.generate(filters)
        filepath = f"{filename}{self._formatter.file_extension()}"
        with open(filepath, "wb") as f:
            f.write(content)
        return filepath


# 使用例: 組み合わせ自由
engine = ReportEngine(
    source=MySQLDataSource("mysql://localhost/mydb"),
    formatter=PdfFormatter(),
)
engine.generate_to_file("monthly_report")

# 新しい組み合わせも既存コード変更なし
engine2 = ReportEngine(
    source=RestApiDataSource("https://api.example.com", "key123"),
    formatter=ExcelFormatter(),
)
```

```java
// Java: OCP の実践例 - 認証パイプライン

// ❌ OCP違反
public class AuthService {
    public boolean authenticate(String method, String credentials) {
        if ("password".equals(method)) {
            // パスワード認証
            String[] parts = credentials.split(":");
            return checkPassword(parts[0], parts[1]);
        } else if ("oauth".equals(method)) {
            // OAuth認証
            return verifyOAuthToken(credentials);
        } else if ("api_key".equals(method)) {
            // APIキー認証
            return validateApiKey(credentials);
        } else if ("certificate".equals(method)) {
            // 証明書認証（追加のたびにここを修正）
            return verifyCertificate(credentials);
        }
        throw new IllegalArgumentException("Unknown method: " + method);
    }
}


// ✅ OCP適用: 認証方法はプラグインとして追加

public interface AuthenticationStrategy {
    boolean authenticate(AuthRequest request);
    boolean supports(String method);
}

public class PasswordAuthentication implements AuthenticationStrategy {
    private final PasswordEncoder encoder;
    private final UserRepository userRepo;

    public PasswordAuthentication(PasswordEncoder encoder, UserRepository userRepo) {
        this.encoder = encoder;
        this.userRepo = userRepo;
    }

    @Override
    public boolean supports(String method) {
        return "password".equals(method);
    }

    @Override
    public boolean authenticate(AuthRequest request) {
        User user = userRepo.findByUsername(request.getUsername());
        if (user == null) return false;
        return encoder.matches(request.getCredentials(), user.getPasswordHash());
    }
}

public class OAuthAuthentication implements AuthenticationStrategy {
    private final OAuthTokenVerifier verifier;

    public OAuthAuthentication(OAuthTokenVerifier verifier) {
        this.verifier = verifier;
    }

    @Override
    public boolean supports(String method) {
        return "oauth".equals(method);
    }

    @Override
    public boolean authenticate(AuthRequest request) {
        return verifier.verify(request.getCredentials());
    }
}

public class ApiKeyAuthentication implements AuthenticationStrategy {
    private final ApiKeyRepository keyRepo;

    public ApiKeyAuthentication(ApiKeyRepository keyRepo) {
        this.keyRepo = keyRepo;
    }

    @Override
    public boolean supports(String method) {
        return "api_key".equals(method);
    }

    @Override
    public boolean authenticate(AuthRequest request) {
        return keyRepo.isValid(request.getCredentials());
    }
}

// 認証サービス: 新しい認証方法が追加されても変更不要
public class AuthService {
    private final List<AuthenticationStrategy> strategies;

    public AuthService(List<AuthenticationStrategy> strategies) {
        this.strategies = strategies;
    }

    public boolean authenticate(String method, AuthRequest request) {
        return strategies.stream()
            .filter(s -> s.supports(method))
            .findFirst()
            .map(s -> s.authenticate(request))
            .orElseThrow(() ->
                new IllegalArgumentException("Unsupported auth method: " + method)
            );
    }
}

// Spring Boot での設定例
@Configuration
public class AuthConfig {
    @Bean
    public AuthService authService(
        PasswordAuthentication password,
        OAuthAuthentication oauth,
        ApiKeyAuthentication apiKey
    ) {
        return new AuthService(List.of(password, oauth, apiKey));
    }
}
```

### 2.4 OCP のもう一つの実現方法: デコレータ

```python
# Python: デコレータによるOCP
class Logger:
    """既存クラスを変更せずにログ機能を追加"""
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name):
        original = getattr(self._wrapped, name)
        if callable(original):
            def wrapper(*args, **kwargs):
                print(f"[LOG] {name} called with {args}")
                result = original(*args, **kwargs)
                print(f"[LOG] {name} returned {result}")
                return result
            return wrapper
        return original

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

# Calculator を変更せずにログ機能を追加
calc = Logger(Calculator())
calc.add(1, 2)
# [LOG] add called with (1, 2)
# [LOG] add returned 3
```

```typescript
// TypeScript: デコレータパターンによるOCP

// 基本インターフェース
interface HttpClient {
  get(url: string): Promise<Response>;
  post(url: string, body: any): Promise<Response>;
}

// 基本実装
class FetchHttpClient implements HttpClient {
  async get(url: string): Promise<Response> {
    return fetch(url);
  }

  async post(url: string, body: any): Promise<Response> {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }
}

// デコレータ1: ログ追加（FetchHttpClient を変更しない）
class LoggingHttpClient implements HttpClient {
  constructor(private inner: HttpClient) {}

  async get(url: string): Promise<Response> {
    console.log(`[GET] ${url}`);
    const start = Date.now();
    const response = await this.inner.get(url);
    console.log(`[GET] ${url} → ${response.status} (${Date.now() - start}ms)`);
    return response;
  }

  async post(url: string, body: any): Promise<Response> {
    console.log(`[POST] ${url}`, body);
    const start = Date.now();
    const response = await this.inner.post(url, body);
    console.log(`[POST] ${url} → ${response.status} (${Date.now() - start}ms)`);
    return response;
  }
}

// デコレータ2: リトライ追加（FetchHttpClient を変更しない）
class RetryHttpClient implements HttpClient {
  constructor(
    private inner: HttpClient,
    private maxRetries: number = 3,
  ) {}

  async get(url: string): Promise<Response> {
    return this.withRetry(() => this.inner.get(url));
  }

  async post(url: string, body: any): Promise<Response> {
    return this.withRetry(() => this.inner.post(url, body));
  }

  private async withRetry(fn: () => Promise<Response>): Promise<Response> {
    let lastError: Error | null = null;
    for (let i = 0; i <= this.maxRetries; i++) {
      try {
        const response = await fn();
        if (response.ok) return response;
        if (response.status >= 500) {
          lastError = new Error(`Server error: ${response.status}`);
          continue;
        }
        return response; // 4xx はリトライしない
      } catch (error) {
        lastError = error as Error;
      }
    }
    throw lastError;
  }
}

// デコレータ3: キャッシュ追加
class CachingHttpClient implements HttpClient {
  private cache = new Map<string, { response: Response; expiry: number }>();

  constructor(
    private inner: HttpClient,
    private ttlMs: number = 60_000,
  ) {}

  async get(url: string): Promise<Response> {
    const cached = this.cache.get(url);
    if (cached && cached.expiry > Date.now()) {
      return cached.response.clone();
    }
    const response = await this.inner.get(url);
    this.cache.set(url, { response: response.clone(), expiry: Date.now() + this.ttlMs });
    return response;
  }

  async post(url: string, body: any): Promise<Response> {
    // POST はキャッシュしない
    return this.inner.post(url, body);
  }
}

// 使用例: デコレータを組み合わせて機能を追加
// 既存の FetchHttpClient は一切変更していない
const client: HttpClient = new CachingHttpClient(
  new RetryHttpClient(
    new LoggingHttpClient(
      new FetchHttpClient()
    ),
    3,
  ),
  30_000,
);

// リクエスト → Caching → Retry → Logging → Fetch の順に処理
await client.get("https://api.example.com/data");
```

```python
# Python: デコレータパターン - ミドルウェアパイプライン

from abc import ABC, abstractmethod
from typing import Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class Request:
    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None


@dataclass
class Response:
    status: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)


# ミドルウェアインターフェース
class Middleware(ABC):
    @abstractmethod
    def process(
        self, request: Request, next_handler: Callable[[Request], Response]
    ) -> Response:
        ...


# 認証ミドルウェア
class AuthMiddleware(Middleware):
    def __init__(self, token_verifier):
        self._verifier = token_verifier

    def process(self, request: Request, next_handler):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token or not self._verifier.verify(token):
            return Response(status=401, body={"error": "Unauthorized"})
        return next_handler(request)


# ログミドルウェア
class LoggingMiddleware(Middleware):
    def process(self, request: Request, next_handler):
        start = time.time()
        print(f"→ {request.method} {request.path}")
        response = next_handler(request)
        elapsed = time.time() - start
        print(f"← {response.status} ({elapsed:.3f}s)")
        return response


# レート制限ミドルウェア
class RateLimitMiddleware(Middleware):
    def __init__(self, max_requests: int, window_seconds: int):
        self._max = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = {}

    def process(self, request: Request, next_handler):
        client_ip = request.headers.get("X-Real-IP", "unknown")
        now = time.time()
        requests = self._requests.setdefault(client_ip, [])
        requests = [t for t in requests if now - t < self._window]
        self._requests[client_ip] = requests

        if len(requests) >= self._max:
            return Response(status=429, body={"error": "Too many requests"})

        requests.append(now)
        return next_handler(request)


# CORS ミドルウェア
class CorsMiddleware(Middleware):
    def __init__(self, allowed_origins: list[str]):
        self._origins = allowed_origins

    def process(self, request: Request, next_handler):
        response = next_handler(request)
        origin = request.headers.get("Origin", "")
        if origin in self._origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE"
        return response


# パイプライン: ミドルウェアを組み合わせ（OCP達成）
class MiddlewarePipeline:
    """新しいミドルウェアは Middleware クラスを追加するだけ"""
    def __init__(self, handler: Callable[[Request], Response]):
        self._handler = handler
        self._middlewares: list[Middleware] = []

    def use(self, middleware: Middleware) -> "MiddlewarePipeline":
        self._middlewares.append(middleware)
        return self

    def handle(self, request: Request) -> Response:
        def build_chain(index: int) -> Callable[[Request], Response]:
            if index >= len(self._middlewares):
                return self._handler
            middleware = self._middlewares[index]
            return lambda req: middleware.process(req, build_chain(index + 1))

        return build_chain(0)(request)


# 使用例
def app_handler(request: Request) -> Response:
    return Response(status=200, body={"message": "Hello!"})

pipeline = (
    MiddlewarePipeline(app_handler)
    .use(LoggingMiddleware())
    .use(CorsMiddleware(["https://example.com"]))
    .use(RateLimitMiddleware(max_requests=100, window_seconds=60))
    .use(AuthMiddleware(token_verifier))
)

response = pipeline.handle(Request(method="GET", path="/api/data"))
```

### 2.5 OCP 違反の検出方法

```
OCP違反を検出する4つのヒューリスティック:

  1. switch/if-else チェーン:
     → 同じ変数に対する type チェックが複数箇所に散在
     → 新しい type を追加するとき、全ての switch を修正する必要がある
     → Shotgun Surgery（散弾銃手術）の兆候

  2. instanceof / type チェック:
     → if (obj instanceof SomeClass) が頻出
     → ポリモーフィズムで解決すべき箇所

  3. 変更履歴の分析:
     → git log で「同じファイルが異なる機能追加で繰り返し修正」
     → → OCP適用の候補

  4. コメント「// 新しい〜を追加する場合はここに追記」:
     → 修正箇所をコメントで示す必要がある = OCP違反
     → 本来は「クラスを追加するだけ」であるべき

  検出コマンド例:
    # 同じファイルが頻繁に変更されている箇所を特定
    git log --format=format: --name-only --since="6 months ago" | \
      sort | uniq -c | sort -rn | head -20
```

---

## 3. SRP と OCP の関係

```
SRP → クラスを小さく分割
  ↓
OCP → 小さなクラスをインターフェースで接続
  ↓
結果: 拡張が容易で変更の影響が局所的な設計

実践の流れ:
  1. SRP で責任を分離
  2. 変化しやすい部分を特定
  3. OCP でインターフェースを設計
  4. 新しい要件はクラスを追加して対応
```

### 3.1 SRP + OCP の連携パターン

```typescript
// SRP と OCP が連携する実践例: 請求書システム

// Step 1: SRP で責任を分離する

// 請求書データ
interface Invoice {
  id: string;
  items: InvoiceItem[];
  customer: Customer;
  issuedAt: Date;
  dueDate: Date;
}

// 税金計算（SRP: 税金計算のみ）
interface TaxCalculator {
  calculate(items: InvoiceItem[]): number;
}

// 割引適用（SRP: 割引計算のみ）
interface DiscountPolicy {
  apply(subtotal: number, customer: Customer): number;
}

// フォーマット（SRP: フォーマットのみ）
interface InvoiceFormatter {
  format(invoice: Invoice, total: number): string;
}

// 送信（SRP: 送信のみ）
interface InvoiceSender {
  send(invoice: Invoice, formatted: string): Promise<void>;
}

// Step 2: OCP で各責任を拡張可能にする

// 税金計算: 国ごとの税制に対応
class JapaneseTaxCalculator implements TaxCalculator {
  calculate(items: InvoiceItem[]): number {
    const subtotal = items.reduce((sum, i) => sum + i.amount, 0);
    return Math.floor(subtotal * 0.10); // 消費税10%
  }
}

class USStateTaxCalculator implements TaxCalculator {
  constructor(private stateRate: number) {}
  calculate(items: InvoiceItem[]): number {
    const subtotal = items.reduce((sum, i) => sum + i.amount, 0);
    return Math.floor(subtotal * this.stateRate);
  }
}

// 割引: ビジネスルールに応じた割引
class VolumeDiscount implements DiscountPolicy {
  apply(subtotal: number, customer: Customer): number {
    if (subtotal > 100000) return subtotal * 0.05; // 5%割引
    return 0;
  }
}

class LoyaltyDiscount implements DiscountPolicy {
  apply(subtotal: number, customer: Customer): number {
    if (customer.memberSince.getFullYear() < 2020) return subtotal * 0.03;
    return 0;
  }
}

class CompositeDiscount implements DiscountPolicy {
  constructor(private policies: DiscountPolicy[]) {}
  apply(subtotal: number, customer: Customer): number {
    return this.policies.reduce(
      (total, policy) => total + policy.apply(subtotal, customer),
      0
    );
  }
}

// フォーマット: 出力形式
class PdfInvoiceFormatter implements InvoiceFormatter {
  format(invoice: Invoice, total: number): string {
    // PDF生成ロジック
    return `<pdf-data>Invoice ${invoice.id}: ¥${total}</pdf-data>`;
  }
}

class HtmlInvoiceFormatter implements InvoiceFormatter {
  format(invoice: Invoice, total: number): string {
    return `<html><h1>Invoice ${invoice.id}</h1><p>Total: ¥${total}</p></html>`;
  }
}

// 送信: 送信手段
class EmailInvoiceSender implements InvoiceSender {
  async send(invoice: Invoice, formatted: string): Promise<void> {
    await emailClient.send(invoice.customer.email, "Invoice", formatted);
  }
}

class FaxInvoiceSender implements InvoiceSender {
  async send(invoice: Invoice, formatted: string): Promise<void> {
    await faxService.send(invoice.customer.faxNumber, formatted);
  }
}

// Step 3: オーケストレーター（SRP: 調整のみ）
class InvoiceService {
  constructor(
    private taxCalc: TaxCalculator,
    private discount: DiscountPolicy,
    private formatter: InvoiceFormatter,
    private sender: InvoiceSender,
  ) {}

  async processInvoice(invoice: Invoice): Promise<void> {
    const subtotal = invoice.items.reduce((sum, i) => sum + i.amount, 0);
    const tax = this.taxCalc.calculate(invoice.items);
    const discountAmount = this.discount.apply(subtotal, invoice.customer);
    const total = subtotal + tax - discountAmount;

    const formatted = this.formatter.format(invoice, total);
    await this.sender.send(invoice, formatted);
  }
}

// 使用例: 日本の顧客向け、PDF形式、メール送信
const jpService = new InvoiceService(
  new JapaneseTaxCalculator(),
  new CompositeDiscount([new VolumeDiscount(), new LoyaltyDiscount()]),
  new PdfInvoiceFormatter(),
  new EmailInvoiceSender(),
);

// 使用例: US顧客向け、HTML形式、FAX送信
const usService = new InvoiceService(
  new USStateTaxCalculator(0.08),
  new VolumeDiscount(),
  new HtmlInvoiceFormatter(),
  new FaxInvoiceSender(),
);

// 新しい税制・割引・フォーマット・送信手段 → クラスを追加するだけ
// InvoiceService は一切変更不要
```

### 3.2 テスト容易性の向上

```typescript
// SRP + OCP がテストを劇的に簡単にする

// テスト用モック
class MockTaxCalculator implements TaxCalculator {
  calculate(items: InvoiceItem[]): number {
    return 1000; // 固定値で予測可能に
  }
}

class MockDiscountPolicy implements DiscountPolicy {
  apply(subtotal: number, customer: Customer): number {
    return 0; // 割引なし
  }
}

class MockInvoiceFormatter implements InvoiceFormatter {
  lastInvoice?: Invoice;
  lastTotal?: number;

  format(invoice: Invoice, total: number): string {
    this.lastInvoice = invoice;
    this.lastTotal = total;
    return "formatted-invoice";
  }
}

class MockInvoiceSender implements InvoiceSender {
  sentInvoices: Array<{ invoice: Invoice; formatted: string }> = [];

  async send(invoice: Invoice, formatted: string): Promise<void> {
    this.sentInvoices.push({ invoice, formatted });
  }
}

// テストコード
describe("InvoiceService", () => {
  let service: InvoiceService;
  let mockFormatter: MockInvoiceFormatter;
  let mockSender: MockInvoiceSender;

  beforeEach(() => {
    mockFormatter = new MockInvoiceFormatter();
    mockSender = new MockInvoiceSender();
    service = new InvoiceService(
      new MockTaxCalculator(),
      new MockDiscountPolicy(),
      mockFormatter,
      mockSender,
    );
  });

  it("should calculate total correctly", async () => {
    const invoice = createTestInvoice([
      { name: "Item A", amount: 5000 },
      { name: "Item B", amount: 3000 },
    ]);

    await service.processInvoice(invoice);

    // subtotal(8000) + tax(1000) - discount(0) = 9000
    expect(mockFormatter.lastTotal).toBe(9000);
  });

  it("should send formatted invoice", async () => {
    const invoice = createTestInvoice([{ name: "Item A", amount: 5000 }]);

    await service.processInvoice(invoice);

    expect(mockSender.sentInvoices).toHaveLength(1);
    expect(mockSender.sentInvoices[0].formatted).toBe("formatted-invoice");
  });
});

// 各コンポーネントも独立してテスト可能
describe("JapaneseTaxCalculator", () => {
  const calc = new JapaneseTaxCalculator();

  it("should calculate 10% tax", () => {
    const items = [{ name: "Item", amount: 10000 }];
    expect(calc.calculate(items)).toBe(1000);
  });
});

describe("VolumeDiscount", () => {
  const discount = new VolumeDiscount();

  it("should apply 5% discount for orders over 100000", () => {
    const customer = createTestCustomer();
    expect(discount.apply(200000, customer)).toBe(10000);
  });

  it("should not apply discount for small orders", () => {
    const customer = createTestCustomer();
    expect(discount.apply(50000, customer)).toBe(0);
  });
});
```

---

## 4. アンチパターンと注意点

```
SRP の過剰適用:
  → 1メソッドだけのクラスが大量発生
  → ファイル数が爆発してナビゲーション困難
  → 対策: 「変更する理由」で分割。メソッド数ではない

OCP の過剰適用:
  → 変更されない部分まで抽象化
  → 不要なインターフェースだらけ
  → 対策: 「実際に変更が発生してから」抽象化する

判断基準:
  「このクラスが変更される理由は何か？」
  → 理由が複数ある → SRP で分割
  「この部分は今後変更される可能性があるか？」
  → ある → OCP でインターフェースを導入
  → ない → そのままでよい（YAGNI）
```

### 4.1 SRP の過剰適用例

```typescript
// ❌ SRP の過剰適用: 不必要な分割

// 1文字列の結合のためだけにクラスを作る必要はない
class StringConcatenator {
  concatenate(a: string, b: string): string {
    return a + b;
  }
}

// 加算のためだけにクラスを作る必要はない
class NumberAdder {
  add(a: number, b: number): number {
    return a + b;
  }
}

// nullチェックのためだけにクラスを作る必要はない
class NullChecker {
  isNull(value: any): boolean {
    return value === null || value === undefined;
  }
}

// ✅ 適切な粒度: 関連する操作をまとめたクラス
class MathUtils {
  static add(a: number, b: number): number { return a + b; }
  static subtract(a: number, b: number): number { return a - b; }
  static multiply(a: number, b: number): number { return a * b; }
  static divide(a: number, b: number): number {
    if (b === 0) throw new Error("Division by zero");
    return a / b;
  }
}
// → 変更理由: 「数学計算のルール変更」→ 1つのアクター
// → メソッドは4つだが責任は1つ = SRP準拠
```

### 4.2 OCP の過剰適用例

```typescript
// ❌ OCP の過剰適用: 変更が発生しない部分まで抽象化

// 環境設定の読み取り: 変更される可能性が低い
interface ConfigReader { read(): Config; }
interface ConfigParser { parse(raw: string): Config; }
interface ConfigValidator { validate(config: Config): void; }
interface ConfigMerger { merge(base: Config, override: Config): Config; }

// → 設定ファイルの読み取り方法が頻繁に変わることはない
// → 4つのインターフェースは過剰

// ✅ 適切な抽象化レベル
class ConfigLoader {
  load(path: string): Config {
    const raw = fs.readFileSync(path, "utf-8");
    const config = JSON.parse(raw);
    this.validate(config);
    return config;
  }

  private validate(config: Config): void {
    if (!config.port) throw new Error("port is required");
    if (!config.dbUrl) throw new Error("dbUrl is required");
  }
}
// → 設定の読み取りが変更される頻度は低い
// → シンプルなクラスで十分
// → 将来変更が必要になったら、その時に抽象化する
```

### 4.3 実務での判断フローチャート

```
SRP 適用の判断フロー:

  クラスの行数 > 300?
    │
    ├── Yes → 変更理由を分析
    │         │
    │         ├── 変更理由が複数 → SRP適用（分割する）
    │         └── 変更理由が1つ → 大きくてもOK（責任は1つ）
    │
    └── No → 複数のドメインを混在させていないか？
              │
              ├── Yes → SRP適用（小さくても分割すべき）
              └── No → 現状維持でOK


OCP 適用の判断フロー:

  同じ種類の変更が2回以上発生した？
    │
    ├── Yes → switch/if-elseの分岐が増えている？
    │         │
    │         ├── Yes → OCP適用（インターフェース導入）
    │         └── No → もう1回変更が来たら適用を検討
    │
    └── No → 変更されていない
              → 現状維持（YAGNI）
              → 抽象化は「投機的」にしない
```

---

## 5. フレームワークにおけるSRP + OCP

```
主要フレームワークでの SRP + OCP の活用例:

  NestJS (TypeScript):
    SRP → Controller, Service, Repository の分離
    OCP → @Injectable() による DI、Guard / Interceptor / Pipe

  Spring Boot (Java):
    SRP → @Controller, @Service, @Repository アノテーション
    OCP → @Bean 定義、@Profile による環境切り替え

  Django (Python):
    SRP → views.py, models.py, serializers.py の分離
    OCP → Middleware クラス、カスタム Backend

  Rails (Ruby):
    SRP → Model, Controller, Service Object パターン
    OCP → Concern モジュール、ActiveSupport::Concern
```

```typescript
// NestJS: SRP + OCP の実践例

// Controller（SRP: HTTPリクエスト処理のみ）
@Controller("orders")
class OrderController {
  constructor(private readonly orderService: OrderService) {}

  @Post()
  async createOrder(@Body() dto: CreateOrderDto): Promise<OrderResponse> {
    return this.orderService.create(dto);
  }

  @Get(":id")
  async getOrder(@Param("id") id: string): Promise<OrderResponse> {
    return this.orderService.findById(id);
  }
}

// Service（SRP: ビジネスロジックのみ）
@Injectable()
class OrderService {
  constructor(
    private readonly repo: OrderRepository,
    @Inject("PAYMENT_GATEWAY") private readonly payment: PaymentGateway,
    @Inject("NOTIFIER") private readonly notifier: Notifier,
  ) {}

  async create(dto: CreateOrderDto): Promise<OrderResponse> {
    const order = Order.create(dto);
    await this.payment.charge(order.total, order.id);
    await this.repo.save(order);
    await this.notifier.notify(order);
    return OrderResponse.fromEntity(order);
  }
}

// Module（OCP: 依存の差し替えが容易）
@Module({
  providers: [
    OrderService,
    {
      provide: "PAYMENT_GATEWAY",
      useClass: process.env.NODE_ENV === "test"
        ? MockPaymentGateway
        : StripePaymentGateway,
    },
    {
      provide: "NOTIFIER",
      useClass: process.env.NODE_ENV === "test"
        ? MockNotifier
        : EmailNotifier,
    },
  ],
})
class OrderModule {}

// Guard（OCP: 認証ロジックをプラグイン的に追加）
@Injectable()
class AuthGuard implements CanActivate {
  canActivate(context: ExecutionContext): boolean {
    const request = context.switchToHttp().getRequest();
    return this.validateToken(request.headers.authorization);
  }
}

// Interceptor（OCP: 横断的関心事をデコレータ的に追加）
@Injectable()
class LoggingInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const now = Date.now();
    return next.handle().pipe(
      tap(() => console.log(`Response time: ${Date.now() - now}ms`)),
    );
  }
}

// Pipe（OCP: バリデーションをプラグイン的に追加）
@Injectable()
class OrderValidationPipe implements PipeTransform {
  transform(value: any): CreateOrderDto {
    const dto = plainToClass(CreateOrderDto, value);
    const errors = validateSync(dto);
    if (errors.length > 0) {
      throw new BadRequestException(errors);
    }
    return dto;
  }
}
```

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| 原則 | 核心 | 実現手段 | 注意 | 検出方法 |
|------|------|---------|------|---------|
| SRP | 1クラス1責任 | 責任の分離、委譲 | 過剰分割に注意 | クラス名・インポートテスト |
| OCP | 拡張は開、修正は閉 | インターフェース、ポリモーフィズム | 必要になってから抽象化 | switch/if-else連鎖の検出 |

### SRP + OCP 適用チェックリスト

```
□ クラスの変更理由が1つだけか（SRP）
□ クラス名が1つの責任を表しているか（SRP）
□ import/依存が1つのドメインに限定されているか（SRP）
□ コンストラクタの引数が5つ以下か（SRP）
□ 同じ種類の分岐が複数箇所に散在していないか（OCP）
□ 新しい種類の追加でコメント「ここに追記」が必要ないか（OCP）
□ テスト時にモックに差し替え可能か（SRP + OCP）
□ 各クラスが独立してテスト可能か（SRP）
□ 変更が1クラスに閉じるか（SRP + OCP）
□ git log で同じファイルが頻繁に修正されていないか（OCP）
```

---

## 次に読むべきガイド

---

## 参考文献
1. Martin, R. "Clean Architecture: A Craftsman's Guide to Software Structure and Design." Chapter 7-8, Prentice Hall, 2017.
2. Martin, R. "The Single Responsibility Principle." The Clean Coder Blog, 2014.
3. Martin, R. "Agile Software Development, Principles, Patterns, and Practices." Prentice Hall, 2003.
4. Meyer, B. "Object-Oriented Software Construction." Prentice Hall, 2nd ed., 1997.
5. Fowler, M. "Refactoring: Improving the Design of Existing Code." Addison-Wesley, 2nd ed., 2018.
6. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
7. Freeman, S. and Pryce, N. "Growing Object-Oriented Software, Guided by Tests." Addison-Wesley, 2009.
