# SRP (Single Responsibility Principle) + OCP (Open/Closed Principle)

> SRP says "there should be only one reason to change," OCP says "extend without modifying." Together, these two principles form the foundation of maintainable design.

## What You Will Learn in This Chapter

- [ ] Understand the correct definition of "responsibility" in SRP
- [ ] Grasp how to achieve OCP through polymorphism
- [ ] Learn practical refactoring techniques
- [ ] Be able to detect SRP and OCP violation patterns
- [ ] Master SRP/OCP application patterns across multiple languages
- [ ] Learn how to apply them incrementally in real-world projects


## Prerequisites

Before reading this guide, you will gain a deeper understanding if you already have the following knowledge:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [SOLID Principles Overview](./00-solid-overview.md)

---

## 1. SRP: Single Responsibility Principle

```
Definition (Robert C. Martin):
  "A class should have only one reason to change."

More precise definition:
  "A class should be responsible to only one actor (stakeholder)."

  Example:
    If an Employee class has:
    - calculatePay()    -> Responsibility of the CFO (Accounting)
    - reportHours()     -> Responsibility of the COO (Operations)
    - save()            -> Responsibility of the CTO (Engineering)

    -> Depends on 3 actors = SRP violation
    -> A change request from Accounting may affect Operations code
```

### 1.1 What "Responsibility" Means in SRP

```
Common misconceptions vs. correct understanding:

  [Wrong] "A class should only have one method"
    -> Judging by method count is incorrect
    -> Even with 100 methods, if the "responsibility is one," SRP is satisfied

  [Wrong] "Do only one thing"
    -> The granularity of "one thing" varies with abstraction level
    -> What counts as "one" is ambiguous

  [Correct] "There should be only one reason to change"
    -> "Who (which actor) wants to change this class?"
    -> If only one actor -> SRP is satisfied

  [More practical] "Responsibility to a single actor"
    -> Actor = business stakeholder
    -> Accounting, HR, Engineering, etc.
    -> Change requests from the same actor should be contained in one class

  Criteria for judging responsibility granularity:
    1. "List three scenarios in which this class would be modified"
    2. If all three come from the same actor -> SRP satisfied
    3. If they come from different actors -> possible SRP violation
```

### 1.2 SRP Refactoring

```typescript
// [Bad] SRP violation: a class with multiple responsibilities
class UserService {
  // Responsibility 1: user creation logic
  createUser(data: CreateUserDto): User {
    // Validation
    if (!data.email.includes("@")) throw new Error("Invalid email");
    if (data.password.length < 8) throw new Error("Password too short");

    // Password hashing
    const hashedPassword = bcrypt.hashSync(data.password, 10);

    // DB persistence
    const user = db.users.create({ ...data, password: hashedPassword });

    // Email delivery
    const html = `<h1>Welcome ${data.name}!</h1>`;
    emailClient.send(data.email, "Welcome", html);

    // Logging
    logger.info(`User created: ${user.id}`);

    return user;
  }
}

// [Good] SRP applied: each class has a single responsibility
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

// Orchestrator
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

### 1.3 SRP Across Multiple Languages

```python
# Python: SRP in practice - e-commerce order processing

# [Bad] SRP violation: one class handles everything about orders
class OrderManager:
    def __init__(self):
        self.db = psycopg2.connect("dbname=shop")

    def create_order(self, customer_id: int, items: list[dict]) -> dict:
        # Validation (responsibility 1)
        if not items:
            raise ValueError("Order must have at least one item")
        for item in items:
            if item["quantity"] <= 0:
                raise ValueError(f"Invalid quantity for {item['name']}")

        # Price calculation (responsibility 2)
        subtotal = sum(i["price"] * i["quantity"] for i in items)
        tax = subtotal * 0.10  # consumption tax
        shipping = 500 if subtotal < 5000 else 0
        total = subtotal + tax + shipping

        # Inventory check (responsibility 3)
        cursor = self.db.cursor()
        for item in items:
            cursor.execute(
                "SELECT stock FROM products WHERE id = %s",
                (item["product_id"],)
            )
            stock = cursor.fetchone()[0]
            if stock < item["quantity"]:
                raise ValueError(f"Insufficient stock for {item['name']}")

        # DB persistence (responsibility 4)
        cursor.execute(
            "INSERT INTO orders (customer_id, total) VALUES (%s, %s) RETURNING id",
            (customer_id, total)
        )
        order_id = cursor.fetchone()[0]
        self.db.commit()

        # Email delivery (responsibility 5)
        import smtplib
        server = smtplib.SMTP("smtp.example.com")
        server.sendmail(
            "shop@example.com",
            f"customer_{customer_id}@example.com",
            f"Your order #{order_id} has been placed. Total: ¥{total}"
        )

        return {"order_id": order_id, "total": total}


# [Good] SRP applied: each class has exactly one responsibility

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
    """Validates order data only"""
    def validate(self, customer_id: int, items: list[OrderItem]) -> None:
        if not items:
            raise ValueError("Order must have at least one item")
        for item in items:
            if item.quantity <= 0:
                raise ValueError(f"Invalid quantity for {item.name}")
            if item.price <= 0:
                raise ValueError(f"Invalid price for {item.name}")


class PriceCalculator:
    """Price calculation only"""
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
    """Inventory availability check only"""
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
    """Order persistence only"""
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
    """Order confirmation notification only"""
    def __init__(self, email_sender):
        self._sender = email_sender

    def notify(self, order: Order) -> None:
        self._sender.send(
            to=f"customer_{order.customer_id}@example.com",
            subject=f"Order confirmation #{order.id}",
            body=f"Thank you for your order. Total: ¥{order.total}"
        )


class CreateOrderUseCase:
    """Orchestration (just composes each responsibility)"""
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
        # 1. Validation
        self._validator.validate(customer_id, items)

        # 2. Inventory check
        self._inventory.check_availability(items)

        # 3. Price calculation
        subtotal, tax, shipping, total = self._calculator.calculate(items)

        # 4. Create and persist the order
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

        # 5. Notification
        self._notifier.notify(order)

        return order
```

```java
// Java: SRP in practice - logging

// [Bad] SRP violation: formatting, writing, and alerting all in one class
public class Logger {
    private final String logFile;
    private final String dbUrl;

    public Logger(String logFile, String dbUrl) {
        this.logFile = logFile;
        this.dbUrl = dbUrl;
    }

    public void log(String level, String message) {
        // Responsibility 1: message formatting
        String timestamp = LocalDateTime.now()
            .format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
        String formatted = String.format("[%s] %s: %s", timestamp, level, message);

        // Responsibility 2: writing to file
        try (FileWriter fw = new FileWriter(logFile, true)) {
            fw.write(formatted + "\n");
        } catch (IOException e) {
            System.err.println("Failed to write log: " + e.getMessage());
        }

        // Responsibility 3: writing to DB
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

        // Responsibility 4: sending alerts (when level is ERROR)
        if ("ERROR".equals(level)) {
            // Slack notification
            HttpClient client = HttpClient.newHttpClient();
            // ... call the Slack API
        }
    }
}


// [Good] SRP applied: each class has one responsibility

// Formatting responsibility
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

// Output destination responsibility (abstracted via interface -> also enables OCP)
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

// Alerting responsibility
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
        // Call the Slack API
    }
}

// Orchestrator
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
// Kotlin: SRP in practice - validation

// [Bad] SRP violation: a single validator class knows about every domain rule
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


// [Good] SRP applied: separate validators per domain

// Generic validation result
sealed class ValidationResult {
    object Valid : ValidationResult()
    data class Invalid(val errors: List<String>) : ValidationResult()
}

// Validator interface
interface Validator<T> {
    fun validate(target: T): ValidationResult
}

// Rule-based validation
interface ValidationRule<T> {
    fun check(target: T): String?  // null = OK, non-null = error message
}

// User validation
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

// Product validation (independent responsibility)
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

// New validation rules just require adding a new ValidationRule
// -> this also aligns with OCP
```

### 1.4 How to Detect SRP Violations

```
Five heuristics for detecting SRP violations:

  1. Class-name test:
     -> Class name contains "And", "Or", "Manager", or "Handler"
     -> Example: UserAndOrderManager -> likely SRP violation
     -> Remedy: separate names per responsibility

  2. Reason-to-change test:
     -> "List three reasons this class might change"
     -> Reasons span different business domains -> SRP violation
     -> Example: "UI change" and "DB change" in the same class

  3. Description test:
     -> If you cannot describe the class purpose in one sentence -> likely SRP violation
     -> "Does A, and B, and C" -> three responsibilities
     -> "Does A" -> one responsibility (SRP satisfied)

  4. Import test:
     -> Imports reference many different libraries -> likely SRP violation
     -> Example: imports DB, HTTP, Email, filesystem
     -> Changes in each library can affect it = multiple change reasons

  5. Constructor test:
     -> More than 5 dependency-injection parameters -> likely SRP violation
     -> Many dependencies = possibly many responsibilities
     -> Orchestrators are exceptions, however
```

```typescript
// Concrete examples of SRP violation detection

// [Check] Class-name test
class UserRegistrationAndNotificationService { } // [Bad] contains "And"
class DataProcessingManager { }                   // [Bad] "Manager" is too vague
class UserRegistrationService { }                 // [Good] one responsibility

// [Check] Import test
// [Bad] too many diverse imports -> sign of SRP violation
import { Database } from './database';
import { SmtpClient } from './email';
import { S3Client } from 'aws-sdk';
import { RedisClient } from 'redis';
import { SlackWebhook } from './slack';
import { PdfGenerator } from './pdf';

class ReportService {
  constructor(
    private db: Database,        // DB dependency
    private smtp: SmtpClient,    // email dependency
    private s3: S3Client,        // storage dependency
    private redis: RedisClient,  // cache dependency
    private slack: SlackWebhook, // notification dependency
    private pdf: PdfGenerator,   // PDF generation dependency
  ) {}
  // -> depends on 6 different concerns = 6 reasons to change
}

// [Good] After applying SRP
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

## 2. OCP: Open/Closed Principle

```
Definition:
  "Software entities should be open for extension
   but closed for modification."

In other words:
  -> When adding new functionality, do not modify existing code
  -> Achieved through polymorphism (interface + implementation classes)

Why it matters:
  -> Modifying existing code risks regressions
  -> You do not have to touch tested code
  -> Fewer conflicts in team development
```

### 2.1 How to Achieve OCP

```
Four patterns for achieving OCP:

  1. Strategy pattern (most fundamental):
     -> Define an interface and add implementation classes
     -> Callers invoke the interface instead of using switch/if

  2. Template Method pattern:
     -> Define the algorithmic skeleton in a base class
     -> Override the details in subclasses

  3. Decorator pattern:
     -> Wrap an existing class to add functionality
     -> No changes to the original class

  4. Plugin / Registry pattern:
     -> Register implementations dynamically at runtime
     -> New implementations are added as plugins

  Criteria for application:
    No change has happened yet -> OCP is not needed (YAGNI)
    The same kind of change occurs 2-3 times -> time to apply OCP
```

### 2.2 OCP Refactoring

```typescript
// [Bad] OCP violation: must be modified every time a new channel is added
class NotificationService {
  send(type: string, message: string, recipient: string): void {
    if (type === "email") {
      // Email delivery
      emailClient.send(recipient, message);
    } else if (type === "sms") {
      // SMS delivery
      smsClient.send(recipient, message);
    } else if (type === "slack") {
      // Slack delivery (this must be modified whenever a new channel is added)
      slackClient.post(recipient, message);
    }
    // Add LINE? Add Discord? -> keep modifying here...
  }
}

// [Good] OCP applied: a new channel just requires adding a class
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

// Adding LINE -> just add a LineChannel class
// NotificationService does not need to change at all

class NotificationService {
  constructor(private channels: NotificationChannel[]) {}

  async sendAll(message: string, recipient: string): Promise<void> {
    await Promise.all(
      this.channels.map(ch => ch.send(message, recipient))
    );
  }
}
```

### 2.3 OCP Across Multiple Languages

```python
# Python: OCP in practice - report engine

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# [Bad] OCP violation: modifications needed for new data sources or formats
class ReportEngine:
    def generate(
        self, source: str, format_type: str, filters: dict
    ) -> str:
        # Data fetching (branches by source type)
        if source == "mysql":
            data = self._fetch_from_mysql(filters)
        elif source == "mongodb":
            data = self._fetch_from_mongodb(filters)
        elif source == "api":
            data = self._fetch_from_api(filters)
        else:
            raise ValueError(f"Unknown source: {source}")

        # Formatting (branches by format type)
        if format_type == "pdf":
            return self._format_as_pdf(data)
        elif format_type == "excel":
            return self._format_as_excel(data)
        elif format_type == "html":
            return self._format_as_html(data)
        else:
            raise ValueError(f"Unknown format: {format_type}")

    # Private methods keep multiplying per source...
    def _fetch_from_mysql(self, filters): ...
    def _fetch_from_mongodb(self, filters): ...
    def _fetch_from_api(self, filters): ...
    def _format_as_pdf(self, data): ...
    def _format_as_excel(self, data): ...
    def _format_as_html(self, data): ...


# [Good] OCP applied: data sources and formatters are extensible

@dataclass
class ReportData:
    """Common representation of report data"""
    headers: list[str]
    rows: list[list[Any]]
    metadata: dict[str, Any]


class DataSource(ABC):
    """Data source abstraction"""
    @abstractmethod
    def fetch(self, filters: dict) -> ReportData: ...


class MySQLDataSource(DataSource):
    def __init__(self, connection_string: str):
        self._conn_str = connection_string

    def fetch(self, filters: dict) -> ReportData:
        # Fetch data from MySQL
        import mysql.connector
        conn = mysql.connector.connect(self._conn_str)
        cursor = conn.cursor()
        query = self._build_query(filters)
        cursor.execute(query)
        headers = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return ReportData(headers=headers, rows=rows, metadata={"source": "mysql"})

    def _build_query(self, filters: dict) -> str:
        # Build query
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
    """Fetch data from a REST API - adding it requires no change to existing code"""
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
    """Formatter abstraction"""
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
        # PDF generation logic
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
        # Excel generation logic
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
    """This class needs no change when new data sources or formats are added"""
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


# Usage: compose freely
engine = ReportEngine(
    source=MySQLDataSource("mysql://localhost/mydb"),
    formatter=PdfFormatter(),
)
engine.generate_to_file("monthly_report")

# A new combination needs no changes to existing code
engine2 = ReportEngine(
    source=RestApiDataSource("https://api.example.com", "key123"),
    formatter=ExcelFormatter(),
)
```

```java
// Java: OCP in practice - authentication pipeline

// [Bad] OCP violation
public class AuthService {
    public boolean authenticate(String method, String credentials) {
        if ("password".equals(method)) {
            // Password authentication
            String[] parts = credentials.split(":");
            return checkPassword(parts[0], parts[1]);
        } else if ("oauth".equals(method)) {
            // OAuth authentication
            return verifyOAuthToken(credentials);
        } else if ("api_key".equals(method)) {
            // API key authentication
            return validateApiKey(credentials);
        } else if ("certificate".equals(method)) {
            // Certificate authentication (this must be modified every time one is added)
            return verifyCertificate(credentials);
        }
        throw new IllegalArgumentException("Unknown method: " + method);
    }
}


// [Good] OCP applied: add authentication methods as plugins

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

// Authentication service: no changes required when new methods are added
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

// Example configuration for Spring Boot
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

### 2.4 Another Way to Achieve OCP: Decorators

```python
# Python: OCP via decorators
class Logger:
    """Add logging without modifying the existing class"""
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

# Add logging without modifying Calculator
calc = Logger(Calculator())
calc.add(1, 2)
# [LOG] add called with (1, 2)
# [LOG] add returned 3
```

```typescript
// TypeScript: OCP via the Decorator pattern

// Base interface
interface HttpClient {
  get(url: string): Promise<Response>;
  post(url: string, body: any): Promise<Response>;
}

// Base implementation
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

// Decorator 1: add logging (without modifying FetchHttpClient)
class LoggingHttpClient implements HttpClient {
  constructor(private inner: HttpClient) {}

  async get(url: string): Promise<Response> {
    console.log(`[GET] ${url}`);
    const start = Date.now();
    const response = await this.inner.get(url);
    console.log(`[GET] ${url} -> ${response.status} (${Date.now() - start}ms)`);
    return response;
  }

  async post(url: string, body: any): Promise<Response> {
    console.log(`[POST] ${url}`, body);
    const start = Date.now();
    const response = await this.inner.post(url, body);
    console.log(`[POST] ${url} -> ${response.status} (${Date.now() - start}ms)`);
    return response;
  }
}

// Decorator 2: add retry (without modifying FetchHttpClient)
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
        return response; // do not retry 4xx
      } catch (error) {
        lastError = error as Error;
      }
    }
    throw lastError;
  }
}

// Decorator 3: add caching
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
    // Do not cache POST
    return this.inner.post(url, body);
  }
}

// Usage: compose decorators to add features
// The existing FetchHttpClient was not modified at all
const client: HttpClient = new CachingHttpClient(
  new RetryHttpClient(
    new LoggingHttpClient(
      new FetchHttpClient()
    ),
    3,
  ),
  30_000,
);

// Request -> Caching -> Retry -> Logging -> Fetch (processed in that order)
await client.get("https://api.example.com/data");
```

```python
# Python: Decorator pattern - middleware pipeline

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


# Middleware interface
class Middleware(ABC):
    @abstractmethod
    def process(
        self, request: Request, next_handler: Callable[[Request], Response]
    ) -> Response:
        ...


# Authentication middleware
class AuthMiddleware(Middleware):
    def __init__(self, token_verifier):
        self._verifier = token_verifier

    def process(self, request: Request, next_handler):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token or not self._verifier.verify(token):
            return Response(status=401, body={"error": "Unauthorized"})
        return next_handler(request)


# Logging middleware
class LoggingMiddleware(Middleware):
    def process(self, request: Request, next_handler):
        start = time.time()
        print(f"-> {request.method} {request.path}")
        response = next_handler(request)
        elapsed = time.time() - start
        print(f"<- {response.status} ({elapsed:.3f}s)")
        return response


# Rate-limiting middleware
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


# CORS middleware
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


# Pipeline: compose middleware (OCP achieved)
class MiddlewarePipeline:
    """Adding new middleware only requires a new Middleware class"""
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


# Usage
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

### 2.5 How to Detect OCP Violations

```
Four heuristics for detecting OCP violations:

  1. switch / if-else chains:
     -> Type checks against the same variable scattered in multiple places
     -> Adding a new type requires modifying every switch
     -> A sign of Shotgun Surgery

  2. instanceof / type checks:
     -> Frequent `if (obj instanceof SomeClass)`
     -> A place that should be resolved with polymorphism

  3. Change-history analysis:
     -> In git log: "the same file is repeatedly modified for different features"
     -> -> candidate for applying OCP

  4. Comments like "// Add new ~ here":
     -> If a comment has to flag the spot to modify = OCP violation
     -> Ideally, "just add a class" should be enough

  Example detection command:
    # Identify files modified most frequently
    git log --format=format: --name-only --since="6 months ago" | \
      sort | uniq -c | sort -rn | head -20
```

---

## 3. The Relationship Between SRP and OCP

```
SRP -> break classes into small pieces
  v
OCP -> connect the small classes via interfaces
  v
Result: a design where extension is easy and the impact of changes is localized

Practical flow:
  1. Separate responsibilities with SRP
  2. Identify the parts that are likely to change
  3. Design interfaces with OCP
  4. Handle new requirements by adding classes
```

### 3.1 SRP + OCP Collaboration Patterns

```typescript
// A practical example where SRP and OCP work together: invoice system

// Step 1: separate responsibilities with SRP

// Invoice data
interface Invoice {
  id: string;
  items: InvoiceItem[];
  customer: Customer;
  issuedAt: Date;
  dueDate: Date;
}

// Tax calculation (SRP: tax calculation only)
interface TaxCalculator {
  calculate(items: InvoiceItem[]): number;
}

// Discount application (SRP: discount calculation only)
interface DiscountPolicy {
  apply(subtotal: number, customer: Customer): number;
}

// Formatting (SRP: formatting only)
interface InvoiceFormatter {
  format(invoice: Invoice, total: number): string;
}

// Sending (SRP: sending only)
interface InvoiceSender {
  send(invoice: Invoice, formatted: string): Promise<void>;
}

// Step 2: make each responsibility extensible with OCP

// Tax calculation: support tax systems by country
class JapaneseTaxCalculator implements TaxCalculator {
  calculate(items: InvoiceItem[]): number {
    const subtotal = items.reduce((sum, i) => sum + i.amount, 0);
    return Math.floor(subtotal * 0.10); // 10% consumption tax
  }
}

class USStateTaxCalculator implements TaxCalculator {
  constructor(private stateRate: number) {}
  calculate(items: InvoiceItem[]): number {
    const subtotal = items.reduce((sum, i) => sum + i.amount, 0);
    return Math.floor(subtotal * this.stateRate);
  }
}

// Discounts: driven by business rules
class VolumeDiscount implements DiscountPolicy {
  apply(subtotal: number, customer: Customer): number {
    if (subtotal > 100000) return subtotal * 0.05; // 5% discount
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

// Formatting: output formats
class PdfInvoiceFormatter implements InvoiceFormatter {
  format(invoice: Invoice, total: number): string {
    // PDF generation logic
    return `<pdf-data>Invoice ${invoice.id}: ¥${total}</pdf-data>`;
  }
}

class HtmlInvoiceFormatter implements InvoiceFormatter {
  format(invoice: Invoice, total: number): string {
    return `<html><h1>Invoice ${invoice.id}</h1><p>Total: ¥${total}</p></html>`;
  }
}

// Sending: delivery channels
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

// Step 3: orchestrator (SRP: coordination only)
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

// Usage: Japanese customer, PDF format, email delivery
const jpService = new InvoiceService(
  new JapaneseTaxCalculator(),
  new CompositeDiscount([new VolumeDiscount(), new LoyaltyDiscount()]),
  new PdfInvoiceFormatter(),
  new EmailInvoiceSender(),
);

// Usage: US customer, HTML format, fax delivery
const usService = new InvoiceService(
  new USStateTaxCalculator(0.08),
  new VolumeDiscount(),
  new HtmlInvoiceFormatter(),
  new FaxInvoiceSender(),
);

// New taxes, discounts, formats, or delivery channels -> just add classes
// InvoiceService requires no changes
```

### 3.2 Improved Testability

```typescript
// SRP + OCP drastically simplify testing

// Test doubles
class MockTaxCalculator implements TaxCalculator {
  calculate(items: InvoiceItem[]): number {
    return 1000; // fixed value for predictability
  }
}

class MockDiscountPolicy implements DiscountPolicy {
  apply(subtotal: number, customer: Customer): number {
    return 0; // no discount
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

// Test code
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

// Each component can be tested independently
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

## 4. Anti-Patterns and Caveats

```
Over-applying SRP:
  -> Huge numbers of classes with only one method
  -> File count explodes, navigation becomes difficult
  -> Remedy: split by "reasons to change," not by method count

Over-applying OCP:
  -> Abstracting parts that never change
  -> Codebase is overrun with unnecessary interfaces
  -> Remedy: abstract only after the change actually occurs

Judgment criteria:
  "What are the reasons this class would change?"
  -> Multiple reasons -> split with SRP
  "Is this part likely to change in the future?"
  -> Yes -> introduce interfaces via OCP
  -> No -> leave it alone (YAGNI)
```

### 4.1 Over-Applying SRP

```typescript
// [Bad] Over-applied SRP: unnecessary fragmentation

// No need for a class just to concatenate two strings
class StringConcatenator {
  concatenate(a: string, b: string): string {
    return a + b;
  }
}

// No need for a class just to add numbers
class NumberAdder {
  add(a: number, b: number): number {
    return a + b;
  }
}

// No need for a class just to null-check
class NullChecker {
  isNull(value: any): boolean {
    return value === null || value === undefined;
  }
}

// [Good] Appropriate granularity: a class that groups related operations
class MathUtils {
  static add(a: number, b: number): number { return a + b; }
  static subtract(a: number, b: number): number { return a - b; }
  static multiply(a: number, b: number): number { return a * b; }
  static divide(a: number, b: number): number {
    if (b === 0) throw new Error("Division by zero");
    return a / b;
  }
}
// -> Reason to change: "rules for mathematical computation" -> one actor
// -> 4 methods but 1 responsibility = SRP satisfied
```

### 4.2 Over-Applying OCP

```typescript
// [Bad] Over-applied OCP: abstracting parts that do not change

// Reading configuration: unlikely to change
interface ConfigReader { read(): Config; }
interface ConfigParser { parse(raw: string): Config; }
interface ConfigValidator { validate(config: Config): void; }
interface ConfigMerger { merge(base: Config, override: Config): Config; }

// -> How a config file is loaded rarely changes
// -> Four interfaces is overkill

// [Good] Appropriate level of abstraction
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
// -> Configuration loading changes infrequently
// -> A simple class is enough
// -> When a change is actually needed, abstract at that time
```

### 4.3 A Decision Flowchart for Real-World Work

```
SRP decision flow:

  Class length > 300 lines?
    |
    +-- Yes -> Analyze reasons to change
    |          |
    |          +-- Multiple reasons -> apply SRP (split)
    |          +-- Single reason    -> large is OK (one responsibility)
    |
    +-- No  -> Does it mix multiple domains?
               |
               +-- Yes -> apply SRP (split even if small)
               +-- No  -> keep as is


OCP decision flow:

  Has the same kind of change happened 2+ times?
    |
    +-- Yes -> Are switch / if-else branches growing?
    |          |
    |          +-- Yes -> apply OCP (introduce interface)
    |          +-- No  -> wait for one more change before applying
    |
    +-- No  -> No change has occurred
               -> keep as is (YAGNI)
               -> do not abstract "speculatively"
```

---

## 5. SRP + OCP in Frameworks

```
Examples of SRP + OCP usage in major frameworks:

  NestJS (TypeScript):
    SRP -> separation of Controller, Service, Repository
    OCP -> DI via @Injectable(), Guards / Interceptors / Pipes

  Spring Boot (Java):
    SRP -> @Controller, @Service, @Repository annotations
    OCP -> @Bean definitions, environment switching via @Profile

  Django (Python):
    SRP -> separation of views.py, models.py, serializers.py
    OCP -> Middleware classes, custom authentication Backends

  Rails (Ruby):
    SRP -> Model, Controller, Service Object pattern
    OCP -> Concern modules, ActiveSupport::Concern
```

```typescript
// NestJS: SRP + OCP in practice

// Controller (SRP: HTTP request handling only)
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

// Service (SRP: business logic only)
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

// Module (OCP: easy to swap dependencies)
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

// Guard (OCP: add authentication as a plugin)
@Injectable()
class AuthGuard implements CanActivate {
  canActivate(context: ExecutionContext): boolean {
    const request = context.switchToHttp().getRequest();
    return this.validateToken(request.headers.authorization);
  }
}

// Interceptor (OCP: add cross-cutting concerns like a decorator)
@Injectable()
class LoggingInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const now = Date.now();
    return next.handle().pipe(
      tap(() => console.log(`Response time: ${Date.now() - now}ms`)),
    );
  }
}

// Pipe (OCP: add validation as a plugin)
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

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Your understanding deepens when, in addition to theory, you actually write code and verify its behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. We recommend firmly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in real-world work?

The knowledge of this topic is used frequently in day-to-day development work. It becomes especially important during code reviews and when designing architecture.

---

## Summary

| Principle | Core idea | How to achieve it | Caveat | How to detect |
|-----------|-----------|-------------------|--------|---------------|
| SRP | One class, one responsibility | Separation of responsibilities, delegation | Avoid over-splitting | Class-name / import tests |
| OCP | Open for extension, closed for modification | Interfaces, polymorphism | Abstract only when needed | Detect switch / if-else chains |

### SRP + OCP Application Checklist

```
[ ] The class has exactly one reason to change (SRP)
[ ] The class name expresses a single responsibility (SRP)
[ ] Imports/dependencies are limited to one domain (SRP)
[ ] The constructor takes 5 or fewer arguments (SRP)
[ ] The same kind of branching is not scattered in multiple places (OCP)
[ ] No "add here" comment is needed when adding a new variant (OCP)
[ ] Components can be swapped with mocks during testing (SRP + OCP)
[ ] Each class can be tested independently (SRP)
[ ] Changes are contained within one class (SRP + OCP)
[ ] The same file is not modified frequently in git log (OCP)
```

---

## Recommended Next Reading

---

## References
1. Martin, R. "Clean Architecture: A Craftsman's Guide to Software Structure and Design." Chapter 7-8, Prentice Hall, 2017.
2. Martin, R. "The Single Responsibility Principle." The Clean Coder Blog, 2014.
3. Martin, R. "Agile Software Development, Principles, Patterns, and Practices." Prentice Hall, 2003.
4. Meyer, B. "Object-Oriented Software Construction." Prentice Hall, 2nd ed., 1997.
5. Fowler, M. "Refactoring: Improving the Design of Existing Code." Addison-Wesley, 2nd ed., 2018.
6. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
7. Freeman, S. and Pryce, N. "Growing Object-Oriented Software, Guided by Tests." Addison-Wesley, 2009.
