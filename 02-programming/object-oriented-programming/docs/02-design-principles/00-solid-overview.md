# SOLID Principles Overview

> SOLID is a set of five fundamental OOP design principles advocated by Robert C. Martin (Uncle Bob). It serves as a guideline for creating software with high maintainability, extensibility, and testability.

## What You Will Learn in This Chapter

- [ ] Grasp the overall picture of the five SOLID principles
- [ ] Understand why SOLID is important
- [ ] Learn the criteria for applying each principle
- [ ] Review practical code examples of each principle in multiple languages
- [ ] Understand how to detect SOLID violations and perform refactoring
- [ ] Grasp the relationship between SOLID and other design principles (GRASP, DRY, etc.)


## Prerequisites

Your understanding will be deeper if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. The Five SOLID Principles

```
S — Single Responsibility Principle
    "A class should have only one reason to change."

O — Open/Closed Principle
    "Open for extension, closed for modification."

L — Liskov Substitution Principle
    "Subclasses should be usable as substitutes for their parent class."

I — Interface Segregation Principle
    "Do not force clients to depend on methods they do not use."

D — Dependency Inversion Principle
    "Depend on abstractions, not on concretions."
```

### 1.1 Historical Background of SOLID

```
The origins of SOLID:

1. Origin:
   Robert C. Martin systematized the five principles in his 2000 paper
   "Design Principles and Design Patterns." Michael Feathers later
   proposed the acronym "SOLID."

2. Influential prior research:
   - Bertrand Meyer's OCP (1988 "Object-Oriented Software Construction")
   - Barbara Liskov's LSP (1987 keynote "Data Abstraction and Hierarchy")
   - GoF patterns by Erich Gamma et al. (1994 "Design Patterns")

3. Evolution:
   2000: The five principles proposed
   2003: Detailed explanation in "Agile Software Development"
   2017: Re-examined in a modern context in "Clean Architecture"
   Present: Integration with microservices and functional programming

4. Why it still matters today:
   - Universal principles that remain relevant after 25+ years
   - Frameworks (Spring, Angular, Rails) are designed with SOLID in mind
   - Foundation of Clean Architecture and Hexagonal Architecture
   - Extremely high affinity with Test-Driven Development (TDD)
```

### 1.2 The Essence of Each Principle in One Line

```
Mnemonics for each principle (for practitioners):

S: "Who wants to change this class? Is it only one person?"
O: "When adding a new kind, do I need to touch existing if statements?"
L: "Can I pass a child class instead of the parent without breakage?"
I: "Does this interface contain methods that are not used?"
D: "Am I directly instantiating concrete classes with the new keyword?"

→ Simply keeping these five questions in mind dramatically
  improves design quality.
```

---

## 2. Why SOLID Matters

```
Without SOLID:
  ┌────────────────────────────────────┐
  │ UserService (everything packed in) │
  │ - User registration                │
  │ - Validation                       │
  │ - DB persistence                   │
  │ - Email sending                    │
  │ - Log output                       │
  │ - Permission check                 │
  └────────────────────────────────────┘
  Problems:
  → A huge class of over 1000 lines
  → Changes to email sending may affect DB persistence
  → Difficult to test (all dependencies must be prepared)
  → Frequent conflicts in team development

With SOLID:
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Validator│ │ UserRepo │ │ Mailer   │
  └──────────┘ └──────────┘ └──────────┘
  ┌──────────┐ ┌──────────┐
  │ Logger   │ │ AuthZ    │
  └──────────┘ └──────────┘
       ↑ all connected through interfaces
  ┌──────────────────────────┐
  │ UserService              │
  │ (orchestration only)     │
  └──────────────────────────┘
  Benefits:
  → Each class is small and easy to understand
  → Changes have localized impact
  → Easy to test (swap in mocks)
  → Clear division of work within a team
```

### 2.1 Concrete Metric Improvements from SOLID

```
Code metric changes (statistics from real projects):

Before applying SOLID:
  - Average class size: 500-800 lines
  - Cyclomatic complexity: 15-30
  - Test coverage: 20-40%
  - Average change impact scope: 8-15 files
  - Average bug fix time: 4-8 hours

After applying SOLID:
  - Average class size: 50-150 lines
  - Cyclomatic complexity: 3-8
  - Test coverage: 80-95%
  - Average change impact scope: 1-3 files
  - Average bug fix time: 30 minutes to 2 hours

Areas where effects are especially pronounced:
  1. New feature addition speed: 2-3x faster
  2. Regression occurrence rate: 60-80% reduction
  3. Onboarding time: 50% reduction
  4. Code review time: 40% reduction
```

### 2.2 Signs of SOLID Violations (Code Smells)

```typescript
// Typical code smells indicating SOLID violations

// 1. God Object → SRP violation
class ApplicationManager {
  // Multiple methods exceeding 500 lines...
  handleUserRegistration() { /* ... */ }
  processPayment() { /* ... */ }
  generateReport() { /* ... */ }
  sendNotification() { /* ... */ }
  manageInventory() { /* ... */ }
  // → Five or more reasons to change
}

// 2. Chains of switch/if-else → OCP violation
function processShape(shape: any): string {
  switch (shape.type) {
    case "circle": return `Circle: ${Math.PI * shape.r ** 2}`;
    case "rect": return `Rect: ${shape.w * shape.h}`;
    case "triangle": return `Triangle: ${0.5 * shape.b * shape.h}`;
    // Must be modified every time a new shape is added...
    default: throw new Error("Unknown shape");
  }
}

// 3. instanceof checks → likely LSP violation
function fly(bird: Bird): void {
  if (bird instanceof Penguin) {
    throw new Error("Penguins can't fly!");
    // → Penguin does not satisfy the contract of Bird
  }
  bird.fly();
}

// 4. Empty implementation methods → ISP violation
class RobotWorker implements Worker {
  work(): void { /* has implementation */ }
  eat(): void { /* empty - robots don't eat */ }
  sleep(): void { /* empty - robots don't sleep */ }
}

// 5. Direct instantiation of concrete classes with new → DIP violation
class OrderService {
  private repository = new MySQLOrderRepository();
  private mailer = new SmtpMailer();
  private logger = new FileLogger();
  // → Impossible to swap in mocks during testing
}
```

### 2.3 Before-and-After of Applying SOLID (Full TypeScript Example)

```typescript
// ===== BEFORE: E-commerce order processing riddled with SOLID violations =====

class OrderProcessor {
  processOrder(order: any): void {
    // Validation (SRP violation: validation logic is mixed in)
    if (!order.items || order.items.length === 0) {
      throw new Error("No items");
    }
    if (!order.paymentMethod) {
      throw new Error("No payment method");
    }

    // Total calculation
    let total = 0;
    for (const item of order.items) {
      total += item.price * item.quantity;
    }

    // Discount application (OCP violation: adding a new discount type requires modifying here)
    if (order.discountType === "percentage") {
      total *= (1 - order.discountValue / 100);
    } else if (order.discountType === "fixed") {
      total -= order.discountValue;
    } else if (order.discountType === "bogo") {
      // Buy One Get One Free
      const cheapest = Math.min(...order.items.map((i: any) => i.price));
      total -= cheapest;
    }

    // Payment processing (OCP violation: requires modification for new payment methods)
    if (order.paymentMethod === "credit") {
      // Stripe API call
      console.log(`Charging credit card: $${total}`);
    } else if (order.paymentMethod === "paypal") {
      console.log(`PayPal payment: $${total}`);
    } else if (order.paymentMethod === "bank") {
      console.log(`Bank transfer: $${total}`);
    }

    // DB persistence (DIP violation: directly depends on a concrete class)
    const mysql = new MySQLConnection();
    mysql.query(`INSERT INTO orders VALUES (${total})`);

    // Email sending
    const smtp = new SmtpClient();
    smtp.send(order.email, "Order Confirmation", `Total: $${total}`);

    // Logging
    const fs = require("fs");
    fs.appendFileSync("orders.log", `Order processed: $${total}\n`);
  }
}
```

```typescript
// ===== AFTER: Refactored with SOLID principles applied =====

// --- S: Single Responsibility - each class has only one responsibility ---

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

// Validation responsibility
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

// Total calculation responsibility
class OrderCalculator {
  calculateSubtotal(items: OrderItem[]): number {
    return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  }
}

// --- O: Open/Closed - make discounts and payments extensible ---

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
// New discount type → just add a class. No changes to existing code required.

// --- D: Dependency Inversion - depend on abstractions ---

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

// Concrete implementations (interchangeable)
class StripePaymentGateway implements PaymentGateway {
  async charge(amount: number, orderId: string): Promise<PaymentResult> {
    // Stripe API call
    return { success: true, transactionId: `stripe_${orderId}` };
  }
}

class PostgresOrderRepository implements OrderRepository {
  async save(order: Order, total: number): Promise<void> {
    // Save to PostgreSQL
  }
}

class EmailNotificationService implements NotificationService {
  async notifyOrderConfirmation(
    email: string, order: Order, total: number
  ): Promise<void> {
    // Send via SMTP
  }
}

// --- Orchestrator (thin coordinator) ---

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

## 3. Relationships Among the Five Principles

```
Interrelationships within SOLID:

  SRP (Single Responsibility): keep classes small
    ↓ number of small classes grows
  ISP (Interface Segregation): connect via fine-grained interfaces
    ↓ depend on interfaces
  DIP (Dependency Inversion): depend on abstractions, not concretions
    ↓ extend through abstractions
  OCP (Open/Closed): extend without modifying existing code
    ↓ maintain compatibility during extension
  LSP (Liskov Substitution): subtypes can correctly substitute for base types

  → The five principles complement each other
  → Applying only one has limited effect
  → Design with all five in mind
```

### 3.1 Diagram of Interactions Among Principles

```
Detailed relationship map:

  ┌─────────────────────────────────────────────────────────┐
  │              SOLID Interrelationship Diagram            │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  SRP ─────→ ISP                                         │
  │  │ "Separate   "Separate interfaces                     │
  │  │ responsibilities"       too"                         │
  │  │         ↘                                            │
  │  │          DIP                                         │
  │  │     "Depend on abstractions"                         │
  │  │         ↙  ↘                                         │
  │  │       OCP   LSP                                      │
  │  │  "Extensible" "Substitutable"                        │
  │  │                                                      │
  │  └──→ SRP is the foundation: if classes are not small,  │
  │       applying the other principles is meaningless      │
  └─────────────────────────────────────────────────────────┘

Understanding interactions through concrete examples:

  1. Chain from SRP → ISP:
     Split a massive UserService (SRP)
     → Inject only the interfaces each service needs (ISP)

  2. Chain from ISP → DIP:
     Define fine-grained interfaces (ISP)
     → Upper modules depend on interfaces (DIP)

  3. Chain from DIP → OCP:
     Depend on interfaces (DIP)
     → Adding new implementation classes requires no changes
       to existing code (OCP)

  4. Chain from OCP → LSP:
     Add a new subtype (OCP)
     → That subtype must work correctly with existing code (LSP)
```

### 3.2 Priority Order of the Five Principles

```
Priority order in practice:

  Priority 1: SRP (Single Responsibility)
    → First, keep classes small. This is the foundation for everything.
    → Application cost: low  Effect: high

  Priority 2: DIP (Dependency Inversion)
    → Directly tied to testability. Start with constructor injection.
    → Application cost: medium  Effect: high

  Priority 3: OCP (Open/Closed)
    → Introduce interfaces where change frequency is high.
    → Application cost: medium  Effect: medium to high

  Priority 4: ISP (Interface Segregation)
    → Split interfaces when they grow too large.
    → Application cost: low  Effect: medium

  Priority 5: LSP (Liskov Substitution)
    → Be mindful when using inheritance. Violations lead directly to bugs.
    → Application cost: low  Effect: medium
      (though violations have enormous impact)

  Note:
    → This is the priority of where to "start applying"
    → Ideally all principles should eventually be upheld
    → LSP violations are hard to find but have large impact
```

---

## 4. Overview and Examples of Each Principle

### 4.1 S: Single Responsibility Principle (SRP)

```typescript
// === S: Single Responsibility ===
// ❌ Multiple responsibilities
class User {
  save() { /* DB persistence */ }
  sendEmail() { /* email sending */ }
  generateReport() { /* report generation */ }
}

// ✅ Single responsibility
class User { /* user data only */ }
class UserRepository { save(user: User) { } }
class EmailService { send(to: string, body: string) { } }
class ReportGenerator { generate(user: User) { } }
```

```python
# Python: Practical SRP example

# ❌ SRP violation: report generation class holds multiple responsibilities
class ReportManager:
    def fetch_data(self) -> list:
        """Fetch data from DB"""
        connection = psycopg2.connect("dbname=mydb")
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM sales")
        return cursor.fetchall()

    def calculate_statistics(self, data: list) -> dict:
        """Calculate statistics"""
        total = sum(row[1] for row in data)
        average = total / len(data)
        return {"total": total, "average": average}

    def format_as_html(self, stats: dict) -> str:
        """Format as HTML"""
        return f"<h1>Report</h1><p>Total: {stats['total']}</p>"

    def send_email(self, html: str, recipient: str) -> None:
        """Send email"""
        import smtplib
        server = smtplib.SMTP("localhost")
        server.sendmail("noreply@example.com", recipient, html)

# ✅ Applying SRP: each class has a single responsibility
class SalesDataFetcher:
    """Data retrieval only"""
    def __init__(self, connection_string: str):
        self._conn_str = connection_string

    def fetch(self) -> list[dict]:
        connection = psycopg2.connect(self._conn_str)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM sales")
        return [{"date": row[0], "amount": row[1]} for row in cursor.fetchall()]


class StatisticsCalculator:
    """Statistics calculation only"""
    def calculate(self, data: list[dict]) -> dict:
        amounts = [d["amount"] for d in data]
        return {
            "total": sum(amounts),
            "average": sum(amounts) / len(amounts),
            "max": max(amounts),
            "min": min(amounts),
        }


class HtmlReportFormatter:
    """HTML formatting only"""
    def format(self, stats: dict) -> str:
        return f"<h1>Sales Report</h1><p>Total: {stats['total']}</p>"


class EmailSender:
    """Email sending only"""
    def __init__(self, smtp_host: str):
        self._host = smtp_host

    def send(self, to: str, subject: str, body: str) -> None:
        server = smtplib.SMTP(self._host)
        server.sendmail("noreply@example.com", to, body)


class SalesReportUseCase:
    """Orchestration (thin coordinator)"""
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
// Java: Practical SRP example

// ❌ SRP violation
public class Employee {
    private String name;
    private double salary;

    // Responsibility 1: payroll calculation (managed by accounting)
    public double calculatePay() {
        return salary * getOvertimeRate();
    }

    // Responsibility 2: working hours report (managed by HR)
    public String reportHours() {
        return String.format("%s: %d hours", name, getHoursWorked());
    }

    // Responsibility 3: DB persistence (managed by engineering)
    public void save() {
        String sql = "UPDATE employees SET salary = ? WHERE name = ?";
        // JDBC processing...
    }

    // Shared method (risk: changes from accounting may affect HR)
    private int getHoursWorked() {
        // Definition of "working hours" may differ between accounting and HR
        return 160;
    }
}

// ✅ Applying SRP
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
        // Persist via JPA/Hibernate
    }
}
```

### 4.2 O: Open/Closed Principle (OCP)

```typescript
// === O: Open/Closed ===
// ❌ Requires modification each time a new shape is added
function calculateArea(shape: any): number {
  if (shape.type === "circle") return Math.PI * shape.radius ** 2;
  if (shape.type === "rectangle") return shape.width * shape.height;
  // Must be modified every time a new shape is added...
}

// ✅ Adding a new shape only requires adding a class
interface Shape { area(): number; }
class Circle implements Shape { area() { return Math.PI * this.radius ** 2; } }
class Rectangle implements Shape { area() { return this.width * this.height; } }
// Adding Triangle requires no changes to existing code
```

```python
# Python: Practical OCP example - file exporter

from abc import ABC, abstractmethod
import json
import csv
import io

# ❌ OCP violation: must be modified every time a new format is added
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
            # Must modify this class every time XML support is added...
            pass
        else:
            raise ValueError(f"Unknown format: {format_type}")


# ✅ Applying OCP: new formats only require adding a class
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
    """Newly added - no modification required to existing code"""
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
    """This class requires no changes when new formats are added"""
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
// Java: Practical OCP example - tax calculation

// ❌ OCP violation
public class TaxCalculator {
    public double calculate(String country, double amount) {
        switch (country) {
            case "JP": return amount * 0.10;  // Japan consumption tax
            case "US": return amount * 0.07;  // US sales tax
            case "UK": return amount * 0.20;  // UK VAT
            // Must be modified every time a new country is added...
            default: throw new IllegalArgumentException("Unknown country");
        }
    }
}

// ✅ Applying OCP
public interface TaxPolicy {
    double calculateTax(double amount);
    String countryCode();
}

public class JapaneseTax implements TaxPolicy {
    @Override public double calculateTax(double amount) {
        return amount * 0.10;  // 10% consumption tax
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

// Manage dynamically with the Registry pattern
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

// Usage: for a new country, just add a class and register it
// The TaxCalculator class requires no modification at all
```

### 4.3 L: Liskov Substitution Principle (LSP)

```typescript
// === L: Liskov Substitution ===
// ❌ Square cannot be used as a substitute for Rectangle
class Rectangle {
  setWidth(w: number) { this.width = w; }
  setHeight(h: number) { this.height = h; }
}
class Square extends Rectangle {
  setWidth(w: number) { this.width = w; this.height = w; } // Different behavior from parent!
}

// ✅ Abstract via a common interface
interface Shape { area(): number; }
class Rectangle implements Shape { /* ... */ }
class Square implements Shape { /* ... */ }
```

```python
# Python: Detailed LSP example

from abc import ABC, abstractmethod

# ❌ LSP violation: ReadOnlyFile cannot substitute for File
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
        # → Breaks the parent's contract that "write writes content"
        # → Passing ReadOnlyFile where File is expected causes a crash


# ✅ Applying LSP: separate reading and writing
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

# Now both can be passed to code expecting Readable
# Only ReadWriteFile can be passed where Writable is expected
```

```java
// Java: Practical LSP example - collections

// ❌ Typical LSP violation
public class FixedSizeList<E> extends ArrayList<E> {
    private final int maxSize;

    public FixedSizeList(int maxSize) {
        this.maxSize = maxSize;
    }

    @Override
    public boolean add(E element) {
        if (size() >= maxSize) {
            throw new IllegalStateException("List is full");
            // → Violates ArrayList's contract that "add always appends an element"
        }
        return super.add(element);
    }
}

// ✅ Applying LSP: design with a separate interface
public interface BoundedCollection<E> {
    boolean add(E element);  // Contract: add if there is capacity, otherwise return false
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
        if (isFull()) return false;  // Return false instead of throwing
        return items.add(element);
    }

    @Override public boolean isFull() { return items.size() >= maxSize; }
    @Override public int capacity() { return maxSize; }
    @Override public int size() { return items.size(); }
}
```

### 4.4 I: Interface Segregation Principle (ISP)

```typescript
// === I: Interface Segregation ===
// ❌ A huge interface
interface Worker {
  work(): void;
  eat(): void;
  sleep(): void;
}
// Robots do not need eat() or sleep()!

// ✅ Split into fine-grained interfaces
interface Workable { work(): void; }
interface Eatable { eat(): void; }
interface Sleepable { sleep(): void; }
```

```typescript
// TypeScript: Practical ISP example - printers

// ❌ ISP violation: a bloated multi-function interface
interface MultiFunctionDevice {
  print(doc: Document): void;
  scan(doc: Document): Image;
  fax(doc: Document): void;
  staple(doc: Document): void;
  copy(doc: Document): Document;
}

// A simple printer is forced to implement every method
class SimplePrinter implements MultiFunctionDevice {
  print(doc: Document): void { /* has implementation */ }
  scan(doc: Document): Image { throw new Error("Not supported"); }
  fax(doc: Document): void { throw new Error("Not supported"); }
  staple(doc: Document): void { throw new Error("Not supported"); }
  copy(doc: Document): Document { throw new Error("Not supported"); }
}

// ✅ Applying ISP: separate interfaces per capability
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

// Simple printer: implements only what it needs
class BasicPrinter implements Printer {
  print(doc: Document): void {
    console.log("Printing document...");
  }
}

// Multi-function office printer: implements multiple interfaces
class OfficePrinter implements Printer, Scanner, Faxer, Stapler {
  print(doc: Document): void { /* ... */ }
  scan(doc: Document): Image { /* ... */ }
  fax(doc: Document): void { /* ... */ }
  staple(doc: Document): void { /* ... */ }
}

// Clients depend only on the capabilities they need
function printReport(printer: Printer): void {
  // Depends only on the Printer interface
  // No knowledge of Scanner or Faxer is required
  printer.print(report);
}
```

```python
# Python: Practical ISP example - authentication providers

from abc import ABC, abstractmethod
from typing import Protocol

# ❌ ISP violation: forces every provider to implement methods they don't need
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


# ✅ Applying ISP: split protocols by concern
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


# Basic authentication: no MFA or token management required
class BasicAuthProvider:
    def authenticate(self, username: str, password: str) -> bool:
        # Password verification only
        return check_password(username, password)

    def authorize(self, user_id: str, resource: str) -> bool:
        return check_permission(user_id, resource)


# OAuth: no MFA needed, but token management is required
class OAuthProvider:
    def authenticate(self, username: str, password: str) -> bool: ...
    def authorize(self, user_id: str, resource: str) -> bool: ...
    def refresh_token(self, token: str) -> str: ...
    def revoke_token(self, token: str) -> None: ...
    def get_user_info(self, token: str) -> dict: ...


# Clients depend only on the protocols they need
def login(auth: Authenticator, username: str, password: str) -> bool:
    return auth.authenticate(username, password)

def check_access(authz: Authorizer, user_id: str, resource: str) -> bool:
    return authz.authorize(user_id, resource)
```

### 4.5 D: Dependency Inversion Principle (DIP)

```typescript
// === D: Dependency Inversion ===
// ❌ Depending on a concretion
class OrderService {
  private db = new MySQLDatabase(); // Directly depends on a concrete class
  save(order: Order) { this.db.insert(order); }
}

// ✅ Depending on an abstraction
interface Database { insert(data: any): void; }
class OrderService {
  constructor(private db: Database) {} // Depends on an interface
  save(order: Order) { this.db.insert(order); }
}
```

```typescript
// TypeScript: Practical DIP example - notification system

// ❌ DIP violation: higher-level module depends on concretions of lower-level modules
class UserRegistrationService {
  // Directly instantiates concrete classes → not testable, not swappable
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

// ✅ Applying DIP: both higher and lower modules depend on abstractions

// Define abstractions (interfaces)
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

// Higher-level module: depends only on abstractions
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

// Lower-level modules: implement the abstractions
class PostgresUserRepository implements UserRepository {
  async save(data: CreateUserDto): Promise<User> { /* Postgres implementation */ }
  async findById(id: string): Promise<User | null> { /* Postgres implementation */ }
}

class SendGridEmailClient implements EmailClient {
  async sendWelcome(to: string): Promise<void> { /* SendGrid implementation */ }
  async sendPasswordReset(to: string, token: string): Promise<void> { /* ... */ }
}

class WinstonLogger implements AppLogger {
  info(message: string): void { /* Winston implementation */ }
  error(message: string, error?: Error): void { /* Winston implementation */ }
}

// Test mocks
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

// Tests: inject mocks to test without external dependencies
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
# Python: Practical DIP example - DI container

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Type

# Abstraction definitions
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

# Concrete implementations
class RedisCache(CacheStore):
    def __init__(self, url: str):
        self._url = url

    def get(self, key: str) -> str | None:
        # Redis implementation
        pass

    def set(self, key: str, value: str, ttl: int = 300) -> None:
        # Redis implementation
        pass

class InMemoryCache(CacheStore):
    """For testing / development"""
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
        # RabbitMQ implementation
        pass

    def subscribe(self, topic: str, handler: callable) -> None:
        # RabbitMQ implementation
        pass

# Simple DI container
class Container:
    """Dependency injection container"""
    def __init__(self):
        self._bindings: dict[type, callable] = {}

    def bind(self, abstract: type, factory: callable) -> None:
        self._bindings[abstract] = factory

    def resolve(self, abstract: type):
        factory = self._bindings.get(abstract)
        if factory is None:
            raise ValueError(f"No binding for {abstract}")
        return factory()

# Configuration per environment
def configure_production(container: Container) -> None:
    container.bind(CacheStore, lambda: RedisCache("redis://prod:6379"))
    container.bind(MessageQueue, lambda: RabbitMQQueue("amqp://prod:5672"))

def configure_testing(container: Container) -> None:
    container.bind(CacheStore, lambda: InMemoryCache())
    container.bind(MessageQueue, lambda: InMemoryQueue())

# Usage
container = Container()
configure_production(container)  # Production environment
# configure_testing(container)  # Test environment

cache: CacheStore = container.resolve(CacheStore)
queue: MessageQueue = container.resolve(MessageQueue)
```

---

## 5. Criteria for Applying SOLID

```
Warning on over-application:
  → SOLID is not needed for a 10-line script
  → Excessive abstraction early in a personal project is harmful
  → Balance with YAGNI (You Ain't Gonna Need It)

When to apply:
  ✓ Production code in team development
  ✓ Systems that are maintained for a long time
  ✓ Systems where testing is important
  ✓ Areas where changes occur frequently

Gradual application:
  1. First, write it simply
  2. When a change occurs, apply SOLID to that area
  3. Consider abstraction on the "third change" (Rule of Three)
```

### 5.1 Application Guide by Project Size

```
Small projects (up to 5,000 lines):
  ┌─────────────────────────────────────────┐
  │ Application level: minimal               │
  │ SRP: just file splitting                 │
  │ OCP: not needed (change patterns unclear)│
  │ LSP: only when using inheritance         │
  │ ISP: not needed (few interfaces)         │
  │ DIP: not needed (tests are easy to write)│
  └─────────────────────────────────────────┘

Medium projects (5,000 to 50,000 lines):
  ┌─────────────────────────────────────────┐
  │ Application level: selective             │
  │ SRP: apply aggressively                  │
  │ OCP: apply in high-change areas          │
  │ LSP: always verify inheritance hierarchies│
  │ ISP: split when fat interfaces appear    │
  │ DIP: apply at external service boundaries│
  └─────────────────────────────────────────┘

Large projects (50,000+ lines):
  ┌─────────────────────────────────────────┐
  │ Application level: comprehensive         │
  │ SRP: strictly enforce across all classes │
  │ OCP: apply throughout domain logic       │
  │ LSP: verify contracts in all subtypes    │
  │ ISP: role-based interface design         │
  │ DIP: comprehensive use with DI containers│
  └─────────────────────────────────────────┘
```

### 5.2 Tools and Metrics for Detecting SOLID Violations

```
Static analysis tools:

  TypeScript/JavaScript:
    - ESLint + @typescript-eslint/recommended
    - SonarQube / SonarCloud
    - CodeClimate

  Java:
    - SonarQube
    - PMD
    - Checkstyle
    - SpotBugs (formerly FindBugs)

  Python:
    - pylint
    - flake8 + flake8-import-order
    - mypy (type checking = useful for verifying DIP/LSP)
    - SonarQube

  C#:
    - ReSharper / Rider
    - Roslyn Analyzers
    - NDepend

Key metrics to watch:
  ┌───────────────────┬────────────────────────────────┐
  │ Metric             │ Signs of SOLID violation       │
  ├───────────────────┼────────────────────────────────┤
  │ Class size > 300   │ Possible SRP violation         │
  │ Method count > 15  │ Possible SRP violation         │
  │ Dependencies > 8   │ Possible SRP + DIP violation   │
  │ Cyclomatic         │ Possible OCP violation         │
  │ complexity > 10    │ (many conditional branches)    │
  │ Inheritance        │ Increased risk of LSP violation│
  │ depth > 4          │                                │
  │ Interface method   │ Possible ISP violation         │
  │ count > 10         │                                │
  │ High coupling      │ Possible DIP violation         │
  └───────────────────┴────────────────────────────────┘
```

### 5.3 Step-by-Step Refactoring Procedure

```typescript
// Step 1: Start simple (ignoring SOLID is OK)
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
// → This is sufficient while the code is small
```

```typescript
// Step 2: A change occurs → apply SRP to that area
// Requirements: "Save Todo to a file" "Also send notifications"

// SRP: separate persistence
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
  constructor(private repo: TodoRepository) {} // DIP: depend on abstraction

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
// Step 3: Third change → design extension points with OCP
// Requirements: "Also save to DB" "Also save to S3"

// OCP: new storage only requires adding a class
class DatabaseTodoRepository implements TodoRepository {
  async save(todos: Todo[]): Promise<void> {
    // Save to PostgreSQL
  }
  async load(): Promise<Todo[]> {
    // Retrieve from PostgreSQL
  }
}

class S3TodoRepository implements TodoRepository {
  async save(todos: Todo[]): Promise<void> {
    // Save to AWS S3
  }
  async load(): Promise<Todo[]> {
    // Retrieve from AWS S3
  }
}

// TodoService requires no changes at all (OCP achieved)
```

---

## 6. Relationship Between SOLID and Other Design Principles

```
Correspondence between SOLID and GRASP:

  GRASP (General Responsibility Assignment Software Patterns):
  ┌────────────────────┬─────────────────────────────────┐
  │ GRASP principle     │ Corresponding SOLID principle   │
  ├────────────────────┼─────────────────────────────────┤
  │ Information Expert  │ SRP (proper assignment of       │
  │                    │ responsibilities)                │
  │ Creator            │ DIP (separate creation responsibility)│
  │ Low Coupling       │ DIP + ISP (achieving loose coupling)│
  │ High Cohesion      │ SRP (maintaining high cohesion)  │
  │ Polymorphism       │ OCP + LSP                        │
  │ Pure Fabrication   │ SRP (introducing synthetic classes)│
  │ Indirection        │ DIP (introducing indirection)    │
  │ Protected Variations│ OCP (isolating change impact)   │
  └────────────────────┴─────────────────────────────────┘

SOLID and DRY/KISS/YAGNI:

  DRY (Don't Repeat Yourself):
    → Complementary to SRP: separating responsibilities naturally leads to DRY
    → Caution: pursuing DRY excessively creates wrong abstractions

  KISS (Keep It Simple, Stupid):
    → A principle that curbs over-application of SOLID
    → Five classes for a 10-line script violates KISS

  YAGNI (You Ain't Gonna Need It):
    → Controls the timing of OCP application
    → Don't abstract on the hunch "this might be needed in the future"
    → Introduce interfaces only after 2-3 actual changes have occurred
```

### 6.1 SOLID and Clean Architecture

```
The role of SOLID in Clean Architecture:

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

  Mapping each layer to SOLID:
    Entities (domain layer):
      → SRP: each entity captures a single business rule
      → LSP: equality of Value Objects

    Use Cases (use case layer):
      → SRP: 1 use case = 1 class
      → OCP: extending use cases

    Interface Adapters (adapter layer):
      → DIP: invert dependency on external services
      → ISP: keep ports small and separate

    Frameworks (framework layer):
      → DIP: inner code does not depend on framework details
      → OCP: swapping frameworks requires no changes to inner layers

  Direction of dependency:
    Outer → Inner (achieved via DIP)
    Frameworks → Adapters → Use Cases → Entities
    Inner code is unaware of the existence of outer layers
```

### 6.2 SOLID and Microservices

```
Microservices architecture and SOLID:

  SRP → service boundaries:
    1 microservice = 1 business domain
    ✓ UserService: user management only
    ✓ OrderService: order management only
    ✓ PaymentService: payment management only
    ✗ MonolithService: everything → SRP violation

  OCP → extending services:
    When adding a new service, do not modify existing services
    → Event-driven architecture
    → Loose coupling via message queues

  LSP → backward compatibility of APIs:
    Version 2 of a service should be backward compatible with v1
    → API versioning
    → Consumer-Driven Contract testing (CDC)

  ISP → API granularity:
    Do not cram all information into a single endpoint
    → GraphQL: clients fetch only the fields they need
    → BFF (Backend For Frontend) pattern

  DIP → inter-service communication:
    Do not directly depend on concrete service URLs
    → Service discovery
    → API Gateway
    → Message broker
```

---

## 7. SOLID and Functional Programming

```
SOLID is an OOP-specific principle, but there are corresponding
concepts in functional programming:

  ┌────────────┬─────────────────────────────────────────┐
  │ SOLID      │ Corresponding FP concept                 │
  ├────────────┼─────────────────────────────────────────┤
  │ SRP        │ Pure functions (perform only one calculation)│
  │ OCP        │ Higher-order functions (pass functions   │
  │            │ to extend behavior)                      │
  │ LSP        │ Type class constraints (Haskell type classes)│
  │ ISP        │ Granular type classes                    │
  │ DIP        │ Function injection (callbacks, DI functions)│
  └────────────┴─────────────────────────────────────────┘
```

```typescript
// TypeScript: SOLID in a functional style

// SRP: a pure function = a single responsibility
const calculateTax = (amount: number, rate: number): number =>
  amount * rate;

const formatCurrency = (amount: number): string =>
  `$${amount.toFixed(2)}`;

// OCP: extend via higher-order functions
type Middleware<T> = (data: T) => T;

const applyMiddlewares = <T>(data: T, middlewares: Middleware<T>[]): T =>
  middlewares.reduce((acc, mw) => mw(acc), data);

// Extensible simply by adding new middleware
const addTimestamp: Middleware<Request> = (req) => ({
  ...req,
  timestamp: Date.now(),
});

const addCorrelationId: Middleware<Request> = (req) => ({
  ...req,
  correlationId: crypto.randomUUID(),
});

// DIP: inject functions
type Fetcher = (url: string) => Promise<Response>;
type Parser<T> = (data: string) => T;

const loadData = async <T>(
  url: string,
  fetch: Fetcher,      // Inject a function type instead of a concretion
  parse: Parser<T>,    // Inject the parser as well
): Promise<T> => {
  const response = await fetch(url);
  const text = await response.text();
  return parse(text);
};

// When testing: pass mock functions
const mockFetch: Fetcher = async () =>
  new Response('{"name":"test"}');

const result = await loadData(
  "https://api.example.com/data",
  mockFetch,
  JSON.parse,
);
```

```python
# Python: SOLID in a functional style

from typing import Callable, TypeVar
from functools import reduce

T = TypeVar("T")

# SRP: pure functions
def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[1]

def hash_password(password: str) -> str:
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

# OCP: extend behavior with higher-order functions
Pipeline = list[Callable[[T], T]]

def execute_pipeline(data: T, steps: Pipeline[T]) -> T:
    return reduce(lambda acc, step: step(acc), steps, data)

# Extensible simply by adding steps
def normalize(text: str) -> str:
    return text.strip().lower()

def remove_special_chars(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9\s]", "", text)

def truncate(text: str) -> str:
    return text[:100]

# Pipeline execution
result = execute_pipeline(
    "  Hello, World! @#$ ",
    [normalize, remove_special_chars, truncate]
)
# → "hello world  "

# DIP: function injection
def process_order(
    save: Callable[[dict], None],
    notify: Callable[[str, str], None],
    order: dict,
) -> None:
    save(order)
    notify(order["email"], f"Order {order['id']} confirmed")

# Production
process_order(save=db_save, notify=email_notify, order=order_data)

# Testing
calls = []
process_order(
    save=lambda o: calls.append(("save", o)),
    notify=lambda e, m: calls.append(("notify", e, m)),
    order={"id": "1", "email": "test@test.com"},
)
assert len(calls) == 2
```

---

## 8. SOLID Implementation Patterns by Language

```
Characteristics of each language and SOLID implementation:

  TypeScript:
    - interface + class enables natural SOLID implementation
    - Structural typing makes ISP flexible
    - DI frameworks: tsyringe, inversify, NestJS

  Java:
    - interface + abstract class are supported at the language level
    - Spring Framework standardizes DIP
    - Annotations (@Inject, @Autowired) for DI

  Python:
    - Protocol (structural subtyping) for ISP
    - ABC for explicit interfaces
    - Duck typing naturally realizes OCP

  Kotlin:
    - sealed class improves LSP safety
    - data class for immutable entities
    - Extension functions for OCP

  Rust:
    - trait for ISP + DIP
    - enum + match for type-safe OCP
    - Ownership forces SRP

  Go:
    - Implicit interface implementation makes ISP extremely natural
    - Culture of small interfaces (io.Reader, io.Writer)
    - Struct embedding (Composition over Inheritance)
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important. Understanding deepens when you not only study theory but also actually write code and observe its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend fully understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently utilized in daily development work. It is especially important during code reviews and architecture design.

---

## Summary

| Principle | In one line | Effect | Means | Caveats |
|------|--------|------|---------|--------|
| SRP | 1 class, 1 responsibility | Localize impact of changes | Separation of responsibilities, delegation | Beware of over-splitting |
| OCP | Open to extension, closed to modification | Stability of existing code | Interfaces, polymorphism | Balance with YAGNI |
| LSP | Substitutability | Correctness of polymorphism | Maintaining pre/post-conditions | Violations are hard to find |
| ISP | Keep interfaces small | Eliminate unnecessary dependencies | Role-based interfaces | Beware of excessive fragmentation |
| DIP | Depend on abstractions | Loose coupling and testability | DI, constructor injection | Unnecessary at small scale |

### Checklist

```
SOLID checklist during design review:

  □ SRP: Is there only one reason to change this class?
  □ SRP: Does the class name represent a single responsibility?
  □ OCP: Does adding a new kind require modifying existing code?
  □ OCP: Are there chains of switch/if-else statements?
  □ LSP: Does a subclass work correctly when used in place of its parent?
  □ LSP: Do subclasses throw exceptions that break the parent's contract?
  □ ISP: Does the interface contain methods that are not used?
  □ ISP: Are clients forced to carry unnecessary dependencies?
  □ DIP: Does a higher-level module depend directly on concrete classes?
  □ DIP: Can mocks be swapped in during testing?
```

---

## Recommended Next Guides

---

## References
1. Martin, R. "Agile Software Development, Principles, Patterns, and Practices." Prentice Hall, 2003.
2. Martin, R. "Clean Architecture: A Craftsman's Guide to Software Structure and Design." Prentice Hall, 2017.
3. Martin, R. "The Principles of OOD." butunclebob.com, 2005.
4. Meyer, B. "Object-Oriented Software Construction." Prentice Hall, 2nd ed., 1997.
5. Liskov, B. "Data Abstraction and Hierarchy." SIGPLAN Notices, 1988.
6. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
7. Feathers, M. "Working Effectively with Legacy Code." Prentice Hall, 2004.
8. Fowler, M. "Refactoring: Improving the Design of Existing Code." Addison-Wesley, 2nd ed., 2018.
