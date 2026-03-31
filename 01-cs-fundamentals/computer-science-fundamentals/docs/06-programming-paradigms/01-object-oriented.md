# Object-Oriented Programming

> The essence of OOP lies in the four pillars of "Encapsulation," "Inheritance," "Polymorphism," and "Abstraction" -- a methodology for managing the complexity of large-scale software. When used appropriately, it dramatically improves code reusability, maintainability, and extensibility, but when overused, it becomes a double-edged sword that increases complexity.

## Learning Objectives

- [ ] Explain the four fundamental principles of OOP and demonstrate implementation patterns for each
- [ ] Understand the SOLID principles and explain violation patterns and improvement methods
- [ ] Demonstrate with concrete examples why composition should be preferred over inheritance
- [ ] Implement representative design patterns
- [ ] Understand approaches for blending OOP with other paradigms
- [ ] Apply best practices for class design in real-world projects


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Imperative Programming](./00-imperative.md)

---

## 1. History and Background of OOP

### 1.1 Birth and Evolution of OOP

Object-oriented programming originated in Simula in the 1960s and was formally systematized with Smalltalk in the 1970s. It was subsequently adopted by major languages such as C++, Java, C#, Python, and Ruby, becoming the dominant paradigm in modern software development.

```
OOP Historical Timeline:

1967    Simula 67
        - Developed by Ole-Johan Dahl and Kristen Nygaard
        - First introduction of the concepts of classes and objects
        - Designed as a simulation language

1972    Smalltalk
        - Developed by Alan Kay at Xerox PARC
        - Pure OOP where "everything is an object"
        - Message passing model
        - The concept of GUI also originated here

1979    C++
        - Developed by Bjarne Stroustrup
        - Added OOP features to the C language
        - Multiple inheritance, templates
        - Widely adopted in industry

1995    Java
        - Sun Microsystems (James Gosling)
        - "Write Once, Run Anywhere"
        - Single inheritance + interfaces
        - Garbage collection

1995    JavaScript
        - Developed by Brendan Eich
        - Prototype-based OOP
        - A different approach from class-based OOP

2000    C#
        - Microsoft (.NET Framework)
        - Design influenced by Java
        - Properties, events, delegates

2004    Scala
        - Fusion of OOP and functional programming
        - Traits (mixins)

2014    Swift
        - Developed by Apple
        - Protocol-oriented programming
        - Emphasis on value types (struct)
```

### 1.2 Fundamental Concepts of OOP

The core of OOP is "modeling the real world." Programs are constructed by representing real-world entities as objects and defining the interactions between those objects.

```python
# Fundamental concept of OOP: modeling the real world

# Example: Domain model for an e-commerce site
# Represent real-world concepts as classes

class Customer:
    """Customer - a person who purchases products"""
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
        self._orders: list['Order'] = []

    def place_order(self, cart: 'ShoppingCart') -> 'Order':
        """Create an order from the cart contents"""
        order = Order(customer=self, items=cart.items.copy())
        self._orders.append(order)
        cart.clear()
        return order

    @property
    def order_history(self) -> list['Order']:
        return self._orders.copy()

class Product:
    """Product - an item for sale"""
    def __init__(self, name: str, price: int, stock: int):
        self.name = name
        self.price = price
        self._stock = stock

    def is_available(self) -> bool:
        return self._stock > 0

    def reduce_stock(self, quantity: int) -> None:
        if quantity > self._stock:
            raise ValueError(f"Insufficient stock: {self._stock} remaining")
        self._stock -= quantity

class ShoppingCart:
    """Shopping cart - a temporary collection of products before purchase"""
    def __init__(self):
        self.items: list[tuple[Product, int]] = []

    def add(self, product: Product, quantity: int = 1) -> None:
        if not product.is_available():
            raise ValueError(f"{product.name} is out of stock")
        self.items.append((product, quantity))

    @property
    def total(self) -> int:
        return sum(p.price * q for p, q in self.items)

    def clear(self) -> None:
        self.items.clear()

class Order:
    """Order - a confirmed purchase"""
    _next_id = 1

    def __init__(self, customer: Customer, items: list):
        self.order_id = Order._next_id
        Order._next_id += 1
        self.customer = customer
        self.items = items
        self.status = "pending"

    @property
    def total(self) -> int:
        return sum(p.price * q for p, q in self.items)

# Usage example
customer = Customer("Taro Tanaka", "tanaka@example.com")
laptop = Product("Laptop", 150000, 10)
mouse = Product("Mouse", 3000, 50)

cart = ShoppingCart()
cart.add(laptop, 1)
cart.add(mouse, 2)
print(f"Cart total: {cart.total} yen")  # 156000 yen

order = customer.place_order(cart)
print(f"Order ID: {order.order_id}, Total: {order.total} yen")
```

---

## 2. The Four Pillars of OOP

### 2.1 Encapsulation

Encapsulation is the mechanism of bundling data (state) and the methods (behavior) that operate on that data into a single unit, while restricting unauthorized external access. It is also known as Information Hiding.

```python
# === Encapsulation Basics ===

# Bad: No encapsulation - data is fully exposed
class BadBankAccount:
    def __init__(self):
        self.balance = 0  # Anyone can access freely
        self.transactions = []

# External code can directly modify state
account = BadBankAccount()
account.balance = -1000000  # Invalid state!
account.transactions = []    # History wiped!


# Good: With encapsulation - data is protected
class BankAccount:
    """Bank account - internal state is properly protected"""

    def __init__(self, account_number: str, initial_balance: int = 0):
        self._account_number = account_number  # protected (by convention)
        self.__balance = initial_balance        # private (name mangling)
        self.__transactions: list[dict] = []
        self.__is_frozen = False

    @property
    def balance(self) -> int:
        """Read-only property for balance"""
        return self.__balance

    @property
    def account_number(self) -> str:
        """Read-only property for account number"""
        return self._account_number

    def deposit(self, amount: int, description: str = "") -> bool:
        """Deposit processing"""
        if self.__is_frozen:
            raise RuntimeError("Account is frozen")
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        self.__balance += amount
        self.__record_transaction("deposit", amount, description)
        return True

    def withdraw(self, amount: int, description: str = "") -> bool:
        """Withdrawal processing"""
        if self.__is_frozen:
            raise RuntimeError("Account is frozen")
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.__balance:
            raise ValueError(f"Insufficient balance (balance: {self.__balance} yen)")

        self.__balance -= amount
        self.__record_transaction("withdrawal", amount, description)
        return True

    def get_statement(self, last_n: int = 10) -> list[dict]:
        """Get transaction statement (last N transactions)"""
        return self.__transactions[-last_n:].copy()  # Return a copy

    def freeze(self) -> None:
        """Freeze the account"""
        self.__is_frozen = True

    def __record_transaction(self, tx_type: str, amount: int, desc: str):
        """Internal method: record a transaction"""
        from datetime import datetime
        self.__transactions.append({
            "type": tx_type,
            "amount": amount,
            "description": desc,
            "timestamp": datetime.now().isoformat(),
            "balance_after": self.__balance
        })

# Usage example
account = BankAccount("1234-5678", 100000)
account.deposit(50000, "Salary deposit")
account.withdraw(30000, "Rent")
print(f"Balance: {account.balance} yen")  # 120000 yen

# Unauthorized operations are not possible
# account.__balance = -1000000  # AttributeError
# account.balance = 0           # Cannot set via property
```

```java
// Encapsulation in Java (access modifiers)
public class Employee {
    // Four levels of access modifiers
    // private:   within the class only
    // (default): within the same package
    // protected: same package + subclasses
    // public:    accessible from anywhere

    private String id;
    private String name;
    private int salary;
    private Department department;

    public Employee(String id, String name, int salary) {
        this.id = id;
        setName(name);
        setSalary(salary);
    }

    // Getter: provides read access
    public String getId() { return id; }
    public String getName() { return name; }
    public int getSalary() { return salary; }

    // Setter: controls write access with validation
    public void setName(String name) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Name is required");
        }
        this.name = name.trim();
    }

    public void setSalary(int salary) {
        if (salary < 0) {
            throw new IllegalArgumentException("Salary must be 0 or greater");
        }
        this.salary = salary;
    }

    // Business logic
    public void raiseSalary(double percentage) {
        if (percentage < 0 || percentage > 50) {
            throw new IllegalArgumentException("Raise percentage must be between 0-50%");
        }
        this.salary = (int)(this.salary * (1 + percentage / 100));
    }
}
```

```typescript
// Encapsulation in TypeScript
class UserAccount {
    private _email: string;
    private _passwordHash: string;
    private _loginAttempts: number = 0;
    private _isLocked: boolean = false;
    readonly createdAt: Date;

    constructor(email: string, password: string) {
        this.validateEmail(email);
        this._email = email;
        this._passwordHash = this.hashPassword(password);
        this.createdAt = new Date();
    }

    // get accessor (read-only)
    get email(): string { return this._email; }
    get isLocked(): boolean { return this._isLocked; }

    // Do not expose the password hash
    // get passwordHash is intentionally not defined

    changeEmail(newEmail: string, currentPassword: string): void {
        if (!this.verifyPassword(currentPassword)) {
            throw new Error("Current password is incorrect");
        }
        this.validateEmail(newEmail);
        this._email = newEmail;
    }

    login(password: string): boolean {
        if (this._isLocked) {
            throw new Error("Account is locked");
        }

        if (this.verifyPassword(password)) {
            this._loginAttempts = 0;
            return true;
        }

        this._loginAttempts++;
        if (this._loginAttempts >= 5) {
            this._isLocked = true;
        }
        return false;
    }

    private validateEmail(email: string): void {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            throw new Error("Invalid email address format");
        }
    }

    private hashPassword(password: string): string {
        // In practice, use bcrypt or similar
        return `hashed_${password}`;
    }

    private verifyPassword(password: string): boolean {
        return this._passwordHash === this.hashPassword(password);
    }
}
```

### 2.2 Inheritance

Inheritance is the mechanism of creating a new class (child class/subclass) by inheriting the functionality of an existing class (parent class/superclass). It is used for code reuse and modeling "is-a" relationships.

```python
# === Basic Inheritance Patterns ===

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Optional


# Base class (parent class)
class Employee(ABC):
    """Base class for employees"""

    _next_id = 1

    def __init__(self, name: str, hire_date: date, base_salary: int):
        self.employee_id = Employee._next_id
        Employee._next_id += 1
        self.name = name
        self.hire_date = hire_date
        self._base_salary = base_salary

    @property
    def years_of_service(self) -> int:
        """Years of service"""
        return (date.today() - self.hire_date).days // 365

    @abstractmethod
    def calculate_pay(self) -> int:
        """Calculate monthly salary (must be implemented by subclasses)"""
        pass

    @abstractmethod
    def get_role(self) -> str:
        """Return the job title"""
        pass

    def __str__(self) -> str:
        return f"{self.get_role()}: {self.name} (ID: {self.employee_id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, id={self.employee_id})"


# Full-time employee
class FullTimeEmployee(Employee):
    """Full-time employee"""

    def __init__(self, name: str, hire_date: date, base_salary: int,
                 bonus_rate: float = 0.1):
        super().__init__(name, hire_date, base_salary)
        self.bonus_rate = bonus_rate

    def calculate_pay(self) -> int:
        # Including seniority-based raise
        seniority_bonus = self.years_of_service * 5000
        return self._base_salary + seniority_bonus

    def calculate_annual_bonus(self) -> int:
        """Calculate annual bonus"""
        return int(self._base_salary * 12 * self.bonus_rate)

    def get_role(self) -> str:
        return "Full-time Employee"


# Contract employee
class ContractEmployee(Employee):
    """Contract employee"""

    def __init__(self, name: str, hire_date: date, base_salary: int,
                 contract_end: date):
        super().__init__(name, hire_date, base_salary)
        self.contract_end = contract_end

    def calculate_pay(self) -> int:
        return self._base_salary

    def is_contract_active(self) -> bool:
        return date.today() <= self.contract_end

    def get_role(self) -> str:
        return "Contract Employee"


# Part-time employee
class PartTimeEmployee(Employee):
    """Part-time employee"""

    def __init__(self, name: str, hire_date: date, hourly_rate: int,
                 hours_per_month: int):
        super().__init__(name, hire_date, 0)
        self.hourly_rate = hourly_rate
        self.hours_per_month = hours_per_month

    def calculate_pay(self) -> int:
        return self.hourly_rate * self.hours_per_month

    def get_role(self) -> str:
        return "Part-time Employee"


# Manager (extension of full-time employee)
class Manager(FullTimeEmployee):
    """Manager - adds management capabilities to full-time employee"""

    def __init__(self, name: str, hire_date: date, base_salary: int,
                 management_allowance: int = 50000):
        super().__init__(name, hire_date, base_salary, bonus_rate=0.15)
        self.management_allowance = management_allowance
        self._subordinates: list[Employee] = []

    def calculate_pay(self) -> int:
        return super().calculate_pay() + self.management_allowance

    def add_subordinate(self, employee: Employee) -> None:
        if employee not in self._subordinates:
            self._subordinates.append(employee)

    def get_subordinates(self) -> list[Employee]:
        return self._subordinates.copy()

    def get_role(self) -> str:
        return "Manager"


# Leveraging polymorphism
def print_payroll(employees: list[Employee]) -> None:
    """Print payroll for all employees - processes uniformly regardless of type"""
    total = 0
    for emp in employees:
        pay = emp.calculate_pay()
        total += pay
        print(f"  {emp} -> Monthly salary: {pay:,} yen")
    print(f"  {'─' * 40}")
    print(f"  Total: {total:,} yen")


# Usage example
employees = [
    Manager("Director Sato", date(2015, 4, 1), 500000),
    FullTimeEmployee("Ichiro Tanaka", date(2020, 4, 1), 300000),
    ContractEmployee("Jiro Suzuki", date(2024, 4, 1), 350000, date(2026, 3, 31)),
    PartTimeEmployee("Hanako Yamada", date(2023, 10, 1), 1200, 80),
]
print_payroll(employees)
```

```java
// Inheritance and interfaces in Java

// Interface: defines a contract (what can be done)
interface Payable {
    int calculatePay();
}

interface Reportable {
    String generateReport();
}

// Abstract class: provides common implementation
abstract class Employee implements Payable, Reportable {
    protected final String name;
    protected final String employeeId;
    protected int baseSalary;

    protected Employee(String name, String employeeId, int baseSalary) {
        this.name = name;
        this.employeeId = employeeId;
        this.baseSalary = baseSalary;
    }

    // Template Method pattern
    @Override
    public final String generateReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Employee Report ===\n");
        sb.append("Name: ").append(name).append("\n");
        sb.append("ID: ").append(employeeId).append("\n");
        sb.append("Monthly salary: ").append(calculatePay()).append(" yen\n");
        appendExtraInfo(sb);  // Customizable by subclasses
        return sb.toString();
    }

    // Subclasses can override to add information to the report
    protected void appendExtraInfo(StringBuilder sb) {
        // Default does nothing
    }

    public abstract String getRole();
}

// Concrete class
class FullTimeEmployee extends Employee {
    private double bonusRate;

    public FullTimeEmployee(String name, String id, int salary, double bonusRate) {
        super(name, id, salary);
        this.bonusRate = bonusRate;
    }

    @Override
    public int calculatePay() {
        return baseSalary;
    }

    public int calculateAnnualBonus() {
        return (int)(baseSalary * 12 * bonusRate);
    }

    @Override
    public String getRole() { return "Full-time Employee"; }

    @Override
    protected void appendExtraInfo(StringBuilder sb) {
        sb.append("Annual bonus: ").append(calculateAnnualBonus()).append(" yen\n");
    }
}
```

### 2.3 Polymorphism

Polymorphism means "many forms" -- it is the mechanism of providing different implementations for the same interface. There are two types: compile-time polymorphism (method overloading) and runtime polymorphism (method overriding).

```python
# === Practical Examples of Polymorphism ===

from abc import ABC, abstractmethod
from typing import Protocol


# 1. Polymorphism via abstract base classes
class NotificationSender(ABC):
    """Abstract interface for sending notifications"""

    @abstractmethod
    def send(self, recipient: str, message: str) -> bool:
        """Send a notification"""
        pass

    @abstractmethod
    def get_channel_name(self) -> str:
        """Return the channel name"""
        pass


class EmailSender(NotificationSender):
    def __init__(self, smtp_server: str):
        self.smtp_server = smtp_server

    def send(self, recipient: str, message: str) -> bool:
        print(f"[Email] Sending to {recipient}: {message}")
        # SMTP send processing...
        return True

    def get_channel_name(self) -> str:
        return "Email"


class SlackSender(NotificationSender):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, recipient: str, message: str) -> bool:
        print(f"[Slack] Sending to #{recipient}: {message}")
        # Webhook POST processing...
        return True

    def get_channel_name(self) -> str:
        return "Slack"


class SMSSender(NotificationSender):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def send(self, recipient: str, message: str) -> bool:
        print(f"[SMS] Sending to {recipient}: {message}")
        # SMS API call...
        return True

    def get_channel_name(self) -> str:
        return "SMS"


class LineSender(NotificationSender):
    def __init__(self, channel_token: str):
        self.channel_token = channel_token

    def send(self, recipient: str, message: str) -> bool:
        print(f"[LINE] Sending to {recipient}: {message}")
        return True

    def get_channel_name(self) -> str:
        return "LINE"


# Notification service that leverages polymorphism
class NotificationService:
    """Manages notification sending uniformly"""

    def __init__(self):
        self._senders: list[NotificationSender] = []

    def register_sender(self, sender: NotificationSender) -> None:
        self._senders.append(sender)

    def broadcast(self, message: str, recipients: dict[str, str]) -> dict:
        """Send notifications via all channels"""
        results = {}
        for sender in self._senders:
            channel = sender.get_channel_name()
            if channel in recipients:
                success = sender.send(recipients[channel], message)
                results[channel] = success
        return results


# Usage example: works without knowing the specific type of sender
service = NotificationService()
service.register_sender(EmailSender("smtp.example.com"))
service.register_sender(SlackSender("https://hooks.slack.com/xxx"))
service.register_sender(SMSSender("api-key-123"))
service.register_sender(LineSender("channel-token-abc"))

results = service.broadcast(
    "A server failure has occurred",
    {
        "Email": "admin@example.com",
        "Slack": "alerts",
        "SMS": "090-1234-5678",
        "LINE": "U1234567890"
    }
)
```

```python
# 2. Polymorphism via Protocol (structural subtyping, Python 3.8+)

from typing import Protocol, runtime_checkable


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects"""
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...


@runtime_checkable
class Validatable(Protocol):
    """Protocol for validatable objects"""
    def validate(self) -> list[str]: ...
    def is_valid(self) -> bool: ...


# Class satisfying the Protocol (no explicit inheritance needed)
class UserProfile:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def to_dict(self) -> dict:
        return {"name": self.name, "age": self.age, "email": self.email}

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def validate(self) -> list[str]:
        errors = []
        if not self.name:
            errors.append("Name is required")
        if self.age < 0 or self.age > 150:
            errors.append("Age is invalid")
        if "@" not in self.email:
            errors.append("Email address is invalid")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


# Generic functions using Protocol
def save_to_file(obj: Serializable, filepath: str) -> None:
    """Save a Serializable object to a file"""
    with open(filepath, 'w') as f:
        f.write(obj.to_json())

def validate_and_report(obj: Validatable) -> None:
    """Validate a Validatable object and report"""
    errors = obj.validate()
    if errors:
        print(f"Validation errors: {', '.join(errors)}")
    else:
        print("Validation successful")

# Usage example
user = UserProfile("Taro Tanaka", 30, "tanaka@example.com")
print(isinstance(user, Serializable))   # True (structural subtyping)
print(isinstance(user, Validatable))    # True
save_to_file(user, "/tmp/user.json")
validate_and_report(user)
```

```typescript
// Polymorphism in TypeScript (structural subtyping is standard)

// Interface definitions
interface Shape {
    area(): number;
    perimeter(): number;
    describe(): string;
}

interface Drawable {
    draw(ctx: CanvasRenderingContext2D): void;
}

// Implementing multiple interfaces
class Circle implements Shape, Drawable {
    constructor(
        public readonly centerX: number,
        public readonly centerY: number,
        public readonly radius: number
    ) {}

    area(): number {
        return Math.PI * this.radius ** 2;
    }

    perimeter(): number {
        return 2 * Math.PI * this.radius;
    }

    describe(): string {
        return `Circle (radius: ${this.radius})`;
    }

    draw(ctx: CanvasRenderingContext2D): void {
        ctx.beginPath();
        ctx.arc(this.centerX, this.centerY, this.radius, 0, Math.PI * 2);
        ctx.stroke();
    }
}

class Rectangle implements Shape, Drawable {
    constructor(
        public readonly x: number,
        public readonly y: number,
        public readonly width: number,
        public readonly height: number
    ) {}

    area(): number {
        return this.width * this.height;
    }

    perimeter(): number {
        return 2 * (this.width + this.height);
    }

    describe(): string {
        return `Rectangle (${this.width} x ${this.height})`;
    }

    draw(ctx: CanvasRenderingContext2D): void {
        ctx.strokeRect(this.x, this.y, this.width, this.height);
    }
}

class Triangle implements Shape {
    constructor(
        public readonly a: number,
        public readonly b: number,
        public readonly c: number
    ) {}

    area(): number {
        // Heron's formula
        const s = (this.a + this.b + this.c) / 2;
        return Math.sqrt(s * (s - this.a) * (s - this.b) * (s - this.c));
    }

    perimeter(): number {
        return this.a + this.b + this.c;
    }

    describe(): string {
        return `Triangle (sides: ${this.a}, ${this.b}, ${this.c})`;
    }
}

// Functions leveraging polymorphism
function printShapeInfo(shapes: Shape[]): void {
    for (const shape of shapes) {
        console.log(`${shape.describe()}: area=${shape.area().toFixed(2)}, perimeter=${shape.perimeter().toFixed(2)}`);
    }
}

function totalArea(shapes: Shape[]): number {
    return shapes.reduce((sum, shape) => sum + shape.area(), 0);
}
```

### 2.4 Abstraction

Abstraction is the concept of hiding complex implementation details and exposing only the essential interface. Users only need to know "what can be done," not "how it is achieved."

```python
# === Practical Example of Abstraction: Database Access Layer ===

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Abstract representation of query results"""
    rows: list[dict[str, Any]]
    row_count: int
    affected_rows: int = 0


class DatabaseConnection(ABC):
    """Abstract interface for database connections

    Users only need to know this interface and don't need to
    be aware of the specific DB (PostgreSQL, MySQL, SQLite, etc.).
    """

    @abstractmethod
    def connect(self) -> None:
        """Connect to the database"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the database"""
        pass

    @abstractmethod
    def execute(self, query: str, params: Optional[tuple] = None) -> QueryResult:
        """Execute an SQL query"""
        pass

    @abstractmethod
    def begin_transaction(self) -> None:
        """Begin a transaction"""
        pass

    @abstractmethod
    def commit(self) -> None:
        """Commit a transaction"""
        pass

    @abstractmethod
    def rollback(self) -> None:
        """Roll back a transaction"""
        pass

    # Can also be used as a context manager (template method)
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        self.disconnect()
        return False


class PostgreSQLConnection(DatabaseConnection):
    """Concrete implementation for PostgreSQL"""

    def __init__(self, host: str, port: int, dbname: str, user: str, password: str):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self._conn = None

    def connect(self) -> None:
        import psycopg2
        self._conn = psycopg2.connect(
            host=self.host, port=self.port,
            dbname=self.dbname, user=self.user, password=self.password
        )

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, query: str, params: Optional[tuple] = None) -> QueryResult:
        cursor = self._conn.cursor()
        cursor.execute(query, params)

        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return QueryResult(rows=rows, row_count=len(rows))
        else:
            return QueryResult(rows=[], row_count=0, affected_rows=cursor.rowcount)

    def begin_transaction(self) -> None:
        self._conn.autocommit = False

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()


class SQLiteConnection(DatabaseConnection):
    """Concrete implementation for SQLite"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None

    def connect(self) -> None:
        import sqlite3
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, query: str, params: Optional[tuple] = None) -> QueryResult:
        cursor = self._conn.cursor()
        cursor.execute(query, params or ())

        if cursor.description:
            rows = [dict(row) for row in cursor.fetchall()]
            return QueryResult(rows=rows, row_count=len(rows))
        else:
            return QueryResult(rows=[], row_count=0, affected_rows=cursor.rowcount)

    def begin_transaction(self) -> None:
        self._conn.execute("BEGIN")

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()


# Repository using the abstracted interface
class UserRepository:
    """User repository - not dependent on a specific DB"""

    def __init__(self, db: DatabaseConnection):
        self._db = db  # Depends on abstraction (DI)

    def find_by_id(self, user_id: int) -> Optional[dict]:
        result = self._db.execute(
            "SELECT * FROM users WHERE id = %s", (user_id,)
        )
        return result.rows[0] if result.rows else None

    def find_all(self) -> list[dict]:
        result = self._db.execute("SELECT * FROM users ORDER BY id")
        return result.rows

    def create(self, name: str, email: str) -> int:
        result = self._db.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s) RETURNING id",
            (name, email)
        )
        return result.rows[0]["id"]


# Usage example: DB implementation is interchangeable
# Development environment
# db = SQLiteConnection(":memory:")
# Production environment
# db = PostgreSQLConnection("db.example.com", 5432, "myapp", "user", "pass")
# repo = UserRepository(db)
```

---

## 3. SOLID Principles

The SOLID principles are five object-oriented design principles proposed by Robert C. Martin (Uncle Bob). They serve as guidelines for building software with high maintainability, extensibility, and testability.

### 3.1 S -- Single Responsibility Principle (SRP)

```
Single Responsibility Principle (SRP):
  "A class should have only one reason to change"

  More precisely:
  "A class should be responsible to only one actor"
  (Actor = a person or organization that requests changes to the class)
```

```python
# Bad: SRP violation - one class has multiple responsibilities
class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def save_to_database(self):
        """Save to database -> persistence responsibility"""
        pass

    def send_welcome_email(self):
        """Send welcome email -> notification responsibility"""
        pass

    def generate_report(self):
        """Generate report -> reporting responsibility"""
        pass

    def validate(self):
        """Validation -> validation responsibility"""
        pass


# Good: SRP compliant - each class has only one responsibility
class User:
    """User domain model (business rules only)"""
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email


class UserValidator:
    """User data validation"""
    @staticmethod
    def validate(user: User) -> list[str]:
        errors = []
        if not user.name or len(user.name) < 2:
            errors.append("Name must be at least 2 characters")
        if "@" not in user.email:
            errors.append("Please enter a valid email address")
        return errors


class UserRepository:
    """User persistence"""
    def __init__(self, db_connection):
        self._db = db_connection

    def save(self, user: User) -> int:
        # DB save processing
        pass

    def find_by_email(self, email: str) -> Optional[User]:
        # DB search processing
        pass


class UserNotifier:
    """User notifications"""
    def __init__(self, email_service):
        self._email_service = email_service

    def send_welcome(self, user: User) -> None:
        self._email_service.send(
            to=user.email,
            subject="Welcome!",
            body=f"Dear {user.name}, thank you for registering."
        )


class UserReportGenerator:
    """User-related report generation"""
    def generate_activity_report(self, user: User) -> str:
        # Report generation processing
        pass
```

### 3.2 O -- Open/Closed Principle (OCP)

```
Open/Closed Principle (OCP):
  "Software entities should be open for extension
   and closed for modification"

  -> Design so that existing code does not need to be modified when adding new features
```

```python
# Bad: OCP violation - existing code must be modified every time a new discount type is added
class DiscountCalculator:
    def calculate(self, order_total: int, discount_type: str) -> int:
        if discount_type == "percentage":
            return int(order_total * 0.9)
        elif discount_type == "fixed":
            return order_total - 1000
        elif discount_type == "member":
            return int(order_total * 0.85)
        # Add an elif here every time a new discount is added...
        else:
            return order_total


# Good: OCP compliant - no need to modify existing code when adding new discounts
class DiscountStrategy(ABC):
    """Abstract base class for discount strategies"""

    @abstractmethod
    def apply(self, order_total: int) -> int:
        pass

    @abstractmethod
    def description(self) -> str:
        pass


class PercentageDiscount(DiscountStrategy):
    def __init__(self, rate: float):
        self.rate = rate

    def apply(self, order_total: int) -> int:
        return int(order_total * (1 - self.rate))

    def description(self) -> str:
        return f"{int(self.rate * 100)}% discount"


class FixedAmountDiscount(DiscountStrategy):
    def __init__(self, amount: int):
        self.amount = amount

    def apply(self, order_total: int) -> int:
        return max(0, order_total - self.amount)

    def description(self) -> str:
        return f"{self.amount} yen off"


class MemberDiscount(DiscountStrategy):
    def __init__(self, member_rank: str):
        self.member_rank = member_rank
        self._rates = {"gold": 0.15, "silver": 0.10, "bronze": 0.05}

    def apply(self, order_total: int) -> int:
        rate = self._rates.get(self.member_rank, 0)
        return int(order_total * (1 - rate))

    def description(self) -> str:
        return f"Member discount ({self.member_rank})"


# Adding new discounts requires no changes to existing code
class CouponDiscount(DiscountStrategy):
    """Coupon discount - added later"""
    def __init__(self, coupon_code: str, discount_rate: float):
        self.coupon_code = coupon_code
        self.discount_rate = discount_rate

    def apply(self, order_total: int) -> int:
        return int(order_total * (1 - self.discount_rate))

    def description(self) -> str:
        return f"Coupon {self.coupon_code}"


class TimeLimitedDiscount(DiscountStrategy):
    """Time-limited discount - added even later"""
    def __init__(self, name: str, rate: float, end_date: date):
        self._name = name
        self._rate = rate
        self._end_date = end_date

    def apply(self, order_total: int) -> int:
        if date.today() <= self._end_date:
            return int(order_total * (1 - self._rate))
        return order_total

    def description(self) -> str:
        return f"Limited-time: {self._name}"


# Code that applies discounts (no modification needed)
class OrderProcessor:
    def apply_discount(self, total: int, strategy: DiscountStrategy) -> int:
        discounted = strategy.apply(total)
        print(f"  {strategy.description()}: {total:,} yen -> {discounted:,} yen")
        return discounted

    def apply_best_discount(self, total: int, strategies: list[DiscountStrategy]) -> int:
        """Apply the best discount"""
        best = min(strategies, key=lambda s: s.apply(total))
        return self.apply_discount(total, best)
```

### 3.3 L -- Liskov Substitution Principle (LSP)

```
Liskov Substitution Principle (LSP):
  "Subtypes must be substitutable for their base types"

  -> When replacing a parent class with a child class in any part
    of the program, the program's correctness must be preserved
```

```python
# Bad: LSP violation - classic "Square-Rectangle" problem
class Rectangle:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value

    def area(self) -> int:
        return self._width * self._height


class Square(Rectangle):
    """Square - width and height are always equal"""
    def __init__(self, side: int):
        super().__init__(side, side)

    @Rectangle.width.setter
    def width(self, value: int):
        self._width = value
        self._height = value  # Height also changes!

    @Rectangle.height.setter
    def height(self, value: int):
        self._width = value  # Width also changes!
        self._height = value


# This function assumes the Rectangle contract
def test_rectangle(rect: Rectangle):
    rect.width = 5
    rect.height = 10
    assert rect.area() == 50  # Fails when Square is passed! (area = 100)


# Good: LSP compliant - designed with a common interface
class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

class Square(Shape):
    def __init__(self, side: float):
        self.side = side

    def area(self) -> float:
        return self.side ** 2
```

```python
# Bad: LSP violation - another classic example
class Bird:
    def fly(self):
        print("Flying")

class Penguin(Bird):
    def fly(self):
        raise NotImplementedError("Penguins cannot fly!")


# Good: LSP compliant - properly separated interfaces
class Bird(ABC):
    @abstractmethod
    def move(self) -> str:
        pass

class FlyingBird(Bird):
    def move(self) -> str:
        return "Flies through the sky"

    def fly(self) -> str:
        return "Flapping wings to fly"

class SwimmingBird(Bird):
    def move(self) -> str:
        return "Swims through water"

    def swim(self) -> str:
        return "Swimming underwater"

class Sparrow(FlyingBird):
    pass

class Penguin(SwimmingBird):
    pass

# Can be handled uniformly as a list of Birds
birds: list[Bird] = [Sparrow(), Penguin()]
for bird in birds:
    print(bird.move())  # All work correctly
```

### 3.4 I -- Interface Segregation Principle (ISP)

```
Interface Segregation Principle (ISP):
  "Clients should not be forced to depend on methods they do not use"

  -> Split large interfaces into smaller interfaces based on usage
```

```python
# Bad: ISP violation - a monolithic interface
class Machine(ABC):
    @abstractmethod
    def print_document(self, doc): pass

    @abstractmethod
    def scan_document(self, doc): pass

    @abstractmethod
    def fax_document(self, doc): pass

    @abstractmethod
    def staple_pages(self, pages): pass


# Even a simple printer is forced to implement all methods
class SimplePrinter(Machine):
    def print_document(self, doc):
        print("Printing...")

    def scan_document(self, doc):
        raise NotImplementedError("Scanning not supported")  # Unnecessary implementation

    def fax_document(self, doc):
        raise NotImplementedError("Fax not supported")  # Unnecessary implementation

    def staple_pages(self, pages):
        raise NotImplementedError("Stapling not supported")  # Unnecessary implementation


# Good: ISP compliant - split into small interfaces
class Printer(ABC):
    @abstractmethod
    def print_document(self, doc) -> None: pass

class Scanner(ABC):
    @abstractmethod
    def scan_document(self, doc) -> bytes: pass

class Fax(ABC):
    @abstractmethod
    def fax_document(self, doc, number: str) -> bool: pass

class Stapler(ABC):
    @abstractmethod
    def staple_pages(self, pages) -> None: pass


# Implement only the necessary interfaces
class SimplePrinter(Printer):
    def print_document(self, doc) -> None:
        print("Printing...")

class MultiFunctionPrinter(Printer, Scanner, Fax):
    def print_document(self, doc) -> None:
        print("Printing...")

    def scan_document(self, doc) -> bytes:
        return b"scanned_data"

    def fax_document(self, doc, number: str) -> bool:
        print(f"Sending fax... {number}")
        return True

# Functions can accept precisely-typed parameters
def do_printing(printer: Printer, doc) -> None:
    printer.print_document(doc)  # Works with both SimplePrinter and MultiFunctionPrinter
```

### 3.5 D -- Dependency Inversion Principle (DIP)

```
Dependency Inversion Principle (DIP):
  "High-level modules should not depend on low-level modules.
   Both should depend on abstractions"

  -> Depend on abstractions (interfaces), not on concrete implementations
```

```python
# Bad: DIP violation - high-level module directly depends on low-level module
class MySQLDatabase:
    def query(self, sql: str) -> list:
        # MySQL-specific processing
        pass

class UserService:
    def __init__(self):
        self.db = MySQLDatabase()  # Direct dependency on concrete class!

    def get_user(self, user_id: int):
        return self.db.query(f"SELECT * FROM users WHERE id = {user_id}")


# Good: DIP compliant - depends on abstraction, uses dependency injection (DI)
class Database(ABC):
    """Abstract interface for database"""
    @abstractmethod
    def query(self, sql: str, params: tuple = ()) -> list: pass

    @abstractmethod
    def execute(self, sql: str, params: tuple = ()) -> int: pass


class MySQLDatabase(Database):
    def __init__(self, connection_string: str):
        self._conn_str = connection_string

    def query(self, sql: str, params: tuple = ()) -> list:
        # MySQL-specific processing
        pass

    def execute(self, sql: str, params: tuple = ()) -> int:
        pass


class PostgreSQLDatabase(Database):
    def __init__(self, connection_string: str):
        self._conn_str = connection_string

    def query(self, sql: str, params: tuple = ()) -> list:
        # PostgreSQL-specific processing
        pass

    def execute(self, sql: str, params: tuple = ()) -> int:
        pass


class InMemoryDatabase(Database):
    """In-memory DB for testing"""
    def __init__(self):
        self._data: dict[str, list] = {}

    def query(self, sql: str, params: tuple = ()) -> list:
        return []

    def execute(self, sql: str, params: tuple = ()) -> int:
        return 0


class UserService:
    """User service - depends on abstraction"""

    def __init__(self, db: Database):  # Receives abstract type (DI)
        self._db = db

    def get_user(self, user_id: int) -> dict:
        results = self._db.query(
            "SELECT * FROM users WHERE id = %s", (user_id,)
        )
        return results[0] if results else None

    def create_user(self, name: str, email: str) -> int:
        return self._db.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s)",
            (name, email)
        )


# Switch via DI based on environment
import os
def create_user_service() -> UserService:
    env = os.getenv("APP_ENV", "development")

    if env == "production":
        db = PostgreSQLDatabase("postgresql://prod-server/myapp")
    elif env == "development":
        db = MySQLDatabase("mysql://localhost/myapp_dev")
    else:  # test
        db = InMemoryDatabase()

    return UserService(db)
```

---

## 4. Inheritance vs Composition

### 4.1 Problems with Inheritance

```python
# === Problems caused by overuse of inheritance ===

# Bad: Deep inheritance tree (Fragile Base Class Problem)
class Animal: pass
class Mammal(Animal): pass
class DomesticMammal(Mammal): pass
class Dog(DomesticMammal): pass
class GuideDog(Dog): pass  # 5 levels of inheritance!

# Problems:
# 1. Changes to the parent class propagate to all child classes (fragile base class problem)
# 2. Must read all levels to understand the hierarchy
# 3. Difficult to add new classifications (a swimming dog? Where does SwimmingDog go?)


# Bad: Diamond problem with multiple inheritance (Python)
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")

class C(A):
    def method(self):
        print("C")

class D(B, C):  # Inherits from both B and C
    pass

d = D()
d.method()  # "B" is called (based on MRO)
# -> Unpredictable behavior
print(D.__mro__)  # Can check the Method Resolution Order
```

### 4.2 Advantages of Composition

```python
# Good: Composition - combine functionality as parts

# Define abilities as classes
class WalkAbility:
    def walk(self, distance: int) -> str:
        return f"Walked {distance}m"

class SwimAbility:
    def swim(self, distance: int) -> str:
        return f"Swam {distance}m"

class FlyAbility:
    def fly(self, distance: int) -> str:
        return f"Flew {distance}m"

class BarkAbility:
    def bark(self) -> str:
        return "Woof woof!"

class GuideAbility:
    def __init__(self):
        self._is_trained = False

    def train(self) -> None:
        self._is_trained = True

    def guide(self, destination: str) -> str:
        if not self._is_trained:
            raise RuntimeError("Training not completed")
        return f"Guiding to {destination}"


# Animal classes hold abilities via composition
class Dog:
    """Dog - has walking and barking abilities"""
    def __init__(self, name: str):
        self.name = name
        self.walker = WalkAbility()
        self.barker = BarkAbility()
        self.swimmer: SwimAbility | None = None  # Optional
        self.guide: GuideAbility | None = None     # Optional

    def make_guide_dog(self) -> None:
        """Make into a guide dog"""
        self.guide = GuideAbility()
        self.guide.train()

    def make_swimmer(self) -> None:
        """Enable swimming"""
        self.swimmer = SwimAbility()


class Duck:
    """Duck - has walking, flying, and swimming abilities"""
    def __init__(self, name: str):
        self.name = name
        self.walker = WalkAbility()
        self.flyer = FlyAbility()
        self.swimmer = SwimAbility()


# Usage example
dog = Dog("Pochi")
print(dog.walker.walk(100))    # Walked 100m
print(dog.barker.bark())       # Woof woof!

dog.make_guide_dog()
print(dog.guide.guide("the station"))   # Guiding to the station

duck = Duck("Quackers")
print(duck.walker.walk(50))    # Walked 50m
print(duck.flyer.fly(200))     # Flew 200m
print(duck.swimmer.swim(30))   # Swam 30m
```

### 4.3 When to Use Inheritance vs Composition

```
Inheritance vs Composition: Guidelines for Choosing

  When to use inheritance:
  +----------------------------------------------------+
  | When there is a clear "is-a" relationship           |
  |    -> Dog is an Animal                              |
  | When the Liskov Substitution Principle can be met   |
  |    -> Subclass fully honors the parent's contract   |
  | When the inheritance hierarchy is shallow (2-3 levels)|
  | When required by a framework                        |
  |    -> Django's Model, View, etc.                    |
  +----------------------------------------------------+

  When to use composition:
  +----------------------------------------------------+
  | When there is a "has-a" relationship                |
  |    -> Car has an Engine                             |
  | When combining multiple functionalities             |
  | When behavior needs to change dynamically at runtime|
  | When you want to swap in mocks for testing          |
  | When the inheritance hierarchy would become deep    |
  +----------------------------------------------------+

  When in doubt, choose composition (GoF maxim):
  "Favor composition over inheritance"
```

### 4.4 Mixin/Trait Pattern

```python
# Mixin: an intermediate approach between inheritance and composition

import json
import logging
from datetime import datetime


class TimestampMixin:
    """Mixin that provides timestamp functionality"""
    created_at: datetime
    updated_at: datetime

    def init_timestamps(self):
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def touch(self):
        """Update the updated_at timestamp to the current time"""
        self.updated_at = datetime.now()


class SerializableMixin:
    """Mixin that provides JSON serialization functionality"""

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict):
        instance = cls.__new__(cls)
        for key, value in data.items():
            setattr(instance, key, value)
        return instance


class LoggableMixin:
    """Mixin that provides logging functionality"""

    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    def log_info(self, message: str):
        self._logger.info(f"[{self.__class__.__name__}] {message}")

    def log_error(self, message: str):
        self._logger.error(f"[{self.__class__.__name__}] {message}")


class ValidatableMixin:
    """Mixin that provides validation functionality"""

    def validate(self) -> list[str]:
        errors = []
        for attr_name in dir(self):
            if attr_name.startswith('validate_'):
                method = getattr(self, attr_name)
                error = method()
                if error:
                    errors.append(error)
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


# Combining mixins
class Article(TimestampMixin, SerializableMixin, LoggableMixin, ValidatableMixin):
    def __init__(self, title: str, content: str, author: str):
        self.title = title
        self.content = content
        self.author = author
        self.init_timestamps()
        self.log_info(f"Created article '{title}'")

    def validate_title(self) -> str | None:
        if not self.title or len(self.title) < 5:
            return "Title must be at least 5 characters"
        return None

    def validate_content(self) -> str | None:
        if not self.content or len(self.content) < 100:
            return "Content must be at least 100 characters"
        return None


# Usage example
article = Article("Python Intro", "Python is..." * 50, "Taro Tanaka")
print(article.to_json())           # SerializableMixin
print(article.is_valid())          # ValidatableMixin
article.touch()                     # TimestampMixin
article.log_info("Article updated") # LoggableMixin
```

---

## 5. Practical Class Design Patterns

### 5.1 Value Object

```python
from dataclasses import dataclass


@dataclass(frozen=True)  # Immutable object
class Money:
    """Value object representing an amount of money"""
    amount: int
    currency: str = "JPY"

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Amount must be 0 or greater")
        if self.currency not in ("JPY", "USD", "EUR"):
            raise ValueError(f"Unsupported currency: {self.currency}")

    def __add__(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("Cannot subtract different currencies")
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: int | float) -> 'Money':
        return Money(int(self.amount * factor), self.currency)

    def format(self) -> str:
        if self.currency == "JPY":
            return f"\u00a5{self.amount:,}"
        elif self.currency == "USD":
            return f"${self.amount / 100:.2f}"
        elif self.currency == "EUR":
            return f"\u20ac{self.amount / 100:.2f}"
        return f"{self.amount} {self.currency}"


@dataclass(frozen=True)
class EmailAddress:
    """Value object for email addresses"""
    value: str

    def __post_init__(self):
        if not self._is_valid_email(self.value):
            raise ValueError(f"Invalid email address: {self.value}")

    @staticmethod
    def _is_valid_email(email: str) -> bool:
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @property
    def domain(self) -> str:
        return self.value.split('@')[1]

    @property
    def local_part(self) -> str:
        return self.value.split('@')[0]


@dataclass(frozen=True)
class DateRange:
    """Value object for date ranges"""
    start: date
    end: date

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("Start date must be before end date")

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end

    def overlaps(self, other: 'DateRange') -> bool:
        return self.start <= other.end and other.start <= self.end


# Usage example
price = Money(1500)
tax = price * 0.1
total = price + tax
print(total.format())  # \u00a51,650

email = EmailAddress("user@example.com")
print(email.domain)  # example.com

period = DateRange(date(2025, 1, 1), date(2025, 12, 31))
print(period.days)  # 364
print(period.contains(date(2025, 6, 15)))  # True
```

### 5.2 Entity

```python
from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass
class OrderItem:
    """Order line item"""
    product_name: str
    unit_price: Money
    quantity: int

    @property
    def subtotal(self) -> Money:
        return self.unit_price * self.quantity


@dataclass
class Order:
    """Order entity - identity is determined by ID"""
    id: UUID = field(default_factory=uuid4)
    customer_name: str = ""
    items: list[OrderItem] = field(default_factory=list)
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.now)

    def add_item(self, product_name: str, unit_price: int, quantity: int) -> None:
        if self.status != "draft":
            raise RuntimeError("Cannot add items to a confirmed order")
        item = OrderItem(product_name, Money(unit_price), quantity)
        self.items.append(item)

    def remove_item(self, index: int) -> None:
        if self.status != "draft":
            raise RuntimeError("Cannot remove items from a confirmed order")
        self.items.pop(index)

    @property
    def total(self) -> Money:
        return Money(sum(item.subtotal.amount for item in self.items))

    def confirm(self) -> None:
        if not self.items:
            raise ValueError("Cannot confirm an order with no items")
        if not self.customer_name:
            raise ValueError("Customer name is not set")
        self.status = "confirmed"

    def cancel(self) -> None:
        if self.status == "shipped":
            raise RuntimeError("Cannot cancel a shipped order")
        self.status = "cancelled"

    def ship(self) -> None:
        if self.status != "confirmed":
            raise RuntimeError("Only confirmed orders can be shipped")
        self.status = "shipped"

    # Entity identity is determined by ID
    def __eq__(self, other) -> bool:
        if not isinstance(other, Order):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


# Usage example
order = Order(customer_name="Taro Tanaka")
order.add_item("Laptop", 150000, 1)
order.add_item("Mouse", 3000, 2)
print(f"Total: {order.total.format()}")  # \u00a5156,000
order.confirm()
order.ship()
```

### 5.3 Service Class

```python
class OrderService:
    """Business logic related to orders"""

    def __init__(
        self,
        order_repo: 'OrderRepository',
        product_repo: 'ProductRepository',
        payment_service: 'PaymentService',
        notification_service: 'NotificationService'
    ):
        self._order_repo = order_repo
        self._product_repo = product_repo
        self._payment = payment_service
        self._notification = notification_service

    def place_order(self, customer_id: str, cart_items: list[dict]) -> Order:
        """Confirm an order"""
        # 1. Stock check
        for item in cart_items:
            product = self._product_repo.find_by_id(item['product_id'])
            if not product or product.stock < item['quantity']:
                raise ValueError(f"Insufficient stock: {item['product_id']}")

        # 2. Create order
        order = Order(customer_name=customer_id)
        for item in cart_items:
            product = self._product_repo.find_by_id(item['product_id'])
            order.add_item(product.name, product.price, item['quantity'])

        order.confirm()

        # 3. Payment processing
        payment_result = self._payment.charge(
            customer_id=customer_id,
            amount=order.total.amount
        )
        if not payment_result.success:
            raise RuntimeError(f"Payment failed: {payment_result.error}")

        # 4. Reduce stock
        for item in cart_items:
            product = self._product_repo.find_by_id(item['product_id'])
            product.reduce_stock(item['quantity'])
            self._product_repo.save(product)

        # 5. Save order
        self._order_repo.save(order)

        # 6. Notification
        self._notification.send(
            customer_id,
            f"Thank you for your order. Order ID: {order.id}"
        )

        return order
```

---

## 6. Modern OOP

### 6.1 Data Classes and Immutability

```python
# Python dataclass
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)  # frozen=True makes it immutable
class Point:
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def translate(self, dx: float, dy: float) -> 'Point':
        """Return a new Point with the translation applied (original is unchanged)"""
        return replace(self, x=self.x + dx, y=self.y + dy)


@dataclass(frozen=True)
class Config:
    """Application configuration (immutable)"""
    database_url: str
    debug: bool = False
    max_connections: int = 10
    allowed_origins: tuple[str, ...] = ("http://localhost:3000",)

    def with_debug(self, debug: bool) -> 'Config':
        """Return a new Config with the debug setting changed"""
        return replace(self, debug=debug)

    def with_max_connections(self, n: int) -> 'Config':
        return replace(self, max_connections=n)
```

```kotlin
// Kotlin data class
data class User(
    val id: Long,
    val name: String,
    val email: String,
    val role: Role = Role.USER
) {
    // data class auto-generates equals, hashCode, toString, copy

    fun withRole(newRole: Role): User = copy(role = newRole)
}

enum class Role { USER, ADMIN, MODERATOR }

// Usage example
val user = User(1, "Taro Tanaka", "tanaka@example.com")
val admin = user.copy(role = Role.ADMIN)  // Immutable update
println(admin)  // User(id=1, name=Taro Tanaka, email=tanaka@example.com, role=ADMIN)
```

```typescript
// Immutable objects in TypeScript
interface UserData {
    readonly id: string;
    readonly name: string;
    readonly email: string;
    readonly createdAt: Date;
}

class ImmutableUser implements UserData {
    readonly id: string;
    readonly name: string;
    readonly email: string;
    readonly createdAt: Date;

    constructor(data: UserData) {
        this.id = data.id;
        this.name = data.name;
        this.email = data.email;
        this.createdAt = data.createdAt;
    }

    // Returns a new instance on modification
    withName(name: string): ImmutableUser {
        return new ImmutableUser({ ...this, name });
    }

    withEmail(email: string): ImmutableUser {
        return new ImmutableUser({ ...this, email });
    }
}
```

### 6.2 Blending OOP and Functional Programming

```python
# Modern OOP: blending with functional programming

from dataclasses import dataclass
from typing import Callable
from functools import reduce


# 1. Data classes + pure functions
@dataclass(frozen=True)
class Transaction:
    amount: int
    category: str
    description: str
    is_income: bool


# Define business logic as pure functions
def total_by_category(
    transactions: list[Transaction],
    category: str
) -> int:
    """Pure function that calculates the total for a specified category"""
    return sum(
        t.amount if t.is_income else -t.amount
        for t in transactions
        if t.category == category
    )


def filter_transactions(
    transactions: list[Transaction],
    predicate: Callable[[Transaction], bool]
) -> list[Transaction]:
    """Higher-order function that extracts transactions matching a condition"""
    return [t for t in transactions if predicate(t)]


def summarize(transactions: list[Transaction]) -> dict[str, int]:
    """Summary by category"""
    categories = set(t.category for t in transactions)
    return {
        cat: total_by_category(transactions, cat)
        for cat in categories
    }


# 2. Method chaining (Fluent Interface)
class TransactionQuery:
    """Query builder for transaction data"""

    def __init__(self, transactions: list[Transaction]):
        self._transactions = transactions

    def income_only(self) -> 'TransactionQuery':
        return TransactionQuery([t for t in self._transactions if t.is_income])

    def expense_only(self) -> 'TransactionQuery':
        return TransactionQuery([t for t in self._transactions if not t.is_income])

    def by_category(self, category: str) -> 'TransactionQuery':
        return TransactionQuery([t for t in self._transactions if t.category == category])

    def above(self, amount: int) -> 'TransactionQuery':
        return TransactionQuery([t for t in self._transactions if t.amount > amount])

    def total(self) -> int:
        return sum(t.amount for t in self._transactions)

    def count(self) -> int:
        return len(self._transactions)

    def to_list(self) -> list[Transaction]:
        return self._transactions.copy()


# Usage example: intuitive queries with method chaining
transactions = [
    Transaction(300000, "Salary", "Monthly salary", True),
    Transaction(80000, "Rent", "Apartment", False),
    Transaction(5000, "Food", "Supermarket", False),
    Transaction(3000, "Food", "Convenience store", False),
    Transaction(50000, "Side job", "Freelance", True),
]

query = TransactionQuery(transactions)
food_total = query.expense_only().by_category("Food").total()
print(f"Food total: {food_total} yen")  # 8000 yen

large_expenses = query.expense_only().above(10000).count()
print(f"Expenses over 10,000 yen: {large_expenses}")  # 1
```

### 6.3 Protocol-Oriented Programming (Swift-style)

```python
# Protocol Oriented Programming (POP) implementation in Python

from typing import Protocol, runtime_checkable


@runtime_checkable
class Equatable(Protocol):
    def __eq__(self, other: object) -> bool: ...

@runtime_checkable
class Hashable(Equatable, Protocol):
    def __hash__(self) -> int: ...

@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...

@runtime_checkable
class Displayable(Protocol):
    def display(self) -> str: ...

@runtime_checkable
class Persistable(Protocol):
    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> 'Persistable': ...


# Class conforming to protocols (no explicit inheritance)
@dataclass(frozen=True)
class Temperature:
    celsius: float

    @property
    def fahrenheit(self) -> float:
        return self.celsius * 9 / 5 + 32

    def display(self) -> str:
        return f"{self.celsius}\u00b0C ({self.fahrenheit:.1f}\u00b0F)"

    def __lt__(self, other: 'Temperature') -> bool:
        return self.celsius < other.celsius

    def __le__(self, other: 'Temperature') -> bool:
        return self.celsius <= other.celsius

    def to_dict(self) -> dict:
        return {"celsius": self.celsius}

    @classmethod
    def from_dict(cls, data: dict) -> 'Temperature':
        return cls(celsius=data["celsius"])


# Protocol-based generic functions
def find_max(items: list[Comparable]) -> Comparable:
    return max(items)

def display_all(items: list[Displayable]) -> None:
    for item in items:
        print(item.display())

temps = [Temperature(20), Temperature(35), Temperature(-5)]
display_all(temps)
hottest = find_max(temps)
print(f"Highest temperature: {hottest.display()}")
```

---

## 7. OOP Anti-patterns and How to Avoid Them

### 7.1 Patterns to Avoid

```
OOP Anti-pattern List:

1. God Object
   Bad: One class manages everything
   Good: Split classes by responsibility

2. Anemic Domain Model
   Bad: Data containers with only getters/setters
   Good: Include business logic within the entity

3. Feature Envy
   Bad: A class excessively accesses another class's data
   Good: Move methods to the class that owns the data

4. Shotgun Surgery
   Bad: One change propagates to many classes
   Good: Consolidate related responsibilities into one class

5. Inappropriate Intimacy
   Bad: Tight coupling between classes that depend on internal details
   Good: Loose coupling through interfaces

6. Dead Code
   Bad: Leaving unused classes or methods in the codebase
   Good: Regularly remove unused code

7. Premature Abstraction
   Bad: Excessive abstraction anticipating future extensions
   Good: YAGNI principle: implement only what is needed now

8. Parallel Inheritance Hierarchies
   Bad: Adding a class to one hierarchy requires adding one to another
   Good: Consolidate with composition
```

```python
# Bad: Anemic Domain Model
class UserAnemic:
    """Just a data container - no business logic"""
    def __init__(self):
        self.name = ""
        self.email = ""
        self.status = ""
        self.login_count = 0
        self.last_login = None

# Logic is scattered across the service layer
class UserServiceAnemic:
    def activate_user(self, user: UserAnemic):
        if user.status == "pending":
            user.status = "active"

    def deactivate_user(self, user: UserAnemic):
        if user.status == "active":
            user.status = "inactive"

    def can_login(self, user: UserAnemic) -> bool:
        return user.status == "active"


# Good: Rich Domain Model
class UserRich:
    """Rich domain model that encapsulates business logic"""

    def __init__(self, name: str, email: str):
        self._name = name
        self._email = email
        self._status = "pending"
        self._login_count = 0
        self._last_login: datetime | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_active(self) -> bool:
        return self._status == "active"

    def activate(self) -> None:
        """Activate the account"""
        if self._status != "pending":
            raise RuntimeError(f"Cannot activate from status '{self._status}'")
        self._status = "active"

    def deactivate(self, reason: str) -> None:
        """Deactivate the account"""
        if self._status != "active":
            raise RuntimeError("Only active accounts can be deactivated")
        self._status = "inactive"

    def login(self) -> None:
        """Login processing"""
        if not self.is_active:
            raise PermissionError("Cannot login with an inactive account")
        self._login_count += 1
        self._last_login = datetime.now()

    def can_login(self) -> bool:
        return self.is_active
```

---

## 8. Language-specific OOP Feature Comparison

```
Language-specific OOP Features:

+----------------+-----------+----------+--------------+------------+
|                | Python    | Java     | TypeScript   | Go         |
+----------------+-----------+----------+--------------+------------+
| Classes        | Yes       | Yes      | Yes          | struct     |
| Inheritance    | Multiple  | Single   | Single       | None       |
| Interfaces     | ABC/      | interface| interface    | interface  |
|                | Protocol  |          |              | (implicit) |
| Access control | Convention| 4 levels | 3 levels     | Upper/     |
|                | (_)       |          | (public etc) | lowercase  |
| Generics       | typing    | Yes      | Yes          | Yes(1.18)  |
| Data classes   | dataclass | record   | N/A          | struct     |
| Null safety    | Optional  | Optional | strictNull   | nil + err  |
| Immutability   | frozen    | final    | readonly     | N/A        |
| Pattern        | match     | sealed   | discriminated| switch     |
| matching       | (3.10+)   | (21+)    | union        | type       |
+----------------+-----------+----------+--------------+------------+

Notable points:
- Go has no classes; it achieves OOP through struct + interface + composition
- Rust uses traits; it has no class inheritance
- Swift recommends Protocol-Oriented Programming
- Kotlin provides concise OOP with data class + sealed class + delegation
```

---

## 9. OOP Design for Testability

```python
# Testable class design

from abc import ABC, abstractmethod
from typing import Protocol


# === Dependency injection for testability ===

class Clock(Protocol):
    """Clock protocol"""
    def now(self) -> datetime: ...


class RealClock:
    """Production clock"""
    def now(self) -> datetime:
        return datetime.now()


class FakeClock:
    """Fake clock for testing"""
    def __init__(self, fixed_time: datetime):
        self._time = fixed_time

    def now(self) -> datetime:
        return self._time

    def advance(self, seconds: int) -> None:
        from datetime import timedelta
        self._time += timedelta(seconds=seconds)


class SessionManager:
    """Session management - designed for testability"""

    def __init__(self, clock: Clock, timeout_minutes: int = 30):
        self._clock = clock  # Inject the clock
        self._timeout_minutes = timeout_minutes
        self._sessions: dict[str, datetime] = {}

    def create_session(self, user_id: str) -> str:
        import uuid
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = self._clock.now()
        return session_id

    def is_valid(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False

        from datetime import timedelta
        created = self._sessions[session_id]
        elapsed = self._clock.now() - created
        return elapsed < timedelta(minutes=self._timeout_minutes)

    def refresh(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id] = self._clock.now()


# Test example
def test_session_expiry():
    """Test session expiration"""
    fake_clock = FakeClock(datetime(2025, 1, 1, 12, 0, 0))
    manager = SessionManager(clock=fake_clock, timeout_minutes=30)

    session_id = manager.create_session("user-1")

    # Session is valid immediately after creation
    assert manager.is_valid(session_id) == True

    # Still valid after 29 minutes
    fake_clock.advance(29 * 60)
    assert manager.is_valid(session_id) == True

    # Expired after 31 minutes
    fake_clock.advance(2 * 60)
    assert manager.is_valid(session_id) == False
    print("Test passed: Session expiration")


def test_session_refresh():
    """Test session refresh"""
    fake_clock = FakeClock(datetime(2025, 1, 1, 12, 0, 0))
    manager = SessionManager(clock=fake_clock, timeout_minutes=30)

    session_id = manager.create_session("user-1")

    # Refresh after 25 minutes
    fake_clock.advance(25 * 60)
    manager.refresh(session_id)

    # Still valid 10 minutes after refresh
    fake_clock.advance(10 * 60)
    assert manager.is_valid(session_id) == True
    print("Test passed: Session refresh")


test_session_expiry()
test_session_refresh()
```

---

## 10. When Not to Use OOP

```
Cases where OOP is not suitable:

1. Small scripts:
   -> Defining a few functions procedurally is sufficient
   -> Class design is overhead

2. Data transformation pipelines:
   -> Functional style (map/filter/reduce) is appropriate
   -> A chain of stateless transformations

3. Numerical/scientific computing:
   -> Primarily array operations with NumPy, etc.
   -> OOP abstraction adds performance overhead

4. Simple CLI tools:
   -> Just parse arguments and execute processing
   -> Classes add unnecessary complexity

5. Configuration file processing:
   -> Dictionaries (dict) or dataclasses are sufficient
   -> Don't make a class if methods aren't needed

6. One-shot data processing:
   -> Writing it as a script is clearer
   -> Don't abstract if reusability isn't needed

Decision criteria:
  - State management needed -> Consider OOP
  - Shared behavior across multiple types -> OOP (polymorphism)
  - Primarily data transformation -> Functional style
  - Small-scale / disposable -> Procedural style
  - Primarily concurrent processing -> Actor model or CSP
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|-----------|
| Four Pillars | Encapsulation, Inheritance, Polymorphism, Abstraction |
| SOLID | Five design principles. Improve maintainability and extensibility |
| SRP | One class, one responsibility. Only one reason to change |
| OCP | Open for extension, closed for modification. Achieved via Strategy pattern, etc. |
| LSP | Subtypes must be substitutable for their base types |
| ISP | Don't force clients to depend on methods they don't use |
| DIP | Depend on abstractions, not on concrete implementations. Achieved via DI |
| Inheritance vs Composition | "Favor composition over inheritance." Prioritize flexibility and loose coupling |
| Value Object | Immutable, compared by equality, represents domain concepts |
| Entity | Identified by ID, has state, has a lifecycle |
| Modern OOP | Blending with functional, emphasis on immutability, lightweight data classes |
| Testability | DI, Protocol, test doubles for easily verifiable designs |
| Anti-patterns | Avoid God Object, Anemic Model, Feature Envy |

---

## Recommended Next Guides

---

## References
1. Martin, R. C. "Clean Architecture." Prentice Hall, 2017.
2. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software (GoF)." Addison-Wesley, 1994.
3. Bloch, J. "Effective Java." 3rd Edition, Addison-Wesley, 2018.
4. Martin, R. C. "Agile Software Development, Principles, Patterns, and Practices." Prentice Hall, 2002.
5. Evans, E. "Domain-Driven Design: Tackling Complexity in the Heart of Software." Addison-Wesley, 2003.
6. Freeman, S. and Pryce, N. "Growing Object-Oriented Software, Guided by Tests." Addison-Wesley, 2009.
7. Kay, A. "The Early History of Smalltalk." ACM SIGPLAN Notices, 1993.
8. Liskov, B. "Data Abstraction and Hierarchy." ACM SIGPLAN Notices, 1988.
9. Meyer, B. "Object-Oriented Software Construction." 2nd Edition, Prentice Hall, 1997.
10. Fowler, M. "Refactoring: Improving the Design of Existing Code." 2nd Edition, Addison-Wesley, 2018.
