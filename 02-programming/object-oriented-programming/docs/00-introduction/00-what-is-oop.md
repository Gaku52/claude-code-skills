# What is OOP?

> Object-Oriented Programming (OOP) is a programming paradigm that "bundles data and the procedures that operate on it into a single unit (an object)." It is the most widely used design approach, ranging from modeling the real world to structuring large-scale software.

## What You Will Learn in This Chapter

- [ ] Understand the essential ideas behind OOP
- [ ] Grasp the relationship between objects and message passing
- [ ] Understand the problems OOP solves and where it applies
- [ ] Experience how OOP's core principles are reflected in real code
- [ ] Compare how OOP is implemented across multiple languages


## Prerequisites

Reading this guide will be easier if you have the following background:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. The Essence of OOP

```
Comparison of programming paradigms:

  Procedural:  data + functions (managed separately)
  OOP:         data + functions = object (integrated)
  Functional:  functions (pipelines that transform data)

The core of OOP:
  "View the world as a collection of objects
   and advance processing through messages exchanged between them."

Alan Kay's definition (designer of Smalltalk):
  1. Everything is an object
  2. Objects communicate by sending messages
  3. Objects have their own memory
  4. Every object is an instance of a class
  5. The class holds shared behavior
```

### 1.1 A Deeper Look at the Definition of OOP

The definition of OOP varies across eras and authors. There are two major schools of thought.

```
Scandinavian School:
  -> Derived from Simula
  -> Emphasizes classes, inheritance, and static typing
  -> Inherited by C++, Java, and C#
  -> "OOP = class-based programming"

American School:
  -> Derived from Smalltalk
  -> Emphasizes message passing and dynamic typing
  -> Inherited by Ruby, Python, and Objective-C
  -> "OOP = messaging between objects"

Modern integrated understanding:
  -> Rather than choosing one, combine elements of both
  -> TypeScript, Kotlin, and Swift incorporate the strengths of both
```

### 1.2 The Four Pillars That Support OOP

OOP has four fundamental principles. They will be covered in detail in later chapters, but let's get an overview here.

```
Four Pillars of OOP:

  1. Encapsulation
     -> Bundle data and behavior into a single unit
     -> Hide internal implementation and expose only a public API
     -> Limit the scope of change impact

  2. Inheritance
     -> Build new classes by inheriting functionality from existing ones
     -> Promote code reuse
     -> Express is-a relationships

  3. Polymorphism
     -> Invoke different implementations through the same interface
     -> The appropriate method is selected at runtime
     -> Enables flexible and extensible code

  4. Abstraction
     -> Hide complex details and expose only essential characteristics
     -> Define contracts with abstract classes or interfaces
     -> Conceal complexity that users don't need to know about
```

```typescript
// TypeScript: demonstrating the four pillars in a single example

// Abstraction: define a shared interface
interface Shape {
  area(): number;
  perimeter(): number;
  describe(): string;
}

// Encapsulation: hide internal data
class Circle implements Shape {
  private readonly _radius: number;

  constructor(radius: number) {
    if (radius <= 0) throw new Error("Radius must be a positive number");
    this._radius = radius;
  }

  // Polymorphism: implementation of the Shape interface
  area(): number {
    return Math.PI * this._radius ** 2;
  }

  perimeter(): number {
    return 2 * Math.PI * this._radius;
  }

  describe(): string {
    return `Circle (radius: ${this._radius})`;
  }

  get radius(): number {
    return this._radius;
  }
}

// Inheritance + polymorphism
class Rectangle implements Shape {
  constructor(
    private readonly width: number,
    private readonly height: number,
  ) {
    if (width <= 0 || height <= 0) {
      throw new Error("Width and height must be positive numbers");
    }
  }

  area(): number {
    return this.width * this.height;
  }

  perimeter(): number {
    return 2 * (this.width + this.height);
  }

  describe(): string {
    return `Rectangle (width: ${this.width}, height: ${this.height})`;
  }
}

// Inheritance: extend Rectangle
class Square extends Rectangle {
  constructor(side: number) {
    super(side, side);
  }

  describe(): string {
    return `Square (side: ${this.perimeter() / 4})`;
  }
}

// Polymorphism: handle different types through the same interface
function printShapeInfo(shapes: Shape[]): void {
  for (const shape of shapes) {
    console.log(`${shape.describe()} - area: ${shape.area().toFixed(2)}`);
  }
}

const shapes: Shape[] = [
  new Circle(5),
  new Rectangle(4, 6),
  new Square(3),
];
printShapeInfo(shapes);
```

---

## 2. Mental Model

```
Procedural mental model:
  "A procedure manual" — a sequence of instructions executed top-down

  1. Retrieve user information
  2. Validate it
  3. Save it to the database
  4. Send an email

OOP mental model:
  "An organization of people with roles" — each person executes their responsibility

  ┌─────────┐    ┌──────────┐    ┌──────────┐
  │  User    │───>│ Validator│───>│ Database │
  │ (data)   │    │ (verify) │    │ (save)   │
  └─────────┘    └──────────┘    └──────────┘
       │                              │
       │         ┌──────────┐         │
       └────────>│ Mailer   │<────────┘
                 │ (notify) │
                 └──────────┘

  Each object:
    - Manages its own data (state)
    - Executes operations within its scope of responsibility
    - Sends messages (method calls) to other objects
```

### 2.1 Shifting Thinking from Procedural to OOP

Procedural and OOP approaches differ fundamentally in how they tackle problems. Let's compare with a concrete example.

```python
# === Procedural approach: user registration ===

# Data managed as a dictionary
users_db = []

def validate_email(email: str) -> bool:
    """Check email validity"""
    return "@" in email and "." in email.split("@")[1]

def validate_password(password: str) -> bool:
    """Check password validity"""
    return len(password) >= 8

def hash_password(password: str) -> str:
    """Hash the password"""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name: str, email: str, password: str) -> dict:
    """Execute user registration steps sequentially"""
    # Step 1: validation
    if not validate_email(email):
        raise ValueError("Invalid email address")
    if not validate_password(password):
        raise ValueError("Password must be at least 8 characters")

    # Step 2: duplicate check
    for user in users_db:
        if user["email"] == email:
            raise ValueError("Email is already registered")

    # Step 3: create the user
    user = {
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
    }

    # Step 4: persist
    users_db.append(user)
    return user

# Issues:
# - Data (users_db) and functions are separated
# - Depends on global state
# - Hard to test (users_db has to be reset)
# - The number of functions grows as features are added
```

```python
# === OOP approach: user registration ===

import hashlib
from dataclasses import dataclass, field
from typing import Optional


class EmailAddress:
    """Email address value object: validation is internalized"""
    def __init__(self, value: str):
        if "@" not in value or "." not in value.split("@")[1]:
            raise ValueError(f"Invalid email address: {value}")
        self._value = value

    @property
    def value(self) -> str:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EmailAddress):
            return self._value == other._value
        return False

    def __hash__(self) -> int:
        return hash(self._value)

    def __str__(self) -> str:
        return self._value


class Password:
    """Password value object: hashing is internalized"""
    def __init__(self, plain_text: str):
        if len(plain_text) < 8:
            raise ValueError("Password must be at least 8 characters")
        self._hash = hashlib.sha256(plain_text.encode()).hexdigest()

    @property
    def hashed(self) -> str:
        return self._hash

    def verify(self, plain_text: str) -> bool:
        return hashlib.sha256(plain_text.encode()).hexdigest() == self._hash


class User:
    """User entity: integrates data and behavior"""
    def __init__(self, name: str, email: EmailAddress, password: Password):
        self._name = name
        self._email = email
        self._password = password

    @property
    def name(self) -> str:
        return self._name

    @property
    def email(self) -> EmailAddress:
        return self._email

    def authenticate(self, plain_password: str) -> bool:
        """Authentication is the user's own responsibility"""
        return self._password.verify(plain_password)


class UserRepository:
    """User repository: handles persistence"""
    def __init__(self):
        self._users: list[User] = []

    def find_by_email(self, email: EmailAddress) -> Optional[User]:
        for user in self._users:
            if user.email == email:
                return user
        return None

    def save(self, user: User) -> None:
        if self.find_by_email(user.email) is not None:
            raise ValueError("Email is already registered")
        self._users.append(user)

    @property
    def count(self) -> int:
        return len(self._users)


class UserRegistrationService:
    """User registration service: orchestrates the use case"""
    def __init__(self, repository: UserRepository):
        self._repository = repository

    def register(self, name: str, email_str: str, password_str: str) -> User:
        # Each object validates within its own area of responsibility
        email = EmailAddress(email_str)        # check email format
        password = Password(password_str)      # check password strength
        user = User(name, email, password)     # create the user
        self._repository.save(user)            # persist (includes duplicate check)
        return user


# Benefits:
# - Each class has a clear responsibility
# - Validation is integrated with data
# - Easy to test (the repository can be swapped with a mock)
# - Features can be managed on a per-class basis
```

### 2.2 Analogies from the Real World

Analogies to the real world are useful for understanding OOP concepts.

```
Restaurant analogy:

  Procedural thinking:
    1. A customer arrives
    2. Show them the menu
    3. Take the order
    4. Pass the order to the kitchen
    5. Cook the food
    6. Deliver the food
    7. Handle payment
    -> Manage every step in a single script

  OOP thinking:
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Customer │───>│ Waiter   │───>│ Kitchen  │
    │ orders   │    │ relays   │    │ cooks    │
    └──────────┘    └──────────┘    └──────────┘
         │                              │
         │         ┌──────────┐         │
         └────────>│ Cashier  │<────────┘
                   │ charges  │
                   └──────────┘

    Each object (person) carries out its own responsibilities:
    - Customer: chooses from the menu, eats, pays
    - Waiter: takes orders, delivers food
    - Kitchen: cooks dishes based on orders
    - Cashier: calculates totals and handles payment

    -> Adding a new menu item only changes the Kitchen
    -> Changing payment methods only changes the Cashier
    -> The scope of change is limited
```

```typescript
// TypeScript: expressing the restaurant analogy in code

interface MenuItem {
  name: string;
  price: number;
  category: "appetizer" | "main" | "dessert" | "drink";
}

class Order {
  private _items: MenuItem[] = [];
  private _status: "pending" | "preparing" | "ready" | "served" = "pending";

  addItem(item: MenuItem): void {
    if (this._status !== "pending") {
      throw new Error("Cannot add items after the order is confirmed");
    }
    this._items.push(item);
  }

  get total(): number {
    return this._items.reduce((sum, item) => sum + item.price, 0);
  }

  get items(): ReadonlyArray<MenuItem> {
    return [...this._items]; // defensive copy
  }

  get status(): string {
    return this._status;
  }

  confirm(): void {
    if (this._items.length === 0) {
      throw new Error("Cannot confirm an empty order");
    }
    this._status = "preparing";
  }

  markReady(): void {
    this._status = "ready";
  }

  markServed(): void {
    this._status = "served";
  }
}

class Customer {
  private _currentOrder: Order | null = null;

  constructor(public readonly name: string) {}

  createOrder(): Order {
    this._currentOrder = new Order();
    return this._currentOrder;
  }

  get currentOrder(): Order | null {
    return this._currentOrder;
  }
}

class Kitchen {
  private _queue: Order[] = [];

  receiveOrder(order: Order): void {
    order.confirm();
    this._queue.push(order);
    console.log(`Kitchen: order accepted (${order.items.length} items)`);
  }

  prepareNext(): Order | null {
    const order = this._queue.shift();
    if (order) {
      order.markReady();
      console.log("Kitchen: the dish is ready");
    }
    return order ?? null;
  }

  get pendingOrders(): number {
    return this._queue.length;
  }
}

class Waiter {
  constructor(
    private readonly name: string,
    private readonly kitchen: Kitchen,
  ) {}

  takeOrder(customer: Customer, items: MenuItem[]): void {
    const order = customer.createOrder();
    for (const item of items) {
      order.addItem(item);
    }
    this.kitchen.receiveOrder(order);
    console.log(`${this.name}: received order from ${customer.name}`);
  }

  serveOrder(order: Order): void {
    order.markServed();
    console.log(`${this.name}: here is your food`);
  }
}

class Cashier {
  private _totalRevenue = 0;

  checkout(order: Order): number {
    const total = order.total;
    const tax = Math.floor(total * 0.1);
    const grandTotal = total + tax;
    this._totalRevenue += grandTotal;
    console.log(`Checkout: subtotal ${total} yen + tax ${tax} yen = ${grandTotal} yen`);
    return grandTotal;
  }

  get totalRevenue(): number {
    return this._totalRevenue;
  }
}
```

---

## 3. The Three Elements of an Object

```
Object = State + Behavior + Identity

  ┌─────────────────────────────────┐
  │        BankAccount              │
  ├─────────────────────────────────┤
  │ State:                          │
  │   - owner: "Taro Tanaka"        │
  │   - balance: 100000             │
  │   - accountNumber: "1234567"    │
  ├─────────────────────────────────┤
  │ Behavior:                       │
  │   - deposit(amount)             │
  │   - withdraw(amount)            │
  │   - getBalance()                │
  ├─────────────────────────────────┤
  │ Identity:                       │
  │   - Memory address: 0x7ff...    │
  │   - Same state != same object   │
  └─────────────────────────────────┘
```

### 3.1 Managing State

State is the data held by an object and can change over time. State management is one of OOP's most important concerns.

```typescript
// TypeScript: practical example of state management - e-commerce shopping cart

interface Product {
  readonly id: string;
  readonly name: string;
  readonly price: number;
  readonly stock: number;
}

interface CartItem {
  readonly product: Product;
  quantity: number;
}

class ShoppingCart {
  // State: list of products in the cart
  private _items: Map<string, CartItem> = new Map();
  // State: cart creation timestamp
  private readonly _createdAt: Date = new Date();
  // State: last-updated timestamp
  private _updatedAt: Date = new Date();

  /**
   * Add a product to the cart
   * Business rule: cannot add more than the available stock
   */
  addItem(product: Product, quantity: number = 1): void {
    if (quantity <= 0) {
      throw new Error("Quantity must be 1 or greater");
    }

    const existing = this._items.get(product.id);
    const currentQty = existing?.quantity ?? 0;
    const newQty = currentQty + quantity;

    if (newQty > product.stock) {
      throw new Error(
        `Insufficient stock: only ${product.stock} of ${product.name} are available`
      );
    }

    this._items.set(product.id, { product, quantity: newQty });
    this._updatedAt = new Date();
  }

  /**
   * Change the quantity of a product
   */
  updateQuantity(productId: string, quantity: number): void {
    if (quantity < 0) {
      throw new Error("Quantity must be 0 or greater");
    }

    if (quantity === 0) {
      this._items.delete(productId);
    } else {
      const item = this._items.get(productId);
      if (!item) {
        throw new Error("Product is not in the cart");
      }
      if (quantity > item.product.stock) {
        throw new Error("Quantity cannot exceed available stock");
      }
      item.quantity = quantity;
    }
    this._updatedAt = new Date();
  }

  /**
   * Remove a product from the cart
   */
  removeItem(productId: string): void {
    if (!this._items.has(productId)) {
      throw new Error("Product is not in the cart");
    }
    this._items.delete(productId);
    this._updatedAt = new Date();
  }

  /**
   * Calculate the subtotal
   */
  get subtotal(): number {
    let total = 0;
    for (const item of this._items.values()) {
      total += item.product.price * item.quantity;
    }
    return total;
  }

  /**
   * Calculate the total including tax
   */
  get totalWithTax(): number {
    return Math.floor(this.subtotal * 1.1);
  }

  /**
   * Return the number of items in the cart
   */
  get itemCount(): number {
    let count = 0;
    for (const item of this._items.values()) {
      count += item.quantity;
    }
    return count;
  }

  /**
   * Return whether the cart is empty
   */
  get isEmpty(): boolean {
    return this._items.size === 0;
  }

  /**
   * Display the cart contents
   */
  display(): string {
    if (this.isEmpty) return "The cart is empty";

    const lines: string[] = ["=== Shopping Cart ==="];
    for (const item of this._items.values()) {
      const lineTotal = item.product.price * item.quantity;
      lines.push(
        `${item.product.name} x${item.quantity} = ¥${lineTotal.toLocaleString()}`
      );
    }
    lines.push("─".repeat(30));
    lines.push(`Subtotal: ¥${this.subtotal.toLocaleString()}`);
    lines.push(`Total (incl. tax): ¥${this.totalWithTax.toLocaleString()}`);
    return lines.join("\n");
  }
}
```

### 3.2 Designing Behavior

Behavior is the set of operations an object exposes to the outside world, serving as a means to change the object's state safely.

```python
# Python: designing behavior - a task management system

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    CANCELLED = "cancelled"


class Task:
    """Task: state-transition rules are embedded as behavior"""

    # Allowed state transitions
    _VALID_TRANSITIONS = {
        TaskStatus.TODO: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
        TaskStatus.IN_PROGRESS: {TaskStatus.IN_REVIEW, TaskStatus.TODO, TaskStatus.CANCELLED},
        TaskStatus.IN_REVIEW: {TaskStatus.DONE, TaskStatus.IN_PROGRESS},
        TaskStatus.DONE: set(),       # cannot change after completion
        TaskStatus.CANCELLED: set(),  # cannot change after cancellation
    }

    def __init__(
        self,
        title: str,
        description: str = "",
        priority: Priority = Priority.MEDIUM,
        due_date: Optional[datetime] = None,
    ):
        if not title.strip():
            raise ValueError("Title is required")

        self._title = title
        self._description = description
        self._priority = priority
        self._status = TaskStatus.TODO
        self._due_date = due_date
        self._created_at = datetime.now()
        self._updated_at = datetime.now()
        self._history: list[tuple[datetime, str]] = []
        self._history.append((self._created_at, f"Task created: {title}"))

    @property
    def title(self) -> str:
        return self._title

    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def priority(self) -> Priority:
        return self._priority

    @property
    def is_overdue(self) -> bool:
        """Determine whether the task is overdue"""
        if self._due_date is None:
            return False
        if self._status in (TaskStatus.DONE, TaskStatus.CANCELLED):
            return False
        return datetime.now() > self._due_date

    def start(self) -> None:
        """Start the task"""
        self._transition_to(TaskStatus.IN_PROGRESS, "Task started")

    def submit_for_review(self) -> None:
        """Submit for review"""
        self._transition_to(TaskStatus.IN_REVIEW, "Submitted for review")

    def complete(self) -> None:
        """Complete the task"""
        self._transition_to(TaskStatus.DONE, "Task completed")

    def cancel(self, reason: str = "") -> None:
        """Cancel the task"""
        self._transition_to(TaskStatus.CANCELLED, f"Cancelled: {reason}")

    def send_back(self, reason: str = "") -> None:
        """Send back (review -> in progress)"""
        self._transition_to(TaskStatus.IN_PROGRESS, f"Sent back: {reason}")

    def update_priority(self, new_priority: Priority) -> None:
        """Change the priority"""
        old = self._priority
        self._priority = new_priority
        self._record_change(f"Priority changed: {old.name} -> {new_priority.name}")

    def _transition_to(self, new_status: TaskStatus, message: str) -> None:
        """Enforce state-transition rules"""
        valid = self._VALID_TRANSITIONS.get(self._status, set())
        if new_status not in valid:
            raise ValueError(
                f"Invalid state transition: {self._status.value} -> {new_status.value}"
            )
        old_status = self._status
        self._status = new_status
        self._record_change(f"{message} ({old_status.value} -> {new_status.value})")

    def _record_change(self, message: str) -> None:
        """Record the change history"""
        now = datetime.now()
        self._updated_at = now
        self._history.append((now, message))

    def get_history(self) -> list[tuple[datetime, str]]:
        """Return the change history (defensive copy)"""
        return list(self._history)

    def __str__(self) -> str:
        overdue = " [overdue]" if self.is_overdue else ""
        return f"[{self._priority.name}] {self._title} ({self._status.value}){overdue}"


# Usage example
task = Task("Implement login feature", priority=Priority.HIGH,
            due_date=datetime.now() + timedelta(days=7))
print(task)                    # [HIGH] Implement login feature (todo)

task.start()
print(task)                    # [HIGH] Implement login feature (in_progress)

task.submit_for_review()
print(task)                    # [HIGH] Implement login feature (in_review)

task.complete()
print(task)                    # [HIGH] Implement login feature (done)

# task.start()  # ValueError: Invalid state transition: done -> in_progress
```

### 3.3 The Importance of Identity

Identity is the concept used to distinguish two objects that share the same state.

```java
// Java: the difference between identity and equality

public class Money {
    private final int amount;
    private final String currency;

    public Money(int amount, String currency) {
        this.amount = amount;
        this.currency = currency;
    }

    // Value object: equal when state matches
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;                    // identity match
        if (obj == null || getClass() != obj.getClass()) return false;
        Money money = (Money) obj;
        return amount == money.amount
            && Objects.equals(currency, money.currency); // compare by state
    }

    @Override
    public int hashCode() {
        return Objects.hash(amount, currency);
    }

    @Override
    public String toString() {
        return amount + " " + currency;
    }
}

public class Customer {
    private final String id;   // identity: distinguished by ID
    private String name;
    private String email;

    public Customer(String id, String name, String email) {
        this.id = id;
        this.name = name;
        this.email = email;
    }

    // Entity: identical when IDs match
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Customer customer = (Customer) obj;
        return Objects.equals(id, customer.id);  // compare by ID only
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}

// Usage examples:
// Money: "equal" when amount and currency match
Money m1 = new Money(1000, "JPY");
Money m2 = new Money(1000, "JPY");
System.out.println(m1.equals(m2));   // true (equal values)
System.out.println(m1 == m2);        // false (different objects)

// Customer: "same person" when IDs match
Customer c1 = new Customer("C001", "Tanaka", "tanaka@example.com");
Customer c2 = new Customer("C001", "Taro Tanaka", "t.tanaka@example.com");
System.out.println(c1.equals(c2));   // true (same ID -> same person)
// -> Still the same person even if name or email changes
```

```typescript
// TypeScript: distinguishing entities from value objects

// Value object: equality determined by state
class Address {
  constructor(
    public readonly prefecture: string,
    public readonly city: string,
    public readonly street: string,
    public readonly zipCode: string,
  ) {}

  equals(other: Address): boolean {
    return (
      this.prefecture === other.prefecture &&
      this.city === other.city &&
      this.street === other.street &&
      this.zipCode === other.zipCode
    );
  }

  toString(): string {
    return `〒${this.zipCode} ${this.prefecture}${this.city}${this.street}`;
  }
}

// Entity: identity determined by ID
class Employee {
  private _name: string;
  private _address: Address;
  private _department: string;

  constructor(
    public readonly employeeId: string,
    name: string,
    address: Address,
    department: string,
  ) {
    this._name = name;
    this._address = address;
    this._department = department;
  }

  get name(): string { return this._name; }
  get address(): Address { return this._address; }
  get department(): string { return this._department; }

  transfer(newDepartment: string): void {
    this._department = newDepartment;
  }

  relocate(newAddress: Address): void {
    this._address = newAddress;
  }

  equals(other: Employee): boolean {
    // Same employee when employee IDs match
    return this.employeeId === other.employeeId;
  }
}
```

### Code Example

```typescript
// TypeScript: a bank account object
class BankAccount {
  // State
  private owner: string;
  private balance: number;
  private readonly accountNumber: string;

  constructor(owner: string, accountNumber: string, initialBalance: number = 0) {
    this.owner = owner;
    this.accountNumber = accountNumber;
    this.balance = initialBalance;
  }

  // Behavior
  deposit(amount: number): void {
    if (amount <= 0) throw new Error("Deposit amount must be a positive number");
    this.balance += amount;
  }

  withdraw(amount: number): void {
    if (amount > this.balance) throw new Error("Insufficient balance");
    this.balance -= amount;
  }

  getBalance(): number {
    return this.balance;
  }
}

// Identity: same state, still different objects
const account1 = new BankAccount("Tanaka", "001", 10000);
const account2 = new BankAccount("Tanaka", "001", 10000);
console.log(account1 === account2); // false (different objects)
```

```python
# Python: same concept
class BankAccount:
    def __init__(self, owner: str, account_number: str, initial_balance: float = 0):
        self._owner = owner
        self._account_number = account_number
        self._balance = initial_balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be a positive number")
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self._balance:
            raise ValueError("Insufficient balance")
        self._balance -= amount

    @property
    def balance(self) -> float:
        return self._balance
```

```java
// Java: same concept
public class BankAccount {
    private String owner;
    private double balance;
    private final String accountNumber;

    public BankAccount(String owner, String accountNumber, double initialBalance) {
        this.owner = owner;
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
    }

    public void deposit(double amount) {
        if (amount <= 0) throw new IllegalArgumentException("Deposit must be positive");
        this.balance += amount;
    }

    public void withdraw(double amount) {
        if (amount > this.balance) throw new IllegalStateException("Insufficient balance");
        this.balance -= amount;
    }

    public double getBalance() { return balance; }
}
```

---

## 4. Message Passing

```
The essence of OOP is message passing:

  Procedural way of thinking:
    result = validate(user_data)    <- pass data to a function

  OOP way of thinking:
    result = validator.validate(user_data)  <- send a message to an object

  Difference:
    Procedural: it is unclear "who" performs the work
    OOP:        the "validator" is responsible for it

  Benefits of message passing:
    1. Clear ownership of responsibility
    2. Implementations can be swapped (polymorphism)
    3. Can be substituted with mocks during tests
```

### 4.1 Practicing Message Passing

Message passing is more than a plain method call. It expresses the idea of "requesting work from an object."

```typescript
// TypeScript: practical example of message passing - a notification system

// Define the "ability" to send a notification
interface NotificationSender {
  send(recipient: string, message: string): Promise<boolean>;
  readonly channelName: string;
}

// Email notification
class EmailSender implements NotificationSender {
  readonly channelName = "Email";

  async send(recipient: string, message: string): Promise<boolean> {
    console.log(`Send email: ${recipient} -> ${message}`);
    // In reality, this would connect to an SMTP server
    return true;
  }
}

// Slack notification
class SlackSender implements NotificationSender {
  readonly channelName = "Slack";

  constructor(private readonly webhookUrl: string) {}

  async send(recipient: string, message: string): Promise<boolean> {
    console.log(`Send Slack: ${recipient} -> ${message}`);
    // In reality, this would call a webhook API
    return true;
  }
}

// SMS notification
class SmsSender implements NotificationSender {
  readonly channelName = "SMS";

  async send(recipient: string, message: string): Promise<boolean> {
    console.log(`Send SMS: ${recipient} -> ${message}`);
    // In reality, this would call an SMS API
    return true;
  }
}

// Notification service: senders can be swapped
class NotificationService {
  private senders: NotificationSender[] = [];

  addSender(sender: NotificationSender): void {
    this.senders.push(sender);
  }

  async notifyAll(recipient: string, message: string): Promise<void> {
    for (const sender of this.senders) {
      try {
        // Message passing: tell each sender to "send"
        const success = await sender.send(recipient, message);
        if (success) {
          console.log(`${sender.channelName}: send succeeded`);
        }
      } catch (error) {
        console.error(`${sender.channelName}: send failed`);
      }
    }
  }
}

// Usage
const service = new NotificationService();
service.addSender(new EmailSender());
service.addSender(new SlackSender("https://hooks.slack.com/..."));
service.addSender(new SmsSender());

// Send the same message across all channels
// Each sender handles it in its own way (polymorphism)
await service.notifyAll("user@example.com", "Server alert triggered");
```

### 4.2 The Relationship Between Message Passing and Polymorphism

```python
# Python: message passing enables polymorphism

from abc import ABC, abstractmethod
from typing import BinaryIO
import json
import csv
import io


class DataExporter(ABC):
    """Abstract base class for data exporters"""

    @abstractmethod
    def export(self, data: list[dict]) -> str:
        """Export data"""
        pass

    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension"""
        pass


class JsonExporter(DataExporter):
    """Export as JSON"""

    def export(self, data: list[dict]) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2)

    def file_extension(self) -> str:
        return ".json"


class CsvExporter(DataExporter):
    """Export as CSV"""

    def export(self, data: list[dict]) -> str:
        if not data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()

    def file_extension(self) -> str:
        return ".csv"


class MarkdownExporter(DataExporter):
    """Export as a Markdown table"""

    def export(self, data: list[dict]) -> str:
        if not data:
            return ""
        headers = list(data[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in data:
            values = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def file_extension(self) -> str:
        return ".md"


class ReportGenerator:
    """Report generator: exporter can be swapped"""

    def __init__(self, exporter: DataExporter):
        self._exporter = exporter

    def generate(self, data: list[dict]) -> str:
        # Message passing: ask the exporter to "export"
        # The exporter decides what format to produce
        return self._exporter.export(data)

    @property
    def output_extension(self) -> str:
        return self._exporter.file_extension()


# Usage example
data = [
    {"name": "Tanaka", "age": 25, "department": "Development"},
    {"name": "Yamada", "age": 30, "department": "Sales"},
]

# Same data, different formats
for exporter in [JsonExporter(), CsvExporter(), MarkdownExporter()]:
    generator = ReportGenerator(exporter)
    print(f"--- {generator.output_extension} ---")
    print(generator.generate(data))
    print()
```

### 4.3 Smalltalk-Style Message Passing

The original notion of message passing advocated by Alan Kay differs conceptually from the method calls of many modern languages.

```
Smalltalk message passing:
  -> Send a message to an object
  -> The object receives the message and decides how to handle it
  -> If no matching method exists, doesNotUnderstand: is invoked

  3 + 4
  -> Sends the message "+" with argument "4" to 3 (an Integer object)
  -> 3 itself decides how to perform the addition

Modern-language method calls:
  -> The compiler/interpreter resolves which method to invoke
  -> Calling a nonexistent method causes a compile error (in statically typed languages)

Ruby (heavily influenced by Smalltalk):
  -> Handles messages via method_missing
  -> Uses dynamic method resolution
```

```python
# Python: a message-passing-like pattern using __getattr__

class FluentQuery:
    """Query builder with a fluent API"""

    def __init__(self):
        self._conditions: list[str] = []
        self._table: str = ""
        self._limit: int | None = None
        self._order_by: str | None = None

    def from_table(self, table: str) -> "FluentQuery":
        self._table = table
        return self

    def where(self, condition: str) -> "FluentQuery":
        self._conditions.append(condition)
        return self

    def limit(self, n: int) -> "FluentQuery":
        self._limit = n
        return self

    def order_by(self, column: str) -> "FluentQuery":
        self._order_by = column
        return self

    def build(self) -> str:
        query = f"SELECT * FROM {self._table}"
        if self._conditions:
            query += " WHERE " + " AND ".join(self._conditions)
        if self._order_by:
            query += f" ORDER BY {self._order_by}"
        if self._limit:
            query += f" LIMIT {self._limit}"
        return query


# Method chaining = a chain of messages
query = (
    FluentQuery()
    .from_table("users")
    .where("age >= 18")
    .where("status = 'active'")
    .order_by("created_at DESC")
    .limit(10)
    .build()
)
print(query)
# SELECT * FROM users WHERE age >= 18 AND status = 'active' ORDER BY created_at DESC LIMIT 10
```

---

## 5. Problems That OOP Solves

```
Without OOP (procedural):
  Problem 1: hard to manage global state
    -> Impossible to trace who changed which variable
    -> OOP: encapsulation contains state inside the object

  Problem 2: code duplication
    -> The same logic gets written repeatedly
    -> OOP: inheritance and composition enable reuse

  Problem 3: unclear blast radius of changes
    -> A single change ripples throughout the system
    -> OOP: interfaces control dependencies

  Problem 4: hard to structure large codebases
    -> Procedural code collapses beyond ~10k lines
    -> OOP: classes and packages provide structure
```

### 5.1 Concrete Examples of Problems and Solutions

```typescript
// TypeScript: concrete examples of the problems OOP solves

// === Problem 1: hard-to-manage global state ===

// Procedural (problematic)
let globalUserCount = 0;
let globalTotalRevenue = 0;
// ... referenced and modified from 100 places -> chaos

// OOP (solution)
class Analytics {
  private _userCount = 0;
  private _totalRevenue = 0;

  incrementUsers(): void {
    this._userCount++;
  }

  addRevenue(amount: number): void {
    if (amount < 0) throw new Error("Revenue must be positive");
    this._totalRevenue += amount;
  }

  get report(): { users: number; revenue: number } {
    return { users: this._userCount, revenue: this._totalRevenue };
  }
}
// -> State changes can only happen through the Analytics object
// -> Invalid changes can be prevented


// === Problem 2: code duplication ===

// Procedural (problematic)
function validateUserEmail(email: string): boolean {
  return /^[\w.-]+@[\w.-]+\.\w+$/.test(email);
}
function validateAdminEmail(email: string): boolean {
  return /^[\w.-]+@[\w.-]+\.\w+$/.test(email); // same logic!
}

// OOP (solution)
class EmailValidator {
  private readonly pattern = /^[\w.-]+@[\w.-]+\.\w+$/;

  validate(email: string): boolean {
    return this.pattern.test(email);
  }
}
// -> Consolidated in one place and reused


// === Problem 3: unclear blast radius of changes ===

// Define the contract through an interface
interface PaymentProcessor {
  charge(amount: number, currency: string): Promise<PaymentResult>;
  refund(transactionId: string): Promise<RefundResult>;
}

interface PaymentResult {
  transactionId: string;
  success: boolean;
}

interface RefundResult {
  success: boolean;
  refundedAmount: number;
}

// Consumers are unaffected even if implementations are swapped
class StripePaymentProcessor implements PaymentProcessor {
  async charge(amount: number, currency: string): Promise<PaymentResult> {
    // Implementation using the Stripe API
    return { transactionId: "stripe_xxx", success: true };
  }

  async refund(transactionId: string): Promise<RefundResult> {
    return { success: true, refundedAmount: 1000 };
  }
}

class PayPayPaymentProcessor implements PaymentProcessor {
  async charge(amount: number, currency: string): Promise<PaymentResult> {
    // Implementation using the PayPay API
    return { transactionId: "paypay_xxx", success: true };
  }

  async refund(transactionId: string): Promise<RefundResult> {
    return { success: true, refundedAmount: 1000 };
  }
}

// Consumer: depends only on PaymentProcessor
class CheckoutService {
  constructor(private processor: PaymentProcessor) {}

  async checkout(amount: number): Promise<string> {
    const result = await this.processor.charge(amount, "JPY");
    if (!result.success) throw new Error("Payment failed");
    return result.transactionId;
  }
}
// -> Swapping the processor implementation does not require changes to CheckoutService


// === Problem 4: structuring large-scale code ===

// Example module layout
// src/
//   domain/
//     entities/User.ts, Order.ts, Product.ts
//     value-objects/Money.ts, Address.ts
//     repositories/UserRepository.ts
//   application/
//     services/OrderService.ts, UserService.ts
//   infrastructure/
//     database/PostgresUserRepository.ts
//     external/StripePaymentProcessor.ts
//   presentation/
//     controllers/OrderController.ts

// -> Classes become units of file-level structure
// -> Dependencies are made explicit by interfaces
```

### 5.2 The Scalability Problem

OOP delivers its value most clearly as projects grow.

```
Project growth and paradigm fit:

  100 lines: procedural is enough
    -> Just a handful of function definitions
    -> Classes would be overkill

  1,000 lines: OOP starts to pay off
    -> State management becomes complex
    -> Modular separation becomes necessary

  10,000 lines: hard to manage without OOP
    -> Tracking global state becomes impossible
    -> Blast radius of changes becomes unpredictable
    -> Writing tests becomes infeasible

  100,000+ lines: OOP + design patterns are mandatory
    -> Layered architecture
    -> Dependency Injection (DI)
    -> Loose coupling via interfaces
    -> Established testing strategy

  Value of OOP in large projects:
    1. Code structuring -> classes / packages / modules
    2. Team division -> class boundary = team boundary
    3. Testability -> unit tests via mocks and stubs
    4. Changeability -> loose coupling via interfaces
    5. Knowledge organization -> domain model expresses business knowledge
```

---

## 6. Where OOP Applies

```
Areas where OOP shines:
  ✓ GUI applications (widget hierarchies)
  ✓ Game development (entities and components)
  ✓ Enterprise applications (business logic)
  ✓ Framework design (providing extension points)
  ✓ Simulation (modeling the real world)

Areas where OOP is a poor fit:
  ✗ Data transformation pipelines -> functional fits better
  ✗ Numerical / scientific computing -> procedural/array-oriented fits better
  ✗ Scripts / glue code -> simple procedural fits better
  ✗ Concurrency -> actor model / functional fits better

Real-world projects:
  -> Combining multiple paradigms is optimal
  -> The OOP + FP multi-paradigm approach is mainstream
```

### 6.1 OOP in GUI Applications

GUI frameworks are a classic use case for OOP. A hierarchy of widgets and an event-driven model map naturally onto OOP.

```typescript
// TypeScript: a simplified GUI component hierarchy

abstract class Widget {
  protected _x: number;
  protected _y: number;
  protected _width: number;
  protected _height: number;
  protected _visible: boolean = true;
  protected _parent: Widget | null = null;
  protected _children: Widget[] = [];

  constructor(x: number, y: number, width: number, height: number) {
    this._x = x;
    this._y = y;
    this._width = width;
    this._height = height;
  }

  addChild(child: Widget): void {
    child._parent = this;
    this._children.push(child);
  }

  abstract render(ctx: CanvasRenderingContext2D): void;

  renderAll(ctx: CanvasRenderingContext2D): void {
    if (!this._visible) return;
    this.render(ctx);
    for (const child of this._children) {
      child.renderAll(ctx);
    }
  }

  hide(): void { this._visible = false; }
  show(): void { this._visible = true; }
}

class Panel extends Widget {
  private _backgroundColor: string;

  constructor(
    x: number, y: number, w: number, h: number,
    backgroundColor: string = "#ffffff",
  ) {
    super(x, y, w, h);
    this._backgroundColor = backgroundColor;
  }

  render(ctx: CanvasRenderingContext2D): void {
    ctx.fillStyle = this._backgroundColor;
    ctx.fillRect(this._x, this._y, this._width, this._height);
  }
}

class Label extends Widget {
  constructor(
    x: number, y: number,
    private _text: string,
    private _fontSize: number = 14,
    private _color: string = "#000000",
  ) {
    super(x, y, 0, 0);
  }

  set text(value: string) { this._text = value; }

  render(ctx: CanvasRenderingContext2D): void {
    ctx.fillStyle = this._color;
    ctx.font = `${this._fontSize}px sans-serif`;
    ctx.fillText(this._text, this._x, this._y);
  }
}

class Button extends Widget {
  private _label: string;
  private _onClick: (() => void) | null = null;
  private _isHovered = false;

  constructor(
    x: number, y: number, w: number, h: number,
    label: string,
  ) {
    super(x, y, w, h);
    this._label = label;
  }

  set onClick(handler: () => void) {
    this._onClick = handler;
  }

  handleClick(mouseX: number, mouseY: number): void {
    if (this.containsPoint(mouseX, mouseY) && this._onClick) {
      this._onClick();
    }
  }

  private containsPoint(px: number, py: number): boolean {
    return (
      px >= this._x && px <= this._x + this._width &&
      py >= this._y && py <= this._y + this._height
    );
  }

  render(ctx: CanvasRenderingContext2D): void {
    ctx.fillStyle = this._isHovered ? "#4488ff" : "#3366cc";
    ctx.fillRect(this._x, this._y, this._width, this._height);
    ctx.fillStyle = "#ffffff";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(
      this._label,
      this._x + this._width / 2,
      this._y + this._height / 2,
    );
  }
}
```

### 6.2 OOP in Game Development

```python
# Python: a simple game entity system

from abc import ABC, abstractmethod
import math


class Vector2D:
    """2D vector"""
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self) -> "Vector2D":
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def distance_to(self, other: "Vector2D") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class GameObject(ABC):
    """Base class for game objects"""
    def __init__(self, position: Vector2D, name: str = ""):
        self.position = position
        self.name = name
        self._active = True

    @property
    def is_active(self) -> bool:
        return self._active

    def deactivate(self) -> None:
        self._active = False

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """Update logic called every frame"""
        pass

    @abstractmethod
    def render(self) -> str:
        """Return a string for rendering"""
        pass


class Player(GameObject):
    """Player character"""
    def __init__(self, position: Vector2D, name: str = "Player"):
        super().__init__(position, name)
        self._hp = 100
        self._max_hp = 100
        self._speed = 5.0
        self._attack_power = 10
        self._level = 1
        self._experience = 0
        self._velocity = Vector2D(0, 0)

    @property
    def hp(self) -> int:
        return self._hp

    @property
    def is_alive(self) -> bool:
        return self._hp > 0

    def move(self, direction: Vector2D) -> None:
        self._velocity = direction.normalized() * self._speed

    def take_damage(self, amount: int) -> None:
        self._hp = max(0, self._hp - amount)
        if self._hp == 0:
            self.deactivate()

    def heal(self, amount: int) -> None:
        self._hp = min(self._max_hp, self._hp + amount)

    def gain_experience(self, amount: int) -> None:
        self._experience += amount
        while self._experience >= self._level * 100:
            self._experience -= self._level * 100
            self._level_up()

    def _level_up(self) -> None:
        self._level += 1
        self._max_hp += 10
        self._hp = self._max_hp
        self._attack_power += 2

    def update(self, delta_time: float) -> None:
        if not self.is_active:
            return
        self.position = self.position + self._velocity * delta_time
        self._velocity = Vector2D(0, 0)  # stop when there is no input

    def render(self) -> str:
        return f"[{self.name}] HP:{self._hp}/{self._max_hp} Lv:{self._level} Pos:({self.position.x:.1f},{self.position.y:.1f})"


class Enemy(GameObject):
    """Enemy character"""
    def __init__(self, position: Vector2D, name: str, hp: int, attack: int, speed: float):
        super().__init__(position, name)
        self._hp = hp
        self._attack = attack
        self._speed = speed
        self._target: Player | None = None

    def set_target(self, player: Player) -> None:
        self._target = player

    def take_damage(self, amount: int) -> None:
        self._hp = max(0, self._hp - amount)
        if self._hp == 0:
            self.deactivate()

    def update(self, delta_time: float) -> None:
        if not self.is_active or self._target is None:
            return

        # Move toward the player
        direction = Vector2D(
            self._target.position.x - self.position.x,
            self._target.position.y - self.position.y,
        )
        self.position = self.position + direction.normalized() * self._speed * delta_time

        # Deal damage if within attack range
        if self.position.distance_to(self._target.position) < 1.0:
            self._target.take_damage(self._attack)

    def render(self) -> str:
        return f"[{self.name}] HP:{self._hp} Pos:({self.position.x:.1f},{self.position.y:.1f})"


class GameWorld:
    """Game world: manages all objects"""
    def __init__(self):
        self._objects: list[GameObject] = []

    def add(self, obj: GameObject) -> None:
        self._objects.append(obj)

    def update(self, delta_time: float) -> None:
        for obj in self._objects:
            if obj.is_active:
                obj.update(delta_time)
        # Remove inactive objects
        self._objects = [obj for obj in self._objects if obj.is_active]

    def render(self) -> str:
        return "\n".join(obj.render() for obj in self._objects if obj.is_active)
```

### 6.3 OOP in Enterprise Applications

```java
// Java: an enterprise-application example (order management)

// Entity
public class Order {
    private final String orderId;
    private final String customerId;
    private final List<OrderLine> lines;
    private OrderStatus status;
    private final LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public Order(String orderId, String customerId) {
        this.orderId = orderId;
        this.customerId = customerId;
        this.lines = new ArrayList<>();
        this.status = OrderStatus.DRAFT;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = this.createdAt;
    }

    public void addLine(Product product, int quantity) {
        if (status != OrderStatus.DRAFT) {
            throw new IllegalStateException("Cannot add items to a confirmed order");
        }
        if (quantity <= 0) {
            throw new IllegalArgumentException("Quantity must be at least 1");
        }
        lines.add(new OrderLine(product, quantity));
        updatedAt = LocalDateTime.now();
    }

    public BigDecimal getTotal() {
        return lines.stream()
            .map(OrderLine::getSubtotal)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    public void confirm() {
        if (lines.isEmpty()) {
            throw new IllegalStateException("Cannot confirm an empty order");
        }
        this.status = OrderStatus.CONFIRMED;
        this.updatedAt = LocalDateTime.now();
    }

    public void cancel() {
        if (status == OrderStatus.SHIPPED || status == OrderStatus.DELIVERED) {
            throw new IllegalStateException("Cannot cancel a shipped order");
        }
        this.status = OrderStatus.CANCELLED;
        this.updatedAt = LocalDateTime.now();
    }

    // getters omitted
}

// Value object
public class OrderLine {
    private final Product product;
    private final int quantity;

    public OrderLine(Product product, int quantity) {
        this.product = product;
        this.quantity = quantity;
    }

    public BigDecimal getSubtotal() {
        return product.getPrice().multiply(BigDecimal.valueOf(quantity));
    }
}

// Enum
public enum OrderStatus {
    DRAFT, CONFIRMED, SHIPPED, DELIVERED, CANCELLED
}

// Service layer
public class OrderService {
    private final OrderRepository orderRepository;
    private final InventoryService inventoryService;
    private final NotificationService notificationService;

    public OrderService(
        OrderRepository orderRepository,
        InventoryService inventoryService,
        NotificationService notificationService
    ) {
        this.orderRepository = orderRepository;
        this.inventoryService = inventoryService;
        this.notificationService = notificationService;
    }

    public void placeOrder(String orderId) {
        Order order = orderRepository.findById(orderId)
            .orElseThrow(() -> new OrderNotFoundException(orderId));

        // Business rule: check inventory
        for (OrderLine line : order.getLines()) {
            inventoryService.reserve(line.getProduct(), line.getQuantity());
        }

        order.confirm();
        orderRepository.save(order);

        // Notify
        notificationService.sendOrderConfirmation(order);
    }
}
```

---

## 7. OOP Styles Across Languages

```
┌──────────────┬───────────────────────────────────────┐
│ Language     │ OOP style                              │
├──────────────┼───────────────────────────────────────┤
│ Java         │ Class-based, pure OOP                  │
│ C++          │ Class-based, multi-paradigm            │
│ Python       │ Class-based, duck typing               │
│ TypeScript   │ Classes + structural typing            │
│ Ruby         │ Pure OOP (everything is an object)     │
│ Kotlin       │ Class-based, data classes              │
│ Swift        │ Protocol-oriented + class-based        │
│ Rust         │ Trait-based (no classes)               │
│ Go           │ Structs + interfaces (no classes)      │
│ JavaScript   │ Prototype-based + class syntax         │
└──────────────┴───────────────────────────────────────┘
```

### 7.1 Comparing OOP Implementations Across Languages

Let's implement the same "shape area calculation" in each language to see how OOP styles differ.

```typescript
// TypeScript: structural typing (duck typing + type safety)
interface HasArea {
  area(): number;
}

class TSCircle implements HasArea {
  constructor(private radius: number) {}
  area(): number { return Math.PI * this.radius ** 2; }
}

class TSRectangle implements HasArea {
  constructor(private width: number, private height: number) {}
  area(): number { return this.width * this.height; }
}

// Structural typing: works even without `implements` if the structure matches
const customShape = {
  area(): number { return 42; }
};

function printArea(shape: HasArea): void {
  console.log(`area: ${shape.area()}`);
}

printArea(new TSCircle(5));
printArea(new TSRectangle(3, 4));
printArea(customShape);  // OK: structure matches
```

```python
# Python: duck typing + Protocol (type hints)
from typing import Protocol
import math


class SupportsArea(Protocol):
    """Something that can compute an area (protocol = structural typing)"""
    def area(self) -> float: ...


class PyCircle:
    def __init__(self, radius: float):
        self._radius = radius

    def area(self) -> float:
        return math.pi * self._radius ** 2


class PyRectangle:
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    def area(self) -> float:
        return self._width * self._height


def print_area(shape: SupportsArea) -> None:
    """Duck typing: accepts anything that has an area() method"""
    print(f"area: {shape.area():.2f}")


print_area(PyCircle(5))
print_area(PyRectangle(3, 4))
```

```go
// Go: interfaces + structs (no classes)
package main

import (
    "fmt"
    "math"
)

// Interface (implicit implementation)
type Shape interface {
    Area() float64
}

// Struct + methods (used in place of classes)
type GoCircle struct {
    Radius float64
}

func (c GoCircle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

type GoRectangle struct {
    Width  float64
    Height float64
}

func (r GoRectangle) Area() float64 {
    return r.Width * r.Height
}

// Accepts anything that satisfies the Shape interface
func PrintArea(s Shape) {
    fmt.Printf("area: %.2f\n", s.Area())
}

func main() {
    PrintArea(GoCircle{Radius: 5})
    PrintArea(GoRectangle{Width: 3, Height: 4})
}
```

```rust
// Rust: traits + structs (no classes, no inheritance)
use std::f64::consts::PI;

trait Shape {
    fn area(&self) -> f64;

    // Default implementations are also possible
    fn describe(&self) -> String {
        format!("area: {:.2}", self.area())
    }
}

struct RustCircle {
    radius: f64,
}

impl Shape for RustCircle {
    fn area(&self) -> f64 {
        PI * self.radius * self.radius
    }
}

struct RustRectangle {
    width: f64,
    height: f64,
}

impl Shape for RustRectangle {
    fn area(&self) -> f64 {
        self.width * self.height
    }
}

// Generic over anything bound by the trait
fn print_area(shape: &dyn Shape) {
    println!("{}", shape.describe());
}

fn main() {
    let circle = RustCircle { radius: 5.0 };
    let rect = RustRectangle { width: 3.0, height: 4.0 };
    print_area(&circle);
    print_area(&rect);
}
```

### 7.2 Prototype-Based OOP (JavaScript)

JavaScript is not class-based; it uses prototype-based OOP. The `class` syntax introduced in ES2015 is syntactic sugar over prototypes.

```javascript
// JavaScript: prototype-based OOP

// === Prototype chain (internal mechanism) ===
const animal = {
  speak() {
    return `${this.name} makes a sound`;
  },
};

const dog = Object.create(animal); // use `animal` as the prototype
dog.name = "Pochi";
dog.bark = function () {
  return `${this.name}: Woof woof!`;
};

console.log(dog.bark());   // Pochi: Woof woof!
console.log(dog.speak());  // Pochi makes a sound (inherited from the prototype)

// === ES2015 class syntax (syntactic sugar) ===
class Animal {
  constructor(name) {
    this.name = name;
  }

  speak() {
    return `${this.name} makes a sound`;
  }
}

class Dog extends Animal {
  bark() {
    return `${this.name}: Woof woof!`;
  }
}

const pochi = new Dog("Pochi");
console.log(pochi.bark());   // Pochi: Woof woof!
console.log(pochi.speak());  // Pochi makes a sound

// Verifying the prototype chain
console.log(pochi instanceof Dog);     // true
console.log(pochi instanceof Animal);  // true
console.log(Object.getPrototypeOf(pochi) === Dog.prototype); // true
```

---

## 8. Criticisms and Limitations of OOP

OOP is not a silver bullet. Understanding its limits enables better design decisions.

```
Major criticisms of OOP:

  1. "You wanted a banana, but what you got was a gorilla holding the banana
     and the entire jungle."
     — Joe Armstrong (creator of Erlang)
     -> Excessive dependencies caused by inheritance
     -> To use one class you drag in its parent, grandparent, and so on
     -> Mitigation: prefer composition and keep inheritance hierarchies shallow

  2. Over-abstraction (over-engineering)
     -> The AbstractSingletonProxyFactoryBean problem
     -> Stuffing simple operations with design patterns
     -> Mitigation: YAGNI (You Ain't Gonna Need It)

  3. Complexity from mutable state
     -> Shared mutable state is the enemy of concurrency
     -> Object state changes unpredictably
     -> Mitigation: immutable objects, adopt functional techniques

  4. Learning cost and productivity
     -> Overhead is heavy for small projects
     -> Design knowledge is required to write OOP "correctly"
     -> Mitigation: use it when it fits

  5. Hard-to-test tight coupling
     -> Many dependencies make tests difficult
     -> Mock hell
     -> Mitigation: dependency injection, loose coupling via interfaces
```

### 8.1 Approaches That Go Beyond OOP's Limits

```typescript
// TypeScript: modern approaches addressing OOP's limits

// === Composition over inheritance ===

// Bad example: a deep inheritance hierarchy
// Animal -> Mammal -> Pet -> Dog -> GoldenRetriever

// Good example: composition (combine capabilities)
interface Walkable {
  walk(): void;
}

interface Swimmable {
  swim(): void;
}

interface Trainable {
  train(command: string): void;
}

class DogBehavior implements Walkable, Swimmable, Trainable {
  constructor(private name: string) {}

  walk(): void {
    console.log(`${this.name} goes for a walk`);
  }

  swim(): void {
    console.log(`${this.name} swims`);
  }

  train(command: string): void {
    console.log(`${this.name} learned "${command}"`);
  }
}


// === Immutable objects + builder pattern ===

class ImmutableConfig {
  readonly host: string;
  readonly port: number;
  readonly database: string;
  readonly maxConnections: number;
  readonly timeout: number;

  private constructor(builder: ConfigBuilder) {
    this.host = builder.host;
    this.port = builder.port;
    this.database = builder.database;
    this.maxConnections = builder.maxConnections;
    this.timeout = builder.timeout;
  }

  static builder(): ConfigBuilder {
    return new ConfigBuilder();
  }

  // Create a new config with some fields changed from the existing one
  withHost(host: string): ImmutableConfig {
    return ImmutableConfig.builder()
      .setHost(host)
      .setPort(this.port)
      .setDatabase(this.database)
      .setMaxConnections(this.maxConnections)
      .setTimeout(this.timeout)
      .build();
  }
}

class ConfigBuilder {
  host = "localhost";
  port = 5432;
  database = "mydb";
  maxConnections = 10;
  timeout = 5000;

  setHost(host: string): this { this.host = host; return this; }
  setPort(port: number): this { this.port = port; return this; }
  setDatabase(db: string): this { this.database = db; return this; }
  setMaxConnections(n: number): this { this.maxConnections = n; return this; }
  setTimeout(ms: number): this { this.timeout = ms; return this; }

  build(): ImmutableConfig {
    // @ts-ignore: private constructor access for builder
    return new ImmutableConfig(this);
  }
}
```

---

## 9. A Roadmap for Learning OOP

```
Roadmap for learning OOP:

  Level 1: core concepts
    □ Relationship between classes and objects
    □ Constructors and instantiation
    □ Fields and methods
    □ Access modifiers (public, private, protected)

  Level 2: the four pillars
    □ Encapsulation (information hiding, bundling)
    □ Inheritance (is-a relationships, method overriding)
    □ Polymorphism (interfaces, abstract classes)
    □ Abstraction (hiding complexity, defining contracts)

  Level 3: design principles
    □ SOLID principles
    □ Tell, Don't Ask
    □ Law of Demeter
    □ Composition over Inheritance

  Level 4: design patterns
    □ Creational patterns (Factory, Builder, Singleton)
    □ Structural patterns (Adapter, Decorator, Proxy)
    □ Behavioral patterns (Strategy, Observer, Command)

  Level 5: architecture
    □ Layered architecture
    □ Domain-Driven Design (DDD)
    □ Clean Architecture
    □ Dependency Injection (DI)

  Level 6: multi-paradigm
    □ Blending OOP + FP
    □ Reactive programming
    □ Actor model
    □ Event sourcing
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Accumulating hands-on experience is most important. Understanding deepens when you go beyond theory and actually write code and observe how it runs.

### Q2: What mistakes do beginners commonly make?

Skipping fundamentals and jumping straight to advanced material. We recommend firmly grasping the foundational concepts explained in this guide before moving on.

### Q3: How is this used in real-world work?

Knowledge of this topic is used frequently in day-to-day development, especially during code reviews and architectural design.

---

## Summary

| Concept | Key point |
|------|---------|
| Essence of OOP | Integrate data and behavior into objects |
| Three elements | State + behavior + identity |
| Four pillars | Encapsulation, inheritance, polymorphism, abstraction |
| Messages | Method calls between objects |
| Strengths | GUIs, games, enterprise, frameworks |
| Limits | Over-abstraction, shared mutable state, inheritance complexity |
| Reality | The OOP + FP multi-paradigm approach is mainstream |
| Learning | Basics -> four pillars -> SOLID -> patterns -> architecture |

---

## Recommended Next Guides

---

## References
1. Kay, A. "The Early History of Smalltalk." ACM SIGPLAN Notices, 1993.
2. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
3. Martin, R. "Clean Code." Prentice Hall, 2008.
4. Bloch, J. "Effective Java." 3rd Ed, Addison-Wesley, 2018.
5. Armstrong, J. "Coders at Work." Apress, 2009.
6. Meyer, B. "Object-Oriented Software Construction." 2nd Ed, Prentice Hall, 1997.
