# OOP vs Other Paradigms

> OOP is not a silver bullet. Modern engineers are expected to understand the strengths and weaknesses of each paradigm—procedural, functional, reactive, the actor model, and more—and apply them appropriately.

## What You Will Learn in This Chapter

- [ ] Grasp the characteristics and domains of applicability of the major paradigms
- [ ] Understand the fundamental design philosophy differences between OOP and functional programming
- [ ] Learn the practical use of multi-paradigm approaches
- [ ] Compare implementations of each paradigm across multiple languages
- [ ] Understand the criteria for selecting a paradigm in real-world projects


## Prerequisites

Reading the following beforehand will deepen your understanding of this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of [History and Evolution of OOP](./01-history-and-evolution.md)

---

## 1. Paradigm Comparison

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│              │ Procedural   │ OOP          │ Functional   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Core concept │ Steps/cmds   │ Objects      │ Funcs/xforms │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Data mgmt    │ Global vars  │ Encapsulation│ Immutable    │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Code reuse   │ Functions    │ Inherit/Comp │ Higher-order │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ State mgmt   │ Var mutation │ Via methods  │ Return new   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Side effects │ Anywhere     │ In methods   │ Minimized    │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Testability  │ △           │ ○           │ ◎           │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Concurrency  │ Difficult    │ Hard(shared) │ Easy(immut)  │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Learning cost│ Low          │ Medium       │ High         │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### 1.1 Detailed Characteristics of Each Paradigm

```
Procedural Programming:
  Core: Describing procedures
  Representative languages: C, Pascal, BASIC, Shell Script
  Characteristics:
    - Execute instructions top to bottom, in order
    - Structure code via functions
    - Data and processing are separated
    - State sharing through global variables
  Advantages:
    - Easy to learn
    - Execution flow is intuitive
    - Low overhead
  Disadvantages:
    - Tends to break down at large scale
    - Hard to guarantee data integrity
    - Code reuse is limited

OOP (Object-Oriented Programming):
  Core: Collaboration between objects
  Representative languages: Java, C#, Python, TypeScript, Kotlin
  Characteristics:
    - Integration of data and behavior
    - Information hiding via encapsulation
    - Reuse through inheritance
    - Flexibility through polymorphism
  Advantages:
    - Strong for large-scale development
    - Clear module boundaries in team development
    - Natural modeling of the real world
  Disadvantages:
    - Risk of over-abstraction
    - Concurrency is hard due to shared mutable state
    - Tends to accumulate boilerplate

Functional Programming:
  Core: Data transformation pipelines
  Representative languages: Haskell, Erlang, Clojure, Elm, F#
  Characteristics:
    - Immutable data
    - Pure functions (same input → always same output)
    - Higher-order functions (functions as arguments or return values)
    - Referential transparency (no side effects)
  Advantages:
    - Easy to test (pure functions)
    - Safe concurrency (no shared state)
    - Easy-to-reason-about code
  Disadvantages:
    - High learning cost
    - Unnatural for domains where state management is essential
    - Performance tuning can be difficult

Reactive Programming:
  Core: Data flow and propagation of changes
  Representatives: RxJS, Reactor, Akka Streams
  Characteristics:
    - Asynchronous data streams
    - Event-driven
    - Backpressure
  Advantages:
    - Declarative description of asynchronous processing
    - Well-suited to UI event handling
  Disadvantages:
    - Debugging is difficult
    - High learning cost

Actor Model:
  Core: Message passing between independent computational units (actors)
  Representatives: Erlang/OTP, Akka, Orleans
  Characteristics:
    - Each actor has its own state
    - Asynchronous sending and receiving of messages
    - Fault isolation (Let it crash)
  Advantages:
    - Well-suited to distributed systems
    - High scalability
    - Excellent fault tolerance
  Disadvantages:
    - Limited guarantee on message ordering
    - Debugging is difficult
```

---

## 2. Solving the Same Problem with Each Paradigm

### Problem: Extract email addresses of adults from a user list

```python
# === Procedural ===
users = [
    {"name": "Tanaka", "age": 25, "email": "tanaka@example.com"},
    {"name": "Yamada", "age": 17, "email": "yamada@example.com"},
    {"name": "Sato", "age": 30, "email": "sato@example.com"},
]

adult_emails = []
for user in users:
    if user["age"] >= 18:
        adult_emails.append(user["email"])
# → Describes the procedure sequentially
```

```python
# === OOP ===
class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def is_adult(self) -> bool:
        return self.age >= 18

class UserRepository:
    def __init__(self, users: list[User]):
        self._users = users

    def get_adult_emails(self) -> list[str]:
        return [u.email for u in self._users if u.is_adult()]

# → Delegates responsibility to objects
repo = UserRepository([
    User("Tanaka", 25, "tanaka@example.com"),
    User("Yamada", 17, "yamada@example.com"),
    User("Sato", 30, "sato@example.com"),
])
adult_emails = repo.get_adult_emails()
```

```python
# === Functional ===
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)  # Immutable
class User:
    name: str
    age: int
    email: str

def is_adult(user: User) -> bool:
    return user.age >= 18

def get_email(user: User) -> str:
    return user.email

users = [
    User("Tanaka", 25, "tanaka@example.com"),
    User("Yamada", 17, "yamada@example.com"),
    User("Sato", 30, "sato@example.com"),
]

adult_emails = list(map(get_email, filter(is_adult, users)))
# → A pipeline of data transformations
```

### Problem 2: Order processing for an e-commerce site

Compare the characteristics of each paradigm with a more complex problem.

```typescript
// === Procedural approach ===

// Data is managed as separate arrays/dictionaries
interface OrderData {
  orderId: string;
  items: { productId: string; quantity: number; price: number }[];
  status: string;
  discount: number;
}

// A collection of functions
function calculateSubtotal(order: OrderData): number {
  let total = 0;
  for (const item of order.items) {
    total += item.price * item.quantity;
  }
  return total;
}

function applyDiscount(order: OrderData): number {
  const subtotal = calculateSubtotal(order);
  return subtotal * (1 - order.discount);
}

function calculateTax(amount: number, rate: number = 0.1): number {
  return Math.floor(amount * rate);
}

function processOrder(order: OrderData): {
  subtotal: number;
  discount: number;
  tax: number;
  total: number;
} {
  const subtotal = calculateSubtotal(order);
  const afterDiscount = applyDiscount(order);
  const tax = calculateTax(afterDiscount);
  return {
    subtotal,
    discount: subtotal - afterDiscount,
    tax,
    total: afterDiscount + tax,
  };
}
// Problem: Data is scattered across functions, state management is difficult
```

```typescript
// === OOP approach ===

class Money {
  constructor(private readonly _amount: number) {
    if (_amount < 0) throw new Error("Amount must be 0 or more");
  }

  get amount(): number { return this._amount; }

  add(other: Money): Money {
    return new Money(this._amount + other._amount);
  }

  subtract(other: Money): Money {
    return new Money(this._amount - other._amount);
  }

  multiply(factor: number): Money {
    return new Money(Math.floor(this._amount * factor));
  }

  toString(): string {
    return `¥${this._amount.toLocaleString()}`;
  }
}

class OrderItem {
  constructor(
    public readonly productId: string,
    public readonly productName: string,
    public readonly unitPrice: Money,
    public readonly quantity: number,
  ) {
    if (quantity <= 0) throw new Error("Quantity must be 1 or more");
  }

  get subtotal(): Money {
    return this.unitPrice.multiply(this.quantity);
  }
}

type OrderStatus = "draft" | "confirmed" | "paid" | "shipped" | "delivered" | "cancelled";

class Order {
  private _items: OrderItem[] = [];
  private _status: OrderStatus = "draft";
  private _discountRate: number = 0;

  constructor(public readonly orderId: string) {}

  addItem(item: OrderItem): void {
    if (this._status !== "draft") {
      throw new Error("Cannot add items to a confirmed order");
    }
    this._items.push(item);
  }

  applyDiscount(rate: number): void {
    if (rate < 0 || rate > 1) throw new Error("Discount rate must be in the range 0–1");
    this._discountRate = rate;
  }

  get subtotal(): Money {
    return this._items.reduce(
      (sum, item) => sum.add(item.subtotal),
      new Money(0),
    );
  }

  get discountAmount(): Money {
    return this.subtotal.multiply(this._discountRate);
  }

  get afterDiscount(): Money {
    return this.subtotal.subtract(this.discountAmount);
  }

  get tax(): Money {
    return this.afterDiscount.multiply(0.1);
  }

  get total(): Money {
    return this.afterDiscount.add(this.tax);
  }

  confirm(): void {
    if (this._items.length === 0) throw new Error("An empty order cannot be confirmed");
    this._status = "confirmed";
  }

  get status(): OrderStatus {
    return this._status;
  }

  get itemCount(): number {
    return this._items.length;
  }

  display(): string {
    const lines = [`Order ${this.orderId} (${this._status})`];
    for (const item of this._items) {
      lines.push(`  ${item.productName} x${item.quantity} = ${item.subtotal}`);
    }
    lines.push(`  Subtotal: ${this.subtotal}`);
    if (this._discountRate > 0) {
      lines.push(`  Discount: -${this.discountAmount} (${this._discountRate * 100}%)`);
    }
    lines.push(`  Tax: ${this.tax}`);
    lines.push(`  Total: ${this.total}`);
    return lines.join("\n");
  }
}
// Advantage: Business rules are encapsulated within the object
```

```typescript
// === Functional approach ===

// All immutable data types
type FPOrderItem = Readonly<{
  productId: string;
  productName: string;
  unitPrice: number;
  quantity: number;
}>;

type FPOrder = Readonly<{
  orderId: string;
  items: ReadonlyArray<FPOrderItem>;
  status: OrderStatus;
  discountRate: number;
}>;

type OrderSummary = Readonly<{
  subtotal: number;
  discount: number;
  tax: number;
  total: number;
}>;

// Pure functions (no side effects, same input → same output)
const itemSubtotal = (item: FPOrderItem): number =>
  item.unitPrice * item.quantity;

const orderSubtotal = (order: FPOrder): number =>
  order.items.reduce((sum, item) => sum + itemSubtotal(item), 0);

const calculateDiscount = (subtotal: number, rate: number): number =>
  Math.floor(subtotal * rate);

const calculateTax = (amount: number, rate: number = 0.1): number =>
  Math.floor(amount * rate);

const summarize = (order: FPOrder): OrderSummary => {
  const subtotal = orderSubtotal(order);
  const discount = calculateDiscount(subtotal, order.discountRate);
  const afterDiscount = subtotal - discount;
  const tax = calculateTax(afterDiscount);
  return {
    subtotal,
    discount,
    tax,
    total: afterDiscount + tax,
  };
};

// Changes to an order return a new object
const addItem = (order: FPOrder, item: FPOrderItem): FPOrder => ({
  ...order,
  items: [...order.items, item],
});

const applyDiscountFP = (order: FPOrder, rate: number): FPOrder => ({
  ...order,
  discountRate: rate,
});

const confirmOrder = (order: FPOrder): FPOrder => ({
  ...order,
  status: "confirmed" as OrderStatus,
});

// Pipeline: function composition
const pipe = <T>(...fns: ((arg: T) => T)[]) =>
  (initial: T): T => fns.reduce((acc, fn) => fn(acc), initial);

// Usage example
const baseOrder: FPOrder = {
  orderId: "ORD001",
  items: [],
  status: "draft",
  discountRate: 0,
};

const processedOrder = pipe<FPOrder>(
  (o) => addItem(o, { productId: "P1", productName: "Book", unitPrice: 1500, quantity: 2 }),
  (o) => addItem(o, { productId: "P2", productName: "Pen", unitPrice: 200, quantity: 5 }),
  (o) => applyDiscountFP(o, 0.1),
  confirmOrder,
)(baseOrder);

const summary = summarize(processedOrder);
// baseOrder is unchanged (immutable)
// processedOrder is a new object
```

---

## 3. OOP vs Functional: Fundamental Differences

```
The Expression Problem:

  OOP: Easy to add new "types", hard to add new "operations"
  FP:  Easy to add new "operations", hard to add new "types"

  Example: Drawing shapes

  OOP:
    Shape (abstract class)
    ├── Circle.draw()      ← Adding a new type (Triangle) is easy
    ├── Rectangle.draw()      just implement draw() in each class
    └── Triangle.draw()    ← But adding a new operation (area())
                              requires modifying all classes

  FP:
    draw(shape) = match shape with    ← Adding a new operation (area()) is easy
    | Circle r -> ...                    just define a new function
    | Rectangle w h -> ...            ← But adding a new type (Triangle)
                                         requires modifying all functions

  → Which is better depends on "what changes frequently"
  → Types grow → OOP
  → Operations grow → FP
```

### 3.1 A Concrete Example of the Expression Problem

```typescript
// TypeScript: The Expression Problem in OOP

// === OOP approach: easy to add new types ===

interface Shape {
  area(): number;
  perimeter(): number;
  draw(): string;
}

class Circle implements Shape {
  constructor(private radius: number) {}
  area(): number { return Math.PI * this.radius ** 2; }
  perimeter(): number { return 2 * Math.PI * this.radius; }
  draw(): string { return `○ (r=${this.radius})`; }
}

class Rectangle implements Shape {
  constructor(private width: number, private height: number) {}
  area(): number { return this.width * this.height; }
  perimeter(): number { return 2 * (this.width + this.height); }
  draw(): string { return `□ (${this.width}x${this.height})`; }
}

// Adding a new type (Triangle): Easy! No changes to existing code
class Triangle implements Shape {
  constructor(
    private base: number,
    private height: number,
    private sideA: number,
    private sideB: number,
  ) {}
  area(): number { return this.base * this.height / 2; }
  perimeter(): number { return this.base + this.sideA + this.sideB; }
  draw(): string { return `△ (base=${this.base})`; }
}

// However, adding a new operation (e.g., serialize)
// requires modifying all classes → difficult
// Add serialize(): string to interface Shape
// → Add implementation to Circle, Rectangle, and Triangle


// === Functional approach: easy to add new operations ===

type FPShape =
  | { type: "circle"; radius: number }
  | { type: "rectangle"; width: number; height: number }
  | { type: "triangle"; base: number; height: number; sideA: number; sideB: number };

// Operation 1
function fpArea(shape: FPShape): number {
  switch (shape.type) {
    case "circle": return Math.PI * shape.radius ** 2;
    case "rectangle": return shape.width * shape.height;
    case "triangle": return shape.base * shape.height / 2;
  }
}

// Operation 2
function fpPerimeter(shape: FPShape): number {
  switch (shape.type) {
    case "circle": return 2 * Math.PI * shape.radius;
    case "rectangle": return 2 * (shape.width + shape.height);
    case "triangle": return shape.base + shape.sideA + shape.sideB;
  }
}

// Adding a new operation (serialize): Easy! No changes to existing code
function fpSerialize(shape: FPShape): string {
  return JSON.stringify(shape);
}

// Adding a new operation (convert to SVG): Easy!
function fpToSvg(shape: FPShape): string {
  switch (shape.type) {
    case "circle":
      return `<circle r="${shape.radius}" />`;
    case "rectangle":
      return `<rect width="${shape.width}" height="${shape.height}" />`;
    case "triangle":
      return `<polygon points="0,${shape.height} ${shape.base},${shape.height} ${shape.base/2},0" />`;
  }
}

// However, adding a new type (pentagon)
// requires modifying all functions → difficult
```

### 3.2 Differences in State Management

```
OOP: Encapsulated mutable state
  account.deposit(1000)   ← Object's internal state changes
  account.withdraw(500)   ← The same object keeps changing

FP: Transforming immutable data
  newAccount = deposit(account, 1000)   ← Returns a new object
  finalAccount = withdraw(newAccount, 500) ← Original account is unchanged

  Advantages of FP:
    - Safe concurrency (no shared state)
    - Time-travel debugging is possible
    - Easy to test (same input → same output)

  Advantages of OOP:
    - Intuitive (close to real-world models)
    - Fits domains where state change is essential, such as GUIs and games
    - Memory efficient (just update the object)
```

```typescript
// TypeScript: Differences in state management — bank account example

// === OOP style: mutable object ===
class MutableBankAccount {
  private _balance: number;
  private _transactions: { amount: number; type: string; date: Date }[] = [];

  constructor(
    public readonly accountNumber: string,
    initialBalance: number = 0,
  ) {
    this._balance = initialBalance;
  }

  deposit(amount: number): void {
    if (amount <= 0) throw new Error("Deposit amount must be positive");
    this._balance += amount;
    this._transactions.push({
      amount,
      type: "deposit",
      date: new Date(),
    });
  }

  withdraw(amount: number): void {
    if (amount > this._balance) throw new Error("Insufficient funds");
    this._balance -= amount;
    this._transactions.push({
      amount: -amount,
      type: "withdrawal",
      date: new Date(),
    });
  }

  get balance(): number {
    return this._balance;
  }

  get transactionCount(): number {
    return this._transactions.length;
  }
}

// Usage: the object's state changes
const account = new MutableBankAccount("001", 10000);
account.deposit(5000);   // balance: 15000 (internals mutate)
account.withdraw(3000);  // balance: 12000 (internals mutate)


// === Functional style: immutable data ===
type ImmutableAccount = Readonly<{
  accountNumber: string;
  balance: number;
  transactions: ReadonlyArray<{
    amount: number;
    type: string;
    timestamp: number;
  }>;
}>;

const createAccount = (accountNumber: string, balance: number = 0): ImmutableAccount => ({
  accountNumber,
  balance,
  transactions: [],
});

const deposit = (account: ImmutableAccount, amount: number): ImmutableAccount => {
  if (amount <= 0) throw new Error("Deposit amount must be positive");
  return {
    ...account,
    balance: account.balance + amount,
    transactions: [
      ...account.transactions,
      { amount, type: "deposit", timestamp: Date.now() },
    ],
  };
};

const withdraw = (account: ImmutableAccount, amount: number): ImmutableAccount => {
  if (amount > account.balance) throw new Error("Insufficient funds");
  return {
    ...account,
    balance: account.balance - amount,
    transactions: [
      ...account.transactions,
      { amount: -amount, type: "withdrawal", timestamp: Date.now() },
    ],
  };
};

// Usage: each operation returns a new state
const acc0 = createAccount("001", 10000);
const acc1 = deposit(acc0, 5000);   // New object (acc0 is unchanged)
const acc2 = withdraw(acc1, 3000);  // New object (acc1 is unchanged)

// Time travel: every state is preserved
console.log(acc0.balance); // 10000 (original state)
console.log(acc1.balance); // 15000 (after deposit)
console.log(acc2.balance); // 12000 (after withdrawal)
```

### 3.3 Managing Side Effects

```python
# Python: Differences in managing side effects

# === Procedural: side effects occur anywhere ===

total_orders = 0  # Global state
log_file = None   # Global resource

def process_order_procedural(order_data: dict) -> dict:
    global total_orders
    total_orders += 1  # Side effect: mutating global state

    # Side effect: file I/O
    with open("orders.log", "a") as f:
        f.write(f"Order: {order_data}\n")

    # Side effect: external API call
    # send_email(order_data["email"], "Order Confirmation")

    result = {
        "order_id": f"ORD-{total_orders}",
        "total": sum(item["price"] * item["qty"] for item in order_data["items"]),
    }
    return result

# Problem: side effects are scattered, and testing is difficult


# === Functional: isolate side effects ===

from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")
E = TypeVar("E")

@dataclass(frozen=True)
class Result:
    """A type representing success/failure"""
    success: bool
    value: object = None
    error: str = ""

    @staticmethod
    def ok(value: object) -> "Result":
        return Result(success=True, value=value)

    @staticmethod
    def fail(error: str) -> "Result":
        return Result(success=False, error=error)


@dataclass(frozen=True)
class OrderInput:
    customer_email: str
    items: tuple  # Immutable tuple


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    subtotal: int
    tax: int
    total: int


# Pure function: no side effects, easy to test
def calculate_order(order_input: OrderInput, order_number: int) -> OrderResult:
    subtotal = sum(item["price"] * item["qty"] for item in order_input.items)
    tax = int(subtotal * 0.1)
    return OrderResult(
        order_id=f"ORD-{order_number:06d}",
        subtotal=subtotal,
        tax=tax,
        total=subtotal + tax,
    )


def validate_order(order_input: OrderInput) -> Result:
    if not order_input.items:
        return Result.fail("Items list is empty")
    if "@" not in order_input.customer_email:
        return Result.fail("Invalid email address")
    return Result.ok(order_input)


# Functions with side effects are placed at the "boundaries"
class OrderProcessor:
    """Side effect boundary: I/O, DB, and external APIs are consolidated here"""

    def __init__(self, db, mailer, logger):
        self._db = db
        self._mailer = mailer
        self._logger = logger

    def process(self, order_input: OrderInput) -> Result:
        # 1. Validation (pure function)
        validation = validate_order(order_input)
        if not validation.success:
            return validation

        # 2. Get order number (side effect: DB)
        order_number = self._db.next_order_number()

        # 3. Calculate (pure function)
        result = calculate_order(order_input, order_number)

        # 4. Save (side effect: DB)
        self._db.save_order(result)

        # 5. Notify (side effect: email)
        self._mailer.send(order_input.customer_email, f"Order Confirmation: {result.order_id}")

        # 6. Log (side effect: file)
        self._logger.info(f"Order processing complete: {result.order_id}")

        return Result.ok(result)
```

---

## 4. Multi-paradigm in Practice

```
Modern best practices:

  "Domain model" → OOP
    Use classes to represent business entities
    Example: User, Order, Product

  "Data transformation" → FP
    Use functions for filtering and transforming data
    Example: map, filter, reduce

  "Side effect management" → FP
    Push I/O and DB operations to the boundaries of functions

  "UI with state" → OOP + Reactive
    Manage component state

Practical example (TypeScript):
```

```typescript
// Domain model: OOP
class Order {
  constructor(
    public readonly id: string,
    public readonly items: OrderItem[],
    public readonly status: OrderStatus,
    public readonly createdAt: Date,
  ) {}

  get totalPrice(): number {
    // FP: data transformation
    return this.items
      .map(item => item.price * item.quantity)
      .reduce((sum, price) => sum + price, 0);
  }

  canCancel(): boolean {
    return this.status === "pending" || this.status === "confirmed";
  }
}

// Data transformation: FP
const getRecentHighValueOrders = (orders: Order[]): Order[] =>
  orders
    .filter(order => order.totalPrice > 10000)
    .filter(order => order.createdAt > thirtyDaysAgo())
    .sort((a, b) => b.totalPrice - a.totalPrice);

// Side effect management: isolated at function boundaries
async function processOrders(repo: OrderRepository): Promise<void> {
  const orders = await repo.findAll();          // Side effect (DB)
  const highValue = getRecentHighValueOrders(orders); // Pure function
  await notifyAdmins(highValue);                // Side effect (notification)
}
```

### 4.1 Choosing a Paradigm per Layer

In real-world applications, the optimal paradigm differs by layer.

```typescript
// TypeScript: choosing a paradigm per layer

// === Domain layer: OOP (entities + value objects) ===

class Email {
  private readonly _value: string;

  constructor(value: string) {
    if (!/^[\w.-]+@[\w.-]+\.\w+$/.test(value)) {
      throw new Error(`Invalid email address: ${value}`);
    }
    this._value = value.toLowerCase();
  }

  get value(): string { return this._value; }
  get domain(): string { return this._value.split("@")[1]; }

  equals(other: Email): boolean {
    return this._value === other._value;
  }

  toString(): string { return this._value; }
}

class Customer {
  private _name: string;
  private _email: Email;
  private _memberSince: Date;
  private _totalPurchases: number = 0;

  constructor(
    public readonly customerId: string,
    name: string,
    email: Email,
  ) {
    this._name = name;
    this._email = email;
    this._memberSince = new Date();
  }

  get name(): string { return this._name; }
  get email(): Email { return this._email; }
  get memberSince(): Date { return this._memberSince; }
  get totalPurchases(): number { return this._totalPurchases; }

  get membershipTier(): "bronze" | "silver" | "gold" | "platinum" {
    if (this._totalPurchases >= 1000000) return "platinum";
    if (this._totalPurchases >= 500000) return "gold";
    if (this._totalPurchases >= 100000) return "silver";
    return "bronze";
  }

  get discountRate(): number {
    switch (this.membershipTier) {
      case "platinum": return 0.15;
      case "gold": return 0.10;
      case "silver": return 0.05;
      case "bronze": return 0;
    }
  }

  recordPurchase(amount: number): void {
    this._totalPurchases += amount;
  }

  updateEmail(newEmail: Email): void {
    this._email = newEmail;
  }
}


// === Application layer: FP (use case orchestration) ===

// Pure function: computing business rules
const calculateOrderTotal = (
  items: { price: number; quantity: number }[],
  discountRate: number,
): { subtotal: number; discount: number; tax: number; total: number } => {
  const subtotal = items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0,
  );
  const discount = Math.floor(subtotal * discountRate);
  const afterDiscount = subtotal - discount;
  const tax = Math.floor(afterDiscount * 0.1);
  return {
    subtotal,
    discount,
    tax,
    total: afterDiscount + tax,
  };
};

// Data transformation pipeline
const getEligibleCustomers = (
  customers: Customer[],
  minTier: string,
): Customer[] => {
  const tierOrder = ["bronze", "silver", "gold", "platinum"];
  const minIndex = tierOrder.indexOf(minTier);
  return customers
    .filter(c => tierOrder.indexOf(c.membershipTier) >= minIndex)
    .sort((a, b) => b.totalPurchases - a.totalPurchases);
};

// Extract the target audience for a campaign email
const getCampaignTargets = (
  customers: Customer[],
  campaign: { minTier: string; minPurchases: number },
): { email: string; name: string; tier: string }[] =>
  customers
    .filter(c => c.totalPurchases >= campaign.minPurchases)
    .filter(c => {
      const tiers = ["bronze", "silver", "gold", "platinum"];
      return tiers.indexOf(c.membershipTier) >= tiers.indexOf(campaign.minTier);
    })
    .map(c => ({
      email: c.email.value,
      name: c.name,
      tier: c.membershipTier,
    }));


// === Infrastructure layer: procedural (explicitly manage side effects) ===

interface CustomerRepository {
  findById(id: string): Promise<Customer | null>;
  findAll(): Promise<Customer[]>;
  save(customer: Customer): Promise<void>;
}

interface EmailService {
  send(to: string, subject: string, body: string): Promise<boolean>;
}

class CampaignService {
  constructor(
    private readonly customerRepo: CustomerRepository,
    private readonly emailService: EmailService,
  ) {}

  async runCampaign(campaign: {
    minTier: string;
    minPurchases: number;
    subject: string;
    bodyTemplate: string;
  }): Promise<{ sent: number; failed: number }> {
    // Side effect: fetch data from DB
    const customers = await this.customerRepo.findAll();

    // Pure function: extract targets
    const targets = getCampaignTargets(customers, campaign);

    // Side effect: send emails
    let sent = 0;
    let failed = 0;
    for (const target of targets) {
      const body = campaign.bodyTemplate
        .replace("{{name}}", target.name)
        .replace("{{tier}}", target.tier);

      const success = await this.emailService.send(
        target.email,
        campaign.subject,
        body,
      );

      if (success) sent++;
      else failed++;
    }

    return { sent, failed };
  }
}
```

### 4.2 Integration with Reactive Programming

```typescript
// TypeScript: concepts of reactive programming

// Observable pattern (simple implementation)
type Observer<T> = (value: T) => void;

class Observable<T> {
  private observers: Observer<T>[] = [];

  subscribe(observer: Observer<T>): () => void {
    this.observers.push(observer);
    // Return an unsubscribe function
    return () => {
      this.observers = this.observers.filter(o => o !== observer);
    };
  }

  protected emit(value: T): void {
    for (const observer of this.observers) {
      observer(value);
    }
  }

  // FP-style transformation methods
  map<U>(transform: (value: T) => U): Observable<U> {
    const result = new Observable<U>();
    this.subscribe(value => {
      result.emit(transform(value));
    });
    return result;
  }

  filter(predicate: (value: T) => boolean): Observable<T> {
    const result = new Observable<T>();
    this.subscribe(value => {
      if (predicate(value)) {
        result.emit(value);
      }
    });
    return result;
  }

  debounce(ms: number): Observable<T> {
    const result = new Observable<T>();
    let timeout: ReturnType<typeof setTimeout> | null = null;
    this.subscribe(value => {
      if (timeout) clearTimeout(timeout);
      timeout = setTimeout(() => result.emit(value), ms);
    });
    return result;
  }
}

// OOP: stateful event source
class SearchInput extends Observable<string> {
  private _value = "";

  get value(): string { return this._value; }

  setValue(newValue: string): void {
    this._value = newValue;
    this.emit(newValue);
  }
}

// FP + Reactive: data transformation pipeline
const searchInput = new SearchInput();

const searchResults = searchInput
  .debounce(300)                          // 300ms debounce
  .filter(query => query.length >= 2)     // 2 characters or more
  .map(query => query.toLowerCase().trim()) // Normalize
  .map(query => `Search: "${query}"`);    // Transform for display

searchResults.subscribe(result => {
  console.log(result);
});

// Usage examples
searchInput.setValue("T");        // Ignored (fewer than 2 characters)
searchInput.setValue("Ty");       // After 300ms: 'Search: "ty"'
searchInput.setValue("TypeScript"); // After 300ms: 'Search: "typescript"'
```

### 4.3 Comparison with the Actor Model

```typescript
// TypeScript: a simple implementation of the actor model

type Message = {
  type: string;
  payload: unknown;
  replyTo?: Actor<unknown>;
};

class Actor<T extends Message> {
  private mailbox: T[] = [];
  private processing = false;

  constructor(
    public readonly name: string,
    private readonly handler: (message: T, self: Actor<T>) => void,
  ) {}

  send(message: T): void {
    this.mailbox.push(message);
    this.processNext();
  }

  private processNext(): void {
    if (this.processing || this.mailbox.length === 0) return;
    this.processing = true;

    const message = this.mailbox.shift()!;
    try {
      this.handler(message, this);
    } catch (error) {
      console.error(`Actor ${this.name}: error`, error);
      // Let it crash: isolate the error
    } finally {
      this.processing = false;
      if (this.mailbox.length > 0) {
        // Process the next message asynchronously
        setTimeout(() => this.processNext(), 0);
      }
    }
  }
}

// A bank account in the actor model
type BankMessage =
  | { type: "deposit"; payload: { amount: number }; replyTo?: Actor<any> }
  | { type: "withdraw"; payload: { amount: number }; replyTo?: Actor<any> }
  | { type: "getBalance"; payload: {}; replyTo: Actor<any> }
  | { type: "balanceResponse"; payload: { balance: number } };

function createBankAccountActor(
  accountId: string,
  initialBalance: number,
): Actor<BankMessage> {
  let balance = initialBalance;

  return new Actor<BankMessage>(`account-${accountId}`, (message, self) => {
    switch (message.type) {
      case "deposit":
        balance += (message.payload as { amount: number }).amount;
        console.log(`[${accountId}] Deposit: ${(message.payload as any).amount} → balance: ${balance}`);
        break;
      case "withdraw":
        const amount = (message.payload as { amount: number }).amount;
        if (amount > balance) {
          console.log(`[${accountId}] Insufficient funds`);
        } else {
          balance -= amount;
          console.log(`[${accountId}] Withdrawal: ${amount} → balance: ${balance}`);
        }
        break;
      case "getBalance":
        if (message.replyTo) {
          message.replyTo.send({
            type: "balanceResponse",
            payload: { balance },
          });
        }
        break;
    }
  });
}

// Usage example
const account = createBankAccountActor("001", 10000);
account.send({ type: "deposit", payload: { amount: 5000 } });
account.send({ type: "withdraw", payload: { amount: 3000 } });

// Advantages of the actor model:
// - Each actor has independent state (no shared state)
// - Messages are processed asynchronously
// - Errors do not affect other actors
// - Naturally extends to distributed systems
```

---

## 5. Selection Guidelines

```
When to use OOP:
  ✓ Modeling a business domain (many entities)
  ✓ GUI frameworks (widget hierarchies)
  ✓ Game entity systems
  ✓ Designing public APIs for frameworks/libraries
  ✓ Team development (shared understanding of structure matters)

When to use functional:
  ✓ Data processing pipelines
  ✓ Concurrent/distributed processing
  ✓ Mathematical/scientific computation
  ✓ Compilers/parsers
  ✓ Stateless transformation logic

When to use procedural:
  ✓ Simple scripts
  ✓ Shell-script-like processing
  ✓ Prototypes/throwaway code
  ✓ Hardware control

Multi-paradigm (recommended):
  → Domain model = OOP
  → Data transformation = FP
  → Configuration/scripting = Procedural
  → Concurrency = Actor model / CSP
```

### 5.1 Decision Flowchart

```
Flowchart for choosing a paradigm:

  Q1: What is the size of the project?
    → Under 100 lines → Procedural
    → 100–1000 lines → OOP or FP
    → Over 1000 lines → Go to Q2

  Q2: What is the main concern?
    → Managing entities (users, orders, etc.) → OOP
    → Transforming/processing data → FP
    → Concurrent/distributed processing → Actor model / FP
    → UI / event handling → OOP + Reactive
    → Systems programming → Procedural + OOP

  Q3: What changes frequently?
    → Types (entities) grow → OOP
    → Operations (business rules) grow → FP
    → Both → Multi-paradigm

  Q4: What is the team's proficiency?
    → Experienced in OOP → OOP + FP elements
    → Experienced in FP → FP + OOP elements
    → Mixed team → Multi-paradigm (with clear guidelines)

  Q5: What is the testing strategy?
    → Emphasis on unit tests → FP (easy to test pure functions)
    → Emphasis on integration tests → OOP (DI with mocks)
    → Both → Multi-paradigm
```

### 5.2 Example of Judgment in a Real Project

```typescript
// TypeScript: example decisions for an e-commerce backend

// === Domain model: OOP ===
// Reason: many entities, business rules bound to each entity
class Product {
  constructor(
    public readonly id: string,
    private _name: string,
    private _price: number,
    private _stock: number,
    private _category: string,
  ) {}

  get name(): string { return this._name; }
  get price(): number { return this._price; }
  get stock(): number { return this._stock; }
  get category(): string { return this._category; }
  get isAvailable(): boolean { return this._stock > 0; }

  reserve(quantity: number): void {
    if (quantity > this._stock) throw new Error("Insufficient stock");
    this._stock -= quantity;
  }

  restock(quantity: number): void {
    this._stock += quantity;
  }

  updatePrice(newPrice: number): void {
    if (newPrice < 0) throw new Error("Price must be 0 or more");
    this._price = newPrice;
  }
}


// === Reports/aggregations: FP ===
// Reason: the main work is data transformation and aggregation
type SalesRecord = {
  productId: string;
  category: string;
  amount: number;
  quantity: number;
  date: Date;
};

// Describe aggregation logic with pure functions
const totalSales = (records: SalesRecord[]): number =>
  records.reduce((sum, r) => sum + r.amount, 0);

const salesByCategory = (records: SalesRecord[]): Map<string, number> =>
  records.reduce((map, r) => {
    const current = map.get(r.category) ?? 0;
    map.set(r.category, current + r.amount);
    return map;
  }, new Map<string, number>());

const topProducts = (records: SalesRecord[], n: number): string[] => {
  const productSales = new Map<string, number>();
  for (const r of records) {
    const current = productSales.get(r.productId) ?? 0;
    productSales.set(r.productId, current + r.amount);
  }
  return [...productSales.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([id]) => id);
};

const monthlySales = (records: SalesRecord[]): Map<string, number> =>
  records.reduce((map, r) => {
    const key = `${r.date.getFullYear()}-${String(r.date.getMonth() + 1).padStart(2, "0")}`;
    const current = map.get(key) ?? 0;
    map.set(key, current + r.amount);
    return map;
  }, new Map<string, number>());


// === Batch processing/scripts: procedural ===
// Reason: procedures are clear and the processing is one-off
async function runDailyReport(
  db: Database,
  mailer: EmailService,
): Promise<void> {
  // 1. Fetch data
  const records = await db.getSalesRecords(new Date());

  // 2. Aggregate (use pure functions)
  const total = totalSales(records);
  const byCategory = salesByCategory(records);
  const topN = topProducts(records, 10);

  // 3. Generate report
  const report = formatReport(total, byCategory, topN);

  // 4. Send email
  await mailer.send("admin@example.com", "Daily Sales Report", report);

  // 5. Log
  console.log(`Daily report sent: sales ${total} yen`);
}
```

---

## 6. Mapping Between Paradigms

```
Conceptual correspondences:

  ┌──────────────────┬────────────────┬────────────────┬─────────────────┐
  │ Concept          │ OOP            │ FP             │ Procedural      │
  ├──────────────────┼────────────────┼────────────────┼─────────────────┤
  │ Data + operation │ Class          │ Module         │ Struct + funcs  │
  │ Code reuse       │ Inheritance    │ Higher-order   │ Function library│
  │ Polymorphism     │ Polymorphism   │ Pattern match  │ Function ptr    │
  │ State management │ Encapsulation  │ Immutable data │ Global variables│
  │ Error handling   │ Exceptions     │ Result/Maybe   │ Error codes     │
  │ Dependency mgmt  │ DI             │ Function args  │ Global refs     │
  │ Collection ops   │ Iterators      │ map/filter     │ for loops       │
  │ Async processing │ Future/Promise │ IO monad       │ Callbacks       │
  └──────────────────┴────────────────┴────────────────┴─────────────────┘
```

### 6.1 Comparison of Error Handling

```typescript
// TypeScript: error handling in each paradigm

// === OOP: exception-based ===
class InsufficientFundsError extends Error {
  constructor(
    public readonly accountId: string,
    public readonly balance: number,
    public readonly requested: number,
  ) {
    super(`Insufficient funds: balance ${balance} yen, requested ${requested} yen`);
    this.name = "InsufficientFundsError";
  }
}

class OOPBankService {
  withdraw(accountId: string, amount: number): void {
    const account = this.findAccount(accountId);
    if (account.balance < amount) {
      throw new InsufficientFundsError(accountId, account.balance, amount);
    }
    account.balance -= amount;
  }

  private findAccount(id: string): { balance: number } {
    return { balance: 10000 }; // Simplified
  }
}

// Caller side
try {
  const service = new OOPBankService();
  service.withdraw("001", 50000);
} catch (e) {
  if (e instanceof InsufficientFundsError) {
    console.log(`Insufficient funds: ${e.balance} yen < ${e.requested} yen`);
  }
}


// === FP: Result type-based ===
type WithdrawError =
  | { type: "INSUFFICIENT_FUNDS"; balance: number; requested: number }
  | { type: "ACCOUNT_NOT_FOUND"; accountId: string }
  | { type: "INVALID_AMOUNT"; amount: number };

type WithdrawResult = Result<{ newBalance: number }, WithdrawError>;

type Result<T, E> =
  | { ok: true; value: T }
  | { ok: false; error: E };

function fpWithdraw(
  balance: number,
  amount: number,
): WithdrawResult {
  if (amount <= 0) {
    return { ok: false, error: { type: "INVALID_AMOUNT", amount } };
  }
  if (balance < amount) {
    return {
      ok: false,
      error: { type: "INSUFFICIENT_FUNDS", balance, requested: amount },
    };
  }
  return { ok: true, value: { newBalance: balance - amount } };
}

// Caller side: handle all cases in a type-safe way
const result = fpWithdraw(10000, 50000);
if (result.ok) {
  console.log(`Withdrawal successful: new balance ${result.value.newBalance} yen`);
} else {
  switch (result.error.type) {
    case "INSUFFICIENT_FUNDS":
      console.log(`Insufficient funds: ${result.error.balance} yen < ${result.error.requested} yen`);
      break;
    case "INVALID_AMOUNT":
      console.log(`Invalid amount: ${result.error.amount} yen`);
      break;
    case "ACCOUNT_NOT_FOUND":
      console.log(`Account not found: ${result.error.accountId}`);
      break;
  }
}
```

### 6.2 Comparison of Collection Operations

```python
# Python: comparison of collection operations across paradigms

from dataclasses import dataclass
from typing import Callable
from functools import reduce

@dataclass(frozen=True)
class Employee:
    name: str
    department: str
    salary: int
    years: int

employees = [
    Employee("Tanaka", "Development", 600000, 5),
    Employee("Yamada", "Sales", 450000, 3),
    Employee("Sato", "Development", 750000, 8),
    Employee("Suzuki", "HR", 500000, 2),
    Employee("Takahashi", "Development", 550000, 4),
    Employee("Ito", "Sales", 400000, 1),
    Employee("Watanabe", "HR", 600000, 6),
]

# === Procedural ===
def get_dev_avg_salary_procedural(employees):
    total = 0
    count = 0
    for emp in employees:
        if emp.department == "Development":
            total += emp.salary
            count += 1
    return total / count if count > 0 else 0

# === OOP ===
class EmployeeAnalytics:
    def __init__(self, employees: list[Employee]):
        self._employees = employees

    def average_salary_by_department(self, department: str) -> float:
        dept_employees = [e for e in self._employees if e.department == department]
        if not dept_employees:
            return 0
        return sum(e.salary for e in dept_employees) / len(dept_employees)

    def top_earners(self, n: int) -> list[Employee]:
        return sorted(self._employees, key=lambda e: e.salary, reverse=True)[:n]

    def department_report(self) -> dict[str, dict]:
        report = {}
        for emp in self._employees:
            if emp.department not in report:
                report[emp.department] = {"count": 0, "total_salary": 0, "names": []}
            report[emp.department]["count"] += 1
            report[emp.department]["total_salary"] += emp.salary
            report[emp.department]["names"].append(emp.name)
        for dept in report:
            report[dept]["avg_salary"] = report[dept]["total_salary"] / report[dept]["count"]
        return report

analytics = EmployeeAnalytics(employees)
print(analytics.average_salary_by_department("Development"))
print(analytics.top_earners(3))

# === Functional ===
from itertools import groupby
from operator import attrgetter

# A pipeline of pure functions
def fp_avg_salary_by_dept(employees: list[Employee], dept: str) -> float:
    dept_salaries = [e.salary for e in employees if e.department == dept]
    return sum(dept_salaries) / len(dept_salaries) if dept_salaries else 0

def fp_top_earners(employees: list[Employee], n: int) -> list[Employee]:
    return sorted(employees, key=attrgetter("salary"), reverse=True)[:n]

def fp_department_summary(employees: list[Employee]) -> dict[str, dict]:
    sorted_emps = sorted(employees, key=attrgetter("department"))
    return {
        dept: {
            "count": len(group := list(emps)),
            "avg_salary": sum(e.salary for e in group) / len(group),
            "names": [e.name for e in group],
        }
        for dept, emps in groupby(sorted_emps, key=attrgetter("department"))
    }

# Function composition
def compose(*functions):
    """Compose functions"""
    def composed(data):
        result = data
        for fn in functions:
            result = fn(result)
        return result
    return composed

# Pipeline: names of the top 2 highest-paid employees in the Development department
pipeline = compose(
    lambda emps: [e for e in emps if e.department == "Development"],
    lambda emps: sorted(emps, key=lambda e: e.salary, reverse=True),
    lambda emps: emps[:2],
    lambda emps: [e.name for e in emps],
)

result = pipeline(employees)
print(result)  # ['Sato', 'Tanaka']
```

---

## 7. Fusion of Paradigms: Modern Trends

```
Modern languages are crossing paradigm boundaries and fusing:

  TypeScript: OOP + FP + structural typing
  Kotlin: OOP + FP + coroutines
  Swift: OOP + protocol-oriented + value types
  Rust: traits + FP + ownership
  Scala: complete fusion of OOP + FP
  Python: OOP + FP + procedural

  The era is moving from "which paradigm to write in"
  to "what is the best tool for this problem"

  Best practices:
    1. Default to immutable data
    2. Design around pure functions
    3. Use OOP when entities are needed
    4. Consolidate side effects at boundaries
    5. Always keep testability in mind
```

```typescript
// TypeScript: a practical example of paradigm fusion

// Functional value object
type Currency = "JPY" | "USD" | "EUR";

const createMoney = (amount: number, currency: Currency = "JPY") => {
  if (amount < 0) throw new Error("Amount must be 0 or more");
  return Object.freeze({ amount, currency });
};

type MoneyType = ReturnType<typeof createMoney>;

const addMoney = (a: MoneyType, b: MoneyType): MoneyType => {
  if (a.currency !== b.currency) throw new Error("Currency mismatch");
  return createMoney(a.amount + b.amount, a.currency);
};

// OOP entity (internally leverages immutable data)
class Invoice {
  private readonly _lines: ReadonlyArray<{
    description: string;
    amount: MoneyType;
  }>;

  constructor(
    public readonly invoiceId: string,
    public readonly customerId: string,
    lines: { description: string; amount: MoneyType }[],
  ) {
    this._lines = Object.freeze([...lines]);
  }

  get total(): MoneyType {
    return this._lines.reduce(
      (sum, line) => addMoney(sum, line.amount),
      createMoney(0),
    );
  }

  get lineCount(): number {
    return this._lines.length;
  }

  // FP-style: return a new Invoice (immutable)
  addLine(description: string, amount: MoneyType): Invoice {
    return new Invoice(
      this.invoiceId,
      this.customerId,
      [...this._lines, { description, amount }],
    );
  }
}

// Functional pipeline
const getOverdueInvoices = (
  invoices: Invoice[],
  dueDate: Date,
): Invoice[] =>
  invoices
    .filter(inv => inv.total.amount > 0)
    .sort((a, b) => b.total.amount - a.total.amount);
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Your understanding deepens not only through theory but also by actually writing code and verifying its behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. We recommend firmly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently used in day-to-day development. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Paradigm | Core | Strengths | Weaknesses |
|-----------|------|---------|------|
| Procedural | Describing procedures | Simple scripts | Breaks down at large scale |
| OOP | Objects | Domain modeling, GUIs | Shared state, concurrency |
| Functional | Data transformation | Concurrency, pipelines | GUIs, state management |
| Reactive | Data flow | Async, UI | Difficult to debug |
| Actor model | Messages | Distribution, fault tolerance | Hard to guarantee ordering |
| Multi | Right tool for the job | Modern large-scale development | Requires judgment |

---

## Next Guides to Read

---

## References
1. Wadler, P. "The Expression Problem." 1998.
2. Martin, R. "Clean Architecture." Prentice Hall, 2017.
3. Odersky, M. "Programming in Scala." Artima, 2021.
4. Armstrong, J. "Programming Erlang." Pragmatic Bookshelf, 2013.
5. Wlaschin, S. "Domain Modeling Made Functional." Pragmatic Bookshelf, 2018.
6. Milewski, B. "Category Theory for Programmers." 2018.
