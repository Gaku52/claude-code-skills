# OOP Anti-Patterns

> Anti-patterns are "common bad designs." This guide explains the traps that OOP developers frequently fall into—such as the God Object, deep inheritance hierarchies, and the Anemic Domain Model—and how to avoid them. Recognizing anti-patterns and being able to refactor them appropriately is an essential skill for stepping up from intermediate to senior engineer.

## What You Will Learn in This Chapter

- [ ] Be able to recognize the major OOP anti-patterns
- [ ] Understand the problems and root causes of each anti-pattern
- [ ] Learn how to resolve them through refactoring
- [ ] Be able to point out anti-patterns during code reviews
- [ ] Acquire preventive design techniques


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding the content of [Behavioral Patterns](./02-behavioral-patterns.md)

---

## 1. God Object

### 1.1 Overview and Symptoms

The God Object is one of the most common and most harmful anti-patterns. It refers to a situation where a single class takes on many of the application's responsibilities, effectively becoming an "omniscient and omnipotent" object.

```
Symptoms:
  → One class knows everything and does everything
  → A class of more than 1,000 lines
  → More than 20 methods
  → Every class depends on this class
  → Method names lack consistency (create, process, handle, manage...)
  → Multiple distinct domain concepts are mixed together

Causes:
  → The accumulation of "just add it here for now"
  → Development without awareness of separation of responsibilities
  → The result of continuing to add code without designing from the start
  → No agreement among team members on class design

Problems:
  → Difficult to change (huge impact scope)
  → Difficult to test (too many dependencies)
  → Frequent conflicts in team development
  → Increased compile time
  → Increased memory usage (unnecessary data also gets loaded)
```

### 1.2 How God Objects Emerge

```
Bloat over time:

 Month 1:  [UserService: 100 lines]   ← Initially appropriate
 Month 3:  [UserService: 300 lines]   ← "Just a little addition"
 Month 6:  [UserService: 800 lines]   ← "Because it's related"
 Month 12: [UserService: 2000 lines]  ← God Object completed
 Month 18: [UserService: 5000 lines]  ← Nobody wants to touch it

Growth of dependencies:
                 ┌──────────┐
    ┌───────────►│          │◄───────────┐
    │            │   God    │            │
    │   ┌──────►│  Object  │◄──────┐   │
    │   │        │          │        │   │
    │   │        └──────────┘        │   │
    │   │         ▲  ▲  ▲           │   │
  [A] [B]       [C][D][E]         [F] [G]

  → Every class depends on the God Object
  → Changes to the God Object ripple through the entire system
```

### 1.3 Concrete Example and Problems

```typescript
// ❌ Typical example of a God Object
class ApplicationManager {
  // User management
  private users: Map<string, User> = new Map();
  private sessions: Map<string, Session> = new Map();

  createUser(data: any) { /* ... */ }
  deleteUser(id: string) { /* ... */ }
  authenticateUser(email: string, password: string) { /* ... */ }
  updateUserProfile(id: string, profile: any) { /* ... */ }
  getUserPermissions(id: string) { /* ... */ }

  // Order management
  private orders: Map<string, Order> = new Map();

  createOrder(userId: string, items: any[]) { /* ... */ }
  cancelOrder(orderId: string) { /* ... */ }
  getOrderHistory(userId: string) { /* ... */ }
  updateOrderStatus(orderId: string, status: string) { /* ... */ }

  // Payment management
  processPayment(orderId: string, card: any) { /* ... */ }
  refundPayment(paymentId: string) { /* ... */ }
  getPaymentHistory(userId: string) { /* ... */ }

  // Notification management
  sendEmail(to: string, body: string) { /* ... */ }
  sendSms(to: string, body: string) { /* ... */ }
  sendPushNotification(userId: string, message: string) { /* ... */ }

  // Reports
  generateMonthlyReport() { /* ... */ }
  generateUserReport(userId: string) { /* ... */ }
  generateSalesReport(from: Date, to: Date) { /* ... */ }

  // Configuration management
  getConfig(key: string) { /* ... */ }
  updateConfig(key: string, value: any) { /* ... */ }

  // Cache management
  clearCache() { /* ... */ }
  warmUpCache() { /* ... */ }

  // ... and more than 100 additional methods
}
```

### 1.4 Incremental Refactoring Procedure

Refactoring a God Object should not be done all at once; it's important to proceed incrementally.

```
Refactoring strategy:

Phase 1: Identify responsibilities
  → Group methods by category
  → Visualize dependencies
  → Determine split boundaries

Phase 2: Extract interfaces
  → Define an interface for each group
  → Have the God Object implement the interfaces
  → Change callers to go through the interfaces

Phase 3: Extract classes
  → Create a new class for each interface
  → Move logic out of the God Object
  → Keep the God Object as a facade (temporarily)

Phase 4: Remove the facade
  → Connect callers directly to the new classes
  → Delete the God Object completely
```

```typescript
// Phase 1: Identify responsibilities (group with comments)

// Phase 2: Extract interfaces
interface IUserService {
  createUser(data: CreateUserDTO): Promise<User>;
  deleteUser(id: string): Promise<void>;
  authenticateUser(email: string, password: string): Promise<AuthResult>;
  updateUserProfile(id: string, profile: ProfileDTO): Promise<User>;
}

interface IOrderService {
  createOrder(userId: string, items: OrderItemDTO[]): Promise<Order>;
  cancelOrder(orderId: string): Promise<void>;
  getOrderHistory(userId: string): Promise<Order[]>;
}

interface IPaymentService {
  processPayment(orderId: string, card: CardDTO): Promise<PaymentResult>;
  refundPayment(paymentId: string): Promise<void>;
}

interface INotificationService {
  sendEmail(to: string, body: string): Promise<void>;
  sendSms(to: string, body: string): Promise<void>;
  sendPushNotification(userId: string, message: string): Promise<void>;
}

// Phase 3: Extract classes
class UserService implements IUserService {
  constructor(
    private readonly userRepo: UserRepository,
    private readonly passwordHasher: PasswordHasher,
    private readonly eventBus: EventBus,
  ) {}

  async createUser(data: CreateUserDTO): Promise<User> {
    const hashedPassword = await this.passwordHasher.hash(data.password);
    const user = new User({
      email: Email.create(data.email),
      password: hashedPassword,
      name: data.name,
    });
    await this.userRepo.save(user);
    await this.eventBus.publish(new UserCreatedEvent(user.id));
    return user;
  }

  async deleteUser(id: string): Promise<void> {
    const user = await this.userRepo.findById(id);
    if (!user) throw new UserNotFoundError(id);
    await this.userRepo.delete(id);
    await this.eventBus.publish(new UserDeletedEvent(id));
  }

  async authenticateUser(email: string, password: string): Promise<AuthResult> {
    const user = await this.userRepo.findByEmail(email);
    if (!user) throw new AuthenticationError("Invalid credentials");
    const isValid = await this.passwordHasher.verify(password, user.password);
    if (!isValid) throw new AuthenticationError("Invalid credentials");
    return { userId: user.id, token: this.generateToken(user) };
  }

  async updateUserProfile(id: string, profile: ProfileDTO): Promise<User> {
    const user = await this.userRepo.findById(id);
    if (!user) throw new UserNotFoundError(id);
    user.updateProfile(profile);
    await this.userRepo.save(user);
    return user;
  }

  private generateToken(user: User): string {
    // JWT generation logic
    return "token";
  }
}

class OrderService implements IOrderService {
  constructor(
    private readonly orderRepo: OrderRepository,
    private readonly userService: IUserService,
    private readonly eventBus: EventBus,
  ) {}

  async createOrder(userId: string, items: OrderItemDTO[]): Promise<Order> {
    const order = Order.create(userId, items);
    await this.orderRepo.save(order);
    await this.eventBus.publish(new OrderCreatedEvent(order.id));
    return order;
  }

  async cancelOrder(orderId: string): Promise<void> {
    const order = await this.orderRepo.findById(orderId);
    if (!order) throw new OrderNotFoundError(orderId);
    order.cancel();
    await this.orderRepo.save(order);
    await this.eventBus.publish(new OrderCancelledEvent(orderId));
  }

  async getOrderHistory(userId: string): Promise<Order[]> {
    return this.orderRepo.findByUserId(userId);
  }
}

// Phase 4: Change callers
// Before: const app = new ApplicationManager(); app.createUser(...);
// After:  use each service directly
class OrderController {
  constructor(
    private readonly orderService: IOrderService,
    private readonly paymentService: IPaymentService,
    private readonly notificationService: INotificationService,
  ) {}

  async placeOrder(req: Request): Promise<Response> {
    const order = await this.orderService.createOrder(req.userId, req.items);
    const payment = await this.paymentService.processPayment(order.id, req.card);
    await this.notificationService.sendEmail(req.userEmail, "Order complete");
    return { orderId: order.id, paymentId: payment.id };
  }
}
```

### 1.5 Rules to Prevent God Objects

```
Preventive measures:
  1. Before adding a new method, ask "Is this the responsibility of this class?"
  2. Clarify responsibility through class naming (Manager/Handler are danger signals)
  3. Consider splitting when a class exceeds 200 lines
  4. Point out mixed responsibilities during code reviews
  5. Monitor class complexity with static analysis tools

Dangerous naming patterns:
  ❌ ApplicationManager
  ❌ SystemHelper
  ❌ Utility
  ❌ CommonService
  ❌ MainController
  → These names imply "anything can go in here"
```

---

## 2. Anemic Domain Model

### 2.1 Overview and Symptoms

The Anemic Domain Model is what Martin Fowler called "the biggest anti-pattern in Domain-Driven Design." It refers to a state where domain objects are nothing more than data containers, and all business logic has leaked out into service classes.

```
Symptoms:
  → Classes have only data (getters/setters)
  → All business logic lives in service classes
  → Domain objects are mere data containers
  → Invariants are not enforced anywhere
  → The same validation logic is scattered across multiple locations

Causes:
  → Separating into "data classes + service classes" becomes an end in itself
  → Confusion between DTOs and domain models
  → The habit of creating classes 1:1 with database tables
  → Procedural programming thinking still remains

Problems:
  → Cannot leverage the advantages of object orientation
  → Duplication of business rules
  → Cannot prevent invalid states
  → Tests need to prepare both service and data
  → Domain knowledge is not expressed in the code
```

### 2.2 Comparison Diagram: Anemic vs Rich

```
Anemic Domain Model:

  ┌─────────────┐         ┌─────────────────┐
  │   Order      │         │  OrderService    │
  │ (data only)  │◄────────│  (logic only)    │
  │             │         │                 │
  │ id          │         │ calculateTotal()│
  │ items[]     │         │ canCancel()     │
  │ status      │         │ cancel()        │
  │ totalPrice  │         │ applyDiscount() │
  │ createdAt   │         │ validate()      │
  └─────────────┘         └─────────────────┘
  * Data and logic are separated → invalid states are possible


Rich Domain Model:

  ┌──────────────────────────┐
  │   Order                   │
  │  (data + logic)           │
  │                          │
  │ - id: OrderId            │
  │ - items: OrderItem[]     │
  │ - status: OrderStatus    │
  │                          │
  │ + addItem(item)          │
  │ + removeItem(itemId)     │
  │ + totalPrice: Money      │
  │ + confirm()              │
  │ + cancel()               │
  │ + applyDiscount(coupon)  │
  └──────────────────────────┘
  * Data and logic are unified → invalid states are prevented by types
```

### 2.3 Concrete Example: Before / After

```typescript
// ❌ Anemic Domain Model
class Order {
  id: string = "";
  items: OrderItem[] = [];
  status: string = "pending";
  totalPrice: number = 0;
  discount: number = 0;
  shippingAddress: string = "";
  createdAt: Date = new Date();
  // Only getters/setters. No logic.
}

class OrderService {
  // All logic is in the service
  calculateTotal(order: Order): number {
    order.totalPrice = order.items.reduce(
      (sum, item) => sum + item.price * item.quantity, 0
    );
    return order.totalPrice;
  }

  applyDiscount(order: Order, discountRate: number): void {
    // Validation is on the service side
    if (discountRate < 0 || discountRate > 0.5) {
      throw new Error("Invalid discount rate");
    }
    order.discount = order.totalPrice * discountRate;
    order.totalPrice -= order.discount;
  }

  canCancel(order: Order): boolean {
    return order.status === "pending" || order.status === "confirmed";
  }

  cancel(order: Order): void {
    if (!this.canCancel(order)) throw new Error("Cannot cancel");
    order.status = "cancelled";
  }

  validate(order: Order): string[] {
    const errors: string[] = [];
    if (order.items.length === 0) errors.push("Items are required");
    if (!order.shippingAddress) errors.push("Shipping address is required");
    if (order.totalPrice < 0) errors.push("Invalid total price");
    return errors;
  }
}

// Problem: it's easy to create invalid states
const order = new Order();
order.status = "shipped";      // Shipped right away?
order.totalPrice = -1000;      // Negative amount?
order.items = [];              // Shipped without items?
// → No errors are raised!
```

```typescript
// ✅ Rich Domain Model
type OrderStatus = "pending" | "confirmed" | "shipped" | "delivered" | "cancelled";

class OrderId {
  constructor(private readonly value: string) {
    if (!value || value.length === 0) {
      throw new Error("OrderId cannot be empty");
    }
  }
  toString(): string { return this.value; }
  equals(other: OrderId): boolean { return this.value === other.value; }
}

class Order {
  private _items: OrderItem[] = [];
  private _status: OrderStatus = "pending";
  private _discount: Money = Money.zero("JPY");
  private _shippingAddress?: Address;
  private readonly _createdAt: Date = new Date();

  constructor(private readonly _id: OrderId) {}

  // Methods that carry business rules
  addItem(item: OrderItem): void {
    if (this._status !== "pending") {
      throw new DomainError("Cannot add to a confirmed order");
    }
    const existing = this._items.find(i => i.productId === item.productId);
    if (existing) {
      existing.increaseQuantity(item.quantity);
    } else {
      this._items.push(item);
    }
  }

  removeItem(productId: string): void {
    if (this._status !== "pending") {
      throw new DomainError("Cannot remove from a confirmed order");
    }
    const index = this._items.findIndex(i => i.productId === productId);
    if (index === -1) throw new DomainError("Item not found");
    this._items.splice(index, 1);
  }

  // Calculation logic lives inside the domain object
  get subtotal(): Money {
    return this._items.reduce(
      (sum, item) => sum.add(item.totalPrice),
      Money.zero("JPY")
    );
  }

  get totalPrice(): Money {
    return this.subtotal.subtract(this._discount);
  }

  applyDiscount(coupon: Coupon): void {
    if (this._status !== "pending") {
      throw new DomainError("Cannot apply a discount to a confirmed order");
    }
    if (!coupon.isValid()) {
      throw new DomainError("Invalid coupon");
    }
    this._discount = coupon.calculateDiscount(this.subtotal);
  }

  setShippingAddress(address: Address): void {
    if (this._status === "shipped" || this._status === "delivered") {
      throw new DomainError("Cannot change the shipping address of a shipped order");
    }
    this._shippingAddress = address;
  }

  // Methods that manage state transitions
  confirm(): void {
    if (this._items.length === 0) {
      throw new DomainError("Cannot confirm an order with no items");
    }
    if (!this._shippingAddress) {
      throw new DomainError("Shipping address is not set");
    }
    if (this._status !== "pending") {
      throw new DomainError(`Cannot confirm a ${this._status} order`);
    }
    this._status = "confirmed";
  }

  ship(): void {
    if (this._status !== "confirmed") {
      throw new DomainError("Only confirmed orders can be shipped");
    }
    this._status = "shipped";
  }

  deliver(): void {
    if (this._status !== "shipped") {
      throw new DomainError("Only shipped orders can be marked as delivered");
    }
    this._status = "delivered";
  }

  cancel(): void {
    if (this._status === "shipped" || this._status === "delivered") {
      throw new DomainError("Cannot cancel a shipped order");
    }
    this._status = "cancelled";
  }

  get id(): OrderId { return this._id; }
  get status(): OrderStatus { return this._status; }
  get items(): ReadonlyArray<OrderItem> { return [...this._items]; }
  get createdAt(): Date { return this._createdAt; }
}

// → Invalid states cannot be created
// → Business rules are consolidated inside the object
// → Testing is straightforward (you only need to test Order)
```

### 2.4 Rich Domain Model Viewed Through State Transitions

```
Order state transitions:

  ┌─────────┐  confirm()  ┌───────────┐  ship()  ┌─────────┐  deliver()  ┌───────────┐
  │ pending  │───────────►│ confirmed │────────►│ shipped │───────────►│ delivered │
  └─────────┘             └───────────┘         └─────────┘            └───────────┘
       │                       │
       │     cancel()          │  cancel()
       │                       │
       ▼                       ▼
  ┌───────────┐          ┌───────────┐
  │ cancelled │          │ cancelled │
  └───────────┘          └───────────┘

  → Business rules are embedded at every state transition
  → Invalid transitions (e.g., pending → delivered) are prevented via exceptions
```

### 2.5 Rich Domain Model Example in Java

```java
// ✅ Rich Domain Model in Java
public class BankAccount {
    private final AccountId id;
    private Money balance;
    private AccountStatus status;
    private final List<Transaction> transactions;

    private BankAccount(AccountId id, Money initialDeposit) {
        if (initialDeposit.isLessThan(Money.of(1000, "JPY"))) {
            throw new DomainException("The minimum deposit is 1000 JPY");
        }
        this.id = id;
        this.balance = initialDeposit;
        this.status = AccountStatus.ACTIVE;
        this.transactions = new ArrayList<>();
        this.transactions.add(Transaction.initialDeposit(initialDeposit));
    }

    public static BankAccount open(AccountId id, Money initialDeposit) {
        return new BankAccount(id, initialDeposit);
    }

    public void deposit(Money amount) {
        ensureActive();
        if (amount.isLessThanOrEqual(Money.zero("JPY"))) {
            throw new DomainException("Deposit amount must be positive");
        }
        this.balance = this.balance.add(amount);
        this.transactions.add(Transaction.deposit(amount));
    }

    public void withdraw(Money amount) {
        ensureActive();
        if (amount.isLessThanOrEqual(Money.zero("JPY"))) {
            throw new DomainException("Withdrawal amount must be positive");
        }
        if (this.balance.isLessThan(amount)) {
            throw new InsufficientFundsException(this.balance, amount);
        }
        this.balance = this.balance.subtract(amount);
        this.transactions.add(Transaction.withdrawal(amount));
    }

    public void freeze() {
        ensureActive();
        this.status = AccountStatus.FROZEN;
    }

    public void close() {
        if (this.balance.isGreaterThan(Money.zero("JPY"))) {
            throw new DomainException("Cannot close an account with a remaining balance");
        }
        this.status = AccountStatus.CLOSED;
    }

    private void ensureActive() {
        if (this.status != AccountStatus.ACTIVE) {
            throw new DomainException("Operation is only allowed on active accounts");
        }
    }

    // Invariants are always guaranteed
    public Money getBalance() { return this.balance; }
    public AccountStatus getStatus() { return this.status; }
    public List<Transaction> getTransactionHistory() {
        return Collections.unmodifiableList(this.transactions);
    }
}
```

### 2.6 Situations Where the Anemic Model Is Appropriate

The Rich Domain Model is not always optimal in every case. In the following situations, the Anemic Model can be a reasonable choice.

```
Situations where the Anemic Model is acceptable:
  → Simple CRUD applications (little business logic)
  → Use as a DTO (Data Transfer Object)
  → Objects for communication with external APIs
  → Read-only data for reports/aggregates

Situations where a Rich Domain Model is essential:
  → Complex business rules exist
  → Managing state transitions is important
  → A shared language with domain experts is needed
  → Invalid states must be reliably prevented
```

---

## 3. Deep Inheritance Hierarchy

### 3.1 Overview and Symptoms

```
Symptoms:
  → Inheritance chains four or more levels deep
  → Changes at each layer affect everything below
  → It becomes unclear which method was defined at which layer
  → Calls like super.super.method()
  → A single change causes unexpected behavior changes

  Entity
  └── LivingEntity
      └── Animal
          └── Mammal
              └── Dog
                  └── GuideDog
                      └── TrainedGuideDog

  → 7 levels! Fixing a bug in TrainedGuideDog
     requires understanding every layer

Solution:
  → Keep inheritance to 2–3 levels
  → Combine features via composition
  → Express type relationships through interfaces
```

### 3.2 Visualizing the Problems of Deep Inheritance

```
The complexity of method resolution:

  Where is TrainedGuideDog.walk() defined?

  TrainedGuideDog   → No walk()
  GuideDog          → No walk()
  Dog               → Has walk()! ... but it calls super.walk()
  Mammal            → Has walk()! ... but it also calls super.move()
  Animal            → Has move()!
  LivingEntity      → None
  Entity            → None

  → "Yo-Yo Problem": you can't understand it without bouncing up and down

The Fragile Base Class Problem:

  Adding a sound() method to Animal:
    → Mammal: no problem
    → Dog: no problem
    → GuideDog: sound() is already defined → unintended override!
    → TrainedGuideDog: behavior changes → tests fail!
```

### 3.3 Refactoring Example

```typescript
// ❌ Deep inheritance
class BaseComponent { /* base features */ }
class StyledComponent extends BaseComponent { /* styling */ }
class InteractiveComponent extends StyledComponent { /* interaction */ }
class AnimatedComponent extends InteractiveComponent { /* animation */ }
class AccessibleComponent extends AnimatedComponent { /* a11y */ }
class MyButton extends AccessibleComponent { /* button */ }

// ✅ Composition + interfaces
interface Stylable { applyStyles(styles: Styles): void; }
interface Interactive { onClick(handler: () => void): void; }
interface Animatable { animate(animation: Animation): void; }
interface Accessible { setAriaLabel(label: string): void; }

class StyleEngine implements Stylable {
  applyStyles(styles: Styles): void {
    // Style application logic
  }
}

class AnimationEngine implements Animatable {
  animate(animation: Animation): void {
    // Animation logic
  }
}

class AccessibilityManager implements Accessible {
  setAriaLabel(label: string): void {
    // ARIA attribute setting logic
  }
}

class MyButton implements Stylable, Interactive, Animatable, Accessible {
  private styleEngine: StyleEngine;
  private animationEngine: AnimationEngine;
  private accessibilityManager: AccessibilityManager;

  constructor(deps: ButtonDeps) {
    this.styleEngine = deps.styleEngine;
    this.animationEngine = deps.animationEngine;
    this.accessibilityManager = deps.accessibilityManager;
  }

  applyStyles(styles: Styles): void {
    this.styleEngine.applyStyles(styles);
  }

  onClick(handler: () => void): void {
    // Click event handling
  }

  animate(animation: Animation): void {
    this.animationEngine.animate(animation);
  }

  setAriaLabel(label: string): void {
    this.accessibilityManager.setAriaLabel(label);
  }
}
```

### 3.4 Mixin Pattern in Python

In Python, the Mixin pattern—which leverages multiple inheritance—can be used as an alternative to deep single inheritance.

```python
# ❌ Deep single inheritance
class BaseModel:
    def save(self): pass

class TimestampedModel(BaseModel):
    created_at: datetime
    updated_at: datetime

class SoftDeletableModel(TimestampedModel):
    deleted_at: datetime | None = None

    def soft_delete(self):
        self.deleted_at = datetime.now()

class AuditableModel(SoftDeletableModel):
    created_by: str
    updated_by: str

class VersionedModel(AuditableModel):
    version: int = 1

class Product(VersionedModel):  # 5 levels of inheritance!
    name: str
    price: float


# ✅ Mixin pattern
class TimestampMixin:
    """Timestamp functionality"""
    created_at: datetime
    updated_at: datetime

    def touch(self):
        self.updated_at = datetime.now()

class SoftDeleteMixin:
    """Soft-delete functionality"""
    deleted_at: datetime | None = None

    def soft_delete(self):
        self.deleted_at = datetime.now()

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

class AuditMixin:
    """Audit log functionality"""
    created_by: str
    updated_by: str

class VersionMixin:
    """Versioning functionality"""
    version: int = 1

    def increment_version(self):
        self.version += 1

class BaseModel:
    """Base model"""
    def save(self): pass

class Product(BaseModel, TimestampMixin, SoftDeleteMixin, AuditMixin, VersionMixin):
    """Combine only the features you need via mixins (inheritance is one level)"""
    name: str
    price: float

# A different model uses a different combination
class LogEntry(BaseModel, TimestampMixin):
    """Log entry: only timestamp is needed"""
    message: str
    level: str
```

### 3.5 When to Use Inheritance vs. Composition

```
Decision criteria:

  ┌─────────────────────┬────────────────────┬────────────────────┐
  │ Criterion           │ Use inheritance    │ Use composition    │
  ├─────────────────────┼────────────────────┼────────────────────┤
  │ Relationship        │ "is-a" relation    │ "has-a" relation   │
  │ Example             │ Cat is an Animal   │ Car has an Engine  │
  │ Direction of reuse  │ Shared behavior    │ Feature composition│
  │ Change frequency    │ Stable base class  │ Features vary      │
  │ Flexibility         │ Decided at compile │ Changeable at run  │
  │ Levels              │ Up to 2–3 levels   │ No limit           │
  └─────────────────────┴────────────────────┴────────────────────┘

  Principle: when in doubt, use composition (Composition over Inheritance)
```

---

## 4. Circular Dependency

### 4.1 Overview and Symptoms

```
Structure of circular dependencies:

  Direct cycle:
    A → B → A

  Indirect cycle:
    A → B → C → A

  Complex cycle:
    A → B → C → D
    ↑           │
    └───────────┘

Symptoms:
  → Compile-time errors or runtime errors
  → Module loading order problems
  → Cannot mock one side in tests
  → Cannot deploy independently
  → Changes propagate in cascade

Causes:
  → Module boundaries were not considered at design time
  → Bidirectional references were added for convenience
  → Insufficient extraction of common modules
```

### 4.2 Concrete Example of Circular Dependency and Its Resolution

```typescript
// ❌ Example of a circular dependency
// user.ts
import { Order } from "./order";

class User {
  orders: Order[] = [];

  getActiveOrders(): Order[] {
    return this.orders.filter(o => o.isActive());
  }

  getTotalSpent(): number {
    return this.orders.reduce((sum, o) => sum + o.totalPrice, 0);
  }
}

// order.ts
import { User } from "./user";  // ← circular dependency!

class Order {
  user: User;
  items: OrderItem[] = [];

  isActive(): boolean {
    return this.status !== "cancelled";
  }

  get totalPrice(): number {
    return this.items.reduce((sum, i) => sum + i.price * i.quantity, 0);
  }

  // A method that uses information from User
  getShippingAddress(): Address {
    return this.user.defaultAddress;  // Depends on User
  }
}
```

```typescript
// ✅ Solution 1: Introduce an interface (dependency inversion)
// interfaces.ts (shared module)
interface IUser {
  id: string;
  defaultAddress: Address;
}

interface IOrder {
  isActive(): boolean;
  totalPrice: number;
}

// user.ts
import { IOrder } from "./interfaces";

class User implements IUser {
  private orders: IOrder[] = [];  // Depends on the interface

  getActiveOrders(): IOrder[] {
    return this.orders.filter(o => o.isActive());
  }
}

// order.ts
import { IUser } from "./interfaces";

class Order implements IOrder {
  private userId: string;  // Reference by ID rather than User object

  constructor(private readonly userProvider: (id: string) => IUser) {}

  getShippingAddress(): Address {
    const user = this.userProvider(this.userId);
    return user.defaultAddress;
  }
}
```

```typescript
// ✅ Solution 2: Introduce an intermediate module
// user.ts - User does not know about Order
class User {
  id: string;
  defaultAddress: Address;
}

// order.ts - Order does not know about User
class Order {
  userId: string;
  items: OrderItem[] = [];

  get totalPrice(): number {
    return this.items.reduce((sum, i) => sum + i.price * i.quantity, 0);
  }
}

// user-order-service.ts - the mediator that knows both
import { User } from "./user";
import { Order } from "./order";

class UserOrderService {
  constructor(
    private userRepo: UserRepository,
    private orderRepo: OrderRepository,
  ) {}

  async getUserActiveOrders(userId: string): Promise<Order[]> {
    const orders = await this.orderRepo.findByUserId(userId);
    return orders.filter(o => o.isActive());
  }

  async getOrderShippingAddress(orderId: string): Promise<Address> {
    const order = await this.orderRepo.findById(orderId);
    const user = await this.userRepo.findById(order.userId);
    return user.defaultAddress;
  }
}
```

```typescript
// ✅ Solution 3: Event-driven architecture
// user.ts
class User {
  deactivate(eventBus: EventBus): void {
    this.status = "inactive";
    eventBus.publish(new UserDeactivatedEvent(this.id));
  }
}

// order.ts - does not reference User directly
class OrderEventHandler {
  constructor(private orderRepo: OrderRepository) {}

  @EventListener(UserDeactivatedEvent)
  async onUserDeactivated(event: UserDeactivatedEvent): Promise<void> {
    const orders = await this.orderRepo.findActiveByUserId(event.userId);
    for (const order of orders) {
      order.cancel();
      await this.orderRepo.save(order);
    }
  }
}
```

### 4.3 The Principle of Dependency Direction

```
Always direct dependencies toward what is more stable:

  Unstable (changes often)
     ↓
  ┌───────────┐
  │ Controller │ ─ → Change frequency: high
  └───────────┘
       │
       ▼
  ┌───────────┐
  │  Service   │ ─ → Change frequency: medium
  └───────────┘
       │
       ▼
  ┌───────────┐
  │  Domain    │ ─ → Change frequency: low
  └───────────┘
       │
       ▼
  ┌───────────┐
  │ Interface  │ ─ → Change frequency: lowest
  └───────────┘
     ↓
  Stable (rarely changes)

  Dependency Inversion Principle (DIP):
    → Depend on interfaces rather than concrete classes
    → High-level modules do not depend on low-level modules
    → Both depend on abstractions
```

---

## 5. Feature Envy

### 5.1 Overview

Feature Envy refers to a situation where a method of one class uses another class's data more than its own.

```
Symptoms:
  → A method calls getters on another object three or more times
  → The method barely uses its own this
  → You find yourself wondering, "Should this method really live in this class?"

Causes:
  → The location of data does not match the location of logic
  → Misplacement of responsibility
```

### 5.2 Concrete Example

```typescript
// ❌ Feature Envy: InvoiceGenerator keeps using Customer's data
class InvoiceGenerator {
  generateGreeting(customer: Customer): string {
    // Uses customer data four times → Feature Envy
    if (customer.getType() === "premium") {
      return `Dear ${customer.getTitle()} ${customer.getLastName()}, ` +
             `thank you for your continued patronage.`;
    }
    return `Dear ${customer.getFirstName()} ${customer.getLastName()}`;
  }

  calculateDiscount(customer: Customer, amount: number): number {
    // Once again, customer data everywhere
    if (customer.getType() === "premium") {
      return amount * customer.getDiscountRate();
    }
    if (customer.getYearsAsCustomer() > 5) {
      return amount * 0.05;
    }
    return 0;
  }
}

// ✅ Refactoring: move the logic to the class that owns the data
class Customer {
  private type: CustomerType;
  private title: string;
  private firstName: string;
  private lastName: string;
  private discountRate: number;
  private registeredAt: Date;

  getGreeting(): string {
    if (this.type === "premium") {
      return `Dear ${this.title} ${this.lastName}, thank you for your continued patronage.`;
    }
    return `Dear ${this.firstName} ${this.lastName}`;
  }

  calculateDiscount(amount: number): number {
    if (this.type === "premium") {
      return amount * this.discountRate;
    }
    if (this.yearsAsCustomer > 5) {
      return amount * 0.05;
    }
    return 0;
  }

  private get yearsAsCustomer(): number {
    return new Date().getFullYear() - this.registeredAt.getFullYear();
  }
}

class InvoiceGenerator {
  generateInvoice(customer: Customer, items: InvoiceItem[]): Invoice {
    const subtotal = this.calculateSubtotal(items);
    const discount = customer.calculateDiscount(subtotal);
    return new Invoice({
      greeting: customer.getGreeting(),
      items,
      subtotal,
      discount,
      total: subtotal - discount,
    });
  }

  private calculateSubtotal(items: InvoiceItem[]): number {
    return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  }
}
```

---

## 6. Shotgun Surgery

### 6.1 Overview

Shotgun Surgery refers to a situation where a single change requires modifying many classes or files. It occurs because a piece of functionality has been scattered across multiple classes.

```
Symptoms:
  → A single specification change requires editing five or more files
  → A "tax rate change" forces modifications to Order, Invoice, Cart, Report, and Export
  → Bugs from missed changes happen frequently
  → "Oh, this also needed to be changed" is a common occurrence

  Specification change: consumption tax 8% → 10%

  Classes that need modification:
    ┌────────┐ ┌───────┐ ┌──────┐ ┌────────┐ ┌────────┐
    │ Order  │ │Invoice│ │ Cart │ │ Report │ │ Export │
    │  8→10  │ │ 8→10  │ │ 8→10 │ │  8→10  │ │  8→10  │
    └────────┘ └───────┘ └──────┘ └────────┘ └────────┘
    → Five places to modify! Miss one and you've got a bug!

Solution:
  → Consolidate related logic into a single class
  → Localize the impact of changes
```

### 6.2 Concrete Example

```typescript
// ❌ Tax rate calculation is scattered
class Order {
  calculateTax(): number {
    return this.subtotal * 0.10;  // ← hard-coded
  }
}

class Invoice {
  getTaxAmount(): number {
    return this.amount * 0.10;  // ← the same value appears here too
  }
}

class Cart {
  estimateTax(): number {
    return this.total * 0.10;  // ← and here
  }
}

class SalesReport {
  calculateTaxTotal(): number {
    return this.sales.reduce((sum, s) => sum + s.amount * 0.10, 0);  // ← and here
  }
}


// ✅ Consolidate tax calculation into one place
class TaxCalculator {
  private static readonly STANDARD_RATE = 0.10;
  private static readonly REDUCED_RATE = 0.08;  // Reduced tax rate

  static calculate(amount: number, type: TaxType = "standard"): number {
    const rate = type === "reduced"
      ? TaxCalculator.REDUCED_RATE
      : TaxCalculator.STANDARD_RATE;
    return Math.floor(amount * rate);
  }

  static getRate(type: TaxType = "standard"): number {
    return type === "reduced"
      ? TaxCalculator.REDUCED_RATE
      : TaxCalculator.STANDARD_RATE;
  }
}

class Order {
  calculateTax(): number {
    return TaxCalculator.calculate(this.subtotal);
  }
}

class Invoice {
  getTaxAmount(): number {
    return TaxCalculator.calculate(this.amount);
  }
}

// → When the tax rate changes, only TaxCalculator needs to be modified
```

---

## 7. Primitive Obsession

### 7.1 Overview

Primitive Obsession is the anti-pattern of continuing to represent domain concepts with primitive types such as string, number, and boolean.

```
Symptoms:
  → Email addresses are strings
  → Amounts are numbers
  → Phone numbers are strings
  → Statuses are strings ("active", "inactive", ...)
  → The same validation is scattered across multiple places

Problems:
  → No type safety (it's easy to confuse one string with another)
  → Duplication of validation
  → Invalid values cannot be prevented
  → The meaning of the domain is not expressed in code
```

### 7.2 Resolution via Value Objects

```typescript
// ❌ Primitive Obsession
function createUser(
  name: string,
  email: string,
  phone: string,
  age: number,
  zipCode: string,
): void {
  // name, email, phone, and zipCode are all strings
  // Even if you get the argument order wrong, there's no compile error!
  // createUser("test@email.com", "John Doe", "100-0001", 25, "090-1234-5678")
  //            ↑ email and name are swapped! But the code still compiles
}

function sendEmail(to: string, subject: string, body: string): void {
  // You have to validate that to is a valid email address... every time
  if (!to.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
    throw new Error(`Invalid email: ${to}`);
  }
  // ...
}

// ✅ Use value objects for type safety
class Email {
  private constructor(private readonly value: string) {}

  static create(value: string): Email {
    const trimmed = value.trim().toLowerCase();
    if (!trimmed.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
      throw new InvalidEmailError(value);
    }
    return new Email(trimmed);
  }

  get domain(): string {
    return this.value.split("@")[1];
  }

  toString(): string { return this.value; }
  equals(other: Email): boolean { return this.value === other.value; }
}

class PhoneNumber {
  private constructor(private readonly value: string) {}

  static create(value: string): PhoneNumber {
    const normalized = value.replace(/[-\s()]/g, "");
    if (!normalized.match(/^0\d{9,10}$/)) {
      throw new InvalidPhoneNumberError(value);
    }
    return new PhoneNumber(normalized);
  }

  get formatted(): string {
    // 090-1234-5678 format
    if (this.value.length === 11) {
      return `${this.value.slice(0,3)}-${this.value.slice(3,7)}-${this.value.slice(7)}`;
    }
    return this.value;
  }

  toString(): string { return this.value; }
  equals(other: PhoneNumber): boolean { return this.value === other.value; }
}

class Money {
  constructor(
    public readonly amount: number,
    public readonly currency: "JPY" | "USD" | "EUR",
  ) {
    if (!Number.isFinite(amount)) throw new Error("Amount must be a finite number");
    if (amount < 0) throw new Error("Amount must be non-negative");
    if (currency === "JPY" && !Number.isInteger(amount)) {
      throw new Error("Japanese yen must be an integer");
    }
  }

  add(other: Money): Money {
    this.ensureSameCurrency(other);
    return new Money(this.amount + other.amount, this.currency);
  }

  subtract(other: Money): Money {
    this.ensureSameCurrency(other);
    const result = this.amount - other.amount;
    if (result < 0) throw new Error("Amount would become negative");
    return new Money(result, this.currency);
  }

  multiply(factor: number): Money {
    return new Money(
      this.currency === "JPY"
        ? Math.floor(this.amount * factor)
        : Math.round(this.amount * factor * 100) / 100,
      this.currency,
    );
  }

  isGreaterThan(other: Money): boolean {
    this.ensureSameCurrency(other);
    return this.amount > other.amount;
  }

  private ensureSameCurrency(other: Money): void {
    if (this.currency !== other.currency) {
      throw new CurrencyMismatchError(this.currency, other.currency);
    }
  }

  static zero(currency: "JPY" | "USD" | "EUR"): Money {
    return new Money(0, currency);
  }

  toString(): string {
    if (this.currency === "JPY") return `¥${this.amount.toLocaleString()}`;
    if (this.currency === "USD") return `$${this.amount.toFixed(2)}`;
    return `€${this.amount.toFixed(2)}`;
  }
}

class ZipCode {
  private constructor(private readonly value: string) {}

  static create(value: string): ZipCode {
    const normalized = value.replace(/-/g, "");
    if (!normalized.match(/^\d{7}$/)) {
      throw new Error(`Invalid zip code: ${value}`);
    }
    return new ZipCode(normalized);
  }

  get formatted(): string {
    return `${this.value.slice(0,3)}-${this.value.slice(3)}`;
  }

  toString(): string { return this.value; }
}

// A function that uses value objects
function createUser(
  name: UserName,
  email: Email,
  phone: PhoneNumber,
  age: Age,
  zipCode: ZipCode,
): void {
  // If the argument order is wrong, you get a compile error!
  // Each value was already validated at creation time
}

function sendEmail(to: Email, subject: string, body: string): void {
  // `to` is guaranteed to be a valid email address. No validation needed!
}
```

### 7.3 Value Objects in Python

```python
from dataclasses import dataclass
from typing import Self
import re

# ✅ Value object in Python (dataclass + __post_init__)
@dataclass(frozen=True)  # frozen=True makes it immutable
class Email:
    value: str

    def __post_init__(self):
        normalized = self.value.strip().lower()
        if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', normalized):
            raise ValueError(f"Invalid email: {self.value}")
        # Even with frozen=True, object.__setattr__ can be used inside __post_init__
        object.__setattr__(self, 'value', normalized)

    @property
    def domain(self) -> str:
        return self.value.split('@')[1]

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Money:
    amount: int  # Smallest unit (1 JPY for Japanese yen)
    currency: str

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Amount must be non-negative")
        if self.currency not in ("JPY", "USD", "EUR"):
            raise ValueError(f"Unsupported currency: {self.currency}")

    def add(self, other: Self) -> Self:
        if self.currency != other.currency:
            raise ValueError(f"Currency mismatch: {self.currency} vs {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def subtract(self, other: Self) -> Self:
        if self.currency != other.currency:
            raise ValueError(f"Currency mismatch")
        return Money(self.amount - other.amount, self.currency)

    def multiply(self, factor: float) -> Self:
        return Money(int(self.amount * factor), self.currency)

    @classmethod
    def zero(cls, currency: str) -> Self:
        return cls(0, currency)

    def __str__(self) -> str:
        if self.currency == "JPY":
            return f"¥{self.amount:,}"
        return f"{self.currency} {self.amount / 100:.2f}"


# Usage example
email = Email("User@Example.COM")
print(email)         # user@example.com
print(email.domain)  # example.com

price = Money(1000, "JPY")
tax = price.multiply(0.10)
total = price.add(tax)
print(total)  # ¥1,100
```

### 7.4 Value Objects in Java (record)

```java
// Value object using a Java 16+ record
public record Email(String value) {
    public Email {
        // Validation inside a compact constructor
        value = value.trim().toLowerCase();
        if (!value.matches("^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$")) {
            throw new IllegalArgumentException("Invalid email: " + value);
        }
    }

    public String domain() {
        return value.split("@")[1];
    }
}

public record Money(long amount, Currency currency) {
    public Money {
        if (amount < 0) {
            throw new IllegalArgumentException("Amount must be non-negative");
        }
        Objects.requireNonNull(currency, "Currency must not be null");
    }

    public Money add(Money other) {
        ensureSameCurrency(other);
        return new Money(this.amount + other.amount, this.currency);
    }

    public Money subtract(Money other) {
        ensureSameCurrency(other);
        return new Money(this.amount - other.amount, this.currency);
    }

    private void ensureSameCurrency(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException("Currency mismatch");
        }
    }

    public static Money zero(Currency currency) {
        return new Money(0, currency);
    }
}
```

---

## 8. Misuse of Singleton

### 8.1 Overview

The Singleton pattern itself is a design pattern, but when overused it becomes an anti-pattern. It is often used as a means of hiding global mutable state.

```
Problems:
  → Hiding global state (effectively a global variable)
  → Difficult to test (hard to mock)
  → Hidden dependencies (not visible in the constructor)
  → Problems with concurrency
  → Difficulty managing lifecycle

Legitimate uses of Singleton:
  → Managing hardware resources (e.g., printer spooler)
  → Read-only caches of configuration
  → Log factories
```

### 8.2 Concrete Example

```typescript
// ❌ Misuse of Singleton
class Database {
  private static instance: Database;
  private connection: Connection;

  private constructor() {
    this.connection = createConnection(/* ... */);
  }

  static getInstance(): Database {
    if (!Database.instance) {
      Database.instance = new Database();
    }
    return Database.instance;
  }

  query(sql: string): Promise<any[]> {
    return this.connection.query(sql);
  }
}

// Caller - hidden dependency
class UserRepository {
  async findById(id: string): Promise<User | null> {
    // The dependency on Database is not visible in the constructor!
    const db = Database.getInstance();
    const rows = await db.query(`SELECT * FROM users WHERE id = '${id}'`);
    return rows[0] ? this.toUser(rows[0]) : null;
  }
}

// Problem when testing
// Testing UserRepository requires an actual database
// It is hard to mock Database.getInstance()


// ✅ Resolved via dependency injection
interface IDatabase {
  query(sql: string): Promise<any[]>;
}

class Database implements IDatabase {
  constructor(private connection: Connection) {}

  async query(sql: string): Promise<any[]> {
    return this.connection.query(sql);
  }
}

class UserRepository {
  // Dependency is explicit
  constructor(private readonly db: IDatabase) {}

  async findById(id: string): Promise<User | null> {
    const rows = await this.db.query(`SELECT * FROM users WHERE id = ?`, [id]);
    return rows[0] ? this.toUser(rows[0]) : null;
  }
}

// For testing — a mock can be injected easily
class MockDatabase implements IDatabase {
  private mockData: Map<string, any[]> = new Map();

  setMockResult(sql: string, result: any[]): void {
    this.mockData.set(sql, result);
  }

  async query(sql: string): Promise<any[]> {
    return this.mockData.get(sql) || [];
  }
}

const mockDb = new MockDatabase();
const repo = new UserRepository(mockDb);  // Inject a test database
```

---

## 9. Poltergeist

### 9.1 Overview

A Poltergeist is a class with little reason for existence—an object that merely invokes methods on other classes and does nothing itself.

```
Symptoms:
  → A class just calls methods on other classes
  → It has no state or logic of its own
  → Names like "Manager," "Controller," or "Handler"
  → It's short-lived, used for an instant and then discarded
  → Methods are 1–2 lines and are all delegation
```

### 9.2 Concrete Example

```typescript
// ❌ Poltergeist: a class that does nothing
class OrderProcessor {
  private orderService: OrderService;
  private paymentService: PaymentService;
  private emailService: EmailService;

  processOrder(order: Order): void {
    this.orderService.validate(order);    // just delegates
    this.paymentService.charge(order);    // just delegates
    this.emailService.sendConfirmation(order);  // just delegates
  }
}

// ✅ Solution 1: Remove the Poltergeist and call directly
// (when the workflow is simple)
async function processOrder(
  order: Order,
  orderService: OrderService,
  paymentService: PaymentService,
  emailService: EmailService,
): Promise<void> {
  await orderService.validate(order);
  await paymentService.charge(order);
  await emailService.sendConfirmation(order);
}

// ✅ Solution 2: A value-adding orchestrator
// (legitimate when it has its own logic such as error handling or transaction management)
class OrderWorkflow {
  constructor(
    private orderService: OrderService,
    private paymentService: PaymentService,
    private emailService: EmailService,
    private logger: Logger,
  ) {}

  async execute(order: Order): Promise<OrderResult> {
    this.logger.info(`Processing order ${order.id}`);

    // Validation
    const validationResult = await this.orderService.validate(order);
    if (!validationResult.isValid) {
      this.logger.warn(`Order ${order.id} validation failed`, validationResult.errors);
      return OrderResult.failure(validationResult.errors);
    }

    // Payment (with retry)
    let paymentResult: PaymentResult;
    try {
      paymentResult = await this.retryWithBackoff(
        () => this.paymentService.charge(order),
        3,
      );
    } catch (error) {
      this.logger.error(`Payment failed for order ${order.id}`, error);
      await this.orderService.markAsFailed(order.id);
      return OrderResult.paymentFailed(error);
    }

    // Confirmation email (the order still succeeds even if this fails)
    try {
      await this.emailService.sendConfirmation(order);
    } catch (error) {
      this.logger.warn(`Email sending failed for order ${order.id}`, error);
      // Email failure does not make the order a failure
    }

    return OrderResult.success(order.id, paymentResult.transactionId);
  }

  private async retryWithBackoff<T>(
    fn: () => Promise<T>,
    maxRetries: number,
  ): Promise<T> {
    // Retry logic
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (error) {
        if (i === maxRetries - 1) throw error;
        await this.delay(Math.pow(2, i) * 1000);
      }
    }
    throw new Error("Unreachable");
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

---

## 10. Lava Flow

### 10.1 Overview

Lava Flow refers to "code that nobody knows why it's there, but everyone is too afraid to touch." Past implementations remain as debris, hardening like cooled lava until they become difficult to remove.

```
Symptoms:
  → "What is this used for?" "I don't know, but deleting it might break things"
  → Large amounts of commented-out code
  → Unused classes and methods
  → TODO comments left in place for years
  → Files named "legacy," "old," "temp," or "test"
  → Duplicated code that does the same thing in multiple ways

Causes:
  → Prototype code shipped straight into production
  → Refactoring was interrupted midway
  → The responsible person left and the knowledge was lost
  → A "don't touch it if it works" culture

Countermeasures:
  → Raise test coverage before deleting
  → Check the last-modified date with git log (untouched for years is a candidate)
  → Use dead-code detection tools
  → Clarify code ownership
  → Periodic "technical debt repayment sprints"
```

```typescript
// ❌ Example of Lava Flow
class OrderProcessor {
  // TODO: refactor this (2019-03-15)
  process(order: Order): void {
    // ...
  }

  // Old version (planned to be deleted after migration completes)
  // processOld(order: Order): void {
  //   const tax = order.total * 0.08;  // old tax rate
  //   // ... 200 lines of commented-out code
  // }

  // There's also processV2...
  processV2(order: Order): void {
    // A "temporary" fix (from 3 years ago)
    if (order.type === "legacy") {
      return this.processLegacy(order);
    }
    // ...
  }

  // Who is using this? Unknown
  private processLegacy(order: Order): void {
    // Code from a 2020 system migration
    // Is it still being used? Unknown...
  }
}
```

---

## 11. Detection and Avoidance

### 11.1 Quantitative Detection Indicators

```
Detection indicators:
  → Class line count > 300 → consider splitting
  → Method count > 15 → consider separating responsibilities
  → Inheritance depth > 3 → switch to composition
  → Dependency count > 7 → consider introducing a facade
  → Use of instanceof → switch to polymorphism
  → Classes with only getters/setters → Rich Domain Model
  → The same validation in three or more places → value object
  → Circular imports → interface segregation
  → Three or more getter calls on another object in a method → Feature Envy

Reference metrics:
  ┌────────────────────────┬─────────┬──────────┬─────────┐
  │ Metric                  │ Good     │ Caution  │ Danger  │
  ├────────────────────────┼─────────┼──────────┼─────────┤
  │ Class line count        │ < 200   │ 200-500  │ > 500   │
  │ Method count            │ < 10    │ 10-20    │ > 20    │
  │ Inheritance depth       │ 1-2     │ 3        │ > 3     │
  │ Cyclomatic complexity   │ < 10    │ 10-20    │ > 20    │
  │ Number of dependencies  │ < 5     │ 5-10     │ > 10    │
  │ Method parameter count  │ < 4     │ 4-6      │ > 6     │
  │ Code coverage           │ > 80%   │ 50-80%   │ < 50%   │
  └────────────────────────┴─────────┴──────────┴─────────┘
```

### 11.2 Tools

```
Static analysis tools:
  → SonarQube: code quality metrics (code smell detection)
  → ESLint: complexity rules (JavaScript/TypeScript)
  → Pylint: code quality checks for Python
  → SpotBugs / PMD: bug pattern detection for Java
  → IDEs: class diagram visualization (IntelliJ, VS Code)

TypeScript / JavaScript:
  → eslint-plugin-sonarjs: code smell detection
  → eslint-plugin-import: circular dependency detection
  → madge: module dependency visualization
  → dependency-cruiser: define and enforce dependency rules

Python:
  → pylint: general code quality
  → radon: cyclomatic complexity measurement
  → vulture: dead code detection
  → pydeps: dependency graph visualization

Java:
  → Checkstyle: coding style
  → ArchUnit: architecture tests
  → JDepend: package dependency analysis
```

### 11.3 Code Review Checklist

```
Anti-pattern checks during code review:

□ God Object
  - Does the new method fit the responsibility of the existing class?
  - Can you describe the class in a single sentence?

□ Anemic Domain Model
  - Do the domain objects have business logic?
  - Is the service class directly manipulating setters on data classes?

□ Deep Inheritance
  - Is inheritance within three levels?
  - Wouldn't composition be more appropriate than inheritance?

□ Circular Dependency
  - Does the new import create a cycle?
  - Is the direction of dependencies between modules appropriate?

□ Feature Envy
  - Is the method using a lot of data from other objects?
  - Is the logic placed in the class that owns the data?

□ Shotgun Surgery
  - Are there other places that need to be modified for this change?
  - Is the same logic present in multiple places?

□ Primitive Obsession
  - Are domain concepts being represented with primitive types?
  - Is validation scattered around?

□ Lava Flow
  - Is there unnecessary commented-out code left behind?
  - Have any unused methods or classes been added?
```

### 11.4 Design Principles

```
Principles:
  1. Small classes (SRP: Single Responsibility Principle)
     → A class should change for only one reason

  2. Shallow inheritance (prefer composition)
     → Don't inherit unless it's an "is-a" relationship

  3. Rich domain model (have behavior)
     → Unify data and logic

  4. Value objects (express constraints via types)
     → Create dedicated types for domain concepts

  5. Make dependency direction one-way (DIP: Dependency Inversion Principle)
     → Depend on abstractions rather than concretions

  6. Interface Segregation Principle (ISP)
     → Don't force clients to depend on methods they don't use

  7. Open/Closed Principle (OCP)
     → Open to extension, closed to modification

  Mapping to SOLID principles:
  ┌──────────────────┬──────────────────────────┐
  │ Anti-pattern     │ Violated SOLID principle  │
  ├──────────────────┼──────────────────────────┤
  │ God Object       │ SRP (Single Responsibility)│
  │ Anemic Model     │ OCP (Open/Closed)          │
  │ Deep Inheritance │ LSP (Liskov Substitution)  │
  │ Circular Deps    │ DIP (Dependency Inversion) │
  │ Feature Envy     │ SRP (Single Responsibility)│
  │ Shotgun Surgery  │ SRP (Single Responsibility)│
  │ Primitive Obsession │ ISP (Interface Segregation)│
  └──────────────────┴──────────────────────────┘
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not through theory alone but by actually writing code and observing how it behaves.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is used frequently in day-to-day development work. It is especially important during code reviews and architecture design.

---

## Summary

| Anti-pattern | Symptoms | Solution | Detection method |
|---------------|------|--------|----------|
| God Object | A class that does everything | Split via SRP | Lines > 300, methods > 15 |
| Anemic Model | Data-only classes | Rich Domain Model | Only getters/setters |
| Deep Inheritance | Four or more levels | Composition | Measure inheritance depth |
| Circular Dependency | A→B→C→A | Introduce interfaces | Dependency graph analysis |
| Feature Envy | Heavy use of another class's data | Move the method | Count of getter calls on other objects |
| Shotgun Surgery | One change edits many files | Consolidate logic | Analyze impact scope of changes |
| Primitive Obsession | Overuse of string/number | Value objects | Duplicate validation |
| Singleton Misuse | Global state | Dependency injection | Frequent use of getInstance() |
| Poltergeist | A class that only delegates | Remove/merge the class | Nothing but 1–2-line methods |
| Lava Flow | Untouchable code | Add tests, then delete | Check last-modified date |

---

## Exercises

### Exercise 1: Splitting a God Object

The `AppController` below is a classic God Object. Split it into five or more classes, clarifying the responsibility of each.

```typescript
class AppController {
  // User authentication
  login(email: string, password: string): Token { /* ... */ }
  logout(token: string): void { /* ... */ }
  refreshToken(token: string): Token { /* ... */ }

  // Profile management
  getProfile(userId: string): Profile { /* ... */ }
  updateProfile(userId: string, data: any): void { /* ... */ }
  uploadAvatar(userId: string, file: File): string { /* ... */ }

  // Product management
  listProducts(filter: any): Product[] { /* ... */ }
  getProduct(id: string): Product { /* ... */ }
  searchProducts(query: string): Product[] { /* ... */ }

  // Cart management
  addToCart(userId: string, productId: string): void { /* ... */ }
  removeFromCart(userId: string, productId: string): void { /* ... */ }
  getCart(userId: string): Cart { /* ... */ }

  // Orders
  checkout(userId: string): Order { /* ... */ }
  getOrderHistory(userId: string): Order[] { /* ... */ }

  // Administrative features
  getSystemStats(): Stats { /* ... */ }
  generateReport(type: string): Report { /* ... */ }
}
```

### Exercise 2: Anemic → Rich Refactoring

Refactor the following Anemic Domain Model into a Rich Domain Model. Ensure that no business rule leaks outside of the service.

```typescript
class Subscription {
  plan: "free" | "basic" | "premium" = "free";
  startDate: Date = new Date();
  endDate: Date | null = null;
  isCancelled: boolean = false;
  paymentMethod: string | null = null;
}

class SubscriptionService {
  upgrade(sub: Subscription, newPlan: string): void {
    if (sub.isCancelled) throw new Error("Cancelled");
    if (newPlan === "free") throw new Error("Cannot downgrade to free");
    sub.plan = newPlan as any;
  }

  cancel(sub: Subscription): void {
    if (sub.isCancelled) throw new Error("Already cancelled");
    sub.isCancelled = true;
    sub.endDate = new Date();
  }

  isActive(sub: Subscription): boolean {
    return !sub.isCancelled &&
           (sub.endDate === null || sub.endDate > new Date());
  }
}
```

### Exercise 3: Resolving a Circular Dependency

The code below has a circular dependency. Introduce an interface to break the cycle.

```typescript
// department.ts
import { Employee } from "./employee";
class Department {
  employees: Employee[] = [];
  manager: Employee;
  getHeadcount(): number { return this.employees.length; }
  getBudget(): number {
    return this.employees.reduce((sum, e) => sum + e.salary, 0);
  }
}

// employee.ts
import { Department } from "./department";  // circular dependency!
class Employee {
  salary: number;
  department: Department;
  getDepartmentName(): string { return this.department.name; }
  getColleagues(): Employee[] { return this.department.employees; }
}
```

### Exercise 4: Resolving Primitive Obsession

Replace the primitive types used in the code below with value objects. Create at least three value objects (any of Email, PhoneNumber, Address).

```typescript
function registerUser(
  name: string,
  email: string,
  phone: string,
  addressLine1: string,
  addressLine2: string,
  city: string,
  zipCode: string,
  country: string,
): User {
  // Validation drags on at the top of the function...
  if (!email.includes("@")) throw new Error("Invalid email");
  if (phone.length < 10) throw new Error("Invalid phone");
  if (zipCode.length !== 7) throw new Error("Invalid zip code");
  // ...
}
```

### Exercise 5: Identifying Anti-Patterns

Identify all the anti-patterns contained in the code below. Also provide a corrective approach for each.

```typescript
class SystemManager {
  private static instance: SystemManager;
  static getInstance() {
    if (!this.instance) this.instance = new SystemManager();
    return this.instance;
  }

  processUserOrder(userId: string, orderId: string) {
    const db = Database.getInstance();
    const user = db.query(`SELECT * FROM users WHERE id = ${userId}`);
    const order = db.query(`SELECT * FROM orders WHERE id = ${orderId}`);

    // Feature Envy: uses user data heavily
    const discount = user.type === "premium" ? user.discountRate :
                     user.yearsActive > 5 ? 0.05 : 0;

    const tax = order.total * 0.10;  // Shotgun Surgery: tax rate hard-coded
    const total = order.total - (order.total * discount) + tax;

    db.query(`UPDATE orders SET total = ${total} WHERE id = ${orderId}`);

    // Also handles notifications, logging, and report updates here...
    this.sendEmail(user.email, `Order confirmed: ${total} yen`);
    this.logActivity(userId, "order_processed");
    this.updateDashboard();
  }

  sendEmail(to: string, body: string) { /* ... */ }
  logActivity(userId: string, action: string) { /* ... */ }
  updateDashboard() { /* ... */ }
  // ... 50 more methods
}
```

**Hint**: at least five anti-patterns are present.

### Exercise 6: Improving Testability

Refactor the class below to eliminate anti-patterns and produce a design that is easy to unit test. Also write at least one piece of test code.

```typescript
class ReportGenerator {
  generate(type: string): string {
    const db = Database.getInstance();
    const data = db.query(`SELECT * FROM ${type}_data`);
    const now = new Date();

    let report = `Report: ${type}\nGenerated: ${now.toISOString()}\n\n`;

    for (const row of data) {
      report += `${row.name}: ${row.value}\n`;
    }

    // Write directly to a file
    const fs = require("fs");
    fs.writeFileSync(`/reports/${type}_${now.getTime()}.txt`, report);

    return report;
  }
}
```

---


## Recommended Next Guides

- Please refer to the other guides in the same category

---

## References
1. Fowler, M. "Refactoring: Improving the Design of Existing Code." 2nd Ed, Addison-Wesley, 2018.
2. Brown, W. "AntiPatterns: Refactoring Software, Architectures, and Projects in Crisis." Wiley, 1998.
3. Evans, E. "Domain-Driven Design: Tackling Complexity in the Heart of Software." Addison-Wesley, 2003.
4. Martin, R. C. "Clean Code: A Handbook of Agile Software Craftsmanship." Prentice Hall, 2008.
5. Martin, R. C. "Agile Software Development, Principles, Patterns, and Practices." Prentice Hall, 2002.
6. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
7. Fowler, M. "AnemicDomainModel." martinfowler.com (2003).
8. Vernon, V. "Implementing Domain-Driven Design." Addison-Wesley, 2013.
