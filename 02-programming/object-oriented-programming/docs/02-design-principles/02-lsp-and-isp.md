# LSP (Liskov Substitution Principle) + ISP (Interface Segregation Principle)

> LSP states "subtypes should be substitutable for their base types," while ISP states "clients should not be forced to depend on methods they do not use." These principles guarantee type correctness and appropriate interface granularity.

## What You Will Learn in This Chapter

- [ ] Understand LSP violation patterns and how to avoid them
- [ ] Grasp proper interface design through ISP
- [ ] Learn practical criteria for design decisions
- [ ] Master the design of robust type hierarchies that combine LSP and ISP
- [ ] Acquire techniques for detecting violations and refactoring in real-world practice


## Prerequisites

Your understanding will deepen if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the content of [SRP (Single Responsibility Principle) + OCP (Open/Closed Principle)](./01-srp-and-ocp.md)

---

## 1. LSP: The Liskov Substitution Principle

```
Definition (Barbara Liskov, 1987):
  "If S is a subtype of T, then objects of type T may be replaced
   with objects of type S without altering the correctness of the program."

In plain terms:
  -> Substituting a subclass where a parent class is used should not break anything
  -> Subclasses must honor the "contract" of the parent class

The contract:
  1. Do not strengthen preconditions (do not narrow the range of acceptance)
  2. Do not weaken postconditions (do not reduce guarantees)
  3. Maintain invariants

Formal definition (relationship with Design by Contract):
  For a subtype S to be a valid substitute for T:
  - Preconditions of S <= Preconditions of T (weaker or equal)
  - Postconditions of S >= Postconditions of T (stronger or equal)
  - S maintains all invariants of T
  - S throws only subtypes of the exceptions thrown by T
```

### 1.1 Historical Background of LSP

```
1987:
  Barbara Liskov first presented this concept in the OOPSLA keynote
  "Data Abstraction and Hierarchy"

1994:
  Liskov and Wing published "A Behavioral Notion of Subtyping,"
  establishing a formal definition of behavioral subtyping

2002:
  Robert C. Martin organized it as part of the SOLID principles
  and restated it in an accessible way for practitioners

Core insight:
  -> Inheritance should be used for "behavioral compatibility,"
     not for "code reuse"
  -> Type hierarchies should be "hierarchies of contracts,"
     not "hierarchies of implementations"
  -> A subtype must not merely possess the methods;
     it must behave correctly in a semantic sense
```

### 1.2 Classic LSP Violation Example: Square and Rectangle

```typescript
// Classic LSP violation example (bad)
class Rectangle {
  constructor(protected width: number, protected height: number) {}

  setWidth(w: number): void { this.width = w; }
  setHeight(h: number): void { this.height = h; }
  area(): number { return this.width * this.height; }
}

class Square extends Rectangle {
  setWidth(w: number): void {
    this.width = w;
    this.height = w; // Since it's a square, keep width and height equal
  }
  setHeight(h: number): void {
    this.width = h;
    this.height = h;
  }
}

// This test breaks = LSP violation
function testRectangle(rect: Rectangle): void {
  rect.setWidth(5);
  rect.setHeight(4);
  console.assert(rect.area() === 20); // With Square, becomes 16!
}

testRectangle(new Rectangle(0, 0)); // OK: 20
testRectangle(new Square(0, 0));    // FAIL: 16 (LSP violation)
```

```typescript
// LSP compliant: abstract through a common interface
interface Shape {
  area(): number;
}

class Rectangle implements Shape {
  constructor(private width: number, private height: number) {}
  area(): number { return this.width * this.height; }
}

class Square implements Shape {
  constructor(private side: number) {}
  area(): number { return this.side * this.side; }
}
// Square does not inherit from Rectangle -> no LSP problem arises
```

### 1.3 Catalog of LSP Violation Patterns

```
Pattern 1: Adding exceptions to methods
  Parent: withdraw(amount) - always succeeds
  Child:  withdraw(amount) - throws on insufficient balance <- strengthened precondition

Pattern 2: Empty implementations
  Parent: save() - persists data
  Child:  save() - does nothing <- weakened postcondition

Pattern 3: Type checks
  if (animal instanceof Dog) {
    animal.fetch();
  }
  -> Polymorphism is broken = sign of LSP violation

Pattern 4: Changing return types
  Parent: findAll() -> always returns an array
  Child:  findAll() -> returns null under certain conditions <- weakened postcondition

Pattern 5: Adding side effects
  Parent: calculate(x) -> returns result (no side effects)
  Child:  calculate(x) -> returns result + writes to log + saves to DB
  -> Unexpected side effects <- invariant violation

Pattern 6: Scope of state changes
  Parent: setName(name) -> changes only the name field
  Child:  setName(name) -> changes name + updatedAt + sends notification
  -> Unexpected state changes from the caller's perspective
```

```python
# LSP violation: empty implementation (bad)
class Bird:
    def fly(self) -> str:
        return "flying"

class Penguin(Bird):
    def fly(self) -> str:
        raise NotImplementedError("Penguins cannot fly")  # LSP violation

# LSP compliant: separate the interfaces
from abc import ABC, abstractmethod

class Bird(ABC):
    @abstractmethod
    def move(self) -> str: ...

class FlyingBird(Bird):
    def move(self) -> str:
        return "flying"

class Penguin(Bird):
    def move(self) -> str:
        return "swimming"  # a legitimate implementation
```

### 1.4 Design by Contract and LSP

```python
# Formalizing LSP using Design by Contract (DbC)
from abc import ABC, abstractmethod
from typing import List


class SortedCollection(ABC):
    """Contract for a sorted collection"""

    @abstractmethod
    def add(self, item: int) -> None:
        """
        Precondition: none (accepts any integer)
        Postcondition: the item is added and the collection remains sorted
        Invariant: the collection is always sorted
        """
        ...

    @abstractmethod
    def get_all(self) -> List[int]:
        """
        Precondition: none
        Postcondition: returns a sorted list
        """
        ...

    def _check_invariant(self) -> bool:
        """Invariant: always sorted"""
        items = self.get_all()
        return all(items[i] <= items[i + 1] for i in range(len(items) - 1))


class AscendingSortedCollection(SortedCollection):
    """LSP compliant: sorted in ascending order"""

    def __init__(self):
        self._items: List[int] = []

    def add(self, item: int) -> None:
        import bisect
        bisect.insort(self._items, item)
        assert self._check_invariant(), "invariant violation"

    def get_all(self) -> List[int]:
        return list(self._items)


class UniqueAscendingSortedCollection(SortedCollection):
    """LSP compliant: ascending sort without duplicates
    Strengthened postcondition (also guarantees deduplication) -> OK
    """

    def __init__(self):
        self._items: List[int] = []

    def add(self, item: int) -> None:
        if item not in self._items:
            import bisect
            bisect.insort(self._items, item)
        assert self._check_invariant(), "invariant violation"

    def get_all(self) -> List[int]:
        return list(self._items)


class BoundedSortedCollection(SortedCollection):
    """LSP violation: restricts the value range (strengthened precondition)"""

    def __init__(self, min_val: int = 0, max_val: int = 100):
        self._items: List[int] = []
        self._min = min_val
        self._max = max_val

    def add(self, item: int) -> None:
        if item < self._min or item > self._max:
            raise ValueError(f"Value must be within {self._min} to {self._max}")
        import bisect
        bisect.insort(self._items, item)

    def get_all(self) -> List[int]:
        return list(self._items)


# Test: verify LSP compliance
def test_sorted_collection(collection: SortedCollection):
    """Test based on the parent class contract"""
    collection.add(5)
    collection.add(1)
    collection.add(3)
    items = collection.get_all()
    assert items == sorted(items), "not sorted!"
    print(f"OK {type(collection).__name__}: {items}")


test_sorted_collection(AscendingSortedCollection())           # OK: [1, 3, 5]
test_sorted_collection(UniqueAscendingSortedCollection())     # OK: [1, 3, 5]
# test_sorted_collection(BoundedSortedCollection(min_val=3))  # FAIL: ValueError
```

### 1.5 Covariance, Contravariance, and LSP

```typescript
// Covariance and contravariance are closely related to LSP

// === Covariant return types (LSP compliant) ===
// Returning a subtype of the parent's return type is OK
class Animal {
  name: string;
  constructor(name: string) { this.name = name; }
}

class Dog extends Animal {
  breed: string;
  constructor(name: string, breed: string) {
    super(name);
    this.breed = breed;
  }
}

class AnimalFactory {
  create(): Animal {
    return new Animal("some animal");
  }
}

class DogFactory extends AnimalFactory {
  // Covariant return type: Dog is a subtype of Animal
  create(): Dog {
    return new Dog("Buddy", "Labrador");
  }
}

// === Contravariant parameters (LSP compliant) ===
// Accepting a supertype of the parent's parameter type is OK
interface AnimalHandler {
  handle(animal: Dog): void;  // accepts only Dog
}

class GeneralAnimalHandler implements AnimalHandler {
  // Accepting a wider type (Animal) is safe
  handle(animal: Animal): void {
    console.log(`Handling ${animal.name}`);
  }
}

// === LSP violation: narrowing parameters ===
// class StrictDogHandler extends GeneralHandler {
//   handle(animal: PurebredDog): void { ... }
//   // BAD: narrowing the parameter (strengthened precondition)
// }
```

```java
// Covariant return types in Java
public class AnimalShelter {
    public Animal adopt() {
        return new Animal("Unknown");
    }
}

public class DogShelter extends AnimalShelter {
    // Covariant return type: return type changed to a subtype (Java 5+)
    @Override
    public Dog adopt() {
        return new Dog("Buddy");
    }
}

// Caller side
AnimalShelter shelter = new DogShelter();
Animal animal = shelter.adopt();  // Returns Dog, usable as Animal -> LSP compliant
```

### 1.6 Detecting LSP Violations in Practice

```typescript
// Five signs that indicate LSP violations

// Sign 1: Presence of instanceof checks
// BAD: polymorphism has collapsed
function processShape(shape: Shape): number {
  if (shape instanceof Circle) {
    return Math.PI * (shape as Circle).radius ** 2;
  } else if (shape instanceof Rectangle) {
    return (shape as Rectangle).width * (shape as Rectangle).height;
  }
  throw new Error("Unknown shape");
}

// GOOD: solve with polymorphism
function processShape(shape: Shape): number {
  return shape.area();  // Each class implements its own area()
}

// Sign 2: NotImplementedError / UnsupportedOperationException
// BAD: subclass rejects the parent's method
class ReadOnlyList<T> extends ArrayList<T> {
  add(item: T): void {
    throw new Error("UnsupportedOperation: read-only list");
  }
}

// GOOD: solve by separating interfaces
interface ReadableList<T> {
  get(index: number): T;
  size(): number;
}

interface WritableList<T> extends ReadableList<T> {
  add(item: T): void;
  remove(index: number): T;
}

// Sign 3: Empty method implementations
// BAD: save() that does nothing
class CacheOnlyRepository implements Repository {
  save(entity: Entity): void {
    // do nothing (cache-only, so no persistence needed)
  }
}

// Sign 4: Branching on type within conditionals
// BAD: branching based on type
function calculatePay(employee: Employee): number {
  switch (employee.type) {
    case "fulltime": return employee.salary;
    case "parttime": return employee.hourlyRate * employee.hours;
    case "contractor": return employee.dailyRate * employee.days;
  }
}

// GOOD: solve with polymorphism
interface Payable {
  calculatePay(): number;
}

class FullTimeEmployee implements Payable {
  constructor(private salary: number) {}
  calculatePay(): number { return this.salary; }
}

class PartTimeEmployee implements Payable {
  constructor(private hourlyRate: number, private hours: number) {}
  calculatePay(): number { return this.hourlyRate * this.hours; }
}

class Contractor implements Payable {
  constructor(private dailyRate: number, private days: number) {}
  calculatePay(): number { return this.dailyRate * this.days; }
}

// Sign 5: Documentation stating "please do not call this method"
// -> A clear sign that there is a problem with the interface's design
```

### 1.7 Refactoring Patterns for LSP Compliance

```python
# Pattern 1: "Extract and Delegate"
# Convert inheritance relationships into composition

# LSP violation: Stack as a subtype of List (bad)
class MyList:
    def __init__(self):
        self._items = []

    def add(self, item):
        self._items.append(item)

    def get(self, index):
        return self._items[index]

    def remove(self, index):
        return self._items.pop(index)

    def size(self):
        return len(self._items)


class Stack(MyList):
    """Is a stack a kind of list? -> No! LSP violation"""
    def push(self, item):
        self.add(item)

    def pop(self):
        return self.remove(self.size() - 1)

    # get(index) is still available -> the stack's contract (LIFO) is broken!


# LSP compliant: implement with composition
class Stack:
    """Stack: LIFO structure"""

    def __init__(self):
        self._items = []  # composition: use a list internally

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("stack is empty")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("stack is empty")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def size(self) -> int:
        return len(self._items)

    # Do not expose get(index) -> preserve the LIFO contract
```

```typescript
// Pattern 2: "Extract Interface"
// Extract a common interface and implement them separately

// LSP violation: mixing persistent and temporary objects (bad)
class PersistentEntity {
  id: string;
  save(): void { /* save to DB */ }
  delete(): void { /* delete from DB */ }
  validate(): boolean { return true; }
}

class TemporaryEntity extends PersistentEntity {
  save(): void { /* do nothing */ }    // BAD: empty implementation
  delete(): void { /* do nothing */ }  // BAD: empty implementation
}

// LSP compliant via interface extraction
interface Validatable {
  validate(): boolean;
}

interface Persistable extends Validatable {
  save(): void;
  delete(): void;
}

class PersistentEntity implements Persistable {
  id: string;
  save(): void { /* save to DB */ }
  delete(): void { /* delete from DB */ }
  validate(): boolean { return true; }
}

class TemporaryEntity implements Validatable {
  validate(): boolean { return true; }
  // Does not have save() or delete() at all -> no LSP problem
}
```

```java
// Pattern 3: "Template Method + Hooks"
// Customize part of the parent class's algorithm in subclasses

public abstract class DataExporter {
    // Template method: skeleton of the algorithm
    public final void export(List<Record> records) {
        validate(records);
        List<String> formatted = format(records);
        String output = join(formatted);
        write(output);
        afterExport(records);  // hook
    }

    // Shared implementation
    protected void validate(List<Record> records) {
        if (records == null || records.isEmpty()) {
            throw new IllegalArgumentException("records are empty");
        }
    }

    // Abstract methods to be implemented by subclasses
    protected abstract List<String> format(List<Record> records);
    protected abstract String join(List<String> formatted);
    protected abstract void write(String output);

    // Optional hook (does nothing by default)
    protected void afterExport(List<Record> records) {
        // Subclasses may override as needed
    }
}

// LSP compliant: customize while honoring the template method contract
public class CsvExporter extends DataExporter {
    @Override
    protected List<String> format(List<Record> records) {
        return records.stream()
            .map(r -> String.join(",", r.getValues()))
            .collect(Collectors.toList());
    }

    @Override
    protected String join(List<String> formatted) {
        return String.join("\n", formatted);
    }

    @Override
    protected void write(String output) {
        Files.writeString(Path.of("export.csv"), output);
    }
}

public class JsonExporter extends DataExporter {
    @Override
    protected List<String> format(List<Record> records) {
        ObjectMapper mapper = new ObjectMapper();
        return records.stream()
            .map(r -> mapper.writeValueAsString(r))
            .collect(Collectors.toList());
    }

    @Override
    protected String join(List<String> formatted) {
        return "[" + String.join(",", formatted) + "]";
    }

    @Override
    protected void write(String output) {
        Files.writeString(Path.of("export.json"), output);
    }

    @Override
    protected void afterExport(List<Record> records) {
        logger.info("JSON export completed: {} records", records.size());
    }
}
```

---

## 2. ISP: The Interface Segregation Principle

```
Definition:
  "Clients should not be forced to depend on methods they do not use."

In plain terms:
  -> Interfaces should be small and focused
  -> Split "fat" interfaces into "thin" interfaces

Why it matters:
  -> Reduces the burden of implementing unnecessary methods
  -> Minimizes the impact of changes
  -> Makes mocking in tests easier
  -> Increases the cohesion of interfaces

Formal criteria for ISP:
  1. Every method in an interface should be semantically
     implementable by every implementer of that interface
  2. Every method in an interface should be needed by every
     client that uses it
  3. If either condition is not met, split the interface
```

### 2.1 Historical Background of ISP

```
1996:
  Robert C. Martin published "The Interface Segregation Principle,"
  deriving the principle from actual design issues in Xerox's printer software

Background of the problem:
  Xerox defined all features of a multifunction printer in one huge interface
  -> Each time a new type of printer was added, unnecessary methods had to be implemented
  -> Increased compilation time (C++)
  -> Bloated tests

Solution:
  Separate interfaces by functionality
  -> Each printer implements only the interfaces it needs
  -> Reduced compilation time
  -> Simplified tests

Lesson:
  -> "Fat" interfaces burden both clients and servers
  -> Smaller interfaces have higher reusability and testability
```

### 2.2 ISP Refactoring: The Device Example

```typescript
// ISP violation: a giant interface (bad)
interface SmartDevice {
  print(doc: Document): void;
  scan(): Image;
  fax(doc: Document, number: string): void;
  copy(doc: Document): Document;
  staple(doc: Document): void;
}

// A simple printer does not need fax, scan, or staple!
class SimplePrinter implements SmartDevice {
  print(doc: Document): void { /* impl */ }
  scan(): Image { throw new Error("Not supported"); } // empty impl...
  fax(): void { throw new Error("Not supported"); }   // empty impl...
  copy(): Document { throw new Error("Not supported"); }
  staple(): void { throw new Error("Not supported"); }
}

// Applying ISP: split into smaller interfaces
interface Printer {
  print(doc: Document): void;
}

interface Scanner {
  scan(): Image;
}

interface Faxer {
  fax(doc: Document, number: string): void;
}

// Implement only the needed interfaces
class SimplePrinter implements Printer {
  print(doc: Document): void { /* impl */ }
}

class MultiFunctionDevice implements Printer, Scanner, Faxer {
  print(doc: Document): void { /* impl */ }
  scan(): Image { /* impl */ return new Image(); }
  fax(doc: Document, number: string): void { /* impl */ }
}

// The caller also depends only on the interfaces it needs
function printReport(printer: Printer): void {
  // Depends only on Printer. Does not know Scanner or Faxer.
  printer.print(report);
}
```

### 2.3 Practical Example: ISP for Repositories

```typescript
// ISP violation: entire CRUD is required (bad)
interface Repository<T> {
  findAll(): Promise<T[]>;
  findById(id: string): Promise<T | null>;
  create(data: Partial<T>): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
}

// Even read-only services see the write methods

// Applying ISP: separate read and write
interface ReadRepository<T> {
  findAll(): Promise<T[]>;
  findById(id: string): Promise<T | null>;
}

interface WriteRepository<T> {
  create(data: Partial<T>): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
}

interface Repository<T> extends ReadRepository<T>, WriteRepository<T> {}

// Read-only service
class ReportService {
  constructor(private repo: ReadRepository<Order>) {}
  // Cannot access write methods = safe
}
```

### 2.4 ISP in Practice: User Management Service

```typescript
// ISP violation: a giant user service interface (bad)
interface UserService {
  // Authentication
  login(email: string, password: string): Promise<AuthToken>;
  logout(token: string): Promise<void>;
  refreshToken(token: string): Promise<AuthToken>;

  // Profile management
  getProfile(userId: string): Promise<UserProfile>;
  updateProfile(userId: string, data: Partial<UserProfile>): Promise<void>;
  uploadAvatar(userId: string, image: Buffer): Promise<string>;

  // Administrator functions
  listAllUsers(): Promise<User[]>;
  banUser(userId: string): Promise<void>;
  unbanUser(userId: string): Promise<void>;
  assignRole(userId: string, role: string): Promise<void>;

  // Notification settings
  getNotificationSettings(userId: string): Promise<NotificationSettings>;
  updateNotificationSettings(userId: string, settings: Partial<NotificationSettings>): Promise<void>;

  // Billing
  getSubscription(userId: string): Promise<Subscription>;
  updateSubscription(userId: string, plan: string): Promise<void>;
  cancelSubscription(userId: string): Promise<void>;
}

// Applying ISP: split interfaces by responsibility
interface AuthenticationService {
  login(email: string, password: string): Promise<AuthToken>;
  logout(token: string): Promise<void>;
  refreshToken(token: string): Promise<AuthToken>;
}

interface ProfileService {
  getProfile(userId: string): Promise<UserProfile>;
  updateProfile(userId: string, data: Partial<UserProfile>): Promise<void>;
  uploadAvatar(userId: string, image: Buffer): Promise<string>;
}

interface AdminService {
  listAllUsers(): Promise<User[]>;
  banUser(userId: string): Promise<void>;
  unbanUser(userId: string): Promise<void>;
  assignRole(userId: string, role: string): Promise<void>;
}

interface NotificationSettingsService {
  getNotificationSettings(userId: string): Promise<NotificationSettings>;
  updateNotificationSettings(
    userId: string,
    settings: Partial<NotificationSettings>
  ): Promise<void>;
}

interface SubscriptionService {
  getSubscription(userId: string): Promise<Subscription>;
  updateSubscription(userId: string, plan: string): Promise<void>;
  cancelSubscription(userId: string): Promise<void>;
}

// Login screen: only needs authentication
class LoginController {
  constructor(private auth: AuthenticationService) {}

  async handleLogin(email: string, password: string): Promise<AuthToken> {
    return this.auth.login(email, password);
  }
}

// User dashboard: only profile and notification settings
class DashboardController {
  constructor(
    private profile: ProfileService,
    private notifications: NotificationSettingsService,
  ) {}

  async getDashboardData(userId: string) {
    const [userProfile, notifSettings] = await Promise.all([
      this.profile.getProfile(userId),
      this.notifications.getNotificationSettings(userId),
    ]);
    return { userProfile, notifSettings };
  }
}

// Admin screen: only admin functions
class AdminController {
  constructor(private admin: AdminService) {}

  async banMaliciousUser(userId: string): Promise<void> {
    await this.admin.banUser(userId);
  }
}
```

### 2.5 ISP in Practice: Separating Event Handlers

```python
from abc import ABC, abstractmethod
from typing import Protocol
from dataclasses import dataclass
from datetime import datetime


# ISP violation: a giant listener that handles every event (bad)
class EventListener(ABC):
    @abstractmethod
    def on_user_created(self, user_id: str) -> None: ...

    @abstractmethod
    def on_user_updated(self, user_id: str) -> None: ...

    @abstractmethod
    def on_user_deleted(self, user_id: str) -> None: ...

    @abstractmethod
    def on_order_created(self, order_id: str) -> None: ...

    @abstractmethod
    def on_order_shipped(self, order_id: str) -> None: ...

    @abstractmethod
    def on_payment_received(self, payment_id: str) -> None: ...

    @abstractmethod
    def on_payment_refunded(self, payment_id: str) -> None: ...


# The email notification service only needs order events,
# but is forced to implement every method
class EmailNotificationService(EventListener):
    def on_user_created(self, user_id: str) -> None:
        pass  # not needed, but required to implement

    def on_user_updated(self, user_id: str) -> None:
        pass  # not needed, but required to implement

    def on_user_deleted(self, user_id: str) -> None:
        pass  # not needed, but required to implement

    def on_order_created(self, order_id: str) -> None:
        self._send_order_confirmation(order_id)

    def on_order_shipped(self, order_id: str) -> None:
        self._send_shipping_notification(order_id)

    def on_payment_received(self, payment_id: str) -> None:
        pass  # not needed

    def on_payment_refunded(self, payment_id: str) -> None:
        self._send_refund_notification(payment_id)


# Applying ISP: separate listeners per event
class UserEventListener(Protocol):
    def on_user_created(self, user_id: str) -> None: ...
    def on_user_updated(self, user_id: str) -> None: ...
    def on_user_deleted(self, user_id: str) -> None: ...


class OrderEventListener(Protocol):
    def on_order_created(self, order_id: str) -> None: ...
    def on_order_shipped(self, order_id: str) -> None: ...


class PaymentEventListener(Protocol):
    def on_payment_received(self, payment_id: str) -> None: ...
    def on_payment_refunded(self, payment_id: str) -> None: ...


# Email notification service: handles only the events it needs
class EmailNotificationService:
    """Handles only order and refund notifications"""

    def on_order_created(self, order_id: str) -> None:
        print(f"Sending order confirmation email: {order_id}")

    def on_order_shipped(self, order_id: str) -> None:
        print(f"Sending shipping notification email: {order_id}")

    def on_payment_refunded(self, payment_id: str) -> None:
        print(f"Sending refund notification email: {payment_id}")


# Audit log service: handles only user events
class AuditLogService:
    """Records audit logs for user operations"""

    def on_user_created(self, user_id: str) -> None:
        print(f"Audit log: user created {user_id}")

    def on_user_updated(self, user_id: str) -> None:
        print(f"Audit log: user updated {user_id}")

    def on_user_deleted(self, user_id: str) -> None:
        print(f"Audit log: user deleted {user_id}")


# Event bus: register listeners with the appropriate interface
class EventBus:
    def __init__(self):
        self._user_listeners: list[UserEventListener] = []
        self._order_listeners: list[OrderEventListener] = []
        self._payment_listeners: list[PaymentEventListener] = []

    def register_user_listener(self, listener: UserEventListener) -> None:
        self._user_listeners.append(listener)

    def register_order_listener(self, listener: OrderEventListener) -> None:
        self._order_listeners.append(listener)

    def register_payment_listener(self, listener: PaymentEventListener) -> None:
        self._payment_listeners.append(listener)

    def emit_order_created(self, order_id: str) -> None:
        for listener in self._order_listeners:
            listener.on_order_created(order_id)

    def emit_user_created(self, user_id: str) -> None:
        for listener in self._user_listeners:
            listener.on_user_created(user_id)


# Usage
bus = EventBus()
bus.register_order_listener(EmailNotificationService())
bus.register_user_listener(AuditLogService())
```

### 2.6 ISP in Practice: File System Operations

```go
package filesystem

import "io"

// ISP violation: a giant filesystem interface (bad)
type FileSystem interface {
    Read(path string) ([]byte, error)
    Write(path string, data []byte) error
    Delete(path string) error
    Rename(old, new string) error
    List(dir string) ([]string, error)
    Mkdir(path string) error
    MkdirAll(path string) error
    Chmod(path string, mode int) error
    Chown(path string, uid, gid int) error
    Stat(path string) (FileInfo, error)
    Symlink(oldname, newname string) error
    ReadLink(path string) (string, error)
    Watch(path string, callback func(Event)) error
}

// Applying ISP: separate by responsibility
type FileReader interface {
    Read(path string) ([]byte, error)
    Stat(path string) (FileInfo, error)
}

type FileWriter interface {
    Write(path string, data []byte) error
    Mkdir(path string) error
    MkdirAll(path string) error
}

type FileDeleter interface {
    Delete(path string) error
}

type DirectoryLister interface {
    List(dir string) ([]string, error)
}

type FilePermissions interface {
    Chmod(path string, mode int) error
    Chown(path string, uid, gid int) error
}

type FileWatcher interface {
    Watch(path string, callback func(Event)) error
}

// Read-only backup service
type BackupService struct {
    reader FileReader
    lister DirectoryLister
}

func NewBackupService(reader FileReader, lister DirectoryLister) *BackupService {
    return &BackupService{reader: reader, lister: lister}
}

func (s *BackupService) BackupDirectory(dir string) ([]BackupEntry, error) {
    files, err := s.lister.List(dir)
    if err != nil {
        return nil, err
    }

    var entries []BackupEntry
    for _, file := range files {
        data, err := s.reader.Read(file)
        if err != nil {
            return nil, err
        }
        info, _ := s.reader.Stat(file)
        entries = append(entries, BackupEntry{
            Path:    file,
            Data:    data,
            Size:    info.Size(),
            ModTime: info.ModTime(),
        })
    }
    return entries, nil
}

// Deploy service requiring write permissions
type DeployService struct {
    reader  FileReader
    writer  FileWriter
    deleter FileDeleter
}

func NewDeployService(
    reader FileReader,
    writer FileWriter,
    deleter FileDeleter,
) *DeployService {
    return &DeployService{reader, writer, deleter}
}
```

---

## 3. The Relationship Between LSP and ISP

```
LSP: guarantees the correctness of subtypes
  -> "Can this class truly substitute for the parent?"
  -> If not -> the interface design is wrong

ISP: optimizes interface granularity
  -> "Is this interface too fine? Too fat?"
  -> Unnecessary methods exist -> split them

LSP violations can often be resolved via ISP:
  Penguin cannot implement Bird.fly()
  -> The Bird interface is too fat
  -> Split into Movable and Flyable (ISP)
  -> Penguin implements only Movable (LSP compliant)

Diagram of the relationship:

  ISP violation            LSP violation
    |                         |
    v                         v
  Fat interfaces  ->  Empty implementations / thrown exceptions required
    |                         |
    v                         v
  Split via ISP  ->  Naturally becomes LSP compliant

In other words:
  ISP plays the role of "preventing" LSP violations.
  Properly segregated interfaces naturally lead to
  designs where LSP violations are less likely to occur.
```

### 3.1 An Integrated Design Example of LSP + ISP

```typescript
// A practical example: a payment system

// === Step 1: Design interfaces with appropriate granularity via ISP ===

interface ChargeablePayment {
  charge(amount: number): Promise<PaymentResult>;
  getChargeLimit(): number;
}

interface RefundablePayment {
  refund(transactionId: string, amount: number): Promise<RefundResult>;
  getRefundPolicy(): RefundPolicy;
}

interface RecurringPayment {
  setupRecurring(interval: string, amount: number): Promise<SubscriptionId>;
  cancelRecurring(subscriptionId: string): Promise<void>;
}

interface PaymentInfoProvider {
  getLastFourDigits(): string;
  getExpirationDate(): string;
  getPaymentType(): string;
}

// === Step 2: Each payment method is implemented in an LSP-compliant way ===

class CreditCardPayment implements
  ChargeablePayment,
  RefundablePayment,
  RecurringPayment,
  PaymentInfoProvider
{
  constructor(
    private cardNumber: string,
    private expiry: string,
    private cvv: string,
  ) {}

  async charge(amount: number): Promise<PaymentResult> {
    // Credit card charge implementation
    return { success: true, transactionId: "cc_" + Date.now() };
  }

  getChargeLimit(): number {
    return 1000000; // 1,000,000 JPY
  }

  async refund(transactionId: string, amount: number): Promise<RefundResult> {
    return { success: true, refundId: "ref_" + Date.now() };
  }

  getRefundPolicy(): RefundPolicy {
    return { maxDays: 30, partialAllowed: true };
  }

  async setupRecurring(interval: string, amount: number): Promise<string> {
    return "sub_" + Date.now();
  }

  async cancelRecurring(subscriptionId: string): Promise<void> {
    // Subscription cancellation
  }

  getLastFourDigits(): string {
    return this.cardNumber.slice(-4);
  }

  getExpirationDate(): string {
    return this.expiry;
  }

  getPaymentType(): string {
    return "credit_card";
  }
}

class BankTransferPayment implements ChargeablePayment, PaymentInfoProvider {
  // Bank transfer: only charges and info provision
  // Does not implement RefundablePayment or RecurringPayment -> ISP compliant
  // -> Not forced to provide empty implementations for "cannot refund" methods -> LSP compliant

  constructor(
    private bankCode: string,
    private accountNumber: string,
  ) {}

  async charge(amount: number): Promise<PaymentResult> {
    return { success: true, transactionId: "bt_" + Date.now() };
  }

  getChargeLimit(): number {
    return 5000000; // 5,000,000 JPY (bank transfers have a high limit)
  }

  getLastFourDigits(): string {
    return this.accountNumber.slice(-4);
  }

  getExpirationDate(): string {
    return "N/A"; // Bank accounts do not expire
  }

  getPaymentType(): string {
    return "bank_transfer";
  }
}

class ConvenienceStorePayment implements ChargeablePayment {
  // Convenience store payment: charges only
  // No refunds, no recurring, no card info

  async charge(amount: number): Promise<PaymentResult> {
    if (amount > 300000) {
      return { success: false, error: "The limit for convenience store payments is 300,000 JPY" };
    }
    return { success: true, transactionId: "cvs_" + Date.now() };
  }

  getChargeLimit(): number {
    return 300000; // 300,000 JPY
  }
}

// === Step 3: Callers depend only on the interfaces they need ===

class CheckoutService {
  // Only needs charging
  async processPayment(
    payment: ChargeablePayment,
    amount: number,
  ): Promise<PaymentResult> {
    const limit = payment.getChargeLimit();
    if (amount > limit) {
      return { success: false, error: `Exceeds payment limit (${limit})` };
    }
    return payment.charge(amount);
  }
}

class RefundService {
  // Only refundable payment methods
  async processRefund(
    payment: RefundablePayment,
    transactionId: string,
    amount: number,
  ): Promise<RefundResult> {
    const policy = payment.getRefundPolicy();
    // Refund processing based on the policy
    return payment.refund(transactionId, amount);
  }
}

class SubscriptionService {
  // Only payment methods that support recurring
  async createSubscription(
    payment: RecurringPayment,
    plan: { interval: string; amount: number },
  ): Promise<string> {
    return payment.setupRecurring(plan.interval, plan.amount);
  }
}
```

---

## 4. Decision Criteria

```
LSP checklist:
  [ ] Does the subclass correctly implement all of the parent's methods?
  [ ] Are there any empty implementations or exception throws (UnsupportedOperation)?
  [ ] Is instanceof type checking unnecessary?
  [ ] Do the parent class's tests also pass for the subclass?
  [ ] Are preconditions not strengthened?
  [ ] Are postconditions not weakened?
  [ ] Are invariants maintained?
  [ ] Are return types covariant? (Returning a subtype is OK)
  [ ] Are parameter types contravariant? (Accepting a supertype is OK)

ISP checklist:
  [ ] Do implementers of the interface use all its methods?
  [ ] Do clients of the interface need every method?
  [ ] Does the interface have five methods or fewer?
  [ ] Does the interface have high cohesion?
  [ ] Does it have only one reason to change? (An SRP-style lens)
  [ ] Is the interface's name specific? ("Service" is too broad)
  [ ] Are mocks easy to create?

Practical guidelines for splitting:
  1. "When implementing this interface,
      can I write meaningful implementations for every method?"
     -> If not -> a split is needed

  2. "When using this interface,
      do I need every method?"
     -> Some are unnecessary -> a split is needed

  3. "Is the change frequency of this interface
      the same across all methods?"
     -> If not -> split by change frequency
```

### 4.1 Avoiding Excessive Segregation

```
Pitfall of ISP: Over-Segregation

Too far (bad):
  interface Readable { read(): string; }
  interface Writable { write(data: string): void; }
  interface Closable { close(): void; }
  interface Flushable { flush(): void; }
  interface Seekable { seek(position: number): void; }
  interface Positionable { getPosition(): number; }
  interface Sizeable { getSize(): number; }
  // ... 7 interfaces; inconvenient to use

Appropriate granularity (good):
  interface ReadableStream {
    read(): string;
    getPosition(): number;
    getSize(): number;
  }

  interface WritableStream {
    write(data: string): void;
    flush(): void;
    getPosition(): number;
  }

  interface Closable {
    close(): void;
  }

  // Three interfaces are enough

A rule of thumb:
  -> Put "methods used together" in the same interface
  -> Put "methods used by different clients" in different interfaces
  -> Be mindful of cohesion
  -> Work backward from actual client use cases
```

---

## 5. LSP and ISP in Testing

```typescript
// LSP-compliant tests: parent class tests pass for subclasses

// Common test pattern
abstract class CollectionTestBase<T extends Collection<number>> {
  abstract createCollection(): T;

  testAddAndContains(): void {
    const collection = this.createCollection();
    collection.add(42);
    assert(collection.contains(42), "added element should be present");
  }

  testRemove(): void {
    const collection = this.createCollection();
    collection.add(42);
    collection.remove(42);
    assert(!collection.contains(42), "removed element should not be present");
  }

  testSize(): void {
    const collection = this.createCollection();
    assert(collection.size() === 0, "initial size is 0");
    collection.add(1);
    collection.add(2);
    assert(collection.size() === 2, "size is 2 after adding two");
  }
}

// LSP compliant: all subclasses pass the same tests
class ArrayListTest extends CollectionTestBase<ArrayList<number>> {
  createCollection() { return new ArrayList<number>(); }
}

class LinkedListTest extends CollectionTestBase<LinkedList<number>> {
  createCollection() { return new LinkedList<number>(); }
}

class HashSetTest extends CollectionTestBase<HashSet<number>> {
  createCollection() { return new HashSet<number>(); }
}

// ISP and mocking: smaller interfaces are easier to mock
// Fat interface: mock creation is tedious
const mockFullRepository: jest.Mocked<Repository<User>> = {
  findAll: jest.fn(),
  findById: jest.fn(),
  create: jest.fn(),
  update: jest.fn(),
  delete: jest.fn(),
  count: jest.fn(),
  findByEmail: jest.fn(),
  search: jest.fn(),
  // ... many methods to mock
};

// Applying ISP: mock only the needed methods
const mockReader: jest.Mocked<ReadRepository<User>> = {
  findAll: jest.fn().mockResolvedValue([]),
  findById: jest.fn().mockResolvedValue(null),
};

// The test is concise and its intent is clear
describe("ReportService", () => {
  it("should generate report from all users", async () => {
    mockReader.findAll.mockResolvedValue([
      { id: "1", name: "Tanaka" },
      { id: "2", name: "Sato" },
    ]);

    const service = new ReportService(mockReader);
    const report = await service.generate();

    expect(mockReader.findAll).toHaveBeenCalled();
    expect(report.userCount).toBe(2);
  });
});
```

```python
# Python: LSP-compliant tests with pytest
import pytest
from abc import ABC, abstractmethod


class ShapeTestBase(ABC):
    """Base test class for Shape LSP tests"""

    @abstractmethod
    def create_shape(self) -> "Shape":
        """Return the Shape instance under test"""
        ...

    def test_area_is_non_negative(self):
        """Area is always non-negative"""
        shape = self.create_shape()
        assert shape.area() >= 0

    def test_area_is_numeric(self):
        """Area returns a numeric value"""
        shape = self.create_shape()
        assert isinstance(shape.area(), (int, float))

    def test_perimeter_is_non_negative(self):
        """Perimeter is always non-negative"""
        shape = self.create_shape()
        assert shape.perimeter() >= 0

    def test_string_representation(self):
        """String representation is non-empty"""
        shape = self.create_shape()
        assert len(str(shape)) > 0


class TestCircle(ShapeTestBase):
    def create_shape(self):
        return Circle(radius=5)

    def test_circle_specific_area(self):
        circle = Circle(radius=10)
        assert abs(circle.area() - 314.159) < 0.01


class TestRectangle(ShapeTestBase):
    def create_shape(self):
        return Rectangle(width=4, height=5)

    def test_rectangle_specific_area(self):
        rect = Rectangle(width=3, height=7)
        assert rect.area() == 21


class TestTriangle(ShapeTestBase):
    def create_shape(self):
        return Triangle(base=6, height=4)

    def test_triangle_specific_area(self):
        tri = Triangle(base=10, height=5)
        assert tri.area() == 25


# All subclasses pass ShapeTestBase's tests = LSP compliant
```

---

## 6. Real-World Anti-Patterns and How to Address Them

### 6.1 The "Interface Everything" Anti-Pattern

```typescript
// Anti-pattern: forcibly creating an interface when there is only one implementation (bad)
interface IUserService {
  getUser(id: string): Promise<User>;
}

class UserService implements IUserService {
  getUser(id: string): Promise<User> { /* ... */ }
}

// -> A proliferation of interfaces with the "I" prefix...
// -> With only one implementation, the benefits of ISP are slim
// -> Code navigation becomes difficult

// Better: create interfaces only when truly necessary
// Interfaces are useful in the following situations:
// 1. Multiple implementations exist (production, testing, development)
// 2. You want to abstract dependencies on external services
// 3. Mocks are needed for testing

// External API dependency: interface is useful
interface PaymentGateway {
  charge(amount: number, token: string): Promise<ChargeResult>;
}

class StripeGateway implements PaymentGateway {
  async charge(amount: number, token: string): Promise<ChargeResult> {
    // Stripe API call
  }
}

class MockPaymentGateway implements PaymentGateway {
  async charge(amount: number, token: string): Promise<ChargeResult> {
    return { success: true, id: "mock_" + Date.now() };
  }
}
```

### 6.2 The "Header Interface" Anti-Pattern

```java
// Header Interface (bad): turn all public methods of a class into an interface
public interface IOrderService {
    Order createOrder(CreateOrderDto dto);
    Order getOrder(String id);
    List<Order> getOrdersByUser(String userId);
    void cancelOrder(String id);
    void updateOrderStatus(String id, OrderStatus status);
    OrderReport generateReport(DateRange range);
    void sendOrderConfirmation(String orderId);
    List<Order> searchOrders(OrderSearchCriteria criteria);
    void archiveOldOrders(int daysOld);
    OrderStats getStatistics(DateRange range);
}

// -> Just a copy of every method from the class
// -> Violates the spirit of ISP (clients do not use every method)

// Design the interface from the client's perspective
// Order creation and management
public interface OrderManagement {
    Order createOrder(CreateOrderDto dto);
    void cancelOrder(String id);
    void updateOrderStatus(String id, OrderStatus status);
}

// Order search and retrieval
public interface OrderQuery {
    Order getOrder(String id);
    List<Order> getOrdersByUser(String userId);
    List<Order> searchOrders(OrderSearchCriteria criteria);
}

// Reporting and analytics
public interface OrderReporting {
    OrderReport generateReport(DateRange range);
    OrderStats getStatistics(DateRange range);
}

// Administrator operations
public interface OrderAdministration {
    void archiveOldOrders(int daysOld);
}

// Notifications
public interface OrderNotification {
    void sendOrderConfirmation(String orderId);
}
```

### 6.3 Language-Specific ISP Implementation Techniques

```python
# Python: implementing ISP with Protocol
from typing import Protocol, runtime_checkable


@runtime_checkable
class Drawable(Protocol):
    """A drawable object"""
    def draw(self, canvas: "Canvas") -> None: ...


@runtime_checkable
class Resizable(Protocol):
    """A resizable object"""
    def resize(self, factor: float) -> None: ...


@runtime_checkable
class Movable(Protocol):
    """A movable object"""
    def move(self, dx: float, dy: float) -> None: ...


@runtime_checkable
class Rotatable(Protocol):
    """A rotatable object"""
    def rotate(self, angle: float) -> None: ...


class Circle:
    """Circle: supports all operations"""
    def __init__(self, x: float, y: float, radius: float):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, canvas: "Canvas") -> None:
        canvas.draw_circle(self.x, self.y, self.radius)

    def resize(self, factor: float) -> None:
        self.radius *= factor

    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    def rotate(self, angle: float) -> None:
        pass  # rotation does not change a circle (a valid implementation)


class TextLabel:
    """Text label: drawing and moving only"""
    def __init__(self, x: float, y: float, text: str):
        self.x = x
        self.y = y
        self.text = text

    def draw(self, canvas: "Canvas") -> None:
        canvas.draw_text(self.x, self.y, self.text)

    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    # Do not implement resize() or rotate() -> ISP compliant


# Caller side: specify only the needed protocols as type hints
def draw_all(items: list[Drawable]) -> None:
    for item in items:
        item.draw(canvas)

def resize_all(items: list[Resizable], factor: float) -> None:
    for item in items:
        item.resize(factor)

def move_all(items: list[Movable], dx: float, dy: float) -> None:
    for item in items:
        item.move(dx, dy)


# Type checking: Protocols work with isinstance too
circle = Circle(0, 0, 10)
label = TextLabel(0, 0, "Hello")

assert isinstance(circle, Drawable)   # True
assert isinstance(circle, Resizable)  # True
assert isinstance(label, Drawable)    # True
assert isinstance(label, Resizable)   # False — safe thanks to ISP
```

```rust
// Rust: ISP is achieved naturally through traits
trait Drawable {
    fn draw(&self, canvas: &mut Canvas);
}

trait Resizable {
    fn resize(&mut self, factor: f64);
}

trait Movable {
    fn move_by(&mut self, dx: f64, dy: f64);
}

trait Rotatable {
    fn rotate(&mut self, angle: f64);
}

struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

// Implement only the needed traits
impl Drawable for Circle {
    fn draw(&self, canvas: &mut Canvas) {
        canvas.draw_circle(self.x, self.y, self.radius);
    }
}

impl Resizable for Circle {
    fn resize(&mut self, factor: f64) {
        self.radius *= factor;
    }
}

impl Movable for Circle {
    fn move_by(&mut self, dx: f64, dy: f64) {
        self.x += dx;
        self.y += dy;
    }
}

struct TextLabel {
    x: f64,
    y: f64,
    text: String,
}

// TextLabel only has Drawable and Movable
impl Drawable for TextLabel {
    fn draw(&self, canvas: &mut Canvas) {
        canvas.draw_text(self.x, self.y, &self.text);
    }
}

impl Movable for TextLabel {
    fn move_by(&mut self, dx: f64, dy: f64) {
        self.x += dx;
        self.y += dy;
    }
}

// Specify required capabilities via trait bounds
fn draw_all(items: &[&dyn Drawable]) {
    for item in items {
        item.draw(&mut canvas);
    }
}

fn resize_all(items: &mut [&mut dyn Resizable], factor: f64) {
    for item in items {
        item.resize(factor);
    }
}

// Combining multiple trait bounds
fn interactive_element<T: Drawable + Movable + Resizable>(element: &mut T) {
    element.draw(&mut canvas);
    element.move_by(10.0, 20.0);
    element.resize(1.5);
    element.draw(&mut canvas);
}
```

---

## 7. Relationships with Other SOLID Principles

```
LSP and other principles:

  SRP <-> LSP:
    SRP violations (multiple responsibilities) -> more prone to LSP violations
    Example: inheriting a class that combines "data persistence" and "notification"
        causes LSP violations in subclasses where one of those responsibilities is unnecessary

  OCP <-> LSP:
    LSP compliance -> easier OCP compliance
    Existing code does not break when new subtypes are added

  ISP <-> LSP:
    ISP compliance -> fewer LSP violations (as discussed above)

  DIP <-> ISP:
    Depend on ISP-segregated interfaces (DIP)
    -> Naturally yields a loosely coupled architecture

ISP and other principles:

  SRP <-> ISP:
    SRP: single responsibility at the class level
    ISP: single responsibility at the interface level
    The same "separation of responsibilities" applied at different levels

  OCP <-> ISP:
    Segregation via ISP -> limited scope of change impact -> contributes to OCP

  DIP <-> ISP:
    ISP gives interfaces appropriate granularity -> optimizes the abstraction layer of DIP
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Beyond theory, actually writing code and observing its behavior deepens your understanding.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the foundational concepts explained in this guide before moving on to the next step.

### Q3: How is this used in real-world practice?

The knowledge on this topic is frequently used in day-to-day development work. It is especially important during code reviews and architectural design.

---

## Summary

| Principle | Core idea | Signs of violation | Solution |
|------|------|------------|--------|
| LSP | Substitutability | Empty implementations, instanceof, added exceptions | Redesign the interface |
| ISP | Appropriate granularity | Unnecessary methods, bloated interfaces | Split the interface |

```
Practical summary of LSP + ISP:

  1. When designing interfaces:
     -> ISP perspective: keep them small and focused
     -> "Can every implementer provide a meaningful implementation for every method?"

  2. When designing inheritance relationships:
     -> LSP perspective: guarantee substitutability
     -> "Do the parent's tests pass for the subclass?"

  3. When refactoring:
     -> Find an LSP violation -> consider splitting via ISP
     -> Find an ISP violation -> check for LSP violations at the same time

  4. When testing:
     -> LSP: re-run parent class tests against subclasses
     -> ISP: judge granularity by how easy it is to create mocks
```

---

## Recommended Next Guides

---

## References
1. Liskov, B. "Data Abstraction and Hierarchy." OOPSLA, 1987.
2. Liskov, B. and Wing, J. "A Behavioral Notion of Subtyping." ACM Transactions on Programming Languages and Systems, 1994.
3. Martin, R. "The Interface Segregation Principle." 1996.
4. Martin, R. "Agile Software Development: Principles, Patterns, and Practices." Prentice Hall, 2002.
5. Meyer, B. "Object-Oriented Software Construction." Prentice Hall, 1997.
6. Bloch, J. "Effective Java." 3rd Edition, Addison-Wesley, 2018.
