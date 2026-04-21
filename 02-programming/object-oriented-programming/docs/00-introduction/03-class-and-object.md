# Classes and Objects

> A class is a "blueprint," and an object is an "instance." Gain a deep understanding of this relationship, how objects are laid out in memory, constructor design patterns, and the proper use of static members.

## What You Will Learn

- [ ] Understand the relationship between classes and objects at the memory level
- [ ] Grasp the design patterns for constructors
- [ ] Learn when to use static members versus instance members
- [ ] Understand the object lifecycle (creation, use, destruction)
- [ ] Grasp the difference between value types and reference types, and copy semantics
- [ ] Be able to apply best practices for class design


## Prerequisites

You will gain a deeper understanding of this guide if you have the following knowledge beforehand:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of the content in [OOP vs Other Paradigms](./02-oop-vs-other-paradigms.md)

---

## 1. The Relationship Between Classes and Objects

```
Class = Blueprint / Type
  -> Exists only once in memory (metadata)
  -> Holds field definitions and method implementations

Object = Instance / Entity
  -> Multiple can exist in memory
  -> Each object has its own state (field values)

  +----------- Class: User -----------+
  | Blueprint (metadata)              |
  |   fields: name, age, email        |
  |   methods: greet(), isAdult()     |
  +-----------------------------------+
           | new User(...)
     +-----+-----+-------------+
     v           v             v
  +--------+ +--------+ +--------+
  | obj_1  | | obj_2  | | obj_3  |
  | Tanaka | | Yamada | | Sato   |
  | age 25 | | age 30 | | age 17 |
  +--------+ +--------+ +--------+
  0x1000      0x2000      0x3000
  Each object has its own dedicated memory region
```

### 1.1 The Three Roles of a Class

A class is not merely a definition of a data structure; it plays three important roles.

```
1. Role as a Type:
   -> Defines the type of a variable
   -> Enables compile-time type checking
   -> Expresses the contract of an interface

2. Role as a Template:
   -> Blueprint for object creation
   -> Defines the layout of fields
   -> Holds method implementations

3. Role as a Module:
   -> Encapsulates related functionality
   -> Provides a namespace
   -> Sets the boundary for access control
```

```typescript
// TypeScript: Example showing the three roles of a class

// 1. As a type: can be used as a variable type annotation
class Product {
  constructor(
    public readonly id: string,
    public readonly name: string,
    public readonly price: number,
    public readonly category: string,
  ) {}

  // 2. As a template: defines fields and methods
  getDisplayPrice(): string {
    return `${this.price.toLocaleString()} yen`;
  }

  isInCategory(category: string): boolean {
    return this.category === category;
  }

  // 3. As a module: groups related functionality
  applyDiscount(rate: number): Product {
    if (rate < 0 || rate > 1) {
      throw new Error("The discount rate must be in the range 0 to 1");
    }
    return new Product(
      this.id,
      this.name,
      Math.floor(this.price * (1 - rate)),
      this.category,
    );
  }
}

// Used as a type
function findExpensiveProducts(products: Product[], threshold: number): Product[] {
  return products.filter(p => p.price >= threshold);
}

// Used as a template (instance creation)
const laptop = new Product("P001", "MacBook Pro", 298000, "Electronics");
const phone = new Product("P002", "iPhone 15", 149800, "Electronics");

// Used as a module (instruct the object)
const discountedLaptop = laptop.applyDiscount(0.1);
console.log(discountedLaptop.getDisplayPrice()); // "268,200 yen"
```

### 1.2 The Three Characteristics of an Object

Every object has three characteristics: "state," "behavior," and "identity."

```python
# Python: The three characteristics of an object

class Employee:
    """Employee class: has state, behavior, and identity"""

    _next_id = 1

    def __init__(self, name: str, department: str, salary: int):
        # Identity: uniquely identifies each object
        self._id = Employee._next_id
        Employee._next_id += 1

        # State: data unique to the object
        self._name = name
        self._department = department
        self._salary = salary
        self._is_active = True

    # Behavior: operations the object can perform

    def promote(self, raise_amount: int) -> None:
        """Promote: increase the salary"""
        if not self._is_active:
            raise RuntimeError(f"{self._name} has already resigned")
        if raise_amount <= 0:
            raise ValueError("The raise amount must be a positive number")
        self._salary += raise_amount

    def transfer(self, new_department: str) -> None:
        """Transfer to a different department"""
        if not self._is_active:
            raise RuntimeError(f"{self._name} has already resigned")
        old = self._department
        self._department = new_department
        print(f"{self._name}: {old} -> {new_department}")

    def resign(self) -> None:
        """Process resignation"""
        self._is_active = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def employee_id(self) -> int:
        return self._id

    def __repr__(self) -> str:
        status = "active" if self._is_active else "resigned"
        return (
            f"Employee(id={self._id}, name='{self._name}', "
            f"dept='{self._department}', salary={self._salary}, status={status})"
        )

    def __eq__(self, other: object) -> bool:
        """Identity comparison: same person if the ID is the same"""
        if not isinstance(other, Employee):
            return NotImplemented
        return self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)


# Usage example
emp1 = Employee("Taro Tanaka", "Development", 500000)
emp2 = Employee("Hanako Yamada", "Sales", 450000)

# Checking state
print(emp1)  # Employee(id=1, name='Taro Tanaka', dept='Development', salary=500000, status=active)

# Executing behavior
emp1.promote(50000)
emp1.transfer("Engineering")  # Taro Tanaka: Development -> Engineering

# Checking identity
emp1_copy = emp1  # Copy of the reference
print(emp1 is emp1_copy)   # True (same object)
print(emp1 == emp1_copy)   # True (same ID)
print(emp1 == emp2)        # False (different IDs)
```

### 1.3 The Class-Object Relationship in Various Languages

```java
// Java: the basic relationship between classes and objects

public class Book {
    // Fields (definition of state)
    private final String isbn;
    private final String title;
    private final String author;
    private int stock;
    private boolean isAvailable;

    // Constructor (initialization)
    public Book(String isbn, String title, String author, int stock) {
        this.isbn = isbn;
        this.title = title;
        this.author = author;
        this.stock = stock;
        this.isAvailable = stock > 0;
    }

    // Methods (definition of behavior)
    public boolean borrow() {
        if (stock <= 0) {
            return false;
        }
        stock--;
        isAvailable = stock > 0;
        return true;
    }

    public void returnBook() {
        stock++;
        isAvailable = true;
    }

    public String getInfo() {
        return String.format(
            "'%s' (%s) by %s [stock: %d]",
            title, isbn, author, stock
        );
    }

    // Identity: determined by ISBN
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Book other)) return false;
        return isbn.equals(other.isbn);
    }

    @Override
    public int hashCode() {
        return isbn.hashCode();
    }
}

// Creating objects from the class
Book book1 = new Book("978-4-xxx", "Design Patterns", "GoF", 3);
Book book2 = new Book("978-4-yyy", "Refactoring", "Fowler", 2);

// Each object has independent state
book1.borrow();  // Only book1's stock decreases
System.out.println(book1.getInfo()); // stock: 2
System.out.println(book2.getInfo()); // stock: 2 (unaffected)
```

```kotlin
// Kotlin: a more concise class definition

class BankAccount(
    val accountNumber: String,
    val owner: String,
    initialBalance: Long = 0
) {
    var balance: Long = initialBalance
        private set  // setter is private

    private val transactions = mutableListOf<String>()

    fun deposit(amount: Long): BankAccount {
        require(amount > 0) { "The deposit amount must be a positive number" }
        balance += amount
        transactions.add("Deposit: +${amount} yen")
        return this
    }

    fun withdraw(amount: Long): BankAccount {
        require(amount > 0) { "The withdrawal amount must be a positive number" }
        require(balance >= amount) { "Insufficient balance (balance: ${balance} yen, withdrawal: ${amount} yen)" }
        balance -= amount
        transactions.add("Withdrawal: -${amount} yen")
        return this
    }

    fun getStatement(): String {
        val header = "=== Account Statement ===\nAccount Number: $accountNumber\nOwner: $owner\n"
        val body = transactions.joinToString("\n")
        val footer = "\nBalance: ${balance} yen"
        return header + body + footer
    }
}

// Object creation and use
val account = BankAccount("1234-5678", "Taro Tanaka", 100000)
account.deposit(50000).withdraw(30000)
println(account.getStatement())
// === Account Statement ===
// Account Number: 1234-5678
// Owner: Taro Tanaka
// Deposit: +50000 yen
// Withdrawal: -30000 yen
// Balance: 120000 yen
```

```rust
// Rust: representing the equivalent of a class using structs and impl blocks

use std::fmt;

struct Rectangle {
    width: f64,
    height: f64,
}

impl Rectangle {
    // Associated function (equivalent to a constructor)
    fn new(width: f64, height: f64) -> Self {
        assert!(width > 0.0, "Width must be a positive number");
        assert!(height > 0.0, "Height must be a positive number");
        Rectangle { width, height }
    }

    fn square(size: f64) -> Self {
        Rectangle::new(size, size)
    }

    // Method (&self means immutable borrow)
    fn area(&self) -> f64 {
        self.width * self.height
    }

    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }

    fn is_square(&self) -> bool {
        (self.width - self.height).abs() < f64::EPSILON
    }

    // &mut self for mutable borrow
    fn scale(&mut self, factor: f64) {
        assert!(factor > 0.0, "The scale factor must be a positive number");
        self.width *= factor;
        self.height *= factor;
    }

    // Consumes self and returns a new value
    fn rotate(self) -> Rectangle {
        Rectangle::new(self.height, self.width)
    }
}

impl fmt::Display for Rectangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rectangle({}x{}, area={})", self.width, self.height, self.area())
    }
}

fn main() {
    let mut rect = Rectangle::new(10.0, 5.0);
    println!("{}", rect);  // Rectangle(10x5, area=50)

    rect.scale(2.0);
    println!("{}", rect);  // Rectangle(20x10, area=200)

    let rotated = rect.rotate(); // rect is moved
    println!("{}", rotated); // Rectangle(10x20, area=200)
    // println!("{}", rect);  // Compile error: rect has been moved
}
```

---

## 2. Memory Layout

```
Memory model of Java/C#:

  Stack                     Heap
  +------------+            +---------------------+
  | user1 ref -+----------->| User object          |
  |  (8 bytes) |            | +-------------------+|
  +------------+            | | header (16 bytes) ||
  | user2 ref -+------+     | | class pointer     ||
  |  (8 bytes) |      |     | | hash code         ||
  +------------+      |     | | lock info         ||
                      |     | +-------------------+|
                      |     | | name: "Tanaka" ref|+-> String object
                      |     | | age: 25           ||
                      |     | | email: ref        ||-> String object
                      |     | +-------------------+|
                      |     +---------------------+
                      +---->| User object          |
                           | name: "Yamada", age:30|
                           +---------------------+

C++ memory model:
  -> Can be placed on either the stack or the heap
  User user1("Tanaka", 25);           // On the stack
  User* user2 = new User("Yamada", 30); // On the heap

Python memory model:
  -> Everything is an object on the heap
  -> All variables are references (pointers)
```

### 2.1 Memory Layout Details

The memory layout of objects varies significantly depending on each language's runtime.

```
Java object header (64-bit JVM, with Compressed Oops enabled):

  +-----------------------------------------+
  | Mark Word (8 bytes)                     |
  |   - Hash code (31 bits)                 |
  |   - GC age (4 bits)                     |
  |   - Lock info (2 bits)                  |
  |   - Biased lock info (1 bit)            |
  +-----------------------------------------+
  | Class Pointer (4 bytes, compressed)     |
  |   -> Pointer to the method table (vtable)|
  +-----------------------------------------+
  | Padding (4 bytes)                       |
  |   -> For 8-byte alignment               |
  +-----------------------------------------+
  | Instance Fields                         |
  |   -> Field values (primitive or reference)|
  +-----------------------------------------+

  Total: 16 bytes for header + fields

  Example: class Point { int x; int y; }
  -> 16 (header) + 4 (x) + 4 (y) = 24 bytes

  Example: class User { String name; int age; String email; }
  -> 16 (header) + 4 (name ref) + 4 (age) + 4 (email ref) + 4 (padding) = 32 bytes

Field reordering:
  The JVM reorders fields for memory efficiency:
  1. double / long  (8 bytes)
  2. int / float    (4 bytes)
  3. short / char   (2 bytes)
  4. byte / boolean (1 byte)
  5. Reference types (4 bytes, Compressed Oops)
```

```java
// Java: estimating and optimizing object size

// Bad example: wasteful memory usage
public class WastefulObject {
    private boolean flag1;    // 1 byte -> 8 bytes (padding)
    private long value;       // 8 bytes
    private boolean flag2;    // 1 byte -> 8 bytes (padding)
    private long timestamp;   // 8 bytes
    // header 16 + 32 = 48 bytes

    // JVM reorders and optimizes:
    // long value;      -> 8 bytes
    // long timestamp;  -> 8 bytes
    // boolean flag1;   -> 1 byte
    // boolean flag2;   -> 1 byte + 6 bytes padding
    // Actual: header 16 + 24 = 40 bytes
}

// Design mindful of memory efficiency
public class EfficientObject {
    private long value;       // 8 bytes
    private long timestamp;   // 8 bytes
    private int count;        // 4 bytes
    private short type;       // 2 bytes
    private boolean flag1;    // 1 byte
    private boolean flag2;    // 1 byte
    // header 16 + 24 = 40 bytes (minimal padding)
}
```

### 2.2 The Method Table (vtable)

```
Where methods are stored:

  Methods are not copied per object.
  They exist only once in the class's metadata region, shared by all instances.

  +------- Class Metadata: Animal ---------+
  | vtable (Virtual Method Table):          |
  |   [0] speak()  -> 0x4000 (Animal.speak)|
  |   [1] move()   -> 0x4100 (Animal.move) |
  |   [2] eat()    -> 0x4200 (Animal.eat)  |
  +-----------------------------------------+

  +------- Class Metadata: Dog ------------+
  | vtable (Virtual Method Table):          |
  |   [0] speak()  -> 0x5000 (Dog.speak)   |  <- Override
  |   [1] move()   -> 0x4100 (Animal.move) |  <- Inherited
  |   [2] eat()    -> 0x5200 (Dog.eat)     |  <- Override
  |   [3] fetch()  -> 0x5300 (Dog.fetch)   |  <- Newly added
  +-----------------------------------------+

  Animal animal = new Dog();
  animal.speak();
  -> animal's class pointer -> Dog's metadata -> vtable[0] -> Dog.speak()
  -> This is the mechanism of "dynamic dispatch"
```

```typescript
// TypeScript: method sharing via the prototype chain

class Animal {
  constructor(public name: string) {}

  speak(): string {
    return `${this.name} makes a sound`;
  }

  move(distance: number): string {
    return `${this.name} moved ${distance}m`;
  }
}

class Dog extends Animal {
  constructor(name: string, public breed: string) {
    super(name);
  }

  speak(): string {
    return `${this.name} (${this.breed}): Woof woof!`;
  }

  fetch(item: string): string {
    return `${this.name} fetched the ${item}`;
  }
}

// JavaScript memory model:
// dog1.__proto__ -> Dog.prototype -> Animal.prototype -> Object.prototype
// Methods live on the prototype chain,
// shared by all instances

const dog1 = new Dog("Pochi", "Shiba");
const dog2 = new Dog("Hachi", "Akita");

// dog1.speak and dog2.speak reference the same function object
console.log(dog1.speak === dog2.speak); // true

// Checking the prototype chain
console.log(dog1 instanceof Dog);    // true
console.log(dog1 instanceof Animal); // true
```

### 2.3 Garbage Collection and the Object Lifecycle

```
Object lifecycle:

  1. Allocation
     -> Allocate memory on the heap with new
     -> Initialize via the constructor

  2. Usage
     -> Method calls, field accesses
     -> Access the object through references

  3. Unreachable
     -> All references are gone
     -> Becomes eligible for GC reclamation

  4. Collection
     -> GC frees memory
     -> Finalizer/destructor is called (language dependent)

  +----------+     +----------+     +------------+     +----------+
  | Allocate | --> |   Use    | --> | Unreachable| --> | Collect  |
  | (new)    |     | (in use) |     | (no refs)  |     | (GC)     |
  +----------+     +----------+     +------------+     +----------+
```

```python
# Python: object lifecycle and garbage collection

import gc
import weakref

class Resource:
    """Class demonstrating resource management"""

    _instance_count = 0

    def __init__(self, name: str):
        self.name = name
        Resource._instance_count += 1
        print(f"[Created] {self.name} (total: {Resource._instance_count})")

    def __del__(self):
        Resource._instance_count -= 1
        print(f"[Destroyed] {self.name} (remaining: {Resource._instance_count})")

    def process(self) -> str:
        return f"{self.name}: processing"


def demonstrate_lifecycle():
    # 1. Allocation
    r1 = Resource("Resource A")
    r2 = Resource("Resource B")
    r3 = Resource("Resource C")

    # 2. Usage
    print(r1.process())
    print(r2.process())

    # 3. Releasing references
    r2 = None  # Reference to Resource B is gone -> eligible for GC

    # Weak reference: check an object's liveness (does not affect refcount)
    weak_r3 = weakref.ref(r3)
    print(f"r3 is alive: {weak_r3() is not None}")  # True

    r3 = None  # Reference to Resource C is gone
    print(f"r3 is alive: {weak_r3() is not None}")  # False

    # 4. Explicit GC execution
    gc.collect()

    print(f"Remaining objects: {Resource._instance_count}")
    # r1 is still in scope, so it remains


# Guaranteed resource release via a context manager
class DatabaseConnection:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._connection = None

    def __enter__(self):
        print(f"Connecting: {self.host}:{self.port}")
        self._connection = f"Connection({self.host}:{self.port})"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Disconnecting: {self.host}:{self.port}")
        self._connection = None
        return False  # Re-raise the exception

    def execute(self, query: str) -> str:
        if self._connection is None:
            raise RuntimeError("Not connected")
        return f"Executed: {query}"


# Manage the lifecycle explicitly using a with statement
with DatabaseConnection("localhost", 5432) as db:
    result = db.execute("SELECT * FROM users")
    print(result)
# <- __exit__ is called automatically here
```

```cpp
// C++: manual memory management and smart pointers

#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Sensor {
public:
    Sensor(const std::string& name, double threshold)
        : name_(name), threshold_(threshold), reading_(0.0) {
        std::cout << "[Created] " << name_ << std::endl;
    }

    ~Sensor() {
        std::cout << "[Destroyed] " << name_ << std::endl;
    }

    // Disallow copying (prevent resource duplication)
    Sensor(const Sensor&) = delete;
    Sensor& operator=(const Sensor&) = delete;

    // Allow moving
    Sensor(Sensor&& other) noexcept
        : name_(std::move(other.name_)),
          threshold_(other.threshold_),
          reading_(other.reading_) {
        std::cout << "[Moved] " << name_ << std::endl;
    }

    void update(double value) {
        reading_ = value;
    }

    bool isAlarm() const {
        return reading_ > threshold_;
    }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    double threshold_;
    double reading_;
};

int main() {
    // 1. Object on the stack (automatically managed)
    {
        Sensor temp("Temperature sensor", 40.0);
        temp.update(42.0);
        std::cout << temp.name() << " alarm: "
                  << (temp.isAlarm() ? "Yes" : "No") << std::endl;
    } // <- The destructor is called automatically when leaving scope

    // 2. unique_ptr (exclusive ownership)
    auto humidity = std::make_unique<Sensor>("Humidity sensor", 80.0);
    humidity->update(75.0);

    // Transfer of ownership
    auto transferred = std::move(humidity);
    // humidity becomes nullptr

    // 3. shared_ptr (shared ownership)
    auto pressure = std::make_shared<Sensor>("Pressure sensor", 1050.0);
    {
        auto shared_ref = pressure;  // Reference count: 2
        std::cout << "Reference count: " << pressure.use_count() << std::endl;
    } // shared_ref goes out of scope -> reference count: 1

    std::cout << "Reference count: " << pressure.use_count() << std::endl;

    return 0;
}
// All smart pointers go out of scope -> destructors are called automatically
```

### 2.4 Reference Counting and Cycle Detection

```
Reference counting (Python, Swift, Rust's Arc):

  Each object counts "how many places reference it."
  When the count reaches 0, it is freed immediately.

  Problem: reference cycles
    A -> B -> A  (they reference each other)
    -> Neither reference count reaches 0
    -> Memory leak

  Solutions:
    1. Use weak references
    2. Cycle detection GC (Python's gc module)
    3. Ownership model (Rust) structurally prevents reference cycles
```

```python
# Python: the reference cycle problem and its mitigations

import gc
import weakref

class Parent:
    def __init__(self, name: str):
        self.name = name
        self.children: list["Child"] = []

    def add_child(self, child: "Child") -> None:
        self.children.append(child)

    def __repr__(self) -> str:
        return f"Parent({self.name})"

    def __del__(self):
        print(f"[GC] Parent({self.name}) destroyed")

class Child:
    def __init__(self, name: str, parent: Parent):
        self.name = name
        # Problem: strong reference -> causes a reference cycle
        # self.parent = parent

        # Solution: use a weak reference
        self._parent_ref = weakref.ref(parent)

    @property
    def parent(self) -> Parent | None:
        return self._parent_ref()

    def __repr__(self) -> str:
        return f"Child({self.name})"

    def __del__(self):
        print(f"[GC] Child({self.name}) destroyed")

def demo_weak_ref():
    p = Parent("Taro")
    c1 = Child("Ichiro", p)
    c2 = Child("Jiro", p)
    p.add_child(c1)
    p.add_child(c2)

    print(c1.parent)  # Parent(Taro)

    del p  # Parent is released
    print(c1.parent)  # None (the weak reference is invalidated)

demo_weak_ref()
gc.collect()
```

---

## 3. Constructors

```
Role of a constructor:
  1. Initialize fields
  2. Establish invariants
  3. Inject dependent objects
```

### 3.1 Basic Constructor Patterns

```typescript
// TypeScript: constructor design patterns

// Basic: only required parameters
class User {
  constructor(
    public readonly name: string,
    public readonly email: string,
  ) {}
}

// Optional parameters
class HttpClient {
  private baseUrl: string;
  private timeout: number;
  private retries: number;

  constructor(baseUrl: string, options?: { timeout?: number; retries?: number }) {
    this.baseUrl = baseUrl;
    this.timeout = options?.timeout ?? 5000;
    this.retries = options?.retries ?? 3;
  }
}

// Factory method (alternative to a constructor)
class Temperature {
  private constructor(private readonly kelvin: number) {}

  static fromCelsius(c: number): Temperature {
    return new Temperature(c + 273.15);
  }

  static fromFahrenheit(f: number): Temperature {
    return new Temperature((f - 32) * 5 / 9 + 273.15);
  }

  toCelsius(): number {
    return this.kelvin - 273.15;
  }
}

// Usage is clear
const temp = Temperature.fromCelsius(100);
// new Temperature(373.15) is not allowed because it is private
```

### 3.2 Establishing Invariants in the Constructor

```typescript
// TypeScript: guaranteeing invariants in the constructor

class EmailAddress {
  private readonly value: string;

  constructor(email: string) {
    const trimmed = email.trim().toLowerCase();

    // Invariant 1: must not be an empty string
    if (trimmed.length === 0) {
      throw new Error("Email address cannot be empty");
    }

    // Invariant 2: must contain @
    if (!trimmed.includes("@")) {
      throw new Error("Email address must contain @");
    }

    // Invariant 3: must have a domain part
    const [local, domain] = trimmed.split("@");
    if (!local || !domain || !domain.includes(".")) {
      throw new Error("Email address format is invalid");
    }

    // Invariant 4: length limit
    if (trimmed.length > 254) {
      throw new Error("Email address is too long (max 254 chars)");
    }

    this.value = trimmed;
  }

  toString(): string {
    return this.value;
  }

  equals(other: EmailAddress): boolean {
    return this.value === other.value;
  }

  getDomain(): string {
    return this.value.split("@")[1];
  }
}

class Age {
  private readonly value: number;

  constructor(value: number) {
    if (!Number.isInteger(value)) {
      throw new Error("Age must be an integer");
    }
    if (value < 0 || value > 150) {
      throw new Error("Age must be in the range 0 to 150");
    }
    this.value = value;
  }

  toNumber(): number {
    return this.value;
  }

  isAdult(): boolean {
    return this.value >= 18;
  }

  isElderly(): boolean {
    return this.value >= 65;
  }
}

class UserProfile {
  constructor(
    public readonly name: string,
    public readonly email: EmailAddress,
    public readonly age: Age,
  ) {
    // Name invariants
    if (name.trim().length === 0) {
      throw new Error("Name cannot be empty");
    }
    if (name.length > 100) {
      throw new Error("Name must be 100 characters or less");
    }
  }

  getInfo(): string {
    return `${this.name} (${this.age.toNumber()} years old) - ${this.email}`;
  }
}

// Usage: invalid data cannot produce an object
try {
  const email = new EmailAddress("invalid-email");
} catch (e) {
  console.error(e); // "Email address must contain @"
}

// Only valid data can produce an object
const profile = new UserProfile(
  "Taro Tanaka",
  new EmailAddress("tanaka@example.com"),
  new Age(30),
);
console.log(profile.getInfo()); // "Taro Tanaka (30 years old) - tanaka@example.com"
```

### 3.3 The Builder Pattern

```java
// Java: the telescoping constructor problem -> Builder pattern
public class Pizza {
    private final int size;          // Required
    private final boolean cheese;    // Optional
    private final boolean pepperoni; // Optional
    private final boolean mushroom;  // Optional

    // Constructors explode in number
    // Pizza(int)
    // Pizza(int, boolean)
    // Pizza(int, boolean, boolean)
    // Pizza(int, boolean, boolean, boolean)

    // -> Solved with the Builder pattern
    private Pizza(Builder builder) {
        this.size = builder.size;
        this.cheese = builder.cheese;
        this.pepperoni = builder.pepperoni;
        this.mushroom = builder.mushroom;
    }

    public static class Builder {
        private final int size;
        private boolean cheese = false;
        private boolean pepperoni = false;
        private boolean mushroom = false;

        public Builder(int size) { this.size = size; }
        public Builder cheese() { this.cheese = true; return this; }
        public Builder pepperoni() { this.pepperoni = true; return this; }
        public Builder mushroom() { this.mushroom = true; return this; }
        public Pizza build() { return new Pizza(this); }
    }
}

// High readability
Pizza pizza = new Pizza.Builder(12)
    .cheese()
    .pepperoni()
    .build();
```

### 3.4 Patterns for Constructing Complex Objects

```typescript
// TypeScript: staged Builder pattern (type-safe version)

// Control the construction stages at the type level
interface NeedsHost {
  host(host: string): NeedsPort;
}

interface NeedsPort {
  port(port: number): OptionalConfig;
}

interface OptionalConfig {
  ssl(enabled: boolean): OptionalConfig;
  timeout(ms: number): OptionalConfig;
  maxRetries(count: number): OptionalConfig;
  build(): DatabaseConfig;
}

class DatabaseConfig {
  private constructor(
    public readonly host: string,
    public readonly port: number,
    public readonly ssl: boolean,
    public readonly timeout: number,
    public readonly maxRetries: number,
  ) {}

  static builder(): NeedsHost {
    return new DatabaseConfigBuilder();
  }

  getConnectionString(): string {
    const protocol = this.ssl ? "ssl" : "tcp";
    return `${protocol}://${this.host}:${this.port}?timeout=${this.timeout}&retries=${this.maxRetries}`;
  }
}

class DatabaseConfigBuilder implements NeedsHost, NeedsPort, OptionalConfig {
  private _host = "";
  private _port = 0;
  private _ssl = false;
  private _timeout = 5000;
  private _maxRetries = 3;

  host(host: string): NeedsPort {
    this._host = host;
    return this;
  }

  port(port: number): OptionalConfig {
    this._port = port;
    return this;
  }

  ssl(enabled: boolean): OptionalConfig {
    this._ssl = enabled;
    return this;
  }

  timeout(ms: number): OptionalConfig {
    this._timeout = ms;
    return this;
  }

  maxRetries(count: number): OptionalConfig {
    this._maxRetries = count;
    return this;
  }

  build(): DatabaseConfig {
    return new (DatabaseConfig as any)(
      this._host,
      this._port,
      this._ssl,
      this._timeout,
      this._maxRetries,
    );
  }
}

// Usage: failing to supply required parameters in order causes a compile error
const config = DatabaseConfig.builder()
  .host("db.example.com")   // NeedsHost -> NeedsPort
  .port(5432)               // NeedsPort -> OptionalConfig
  .ssl(true)                // OptionalConfig -> OptionalConfig
  .timeout(10000)           // OptionalConfig -> OptionalConfig
  .build();                 // OptionalConfig -> DatabaseConfig

console.log(config.getConnectionString());
// "ssl://db.example.com:5432?timeout=10000&retries=3"

// Compile error: port() cannot be called before host()
// DatabaseConfig.builder().port(5432);  // error
```

```python
# Python: construction pattern using dataclass + factory methods

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass(frozen=True)  # frozen=True makes an immutable object
class Task:
    title: str
    description: str
    priority: Priority
    assignee: str
    due_date: datetime
    tags: tuple[str, ...] = ()  # tuple guarantees immutability
    created_at: datetime = field(default_factory=datetime.now)
    task_id: str = field(default_factory=lambda: f"TASK-{id(object()):08x}")

    def __post_init__(self):
        """Validate invariants"""
        if not self.title.strip():
            raise ValueError("Task title cannot be empty")
        if self.due_date < self.created_at:
            raise ValueError("Due date must be after the creation date")

    # Factory methods
    @classmethod
    def create_bug(cls, title: str, assignee: str,
                   severity: str = "medium") -> "Task":
        """Create a bug report task"""
        priority = {
            "low": Priority.LOW,
            "medium": Priority.MEDIUM,
            "high": Priority.HIGH,
            "critical": Priority.CRITICAL,
        }.get(severity, Priority.MEDIUM)

        return cls(
            title=f"[Bug] {title}",
            description=f"Bug report: {title}",
            priority=priority,
            assignee=assignee,
            due_date=datetime.now() + timedelta(days=7),
            tags=("bug", severity),
        )

    @classmethod
    def create_feature(cls, title: str, assignee: str,
                       sprint_days: int = 14) -> "Task":
        """Create a feature development task"""
        return cls(
            title=f"[Feature] {title}",
            description=f"Feature development: {title}",
            priority=Priority.MEDIUM,
            assignee=assignee,
            due_date=datetime.now() + timedelta(days=sprint_days),
            tags=("feature", "development"),
        )

    def is_overdue(self) -> bool:
        return datetime.now() > self.due_date

    def days_until_due(self) -> int:
        delta = self.due_date - datetime.now()
        return max(0, delta.days)


# Usage example
bug = Task.create_bug("Login screen crashes", "Tanaka", "critical")
feature = Task.create_feature("Dashboard feature", "Yamada")

print(f"{bug.title} - {bug.days_until_due()} days until due")
print(f"{feature.title} - priority: {feature.priority.value}")
```

### 3.5 Copy Constructors and clone

```java
// Java: copy constructor vs. clone

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Playlist {
    private final String name;
    private final String owner;
    private final List<String> songs;

    public Playlist(String name, String owner) {
        this.name = name;
        this.owner = owner;
        this.songs = new ArrayList<>();
    }

    // Copy constructor (preferred over clone)
    public Playlist(Playlist other) {
        this.name = other.name + " (copy)";
        this.owner = other.owner;
        this.songs = new ArrayList<>(other.songs); // Defensive copy
    }

    // Copy with a specific attribute changed
    public Playlist withName(String newName) {
        Playlist copy = new Playlist(this);
        // Direct field access (within the same class)
        // When name is final, you need reflection or a different constructor
        return copy;
    }

    public void addSong(String song) {
        songs.add(song);
    }

    public List<String> getSongs() {
        return Collections.unmodifiableList(songs); // Defensive copy
    }

    public String getName() { return name; }
    public String getOwner() { return owner; }

    @Override
    public String toString() {
        return String.format("Playlist('%s' by %s, %d songs)", name, owner, songs.size());
    }
}

// Usage example
Playlist original = new Playlist("Favorites", "Tanaka");
original.addSong("Song A");
original.addSong("Song B");

Playlist copy = new Playlist(original);
copy.addSong("Song C");

System.out.println(original); // Playlist('Favorites' by Tanaka, 2 songs)
System.out.println(copy);     // Playlist('Favorites (copy)' by Tanaka, 3 songs)
// -> Independent objects
```

---

## 4. Static Members vs. Instance Members

```
+------------------+--------------------+--------------------+
|                  | Static              | Instance            |
+------------------+--------------------+--------------------+
| Belongs to       | The class           | The object          |
+------------------+--------------------+--------------------+
| Memory           | One per class area  | One per object      |
+------------------+--------------------+--------------------+
| Access           | ClassName.member    | instance.member     |
+------------------+--------------------+--------------------+
| this reference   | No                  | Yes                 |
+------------------+--------------------+--------------------+
| Use cases        | Utilities           | Object-specific     |
|                  | Factory methods     | state and behavior  |
|                  | Constants           |                     |
+------------------+--------------------+--------------------+
```

### 4.1 Python Class Methods, Static Methods, and Instance Methods

```python
# Python: class method vs. instance method
class Counter:
    _instance_count = 0  # Class variable (shared by all instances)

    def __init__(self, name: str):
        self.name = name         # Instance variable
        self.count = 0           # Instance variable
        Counter._instance_count += 1

    def increment(self):          # Instance method
        self.count += 1

    @classmethod
    def get_instance_count(cls):  # Class method
        return cls._instance_count

    @staticmethod
    def is_valid_name(name: str) -> bool:  # Static method
        return len(name) > 0 and name.isalpha()

c1 = Counter("alpha")
c2 = Counter("beta")
print(Counter.get_instance_count())  # 2
print(Counter.is_valid_name("test")) # True
```

### 4.2 Proper Use of Static Members

```typescript
// TypeScript: patterns for leveraging static members

// Pattern 1: defining constants
class HttpStatus {
  static readonly OK = 200;
  static readonly NOT_FOUND = 404;
  static readonly INTERNAL_SERVER_ERROR = 500;

  static isSuccess(code: number): boolean {
    return code >= 200 && code < 300;
  }

  static isClientError(code: number): boolean {
    return code >= 400 && code < 500;
  }

  static isServerError(code: number): boolean {
    return code >= 500 && code < 600;
  }
}

// Pattern 2: factory methods
class Color {
  private constructor(
    public readonly r: number,
    public readonly g: number,
    public readonly b: number,
    public readonly a: number = 1.0,
  ) {}

  // Named factory methods
  static fromRGB(r: number, g: number, b: number): Color {
    return new Color(
      Math.max(0, Math.min(255, r)),
      Math.max(0, Math.min(255, g)),
      Math.max(0, Math.min(255, b)),
    );
  }

  static fromHex(hex: string): Color {
    const match = hex.replace("#", "").match(/.{2}/g);
    if (!match || match.length < 3) {
      throw new Error(`Invalid hex color: ${hex}`);
    }
    return new Color(
      parseInt(match[0], 16),
      parseInt(match[1], 16),
      parseInt(match[2], 16),
    );
  }

  static fromHSL(h: number, s: number, l: number): Color {
    // HSL -> RGB conversion logic
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;

    let r = 0, g = 0, b = 0;
    if (h < 60)       { r = c; g = x; }
    else if (h < 120) { r = x; g = c; }
    else if (h < 180) { g = c; b = x; }
    else if (h < 240) { g = x; b = c; }
    else if (h < 300) { r = x; b = c; }
    else              { r = c; b = x; }

    return new Color(
      Math.round((r + m) * 255),
      Math.round((g + m) * 255),
      Math.round((b + m) * 255),
    );
  }

  // Constants for commonly used colors
  static readonly RED = Color.fromRGB(255, 0, 0);
  static readonly GREEN = Color.fromRGB(0, 255, 0);
  static readonly BLUE = Color.fromRGB(0, 0, 255);
  static readonly WHITE = Color.fromRGB(255, 255, 255);
  static readonly BLACK = Color.fromRGB(0, 0, 0);

  toHex(): string {
    const hex = (n: number) => n.toString(16).padStart(2, "0");
    return `#${hex(this.r)}${hex(this.g)}${hex(this.b)}`;
  }

  mix(other: Color, ratio: number = 0.5): Color {
    return Color.fromRGB(
      Math.round(this.r * (1 - ratio) + other.r * ratio),
      Math.round(this.g * (1 - ratio) + other.g * ratio),
      Math.round(this.b * (1 - ratio) + other.b * ratio),
    );
  }
}

// Pattern 3: singleton (a single unique instance)
class AppConfig {
  private static instance: AppConfig | null = null;

  private constructor(
    public readonly appName: string,
    public readonly version: string,
    public readonly debug: boolean,
  ) {}

  static getInstance(): AppConfig {
    if (AppConfig.instance === null) {
      AppConfig.instance = new AppConfig(
        "MyApp",
        "1.0.0",
        process.env.NODE_ENV !== "production",
      );
    }
    return AppConfig.instance;
  }

  // For testing: reset the instance
  static resetForTesting(): void {
    AppConfig.instance = null;
  }
}

// Pattern 4: registry (managing objects)
class EventBus {
  private static handlers = new Map<string, Set<Function>>();

  static on(event: string, handler: Function): void {
    if (!EventBus.handlers.has(event)) {
      EventBus.handlers.set(event, new Set());
    }
    EventBus.handlers.get(event)!.add(handler);
  }

  static off(event: string, handler: Function): void {
    EventBus.handlers.get(event)?.delete(handler);
  }

  static emit(event: string, ...args: unknown[]): void {
    const handlers = EventBus.handlers.get(event);
    if (handlers) {
      for (const handler of handlers) {
        handler(...args);
      }
    }
  }

  static clear(): void {
    EventBus.handlers.clear();
  }
}
```

### 4.3 Caveats and Anti-Patterns with Static Members

```
Risks of static members:

1. Becomes hard to test:
   -> Static methods are hard to mock
   -> Dependency injection is not possible
   -> State is shared across tests

2. Concurrency issues:
   -> Static fields are shared across all threads
   -> Proper synchronization is required

3. Overuse:
   -> "Why not just make everything static?" -> regression to procedural programming
   -> The benefits of OOP (e.g., polymorphism) are lost

Decision criteria:
  static is appropriate when:
    -> Factory methods
    -> Utility functions (Math.max, Collections.sort)
    -> Constants
    -> Pure functions that do not depend on object state

  static is inappropriate when:
    -> Business logic (hurts testability)
    -> Stateful processing (concurrency issues)
    -> Processing that requires interface implementation
```

```java
// Java: avoiding overuse of static methods

// Bad example: overuse of static methods
public class UserService {
    public static User findById(long id) {
        // Directly accesses the database -> hard to test
        return Database.query("SELECT * FROM users WHERE id = ?", id);
    }

    public static void updateEmail(long userId, String email) {
        // Calls between static methods -> cannot be mocked
        User user = findById(userId);
        Database.execute("UPDATE users SET email = ? WHERE id = ?", email, userId);
    }
}

// Good example: instance methods + dependency injection
public class UserService {
    private final UserRepository repository;
    private final EmailValidator validator;

    public UserService(UserRepository repository, EmailValidator validator) {
        this.repository = repository;
        this.validator = validator;
    }

    public User findById(long id) {
        return repository.findById(id)
            .orElseThrow(() -> new UserNotFoundException(id));
    }

    public void updateEmail(long userId, String email) {
        if (!validator.isValid(email)) {
            throw new InvalidEmailException(email);
        }
        User user = findById(userId);
        user.setEmail(email);
        repository.save(user);
    }
}

// In tests: mocks can be injected
// UserService service = new UserService(mockRepo, mockValidator);
```

---

## 5. Value Types vs. Reference Types

```
Reference type:
  -> The variable holds a reference (pointer) to the object
  -> Assignment copies the pointer (shallow copy)
  -> Java classes, all Python objects

  user1 = User("Tanaka")
  user2 = user1        <- user1 and user2 reference the same object
  user2.name = "Yamada"  <- user1.name also becomes "Yamada"!

Value type:
  -> The variable holds the value itself
  -> Assignment copies the value (deep copy)
  -> C# struct, Swift struct, Rust non-reference types

  var point1 = Point(x: 1, y: 2)
  var point2 = point1  <- point2 is a copy of point1
  point2.x = 10        <- point1.x remains 1
```

### 5.1 Swift: Clear Distinction Between Value Types and Reference Types

```swift
// Swift: clear distinction between value types and reference types
struct Point {          // Value type
    var x: Double
    var y: Double
}

class Circle {          // Reference type
    var center: Point
    var radius: Double

    init(center: Point, radius: Double) {
        self.center = center
        self.radius = radius
    }
}

var p1 = Point(x: 0, y: 0)
var p2 = p1             // Copy
p2.x = 10
print(p1.x)            // 0 (unchanged)

let c1 = Circle(center: Point(x: 0, y: 0), radius: 5)
let c2 = c1             // Shared reference
c2.radius = 10
print(c1.radius)        // 10 (changed!)
```

### 5.2 How Each Language Handles Value and Reference Types

```
+----------+----------------------+----------------------+
| Language | Value types          | Reference types      |
+----------+----------------------+----------------------+
| Java     | Primitive types      | Classes (all)        |
|          | (int, double, etc.)  |                      |
+----------+----------------------+----------------------+
| C#       | struct, enum,        | class, interface,    |
|          | primitive types      | delegate, array      |
+----------+----------------------+----------------------+
| Swift    | struct, enum, tuple  | class, closure       |
+----------+----------------------+----------------------+
| Kotlin   | inline class         | class (all)          |
|          | (optimized on JVM)   |                      |
+----------+----------------------+----------------------+
| Rust     | Everything by default| Box, Rc, Arc,        |
|          | (when Copy trait)    | reference (&T)       |
+----------+----------------------+----------------------+
| Python   | None (all reference) | All objects          |
|          | * int, str are immutable|                   |
+----------+----------------------+----------------------+
| Go       | struct, basic types  | pointers, slices,    |
|          |                      | maps, channels       |
+----------+----------------------+----------------------+
```

```csharp
// C#: distinguishing struct (value type) from class (reference type)

// Value type: suited to small, immutable data
public readonly struct Vector2D
{
    public double X { get; }
    public double Y { get; }

    public Vector2D(double x, double y)
    {
        X = x;
        Y = y;
    }

    public double Magnitude => Math.Sqrt(X * X + Y * Y);

    public Vector2D Normalize()
    {
        var mag = Magnitude;
        return mag > 0 ? new Vector2D(X / mag, Y / mag) : this;
    }

    public static Vector2D operator +(Vector2D a, Vector2D b)
        => new Vector2D(a.X + b.X, a.Y + b.Y);

    public static Vector2D operator -(Vector2D a, Vector2D b)
        => new Vector2D(a.X - b.X, a.Y - b.Y);

    public static Vector2D operator *(Vector2D v, double scalar)
        => new Vector2D(v.X * scalar, v.Y * scalar);

    public static double Dot(Vector2D a, Vector2D b)
        => a.X * b.X + a.Y * b.Y;

    public override string ToString() => $"({X:F2}, {Y:F2})";
}

// Reference type: suited to complex stateful objects
public class Particle
{
    public Vector2D Position { get; private set; }
    public Vector2D Velocity { get; private set; }
    public double Mass { get; }
    public bool IsActive { get; private set; }

    public Particle(Vector2D position, Vector2D velocity, double mass)
    {
        Position = position;
        Velocity = velocity;
        Mass = mass;
        IsActive = true;
    }

    public void Update(double deltaTime)
    {
        if (!IsActive) return;
        Position = Position + Velocity * deltaTime;
    }

    public void ApplyForce(Vector2D force)
    {
        if (!IsActive) return;
        // F = ma -> a = F/m -> v += a * dt
        Velocity = Velocity + force * (1.0 / Mass);
    }

    public void Deactivate() => IsActive = false;
}

// Usage example
var v1 = new Vector2D(3, 4);
var v2 = v1; // Copy (value type)
// Changing v2 does not affect v1

var p1 = new Particle(new Vector2D(0, 0), new Vector2D(1, 0), 1.0);
var p2 = p1; // Reference copy (reference type)
p2.Update(1.0); // Affects p1 as well (same object)
```

### 5.3 Copy-on-Write (COW) Optimization

```
Copy-on-Write:
  -> On copy, the reference is shared; the actual copy is made only on mutation
  -> Balances memory efficiency with value-type semantics

  Swift's Array, String, and Dictionary use COW internally:

  var array1 = [1, 2, 3]
  var array2 = array1   <- At this point, memory is shared
  array2.append(4)      <- The copy occurs only here

  +------+
  |array1+-> [1, 2, 3]     (shared state)
  |array2+-+
  +------+

  After array2.append(4):

  +------+
  |array1+-> [1, 2, 3]     (original data)
  |array2+-> [1, 2, 3, 4]  (new copied data)
  +------+
```

```swift
// Swift: an example Copy-on-Write implementation

final class Storage<T> {
    var value: T

    init(_ value: T) {
        self.value = value
    }

    func copy() -> Storage<T> {
        return Storage(value)
    }
}

struct COWWrapper<T> {
    private var storage: Storage<T>

    init(_ value: T) {
        self.storage = Storage(value)
    }

    var value: T {
        get { storage.value }
        set {
            // Copy only when there are other referrers
            if !isKnownUniquelyReferenced(&storage) {
                storage = storage.copy()
            }
            storage.value = newValue
        }
    }
}

// Usage example
var a = COWWrapper([1, 2, 3])
var b = a  // Share reference (no copy yet)

// The copy occurs only when b is mutated
b.value = [1, 2, 3, 4]
print(a.value)  // [1, 2, 3] (unaffected)
print(b.value)  // [1, 2, 3, 4]
```

---

## 6. Equality and Identity

```
Identity:
  -> Whether two variables point to "the same object"
  -> Memory address comparison
  -> Java: ==, Python: is, JavaScript: === (between objects)

Equality:
  -> Whether two objects have "the same value"
  -> Logical equality
  -> Java: .equals(), Python: ==, JavaScript: custom implementation

  +------+      +----------+
  | a ---+----->| "Hello"  |  a and b are identical (same object)
  | b ---+----->|          |  a and b are equal (same value)
  +------+      +----------+

  +------+      +----------+
  | a ---+----->| "Hello"  |  a and b are not identical (different objects)
  | b ---+--+   +----------+  a and b are equal (same value)
  +------+  |   +----------+
            +-->| "Hello"  |
                +----------+
```

```java
// Java: correctly implementing equals and hashCode

import java.util.Objects;

public class Money implements Comparable<Money> {
    private final long amount;  // Held in the smallest unit (sen)
    private final String currency;

    public Money(long amount, String currency) {
        this.amount = amount;
        this.currency = Objects.requireNonNull(currency);
    }

    // Equality definition: equal if amount and currency match
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;  // Identity check (optimization)
        if (!(obj instanceof Money other)) return false;
        return amount == other.amount
            && currency.equals(other.currency);
    }

    // When overriding equals, always override hashCode too
    // Equal objects must return the same hashCode
    @Override
    public int hashCode() {
        return Objects.hash(amount, currency);
    }

    @Override
    public int compareTo(Money other) {
        if (!currency.equals(other.currency)) {
            throw new IllegalArgumentException("Currencies differ");
        }
        return Long.compare(amount, other.amount);
    }

    @Override
    public String toString() {
        return String.format("%d.%02d %s", amount / 100, amount % 100, currency);
    }
}

// Usage example
Money m1 = new Money(10000, "JPY");
Money m2 = new Money(10000, "JPY");
Money m3 = m1;

System.out.println(m1 == m2);      // false (different objects)
System.out.println(m1 == m3);      // true (same object)
System.out.println(m1.equals(m2)); // true (equal)
System.out.println(m1.equals(m3)); // true (equal)

// Works correctly with HashSet / HashMap
Set<Money> set = new HashSet<>();
set.add(m1);
set.add(m2);
System.out.println(set.size()); // 1 (deduped because equal)
```

```python
# Python: implementing __eq__ and __hash__

from dataclasses import dataclass

@dataclass(frozen=True)  # frozen=True auto-generates __eq__ and __hash__
class Coordinate:
    latitude: float
    longitude: float

    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be in the range -90 to 90: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be in the range -180 to 180: {self.longitude}")

    def distance_to(self, other: "Coordinate") -> float:
        """Compute the distance between two points using the Haversine formula (km)"""
        import math
        R = 6371  # Earth's radius (km)
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

# Equality checks
tokyo = Coordinate(35.6762, 139.6503)
tokyo2 = Coordinate(35.6762, 139.6503)
osaka = Coordinate(34.6937, 135.5023)

print(tokyo == tokyo2)          # True (equal)
print(tokyo is tokyo2)          # False (not identical)
print(tokyo == osaka)           # False (not equal)

# Usable as a dictionary key
cities = {
    tokyo: "Tokyo",
    osaka: "Osaka",
}
print(cities[tokyo2])  # "Tokyo" (accessed with an equal key)

# Distance calculation
print(f"Tokyo-Osaka: {tokyo.distance_to(osaka):.1f} km")
```

---

## 7. Object Composition

```
Composition:
  -> A "has-a" relationship where an object holds another object
  -> More flexible and recommended over inheritance (is-a)
  -> Composition can be changed at runtime

  +------------+     +------------+
  |   Car      |     |   Engine   |
  |            |---->|            |  Car "has" an Engine
  | engine     |     | start()    |
  | start()    |     | stop()     |
  +------------+     +------------+
        |
        +---->+------------+
        |     | Wheels[]   |
        |     | rotate()   |
        |     +------------+
        |
        +---->+------------+
              | GPS        |
              | navigate() |
              +------------+
```

```typescript
// TypeScript: flexible design via composition

// Classes with individual responsibilities
class Logger {
  constructor(private readonly prefix: string) {}

  info(message: string): void {
    console.log(`[INFO][${this.prefix}] ${message}`);
  }

  error(message: string): void {
    console.error(`[ERROR][${this.prefix}] ${message}`);
  }

  warn(message: string): void {
    console.warn(`[WARN][${this.prefix}] ${message}`);
  }
}

class Validator {
  private rules = new Map<string, (value: unknown) => string | null>();

  addRule(field: string, validate: (value: unknown) => string | null): this {
    this.rules.set(field, validate);
    return this;
  }

  validate(data: Record<string, unknown>): string[] {
    const errors: string[] = [];
    for (const [field, rule] of this.rules) {
      const error = rule(data[field]);
      if (error) {
        errors.push(`${field}: ${error}`);
      }
    }
    return errors;
  }
}

class EventEmitter<T extends Record<string, unknown[]>> {
  private handlers = new Map<string, Set<Function>>();

  on<K extends keyof T>(event: K, handler: (...args: T[K]) => void): void {
    if (!this.handlers.has(event as string)) {
      this.handlers.set(event as string, new Set());
    }
    this.handlers.get(event as string)!.add(handler);
  }

  emit<K extends keyof T>(event: K, ...args: T[K]): void {
    const handlers = this.handlers.get(event as string);
    if (handlers) {
      for (const handler of handlers) {
        handler(...args);
      }
    }
  }
}

// Composition: build complex features by combining individual classes
type UserEvents = {
  "user:created": [user: { name: string; email: string }];
  "user:updated": [userId: string, changes: Record<string, unknown>];
  "user:deleted": [userId: string];
};

class UserService {
  private readonly logger: Logger;
  private readonly validator: Validator;
  private readonly events: EventEmitter<UserEvents>;
  private readonly users = new Map<string, { name: string; email: string }>();

  constructor() {
    // Composition: hold each component internally
    this.logger = new Logger("UserService");

    this.validator = new Validator()
      .addRule("name", (v) =>
        typeof v === "string" && v.length > 0 ? null : "Name is required"
      )
      .addRule("email", (v) =>
        typeof v === "string" && v.includes("@") ? null : "A valid email address is required"
      );

    this.events = new EventEmitter<UserEvents>();
  }

  onUserCreated(handler: (user: { name: string; email: string }) => void): void {
    this.events.on("user:created", handler);
  }

  createUser(name: string, email: string): string {
    // Validation (delegated to Validator)
    const errors = this.validator.validate({ name, email });
    if (errors.length > 0) {
      this.logger.error(`Validation error: ${errors.join(", ")}`);
      throw new Error(`Validation error: ${errors.join(", ")}`);
    }

    // Create user
    const id = crypto.randomUUID();
    const user = { name, email };
    this.users.set(id, user);

    // Log output (delegated to Logger)
    this.logger.info(`User created: ${name} (${email})`);

    // Emit event (delegated to EventEmitter)
    this.events.emit("user:created", user);

    return id;
  }

  getUser(id: string): { name: string; email: string } | undefined {
    return this.users.get(id);
  }
}

// Usage example
const service = new UserService();
service.onUserCreated((user) => {
  console.log(`New user notification: ${user.name}`);
});

const userId = service.createUser("Taro Tanaka", "tanaka@example.com");
```

```python
# Python: practical example of composition (notification system)

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

# Notification destination (Strategy pattern)
class NotificationSender(Protocol):
    def send(self, recipient: str, subject: str, body: str) -> bool: ...

class EmailSender:
    def __init__(self, smtp_host: str, smtp_port: int):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def send(self, recipient: str, subject: str, body: str) -> bool:
        print(f"[Email] To: {recipient}, Subject: {subject}")
        return True

class SlackSender:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, recipient: str, subject: str, body: str) -> bool:
        print(f"[Slack] Channel: {recipient}, Message: {subject}")
        return True

class SmsSender:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def send(self, recipient: str, subject: str, body: str) -> bool:
        print(f"[SMS] To: {recipient}, Message: {body[:100]}")
        return True

# Template engine
@dataclass
class MessageTemplate:
    subject_template: str
    body_template: str

    def render(self, **kwargs: str) -> tuple[str, str]:
        subject = self.subject_template.format(**kwargs)
        body = self.body_template.format(**kwargs)
        return subject, body

# Notification log
@dataclass
class NotificationLog:
    entries: list[dict] = field(default_factory=list)

    def record(self, channel: str, recipient: str,
               subject: str, success: bool) -> None:
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "channel": channel,
            "recipient": recipient,
            "subject": subject,
            "success": success,
        })

    def get_failures(self) -> list[dict]:
        return [e for e in self.entries if not e["success"]]

# Composition: notification service
class NotificationService:
    """Composition pattern: combine components to provide notification features"""

    def __init__(self):
        self._senders: dict[str, NotificationSender] = {}
        self._templates: dict[str, MessageTemplate] = {}
        self._log = NotificationLog()

    def register_sender(self, name: str, sender: NotificationSender) -> None:
        self._senders[name] = sender

    def register_template(self, name: str, template: MessageTemplate) -> None:
        self._templates[name] = template

    def send(self, channel: str, recipient: str,
             template_name: str, **kwargs: str) -> bool:
        sender = self._senders.get(channel)
        if sender is None:
            raise ValueError(f"Unregistered channel: {channel}")

        template = self._templates.get(template_name)
        if template is None:
            raise ValueError(f"Unregistered template: {template_name}")

        subject, body = template.render(**kwargs)
        success = sender.send(recipient, subject, body)
        self._log.record(channel, recipient, subject, success)
        return success

    def broadcast(self, template_name: str,
                  recipients: dict[str, str], **kwargs: str) -> dict[str, bool]:
        results = {}
        for channel, recipient in recipients.items():
            results[channel] = self.send(channel, recipient, template_name, **kwargs)
        return results

    @property
    def log(self) -> NotificationLog:
        return self._log


# Usage example
service = NotificationService()

# Register components
service.register_sender("email", EmailSender("smtp.example.com", 587))
service.register_sender("slack", SlackSender("https://hooks.slack.com/xxx"))
service.register_sender("sms", SmsSender("api-key-123"))

service.register_template("welcome", MessageTemplate(
    subject_template="Welcome, {name}!",
    body_template="{name}, your account has been created successfully.",
))

# Broadcast to multiple channels at once
results = service.broadcast(
    "welcome",
    recipients={
        "email": "tanaka@example.com",
        "slack": "#new-users",
        "sms": "090-1234-5678",
    },
    name="Taro Tanaka",
)

print(f"Send results: {results}")
print(f"Failure count: {len(service.log.get_failures())}")
```

---

## 8. Best Practices for Class Design

### 8.1 The Single Responsibility Principle (SRP)

```
A class should change for only one reason:

  Bad example: UserManager holding all responsibilities
    class UserManager {
      createUser()
      deleteUser()
      sendEmail()        <- sending email is a different responsibility
      generateReport()   <- report generation is a different responsibility
      validateInput()    <- validation is a different responsibility
    }

  Good example: separate the responsibilities
    class UserRepository { create() / delete() / find() }
    class EmailService { send() }
    class ReportGenerator { generate() }
    class UserValidator { validate() }
    class UserService { // composes and uses these }
```

### 8.2 Increase Cohesion

```typescript
// TypeScript: highly cohesive class design

// Low cohesion: mixing unrelated methods
class Utils {
  static formatDate(date: Date): string { /* ... */ return ""; }
  static parseJSON(json: string): unknown { /* ... */ return {}; }
  static calculateTax(amount: number): number { /* ... */ return 0; }
  static sendEmail(to: string, body: string): void { /* ... */ }
  static resizeImage(path: string, width: number): void { /* ... */ }
}

// High cohesion: related data and methods grouped in one class
class DateRange {
  constructor(
    public readonly start: Date,
    public readonly end: Date,
  ) {
    if (start > end) {
      throw new Error("Start date must be before end date");
    }
  }

  contains(date: Date): boolean {
    return date >= this.start && date <= this.end;
  }

  overlaps(other: DateRange): boolean {
    return this.start <= other.end && other.start <= this.end;
  }

  getDurationDays(): number {
    const diff = this.end.getTime() - this.start.getTime();
    return Math.ceil(diff / (1000 * 60 * 60 * 24));
  }

  intersection(other: DateRange): DateRange | null {
    const start = new Date(Math.max(this.start.getTime(), other.start.getTime()));
    const end = new Date(Math.min(this.end.getTime(), other.end.getTime()));
    if (start > end) return null;
    return new DateRange(start, end);
  }

  extend(days: number): DateRange {
    const newEnd = new Date(this.end);
    newEnd.setDate(newEnd.getDate() + days);
    return new DateRange(this.start, newEnd);
  }

  format(locale: string = "ja-JP"): string {
    const fmt = (d: Date) => d.toLocaleDateString(locale);
    return `${fmt(this.start)} - ${fmt(this.end)} (${this.getDurationDays()} days)`;
  }
}

// Usage example
const vacation = new DateRange(
  new Date("2025-08-10"),
  new Date("2025-08-20"),
);
const holiday = new DateRange(
  new Date("2025-08-15"),
  new Date("2025-08-16"),
);

console.log(vacation.format());           // "2025/8/10 - 2025/8/20 (10 days)"
console.log(vacation.overlaps(holiday));   // true
console.log(vacation.contains(new Date("2025-08-12"))); // true

const overlap = vacation.intersection(holiday);
console.log(overlap?.format());           // "2025/8/15 - 2025/8/16 (1 day)"
```

### 8.3 Practical Example: Product Management for an E-Commerce Site

```typescript
// TypeScript: practical class design (e-commerce site)

// Value objects
class Money {
  constructor(
    public readonly amount: number,
    public readonly currency: string,
  ) {
    if (amount < 0) throw new Error("Amount must be 0 or greater");
  }

  add(other: Money): Money {
    this.assertSameCurrency(other);
    return new Money(this.amount + other.amount, this.currency);
  }

  subtract(other: Money): Money {
    this.assertSameCurrency(other);
    if (this.amount < other.amount) {
      throw new Error("Result would be negative");
    }
    return new Money(this.amount - other.amount, this.currency);
  }

  multiply(factor: number): Money {
    return new Money(Math.floor(this.amount * factor), this.currency);
  }

  private assertSameCurrency(other: Money): void {
    if (this.currency !== other.currency) {
      throw new Error(`Currencies differ: ${this.currency} vs ${other.currency}`);
    }
  }

  format(): string {
    return `${this.amount.toLocaleString()} ${this.currency}`;
  }

  equals(other: Money): boolean {
    return this.amount === other.amount && this.currency === other.currency;
  }
}

class Quantity {
  constructor(public readonly value: number) {
    if (!Number.isInteger(value) || value < 0) {
      throw new Error("Quantity must be a non-negative integer");
    }
  }

  add(n: number): Quantity {
    return new Quantity(this.value + n);
  }

  subtract(n: number): Quantity {
    return new Quantity(this.value - n);
  }

  isZero(): boolean {
    return this.value === 0;
  }
}

// Entity
class Product {
  private _stock: Quantity;

  constructor(
    public readonly id: string,
    public readonly name: string,
    public readonly price: Money,
    stock: number,
    public readonly category: string,
  ) {
    this._stock = new Quantity(stock);
  }

  get stock(): Quantity {
    return this._stock;
  }

  isAvailable(): boolean {
    return !this._stock.isZero();
  }

  reserve(quantity: number): void {
    if (this._stock.value < quantity) {
      throw new Error(`Insufficient stock: ${this.name} (stock: ${this._stock.value}, requested: ${quantity})`);
    }
    this._stock = this._stock.subtract(quantity);
  }

  restock(quantity: number): void {
    this._stock = this._stock.add(quantity);
  }
}

// Cart item
class CartItem {
  constructor(
    public readonly product: Product,
    private _quantity: Quantity,
  ) {}

  get quantity(): number {
    return this._quantity.value;
  }

  getSubtotal(): Money {
    return this.product.price.multiply(this._quantity.value);
  }

  updateQuantity(newQuantity: number): CartItem {
    return new CartItem(this.product, new Quantity(newQuantity));
  }
}

// Shopping cart
class ShoppingCart {
  private items = new Map<string, CartItem>();

  addItem(product: Product, quantity: number = 1): void {
    if (!product.isAvailable()) {
      throw new Error(`${product.name} is out of stock`);
    }
    const existing = this.items.get(product.id);
    if (existing) {
      const newQty = existing.quantity + quantity;
      this.items.set(product.id, existing.updateQuantity(newQty));
    } else {
      this.items.set(product.id, new CartItem(product, new Quantity(quantity)));
    }
  }

  removeItem(productId: string): void {
    this.items.delete(productId);
  }

  getTotal(): Money {
    let total = new Money(0, "JPY");
    for (const item of this.items.values()) {
      total = total.add(item.getSubtotal());
    }
    return total;
  }

  getItemCount(): number {
    let count = 0;
    for (const item of this.items.values()) {
      count += item.quantity;
    }
    return count;
  }

  getSummary(): string {
    const lines: string[] = ["=== Shopping Cart ==="];
    for (const item of this.items.values()) {
      lines.push(
        `  ${item.product.name} x ${item.quantity} = ${item.getSubtotal().format()}`
      );
    }
    lines.push(`  Total: ${this.getTotal().format()} (${this.getItemCount()} items)`);
    return lines.join("\n");
  }

  isEmpty(): boolean {
    return this.items.size === 0;
  }

  clear(): void {
    this.items.clear();
  }
}

// Usage example
const laptop = new Product("P001", "MacBook Pro", new Money(298000, "JPY"), 5, "Electronics");
const mouse = new Product("P002", "Magic Mouse", new Money(13800, "JPY"), 20, "Accessories");
const keyboard = new Product("P003", "HHKB", new Money(35200, "JPY"), 3, "Accessories");

const cart = new ShoppingCart();
cart.addItem(laptop, 1);
cart.addItem(mouse, 2);
cart.addItem(keyboard, 1);

console.log(cart.getSummary());
// === Shopping Cart ===
//   MacBook Pro x 1 = 298,000 JPY
//   Magic Mouse x 2 = 27,600 JPY
//   HHKB x 1 = 35,200 JPY
//   Total: 360,800 JPY (4 items)
```

---

## 9. Metaclasses and Reflection

```
Metaclass:
  -> "The class of a class"
  -> Defines the behavior of the class itself
  -> In Python, type is the default metaclass

Reflection:
  -> Inspect and manipulate class/object structure at runtime
  -> The foundation of metaprogramming
```

```python
# Python: controlling class behavior with a metaclass

class SingletonMeta(type):
    """Implement the singleton pattern as a metaclass"""
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class AppConfig(metaclass=SingletonMeta):
    def __init__(self):
        self.settings: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self.settings[key] = value

    def get(self, key: str, default: str = "") -> str:
        return self.settings.get(key, default)

# No matter how many times you instantiate, you get the same object
config1 = AppConfig()
config2 = AppConfig()
config1.set("debug", "true")
print(config1 is config2)         # True
print(config2.get("debug"))       # "true"


# Reflection example
class ValidationMeta(type):
    """Metaclass that auto-detects fields with validation"""

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        # Auto-collect methods prefixed with _validate_
        validators = {}
        for attr_name, attr_value in namespace.items():
            if attr_name.startswith("_validate_") and callable(attr_value):
                field_name = attr_name[len("_validate_"):]
                validators[field_name] = attr_value

        cls._validators = validators
        return cls

class ValidatedModel(metaclass=ValidationMeta):
    def __setattr__(self, name: str, value):
        validator = self.__class__._validators.get(name)
        if validator:
            validator(self, value)
        super().__setattr__(name, value)

class User(ValidatedModel):
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def _validate_name(self, value: str):
        if not isinstance(value, str) or len(value) == 0:
            raise ValueError("Name cannot be an empty string")

    def _validate_age(self, value: int):
        if not isinstance(value, int) or value < 0 or value > 150:
            raise ValueError(f"Age is out of range: {value}")

    def _validate_email(self, value: str):
        if not isinstance(value, str) or "@" not in value:
            raise ValueError(f"Invalid email address: {value}")

# Usage example
user = User("Tanaka", 30, "tanaka@example.com")
try:
    user.age = -5  # ValueError: Age is out of range: -5
except ValueError as e:
    print(e)
```

```java
// Java: a practical use of reflection

import java.lang.reflect.*;
import java.lang.annotation.*;

// Custom annotations
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
@interface JsonField {
    String name() default "";
}

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
@interface Required {}

public class SimpleJsonSerializer {

    public static String toJson(Object obj) throws IllegalAccessException {
        StringBuilder sb = new StringBuilder("{");
        Field[] fields = obj.getClass().getDeclaredFields();
        boolean first = true;

        for (Field field : fields) {
            field.setAccessible(true);

            JsonField jsonField = field.getAnnotation(JsonField.class);
            if (jsonField == null) continue;

            String name = jsonField.name().isEmpty()
                ? field.getName()
                : jsonField.name();

            Object value = field.get(obj);

            if (!first) sb.append(",");
            first = false;

            sb.append("\"").append(name).append("\":");
            if (value instanceof String) {
                sb.append("\"").append(value).append("\"");
            } else {
                sb.append(value);
            }
        }

        sb.append("}");
        return sb.toString();
    }

    public static void validate(Object obj) throws Exception {
        for (Field field : obj.getClass().getDeclaredFields()) {
            field.setAccessible(true);
            if (field.isAnnotationPresent(Required.class)) {
                Object value = field.get(obj);
                if (value == null || (value instanceof String && ((String) value).isEmpty())) {
                    throw new IllegalStateException(
                        field.getName() + " is a required field"
                    );
                }
            }
        }
    }
}

// Usage example
class UserDto {
    @JsonField(name = "user_name")
    @Required
    String name;

    @JsonField
    @Required
    String email;

    @JsonField
    int age;

    UserDto(String name, String email, int age) {
        this.name = name;
        this.email = email;
        this.age = age;
    }
}

UserDto user = new UserDto("Tanaka", "tanaka@example.com", 30);
SimpleJsonSerializer.validate(user);  // OK
String json = SimpleJsonSerializer.toJson(user);
// {"user_name":"Tanaka","email":"tanaka@example.com","age":30}
```

---

## 10. Kinds of Classes and Special Classes

```
+------------------+--------------------------------------+
| Kind             | Description                          |
+------------------+--------------------------------------+
| Concrete class   | A regular class. Can be instantiated |
+------------------+--------------------------------------+
| Abstract class   | Cannot be instantiated directly.     |
|                  | Provides shared features to subclasses|
+------------------+--------------------------------------+
| Interface        | Defines only method signatures       |
|                  | No implementations (Java 8+ allows default)|
+------------------+--------------------------------------+
| sealed class     | Restricts subclasses to a known set  |
|                  | Works well with pattern matching     |
+------------------+--------------------------------------+
| data class       | Holds data. Auto equals/hashCode     |
|                  | Kotlin data class, Java record       |
+------------------+--------------------------------------+
| enum class       | Defines a finite set of constant     |
|                  | instances                            |
+------------------+--------------------------------------+
| Inner class      | A class defined inside another class |
+------------------+--------------------------------------+
| Anonymous class  | A class without a name. One-off use  |
+------------------+--------------------------------------+
```

```kotlin
// Kotlin: leveraging various kinds of classes

// sealed class: the set of subclasses is closed
sealed class Shape {
    abstract fun area(): Double
    abstract fun perimeter(): Double

    data class Circle(val radius: Double) : Shape() {
        override fun area() = Math.PI * radius * radius
        override fun perimeter() = 2 * Math.PI * radius
    }

    data class Rectangle(val width: Double, val height: Double) : Shape() {
        override fun area() = width * height
        override fun perimeter() = 2 * (width + height)
    }

    data class Triangle(val a: Double, val b: Double, val c: Double) : Shape() {
        override fun area(): Double {
            val s = (a + b + c) / 2
            return Math.sqrt(s * (s - a) * (s - b) * (s - c))
        }
        override fun perimeter() = a + b + c
    }
}

// sealed class + when expression = exhaustive pattern matching
fun describeShape(shape: Shape): String = when (shape) {
    is Shape.Circle -> "Circle of radius ${shape.radius} (area: ${shape.area():.2f})"
    is Shape.Rectangle -> "Rectangle ${shape.width}x${shape.height}"
    is Shape.Triangle -> "Triangle (perimeter: ${shape.perimeter():.2f})"
    // All subclasses are covered, so else is not needed
}

// data class: value object
data class Address(
    val postalCode: String,
    val prefecture: String,
    val city: String,
    val street: String,
    val building: String? = null,
) {
    fun toSingleLine(): String {
        val parts = listOfNotNull(
            "Postal $postalCode",
            prefecture,
            city,
            street,
            building,
        )
        return parts.joinToString(" ")
    }
}

// enum class: enumeration
enum class OrderStatus(val label: String, val isFinal: Boolean) {
    PENDING("Order received", false),
    CONFIRMED("Confirmed", false),
    SHIPPING("Shipping", false),
    DELIVERED("Delivered", true),
    CANCELLED("Cancelled", true);

    fun canTransitionTo(next: OrderStatus): Boolean = when (this) {
        PENDING -> next == CONFIRMED || next == CANCELLED
        CONFIRMED -> next == SHIPPING || next == CANCELLED
        SHIPPING -> next == DELIVERED
        DELIVERED, CANCELLED -> false
    }
}

// object: singleton
object IdGenerator {
    private var counter = 0L

    @Synchronized
    fun nextId(): String {
        counter++
        return "ID-${counter.toString().padStart(8, '0')}"
    }
}

// companion object: equivalent to static members
class User private constructor(
    val id: String,
    val name: String,
    val email: String,
) {
    companion object {
        fun create(name: String, email: String): User {
            require(name.isNotBlank()) { "Name cannot be empty" }
            require("@" in email) { "A valid email address is required" }
            return User(IdGenerator.nextId(), name, email)
        }
    }
}

// Usage example
val shapes = listOf(
    Shape.Circle(5.0),
    Shape.Rectangle(3.0, 4.0),
    Shape.Triangle(3.0, 4.0, 5.0),
)
shapes.forEach { println(describeShape(it)) }

val addr = Address("100-0001", "Tokyo", "Chiyoda", "Chiyoda 1-1")
println(addr.toSingleLine())

val user = User.create("Taro Tanaka", "tanaka@example.com")
println("${user.id}: ${user.name}")
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is most important. Rather than only studying the theory, writing real code and observing its behavior will deepen your understanding.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping straight into advanced topics. We recommend fully understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Key Point |
|------|---------|
| Class | A blueprint. One per class in memory as metadata |
| Object | An instance. Many can exist on the heap |
| Three characteristics | State, behavior, and identity |
| Memory layout | vtable, header, field layout |
| Constructor | Initialization + establishing invariants |
| Builder | Build complex objects in stages |
| Static member | Belongs to the class. Utilities/factories |
| Value vs. reference type | Differences in copy semantics |
| Equality vs. identity | Correct implementation of equals/hashCode |
| Composition | A has-a relationship. More flexible than inheritance |
| Metaclass | Controls the behavior of the class |
| Kinds of classes | Concrete, abstract, sealed, data, enum |

---

## Recommended Next Guides

---

## References
1. Bloch, J. "Effective Java." 3rd Ed, Addison-Wesley, 2018.
2. Eckel, B. "Thinking in Java." 4th Ed, Prentice Hall, 2006.
3. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
4. Meyer, B. "Object-Oriented Software Construction." 2nd Ed, Prentice Hall, 1997.
5. Evans, E. "Domain-Driven Design." Addison-Wesley, 2003.
6. Fowler, M. "Patterns of Enterprise Application Architecture." Addison-Wesley, 2002.
