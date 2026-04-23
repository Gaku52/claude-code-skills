# Inheritance

> Inheritance is a mechanism that "creates a new class by taking over the functionality of an existing class." It is powerful but easy to misuse, and in modern OOP, the principle "prefer composition over inheritance" has become the norm.

## What You Will Learn in This Chapter

- [ ] Understand the mechanism of inheritance and its representation in memory
- [ ] Grasp appropriate uses of inheritance and its pitfalls
- [ ] Learn the differences between abstract classes and interfaces
- [ ] Understand the problems with multiple inheritance and the solutions in each language
- [ ] Practically grasp the criteria for choosing between inheritance and composition
- [ ] Learn how to utilize the Template Method pattern


## Prerequisites

Reading this guide will be easier if you have the following knowledge:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding the content of [Encapsulation](./00-encapsulation.md)

---

## 1. The Basics of Inheritance

```
Inheritance:
  -> A mechanism where a child class (subclass)
    takes over the fields and methods of a parent class (superclass)

  Animal (parent class)
  |-- name: string
  |-- sound(): string
  +-- move(): void
       ^ inheritance
  +-----+------+
  Dog          Cat
  |-- breed    |-- indoor
  +-- fetch()  +-- purr()

  Dog automatically has Animal's name, sound(), move()
  + adds its own breed, fetch()
```

### 1.1 Basic Inheritance Syntax (Comparison Across Languages)

```typescript
// TypeScript: basic inheritance
class Animal {
  constructor(
    protected name: string,
    protected age: number,
  ) {}

  speak(): string {
    return `${this.name} is making a sound`;
  }

  toString(): string {
    return `${this.name} (${this.age} years old)`;
  }
}

class Dog extends Animal {
  constructor(name: string, age: number, private breed: string) {
    super(name, age); // Call the parent's constructor
  }

  // Override (overwrite the parent's method)
  speak(): string {
    return `${this.name} "Woof!"`;
  }

  fetch(): string {
    return `${this.name} fetched the ball`;
  }
}

class Cat extends Animal {
  speak(): string {
    return `${this.name} "Meow"`;
  }
}

const dog = new Dog("ポチ", 3, "柴犬");
const cat = new Cat("タマ", 5);
console.log(dog.speak()); // ポチ "Woof!"
console.log(cat.speak()); // タマ "Meow"
```

```java
// Java: basic inheritance
public abstract class Vehicle {
    protected String name;
    protected int year;
    protected double fuelLevel;

    public Vehicle(String name, int year) {
        this.name = name;
        this.year = year;
        this.fuelLevel = 100.0;
    }

    // Common method
    public String getInfo() {
        return String.format("%s (%d model) fuel: %.1f%%", name, year, fuelLevel);
    }

    // Abstract method (must be implemented in subclass)
    public abstract double getFuelEfficiency();

    // Template method
    public final void startEngine() {
        if (fuelLevel <= 0) {
            System.out.println("No fuel");
            return;
        }
        performPreCheck();
        ignite();
        System.out.println(name + "'s engine has started");
    }

    protected void performPreCheck() {
        System.out.println("Performing basic check...");
    }

    protected abstract void ignite();
}

public class Car extends Vehicle {
    private int doorCount;

    public Car(String name, int year, int doorCount) {
        super(name, year);
        this.doorCount = doorCount;
    }

    @Override
    public double getFuelEfficiency() {
        return 15.0; // km/L
    }

    @Override
    protected void ignite() {
        System.out.println("Starter motor engaged");
    }

    @Override
    protected void performPreCheck() {
        super.performPreCheck(); // Also run the parent's processing
        System.out.println("Door lock check: " + doorCount + " doors");
    }
}

public class Motorcycle extends Vehicle {
    private boolean hasSidecar;

    public Motorcycle(String name, int year, boolean hasSidecar) {
        super(name, year);
        this.hasSidecar = hasSidecar;
    }

    @Override
    public double getFuelEfficiency() {
        return hasSidecar ? 20.0 : 30.0;
    }

    @Override
    protected void ignite() {
        System.out.println("Kick start");
    }
}

// Usage example
Vehicle car = new Car("Toyota Corolla", 2024, 4);
Vehicle bike = new Motorcycle("Honda CB400", 2023, false);

car.startEngine();
// Performing basic check...
// Door lock check: 4 doors
// Starter motor engaged
// Toyota Corolla's engine has started

bike.startEngine();
// Performing basic check...
// Kick start
// Honda CB400's engine has started
```

```python
# Python: basic inheritance
class Employee:
    """Base class for employees"""

    def __init__(self, name: str, employee_id: str, base_salary: float):
        self.name = name
        self.employee_id = employee_id
        self.base_salary = base_salary
        self._benefits: list[str] = ["Health insurance", "Welfare pension"]

    def calculate_pay(self) -> float:
        """Calculate monthly salary"""
        return self.base_salary

    def get_benefits(self) -> list[str]:
        """Get the list of benefits"""
        return self._benefits.copy()

    def __str__(self) -> str:
        return f"{self.name} (ID: {self.employee_id})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}', id='{self.employee_id}')"


class FullTimeEmployee(Employee):
    """Full-time employee"""

    def __init__(self, name: str, employee_id: str, base_salary: float,
                 bonus_rate: float = 0.2):
        super().__init__(name, employee_id, base_salary)
        self.bonus_rate = bonus_rate
        self._benefits.extend(["Retirement allowance", "Housing allowance"])

    def calculate_pay(self) -> float:
        """Base salary + bonus portion"""
        return self.base_salary * (1 + self.bonus_rate)

    def calculate_annual_bonus(self) -> float:
        """Annual bonus"""
        return self.base_salary * self.bonus_rate * 2  # summer and winter


class PartTimeEmployee(Employee):
    """Part-time employee"""

    def __init__(self, name: str, employee_id: str,
                 hourly_rate: float, hours_per_month: float):
        # base_salary is calculated as hourly rate x hours
        super().__init__(name, employee_id, hourly_rate * hours_per_month)
        self.hourly_rate = hourly_rate
        self.hours_per_month = hours_per_month
        # Part-time employees have limited benefits
        self._benefits = ["Health insurance"]

    def calculate_pay(self) -> float:
        """Hourly rate x hours"""
        return self.hourly_rate * self.hours_per_month


class Manager(FullTimeEmployee):
    """Manager (further inherited from full-time employee)"""

    def __init__(self, name: str, employee_id: str, base_salary: float,
                 bonus_rate: float = 0.3, team_size: int = 0):
        super().__init__(name, employee_id, base_salary, bonus_rate)
        self.team_size = team_size
        self._benefits.append("Managerial allowance")

    def calculate_pay(self) -> float:
        """Base salary + bonus + management allowance"""
        base_pay = super().calculate_pay()
        management_allowance = 50000 * (self.team_size // 5)  # 50,000 yen per 5 people
        return base_pay + management_allowance


# Usage example: working with polymorphism
employees: list[Employee] = [
    FullTimeEmployee("田中太郎", "FT001", 350000),
    PartTimeEmployee("鈴木花子", "PT001", 1200, 80),
    Manager("佐藤部長", "MG001", 500000, team_size=12),
]

for emp in employees:
    print(f"{emp}: ¥{emp.calculate_pay():,.0f}")
# 田中太郎 (ID: FT001): ¥420,000
# 鈴木花子 (ID: PT001): ¥96,000
# 佐藤部長 (ID: MG001): ¥750,000
```

### 1.2 Memory Layout of Inheritance

```
Representation of inherited objects in memory:

  Memory layout of a Manager object:
  +--------------------------------------+
  | vptr -> Manager's vtable             | <- virtual function table pointer
  +--------------------------------------+
  | name: "佐藤部長"                     | <- Employee's fields
  | employee_id: "MG001"                 |
  | base_salary: 500000                  |
  | _benefits: [...]                     |
  +--------------------------------------+
  | bonus_rate: 0.3                      | <- FullTimeEmployee's field
  +--------------------------------------+
  | team_size: 12                        | <- Manager's field
  +--------------------------------------+

  Inheritance chain:
  Employee -> FullTimeEmployee -> Manager

  Fields at each level are laid out contiguously
  -> The deeper the inheritance, the larger the object size
```

---

## 2. Method Override and super

```
Override:
  -> Redefining a parent class method in a child class
  -> Dynamic dispatch: the method of the actual type is called at runtime

Role of super:
  -> Explicitly calls the parent class's constructor/method
  -> "Parent's processing + additional processing" pattern

Three override patterns:
  1. Complete replacement: Entirely replace the parent's method with a new implementation
  2. Extension: Call super() and then add extra processing
  3. Conditional branching: Use the parent's implementation or your own depending on conditions
```

```python
# Python: How to use super() and three override patterns
class Shape:
    def __init__(self, color: str = "black"):
        self.color = color

    def area(self) -> float:
        raise NotImplementedError

    def describe(self) -> str:
        return f"A {self.color} {type(self).__name__}"

    def validate(self) -> bool:
        """Validate whether the shape is valid"""
        return True

class Circle(Shape):
    def __init__(self, radius: float, color: str = "black"):
        super().__init__(color)  # Initialize parent
        self.radius = radius

    # Pattern 1: Complete replacement
    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    # Pattern 2: Extension (super() + additional processing)
    def describe(self) -> str:
        return f"{super().describe()}, radius {self.radius}"

    # Pattern 3: Conditional branching
    def validate(self) -> bool:
        if self.radius <= 0:
            return False
        return super().validate()

class Rectangle(Shape):
    def __init__(self, width: float, height: float, color: str = "black"):
        super().__init__(color)
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def describe(self) -> str:
        return f"{super().describe()}, {self.width}x{self.height}"

    def validate(self) -> bool:
        if self.width <= 0 or self.height <= 0:
            return False
        return super().validate()

class Square(Rectangle):
    """A square is a special case of rectangle (but caution is required)"""
    def __init__(self, side: float, color: str = "black"):
        super().__init__(side, side, color)

    def describe(self) -> str:
        # Completely replace the parent's describe
        return f"A {self.color} square, side {self.width}"
```

### 2.1 Advanced Usage of super()

```python
# Python: super() in cooperative multiple inheritance
class Loggable:
    """Mixin that provides logging functionality"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log: list[str] = []

    def log(self, message: str) -> None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.append(f"[{timestamp}] {message}")

    def get_log(self) -> list[str]:
        return self._log.copy()


class Validatable:
    """Mixin that provides validation functionality"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._errors: list[str] = []

    def add_error(self, error: str) -> None:
        self._errors.append(error)

    def is_valid(self) -> bool:
        self._errors.clear()
        self.validate()
        return len(self._errors) == 0

    def validate(self) -> None:
        """Override in subclass to add validation rules"""
        pass


class Product(Loggable, Validatable):
    """Product class (inherits from multiple Mixins)"""
    def __init__(self, name: str, price: float):
        super().__init__()  # All __init__ are called according to MRO
        self.name = name
        self.price = price
        self.log(f"Product created: {name}")

    def validate(self) -> None:
        super().validate()  # Call Validatable.validate()
        if not self.name:
            self.add_error("Product name is required")
        if self.price < 0:
            self.add_error("Price must be 0 or more")
        if self.price > 10_000_000:
            self.add_error("Price exceeds the upper limit")

    def update_price(self, new_price: float) -> None:
        old_price = self.price
        self.price = new_price
        self.log(f"Price changed: {old_price} -> {new_price}")


# Usage example
product = Product("Laptop PC", 150000)
print(product.is_valid())    # True
print(product.get_log())     # ['[2024-...] Product created: Laptop PC']

product.update_price(-100)
print(product.is_valid())    # False (price is negative)

# Check the MRO
print(Product.__mro__)
# (Product, Loggable, Validatable, object)
```

```typescript
// TypeScript: advanced patterns for using super

abstract class Component {
  protected children: Component[] = [];
  protected parent: Component | null = null;

  constructor(protected id: string) {}

  addChild(child: Component): void {
    child.parent = this;
    this.children.push(child);
  }

  // Template method
  render(): string {
    const self = this.renderSelf();
    const children = this.children.map(c => c.render()).join("\n");
    return children ? `${self}\n${children}` : self;
  }

  protected abstract renderSelf(): string;

  // Lifecycle hooks
  mount(): void {
    this.onBeforeMount();
    this.children.forEach(c => c.mount());
    this.onMounted();
  }

  protected onBeforeMount(): void {}
  protected onMounted(): void {}
}

class Panel extends Component {
  constructor(id: string, private title: string) {
    super(id);
  }

  protected renderSelf(): string {
    return `<panel id="${this.id}" title="${this.title}">`;
  }

  protected onMounted(): void {
    console.log(`Panel "${this.title}" mounted`);
  }
}

class Button extends Component {
  constructor(id: string, private label: string, private onClick: () => void) {
    super(id);
  }

  protected renderSelf(): string {
    return `<button id="${this.id}">${this.label}</button>`;
  }

  // Call super.mount() and then add an event handler
  mount(): void {
    super.mount();
    console.log(`Registered click handler for Button "${this.label}"`);
  }
}

class Form extends Panel {
  private fields: Map<string, string> = new Map();

  constructor(id: string, title: string) {
    super(id, title);
  }

  addField(name: string, defaultValue: string = ""): void {
    this.fields.set(name, defaultValue);
  }

  // Extend the parent's renderSelf
  protected renderSelf(): string {
    const base = super.renderSelf();
    const fieldsHtml = Array.from(this.fields.entries())
      .map(([name, value]) => `  <input name="${name}" value="${value}" />`)
      .join("\n");
    return `${base}\n${fieldsHtml}`;
  }

  protected onMounted(): void {
    super.onMounted(); // Call Panel.onMounted()
    console.log(`Form fields: ${this.fields.size}`);
  }
}

// Usage example
const form = new Form("login-form", "Login");
form.addField("username");
form.addField("password");
form.addChild(new Button("submit-btn", "Login", () => {}));
form.mount();
// Panel "Login" mounted
// Form fields: 2
// Registered click handler for Button "Login"
```

---

## 3. Kinds of Inheritance

```
Single Inheritance:
  -> Only one parent class can be inherited
  -> Java, C#, Swift, Kotlin, Ruby
  -> Simple, but expressiveness is limited

Multiple Inheritance:
  -> Multiple parent classes can be inherited
  -> C++, Python
  -> Powerful, but the diamond problem arises

  +---------+
  | Animal  | <- diamond problem
  +----+----+
  +----+----+
  v         v
+-----+  +------+
| Fly |  | Swim |
+--+--+  +--+---+
   +----+---+
        v
  +----------+
  | FlyFish  | <- From which should Animal's methods be inherited?
  +----------+

Multiple implementation via interfaces:
  -> Java, C#, TypeScript, Kotlin, Swift
  -> Do not carry implementations (Java 8+ default methods are an exception)
  -> Partially avoids the diamond problem

Mixin / Trait:
  -> Ruby (module), Scala (trait), Rust (trait), Kotlin (interface + default)
  -> Contains implementations, but state (fields) is restricted
  -> Safely provides the benefits of multiple inheritance
```

### 3.1 Python's MRO (Method Resolution Order)

```python
# Python: solving the diamond problem with MRO (Method Resolution Order)
class Animal:
    def move(self):
        return "Move"

    def breathe(self):
        return "Breathe"

class Flyer(Animal):
    def move(self):
        return "Fly"

    def take_off(self):
        return "Take off"

class Swimmer(Animal):
    def move(self):
        return "Swim"

    def dive(self):
        return "Dive"

class FlyingFish(Flyer, Swimmer):
    pass

fish = FlyingFish()
print(fish.move())      # "Fly" (MRO: FlyingFish -> Flyer -> Swimmer -> Animal)
print(fish.breathe())   # "Breathe" (inherited from Animal)
print(fish.take_off())  # "Take off" (inherited from Flyer)
print(fish.dive())      # "Dive" (inherited from Swimmer)

# Check the MRO
print(FlyingFish.__mro__)
# (FlyingFish, Flyer, Swimmer, Animal, object)
# -> The order is determined by the C3 linearization algorithm
```

### 3.2 Details of the C3 Linearization Algorithm

```
C3 Linearization:
  The algorithm Python uses to determine MRO

Rules:
  1. A child class always comes before parent classes
  2. When there are multiple parents, preserve the declaration order
  3. Contradictory orderings are not allowed (TypeError is raised)

Example: For class D(B, C) where B(A), C(A)
  L[D] = D + merge(L[B], L[C], [B, C])
  L[B] = B, A, object
  L[C] = C, A, object
  merge([B, A, object], [C, A, object], [B, C])
  = B + merge([A, object], [C, A, object], [C])
  = B, C + merge([A, object], [A, object])
  = B, C, A + merge([object], [object])
  = B, C, A, object
  -> D's MRO = [D, B, C, A, object]
```

```python
# Example of contradictory MRO (TypeError is raised)
class A:
    pass

class B(A):
    pass

class C(A, B):  # A comes before B, but B inherits from A
    pass
# TypeError: Cannot create a consistent method resolution order (MRO)
# -> because the ordering of A and B contradicts
```

### 3.3 How Each Language Handles Multiple Inheritance

```java
// Java: multiple implementation via default methods in interfaces
interface Flyable {
    default String move() {
        return "Fly";
    }

    String altitude();
}

interface Swimmable {
    default String move() {
        return "Swim";
    }

    String depth();
}

// When implementing both interfaces,
// default methods with the same name must be explicitly overridden
class Duck implements Flyable, Swimmable {
    @Override
    public String move() {
        // Choose one, or write your own implementation
        return "Can " + Flyable.super.move() + " and also " + Swimmable.super.move();
    }

    @Override
    public String altitude() {
        return "100m";
    }

    @Override
    public String depth() {
        return "5m";
    }
}

Duck duck = new Duck();
System.out.println(duck.move()); // "Can Fly and also Swim"
```

```kotlin
// Kotlin: default implementations in interfaces
interface Logger {
    fun log(message: String) {
        println("[LOG] $message")
    }
}

interface Auditable {
    fun audit(action: String) {
        println("[AUDIT] $action")
    }

    fun log(message: String) {
        println("[AUDIT-LOG] $message")
    }
}

class UserService : Logger, Auditable {
    // Methods with the same name collide, so we must explicitly override
    override fun log(message: String) {
        super<Logger>.log(message)      // Call Logger's implementation
        super<Auditable>.audit(message) // Also call Auditable's audit
    }

    fun createUser(name: String) {
        log("User created: $name")
    }
}
```

```ruby
# Ruby: alternative to multiple inheritance using Modules (Mixins)
module Serializable
  def serialize
    instance_variables.each_with_object({}) do |var, hash|
      hash[var.to_s.delete('@')] = instance_variable_get(var)
    end
  end

  def to_json
    require 'json'
    JSON.generate(serialize)
  end
end

module Cacheable
  def cache_key
    "#{self.class.name}:#{object_id}"
  end

  def cached?
    @_cached ||= false
  end

  def mark_cached!
    @_cached = true
  end
end

module Auditable
  def audit_trail
    @_audit_trail ||= []
  end

  def record_change(field, old_value, new_value)
    audit_trail << {
      field: field,
      old: old_value,
      new: new_value,
      at: Time.now
    }
  end
end

class User
  include Serializable
  include Cacheable
  include Auditable

  attr_reader :name, :email

  def initialize(name, email)
    @name = name
    @email = email
  end

  def update_email(new_email)
    record_change(:email, @email, new_email)
    @email = new_email
  end
end

user = User.new("田中", "tanaka@example.com")
puts user.to_json        # {"name":"田中","email":"tanaka@example.com"}
puts user.cache_key      # "User:12345"
user.update_email("new@example.com")
puts user.audit_trail    # [{field: :email, old: "tanaka@...", new: "new@...", ...}]

# Check the inheritance chain
puts User.ancestors
# [User, Auditable, Cacheable, Serializable, Object, Kernel, BasicObject]
```

---

## 4. Pitfalls of Inheritance

```
Problem 1: Fragile Base Class Problem
  -> A change to the parent class breaks child classes

Problem 2: Inappropriate is-a relationships
  -> Is a square is-a rectangle? (violates the Liskov Substitution Principle)

Problem 3: Deep inheritance hierarchies
  -> Inheritance of 3 or more levels is hard to understand
  -> Entity -> LivingEntity -> Animal -> Mammal -> Dog -> GuideDog
  -> Changes at each layer affect everything below

Problem 4: Breakdown of encapsulation by inheritance
  -> A child class depends on the parent's implementation details
  -> Direct access to protected fields

Problem 5: The Gorilla-Banana problem
  "All I wanted was a banana, but I got a gorilla holding the banana
   and the entire jungle along with it."
  -> When you inherit, all unnecessary functionality comes along
```

### 4.1 The Fragile Base Class Problem

```java
// Example of the fragile base class problem (from Effective Java Item 18)
public class HashSet<E> {
    private int addCount = 0;

    public boolean add(E e) {
        addCount++;
        // ... actual addition processing
        return true;
    }

    public boolean addAll(Collection<E> c) {
        // Implementation that calls add() internally
        for (E e : c) add(e);
        return true;
    }

    public int getAddCount() { return addCount; }
}

// A problematic subclass
public class InstrumentedHashSet<E> extends HashSet<E> {
    private int addCount = 0;

    @Override
    public boolean add(E e) {
        addCount++;
        return super.add(e);
    }

    @Override
    public boolean addAll(Collection<E> c) {
        addCount += c.size();
        return super.addAll(c); // super.addAll() calls add()!
    }
    // addAll({a, b, c}) -> addCount = 6 (expected 3)
    // -> super.addAll() internally calls add(), resulting in double counting
}
```

```java
// Solution: use composition (recommended by Effective Java)
public class InstrumentedSet<E> {
    private final Set<E> set;  // composition
    private int addCount = 0;

    public InstrumentedSet(Set<E> set) {
        this.set = set;
    }

    public boolean add(E e) {
        addCount++;
        return set.add(e);  // delegation
    }

    public boolean addAll(Collection<E> c) {
        addCount += c.size();
        return set.addAll(c);  // does not depend on set's internal implementation
    }

    public int getAddCount() { return addCount; }

    // Delegate the necessary Set methods
    public boolean contains(Object o) { return set.contains(o); }
    public int size() { return set.size(); }
    public Iterator<E> iterator() { return set.iterator(); }
}

// Usage example: works with any Set implementation
InstrumentedSet<String> s1 = new InstrumentedSet<>(new HashSet<>());
InstrumentedSet<String> s2 = new InstrumentedSet<>(new TreeSet<>());
InstrumentedSet<String> s3 = new InstrumentedSet<>(new LinkedHashSet<>());
```

### 4.2 Inappropriate is-a Relationships (The Square-Rectangle Problem)

```typescript
// BAD: Square extends Rectangle: a classic LSP violation
class Rectangle {
  constructor(protected width: number, protected height: number) {}

  setWidth(w: number): void {
    this.width = w;
  }

  setHeight(h: number): void {
    this.height = h;
  }

  area(): number {
    return this.width * this.height;
  }
}

class Square extends Rectangle {
  constructor(side: number) {
    super(side, side);
  }

  // For a square, changing the width also changes the height (different behavior from parent)
  setWidth(w: number): void {
    this.width = w;
    this.height = w; // <- side effect not present in parent class
  }

  setHeight(h: number): void {
    this.width = h;
    this.height = h; // <- side effect not present in parent class
  }
}

// Problem: when used as a Rectangle, it does not behave as expected
function doubleWidth(rect: Rectangle): void {
  const originalHeight = rect.area() / rect.area(); // preserve original height
  rect.setWidth(rect.area() / 10); // intended to change only the width
  // For a Square, the height changes too!
}

// GOOD: abstract via a common interface
interface Shape {
  area(): number;
  perimeter(): number;
}

class ImmutableRectangle implements Shape {
  constructor(readonly width: number, readonly height: number) {}
  area(): number { return this.width * this.height; }
  perimeter(): number { return 2 * (this.width + this.height); }
}

class ImmutableSquare implements Shape {
  constructor(readonly side: number) {}
  area(): number { return this.side ** 2; }
  perimeter(): number { return 4 * this.side; }
}
```

### 4.3 The Problem of Deep Inheritance Hierarchies

```
Risks of deep inheritance hierarchies:

  Level 0: Entity
  Level 1: +-- LivingEntity
  Level 2:     +-- Animal
  Level 3:         +-- Mammal
  Level 4:             +-- Canine
  Level 5:                 +-- Dog
  Level 6:                     +-- GuideDog

  Problems:
  1. Changing Level 2 (Animal) -> affects all of Levels 3-6
  2. A bug in GuideDog may have its root cause at Level 1
  3. When adding a new kind of dog, it's unclear at which level to add it
  4. Testing requires setup for all levels

  Recommendation: up to 2-3 levels maximum
  -> Beyond that, switch to composition
```

```python
# Example of improving a deep inheritance hierarchy with composition

# BAD: deep inheritance hierarchy
class Entity:
    def __init__(self, id: str):
        self.id = id

class LivingEntity(Entity):
    def __init__(self, id: str, health: float):
        super().__init__(id)
        self.health = health

class Animal(LivingEntity):
    def __init__(self, id: str, health: float, species: str):
        super().__init__(id, health)
        self.species = species

class Pet(Animal):
    def __init__(self, id: str, health: float, species: str, owner: str):
        super().__init__(id, health, species)
        self.owner = owner

class Dog(Pet):
    def __init__(self, id: str, health: float, owner: str, breed: str):
        super().__init__(id, health, "Dog", owner)
        self.breed = breed

class GuideDog(Dog):
    def __init__(self, id: str, health: float, owner: str, breed: str,
                 handler: str, certification: str):
        super().__init__(id, health, owner, breed)
        self.handler = handler
        self.certification = certification


# GOOD: improved with composition (shallow inheritance + functions as components)
from dataclasses import dataclass
from typing import Optional

@dataclass
class Identity:
    id: str
    created_at: Optional[str] = None

@dataclass
class HealthStatus:
    current_hp: float
    max_hp: float

    @property
    def is_alive(self) -> bool:
        return self.current_hp > 0

    def take_damage(self, amount: float) -> None:
        self.current_hp = max(0, self.current_hp - amount)

@dataclass
class OwnerInfo:
    owner_name: str
    owner_contact: str

@dataclass
class GuideDogCertification:
    handler_name: str
    certification_id: str
    expires_at: str

class AnimalV2:
    """Shallow structure: assemble functionality through composition"""
    def __init__(self, identity: Identity, species: str, breed: str):
        self.identity = identity
        self.species = species
        self.breed = breed
        self.health: Optional[HealthStatus] = None
        self.owner: Optional[OwnerInfo] = None
        self.guide_cert: Optional[GuideDogCertification] = None

    def is_guide_dog(self) -> bool:
        return self.guide_cert is not None

    def is_pet(self) -> bool:
        return self.owner is not None


# Usage example
guide_dog = AnimalV2(
    identity=Identity(id="GD-001"),
    species="Dog",
    breed="Labrador",
)
guide_dog.health = HealthStatus(current_hp=100, max_hp=100)
guide_dog.owner = OwnerInfo(owner_name="佐藤", owner_contact="090-XXXX")
guide_dog.guide_cert = GuideDogCertification(
    handler_name="田中",
    certification_id="CERT-2024-001",
    expires_at="2026-12-31",
)
```

---

## 5. Abstract Classes

```
Abstract class:
  -> A class that cannot be instantiated
  -> Defines common implementation + implementation obligations for subclasses
  -> Foundation for the "Template Method pattern"

Uses:
  - Provides common fields and some method implementations
  - Forces subclasses to implement specific methods
  - When the is-a relationship is clear

Abstract class vs interface:
  Abstract class: "what it is" + common implementation
  Interface: only "what it can do"
```

### 5.1 Basics of Abstract Classes

```typescript
// TypeScript: abstract classes
abstract class DatabaseConnection {
  protected connected: boolean = false;
  protected queryCount: number = 0;

  // Common implementation
  async query(sql: string): Promise<any[]> {
    if (!this.connected) {
      await this.connect();
    }
    this.queryCount++;
    const startTime = Date.now();
    const result = await this.executeQuery(sql);
    const duration = Date.now() - startTime;
    console.log(`Query #${this.queryCount} took ${duration}ms`);
    return result;
  }

  // Transaction management (template method)
  async withTransaction<T>(fn: () => Promise<T>): Promise<T> {
    await this.beginTransaction();
    try {
      const result = await fn();
      await this.commitTransaction();
      return result;
    } catch (error) {
      await this.rollbackTransaction();
      throw error;
    }
  }

  // Abstract methods that subclasses must implement
  abstract connect(): Promise<void>;
  abstract disconnect(): Promise<void>;
  protected abstract executeQuery(sql: string): Promise<any[]>;
  protected abstract beginTransaction(): Promise<void>;
  protected abstract commitTransaction(): Promise<void>;
  protected abstract rollbackTransaction(): Promise<void>;
}

class PostgresConnection extends DatabaseConnection {
  private client: any; // pg.Client

  async connect(): Promise<void> {
    // PostgreSQL-specific connection handling
    this.client = {}; // new pg.Client(connectionString)
    // await this.client.connect();
    this.connected = true;
    console.log("Connected to PostgreSQL");
  }

  async disconnect(): Promise<void> {
    // await this.client.end();
    this.connected = false;
    console.log("Disconnected from PostgreSQL");
  }

  protected async executeQuery(sql: string): Promise<any[]> {
    // const result = await this.client.query(sql);
    // return result.rows;
    console.log(`PostgreSQL exec: ${sql}`);
    return [];
  }

  protected async beginTransaction(): Promise<void> {
    await this.executeQuery("BEGIN");
  }

  protected async commitTransaction(): Promise<void> {
    await this.executeQuery("COMMIT");
  }

  protected async rollbackTransaction(): Promise<void> {
    await this.executeQuery("ROLLBACK");
  }
}

class SQLiteConnection extends DatabaseConnection {
  private db: any;

  async connect(): Promise<void> {
    // SQLite-specific connection handling
    this.db = {}; // new sqlite3.Database(path)
    this.connected = true;
    console.log("Connected to SQLite");
  }

  async disconnect(): Promise<void> {
    // this.db.close();
    this.connected = false;
  }

  protected async executeQuery(sql: string): Promise<any[]> {
    console.log(`SQLite exec: ${sql}`);
    return [];
  }

  protected async beginTransaction(): Promise<void> {
    await this.executeQuery("BEGIN TRANSACTION");
  }

  protected async commitTransaction(): Promise<void> {
    await this.executeQuery("COMMIT TRANSACTION");
  }

  protected async rollbackTransaction(): Promise<void> {
    await this.executeQuery("ROLLBACK TRANSACTION");
  }
}
```

### 5.2 The Template Method Pattern

```python
# Python: a practical example of the Template Method pattern
from abc import ABC, abstractmethod
from typing import Any
import time


class DataPipeline(ABC):
    """Abstract base class for a data processing pipeline"""

    def run(self, source: str) -> dict[str, Any]:
        """Template method: defines the flow of processing"""
        start_time = time.time()

        # 1. Data acquisition
        raw_data = self.extract(source)
        print(f"  acquired: {len(raw_data)} items")

        # 2. Validation
        valid_data = self.validate(raw_data)
        print(f"  valid: {len(valid_data)} items")

        # 3. Data transformation
        transformed = self.transform(valid_data)
        print(f"  transformation complete")

        # 4. Data saving
        result = self.load(transformed)
        print(f"  save complete")

        # 5. Post-processing (optional: hook)
        self.on_complete(result)

        elapsed = time.time() - start_time
        return {
            "source": source,
            "input_count": len(raw_data),
            "output_count": len(valid_data),
            "elapsed_seconds": round(elapsed, 2),
            "result": result,
        }

    # Required: subclasses must implement
    @abstractmethod
    def extract(self, source: str) -> list[dict]:
        """Acquire data from the data source"""
        ...

    @abstractmethod
    def transform(self, data: list[dict]) -> list[dict]:
        """Transform the data"""
        ...

    @abstractmethod
    def load(self, data: list[dict]) -> Any:
        """Write data to the destination"""
        ...

    # Optional: has a default implementation (can be overridden)
    def validate(self, data: list[dict]) -> list[dict]:
        """Default validation (excludes None/empty)"""
        return [d for d in data if d]

    def on_complete(self, result: Any) -> None:
        """Hook at completion (by default, does nothing)"""
        pass


class CsvToJsonPipeline(DataPipeline):
    """Pipeline that reads a CSV file and converts it to JSON"""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def extract(self, source: str) -> list[dict]:
        import csv
        with open(source, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def validate(self, data: list[dict]) -> list[dict]:
        # Parent's validation + additional rules
        valid = super().validate(data)
        return [d for d in valid if all(v.strip() for v in d.values())]

    def transform(self, data: list[dict]) -> list[dict]:
        # Convert numeric string fields
        for row in data:
            for key, value in row.items():
                try:
                    row[key] = int(value)
                except (ValueError, TypeError):
                    try:
                        row[key] = float(value)
                    except (ValueError, TypeError):
                        pass
        return data

    def load(self, data: list[dict]) -> str:
        import json
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return self.output_path

    def on_complete(self, result: Any) -> None:
        print(f"  -> saved to {result}")


class ApiToDbPipeline(DataPipeline):
    """Pipeline that fetches data from a REST API and saves it to a DB"""

    def __init__(self, db_connection: Any):
        self.db = db_connection

    def extract(self, source: str) -> list[dict]:
        import urllib.request
        import json
        with urllib.request.urlopen(source) as response:
            return json.loads(response.read())

    def transform(self, data: list[dict]) -> list[dict]:
        # Map the API response to the DB schema
        return [
            {
                "external_id": item.get("id"),
                "name": item.get("name", "").strip(),
                "email": item.get("email", "").lower(),
                "created_at": item.get("created_at"),
            }
            for item in data
        ]

    def load(self, data: list[dict]) -> int:
        # Bulk save to the DB
        count = 0
        for row in data:
            # self.db.insert("users", row)
            count += 1
        return count

    def on_complete(self, result: Any) -> None:
        print(f"  -> saved {result} rows to the DB")
```

---

## 6. Inheritance vs. Composition

```
When to use inheritance:
  OK Clear is-a relationship
  OK Subclass fully satisfies the parent's interface (LSP)
  OK Framework requires inheritance (Android's Activity, etc.)
  OK Template Method pattern

When to avoid inheritance:
  NG The only goal is code reuse
  NG has-a relationships (use composition)
  NG Requires 3 or more levels of inheritance
  NG You only want to use some of the parent class's methods

Decision criterion:
  "Can this subclass be used as a replacement in every place
   the parent class is used?"
  -> Yes -> inheritance is appropriate
  -> No  -> use composition
```

### 6.1 Practical Comparison: Inheritance vs. Composition

```typescript
// BAD: misuse of inheritance - inheritance for code reuse
class ArrayList<T> {
  protected items: T[] = [];

  add(item: T): void {
    this.items.push(item);
  }

  get(index: number): T {
    return this.items[index];
  }

  size(): number {
    return this.items.length;
  }

  remove(index: number): T {
    return this.items.splice(index, 1)[0];
  }
}

// Stack is not an ArrayList (there is no is-a relationship)
// A Stack only supports "add/remove at the top"; random access is unnecessary
class Stack<T> extends ArrayList<T> {
  push(item: T): void {
    this.add(item);
  }

  pop(): T | undefined {
    if (this.size() === 0) return undefined;
    return this.remove(this.size() - 1);
  }

  peek(): T | undefined {
    if (this.size() === 0) return undefined;
    return this.get(this.size() - 1);
  }

  // Problem: get(), remove() etc. become publicly exposed
  // stack.get(0) or stack.remove(3) are possible = breaks the Stack contract
}


// GOOD: implement with composition
class StackV2<T> {
  private items: T[] = []; // hide the internal implementation

  push(item: T): void {
    this.items.push(item);
  }

  pop(): T | undefined {
    return this.items.pop();
  }

  peek(): T | undefined {
    return this.items[this.items.length - 1];
  }

  size(): number {
    return this.items.length;
  }

  isEmpty(): boolean {
    return this.items.length === 0;
  }

  // get(), remove() are not exposed = the Stack contract is preserved
}
```

```python
# Inheritance vs. composition: designing a game character

# BAD: inheritance - combinations explode
class Character:
    pass

class Warrior(Character):
    def attack(self):
        return "Attack with sword"

class Mage(Character):
    def cast_spell(self):
        return "Cast a spell"

class Archer(Character):
    def shoot(self):
        return "Attack with bow"

# I want a magic warrior -> WarriorMage? How do I inherit?
# A mage who uses a bow -> MageArcher?
# A character who can do everything -> WarriorMageArcher??
# -> Combinations explode as 2^n


# GOOD: composition - assemble abilities as parts
from abc import ABC, abstractmethod
from typing import Optional


class Ability(ABC):
    """Abstract base class for an ability"""
    @abstractmethod
    def use(self, user_name: str) -> str:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...


class SwordSkill(Ability):
    def __init__(self, damage: int = 10):
        self.damage = damage

    def use(self, user_name: str) -> str:
        return f"{user_name} attacks with a sword! {self.damage} damage"

    def get_name(self) -> str:
        return "Swordsmanship"


class MagicSkill(Ability):
    def __init__(self, mana_cost: int = 5):
        self.mana_cost = mana_cost

    def use(self, user_name: str) -> str:
        return f"{user_name} cast a spell! MP-{self.mana_cost}"

    def get_name(self) -> str:
        return "Magic"


class ArcherySkill(Ability):
    def __init__(self, range_bonus: int = 20):
        self.range_bonus = range_bonus

    def use(self, user_name: str) -> str:
        return f"{user_name} attacks with a bow! Range+{self.range_bonus}"

    def get_name(self) -> str:
        return "Archery"


class HealingSkill(Ability):
    def __init__(self, heal_amount: int = 15):
        self.heal_amount = heal_amount

    def use(self, user_name: str) -> str:
        return f"{user_name} heals! HP+{self.heal_amount}"

    def get_name(self) -> str:
        return "Healing"


class GameCharacter:
    """A composition-based character"""
    def __init__(self, name: str, hp: int = 100, mp: int = 50):
        self.name = name
        self.hp = hp
        self.mp = mp
        self._abilities: list[Ability] = []

    def add_ability(self, ability: Ability) -> "GameCharacter":
        """Supports method chaining"""
        self._abilities.append(ability)
        return self

    def use_ability(self, index: int) -> str:
        if 0 <= index < len(self._abilities):
            return self._abilities[index].use(self.name)
        return f"Ability {index} does not exist"

    def list_abilities(self) -> list[str]:
        return [a.get_name() for a in self._abilities]

    def __str__(self) -> str:
        abilities = ", ".join(self.list_abilities())
        return f"{self.name} (HP:{self.hp} MP:{self.mp}) [{abilities}]"


# Can be freely combined
warrior = GameCharacter("Warrior", hp=150).add_ability(SwordSkill(damage=15))
mage = GameCharacter("Mage", mp=100).add_ability(MagicSkill()).add_ability(HealingSkill())
magic_warrior = (GameCharacter("Magic Warrior", hp=120, mp=70)
    .add_ability(SwordSkill(damage=12))
    .add_ability(MagicSkill(mana_cost=8)))
all_rounder = (GameCharacter("All-rounder", hp=100, mp=80)
    .add_ability(SwordSkill())
    .add_ability(MagicSkill())
    .add_ability(ArcherySkill())
    .add_ability(HealingSkill()))

print(all_rounder)
# All-rounder (HP:100 MP:80) [Swordsmanship, Magic, Archery, Healing]
print(all_rounder.use_ability(1))
# All-rounder cast a spell! MP-5
```

### 6.2 The Delegation Pattern

```kotlin
// Kotlin: delegation via the by keyword
interface Printer {
    fun print(content: String)
}

interface Scanner {
    fun scan(): String
}

class LaserPrinter : Printer {
    override fun print(content: String) {
        println("Laser print: $content")
    }
}

class FlatbedScanner : Scanner {
    override fun scan(): String {
        return "Scanned data"
    }
}

// Kotlin's by keyword lets you write delegation concisely
class MultiFunctionDevice(
    printer: Printer,
    scanner: Scanner
) : Printer by printer, Scanner by scanner {
    // print() and scan() are automatically delegated
    // They can also be overridden as needed

    fun copyDocument() {
        val data = scan()
        print(data)
    }
}

val device = MultiFunctionDevice(LaserPrinter(), FlatbedScanner())
device.print("Hello")    // Laser print: Hello
device.scan()             // Scanned data
device.copyDocument()     // scan -> print
```

---

## 7. Best Practices for Inheritance

### 7.1 Checklist When Using Inheritance

```
Checklist before introducing inheritance:

[] Does an is-a relationship hold naturally?
  "Dog is-a Animal" -> OK
  "Stack is-a ArrayList" -> NG

[] Does it satisfy the Liskov Substitution Principle?
  "In every place the parent class is used, can the subclass be substituted and still work correctly?"
  -> Square extends Rectangle is NG (the side effects of setWidth differ)

[] Do all parent-class methods make sense in the subclass?
  -> Avoiding the "Gorilla-Banana problem"

[] Does the inheritance hierarchy fit within 3 levels?
  -> 3 or more levels -> consider composition

[] Is the parent class stable?
  -> Frequently changed parent class -> risk of fragile base class problem

[] Is it easy to test?
  -> Requires huge setup of the parent class -> composition is recommended
```

### 7.2 Controlling Inheritance with sealed / final

```typescript
// TypeScript: patterns for restricting inheritance

// Intentionally prohibit inheritance (since TypeScript has no final, done by convention)
class Configuration {
  // There is no @final decorator, so indicate intent via a comment
  /** @final Please do not inherit from this class */
  private constructor(private readonly settings: Map<string, string>) {}

  static create(settings: Record<string, string>): Configuration {
    return new Configuration(new Map(Object.entries(settings)));
  }

  get(key: string): string | undefined {
    return this.settings.get(key);
  }
}
```

```java
// Java: inheritance control via final and sealed

// final: completely prohibits inheritance
public final class ImmutablePoint {
    private final double x;
    private final double y;

    public ImmutablePoint(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() { return x; }
    public double getY() { return y; }

    public double distanceTo(ImmutablePoint other) {
        return Math.sqrt(
            Math.pow(this.x - other.x, 2) +
            Math.pow(this.y - other.y, 2)
        );
    }
}
// class ExtendedPoint extends ImmutablePoint {} // compile error!

// sealed (Java 17+): only permitted classes can inherit
public sealed class Shape permits Circle, Rectangle, Triangle {
    public abstract double area();
}

public final class Circle extends Shape {
    private final double radius;
    public Circle(double radius) { this.radius = radius; }
    public double area() { return Math.PI * radius * radius; }
}

public final class Rectangle extends Shape {
    private final double width, height;
    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }
    public double area() { return width * height; }
}

public final class Triangle extends Shape {
    private final double base, height;
    public Triangle(double base, double height) {
        this.base = base;
        this.height = height;
    }
    public double area() { return 0.5 * base * height; }
}

// Safely branch with pattern matching (Java 21+)
public String describeShape(Shape shape) {
    return switch (shape) {
        case Circle c    -> "Circle (radius: " + c.getRadius() + ")";
        case Rectangle r -> "Rectangle (" + r.getWidth() + " x " + r.getHeight() + ")";
        case Triangle t  -> "Triangle (base: " + t.getBase() + ")";
        // Since it's sealed, all patterns are covered -> default is unnecessary
    };
}
```

```kotlin
// Kotlin: sealed class (implementing ADT: Algebraic Data Type)
sealed class Result<out T> {
    data class Success<T>(val value: T) : Result<T>()
    data class Failure(val error: Throwable) : Result<Nothing>()
    data object Loading : Result<Nothing>()
}

fun <T> handleResult(result: Result<T>) {
    when (result) {
        is Result.Success -> println("Success: ${result.value}")
        is Result.Failure -> println("Failure: ${result.error.message}")
        is Result.Loading -> println("Loading...")
        // Since it's sealed, all patterns are covered -> else is unnecessary
    }
}

// Usage example
val result: Result<String> = Result.Success("Data fetch complete")
handleResult(result) // Success: Data fetch complete
```

---

## 8. Practical Case Study: Inheritance in a Web Framework

```python
# Django's class-based views: inheritance required by the framework
from django.views import View
from django.views.generic import ListView, CreateView, DetailView
from django.http import JsonResponse


# Pattern 1: basic View inheritance
class HealthCheckView(View):
    """Health check API"""

    def get(self, request):
        return JsonResponse({
            "status": "healthy",
            "version": "1.0.0",
        })


# Pattern 2: using generic views
class ArticleListView(ListView):
    """Article list (using the Template Method pattern)"""
    model = Article               # template variable
    template_name = "articles/list.html"
    paginate_by = 20
    ordering = ["-created_at"]

    def get_queryset(self):
        """Hook: customize the queryset"""
        qs = super().get_queryset()
        category = self.request.GET.get("category")
        if category:
            qs = qs.filter(category=category)
        return qs

    def get_context_data(self, **kwargs):
        """Hook: add to the template context"""
        context = super().get_context_data(**kwargs)
        context["categories"] = Category.objects.all()
        return context


# Pattern 3: using Mixins
class LoginRequiredMixin:
    """Login-required Mixin"""
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)
        return super().dispatch(request, *args, **kwargs)


class RateLimitMixin:
    """Rate-limit Mixin"""
    rate_limit = 100  # requests/minute

    def dispatch(self, request, *args, **kwargs):
        # Rate limit check (simplified)
        client_ip = request.META.get("REMOTE_ADDR")
        # if is_rate_limited(client_ip, self.rate_limit):
        #     return JsonResponse({"error": "Too many requests"}, status=429)
        return super().dispatch(request, *args, **kwargs)


class ProtectedArticleCreateView(LoginRequiredMixin, RateLimitMixin, CreateView):
    """Protected article creation view (combining multiple Mixins)"""
    model = Article
    fields = ["title", "content", "category"]
    template_name = "articles/create.html"
    success_url = "/articles/"

    # MRO: ProtectedArticleCreateView -> LoginRequiredMixin
    #   -> RateLimitMixin -> CreateView -> ...
    # Order of dispatch() calls:
    # 1. LoginRequiredMixin.dispatch() -> login check
    # 2. RateLimitMixin.dispatch() -> rate limit check
    # 3. CreateView.dispatch() -> actual request processing
```

```typescript
// React components: migrating from class-based to function-based

// Old: class-based component (inheritance-based)
class UserProfile extends React.Component<UserProfileProps, UserProfileState> {
  constructor(props: UserProfileProps) {
    super(props);
    this.state = { user: null, loading: true };
  }

  async componentDidMount() {
    const user = await fetchUser(this.props.userId);
    this.setState({ user, loading: false });
  }

  componentDidUpdate(prevProps: UserProfileProps) {
    if (prevProps.userId !== this.props.userId) {
      this.setState({ loading: true });
      fetchUser(this.props.userId).then(user => {
        this.setState({ user, loading: false });
      });
    }
  }

  render() {
    if (this.state.loading) return <div>Loading...</div>;
    return <div>{this.state.user?.name}</div>;
  }
}

// Modern: function component (composition-based)
function UserProfileV2({ userId }: { userId: string }) {
  // Custom hook = logic reuse through composition
  const { data: user, loading, error } = useUser(userId);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  return <div>{user?.name}</div>;
}

// Custom hook: reuse logic without inheritance
function useUser(userId: string) {
  const [data, setData] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchUser(userId)
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [userId]);

  return { data, loading, error };
}
// -> With no inheritance, any component can reuse logic by simply calling useUser()
```

---

## 9. Comparison Table of Inheritance Features by Language

```
+-------------+------+------+--------+--------+--------+-------+
| Feature      | Java | C#   | Python | C++    | Kotlin | Rust  |
+-------------+------+------+--------+--------+--------+-------+
| Single inh.  |  OK  |  OK  |  OK    |  OK    |  OK    |  NG* |
| Multiple inh.|  NG  |  NG  |  OK    |  OK    |  NG    |  NG  |
| Interface    |  OK  |  OK  | ABC    | pure virt| OK  | trait |
| default impl |  OK  |  OK  |  -     |  -     |  OK    |  OK  |
| Mixin        |  NG  |  NG  |  OK    |  NG    |  NG    |  NG  |
| abstract     |  OK  |  OK  | ABC    |  OK    |  OK    |  -   |
| final class  |  OK  |sealed|  -     |  -     | default|  -   |
| sealed class |  OK* |  OK  |  -     |  -     |  OK    |  -   |
| delegation(by)| NG  |  NG  |  NG    |  NG    |  OK    |  NG  |
| virtual dflt |  NG  | OK(virtual) | OK | OK(virtual)| OK(open)|  -   |
+-------------+------+------+--------+--------+--------+-------+

* Rust has no inheritance; type composition via trait is the norm
* Java's sealed is 17+
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is most important. Rather than just theory, your understanding deepens by actually writing code and verifying its behavior.

### Q2: What are the common mistakes beginners make?

Skipping the fundamentals and jumping into applied topics. It's recommended to firmly understand the basic concepts explained in this guide before moving on to the next step.

### Q3: How is it utilized in practical work?

Knowledge of this topic is frequently utilized in day-to-day development work. It is especially important during code reviews and architecture design.

---

## Summary

| Concept | Key Point |
|------|---------|
| Inheritance | Take over and extend the parent's functionality |
| Override | Redefine parent methods (3 patterns: complete replacement/extension/conditional branching) |
| Multiple inheritance | Beware of the diamond problem. Solved via MRO/Interface/Mixin |
| Abstract class | Common implementation + implementation obligation. The foundation of the Template Method pattern |
| Composition | has-a relationship. More flexible and safer than inheritance |
| Principle | "Composition over inheritance." Use inheritance only when the is-a relationship is clear |
| sealed/final | Restrict inheritance to increase safety |
| Testing | Composition makes it easier to swap in mocks |

---

## Guides to Read Next

---

## References
1. Bloch, J. "Effective Java." Item 18: Favor composition over inheritance. 2018.
2. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
3. Martin, R. "Clean Architecture." Prentice Hall, 2017.
4. Sandi Metz. "Practical Object-Oriented Design in Ruby." 2nd edition, 2018.
5. Joshua Kerievsky. "Refactoring to Patterns." Addison-Wesley, 2004.
6. Eric Freeman et al. "Head First Design Patterns." O'Reilly, 2nd edition, 2020.
