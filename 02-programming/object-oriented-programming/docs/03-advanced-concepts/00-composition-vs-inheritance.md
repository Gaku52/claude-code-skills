# Composition vs Inheritance

> "Favor composition over inheritance" has been an iron rule since GoF. However, rather than blindly using composition, it is important to understand the tradeoffs between the two and use them appropriately.

## What You Will Learn in This Chapter

- [ ] Understand the essential difference between composition and inheritance
- [ ] Grasp the reasons behind "favor composition over inheritance"
- [ ] Learn practical criteria for choosing between them
- [ ] Master the use of composition in design patterns
- [ ] Understand when inheritance is appropriate and its design guidelines


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. The Essential Difference

```
Inheritance: is-a relationship
  -> Dog is-a Animal
  -> Inherits everything from the parent (strong coupling)
  -> Relationships fixed at compile time

Composition: has-a relationship
  -> Car has-a Engine
  -> Combines the necessary parts (loose coupling)
  -> Parts can be swapped at runtime

  Inheritance:
    +---------+
    | Animal  |
    +----+----+
         | is-a
    +----+----+
    |   Dog   |
    +---------+

  Composition:
    +---------+     +--------+
    |   Car   |---->| Engine |  has-a
    |         |---->| Wheels |  has-a
    |         |---->| GPS    |  has-a
    +---------+     +--------+

Delegation:
  -> A form of composition
  -> Forwards method calls to an internal object
  -> Rather than "handling it yourself," you "ask what you have"

Aggregation:
  -> A weaker form of composition
  -> The "part" can exist independently
  -> Car has-a Driver (a driver exists even without a car)

  Composition: parts live and die with the owner
  Aggregation: parts exist independently of the owner
```

### 1.1 Structural Problems of Inheritance

```
Inheritance mechanism:

  class Dog extends Animal {
    // Dog automatically inherits everything from Animal
    // 1. public methods -> exposed as-is
    // 2. protected methods -> accessible
    // 3. private methods -> not accessible but still exist
    // 4. fields -> all inherited
  }

Problems:

  1. Breaking encapsulation:
     -> Can access protected fields
     -> Depends on the internal implementation of the parent class
     -> Refactoring the parent class becomes difficult

  2. Fragile Base Class Problem:
     -> Changes to the parent class break subclasses unexpectedly
     -> Because subclasses depend on the parent's implementation details

  3. Tight coupling:
     -> Forced to inherit the entire interface of the parent class
     -> All methods, including unnecessary ones, are exposed

  4. Single inheritance constraint (Java, C#, TypeScript):
     -> Can only have one parent class
     -> Cannot combine multiple behaviors
```

### 1.2 Concrete Example of the Fragile Base Class Problem

```java
// BAD: Real example of the fragile base class problem

// A "counting set" that extends Java's HashSet
public class CountingSet<E> extends HashSet<E> {
    private int addCount = 0;

    @Override
    public boolean add(E e) {
        addCount++;
        return super.add(e);
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        addCount += c.size();
        return super.addAll(c);
    }

    public int getAddCount() {
        return addCount;
    }
}

// Try using it
CountingSet<String> s = new CountingSet<>();
s.addAll(Arrays.asList("A", "B", "C"));
System.out.println(s.getAddCount()); // Expected: 3, Actual: 6 !!!

// Why is it 6?
// HashSet.addAll() internally calls add()!
// +3 from addAll(), +3 from add() x 3 = total 6
// -> We depended on the parent class's implementation details

// GOOD: Solved with composition
public class CountingSet<E> implements Set<E> {
    private final Set<E> delegate = new HashSet<>();
    private int addCount = 0;

    @Override
    public boolean add(E e) {
        addCount++;
        return delegate.add(e);  // Delegation
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        addCount += c.size();
        return delegate.addAll(c);  // Delegation (no effect even if add() is called internally)
    }

    @Override
    public int size() { return delegate.size(); }

    @Override
    public boolean contains(Object o) { return delegate.contains(o); }

    // ... all other Set methods also delegate to delegate

    public int getAddCount() {
        return addCount;
    }
}

CountingSet<String> s = new CountingSet<>();
s.addAll(Arrays.asList("A", "B", "C"));
System.out.println(s.getAddCount()); // 3 (correct!)
```

---

## 2. Why "Favor Composition Over Inheritance"

```
Problems with inheritance:
  1. Tight coupling: changes to the parent ripple through all subclasses
  2. Breaking encapsulation: children depend on parent's implementation details
  3. Lack of flexibility: behavior cannot be changed at runtime
  4. Combinatorial explosion:

  Example: Game characters
  Designed with inheritance:
    Character
    |-- Warrior
    |   |-- FireWarrior
    |   |-- IceWarrior
    |   +-- FlyingWarrior
    |-- Mage
    |   |-- FireMage
    |   |-- IceMage
    |   +-- FlyingMage
    +-- Archer
        |-- FireArcher
        |-- IceArcher
        +-- FlyingArcher
    -> 3 attributes x 3 classes = 9 classes
    -> Adding a new attribute = +3 classes, adding a new class = +3 classes

  Designed with composition:
    Character
    |-- has-a: AttackStyle (Warrior, Mage, Archer)
    |-- has-a: Element (Fire, Ice, Lightning)
    +-- has-a: Movement (Walk, Fly, Teleport)
    -> 3 + 3 + 3 = 9 components
    -> Adding a new attribute = +1 component
```

### 2.1 Refactoring with Composition

```typescript
// BAD inheritance: class explosion
class Animal {
  eat(): void { console.log("eat"); }
}
class FlyingAnimal extends Animal {
  fly(): void { console.log("fly"); }
}
class SwimmingAnimal extends Animal {
  swim(): void { console.log("swim"); }
}
class FlyingSwimmingAnimal extends ??? {
  // Multiple inheritance is not possible!
}

// GOOD composition: flexible combinations
interface MovementAbility {
  move(): string;
}

class Flying implements MovementAbility {
  move(): string { return "fly through the sky"; }
}

class Swimming implements MovementAbility {
  move(): string { return "swim underwater"; }
}

class Walking implements MovementAbility {
  move(): string { return "walk on the ground"; }
}

class Animal {
  private abilities: MovementAbility[] = [];

  addAbility(ability: MovementAbility): void {
    this.abilities.push(ability);
  }

  moveAll(): string[] {
    return this.abilities.map(a => a.move());
  }
}

// Duck: can fly + swim + walk
const duck = new Animal();
duck.addAbility(new Flying());
duck.addAbility(new Swimming());
duck.addAbility(new Walking());
console.log(duck.moveAll()); // ["fly through the sky", "swim underwater", "walk on the ground"]
```

### 2.2 Game Character Composition Design

```typescript
// ECS (Entity-Component-System) style composition design

// === Components (behavior parts) ===
interface AttackBehavior {
  attack(target: string): string;
  getRange(): number;
}

interface DefenseBehavior {
  defend(): string;
  getArmor(): number;
}

interface MovementBehavior {
  move(direction: string): string;
  getSpeed(): number;
}

interface ElementalPower {
  element(): string;
  specialAttack(target: string): string;
}

// === Concrete components ===
class SwordAttack implements AttackBehavior {
  attack(target: string): string {
    return `Slashed ${target} with a sword!`;
  }
  getRange(): number { return 1; }
}

class BowAttack implements AttackBehavior {
  attack(target: string): string {
    return `Shot ${target} with a bow and arrow!`;
  }
  getRange(): number { return 10; }
}

class MagicAttack implements AttackBehavior {
  attack(target: string): string {
    return `Attacked ${target} with magic!`;
  }
  getRange(): number { return 5; }
}

class ShieldDefense implements DefenseBehavior {
  defend(): string { return "Defended with a shield!"; }
  getArmor(): number { return 50; }
}

class DodgeDefense implements DefenseBehavior {
  defend(): string { return "Nimbly dodged!"; }
  getArmor(): number { return 10; }
}

class MagicBarrier implements DefenseBehavior {
  defend(): string { return "Deployed a magic barrier!"; }
  getArmor(): number { return 30; }
}

class WalkMovement implements MovementBehavior {
  move(direction: string): string { return `Walked ${direction}`; }
  getSpeed(): number { return 3; }
}

class FlyMovement implements MovementBehavior {
  move(direction: string): string { return `Flew ${direction}`; }
  getSpeed(): number { return 8; }
}

class TeleportMovement implements MovementBehavior {
  move(direction: string): string { return `Teleported ${direction}`; }
  getSpeed(): number { return 100; }
}

class FirePower implements ElementalPower {
  element(): string { return "Fire"; }
  specialAttack(target: string): string {
    return `Burned ${target} to ashes with fire!`;
  }
}

class IcePower implements ElementalPower {
  element(): string { return "Ice"; }
  specialAttack(target: string): string {
    return `Froze ${target} with ice!`;
  }
}

class LightningPower implements ElementalPower {
  element(): string { return "Lightning"; }
  specialAttack(target: string): string {
    return `Struck ${target} with lightning!`;
  }
}

// === Character (composition) ===
class Character {
  constructor(
    public name: string,
    private attackBehavior: AttackBehavior,
    private defenseBehavior: DefenseBehavior,
    private movementBehavior: MovementBehavior,
    private elementalPower?: ElementalPower,
  ) {}

  performAttack(target: string): string {
    return this.attackBehavior.attack(target);
  }

  performDefense(): string {
    return this.defenseBehavior.defend();
  }

  performMove(direction: string): string {
    return this.movementBehavior.move(direction);
  }

  performSpecial(target: string): string {
    if (!this.elementalPower) {
      return "No special ability";
    }
    return this.elementalPower.specialAttack(target);
  }

  // Behavior can be changed at runtime!
  setAttackBehavior(attack: AttackBehavior): void {
    this.attackBehavior = attack;
  }

  setElementalPower(power: ElementalPower): void {
    this.elementalPower = power;
  }

  describe(): string {
    const parts = [
      `[${this.name}]`,
      `Attack: ${this.attackBehavior.constructor.name}`,
      `Defense: ${this.defenseBehavior.constructor.name}`,
      `Movement: ${this.movementBehavior.constructor.name}`,
    ];
    if (this.elementalPower) {
      parts.push(`Element: ${this.elementalPower.element()}`);
    }
    return parts.join(" / ");
  }
}

// === Usage examples ===
// Fire warrior: sword + shield + walking + fire
const fireWarrior = new Character(
  "Fire Warrior",
  new SwordAttack(),
  new ShieldDefense(),
  new WalkMovement(),
  new FirePower(),
);

// Ice mage: magic + magic barrier + teleport + ice
const iceMage = new Character(
  "Ice Mage",
  new MagicAttack(),
  new MagicBarrier(),
  new TeleportMovement(),
  new IcePower(),
);

// Lightning archer: bow + dodge + fly + lightning
const lightningArcher = new Character(
  "Lightning Archer",
  new BowAttack(),
  new DodgeDefense(),
  new FlyMovement(),
  new LightningPower(),
);

// Change equipment during the game!
fireWarrior.setAttackBehavior(new BowAttack()); // Switch to bow
fireWarrior.setElementalPower(new IcePower());   // Change element

console.log(fireWarrior.performAttack("Dragon"));  // "Shot Dragon with a bow and arrow!"
console.log(fireWarrior.performSpecial("Dragon")); // "Froze Dragon with ice!"
```

### 2.3 Composition in Python

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Optional


# === Component definitions ===
class Renderer(Protocol):
    """Rendering strategy"""
    def render(self, data: dict) -> str: ...


class Validator(Protocol):
    """Validation strategy"""
    def validate(self, data: dict) -> list[str]: ...


class Serializer(Protocol):
    """Serialization strategy"""
    def serialize(self, data: dict) -> str: ...
    def deserialize(self, raw: str) -> dict: ...


class Logger(Protocol):
    """Log output strategy"""
    def info(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...


# === Component implementations ===
class HtmlRenderer:
    def render(self, data: dict) -> str:
        rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data.items()
        )
        return f"<table>{rows}</table>"


class MarkdownRenderer:
    def render(self, data: dict) -> str:
        header = "| Key | Value |\n|-----|-------|\n"
        rows = "\n".join(f"| {k} | {v} |" for k, v in data.items())
        return header + rows


class JsonRenderer:
    def render(self, data: dict) -> str:
        import json
        return json.dumps(data, indent=2, ensure_ascii=False)


class StrictValidator:
    """Strict validation"""
    def __init__(self, required_fields: list[str]):
        self.required_fields = required_fields

    def validate(self, data: dict) -> list[str]:
        errors = []
        for field_name in self.required_fields:
            if field_name not in data or not data[field_name]:
                errors.append(f"'{field_name}' is required")
        return errors


class LenientValidator:
    """Lenient validation (warnings only)"""
    def validate(self, data: dict) -> list[str]:
        return []  # Always OK


class JsonSerializer:
    def serialize(self, data: dict) -> str:
        import json
        return json.dumps(data, ensure_ascii=False)

    def deserialize(self, raw: str) -> dict:
        import json
        return json.loads(raw)


class ConsoleLogger:
    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def error(self, message: str) -> None:
        print(f"[ERROR] {message}")


class NullLogger:
    """Logger that outputs nothing (for testing)"""
    def info(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass


# === Build with composition ===
@dataclass
class ReportGenerator:
    """Report generator: built by combining components"""
    renderer: Renderer
    validator: Validator
    serializer: Serializer
    logger: Logger

    def generate(self, data: dict) -> str:
        self.logger.info(f"Report generation started: {len(data)} items")

        # Validation
        errors = self.validator.validate(data)
        if errors:
            for error in errors:
                self.logger.error(f"Validation error: {error}")
            raise ValueError(f"Validation failed: {errors}")

        # Rendering
        rendered = self.renderer.render(data)
        self.logger.info(f"Rendering completed: {len(rendered)} chars")

        return rendered

    def save(self, data: dict, filepath: str) -> None:
        serialized = self.serializer.serialize(data)
        with open(filepath, "w") as f:
            f.write(serialized)
        self.logger.info(f"Save completed: {filepath}")

    def load(self, filepath: str) -> dict:
        with open(filepath) as f:
            raw = f.read()
        return self.serializer.deserialize(raw)


# === Usage example: combine different components for different purposes ===

# Production: HTML + strict validation + JSON + console log
production_report = ReportGenerator(
    renderer=HtmlRenderer(),
    validator=StrictValidator(["title", "author", "date"]),
    serializer=JsonSerializer(),
    logger=ConsoleLogger(),
)

# Development: Markdown + lenient validation + JSON + silent log
dev_report = ReportGenerator(
    renderer=MarkdownRenderer(),
    validator=LenientValidator(),
    serializer=JsonSerializer(),
    logger=NullLogger(),
)

# Test: JSON + lenient validation + JSON + silent log
test_report = ReportGenerator(
    renderer=JsonRenderer(),
    validator=LenientValidator(),
    serializer=JsonSerializer(),
    logger=NullLogger(),
)

# Same ReportGenerator class, but completely different behavior
data = {"title": "Monthly Report", "author": "Tanaka", "date": "2026-01-01"}
print(production_report.generate(data))  # HTML format
print(dev_report.generate(data))         # Markdown format
print(test_report.generate(data))        # JSON format
```

---

## 3. Relationship with the Strategy Pattern

```typescript
// Composition + Strategy = change behavior at runtime

interface SortStrategy {
  sort<T>(data: T[], compareFn: (a: T, b: T) => number): T[];
}

class QuickSort implements SortStrategy {
  sort<T>(data: T[], compareFn: (a: T, b: T) => number): T[] {
    // Quicksort implementation
    return [...data].sort(compareFn);
  }
}

class MergeSort implements SortStrategy {
  sort<T>(data: T[], compareFn: (a: T, b: T) => number): T[] {
    // Merge sort implementation
    return [...data].sort(compareFn);
  }
}

class DataProcessor {
  // Composition: strategy is injected from outside
  constructor(private sortStrategy: SortStrategy) {}

  // Strategy can be changed at runtime
  setSortStrategy(strategy: SortStrategy): void {
    this.sortStrategy = strategy;
  }

  process(data: number[]): number[] {
    return this.sortStrategy.sort(data, (a, b) => a - b);
  }
}

const processor = new DataProcessor(new QuickSort());
processor.process([3, 1, 4, 1, 5]);
// When data size grows, change the strategy
processor.setSortStrategy(new MergeSort());
```

### 3.1 Design Patterns and Composition

```
Design patterns that leverage composition:

  Strategy pattern:
    -> Makes behavior interchangeable
    -> Examples: sort algorithms, authentication strategies

  Decorator pattern:
    -> Adds functionality to existing objects
    -> Examples: stream processing, middleware

  Observer pattern:
    -> Event notification
    -> Examples: UI events, Pub/Sub

  Composite pattern:
    -> Tree structure representation
    -> Examples: UI components, file systems

  Bridge pattern:
    -> Separates abstraction from implementation
    -> Example: platform-specific rendering

  State pattern:
    -> Changes behavior based on state
    -> Examples: workflow, TCP connection

  Chain of Responsibility pattern:
    -> Chain of processing
    -> Examples: middleware chain, validation
```

### 3.2 Decorator Pattern (Application of Composition)

```typescript
// Decorator pattern: extend functionality through composition, not inheritance

interface Logger {
  log(message: string): void;
}

class ConsoleLogger implements Logger {
  log(message: string): void {
    console.log(message);
  }
}

// Decorator: wraps the base logger to add functionality
class TimestampLogger implements Logger {
  constructor(private inner: Logger) {}

  log(message: string): void {
    const timestamp = new Date().toISOString();
    this.inner.log(`[${timestamp}] ${message}`);
  }
}

class PrefixLogger implements Logger {
  constructor(private inner: Logger, private prefix: string) {}

  log(message: string): void {
    this.inner.log(`${this.prefix} ${message}`);
  }
}

class JsonLogger implements Logger {
  constructor(private inner: Logger) {}

  log(message: string): void {
    this.inner.log(JSON.stringify({
      message,
      timestamp: new Date().toISOString(),
      level: "info",
    }));
  }
}

class FilterLogger implements Logger {
  constructor(private inner: Logger, private minLevel: string) {}

  log(message: string): void {
    // Filtering logic
    if (this.shouldLog(message)) {
      this.inner.log(message);
    }
  }

  private shouldLog(message: string): boolean {
    // Simple filtering
    return !message.startsWith("[DEBUG]");
  }
}

// Combine decorators
const logger = new TimestampLogger(
  new PrefixLogger(
    new FilterLogger(
      new ConsoleLogger(),
      "info"
    ),
    "[MyApp]"
  )
);

logger.log("Hello, World!");
// -> [2026-01-15T10:30:00.000Z] [MyApp] Hello, World!

// Trying to do the same with inheritance would produce:
// TimestampConsoleLogger, TimestampFileLogger,
// PrefixConsoleLogger, PrefixFileLogger,
// TimestampPrefixConsoleLogger, TimestampPrefixFileLogger...
// -> Class explosion!
```

```python
# Python: composition using decorators (functions)
from functools import wraps
from typing import Callable, Any
import time
import logging


def with_logging(func: Callable) -> Callable:
    """Decorator that adds log output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} returned {result}")
        return result
    return wrapper


def with_timing(func: Callable) -> Callable:
    """Decorator that adds execution time measurement"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def with_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator that adds retry"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def with_cache(func: Callable) -> Callable:
    """Decorator that caches results"""
    cache = {}
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper


# Combine decorators (composition)
@with_logging
@with_timing
@with_retry(max_retries=3)
@with_cache
def fetch_data(url: str) -> dict:
    """Fetch data from an external API"""
    import requests
    response = requests.get(url)
    return response.json()

# When executed:
# 1. Log output (with_logging)
# 2. Start time measurement (with_timing)
# 3. Retry handling (with_retry)
# 4. Check cache (with_cache)
# 5. Actual function execution
```

### 3.3 State Pattern (State Management with Composition)

```typescript
// State pattern: hold the state object via composition

interface OrderState {
  name: string;
  canConfirm(): boolean;
  canShip(): boolean;
  canCancel(): boolean;
  canDeliver(): boolean;
  confirm(order: Order): void;
  ship(order: Order): void;
  cancel(order: Order): void;
  deliver(order: Order): void;
}

class PendingState implements OrderState {
  name = "pending";
  canConfirm() { return true; }
  canShip() { return false; }
  canCancel() { return true; }
  canDeliver() { return false; }

  confirm(order: Order): void {
    console.log("Order confirmed");
    order.setState(new ConfirmedState());
  }
  ship(order: Order): void {
    throw new Error("Cannot ship an unconfirmed order");
  }
  cancel(order: Order): void {
    console.log("Order cancelled");
    order.setState(new CancelledState());
  }
  deliver(order: Order): void {
    throw new Error("Cannot deliver an unconfirmed order");
  }
}

class ConfirmedState implements OrderState {
  name = "confirmed";
  canConfirm() { return false; }
  canShip() { return true; }
  canCancel() { return true; }
  canDeliver() { return false; }

  confirm(order: Order): void {
    throw new Error("Already confirmed");
  }
  ship(order: Order): void {
    console.log("Order shipped");
    order.setState(new ShippedState());
  }
  cancel(order: Order): void {
    console.log("Confirmed order cancelled (refund process started)");
    order.setState(new CancelledState());
  }
  deliver(order: Order): void {
    throw new Error("Cannot deliver before shipping");
  }
}

class ShippedState implements OrderState {
  name = "shipped";
  canConfirm() { return false; }
  canShip() { return false; }
  canCancel() { return false; }
  canDeliver() { return true; }

  confirm(order: Order): void { throw new Error("Already shipped"); }
  ship(order: Order): void { throw new Error("Already shipped"); }
  cancel(order: Order): void { throw new Error("Cannot cancel a shipped order"); }
  deliver(order: Order): void {
    console.log("Order delivered");
    order.setState(new DeliveredState());
  }
}

class DeliveredState implements OrderState {
  name = "delivered";
  canConfirm() { return false; }
  canShip() { return false; }
  canCancel() { return false; }
  canDeliver() { return false; }

  confirm() { throw new Error("Already delivered"); }
  ship() { throw new Error("Already delivered"); }
  cancel() { throw new Error("Cannot cancel a delivered order"); }
  deliver() { throw new Error("Already delivered"); }
}

class CancelledState implements OrderState {
  name = "cancelled";
  canConfirm() { return false; }
  canShip() { return false; }
  canCancel() { return false; }
  canDeliver() { return false; }

  confirm() { throw new Error("Already cancelled"); }
  ship() { throw new Error("Already cancelled"); }
  cancel() { throw new Error("Already cancelled"); }
  deliver() { throw new Error("Already cancelled"); }
}

class Order {
  private state: OrderState = new PendingState(); // Composition

  setState(state: OrderState): void {
    console.log(`State change: ${this.state.name} -> ${state.name}`);
    this.state = state;
  }

  confirm(): void { this.state.confirm(this); }
  ship(): void { this.state.ship(this); }
  cancel(): void { this.state.cancel(this); }
  deliver(): void { this.state.deliver(this); }

  getStatus(): string { return this.state.name; }
}

// Usage example
const order = new Order();
console.log(order.getStatus());  // "pending"
order.confirm();                  // State change: pending -> confirmed
order.ship();                     // State change: confirmed -> shipped
order.deliver();                  // State change: shipped -> delivered
// order.cancel();                // Error: Cannot cancel a delivered order
```

---

## 4. When Inheritance Is Appropriate

```
When to use inheritance:
  - Clear is-a relationship (List is a Collection)
  - Framework extension points (AbstractController)
  - Template method pattern
  - The subclass semantically satisfies all methods of the parent
  - Type hierarchy is stable (does not change frequently)

When to use composition:
  - has-a relationship (Car has an Engine)
  - Combinations of behaviors are needed
  - Want to change behavior at runtime
  - Want to combine multiple "features"
  - Want to replace with a mock during testing

When in doubt:
  -> Choose composition (safer)
  -> Ask yourself: "Is this class really a 'kind of' the parent?"
  -> Avoid "inheritance just for code reuse"
```

### 4.1 Template Method Pattern (Appropriate Use of Inheritance)

```python
from abc import ABC, abstractmethod
from typing import Any


class ETLPipeline(ABC):
    """Base class for an ETL (Extract-Transform-Load) pipeline

    Template method pattern: defines the skeleton of an algorithm
    and delegates the concrete steps to subclasses.
    """

    def run(self) -> dict:
        """Template method: overall ETL flow (final)"""
        self._log("Pipeline started")

        # 1. Data extraction
        raw_data = self.extract()
        self._log(f"Extraction complete: {len(raw_data)} records")

        # 2. Data transformation
        transformed = self.transform(raw_data)
        self._log(f"Transformation complete: {len(transformed)} records")

        # 3. Validation (optional hook)
        valid_data = self.validate(transformed)
        self._log(f"Validation complete: {len(valid_data)} records")

        # 4. Data loading
        result = self.load(valid_data)
        self._log(f"Load complete")

        # 5. Post-processing (optional hook)
        self.after_load(result)

        return result

    @abstractmethod
    def extract(self) -> list[dict]:
        """Extract data (to be implemented by subclass)"""
        ...

    @abstractmethod
    def transform(self, data: list[dict]) -> list[dict]:
        """Transform data (to be implemented by subclass)"""
        ...

    @abstractmethod
    def load(self, data: list[dict]) -> dict:
        """Load data (to be implemented by subclass)"""
        ...

    def validate(self, data: list[dict]) -> list[dict]:
        """Validation (default: all pass)"""
        return data

    def after_load(self, result: dict) -> None:
        """Post-processing (default: do nothing)"""
        pass

    def _log(self, message: str) -> None:
        print(f"[{self.__class__.__name__}] {message}")


# Subclass: CSV -> PostgreSQL ETL
class CsvToPostgresETL(ETLPipeline):
    def __init__(self, csv_path: str, db_connection):
        self.csv_path = csv_path
        self.db = db_connection

    def extract(self) -> list[dict]:
        import csv
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            return list(reader)

    def transform(self, data: list[dict]) -> list[dict]:
        # Type conversion and cleansing
        for row in data:
            row["price"] = float(row.get("price", 0))
            row["name"] = row.get("name", "").strip()
        return data

    def validate(self, data: list[dict]) -> list[dict]:
        # Only data with positive prices
        return [row for row in data if row["price"] > 0]

    def load(self, data: list[dict]) -> dict:
        # INSERT into PostgreSQL
        count = 0
        for row in data:
            self.db.execute(
                "INSERT INTO products (name, price) VALUES (%s, %s)",
                (row["name"], row["price"]),
            )
            count += 1
        return {"inserted": count}


# Subclass: API -> Elasticsearch ETL
class ApiToElasticsearchETL(ETLPipeline):
    def __init__(self, api_url: str, es_client):
        self.api_url = api_url
        self.es = es_client

    def extract(self) -> list[dict]:
        import requests
        response = requests.get(self.api_url)
        return response.json()["results"]

    def transform(self, data: list[dict]) -> list[dict]:
        # Transform documents for Elasticsearch
        return [
            {
                "_index": "products",
                "_id": item["id"],
                "_source": {
                    "name": item["name"],
                    "price": item["price"],
                    "category": item.get("category", "uncategorized"),
                },
            }
            for item in data
        ]

    def load(self, data: list[dict]) -> dict:
        # Bulk insert into Elasticsearch
        from elasticsearch.helpers import bulk
        success, errors = bulk(self.es, data)
        return {"success": success, "errors": len(errors)}

    def after_load(self, result: dict) -> None:
        # Refresh the index
        self.es.indices.refresh(index="products")
```

### 4.2 Framework Extension (Appropriate Use of Inheritance)

```typescript
// Extending base classes provided by a framework
// -> This is a case where inheritance is appropriate

// React class components (historical example)
abstract class Component<P, S> {
  constructor(public props: P) {}
  abstract render(): VNode;

  setState(newState: Partial<S>): void {
    // Framework-internal processing
  }

  componentDidMount(): void {}
  componentWillUnmount(): void {}
  shouldComponentUpdate(nextProps: P, nextState: S): boolean {
    return true;
  }
}

// Inherit as a framework extension point
class UserProfile extends Component<UserProps, UserState> {
  componentDidMount(): void {
    this.fetchUser(this.props.userId);
  }

  render(): VNode {
    // UI rendering
  }
}

// Express middleware base class (hypothetical example)
abstract class Middleware {
  abstract handle(req: Request, res: Response, next: NextFunction): void;

  protected sendError(res: Response, status: number, message: string): void {
    res.status(status).json({ error: message });
  }
}

class AuthMiddleware extends Middleware {
  handle(req: Request, res: Response, next: NextFunction): void {
    const token = req.headers.authorization;
    if (!token) {
      this.sendError(res, 401, "Authentication required");
      return;
    }
    // Token validation...
    next();
  }
}

class RateLimitMiddleware extends Middleware {
  private requests = new Map<string, number[]>();

  handle(req: Request, res: Response, next: NextFunction): void {
    const ip = req.ip;
    const now = Date.now();
    const windowMs = 60000; // 1 minute

    const reqs = this.requests.get(ip) ?? [];
    const recent = reqs.filter(t => now - t < windowMs);

    if (recent.length >= 100) {
      this.sendError(res, 429, "Rate limit exceeded");
      return;
    }

    recent.push(now);
    this.requests.set(ip, recent);
    next();
  }
}
```

---

## 5. Decision Flowchart: Composition vs Inheritance

```
Decision flowchart:

  Q1: "Is B a kind of A?" (is-a relationship?)
  |
  |-- No -> Composition
  |
  +-- Yes
      |
      Q2: "Can B correctly implement all methods of A?"
      |
      |-- No -> Composition (+ separate interfaces via ISP)
      |
      +-- Yes
          |
          Q3: "Does B need to depend on A's implementation details?"
          |
          |-- Yes -> Inheritance (but minimize use of protected)
          |
          +-- No
              |
              Q4: "Does B's behavior need to change at runtime?"
              |
              |-- Yes -> Composition (Strategy pattern)
              |
              +-- No
                  |
                  Q5: "Is the type hierarchy stable?"
                  |
                  |-- Yes -> Inheritance is OK
                  |
                  +-- No -> Composition (prepare for future changes)

Concrete decision examples:

  ArrayList extends AbstractList -> OK inheritance (is-a + stable type hierarchy)
  Stack extends Vector -> BAD inheritance (Stack is not a Vector)
  CountingSet extends HashSet -> BAD inheritance (fragile base class problem)
  Button extends Component -> OK inheritance (framework extension)
  Car has-a Engine -> OK composition (has-a relationship)
  Logger has-a Formatter -> OK composition (runtime changes)
```

### 5.1 Common Cases in Practice

```typescript
// Case 1: Log output customization
// BAD inheritance
class FileLogger extends ConsoleLogger { ... }
class JsonLogger extends FileLogger { ... }
// -> Loggers are not in an is-a relationship; they differ in output destination

// GOOD composition
class Logger {
  constructor(
    private transport: LogTransport,  // Output destination
    private formatter: LogFormatter,  // Format
    private filter: LogFilter,        // Filter
  ) {}
}

// Case 2: HTTP client authentication
// BAD inheritance
class AuthenticatedHttpClient extends HttpClient { ... }
class OAuthHttpClient extends AuthenticatedHttpClient { ... }

// GOOD composition
class HttpClient {
  constructor(private auth: AuthStrategy) {}
  // BasicAuth, BearerToken, OAuth, NoAuth are interchangeable
}

// Case 3: Validation logic
// BAD inheritance
class EmailValidator extends StringValidator { ... }
class StrongPasswordValidator extends PasswordValidator { ... }

// GOOD composition
class CompositeValidator implements Validator {
  constructor(private validators: Validator[]) {}
  validate(value: string): ValidationResult {
    const errors = this.validators
      .map(v => v.validate(value))
      .filter(r => !r.isValid);
    return errors.length === 0
      ? { isValid: true }
      : { isValid: false, errors: errors.flatMap(r => r.errors) };
  }
}

// Combine validation rules
const passwordValidator = new CompositeValidator([
  new MinLengthValidator(8),
  new MaxLengthValidator(100),
  new ContainsUppercaseValidator(),
  new ContainsLowercaseValidator(),
  new ContainsDigitValidator(),
  new ContainsSpecialCharValidator(),
]);

// Case 4: Data repository
// BAD inheritance
class CachedUserRepository extends PostgresUserRepository { ... }
// -> Caching is not a persistence strategy

// GOOD composition (Decorator pattern)
class CachedRepository<T> implements Repository<T> {
  constructor(
    private inner: Repository<T>,
    private cache: CacheStore,
  ) {}

  async findById(id: string): Promise<T | null> {
    const cached = await this.cache.get(id);
    if (cached) return cached;
    const result = await this.inner.findById(id);
    if (result) await this.cache.set(id, result);
    return result;
  }
}

const userRepo = new CachedRepository(
  new PostgresUserRepository(db),
  new RedisCache(redis),
);
```

---

## 6. Composition Support Features per Language

```
Composition support in each language:

  Rust:
    -> Traits + impl -> explicit composition
    -> No inheritance (design decision)
    -> derive macros -> automatic implementations

  Go:
    -> Embedding -> syntactic sugar for delegation
    -> Interfaces are implicit
    -> No inheritance (design decision)

  Kotlin:
    -> by keyword -> syntactic sugar for delegation
    -> data class -> automatic generation of value objects

  Swift:
    -> Protocol extensions -> default implementations on protocols
    -> Protocol composition -> composing protocols

  TypeScript:
    -> Mixins -> composition via class expressions
    -> Intersection types -> composition at the type level
```

```go
// Go: composition via embedding
type Logger struct{}

func (l *Logger) Log(msg string) {
    fmt.Printf("[LOG] %s\n", msg)
}

type Metrics struct{}

func (m *Metrics) RecordLatency(duration time.Duration) {
    fmt.Printf("[METRICS] latency: %v\n", duration)
}

// Composition via embedding (syntactic sugar for delegation)
type Service struct {
    Logger   // Embedded: Service.Log() becomes available
    Metrics  // Embedded: Service.RecordLatency() becomes available
    db *sql.DB
}

func (s *Service) GetUser(id string) (*User, error) {
    start := time.Now()
    s.Log(fmt.Sprintf("Getting user: %s", id))  // Method from Logger

    var user User
    err := s.db.QueryRow("SELECT * FROM users WHERE id = $1", id).
        Scan(&user.ID, &user.Name)

    s.RecordLatency(time.Since(start))  // Method from Metrics
    return &user, err
}
```

```kotlin
// Kotlin: delegation with the 'by' keyword
interface Printer {
    fun print(message: String)
}

class ConsolePrinter : Printer {
    override fun print(message: String) {
        println(message)
    }
}

// Delegate with 'by': delegates processing to printer
class TimestampPrinter(private val printer: Printer) : Printer by printer {
    // print() is automatically delegated to printer

    // Override as needed
    override fun print(message: String) {
        val timestamp = java.time.LocalDateTime.now()
        printer.print("[$timestamp] $message")
    }
}

// Delegating multiple interfaces
interface Logger {
    fun log(message: String)
}

interface Cache {
    fun get(key: String): String?
    fun set(key: String, value: String)
}

class MyService(
    logger: Logger,
    cache: Cache,
) : Logger by logger, Cache by cache {
    // All methods of Logger and Cache are automatically delegated
    // This class only defines additional business logic

    fun processRequest(key: String): String {
        log("Processing request for key: $key")
        val cached = get(key)
        if (cached != null) {
            log("Cache hit for key: $key")
            return cached
        }
        val result = "computed_result"
        set(key, result)
        return result
    }
}
```

```rust
// Rust: composition via traits (no inheritance)
trait Drawable {
    fn draw(&self);
}

trait Clickable {
    fn on_click(&mut self);
}

trait Resizable {
    fn resize(&mut self, width: u32, height: u32);
}

// Implement multiple traits (composition-like)
struct Button {
    label: String,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    click_count: u32,
}

impl Drawable for Button {
    fn draw(&self) {
        println!("Drawing button '{}' at ({}, {})", self.label, self.x, self.y);
    }
}

impl Clickable for Button {
    fn on_click(&mut self) {
        self.click_count += 1;
        println!("Button '{}' clicked! (count: {})", self.label, self.click_count);
    }
}

impl Resizable for Button {
    fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

// Dynamic dispatch via trait objects
fn draw_all(items: &[&dyn Drawable]) {
    for item in items {
        item.draw();
    }
}

// Constrain generics via trait bounds
fn interactive<T: Drawable + Clickable + Resizable>(widget: &mut T) {
    widget.draw();
    widget.on_click();
    widget.resize(200, 100);
    widget.draw();
}
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that meets the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Also write test code

```python
# Exercise 1: basic implementation template
class Exercise1:
    """Exercise in basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main logic for data processing"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve the processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Test
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation to add the following features.

```python
# Exercise 2: advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise in advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Look up by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Test
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient: {slow_time:.4f}s")
    print(f"Efficient:   {fast_time:.6f}s")
    print(f"Speedup:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be mindful of the algorithm's computational complexity
- Choose the appropriate data structure
- Measure the effect with benchmarks
---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Beyond theory, actually writing code and verifying its behavior deepens understanding.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architectural design.

---

## Summary

| Aspect | Inheritance | Composition |
|--------|-------------|-------------|
| Relationship | is-a | has-a |
| Coupling | Tight | Loose |
| Flexibility | Low | High |
| Runtime change | Not possible | Possible |
| Recommendation | Limited | Preferred |
| Testability | Low | High |
| Reusability | Depends on type hierarchy | Independently reusable |

```
Practical guidelines:

  1. Default to composition
     -> When in doubt, choose composition
     -> Changing later from composition to inheritance is easier than the reverse

  2. Conditions for using inheritance:
     -> There is a clear is-a relationship
     -> All methods of the parent class are meaningful in the subclass (LSP compliant)
     -> The type hierarchy is stable
     -> The framework requires it

  3. Avoid "inheritance for code reuse"
     -> If you only want shared code, use utility classes or helper functions
     -> For behavior reuse, use traits/mixins

  4. Keep inheritance depth to 2-3 levels
     -> Deep inheritance trees are hard to understand
     -> "A -> B -> C -> D -> E" is a red flag

  5. Interfaces over inheritance
     -> Interfaces are sufficient when only type compatibility is needed
     -> Share implementation via composition + delegation
```

---

## Further Reading

---

## References
1. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994. (Favor composition over inheritance)
2. Bloch, J. "Effective Java." Item 18: Favor composition over inheritance. 3rd Edition, 2018.
3. Martin, R. "Clean Architecture." Prentice Hall, 2017.
4. Sandi Metz. "Practical Object-Oriented Design in Ruby." 2nd Edition, 2018.
5. The Go Programming Language Specification. "Embedding." golang.org.
6. The Rust Programming Language. "Traits." doc.rust-lang.org.
