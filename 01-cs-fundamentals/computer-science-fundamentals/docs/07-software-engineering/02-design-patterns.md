# Design Patterns

> Design patterns are "crystallized wisdom of predecessors" -- reusable solutions to common problems.
> Knowing patterns expands your design vocabulary and facilitates smoother team communication.

## Learning Objectives

- [ ] Explain the classification and intent of all 23 GoF patterns
- [ ] Select and implement patterns from the Creational, Structural, and Behavioral categories
- [ ] Compare architecture patterns (MVC / MVVM / Repository)
- [ ] Recognize anti-patterns and propose countermeasures
- [ ] Make informed decisions about pattern application in real codebases


## Prerequisites

Having the following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of [Software Testing](./01-testing.md)

---

## 1. The Significance of Design Patterns

### 1.1 What Are Design Patterns?

Design patterns are systematic catalogings of recurring problems in software design
and their general-purpose solutions.
In 1994, Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides
(commonly known as the Gang of Four, abbreviated GoF) organized
23 patterns in their book
"Design Patterns: Elements of Reusable Object-Oriented Software,"
which became the starting point.

Design patterns prevent "reinventing the wheel" and serve as tools for reusing proven designs.
However, patterns should not be copied verbatim;
they should be understood as templates to be adapted to the context of the problem.

### 1.2 Three Benefits of Learning Patterns

```
+--------------------------------------------------------------+
|          Three Benefits of Learning Design Patterns           |
+--------------------------------------------------------------+
|                                                              |
|  1. Acquiring a Common Vocabulary                            |
|     By saying "Let's implement this with Observer,"          |
|     every team member instantly understands the design intent |
|                                                              |
|  2. Faster Design Decisions                                  |
|     The appropriate structure comes to mind the moment        |
|     you see a problem, greatly reducing time spent            |
|     thinking from scratch                                    |
|                                                              |
|  3. Improved Maintainability and Extensibility               |
|     Designs that follow patterns are resilient to change      |
|     They naturally align with the SOLID principles           |
|                                                              |
+--------------------------------------------------------------+
```

### 1.3 Classification of GoF Patterns

The 23 GoF patterns are classified into 3 categories based on their purpose.

| Category | Purpose | Count | Examples |
|----------|---------|-------|----------|
| Creational | Make object creation mechanisms flexible | 5 | Singleton, Factory Method, Abstract Factory, Builder, Prototype |
| Structural | Organize the composition of classes and objects | 7 | Adapter, Bridge, Composite, Decorator, Facade, Flyweight, Proxy |
| Behavioral | Organize responsibility distribution and communication between objects | 11 | Observer, Strategy, Command, Iterator, State, Template Method, Visitor, Chain of Responsibility, Mediator, Memento, Interpreter |

```
+-------------------------------------------------------------+
|                   GoF 23 Patterns Overview                   |
+---------------+----------------+----------------------------+
| Creational(5) | Structural(7)  |     Behavioral(11)         |
+---------------+----------------+----------------------------+
| Singleton     | Adapter        | Chain of Responsibility    |
| Factory Method| Bridge         | Command                    |
| Abstract Fctry| Composite      | Interpreter                |
| Builder       | Decorator      | Iterator                   |
| Prototype     | Facade         | Mediator                   |
|               | Flyweight      | Memento                    |
|               | Proxy          | Observer                   |
|               |                | State                      |
|               |                | Strategy                   |
|               |                | Template Method            |
|               |                | Visitor                    |
+---------------+----------------+----------------------------+
```

### 1.4 Components of a Pattern

Each pattern is described by the following 4 elements:

1. **Name**: The name that becomes part of the design vocabulary
2. **Problem**: In what situations should it be used
3. **Solution**: Description of relationships, responsibilities, and collaboration between elements
4. **Consequences**: Trade-offs of applying the pattern

This chapter will explain each pattern while explicitly stating these 4 elements.

---

## 2. Creational Patterns

Creational patterns abstract the object creation process,
making systems flexible in how objects are created, composed, and represented.
Instead of calling `new` (or `ClassName()` in Python) directly,
separating the creation logic achieves designs that are resilient to change.

### 2.1 Singleton Pattern

#### Intent

Ensure that a class has only one instance across the entire system,
and provide a global access point to that instance.

#### Problem

With database connection pools, log managers, configuration objects, etc.,
having multiple instances can cause inconsistencies or resource waste.
Such objects should exist only once across the entire application.

#### Solution

```
+---------------------------------+
|         Singleton               |
+---------------------------------+
| - _instance: Singleton = None   |
+---------------------------------+
| + __new__(cls): Singleton       |
| + get_instance(): Singleton     |
| + operation(): void             |
+---------------------------------+
         ^
         | sole instance
         |
    client code
```

#### Python Implementation

```python
import threading


class DatabaseConnection:
    """Thread-safe Singleton pattern implementation example.

    Overrides the __new__ method and uses a Lock to ensure
    that only one instance is created even in multi-threaded environments.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, host: str = "localhost", port: int = 5432):
        if self._initialized:
            return
        self.host = host
        self.port = port
        self.connection = None
        self._initialized = True

    def connect(self) -> str:
        self.connection = f"Connected to {self.host}:{self.port}"
        return self.connection

    def disconnect(self) -> None:
        self.connection = None


# Usage example
db1 = DatabaseConnection("db.example.com", 5432)
db2 = DatabaseConnection("other.host.com", 3306)

assert db1 is db2               # Same instance
assert db1.host == "db.example.com"  # The first initialization value is preserved

db1.connect()
print(db2.connection)  # "Connected to db.example.com:5432"
```

#### More Practical Singleton in Python: Module-Level Variables

In Python, the module itself acts as a Singleton.
Therefore, managing configuration at the module level as shown below is the most natural approach.

```python
# config.py -- The module itself is a Singleton
_settings: dict = {}


def load(path: str) -> None:
    """Load a configuration file."""
    import json
    with open(path) as f:
        _settings.update(json.load(f))


def get(key: str, default=None):
    """Retrieve a configuration value."""
    return _settings.get(key, default)
```

#### Consequences (Trade-offs)

| Advantages | Disadvantages |
|-----------|--------------|
| Guarantees instance uniqueness | Introduces global state, making testing difficult |
| Enables lazy initialization | Requires caution for multi-thread race conditions |
| Reduces memory usage | Dependencies tend to become implicit |

#### When to Avoid Singleton

Singleton is convenient, but overuse introduces the same problems as global variables.
When testability is important, prefer Dependency Injection (DI)
and manage the lifetime as "singleton" on the DI container side.

---

### 2.2 Factory Method Pattern

#### Intent

Delegate object creation to subclasses,
allowing dynamic determination of which class to instantiate.

#### Problem

Consider a notification system as an example. Email notifications, SMS notifications, push notifications --
changing the creation logic every time a new notification type is added violates the Open-Closed Principle.
A mechanism is needed so that existing code does not need to be modified when adding new notification types.

#### Solution

```
+----------------------+
|   NotificationFactory |  (Creator)
|   <<abstract>>       |
+----------------------+
| + create(): Notif.   | <- Factory Method
| + send(msg): void    |
+------+---------------+
       | inheritance
  +----+------+----------------+
  v           v                v
+------+  +-------+  +-------------+
|Email |  | SMS   |  | Push        |
|Fctry |  | Fctry |  | Fctry       |
+--+---+  +--+----+  +--+----------+
   | create  | create   | create
   v         v          v
+------+  +-------+  +-------------+
|Email |  | SMS   |  | Push        |
|Notif |  | Notif |  | Notification|
+------+  +-------+  +-------------+
```

#### Python Implementation

```python
from abc import ABC, abstractmethod


class Notification(ABC):
    """Base class for notifications."""

    @abstractmethod
    def send(self, message: str) -> str:
        """Send a message."""
        ...


class EmailNotification(Notification):
    def __init__(self, recipient: str):
        self.recipient = recipient

    def send(self, message: str) -> str:
        return f"Email to {self.recipient}: {message}"


class SMSNotification(Notification):
    def __init__(self, phone_number: str):
        self.phone_number = phone_number

    def send(self, message: str) -> str:
        return f"SMS to {self.phone_number}: {message}"


class PushNotification(Notification):
    def __init__(self, device_token: str):
        self.device_token = device_token

    def send(self, message: str) -> str:
        return f"Push to {self.device_token}: {message}"


class NotificationFactory:
    """Notification object creation using the Factory Method pattern.

    Uses a dictionary-based registry to ensure extensibility.
    To add a new notification type, simply call register().
    """

    _creators: dict[str, type[Notification]] = {}

    @classmethod
    def register(cls, notification_type: str, creator: type[Notification]) -> None:
        """Register a notification type."""
        cls._creators[notification_type] = creator

    @classmethod
    def create(cls, notification_type: str, **kwargs) -> Notification:
        """Create an instance from a registered notification type."""
        creator = cls._creators.get(notification_type)
        if creator is None:
            raise ValueError(
                f"Unknown notification type: {notification_type}. "
                f"Available: {list(cls._creators.keys())}"
            )
        return creator(**kwargs)


# Register types
NotificationFactory.register("email", EmailNotification)
NotificationFactory.register("sms", SMSNotification)
NotificationFactory.register("push", PushNotification)

# Usage example
notif = NotificationFactory.create("email", recipient="user@example.com")
print(notif.send("Hello!"))  # "Email to user@example.com: Hello!"

notif2 = NotificationFactory.create("sms", phone_number="+81-90-1234-5678")
print(notif2.send("Verification code: 1234"))
```

#### Consequences (Trade-offs)

| Advantages | Disadvantages |
|-----------|--------------|
| Separation of creation and usage (loose coupling) | Number of classes increases |
| Conforms to the Open-Closed Principle | Can be over-engineering for simple cases |
| Easy to swap in mocks during testing | Increased indirection makes tracing harder |

---

### 2.3 Builder Pattern

#### Intent

Separate the construction process of complex objects,
enabling different representations to be produced from the same construction process.

#### Problem

Objects with many constructor arguments (e.g., HTTP requests, SQL queries,
UI components) are prone to argument ordering mistakes and poor readability.
Additionally, when there are many optional arguments, constructor overloads
explode (the Telescoping Constructor problem).

#### Python Implementation

```python
from dataclasses import dataclass, field


@dataclass
class HttpRequest:
    """A constructed HTTP request object."""
    method: str = "GET"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: str | None = None
    timeout: int = 30
    retries: int = 0
    auth_token: str | None = None


class HttpRequestBuilder:
    """HTTP request construction using the Builder pattern.

    Builds the request step by step via method chaining,
    and returns the final HttpRequest object with build().
    """

    def __init__(self):
        self._method = "GET"
        self._url = ""
        self._headers: dict[str, str] = {}
        self._body: str | None = None
        self._timeout = 30
        self._retries = 0
        self._auth_token: str | None = None

    def method(self, method: str) -> "HttpRequestBuilder":
        self._method = method.upper()
        return self

    def url(self, url: str) -> "HttpRequestBuilder":
        self._url = url
        return self

    def header(self, key: str, value: str) -> "HttpRequestBuilder":
        self._headers[key] = value
        return self

    def body(self, body: str) -> "HttpRequestBuilder":
        self._body = body
        return self

    def timeout(self, seconds: int) -> "HttpRequestBuilder":
        self._timeout = seconds
        return self

    def retries(self, count: int) -> "HttpRequestBuilder":
        self._retries = count
        return self

    def auth(self, token: str) -> "HttpRequestBuilder":
        self._auth_token = token
        return self

    def build(self) -> HttpRequest:
        if not self._url:
            raise ValueError("URL is required")
        if self._auth_token:
            self._headers["Authorization"] = f"Bearer {self._auth_token}"
        return HttpRequest(
            method=self._method,
            url=self._url,
            headers=self._headers,
            body=self._body,
            timeout=self._timeout,
            retries=self._retries,
            auth_token=self._auth_token,
        )


# Usage example: Build readably via method chaining
request = (
    HttpRequestBuilder()
    .method("POST")
    .url("https://api.example.com/users")
    .header("Content-Type", "application/json")
    .body('{"name": "Alice", "age": 30}')
    .auth("my-secret-token")
    .timeout(10)
    .retries(3)
    .build()
)

print(request.method)   # "POST"
print(request.url)      # "https://api.example.com/users"
print(request.headers)  # {"Content-Type": "application/json", "Authorization": "Bearer my-secret-token"}
print(request.retries)  # 3


# Comparison: Without a Builder (poor readability)
# request = HttpRequest("POST", "https://api.example.com/users",
#     {"Content-Type": "application/json", "Authorization": "Bearer my-secret-token"},
#     '{"name": "Alice", "age": 30}', 10, 3, "my-secret-token")
```

#### Consequences (Trade-offs)

| Advantages | Disadvantages |
|-----------|--------------|
| Enables step-by-step construction of complex objects | Increases code due to the Builder class |
| High readability with method chaining | Overkill for simple objects |
| Validation can be centralized in build() | Requires ingenuity when combined with immutable design |

---

### 2.4 Prototype Pattern

#### Intent

Create new objects by copying (cloning) existing objects.
Effective for objects that are expensive to create or have complex configurations.

#### Problem

When generating a large number of enemy characters in a game, building from scratch each time is inefficient.
It is more efficient to copy a template object and modify only the necessary parts.

#### Python Implementation

```python
import copy
from dataclasses import dataclass, field


@dataclass
class GameCharacter:
    """Game character prototype."""
    name: str
    hp: int
    attack: int
    defense: int
    skills: list[str] = field(default_factory=list)
    position: dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})

    def clone(self) -> "GameCharacter":
        """Create a complete copy via deep copy."""
        return copy.deepcopy(self)


# Create a template
goblin_template = GameCharacter(
    name="Goblin",
    hp=50,
    attack=10,
    defense=5,
    skills=["slash", "dodge"],
)

# Generate clones from the prototype
goblin1 = goblin_template.clone()
goblin1.name = "Goblin A"
goblin1.position = {"x": 10, "y": 20}

goblin2 = goblin_template.clone()
goblin2.name = "Goblin B"
goblin2.position = {"x": 30, "y": 40}
goblin2.skills.append("poison")  # Does not affect the template

print(goblin_template.skills)  # ["slash", "dodge"]
print(goblin2.skills)          # ["slash", "dodge", "poison"]
```

#### Consequences (Trade-offs)

| Advantages | Disadvantages |
|-----------|--------------|
| Simplifies creation of complex objects | Beware of deep copy costs |
| Prototypes can be modified dynamically at runtime | Requires caution with objects that have circular references |
| Generates diverse objects without subclassing | Must ensure independence after copying |

### 2.5 Creational Patterns Comparison

| Pattern | Primary Purpose | Use Case | Typical Python Implementation |
|---------|----------------|----------|------------------------------|
| Singleton | Restrict to a single instance | DB connections, config, logging | `__new__` / module variables |
| Factory Method | Delegate/abstract creation | Notifications, documents, UI components | Registry dict + `create()` |
| Abstract Factory | Create families of related objects at once | GUI themes, DB driver groups | Abstract base class groups |
| Builder | Step-by-step construction of complex objects | HTTP requests, SQL, config | Method chaining + `build()` |
| Prototype | Create by copying existing objects | Game characters, config templates | `copy.deepcopy()` |

---

## 3. Structural Patterns

Structural patterns deal with how to compose classes and objects to form larger structures.
They resolve interface mismatches, add new functionality dynamically,
or wrap complex subsystems with simple interfaces.

### 3.1 Adapter Pattern

#### Intent

Convert the interface of an existing class into another interface that the client expects.
It is a "converter" that enables incompatible interfaces to work together.

#### Problem

A legacy system's API returns XML, but the new system expects JSON.
You want to connect old and new without rewriting the legacy system.

#### Solution

```
+----------+       +--------------+       +----------+
|  Client  |------>|   Adapter    |------>|  Adaptee |
|          |       | (conversion  |       | (existing|
| expects  |       |  layer)      |       |  API)    |
| JSON     |       | XML->JSON    |       | returns  |
|          |       | conversion   |       | XML      |
+----------+       +--------------+       +----------+
```

#### Python Implementation

```python
from abc import ABC, abstractmethod
import json
from xml.etree import ElementTree as ET


class ModernAPI(ABC):
    """JSON interface expected by the new system."""

    @abstractmethod
    def get_data(self) -> dict:
        ...


class LegacyXMLService:
    """Legacy system: returns data as XML."""

    def fetch_xml(self) -> str:
        return """
        <users>
            <user>
                <name>Alice</name>
                <age>30</age>
            </user>
            <user>
                <name>Bob</name>
                <age>25</age>
            </user>
        </users>
        """


class XMLToJSONAdapter(ModernAPI):
    """Adapter: Adapts a legacy XML service to the JSON interface.

    Holds the Adaptee (LegacyXMLService) internally
    and implements the ModernAPI interface expected by the client.
    """

    def __init__(self, legacy_service: LegacyXMLService):
        self._legacy = legacy_service

    def get_data(self) -> dict:
        xml_str = self._legacy.fetch_xml()
        root = ET.fromstring(xml_str)
        users = []
        for user_elem in root.findall("user"):
            users.append({
                "name": user_elem.findtext("name", ""),
                "age": int(user_elem.findtext("age", "0")),
            })
        return {"users": users, "count": len(users)}


# Usage example
legacy = LegacyXMLService()
adapter = XMLToJSONAdapter(legacy)
data = adapter.get_data()
print(json.dumps(data, indent=2))
# {
#   "users": [
#     {"name": "Alice", "age": 30},
#     {"name": "Bob", "age": 25}
#   ],
#   "count": 2
# }
```

#### Consequences (Trade-offs)

| Advantages | Disadvantages |
|-----------|--------------|
| Ensures compatibility without modifying existing code | Adapter classes increase in number |
| Separates conversion in line with the Single Responsibility Principle | Excessive use leads to wrapper hell |
| Allows swapping out the legacy part during testing | Slight performance overhead |

---

### 3.2 Decorator Pattern

#### Intent

Dynamically add new responsibilities to an object.
An alternative to subclassing for extending functionality,
allowing flexible combination of features while respecting the Single Responsibility Principle.

#### Problem

Consider a coffee shop system. You want to freely combine toppings like milk, sugar,
and whipped cream with a base coffee.
Creating subclasses for all combinations leads to a class explosion
(CoffeeWithMilk, CoffeeWithSugar, CoffeeWithMilkAndSugar, ...).

#### Solution

```
+---------------------+
|  Component (ABC)    |
|  + cost(): float    |
|  + description(): str|
+-----+---------------+
      |
  +---+--------------+
  v                   v
+-----------+   +------------------+
| Concrete  |   | Decorator (ABC)  |
| Coffee    |   | wraps Component  |
+-----------+   +--+---------------+
                   |
          +--------+----------+
          v        v          v
       +------+ +------+ +--------+
       | Milk | |Sugar | |Whipped |
       | Dec. | | Dec. | |Cream D.|
       +------+ +------+ +--------+
```

#### Python Implementation

```python
from abc import ABC, abstractmethod


class Beverage(ABC):
    """Base class for beverages."""

    @abstractmethod
    def cost(self) -> float:
        ...

    @abstractmethod
    def description(self) -> str:
        ...


class Coffee(Beverage):
    """Base coffee."""

    def cost(self) -> float:
        return 300.0

    def description(self) -> str:
        return "Coffee"


class Espresso(Beverage):
    """Base espresso."""

    def cost(self) -> float:
        return 350.0

    def description(self) -> str:
        return "Espresso"


class BeverageDecorator(Beverage):
    """Base class for Decorators. Holds a Beverage internally."""

    def __init__(self, beverage: Beverage):
        self._beverage = beverage


class MilkDecorator(BeverageDecorator):
    def cost(self) -> float:
        return self._beverage.cost() + 50.0

    def description(self) -> str:
        return self._beverage.description() + " + Milk"


class SugarDecorator(BeverageDecorator):
    def cost(self) -> float:
        return self._beverage.cost() + 30.0

    def description(self) -> str:
        return self._beverage.description() + " + Sugar"


class WhippedCreamDecorator(BeverageDecorator):
    def cost(self) -> float:
        return self._beverage.cost() + 80.0

    def description(self) -> str:
        return self._beverage.description() + " + Whipped Cream"


# Usage example: Freely combine Decorators
order = Coffee()
order = MilkDecorator(order)
order = SugarDecorator(order)
order = WhippedCreamDecorator(order)

print(order.description())  # "Coffee + Milk + Sugar + Whipped Cream"
print(f"Total: {order.cost()} yen")  # "Total: 460.0 yen"

# Relationship with Python's decorator syntax (@decorator):
# Python's @decorator is a function/class wrapping feature,
# which is conceptually similar to but distinct from the GoF Decorator pattern.
# However, you can use @decorator to implement the GoF Decorator pattern.
```

#### Python Decorator (@) Used for the Decorator Pattern

```python
def logging_decorator(func):
    """A Python decorator that automatically adds logging to function calls."""
    def wrapper(*args, **kwargs):
        print(f"[LOG] Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"[LOG] {func.__name__} returned {result}")
        return result
    return wrapper


def retry_decorator(max_retries: int = 3):
    """A Python decorator that adds retry functionality."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"[RETRY] Attempt {attempt + 1} failed: {e}")
        return wrapper
    return decorator


@logging_decorator
@retry_decorator(max_retries=3)
def fetch_data(url: str) -> str:
    return f"Data from {url}"
```

---

### 3.3 Facade Pattern

#### Intent

Provide a unified and simplified interface to a complex subsystem.
Clients no longer need to know the details of the subsystem.

#### Problem

Order processing in an online shop involves multiple subsystems such as
inventory checking, payment processing, shipping arrangement, and email notification.
If client code directly manipulates all of these,
it becomes tightly coupled and difficult to change.

#### Solution

```
                  +---------------------+
  Client -------->|   OrderFacade       |
                  | place_order()       |
                  +--+------+------+---+
                     |      |      |
           +---------+      |      +----------+
           v                v                  v
   +--------------+ +--------------+ +--------------+
   |  Inventory   | |  Payment     | |  Shipping    |
   |  Service     | |  Service     | |  Service     |
   +--------------+ +--------------+ +--------------+
```

#### Python Implementation

```python
class InventoryService:
    """Inventory management subsystem."""

    def check_stock(self, product_id: str) -> bool:
        print(f"[Inventory] Checking stock for {product_id}")
        return True  # In stock

    def reserve(self, product_id: str, qty: int) -> str:
        print(f"[Inventory] Reserved {qty} units of {product_id}")
        return f"RESERVE-{product_id}-{qty}"


class PaymentService:
    """Payment subsystem."""

    def authorize(self, amount: float, card_token: str) -> str:
        print(f"[Payment] Authorized {amount} yen with card {card_token[:4]}****")
        return "AUTH-12345"

    def capture(self, auth_id: str) -> bool:
        print(f"[Payment] Captured payment {auth_id}")
        return True


class ShippingService:
    """Shipping subsystem."""

    def calculate_cost(self, address: str) -> float:
        print(f"[Shipping] Calculating cost to {address}")
        return 500.0

    def create_shipment(self, product_id: str, address: str) -> str:
        print(f"[Shipping] Created shipment for {product_id} to {address}")
        return "SHIP-67890"


class NotificationService:
    """Notification subsystem."""

    def send_confirmation(self, email: str, order_id: str) -> None:
        print(f"[Notification] Sent confirmation for {order_id} to {email}")


class OrderFacade:
    """Facade: A unified interface that hides the complexity of order processing.

    Clients only need to call place_order().
    Internally, 4 subsystems cooperate.
    """

    def __init__(self):
        self._inventory = InventoryService()
        self._payment = PaymentService()
        self._shipping = ShippingService()
        self._notification = NotificationService()

    def place_order(
        self,
        product_id: str,
        qty: int,
        card_token: str,
        address: str,
        email: str,
    ) -> dict:
        # Step 1: Check inventory
        if not self._inventory.check_stock(product_id):
            raise RuntimeError(f"Product {product_id} is out of stock")

        # Step 2: Reserve inventory
        reservation = self._inventory.reserve(product_id, qty)

        # Step 3: Calculate shipping cost
        shipping_cost = self._shipping.calculate_cost(address)

        # Step 4: Process payment
        total = qty * 1000 + shipping_cost  # Assumed unit price of 1000 yen
        auth_id = self._payment.authorize(total, card_token)
        self._payment.capture(auth_id)

        # Step 5: Arrange shipping
        shipment_id = self._shipping.create_shipment(product_id, address)

        # Step 6: Send confirmation email
        order_id = f"ORDER-{reservation}-{shipment_id}"
        self._notification.send_confirmation(email, order_id)

        return {
            "order_id": order_id,
            "total": total,
            "shipment_id": shipment_id,
        }


# Usage example: The client only needs to know the Facade
facade = OrderFacade()
result = facade.place_order(
    product_id="ITEM-001",
    qty=2,
    card_token="tok_visa_4242424242424242",
    address="Shibuya, Tokyo...",
    email="customer@example.com",
)
print(f"Order complete: {result['order_id']}")
```

---

### 3.4 Composite Pattern

#### Intent

Compose objects into tree structures and allow individual objects
and their collections to be treated through the same interface.

#### Problem

In a file system, you want to treat files (leaves) and directories (branches)
uniformly. Directories can contain both files and directories.
You want to perform size calculations and display recursively,
but different interfaces for files and directories make handling cumbersome.

#### Python Implementation

```python
from abc import ABC, abstractmethod


class FileSystemNode(ABC):
    """File system node (Component)."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def size(self) -> int:
        """Return the size in bytes."""
        ...

    @abstractmethod
    def display(self, indent: int = 0) -> str:
        """Return a string for tree display."""
        ...


class File(FileSystemNode):
    """File (Leaf)."""

    def __init__(self, name: str, size_bytes: int):
        super().__init__(name)
        self._size = size_bytes

    def size(self) -> int:
        return self._size

    def display(self, indent: int = 0) -> str:
        return " " * indent + f"[File] {self.name} ({self._size} bytes)"


class Directory(FileSystemNode):
    """Directory (Composite). Has child elements."""

    def __init__(self, name: str):
        super().__init__(name)
        self._children: list[FileSystemNode] = []

    def add(self, node: FileSystemNode) -> None:
        self._children.append(node)

    def remove(self, node: FileSystemNode) -> None:
        self._children.remove(node)

    def size(self) -> int:
        return sum(child.size() for child in self._children)

    def display(self, indent: int = 0) -> str:
        lines = [" " * indent + f"[Dir] {self.name} ({self.size()} bytes)"]
        for child in self._children:
            lines.append(child.display(indent + 2))
        return "\n".join(lines)


# Usage example
root = Directory("project")
src = Directory("src")
src.add(File("main.py", 2048))
src.add(File("utils.py", 1024))

tests = Directory("tests")
tests.add(File("test_main.py", 512))

root.add(src)
root.add(tests)
root.add(File("README.md", 256))

print(root.display())
# [Dir] project (3840 bytes)
#   [Dir] src (3072 bytes)
#     [File] main.py (2048 bytes)
#     [File] utils.py (1024 bytes)
#   [Dir] tests (512 bytes)
#     [File] test_main.py (512 bytes)
#   [File] README.md (256 bytes)

print(f"Total size: {root.size()} bytes")  # 3840
```

### 3.5 Structural Patterns Comparison

| Pattern | Primary Purpose | Use Case | Keywords |
|---------|----------------|----------|----------|
| Adapter | Interface conversion | Legacy integration, external API integration | Wrapper, conversion |
| Bridge | Separation of abstraction and implementation | Platform independence, drivers | Separation, independent variation |
| Composite | Unified operations on tree structures | File systems, UI trees, organization charts | Recursion, part-whole |
| Decorator | Dynamic feature addition | Streams, middleware, logging | Wrapping, layering |
| Facade | Hiding complexity | Subsystem integration, API gateways | Simplification, unified entry point |
| Flyweight | Memory sharing | Character rendering, game tiles | Sharing, lightweight |
| Proxy | Access control / proxy | Cache, lazy loading, authorization checks | Proxy, control |

---

## 4. Behavioral Patterns

Behavioral patterns deal with communication between objects and how responsibilities are distributed.
They cover algorithm switching, event notifications, command execution and cancellation,
and organize how objects cooperate with each other.

### 4.1 Observer Pattern

#### Intent

Define a one-to-many dependency between objects so that
when one object changes state,
all dependent objects are automatically notified and updated.

#### Problem

In a stock monitoring system, you want to update multiple components
such as chart display, alert notifications, and log recording whenever stock prices change.
Directly calling each component creates tight coupling,
making it difficult to add new components.

#### Solution

```
+-------------------+       +--------------------+
|   Subject         |       |   Observer (ABC)   |
| (Observable)      |       |                    |
+-------------------+       +--------------------+
| - observers: list |<>---->| + update(data)     |
| + attach(obs)     |  1..* +--------+-----------+
| + detach(obs)     |                |
| + notify()        |        +-------+----------+
+-------------------+        v       v          v
                          +------+ +------+ +------+
                          |Chart | |Alert | |Logger|
                          |Obs.  | |Obs.  | |Obs.  |
                          +------+ +------+ +------+
```

#### Python Implementation

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class Observer(ABC):
    """Observer interface."""

    @abstractmethod
    def update(self, event: str, data: Any) -> None:
        ...


class EventEmitter:
    """A generic Observable (Subject) implementation.

    Manages Observers per event name,
    notifying only the relevant Observers when a specific event occurs.
    """

    def __init__(self):
        self._observers: dict[str, list[Observer]] = {}

    def on(self, event: str, observer: Observer) -> None:
        """Register an Observer for an event."""
        self._observers.setdefault(event, []).append(observer)

    def off(self, event: str, observer: Observer) -> None:
        """Unregister an Observer from an event."""
        if event in self._observers:
            self._observers[event].remove(observer)

    def emit(self, event: str, data: Any = None) -> None:
        """Fire an event and notify all registered Observers."""
        for observer in self._observers.get(event, []):
            observer.update(event, data)


@dataclass
class StockPrice:
    symbol: str
    price: float
    change: float


class ChartObserver(Observer):
    """Updates the chart display."""

    def update(self, event: str, data: Any) -> None:
        if isinstance(data, StockPrice):
            direction = "^" if data.change > 0 else "v"
            print(f"[Chart] {data.symbol}: {data.price:.2f} {direction}")


class AlertObserver(Observer):
    """Price alert notification."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def update(self, event: str, data: Any) -> None:
        if isinstance(data, StockPrice) and abs(data.change) > self.threshold:
            print(f"[ALERT] {data.symbol} moved {data.change:+.2f}% !")


class LogObserver(Observer):
    """Trade log recording."""

    def __init__(self):
        self.log: list[str] = []

    def update(self, event: str, data: Any) -> None:
        if isinstance(data, StockPrice):
            entry = f"{data.symbol},{data.price},{data.change}"
            self.log.append(entry)
            print(f"[Log] Recorded: {entry}")


# Usage example
stock_feed = EventEmitter()

chart = ChartObserver()
alert = AlertObserver(threshold=2.0)
logger = LogObserver()

stock_feed.on("price_update", chart)
stock_feed.on("price_update", alert)
stock_feed.on("price_update", logger)

# Simulate stock price updates
stock_feed.emit("price_update", StockPrice("AAPL", 178.50, +1.2))
# [Chart] AAPL: 178.50 ^
# [Log] Recorded: AAPL,178.5,1.2

stock_feed.emit("price_update", StockPrice("GOOG", 141.80, -3.5))
# [Chart] GOOG: 141.80 v
# [ALERT] GOOG moved -3.50% !
# [Log] Recorded: GOOG,141.8,-3.5
```

---

### 4.2 Strategy Pattern

#### Intent

Define a family of algorithms, encapsulate each one, and make them interchangeable.
Algorithms can be switched without changing client code.

#### Problem

In an e-commerce site's discount calculation, there are multiple discount rules
such as regular discount, member discount, and seasonal discount.
Implementing with an `if-elif` chain requires modifying existing code
every time a new discount rule is added or changed.

#### Python Implementation

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


class DiscountStrategy(ABC):
    """Base class for discount strategies."""

    @abstractmethod
    def calculate(self, price: float) -> float:
        """Return the price after discount."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Return a description of the strategy."""
        ...


class NoDiscount(DiscountStrategy):
    def calculate(self, price: float) -> float:
        return price

    def description(self) -> str:
        return "No discount"


class PercentageDiscount(DiscountStrategy):
    def __init__(self, percentage: float):
        self._percentage = percentage

    def calculate(self, price: float) -> float:
        return price * (1 - self._percentage / 100)

    def description(self) -> str:
        return f"{self._percentage}% OFF"


class FixedAmountDiscount(DiscountStrategy):
    def __init__(self, amount: float):
        self._amount = amount

    def calculate(self, price: float) -> float:
        return max(0, price - self._amount)

    def description(self) -> str:
        return f"{self._amount} yen off"


class BuyNGetFreeDiscount(DiscountStrategy):
    """Buy N, get 1 free."""

    def __init__(self, buy_count: int):
        self._buy = buy_count

    def calculate(self, price: float) -> float:
        return price * self._buy / (self._buy + 1)

    def description(self) -> str:
        return f"Buy {self._buy}, get 1 free"


@dataclass
class ShoppingCart:
    """A cart with swappable discount strategies using the Strategy pattern."""

    items: list[tuple[str, float]]
    strategy: DiscountStrategy

    def set_strategy(self, strategy: DiscountStrategy) -> None:
        """Dynamically change the discount strategy."""
        self.strategy = strategy

    def subtotal(self) -> float:
        return sum(price for _, price in self.items)

    def total(self) -> float:
        return self.strategy.calculate(self.subtotal())

    def receipt(self) -> str:
        lines = ["--- Receipt ---"]
        for name, price in self.items:
            lines.append(f"  {name}: {price:.0f} yen")
        lines.append(f"  Subtotal: {self.subtotal():.0f} yen")
        lines.append(f"  {self.strategy.description()}")
        lines.append(f"  Total: {self.total():.0f} yen")
        return "\n".join(lines)


# Usage example
cart = ShoppingCart(
    items=[("Python Textbook", 3000), ("Notebook", 500), ("Pen", 200)],
    strategy=NoDiscount(),
)
print(cart.receipt())
# Total: 3700 yen

cart.set_strategy(PercentageDiscount(20))
print(cart.receipt())
# 20% OFF -> Total: 2960 yen

cart.set_strategy(FixedAmountDiscount(500))
print(cart.receipt())
# 500 yen off -> Total: 3200 yen
```

#### Comparing the Strategy Pattern with a Functional Approach

In Python, the Strategy pattern can also be implemented using functions instead of classes.

```python
from typing import Callable

# Functional Strategy
DiscountFunc = Callable[[float], float]

def no_discount(price: float) -> float:
    return price

def percentage_off(pct: float) -> DiscountFunc:
    return lambda price: price * (1 - pct / 100)

def fixed_off(amount: float) -> DiscountFunc:
    return lambda price: max(0, price - amount)

# Usage example
apply_discount: DiscountFunc = percentage_off(15)
print(apply_discount(1000))  # 850.0
```

The following table shows when to use the class-based vs. function-based approach.

| Aspect | Class-based | Function-based |
|--------|-----------|----------------|
| State retention | Naturally held in fields | Can be held via closures |
| Adding descriptions | description() method | docstring or managed separately |
| Testability | Easy to mock | Equally easy |
| Extensibility | Easy to add new methods | Easy to add new functions |
| Recommended for | Complex strategies, multiple methods | Simple transformations, one-liners |

---

### 4.3 Command Pattern

#### Intent

Encapsulate requests as objects, enabling
execution, cancellation, redo, and queuing of operations.

#### Problem

You want to implement Undo/Redo functionality in a text editor.
You need to record "what was done" for each operation and be able to execute the reverse operation.

#### Python Implementation

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class Command(ABC):
    """Base class for commands."""

    @abstractmethod
    def execute(self) -> None:
        ...

    @abstractmethod
    def undo(self) -> None:
        ...

    @abstractmethod
    def description(self) -> str:
        ...


class TextEditor:
    """The text editor body (Receiver)."""

    def __init__(self):
        self.content: str = ""
        self.cursor: int = 0

    def insert(self, text: str, position: int) -> None:
        self.content = self.content[:position] + text + self.content[position:]
        self.cursor = position + len(text)

    def delete(self, position: int, length: int) -> str:
        deleted = self.content[position:position + length]
        self.content = self.content[:position] + self.content[position + length:]
        self.cursor = position
        return deleted

    def __repr__(self) -> str:
        return f'TextEditor("{self.content}")'


class InsertCommand(Command):
    def __init__(self, editor: TextEditor, text: str, position: int):
        self._editor = editor
        self._text = text
        self._position = position

    def execute(self) -> None:
        self._editor.insert(self._text, self._position)

    def undo(self) -> None:
        self._editor.delete(self._position, len(self._text))

    def description(self) -> str:
        return f'Insert "{self._text}" at {self._position}'


class DeleteCommand(Command):
    def __init__(self, editor: TextEditor, position: int, length: int):
        self._editor = editor
        self._position = position
        self._length = length
        self._deleted_text: str = ""

    def execute(self) -> None:
        self._deleted_text = self._editor.delete(self._position, self._length)

    def undo(self) -> None:
        self._editor.insert(self._deleted_text, self._position)

    def description(self) -> str:
        return f'Delete {self._length} chars at {self._position}'


class CommandHistory:
    """Invoker that manages Undo/Redo."""

    def __init__(self):
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []

    def execute(self, command: Command) -> None:
        command.execute()
        self._undo_stack.append(command)
        self._redo_stack.clear()  # Clear redo history after a new operation

    def undo(self) -> None:
        if not self._undo_stack:
            print("Nothing to undo")
            return
        command = self._undo_stack.pop()
        command.undo()
        self._redo_stack.append(command)
        print(f"Undo: {command.description()}")

    def redo(self) -> None:
        if not self._redo_stack:
            print("Nothing to redo")
            return
        command = self._redo_stack.pop()
        command.execute()
        self._undo_stack.append(command)
        print(f"Redo: {command.description()}")


# Usage example
editor = TextEditor()
history = CommandHistory()

history.execute(InsertCommand(editor, "Hello", 0))
print(editor)  # TextEditor("Hello")

history.execute(InsertCommand(editor, " World", 5))
print(editor)  # TextEditor("Hello World")

history.execute(DeleteCommand(editor, 5, 6))
print(editor)  # TextEditor("Hello")

history.undo()  # Undo: Delete 6 chars at 5
print(editor)  # TextEditor("Hello World")

history.undo()  # Undo: Insert " World" at 5
print(editor)  # TextEditor("Hello")

history.redo()  # Redo: Insert " World" at 5
print(editor)  # TextEditor("Hello World")
```

---

### 4.4 Iterator Pattern

#### Intent

Provide a way to access the elements of a collection sequentially without exposing its internal structure.

#### Problem

When traversing data structures like binary search trees or graphs,
embedding traversal logic (depth-first, breadth-first, in-order, etc.) into the data structure itself
violates the Single Responsibility Principle. We want to externalize traversal methods.

#### Python Implementation

Python supports the Iterator pattern at the language level
through implementation of `__iter__` and `__next__`.

```python
from collections import deque
from typing import Iterator, Generic, TypeVar

T = TypeVar("T")


class BinaryTreeNode(Generic[T]):
    """Binary tree node."""

    def __init__(self, value: T):
        self.value = value
        self.left: "BinaryTreeNode[T] | None" = None
        self.right: "BinaryTreeNode[T] | None" = None


class InOrderIterator:
    """In-order traversal (left -> root -> right) Iterator."""

    def __init__(self, root: BinaryTreeNode | None):
        self._stack: list[BinaryTreeNode] = []
        self._push_left(root)

    def _push_left(self, node: BinaryTreeNode | None) -> None:
        while node:
            self._stack.append(node)
            node = node.left

    def __iter__(self):
        return self

    def __next__(self):
        if not self._stack:
            raise StopIteration
        node = self._stack.pop()
        self._push_left(node.right)
        return node.value


class BreadthFirstIterator:
    """Breadth-first traversal Iterator."""

    def __init__(self, root: BinaryTreeNode | None):
        self._queue: deque[BinaryTreeNode] = deque()
        if root:
            self._queue.append(root)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._queue:
            raise StopIteration
        node = self._queue.popleft()
        if node.left:
            self._queue.append(node.left)
        if node.right:
            self._queue.append(node.right)
        return node.value


class BinaryTree(Generic[T]):
    """A binary tree that provides multiple traversal methods."""

    def __init__(self, root: BinaryTreeNode[T] | None = None):
        self.root = root

    def in_order(self) -> InOrderIterator:
        return InOrderIterator(self.root)

    def breadth_first(self) -> BreadthFirstIterator:
        return BreadthFirstIterator(self.root)

    def __iter__(self):
        return self.in_order()


# Usage example: Build a tree
#       4
#      / \
#     2   6
#    / \ / \
#   1  3 5  7
root = BinaryTreeNode(4)
root.left = BinaryTreeNode(2)
root.right = BinaryTreeNode(6)
root.left.left = BinaryTreeNode(1)
root.left.right = BinaryTreeNode(3)
root.right.left = BinaryTreeNode(5)
root.right.right = BinaryTreeNode(7)

tree = BinaryTree(root)

print("In-order:", list(tree.in_order()))
# In-order: [1, 2, 3, 4, 5, 6, 7]

print("BFS:", list(tree.breadth_first()))
# BFS: [4, 2, 6, 1, 3, 5, 7]

# Can also be used in a for loop (__iter__ returns in_order)
for value in tree:
    print(value, end=" ")
# 1 2 3 4 5 6 7
```

### 4.5 Behavioral Patterns Comparison

| Pattern | Primary Purpose | Use Case | Keywords |
|---------|----------------|----------|----------|
| Observer | Notification of state changes | Events, PubSub, reactive | Notification, subscription, one-to-many |
| Strategy | Algorithm interchange | Discount calculation, sorting, authentication methods | Interchangeable, policy |
| Command | Objectification of operations | Undo/Redo, queuing, macros | Execute, undo, history |
| Iterator | Sequential access | Collection traversal, streams | Traversal, cursor |
| State | Behavior change based on state | Workflows, TCP connections, UI state | State transition, finite automaton |
| Template Method | Define the skeleton of processing | Frameworks, ETL, tests | Skeleton, hook, inheritance |
| Chain of Responsibility | Chain of processing | Middleware, authorization chains | Chain, pipeline |
| Mediator | Mediation between objects | Chat rooms, control towers | Mediation, hub |
| Memento | Save and restore state | Snapshots, checkpoints | Save, restore |
| Visitor | Separation of data structures and processing | AST traversal, report generation | Double dispatch |

---

## 5. Architecture Patterns

While GoF design patterns deal with class-level design,
architecture patterns deal with the overall structure of an application.
Here we will explain 3 representative patterns.

### 5.1 MVC (Model-View-Controller)

#### Overview

Separates an application into 3 roles:

- **Model**: Business logic and data
- **View**: User interface (display)
- **Controller**: Receives user input and mediates between Model and View

```
+----------+     User actions      +--------------+
|          | --------------------> |              |
|   View   |                      |  Controller  |
| (display)|  <------------------ |  (control)   |
|          |     Update display    |              |
+----+-----+                      +------+-------+
     |                                   |
     |  Reference data                   | Operate Model
     |                                   |
     |         +--------------+          |
     +-------->|    Model     |<---------+
               | (data /      |
               |  business    |
               |  logic)      |
               +--------------+
```

#### Characteristics

- Widely adopted in web frameworks (Django, Ruby on Rails, Spring MVC)
- Separation of Model and View enables displaying the same data in multiple Views
- Model logic can be unit tested by swapping the View during testing

### 5.2 MVVM (Model-View-ViewModel)

#### Overview

A derivative of MVC that leverages data binding to synchronize View and ViewModel.

- **Model**: Data and business logic
- **View**: UI display
- **ViewModel**: Transforms and manages data for display in the View. Holds View state

```
+----------+     Data binding          +--------------+
|          | <========================>|              |
|   View   |     (Two-way sync)        |  ViewModel   |
|          |                           |              |
+----------+                           +------+-------+
                                              |
                                              | Data operations
                                              v
                                       +--------------+
                                       |    Model     |
                                       +--------------+
```

#### Characteristics

- Widely adopted in frontend frameworks (Vue.js, SwiftUI, WPF)
- Data binding eliminates the need for manual View updates
- ViewModel does not depend on the View, making it easy to test

### 5.3 Repository Pattern

#### Overview

Abstracts data access logic and creates an intermediary layer between
the domain model and the database. Business logic no longer needs to know
the details of how data is retrieved or stored.

```
+---------------+     +--------------------+     +--------------+
|  Business     |---->|   Repository       |---->|  Database    |
|  Logic        |     |   (Interface)      |     |  / ORM /     |
|               |<----|                    |<----|  API / File  |
+---------------+     +--------------------+     +--------------+
```

#### Python Implementation

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Generic

T = TypeVar("T")


@dataclass
class User:
    id: int
    name: str
    email: str


class UserRepository(ABC):
    """User repository interface."""

    @abstractmethod
    def find_by_id(self, user_id: int) -> User | None:
        ...

    @abstractmethod
    def find_by_email(self, email: str) -> User | None:
        ...

    @abstractmethod
    def find_all(self) -> list[User]:
        ...

    @abstractmethod
    def save(self, user: User) -> None:
        ...

    @abstractmethod
    def delete(self, user_id: int) -> None:
        ...


class InMemoryUserRepository(UserRepository):
    """In-memory implementation for testing."""

    def __init__(self):
        self._store: dict[int, User] = {}

    def find_by_id(self, user_id: int) -> User | None:
        return self._store.get(user_id)

    def find_by_email(self, email: str) -> User | None:
        for user in self._store.values():
            if user.email == email:
                return user
        return None

    def find_all(self) -> list[User]:
        return list(self._store.values())

    def save(self, user: User) -> None:
        self._store[user.id] = user

    def delete(self, user_id: int) -> None:
        self._store.pop(user_id, None)


class UserService:
    """Business logic layer. Depends on Repository."""

    def __init__(self, repo: UserRepository):
        self._repo = repo

    def register(self, user_id: int, name: str, email: str) -> User:
        existing = self._repo.find_by_email(email)
        if existing:
            raise ValueError(f"Email {email} is already registered")
        user = User(id=user_id, name=name, email=email)
        self._repo.save(user)
        return user

    def get_user(self, user_id: int) -> User:
        user = self._repo.find_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        return user


# Usage example
repo = InMemoryUserRepository()
service = UserService(repo)

alice = service.register(1, "Alice", "alice@example.com")
bob = service.register(2, "Bob", "bob@example.com")

print(service.get_user(1))   # User(id=1, name='Alice', email='alice@example.com')
print(repo.find_all())       # [User(...), User(...)]

# Use InMemoryUserRepository for testing,
# swap to SQLAlchemyUserRepository etc. for production
```

### 5.4 Architecture Patterns Comparison

| Pattern | View-Logic Coupling | Data Sync | Primary Use | Testability |
|---------|-------------------|-----------|-------------|-------------|
| MVC | Controller mediates | Manual (via Controller) | Web (Django, Rails) | Easy to test Model |
| MVVM | Data binding | Automatic (two-way binding) | SPA (Vue), Mobile (SwiftUI) | Easy to test ViewModel |
| MVP | Presenter mediates | Manual (via Presenter) | Android (legacy), Desktop | Easy to test Presenter |
| Repository | Separation of concerns | Via Repository | DDD, Clean Architecture | Easy by swapping Repository |

---

## 6. How to Choose Patterns

### 6.1 Decision Flowchart

Pattern selection is determined by the nature of the problem you face.

```
What is the type of problem?
|
+-- Problems related to object "creation"
|   +-- Want to restrict to a single instance    -> Singleton
|   +-- Want to dynamically decide which class   -> Factory Method
|   +-- Need to create related object groups     -> Abstract Factory
|   +-- Need step-by-step construction of        -> Builder
|   |   complex objects
|   +-- Want to copy an existing object           -> Prototype
|
+-- Problems related to class/object "structure"
|   +-- Need to resolve interface mismatches      -> Adapter
|   +-- Want to add features dynamically          -> Decorator
|   +-- Want to hide complexity                   -> Facade
|   +-- Want to treat tree structures uniformly   -> Composite
|   +-- Want to control/proxy access              -> Proxy
|
+-- Problems related to "behavior" between objects
    +-- Want to notify multiple objects of         -> Observer
    |   state changes
    +-- Want to make algorithms interchangeable    -> Strategy
    +-- Want to manage execution/undo of           -> Command
    |   operations
    +-- Want to traverse a collection              -> Iterator
    +-- Want to change behavior based on state     -> State
```

### 6.2 Principles for Pattern Selection

1. **Problem-first**: Do not design with patterns in mind. Clarify the problem first, then search for patterns
2. **Minimal application**: Apply the minimum necessary patterns. Be cautious when combining multiple patterns
3. **YAGNI**: Avoid applying patterns for future extensibility. Only what is needed now
4. **Team understanding**: Patterns that team members cannot understand increase maintenance costs
5. **Language characteristics**: Leverage Python's duck typing, first-class functions, etc., and avoid excessive class hierarchies

### 6.3 Simplification of Patterns in Python

Python's dynamic typing and first-class functions make some class hierarchies
that were necessary in Java/C++ unnecessary.

| GoF Pattern | Java Implementation | Python Simplification |
|-------------|--------------------|-----------------------|
| Strategy | Interface + implementation classes | Pass functions as arguments |
| Command | Command interface + implementation | Function/lambda + list |
| Observer | Observer interface + Subject | List of callback functions |
| Singleton | Private constructor + static | Module-level variable |
| Factory Method | Abstract class + subclasses | Dictionary + function/class |
| Iterator | Iterator interface implementation | `__iter__` / `__next__` / generators |
| Template Method | Abstract class + inheritance | Higher-order functions or mixins |

---

## 7. Anti-Patterns

### 7.1 God Object

#### Description

A class where one class handles most of the system's functionality,
knowing everything and doing everything.
It completely violates the Single Responsibility Principle (SRP).

#### Symptoms

- The class code grows to hundreds or thousands of lines
- Has 20 or more methods
- Responsibilities from different domains are mixed in a single class
- Every change requires modifying that class

#### Bad Example

```python
# Anti-pattern: God Object
class Application:
    """A class that does everything (example of what NOT to do)."""

    def __init__(self):
        self.users = []
        self.products = []
        self.orders = []
        self.db_connection = None
        self.email_client = None
        self.cache = {}
        self.log_file = None

    def connect_to_database(self): ...
    def close_database(self): ...
    def create_user(self, name, email): ...
    def delete_user(self, user_id): ...
    def authenticate_user(self, email, password): ...
    def add_product(self, name, price): ...
    def update_product(self, product_id, **kwargs): ...
    def search_products(self, query): ...
    def create_order(self, user_id, items): ...
    def calculate_shipping(self, address): ...
    def process_payment(self, order_id, card): ...
    def send_confirmation_email(self, user_id, order_id): ...
    def generate_report(self, report_type): ...
    def export_to_csv(self, data): ...
    def clear_cache(self): ...
    def write_log(self, message): ...
    # ... goes on and on
```

#### Improvement

Split classes by responsibility and have them collaborate loosely.

```python
# Improved: Split classes by responsibility
class UserService:
    """Specialized for user management."""
    def create(self, name: str, email: str) -> "User": ...
    def authenticate(self, email: str, password: str) -> bool: ...

class ProductService:
    """Specialized for product management."""
    def add(self, name: str, price: float) -> "Product": ...
    def search(self, query: str) -> list["Product"]: ...

class OrderService:
    """Specialized for order management."""
    def create(self, user_id: int, items: list) -> "Order": ...
    def process_payment(self, order_id: int, card: str) -> bool: ...

class NotificationService:
    """Specialized for notifications."""
    def send_confirmation(self, user_id: int, order_id: int) -> None: ...
```

---

### 7.2 Golden Hammer

#### Description

Derived from the saying "If all you have is a hammer, everything looks like a nail."
When you become proficient with a specific pattern or technology,
you tend to apply that pattern to every problem.

#### Symptoms

- Applying overly complex patterns to simple problems
- Using the Strategy pattern "just in case," but there is only one strategy
- Using a Factory, but there is only one class to create
- Using Observer, but there is only one Observer

#### Countermeasures

1. **Choose solutions proportionate to the problem's complexity**: Apply patterns only when the problem is complex
2. **Follow YAGNI**: Avoid adding patterns now for the future
3. **Introduce through refactoring**: Implement simply first, then introduce patterns when the need arises

```python
# Example of Golden Hammer: Unnecessary Factory
# If there's only one class to create, a Factory is unnecessary

# Over-engineering (overkill)
class LoggerFactory:
    @staticmethod
    def create(logger_type: str):
        if logger_type == "console":
            return ConsoleLogger()
        raise ValueError(f"Unknown: {logger_type}")

logger = LoggerFactory.create("console")  # Always only "console"

# Appropriate (simple)
logger = ConsoleLogger()  # Direct instantiation is sufficient
```

---

### 7.3 Cargo Cult Programming

#### Description

Blindly copying patterns and practices without understanding their intent,
simply because "it was used in a famous project" or "a senior wrote it."
It may look correct on the surface, but lacks fundamental understanding.

#### Symptoms

- Mimicking only the "form" of a pattern without having a problem to solve
- Defining interfaces for every class, but there is always only one implementation
- Knowing the names of design patterns but being unable to explain their trade-offs

#### Countermeasures

1. Ask "why" before applying a pattern
2. Understand the trade-offs (both advantages and disadvantages) of the pattern
3. Compare with the version written without the pattern and confirm the pattern's value

---

## 8. Exercises

### 8.1 Beginner Level

**Problem 1**: For the following requirements, identify which design pattern is appropriate and explain why.

(a) You want to share configuration file contents across the entire application
(b) You want to choose the log output destination from file, console, or remote server
(c) An external library's interface does not match your system, so you need to convert it

<details>
<summary>Example Answer</summary>

(a) **Singleton Pattern** (or module-level variables).
The configuration object should be unique across the application,
and the same configuration needs to be accessed from anywhere.
In Python, module-level variables are often sufficient.

(b) **Strategy Pattern**.
By making the logging algorithm (output destination) interchangeable,
the destination can be switched at runtime.
In Python, the logging module's Handler follows exactly this structure.

(c) **Adapter Pattern**.
Create a wrapper that converts to the interface your system expects
without modifying the existing library.

</details>

**Problem 2**: List 3 problems with the Singleton pattern and describe countermeasures for each.

<details>
<summary>Example Answer</summary>

1. **Difficult to test**: State is shared between tests due to global state
   -> Countermeasure: Use Dependency Injection (DI) and pass mocks during testing

2. **Multi-thread race conditions**: Multiple threads may attempt instance creation simultaneously
   -> Countermeasure: Double-checked locking with a Lock, or use module-level variables

3. **Hidden dependencies**: Code that directly references a Singleton has implicit dependencies
   -> Countermeasure: Use constructor injection to explicitly declare dependencies

</details>

### 8.2 Intermediate Level

**Problem 3**: Implement a logging system using the Decorator pattern that satisfies the following specifications:

- Base log output (display strings to the console)
- A Decorator that adds timestamps
- A Decorator that adds log levels (INFO, WARN, ERROR)
- A Decorator that converts to JSON format
- These can be freely combined

<details>
<summary>Example Answer</summary>

```python
from abc import ABC, abstractmethod
from datetime import datetime
import json


class Logger(ABC):
    @abstractmethod
    def log(self, message: str) -> str:
        ...


class ConsoleLogger(Logger):
    def log(self, message: str) -> str:
        print(message)
        return message


class LogDecorator(Logger):
    def __init__(self, logger: Logger):
        self._logger = logger


class TimestampDecorator(LogDecorator):
    def log(self, message: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self._logger.log(f"[{timestamp}] {message}")


class LevelDecorator(LogDecorator):
    def __init__(self, logger: Logger, level: str = "INFO"):
        super().__init__(logger)
        self._level = level

    def log(self, message: str) -> str:
        return self._logger.log(f"[{self._level}] {message}")


class JsonDecorator(LogDecorator):
    def log(self, message: str) -> str:
        payload = json.dumps({"message": message}, ensure_ascii=False)
        return self._logger.log(payload)


# Combination example
logger = ConsoleLogger()
logger = TimestampDecorator(logger)
logger = LevelDecorator(logger, "ERROR")
logger.log("Database connection failed")
# [ERROR] [2026-01-15 10:30:00] Database connection failed
```

</details>

**Problem 4**: Use the Command pattern to implement Undo/Redo functionality
for a simple calculator (four arithmetic operations).

<details>
<summary>Example Answer</summary>

```python
from abc import ABC, abstractmethod


class CalculatorCommand(ABC):
    @abstractmethod
    def execute(self) -> float: ...
    @abstractmethod
    def undo(self) -> float: ...


class Calculator:
    def __init__(self):
        self.value = 0.0
        self._history: list[CalculatorCommand] = []
        self._redo_stack: list[CalculatorCommand] = []

    def execute(self, command: "CalculatorCommand") -> float:
        result = command.execute()
        self._history.append(command)
        self._redo_stack.clear()
        return result

    def undo(self) -> float:
        if self._history:
            cmd = self._history.pop()
            cmd.undo()
            self._redo_stack.append(cmd)
        return self.value

    def redo(self) -> float:
        if self._redo_stack:
            cmd = self._redo_stack.pop()
            cmd.execute()
            self._history.append(cmd)
        return self.value


class AddCommand(CalculatorCommand):
    def __init__(self, calc: Calculator, operand: float):
        self._calc = calc
        self._operand = operand

    def execute(self) -> float:
        self._calc.value += self._operand
        return self._calc.value

    def undo(self) -> float:
        self._calc.value -= self._operand
        return self._calc.value


class MultiplyCommand(CalculatorCommand):
    def __init__(self, calc: Calculator, operand: float):
        self._calc = calc
        self._operand = operand
        self._prev_value = 0.0

    def execute(self) -> float:
        self._prev_value = self._calc.value
        self._calc.value *= self._operand
        return self._calc.value

    def undo(self) -> float:
        self._calc.value = self._prev_value
        return self._calc.value


calc = Calculator()
calc.execute(AddCommand(calc, 10))     # 10.0
calc.execute(AddCommand(calc, 5))      # 15.0
calc.execute(MultiplyCommand(calc, 3)) # 45.0
calc.undo()                            # 15.0
calc.undo()                            # 10.0
calc.redo()                            # 15.0
```

</details>

### 8.3 Advanced Level

**Problem 5**: Design and implement a plugin system that satisfies the following requirements.
Combine multiple design patterns.

Requirements:
- Dynamic registration and unregistration of plugins (Factory + Registry)
- Control over plugin execution order (Chain of Responsibility)
- Logging of plugin execution results (Observer)
- Execute all plugins with a single command (Facade)

Hint: First design the interface for each component,
then combine patterns to compose the whole.

<details>
<summary>Design Direction</summary>

```python
from abc import ABC, abstractmethod
from typing import Any


class Plugin(ABC):
    """Base class for plugins."""
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def priority(self) -> int: ...
    @abstractmethod
    def execute(self, context: dict) -> dict: ...


class PluginRegistry:
    """Factory + Registry: Manages plugin registration and creation."""
    _plugins: dict[str, type[Plugin]] = {}

    @classmethod
    def register(cls, plugin_class: type[Plugin]) -> None:
        instance = plugin_class()
        cls._plugins[instance.name()] = plugin_class

    @classmethod
    def create(cls, name: str) -> Plugin:
        return cls._plugins[name]()

    @classmethod
    def create_all(cls) -> list[Plugin]:
        plugins = [cls.create(name) for name in cls._plugins]
        return sorted(plugins, key=lambda p: p.priority())


class PluginEventBus:
    """Observer: Notifies plugin execution events."""
    _listeners: list = []

    @classmethod
    def subscribe(cls, listener) -> None:
        cls._listeners.append(listener)

    @classmethod
    def publish(cls, event: str, data: Any) -> None:
        for listener in cls._listeners:
            listener(event, data)


class PluginEngine:
    """Facade: Unified entry point for the plugin system."""
    def __init__(self):
        self._plugins = PluginRegistry.create_all()

    def run(self, context: dict) -> dict:
        for plugin in self._plugins:
            PluginEventBus.publish("before", {"plugin": plugin.name()})
            context = plugin.execute(context)
            PluginEventBus.publish("after", {"plugin": plugin.name(), "context": context})
        return context
```

The implementation of concrete plugins and test code is left as an exercise for the reader.
The important point is understanding that each pattern has an independent responsibility,
and that combining them enables building flexible systems.

</details>

---

## 9. Frequently Asked Questions (FAQ)

### Q1: When should I learn design patterns?

The optimal time to learn design patterns is after gaining some coding experience.
As a guideline, you can learn effectively if you meet the following conditions:

- You understand the basics of object-oriented programming (classes, inheritance, polymorphism)
- You have experience writing programs of several thousand lines or more
- You have felt "Could this code be better structured?"

Even if beginners memorize patterns, they are hard to apply in practice.
First, write code and experience the "pain,"
then learn patterns as a means to solve that pain -- this is the most effective approach.

### Q2: Do I need to memorize all 23 patterns?

You do not need to memorize all of them. What matters is the following:

1. **Understanding the categories**: Understand the 3 categories -- Creational, Structural, Behavioral
2. **Mastery of frequently used patterns**: The following 8-10 patterns appear frequently in practice
   - Singleton, Factory Method, Builder (Creational)
   - Adapter, Decorator, Facade (Structural)
   - Observer, Strategy, Command, Iterator (Behavioral)
3. **Awareness as a toolkit**: For the remaining patterns, know that "for this kind of problem, this kind of solution exists" and be able to look up details when needed

### Q3: I heard patterns are unnecessary in Python -- is that true?

It is half right and half wrong.

Python's dynamic typing, duck typing, first-class functions, and decorator syntax
do make "boilerplate patterns" that were necessary in Java/C++ unnecessary in some cases.
For example, the Strategy pattern can be achieved by simply passing functions instead of creating classes.

However, the essence of patterns is not "implementation methods" but "design intent."
By conveying "this is designed with the Strategy concept,"
team members can instantly understand the design intent.
Even when the language changes, the "concepts" of patterns remain valid.

What matters in Python is to understand the intent of patterns
and implement them in a Pythonic way.
Avoid directly porting Java pattern implementations to Python.

### Q4: What should I do when I am unsure about applying a pattern?

When in doubt, the following steps can help:

1. **Write it simply first**: Implement without patterns
2. **Refactor when you feel pain**: When you notice duplication, growing conditional branches, or difficulty making changes
3. **Match against pattern intents**: Check which pattern's "problem" corresponds to the pain you felt
4. **Evaluate trade-offs**: Weigh the increased complexity from applying the pattern against the flexibility gained
5. **Discuss with the team**: Adapt pattern application to the team's understanding level

"When in doubt, keep it simple" is the safest principle.

### Q5: How do design patterns change with microservices?

In microservice architectures, distributed system-specific patterns
become important in addition to GoF patterns:

- **Circuit Breaker**: Block calls to failing services to prevent cascading failures
- **Saga**: Manage distributed transactions as a series of local transactions
- **CQRS**: Separate command (write) and query (read) models
- **Event Sourcing**: Record state changes as a sequence of events
- **API Gateway**: Integrate access to multiple microservices (distributed version of Facade)
- **Sidecar**: Handle cross-cutting concerns in a process attached to the service

GoF patterns are "intra-class/intra-process" designs,
while distributed patterns are "inter-service/inter-process" designs.
They are not mutually exclusive; they simply operate at different layers.

---

## 10. Pattern Combinations and Practical Guidelines

### 10.1 Commonly Used Pattern Combinations

In real projects, multiple patterns are often combined.
Below are representative combinations.

| Combination | Typical Use | Description |
|-------------|-----------|-------------|
| Factory + Strategy | Plugin systems | Factory creates Strategies, switched at runtime |
| Observer + Command | Event-driven UI | Events (Observer) trigger commands (Command) |
| Composite + Iterator | Tree traversal | Iterate over Composite structures sequentially |
| Facade + Adapter | Legacy integration | Adapter converts, Facade provides unified entry point |
| Builder + Factory | Complex object groups | Factory decides which Builder to use |
| Decorator + Strategy | Middleware pipeline | Layer with Decorators, each layer processes with Strategy |

### 10.2 When to Introduce Patterns Through Refactoring

The principle is "introduce when needed" rather than "include from the start."
Consider introducing patterns when the following signals appear:

1. **Same conditional branches appear in multiple places** -> Strategy / State
2. **Adding a new type requires modifying existing code** -> Factory Method
3. **Notifications between objects become complex** -> Observer / Mediator
4. **Undo functionality is requested** -> Command / Memento
5. **External API connections increase** -> Adapter / Facade
6. **Constructor has 5 or more arguments** -> Builder

---


## FAQ

### Q1: What is the most important point in learning this topic?

Building practical experience is most important. Understanding deepens not just through theory, but by actually writing and running code.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend solidly understanding the fundamental concepts explained in this guide before moving to the next step.

### Q3: How is this used in professional practice?

The knowledge of this topic is frequently used in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## 11. Summary

### 11.1 What You Learned in This Chapter

| Category | Patterns Learned | Core Concept |
|----------|-----------------|--------------|
| Creational | Singleton, Factory Method, Builder, Prototype | Flexibility and control over object creation |
| Structural | Adapter, Decorator, Facade, Composite | Composition of structures and managing complexity |
| Behavioral | Observer, Strategy, Command, Iterator | Responsibility distribution and communication organization |
| Architecture | MVC, MVVM, Repository | Structuring the entire application |

### 11.2 Pattern Mastery Roadmap

```
Phase 1: Foundational Understanding (content of this chapter)
  +-- Understand the 3 GoF categories
  +-- Be able to explain the intent of 8-10 frequently used patterns
  +-- Transcribe Python implementation examples

Phase 2: Practical Application
  +-- Discover patterns in existing code
  +-- Introduce patterns through refactoring
  +-- Experience combining patterns

Phase 3: Design Judgment
  +-- Select appropriate patterns for problems
  +-- Judge application/non-application of patterns based on trade-offs
  +-- Explain the intent of patterns to the team

Phase 4: Application and Advanced Topics
  +-- Understand distributed system patterns
  +-- Learn Domain-Driven Design (DDD) patterns
  +-- Compare with functional programming patterns
```

### 11.3 Mindset for Design

1. **Patterns are means, not ends**: Be careful that using patterns does not become a goal in itself
2. **Prioritize simplicity**: The simplest design is often the best design
3. **Understand the problem before searching for patterns**: Browsing pattern catalogs to find application targets is putting the cart before the horse
4. **Always be aware of trade-offs**: Every pattern has both advantages and disadvantages
5. **Adapt to the team's context**: Consider the team's technical ability, project size, and maintenance period

---

## Recommended Next Reading

---

## References

1. Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
   - The original work on GoF design patterns. Contains definitions and C++/Smalltalk implementation examples for 23 patterns.

2. Freeman, E., Robson, E., Bates, B., & Sierra, K. (2020). *Head First Design Patterns* (2nd Edition). O'Reilly Media.
   - An introductory book that teaches patterns through illustrations and dialogue. Java-based but ideal for conceptual understanding.

3. Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley.
   - A collection of architecture patterns for enterprise applications. Systematizes Repository, Unit of Work, Data Mapper, etc.

4. Martin, R. C. (2017). *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall.
   - Explains the relationship between SOLID principles and architecture patterns. Useful for the Dependency Inversion Principle and pattern application judgment.

5. Buschmann, F. et al. (1996). *Pattern-Oriented Software Architecture Volume 1: A System of Patterns*. Wiley.
   - A book that systematizes architecture-level patterns (MVC, Pipes and Filters, Broker, etc.).
