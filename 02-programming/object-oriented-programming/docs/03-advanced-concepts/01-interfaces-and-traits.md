# Interfaces and Traits

> Interfaces define "contracts" while traits provide "reusable behavior." Understand the differences in implementation across languages and the relationship with duck typing.

## What You Will Learn in This Chapter

- [ ] Understand the difference between interfaces and traits
- [ ] Grasp implementation methods in each language
- [ ] Learn the relationship between structural typing and duck typing
- [ ] Master best practices for interface design
- [ ] Understand how differences in type systems affect design


## Prerequisites

Reading this guide will be more meaningful if you have the following knowledge:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding the content of [Composition vs Inheritance](./00-composition-vs-inheritance.md)

---

## 1. Interfaces vs Traits vs Abstract Classes

```
┌──────────────┬────────────────┬────────────────┬────────────────┐
│              │ Interface      │ Trait          │ Abstract Class │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ Method decl. │ Yes            │ Yes            │ Yes            │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ Default impl │ Partial(lang)  │ Yes            │ Yes            │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ Fields       │ No             │ Partial(lang)  │ Yes            │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ Multi impl.  │ Yes            │ Yes            │ No             │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ Constructor  │ No             │ No             │ Yes            │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ Access mod.  │ public only    │ Partial(lang)  │ All allowed    │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ Languages    │ Java, TS, Go   │ Rust,Scala,PHP │ Java, Python   │
└──────────────┴────────────────┴────────────────┴────────────────┘

Selection guidelines:
  Interface: When you want to define a contract of "what can be done"
  Trait: When you want to provide reusable behavior implementations
  Abstract Class: When you want to share common state and partial implementations
```

### 1.1 Relationships Between Concepts

```
Three layers of the type system:

  1. Contract Layer
     - Interface: Declares "what can be done"
     - Only method signatures
     - Holds no implementation (in principle)

  2. Behavior Layer
     - Trait: Defines reusable behavior
     - Provides default implementations
     - Holds no state (in principle)

  3. Implementation Layer
     - Abstract class / Concrete class: Complete implementation
     - Holds state (fields)
     - Has constructors

  Evolution of implementations:
    Interface (declarations only)
         ↓ Adds default methods
    Trait (declarations + default implementations)
         ↓ Adds state
    Abstract class (declarations + implementations + state)
         ↓ All methods implemented
    Concrete class (complete implementation)

  Recent language trends:
    - Boundary between interfaces and traits has blurred
    - Java 8+: Default methods on interfaces
    - Kotlin: Properties on interfaces
    - Swift: Protocol extensions
    - PHP 8: Traits close to interfaces
```

---

## 2. Implementation in Each Language

### 2.1 Java: Interfaces

```java
// Java: Interface (with default methods)
public interface Comparable<T> {
    int compareTo(T other);
}

public interface Printable {
    void print();

    // Default method (Java 8+)
    default void printWithBorder() {
        System.out.println("================");
        print();
        System.out.println("================");
    }
}

// Implement multiple interfaces
public class Product implements Comparable<Product>, Printable {
    private String name;
    private int price;

    @Override
    public int compareTo(Product other) {
        return Integer.compare(this.price, other.price);
    }

    @Override
    public void print() {
        System.out.printf("%s: ¥%d%n", name, price);
    }
}
```

```java
// Java: Advanced usage of interfaces

// 1. Functional interface (SAM: Single Abstract Method)
@FunctionalInterface
public interface Predicate<T> {
    boolean test(T t);

    // Composition via default methods
    default Predicate<T> and(Predicate<T> other) {
        return t -> this.test(t) && other.test(t);
    }

    default Predicate<T> or(Predicate<T> other) {
        return t -> this.test(t) || other.test(t);
    }

    default Predicate<T> negate() {
        return t -> !this.test(t);
    }

    // static factory method
    static <T> Predicate<T> isEqual(Object targetRef) {
        return t -> Objects.equals(t, targetRef);
    }
}

// Use with lambda expressions
Predicate<String> isNotEmpty = s -> !s.isEmpty();
Predicate<String> isLongEnough = s -> s.length() >= 8;
Predicate<String> isValidPassword = isNotEmpty.and(isLongEnough);

// 2. sealed interface (Java 17+)
public sealed interface Shape
    permits Circle, Rectangle, Triangle {
    double area();
    double perimeter();
}

public record Circle(double radius) implements Shape {
    @Override
    public double area() { return Math.PI * radius * radius; }
    @Override
    public double perimeter() { return 2 * Math.PI * radius; }
}

public record Rectangle(double width, double height) implements Shape {
    @Override
    public double area() { return width * height; }
    @Override
    public double perimeter() { return 2 * (width + height); }
}

public record Triangle(double a, double b, double c) implements Shape {
    @Override
    public double area() {
        double s = (a + b + c) / 2;
        return Math.sqrt(s * (s - a) * (s - b) * (s - c));
    }
    @Override
    public double perimeter() { return a + b + c; }
}

// Pattern matching (Java 21+)
public String describeShape(Shape shape) {
    return switch (shape) {
        case Circle c -> "Circle with radius " + c.radius();
        case Rectangle r -> r.width() + "x" + r.height() + " rectangle";
        case Triangle t -> "Triangle (sides: " + t.a() + ", " + t.b() + ", " + t.c() + ")";
    };
}

// 3. Default method conflict in interfaces
public interface A {
    default String greet() { return "Hello from A"; }
}

public interface B {
    default String greet() { return "Hello from B"; }
}

// When implementing both, explicit override is required
public class C implements A, B {
    @Override
    public String greet() {
        // Explicitly choose one
        return A.super.greet();
    }
}
```

### 2.2 Rust: Traits

```rust
// Rust: Traits (interface + default implementation + generics constraints)
trait Summary {
    fn summarize_author(&self) -> String;

    // Default implementation
    fn summarize(&self) -> String {
        format!("(Breaking news from {}...)", self.summarize_author())
    }
}

struct Article {
    title: String,
    author: String,
    content: String,
}

impl Summary for Article {
    fn summarize_author(&self) -> String {
        self.author.clone()
    }

    // summarize() uses the default implementation
}

// Trait bound: used as a generics constraint
fn notify(item: &impl Summary) {
    println!("Breaking: {}", item.summarize());
}

// Combining multiple traits
fn display_and_summarize(item: &(impl Summary + std::fmt::Display)) {
    println!("{}", item);
    println!("{}", item.summarize());
}
```

```rust
// Rust: Advanced usage of traits

// 1. Associated Types
trait Iterator {
    type Item;  // Associated type: implementer specifies the concrete type

    fn next(&mut self) -> Option<Self::Item>;

    // Default implementation: method using the associated type
    fn count(mut self) -> usize
    where
        Self: Sized,
    {
        let mut count = 0;
        while self.next().is_some() {
            count += 1;
        }
        count
    }
}

struct Counter {
    count: u32,
    max: u32,
}

impl Iterator for Counter {
    type Item = u32;  // Elements of this Iterator are u32

    fn next(&mut self) -> Option<u32> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}

// 2. Supertraits (trait inheritance)
trait Animal {
    fn name(&self) -> &str;
}

trait Pet: Animal {  // Pet requires Animal as a supertrait
    fn cuddle(&self) -> String {
        format!("Petting {}", self.name())
    }
}

struct Dog {
    name: String,
}

impl Animal for Dog {
    fn name(&self) -> &str {
        &self.name
    }
}

impl Pet for Dog {
    // cuddle() uses the default implementation
}

// 3. Trait objects (dynamic dispatch)
fn print_summaries(items: &[&dyn Summary]) {
    for item in items {
        println!("{}", item.summarize());
    }
}

// 4. Blanket implementation
// Automatically implements ToString for all types that implement Display
impl<T: std::fmt::Display> ToString for T {
    fn to_string(&self) -> String {
        format!("{}", self)
    }
}

// 5. From/Into traits (type conversion)
struct Celsius(f64);
struct Fahrenheit(f64);

impl From<Celsius> for Fahrenheit {
    fn from(c: Celsius) -> Self {
        Fahrenheit(c.0 * 9.0 / 5.0 + 32.0)
    }
}

// Into is automatically derived from From
let c = Celsius(100.0);
let f: Fahrenheit = c.into();  // Fahrenheit(212.0)

// 6. Derive macro (automatic trait implementation)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Point {
    x: i32,
    y: i32,
}
// Debug, Clone, PartialEq, Eq, Hash are automatically implemented

// 7. Newtype pattern (implementing traits for external types)
struct Meters(f64);

impl std::fmt::Display for Meters {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}m", self.0)
    }
}

impl std::ops::Add for Meters {
    type Output = Meters;
    fn add(self, other: Meters) -> Meters {
        Meters(self.0 + other.0)
    }
}
```

### 2.3 Go: Implicit Interfaces

```go
// Go: Structural typing (implicitly satisfies interfaces)
type Writer interface {
    Write(p []byte) (n int, err error)
}

type Reader interface {
    Read(p []byte) (n int, err error)
}

// ReadWriter is a composition of Writer and Reader
type ReadWriter interface {
    Reader
    Writer
}

// MyBuffer satisfies Writer "without declaration"
type MyBuffer struct {
    data []byte
}

func (b *MyBuffer) Write(p []byte) (int, error) {
    b.data = append(b.data, p...)
    return len(p), nil
}

// No "implements Writer" needed (satisfies implicitly)
var w Writer = &MyBuffer{}
```

```go
// Go: Advanced usage of interfaces

// 1. Small interfaces (Go's philosophy)
// Go interfaces are usually 1-3 methods
type Stringer interface {
    String() string
}

type Closer interface {
    Close() error
}

type ReadCloser interface {
    Reader
    Closer
}

// 2. Empty interface (any / interface{})
func PrintAnything(v any) {
    fmt.Println(v)
}

// 3. Type assertion
func Process(r Reader) {
    // Check if r also satisfies Closer
    if closer, ok := r.(Closer); ok {
        defer closer.Close()
    }

    // Type switch
    switch v := r.(type) {
    case *os.File:
        fmt.Println("File:", v.Name())
    case *bytes.Buffer:
        fmt.Println("Buffer:", v.Len(), "bytes")
    default:
        fmt.Println("Unknown Reader")
    }
}

// 4. Interface composition pattern
type Handler interface {
    Handle(ctx context.Context, req Request) (Response, error)
}

type Middleware func(Handler) Handler

// Chain of middleware
func Chain(h Handler, middlewares ...Middleware) Handler {
    for i := len(middlewares) - 1; i >= 0; i-- {
        h = middlewaresi
    }
    return h
}

// Logging middleware
func LoggingMiddleware(next Handler) Handler {
    return HandlerFunc(func(ctx context.Context, req Request) (Response, error) {
        start := time.Now()
        resp, err := next.Handle(ctx, req)
        log.Printf("handled in %v", time.Since(start))
        return resp, err
    })
}

// Authentication middleware
func AuthMiddleware(next Handler) Handler {
    return HandlerFunc(func(ctx context.Context, req Request) (Response, error) {
        token := req.Header("Authorization")
        if token == "" {
            return nil, ErrUnauthorized
        }
        // Token validation...
        return next.Handle(ctx, req)
    })
}

// Usage example
handler := Chain(myHandler, LoggingMiddleware, AuthMiddleware)

// 5. Compile-time interface conformance check
// Idiom to ensure a struct satisfies an interface
var _ Writer = (*MyBuffer)(nil)
var _ Reader = (*MyBuffer)(nil)
// Compile error if MyBuffer does not satisfy Writer/Reader
```

### 2.4 TypeScript: Structural Typing

```typescript
// TypeScript: Structural Typing
interface Loggable {
  toLogString(): string;
}

// No explicit 'implements' needed; matching structure is enough
class User {
  constructor(public name: string, public email: string) {}

  toLogString(): string {
    return `User(${this.name}, ${this.email})`;
  }
}

// User does not explicitly implement Loggable, but since it
// has toLogString() it can be used as Loggable
function log(item: Loggable): void {
  console.log(item.toLogString());
}

log(new User("Tanaka", "tanaka@example.com")); // OK
```

```typescript
// TypeScript: Advanced interface usage

// 1. Generic interfaces
interface Repository<T> {
  findById(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: string): Promise<void>;
}

interface Identifiable {
  id: string;
}

// Generic constraint
interface CrudRepository<T extends Identifiable> extends Repository<T> {
  update(id: string, data: Partial<T>): Promise<T>;
}

// 2. Index signatures
interface Dictionary<T> {
  [key: string]: T;
}

const scores: Dictionary<number> = {
  math: 90,
  english: 85,
  science: 92,
};

// 3. Call signatures
interface Formatter {
  (value: unknown): string;
  locale: string;
}

const jsonFormatter: Formatter = Object.assign(
  (value: unknown) => JSON.stringify(value),
  { locale: "ja-JP" },
);

// 4. Intersection types (type composition)
interface HasName {
  name: string;
}

interface HasEmail {
  email: string;
}

interface HasAge {
  age: number;
}

// Compose with intersection type
type UserInfo = HasName & HasEmail & HasAge;

const user: UserInfo = {
  name: "Tanaka",
  email: "tanaka@example.com",
  age: 30,
};

// 5. Conditional Types with interfaces
interface ApiResponse<T> {
  data: T;
  status: number;
  message: string;
}

type UnwrapResponse<T> = T extends ApiResponse<infer U> ? U : never;

type UserData = UnwrapResponse<ApiResponse<User>>; // User

// 6. Mapped Types
interface User {
  id: string;
  name: string;
  email: string;
  age: number;
}

// Make all fields optional
type PartialUser = Partial<User>;

// Make all fields readonly
type ReadonlyUser = Readonly<User>;

// Pick specific fields only
type UserPreview = Pick<User, "id" | "name">;

// Exclude specific fields
type UserWithoutId = Omit<User, "id">;

// 7. Template Literal Types with interfaces
type EventName = "click" | "hover" | "focus";
type HandlerName = `on${Capitalize<EventName>}`;
// "onClick" | "onHover" | "onFocus"

interface EventHandlers {
  onClick(event: MouseEvent): void;
  onHover(event: MouseEvent): void;
  onFocus(event: FocusEvent): void;
}
```

### 2.5 Python: Protocol (Structural Subtyping)

```python
# Python: Interface via Protocol (Python 3.8+)
from typing import Protocol, runtime_checkable


# Protocol: Structural typing (type-safe version of duck typing)
class Renderable(Protocol):
    """Contract for renderable objects"""
    def render(self) -> str: ...


class HtmlComponent:
    """No need to explicitly implement the Protocol"""
    def __init__(self, tag: str, content: str):
        self.tag = tag
        self.content = content

    def render(self) -> str:
        return f"<{self.tag}>{self.content}</{self.tag}>"


class MarkdownText:
    """Also Renderable since it has render()"""
    def __init__(self, text: str):
        self.text = text

    def render(self) -> str:
        return self.text


# Accepts anything that has render()
def display(item: Renderable) -> None:
    print(item.render())


display(HtmlComponent("h1", "Hello"))  # <h1>Hello</h1>
display(MarkdownText("# Hello"))       # # Hello
```

```python
# Python: Advanced usage of Protocol

from typing import Protocol, runtime_checkable, TypeVar, Generic
from abc import abstractmethod


# 1. runtime_checkable: usable with isinstance()
@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...


class FileWrapper:
    def __init__(self, path: str):
        self.file = open(path)

    def close(self) -> None:
        self.file.close()


# Can be checked with isinstance
wrapper = FileWrapper("test.txt")
assert isinstance(wrapper, Closable)  # True


# 2. Generic Protocol
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class Comparable(Protocol[T]):
    """Comparable object"""
    def __lt__(self, other: T) -> bool: ...
    def __le__(self, other: T) -> bool: ...
    def __gt__(self, other: T) -> bool: ...
    def __ge__(self, other: T) -> bool: ...


class SupportsAdd(Protocol[T_co]):
    """Object supporting addition"""
    def __add__(self, other: "SupportsAdd[T_co]") -> T_co: ...


# 3. Protocol methods cannot have default implementations,
#    but can be combined with Mixins
class EqualityMixin:
    """Mixin providing default equality comparison"""
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class HashableMixin(EqualityMixin):
    """Default hash implementation"""
    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))


# 4. Dependency injection with Protocol
class UserRepository(Protocol):
    async def find_by_id(self, user_id: str) -> dict | None: ...
    async def save(self, user: dict) -> None: ...


class EmailSender(Protocol):
    async def send(self, to: str, subject: str, body: str) -> None: ...


class Logger(Protocol):
    def info(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...


class UserService:
    """Depends on Protocol (not on concrete classes)"""
    def __init__(
        self,
        repo: UserRepository,
        email: EmailSender,
        logger: Logger,
    ):
        self.repo = repo
        self.email = email
        self.logger = logger

    async def register(self, name: str, email_addr: str) -> dict:
        self.logger.info(f"Registering user: {email_addr}")
        user = {"name": name, "email": email_addr}
        await self.repo.save(user)
        await self.email.send(email_addr, "Welcome!", "Thank you for registering")
        return user


# Mocks for testing (satisfying the Protocol is enough)
class MockUserRepository:
    def __init__(self):
        self.users: list[dict] = []

    async def find_by_id(self, user_id: str) -> dict | None:
        return next((u for u in self.users if u.get("id") == user_id), None)

    async def save(self, user: dict) -> None:
        self.users.append(user)


class MockEmailSender:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, to: str, subject: str, body: str) -> None:
        self.sent.append({"to": to, "subject": subject, "body": body})


class MockLogger:
    def __init__(self):
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(f"[INFO] {message}")

    def error(self, message: str) -> None:
        self.messages.append(f"[ERROR] {message}")


# Test
import asyncio

async def test_register():
    repo = MockUserRepository()
    email = MockEmailSender()
    logger = MockLogger()

    service = UserService(repo, email, logger)
    user = await service.register("Tanaka", "tanaka@example.com")

    assert len(repo.users) == 1
    assert len(email.sent) == 1
    assert email.sent[0]["to"] == "tanaka@example.com"
    assert "[INFO] Registering user: tanaka@example.com" in logger.messages

asyncio.run(test_register())
```

### 2.6 Scala: Traits

```scala
// Scala: Traits (interface + default implementation + state)
trait Greeter {
  // Abstract method
  def name: String

  // Default implementation
  def greet(): String = s"Hello, $name!"
}

trait Logger {
  // Traits can hold state
  var logLevel: String = "INFO"

  def log(message: String): Unit = {
    println(s"[$logLevel] $message")
  }
}

trait Serializable {
  def toJson: String
}

// Compose multiple traits
class User(val name: String, val email: String)
    extends Greeter
    with Logger
    with Serializable {

  override def toJson: String =
    s"""{"name": "$name", "email": "$email"}"""
}

// self-type: declare dependency
trait Repository {
  self: Logger =>  // Repository requires Logger
  def save(data: String): Unit = {
    log(s"Saving: $data")
    // Persistence logic
  }
}

// Dynamic mixin (add trait when instantiating)
val user = new User("Tanaka", "tanaka@example.com") with Serializable {
  override def toJson: String = s"""{"name": "$name"}"""
}

// Cake pattern (DI)
trait UserRepositoryComponent {
  val userRepository: UserRepository
  trait UserRepository {
    def findById(id: String): Option[User]
  }
}

trait UserServiceComponent {
  self: UserRepositoryComponent =>
  val userService: UserService
  class UserService {
    def getUser(id: String): Option[User] =
      userRepository.findById(id)
  }
}
```

### 2.7 Swift: Protocols

```swift
// Swift: Protocols (interface + Protocol Extensions)
protocol Drawable {
    func draw()
}

protocol Resizable {
    var width: Double { get set }
    var height: Double { get set }
    func resize(by factor: Double)
}

// Protocol Extension: provide default implementations
extension Resizable {
    func resize(by factor: Double) {
        width *= factor
        height *= factor
    }

    var area: Double {
        return width * height
    }
}

// Protocol Composition
typealias InteractiveElement = Drawable & Resizable

struct Button: InteractiveElement {
    var label: String
    var width: Double
    var height: Double

    func draw() {
        print("Drawing button: \(label) (\(width)x\(height))")
    }
}

// Associated Types
protocol Container {
    associatedtype Item  // Associated type
    var count: Int { get }
    mutating func append(_ item: Item)
    subscript(i: Int) -> Item { get }
}

struct Stack<Element>: Container {
    typealias Item = Element  // Specify the associated type (can be omitted if inferable)
    var items: [Element] = []
    var count: Int { items.count }
    mutating func append(_ item: Element) { items.append(item) }
    subscript(i: Int) -> Element { items[i] }
}

// Generic constraint via where clause
func allEqual<C: Container>(_ container: C) -> Bool
    where C.Item: Equatable {
    if container.count < 2 { return true }
    for i in 1..<container.count {
        if container[i] != container[0] { return false }
    }
    return true
}

// Existential Types (any keyword, Swift 5.7+)
func printAll(_ items: [any Drawable]) {
    for item in items {
        item.draw()
    }
}

// Opaque Types (some keyword)
func makeShape() -> some Drawable {
    return Button(label: "OK", width: 100, height: 40)
}
```

---

## 3. Duck Typing

```
"If it walks like a duck and quacks like a duck, it's a duck"

Nominal Typing:
  - Only types explicitly implementing/extending are compatible
  - Java, C#, Swift

Structural Typing:
  - Compatible if structure (methods/properties) matches
  - TypeScript, Go

Duck Typing:
  - Can be called if method exists at runtime
  - Python, Ruby, JavaScript

Strictness of type checking:
  Nominal typing > Structural typing > Duck typing

Tradeoff between safety and flexibility:
  Nominal typing: High safety / Low flexibility
  Structural typing: Medium safety / Medium flexibility
  Duck typing: Low safety / High flexibility
```

```python
# Python: Duck typing
class Duck:
    def quack(self):
        return "Quack quack"

class Person:
    def quack(self):
        return "(Human imitating) Quack quack"

class RubberDuck:
    def quack(self):
        return "Squeak squeak"

# No type declaration; accepts anything that has quack()
def make_it_quack(thing):
    print(thing.quack())

make_it_quack(Duck())       # Quack quack
make_it_quack(Person())     # (Human imitating) Quack quack
make_it_quack(RubberDuck()) # Squeak squeak

# Protocol (Python 3.8+): make duck typing type-safe via type hints
from typing import Protocol

class Quackable(Protocol):
    def quack(self) -> str: ...

def make_it_quack_typed(thing: Quackable) -> None:
    print(thing.quack())
```

```ruby
# Ruby: Duck typing
class Logger
  def write(message)
    puts "[LOG] #{message}"
  end
end

class FileWriter
  def initialize(path)
    @file = File.open(path, 'a')
  end

  def write(message)
    @file.puts(message)
  end

  def close
    @file.close
  end
end

class NullWriter
  def write(message)
    # Do nothing
  end
end

# Accepts anything that has write()
def process(writer, data)
  writer.write("Processing: #{data}")
  # Does not care about the concrete type of writer
end

process(Logger.new, "test data")
process(FileWriter.new("output.log"), "test data")
process(NullWriter.new, "test data")

# Use respond_to? to check for method existence
def safe_write(writer, message)
  if writer.respond_to?(:write)
    writer.write(message)
  else
    puts "Warning: writer does not support write"
  end
end
```

### 3.1 Comparison of Typing Styles

```typescript
// TypeScript: Benefits and caveats of structural typing

// Benefit 1: Compatibility with third-party libraries
// Interface defined by library A
interface PointA {
  x: number;
  y: number;
}

// Different interface defined by library B
interface PointB {
  x: number;
  y: number;
}

// Compatible if structure is the same even with different names
function distanceA(p: PointA): number {
  return Math.sqrt(p.x ** 2 + p.y ** 2);
}

const pointB: PointB = { x: 3, y: 4 };
distanceA(pointB); // OK (structural typing)
// In Java: compile error (nominal typing)

// Caveat: Same structure but different meaning
interface UserId {
  value: string;
}

interface ProductId {
  value: string;
}

function findUser(id: UserId): User { /* ... */ }
function findProduct(id: ProductId): Product { /* ... */ }

const userId: UserId = { value: "user-123" };
const productId: ProductId = { value: "product-456" };

findUser(productId); // Compiles in TypeScript! (same structure)
// - Semantically wrong
// - Solved with Branded Types

// Branded Types: achieve nominal typing with structural typing
type Brand<T, B extends string> = T & { __brand: B };
type StrictUserId = Brand<string, "UserId">;
type StrictProductId = Brand<string, "ProductId">;

function findUserStrict(id: StrictUserId): User { /* ... */ }
function findProductStrict(id: StrictProductId): Product { /* ... */ }

const strictUserId = "user-123" as StrictUserId;
const strictProductId = "product-456" as StrictProductId;

// findUserStrict(strictProductId); // Compile error!
findUserStrict(strictUserId);        // OK
```

---

## 4. Best Practices for Interface Design

```
1. Keep them small (ISP-compliant):
   - Ideally 1-5 methods
   - "Can every implementer semantically implement
      all methods of this interface?"

2. Design from the client's perspective:
   - Think from users, not from implementers
   - "Does a client exist that needs all methods of this interface?"

3. Convey intent through names:
   - Suffixes like -able, -er, -or
   - Comparable, Serializer, Validator
   - "Can do X", "Thing that does X"

4. Stable contracts:
   - Interfaces are hard to change
   - Don't aim for perfection from the start; add gradually

5. Consider testability:
   - Abstract external dependencies with interfaces
   - Granularity that makes mocking easy

6. Use domain language:
   - Business terms over technical terms
   - interface OrderProcessor > interface DataHandler
```

```typescript
// Good and bad examples of interface design

// Bad example: massive interface
interface DataManager {
  fetch(url: string): Promise<any>;
  save(data: any): Promise<void>;
  delete(id: string): Promise<void>;
  validate(data: any): boolean;
  transform(data: any): any;
  cache(key: string, data: any): void;
  notify(message: string): void;
  log(message: string): void;
  compress(data: any): Buffer;
  encrypt(data: any): Buffer;
}

// Bad example: overly technical name
interface IDataAccessObject {
  executeSQL(query: string): Promise<any>;
  commitTransaction(): Promise<void>;
  rollbackTransaction(): Promise<void>;
}

// Good example: small, domain-oriented
interface OrderRepository {
  findById(id: string): Promise<Order | null>;
  findByUserId(userId: string): Promise<Order[]>;
  save(order: Order): Promise<void>;
}

interface OrderValidator {
  validate(order: Order): ValidationResult;
}

interface PaymentProcessor {
  processPayment(order: Order): Promise<PaymentResult>;
}

interface OrderNotifier {
  notifyOrderCreated(order: Order): Promise<void>;
  notifyOrderShipped(order: Order): Promise<void>;
}

// Good example: functional interfaces
interface Predicate<T> {
  test(value: T): boolean;
}

interface Transformer<I, O> {
  transform(input: I): O;
}

interface AsyncHandler<I, O> {
  handle(input: I): Promise<O>;
}
```

---

## 5. Selection Guide

```
Interface:
  - Defines contract of "what can be done"
  - No implementation (or minimal default)
  - When multiple implementations are needed
  - Enforce common behavior across different types

Trait:
  - Unit of reusable behavior
  - Actively provides default implementations
  - Mixin-style usage
  - Eliminate code duplication while composing flexibly

Abstract class:
  - Common state (fields) + partial implementation
  - Template Method pattern
  - When the is-a relationship is clear
  - When constructor initialization is needed

Recommendations by language:
  Java: Interfaces (leveraging default methods)
  TypeScript: Interfaces (leveraging structural typing)
  Go: Interfaces (small, implicit)
  Rust: Traits (the only abstraction mechanism)
  Python: Protocol (type-safe duck typing)
  Scala: Traits (flexibility with state)
  Swift: Protocols (leveraging Protocol Extensions)
```

### 5.1 Practical Decision Criteria

```
Decision flowchart:

  Q1: "Do you need to share state (fields)?"
  │
  ├── Yes - Abstract class or composition
  │         Q1a: "Is the is-a relationship clear?"
  │         ├── Yes - Abstract class
  │         └── No - Composition
  │
  └── No
      │
      Q2: "Do you want to provide default implementations?"
      │
      ├── Yes - Trait / Interface (default methods)
      │
      └── No - Interface (pure contract)

  Concrete scenarios:

  Scenario 1: Abstracting DB connections
  - Interface
  - Multiple implementations (MySQL, Postgres, SQLite)
  - Implementation classes hold state

  Scenario 2: Logging helper
  - Trait / Mixin
  - Provide default implementations
  - Used cross-cuttingly across many classes

  Scenario 3: Base for UI components
  - Abstract class (provided by the framework)
  - Common state (width, height, visible)
  - Template methods (render, update)

  Scenario 4: Type constraints
  - Interface / trait bound
  - Used as constraint on generics
  - "T satisfies Comparable"
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement appropriate error handling
- Also write test code

```python
# Exercise 1: Template for basic implementation
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
        """Main data processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should be raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Pattern

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced pattern
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
        """Find by key"""
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

# Tests
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
# Exercise 3: Performance optimization
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

    print(f"Inefficient version: {slow_time:.4f} sec")
    print(f"Efficient version:   {fast_time:.6f} sec")
    print(f"Speedup ratio: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be mindful of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks
---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is most important. Understanding deepens not only through theory but also by writing actual code and verifying behavior.

### Q2: What mistakes do beginners often make?

Skipping fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development. It is especially important during code reviews and architectural design.

---

## Summary

| Concept | Characteristics | Representative Languages |
|---------|-----------------|--------------------------|
| Interface | Defines a contract | Java, TS, Go |
| Trait | Reusable behavior | Rust, Scala, PHP |
| Structural typing | Compatible if structure matches | TS, Go |
| Duck typing | Method check at runtime | Python, Ruby |
| Protocol | Type-safe duck typing | Python, Swift |

```
Practical guidelines:

  1. Interfaces are contracts
     - Define "what can be done"
     - "How it is implemented" is up to the implementer

  2. Small interfaces are good interfaces
     - A one-method interface is the most reusable
     - Go's io.Reader, io.Writer are great examples

  3. Leverage language characteristics
     - TypeScript: structural typing - 'implements' can be omitted
     - Go: implicit interfaces - can conform retroactively
     - Rust: trait bounds - use as constraints on generics
     - Python: Protocol - adds type safety to duck typing

  4. Be mindful of testing
     - Abstract external dependencies with interfaces
     - Granularity that makes mock creation easy
```

---

## 6. Interface Evolution Patterns

```
Versioning interfaces:

  Problem: Adding a method to an interface
           breaks all existing implementations

  Solution 1: Default methods (Java 8+)
    - Add methods without breaking existing implementations
    - However, keep default implementations minimal

  Solution 2: Interface splitting
    - Extend with V1 + additional interfaces
    - UserService -> UserService + UserServiceV2

  Solution 3: Adapter pattern
    - Adapt old interfaces to new ones
    - Provide a migration period to switch gradually

  Recommended rules:
    1. Do not change interfaces after release, in principle
    2. Add new features as new interfaces
    3. Use default methods only for backward compatibility
    4. Leverage @Deprecated to migrate in stages
```

```java
// Java: Interface evolution pattern

// V1: Initial release
public interface PaymentGateway {
    PaymentResult charge(String customerId, BigDecimal amount);
    PaymentResult refund(String transactionId);
}

// V2: Add new features (maintain backward compatibility via default methods)
public interface PaymentGateway {
    PaymentResult charge(String customerId, BigDecimal amount);
    PaymentResult refund(String transactionId);

    // Added in V2: backward compatible default implementation
    default PaymentResult chargeWithCurrency(
            String customerId, BigDecimal amount, Currency currency) {
        // By default, call charge without currency conversion
        return charge(customerId, amount);
    }

    // Added in V2: subscription support
    default SubscriptionResult subscribe(
            String customerId, String planId) {
        throw new UnsupportedOperationException(
            "This gateway does not support subscriptions");
    }
}

// Alternate pattern: split interfaces
public interface SubscriptionGateway extends PaymentGateway {
    SubscriptionResult subscribe(String customerId, String planId);
    void cancelSubscription(String subscriptionId);
}
```

```typescript
// TypeScript: Interface extension pattern

// Declaration Merging
// Interfaces with the same name are automatically merged
interface Config {
  host: string;
  port: number;
}

// Added in another place (convenient for extending libraries)
interface Config {
  ssl: boolean;
  timeout: number;
}

// Merged result: { host, port, ssl, timeout }
const config: Config = {
  host: "localhost",
  port: 3000,
  ssl: true,
  timeout: 5000,
};

// Module Augmentation
// Add custom properties to express Request
declare module "express" {
  interface Request {
    user?: {
      id: string;
      role: string;
    };
  }
}

// Extend global types
declare global {
  interface Window {
    myApp: {
      version: string;
      config: Config;
    };
  }
}
```

```go
// Go: Gradual interface extension

// Base interface
type Storage interface {
    Get(key string) ([]byte, error)
    Put(key string, value []byte) error
    Delete(key string) error
}

// Extended interface: batch operations support
type BatchStorage interface {
    Storage
    BatchGet(keys []string) (map[string][]byte, error)
    BatchPut(items map[string][]byte) error
}

// Extended interface: TTL support
type TTLStorage interface {
    Storage
    PutWithTTL(key string, value []byte, ttl time.Duration) error
    GetTTL(key string) (time.Duration, error)
}

// Check for extension support at runtime
func StoreData(s Storage, key string, value []byte, ttl time.Duration) error {
    // Store with TTL if TTL-compatible storage
    if ts, ok := s.(TTLStorage); ok {
        return ts.PutWithTTL(key, value, ttl)
    }
    // Fall back to regular Put
    return s.Put(key, value)
}

// Storage implementation for testing
type MemoryStorage struct {
    data map[string][]byte
    mu   sync.RWMutex
}

func NewMemoryStorage() *MemoryStorage {
    return &MemoryStorage{data: make(map[string][]byte)}
}

func (m *MemoryStorage) Get(key string) ([]byte, error) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    v, ok := m.data[key]
    if !ok {
        return nil, fmt.Errorf("key not found: %s", key)
    }
    return v, nil
}

func (m *MemoryStorage) Put(key string, value []byte) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.data[key] = value
    return nil
}

func (m *MemoryStorage) Delete(key string) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    delete(m.data, key)
    return nil
}

// Compile-time check
var _ Storage = (*MemoryStorage)(nil)
```

---

## 7. Leveraging Interfaces in Testing

```
Testing strategy:

  1. Mocking using interfaces
     - Abstract external dependencies (DB, API, files) with interfaces
     - Inject mock implementations at test time
     - Faster test execution + isolation

  2. Kinds of test doubles:
     - Stub: returns fixed values
     - Mock: verifies calls
     - Fake: simple alternative implementation
     - Spy: records calls while delegating to the real object

  3. Interface design and testability:
     - Small interfaces are easy to mock
     - Single-method interfaces are easiest to test
     - As methods grow, mock creation becomes cumbersome
```

```python
# Python: Testing strategy using Protocol

from typing import Protocol
from dataclasses import dataclass, field
import pytest


# Protocol definitions in production code
class Clock(Protocol):
    def now(self) -> float: ...

class RandomGenerator(Protocol):
    def random(self) -> float: ...

class NotificationSender(Protocol):
    def send(self, recipient: str, message: str) -> bool: ...


# Fake implementations for testing
class FakeClock:
    """For testing: returns a fixed time"""
    def __init__(self, fixed_time: float = 1000.0):
        self._time = fixed_time

    def now(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


class FakeRandom:
    """For testing: returns predetermined values in order"""
    def __init__(self, values: list[float]):
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


@dataclass
class SpyNotificationSender:
    """For testing: spy that records sends"""
    sent: list[dict] = field(default_factory=list)
    should_succeed: bool = True

    def send(self, recipient: str, message: str) -> bool:
        self.sent.append({
            "recipient": recipient,
            "message": message,
        })
        return self.should_succeed


# Production code
class CouponService:
    def __init__(
        self,
        clock: Clock,
        rng: RandomGenerator,
        notifier: NotificationSender,
    ):
        self.clock = clock
        self.rng = rng
        self.notifier = notifier

    def issue_coupon(self, user_email: str) -> str:
        code = f"COUPON-{int(self.rng.random() * 10000):04d}"
        expiry = self.clock.now() + 86400  # 24 hours later
        self.notifier.send(
            user_email,
            f"Coupon code: {code} (expires: {expiry})",
        )
        return code


# Tests
def test_issue_coupon():
    clock = FakeClock(1700000000.0)
    rng = FakeRandom([0.5678])
    notifier = SpyNotificationSender()

    service = CouponService(clock, rng, notifier)
    code = service.issue_coupon("user@example.com")

    assert code == "COUPON-5678"
    assert len(notifier.sent) == 1
    assert notifier.sent[0]["recipient"] == "user@example.com"
    assert "COUPON-5678" in notifier.sent[0]["message"]


def test_issue_coupon_notification_failure():
    clock = FakeClock()
    rng = FakeRandom([0.1234])
    notifier = SpyNotificationSender(should_succeed=False)

    service = CouponService(clock, rng, notifier)
    code = service.issue_coupon("user@example.com")

    # Coupon is still issued even if notification fails
    assert code == "COUPON-1234"
    assert len(notifier.sent) == 1
```

```rust
// Rust: Testing strategy using traits

use std::collections::HashMap;

// Trait definitions in production code
trait UserStore {
    fn find_by_id(&self, id: &str) -> Option<User>;
    fn save(&mut self, user: &User) -> Result<(), StoreError>;
}

trait EmailService {
    fn send(&self, to: &str, subject: &str, body: &str) -> Result<(), EmailError>;
}

#[derive(Debug, Clone)]
struct User {
    id: String,
    name: String,
    email: String,
}

// Mock implementations for testing
#[cfg(test)]
mod tests {
    use super::*;

    struct MockUserStore {
        users: HashMap<String, User>,
        save_calls: Vec<User>,
    }

    impl MockUserStore {
        fn new() -> Self {
            Self {
                users: HashMap::new(),
                save_calls: Vec::new(),
            }
        }

        fn with_user(mut self, user: User) -> Self {
            self.users.insert(user.id.clone(), user);
            self
        }
    }

    impl UserStore for MockUserStore {
        fn find_by_id(&self, id: &str) -> Option<User> {
            self.users.get(id).cloned()
        }

        fn save(&mut self, user: &User) -> Result<(), StoreError> {
            self.save_calls.push(user.clone());
            self.users.insert(user.id.clone(), user.clone());
            Ok(())
        }
    }

    struct MockEmailService {
        sent: Vec<(String, String, String)>,
        should_fail: bool,
    }

    impl MockEmailService {
        fn new() -> Self {
            Self {
                sent: Vec::new(),
                should_fail: false,
            }
        }
    }

    impl EmailService for MockEmailService {
        fn send(&self, to: &str, subject: &str, body: &str) -> Result<(), EmailError> {
            if self.should_fail {
                return Err(EmailError::SendFailed);
            }
            // Note: actual tests need a mutable reference,
            // so in practice use something like RefCell
            Ok(())
        }
    }

    #[test]
    fn test_find_existing_user() {
        let store = MockUserStore::new().with_user(User {
            id: "user-1".to_string(),
            name: "Tanaka".to_string(),
            email: "tanaka@example.com".to_string(),
        });

        let user = store.find_by_id("user-1");
        assert!(user.is_some());
        assert_eq!(user.unwrap().name, "Tanaka");
    }

    #[test]
    fn test_find_nonexistent_user() {
        let store = MockUserStore::new();
        let user = store.find_by_id("nonexistent");
        assert!(user.is_none());
    }
}
```

---

## Recommended Next Guides

---

## References
1. Odersky, M. "Scalable Component Abstractions." OOPSLA, 2005.
2. The Rust Programming Language. "Traits." doc.rust-lang.org.
3. Bloch, J. "Effective Java." 3rd Edition, Addison-Wesley, 2018.
4. The Go Programming Language Specification. "Interface types." golang.org.
5. Python PEP 544. "Protocols: Structural subtyping." 2017.
6. Apple Developer Documentation. "Protocols." developer.apple.com.
7. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
8. Martin, R.C. "Clean Architecture." Prentice Hall, 2017.
