# JVM Language Comparison (Java, Kotlin, Scala, Clojure)

> A family of languages that run on the JVM (Java Virtual Machine). They share Java's massive ecosystem while each evolving with a different philosophy.

## Learning Objectives

- [ ] Understand the JVM language ecosystem and compatibility
- [ ] Understand the characteristics and appropriate use of each language
- [ ] Understand JVM architecture and performance characteristics
- [ ] Be able to compare type system differences across languages
- [ ] Have criteria for language selection in real-world projects
- [ ] Understand the latest developments such as GraalVM and virtual threads


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Systems Language Comparison (C, C++, Rust, Go, Zig)](./01-systems-languages.md)

---

## 1. Comparison Table

```
┌──────────────┬──────────┬──────────┬──────────┬──────────┐
│              │ Java     │ Kotlin   │ Scala    │ Clojure  │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Year Created │ 1995     │ 2011     │ 2003     │ 2007     │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Designer     │ Gosling  │ JetBrains│ Odersky  │ Hickey   │
│              │ (Sun)    │          │          │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Paradigm     │ OOP      │ OOP+FP  │ OOP+FP  │ FP       │
│              │          │ Multi   │ Multi   │ Lisp     │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Typing       │ Static   │ Static   │ Static   │ Dynamic  │
│              │ Nominal  │ Strong   │ Strongest│          │
│              │          │ Inference│ Inference│          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Null Safety  │ None     │ Yes      │ Option  │ nil      │
│              │ (NPE)   │ (Built-in)│         │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Primary      │ Enter-   │ Android │ Data     │ Web      │
│ Use Cases    │ prise    │ Server  │ Distrib. │ Data     │
│              │ Backend  │ Side    │ Process. │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Major FW     │ Spring   │ Ktor    │ Akka     │ Ring     │
│              │ Quarkus  │ Spring  │ Play     │ Luminus  │
│              │ Micronaut│ Exposed │ ZIO      │ Pedestal │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Build Tool   │ Gradle   │ Gradle  │ sbt      │ Leiningen│
│              │ Maven    │ Maven   │ Mill     │ deps.edn │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Test FW      │ JUnit    │ Kotest  │ ScalaTest│ clojure  │
│              │ Mockito  │ MockK   │ Specs2   │ .test    │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Learning     │ Medium   │ Low     │ High     │ High     │
│ Curve        │          │ (if Java│          │ (Lisp    │
│              │          │  exp.)  │          │  family) │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Verbosity    │ Somewhat │ Low     │ Lowest   │ Low      │
│              │ High     │         │          │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Java Interop │ 100%    │ 100%    │ 95%     │ Good     │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Job Market   │ Highest  │ Rapidly  │ Declining│ Niche    │
│              │          │ Growing  │ Trend    │          │
└──────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 2. Understanding JVM Architecture

### 2.1 How the JVM Works

```
┌────────────────────────────────────────────────────────┐
│                   JVM Architecture                     │
├────────────────────────────────────────────────────────┤
│  .java / .kt / .scala / .clj                          │
│       ↓ Compilation                                    │
│  .class files (JVM bytecode)                           │
│       ↓ Class Loading                                  │
│  ┌────────────────────────────────────────────┐       │
│  │  Class Loader                               │       │
│  │  - Bootstrap / Extension / Application      │       │
│  └────────────────────────────────────────────┘       │
│       ↓                                                │
│  ┌────────────────────────────────────────────┐       │
│  │  Runtime Data Areas                         │       │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐ │       │
│  │  │Heap  │ │Stack │ │Method│ │ PC       │ │       │
│  │  │(GC)  │ │      │ │ Area │ │ Register │ │       │
│  │  └──────┘ └──────┘ └──────┘ └──────────┘ │       │
│  └────────────────────────────────────────────┘       │
│       ↓                                                │
│  ┌────────────────────────────────────────────┐       │
│  │  Execution Engine                           │       │
│  │  - Interpreter → JIT Compiler (C1/C2)      │       │
│  │  - Garbage Collector (G1, ZGC, Shenandoah) │       │
│  └────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────┘
```

### 2.2 Advantages of the JVM

```
1. Write Once, Run Anywhere (WORA)
   - The same bytecode runs on all platforms
   - Linux, macOS, Windows, Docker containers

2. JIT Compiler Optimizations
   - C1: Client-oriented, fast compilation
   - C2: Server-oriented, deep optimization
   - Speculative optimization based on execution patterns
   - Inlining, loop unrolling, escape analysis

3. GC (Garbage Collector) Evolution
   - G1 GC: Default (Java 9+), balanced
   - ZGC: Low-latency (<1ms pause), for large heaps
   - Shenandoah: Low-latency, developed by Red Hat
   - Epsilon: No-op GC (for benchmarking)

4. Rich Ecosystem
   - Maven Central: 500,000+ libraries
   - Spring, Hibernate, Apache project family
   - 30 years of accumulated knowledge and best practices

5. Monitoring & Profiling
   - JMX, JFR (Java Flight Recorder)
   - VisualVM, JProfiler, async-profiler
   - Micrometer + Prometheus + Grafana
```

---

## 3. Detailed Comparison of Each Language

### 3.1 Java — Balancing Stability and Evolution

```java
// Java 21+: Modern Java has evolved significantly

// Record (Java 14+) — Immutable data classes
public record User(String name, int age, List<String> tags) {
    // Compact constructor for validation
    public User {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        if (age < 0) {
            throw new IllegalArgumentException("age must be non-negative");
        }
        tags = List.copyOf(tags);  // Defensive copy (immutable list)
    }

    public boolean isAdult() {
        return age >= 18;
    }
}

// Sealed Classes (Java 17+) — Restricting inheritance
public sealed interface Shape
    permits Circle, Rectangle, Triangle {

    double area();
}

public record Circle(double radius) implements Shape {
    public double area() { return Math.PI * radius * radius; }
}

public record Rectangle(double width, double height) implements Shape {
    public double area() { return width * height; }
}

public record Triangle(double base, double height) implements Shape {
    public double area() { return base * height / 2; }
}

// Pattern Matching (Java 21+)
public String describe(Shape shape) {
    return switch (shape) {
        case Circle c when c.radius() > 10 -> "Large circle: r=" + c.radius();
        case Circle c -> "Circle: r=" + c.radius();
        case Rectangle r when r.width() == r.height() -> "Square: " + r.width();
        case Rectangle r -> "Rectangle: " + r.width() + "x" + r.height();
        case Triangle t -> "Triangle: base=" + t.base();
    };
}

// Virtual Threads (Java 21+) — Lightweight threads (Project Loom)
public List<String> fetchAllUrls(List<String> urls) throws Exception {
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
        List<Future<String>> futures = urls.stream()
            .map(url -> executor.submit(() -> fetchUrl(url)))
            .toList();

        List<String> results = new ArrayList<>();
        for (var future : futures) {
            results.add(future.get());
        }
        return results;
    }
}

// Structured Concurrency (Preview, Java 21+)
public record UserProfile(User user, List<Order> orders) {}

public UserProfile fetchUserProfile(long userId) throws Exception {
    try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
        Subtask<User> userTask = scope.fork(() -> findUser(userId));
        Subtask<List<Order>> ordersTask = scope.fork(() -> findOrders(userId));

        scope.join().throwIfFailed();

        return new UserProfile(userTask.get(), ordersTask.get());
    }
}

// Stream API usage
public Map<String, Double> averageAgeByDepartment(List<Employee> employees) {
    return employees.stream()
        .collect(Collectors.groupingBy(
            Employee::department,
            Collectors.averagingInt(Employee::age)
        ));
}

// Text Blocks (Java 15+)
public String generateJson(User user) {
    return """
        {
            "name": "%s",
            "age": %d,
            "isAdult": %b
        }
        """.formatted(user.name(), user.age(), user.isAdult());
}

// Proper use of Optional
public Optional<User> findActiveUser(String name) {
    return userRepository.findByName(name)
        .filter(User::isActive)
        .map(user -> enrichWithProfile(user));
}

// var (local variable type inference, Java 10+)
var users = List.of(
    new User("Alice", 30, List.of("admin")),
    new User("Bob", 25, List.of("user"))
);

var adultNames = users.stream()
    .filter(User::isAdult)
    .map(User::name)
    .sorted()
    .toList();

// String Templates (Preview, Java 22+)
// var message = STR."Hello, \{user.name()}! You are \{user.age()} years old.";
```

### 3.2 Kotlin — Pursuing Conciseness and Safety

```kotlin
// Kotlin: Developed by JetBrains, official Android language
// 100% compatible with Java while being significantly more concise and safe

// Data class (auto-generates equals, hashCode, toString, copy)
data class User(
    val name: String,
    val age: Int,
    val email: String? = null,  // Nullable (explicit)
    val tags: List<String> = emptyList()
) {
    fun isAdult(): Boolean = age >= 18

    // Partial modification with copy method
    fun withTag(tag: String): User = copy(tags = tags + tag)
}

// Null Safety — Kotlin's greatest weapon
fun processUser(name: String?) {
    // ?. Safe call operator
    val length = name?.length  // null if name is null

    // ?: Elvis operator
    val displayName = name ?: "Unknown"

    // ?.let executes only when non-null
    name?.let { n ->
        println("Name is: $n")
    }

    // !! Non-null assertion (minimize usage as it can cause NPE)
    // val forcedLength = name!!.length

    // Smart cast
    if (name != null) {
        // Here name is String type (non-null)
        println(name.length)
    }
}

// Sealed class + when — Exhaustive pattern matching
sealed interface Result<out T> {
    data class Success<T>(val value: T) : Result<T>
    data class Failure(val error: Throwable) : Result<Nothing>
    data object Loading : Result<Nothing>
}

fun <T> handleResult(result: Result<T>): String = when (result) {
    is Result.Success -> "Success: ${result.value}"
    is Result.Failure -> "Failure: ${result.error.message}"
    is Result.Loading -> "Loading..."
    // No else needed since it's sealed (compiler verifies exhaustiveness)
}

// Extension functions — Add methods to existing classes
fun String.toSlug(): String =
    this.lowercase()
        .replace(Regex("[^a-z0-9\\s-]"), "")
        .replace(Regex("[\\s]+"), "-")
        .trim('-')

// "Hello World!".toSlug() → "hello-world"

// Higher-order functions and lambdas
fun <T> List<T>.customFilter(predicate: (T) -> Boolean): List<T> {
    val result = mutableListOf<T>()
    for (item in this) {
        if (predicate(item)) {
            result.add(item)
        }
    }
    return result
}

// Scope functions (let, run, with, apply, also)
fun createUser(): User {
    return User(name = "Alice", age = 30).apply {
        println("Created user: $name")  // this is User
    }.also { user ->
        auditLog.log("User created: ${user.name}")  // it/user is User
    }
}

// Coroutines — Structured asynchronous processing
import kotlinx.coroutines.*

suspend fun fetchUserProfile(userId: Long): UserProfile = coroutineScope {
    val userDeferred = async { userService.findUser(userId) }
    val ordersDeferred = async { orderService.findOrders(userId) }

    UserProfile(
        user = userDeferred.await(),
        orders = ordersDeferred.await()
    )
}

// Flow — Reactive streams
import kotlinx.coroutines.flow.*

fun observeUsers(): Flow<List<User>> = flow {
    while (true) {
        val users = userRepository.findAll()
        emit(users)
        delay(5000)  // Update every 5 seconds
    }
}.distinctUntilChanged()
 .catch { e -> emit(emptyList()) }  // Error handling

// DSL Builder
class HtmlBuilder {
    private val elements = mutableListOf<String>()

    fun h1(text: String) { elements.add("<h1>$text</h1>") }
    fun p(text: String) { elements.add("<p>$text</p>") }
    fun ul(block: UlBuilder.() -> Unit) {
        val builder = UlBuilder()
        builder.block()
        elements.add(builder.build())
    }

    fun build(): String = elements.joinToString("\n")
}

class UlBuilder {
    private val items = mutableListOf<String>()
    fun li(text: String) { items.add("<li>$text</li>") }
    fun build(): String = "<ul>\n${items.joinToString("\n")}\n</ul>"
}

fun html(block: HtmlBuilder.() -> Unit): String {
    val builder = HtmlBuilder()
    builder.block()
    return builder.build()
}

// DSL usage example
val page = html {
    h1("User List")
    p("Active users:")
    ul {
        li("Alice")
        li("Bob")
        li("Carol")
    }
}

// Delegation
interface Repository<T> {
    fun findAll(): List<T>
    fun findById(id: Long): T?
}

class CachedRepository<T>(
    private val delegate: Repository<T>
) : Repository<T> by delegate {
    private val cache = mutableMapOf<Long, T>()

    override fun findById(id: Long): T? {
        return cache.getOrPut(id) { delegate.findById(id) ?: return null }
    }
}

// Value class (lightweight wrapper that gets inlined)
@JvmInline
value class UserId(val value: Long)

@JvmInline
value class OrderId(val value: Long)

fun getUser(id: UserId): User = TODO()  // Cannot confuse UserId and OrderId
fun getOrder(id: OrderId): Order = TODO()

// Kotlin Multiplatform (KMP)
// Switch platform-specific implementations with expect/actual
// expect fun platformName(): String
// actual fun platformName(): String = "JVM" / "JS" / "Native"
```

### 3.3 Scala — The Ultimate in Expressiveness and Type Safety

```scala
// Scala 3: Based on the Dotty compiler, significantly more concise

// case class (equivalent to Java's record, but more powerful)
case class User(name: String, age: Int, tags: List[String] = Nil):
  def isAdult: Boolean = age >= 18
  def withTag(tag: String): User = copy(tags = tags :+ tag)

// enum (Scala 3 algebraic data types)
enum Shape:
  case Circle(radius: Double)
  case Rectangle(width: Double, height: Double)
  case Triangle(base: Double, height: Double)

// Pattern matching (Scala's core feature)
def area(shape: Shape): Double = shape match
  case Shape.Circle(r) => Math.PI * r * r
  case Shape.Rectangle(w, h) => w * h
  case Shape.Triangle(b, h) => b * h / 2

// Pattern matching with guards
def describe(shape: Shape): String = shape match
  case Shape.Circle(r) if r > 10 => s"Large circle: r=$r"
  case Shape.Circle(r) => s"Circle: r=$r"
  case Shape.Rectangle(w, h) if w == h => s"Square: $w"
  case Shape.Rectangle(w, h) => s"Rectangle: ${w}x$h"
  case Shape.Triangle(b, h) => s"Triangle: base=$b"

// Advanced type system
// Union types (Scala 3)
type StringOrInt = String | Int

def process(value: StringOrInt): String = value match
  case s: String => s.toUpperCase
  case i: Int => i.toString

// Intersection types
trait Printable:
  def print(): Unit

trait Serializable:
  def serialize(): Array[Byte]

def sendToPrinter(item: Printable & Serializable): Unit =
  val bytes = item.serialize()
  item.print()

// Opaque Types (zero-cost type wrappers)
object Types:
  opaque type UserId = Long
  opaque type Email = String

  object UserId:
    def apply(value: Long): UserId = value
  extension (id: UserId)
    def value: Long = id

  object Email:
    def apply(value: String): Option[Email] =
      if value.contains("@") then Some(value) else None
  extension (email: Email)
    def value: String = email

// Given / Using (Scala 3 implicit parameters)
// Contextual Abstraction
trait JsonEncoder[A]:
  def encode(value: A): String

given JsonEncoder[User] with
  def encode(user: User): String =
    s"""{"name":"${user.name}","age":${user.age}}"""

given JsonEncoder[Int] with
  def encode(value: Int): String = value.toString

def toJsonA(using encoder: JsonEncoder[A]): String =
  encoder.encode(value)

// Usage example
val json = toJson(User("Alice", 30))
// → {"name":"Alice","age":30}

// For comprehensions (monadic composition)
def findUserOrders(userId: Long): Option[List[Order]] =
  for
    user <- userRepository.findById(userId)
    if user.isActive
    orders <- orderRepository.findByUser(user)
  yield orders

// Asynchronous processing with Future
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

def fetchUserProfile(userId: Long): Future[UserProfile] =
  val userFuture = Future(userService.findUser(userId))
  val ordersFuture = Future(orderService.findOrders(userId))

  for
    user <- userFuture
    orders <- ordersFuture
  yield UserProfile(user, orders)

// Effectful programming with ZIO (cutting edge of Scala ecosystem)
import zio.*

def fetchUser(id: Long): ZIO[UserService, AppError, User] =
  ZIO.serviceWithZIOUserService)

def program: ZIO[UserService & OrderService, AppError, UserProfile] =
  for
    user <- fetchUser(42)
    orders <- fetchOrders(user.id)
  yield UserProfile(user, orders)

// Collection operations (Scala's crown jewel)
val employees = List(
  Employee("Alice", 30, "Engineering", 80000),
  Employee("Bob", 25, "Marketing", 60000),
  Employee("Carol", 35, "Engineering", 90000),
  Employee("Dave", 28, "Marketing", 65000),
  Employee("Eve", 32, "Engineering", 85000)
)

// Average salary by department (top 3 only)
val result = employees
  .groupBy(_.department)
  .view
  .mapValues(emps => emps.map(_.salary).sum.toDouble / emps.size)
  .toList
  .sortBy(-_._2)
  .take(3)
// → List(("Engineering", 85000.0), ("Marketing", 62500.0))

// Type-level programming (advanced)
// Verify state at compile time with Phantom Types
sealed trait DoorState
sealed trait Open extends DoorState
sealed trait Closed extends DoorState

class Door[S <: DoorState]:
  def open(using ev: S =:= Closed): Door[Open] = Door[Open]
  def close(using ev: S =:= Open): Door[Closed] = Door[Closed]

val door = Door[Closed]
val openDoor = door.open       // OK
// val invalid = openDoor.open  // Compile error! Already open

// Extension Methods (Scala 3)
extension (s: String)
  def words: List[String] = s.split("\\s+").toList
  def wordCount: Int = words.length

"Hello World Scala".wordCount  // → 3

// Match Types (Scala 3) — Type-level pattern matching
type Elem[X] = X match
  case String => Char
  case Array[t] => t
  case Iterable[t] => t

// Elem[String] =:= Char
// Elem[Array[Int]] =:= Int
// Elem[List[Double]] =:= Double
```

### 3.4 Clojure — Data-Oriented Programming

```clojure
;; Clojure: A modern Lisp on the JVM
;; Design philosophy: Simple, immutable, data-centric

;; Everything is an expression
;; S-expression: (function arg1 arg2 ...)
(println "Hello, World!")

;; Immutable data structures (persistent data structures)
(def user {:name "Alice"
           :age 30
           :tags ["admin" "developer"]})

;; assoc: Add/update a key (original data is unchanged)
(def updated-user (assoc user :email "alice@example.com"))
;; user remains unchanged

;; update: Apply a function to update
(def older-user (update user :age inc))
;; → {:name "Alice", :age 31, :tags ["admin" "developer"]}

;; Updating nested data
(def company {:name "Acme"
              :address {:city "Tokyo"
                        :zip "100-0001"}})

(def updated (assoc-in company [:address :city] "Osaka"))
(def with-floor (update-in company [:address] assoc :floor 5))

;; Threading macros — Data transformation pipelines
;; -> (thread-first): Pass as the first argument
(-> "Hello, World!"
    .toUpperCase
    (.replace "," "")
    (.split " ")
    first)
;; → "HELLO"

;; ->> (thread-last): Pass as the last argument
(->> (range 1 101)
     (filter odd?)
     (map #(* % %))
     (reduce +))
;; → 1^2 + 3^2 + 5^2 + ... + 99^2

;; Higher-order functions and transducers
(def users [{:name "Alice" :age 30 :active true}
            {:name "Bob"   :age 17 :active true}
            {:name "Carol" :age 25 :active false}
            {:name "Dave"  :age 35 :active true}])

;; Traditional approach (intermediate collections are created)
(->> users
     (filter :active)
     (filter #(>= (:age %) 18))
     (map :name)
     (sort))
;; → ("Alice" "Dave")

;; Transducers (no intermediate collections, efficient)
(def xf (comp
         (filter :active)
         (filter #(>= (:age %) 18))
         (map :name)))

(into [] xf users)
;; → ["Alice" "Dave"]

;; Multimethods — Flexible polymorphism
(defmulti area :shape)

(defmethod area :circle [{:keys [radius]}]
  (* Math/PI radius radius))

(defmethod area :rectangle [{:keys [width height]}]
  (* width height))

(defmethod area :triangle [{:keys [base height]}]
  (/ (* base height) 2))

(area {:shape :circle :radius 5})        ;; → 78.54
(area {:shape :rectangle :width 4 :height 5})  ;; → 20

;; Protocols — A concept close to Java interfaces
(defprotocol Summarizable
  (summarize [this]))

(defrecord User [name age]
  Summarizable
  (summarize [this]
    (str name " (age " age ")")))

(defrecord Article [title author]
  Summarizable
  (summarize [this]
    (str "\"" title "\" by " author)))

(summarize (->User "Alice" 30))              ;; → "Alice (age 30)"
(summarize (->Article "Intro to FP" "Bob"))  ;; → "\"Intro to FP\" by Bob"

;; Spec — Data validation and generation
(require '[clojure.spec.alpha :as s])

(s/def ::name (s/and string? #(> (count %) 0)))
(s/def ::age (s/and int? #(>= % 0) #(<= % 150)))
(s/def ::email (s/and string? #(re-matches #".+@.+\..+" %)))
(s/def ::user (s/keys :req-un [::name ::age]
                      :opt-un [::email]))

(s/valid? ::user {:name "Alice" :age 30})             ;; → true
(s/valid? ::user {:name "" :age 30})                   ;; → false
(s/explain ::user {:name "" :age 30})                  ;; Shows error reason

;; Atom — Safe shared state management
(def counter (atom 0))
(swap! counter inc)     ;; Atomically +1
(swap! counter + 10)    ;; Atomically +10
@counter                ;; → 11 (dereference)

;; Agent — Asynchronous state updates
(def log-agent (agent []))
(send log-agent conj "Event 1")
(send log-agent conj "Event 2")
;; Processed sequentially on a separate thread

;; core.async — CSP-style concurrency
(require '[clojure.core.async :as async])

(let [ch (async/chan 10)]
  ;; Sender side
  (async/go
    (doseq [i (range 5)]
      (async/>! ch i))
    (async/close! ch))

  ;; Receiver side
  (async/go-loop []
    (when-let [v (async/<! ch)]
      (println "Received:" v)
      (recur))))

;; REPL-Driven Development — The essence of Clojure
;; 1. Explore data in the REPL
;; 2. Define and test functions
;; 3. Build data transformation pipelines incrementally
;; 4. Organize into namespaces
;; → No "code → compile → run" cycle needed
```

---

## 4. Java's Modern Evolution (Java 17 - 25)

```
Java 17 (LTS, 2021):
  - Sealed Classes (restricting inheritance)
  - Pattern Matching for instanceof
  - Text Blocks
  - Records

Java 21 (LTS, 2023):
  - Virtual Threads (Project Loom) — Lightweight threads
  - Pattern Matching for switch
  - Record Patterns
  - Sequenced Collections
  - String Templates (Preview)

Java 22-25 (2024-2025):
  - Structured Concurrency
  - Scoped Values (improvement over ThreadLocal)
  - Stream Gatherers (custom stream operations)
  - Foreign Function & Memory API (JNI replacement)
  - Vector API (SIMD operations)
  - Value Types (Project Valhalla, Preview)

Key Trends:
  → Java is evolving rapidly with 6-month release cycles
  → Tendency for Java to adopt good features from Kotlin
  → Records, Sealed Classes, Pattern Matching are modernizing Java
  → Virtual Threads are changing the concurrency paradigm
```

---

## 5. Framework Comparison

### 5.1 Web Frameworks

```java
// Java: Spring Boot — The king of enterprise
@RestController
@RequestMapping("/api/users")
public class UserController {

    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping
    public List<UserDto> listUsers(@RequestParam(defaultValue = "0") int page) {
        return userService.findAll(PageRequest.of(page, 20))
            .stream()
            .map(UserDto::from)
            .toList();
    }

    @GetMapping("/{id}")
    public ResponseEntity<UserDto> getUser(@PathVariable Long id) {
        return userService.findById(id)
            .map(UserDto::from)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public UserDto createUser(@Valid @RequestBody CreateUserRequest request) {
        var user = userService.create(request);
        return UserDto.from(user);
    }
}

// Spring Data JPA — Auto-implemented repositories
public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByEmail(String email);
    List<User> findByAgeGreaterThan(int age);

    @Query("SELECT u FROM User u WHERE u.department = :dept AND u.active = true")
    List<User> findActiveByDepartment(@Param("dept") String department);
}
```

```kotlin
// Kotlin: Ktor — Lightweight async web framework
fun Application.module() {
    install(ContentNegotiation) {
        json(Json {
            prettyPrint = true
            ignoreUnknownKeys = true
        })
    }

    install(StatusPages) {
        exception<NotFoundException> { call, cause ->
            call.respond(HttpStatusCode.NotFound, ErrorResponse(cause.message ?: "Not found"))
        }
    }

    routing {
        route("/api/users") {
            get {
                val users = userService.findAll()
                call.respond(users.map { it.toDto() })
            }

            get("/{id}") {
                val id = call.parameters["id"]?.toLongOrNull()
                    ?: throw BadRequestException("Invalid ID")
                val user = userService.findById(id)
                    ?: throw NotFoundException("User not found")
                call.respond(user.toDto())
            }

            post {
                val request = call.receive<CreateUserRequest>()
                val user = userService.create(request)
                call.respond(HttpStatusCode.Created, user.toDto())
            }
        }
    }
}

// Exposed — Kotlin's type-safe SQL DSL
object Users : Table("users") {
    val id = long("id").autoIncrement()
    val name = varchar("name", 255)
    val age = integer("age")
    val email = varchar("email", 255).nullable()
    override val primaryKey = PrimaryKey(id)
}

fun findActiveAdults(): List<UserRow> = transaction {
    Users.select { (Users.age greaterEq 18) }
        .orderBy(Users.name)
        .map { row ->
            UserRow(
                id = row[Users.id],
                name = row[Users.name],
                age = row[Users.age],
                email = row[Users.email]
            )
        }
}
```

```scala
// Scala: ZIO HTTP + Tapir — Type-safe API definition
import sttp.tapir.*
import sttp.tapir.json.zio.*
import zio.*
import zio.json.*

// Type-safe API endpoint definition
val getUserEndpoint = endpoint
  .get
  .in("api" / "users" / pathLong)
  .out(jsonBody[UserDto])
  .errorOut(statusCode(StatusCode.NotFound))

val listUsersEndpoint = endpoint
  .get
  .in("api" / "users")
  .in(queryInt.default(0))
  .out(jsonBody[List[UserDto]])

// Endpoint implementation
val getUserRoute = getUserEndpoint.zServerLogic { id =>
  userService.findById(id)
    .map(_.toDto)
    .mapError(_ => StatusCode.NotFound)
}

// Auto-generated OpenAPI documentation
val docs = OpenAPIDocsInterpreter()
  .toOpenAPI(List(getUserEndpoint, listUsersEndpoint), "User API", "1.0.0")
```

```clojure
;; Clojure: Ring + Compojure — Function-based web framework
(ns myapp.handler
  (:require [compojure.core :refer [defroutes GET POST]]
            [compojure.route :as route]
            [ring.middleware.json :refer [wrap-json-response wrap-json-body]]
            [ring.util.response :as response]))

(defroutes app-routes
  (GET "/api/users" []
    (response/response (user-service/find-all)))

  (GET "/api/users/:id" [id]
    (if-let [user (user-service/find-by-id (parse-long id))]
      (response/response user)
      (response/not-found {:error "User not found"})))

  (POST "/api/users" {body :body}
    (let [user (user-service/create body)]
      (-> (response/response user)
          (response/status 201))))

  (route/not-found {:error "Not found"}))

(def app
  (-> app-routes
      wrap-json-body
      wrap-json-response))
```

### 5.2 Data Processing Frameworks

```scala
// Scala: Apache Spark — The standard for large-scale data processing
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val spark = SparkSession.builder()
  .appName("UserAnalytics")
  .master("local[*]")
  .getOrCreate()

import spark.implicits._

// CSV reading
val users = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("users.csv")

// Data transformation and aggregation
val result = users
  .filter($"age" >= 18)
  .groupBy($"department")
  .agg(
    count("*").alias("count"),
    avg("salary").alias("avg_salary"),
    max("salary").alias("max_salary")
  )
  .orderBy(desc("avg_salary"))

result.show()
// +------------+-----+----------+----------+
// | department |count|avg_salary|max_salary|
// +------------+-----+----------+----------+
// |Engineering |  150|    95000 |   180000 |
// |Marketing   |   80|    72000 |   130000 |
// +------------+-----+----------+----------+

// Dataset API (type-safe)
case class UserEvent(userId: Long, action: String, timestamp: Long)

val events = spark.read.parquet("events.parquet").as[UserEvent]

val activeUsers = events
  .filter(_.action == "login")
  .groupByKey(_.userId)
  .count()
  .filter(_._2 >= 5)  // Users who logged in 5 or more times
```

---

## 6. Test Strategy Comparison

```java
// Java: JUnit 5 + AssertJ + Mockito
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.*;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

@DisplayName("UserService Tests")
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private UserService userService;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    @DisplayName("Should return active users")
    void shouldReturnActiveUsers() {
        var users = List.of(
            new User("Alice", 30, true),
            new User("Bob", 25, false),
            new User("Carol", 35, true)
        );
        when(userRepository.findAll()).thenReturn(users);

        var result = userService.findActiveUsers();

        assertThat(result)
            .hasSize(2)
            .extracting(User::name)
            .containsExactly("Alice", "Carol");
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 17})
    @DisplayName("Minors should not be adults")
    void shouldNotBeAdult(int age) {
        var user = new User("Test", age, true);
        assertThat(user.isAdult()).isFalse();
    }
}
```

```kotlin
// Kotlin: Kotest — Diverse test styles
import io.kotest.core.spec.style.DescribeSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.collections.shouldContainExactly
import io.mockk.every
import io.mockk.mockk

class UserServiceTest : DescribeSpec({
    val userRepository = mockk<UserRepository>()
    val userService = UserService(userRepository)

    describe("findActiveUsers") {
        it("should return only active users") {
            val users = listOf(
                User("Alice", 30, active = true),
                User("Bob", 25, active = false),
                User("Carol", 35, active = true)
            )
            every { userRepository.findAll() } returns users

            val result = userService.findActiveUsers()

            result shouldHaveSize 2
            result.map { it.name } shouldContainExactly listOf("Alice", "Carol")
        }

        context("when no users exist") {
            it("should return an empty list") {
                every { userRepository.findAll() } returns emptyList()

                val result = userService.findActiveUsers()

                result shouldHaveSize 0
            }
        }
    }

    describe("isAdult") {
        withData(
            nameFn = { "age=${it.first} → isAdult=${it.second}" },
            (17 to false),
            (18 to true),
            (0 to false),
            (100 to true)
        ) { (age, expected) ->
            User("Test", age).isAdult() shouldBe expected
        }
    }
})
```

```clojure
;; Clojure: clojure.test + test.check (property-based testing)
(ns myapp.user-test
  (:require [clojure.test :refer [deftest testing is are]]
            [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [myapp.user :as user]))

(deftest test-find-active-users
  (testing "should return only active users"
    (let [users [{:name "Alice" :age 30 :active true}
                 {:name "Bob"   :age 25 :active false}
                 {:name "Carol" :age 35 :active true}]
          result (user/find-active users)]
      (is (= 2 (count result)))
      (is (= ["Alice" "Carol"] (map :name result)))))

  (testing "when users are empty"
    (is (empty? (user/find-active [])))))

;; Property-based testing
(def user-gen
  (gen/hash-map
    :name gen/string-alphanumeric
    :age (gen/choose 0 150)
    :active gen/boolean))

(def active-users-subset-property
  (prop/for-all [users (gen/vector user-gen)]
    (let [active (user/find-active users)]
      ;; Active users are a subset of the original users
      (every? #(some #{%} users) active))))

(tc/quick-check 1000 active-users-subset-property)
```

---

## 7. GraalVM and Native Images

```
GraalVM: A high-performance JVM developed by Oracle
  → JIT compiler improvements (Graal compiler)
  → Native Image: AOT compilation for faster startup
  → Polyglot execution (Java, JavaScript, Python, Ruby, R, LLVM)

Native Image Benefits:
  - Startup time: Seconds → Tens of milliseconds
  - Memory usage: Hundreds of MB → Tens of MB
  - Package size: No JRE needed (standalone executable binary)

Constraints:
  - Reflection limitations (configuration required)
  - Dynamic class loading restrictions
  - Long compilation times

Supported Frameworks:
  - Quarkus: Native Image first
  - Micronaut: Compile-time DI
  - Spring Native / Spring Boot 3.x
  - Helidon: Oracle's lightweight framework
```

```java
// Quarkus: Optimized for GraalVM Native Image
@Path("/api/users")
@Produces(MediaType.APPLICATION_JSON)
public class UserResource {

    @Inject
    UserService userService;

    @GET
    public List<UserDto> list() {
        return userService.findAll().stream()
            .map(UserDto::from)
            .toList();
    }

    @GET
    @Path("/{id}")
    public Response get(@PathParam("id") Long id) {
        return userService.findById(id)
            .map(u -> Response.ok(UserDto.from(u)).build())
            .orElse(Response.status(Response.Status.NOT_FOUND).build());
    }
}

// Build command:
// ./mvnw package -Pnative
// → Startup time: ~0.02s (normal JVM is ~1s)
// → Memory: ~30MB (normal JVM is ~200MB)
```

---

## 8. Detailed Selection Guidelines

```
Q1: What is the team size and experience?
├── Large team (20+ people), rich Java experience → Java (Spring Boot)
│   Reason: Easy to recruit, abundant documentation and knowledge
├── Medium team (5-20 people), modern orientation → Kotlin (Ktor/Spring)
│   Reason: Improved productivity while maintaining Java compatibility
├── Small team (1-5 people), FP-oriented → Scala or Clojure
│   Reason: Small teams can leverage high expressiveness
└── Including Android development → Kotlin (the only choice)

Q2: What are the performance requirements?
├── Low-latency required → Java (Virtual Threads) or Kotlin (Coroutines)
├── Large-scale data processing → Scala (Spark)
├── Fast startup important → Java/Kotlin + GraalVM Native Image
└── Throughput-focused → Java (Reactor) or Kotlin (Flow)

Q3: What type of project is it?
├── Enterprise CRUD → Java (Spring Boot) or Kotlin (Spring)
├── Microservices → Kotlin (Ktor) or Java (Quarkus)
├── Data pipeline → Scala (Spark) or Java (Flink)
├── Event-driven system → Scala (Akka/ZIO) or Kotlin (Coroutines)
├── REPL-driven exploratory development → Clojure
└── Android app → Kotlin (Google official)

Q4: What about long-term maintainability?
├── 10+ years of operation → Java
│   Reason: Backward compatibility guarantees, LTS releases
├── 5-10 years → Kotlin or Java
│   Reason: JetBrains + Google backing
├── Technical challenge → Scala
│   Reason: Highest expressiveness but high team education cost
└── Startup → Kotlin or Clojure
    Reason: High productivity with small teams
```

---

## 9. Java ⇄ Kotlin Migration Guide

### 9.1 Incremental Migration Steps

```
1. Create new files in Kotlin (can coexist in a Java project)
2. Convert test code to Kotlin (low risk)
3. Convert utility classes to Kotlin
4. Implement all new features in Kotlin
5. Gradually convert existing Java code (leverage IntelliJ's auto-conversion)

Notes:
  - Just adding the Kotlin plugin to build.gradle enables coexistence
  - Use @JvmStatic, @JvmOverloads when calling Kotlin from Java
  - Be aware of platform types when calling Java from Kotlin
```

### 9.2 Key Conversion Patterns

```java
// Java: POJO (lots of boilerplate)
public class User {
    private final String name;
    private final int age;
    private final String email;

    public User(String name, int age, String email) {
        this.name = name;
        this.age = age;
        this.email = email;
    }

    public String getName() { return name; }
    public int getAge() { return age; }
    public String getEmail() { return email; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        User user = (User) o;
        return age == user.age &&
            Objects.equals(name, user.name) &&
            Objects.equals(email, user.email);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, age, email);
    }

    @Override
    public String toString() {
        return "User{name='%s', age=%d, email='%s'}".formatted(name, age, email);
    }
}
```

```kotlin
// Kotlin: data class (equivalent functionality in one line)
data class User(val name: String, val age: Int, val email: String?)
// equals, hashCode, toString, copy, componentN are auto-generated
```

```java
// Java: Chain of null checks
public String getUserCity(Order order) {
    if (order != null) {
        User user = order.getUser();
        if (user != null) {
            Address address = user.getAddress();
            if (address != null) {
                return address.getCity();
            }
        }
    }
    return "Unknown";
}
```

```kotlin
// Kotlin: Safe call operator
fun getUserCity(order: Order?): String =
    order?.user?.address?.city ?: "Unknown"
```

```java
// Java: Limitations of Stream API type inference
List<String> names = users.stream()
    .filter(u -> u.getAge() >= 18)
    .map(User::getName)
    .sorted()
    .collect(Collectors.toList());
```

```kotlin
// Kotlin: More natural collection operations
val names = users
    .filter { it.age >= 18 }
    .map { it.name }
    .sorted()
// No toList() needed (already a List)
```

---

## 10. Practical Project Structure

### 10.1 Spring Boot + Kotlin Project

```
myapp/
├── build.gradle.kts         # Gradle Kotlin DSL
├── settings.gradle.kts
├── src/
│   ├── main/
│   │   ├── kotlin/
│   │   │   └── com/example/myapp/
│   │   │       ├── MyAppApplication.kt
│   │   │       ├── config/
│   │   │       │   ├── SecurityConfig.kt
│   │   │       │   └── WebConfig.kt
│   │   │       ├── controller/
│   │   │       │   └── UserController.kt
│   │   │       ├── service/
│   │   │       │   └── UserService.kt
│   │   │       ├── repository/
│   │   │       │   └── UserRepository.kt
│   │   │       ├── model/
│   │   │       │   ├── entity/
│   │   │       │   │   └── User.kt
│   │   │       │   └── dto/
│   │   │       │       ├── UserDto.kt
│   │   │       │       └── CreateUserRequest.kt
│   │   │       └── exception/
│   │   │           └── GlobalExceptionHandler.kt
│   │   └── resources/
│   │       ├── application.yml
│   │       └── db/migration/
│   │           └── V1__create_users.sql
│   └── test/
│       └── kotlin/
│           └── com/example/myapp/
│               ├── controller/
│               │   └── UserControllerTest.kt
│               └── service/
│                   └── UserServiceTest.kt
├── Dockerfile
└── docker-compose.yml
```


---

## Hands-On Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement appropriate error handling
- Write test code as well

```python
# Exercise 1: Basic Implementation Template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Get processing results"""
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
        assert False, "Should have raised an exception"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced Patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

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
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Delete by key"""
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
# Exercise 3: Performance Optimization
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

    print(f"Inefficient version: {slow_time:.4f}s")
    print(f"Efficient version:   {fast_time:.6f}s")
    print(f"Speedup factor: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be mindful of algorithmic complexity
- Choose appropriate data structures
- Measure effectiveness with benchmarks
---


## FAQ

### Q1: What is the most important point for learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this applied in professional practice?

The knowledge in this topic is frequently used in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Language | Philosophy | Best Use Case | Status in 2025 |
|------|------|----------|-------------|
| Java | Stability & Enterprise | Large-scale business systems | Major evolution with Java 21. Virtual Threads are revolutionary |
| Kotlin | Concise, Safe, Practical | Android, Server-side | Expanding to iOS/Web with KMP. Momentum continues |
| Scala | Expressiveness, Type Safety, FP | Data processing, Distributed systems | Fresh start with Scala 3. Spark ecosystem remains strong |
| Clojure | Simple, Immutable, REPL | Data processing, Web | Niche but with strong following. Pioneer of REPL-driven development |

### Quick Selection Guide

```
When in doubt, Java — The safest choice, easiest to find talent
For modern code, Kotlin — Java's strengths + modern features
For data processing, Scala — The power of the Spark ecosystem
For philosophy, Clojure — The ultimate in data-oriented programming
For Android, Kotlin — Google official, the only choice
```

---

## Recommended Next Guides

---

## References
1. Odersky, M. "Programming in Scala." 5th Ed, Artima, 2023.
2. Jemerov, D. & Isakova, S. "Kotlin in Action." 2nd Ed, Manning, 2024.
3. Bloch, J. "Effective Java." 3rd Ed, Addison-Wesley, 2018.
4. Hickey, R. "Clojure - Rationale." clojure.org.
5. Horstmann, C. "Core Java." Vol 1-2, 12th Ed, Pearson, 2022.
6. "State of Developer Ecosystem 2024." JetBrains.
7. "Spring Framework Documentation." spring.io.
8. "Kotlin Multiplatform." kotlinlang.org.
9. "GraalVM Reference Manual." graalvm.org.
10. Emerick, C., Carper, B. & Grand, C. "Clojure Programming." O'Reilly, 2012.
