# JVM 言語比較（Java, Kotlin, Scala, Clojure）

> JVM（Java Virtual Machine）上で動く言語群。Javaの巨大なエコシステムを共有しつつ、それぞれが異なる哲学で進化。

## この章で学ぶこと

- [ ] JVM 言語のエコシステムと互換性を理解する
- [ ] 各言語の特徴と使い分けを把握する
- [ ] JVM のアーキテクチャとパフォーマンス特性を理解する
- [ ] 各言語の型システムの違いを比較できる
- [ ] 実務プロジェクトでの言語選択の判断基準を持つ
- [ ] GraalVM やバーチャルスレッドなど最新動向を把握する

---

## 1. 比較表

```
┌──────────────┬──────────┬──────────┬──────────┬──────────┐
│              │ Java     │ Kotlin   │ Scala    │ Clojure  │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 登場年        │ 1995     │ 2011     │ 2003     │ 2007     │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 設計者        │ Gosling  │ JetBrains│ Odersky  │ Hickey   │
│              │ (Sun)    │          │          │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ パラダイム    │ OOP      │ OOP+FP  │ OOP+FP  │ FP       │
│              │          │ マルチ   │ マルチ   │ Lisp系   │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 型付け        │ 静的     │ 静的     │ 静的     │ 動的     │
│              │ nominal  │ 型推論強 │ 型推論最強│          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Null安全      │ なし     │ あり     │ Option  │ nil      │
│              │ (NPE)   │ (言語組込)│          │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 主な用途      │ 企業     │ Android │ データ   │ Web      │
│              │ バックエンド│ サーバー │ 分散処理 │ データ   │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 主要FW       │ Spring   │ Ktor    │ Akka     │ Ring     │
│              │ Quarkus  │ Spring  │ Play     │ Luminus  │
│              │ Micronaut│ Exposed │ ZIO      │ Pedestal │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ ビルドツール  │ Gradle   │ Gradle  │ sbt      │ Leiningen│
│              │ Maven    │ Maven   │ Mill     │ deps.edn │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ テストFW     │ JUnit    │ Kotest  │ ScalaTest│ clojure  │
│              │ Mockito  │ MockK   │ Specs2   │ .test    │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 学習コスト    │ 中程度   │ 低い    │ 高い     │ 高い     │
│              │          │ (Java経験│          │ (Lisp系) │
│              │          │  あれば) │          │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 冗長性        │ やや高い │ 低い    │ 最も低い │ 低い     │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Java相互運用  │ 100%    │ 100%    │ 95%     │ 良好     │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 求人数        │ 最も多い │ 急増中   │ 減少傾向 │ ニッチ   │
└──────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 2. JVM アーキテクチャの理解

### 2.1 JVM の仕組み

```
┌────────────────────────────────────────────────────────┐
│                   JVM Architecture                     │
├────────────────────────────────────────────────────────┤
│  .java / .kt / .scala / .clj                          │
│       ↓ コンパイル                                     │
│  .class ファイル（JVMバイトコード）                      │
│       ↓ クラスローディング                              │
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

### 2.2 JVM の利点

```
1. Write Once, Run Anywhere（WORA）
   - 同一のバイトコードが全プラットフォームで動作
   - Linux, macOS, Windows, Docker コンテナ

2. JIT コンパイラによる最適化
   - C1: クライアント向け、速いコンパイル
   - C2: サーバー向け、深い最適化
   - 実行パターンに基づく投機的最適化
   - インライン化、ループアンローリング、エスケープ分析

3. GC（ガベージコレクタ）の進化
   - G1 GC: デフォルト（Java 9+）、バランス型
   - ZGC: 低遅延（<1ms pause）、大ヒープ向け
   - Shenandoah: 低遅延、Red Hat 開発
   - Epsilon: No-op GC（ベンチマーク用）

4. 豊富なエコシステム
   - Maven Central: 50万+ ライブラリ
   - Spring, Hibernate, Apache プロジェクト群
   - 30年の蓄積された知見とベストプラクティス

5. 監視・プロファイリング
   - JMX, JFR (Java Flight Recorder)
   - VisualVM, JProfiler, async-profiler
   - Micrometer + Prometheus + Grafana
```

---

## 3. 各言語の詳細比較

### 3.1 Java — 安定と進化のバランス

```java
// Java 21+: モダン Java は大きく進化した

// Record（Java 14+）— 不変データクラス
public record User(String name, int age, List<String> tags) {
    // コンパクトコンストラクタでバリデーション
    public User {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        if (age < 0) {
            throw new IllegalArgumentException("age must be non-negative");
        }
        tags = List.copyOf(tags);  // 防御的コピー（不変リスト）
    }

    public boolean isAdult() {
        return age >= 18;
    }
}

// Sealed Classes（Java 17+）— 継承の制限
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

// パターンマッチ（Java 21+）
public String describe(Shape shape) {
    return switch (shape) {
        case Circle c when c.radius() > 10 -> "大きな円: r=" + c.radius();
        case Circle c -> "円: r=" + c.radius();
        case Rectangle r when r.width() == r.height() -> "正方形: " + r.width();
        case Rectangle r -> "長方形: " + r.width() + "x" + r.height();
        case Triangle t -> "三角形: base=" + t.base();
    };
}

// Virtual Threads（Java 21+）— 軽量スレッド（Project Loom）
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

// Structured Concurrency（Preview, Java 21+）
public record UserProfile(User user, List<Order> orders) {}

public UserProfile fetchUserProfile(long userId) throws Exception {
    try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
        Subtask<User> userTask = scope.fork(() -> findUser(userId));
        Subtask<List<Order>> ordersTask = scope.fork(() -> findOrders(userId));

        scope.join().throwIfFailed();

        return new UserProfile(userTask.get(), ordersTask.get());
    }
}

// Stream API の活用
public Map<String, Double> averageAgeByDepartment(List<Employee> employees) {
    return employees.stream()
        .collect(Collectors.groupingBy(
            Employee::department,
            Collectors.averagingInt(Employee::age)
        ));
}

// テキストブロック（Java 15+）
public String generateJson(User user) {
    return """
        {
            "name": "%s",
            "age": %d,
            "isAdult": %b
        }
        """.formatted(user.name(), user.age(), user.isAdult());
}

// Optional の適切な使い方
public Optional<User> findActiveUser(String name) {
    return userRepository.findByName(name)
        .filter(User::isActive)
        .map(user -> enrichWithProfile(user));
}

// var（ローカル変数型推論、Java 10+）
var users = List.of(
    new User("Alice", 30, List.of("admin")),
    new User("Bob", 25, List.of("user"))
);

var adultNames = users.stream()
    .filter(User::isAdult)
    .map(User::name)
    .sorted()
    .toList();

// String Templates（Preview, Java 22+）
// var message = STR."Hello, \{user.name()}! You are \{user.age()} years old.";
```

### 3.2 Kotlin — 簡潔さと安全性の追求

```kotlin
// Kotlin: JetBrains が開発、Android 公式言語
// Java との 100% 互換を保ちつつ大幅に簡潔・安全

// データクラス（equals, hashCode, toString, copy を自動生成）
data class User(
    val name: String,
    val age: Int,
    val email: String? = null,  // Nullable（明示的）
    val tags: List<String> = emptyList()
) {
    fun isAdult(): Boolean = age >= 18

    // copy メソッドで部分的な変更
    fun withTag(tag: String): User = copy(tags = tags + tag)
}

// Null安全 — Kotlin の最大の武器
fun processUser(name: String?) {
    // ?. セーフコール演算子
    val length = name?.length  // nameがnullならnull

    // ?: エルビス演算子
    val displayName = name ?: "Unknown"

    // ?.let でnullでない場合のみ実行
    name?.let { n ->
        println("Name is: $n")
    }

    // !! 非null断言（NPE の可能性があるので最小限に）
    // val forcedLength = name!!.length

    // スマートキャスト
    if (name != null) {
        // ここでは name は String 型（non-null）
        println(name.length)
    }
}

// Sealed class + when — 網羅的パターンマッチ
sealed interface Result<out T> {
    data class Success<T>(val value: T) : Result<T>
    data class Failure(val error: Throwable) : Result<Nothing>
    data object Loading : Result<Nothing>
}

fun <T> handleResult(result: Result<T>): String = when (result) {
    is Result.Success -> "成功: ${result.value}"
    is Result.Failure -> "失敗: ${result.error.message}"
    is Result.Loading -> "読み込み中..."
    // sealed なので else が不要（コンパイラが網羅性を検証）
}

// 拡張関数 — 既存クラスにメソッドを追加
fun String.toSlug(): String =
    this.lowercase()
        .replace(Regex("[^a-z0-9\\s-]"), "")
        .replace(Regex("[\\s]+"), "-")
        .trim('-')

// "Hello World!".toSlug() → "hello-world"

// 高階関数とラムダ
fun <T> List<T>.customFilter(predicate: (T) -> Boolean): List<T> {
    val result = mutableListOf<T>()
    for (item in this) {
        if (predicate(item)) {
            result.add(item)
        }
    }
    return result
}

// スコープ関数（let, run, with, apply, also）
fun createUser(): User {
    return User(name = "Alice", age = 30).apply {
        println("Created user: $name")  // this は User
    }.also { user ->
        auditLog.log("User created: ${user.name}")  // it/user は User
    }
}

// Coroutines — 構造化された非同期処理
import kotlinx.coroutines.*

suspend fun fetchUserProfile(userId: Long): UserProfile = coroutineScope {
    val userDeferred = async { userService.findUser(userId) }
    val ordersDeferred = async { orderService.findOrders(userId) }

    UserProfile(
        user = userDeferred.await(),
        orders = ordersDeferred.await()
    )
}

// Flow — リアクティブストリーム
import kotlinx.coroutines.flow.*

fun observeUsers(): Flow<List<User>> = flow {
    while (true) {
        val users = userRepository.findAll()
        emit(users)
        delay(5000)  // 5秒ごとに更新
    }
}.distinctUntilChanged()
 .catch { e -> emit(emptyList()) }  // エラーハンドリング

// DSL ビルダー
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

// DSL の使用例
val page = html {
    h1("ユーザー一覧")
    p("アクティブなユーザー:")
    ul {
        li("Alice")
        li("Bob")
        li("Carol")
    }
}

// Delegation（委譲）
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

// Value class（インライン化される軽量ラッパー）
@JvmInline
value class UserId(val value: Long)

@JvmInline
value class OrderId(val value: Long)

fun getUser(id: UserId): User = TODO()  // UserId と OrderId を混同しない
fun getOrder(id: OrderId): Order = TODO()

// Kotlin Multiplatform（KMP）
// expect/actual で各プラットフォーム固有の実装を切り替え
// expect fun platformName(): String
// actual fun platformName(): String = "JVM" / "JS" / "Native"
```

### 3.3 Scala — 表現力と型安全性の極致

```scala
// Scala 3: Dotty コンパイラベース、大幅に簡潔化

// case class（Javaのrecordに相当、より強力）
case class User(name: String, age: Int, tags: List[String] = Nil):
  def isAdult: Boolean = age >= 18
  def withTag(tag: String): User = copy(tags = tags :+ tag)

// enum（Scala 3の代数的データ型）
enum Shape:
  case Circle(radius: Double)
  case Rectangle(width: Double, height: Double)
  case Triangle(base: Double, height: Double)

// パターンマッチ（Scalaの核心機能）
def area(shape: Shape): Double = shape match
  case Shape.Circle(r) => Math.PI * r * r
  case Shape.Rectangle(w, h) => w * h
  case Shape.Triangle(b, h) => b * h / 2

// ガード付きパターンマッチ
def describe(shape: Shape): String = shape match
  case Shape.Circle(r) if r > 10 => s"大きな円: r=$r"
  case Shape.Circle(r) => s"円: r=$r"
  case Shape.Rectangle(w, h) if w == h => s"正方形: $w"
  case Shape.Rectangle(w, h) => s"長方形: ${w}x$h"
  case Shape.Triangle(b, h) => s"三角形: base=$b"

// 高度な型システム
// Union型（Scala 3）
type StringOrInt = String | Int

def process(value: StringOrInt): String = value match
  case s: String => s.toUpperCase
  case i: Int => i.toString

// Intersection型
trait Printable:
  def print(): Unit

trait Serializable:
  def serialize(): Array[Byte]

def sendToPrinter(item: Printable & Serializable): Unit =
  val bytes = item.serialize()
  item.print()

// Opaque Types（ゼロコスト型ラッパー）
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

// Given / Using（Scala 3の暗黙パラメータ）
// Contextual Abstraction
trait JsonEncoder[A]:
  def encode(value: A): String

given JsonEncoder[User] with
  def encode(user: User): String =
    s"""{"name":"${user.name}","age":${user.age}}"""

given JsonEncoder[Int] with
  def encode(value: Int): String = value.toString

def toJson[A](value: A)(using encoder: JsonEncoder[A]): String =
  encoder.encode(value)

// 使用例
val json = toJson(User("Alice", 30))
// → {"name":"Alice","age":30}

// for内包表記（モナディック合成）
def findUserOrders(userId: Long): Option[List[Order]] =
  for
    user <- userRepository.findById(userId)
    if user.isActive
    orders <- orderRepository.findByUser(user)
  yield orders

// Future を使った非同期処理
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

def fetchUserProfile(userId: Long): Future[UserProfile] =
  val userFuture = Future(userService.findUser(userId))
  val ordersFuture = Future(orderService.findOrders(userId))

  for
    user <- userFuture
    orders <- ordersFuture
  yield UserProfile(user, orders)

// ZIO による効果的プログラミング（Scala エコシステムの最前線）
import zio.*

def fetchUser(id: Long): ZIO[UserService, AppError, User] =
  ZIO.serviceWithZIO[UserService](_.findById(id))

def program: ZIO[UserService & OrderService, AppError, UserProfile] =
  for
    user <- fetchUser(42)
    orders <- fetchOrders(user.id)
  yield UserProfile(user, orders)

// コレクション操作（Scalaの真骨頂）
val employees = List(
  Employee("Alice", 30, "Engineering", 80000),
  Employee("Bob", 25, "Marketing", 60000),
  Employee("Carol", 35, "Engineering", 90000),
  Employee("Dave", 28, "Marketing", 65000),
  Employee("Eve", 32, "Engineering", 85000)
)

// 部門別の平均給与（トップ3のみ）
val result = employees
  .groupBy(_.department)
  .view
  .mapValues(emps => emps.map(_.salary).sum.toDouble / emps.size)
  .toList
  .sortBy(-_._2)
  .take(3)
// → List(("Engineering", 85000.0), ("Marketing", 62500.0))

// 型レベルプログラミング（高度）
// Phantom Types でコンパイル時に状態を検証
sealed trait DoorState
sealed trait Open extends DoorState
sealed trait Closed extends DoorState

class Door[S <: DoorState]:
  def open(using ev: S =:= Closed): Door[Open] = Door[Open]
  def close(using ev: S =:= Open): Door[Closed] = Door[Closed]

val door = Door[Closed]
val openDoor = door.open       // OK
// val invalid = openDoor.open  // コンパイルエラー！ すでに開いている

// Extension Methods（Scala 3）
extension (s: String)
  def words: List[String] = s.split("\\s+").toList
  def wordCount: Int = words.length

"Hello World Scala".wordCount  // → 3

// Match Types（Scala 3）— 型レベルパターンマッチ
type Elem[X] = X match
  case String => Char
  case Array[t] => t
  case Iterable[t] => t

// Elem[String] =:= Char
// Elem[Array[Int]] =:= Int
// Elem[List[Double]] =:= Double
```

### 3.4 Clojure — データ指向プログラミング

```clojure
;; Clojure: JVM上のモダンLisp
;; 設計哲学: シンプル、不変、データ中心

;; すべてが式（expression）
;; S式: (関数 引数1 引数2 ...)
(println "Hello, World!")

;; 不変データ構造（永続的データ構造）
(def user {:name "Alice"
           :age 30
           :tags ["admin" "developer"]})

;; assoc: キーを追加/更新（元のデータは変更されない）
(def updated-user (assoc user :email "alice@example.com"))
;; user は変わらない

;; update: 関数を適用して更新
(def older-user (update user :age inc))
;; → {:name "Alice", :age 31, :tags ["admin" "developer"]}

;; ネストしたデータの更新
(def company {:name "Acme"
              :address {:city "Tokyo"
                        :zip "100-0001"}})

(def updated (assoc-in company [:address :city] "Osaka"))
(def with-floor (update-in company [:address] assoc :floor 5))

;; スレッディングマクロ — データ変換パイプライン
;; -> (thread-first): 最初の引数として渡す
(-> "Hello, World!"
    .toUpperCase
    (.replace "," "")
    (.split " ")
    first)
;; → "HELLO"

;; ->> (thread-last): 最後の引数として渡す
(->> (range 1 101)
     (filter odd?)
     (map #(* % %))
     (reduce +))
;; → 1^2 + 3^2 + 5^2 + ... + 99^2

;; 高階関数とトランスデューサ
(def users [{:name "Alice" :age 30 :active true}
            {:name "Bob"   :age 17 :active true}
            {:name "Carol" :age 25 :active false}
            {:name "Dave"  :age 35 :active true}])

;; 従来のアプローチ（中間コレクションが生成される）
(->> users
     (filter :active)
     (filter #(>= (:age %) 18))
     (map :name)
     (sort))
;; → ("Alice" "Dave")

;; トランスデューサ（中間コレクションなし、効率的）
(def xf (comp
         (filter :active)
         (filter #(>= (:age %) 18))
         (map :name)))

(into [] xf users)
;; → ["Alice" "Dave"]

;; マルチメソッド — 柔軟なポリモーフィズム
(defmulti area :shape)

(defmethod area :circle [{:keys [radius]}]
  (* Math/PI radius radius))

(defmethod area :rectangle [{:keys [width height]}]
  (* width height))

(defmethod area :triangle [{:keys [base height]}]
  (/ (* base height) 2))

(area {:shape :circle :radius 5})        ;; → 78.54
(area {:shape :rectangle :width 4 :height 5})  ;; → 20

;; プロトコル — Java インターフェースに近い概念
(defprotocol Summarizable
  (summarize [this]))

(defrecord User [name age]
  Summarizable
  (summarize [this]
    (str name " (" age "歳)")))

(defrecord Article [title author]
  Summarizable
  (summarize [this]
    (str "\"" title "\" by " author)))

(summarize (->User "Alice" 30))       ;; → "Alice (30歳)"
(summarize (->Article "FP入門" "Bob")) ;; → "\"FP入門\" by Bob"

;; Spec — データのバリデーションと生成
(require '[clojure.spec.alpha :as s])

(s/def ::name (s/and string? #(> (count %) 0)))
(s/def ::age (s/and int? #(>= % 0) #(<= % 150)))
(s/def ::email (s/and string? #(re-matches #".+@.+\..+" %)))
(s/def ::user (s/keys :req-un [::name ::age]
                      :opt-un [::email]))

(s/valid? ::user {:name "Alice" :age 30})             ;; → true
(s/valid? ::user {:name "" :age 30})                   ;; → false
(s/explain ::user {:name "" :age 30})                  ;; エラー理由を表示

;; Atom — 安全な共有状態の管理
(def counter (atom 0))
(swap! counter inc)     ;; アトミックに +1
(swap! counter + 10)    ;; アトミックに +10
@counter                ;; → 11 （デリファレンス）

;; Agent — 非同期の状態更新
(def log-agent (agent []))
(send log-agent conj "Event 1")
(send log-agent conj "Event 2")
;; 別スレッドで順次処理される

;; core.async — CSP スタイルの並行処理
(require '[clojure.core.async :as async])

(let [ch (async/chan 10)]
  ;; 送信側
  (async/go
    (doseq [i (range 5)]
      (async/>! ch i))
    (async/close! ch))

  ;; 受信側
  (async/go-loop []
    (when-let [v (async/<! ch)]
      (println "Received:" v)
      (recur))))

;; REPL駆動開発 — Clojureの真髄
;; 1. REPLでデータを探索
;; 2. 関数を定義・テスト
;; 3. データ変換パイプラインを段階的に構築
;; 4. 名前空間にまとめる
;; → 「コード → コンパイル → 実行」のサイクルが不要
```

---

## 4. Java のモダンな進化（Java 17 - 25）

```
Java 17 (LTS, 2021):
  - Sealed Classes（継承の制限）
  - Pattern Matching for instanceof
  - テキストブロック
  - Records

Java 21 (LTS, 2023):
  - Virtual Threads（Project Loom）— 軽量スレッド
  - Pattern Matching for switch
  - Record Patterns
  - Sequenced Collections
  - String Templates（Preview）

Java 22-25 (2024-2025):
  - Structured Concurrency（構造化された並行処理）
  - Scoped Values（ThreadLocal の改善）
  - Stream Gatherers（カスタムストリーム操作）
  - Foreign Function & Memory API（JNI 代替）
  - Vector API（SIMD 操作）
  - Value Types（Project Valhalla, Preview）

重要なトレンド:
  → Java は6ヶ月リリースサイクルで急速に進化
  → Kotlin の良い機能を Java が取り入れる傾向
  → Records, Sealed Classes, Pattern Matching が Java をモダン化
  → Virtual Threads で並行処理のパラダイムが変わる
```

---

## 5. フレームワーク比較

### 5.1 Web フレームワーク

```java
// Java: Spring Boot — エンタープライズの王者
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

// Spring Data JPA — リポジトリの自動実装
public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByEmail(String email);
    List<User> findByAgeGreaterThan(int age);

    @Query("SELECT u FROM User u WHERE u.department = :dept AND u.active = true")
    List<User> findActiveByDepartment(@Param("dept") String department);
}
```

```kotlin
// Kotlin: Ktor — 軽量な非同期 Web フレームワーク
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

// Exposed — Kotlin の型安全な SQL DSL
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
// Scala: ZIO HTTP + Tapir — 型安全な API 定義
import sttp.tapir.*
import sttp.tapir.json.zio.*
import zio.*
import zio.json.*

// API エンドポイントの型安全な定義
val getUserEndpoint = endpoint
  .get
  .in("api" / "users" / path[Long]("id"))
  .out(jsonBody[UserDto])
  .errorOut(statusCode(StatusCode.NotFound))

val listUsersEndpoint = endpoint
  .get
  .in("api" / "users")
  .in(query[Int]("page").default(0))
  .out(jsonBody[List[UserDto]])

// エンドポイントの実装
val getUserRoute = getUserEndpoint.zServerLogic { id =>
  userService.findById(id)
    .map(_.toDto)
    .mapError(_ => StatusCode.NotFound)
}

// OpenAPI ドキュメントの自動生成
val docs = OpenAPIDocsInterpreter()
  .toOpenAPI(List(getUserEndpoint, listUsersEndpoint), "User API", "1.0.0")
```

```clojure
;; Clojure: Ring + Compojure — 関数ベースの Web フレームワーク
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

### 5.2 データ処理フレームワーク

```scala
// Scala: Apache Spark — 大規模データ処理の標準
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val spark = SparkSession.builder()
  .appName("UserAnalytics")
  .master("local[*]")
  .getOrCreate()

import spark.implicits._

// CSV 読み込み
val users = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("users.csv")

// データ変換と集計
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

// Dataset API（型安全）
case class UserEvent(userId: Long, action: String, timestamp: Long)

val events = spark.read.parquet("events.parquet").as[UserEvent]

val activeUsers = events
  .filter(_.action == "login")
  .groupByKey(_.userId)
  .count()
  .filter(_._2 >= 5)  // 5回以上ログインしたユーザー
```

---

## 6. テスト戦略の比較

```java
// Java: JUnit 5 + AssertJ + Mockito
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.*;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

@DisplayName("UserService テスト")
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
    @DisplayName("アクティブユーザーの一覧を取得する")
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
    @DisplayName("未成年はアダルトではない")
    void shouldNotBeAdult(int age) {
        var user = new User("Test", age, true);
        assertThat(user.isAdult()).isFalse();
    }
}
```

```kotlin
// Kotlin: Kotest — 多様なテストスタイル
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
        it("アクティブユーザーのみを返す") {
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

        context("ユーザーが存在しない場合") {
            it("空のリストを返す") {
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
;; Clojure: clojure.test + test.check（プロパティベーステスト）
(ns myapp.user-test
  (:require [clojure.test :refer [deftest testing is are]]
            [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [myapp.user :as user]))

(deftest test-find-active-users
  (testing "アクティブユーザーのみを返す"
    (let [users [{:name "Alice" :age 30 :active true}
                 {:name "Bob"   :age 25 :active false}
                 {:name "Carol" :age 35 :active true}]
          result (user/find-active users)]
      (is (= 2 (count result)))
      (is (= ["Alice" "Carol"] (map :name result)))))

  (testing "ユーザーが空の場合"
    (is (empty? (user/find-active [])))))

;; プロパティベーステスト
(def user-gen
  (gen/hash-map
    :name gen/string-alphanumeric
    :age (gen/choose 0 150)
    :active gen/boolean))

(def active-users-subset-property
  (prop/for-all [users (gen/vector user-gen)]
    (let [active (user/find-active users)]
      ;; アクティブユーザーは元のユーザーの部分集合
      (every? #(some #{%} users) active))))

(tc/quick-check 1000 active-users-subset-property)
```

---

## 7. GraalVM とネイティブイメージ

```
GraalVM: Oracle が開発する高性能 JVM
  → JIT コンパイラの改善（Graal コンパイラ）
  → Native Image: AOT コンパイルで起動を高速化
  → 多言語実行（Java, JavaScript, Python, Ruby, R, LLVM）

Native Image の利点:
  - 起動時間: 数秒 → 数十ミリ秒
  - メモリ使用量: 数百MB → 数十MB
  - パッケージサイズ: JRE不要（単体実行可能バイナリ）

制約:
  - リフレクションの制限（設定が必要）
  - 動的クラスローディングの制限
  - コンパイル時間が長い

対応フレームワーク:
  - Quarkus: Native Image ファースト
  - Micronaut: コンパイル時DI
  - Spring Native / Spring Boot 3.x
  - Helidon: Oracle の軽量フレームワーク
```

```java
// Quarkus: GraalVM Native Image に最適化
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

// ビルドコマンド:
// ./mvnw package -Pnative
// → 起動時間: ~0.02秒（通常のJVMは ~1秒）
// → メモリ: ~30MB（通常のJVMは ~200MB）
```

---

## 8. 選択指針の詳細

```
Q1: チームの規模と経験は？
├── 大規模チーム(20+人), Java経験豊富 → Java (Spring Boot)
│   理由: 人材確保が容易、ドキュメント・知見が豊富
├── 中規模チーム(5-20人), モダン志向 → Kotlin (Ktor/Spring)
│   理由: Javaとの互換性を保ちつつ生産性向上
├── 小規模チーム(1-5人), FP志向 → Scala or Clojure
│   理由: 少人数で高い表現力を活かせる
└── Android開発含む → Kotlin（一択）

Q2: パフォーマンス要件は？
├── 低遅延が必須 → Java (Virtual Threads) or Kotlin (Coroutines)
├── 大規模データ処理 → Scala (Spark)
├── 起動速度が重要 → Java/Kotlin + GraalVM Native Image
└── スループット重視 → Java (Reactor) or Kotlin (Flow)

Q3: プロジェクトの種類は？
├── エンタープライズ CRUD → Java (Spring Boot) or Kotlin (Spring)
├── マイクロサービス → Kotlin (Ktor) or Java (Quarkus)
├── データパイプライン → Scala (Spark) or Java (Flink)
├── イベント駆動システム → Scala (Akka/ZIO) or Kotlin (Coroutines)
├── REPL駆動の探索的開発 → Clojure
└── Android アプリ → Kotlin（Google公式）

Q4: 長期メンテナンス性は？
├── 10年以上の運用 → Java
│   理由: 後方互換性の保証、LTSリリース
├── 5-10年 → Kotlin or Java
│   理由: JetBrains + Google の支援
├── 技術的挑戦 → Scala
│   理由: 表現力最高だがチーム教育コスト高い
└── スタートアップ → Kotlin or Clojure
    理由: 少人数で高い生産性
```

---

## 9. Java ⇄ Kotlin 移行ガイド

### 9.1 段階的移行の手順

```
1. 新規ファイルを Kotlin で作成（Java プロジェクトに混在可能）
2. テストコードを Kotlin に変換（リスクが低い）
3. ユーティリティクラスを Kotlin に変換
4. 新機能はすべて Kotlin で実装
5. 段階的に既存 Java コードを変換（IntelliJ の自動変換を活用）

注意点:
  - build.gradle に kotlin プラグインを追加するだけで共存可能
  - Java から Kotlin を呼び出す際は @JvmStatic, @JvmOverloads を活用
  - Kotlin から Java を呼び出す際はプラットフォーム型に注意
```

### 9.2 主要な変換パターン

```java
// Java: POJO（ボイラープレートが多い）
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
// Kotlin: data class（1行で同等の機能）
data class User(val name: String, val age: Int, val email: String?)
// equals, hashCode, toString, copy, componentN が自動生成
```

```java
// Java: null チェックの連鎖
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
// Kotlin: セーフコール演算子
fun getUserCity(order: Order?): String =
    order?.user?.address?.city ?: "Unknown"
```

```java
// Java: Stream APIの型推論の限界
List<String> names = users.stream()
    .filter(u -> u.getAge() >= 18)
    .map(User::getName)
    .sorted()
    .collect(Collectors.toList());
```

```kotlin
// Kotlin: コレクション操作がより自然
val names = users
    .filter { it.age >= 18 }
    .map { it.name }
    .sorted()
// toList() 不要（すでにList）
```

---

## 10. 実践的なプロジェクト構成

### 10.1 Spring Boot + Kotlin プロジェクト

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

## まとめ

| 言語 | 哲学 | 最適な場面 | 2025年の状況 |
|------|------|----------|-------------|
| Java | 安定・エンタープライズ | 大規模業務システム | Java 21 で大幅進化。Virtual Threads が革命的 |
| Kotlin | 簡潔・安全・実用的 | Android, サーバーサイド | KMP で iOS/Web にも展開。勢い継続 |
| Scala | 表現力・型安全・FP | データ処理, 分散システム | Scala 3 で再出発。Spark エコシステム健在 |
| Clojure | シンプル・不変・REPL | データ処理, Web | ニッチだが根強い支持。REPL駆動開発の先駆者 |

### 選択の一言ガイド

```
迷ったら Java — 最も安全な選択、人材も確保しやすい
モダンに書きたいなら Kotlin — Java の良さ + 現代的な機能
データ処理なら Scala — Spark エコシステムの強さ
哲学を求めるなら Clojure — データ指向プログラミングの極致
Android なら Kotlin — Google 公式、一択
```

---

## 次に読むべきガイド
→ [[03-functional-languages.md]] -- 関数型言語比較

---

## 参考文献
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
