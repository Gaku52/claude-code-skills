# インターフェースとトレイト

> インターフェースは「契約」を定義し、トレイトは「再利用可能な振る舞い」を提供する。各言語での実装の違いと、ダックタイピングとの関係を理解する。

## この章で学ぶこと

- [ ] インターフェースとトレイトの違いを理解する
- [ ] 各言語での実装方法を把握する
- [ ] 構造的型付けとダックタイピングの関係を学ぶ
- [ ] インターフェース設計のベストプラクティスを習得する
- [ ] 型システムの違いが設計に与える影響を理解する

---

## 1. インターフェース vs トレイト vs 抽象クラス

```
┌──────────────┬────────────────┬────────────────┬────────────────┐
│              │ インターフェース│ トレイト        │ 抽象クラス     │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ メソッド宣言 │ ○             │ ○             │ ○             │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ デフォルト実装│ △(言語による) │ ○             │ ○             │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ フィールド   │ ×             │ △(言語による) │ ○             │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ 多重実装     │ ○             │ ○             │ ×             │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ コンストラクタ│ ×             │ ×             │ ○             │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ アクセス修飾子│ public のみ   │ △(言語による) │ 全て可能       │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ 代表言語     │ Java, TS, Go  │ Rust, Scala,PHP│ Java, Python   │
└──────────────┴────────────────┴────────────────┴────────────────┘

選択のガイドライン:
  インターフェース: 「何ができるか」の契約を定義したい
  トレイト: 再利用可能な振る舞いの実装を提供したい
  抽象クラス: 共通の状態と部分的な実装を共有したい
```

### 1.1 概念の関係性

```
型システムの3つの層:

  1. 契約層（Contract Layer）
     → インターフェース: 「何ができるか」を宣言
     → メソッドシグネチャのみ
     → 実装を持たない（原則）

  2. 振る舞い層（Behavior Layer）
     → トレイト: 再利用可能な振る舞いを定義
     → デフォルト実装を提供
     → 状態は持たない（原則）

  3. 実装層（Implementation Layer）
     → 抽象クラス / 具象クラス: 完全な実装
     → 状態（フィールド）を持つ
     → コンストラクタを持つ

  実装の進化:
    インターフェース（宣言のみ）
         ↓ デフォルトメソッド追加
    トレイト（宣言 + デフォルト実装）
         ↓ 状態の追加
    抽象クラス（宣言 + 実装 + 状態）
         ↓ 全メソッド実装
    具象クラス（完全な実装）

  近年の言語の傾向:
    → インターフェースとトレイトの境界が曖昧化
    → Java 8+: インターフェースにデフォルトメソッド
    → Kotlin: インターフェースにプロパティ
    → Swift: プロトコルエクステンション
    → PHP 8: インターフェースに近いトレイト
```

---

## 2. 各言語の実装

### 2.1 Java: インターフェース

```java
// Java: インターフェース（デフォルトメソッド付き）
public interface Comparable<T> {
    int compareTo(T other);
}

public interface Printable {
    void print();

    // デフォルトメソッド（Java 8+）
    default void printWithBorder() {
        System.out.println("================");
        print();
        System.out.println("================");
    }
}

// 複数のインターフェースを実装
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
// Java: インターフェースの高度な使い方

// 1. 関数型インターフェース（SAM: Single Abstract Method）
@FunctionalInterface
public interface Predicate<T> {
    boolean test(T t);

    // デフォルトメソッドで合成
    default Predicate<T> and(Predicate<T> other) {
        return t -> this.test(t) && other.test(t);
    }

    default Predicate<T> or(Predicate<T> other) {
        return t -> this.test(t) || other.test(t);
    }

    default Predicate<T> negate() {
        return t -> !this.test(t);
    }

    // static ファクトリーメソッド
    static <T> Predicate<T> isEqual(Object targetRef) {
        return t -> Objects.equals(t, targetRef);
    }
}

// ラムダ式で使用
Predicate<String> isNotEmpty = s -> !s.isEmpty();
Predicate<String> isLongEnough = s -> s.length() >= 8;
Predicate<String> isValidPassword = isNotEmpty.and(isLongEnough);

// 2. sealed インターフェース（Java 17+）
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

// パターンマッチング（Java 21+）
public String describeShape(Shape shape) {
    return switch (shape) {
        case Circle c -> "半径 " + c.radius() + " の円";
        case Rectangle r -> r.width() + "x" + r.height() + " の長方形";
        case Triangle t -> "三角形（辺: " + t.a() + ", " + t.b() + ", " + t.c() + "）";
    };
}

// 3. インターフェースのデフォルトメソッド競合
public interface A {
    default String greet() { return "Hello from A"; }
}

public interface B {
    default String greet() { return "Hello from B"; }
}

// 両方を実装する場合、明示的にオーバーライドが必要
public class C implements A, B {
    @Override
    public String greet() {
        // 明示的にどちらかを選ぶ
        return A.super.greet();
    }
}
```

### 2.2 Rust: トレイト

```rust
// Rust: トレイト（インターフェース + デフォルト実装 + ジェネリクス制約）
trait Summary {
    fn summarize_author(&self) -> String;

    // デフォルト実装
    fn summarize(&self) -> String {
        format!("({}からの新着...)", self.summarize_author())
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

    // summarize() はデフォルト実装を使用
}

// トレイト境界: ジェネリクスの制約として使用
fn notify(item: &impl Summary) {
    println!("速報: {}", item.summarize());
}

// 複数トレイトの組み合わせ
fn display_and_summarize(item: &(impl Summary + std::fmt::Display)) {
    println!("{}", item);
    println!("{}", item.summarize());
}
```

```rust
// Rust: トレイトの高度な使い方

// 1. 関連型（Associated Types）
trait Iterator {
    type Item;  // 関連型: 実装者が具体的な型を指定

    fn next(&mut self) -> Option<Self::Item>;

    // デフォルト実装: 関連型を使ったメソッド
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
    type Item = u32;  // この Iterator の要素は u32

    fn next(&mut self) -> Option<u32> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}

// 2. スーパートレイト（トレイトの継承）
trait Animal {
    fn name(&self) -> &str;
}

trait Pet: Animal {  // Pet は Animal のスーパートレイトを要求
    fn cuddle(&self) -> String {
        format!("{}をなでなで", self.name())
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
    // cuddle() はデフォルト実装を使用
}

// 3. トレイトオブジェクト（動的ディスパッチ）
fn print_summaries(items: &[&dyn Summary]) {
    for item in items {
        println!("{}", item.summarize());
    }
}

// 4. ブランケット実装（Blanket Implementation）
// Display を実装するすべての型に ToString を自動実装
impl<T: std::fmt::Display> ToString for T {
    fn to_string(&self) -> String {
        format!("{}", self)
    }
}

// 5. From/Into トレイト（型変換）
struct Celsius(f64);
struct Fahrenheit(f64);

impl From<Celsius> for Fahrenheit {
    fn from(c: Celsius) -> Self {
        Fahrenheit(c.0 * 9.0 / 5.0 + 32.0)
    }
}

// Into は From から自動導出される
let c = Celsius(100.0);
let f: Fahrenheit = c.into();  // Fahrenheit(212.0)

// 6. Derive マクロ（トレイトの自動実装）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Point {
    x: i32,
    y: i32,
}
// Debug, Clone, PartialEq, Eq, Hash が自動実装される

// 7. Newtype パターン（外部の型にトレイトを実装）
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

### 2.3 Go: 暗黙的インターフェース

```go
// Go: 構造的型付け（暗黙的にインターフェースを満たす）
type Writer interface {
    Write(p []byte) (n int, err error)
}

type Reader interface {
    Read(p []byte) (n int, err error)
}

// ReadWriter は Writer と Reader の合成
type ReadWriter interface {
    Reader
    Writer
}

// MyBuffer は Writer を「宣言なしに」満たす
type MyBuffer struct {
    data []byte
}

func (b *MyBuffer) Write(p []byte) (int, error) {
    b.data = append(b.data, p...)
    return len(p), nil
}

// implements Writer とは書かない（暗黙的に満たす）
var w Writer = &MyBuffer{}
```

```go
// Go: インターフェースの高度な使い方

// 1. 小さなインターフェース（Go の哲学）
// Go のインターフェースは通常 1-3 メソッド
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

// 2. 空インターフェース（any / interface{}）
func PrintAnything(v any) {
    fmt.Println(v)
}

// 3. 型アサーション
func Process(r Reader) {
    // r が Closer も満たすか確認
    if closer, ok := r.(Closer); ok {
        defer closer.Close()
    }

    // 型switch
    switch v := r.(type) {
    case *os.File:
        fmt.Println("ファイル:", v.Name())
    case *bytes.Buffer:
        fmt.Println("バッファ:", v.Len(), "バイト")
    default:
        fmt.Println("不明なReader")
    }
}

// 4. インターフェースの合成パターン
type Handler interface {
    Handle(ctx context.Context, req Request) (Response, error)
}

type Middleware func(Handler) Handler

// ミドルウェアの連鎖
func Chain(h Handler, middlewares ...Middleware) Handler {
    for i := len(middlewares) - 1; i >= 0; i-- {
        h = middlewares[i](h)
    }
    return h
}

// ロギングミドルウェア
func LoggingMiddleware(next Handler) Handler {
    return HandlerFunc(func(ctx context.Context, req Request) (Response, error) {
        start := time.Now()
        resp, err := next.Handle(ctx, req)
        log.Printf("handled in %v", time.Since(start))
        return resp, err
    })
}

// 認証ミドルウェア
func AuthMiddleware(next Handler) Handler {
    return HandlerFunc(func(ctx context.Context, req Request) (Response, error) {
        token := req.Header("Authorization")
        if token == "" {
            return nil, ErrUnauthorized
        }
        // トークン検証...
        return next.Handle(ctx, req)
    })
}

// 使用例
handler := Chain(myHandler, LoggingMiddleware, AuthMiddleware)

// 5. コンパイル時のインターフェース準拠チェック
// 構造体がインターフェースを満たすことを保証するイディオム
var _ Writer = (*MyBuffer)(nil)
var _ Reader = (*MyBuffer)(nil)
// MyBuffer が Writer/Reader を満たさない場合、コンパイルエラー
```

### 2.4 TypeScript: 構造的型付け

```typescript
// TypeScript: 構造的型付け（Structural Typing）
interface Loggable {
  toLogString(): string;
}

// 明示的に implements しなくても、構造が合えばOK
class User {
  constructor(public name: string, public email: string) {}

  toLogString(): string {
    return `User(${this.name}, ${this.email})`;
  }
}

// User は Loggable を明示的に implements していないが、
// toLogString() を持つので Loggable として使える
function log(item: Loggable): void {
  console.log(item.toLogString());
}

log(new User("田中", "tanaka@example.com")); // OK
```

```typescript
// TypeScript: インターフェースの高度な使い方

// 1. ジェネリックインターフェース
interface Repository<T> {
  findById(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: string): Promise<void>;
}

interface Identifiable {
  id: string;
}

// ジェネリクス制約
interface CrudRepository<T extends Identifiable> extends Repository<T> {
  update(id: string, data: Partial<T>): Promise<T>;
}

// 2. インデックスシグネチャ
interface Dictionary<T> {
  [key: string]: T;
}

const scores: Dictionary<number> = {
  math: 90,
  english: 85,
  science: 92,
};

// 3. 呼び出しシグネチャ
interface Formatter {
  (value: unknown): string;
  locale: string;
}

const jsonFormatter: Formatter = Object.assign(
  (value: unknown) => JSON.stringify(value),
  { locale: "ja-JP" },
);

// 4. インターセクション型（型の合成）
interface HasName {
  name: string;
}

interface HasEmail {
  email: string;
}

interface HasAge {
  age: number;
}

// インターセクション型で合成
type UserInfo = HasName & HasEmail & HasAge;

const user: UserInfo = {
  name: "田中",
  email: "tanaka@example.com",
  age: 30,
};

// 5. Conditional Types とインターフェース
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

// 全フィールドをオプショナルに
type PartialUser = Partial<User>;

// 全フィールドを読み取り専用に
type ReadonlyUser = Readonly<User>;

// 特定のフィールドのみ取得
type UserPreview = Pick<User, "id" | "name">;

// 特定のフィールドを除外
type UserWithoutId = Omit<User, "id">;

// 7. Template Literal Types とインターフェース
type EventName = "click" | "hover" | "focus";
type HandlerName = `on${Capitalize<EventName>}`;
// "onClick" | "onHover" | "onFocus"

interface EventHandlers {
  onClick(event: MouseEvent): void;
  onHover(event: MouseEvent): void;
  onFocus(event: FocusEvent): void;
}
```

### 2.5 Python: Protocol（構造的サブタイピング）

```python
# Python: Protocol によるインターフェース（Python 3.8+）
from typing import Protocol, runtime_checkable


# Protocol: 構造的型付け（ダックタイピングの型安全版）
class Renderable(Protocol):
    """レンダリング可能なオブジェクトの契約"""
    def render(self) -> str: ...


class HtmlComponent:
    """Protocol を明示的に実装する必要なし"""
    def __init__(self, tag: str, content: str):
        self.tag = tag
        self.content = content

    def render(self) -> str:
        return f"<{self.tag}>{self.content}</{self.tag}>"


class MarkdownText:
    """これも render() を持つので Renderable"""
    def __init__(self, text: str):
        self.text = text

    def render(self) -> str:
        return self.text


# render() を持つ何でも受け取れる
def display(item: Renderable) -> None:
    print(item.render())


display(HtmlComponent("h1", "Hello"))  # <h1>Hello</h1>
display(MarkdownText("# Hello"))       # # Hello
```

```python
# Python: Protocol の高度な使い方

from typing import Protocol, runtime_checkable, TypeVar, Generic
from abc import abstractmethod


# 1. runtime_checkable: isinstance() で使える
@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...


class FileWrapper:
    def __init__(self, path: str):
        self.file = open(path)

    def close(self) -> None:
        self.file.close()


# isinstance でチェック可能
wrapper = FileWrapper("test.txt")
assert isinstance(wrapper, Closable)  # True


# 2. ジェネリック Protocol
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class Comparable(Protocol[T]):
    """比較可能なオブジェクト"""
    def __lt__(self, other: T) -> bool: ...
    def __le__(self, other: T) -> bool: ...
    def __gt__(self, other: T) -> bool: ...
    def __ge__(self, other: T) -> bool: ...


class SupportsAdd(Protocol[T_co]):
    """加算可能なオブジェクト"""
    def __add__(self, other: "SupportsAdd[T_co]") -> T_co: ...


# 3. Protocol のメソッドにデフォルト実装は持てないが、
#    Mixin と組み合わせて使える
class EqualityMixin:
    """等値比較のデフォルト実装を提供するMixin"""
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class HashableMixin(EqualityMixin):
    """ハッシュのデフォルト実装"""
    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))


# 4. Protocol を使った依存性注入
class UserRepository(Protocol):
    async def find_by_id(self, user_id: str) -> dict | None: ...
    async def save(self, user: dict) -> None: ...


class EmailSender(Protocol):
    async def send(self, to: str, subject: str, body: str) -> None: ...


class Logger(Protocol):
    def info(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...


class UserService:
    """Protocol に依存（具象クラスに依存しない）"""
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
        await self.email.send(email_addr, "Welcome!", "ご登録ありがとうございます")
        return user


# テスト用のモック（Protocol を満たせばOK）
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


# テスト
import asyncio

async def test_register():
    repo = MockUserRepository()
    email = MockEmailSender()
    logger = MockLogger()

    service = UserService(repo, email, logger)
    user = await service.register("田中", "tanaka@example.com")

    assert len(repo.users) == 1
    assert len(email.sent) == 1
    assert email.sent[0]["to"] == "tanaka@example.com"
    assert "[INFO] Registering user: tanaka@example.com" in logger.messages

asyncio.run(test_register())
```

### 2.6 Scala: トレイト

```scala
// Scala: トレイト（インターフェース + デフォルト実装 + 状態）
trait Greeter {
  // 抽象メソッド
  def name: String

  // デフォルト実装
  def greet(): String = s"Hello, $name!"
}

trait Logger {
  // トレイトは状態を持てる
  var logLevel: String = "INFO"

  def log(message: String): Unit = {
    println(s"[$logLevel] $message")
  }
}

trait Serializable {
  def toJson: String
}

// 複数のトレイトを合成
class User(val name: String, val email: String)
    extends Greeter
    with Logger
    with Serializable {

  override def toJson: String =
    s"""{"name": "$name", "email": "$email"}"""
}

// self-type: 依存関係の宣言
trait Repository {
  self: Logger =>  // Repository は Logger を必要とする
  def save(data: String): Unit = {
    log(s"Saving: $data")
    // 永続化処理
  }
}

// 動的ミックスイン（インスタンス生成時にトレイトを追加）
val user = new User("田中", "tanaka@example.com") with Serializable {
  override def toJson: String = s"""{"name": "$name"}"""
}

// ケーキパターン（DI）
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

### 2.7 Swift: プロトコル

```swift
// Swift: プロトコル（インターフェース + Protocol Extensions）
protocol Drawable {
    func draw()
}

protocol Resizable {
    var width: Double { get set }
    var height: Double { get set }
    func resize(by factor: Double)
}

// Protocol Extension: デフォルト実装を提供
extension Resizable {
    func resize(by factor: Double) {
        width *= factor
        height *= factor
    }

    var area: Double {
        return width * height
    }
}

// プロトコル合成（Protocol Composition）
typealias InteractiveElement = Drawable & Resizable

struct Button: InteractiveElement {
    var label: String
    var width: Double
    var height: Double

    func draw() {
        print("Drawing button: \(label) (\(width)x\(height))")
    }
}

// Associated Types（関連型）
protocol Container {
    associatedtype Item  // 関連型
    var count: Int { get }
    mutating func append(_ item: Item)
    subscript(i: Int) -> Item { get }
}

struct Stack<Element>: Container {
    typealias Item = Element  // 関連型の指定（推論可能なら省略可）
    var items: [Element] = []
    var count: Int { items.count }
    mutating func append(_ item: Element) { items.append(item) }
    subscript(i: Int) -> Element { items[i] }
}

// where句でジェネリクス制約
func allEqual<C: Container>(_ container: C) -> Bool
    where C.Item: Equatable {
    if container.count < 2 { return true }
    for i in 1..<container.count {
        if container[i] != container[0] { return false }
    }
    return true
}

// Existential Types（any キーワード、Swift 5.7+）
func printAll(_ items: [any Drawable]) {
    for item in items {
        item.draw()
    }
}

// Opaque Types（some キーワード）
func makeShape() -> some Drawable {
    return Button(label: "OK", width: 100, height: 40)
}
```

---

## 3. ダックタイピング

```
「アヒルのように歩き、アヒルのように鳴くなら、それはアヒルだ」

名前的型付け（Nominal Typing）:
  → 明示的に implements/extends した型のみ互換
  → Java, C#, Swift

構造的型付け（Structural Typing）:
  → 構造（メソッド/プロパティ）が合えば互換
  → TypeScript, Go

ダックタイピング（Duck Typing）:
  → 実行時にメソッドが存在すれば呼べる
  → Python, Ruby, JavaScript

型チェックの厳密さ:
  名前的型付け > 構造的型付け > ダックタイピング

安全性と柔軟性のトレードオフ:
  名前的型付け: 安全性 高 / 柔軟性 低
  構造的型付け: 安全性 中 / 柔軟性 中
  ダックタイピング: 安全性 低 / 柔軟性 高
```

```python
# Python: ダックタイピング
class Duck:
    def quack(self):
        return "ガーガー"

class Person:
    def quack(self):
        return "（人間が真似する）ガーガー"

class RubberDuck:
    def quack(self):
        return "キュッキュッ"

# 型宣言なしに、quack() を持つ何でも渡せる
def make_it_quack(thing):
    print(thing.quack())

make_it_quack(Duck())       # ガーガー
make_it_quack(Person())     # （人間が真似する）ガーガー
make_it_quack(RubberDuck()) # キュッキュッ

# Protocol（Python 3.8+）: 型ヒントでダックタイピングを型安全に
from typing import Protocol

class Quackable(Protocol):
    def quack(self) -> str: ...

def make_it_quack_typed(thing: Quackable) -> None:
    print(thing.quack())
```

```ruby
# Ruby: ダックタイピング
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
    # 何もしない
  end
end

# write() を持つ何でも渡せる
def process(writer, data)
  writer.write("Processing: #{data}")
  # writerの具体的な型を気にしない
end

process(Logger.new, "test data")
process(FileWriter.new("output.log"), "test data")
process(NullWriter.new, "test data")

# respond_to? でメソッドの存在を確認
def safe_write(writer, message)
  if writer.respond_to?(:write)
    writer.write(message)
  else
    puts "Warning: writer does not support write"
  end
end
```

### 3.1 各型付け方式の比較

```typescript
// TypeScript: 構造的型付けの利点と注意点

// 利点1: サードパーティライブラリとの互換性
// ライブラリAが定義したインターフェース
interface PointA {
  x: number;
  y: number;
}

// ライブラリBが定義した別のインターフェース
interface PointB {
  x: number;
  y: number;
}

// 名前が違っても構造が同じなら互換
function distanceA(p: PointA): number {
  return Math.sqrt(p.x ** 2 + p.y ** 2);
}

const pointB: PointB = { x: 3, y: 4 };
distanceA(pointB); // ✅ OK（構造的型付け）
// Java なら: ❌ コンパイルエラー（名前的型付け）

// 注意点: 構造が同じでも意味が異なる場合
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

findUser(productId); // ✅ TypeScriptではコンパイル通る！（構造が同じ）
// → 意味的には間違い
// → Branded Types で解決

// Branded Types: 構造的型付けで名前的型付けを実現
type Brand<T, B extends string> = T & { __brand: B };
type StrictUserId = Brand<string, "UserId">;
type StrictProductId = Brand<string, "ProductId">;

function findUserStrict(id: StrictUserId): User { /* ... */ }
function findProductStrict(id: StrictProductId): Product { /* ... */ }

const strictUserId = "user-123" as StrictUserId;
const strictProductId = "product-456" as StrictProductId;

// findUserStrict(strictProductId); // ❌ コンパイルエラー！
findUserStrict(strictUserId);        // ✅ OK
```

---

## 4. インターフェース設計のベストプラクティス

```
1. 小さく保つ（ISP準拠）:
   → メソッド数は1-5個が理想
   → 「このインターフェースの全メソッドを
      すべての実装者が意味的に実装できるか？」

2. クライアント視点で設計:
   → 実装者ではなく利用者の観点で
   → 「このインターフェースのメソッドが
      すべて必要なクライアントは存在するか？」

3. 名前で意図を伝える:
   → -able, -er, -or サフィックス
   → Comparable, Serializer, Validator
   → 「〜できる」「〜するもの」

4. 安定した契約:
   → インターフェースは変更しにくい
   → 最初から完璧を目指さず、少しずつ追加

5. テスタビリティを考慮:
   → 外部依存をインターフェースで抽象化
   → モックを作りやすい粒度に

6. ドメインの言葉を使う:
   → 技術用語より業務用語
   → interface OrderProcessor > interface DataHandler
```

```typescript
// インターフェース設計の良い例と悪い例

// ❌ 悪い例: 巨大なインターフェース
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

// ❌ 悪い例: 技術的すぎる名前
interface IDataAccessObject {
  executeSQL(query: string): Promise<any>;
  commitTransaction(): Promise<void>;
  rollbackTransaction(): Promise<void>;
}

// ✅ 良い例: 小さく、ドメイン志向
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

// ✅ 良い例: 関数型インターフェース
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

## 5. 選択指針

```
インターフェース:
  → 「何ができるか」の契約を定義
  → 実装は持たない（またはデフォルト最小限）
  → 多重実装が必要な場合
  → 異なる型に共通の振る舞いを強制

トレイト:
  → 再利用可能な振る舞いの単位
  → デフォルト実装を積極的に提供
  → ミックスイン的な使い方
  → コードの重複を排除しつつ柔軟に合成

抽象クラス:
  → 共通の状態（フィールド）+ 部分的な実装
  → テンプレートメソッドパターン
  → is-a 関係が明確な場合
  → コンストラクタでの初期化が必要

言語別の推奨:
  Java: インターフェース（デフォルトメソッド活用）
  TypeScript: インターフェース（構造的型付けを活用）
  Go: インターフェース（小さく、暗黙的に）
  Rust: トレイト（唯一の抽象化メカニズム）
  Python: Protocol（型安全なダックタイピング）
  Scala: トレイト（状態も持てる柔軟さ）
  Swift: プロトコル（Protocol Extension 活用）
```

### 5.1 実務での判断基準

```
判断フローチャート:

  Q1: 「状態（フィールド）の共有が必要か？」
  │
  ├── Yes → 抽象クラス or コンポジション
  │         Q1a: 「is-a 関係が明確か？」
  │         ├── Yes → 抽象クラス
  │         └── No → コンポジション
  │
  └── No
      │
      Q2: 「デフォルト実装を提供したいか？」
      │
      ├── Yes → トレイト / インターフェース（デフォルトメソッド）
      │
      └── No → インターフェース（純粋な契約）

  具体的なシナリオ:

  シナリオ1: DB接続の抽象化
  → インターフェース
  → 複数実装（MySQL, Postgres, SQLite）
  → 状態は実装クラスが持つ

  シナリオ2: ログ出力のヘルパー
  → トレイト / ミックスイン
  → デフォルト実装を提供
  → 多くのクラスで横断的に使用

  シナリオ3: UIコンポーネントの基底
  → 抽象クラス（フレームワーク提供）
  → 共通の状態（width, height, visible）
  → テンプレートメソッド（render, update）

  シナリオ4: 型の制約
  → インターフェース / トレイト境界
  → ジェネリクスの制約として使用
  → 「T は Comparable を満たす」
```

---

## まとめ

| 概念 | 特徴 | 代表言語 |
|------|------|---------|
| インターフェース | 契約の定義 | Java, TS, Go |
| トレイト | 再利用可能な振る舞い | Rust, Scala, PHP |
| 構造的型付け | 構造が合えば互換 | TS, Go |
| ダックタイピング | 実行時にメソッド確認 | Python, Ruby |
| Protocol | 型安全なダックタイピング | Python, Swift |

```
実践的な指針:

  1. インターフェースは契約
     → 「何ができるか」を定義する
     → 「どう実装するか」は実装者の自由

  2. 小さいインターフェースは良いインターフェース
     → 1メソッドのインターフェースは最も再利用しやすい
     → Go の io.Reader, io.Writer が好例

  3. 言語の特性を活かす
     → TypeScript: 構造的型付け → implements は省略可能
     → Go: 暗黙的インターフェース → 後から適合可能
     → Rust: トレイト境界 → ジェネリクスの制約として活用
     → Python: Protocol → ダックタイピングに型安全性を追加

  4. テストを意識する
     → 外部依存はインターフェースで抽象化
     → モック作成が容易な粒度に
```

---

## 6. インターフェースの進化パターン

```
インターフェースのバージョニング:

  問題: インターフェースにメソッドを追加すると
       既存の実装がすべて壊れる

  解決策 1: デフォルトメソッド（Java 8+）
    → 既存実装を壊さずにメソッドを追加
    → ただし、デフォルト実装は最小限に

  解決策 2: インターフェース分割
    → V1 + 追加インターフェースで拡張
    → UserService → UserService + UserServiceV2

  解決策 3: アダプターパターン
    → 旧インターフェースを新インターフェースに適合
    → 移行期間を設けて段階的に切り替え

  推奨ルール:
    1. インターフェースは公開後、原則変更しない
    2. 新機能は新インターフェースとして追加
    3. デフォルトメソッドは後方互換のためだけに使う
    4. 非推奨（@Deprecated）を活用して段階的に移行
```

```java
// Java: インターフェースの進化パターン

// V1: 初期リリース
public interface PaymentGateway {
    PaymentResult charge(String customerId, BigDecimal amount);
    PaymentResult refund(String transactionId);
}

// V2: 新機能を追加（デフォルトメソッドで後方互換を維持）
public interface PaymentGateway {
    PaymentResult charge(String customerId, BigDecimal amount);
    PaymentResult refund(String transactionId);

    // V2で追加: デフォルト実装で後方互換
    default PaymentResult chargeWithCurrency(
            String customerId, BigDecimal amount, Currency currency) {
        // デフォルトでは通貨変換なしで charge を呼ぶ
        return charge(customerId, amount);
    }

    // V2で追加: サブスクリプション対応
    default SubscriptionResult subscribe(
            String customerId, String planId) {
        throw new UnsupportedOperationException(
            "This gateway does not support subscriptions");
    }
}

// 別パターン: インターフェース分割
public interface SubscriptionGateway extends PaymentGateway {
    SubscriptionResult subscribe(String customerId, String planId);
    void cancelSubscription(String subscriptionId);
}
```

```typescript
// TypeScript: インターフェースの拡張パターン

// 宣言のマージ（Declaration Merging）
// 同名インターフェースは自動的にマージされる
interface Config {
  host: string;
  port: number;
}

// 別の場所で追加（ライブラリの拡張に便利）
interface Config {
  ssl: boolean;
  timeout: number;
}

// マージ結果: { host, port, ssl, timeout }
const config: Config = {
  host: "localhost",
  port: 3000,
  ssl: true,
  timeout: 5000,
};

// モジュール拡張（Module Augmentation）
// express の Request に独自プロパティを追加
declare module "express" {
  interface Request {
    user?: {
      id: string;
      role: string;
    };
  }
}

// グローバル型の拡張
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
// Go: インターフェースの段階的拡張

// 基本インターフェース
type Storage interface {
    Get(key string) ([]byte, error)
    Put(key string, value []byte) error
    Delete(key string) error
}

// 拡張インターフェース: バッチ操作対応
type BatchStorage interface {
    Storage
    BatchGet(keys []string) (map[string][]byte, error)
    BatchPut(items map[string][]byte) error
}

// 拡張インターフェース: TTL対応
type TTLStorage interface {
    Storage
    PutWithTTL(key string, value []byte, ttl time.Duration) error
    GetTTL(key string) (time.Duration, error)
}

// 実行時に拡張機能の有無を確認
func StoreData(s Storage, key string, value []byte, ttl time.Duration) error {
    // TTL対応ストレージなら TTL 付きで保存
    if ts, ok := s.(TTLStorage); ok {
        return ts.PutWithTTL(key, value, ttl)
    }
    // 非対応なら通常の Put
    return s.Put(key, value)
}

// テスト用のストレージ実装
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

// コンパイル時チェック
var _ Storage = (*MemoryStorage)(nil)
```

---

## 7. テストにおけるインターフェースの活用

```
テスト戦略:

  1. インターフェースを使ったモック
     → 外部依存（DB、API、ファイル）をインターフェースで抽象化
     → テスト時にモック実装を注入
     → テストの実行速度向上 + 独立性確保

  2. テストダブルの種類:
     → スタブ（Stub）: 固定値を返す
     → モック（Mock）: 呼び出しを検証する
     → フェイク（Fake）: 簡易的な代替実装
     → スパイ（Spy）: 呼び出しを記録しつつ本物に委譲

  3. インターフェース設計とテスタビリティ:
     → 小さなインターフェースはモックが楽
     → 1メソッドインターフェースは最もテストしやすい
     → メソッドが増えるとモック作成が煩雑に
```

```python
# Python: Protocol を使ったテスト戦略

from typing import Protocol
from dataclasses import dataclass, field
import pytest


# プロダクションコードの Protocol 定義
class Clock(Protocol):
    def now(self) -> float: ...

class RandomGenerator(Protocol):
    def random(self) -> float: ...

class NotificationSender(Protocol):
    def send(self, recipient: str, message: str) -> bool: ...


# テスト用のフェイク実装
class FakeClock:
    """テスト用: 固定時刻を返す"""
    def __init__(self, fixed_time: float = 1000.0):
        self._time = fixed_time

    def now(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


class FakeRandom:
    """テスト用: 事前に決めた値を順番に返す"""
    def __init__(self, values: list[float]):
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


@dataclass
class SpyNotificationSender:
    """テスト用: 送信を記録するスパイ"""
    sent: list[dict] = field(default_factory=list)
    should_succeed: bool = True

    def send(self, recipient: str, message: str) -> bool:
        self.sent.append({
            "recipient": recipient,
            "message": message,
        })
        return self.should_succeed


# プロダクションコード
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
        expiry = self.clock.now() + 86400  # 24時間後
        self.notifier.send(
            user_email,
            f"クーポンコード: {code}（有効期限: {expiry}）",
        )
        return code


# テスト
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

    # 通知が失敗してもクーポンは発行される
    assert code == "COUPON-1234"
    assert len(notifier.sent) == 1
```

```rust
// Rust: トレイトを使ったテスト戦略

use std::collections::HashMap;

// プロダクションコードのトレイト定義
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

// テスト用のモック実装
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
            // 注: テストではmutable参照が必要なため、
            // 実際にはRefCellなどを使う
            Ok(())
        }
    }

    #[test]
    fn test_find_existing_user() {
        let store = MockUserStore::new().with_user(User {
            id: "user-1".to_string(),
            name: "田中".to_string(),
            email: "tanaka@example.com".to_string(),
        });

        let user = store.find_by_id("user-1");
        assert!(user.is_some());
        assert_eq!(user.unwrap().name, "田中");
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

## 次に読むべきガイド
→ [[02-mixins-and-multiple-inheritance.md]] — ミックスインと多重継承

---

## 参考文献
1. Odersky, M. "Scalable Component Abstractions." OOPSLA, 2005.
2. The Rust Programming Language. "Traits." doc.rust-lang.org.
3. Bloch, J. "Effective Java." 3rd Edition, Addison-Wesley, 2018.
4. The Go Programming Language Specification. "Interface types." golang.org.
5. Python PEP 544. "Protocols: Structural subtyping." 2017.
6. Apple Developer Documentation. "Protocols." developer.apple.com.
7. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
8. Martin, R.C. "Clean Architecture." Prentice Hall, 2017.
