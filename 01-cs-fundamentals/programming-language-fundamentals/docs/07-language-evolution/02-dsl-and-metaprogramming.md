# DSL and Metaprogramming

> A DSL (Domain-Specific Language) is "a language specialized for a particular problem domain," and metaprogramming is "the technique of generating or transforming programs with programs." These two concepts are two sides of the same coin and serve as essential techniques for building highly expressive software. This chapter covers both theory and practice, from DSL design principles to various metaprogramming approaches.

## What You Will Learn in This Chapter

- [ ] Understand the differences between internal and external DSLs and their design trade-offs
- [ ] Grasp the major metaprogramming techniques (macros, reflection, code generation)
- [ ] Implement DSL construction techniques in various languages (Kotlin, Ruby, Scala, Swift, Rust)
- [ ] Distinguish between types of macro systems (text substitution, syntactic macros, procedural macros)
- [ ] Assess the appropriate scope and risks of metaprogramming
- [ ] Apply type-safe DSL design patterns (Builder, Phantom Type)


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Modern Language Features](./01-modern-language-features.md)

---

## 1. Foundational Theory of DSLs

### 1.1 What Is a DSL?

A DSL (Domain-Specific Language) is a language optimized for a specific problem domain. While general-purpose programming languages (GPL: General-Purpose Language) are Turing-complete and can express any computation, DSLs intentionally restrict expressiveness to maximize productivity and safety within a particular domain.

```
DSL vs GPL positioning:

  Range of expressiveness
  <---------------------------------------->
  Narrow (specialized)           Wide (general-purpose)

  +------+  +-------+  +--------+  +----------+
  |Regex |  | SQL   |  |Terraform| | Python   |
  |      |  |       |  | / HCL  | | Java     |
  +------+  +-------+  +--------+ | Rust     |
                                   | etc.     |
  DSL                              +----------+
  - Learning cost: Low              GPL
  - Expressiveness: Limited         - Learning cost: High
  - Safety: High                    - Expressiveness: Unlimited
  - Optimization: Easy              - Safety: Language-dependent
                                    - Optimization: Difficult
```

### 1.2 DSL Classification System

```
DSL classification:

  +--------------------------------------------------+
  |                    DSL                            |
  +----------------------+---------------------------+
  |    External DSL      |       Internal DSL        |
  |  (External DSL)      |   (Internal DSL /         |
  |                      |    Embedded DSL)           |
  +----------------------+---------------------------+
  | - Requires custom    | - Uses host language       |
  |   parser             |   syntax                   |
  | - Defines custom     | - Builds DSL with          |
  |   syntax             |   language features        |
  | - Requires tooling   | - Natural IDE support      |
  |                      |                            |
  | Examples:            | Examples:                  |
  | - SQL                | - RSpec (Ruby)             |
  | - HTML/CSS           | - kotlinx.html (Kotlin)    |
  | - Regular expressions| - SwiftUI (Swift)          |
  | - GraphQL            | - Ktor routing (Kotlin)    |
  | - Terraform HCL      | - ScalaTest (Scala)       |
  | - Protocol Buffers   | - Diesel (Rust)            |
  +----------------------+---------------------------+
```

### 1.3 Anatomy of Famous External DSLs

#### SQL: A DSL for Data Queries

SQL is one of the most successful external DSLs, designed in the 1970s based on E.F. Codd's relational model.

```sql
-- SQL: Declarative data querying
-- Describes "what to retrieve," while the DB engine determines "how to retrieve it"

SELECT
    u.name,
    COUNT(o.id) AS order_count,
    SUM(o.total_amount) AS total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2024-01-01'
GROUP BY u.name
HAVING COUNT(o.id) >= 3
ORDER BY total_spent DESC
LIMIT 10;

-- Writing this in a general-purpose language would require dozens of lines of loops and filtering
```

#### GraphQL: A DSL for API Queries

```graphql
# GraphQL: Client-driven API querying
query GetUserDashboard($userId: ID!) {
  user(id: $userId) {
    name
    email
    orders(last: 5) {
      edges {
        node {
          id
          total
          status
          items {
            product { name }
            quantity
          }
        }
      }
    }
    notifications(unreadOnly: true) {
      message
      createdAt
    }
  }
}
```

#### Terraform HCL: A DSL for Infrastructure Definition

```hcl
# Terraform: Declaring infrastructure as code
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "production-vpc"
  }
}

resource "aws_subnet" "web" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "ap-northeast-1a"
}

resource "aws_instance" "web_server" {
  ami           = "ami-0abcdef1234567890"
  instance_type = "t3.medium"
  subnet_id     = aws_subnet.web.id

  tags = {
    Name        = "web-server"
    Environment = "production"
  }
}
```

---

## 2. Internal DSL Design Patterns

### 2.1 Language Features That Enable Internal DSLs

The expressiveness of internal DSLs depends heavily on the syntactic flexibility of the host language. The following language features facilitate internal DSL construction.

```
Language features supporting internal DSLs:

  +----------------------------+-------------------------------+
  |  Language Feature          |  Supported Languages          |
  +----------------------------+-------------------------------+
  |  Lambda / Closures         |  All modern languages         |
  |  Extension functions       |  Kotlin, Swift, Rust (trait)  |
  |  Operator overloading      |  Kotlin, Scala, Rust, Swift   |
  |  Implicit parameters       |  Scala (given/using)          |
  |  Lambda with receiver      |  Kotlin                       |
  |  Trailing closures         |  Kotlin, Swift, Ruby          |
  |  Method missing            |  Ruby (method_missing)        |
  |  Macros                    |  Rust, Elixir, Scala 3        |
  |  Result Builder            |  Swift                        |
  |  Parentheses omission      |  Ruby, Scala                  |
  +----------------------------+-------------------------------+
```

### 2.2 Building Internal DSLs in Kotlin

Kotlin is one of the most suitable languages for internal DSL construction. By combining "lambda with receiver," "extension functions," "infix functions," and "operator overloading," you can build DSLs that resemble natural language.

**Code Example 1: Kotlin Type-Safe HTML DSL**

```kotlin
// --- Type-safe HTML DSL implementation ---

// HTML element interface
interface Element {
    fun render(builder: StringBuilder, indent: String)
}

// Text node
class TextElement(val text: String) : Element {
    override fun render(builder: StringBuilder, indent: String) {
        builder.append("$indent$text\n")
    }
}

// HTML tag element
@DslMarker
annotation class HtmlTagMarker

@HtmlTagMarker
open class Tag(val name: String) : Element {
    val children = mutableListOf<Element>()
    val attributes = mutableMapOf<String, String>()

    // Unary plus operator to add text
    operator fun String.unaryPlus() {
        children.add(TextElement(this))
    }

    // Attribute setting
    fun attribute(key: String, value: String) {
        attributes[key] = value
    }

    override fun render(builder: StringBuilder, indent: String) {
        val attrs = if (attributes.isEmpty()) ""
            else " " + attributes.entries.joinToString(" ") { "${it.key}=\"${it.value}\"" }

        builder.append("$indent<$name$attrs>\n")
        children.forEach { it.render(builder, "$indent  ") }
        builder.append("$indent</$name>\n")
    }
}

// Concrete tag classes
class HTML : Tag("html") {
    fun head(init: Head.() -> Unit) = initTag(Head(), init)
    fun body(init: Body.() -> Unit) = initTag(Body(), init)
}

class Head : Tag("head") {
    fun title(init: Title.() -> Unit) = initTag(Title(), init)
    fun meta(charset: String) {
        val meta = Tag("meta")
        meta.attribute("charset", charset)
        children.add(meta)
    }
}

class Title : Tag("title")

class Body : Tag("body") {
    fun h1(init: Tag.() -> Unit) = initTag(Tag("h1"), init)
    fun h2(init: Tag.() -> Unit) = initTag(Tag("h2"), init)
    fun p(init: Tag.() -> Unit) = initTag(Tag("p"), init)
    fun div(init: Div.() -> Unit) = initTag(Div(), init)
    fun ul(init: UL.() -> Unit) = initTag(UL(), init)
    fun a(href: String, init: Tag.() -> Unit) {
        val tag = Tag("a")
        tag.attribute("href", href)
        tag.init()
        children.add(tag)
    }
}

class Div : Tag("div") {
    fun h1(init: Tag.() -> Unit) = initTag(Tag("h1"), init)
    fun p(init: Tag.() -> Unit) = initTag(Tag("p"), init)
}

class UL : Tag("ul") {
    fun li(init: Tag.() -> Unit) = initTag(Tag("li"), init)
}

// Tag initialization helper
fun <T : Tag> Tag.initTag(tag: T, init: T.() -> Unit): T {
    tag.init()
    children.add(tag)
    return tag
}

// DSL entry point
fun html(init: HTML.() -> Unit): HTML {
    val html = HTML()
    html.init()
    return html
}

// --- DSL usage example ---
val page = html {
    head {
        meta("UTF-8")
        title { +"My Page" }
    }
    body {
        h1 { +"User Dashboard" }
        div {
            h1 { +"Recent Orders" }
            p { +"You can view your order history" }
        }
        ul {
            li { +"Order #001 - In Transit" }
            li { +"Order #002 - Completed" }
            li { +"Order #003 - Processing" }
        }
        a("https://example.com/orders") {
            +"View All Orders"
        }
    }
}

fun main() {
    val sb = StringBuilder()
    page.render(sb, "")
    println(sb)
}
```

### 2.3 Internal DSLs in Ruby

Ruby is a pioneer in internal DSL construction, with numerous success stories including Rails, RSpec, and Rake.

**Code Example 2: Ruby Test DSL (RSpec-style)**

```ruby
# --- Mini test framework DSL ---

module MiniSpec
  class Context
    attr_reader :description, :examples, :before_blocks

    def initialize(description, &block)
      @description = description
      @examples = []
      @before_blocks = []
      instance_eval(&block)
    end

    def before(&block)
      @before_blocks << block
    end

    def it(description, &block)
      @examples << Example.new(description, @before_blocks, &block)
    end

    def context(description, &block)
      @examples << Context.new("  #{description}", &block)
    end

    def run(indent = "")
      puts "#{indent}#{description}"
      @examples.each do |example|
        case example
        when Example
          example.run("#{indent}  ")
        when Context
          example.run("#{indent}  ")
        end
      end
    end
  end

  class Example
    def initialize(description, before_blocks, &block)
      @description = description
      @before_blocks = before_blocks
      @block = block
    end

    def run(indent = "")
      @before_blocks.each { |b| instance_eval(&b) }
      instance_eval(&@block)
      puts "#{indent}OK: #{@description}"
    rescue => e
      puts "#{indent}FAIL: #{@description} - #{e.message}"
    end

    def expect(actual)
      Expectation.new(actual)
    end
  end

  class Expectation
    def initialize(actual)
      @actual = actual
    end

    def to(matcher)
      unless matcher.matches?(@actual)
        raise "Expected #{matcher.expected}, got #{@actual}"
      end
    end
  end

  # Matchers
  class EqMatcher
    attr_reader :expected
    def initialize(expected) = @expected = expected
    def matches?(actual) = actual == expected
  end

  class BeEmptyMatcher
    def expected = "empty collection"
    def matches?(actual) = actual.empty?
  end
end

def describe(subject, &block)
  context = MiniSpec::Context.new(subject, &block)
  context.run
end

def eq(value) = MiniSpec::EqMatcher.new(value)
def be_empty = MiniSpec::BeEmptyMatcher.new

# --- DSL usage example ---
describe "Calculator" do
  before do
    @calc = Calculator.new
  end

  it "adds two numbers" do
    expect(@calc.add(2, 3)).to eq(5)
  end

  it "subtracts two numbers" do
    expect(@calc.subtract(5, 3)).to eq(2)
  end

  context "with negative numbers" do
    it "handles negative addition" do
      expect(@calc.add(-1, -2)).to eq(-3)
    end
  end
end
```

### 2.4 Internal DSLs in Rust (Macro-Based)

Rust's internal DSLs are built using its macro system. There are two types: declarative macros (`macro_rules!`) and procedural macros.

**Code Example 3: Rust Test Matcher DSL**

```rust
// --- Declarative macro-based assertion DSL ---

macro_rules! assert_that {
    ($actual:expr, is equal_to $expected:expr) => {
        assert_eq!($actual, $expected,
            "Expected: {:?}, Actual: {:?}", $expected, $actual);
    };
    ($actual:expr, is greater_than $expected:expr) => {
        assert!($actual > $expected,
            "{:?} > {:?} is false", $actual, $expected);
    };
    ($actual:expr, is less_than $expected:expr) => {
        assert!($actual < $expected,
            "{:?} < {:?} is false", $actual, $expected);
    };
    ($actual:expr, contains $expected:expr) => {
        assert!($actual.contains(&$expected),
            "{:?} does not contain {:?}", $actual, $expected);
    };
    ($actual:expr, is empty) => {
        assert!($actual.is_empty(),
            "{:?} is not empty", $actual);
    };
    ($actual:expr, has length $expected:expr) => {
        assert_eq!($actual.len(), $expected,
            "Length is {:?} instead of {:?}", $actual.len(), $expected);
    };
}

// --- SQL builder DSL ---
macro_rules! sql {
    (SELECT $($col:ident),+ FROM $table:ident) => {
        {
            let columns = vec![$(stringify!($col)),+];
            format!("SELECT {} FROM {}", columns.join(", "), stringify!($table))
        }
    };
    (SELECT $($col:ident),+ FROM $table:ident WHERE $field:ident = $val:expr) => {
        {
            let columns = vec![$(stringify!($col)),+];
            format!("SELECT {} FROM {} WHERE {} = '{}'",
                columns.join(", "), stringify!($table),
                stringify!($field), $val)
        }
    };
}

// --- Usage example ---
#[cfg(test)]
mod tests {
    #[test]
    fn test_assertions() {
        let numbers = vec![1, 2, 3, 4, 5];

        assert_that!(numbers.len(), is equal_to 5);
        assert_that!(numbers[0], is less_than 10);
        assert_that!(numbers, contains 3);
        assert_that!(numbers, has length 5);

        let empty: Vec<i32> = vec![];
        assert_that!(empty, is empty);
    }

    #[test]
    fn test_sql_builder() {
        let query1 = sql!(SELECT name, email FROM users);
        assert_eq!(query1, "SELECT name, email FROM users");

        let query2 = sql!(SELECT name FROM users WHERE age = "25");
        assert_eq!(query2, "SELECT name FROM users WHERE age = '25'");
    }
}
```

### 2.5 Internal DSLs in Swift (Result Builder)

Swift's Result Builder (formerly Function Builder) is the foundational technology behind SwiftUI's declarative UI.

**Code Example 4: Swift Configuration DSL Using Result Builder**

```swift
// --- Configuration DSL using Result Builder ---

struct ServerConfig {
    var host: String = "localhost"
    var port: Int = 8080
    var routes: [Route] = []
    var middleware: [Middleware] = []
}

struct Route {
    let method: HTTPMethod
    let path: String
    let handler: String
}

struct Middleware {
    let name: String
    let priority: Int
}

enum HTTPMethod { case get, post, put, delete }

// Result Builder definition
@resultBuilder
struct RouteBuilder {
    static func buildBlock(_ components: Route...) -> [Route] {
        components
    }
    static func buildOptional(_ component: [Route]?) -> [Route] {
        component ?? []
    }
    static func buildEither(first component: [Route]) -> [Route] {
        component
    }
    static func buildEither(second component: [Route]) -> [Route] {
        component
    }
}

@resultBuilder
struct MiddlewareBuilder {
    static func buildBlock(_ components: Middleware...) -> [Middleware] {
        components
    }
}

@resultBuilder
struct ConfigBuilder {
    static func buildBlock(_ components: ConfigComponent...) -> ServerConfig {
        var config = ServerConfig()
        for component in components {
            component.apply(to: &config)
        }
        return config
    }
}

protocol ConfigComponent {
    func apply(to config: inout ServerConfig)
}

struct HostSetting: ConfigComponent {
    let value: String
    func apply(to config: inout ServerConfig) { config.host = value }
}

struct PortSetting: ConfigComponent {
    let value: Int
    func apply(to config: inout ServerConfig) { config.port = value }
}

struct RoutesSetting: ConfigComponent {
    let routes: [Route]
    func apply(to config: inout ServerConfig) { config.routes = routes }
}

// DSL functions
func host(_ value: String) -> HostSetting { HostSetting(value: value) }
func port(_ value: Int) -> PortSetting { PortSetting(value: value) }

func routes(@RouteBuilder _ content: () -> [Route]) -> RoutesSetting {
    RoutesSetting(routes: content())
}

func get(_ path: String, handler: String) -> Route {
    Route(method: .get, path: path, handler: handler)
}

func post(_ path: String, handler: String) -> Route {
    Route(method: .post, path: path, handler: handler)
}

// --- DSL usage example ---
@ConfigBuilder
func createConfig() -> ServerConfig {
    host("api.example.com")
    port(443)
    routes {
        get("/users", handler: "UserController.index")
        get("/users/:id", handler: "UserController.show")
        post("/users", handler: "UserController.create")
    }
}

let config = createConfig()
```

---

## 3. The Taxonomy of Metaprogramming

### 3.1 Classification of Metaprogramming

Metaprogramming is the practice of writing "programs that manipulate programs." It can be classified by the timing and method of manipulation as follows.

```
Metaprogramming classification:

  By timing:
  +-----------------------------------------------------+
  |                                                      |
  |  Compile-time        Build-time        Runtime       |
  |  (Compile-time)      (Build-time)    (Runtime)       |
  |                                                      |
  |  +------------+    +------------+  +------------+    |
  |  | Macros     |    | Code gen   |  | Reflection |    |
  |  | (Rust,     |    | (go gen,   |  | (Java,     |    |
  |  |  Elixir,   |    |  protobuf, |  |  Go,       |    |
  |  |  Scala 3)  |    |  Swagger)  |  |  C#,       |    |
  |  +------------+    +------------+  |  Python)   |    |
  |  | const eval |    | Template   |  +------------+    |
  |  | (Rust,     |    | Engine     |  | eval       |    |
  |  |  Zig)      |    | (ERB,     |  | (JS,       |    |
  |  |            |    |  Jinja2)   |  |  Python,   |    |
  |  |            |    |            |  |  Ruby)     |    |
  |  +------------+    +------------+  +------------+    |
  |                                                      |
  |  Safety: High       Safety: Medium    Safety: Low    |
  |  Performance: Best  Performance: Good Performance: Low|
  +-----------------------------------------------------+
```

### 3.2 Comparison of Macro Systems

```
Evolution of macro systems:

  Text substitution macros   Syntactic macros        Procedural macros
  (C preprocessor)          (Scheme, Elixir)        (Rust, Scala 3)
  +----------------+       +----------------+    +----------------+
  | #define MAX(a,b)|      | defmacro       |    | #[derive()]    |
  | ((a)>(b)?(a):(b))|    |   unless ...   |    | #[proc_macro]  |
  |                |       |   quote do     |    | fn my_macro    |
  | Problems:      |       |     ...        |    | (input:        |
  | - No type safety|      |   end          |    |  TokenStream)  |
  | - Hard to debug |      |                |    | -> TokenStream |
  | - Name clashes  |      | Benefits:      |    |                |
  |                |       | - Hygienic     |    | Benefits:      |
  |                |       | - Syntax-aware |    | - Type-safe    |
  |                |       |                |    | - IDE support  |
  +----------------+       +----------------+    | - Error msgs   |
                                                 +----------------+
  1970s                    1990s               2015+
```

### 3.3 Rust's Macro System in Detail

Rust provides three types of macros.

#### Declarative Macros (macro_rules!)

```rust
// --- Pattern-based macros ---

// Simplified implementation of the vec! macro
macro_rules! my_vec {
    // Empty vector
    () => {
        Vec::new()
    };
    // Element enumeration
    ($($element:expr),+ $(,)?) => {
        {
            let mut v = Vec::new();
            $(v.push($element);)+
            v
        }
    };
    // Repeat initialization [value; count]
    ($element:expr; $count:expr) => {
        vec![$element; $count]
    };
}

// HashMap literal macro
macro_rules! hashmap {
    ($($key:expr => $value:expr),* $(,)?) => {
        {
            let mut map = std::collections::HashMap::new();
            $(map.insert($key, $value);)*
            map
        }
    };
}

// Usage example
let v = my_vec![1, 2, 3, 4, 5];
let config = hashmap! {
    "host" => "localhost",
    "port" => "8080",
    "debug" => "true",
};
```

#### Procedural Macros (Derive Macro)

```rust
// --- Procedural macro definition (separate crate) ---
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

/// Derive macro for field validation
#[proc_macro_derive(Validate, attributes(validate))]
pub fn derive_validate(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let validations = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let field_validations: Vec<_> = fields.named.iter()
                    .filter_map(|f| {
                        let field_name = f.ident.as_ref()?;
                        let field_str = field_name.to_string();

                        // Look for the #[validate(not_empty)] attribute
                        for attr in &f.attrs {
                            if attr.path().is_ident("validate") {
                                return Some(quote! {
                                    if self.#field_name.is_empty() {
                                        errors.push(format!(
                                            "'{}' cannot be empty", #field_str
                                        ));
                                    }
                                });
                            }
                        }
                        None
                    })
                    .collect();
                quote! { #(#field_validations)* }
            }
            _ => quote! {},
        },
        _ => panic!("Validate only supports structs"),
    };

    let expanded = quote! {
        impl #name {
            pub fn validate(&self) -> Result<(), Vec<String>> {
                let mut errors = Vec::new();
                #validations
                if errors.is_empty() {
                    Ok(())
                } else {
                    Err(errors)
                }
            }
        }
    };

    TokenStream::from(expanded)
}

// --- Usage side ---
#[derive(Validate)]
struct UserForm {
    #[validate(not_empty)]
    name: String,
    #[validate(not_empty)]
    email: String,
    bio: String,  // No validation
}

fn main() {
    let form = UserForm {
        name: String::new(),
        email: "test@example.com".into(),
        bio: String::new(),
    };

    match form.validate() {
        Ok(()) => println!("Validation succeeded"),
        Err(errors) => {
            for error in errors {
                println!("Error: {}", error);
            }
        }
    }
}
```

### 3.4 Metaprogramming in Python

#### Decorator Pattern

```python
# --- Python: Systematic explanation of decorators ---
import functools
import time
import logging
from typing import TypeVar, Callable, Any

T = TypeVar('T')

# --- 1. Simple decorator ---
def timer(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that measures function execution time"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"Execution time of {func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

# --- 2. Parameterized decorator ---
def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """Retry decorator (with parameters)"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logging.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}"
                    )
                    if attempt < max_attempts:
                        time.sleep(delay * attempt)  # Exponential backoff
            raise last_exception
        return wrapper
    return decorator

# --- 3. Decorator composition ---
def validate_args(**validators):
    """Argument validation decorator"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Argument '{param_name}' value {value!r} is invalid"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# --- Usage example ---
@timer
@retry(max_attempts=3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
@validate_args(user_id=lambda x: isinstance(x, int) and x > 0)
def fetch_user_data(user_id: int) -> dict:
    """Fetch user data"""
    # Simulating an API call
    return {"id": user_id, "name": "Taro Tanaka"}
```

#### Metaclasses

```python
# --- ORM implementation using metaclasses ---

class Field:
    """Base class for database fields"""
    def __init__(self, field_type: str, primary_key: bool = False,
                 nullable: bool = True, max_length: int = None):
        self.field_type = field_type
        self.primary_key = primary_key
        self.nullable = nullable
        self.max_length = max_length
        self.name = None  # Set by metaclass

class IntegerField(Field):
    def __init__(self, **kwargs):
        super().__init__("INTEGER", **kwargs)

class StringField(Field):
    def __init__(self, max_length: int = 255, **kwargs):
        super().__init__("VARCHAR", max_length=max_length, **kwargs)

class ModelMeta(type):
    """Metaclass for ORM models"""
    def __new__(mcs, name, bases, namespace):
        fields = {}
        for key, value in namespace.items():
            if isinstance(value, Field):
                value.name = key
                fields[key] = value

        namespace['_fields'] = fields
        namespace['_table_name'] = name.lower() + 's'

        cls = super().__new__(mcs, name, bases, namespace)

        # Auto-generate CREATE TABLE SQL
        if fields:
            columns = []
            for fname, field in fields.items():
                col_def = f"{fname} {field.field_type}"
                if field.max_length:
                    col_def += f"({field.max_length})"
                if field.primary_key:
                    col_def += " PRIMARY KEY"
                if not field.nullable:
                    col_def += " NOT NULL"
                columns.append(col_def)
            cls._create_sql = f"CREATE TABLE {cls._table_name} (\n  " + \
                              ",\n  ".join(columns) + "\n);"

        return cls

class Model(metaclass=ModelMeta):
    """Base class for ORM models"""
    def __init__(self, **kwargs):
        for name, field in self._fields.items():
            setattr(self, name, kwargs.get(name))

    @classmethod
    def create_table_sql(cls) -> str:
        return cls._create_sql

    def insert_sql(self) -> str:
        cols = []
        vals = []
        for name in self._fields:
            value = getattr(self, name)
            if value is not None:
                cols.append(name)
                vals.append(f"'{value}'" if isinstance(value, str) else str(value))
        return (f"INSERT INTO {self._table_name} "
                f"({', '.join(cols)}) VALUES ({', '.join(vals)});")

# --- Usage example ---
class User(Model):
    id = IntegerField(primary_key=True, nullable=False)
    name = StringField(max_length=100, nullable=False)
    email = StringField(max_length=255, nullable=False)
    bio = StringField(max_length=500)

print(User.create_table_sql())
# CREATE TABLE users (
#   id INTEGER PRIMARY KEY NOT NULL,
#   name VARCHAR(100) NOT NULL,
#   email VARCHAR(255) NOT NULL,
#   bio VARCHAR(500)
# );

user = User(id=1, name="Taro Tanaka", email="tanaka@example.com")
print(user.insert_sql())
# INSERT INTO users (id, name, email) VALUES (1, 'Taro Tanaka', 'tanaka@example.com');
```

---

## 4. Elixir's Macro System

### 4.1 Macros as AST Transformation

Elixir's macros inherit the Lisp tradition and directly manipulate the AST (Abstract Syntax Tree). They achieve hygienic macros through `quote` and `unquote`.

**Code Example 5: DSL Using Elixir Macros**

```elixir
# --- Elixir: Routing DSL ---

defmodule Router do
  defmacro __using__(_opts) do
    quote do
      import Router
      Module.register_attribute(__MODULE__, :routes, accumulate: true)
      @before_compile Router
    end
  end

  # Macro for GET requests
  defmacro get(path, controller, action) do
    quote do
      @routes {:get, unquote(path), unquote(controller), unquote(action)}
    end
  end

  # Macro for POST requests
  defmacro post(path, controller, action) do
    quote do
      @routes {:post, unquote(path), unquote(controller), unquote(action)}
    end
  end

  # Generate the routing table at compile time
  defmacro __before_compile__(env) do
    routes = Module.get_attribute(env.module, :routes)

    match_clauses = for {method, path, controller, action} <- routes do
      quote do
        def match(unquote(method), unquote(path)) do
          unquote(controller).unquote(action)()
        end
      end
    end

    quote do
      unquote_splicing(match_clauses)

      def match(method, path) do
        {:error, "No matching route for #{method} #{path}"}
      end
    end
  end
end

# --- Usage example ---
defmodule MyApp.Router do
  use Router

  get  "/",        MyApp.PageController, :index
  get  "/users",   MyApp.UserController, :index
  post "/users",   MyApp.UserController, :create
  get  "/users/:id", MyApp.UserController, :show
end

# At compile time, code equivalent to the following is generated:
# def match(:get, "/") do ... end
# def match(:get, "/users") do ... end
# def match(:post, "/users") do ... end
# ...
```

---

## 5. Reflection and Introspection

### 5.1 Overview of Reflection

Reflection is the ability of a program to inspect and manipulate its own structure (types, fields, methods) at runtime.

```
Capabilities of reflection:

  +-----------------------------------------------------+
  |                 Reflection                           |
  +---------------------+-------------------------------+
  |   Introspection      |      Intercession            |
  |   (Introspection)    |      (Intercession)          |
  +---------------------+-------------------------------+
  | - Retrieve type info | - Dynamic method invocation  |
  | - List fields        | - Modify field values        |
  | - List methods       | - Dynamic proxy generation   |
  | - Get annotations    | - Dynamic class creation     |
  |                      |                              |
  | Safety: Relatively   | Safety: Requires caution     |
  |   safe               |                              |
  | Use cases:           | Use cases: DI, AOP, ORM     |
  |   Serialization,     |                              |
  |   Debugging          |                              |
  +---------------------+-------------------------------+
```

### 5.2 Reflection in Go

```go
package main

import (
    "fmt"
    "reflect"
    "strings"
)

// Serialization using struct tags
type User struct {
    ID    int    `json:"id" db:"user_id"`
    Name  string `json:"name" db:"user_name" validate:"required"`
    Email string `json:"email" db:"email" validate:"required,email"`
    Age   int    `json:"age,omitempty" db:"age" validate:"min=0,max=150"`
}

// Generic validator that reads struct tags via reflection
func Validate(v interface{}) []string {
    var errors []string
    val := reflect.ValueOf(v)
    typ := val.Type()

    for i := 0; i < val.NumField(); i++ {
        field := typ.Field(i)
        value := val.Field(i)
        tag := field.Tag.Get("validate")

        if tag == "" {
            continue
        }

        rules := strings.Split(tag, ",")
        for _, rule := range rules {
            switch {
            case rule == "required":
                if value.IsZero() {
                    errors = append(errors,
                        fmt.Sprintf("%s is required", field.Name))
                }
            case strings.HasPrefix(rule, "min="):
                // Minimum value check for numbers
                var min int
                fmt.Sscanf(rule, "min=%d", &min)
                if value.Kind() == reflect.Int && value.Int() < int64(min) {
                    errors = append(errors,
                        fmt.Sprintf("%s must be at least %d", field.Name, min))
                }
            case strings.HasPrefix(rule, "max="):
                var max int
                fmt.Sscanf(rule, "max=%d", &max)
                if value.Kind() == reflect.Int && value.Int() > int64(max) {
                    errors = append(errors,
                        fmt.Sprintf("%s must be at most %d", field.Name, max))
                }
            }
        }
    }
    return errors
}

// Auto-generate INSERT SQL via reflection
func GenerateInsertSQL(tableName string, v interface{}) string {
    val := reflect.ValueOf(v)
    typ := val.Type()

    var columns, placeholders []string
    for i := 0; i < val.NumField(); i++ {
        field := typ.Field(i)
        dbTag := field.Tag.Get("db")
        if dbTag != "" {
            columns = append(columns, dbTag)
            placeholders = append(placeholders, fmt.Sprintf("$%d", i+1))
        }
    }

    return fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)",
        tableName,
        strings.Join(columns, ", "),
        strings.Join(placeholders, ", "))
}

func main() {
    user := User{Name: "", Email: "invalid", Age: -5}
    errors := Validate(user)
    for _, e := range errors {
        fmt.Println("Validation error:", e)
    }

    sql := GenerateInsertSQL("users", User{})
    fmt.Println("Generated SQL:", sql)
    // INSERT INTO users (user_id, user_name, email, age) VALUES ($1, $2, $3, $4)
}
```

### 5.3 Reflection Comparison Table

| Feature | Java | Go | Python | C# | Rust |
|---------|------|-----|--------|-----|------|
| Type info retrieval | `Class<?>` | `reflect.Type` | `type()` | `Type` | None (compile-time only) |
| Field access | `Field` | `reflect.Value` | `getattr()` | `FieldInfo` | None |
| Method invocation | `Method.invoke()` | `Value.Call()` | `getattr()+()` | `MethodInfo.Invoke()` | None |
| Annotations | `@Annotation` | Struct tags | Decorators | `[Attribute]` | `#[derive()]` (macros) |
| Dynamic proxy | `Proxy` | None | `__getattr__` | `DispatchProxy` | None |
| Performance | Low | Low | Low | Low | N/A (zero-cost) |
| Type safety | Low | Low | Low | Low | High (macros as alternative) |

---

## 6. Code Generation

### 6.1 Code Generation Approaches

Code generation is the technique of automatically generating actual source code from templates or schemas. Unlike reflection, generated code exists as ordinary source files, achieving both type safety and runtime performance.

```
Code generation classification:

  +--------------------------------------------------+
  |              Code generation categories           |
  +--------------+---------------+------------------+
  | Schema-driven| Template      | AST manipulation |
  |              | engine        |                  |
  +--------------+---------------+------------------+
  | - Protocol   | - Go text/    | - Rust proc_macro|
  |   Buffers    |   template    | - babel plugin   |
  | - OpenAPI    | - Jinja2      | - ts-morph       |
  | - GraphQL    | - ERB/EJS     | - JavaPoet       |
  |   codegen    |               |                  |
  +--------------+---------------+------------------+
  | Input: IDL   | Input:        | Input: Source    |
  | Output: Type |   templates   |   code           |
  |   definitions| Output: Any   | Output:          |
  |              |               |   Transformed AST|
  +--------------+---------------+------------------+
```

### 6.2 Using Go Generate

```go
// Example of code generation using go:generate

//go:generate stringer -type=Color

type Color int

const (
    Red Color = iota
    Green
    Blue
    Yellow
)

// The stringer tool auto-generates:
// func (c Color) String() string { ... }
// Returns "Red", "Green", "Blue", "Yellow"
```

### 6.3 Code Generation with Protocol Buffers

```protobuf
// user.proto: Schema definition
syntax = "proto3";
package user;

option go_package = "./pb";

message User {
  int64 id = 1;
  string name = 2;
  string email = 3;
  repeated Order orders = 4;
}

message Order {
  int64 id = 1;
  double total_amount = 2;
  OrderStatus status = 3;
}

enum OrderStatus {
  PENDING = 0;
  CONFIRMED = 1;
  SHIPPED = 2;
  DELIVERED = 3;
  CANCELLED = 4;
}

service UserService {
  rpc GetUser(GetUserRequest) returns (User);
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
  rpc CreateUser(CreateUserRequest) returns (User);
}

message GetUserRequest { int64 id = 1; }
message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
}
message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
}
message CreateUserRequest {
  string name = 1;
  string email = 2;
}
```

```
Code generated from protoc:

  user.proto
      |
      +--> user.pb.go     (Go structs + serialization)
      +--> user_pb2.py    (Python classes)
      +--> User.java      (Java classes)
      +--> user.rs        (Rust structs + enums)
      +--> user.ts        (TypeScript interfaces)

  Consistent type definitions for multiple languages from a single schema
```

---

## 7. DSL Design Best Practices

### 7.1 DSL Design Principles

```
7 principles of DSL design:

  +--------------------------------------------------+
  | 1. Use domain vocabulary                          |
  |    Adopt domain experts' terminology, not         |
  |    technical jargon                               |
  +--------------------------------------------------+
  | 2. Principle of least surprise                    |
  |    DSL behavior should not defy the reader's      |
  |    intuition                                      |
  +--------------------------------------------------+
  | 3. Progressive disclosure                         |
  |    Basic usage should be easy, advanced usage      |
  |    should also be possible                        |
  +--------------------------------------------------+
  | 4. Type safety                                    |
  |    Design for detecting errors at compile time    |
  +--------------------------------------------------+
  | 5. Error message quality                          |
  |    Return meaningful errors at the DSL level      |
  +--------------------------------------------------+
  | 6. Composability                                  |
  |    DSL components should be combinable into       |
  |    larger structures                              |
  +--------------------------------------------------+
  | 7. Escape hatch                                   |
  |    Provide an escape route for cases the DSL      |
  |    cannot express                                 |
  +--------------------------------------------------+
```

### 7.2 Type-Safe DSLs with Phantom Types

Phantom Types use type parameters not for data storage but for state tracking. This allows preventing invalid operations at compile time.

```rust
// --- Managing builder state with Phantom Types ---

use std::marker::PhantomData;

// Types representing state (never actually instantiated)
struct NoHost;
struct HasHost;
struct NoPort;
struct HasPort;

// Server configuration builder (tracking state via type parameters)
struct ServerConfigBuilder<HostState, PortState> {
    host: Option<String>,
    port: Option<u16>,
    tls: bool,
    _host_state: PhantomData<HostState>,
    _port_state: PhantomData<PortState>,
}

impl ServerConfigBuilder<NoHost, NoPort> {
    fn new() -> Self {
        ServerConfigBuilder {
            host: None,
            port: None,
            tls: false,
            _host_state: PhantomData,
            _port_state: PhantomData,
        }
    }
}

impl<P> ServerConfigBuilder<NoHost, P> {
    // Setting host changes HostState from NoHost to HasHost
    fn host(self, host: &str) -> ServerConfigBuilder<HasHost, P> {
        ServerConfigBuilder {
            host: Some(host.to_string()),
            port: self.port,
            tls: self.tls,
            _host_state: PhantomData,
            _port_state: PhantomData,
        }
    }
}

impl<H> ServerConfigBuilder<H, NoPort> {
    // Setting port changes PortState from NoPort to HasPort
    fn port(self, port: u16) -> ServerConfigBuilder<H, HasPort> {
        ServerConfigBuilder {
            host: self.host,
            port: Some(port),
            tls: self.tls,
            _host_state: PhantomData,
            _port_state: PhantomData,
        }
    }
}

impl<H, P> ServerConfigBuilder<H, P> {
    fn tls(mut self, enabled: bool) -> Self {
        self.tls = enabled;
        self
    }
}

// build() can only be called when both host and port are set
impl ServerConfigBuilder<HasHost, HasPort> {
    fn build(self) -> ServerConfig {
        ServerConfig {
            host: self.host.unwrap(),
            port: self.port.unwrap(),
            tls: self.tls,
        }
    }
}

struct ServerConfig {
    host: String,
    port: u16,
    tls: bool,
}

fn main() {
    // Correct usage: compiles successfully
    let config = ServerConfigBuilder::new()
        .host("api.example.com")
        .port(443)
        .tls(true)
        .build();

    // Invalid usage: compile error
    // let bad = ServerConfigBuilder::new()
    //     .host("api.example.com")
    //     .build();  // Error: HasPort required but got NoPort
}
```

---

## 8. Risks and Countermeasures in Metaprogramming

### 8.1 Benefits and Drawbacks of Metaprogramming

```
Benefits vs drawbacks comparison:

  +----------------------+------------------------------+
  |       Benefits       |         Drawbacks            |
  +----------------------+------------------------------+
  | Boilerplate reduction| Reduced readability          |
  |                      |   (increased "magic")        |
  | DRY principle        | Debugging difficulty         |
  | DSL expressiveness   | Increased compile time       |
  | Compile-time checks  | Limited IDE support          |
  | Consistency guarantee| Higher learning curve        |
  | Perf. optimization   | Degraded error messages      |
  +----------------------+------------------------------+
```

### 8.2 Anti-Pattern 1: "Too Much Magic in the DSL"

```ruby
# --- Bad example: Excessive use of method_missing ---
class MagicQuery
  def method_missing(name, *args)
    if name.to_s.start_with?("find_by_")
      field = name.to_s.sub("find_by_", "")
      # ... dynamically generate query
    elsif name.to_s.start_with?("order_by_")
      field = name.to_s.sub("order_by_", "")
      # ... dynamically sort
    else
      super
    end
  end
end

# query.find_by_name_and_email_order_by_created_at_desc(...)
# -> Impossible to trace what is happening
# -> IDE completion does not work
# -> Typos are not detected until runtime

# --- Good example: Explicit method chaining ---
class TypedQuery
  def where(field:, value:) = # ...
  def order_by(field:, direction: :asc) = # ...
  def limit(n) = # ...
end

# query.where(field: :name, value: "Tanaka")
#       .order_by(field: :created_at, direction: :desc)
#       .limit(10)
# -> IDE completion works
# -> Typos are detected at compile/syntax check time
```

**Lesson:** Balance expressiveness and type safety in your DSL. Prefer "explicit structure" over "implicit rules."

### 8.3 Anti-Pattern 2: "Unnecessary Use of Macros"

```rust
// --- Bad example: Using macros when a function would suffice ---
macro_rules! add_numbers {
    ($a:expr, $b:expr) => {
        $a + $b
    };
}

let sum = add_numbers!(3, 5);  // No need for a macro

// --- Good example: Using a plain function ---
fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

let sum = add_numbers(3, 5);  // Simple and type-safe

// --- Cases where macros are justified ---
// 1. Variadic arguments are needed
macro_rules! sum_all {
    ($($x:expr),+) => {
        0 $(+ $x)+
    };
}
let total = sum_all!(1, 2, 3, 4, 5);

// 2. Code needs to be generated at compile time
// 3. Syntax extension is needed (DSL)
// 4. Significant boilerplate reduction
```

**Lesson:** Always first consider "Can this be solved with regular functions, generics, or traits?" Macros should be the last resort.

### 8.4 Anti-Pattern 3: "Abuse of eval"

```python
# --- Bad example: Dynamic code execution via eval ---
def calculate(expression: str) -> float:
    return eval(expression)  # Security risk!

# Passing user input directly to eval creates an arbitrary code execution vulnerability
# calculate("__import__('os').system('rm -rf /')")

# --- Good example: Using a parser ---
import ast
import operator

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def safe_calculate(expression: str) -> float:
    """Safe expression evaluator"""
    tree = ast.parse(expression, mode='eval')

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
            left = eval_node(node.left)
            right = eval_node(node.right)
            return SAFE_OPERATORStype(node.op)
        else:
            raise ValueError(f"Unsafe expression: {ast.dump(node)}")

    return eval_node(tree)

print(safe_calculate("(3 + 5) * 2"))  # 16.0
```

**Lesson:** In principle, do not use `eval` in production code. Passing user input to `eval` directly leads to arbitrary code execution vulnerabilities.

---

## 9. Metaprogramming in TypeScript

### 9.1 Decorators (Experimental / Stage 3)

TypeScript decorators are based on the ES Decorators proposal (Stage 3).

```typescript
// --- TypeScript class decorators ---

// Method decorator: Logging
function Log(
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
): PropertyDescriptor {
    const originalMethod = descriptor.value;

    descriptor.value = function (...args: any[]) {
        console.log(`[LOG] ${propertyKey} called: args=${JSON.stringify(args)}`);
        const result = originalMethod.apply(this, args);
        console.log(`[LOG] ${propertyKey} return value: ${JSON.stringify(result)}`);
        return result;
    };

    return descriptor;
}

// Method decorator: Performance measurement
function Measure(
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
): PropertyDescriptor {
    const originalMethod = descriptor.value;

    descriptor.value = async function (...args: any[]) {
        const start = performance.now();
        const result = await originalMethod.apply(this, args);
        const elapsed = performance.now() - start;
        console.log(`[PERF] ${propertyKey}: ${elapsed.toFixed(2)}ms`);
        return result;
    };

    return descriptor;
}

// Property decorator: Validation
function MinLength(min: number) {
    return function (target: any, propertyKey: string) {
        let value: string;

        Object.defineProperty(target, propertyKey, {
            get: () => value,
            set: (newValue: string) => {
                if (newValue.length < min) {
                    throw new Error(
                        `${propertyKey} must be at least ${min} characters`
                    );
                }
                value = newValue;
            },
        });
    };
}

// Usage example
class UserService {
    @MinLength(2)
    username: string = "";

    @Log
    @Measure
    async findUser(id: number): Promise<User | null> {
        // Simulating a DB search
        return { id, name: "Tanaka", email: "tanaka@example.com" };
    }
}
```

### 9.2 Type-Level Programming

TypeScript's type system is Turing-complete, enabling metaprogramming at the type level.

```typescript
// --- TypeScript type-level programming ---

// Conditional Types
type IsString<T> = T extends string ? true : false;

type A = IsString<string>;  // true
type B = IsString<number>;  // false

// Mapped Types
type Readonly<T> = { readonly [K in keyof T]: T[K] };
type Optional<T> = { [K in keyof T]?: T[K] };
type Required<T> = { [K in keyof T]-?: T[K] };

// Template Literal Types
type HTTPMethod = "GET" | "POST" | "PUT" | "DELETE";
type APIPath = `/api/${string}`;
type Endpoint = `${HTTPMethod} ${APIPath}`;
// "GET /api/users" | "POST /api/users" | ... is of type Endpoint

// Recursive Types (Deep Readonly)
type DeepReadonly<T> = {
    readonly [K in keyof T]: T[K] extends object
        ? DeepReadonly<T[K]>
        : T[K];
};

// Auto-generation of path parameter types
type PathParams<T extends string> =
    T extends `${string}:${infer Param}/${infer Rest}`
        ? { [K in Param | keyof PathParams<Rest>]: string }
        : T extends `${string}:${infer Param}`
            ? { [K in Param]: string }
            : {};

// Extract { userId: string; postId: string } from "/users/:userId/posts/:postId"
type UserPostParams = PathParams<"/users/:userId/posts/:postId">;

// Type-safe API client
interface APIRoutes {
    "GET /api/users": { response: User[]; params: {} };
    "GET /api/users/:id": { response: User; params: { id: string } };
    "POST /api/users": { response: User; params: {}; body: CreateUserDTO };
}

async function apiCall<T extends keyof APIRoutes>(
    endpoint: T,
    ...args: APIRoutes[T] extends { body: infer B }
        ? [params: APIRoutes[T]["params"], body: B]
        : [params: APIRoutes[T]["params"]]
): Promise<APIRoutes[T]["response"]> {
    // Implementation
    throw new Error("not implemented");
}

// Type-safe invocations
// apiCall("GET /api/users", {});
// apiCall("GET /api/users/:id", { id: "123" });
// apiCall("POST /api/users", {}, { name: "Tanaka", email: "..." });
```

---

## 10. Exercises

### 10.1 Beginner: Understanding Concepts

**Exercise 1:** Classify each of the following DSLs as "external DSL" or "internal DSL."
1. SQL
2. SwiftUI
3. Terraform HCL
4. RSpec
5. GraphQL
6. Gradle Kotlin DSL
7. Protocol Buffers

**Exercise 2:** Classify the following metaprogramming techniques by their timing of application (compile-time, build-time, or runtime).
1. Rust `derive` macros
2. Python decorators
3. Go `go generate`
4. Java reflection
5. Elixir macros
6. Protocol Buffers code generation

**Exercise 3:** Predict the output of the following code.

```python
def trace(func):
    def wrapper(*args):
        print(f"Call: {func.__name__}({args})")
        result = func(*args)
        print(f"Return: {result}")
        return result
    return wrapper

@trace
def add(a, b):
    return a + b

@trace
def multiply(a, b):
    return a * b

result = multiply(add(2, 3), 4)
```

### 10.2 Intermediate: Implementation

**Exercise 4:** Implement a configuration file DSL in any language that satisfies the following requirements.

```
Requirements:
- Can express nested configuration structures
- Is type-safe (type checking for strings, numbers, booleans)
- Supports environment variable references (${ENV_VAR} notation)
- Has validation functionality

Usage example:
config {
    server {
        host = "localhost"
        port = 8080
        tls = true
    }
    database {
        url = "${DATABASE_URL}"
        pool_size = 10
    }
}
```

**Exercise 5:** Implement the following JSON literal DSL using Rust declarative macros (`macro_rules!`).

```rust
let data = json!({
    "name": "Taro Tanaka",
    "age": 30,
    "hobbies": ["reading", "programming"],
    "address": {
        "city": "Tokyo",
        "zip": "100-0001"
    }
});
```

### 10.3 Advanced: Design and Analysis

**Exercise 6:** Compare the following three metaprogramming approaches and discuss the optimal use cases for each.

1. Rust procedural macros
2. Python metaclasses
3. TypeScript type-level programming

Comparison criteria:
- Type safety
- Runtime overhead
- Ease of debugging
- Expressiveness
- Learning cost

**Exercise 7:** Choose one domain from a real project and design a DSL specialized for that domain (implementation not required).

Deliverables:
- Description of the domain
- DSL grammar specification (BNF notation)
- Usage examples (at least 3)
- Rationale for choosing internal vs external DSL
- Comparison with existing DSLs (SQL, GraphQL, etc.)

---

## 11. Advanced Topics

### 11.1 Effect Systems and DSLs

An effect system is a mechanism that tracks function side effects through types. It is drawing attention as a next-generation foundation technology for DSLs.

```
Concept of effect systems:

  Traditional functions:
    fn read_file(path: &str) -> String
    // Cannot tell from the type whether it has side effects (I/O)

  With effect system:
    fn read_file(path: &str) -> String / IO + Error
    // Side effects are explicitly part of the type

  Application to DSLs:
  +--------------------------------------+
  | Effect = Making DSL operations       |
  |          first-class citizens        |
  |                                      |
  | database {                           |
  |   let user = query(User, id: 1)     |
  |   // ^ Database effect               |
  |   log("User retrieved: ${user.name}")|
  |   // ^ Log effect                    |
  | }                                    |
  | // Outside the database block,       |
  | // Database effect cannot be used    |
  +--------------------------------------+
```

### 11.2 Language Workbenches

A language workbench is a development environment that provides unified support for DSL definition, implementation, and IDE integration.

Representative examples:
- **JetBrains MPS**: DSL construction via projectional editing. No syntax ambiguity since it is not text-based.
- **Xtext**: Automatically builds parsers, editors, and code generators from grammar definitions on the Eclipse platform.
- **Spoofax**: A language definition framework. Grammar, type systems, and transformation rules are described declaratively.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important aspect. Understanding deepens not only through theory but also by actually writing and running code.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before proceeding to the next step.

### Q3: How is this applied in practice?

The knowledge from this topic is frequently applied in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 12. Summary

### 12.1 Comprehensive Comparison of Techniques

| Technique | Timing | Representative Languages | Type Safety | Performance | Use Cases |
|-----------|--------|-------------------------|-------------|-------------|-----------|
| Declarative macros | Compile-time | Rust, Elixir | Medium | Optimal | DSL, boilerplate reduction |
| Procedural macros | Compile-time | Rust, Scala 3 | High | Optimal | derive, attribute macros |
| Decorators | Runtime | Python, TS | Low | Low | AOP, logging, auth |
| Metaclasses | Runtime | Python | Low | Low | ORM, validation |
| Reflection | Runtime | Java, Go, C# | Low | Low | Serialization, DI |
| Code generation | Build-time | Go, protobuf | High | Optimal | API definitions, type generation |
| Result Builder | Compile-time | Swift | High | Optimal | SwiftUI, declarative DSL |
| Type-level | Compile-time | TypeScript | Highest | Optimal | Type-safe API, path inference |

### 12.2 Decision Flowchart

```
Choosing a metaprogramming technique:

  "I want to reduce boilerplate"
     |
     +- Can it be resolved at compile time?
     |    +- Yes -> Macros or derive (Rust)
     |    |         Result Builder (Swift)
     |    |         Type-level (TypeScript)
     |    +- No -> Proceed to runtime techniques
     |
     +- Do you need common definitions across multiple languages?
     |    +- Yes -> Code generation (protobuf, OpenAPI)
     |
     +- Cross-cutting concerns (logging, auth, metrics)?
     |    +- Yes -> Decorators (Python, TS)
     |             Aspects (Java / Spring AOP)
     |
     +- Do you need dynamic structural inspection?
          +- Yes -> Reflection (Java, Go, C#)
```

---

## 13. FAQ (Frequently Asked Questions)

### Q1: Should I choose an internal DSL or an external DSL?

**A:** It depends on the project situation. Internal DSLs have lower development costs and can directly leverage the host language's tools (IDE, debugger, test framework), making them the first choice in most cases. External DSLs are chosen when non-engineers (domain experts) need to write the DSL directly, or when a standard DSL already exists (SQL, GraphQL). External DSLs require building the parser, error messages, and all tooling from scratch, resulting in higher development and maintenance costs.

### Q2: At what project scale should metaprogramming be introduced?

**A:** The decision to introduce metaprogramming should be based not on scale but on the frequency of "repeating patterns." If the same boilerplate appears in three or more places with an expectation of growth, it is worth considering metaprogramming. However, the prerequisite is that all team members can understand the technique. "Macros only I can understand" become technical debt for the team.

### Q3: How do Rust macros differ from C macros?

**A:** They are fundamentally different. C macros (`#define`) are text substitutions executed before type checking. They lack type safety, suffer from name collisions (unintended variable name substitution), and are extremely difficult to debug. Rust macros, on the other hand, operate on syntax trees (token streams). Hygienic macros prevent name collisions, and type checking occurs after macro expansion, preserving type safety. Furthermore, procedural macros allow customization of compiler error messages.

### Q4: How practical is TypeScript's type-level programming?

**A:** It is highly practical for library type definitions. Modern TypeScript libraries such as tRPC, Zod, and Prisma actively leverage type-level programming. However, heavy use of complex type-level programming in application code results in cryptic type error messages and slower compilation. In practice, the ideal division is: "library authors use it aggressively, and library consumers transparently benefit from it."

---

## Recommended Next Reading


---

## References

1. Fowler, M. "Domain-Specific Languages." Addison-Wesley, 2010. - A seminal work comprehensively covering DSL design patterns. Systematizes the classification and implementation techniques of internal and external DSLs.
2. Rust Reference. "Macros." The Rust Programming Language. - The official reference for Rust's declarative and procedural macros. Essential for understanding the design philosophy of the macro system.
3. Odersky, M. et al. "Scala 3 Reference: Metaprogramming." EPFL. - Explains Scala 3's inline metaprogramming and macro system. Details the concepts of quotes and splices.
4. Van Rossum, G. "PEP 3119 -- Introducing Abstract Base Classes." Python Software Foundation. - Explains the design rationale of Python's metaclass protocol. Records the discussion leading to the introduction of ABC metaclasses.
5. Thomas, D. "Metaprogramming Ruby." Pragmatic Bookshelf, 2nd Edition, 2014. - A practical guide to Ruby metaprogramming techniques (method_missing, class_eval, define_method, etc.).

---

## Glossary

| Term | Description |
|------|-------------|
| DSL (Domain-Specific Language) | A language specialized for a particular problem domain |
| GPL (General-Purpose Language) | A general-purpose programming language |
| Internal DSL (Internal/Embedded DSL) | A DSL built using the host language's syntax |
| External DSL (External DSL) | A DSL with its own parser and syntax |
| Metaprogramming | Writing programs that manipulate or generate programs |
| Macro | A mechanism for generating or transforming code at compile time |
| Hygienic Macro | A macro system that prevents name collisions |
| Reflection | The ability to inspect and manipulate type information at runtime |
| Decorator | A metaprogramming pattern that transforms functions or classes |
| Metaclass | A class that generates classes |
| Phantom Type | A technique that tracks state using type parameters not used for data storage |
| Result Builder | Swift's feature for building declarative DSLs |
| Code Generation | Automatically generating source code from schemas or templates |
| AST (Abstract Syntax Tree) | A tree structure representing the structure of source code |
| Language Workbench | A development environment for unified DSL definition, implementation, and IDE integration |
