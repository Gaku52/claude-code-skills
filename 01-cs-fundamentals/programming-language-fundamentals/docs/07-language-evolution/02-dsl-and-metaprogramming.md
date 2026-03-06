# DSL とメタプログラミング

> DSL（Domain-Specific Language）は「特定の問題領域に特化した言語」であり、メタプログラミングは「プログラムでプログラムを生成・変換する」技術である。この2つは表裏一体の関係にあり、表現力の高いソフトウェアを構築するための重要な手法である。本章では DSL の設計原則からメタプログラミングの各手法まで、理論と実践の両面から解説する。

## この章で学ぶこと

- [ ] 内部DSLと外部DSLの違い、設計上のトレードオフを理解する
- [ ] メタプログラミングの主要手法（マクロ・リフレクション・コード生成）を把握する
- [ ] 各言語での DSL 構築テクニック（Kotlin, Ruby, Scala, Swift, Rust）を実装できる
- [ ] マクロシステムの種類（テキスト置換・構文マクロ・手続き型マクロ）を区別できる
- [ ] メタプログラミングの適切な適用範囲と危険性を判断できる
- [ ] 型安全な DSL の設計パターン（Builder, Phantom Type）を活用できる

---

## 1. DSL の基礎理論

### 1.1 DSLとは何か

DSL（Domain-Specific Language）は、特定の問題領域（ドメイン）に最適化された言語である。汎用プログラミング言語（GPL: General-Purpose Language）がチューリング完全であらゆる計算を表現できるのに対し、DSLは意図的に表現力を制限することで、特定の領域での生産性と安全性を最大化する。

```
DSL vs GPL の位置づけ:

  表現力の範囲
  ←────────────────────────────────────────→
  狭い（特化）                      広い（汎用）

  ┌────┐  ┌─────┐  ┌──────┐  ┌──────────┐
  │正規 │  │ SQL │  │Terraform│ │ Python   │
  │表現 │  │     │  │ / HCL │  │ Java     │
  └────┘  └─────┘  └──────┘  │ Rust     │
                              │ etc.     │
  DSL                         └──────────┘
  ・学習コスト: 低               GPL
  ・表現力: 限定的               ・学習コスト: 高
  ・安全性: 高                   ・表現力: 無制限
  ・最適化: 容易                 ・安全性: 言語依存
                                ・最適化: 困難
```

### 1.2 DSLの分類体系

```
DSLの分類:

  ┌──────────────────────────────────────────────────┐
  │                    DSL                           │
  ├──────────────────────┬─────────────────────────┤
  │    外部DSL           │       内部DSL            │
  │  (External DSL)      │   (Internal DSL /       │
  │                      │    Embedded DSL)         │
  ├──────────────────────┼─────────────────────────┤
  │ ・独自のパーサーが必要  │ ・ホスト言語の構文を利用  │
  │ ・独自の構文を定義     │ ・言語機能で DSL を構築   │
  │ ・ツールが必要         │ ・IDE サポートが自然      │
  │                      │                         │
  │ 例:                   │ 例:                      │
  │ ・SQL                │ ・RSpec (Ruby)           │
  │ ・HTML/CSS           │ ・kotlinx.html (Kotlin)  │
  │ ・正規表現            │ ・SwiftUI (Swift)        │
  │ ・GraphQL            │ ・Ktor routing (Kotlin)  │
  │ ・Terraform HCL      │ ・ScalaTest (Scala)      │
  │ ・Protocol Buffers   │ ・Diesel (Rust)          │
  └──────────────────────┴─────────────────────────┘
```

### 1.3 有名な外部DSLの解剖

#### SQL: データ問い合わせのDSL

SQLは1970年代に E.F. Codd の関係モデルに基づいて設計された、最も成功した外部DSLの一つである。

```sql
-- SQL: 宣言的なデータ問い合わせ
-- 「何を取得するか」を記述し、「どう取得するか」はDBエンジンが決定

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

-- これを汎用言語で書くと数十行のループとフィルタリングが必要
```

#### GraphQL: API問い合わせのDSL

```graphql
# GraphQL: クライアント駆動のAPI問い合わせ
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

#### Terraform HCL: インフラ定義のDSL

```hcl
# Terraform: インフラをコードとして宣言
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

## 2. 内部DSLの設計パターン

### 2.1 内部DSLを可能にする言語機能

内部DSLの表現力は、ホスト言語の構文的柔軟性に大きく依存する。以下の言語機能が内部DSLの構築を容易にする。

```
内部DSLを支える言語機能:

  ┌────────────────────────────────────────────────────────┐
  │  言語機能              │  対応言語                      │
  ├────────────────────────┼───────────────────────────────┤
  │  ラムダ式 / クロージャ   │  全モダン言語                  │
  │  拡張関数              │  Kotlin, Swift, Rust (trait)  │
  │  演算子オーバーロード    │  Kotlin, Scala, Rust, Swift   │
  │  暗黙の引数            │  Scala (given/using)          │
  │  レシーバ付きラムダ      │  Kotlin                      │
  │  トレイリングクロージャ   │  Kotlin, Swift, Ruby          │
  │  メソッドミッシング      │  Ruby (method_missing)        │
  │  マクロ                │  Rust, Elixir, Scala 3        │
  │  Result Builder        │  Swift                        │
  │  括弧省略              │  Ruby, Scala                  │
  └────────────────────────┴───────────────────────────────┘
```

### 2.2 Kotlin での内部DSL構築

Kotlin は内部DSL構築に最も適した言語の一つである。「レシーバ付きラムダ」「拡張関数」「中置関数」「演算子オーバーロード」を組み合わせることで、自然言語に近いDSLを構築できる。

**コード例1: Kotlin の型安全HTML DSL**

```kotlin
// --- 型安全な HTML DSL の実装 ---

// HTML要素のインターフェース
interface Element {
    fun render(builder: StringBuilder, indent: String)
}

// テキストノード
class TextElement(val text: String) : Element {
    override fun render(builder: StringBuilder, indent: String) {
        builder.append("$indent$text\n")
    }
}

// HTMLタグ要素
@DslMarker
annotation class HtmlTagMarker

@HtmlTagMarker
open class Tag(val name: String) : Element {
    val children = mutableListOf<Element>()
    val attributes = mutableMapOf<String, String>()

    // テキストを追加する単項プラス演算子
    operator fun String.unaryPlus() {
        children.add(TextElement(this))
    }

    // 属性設定
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

// 具体的なタグクラス
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

// タグ初期化のヘルパー
fun <T : Tag> Tag.initTag(tag: T, init: T.() -> Unit): T {
    tag.init()
    children.add(tag)
    return tag
}

// DSLのエントリポイント
fun html(init: HTML.() -> Unit): HTML {
    val html = HTML()
    html.init()
    return html
}

// --- DSLの使用例 ---
val page = html {
    head {
        meta("UTF-8")
        title { +"マイページ" }
    }
    body {
        h1 { +"ユーザーダッシュボード" }
        div {
            h1 { +"最近の注文" }
            p { +"注文履歴を確認できます" }
        }
        ul {
            li { +"注文 #001 - 配送中" }
            li { +"注文 #002 - 完了" }
            li { +"注文 #003 - 処理中" }
        }
        a("https://example.com/orders") {
            +"全ての注文を見る"
        }
    }
}

fun main() {
    val sb = StringBuilder()
    page.render(sb, "")
    println(sb)
}
```

### 2.3 Ruby での内部DSL

Ruby は内部DSL構築のパイオニアであり、Rails, RSpec, Rake など数多くの成功事例がある。

**コード例2: Ruby のテストDSL（RSpec風）**

```ruby
# --- ミニテストフレームワークDSL ---

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

  # マッチャー
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

# --- DSLの使用例 ---
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

### 2.4 Rust での内部DSL（マクロベース）

Rust の内部DSLはマクロシステムを活用して構築される。宣言的マクロ（`macro_rules!`）と手続き型マクロの2種類がある。

**コード例3: Rust のテスト用マッチャーDSL**

```rust
// --- 宣言的マクロによるアサーションDSL ---

macro_rules! assert_that {
    ($actual:expr, is equal_to $expected:expr) => {
        assert_eq!($actual, $expected,
            "期待値: {:?}, 実際値: {:?}", $expected, $actual);
    };
    ($actual:expr, is greater_than $expected:expr) => {
        assert!($actual > $expected,
            "{:?} > {:?} が偽", $actual, $expected);
    };
    ($actual:expr, is less_than $expected:expr) => {
        assert!($actual < $expected,
            "{:?} < {:?} が偽", $actual, $expected);
    };
    ($actual:expr, contains $expected:expr) => {
        assert!($actual.contains(&$expected),
            "{:?} に {:?} が含まれていません", $actual, $expected);
    };
    ($actual:expr, is empty) => {
        assert!($actual.is_empty(),
            "{:?} は空ではありません", $actual);
    };
    ($actual:expr, has length $expected:expr) => {
        assert_eq!($actual.len(), $expected,
            "長さが {:?} ではなく {:?}", $expected, $actual.len());
    };
}

// --- SQLビルダーDSL ---
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

// --- 使用例 ---
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

### 2.5 Swift での内部DSL（Result Builder）

Swift の Result Builder（旧称 Function Builder）は SwiftUI の宣言的UIの基盤技術である。

**コード例4: Swift の Result Builder による設定DSL**

```swift
// --- Result Builder による設定DSL ---

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

// Result Builder定義
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

// DSL関数
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

// --- DSLの使用例 ---
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

## 3. メタプログラミングの体系

### 3.1 メタプログラミングの分類

メタプログラミングとは「プログラムを操作するプログラム」を書くことである。操作のタイミングと手法により、以下のように分類される。

```
メタプログラミングの分類:

  タイミング別:
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  コンパイル時          ビルド時          実行時        │
  │  (Compile-time)      (Build-time)    (Runtime)     │
  │                                                     │
  │  ┌────────────┐    ┌────────────┐  ┌────────────┐ │
  │  │ マクロ      │    │ コード生成   │  │リフレクション│ │
  │  │ (Rust,     │    │ (go gen,   │  │ (Java,     │ │
  │  │  Elixir,   │    │  protobuf, │  │  Go,       │ │
  │  │  Scala 3)  │    │  Swagger)  │  │  C#,       │ │
  │  ├────────────┤    ├────────────┤  │  Python)   │ │
  │  │ const eval │    │ Template   │  ├────────────┤ │
  │  │ (Rust,     │    │ Engine     │  │ eval       │ │
  │  │  Zig)      │    │ (ERB,     │  │ (JS,       │ │
  │  │            │    │  Jinja2)   │  │  Python,   │ │
  │  │            │    │            │  │  Ruby)     │ │
  │  └────────────┘    └────────────┘  └────────────┘ │
  │                                                     │
  │  安全性: 高           安全性: 中        安全性: 低    │
  │  パフォーマンス: 最適  パフォーマンス: 良  パフォーマンス: 低│
  └─────────────────────────────────────────────────────┘
```

### 3.2 マクロシステムの比較

```
マクロシステムの進化:

  テキスト置換マクロ        構文マクロ           手続き型マクロ
  (C プリプロセッサ)       (Scheme, Elixir)    (Rust, Scala 3)
  ┌──────────────┐       ┌──────────────┐    ┌──────────────┐
  │ #define MAX(a,b)│     │ defmacro     │    │ #[derive()]  │
  │ ((a)>(b)?(a):(b))│   │   unless ... │    │ #[proc_macro]│
  │              │       │   quote do   │    │ fn my_macro  │
  │ 問題:        │       │     ...      │    │ (input:      │
  │ ・型安全性なし │       │   end        │    │  TokenStream)│
  │ ・デバッグ困難 │       │              │    │ -> TokenStream│
  │ ・名前衝突    │       │ 利点:        │    │              │
  │              │       │ ・衛生的      │    │ 利点:        │
  │              │       │ ・構文認識    │    │ ・型安全     │
  │              │       │              │    │ ・IDE連携    │
  └──────────────┘       └──────────────┘    │ ・エラー表示 │
                                             └──────────────┘
  1970s                  1990s               2015+
```

### 3.3 Rustのマクロシステム詳解

Rust は3種類のマクロを提供する。

#### 宣言的マクロ（macro_rules!）

```rust
// --- パターンベースのマクロ ---

// vec! マクロの簡略化実装
macro_rules! my_vec {
    // 空のベクタ
    () => {
        Vec::new()
    };
    // 要素列挙
    ($($element:expr),+ $(,)?) => {
        {
            let mut v = Vec::new();
            $(v.push($element);)+
            v
        }
    };
    // 繰り返し初期化  [value; count]
    ($element:expr; $count:expr) => {
        vec![$element; $count]
    };
}

// HashMap リテラルマクロ
macro_rules! hashmap {
    ($($key:expr => $value:expr),* $(,)?) => {
        {
            let mut map = std::collections::HashMap::new();
            $(map.insert($key, $value);)*
            map
        }
    };
}

// 使用例
let v = my_vec![1, 2, 3, 4, 5];
let config = hashmap! {
    "host" => "localhost",
    "port" => "8080",
    "debug" => "true",
};
```

#### 手続き型マクロ（Derive Macro）

```rust
// --- 手続き型マクロの定義（別クレート） ---
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

/// フィールドのバリデーション用 derive マクロ
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

                        // #[validate(not_empty)] 属性を探す
                        for attr in &f.attrs {
                            if attr.path().is_ident("validate") {
                                return Some(quote! {
                                    if self.#field_name.is_empty() {
                                        errors.push(format!(
                                            "'{}' は空にできません", #field_str
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
        _ => panic!("Validate は構造体のみサポート"),
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

// --- 使用側 ---
#[derive(Validate)]
struct UserForm {
    #[validate(not_empty)]
    name: String,
    #[validate(not_empty)]
    email: String,
    bio: String,  // バリデーションなし
}

fn main() {
    let form = UserForm {
        name: String::new(),
        email: "test@example.com".into(),
        bio: String::new(),
    };

    match form.validate() {
        Ok(()) => println!("バリデーション成功"),
        Err(errors) => {
            for error in errors {
                println!("エラー: {}", error);
            }
        }
    }
}
```

### 3.4 Pythonのメタプログラミング

#### デコレータパターン

```python
# --- Python: デコレータの体系的な解説 ---
import functools
import time
import logging
from typing import TypeVar, Callable, Any

T = TypeVar('T')

# --- 1. シンプルなデコレータ ---
def timer(func: Callable[..., T]) -> Callable[..., T]:
    """関数の実行時間を計測するデコレータ"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__} の実行時間: {elapsed:.4f}秒")
        return result
    return wrapper

# --- 2. パラメータ付きデコレータ ---
def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """リトライデコレータ（パラメータ付き）"""
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
                        f"{func.__name__} 試行 {attempt}/{max_attempts} 失敗: {e}"
                    )
                    if attempt < max_attempts:
                        time.sleep(delay * attempt)  # 指数バックオフ
            raise last_exception
        return wrapper
    return decorator

# --- 3. デコレータの合成 ---
def validate_args(**validators):
    """引数バリデーションデコレータ"""
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
                            f"引数 '{param_name}' の値 {value!r} は無効です"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# --- 使用例 ---
@timer
@retry(max_attempts=3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
@validate_args(user_id=lambda x: isinstance(x, int) and x > 0)
def fetch_user_data(user_id: int) -> dict:
    """ユーザーデータを取得する"""
    # API呼び出しのシミュレーション
    return {"id": user_id, "name": "田中太郎"}
```

#### メタクラス

```python
# --- メタクラスによるORMの実装 ---

class Field:
    """データベースフィールドの基底クラス"""
    def __init__(self, field_type: str, primary_key: bool = False,
                 nullable: bool = True, max_length: int = None):
        self.field_type = field_type
        self.primary_key = primary_key
        self.nullable = nullable
        self.max_length = max_length
        self.name = None  # メタクラスで設定

class IntegerField(Field):
    def __init__(self, **kwargs):
        super().__init__("INTEGER", **kwargs)

class StringField(Field):
    def __init__(self, max_length: int = 255, **kwargs):
        super().__init__("VARCHAR", max_length=max_length, **kwargs)

class ModelMeta(type):
    """ORMモデルのメタクラス"""
    def __new__(mcs, name, bases, namespace):
        fields = {}
        for key, value in namespace.items():
            if isinstance(value, Field):
                value.name = key
                fields[key] = value

        namespace['_fields'] = fields
        namespace['_table_name'] = name.lower() + 's'

        cls = super().__new__(mcs, name, bases, namespace)

        # CREATE TABLE SQL を自動生成
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
    """ORMモデルの基底クラス"""
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

# --- 使用例 ---
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

user = User(id=1, name="田中太郎", email="tanaka@example.com")
print(user.insert_sql())
# INSERT INTO users (id, name, email) VALUES (1, '田中太郎', 'tanaka@example.com');
```

---

## 4. Elixir のマクロシステム

### 4.1 AST変換としてのマクロ

Elixir のマクロは Lisp の伝統を受け継ぎ、AST（抽象構文木）を直接操作する。`quote` と `unquote` によって衛生的なマクロを実現している。

**コード例5: Elixir のマクロによるDSL**

```elixir
# --- Elixir: ルーティングDSL ---

defmodule Router do
  defmacro __using__(_opts) do
    quote do
      import Router
      Module.register_attribute(__MODULE__, :routes, accumulate: true)
      @before_compile Router
    end
  end

  # GET リクエストのマクロ
  defmacro get(path, controller, action) do
    quote do
      @routes {:get, unquote(path), unquote(controller), unquote(action)}
    end
  end

  # POST リクエストのマクロ
  defmacro post(path, controller, action) do
    quote do
      @routes {:post, unquote(path), unquote(controller), unquote(action)}
    end
  end

  # コンパイル時にルーティングテーブルを生成
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
        {:error, "#{method} #{path} にマッチするルートがありません"}
      end
    end
  end
end

# --- 使用例 ---
defmodule MyApp.Router do
  use Router

  get  "/",        MyApp.PageController, :index
  get  "/users",   MyApp.UserController, :index
  post "/users",   MyApp.UserController, :create
  get  "/users/:id", MyApp.UserController, :show
end

# コンパイル時に以下と等価なコードが生成される:
# def match(:get, "/") do ... end
# def match(:get, "/users") do ... end
# def match(:post, "/users") do ... end
# ...
```

---

## 5. リフレクションとイントロスペクション

### 5.1 リフレクションの概要

リフレクション（Reflection）とは、プログラムが実行時に自身の構造（型、フィールド、メソッド）を検査・操作する能力である。

```
リフレクションの機能:

  ┌─────────────────────────────────────────────────────┐
  │                 リフレクション                        │
  ├─────────────────────┬───────────────────────────────┤
  │   イントロスペクション  │      インターセッション        │
  │   (Introspection)     │      (Intercession)         │
  ├─────────────────────┼───────────────────────────────┤
  │ ・型情報の取得        │ ・動的メソッド呼び出し          │
  │ ・フィールド一覧      │ ・フィールド値の変更           │
  │ ・メソッド一覧        │ ・動的プロキシ生成             │
  │ ・アノテーション取得   │ ・クラス動的生成               │
  │                      │                              │
  │ 安全性: 比較的安全     │ 安全性: 注意が必要             │
  │ 用途: シリアライズ、   │ 用途: DI、AOP、ORM           │
  │       デバッグ         │                              │
  └─────────────────────┴───────────────────────────────┘
```

### 5.2 Go のリフレクション

```go
package main

import (
    "fmt"
    "reflect"
    "strings"
)

// 構造体タグを使ったシリアライズ
type User struct {
    ID    int    `json:"id" db:"user_id"`
    Name  string `json:"name" db:"user_name" validate:"required"`
    Email string `json:"email" db:"email" validate:"required,email"`
    Age   int    `json:"age,omitempty" db:"age" validate:"min=0,max=150"`
}

// リフレクションで構造体タグを読み取る汎用バリデータ
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
                        fmt.Sprintf("%s は必須です", field.Name))
                }
            case strings.HasPrefix(rule, "min="):
                // 数値の最小値チェック
                var min int
                fmt.Sscanf(rule, "min=%d", &min)
                if value.Kind() == reflect.Int && value.Int() < int64(min) {
                    errors = append(errors,
                        fmt.Sprintf("%s は %d 以上である必要があります", field.Name, min))
                }
            case strings.HasPrefix(rule, "max="):
                var max int
                fmt.Sscanf(rule, "max=%d", &max)
                if value.Kind() == reflect.Int && value.Int() > int64(max) {
                    errors = append(errors,
                        fmt.Sprintf("%s は %d 以下である必要があります", field.Name, max))
                }
            }
        }
    }
    return errors
}

// リフレクションで INSERT SQL を自動生成
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
        fmt.Println("バリデーションエラー:", e)
    }

    sql := GenerateInsertSQL("users", User{})
    fmt.Println("生成SQL:", sql)
    // INSERT INTO users (user_id, user_name, email, age) VALUES ($1, $2, $3, $4)
}
```

### 5.3 リフレクションの比較表

| 特性 | Java | Go | Python | C# | Rust |
|------|------|-----|--------|-----|------|
| 型情報取得 | `Class<?>` | `reflect.Type` | `type()` | `Type` | なし（コンパイル時のみ） |
| フィールドアクセス | `Field` | `reflect.Value` | `getattr()` | `FieldInfo` | なし |
| メソッド呼び出し | `Method.invoke()` | `Value.Call()` | `getattr()+()` | `MethodInfo.Invoke()` | なし |
| アノテーション | `@Annotation` | 構造体タグ | デコレータ | `[Attribute]` | `#[derive()]`（マクロ） |
| 動的プロキシ | `Proxy` | なし | `__getattr__` | `DispatchProxy` | なし |
| パフォーマンス | 低 | 低 | 低 | 低 | N/A（ゼロコスト） |
| 型安全性 | 低 | 低 | 低 | 低 | 高（マクロで代替） |

---

## 6. コード生成

### 6.1 コード生成の手法

コード生成は、テンプレートやスキーマから実際のソースコードを自動生成する手法である。リフレクションと異なり、生成されたコードは通常のソースファイルとして存在するため、型安全性と実行時パフォーマンスを両立できる。

```
コード生成のアプローチ:

  ┌──────────────────────────────────────────────────┐
  │              コード生成の分類                      │
  ├──────────────┬───────────────┬──────────────────┤
  │ スキーマ駆動   │ テンプレート    │ AST操作          │
  │              │ エンジン       │                  │
  ├──────────────┼───────────────┼──────────────────┤
  │ ・Protocol   │ ・Go text/    │ ・Rust proc_macro│
  │   Buffers    │   template    │ ・babel plugin   │
  │ ・OpenAPI    │ ・Jinja2      │ ・ts-morph       │
  │ ・GraphQL    │ ・ERB/EJS     │ ・JavaPoet       │
  │   codegen    │               │                  │
  ├──────────────┼───────────────┼──────────────────┤
  │ 入力: IDL    │ 入力: テンプレ  │ 入力: ソースコード │
  │ 出力: 型定義  │ 出力: 任意     │ 出力: 変換済みAST │
  └──────────────┴───────────────┴──────────────────┘
```

### 6.2 Go generate の活用

```go
// go:generate を使ったコード生成の例

//go:generate stringer -type=Color

type Color int

const (
    Red Color = iota
    Green
    Blue
    Yellow
)

// stringer ツールが以下を自動生成:
// func (c Color) String() string { ... }
// "Red", "Green", "Blue", "Yellow" を返す
```

### 6.3 Protocol Buffers によるコード生成

```protobuf
// user.proto: スキーマ定義
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
protoc から生成されるコード:

  user.proto
      │
      ├──→ user.pb.go     (Go の構造体 + シリアライズ)
      ├──→ user_pb2.py    (Python のクラス)
      ├──→ User.java      (Java のクラス)
      ├──→ user.rs        (Rust の struct + enum)
      └──→ user.ts        (TypeScript の interface)

  1つのスキーマから複数言語の型定義を一貫して生成
```

---

## 7. DSL設計のベストプラクティス

### 7.1 DSL設計の原則

```
DSL設計の7つの原則:

  ┌──────────────────────────────────────────────────┐
  │ 1. ドメインの言葉を使う                            │
  │    技術用語ではなく、ドメインエキスパートの語彙を採用   │
  ├──────────────────────────────────────────────────┤
  │ 2. 最小驚き原則                                   │
  │    DSLの動作は、読み手の直感に反しないこと             │
  ├──────────────────────────────────────────────────┤
  │ 3. 段階的開示                                     │
  │    基本的な使用は簡単に、高度な使用も可能に            │
  ├──────────────────────────────────────────────────┤
  │ 4. 型安全性                                       │
  │    コンパイル時に誤りを検出できる設計                  │
  ├──────────────────────────────────────────────────┤
  │ 5. エラーメッセージの品質                           │
  │    DSLレベルで意味のあるエラーを返す                  │
  ├──────────────────────────────────────────────────┤
  │ 6. コンポーザビリティ                              │
  │    DSLの部品を組み合わせてより大きな構造を作れる       │
  ├──────────────────────────────────────────────────┤
  │ 7. エスケープハッチ                                │
  │    DSLでは表現できないケースに対する脱出路を用意する    │
  └──────────────────────────────────────────────────┘
```

### 7.2 Phantom Type による型安全DSL

Phantom Type（幽霊型）は、型パラメータをデータの格納ではなく状態の追跡に使用するテクニックである。これにより、不正な操作をコンパイル時に防止できる。

```rust
// --- Phantom Type でビルダーの状態を型で管理 ---

use std::marker::PhantomData;

// 状態を表す型（実際にはインスタンス化されない）
struct NoHost;
struct HasHost;
struct NoPort;
struct HasPort;

// サーバー設定ビルダー（状態を型パラメータで追跡）
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
    // host を設定すると、HostState が NoHost → HasHost に変わる
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
    // port を設定すると、PortState が NoPort → HasPort に変わる
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

// build() は host と port が両方設定済みの場合のみ呼べる
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
    // 正しい使用: コンパイル通過
    let config = ServerConfigBuilder::new()
        .host("api.example.com")
        .port(443)
        .tls(true)
        .build();

    // 不正な使用: コンパイルエラー
    // let bad = ServerConfigBuilder::new()
    //     .host("api.example.com")
    //     .build();  // エラー: HasPort が必要だが NoPort
}
```

---

## 8. メタプログラミングのリスクと対策

### 8.1 メタプログラミングの利点と問題点

```
利点と問題点の対比:

  ┌──────────────────────┬──────────────────────────────┐
  │       利点            │         問題点                │
  ├──────────────────────┼──────────────────────────────┤
  │ ボイラープレート削減    │ 可読性の低下（「魔法」の増加） │
  │ DRY原則の徹底          │ デバッグの困難さ              │
  │ DSLによる表現力向上     │ コンパイル時間の増大          │
  │ コンパイル時検証        │ IDE サポートの制限            │
  │ 一貫性の保証           │ 学習コストの上昇              │
  │ パフォーマンス最適化    │ エラーメッセージの劣化         │
  └──────────────────────┴──────────────────────────────┘
```

### 8.2 アンチパターン1: 「魔法が多すぎるDSL」

```ruby
# --- 悪い例: method_missing の過度な使用 ---
class MagicQuery
  def method_missing(name, *args)
    if name.to_s.start_with?("find_by_")
      field = name.to_s.sub("find_by_", "")
      # ... 動的にクエリ生成
    elsif name.to_s.start_with?("order_by_")
      field = name.to_s.sub("order_by_", "")
      # ... 動的にソート
    else
      super
    end
  end
end

# query.find_by_name_and_email_order_by_created_at_desc(...)
# → 何が起きているのか追跡不能
# → IDEの補完が効かない
# → タイプミスが実行時まで検出できない

# --- 良い例: 明示的なメソッドチェーン ---
class TypedQuery
  def where(field:, value:) = # ...
  def order_by(field:, direction: :asc) = # ...
  def limit(n) = # ...
end

# query.where(field: :name, value: "田中")
#       .order_by(field: :created_at, direction: :desc)
#       .limit(10)
# → IDE補完が効く
# → タイプミスはコンパイル/構文チェック時に検出
```

**教訓:** DSLの表現力と型安全性のバランスを取る。「暗黙のルール」よりも「明示的な構造」を優先する。

### 8.3 アンチパターン2: 「不必要なマクロの使用」

```rust
// --- 悪い例: 関数で十分なのにマクロを使う ---
macro_rules! add_numbers {
    ($a:expr, $b:expr) => {
        $a + $b
    };
}

let sum = add_numbers!(3, 5);  // マクロの必要性なし

// --- 良い例: 普通の関数を使う ---
fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

let sum = add_numbers(3, 5);  // シンプルで型安全

// --- マクロが正当化されるケース ---
// 1. 可変長引数が必要
macro_rules! sum_all {
    ($($x:expr),+) => {
        0 $(+ $x)+
    };
}
let total = sum_all!(1, 2, 3, 4, 5);

// 2. コンパイル時にコードを生成する必要がある
// 3. 構文の拡張が必要（DSL）
// 4. ボイラープレートの大幅な削減
```

**教訓:** 「通常の関数・ジェネリクス・トレイトで解決できないか？」をまず検討する。マクロは最後の手段である。