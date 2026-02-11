# DSL とメタプログラミング

> DSL（Domain-Specific Language）は「特定の問題領域に特化した言語」。メタプログラミングは「プログラムでプログラムを生成・変換する」技術。

## この章で学ぶこと

- [ ] 内部DSLと外部DSLの違いを理解する
- [ ] メタプログラミングの手法と活用場面を把握する

---

## 1. DSL の種類

```
外部DSL（独自の構文を持つ）:
  SQL:         SELECT name FROM users WHERE age > 18
  正規表現:    ^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,}$
  HTML/CSS:    <div class="container">...</div>
  YAML/JSON:   設定ファイル
  GraphQL:     query { user(id: 1) { name } }
  Terraform:   resource "aws_instance" "web" { ... }

内部DSL（ホスト言語の構文を活用）:
  Ruby:        Rails のルーティング、RSpec
  Kotlin:      Ktor、Gradle Kotlin DSL
  Scala:       Akka、sbt
  Swift:       SwiftUI
```

### 内部DSLの例

```ruby
# Ruby: RSpec（テストDSL）
describe User do
  context "when new" do
    it "has no posts" do
      expect(User.new.posts).to be_empty
    end
  end
end

# Rails: ルーティングDSL
Rails.application.routes.draw do
  resources :users do
    resources :posts
  end
  root "pages#home"
end
```

```kotlin
// Kotlin: 型安全なHTML DSL
html {
    body {
        h1 { +"Welcome" }
        ul {
            for (item in items) {
                li { +item.name }
            }
        }
    }
}

// Gradle Kotlin DSL
dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    testImplementation("org.junit.jupiter:junit-jupiter")
}
```

---

## 2. メタプログラミング

```
メタプログラミング = プログラムでプログラムを操作

手法:
  1. マクロ（コンパイル時にコード生成）
  2. リフレクション（実行時に型情報にアクセス）
  3. コード生成（テンプレートからコードを生成）
  4. eval（文字列をコードとして実行）
```

```rust
// Rust: derive マクロ（コンパイル時コード生成）
#[derive(Debug, Clone, Serialize, Deserialize)]
struct User {
    name: String,
    age: u32,
}
// → Debug, Clone, Serialize の実装が自動生成される

// 手続き型マクロ（高度なコード生成）
#[tokio::main]
async fn main() {
    // → fn main() { tokio::runtime::Runtime::new()... } に展開
}
```

```python
# Python: デコレータ（関数を変換する関数）
def retry(max_attempts=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
        return wrapper
    return decorator

@retry(max_attempts=5)
def fetch_data(url):
    return requests.get(url)

# メタクラス（クラスを生成するクラス）
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
```

```elixir
# Elixir: マクロ（AST変換）
defmodule MyApp do
  # unless は実際にはマクロ
  defmacro unless(condition, do: block) do
    quote do
      if !unquote(condition) do
        unquote(block)
      end
    end
  end
end
```

---

## 3. メタプログラミングの注意点

```
利点:
  ✓ ボイラープレートの削減
  ✓ DSLによる表現力の向上
  ✓ コンパイル時の検証・最適化

リスク:
  ✗ 可読性の低下（「魔法」が多すぎる）
  ✗ デバッグの困難さ
  ✗ コンパイル時間の増加
  ✗ IDE サポートの制限

原則:
  「メタプログラミングは最後の手段。まず通常のコードで解決を試みる」
```

---

## まとめ

| 手法 | タイミング | 代表言語 |
|------|----------|---------|
| マクロ | コンパイル時 | Rust, Elixir, C |
| デコレータ | 実行時 | Python, TypeScript |
| リフレクション | 実行時 | Java, Go, C# |
| コード生成 | ビルド時 | Go(generate), Rust(build.rs) |

---

## 次に読むべきガイド
→ [[03-future-of-languages.md]] — 言語の未来

---

## 参考文献
1. Fowler, M. "Domain-Specific Languages." Addison-Wesley, 2010.
