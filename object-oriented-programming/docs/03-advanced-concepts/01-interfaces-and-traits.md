# インターフェースとトレイト

> インターフェースは「契約」を定義し、トレイトは「再利用可能な振る舞い」を提供する。各言語での実装の違いと、ダックタイピングとの関係を理解する。

## この章で学ぶこと

- [ ] インターフェースとトレイトの違いを理解する
- [ ] 各言語での実装方法を把握する
- [ ] 構造的型付けとダックタイピングの関係を学ぶ

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
│ 代表言語     │ Java, TS, Go  │ Rust, Scala,PHP│ Java, Python   │
└──────────────┴────────────────┴────────────────┴────────────────┘
```

---

## 2. 各言語の実装

### Java: インターフェース

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

### Rust: トレイト

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

### Go: 暗黙的インターフェース

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

### TypeScript: 構造的型付け

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

---

## 4. 選択指針

```
インターフェース:
  → 「何ができるか」の契約を定義
  → 実装は持たない（またはデフォルト最小限）
  → 多重実装が必要な場合

トレイト:
  → 再利用可能な振る舞いの単位
  → デフォルト実装を積極的に提供
  → ミックスイン的な使い方

抽象クラス:
  → 共通の状態（フィールド）+ 部分的な実装
  → テンプレートメソッドパターン
  → is-a 関係が明確な場合
```

---

## まとめ

| 概念 | 特徴 | 代表言語 |
|------|------|---------|
| インターフェース | 契約の定義 | Java, TS, Go |
| トレイト | 再利用可能な振る舞い | Rust, Scala, PHP |
| 構造的型付け | 構造が合えば互換 | TS, Go |
| ダックタイピング | 実行時にメソッド確認 | Python, Ruby |

---

## 次に読むべきガイド
→ [[02-mixins-and-multiple-inheritance.md]] — ミックスインと多重継承

---

## 参考文献
1. Odersky, M. "Scalable Component Abstractions." OOPSLA, 2005.
2. The Rust Programming Language. "Traits." doc.rust-lang.org.
