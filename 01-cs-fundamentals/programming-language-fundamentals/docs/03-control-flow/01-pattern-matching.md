# パターンマッチ

> パターンマッチは「データの構造に基づいて分岐する」強力な制御構造。switch文の進化版であり、関数型プログラミングの中心的な機能。

## この章で学ぶこと

- [ ] パターンマッチの種類と表現力を理解する
- [ ] 網羅性チェックの重要性を理解する
- [ ] 各言語のパターンマッチ機能を比較できる

---

## 1. パターンマッチの基本

```rust
// Rust: match による構造的パターンマッチ

// リテラルパターン
match x {
    1 => println!("one"),
    2 | 3 => println!("two or three"),  // OR パターン
    4..=9 => println!("four to nine"),  // 範囲パターン
    _ => println!("other"),             // ワイルドカード
}

// 構造体の分解
struct Point { x: i32, y: i32 }
match point {
    Point { x: 0, y: 0 } => println!("origin"),
    Point { x, y: 0 } => println!("on x-axis at {}", x),
    Point { x: 0, y } => println!("on y-axis at {}", y),
    Point { x, y } => println!("({}, {})", x, y),
}

// 列挙型の分解
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    Color(u8, u8, u8),
}

match msg {
    Message::Quit => println!("quit"),
    Message::Move { x, y } => println!("move to ({}, {})", x, y),
    Message::Write(text) => println!("text: {}", text),
    Message::Color(r, g, b) => println!("color: #{:02x}{:02x}{:02x}", r, g, b),
}

// ネストしたパターン
match value {
    Some(Some(x)) if x > 0 => println!("positive: {}", x),
    Some(Some(x)) => println!("non-positive: {}", x),
    Some(None) => println!("inner none"),
    None => println!("outer none"),
}

// ガード条件
match num {
    n if n < 0 => println!("negative"),
    n if n == 0 => println!("zero"),
    n => println!("positive: {}", n),
}

// 束縛（@ パターン）
match age {
    n @ 0..=12 => println!("child: {}", n),
    n @ 13..=17 => println!("teen: {}", n),
    n @ 18.. => println!("adult: {}", n),
    _ => unreachable!(),
}
```

---

## 2. 各言語のパターンマッチ

### Python（3.10+ match文）

```python
# Python 3.10: Structural Pattern Matching
match command:
    case "quit":
        sys.exit()
    case "hello" | "hi":
        print("Hello!")
    case str(s) if s.startswith("/"):
        handle_command(s)
    case _:
        print("Unknown")

# 構造的パターン
match point:
    case (0, 0):
        print("Origin")
    case (x, 0):
        print(f"X-axis at {x}")
    case (0, y):
        print(f"Y-axis at {y}")
    case (x, y):
        print(f"({x}, {y})")

# クラスパターン
match event:
    case Click(position=(x, y)) if x > 100:
        print(f"Right click at ({x}, {y})")
    case KeyPress(key="Enter"):
        print("Enter pressed")
    case KeyPress(key=k):
        print(f"Key: {k}")
```

### TypeScript（判別ユニオン + switch）

```typescript
// TypeScript: 判別ユニオンで擬似パターンマッチ
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rect"; width: number; height: number };

function area(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            return Math.PI * shape.radius ** 2;
        case "rect":
            return shape.width * shape.height;
    }
    // TypeScript は網羅性をチェック（kind の全値を処理しないとエラー）
}

// ts-pattern ライブラリでより強力なパターンマッチ
import { match, P } from 'ts-pattern';

const result = match(shape)
    .with({ kind: "circle", radius: P.when(r => r > 10) }, s =>
        `Large circle: ${s.radius}`)
    .with({ kind: "circle" }, s =>
        `Small circle: ${s.radius}`)
    .with({ kind: "rect" }, s =>
        `Rectangle: ${s.width}x${s.height}`)
    .exhaustive();  // 網羅性チェック
```

### Haskell（パターンマッチの元祖）

```haskell
-- Haskell: 関数定義でのパターンマッチ
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- case 式
describe :: [a] -> String
describe xs = case xs of
    []     -> "empty"
    [_]    -> "singleton"
    [_,_]  -> "pair"
    _      -> "many"

-- リストのパターン
head' :: [a] -> a
head' (x:_) = x
head' []    = error "empty list"

-- タプルのパターン
addVectors :: (Double, Double) -> (Double, Double) -> (Double, Double)
addVectors (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)
```

### Scala

```scala
// Scala: 最も表現力豊かなパターンマッチの1つ
val result = x match {
  case 1 => "one"
  case n if n > 0 => s"positive: $n"
  case _ => "other"
}

// case class の分解
case class Person(name: String, age: Int)

person match {
  case Person("Alice", _) => "Found Alice"
  case Person(name, age) if age >= 18 => s"$name is an adult"
  case Person(name, age) => s"$name is $age years old"
}

// 型パターン
def describe(x: Any): String = x match {
  case i: Int if i > 0 => s"positive int: $i"
  case s: String => s"string: $s"
  case _: Boolean => "boolean"
  case _ => "unknown"
}
```

---

## 3. 網羅性チェック（Exhaustiveness Checking）

```
網羅性チェック = 「全てのケースが処理されているか」をコンパイル時に検証

なぜ重要か？
  → 新しいバリアントを追加した時、処理漏れをコンパイルエラーで検出
  → 実行時の予期しない動作を防止
```

```rust
// Rust: 網羅性チェック
enum Color { Red, Green, Blue }

fn describe(c: Color) -> &'static str {
    match c {
        Color::Red => "red",
        Color::Green => "green",
        // Color::Blue が未処理 → コンパイルエラー
    }
}

// 新しいバリアント追加時
enum Color { Red, Green, Blue, Yellow }  // Yellow 追加
// → 全ての match 文でコンパイルエラーが発生
// → 処理漏れを確実に検出
```

```typescript
// TypeScript: never 型による網羅性チェック
type Color = "red" | "green" | "blue";

function describe(c: Color): string {
    switch (c) {
        case "red": return "Red";
        case "green": return "Green";
        case "blue": return "Blue";
        default:
            const _exhaustive: never = c;
            // "yellow" を追加すると、ここでコンパイルエラー
            return _exhaustive;
    }
}
```

---

## 4. パターンマッチのアンチパターン

```
❌ ワイルドカードの過剰使用
match color {
    Color::Red => "red",
    _ => "other",  // Green, Blue の個別処理を忘れる可能性
}

✅ 全バリアントを明示的に処理
match color {
    Color::Red => "red",
    Color::Green => "green",
    Color::Blue => "blue",
}

❌ パターンの順序ミス
match n {
    _ => "any",     // 全てマッチ → 以下は到達不能
    1 => "one",     // 到達不能
}
```

---

## まとめ

| 言語 | パターンマッチ | 網羅性チェック | 特徴 |
|------|-------------|-------------|------|
| Rust | match（式） | コンパイル時 | 最も安全 |
| Haskell | case / 関数定義 | コンパイル時 | 元祖 |
| Scala | match（式） | コンパイル時 | 表現力最大 |
| Python | match（3.10+） | なし | 構造的パターン |
| TypeScript | switch + never | 型レベル | 判別ユニオン |

---

## 次に読むべきガイド
→ [[02-error-handling.md]] — エラーハンドリング

---

## 参考文献
1. "Rust By Example: Pattern matching." doc.rust-lang.org.
2. "PEP 634: Structural Pattern Matching." python.org, 2021.
