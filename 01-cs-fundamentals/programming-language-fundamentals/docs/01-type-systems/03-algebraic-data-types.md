# 代数的データ型（Algebraic Data Types）

> ADT は「直積型（AND）と直和型（OR）の組み合わせ」でデータを正確にモデリングする手法。不正な状態を型で表現不可能にする。

## この章で学ぶこと

- [ ] 直積型と直和型の概念を理解する
- [ ] パターンマッチとの組み合わせを活用できる
- [ ] 「不正な状態を表現不可能にする」設計ができる

---

## 1. 直積型（Product Types）

```
直積型 = 「AかつB」（AND）
  → 全てのフィールドを同時に持つ

  構造体 / レコード / タプル / クラス
```

```rust
// Rust: 構造体（直積型）
struct User {
    name: String,      // AND
    age: u32,          // AND
    email: String,     // AND
}
// User = String × u32 × String
// 取りうる値の数 = name の値の数 × age の値の数 × email の値の数

// タプル（無名の直積型）
let point: (f64, f64) = (3.0, 4.0);
```

```typescript
// TypeScript: インターフェース（直積型）
interface User {
    name: string;     // AND
    age: number;      // AND
    email: string;    // AND
}

// タプル型
type Point = [number, number];
```

---

## 2. 直和型（Sum Types / Tagged Unions）

```
直和型 = 「AまたはB」（OR）
  → 複数の候補のうち1つだけを持つ

  列挙型 / ユニオン型 / バリアント
```

```rust
// Rust: enum（直和型の最も洗練された形）
enum Shape {
    Circle(f64),                    // 半径
    Rectangle(f64, f64),            // 幅, 高さ
    Triangle(f64, f64, f64),        // 3辺
}
// Shape = Circle(f64) + Rectangle(f64, f64) + Triangle(f64, f64, f64)
// いずれか1つだけ

// Option: 「値があるかないか」を型で表現
enum Option<T> {
    Some(T),
    None,
}

// Result: 「成功か失敗か」を型で表現
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

```typescript
// TypeScript: ユニオン型 + 判別フィールド
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; a: number; b: number; c: number };

// 判別ユニオン（Discriminated Union）
function area(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            return shape.width * shape.height;
        case "triangle": {
            const s = (shape.a + shape.b + shape.c) / 2;
            return Math.sqrt(s * (s-shape.a) * (s-shape.b) * (s-shape.c));
        }
    }
}
```

```haskell
-- Haskell: data 宣言（ADTの本家）
data Shape
    = Circle Double
    | Rectangle Double Double
    | Triangle Double Double Double

area :: Shape -> Double
area (Circle r)        = pi * r * r
area (Rectangle w h)   = w * h
area (Triangle a b c)  = let s = (a + b + c) / 2
                          in sqrt (s * (s-a) * (s-b) * (s-c))
```

---

## 3. パターンマッチ

```rust
// Rust: match による網羅的パターンマッチ
fn describe(shape: &Shape) -> String {
    match shape {
        Shape::Circle(r) => format!("Circle with radius {}", r),
        Shape::Rectangle(w, h) => format!("{}x{} rectangle", w, h),
        Shape::Triangle(a, b, c) => format!("Triangle ({}, {}, {})", a, b, c),
    }
    // 全バリアントを処理しないとコンパイルエラー（網羅性チェック）
}

// Option のパターンマッチ
fn greet(name: Option<&str>) -> String {
    match name {
        Some(n) => format!("Hello, {}!", n),
        None => "Hello, stranger!".to_string(),
    }
}

// if let（単一パターン）
if let Some(name) = get_name() {
    println!("Found: {}", name);
}

// ネストしたパターン
match result {
    Ok(Some(value)) if value > 0 => println!("Positive: {}", value),
    Ok(Some(value)) => println!("Non-positive: {}", value),
    Ok(None) => println!("No value"),
    Err(e) => println!("Error: {}", e),
}
```

---

## 4. 「不正な状態を表現不可能にする」

### 悪い設計（不正な状態が可能）

```typescript
// ❌ 不正な状態が表現可能
interface Connection {
    status: "disconnected" | "connecting" | "connected" | "error";
    socket?: WebSocket;       // connected の時だけ存在
    error?: Error;            // error の時だけ存在
    retryCount?: number;      // error の時だけ意味がある
}

// 問題: status: "disconnected" なのに socket がある状態が作れてしまう
const invalid: Connection = {
    status: "disconnected",
    socket: new WebSocket("ws://..."),  // 不正だが型エラーにならない
};
```

### 良い設計（不正な状態が表現不可能）

```typescript
// ✅ 不正な状態が型で表現できない
type Connection =
    | { status: "disconnected" }
    | { status: "connecting" }
    | { status: "connected"; socket: WebSocket }
    | { status: "error"; error: Error; retryCount: number };

// status: "disconnected" に socket を持たせることが型レベルで不可能
```

```rust
// Rust: enum で状態遷移を厳密にモデリング
enum ConnectionState {
    Disconnected,
    Connecting,
    Connected { socket: TcpStream },
    Error { error: io::Error, retry_count: u32 },
}

// Connected 状態でしか socket にアクセスできない
fn send_data(state: &ConnectionState, data: &[u8]) -> Result<(), String> {
    match state {
        ConnectionState::Connected { socket } => {
            // socket を安全に使用
            Ok(())
        }
        _ => Err("Not connected".to_string()),
    }
}
```

### 実践例: API レスポンス

```typescript
// ✅ ローディング状態を ADT で表現
type AsyncData<T> =
    | { state: "idle" }
    | { state: "loading" }
    | { state: "success"; data: T }
    | { state: "error"; error: Error };

function renderUser(user: AsyncData<User>) {
    switch (user.state) {
        case "idle":
            return <div>Press load</div>;
        case "loading":
            return <Spinner />;
        case "success":
            return <UserCard user={user.data} />;
        case "error":
            return <ErrorMessage error={user.error} />;
    }
}
```

---

## 5. Null の問題と Option/Maybe

```
「10億ドルの間違い」（Tony Hoare, null の発明者）

問題: null は型システムの穴
  Java:    String name = null;  // NullPointerException の温床
  JS:      let x = null;       // TypeError: Cannot read property ...

解決: Option型 / Maybe型
  Rust:    Option<String>  → Some("Gaku") | None
  Haskell: Maybe String    → Just "Gaku" | Nothing
  Swift:   String?         → "Gaku" | nil

  値がないことを「型で明示」し、
  パターンマッチで安全に処理を強制する
```

```rust
// Rust: null がない。Option で明示
fn find_user(id: u32) -> Option<User> {
    if id == 1 {
        Some(User { name: "Gaku".into(), age: 30 })
    } else {
        None
    }
}

// 使う側は None の可能性を必ず処理する
match find_user(1) {
    Some(user) => println!("Found: {}", user.name),
    None => println!("Not found"),
}

// メソッドチェーン
let name = find_user(1)
    .map(|u| u.name)
    .unwrap_or("Unknown".into());

// ? 演算子（エラー伝播）
fn get_user_name(id: u32) -> Option<String> {
    let user = find_user(id)?;  // None なら早期リターン
    Some(user.name)
}
```

---

## 実践演習

### 演習1: [基礎] — 信号機をADTでモデリング
交通信号機の状態（赤・黄・青）を Rust の enum または TypeScript の判別ユニオンで実装する。

### 演習2: [応用] — JSON パーサーの型定義
JSON の値（null, bool, number, string, array, object）を ADT で定義する。

### 演習3: [発展] — 状態機械の実装
HTTP リクエストの状態遷移（Idle → Sending → Success/Error → Idle）を ADT で実装し、不正な遷移を型で防止する。

---

## まとめ

| 概念 | 説明 | 例 |
|------|------|------|
| 直積型 | A かつ B（全フィールド） | struct, interface |
| 直和型 | A または B（1つだけ） | enum, union type |
| パターンマッチ | 網羅的な分岐処理 | match, switch |
| Option/Maybe | null の安全な代替 | Some/None |
| 状態モデリング | 不正な状態を型で防止 | 判別ユニオン |

---

## 次に読むべきガイド
→ [[../02-memory-models/00-stack-and-heap.md]] — メモリモデル

---

## 参考文献
1. Pierce, B. "Types and Programming Languages." MIT Press, 2002.
2. Wlaschin, S. "Domain Modeling Made Functional." Pragmatic Bookshelf, 2018.
