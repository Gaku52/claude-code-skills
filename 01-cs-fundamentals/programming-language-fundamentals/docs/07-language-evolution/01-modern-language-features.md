# モダン言語の共通機能

> 2010年代以降の主要言語は、過去数十年の研究と失敗から学び、共通する「ベストプラクティス」を標準装備として取り込んでいる。本章ではこれらの共通機能を網羅的に解説し、各機能の理論的背景、言語間比較、実装パターンまで深掘りする。

## この章で学ぶこと

- [ ] モダン言語に共通する10大機能を正確に把握する
- [ ] 各機能がどの言語・研究に由来するか歴史的系譜を理解する
- [ ] 型推論・Null安全・パターンマッチなどの本質的な仕組みを説明できる
- [ ] 言語選択時に機能の成熟度を評価できるようになる
- [ ] async/await・ADT・イミュータビリティの設計意図を実コードで示せる
- [ ] 2020年代のトレンド（段階的型付け・AI協調・Wasm）を展望できる

---

## 1. モダン言語とは何か：定義と背景

### 1.1 「モダン言語」の定義

「モダン言語」に厳密な定義は存在しないが、ここでは以下の条件を満たす言語を指す。

```
モダン言語の条件:
+----------------------------------------------------+
| 1. 静的型付け or 段階的型付けを備える                 |
| 2. メモリ安全性を言語レベルで保証する仕組みがある       |
| 3. Null安全またはOption/Maybe型を提供する             |
| 4. パッケージマネージャが公式に統合されている           |
| 5. 非同期処理の言語レベルサポートがある                |
| 6. パターンマッチまたは類似の構造分解機能がある         |
| 7. 充実したツールチェーン（フォーマッタ・リンター）     |
+----------------------------------------------------+
```

### 1.2 言語世代の分類

```
+------------------+------------------+------------------+------------------+
|   第1世代        |   第2世代        |   第3世代        |   第4世代        |
|   (1950-70s)     |   (1980-90s)     |   (2000-10s)     |   (2010s-)       |
+------------------+------------------+------------------+------------------+
| FORTRAN          | C++              | Java             | Rust (2015)      |
| LISP             | Perl             | C#               | Kotlin (2016)    |
| COBOL            | Python           | Scala            | Swift (2014)     |
| Algol            | Ruby             | Go               | TypeScript(2012) |
| ML               | Haskell          | Clojure          | Zig (2016)       |
+------------------+------------------+------------------+------------------+
| 計算の基礎       | OOP / スクリプト  | VM / GC成熟      | 安全性+生産性    |
+------------------+------------------+------------------+------------------+
```

第4世代の言語が共通して採用する機能群が、本章の主題である。

### 1.3 収斂進化としての言語設計

生物学における収斂進化（異なる系統の生物が類似した形質を独立に発達させる現象）と同様に、異なる設計思想を持つ言語が同じ機能に到達している。これは偶然ではなく、ソフトウェア工学の大規模な社会実験の結果として「何が本当に有用か」が明らかになった証拠である。

```
       関数型言語の系譜                 手続き型/OOP言語の系譜
       ┌─────────────┐                 ┌─────────────┐
       │ ML (1973)   │                 │ C (1972)    │
       └──────┬──────┘                 └──────┬──────┘
              │                                │
       ┌──────┴──────┐                 ┌──────┴──────┐
       │Haskell(1990)│                 │ C++ (1985)  │
       └──────┬──────┘                 └──────┬──────┘
              │                                │
              │    ┌───────────────────┐       │
              └───►│ 収斂進化の結果     │◄──────┘
                   │ ・型推論           │
                   │ ・Null安全         │
                   │ ・パターンマッチ    │
                   │ ・不変デフォルト    │
                   │ ・ADT             │
                   └───────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐  ┌────────┐  ┌────────┐
         │ Rust   │  │ Kotlin │  │ Swift  │
         └────────┘  └────────┘  └────────┘
```

---

## 2. モダン言語の10大標準機能

### 2.1 型推論（Type Inference）

#### 2.1.1 概要と歴史

型推論とは、プログラマが明示的に型を書かなくても、コンパイラが文脈から型を自動的に決定する仕組みである。

**歴史的系譜:**
- 1958年: Hindley が論理学で型推論の基礎理論を提示
- 1973年: ML言語で Algorithm W（Hindley-Milner 型推論）を実装
- 1990年: Haskell が完全な型推論を標準装備
- 2004年: C# 3.0 で `var` キーワード導入
- 2012年: TypeScript が Flow Analysis ベースの型推論を採用
- 2015年: Rust が局所型推論 + trait制約推論を実装

#### 2.1.2 型推論のアルゴリズム分類

```
型推論アルゴリズムの分類:

┌───────────────────────────────────────────────────┐
│              完全型推論                             │
│    (プログラム全体の型注釈が不要)                     │
│    例: Haskell, ML, OCaml                          │
├───────────────────────────────────────────────────┤
│              局所型推論                             │
│    (関数シグネチャは必要、本体内は推論)               │
│    例: Rust, Scala, TypeScript                     │
├───────────────────────────────────────────────────┤
│              限定的型推論                           │
│    (変数初期化時のみ推論)                            │
│    例: C++ (auto), Java (var), Go (:=)             │
├───────────────────────────────────────────────────┤
│              段階的型推論                           │
│    (型注釈を任意に追加可能)                          │
│    例: TypeScript, mypy (Python)                   │
└───────────────────────────────────────────────────┘
```

#### 2.1.3 言語別コード比較

**コード例1: 各言語の型推論**

```rust
// Rust: 局所型推論
fn calculate_total(items: Vec<f64>) -> f64 {
    let subtotal = items.iter().sum::<f64>();  // f64 と推論
    let tax_rate = 0.1;                        // f64 と推論
    let tax = subtotal * tax_rate;             // f64 と推論
    subtotal + tax                             // 戻り値型は明示
}
```

```kotlin
// Kotlin: 局所型推論 + スマートキャスト
fun processInput(input: Any): String {
    val result = when (input) {          // String と推論
        is Int -> "整数: ${input * 2}"   // スマートキャスト: Any -> Int
        is String -> "文字列: ${input.uppercase()}"
        is List<*> -> "リスト: ${input.size}件"
        else -> "不明な型"
    }
    return result
}
```

```typescript
// TypeScript: Flow Analysis ベースの型推論
function processData(data: unknown) {
    if (typeof data === "string") {
        // この時点で data は string に絞り込まれる
        console.log(data.toUpperCase());
    } else if (Array.isArray(data)) {
        // この時点で data は any[] に絞り込まれる
        console.log(data.length);
    }
}

// 複雑な推論
const transform = <T, U>(arr: T[], fn: (item: T) => U) => arr.map(fn);
const result = transform([1, 2, 3], x => x.toString());
// result の型: string[] と推論される
```

```go
// Go: 限定的型推論（短縮変数宣言）
func processOrder() {
    price := 100.0          // float64 と推論
    quantity := 3            // int と推論
    total := price * float64(quantity)  // float64 と推論

    // 構造体リテラルも推論される
    order := struct {
        Total    float64
        Currency string
    }{total, "JPY"}

    fmt.Println(order)
}
```

```swift
// Swift: 双方向型推論
let numbers = [1, 2, 3, 4, 5]          // [Int] と推論
let doubled = numbers.map { $0 * 2 }    // [Int] と推論

// ジェネリクスと組み合わせた推論
func findFirst<T: Comparable>(_ array: [T], where predicate: (T) -> Bool) -> T? {
    array.first(where: predicate)
}

let firstEven = findFirst(numbers) { $0 % 2 == 0 }  // Int? と推論
```

#### 2.1.4 型推論の比較表

| 特性 | Rust | TypeScript | Kotlin | Go | Swift | Haskell |
|------|------|-----------|--------|-----|-------|---------|
| 推論範囲 | 局所 | 段階的 | 局所 | 限定 | 局所 | 完全 |
| 関数シグネチャ | 必須 | 任意 | 必須 | 必須 | 必須 | 任意 |
| ジェネリクス推論 | 強力 | 強力 | 強力 | 限定 | 強力 | 完全 |
| 相互再帰推論 | 不可 | 不可 | 不可 | 不可 | 不可 | 可能 |
| 型推論速度 | 速い | 中程度 | 速い | 速い | 中程度 | 遅い場合あり |
| エラーメッセージ | 優秀 | 良好 | 良好 | 簡潔 | 良好 | 難解な場合あり |

### 2.2 Null安全（Null Safety）

#### 2.2.1 「10億ドルの間違い」

Tony Hoare は 1965年に ALGOL W で null 参照を導入した。2009年の QCon 講演で彼はこれを「10億ドルの間違い」と呼んだ。null 参照はあらゆるプログラミング言語に伝播し、数え切れないバグ・クラッシュ・セキュリティ脆弱性の原因となった。

```
Null参照の問題:

  従来のコード（Java）:
  ┌─────────────────────────────────┐
  │ String name = user.getName();   │ ← user が null なら NPE
  │ int len = name.length();        │ ← name が null なら NPE
  │ // 2箇所の爆弾が潜んでいる      │
  └─────────────────────────────────┘

  モダンなコード（Kotlin）:
  ┌─────────────────────────────────┐
  │ val name: String? = user?.name  │ ← 型で null可能性を表現
  │ val len: Int = name?.length ?: 0│ ← 安全な呼び出し + デフォルト値
  │ // コンパイル時に安全性を保証    │
  └─────────────────────────────────┘
```

#### 2.2.2 Null安全の実装パターン

**コード例2: 各言語のNull安全**

```rust
// Rust: Option<T> 型
fn find_user(id: u64) -> Option<User> {
    let db = get_database();
    db.users.get(&id).cloned()
}

fn get_user_email(id: u64) -> String {
    // パターンマッチによる安全な分岐
    match find_user(id) {
        Some(user) => user.email,
        None => String::from("unknown@example.com"),
    }

    // メソッドチェーンによる簡潔な記述
    // find_user(id)
    //     .map(|u| u.email)
    //     .unwrap_or_else(|| String::from("unknown@example.com"))
}

// ? 演算子による早期リターン
fn process_order(user_id: u64, item_id: u64) -> Option<Receipt> {
    let user = find_user(user_id)?;        // None なら即 return None
    let item = find_item(item_id)?;        // None なら即 return None
    let payment = process_payment(&user)?;  // None なら即 return None
    Some(Receipt::new(user, item, payment))
}
```

```kotlin
// Kotlin: Nullable型 (?記法)
fun processUserProfile(userId: Long): String {
    val user: User? = findUser(userId)

    // 安全呼び出し演算子 ?.
    val nameLength: Int? = user?.name?.length

    // エルビス演算子 ?:
    val displayName: String = user?.name ?: "Anonymous"

    // スマートキャスト（nullチェック後は自動的にNon-null扱い）
    if (user != null) {
        // ここでは user は User 型（User? ではない）
        println("Welcome, ${user.name}")
    }

    // let スコープ関数との組み合わせ
    return user?.let { u ->
        "${u.name} (${u.email})"
    } ?: "ユーザーが見つかりません"
}
```

```swift
// Swift: Optional型
func fetchAndDisplayUser(id: Int) -> String {
    guard let user = findUser(id: id) else {
        return "ユーザーが見つかりません"
    }
    // guard let 以降、user は非Optional として使用可能

    // Optional chaining
    let city: String? = user.address?.city

    // nil合体演算子
    let displayCity = city ?? "未設定"

    // if let による安全なアンラップ
    if let email = user.email {
        sendNotification(to: email)
    }

    return "\(user.name) - \(displayCity)"
}
```

```typescript
// TypeScript: strictNullChecks
function processConfig(config: Config | undefined): Result {
    // Optional chaining
    const timeout = config?.network?.timeout ?? 3000;

    // 型ガード
    if (config === undefined) {
        return { status: "error", message: "設定がありません" };
    }
    // 以降 config は Config 型に絞り込まれる

    // Non-null assertion（使用は最小限に）
    // const value = config!.value;  // 危険: 実行時エラーの可能性

    return { status: "ok", data: applyConfig(config) };
}
```

#### 2.2.3 Null安全パターンの比較表

| 特性 | Rust (Option) | Kotlin (?) | Swift (Optional) | TypeScript (strict) |
|------|-------------|-----------|-----------------|-------------------|
| 表現方法 | `Option<T>` | `T?` | `T?` / `Optional<T>` | `T \| undefined` |
| 安全なアクセス | `.map()` / `?` | `?.` | `?.` / `if let` | `?.` |
| デフォルト値 | `.unwrap_or()` | `?:` | `??` | `??` |
| 強制アンラップ | `.unwrap()` (panic) | `!!` (例外) | `!` (trap) | `!` (型アサーション) |
| パターンマッチ | `match` / `if let` | `when` | `switch` / `if let` | 型ガード |
| コンパイル時保証 | 完全 | 完全 | 完全 | tsconfig依存 |

### 2.3 パターンマッチ（Pattern Matching）

#### 2.3.1 パターンマッチの本質

パターンマッチは単なる `switch` 文の拡張ではない。データの構造を分解しながら条件分岐する強力な機構であり、以下の3つの機能を統合したものである。

1. **条件分岐**: 値の種類による分岐
2. **分解束縛（Destructuring）**: 複合データから要素を取り出す
3. **網羅性検査（Exhaustiveness Check）**: 全パターンを網羅しているかコンパイル時に検証

```
パターンマッチの構造:

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  条件分岐     │  +  │  分解束縛     │  +  │  網羅性検査   │
  │  (if/switch)  │     │  (destruct)  │     │  (exhaustive) │
  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  パターンマッチ      │
                    │  match / when      │
                    └───────────────────┘
```

#### 2.3.2 パターンの種類

**コード例3: Rustのパターンマッチ全パターン**

```rust
// --- 様々なパターンの種類 ---

enum Shape {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Triangle { base: f64, height: f64 },
    Polygon { sides: Vec<f64> },
}

fn describe_shape(shape: &Shape) -> String {
    match shape {
        // 構造体パターン + リテラルパターン
        Shape::Circle { radius } if *radius == 0.0 => {
            "点（半径ゼロの円）".to_string()
        }

        // 構造体パターン + ガード節
        Shape::Circle { radius } if *radius > 100.0 => {
            format!("巨大な円（半径: {:.1}）", radius)
        }

        // 構造体パターン + 変数束縛
        Shape::Circle { radius } => {
            format!("円（半径: {:.1}, 面積: {:.1}）", radius, std::f64::consts::PI * radius * radius)
        }

        // 複数フィールドの分解
        Shape::Rectangle { width, height } if (width - height).abs() < f64::EPSILON => {
            format!("正方形（辺: {:.1}）", width)
        }

        Shape::Rectangle { width, height } => {
            format!("長方形（{:.1} x {:.1}）", width, height)
        }

        // ネストしたパターン
        Shape::Triangle { base, height } => {
            format!("三角形（面積: {:.1}）", base * height / 2.0)
        }

        // ワイルドカードパターン
        Shape::Polygon { sides } if sides.is_empty() => {
            "無効な多角形".to_string()
        }

        Shape::Polygon { sides } => {
            format!("{}角形", sides.len())
        }
    }
}

// --- タプルパターン ---
fn classify_point(x: i32, y: i32) -> &'static str {
    match (x.signum(), y.signum()) {
        (1, 1)   => "第1象限",
        (-1, 1)  => "第2象限",
        (-1, -1) => "第3象限",
        (1, -1)  => "第4象限",
        (0, 0)   => "原点",
        (0, _)   => "Y軸上",
        (_, 0)   => "X軸上",
        _         => unreachable!(),
    }
}

// --- OR パターン ---
fn is_weekend(day: &str) -> bool {
    matches!(day, "Saturday" | "Sunday")
}
```

### 2.4 async/await（非同期処理）

#### 2.4.1 非同期処理の進化

```
非同期処理の進化:

  レベル1: コールバック地獄         レベル2: Promise/Future
  ┌────────────────────┐          ┌────────────────────┐
  │ fetchUser(id, (u)=>│          │ fetchUser(id)      │
  │   fetchOrders(u,   │          │   .then(u =>       │
  │     (orders) =>    │   →→→    │     fetchOrders(u))│
  │       render(      │          │   .then(orders =>  │
  │         orders)    │          │     render(orders))│
  │   )                │          │   .catch(handleErr)│
  │ )                  │          │                    │
  └────────────────────┘          └────────────────────┘
         ↓                                ↓
  レベル3: async/await             レベル4: 構造化並行性
  ┌────────────────────┐          ┌────────────────────┐
  │ async fn process() │          │ async fn process() │
  │   let u =          │          │   let (u, c) =     │
  │     fetchUser(id)  │   →→→    │     join!(         │
  │     .await;        │          │       fetchUser(), │
  │   let orders =     │          │       fetchConfig()│
  │     fetchOrders(u) │          │     ).await;       │
  │     .await;        │          │   // 並行実行       │
  │   render(orders);  │          │   process(u, c);   │
  └────────────────────┘          └────────────────────┘
```

#### 2.4.2 各言語の async/await 実装

**コード例4: 各言語の非同期処理**

```rust
// Rust: Zero-cost async/await
use tokio;

#[derive(Debug)]
struct User { name: String, email: String }
#[derive(Debug)]
struct Order { id: u64, amount: f64 }

async fn fetch_user(id: u64) -> Result<User, Box<dyn std::error::Error>> {
    // HTTP リクエストのシミュレーション
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(User { name: "田中".into(), email: "tanaka@example.com".into() })
}

async fn fetch_orders(user: &User) -> Result<Vec<Order>, Box<dyn std::error::Error>> {
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    Ok(vec![
        Order { id: 1, amount: 1500.0 },
        Order { id: 2, amount: 3200.0 },
    ])
}

// 構造化並行性: 複数の非同期タスクを同時実行
async fn dashboard(user_id: u64) -> Result<(), Box<dyn std::error::Error>> {
    let user = fetch_user(user_id).await?;

    // join! で並行実行
    let (orders, notifications) = tokio::join!(
        fetch_orders(&user),
        fetch_notifications(&user),
    );

    println!("ユーザー: {:?}", user);
    println!("注文数: {}", orders?.len());
    Ok(())
}
```

```python
# Python: asyncio ベースの async/await
import asyncio
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str

@dataclass
class Order:
    id: int
    amount: float

async def fetch_user(user_id: int) -> User:
    await asyncio.sleep(0.1)  # I/O シミュレーション
    return User(name="田中", email="tanaka@example.com")

async def fetch_orders(user: User) -> list[Order]:
    await asyncio.sleep(0.05)
    return [Order(id=1, amount=1500.0), Order(id=2, amount=3200.0)]

async def fetch_notifications(user: User) -> list[str]:
    await asyncio.sleep(0.08)
    return ["新着メッセージ", "セール情報"]

# 構造化並行性: TaskGroup (Python 3.11+)
async def dashboard(user_id: int):
    user = await fetch_user(user_id)

    # gather で並行実行
    orders, notifications = await asyncio.gather(
        fetch_orders(user),
        fetch_notifications(user),
    )

    print(f"ユーザー: {user.name}")
    print(f"注文数: {len(orders)}")
    print(f"通知数: {len(notifications)}")

asyncio.run(dashboard(1))
```

```typescript
// TypeScript: Promise ベースの async/await
interface User {
    name: string;
    email: string;
}

interface Order {
    id: number;
    amount: number;
}

async function fetchUser(id: number): Promise<User> {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
}

async function fetchOrders(user: User): Promise<Order[]> {
    const response = await fetch(`/api/orders?email=${user.email}`);
    return response.json();
}

// 構造化並行性: Promise.all / Promise.allSettled
async function dashboard(userId: number): Promise<void> {
    const user = await fetchUser(userId);

    // Promise.allSettled: 一部失敗しても全結果を取得
    const [ordersResult, notificationsResult] = await Promise.allSettled([
        fetchOrders(user),
        fetchNotifications(user),
    ]);

    if (ordersResult.status === "fulfilled") {
        console.log(`注文数: ${ordersResult.value.length}`);
    }
}
```

### 2.5 代数的データ型（Algebraic Data Types）

#### 2.5.1 ADTの数学的基礎

代数的データ型は、型を「代数」（加算と乗算）として扱う考え方である。

```
代数的データ型の構造:

  直積型 (Product Type) = AND
  ┌─────────────────────────────────┐
  │ struct Point { x: f64, y: f64 } │
  │ 値の数 = f64の値 × f64の値        │
  │ = AND結合                        │
  └─────────────────────────────────┘

  直和型 (Sum Type) = OR
  ┌─────────────────────────────────┐
  │ enum Shape {                    │
  │   Circle(f64),                  │
  │   Rectangle(f64, f64),          │
  │ }                               │
  │ 値の数 = f64の値 + (f64 × f64)   │
  │ = OR結合                        │
  └─────────────────────────────────┘

  組み合わせ:
  ┌─────────────────────────────────┐
  │ 直積型 × 直和型 = 豊かな表現力    │
  │ Result<T, E> = Ok(T) | Err(E)  │
  │ Option<T>    = Some(T) | None  │
  └─────────────────────────────────┘
```

#### 2.5.2 ADTによるドメインモデリング

**コード例5: ADTを活用したドメインモデリング**

```rust
// --- ECサイトの注文状態をADTで厳密にモデリング ---

// 直和型: 注文の状態を完全に列挙
enum OrderStatus {
    // 各バリアントが異なるデータを持てる
    Pending {
        created_at: DateTime<Utc>,
    },
    Confirmed {
        confirmed_at: DateTime<Utc>,
        estimated_delivery: DateTime<Utc>,
    },
    Shipped {
        shipped_at: DateTime<Utc>,
        tracking_number: String,
        carrier: Carrier,
    },
    Delivered {
        delivered_at: DateTime<Utc>,
        signed_by: Option<String>,
    },
    Cancelled {
        cancelled_at: DateTime<Utc>,
        reason: CancellationReason,
        refund_amount: Option<Money>,
    },
}

// 直和型: キャンセル理由
enum CancellationReason {
    CustomerRequest,
    OutOfStock,
    PaymentFailed,
    FraudDetected,
}

// 直和型: 配送業者
enum Carrier {
    YamatoTransport,
    SagawaExpress,
    JapanPost,
}

// 直積型: 金額（通貨付き）
struct Money {
    amount: u64,      // セント/円単位
    currency: Currency,
}

enum Currency { JPY, USD, EUR }

// パターンマッチとADTの組み合わせ
fn get_order_summary(status: &OrderStatus) -> String {
    match status {
        OrderStatus::Pending { created_at } => {
            format!("注文受付中（{}）", created_at.format("%Y/%m/%d"))
        }
        OrderStatus::Confirmed { estimated_delivery, .. } => {
            format!("確認済み - 配達予定: {}", estimated_delivery.format("%m/%d"))
        }
        OrderStatus::Shipped { tracking_number, carrier, .. } => {
            let carrier_name = match carrier {
                Carrier::YamatoTransport => "ヤマト運輸",
                Carrier::SagawaExpress => "佐川急便",
                Carrier::JapanPost => "日本郵便",
            };
            format!("配送中 [{}] 追跡: {}", carrier_name, tracking_number)
        }
        OrderStatus::Delivered { signed_by, .. } => {
            match signed_by {
                Some(name) => format!("配達完了（受取人: {}）", name),
                None => "配達完了（置き配）".to_string(),
            }
        }
        OrderStatus::Cancelled { reason, refund_amount, .. } => {
            let reason_text = match reason {
                CancellationReason::CustomerRequest => "お客様のご要望",
                CancellationReason::OutOfStock => "在庫切れ",
                CancellationReason::PaymentFailed => "決済エラー",
                CancellationReason::FraudDetected => "不正検知",
            };
            match refund_amount {
                Some(money) => format!("キャンセル済み（理由: {}, 返金: {}円）", reason_text, money.amount),
                None => format!("キャンセル済み（理由: {}）", reason_text),
            }
        }
    }
}
```

### 2.6 イミュータビリティ優先（Immutability by Default）

#### 2.6.1 なぜイミュータビリティが重要か

可変状態（Mutable State）は、以下の問題を引き起こす主要な原因である。

1. **競合状態（Race Condition）**: 複数スレッドが同じ変数を同時に変更
2. **予期しない副作用**: 関数が引数を変更してしまう
3. **推論の困難さ**: 変数の値がいつ変わるか追跡しにくい
4. **テストの困難さ**: 状態に依存するテストは順序依存になりやすい

```
可変 vs 不変の世界観:

  可変状態の世界                     不変状態の世界
  ┌────────────────────┐           ┌────────────────────┐
  │ 変数A ──→ 値1      │           │ 値A = 1 (固定)      │
  │   ↓ (代入)         │           │                    │
  │ 変数A ──→ 値2      │           │ 値B = f(値A) = 2   │
  │   ↓ (副作用)       │           │                    │
  │ 変数A ──→ 値3      │           │ 値C = g(値B) = 3   │
  │                    │           │                    │
  │ 「今の値は何？」     │           │ 「全ての値が追跡可能」│
  │  → デバッグが困難   │           │  → 推論が容易       │
  └────────────────────┘           └────────────────────┘
```

#### 2.6.2 各言語の不変性サポート

| 言語 | 不変宣言 | 可変宣言 | デフォルト | 深い不変性 |
|------|---------|---------|-----------|-----------|
| Rust | `let` | `let mut` | 不変 | 部分的（内部可変性） |
| Kotlin | `val` | `var` | - | `List` vs `MutableList` |
| Swift | `let` | `var` | - | 値型は深い不変性 |
| TypeScript | `const` | `let` | - | `readonly` / `as const` |
| Scala | `val` | `var` | - | 不変コレクション標準 |
| Haskell | (全て) | `IORef` | 不変 | 完全 |

---

## 3. 言語横断的な機能マップ

### 3.1 機能採用マトリクス

以下の表は、主要モダン言語がどの機能を採用しているかを示す。

| 機能 | Rust | Kotlin | Swift | TypeScript | Go | Python | C# |
|------|------|--------|-------|-----------|-----|--------|-----|
| 型推論 | ● | ● | ● | ● | ○ | ○ | ○ |
| Null安全 | ● | ● | ● | ● | △ | △ | ○ |
| パターンマッチ | ● | ● | ● | △ | △ | ○ | ● |
| async/await | ● | ● | ● | ● | ○* | ● | ● |
| ADT | ● | ● | ● | ○ | △ | △ | ○ |
| 不変デフォルト | ● | △ | △ | △ | △ | × | △ |
| クロージャ | ● | ● | ● | ● | ● | ● | ● |
| パッケージ管理 | ● | ● | ● | ● | ● | ● | ● |
| フォーマッタ | ● | ○ | ○ | ○ | ● | ○ | ○ |
| エラーメッセージ | ● | ○ | ○ | ○ | ○ | ○ | ○ |

凡例: ● = 充実 / ○ = あり / △ = 限定的 / × = なし / * = goroutine方式

---

## 4. クロージャとラムダ式

### 4.1 クロージャの本質

クロージャ（Closure）は、関数が定義された環境（レキシカルスコープ）の変数を「閉じ込める」（close over）ことで、関数を値として持ち運べる仕組みである。1958年の LISP に由来し、現在ではほぼ全てのモダン言語が第一級関数とクロージャをサポートしている。

```
クロージャの概念:

  ┌───────── 外側のスコープ ─────────┐
  │  let factor = 2;                 │
  │                                  │
  │  ┌───── クロージャ ──────┐       │
  │  │  |x| x * factor      │       │
  │  │  ↑ factor を捕捉      │       │
  │  └──────────────────────┘       │
  │                                  │
  └──────────────────────────────────┘

  クロージャが factor を「閉じ込める」ため、
  外側のスコープが終了しても factor にアクセスできる
```

### 4.2 Rustにおけるクロージャの所有権モデル

Rustのクロージャは所有権システムと統合されており、捕捉方法が3種類ある。

```rust
fn demonstrate_closure_capture() {
    let data = vec![1, 2, 3];

    // Fn: 不変参照で捕捉（共有借用）
    let print_data = || {
        println!("データ: {:?}", data);  // &data
    };
    print_data();
    print_data(); // 複数回呼べる

    // FnMut: 可変参照で捕捉（排他借用）
    let mut count = 0;
    let mut increment = || {
        count += 1;  // &mut count
        println!("カウント: {}", count);
    };
    increment();
    increment();

    // FnOnce: 所有権を移動して捕捉
    let name = String::from("Rust");
    let consume = move || {
        println!("消費: {}", name);  // name の所有権を移動
        drop(name);                  // 所有権を消費
    };
    consume();
    // consume(); // エラー: 既に消費済み
}
```

---

## 5. パッケージマネージャと統合ツールチェーン

### 5.1 パッケージマネージャの進化

パッケージマネージャは言語エコシステムの中核であり、モダン言語では公式ツールとして統合されている。

```
パッケージマネージャの世代:

  第1世代: 手動管理             第2世代: 外部ツール
  ┌──────────────────┐        ┌──────────────────┐
  │ ・手動ダウンロード  │        │ ・CPAN (Perl)    │
  │ ・Makefile        │  →→→   │ ・RubyGems       │
  │ ・#include        │        │ ・pip (Python)   │
  │ ・リンカ設定       │        │ ・npm (Node.js)  │
  └──────────────────┘        └──────────────────┘
         ↓                           ↓
  第3世代: 言語統合              第4世代: ワークスペース
  ┌──────────────────┐        ┌──────────────────┐
  │ ・Cargo (Rust)   │        │ ・Cargo workspace│
  │ ・go mod (Go)    │  →→→   │ ・pnpm workspace │
  │ ・Swift PM       │        │ ・Turborepo      │
  │ ・Mix (Elixir)   │        │ ・Nx             │
  └──────────────────┘        └──────────────────┘
```

### 5.2 ツールチェーン比較表

| 機能 | Rust (Cargo) | Go (go mod) | Swift (SPM) | Node (npm/pnpm) |
|------|-------------|-------------|-------------|-----------------|
| パッケージ管理 | Cargo.toml | go.mod | Package.swift | package.json |
| ビルド | `cargo build` | `go build` | `swift build` | `npm run build` |
| テスト | `cargo test` | `go test` | `swift test` | `npm test` |
| フォーマット | `cargo fmt` | `gofmt` | - | `prettier` |
| リンター | `cargo clippy` | `go vet` | `swiftlint` | `eslint` |
| ベンチマーク | `cargo bench` | `go test -bench` | - | 外部ツール |
| ドキュメント | `cargo doc` | `godoc` | DocC | `typedoc` |
| ロックファイル | Cargo.lock | go.sum | Package.resolved | package-lock.json |
| レジストリ | crates.io | proxy.golang.org | - | npmjs.com |
| ワークスペース | 対応 | 対応 | 対応 | 対応 (pnpm) |

---

## 6. エラーハンドリングの革新

### 6.1 例外からResult型へ

従来の例外ベースのエラーハンドリングには、以下の問題がある。

1. **制御フローの隠蔽**: どの関数が例外を投げるか型から判別できない
2. **パフォーマンスコスト**: 例外のスタックトレース構築は高コスト
3. **コンポーザビリティの低さ**: try-catch のネストは複雑化しやすい

```
エラーハンドリングの進化:

  レベル1: エラーコード      レベル2: 例外
  ┌──────────────────┐     ┌──────────────────┐
  │ int err = open() │     │ try {            │
  │ if (err < 0) {   │     │   file = open()  │
  │   // エラー処理   │     │ } catch (e) {    │
  │ }                │     │   // エラー処理   │
  │ // 忘れがち！     │     │ }                │
  └──────────────────┘     │ // 型情報なし     │
         ↓                 └──────────────────┘
                                   ↓
  レベル3: Result型           レベル4: 効果システム
  ┌──────────────────┐      ┌──────────────────┐
  │ let file =       │      │ fn open()        │
  │   open(path)?;   │      │   : IO + Error   │
  │ // 型で明示       │      │ // 副作用を型で   │
  │ // ?で早期リターン │      │ //   完全追跡     │
  └──────────────────┘      └──────────────────┘
```

### 6.2 Result型の言語間比較

```rust
// Rust: Result<T, E>
use std::fs;
use std::io;
use std::num::ParseIntError;

#[derive(Debug)]
enum AppError {
    IoError(io::Error),
    ParseError(ParseIntError),
    ValidationError(String),
}

impl From<io::Error> for AppError {
    fn from(e: io::Error) -> Self { AppError::IoError(e) }
}
impl From<ParseIntError> for AppError {
    fn from(e: ParseIntError) -> Self { AppError::ParseError(e) }
}

fn read_config_value(path: &str, key: &str) -> Result<i32, AppError> {
    let content = fs::read_to_string(path)?;   // io::Error → AppError
    let line = content
        .lines()
        .find(|l| l.starts_with(key))
        .ok_or_else(|| AppError::ValidationError(
            format!("キー '{}' が見つかりません", key)
        ))?;
    let value: i32 = line
        .split('=')
        .nth(1)
        .ok_or_else(|| AppError::ValidationError("不正な形式".into()))?
        .trim()
        .parse()?;   // ParseIntError → AppError
    Ok(value)
}
```

```kotlin
// Kotlin: sealed class による Result 相当
sealed class Result<out T> {
    data class Success<T>(val value: T) : Result<T>()
    data class Failure(val error: AppError) : Result<Nothing>()

    fun <R> map(transform: (T) -> R): Result<R> = when (this) {
        is Success -> Success(transform(value))
        is Failure -> this
    }

    fun <R> flatMap(transform: (T) -> Result<R>): Result<R> = when (this) {
        is Success -> transform(value)
        is Failure -> this
    }
}

// kotlin.Result も標準で用意されている
fun readConfigValue(path: String, key: String): Result<Int> {
    return try {
        val content = java.io.File(path).readText()
        val line = content.lines().find { it.startsWith(key) }
            ?: return Result.Failure(AppError.NotFound("キー '$key' が見つかりません"))
        val value = line.split("=")[1].trim().toInt()
        Result.Success(value)
    } catch (e: Exception) {
        Result.Failure(AppError.IoError(e.message ?: "不明なエラー"))
    }
}
```

---

## 7. 充実したエラーメッセージ

### 7.1 教育的コンパイラの登場

Rust と Elm は、エラーメッセージの品質で革命を起こした。従来のコンパイラが暗号的なエラーを出していたのに対し、これらの言語は「何が問題か」「なぜ問題か」「どう修正するか」の3点を明確に伝える。

```
従来のエラーメッセージ (C++):
┌───────────────────────────────────────────────────────┐
│ error: no matching function for call to               │
│ 'std::vector<std::__cxx11::basic_string<char,         │
│ std::char_traits<char>, std::allocator<char>>,         │
│ std::allocator<std::__cxx11::basic_string<char>>>::    │
│ push_back(int)'                                       │
│ → 何を言っているのか分からない                           │
└───────────────────────────────────────────────────────┘

Rustのエラーメッセージ:
┌───────────────────────────────────────────────────────┐
│ error[E0308]: mismatched types                        │
│  --> src/main.rs:5:20                                 │
│   |                                                   │
│ 5 |     names.push(42);                               │
│   |           ---- ^^ expected `String`, found `i32`  │
│   |           |                                       │
│   |           arguments to this method are incorrect  │
│   |                                                   │
│   = note: expected type `String`                      │
│              found type `i32`                         │
│   = help: try: `names.push(42.to_string())`           │
│                                                       │
│ → 問題箇所、原因、修正方法を明示                         │
└───────────────────────────────────────────────────────┘
```

---

## 8. トレンドの方向性

### 8.1 2010年代のトレンド

2010年代は「安全性の時代」と呼べる。以下の3つの大きな流れがあった。

**型安全性の強化:**
- TypeScript (2012): JavaScript に段階的型付けを追加
- Kotlin (2016): Java の問題点を解消した JVM 言語
- Rust (2015): メモリ安全性をコンパイル時に保証

**Null安全の普及:**
- 「10億ドルの間違い」の認識が広がり、主要言語が Null安全を導入
- Kotlin の `?` 記法、Swift の Optional、TypeScript の strictNullChecks

**並行処理の言語レベルサポート:**
- Go の goroutine + channel（CSPモデル）
- Rust の所有権による安全な並行性
- Kotlin の coroutine

### 8.2 2020年代のトレンド

**AI統合:**
- GitHub Copilot（2021）の登場で「AIと協調するコード」の時代へ
- 型情報が充実した言語ほど AI による補完精度が高い
- LLM による型推論の強化可能性

**エッジ / WebAssembly対応:**
- Wasm はブラウザ外でも実行される汎用バイナリ形式へ進化
- Rust、Go、C#、Kotlin が Wasm ターゲットをサポート
- エッジコンピューティングでの軽量ランタイム需要

**段階的型付けの成熟:**
- Python + mypy/Pyright: 動的言語に型を後付け
- Ruby + Sorbet/RBS: 漸進的型付けの導入
- PHP + PHPStan: 静的解析の高度化

**エラー回復パターンの普及:**
- Result/Either型の標準採用が進む
- Go 2.0 でのエラーハンドリング改善提案
- Java の sealed interface による直和型表現

### 8.3 全体の流れ

```
言語設計の進化の方向性:

  1960-1980        1980-2000        2000-2015        2015-現在
  ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐
  │ 自由  │  →→→  │ 構造  │  →→→  │ 安全  │  →→→  │ 生産性│
  │      │        │      │        │      │        │+AI協調│
  └──────┘        └──────┘        └──────┘        └──────┘

  ・アセンブリ      ・構造化         ・型安全性       ・AI補完
  ・GOTO自由       ・OOP           ・メモリ安全     ・段階的型付け
  ・型なし         ・例外処理       ・Null安全      ・Wasm
  ・手動メモリ     ・GC            ・Result型      ・効果システム

  「何でもできる」→「秩序を作る」→「安全を保証」→「賢く支援」
```

---

## 9. アンチパターン

### 9.1 アンチパターン1: 「全機能を使い倒す症候群」

モダン言語の機能は強力だが、全てを一度に適用しようとすると可読性が著しく低下する。

```rust
// --- 悪い例: 過度にメソッドチェーンを連結 ---
fn bad_example(data: &[Order]) -> HashMap<String, Vec<(String, f64)>> {
    data.iter()
        .filter(|o| o.status != OrderStatus::Cancelled)
        .filter(|o| o.created_at > Utc::now() - Duration::days(30))
        .map(|o| (o.category.clone(), o.items.clone()))
        .flat_map(|(cat, items)| items.into_iter().map(move |i| (cat.clone(), i)))
        .map(|(cat, item)| (cat, (item.name, item.price * (1.0 - item.discount))))
        .fold(HashMap::new(), |mut acc, (cat, item)| {
            acc.entry(cat).or_insert_with(Vec::new).push(item);
            acc
        })
}

// --- 良い例: 意味のある中間変数を使う ---
fn good_example(data: &[Order]) -> HashMap<String, Vec<(String, f64)>> {
    let thirty_days_ago = Utc::now() - Duration::days(30);

    let active_orders: Vec<&Order> = data.iter()
        .filter(|o| o.status != OrderStatus::Cancelled)
        .filter(|o| o.created_at > thirty_days_ago)
        .collect();

    let mut result: HashMap<String, Vec<(String, f64)>> = HashMap::new();

    for order in active_orders {
        for item in &order.items {
            let discounted_price = item.price * (1.0 - item.discount);
            result
                .entry(order.category.clone())
                .or_insert_with(Vec::new)
                .push((item.name.clone(), discounted_price));
        }
    }

    result
}
```

**教訓:** メソッドチェーンは3-4段までが適切。それ以上は中間変数を使って意図を明確にする。

### 9.2 アンチパターン2: 「`.unwrap()` 乱用によるNull安全の無力化」

Option型やResult型があっても、強制アンラップを多用すると安全性が失われる。

```rust
// --- 悪い例: unwrap() の乱用 ---
fn bad_process(config_path: &str) -> String {
    let content = std::fs::read_to_string(config_path).unwrap();  // パニック！
    let config: Config = serde_json::from_str(&content).unwrap(); // パニック！
    let db_url = config.database.unwrap().url.unwrap();           // パニック！
    let conn = Database::connect(&db_url).unwrap();               // パニック！
    conn.query("SELECT 1").unwrap().to_string()                   // パニック！
}

// --- 良い例: 適切なエラーハンドリング ---
fn good_process(config_path: &str) -> Result<String, AppError> {
    let content = std::fs::read_to_string(config_path)
        .map_err(|e| AppError::Io(format!("設定ファイル読み込み失敗: {}", e)))?;

    let config: Config = serde_json::from_str(&content)
        .map_err(|e| AppError::Parse(format!("JSON パース失敗: {}", e)))?;

    let db_url = config.database
        .as_ref()
        .and_then(|db| db.url.as_ref())
        .ok_or_else(|| AppError::Config("DB URLが設定されていません".into()))?;

    let conn = Database::connect(db_url)
        .map_err(|e| AppError::Database(format!("DB接続失敗: {}", e)))?;

    conn.query("SELECT 1")
        .map(|r| r.to_string())
        .map_err(|e| AppError::Database(format!("クエリ失敗: {}", e)))
}
```

**教訓:** `unwrap()` / `!!` / `!` の使用は、「ここで失敗するなら即座にプログラムを停止すべき」と確信できる場面に限定する。

### 9.3 アンチパターン3: 「型推論への過度な依存」

型推論があるからといって、一切の型注釈を書かないと、コードの可読性が低下する。

```typescript
// --- 悪い例: 型注釈が一切ない ---
const process = (data) =>
    data.filter(x => x.active).map(x => ({
        ...x,
        score: x.points * (x.bonus ? 1.5 : 1.0)
    })).sort((a, b) => b.score - a.score);

// --- 良い例: 関数境界に型注釈を付ける ---
interface User {
    name: string;
    active: boolean;
    points: number;
    bonus: boolean;
}

interface ScoredUser extends User {
    score: number;
}

const process = (data: User[]): ScoredUser[] =>
    data
        .filter(x => x.active)
        .map(x => ({
            ...x,
            score: x.points * (x.bonus ? 1.5 : 1.0)
        }))
        .sort((a, b) => b.score - a.score);
```

**教訓:** 「公開API」「関数シグネチャ」「複雑な型」には明示的な型注釈を付ける。ローカル変数は推論に任せてよい。

---

## 10. 演習問題

### 10.1 初級：概念の理解

**演習1:** 以下の各機能について、「なぜ必要か」を2-3文で説明せよ。
1. 型推論
2. Null安全
3. パターンマッチ
4. イミュータビリティ優先

**演習2:** 以下の機能がどの言語に由来するか、系譜を示せ。
- async/await
- Option/Maybe型
- 代数的データ型（ADT）

**演習3:** 以下のコードにはNull安全上の問題がある。修正せよ。

```typescript
function getCity(user: any): string {
    return user.address.city.toUpperCase();
}
```

**ヒント:** Optional chaining (`?.`)、nullish coalescing (`??`)、型定義を活用する。

### 10.2 中級：実装

**演習4:** 以下の要件を満たす型安全な設定ローダーを、任意のモダン言語で実装せよ。

要件:
- ファイルから設定を読み込む
- 必須キーと任意キーを型で区別する
- エラー時は具体的なエラーメッセージを返す（例外ではなくResult/Either型で）
- 環境変数によるオーバーライドをサポートする

```
設計のヒント:

  ┌─────────────────────────────────────────┐
  │ ConfigLoader                            │
  │ ┌─────────────────┐ ┌────────────────┐ │
  │ │ FileSource      │ │ EnvSource      │ │
  │ │ ・TOML/YAML読込  │ │ ・環境変数読込   │ │
  │ └────────┬────────┘ └───────┬────────┘ │
  │          └────────┬─────────┘          │
  │                   ▼                    │
  │          ┌────────────────┐            │
  │          │ Merge & Validate│            │
  │          └────────┬───────┘            │
  │                   ▼                    │
  │          Result<Config, ConfigError>    │
  └─────────────────────────────────────────┘
```

**演習5:** ECサイトの「買い物かご」の状態遷移を代数的データ型でモデリングせよ。

状態: 空のかご → 商品追加 → クーポン適用 → 注文確定 → 支払い完了

各状態遷移において:
- 空のかごから注文確定への直接遷移は不可能
- クーポンは1つまで適用可能
- 支払い完了後の状態変更は不可能

### 10.3 上級：設計と分析

**演習6:** 以下の3言語（Rust, Kotlin, TypeScript）で同じAPIクライアントを実装し、各言語の特徴を比較せよ。

```
要件:
┌─────────────────────────────────────────┐
│ APIクライアントの要件:                    │
│ 1. HTTPリクエスト送信（GET/POST）         │
│ 2. レスポンスのJSON/パース                │
│ 3. リトライロジック（最大3回）             │
│ 4. タイムアウト処理                      │
│ 5. エラーハンドリング                     │
│    - ネットワークエラー                   │
│    - パースエラー                         │
│    - APIエラー（4xx/5xx）                 │
│ 6. 型安全なレスポンス                     │
└─────────────────────────────────────────┘
```

比較すべき観点:
1. コード量
2. 型安全性の強さ
3. エラーハンドリングの明確さ
4. 非同期処理の記述の自然さ
5. テスタビリティ

**演習7:** 「もし2030年に新しい汎用プログラミング言語を設計するなら、どの機能を採用するか」を論じよ。

考慮すべき点:
- 本章で紹介した10大機能のうちどれを採用するか
- 既存の言語が解決できていない問題は何か
- AI時代のプログラミング言語に必要な要件は何か

---

## 11. モダン言語選択ガイド

### 11.1 ユースケース別推奨言語

```
ユースケース別の最適言語:

  ┌─────────────────────┬──────────────────────────────┐
  │ ユースケース         │ 推奨言語 (理由)              │
  ├─────────────────────┼──────────────────────────────┤
  │ システムプログラミング │ Rust (メモリ安全+性能)       │
  │ Web バックエンド     │ Go / Kotlin / TypeScript     │
  │ Web フロントエンド   │ TypeScript (事実上の標準)     │
  │ モバイル (iOS)      │ Swift                        │
  │ モバイル (Android)  │ Kotlin                       │
  │ モバイル (クロス)   │ Kotlin Multiplatform / Flutter│
  │ データサイエンス     │ Python (エコシステム)         │
  │ CLIツール           │ Rust / Go                    │
  │ インフラ・DevOps    │ Go                           │
  │ ゲーム開発          │ C# (Unity) / Rust (Bevy)     │
  │ 分散システム        │ Go / Erlang/Elixir           │
  │ 教育               │ Python / Haskell             │
  └─────────────────────┴──────────────────────────────┘
```

### 11.2 チーム規模と言語選択

小規模チーム（1-5人）では表現力の高い言語（Kotlin, Swift, TypeScript）が生産性を最大化する。大規模チーム（50人以上）では制約の強い言語（Rust, Go）がコードベースの一貫性を保ちやすい。中規模チーム（5-50人）ではプロジェクトの性質に応じた選択が重要である。

---

## 12. FAQ（よくある質問）

### Q1: 型推論があるなら、型注釈は一切不要ですか？

**A:** いいえ。型推論はローカル変数やラムダ式で効力を発揮するが、関数の公開インターフェース（引数型・戻り値型）には明示的な型注釈を付けるべきである。理由は3つある。第一に、コードの読み手に意図を伝える文書としての役割。第二に、コンパイラのエラーメッセージが明確になる。第三に、型推論の計算量を抑制でき、コンパイル速度が向上する。Rust や Kotlin が関数シグネチャの型注釈を必須としているのは、この設計判断の表れである。

### Q2: Null安全な言語を使えば、NullPointerException は完全になくなりますか？

**A:** 原理的にはイエスだが、実務上は注意点がある。Kotlin の `!!`（非null表明演算子）や Swift の `!`（強制アンラップ）を使えば実行時エラーは起こり得る。また、Java や Objective-C との相互運用時には、Null安全の保証が外れる領域がある。重要なのは、言語が「null の可能性」を型システムで明示的にし、開発者に意識的な判断を要求する点にある。`!!` を使うなら「ここは null にならないと確信している」という意思表示になり、コードレビューで検出しやすくなる。

### Q3: Go には型推論もパターンマッチも限定的ですが、なぜモダン言語に分類されるのですか？

**A:** Go は意図的に機能を制限している。Rob Pike は「Go は機能を追加する言語ではなく、機能を削る言語だ」と述べている。Go のモダン性は、goroutine による並行処理の言語レベルサポート、`gofmt` による統一フォーマット、`go mod` による依存管理、充実したツールチェーン、速いコンパイル速度にある。これらは「少ない機能で高い生産性」というアプローチであり、大規模チームでのコードベース管理において特に効果を発揮する。Go 1.18 でジェネリクスが追加され、Go 1.21 以降も段階的に機能が拡充されている。

### Q4: async/await はどの言語の実装が最も優れていますか？

**A:** 一概には言えないが、各言語の特徴を整理する。Rust の async/await はゼロコスト抽象化を実現し、ランタイムを選択できる柔軟性があるが、Pin や Lifetime との組み合わせで学習コストが高い。Kotlin の coroutine は構造化並行性が美しく、キャンセレーションの伝播が自然。TypeScript/JavaScript の async/await はシンプルで直感的だが、シングルスレッドのイベントループ上で動作するため CPU バウンドな処理には向かない。Swift の async/await は Actor モデルとの統合が先進的。用途に応じて最適な実装は異なる。

### Q5: 「イミュータビリティ優先」は本当にパフォーマンスに影響しないのですか？

**A:** ナイーブに不変データ構造を使うとコピーコストが発生するが、モダン言語は様々な最適化でこれを軽減している。Rust はムーブセマンティクスにより「最後の使用者がデータを再利用」できる。関数型データ構造（永続データ構造）は構造共有によりコピーを最小化する。また、コンパイラの最適化（コピー省略、インライン化）により、実行時のオーバーヘッドは多くの場合無視できる水準に収まる。並行処理では不変データのほうがロック不要で高速になる場面も多い。

---

## 13. まとめ

### 13.1 機能由来の一覧表

| 機能 | 由来 | 年代 | 採用した主要言語 |
|------|------|------|----------------|
| 型推論 | ML → Hindley-Milner | 1973 | Rust, TS, Go, Kotlin, Swift, Haskell |
| Null安全 | Haskell (Maybe) | 1990 | Rust, Kotlin, Swift, TS, C# |
| パターンマッチ | ML | 1973 | Rust, Scala, Python 3.10, C# 8, Kotlin |
| async/await | C# | 2012 | JS, Python, Rust, Kotlin, Swift |
| ADT | ML → Haskell | 1973/1990 | Rust, TS, Swift, Kotlin, Scala |
| 不変デフォルト | Haskell → Erlang | 1990 | Rust, Kotlin, Swift, Scala |
| クロージャ | LISP | 1958 | 全モダン言語 |
| パッケージ管理 | CPAN (Perl) | 1995 | 全モダン言語 |
| フォーマッタ | gofmt (Go) | 2009 | Rust, Zig, Deno |
| 教育的エラー | Elm → Rust | 2012/2015 | Rust, Elm, Gleam |

### 13.2 核心メッセージ

```
モダン言語の本質:

  ┌──────────────────────────────────────────────┐
  │                                              │
  │  「バグを見つける」から「バグを防ぐ」へ          │
  │                                              │
  │  ・型推論       → 型の恩恵を低コストで享受       │
  │  ・Null安全     → NullPointerException を根絶  │
  │  ・パターンマッチ → 網羅性をコンパイラが保証      │
  │  ・ADT          → 不正な状態を型で表現不能にする  │
  │  ・Result型     → エラーを無視できない構造にする  │
  │  ・不変デフォルト → 状態変更による事故を削減       │
  │                                              │
  │  共通原理: 「コンパイル時に検出できるものは、      │
  │            実行時まで持ち越さない」              │
  │                                              │
  └──────────────────────────────────────────────┘
```

---

## 次に読むべきガイド

- [[02-dsl-and-metaprogramming.md]] - DSLとメタプログラミング
- [[03-future-of-languages.md]] - プログラミング言語の未来

---

## 参考文献

1. Pierce, B.C. "Types and Programming Languages." MIT Press, 2002. - 型システムの理論的基礎を網羅した名著。型推論アルゴリズム（Hindley-Milner）の詳細な解説を含む。
2. Hoare, C.A.R. "Null References: The Billion Dollar Mistake." QCon London, 2009. - Null参照の発明者自身による反省的講演。Null安全の必要性を理解する上で必読。
3. Matsakis, N. and Klock, F. "The Rust Programming Language." No Starch Press, 2019. - Rustの所有権システム、パターンマッチ、Result型を含む包括的解説。モダン言語機能の実例集として優秀。
4. Odersky, M., Spoon, L., and Venners, B. "Programming in Scala." Artima Press, 2021. - 関数型プログラミングとOOPの融合、ADTの実践的活用を解説。
5. Bloch, J. "Effective Java." 3rd Edition, Addison-Wesley, 2018. - Java におけるモダンなプログラミング慣行。イミュータビリティの重要性を論じた章は言語を問わず参考になる。

---

## 用語集

| 用語 | 説明 |
|------|------|
| 型推論 (Type Inference) | コンパイラが文脈から型を自動決定する機構 |
| Null安全 (Null Safety) | Null参照によるランタイムエラーを型で防止する仕組み |
| パターンマッチ (Pattern Matching) | データ構造を分解しながら条件分岐する機構 |
| 代数的データ型 (ADT) | 直和型と直積型を組み合わせたデータ型 |
| 直和型 (Sum Type) | 複数の型のいずれかを取る型（OR結合） |
| 直積型 (Product Type) | 複数の型を同時に持つ型（AND結合） |
| イミュータビリティ (Immutability) | 値が作成後に変更されない性質 |
| クロージャ (Closure) | 環境を捕捉する関数オブジェクト |
| 段階的型付け (Gradual Typing) | 静的型付けと動的型付けを混在させる手法 |
| 構造化並行性 (Structured Concurrency) | 並行タスクのライフタイムを構造的に管理する手法 |
| 収斂進化 (Convergent Evolution) | 異なる系統が独立に類似した特徴を発達させる現象 |
| ゼロコスト抽象化 (Zero-Cost Abstraction) | 抽象化が実行時のオーバーヘッドを生じない設計 |
| 網羅性検査 (Exhaustiveness Check) | パターンマッチが全ケースを網羅しているかの静的検証 |
