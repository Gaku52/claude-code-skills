# 分岐とループ

> 制御フローはプログラムの「実行順序を変える」仕組み。分岐とループの設計は言語の哲学を反映する。

## この章で学ぶこと

- [ ] 各言語の分岐構文の違いと設計思想を理解する
- [ ] ループの種類と使い分けを把握する
- [ ] 式ベースの制御フローを理解する
- [ ] 早期リターンとガード節を適切に使いこなせる
- [ ] ループ最適化のテクニックを把握する
- [ ] 制御フローの設計判断を言語横断的に比較できる

---

## 1. 分岐（Branching）

### 1.1 if 文 vs if 式

プログラミング言語における `if` の設計は、「文（statement）」として扱うか「式（expression）」として扱うかで大きく異なる。文は副作用のために実行され、式は値を返す。

```python
# Python: if は文（statement）
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

# 三項演算子（式）— 条件式
grade = "A" if score >= 90 else "B" if score >= 80 else "C"

# 三項演算子の実務的な使い方
status = "active" if user.is_verified else "pending"
display_name = user.nickname if user.nickname else user.email
max_val = a if a > b else b

# 複雑な条件は三項演算子を避け、通常の if 文を使う
# ❌ 読みにくい
result = "A" if x > 90 else "B" if x > 80 else "C" if x > 70 else "D" if x > 60 else "F"

# ✅ 読みやすい
if x > 90:
    result = "A"
elif x > 80:
    result = "B"
elif x > 70:
    result = "C"
elif x > 60:
    result = "D"
else:
    result = "F"
```

```rust
// Rust: if は式（expression）→ 値を返す
let grade = if score >= 90 {
    "A"
} else if score >= 80 {
    "B"
} else if score >= 70 {
    "C"
} else if score >= 60 {
    "D"
} else {
    "F"
};  // セミコロンで束縛

// 1行で使える（短い条件の場合）
let abs_val = if x >= 0 { x } else { -x };
let min_val = if a < b { a } else { b };

// ブロック内で複数の文を含む場合、最後の式が値になる
let description = if score >= 90 {
    let prefix = "Excellent";
    let emoji = "🌟";
    format!("{} {}", prefix, emoji)  // これが返る値
} else {
    "Keep trying".to_string()
};

// if let — Option や Result のパターンマッチ簡略版
let config_value = if let Some(val) = config.get("timeout") {
    val.parse::<u64>().unwrap_or(30)
} else {
    30  // デフォルト値
};
```

```kotlin
// Kotlin: if も when も式
val grade = if (score >= 90) "A" else if (score >= 80) "B" else "C"

// 複数行の場合、最後の式が値になる
val result = if (score >= 90) {
    println("Great job!")
    "A"  // この値が返る
} else {
    println("Keep going")
    "B"
}

// when 式（Kotlin 独自の強力な分岐）
val result = when {
    score >= 90 -> "A"
    score >= 80 -> "B"
    score >= 70 -> "C"
    score >= 60 -> "D"
    else -> "F"
}

// when を値ベースで使用
val typeDescription = when (val day = getDayOfWeek()) {
    "Mon", "Tue", "Wed", "Thu", "Fri" -> "Weekday"
    "Sat", "Sun" -> "Weekend"
    else -> "Unknown"
}

// when で型チェック
fun describe(obj: Any): String = when (obj) {
    is Int -> "Integer: $obj"
    is String -> "String of length ${obj.length}"
    is List<*> -> "List of size ${obj.size}"
    else -> "Unknown type"
}

// when で範囲チェック
val category = when (age) {
    in 0..12 -> "Child"
    in 13..17 -> "Teen"
    in 18..64 -> "Adult"
    in 65..Int.MAX_VALUE -> "Senior"
    else -> "Invalid"
}
```

```swift
// Swift: if は文だが、if let / guard let が強力
let score = 85

// if let（Optional バインディング）
if let username = optionalUsername {
    print("Hello, \(username)")
} else {
    print("No username")
}

// guard let（早期リターン向け）
func processUser(_ user: User?) -> String {
    guard let user = user else {
        return "No user"
    }
    guard user.isActive else {
        return "Inactive user"
    }
    return "Processing \(user.name)"
}

// Swift 5.9+: if 式として使用可能
let grade = if score >= 90 { "A" } else if score >= 80 { "B" } else { "C" }
```

```scala
// Scala: if は式（常に値を返す）
val grade = if (score >= 90) "A"
            else if (score >= 80) "B"
            else if (score >= 70) "C"
            else "F"

// ブロック式
val result = if (condition) {
  val computed = heavyComputation()
  computed * 2  // 最後の式が返る値
} else {
  defaultValue
}
```

```haskell
-- Haskell: if は式（else が必須）
grade = if score >= 90 then "A"
        else if score >= 80 then "B"
        else if score >= 70 then "C"
        else "F"

-- ガード（関数定義での条件分岐）
bmiCategory bmi
  | bmi < 18.5 = "Underweight"
  | bmi < 25.0 = "Normal"
  | bmi < 30.0 = "Overweight"
  | otherwise   = "Obese"
```

### 1.2 文 vs 式の設計哲学比較

```
文ベース（Statement-based）の言語:
  C, Java, Python, JavaScript, Go
  → if は値を返さない。副作用（代入など）で結果を伝える
  → 三項演算子（? :）が式としての分岐を補完

式ベース（Expression-based）の言語:
  Rust, Kotlin, Scala, Haskell, OCaml, F#, Elixir
  → if は値を返す。変数束縛と自然に組み合わせられる
  → 「全ての構文が値を返す」一貫性

式ベースの利点:
  1. 変数の不変性を保ちやすい（let x = if ... で一度だけ代入）
  2. 初期化忘れがない（else がないとコンパイルエラー）
  3. 関数型スタイルとの親和性が高い
  4. 型推論が効きやすい

文ベースの利点:
  1. 馴染みやすい（C から続く伝統）
  2. 副作用を明示的に分離できる
  3. void（値なし）の分岐が自然
```

### 1.3 switch / match

```javascript
// JavaScript: switch文（fall-through に注意）
switch (day) {
    case "Mon": case "Tue": case "Wed":
    case "Thu": case "Fri":
        type = "Weekday";
        break;      // break 忘れ → fall-through（バグの温床）
    case "Sat": case "Sun":
        type = "Weekend";
        break;
    default:
        type = "Unknown";
}

// switch の実務的な使用例: HTTPステータスコードの分類
function categorizeStatus(code) {
    switch (true) {
        case code >= 200 && code < 300:
            return "Success";
        case code >= 300 && code < 400:
            return "Redirect";
        case code >= 400 && code < 500:
            return "Client Error";
        case code >= 500:
            return "Server Error";
        default:
            return "Informational";
    }
}

// switch の代替: オブジェクトマップ（推奨パターン）
const statusMessages = {
    200: "OK",
    201: "Created",
    204: "No Content",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    500: "Internal Server Error",
};
const message = statusMessages[code] ?? "Unknown Status";
```

```typescript
// TypeScript: switch で型の絞り込み（Narrowing）
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; base: number; height: number };

function area(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            // ここでは shape は { kind: "circle"; radius: number }
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            // ここでは shape は { kind: "rectangle"; width: number; height: number }
            return shape.width * shape.height;
        case "triangle":
            return (shape.base * shape.height) / 2;
        default:
            // 網羅性チェック（never 型）
            const _exhaustive: never = shape;
            return _exhaustive;
    }
}
```

```rust
// Rust: match式（網羅性チェック + パターンマッチ）
let type_str = match day {
    "Mon" | "Tue" | "Wed" | "Thu" | "Fri" => "Weekday",
    "Sat" | "Sun" => "Weekend",
    _ => "Unknown",
};
// 全パターンを網羅しないとコンパイルエラー
// fall-through なし（安全）

// match で複雑な条件分岐
let message = match status_code {
    200 => "OK",
    201 => "Created",
    204 => "No Content",
    301 | 302 => "Redirect",
    400 => "Bad Request",
    401 => "Unauthorized",
    403 => "Forbidden",
    404 => "Not Found",
    405 => "Method Not Allowed",
    500 => "Internal Server Error",
    502 | 503 => "Service Unavailable",
    code @ 100..=199 => "Informational",
    code @ 200..=299 => "Success",
    code @ 300..=399 => "Redirection",
    code @ 400..=499 => "Client Error",
    code @ 500..=599 => "Server Error",
    _ => "Unknown",
};
```

```go
// Go: switch（break不要、fall-through は明示的）
switch day {
case "Mon", "Tue", "Wed", "Thu", "Fri":
    typeStr = "Weekday"
case "Sat", "Sun":
    typeStr = "Weekend"
default:
    typeStr = "Unknown"
}
// break 不要（自動的に抜ける）
// fallthrough キーワードで明示的に fall-through

// 条件式なしの switch（if-else チェーンの代替）
switch {
case score >= 90:
    grade = "A"
case score >= 80:
    grade = "B"
case score >= 70:
    grade = "C"
case score >= 60:
    grade = "D"
default:
    grade = "F"
}

// 型 switch（Go のインターフェース型アサーション）
func describe(i interface{}) string {
    switch v := i.(type) {
    case int:
        return fmt.Sprintf("Integer: %d", v)
    case string:
        return fmt.Sprintf("String: %s", v)
    case bool:
        return fmt.Sprintf("Boolean: %t", v)
    case []int:
        return fmt.Sprintf("Int slice of length %d", len(v))
    default:
        return fmt.Sprintf("Unknown type: %T", v)
    }
}
```

```c
// C: switch文（整数型のみ、fall-through がデフォルト）
switch (command) {
    case CMD_START:
        initialize();
        break;
    case CMD_STOP:
        cleanup();
        break;
    case CMD_PAUSE:
    case CMD_SUSPEND:  // fall-through（意図的）
        pause();
        break;
    default:
        fprintf(stderr, "Unknown command: %d\n", command);
        break;
}

// C の switch の制限
// - 整数型（int, char, enum）のみ
// - 文字列の比較はできない
// - 範囲指定はできない（GCC拡張を除く）
```

```java
// Java: switch 式（Java 14+）
// 従来の switch 文
String typeStr;
switch (day) {
    case "Mon": case "Tue": case "Wed":
    case "Thu": case "Fri":
        typeStr = "Weekday";
        break;
    case "Sat": case "Sun":
        typeStr = "Weekend";
        break;
    default:
        typeStr = "Unknown";
}

// Java 14+: switch 式（アロー構文）
String typeStr = switch (day) {
    case "Mon", "Tue", "Wed", "Thu", "Fri" -> "Weekday";
    case "Sat", "Sun" -> "Weekend";
    default -> "Unknown";
};

// Java 14+: switch 式でブロック使用（yield で値を返す）
String result = switch (statusCode) {
    case 200 -> "OK";
    case 404 -> "Not Found";
    default -> {
        logger.warn("Unexpected status: " + statusCode);
        yield "Unknown";
    }
};

// Java 21+: パターンマッチング switch
String describe(Object obj) {
    return switch (obj) {
        case Integer i when i > 0 -> "Positive integer: " + i;
        case Integer i -> "Non-positive integer: " + i;
        case String s -> "String: " + s;
        case null -> "null";
        default -> "Unknown: " + obj;
    };
}
```

### 1.4 条件演算子と条件式

```python
# Python: 条件式（三項演算子相当）
result = value_if_true if condition else value_if_false

# 実務例
display = f"{count} item{'s' if count != 1 else ''}"
log_level = "DEBUG" if is_development else "INFO"
timeout = custom_timeout if custom_timeout is not None else default_timeout

# Python: Walrus演算子（:=）— Python 3.8+
# 代入と条件判定を同時に行う
if (n := len(data)) > 10:
    print(f"Data too long: {n}")

# while ループでの活用
while (line := file.readline()) != "":
    process(line)

# リスト内包表記での活用
results = [y for x in data if (y := expensive_computation(x)) is not None]
```

```c
// C / C++ / JavaScript / Java: 三項演算子
int abs_val = (x >= 0) ? x : -x;
const char* msg = (err == 0) ? "Success" : "Error";

// ネストした三項演算子（非推奨 — 読みにくい）
const char* grade = (score >= 90) ? "A"
                  : (score >= 80) ? "B"
                  : (score >= 70) ? "C"
                  : "F";
```

```ruby
# Ruby: 多彩な条件式
# if 修飾子（後置 if）
puts "Adult" if age >= 18
log.warn("Low memory") if memory_usage > 0.9

# unless（否定条件）
raise "Not found" unless record
send_notification unless user.opted_out?

# 三項演算子
status = active? ? "Active" : "Inactive"

# case-when（パターンマッチ風）
result = case score
         when 90..100 then "A"
         when 80..89  then "B"
         when 70..79  then "C"
         when 60..69  then "D"
         else              "F"
         end

# case-in（Ruby 3.0+ パターンマッチ）
case user
in { name: String => name, age: (18..) => age }
  puts "#{name} is an adult (#{age})"
in { name: String => name, age: }
  puts "#{name} is a minor (#{age})"
end
```

---

## 2. ループ

### 2.1 for ループの進化

```c
// C: 古典的な for ループ
for (int i = 0; i < 10; i++) {
    printf("%d\n", i);
}

// C: 配列の走査（インデックスベース）
int arr[] = {10, 20, 30, 40, 50};
int len = sizeof(arr) / sizeof(arr[0]);
for (int i = 0; i < len; i++) {
    printf("arr[%d] = %d\n", i, arr[i]);
}

// C: 逆順走査
for (int i = len - 1; i >= 0; i--) {
    printf("arr[%d] = %d\n", i, arr[i]);
}

// C: 2重ループ（行列処理）
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        matrix[i][j] = i * cols + j;
    }
}
```

```python
# Python: for-in（イテレータベース）
for i in range(10):
    print(i)

for item in collection:
    print(item)

# enumerate（インデックス付き）
for i, item in enumerate(collection):
    print(f"{i}: {item}")

# enumerate の開始インデックス指定
for i, line in enumerate(lines, start=1):
    print(f"Line {i}: {line}")

# zip（並行イテレーション）
for name, age in zip(names, ages):
    print(f"{name}: {age}")

# zip_longest（長さが異なるイテラブルの結合）
from itertools import zip_longest
for a, b in zip_longest([1, 2, 3], [10, 20], fillvalue=0):
    print(f"{a}, {b}")  # (1,10), (2,20), (3,0)

# reversed（逆順）
for item in reversed(collection):
    print(item)

# sorted（ソート順）
for item in sorted(collection, key=lambda x: x.name):
    print(item)

# itertools の活用
from itertools import product, combinations, permutations, chain

# デカルト積（全ての組み合わせ）
for x, y in product(range(3), range(3)):
    print(f"({x}, {y})")

# 組み合わせ
for a, b in combinations([1, 2, 3, 4], 2):
    print(f"{a}, {b}")  # (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)

# 順列
for a, b in permutations([1, 2, 3], 2):
    print(f"{a}, {b}")  # (1,2), (1,3), (2,1), (2,3), (3,1), (3,2)

# チェーン（複数のイテラブルを連結）
for item in chain([1, 2], [3, 4], [5, 6]):
    print(item)  # 1, 2, 3, 4, 5, 6

# 辞書の走査
for key, value in config.items():
    print(f"{key} = {value}")

# 辞書内包表記（ループ + 変換）
squared = {x: x**2 for x in range(10)}
filtered = {k: v for k, v in data.items() if v > threshold}
```

```rust
// Rust: for-in（所有権を意識）
for item in &collection {     // 不変借用（コレクション保持）
    println!("{}", item);
}

for item in &mut collection { // 可変借用（要素を変更）
    *item += 1;
}

for item in collection {      // ムーブ（コレクション消費）
    println!("{}", item);
}
// collection はもう使えない

// レンジ
for i in 0..10 {        // 0〜9
    println!("{}", i);
}
for i in 0..=10 {       // 0〜10（inclusive）
    println!("{}", i);
}

// ステップ（step_by）
for i in (0..100).step_by(5) {
    println!("{}", i);  // 0, 5, 10, ..., 95
}

// 逆順
for i in (0..10).rev() {
    println!("{}", i);  // 9, 8, 7, ..., 0
}

// enumerate（インデックス付き）
for (i, item) in collection.iter().enumerate() {
    println!("{}: {}", i, item);
}

// zip（並行イテレーション）
for (name, age) in names.iter().zip(ages.iter()) {
    println!("{}: {}", name, age);
}

// windows と chunks
let data = vec![1, 2, 3, 4, 5, 6, 7, 8];

// スライディングウィンドウ
for window in data.windows(3) {
    println!("{:?}", window);  // [1,2,3], [2,3,4], [3,4,5], ...
}

// チャンク分割
for chunk in data.chunks(3) {
    println!("{:?}", chunk);  // [1,2,3], [4,5,6], [7,8]
}
```

```go
// Go: for だけ（while も loop も for で表現）
for i := 0; i < 10; i++ {  // C風 for
    fmt.Println(i)
}

for condition {              // while 相当
    // ...
}

for {                        // 無限ループ
    // ...
}

for i, v := range slice {   // for-range（スライス）
    fmt.Println(i, v)
}

// for-range（マップ）
for key, value := range myMap {
    fmt.Printf("%s: %v\n", key, value)
}

// for-range（文字列）— rune（Unicodeコードポイント）単位
for i, r := range "Hello, 世界" {
    fmt.Printf("index=%d, rune=%c\n", i, r)
}

// for-range（チャネル）
for msg := range ch {
    fmt.Println("Received:", msg)
}

// インデックスのみ（値を捨てる）
for i := range slice {
    fmt.Println(i)
}

// 値のみ（インデックスを捨てる）
for _, v := range slice {
    fmt.Println(v)
}
```

```javascript
// JavaScript: 4種類の for ループ

// 1. 古典的 for
for (let i = 0; i < 10; i++) {
    console.log(i);
}

// 2. for...in（オブジェクトのキーを走査 — 配列には非推奨）
for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
        console.log(`${key}: ${obj[key]}`);
    }
}

// 3. for...of（イテラブルの値を走査 — ES6+）
for (const item of array) {
    console.log(item);
}

// 4. forEach メソッド（配列専用）
array.forEach((item, index) => {
    console.log(`${index}: ${item}`);
});

// for...of の応用
// Map の走査
for (const [key, value] of map) {
    console.log(`${key}: ${value}`);
}

// Set の走査
for (const item of set) {
    console.log(item);
}

// entries()（インデックス付き）
for (const [i, item] of array.entries()) {
    console.log(`${i}: ${item}`);
}

// 注意: for...in vs for...of
// for...in: プロトタイプチェーンのプロパティも列挙（危険）
// for...of: Symbol.iterator プロトコルに従う（安全）
```

### 2.2 while と loop

```rust
// Rust: loop（無限ループ、値を返せる）
let result = loop {
    let input = get_input();
    if input.is_valid() {
        break input.value();  // break で値を返す
    }
    println!("Invalid input, try again");
};

// ラベル付きループ（ネストしたループの制御）
'outer: for i in 0..10 {
    for j in 0..10 {
        if i + j > 15 {
            break 'outer;  // 外側のループを脱出
        }
        if j % 2 == 0 {
            continue;  // 内側のループの次の反復へ
        }
        println!("{}, {}", i, j);
    }
}

// while let（パターンマッチ付き）
while let Some(item) = iterator.next() {
    println!("{}", item);
}

// while let チェーン（Rust 1.64+）
while let Some(item) = stack.pop() {
    match item {
        Item::Value(v) => results.push(v),
        Item::SubList(list) => {
            for sub_item in list.into_iter().rev() {
                stack.push(sub_item);
            }
        }
    }
}

// loop + match パターン（状態機械の実装）
enum State { Init, Running, Paused, Done }

let mut state = State::Init;
loop {
    state = match state {
        State::Init => {
            initialize();
            State::Running
        }
        State::Running => {
            if should_pause() {
                State::Paused
            } else if is_complete() {
                State::Done
            } else {
                process_next();
                State::Running
            }
        }
        State::Paused => {
            wait_for_resume();
            State::Running
        }
        State::Done => break,
    };
}
```

```python
# Python: while ループ

# 基本的な while
count = 0
while count < 10:
    print(count)
    count += 1

# while + else（ループが正常終了した場合に else が実行される）
def find_item(items, target):
    i = 0
    while i < len(items):
        if items[i] == target:
            print(f"Found at index {i}")
            break
        i += 1
    else:
        # break で抜けなかった場合に実行
        print("Not found")

# do-while 相当（Python には do-while がない）
while True:
    user_input = input("Enter a number (0 to quit): ")
    if user_input == "0":
        break
    process(user_input)

# イテレータを while で消費
it = iter(data)
while (chunk := list(islice(it, 100))):
    process_batch(chunk)
```

```go
// Go: for だけで全てのループを表現

// while 相当
for condition {
    // ...
}

// do-while 相当
for {
    doSomething()
    if !condition {
        break
    }
}

// ラベル付きループ
OuterLoop:
    for i := 0; i < 10; i++ {
        for j := 0; j < 10; j++ {
            if i+j > 15 {
                break OuterLoop
            }
        }
    }
```

```java
// Java: do-while（少なくとも1回実行）
Scanner scanner = new Scanner(System.in);
String input;
do {
    System.out.print("Enter command: ");
    input = scanner.nextLine();
    processCommand(input);
} while (!input.equals("quit"));

// Java: 拡張 for ループ（for-each）
for (String item : collection) {
    System.out.println(item);
}

// Java: ラベル付き break/continue
outer:
for (int i = 0; i < matrix.length; i++) {
    for (int j = 0; j < matrix[i].length; j++) {
        if (matrix[i][j] == target) {
            System.out.printf("Found at (%d, %d)%n", i, j);
            break outer;
        }
    }
}
```

### 2.3 イテレータメソッド（関数型スタイル）

```typescript
// TypeScript: メソッドチェーン
const result = numbers
    .filter(n => n > 0)
    .map(n => n * 2)
    .reduce((sum, n) => sum + n, 0);

// vs 命令型
let result = 0;
for (const n of numbers) {
    if (n > 0) {
        result += n * 2;
    }
}

// 実務的なメソッドチェーンの例
interface User {
    name: string;
    age: number;
    department: string;
    isActive: boolean;
}

// 部門ごとのアクティブユーザー数を集計
const departmentCounts = users
    .filter(user => user.isActive)
    .reduce((acc, user) => {
        acc[user.department] = (acc[user.department] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);

// グループ化（Object.groupBy — ES2024）
const grouped = Object.groupBy(users, user => user.department);

// flatMap の活用
const allTags = articles
    .flatMap(article => article.tags)
    .filter((tag, i, arr) => arr.indexOf(tag) === i);  // 重複除去

// find と findIndex
const firstAdmin = users.find(u => u.role === "admin");
const adminIndex = users.findIndex(u => u.role === "admin");

// some と every
const hasAdmin = users.some(u => u.role === "admin");
const allActive = users.every(u => u.isActive);
```

```rust
// Rust: イテレータ（ゼロコスト抽象化）
let result: i32 = numbers.iter()
    .filter(|&&n| n > 0)
    .map(|&n| n * 2)
    .sum();
// コンパイル後は手書きのループと同等の性能

// 実務的なイテレータの例
struct Employee {
    name: String,
    department: String,
    salary: u64,
}

// 部門ごとの平均給与
let dept_averages: HashMap<String, f64> = employees.iter()
    .fold(HashMap::new(), |mut acc, emp| {
        let entry = acc.entry(emp.department.clone())
            .or_insert((0u64, 0u64));
        entry.0 += emp.salary;
        entry.1 += 1;
        acc
    })
    .into_iter()
    .map(|(dept, (total, count))| {
        (dept, total as f64 / count as f64)
    })
    .collect();

// partition: 条件で2つに分割
let (evens, odds): (Vec<i32>, Vec<i32>) = numbers.iter()
    .partition(|&&n| n % 2 == 0);

// unzip: ペアのイテレータを2つのコレクションに分割
let (names, ages): (Vec<&str>, Vec<u32>) = people.iter()
    .map(|p| (p.name.as_str(), p.age))
    .unzip();

// チェーンの遅延評価を活用した効率的な検索
let first_match = huge_dataset.iter()
    .filter(|item| expensive_check(item))
    .map(|item| transform(item))
    .next();  // 最初の1つだけ計算（残りは評価されない）
```

```python
# Python: 内包表記（Pythonic なループ）

# リスト内包表記
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]

# ネストした内包表記
pairs = [(x, y) for x in range(3) for y in range(3) if x != y]

# 辞書内包表記
word_lengths = {word: len(word) for word in words}

# 集合内包表記
unique_lengths = {len(word) for word in words}

# ジェネレータ式（メモリ効率が良い）
total = sum(x**2 for x in range(1000000))

# map, filter, reduce（関数型スタイル）
from functools import reduce

squared = list(map(lambda x: x**2, numbers))
positive = list(filter(lambda x: x > 0, numbers))
total = reduce(lambda acc, x: acc + x, numbers, 0)

# 実務では内包表記の方が Pythonic
# map/filter よりリスト内包表記が推奨される
squared = [x**2 for x in numbers]              # map 相当
positive = [x for x in numbers if x > 0]       # filter 相当
```

---

## 3. 早期リターンとガード節

### 3.1 ガード節パターン

```rust
// ガード節パターン（ネストを減らす）
// ❌ ネストが深い
fn process(input: Option<&str>) -> Result<String, Error> {
    if let Some(s) = input {
        if !s.is_empty() {
            if s.len() < 100 {
                Ok(s.to_uppercase())
            } else {
                Err(Error::TooLong)
            }
        } else {
            Err(Error::Empty)
        }
    } else {
        Err(Error::Missing)
    }
}

// ✅ ガード節で早期リターン
fn process(input: Option<&str>) -> Result<String, Error> {
    let s = input.ok_or(Error::Missing)?;
    if s.is_empty() { return Err(Error::Empty); }
    if s.len() >= 100 { return Err(Error::TooLong); }
    Ok(s.to_uppercase())
}
```

```python
# Python: ガード節

# ❌ ネストが深い
def process_order(order):
    if order is not None:
        if order.is_valid():
            if order.has_items():
                if order.payment_verified():
                    return ship_order(order)
                else:
                    return {"error": "Payment not verified"}
            else:
                return {"error": "No items"}
        else:
            return {"error": "Invalid order"}
    else:
        return {"error": "No order"}

# ✅ ガード節で早期リターン
def process_order(order):
    if order is None:
        return {"error": "No order"}
    if not order.is_valid():
        return {"error": "Invalid order"}
    if not order.has_items():
        return {"error": "No items"}
    if not order.payment_verified():
        return {"error": "Payment not verified"}
    return ship_order(order)
```

```go
// Go: ガード節が標準スタイル
func processUser(userID string) (*User, error) {
    if userID == "" {
        return nil, fmt.Errorf("empty user ID")
    }

    user, err := db.FindUser(userID)
    if err != nil {
        return nil, fmt.Errorf("find user %s: %w", userID, err)
    }

    if !user.IsActive {
        return nil, fmt.Errorf("user %s is not active", userID)
    }

    if user.IsLocked {
        return nil, fmt.Errorf("user %s is locked", userID)
    }

    return user, nil
}
```

```typescript
// TypeScript: ガード節 + 型の絞り込み
function processInput(input: unknown): string {
    if (input === null || input === undefined) {
        return "No input";
    }
    if (typeof input !== "string") {
        return "Not a string";
    }
    // ここで input は string 型に絞り込まれている
    if (input.length === 0) {
        return "Empty string";
    }
    if (input.length > 100) {
        return "Too long";
    }
    return input.toUpperCase();
}
```

### 3.2 ループ内の continue と break

```python
# continue: 現在の反復をスキップ
for item in items:
    if not item.is_valid():
        continue  # 無効なアイテムをスキップ
    if item.is_deleted():
        continue  # 削除済みをスキップ
    process(item)

# break: ループを脱出
for item in items:
    if item.matches(target):
        result = item
        break
else:
    # break で抜けなかった場合（見つからなかった場合）
    result = None
```

```rust
// Rust: ラベル付き break/continue
'search: for row in &matrix {
    for &cell in row {
        if cell == target {
            println!("Found: {}", cell);
            break 'search;  // 外側のループも脱出
        }
    }
}

// break で値を返す（loop の場合）
let found = 'outer: loop {
    for item in &collection {
        if item.matches(&criteria) {
            break 'outer Some(item);  // 外側の loop から値を返す
        }
    }
    break None;  // 見つからなかった場合
};
```

---

## 4. 制御フローの高度なパターン

### 4.1 テーブル駆動ディスパッチ

```python
# if-elif チェーンの代替: 辞書ディスパッチ
def handle_command(command: str, args: list[str]) -> str:
    handlers = {
        "help": lambda args: show_help(),
        "list": lambda args: list_items(),
        "add": lambda args: add_item(args[0]) if args else "Missing argument",
        "remove": lambda args: remove_item(args[0]) if args else "Missing argument",
        "search": lambda args: search_items(" ".join(args)),
    }

    handler = handlers.get(command)
    if handler is None:
        return f"Unknown command: {command}"
    return handler(args)
```

```typescript
// TypeScript: ディスパッチマップ
type Handler = (req: Request) => Response;

const routes: Record<string, Handler> = {
    "/api/users": handleUsers,
    "/api/posts": handlePosts,
    "/api/comments": handleComments,
};

function router(req: Request): Response {
    const handler = routes[req.path];
    if (!handler) {
        return new Response("Not Found", { status: 404 });
    }
    return handler(req);
}
```

```go
// Go: ディスパッチテーブル
type CommandHandler func(args []string) error

var commands = map[string]CommandHandler{
    "start":   handleStart,
    "stop":    handleStop,
    "status":  handleStatus,
    "restart": handleRestart,
}

func dispatch(name string, args []string) error {
    handler, ok := commands[name]
    if !ok {
        return fmt.Errorf("unknown command: %s", name)
    }
    return handler(args)
}
```

### 4.2 状態機械パターン

```rust
// Rust: 列挙型による状態機械
enum ConnectionState {
    Disconnected,
    Connecting { attempt: u32, max_attempts: u32 },
    Connected { session_id: String },
    Disconnecting,
}

fn handle_event(state: ConnectionState, event: Event) -> ConnectionState {
    match (state, event) {
        (ConnectionState::Disconnected, Event::Connect) => {
            ConnectionState::Connecting { attempt: 1, max_attempts: 5 }
        }
        (ConnectionState::Connecting { attempt, max_attempts }, Event::Success(session)) => {
            ConnectionState::Connected { session_id: session }
        }
        (ConnectionState::Connecting { attempt, max_attempts }, Event::Failure) => {
            if attempt < max_attempts {
                ConnectionState::Connecting { attempt: attempt + 1, max_attempts }
            } else {
                ConnectionState::Disconnected
            }
        }
        (ConnectionState::Connected { .. }, Event::Disconnect) => {
            ConnectionState::Disconnecting
        }
        (ConnectionState::Disconnecting, Event::Done) => {
            ConnectionState::Disconnected
        }
        (state, _) => state,  // 無関係なイベントは無視
    }
}
```

```python
# Python: 状態機械をクラスで実装
from enum import Enum, auto

class State(Enum):
    IDLE = auto()
    PROCESSING = auto()
    WAITING = auto()
    ERROR = auto()
    DONE = auto()

class StateMachine:
    def __init__(self):
        self.state = State.IDLE
        self._transitions = {
            State.IDLE: {
                "start": self._start_processing,
            },
            State.PROCESSING: {
                "complete": self._complete,
                "error": self._handle_error,
                "wait": self._wait,
            },
            State.WAITING: {
                "resume": self._resume,
                "timeout": self._handle_error,
            },
            State.ERROR: {
                "retry": self._start_processing,
                "abort": self._abort,
            },
        }

    def handle_event(self, event: str):
        transitions = self._transitions.get(self.state, {})
        handler = transitions.get(event)
        if handler is None:
            raise ValueError(
                f"Invalid event '{event}' in state {self.state}"
            )
        handler()

    def _start_processing(self):
        print("Starting...")
        self.state = State.PROCESSING

    def _complete(self):
        print("Done!")
        self.state = State.DONE

    def _handle_error(self):
        print("Error occurred")
        self.state = State.ERROR

    def _wait(self):
        print("Waiting...")
        self.state = State.WAITING

    def _resume(self):
        print("Resuming...")
        self.state = State.PROCESSING

    def _abort(self):
        print("Aborted")
        self.state = State.DONE
```

### 4.3 再帰 vs ループ

```python
# 再帰による木構造の走査
def tree_sum(node):
    if node is None:
        return 0
    return node.value + tree_sum(node.left) + tree_sum(node.right)

# ループ（スタックを使った明示的な走査）
def tree_sum_iterative(root):
    if root is None:
        return 0
    total = 0
    stack = [root]
    while stack:
        node = stack.pop()
        total += node.value
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return total
```

```rust
// Rust: 末尾再帰の最適化（手動）
// 再帰版
fn factorial(n: u64) -> u64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

// ループ版（スタックオーバーフローの心配なし）
fn factorial_iter(n: u64) -> u64 {
    (1..=n).product()
}

// アキュムレータパターン（末尾再帰風）
fn factorial_acc(n: u64, acc: u64) -> u64 {
    if n <= 1 { acc } else { factorial_acc(n - 1, n * acc) }
}
```

```haskell
-- Haskell: 末尾再帰最適化（TCO）
-- 非末尾再帰（スタックを消費）
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- 末尾再帰（アキュムレータ）
factorial' :: Integer -> Integer
factorial' n = go n 1
  where
    go 0 acc = acc
    go n acc = go (n - 1) (n * acc)

-- fold で表現（最も簡潔）
factorial'' :: Integer -> Integer
factorial'' n = foldl' (*) 1 [1..n]
```

---

## 5. ループ最適化テクニック

### 5.1 ループ不変式の外出し

```python
# ❌ ループ内で毎回計算
for i in range(len(data)):
    normalized = data[i] / sum(data)  # sum(data) が毎回計算される
    results.append(normalized)

# ✅ ループ外で一度だけ計算
total = sum(data)
for i in range(len(data)):
    normalized = data[i] / total
    results.append(normalized)

# さらに Pythonic に
total = sum(data)
results = [x / total for x in data]
```

### 5.2 ループの展開とバッチ処理

```python
# バッチ処理（データベース操作など）
# ❌ 1件ずつ INSERT（遅い）
for item in items:
    db.execute("INSERT INTO table VALUES (?)", (item,))

# ✅ バッチ INSERT（高速）
BATCH_SIZE = 1000
for i in range(0, len(items), BATCH_SIZE):
    batch = items[i:i + BATCH_SIZE]
    db.executemany("INSERT INTO table VALUES (?)", [(item,) for item in batch])
```

```rust
// Rust: chunks によるバッチ処理
let data: Vec<Record> = load_records();

for chunk in data.chunks(100) {
    db.bulk_insert(chunk)?;
}

// par_chunks で並列バッチ処理（rayon）
use rayon::prelude::*;
data.par_chunks(100)
    .for_each(|chunk| {
        process_batch(chunk);
    });
```

### 5.3 短絡評価の活用

```python
# 短絡評価: 条件が確定した時点で評価を停止
# any() — 最初の True で停止
has_error = any(item.is_error() for item in items)

# all() — 最初の False で停止
all_valid = all(item.is_valid() for item in items)

# 大量データの場合、ジェネレータ式で遅延評価
has_match = any(
    expensive_check(item)
    for item in huge_dataset
)  # 最初のマッチで停止、全件チェックしない
```

---

## 6. 言語間の制御フロー設計比較

### 6.1 例外的制御フロー

```python
# Python: for-else（ループが break せずに終了した場合に else が実行）
def find_prime_factor(n):
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return i
    return n  # 素数の場合

# Python: try-except を制御フローに使う（Pythonic）
# EAFP: Easier to Ask Forgiveness than Permission
try:
    value = dictionary[key]
except KeyError:
    value = default_value

# vs LBYL: Look Before You Leap
if key in dictionary:
    value = dictionary[key]
else:
    value = default_value

# 推奨: dict.get() を使う
value = dictionary.get(key, default_value)
```

```go
// Go: defer（関数終了時に実行）— 制御フローの一種
func processFile(path string) error {
    f, err := os.Open(path)
    if err != nil {
        return err
    }
    defer f.Close()  // 関数終了時に確実にクローズ

    // 複数の defer は LIFO（後入れ先出し）で実行
    defer fmt.Println("Step 3")
    defer fmt.Println("Step 2")
    defer fmt.Println("Step 1")
    // 出力: Step 1, Step 2, Step 3

    return processData(f)
}

// Go: panic/recover（例外的な状況のみ）
func safeDivide(a, b float64) (result float64, err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()

    if b == 0 {
        panic("division by zero")
    }
    return a / b, nil
}
```

### 6.2 コルーチンと制御フロー

```python
# Python: async/await による制御フロー
import asyncio

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

async def fetch_one(session, url):
    async with session.get(url) as response:
        return await response.json()

# 非同期イテレーション
async def process_stream(stream):
    async for chunk in stream:
        await process_chunk(chunk)
```

```rust
// Rust: async/await
async fn fetch_all(urls: Vec<String>) -> Vec<Result<String, Error>> {
    let futures: Vec<_> = urls.iter()
        .map(|url| fetch_one(url))
        .collect();

    futures::future::join_all(futures).await
}

async fn fetch_one(url: &str) -> Result<String, Error> {
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}
```

---

## 7. アンチパターンとベストプラクティス

### 7.1 よくあるアンチパターン

```python
# ❌ アンチパターン1: フラグ変数の乱用
found = False
for item in items:
    if item == target:
        found = True
        break
if found:
    process(item)

# ✅ 改善: 早期リターンまたは組み込み関数
if target in items:
    process(target)

# または
try:
    index = items.index(target)
    process(items[index])
except ValueError:
    pass
```

```python
# ❌ アンチパターン2: 深いネスト
def validate_and_process(data):
    if data is not None:
        if isinstance(data, dict):
            if "type" in data:
                if data["type"] in VALID_TYPES:
                    if "payload" in data:
                        return process(data["payload"])
                    else:
                        return Error("Missing payload")
                else:
                    return Error("Invalid type")
            else:
                return Error("Missing type")
        else:
            return Error("Not a dict")
    else:
        return Error("No data")

# ✅ 改善: ガード節
def validate_and_process(data):
    if data is None:
        return Error("No data")
    if not isinstance(data, dict):
        return Error("Not a dict")
    if "type" not in data:
        return Error("Missing type")
    if data["type"] not in VALID_TYPES:
        return Error("Invalid type")
    if "payload" not in data:
        return Error("Missing payload")
    return process(data["payload"])
```

```javascript
// ❌ アンチパターン3: callback hell
getUser(userId, (err, user) => {
    if (err) return handleError(err);
    getOrders(user.id, (err, orders) => {
        if (err) return handleError(err);
        getOrderDetails(orders[0].id, (err, details) => {
            if (err) return handleError(err);
            processDetails(details);
        });
    });
});

// ✅ 改善: async/await
async function processUserOrder(userId) {
    try {
        const user = await getUser(userId);
        const orders = await getOrders(user.id);
        const details = await getOrderDetails(orders[0].id);
        return processDetails(details);
    } catch (err) {
        handleError(err);
    }
}
```

```python
# ❌ アンチパターン4: ループ内の不必要な再計算
for i in range(len(items)):
    for j in range(len(items)):
        distance = compute_distance(items[i], items[j])
        if distance < threshold:
            pairs.append((i, j))

# ✅ 改善: 対称性を活用して計算量を半減
for i in range(len(items)):
    for j in range(i + 1, len(items)):  # j > i のみ計算
        distance = compute_distance(items[i], items[j])
        if distance < threshold:
            pairs.append((i, j))
            pairs.append((j, i))  # 対称なペアも追加（必要な場合）
```

### 7.2 ベストプラクティスまとめ

```
1. ネストを浅く保つ
   → ガード節（早期リターン）を活用
   → 最大3段階のネストを目安に

2. 式ベースの分岐を活用する
   → Rust/Kotlin: if 式、match 式
   → 変数の不変性を保つ

3. 適切なループ構文を選ぶ
   → イテレータベース（for-in）が現代の主流
   → 関数型メソッドチェーン（map/filter/reduce）を活用
   → インデックスベースの for は最後の手段

4. 短絡評価を意識する
   → any/all、&&/|| の短絡評価を活用
   → 重い計算は遅延評価で

5. テーブル駆動にする
   → 長い if-elif/switch チェーンはディスパッチテーブルに
   → 保守性と拡張性が向上

6. 網羅性を保証する
   → match/switch で全ケースを明示的に処理
   → ワイルドカードの安易な使用を避ける
   → TypeScript: never 型で網羅性チェック

7. ループ最適化を意識する
   → ループ不変式の外出し
   → バッチ処理の活用
   → 不必要な再計算の排除
```

---

## まとめ

| 構文 | 文 vs 式 | 特徴 |
|------|---------|------|
| if (Python, JS) | 文 | 古典的、三項演算子は式 |
| if (Rust, Kotlin) | 式 | 値を返せる |
| switch (JS, C) | 文 | fall-through に注意 |
| switch (Java 14+) | 式 | アロー構文、yield |
| match (Rust) | 式 | 網羅性チェック、安全 |
| when (Kotlin) | 式 | 範囲、型チェック対応 |
| for-in | - | イテレータベース（現代の主流） |
| for-range (Go) | - | スライス、マップ、チャネル対応 |
| .filter().map() | 式 | 関数型スタイル |
| loop (Rust) | 式 | break で値を返せる |
| while let (Rust) | - | パターンマッチ付きループ |

### 言語ごとのループ構文比較

| 言語 | C風 for | for-in | while | do-while | 無限ループ | ラベル |
|------|---------|--------|-------|----------|-----------|--------|
| C | `for(;;)` | - | `while` | `do-while` | `for(;;)` | goto |
| Java | `for(;;)` | `for(:)` | `while` | `do-while` | `while(true)` | label |
| Python | - | `for-in` | `while` | - | `while True` | - |
| JavaScript | `for(;;)` | `for-of` | `while` | `do-while` | `while(true)` | label |
| Rust | - | `for-in` | `while` | - | `loop` | 'label |
| Go | `for` | `for-range` | `for` | `for{...if}` | `for{}` | label |
| Kotlin | `for(;;)` | `for-in` | `while` | `do-while` | `while(true)` | label |

---

## 次に読むべきガイド
→ [[01-pattern-matching.md]] — パターンマッチ

---

## 参考文献
1. Scott, M. "Programming Language Pragmatics." 4th Ed, Ch.6, Morgan Kaufmann, 2015.
2. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.3, 2023.
3. Bloch, J. "Effective Java." 3rd Ed, Item 58-65, Addison-Wesley, 2018.
4. Van Rossum, G. "PEP 20 -- The Zen of Python." python.org, 2004.
5. Donovan, A. & Kernighan, B. "The Go Programming Language." Ch.1, Addison-Wesley, 2015.
6. Jemerov, D. & Isakova, S. "Kotlin in Action." Ch.2, Manning, 2017.
7. Martin, R. "Clean Code." Ch.7 (Error Handling), Prentice Hall, 2008.
8. "Rust By Example: Flow of Control." doc.rust-lang.org.
9. Lipovaca, M. "Learn You a Haskell for Great Good!" Ch.4, No Starch Press, 2011.
10. Odersky, M., Spoon, L. & Venners, B. "Programming in Scala." 5th Ed, Ch.7, Artima, 2021.
