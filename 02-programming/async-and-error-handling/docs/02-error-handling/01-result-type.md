# Result型

> Result型は「成功または失敗」を型で表現する手法。例外を使わずにエラーを明示的に扱い、コンパイラが「エラー処理忘れ」を検出する。Rust, Go, TypeScript での実装を比較する。

## この章で学ぶこと

- [ ] Result型の仕組みと例外との違いを理解する
- [ ] 各言語でのResult型の実装を把握する
- [ ] Result型のメリットとデメリットを学ぶ
- [ ] モナド的なチェーン操作（map, flatMap, andThen）を理解する
- [ ] Result型とOption/Maybe型の関係を把握する
- [ ] 実務でのResult型の導入パターンを学ぶ

---

## 1. 例外 vs Result型

### 1.1 基本的な違い

```
例外:
  function getUser(id: string): User {
    // エラーが発生する可能性が型から見えない
    // 呼び出し側は try/catch を忘れるかもしれない
  }

Result型:
  function getUser(id: string): Result<User, AppError> {
    // 型を見るだけで「失敗の可能性がある」と分かる
    // コンパイラがエラー処理を強制できる
  }

比較:
  ┌──────────────┬──────────────────┬──────────────────┐
  │              │ 例外             │ Result型         │
  ├──────────────┼──────────────────┼──────────────────┤
  │ エラーの可視性│ 型に現れない     │ 型に現れる       │
  ├──────────────┼──────────────────┼──────────────────┤
  │ 処理の強制   │ なし             │ コンパイラが強制 │
  ├──────────────┼──────────────────┼──────────────────┤
  │ コード       │ try/catch        │ match/map/unwrap │
  ├──────────────┼──────────────────┼──────────────────┤
  │ パフォーマンス│ スタック巻き戻し │ 通常の戻り値     │
  └──────────────┴──────────────────┴──────────────────┘
```

### 1.2 なぜResult型が注目されるのか

```
Result型が注目される理由:

  1. 型安全性
     → エラーの種類が型として明示される
     → コンパイラがエラーハンドリングの漏れを検出
     → IDE の補完が効く

  2. 明示性
     → 関数のシグネチャを見るだけで失敗の可能性が分かる
     → 隠れた制御フロー（例外の伝播）がない
     → コードレビューが容易

  3. パフォーマンス
     → スタックアンワインドが不要
     → スタックトレースの構築が不要
     → 通常の関数リターンと同じコスト

  4. 合成可能性（Composability）
     → map, flatMap, andThen でチェーン処理
     → 関数型プログラミングとの親和性
     → パイプライン処理に適している

  5. 予測可能性
     → エラーパスが明確
     → テストが書きやすい
     → デバッグが容易
```

### 1.3 Result型の数学的背景

```
Result型の背景にある概念:

  直和型（Sum Type / Tagged Union）:
    Result<T, E> = Ok(T) | Err(E)
    → T か E のどちらか一方を必ず持つ

  これは代数的データ型（ADT）の一種:
    → Haskell: Either a b = Left a | Right b
    → Rust: enum Result<T, E> { Ok(T), Err(E) }
    → Scala: Either[L, R] = Left[L] | Right[R]
    → TypeScript: { ok: true; value: T } | { ok: false; error: E }

  モナド（Monad）としての Result:
    → flatMap (andThen) で連鎖可能
    → エラーが発生した時点で短絡（Short-circuit）
    → 例外の try/catch と同等の表現力を持つ
```

---

## 2. Rust の Result

### 2.1 基本的な使い方

```rust
// Rust: Result<T, E> は標準ライブラリの型
use std::fs;
use std::io;

fn read_config(path: &str) -> Result<Config, ConfigError> {
    let content = fs::read_to_string(path)
        .map_err(|e| ConfigError::IoError(e))?;  // ? で早期リターン

    let config: Config = serde_json::from_str(&content)
        .map_err(|e| ConfigError::ParseError(e.to_string()))?;

    if config.port == 0 {
        return Err(ConfigError::ValidationError("port must be > 0".into()));
    }

    Ok(config)
}

// エラー型の定義
#[derive(Debug)]
enum ConfigError {
    IoError(io::Error),
    ParseError(String),
    ValidationError(String),
}

// 使い方
fn main() {
    match read_config("config.json") {
        Ok(config) => println!("Port: {}", config.port),
        Err(ConfigError::IoError(e)) => eprintln!("File error: {}", e),
        Err(ConfigError::ParseError(e)) => eprintln!("Parse error: {}", e),
        Err(ConfigError::ValidationError(e)) => eprintln!("Validation: {}", e),
    }

    // ? 演算子でチェーン（呼び出し元にエラーを伝播）
    // → try/catch の代わりに型でエラーが伝播する
}

// Result のメソッドチェーン
fn process() -> Result<String, Error> {
    read_file("input.txt")?
        .lines()
        .map(|line| parse_line(line))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .map(|item| format_item(item))
        .collect::<Result<String, _>>()
}
```

### 2.2 ? 演算子の詳細

```rust
// ? 演算子は以下の糖衣構文:
fn read_file(path: &str) -> Result<String, io::Error> {
    // これは:
    let content = fs::read_to_string(path)?;

    // 以下と等価:
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return Err(e.into()),  // From トレイトで変換
    };

    Ok(content)
}

// From トレイトによるエラー型の自動変換
impl From<io::Error> for AppError {
    fn from(e: io::Error) -> Self {
        AppError::Io(e)
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError::Parse(e.to_string())
    }
}

// From を実装すると ? 演算子で自動変換される
fn load_config(path: &str) -> Result<Config, AppError> {
    let content = fs::read_to_string(path)?;  // io::Error → AppError
    let config: Config = serde_json::from_str(&content)?;  // serde::Error → AppError
    Ok(config)
}
```

### 2.3 Result のメソッド一覧

```rust
// Result<T, E> の主要メソッド

fn demonstrate_result_methods() {
    let ok_val: Result<i32, String> = Ok(42);
    let err_val: Result<i32, String> = Err("error".to_string());

    // ========== 値の取り出し ==========

    // unwrap: Ok なら値を返す、Err ならパニック
    let value = ok_val.unwrap();  // 42
    // let value = err_val.unwrap();  // パニック！本番コードでは使わない

    // unwrap_or: Err の場合のデフォルト値
    let value = err_val.unwrap_or(0);  // 0

    // unwrap_or_else: Err の場合にクロージャで値を生成
    let value = err_val.unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        0
    });

    // expect: unwrap と同じだがパニックメッセージを指定
    let value = ok_val.expect("Config must be valid");

    // ========== 変換 ==========

    // map: Ok の値を変換
    let mapped: Result<String, String> = ok_val.map(|v| v.to_string());

    // map_err: Err の値を変換
    let mapped: Result<i32, i32> = err_val.map_err(|e| e.len() as i32);

    // and_then (flatMap): Ok の値に関数を適用（チェーン）
    let chained: Result<String, String> = ok_val.and_then(|v| {
        if v > 0 {
            Ok(v.to_string())
        } else {
            Err("must be positive".to_string())
        }
    });

    // or_else: Err の場合に別の Result を返す
    let recovered: Result<i32, String> = err_val.or_else(|e| {
        eprintln!("Recovering from: {}", e);
        Ok(0)
    });

    // ========== 判定 ==========

    // is_ok / is_err
    assert!(ok_val.is_ok());
    assert!(err_val.is_err());

    // ========== Option との変換 ==========

    // ok(): Result<T, E> → Option<T>
    let opt: Option<i32> = ok_val.ok();  // Some(42)
    let opt: Option<i32> = err_val.ok();  // None

    // err(): Result<T, E> → Option<E>
    let opt: Option<String> = err_val.err();  // Some("error")

    // transpose: Result<Option<T>, E> → Option<Result<T, E>>
    let x: Result<Option<i32>, String> = Ok(Some(42));
    let y: Option<Result<i32, String>> = x.transpose();  // Some(Ok(42))
}

// collect で Vec<Result<T, E>> → Result<Vec<T>, E>
fn parse_all(inputs: &[&str]) -> Result<Vec<i32>, String> {
    inputs
        .iter()
        .map(|s| s.parse::<i32>().map_err(|e| e.to_string()))
        .collect()  // 最初の Err で短絡
}
```

### 2.4 thiserror と anyhow

```rust
// thiserror: ライブラリ向けの構造化エラー
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("User not found: {user_id}")]
    UserNotFound { user_id: String },

    #[error("Email already exists: {email}")]
    EmailAlreadyExists { email: String },

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Database error")]
    Database(#[from] sqlx::Error),

    #[error("External API error")]
    ExternalApi(#[from] reqwest::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

// anyhow: アプリケーション向けの柔軟なエラー
use anyhow::{Context, Result, bail, ensure};

fn load_config(path: &str) -> Result<Config> {
    let content = fs::read_to_string(path)
        .context("Failed to read config file")?;  // コンテキスト追加

    let config: Config = serde_json::from_str(&content)
        .context("Failed to parse config")?;

    ensure!(config.port > 0, "Port must be positive, got {}", config.port);
    // ensure! は条件が false なら Err を返す

    if config.host.is_empty() {
        bail!("Host cannot be empty");
        // bail! は即座に Err を返す
    }

    Ok(config)
}

// thiserror vs anyhow の使い分け:
// thiserror: ライブラリ（呼び出し側がエラーの種類を判別する必要がある）
// anyhow: アプリケーション（エラーの詳細は人間向けメッセージで十分）
```

---

## 3. Go のエラー

### 3.1 基本パターン

```go
// Go: 多値返却でエラーを返す
func readConfig(path string) (*Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("failed to read config: %w", err)
    }

    var config Config
    if err := json.Unmarshal(data, &config); err != nil {
        return nil, fmt.Errorf("failed to parse config: %w", err)
    }

    if config.Port == 0 {
        return nil, errors.New("port must be > 0")
    }

    return &config, nil
}

// 使い方
func main() {
    config, err := readConfig("config.json")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Port: %d\n", config.Port)
}

// Go のエラー処理の特徴:
// → エラーは値（error インターフェース）
// → if err != nil が頻出（賛否あり）
// → errors.Is, errors.As でエラーの判定
// → fmt.Errorf("%w", err) でエラーのラッピング
```

### 3.2 カスタムエラー型

```go
// カスタムエラー型の定義
type NotFoundError struct {
    Resource string
    ID       string
}

func (e *NotFoundError) Error() string {
    return fmt.Sprintf("%s not found: %s", e.Resource, e.ID)
}

type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error on %s: %s", e.Field, e.Message)
}

// sentinel error（定数エラー）
var (
    ErrNotFound      = errors.New("not found")
    ErrUnauthorized  = errors.New("unauthorized")
    ErrInternalError = errors.New("internal error")
)

// エラーの判定
func handleError(err error) {
    // errors.Is: sentinel error との比較
    if errors.Is(err, ErrNotFound) {
        fmt.Println("Resource not found")
        return
    }

    // errors.As: カスタムエラー型の判定
    var validationErr *ValidationError
    if errors.As(err, &validationErr) {
        fmt.Printf("Validation failed: %s - %s\n",
            validationErr.Field, validationErr.Message)
        return
    }

    // 未知のエラー
    fmt.Printf("Unknown error: %v\n", err)
}
```

### 3.3 エラーのラッピングチェーン

```go
// Go 1.13+: エラーのラッピング
func getUser(id string) (*User, error) {
    row := db.QueryRow("SELECT * FROM users WHERE id = ?", id)
    var user User
    if err := row.Scan(&user.ID, &user.Name, &user.Email); err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, fmt.Errorf("getUser(%s): %w", id, ErrNotFound)
        }
        return nil, fmt.Errorf("getUser(%s): database error: %w", id, err)
    }
    return &user, nil
}

func createOrder(userID string, items []Item) (*Order, error) {
    user, err := getUser(userID)
    if err != nil {
        return nil, fmt.Errorf("createOrder: %w", err)
    }
    // ...
    return &Order{UserID: user.ID}, nil
}

// エラーチェーンの例:
// "createOrder: getUser(user-123): not found"
// errors.Is(err, ErrNotFound) → true（チェーンを辿って判定）

// Go 1.20+: errors.Join で複数エラーの結合
func validateOrder(order *Order) error {
    var errs []error

    if order.UserID == "" {
        errs = append(errs, &ValidationError{Field: "userID", Message: "required"})
    }
    if len(order.Items) == 0 {
        errs = append(errs, &ValidationError{Field: "items", Message: "at least one item required"})
    }
    if order.Total < 0 {
        errs = append(errs, &ValidationError{Field: "total", Message: "must be non-negative"})
    }

    if len(errs) > 0 {
        return errors.Join(errs...)
    }
    return nil
}
```

### 3.4 Go のエラー処理の議論

```
Go のエラー処理に対する議論:

  賛成派:
  → シンプルで明示的
  → エラーを無視しにくい（lint ツールで検出）
  → スタックトレースのコストがない
  → エラーの伝播が透明

  反対派:
  → if err != nil のボイラープレート
  → エラーハンドリングがコードの大部分を占める
  → Result型の map/flatMap のような合成ができない
  → 型によるエラーの網羅性チェックがない

  Go 2 の提案（ドラフト）:
  → check/handle 構文（2018年提案、未採用）
  → try 組み込み関数（2019年提案、却下）
  → 結局 if err != nil が残り続ける

  実務的な対策:
  → ヘルパー関数でボイラープレートを軽減
  → errgroup でゴルーチンのエラー集約
  → 構造化ログでエラーのコンテキストを補強
```

---

## 4. TypeScript での Result型

### 4.1 シンプルな実装

```typescript
// TypeScript: Result型の実装
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

// 使用例
function parseJson<T>(text: string): Result<T, string> {
  try {
    return ok(JSON.parse(text));
  } catch (e) {
    return err(`Invalid JSON: ${(e as Error).message}`);
  }
}

function validateUser(data: unknown): Result<User, ValidationError> {
  if (!data || typeof data !== "object") {
    return err({ field: "root", message: "Invalid data" });
  }
  const { name, email } = data as any;
  if (!name) return err({ field: "name", message: "Name is required" });
  if (!email?.includes("@")) return err({ field: "email", message: "Invalid email" });
  return ok({ name, email } as User);
}

// チェーン的な使い方
function processInput(input: string): Result<User, string> {
  const jsonResult = parseJson<unknown>(input);
  if (!jsonResult.ok) return err(jsonResult.error);

  const userResult = validateUser(jsonResult.value);
  if (!userResult.ok) return err(userResult.error.message);

  return userResult;
}
```

### 4.2 高機能な Result クラスの実装

```typescript
// より高機能な Result 実装
class Result<T, E> {
    private constructor(
        private readonly _ok: boolean,
        private readonly _value?: T,
        private readonly _error?: E,
    ) {}

    static ok<T>(value: T): Result<T, never> {
        return new Result(true, value);
    }

    static err<E>(error: E): Result<never, E> {
        return new Result(false, undefined, error);
    }

    // 例外を Result に変換
    static fromThrowable<T>(fn: () => T): Result<T, Error> {
        try {
            return Result.ok(fn());
        } catch (e) {
            return Result.err(e instanceof Error ? e : new Error(String(e)));
        }
    }

    // async 版
    static async fromPromise<T>(promise: Promise<T>): Promise<Result<T, Error>> {
        try {
            return Result.ok(await promise);
        } catch (e) {
            return Result.err(e instanceof Error ? e : new Error(String(e)));
        }
    }

    isOk(): this is Result<T, never> {
        return this._ok;
    }

    isErr(): this is Result<never, E> {
        return !this._ok;
    }

    // Ok の値を変換
    map<U>(fn: (value: T) => U): Result<U, E> {
        if (this._ok) {
            return Result.ok(fn(this._value!));
        }
        return Result.err(this._error!);
    }

    // Err の値を変換
    mapErr<F>(fn: (error: E) => F): Result<T, F> {
        if (this._ok) {
            return Result.ok(this._value!);
        }
        return Result.err(fn(this._error!));
    }

    // flatMap / andThen: Result を返す関数をチェーン
    andThen<U>(fn: (value: T) => Result<U, E>): Result<U, E> {
        if (this._ok) {
            return fn(this._value!);
        }
        return Result.err(this._error!);
    }

    // Err の場合にリカバリー
    orElse<F>(fn: (error: E) => Result<T, F>): Result<T, F> {
        if (this._ok) {
            return Result.ok(this._value!);
        }
        return fn(this._error!);
    }

    // 値の取り出し
    unwrap(): T {
        if (this._ok) return this._value!;
        throw new Error(`Called unwrap on Err: ${this._error}`);
    }

    unwrapOr(defaultValue: T): T {
        return this._ok ? this._value! : defaultValue;
    }

    unwrapOrElse(fn: (error: E) => T): T {
        return this._ok ? this._value! : fn(this._error!);
    }

    // パターンマッチ
    match<U>(handlers: { ok: (value: T) => U; err: (error: E) => U }): U {
        if (this._ok) {
            return handlers.ok(this._value!);
        }
        return handlers.err(this._error!);
    }

    // Option への変換
    toOption(): T | undefined {
        return this._ok ? this._value : undefined;
    }
}

// 使用例
const result = Result.fromThrowable(() => JSON.parse('{"name": "test"}'))
    .map(data => data as { name: string })
    .andThen(data => {
        if (!data.name) return Result.err(new Error("Name required"));
        return Result.ok(data);
    })
    .mapErr(e => `Validation failed: ${e.message}`);

result.match({
    ok: data => console.log(`User: ${data.name}`),
    err: msg => console.error(msg),
});
```

### 4.3 neverthrow ライブラリ

```typescript
// neverthrow: TypeScript の人気 Result 型ライブラリ
import { ok, err, Result, ResultAsync } from 'neverthrow';

// 基本的な使い方
function divide(a: number, b: number): Result<number, string> {
    if (b === 0) return err("Division by zero");
    return ok(a / b);
}

// チェーン
function calculateAverage(numbers: number[]): Result<number, string> {
    if (numbers.length === 0) return err("Empty array");

    const sum = numbers.reduce((a, b) => a + b, 0);
    return divide(sum, numbers.length);
}

// map, mapErr, andThen
const result = calculateAverage([10, 20, 30])
    .map(avg => avg.toFixed(2))
    .mapErr(e => `Calculation error: ${e}`);

// ResultAsync: 非同期版
function fetchUser(id: string): ResultAsync<User, ApiError> {
    return ResultAsync.fromPromise(
        fetch(`/api/users/${id}`).then(r => r.json()),
        (e) => new ApiError("Fetch failed", e as Error)
    );
}

function fetchUserOrders(userId: string): ResultAsync<Order[], ApiError> {
    return fetchUser(userId)
        .andThen(user => {
            return ResultAsync.fromPromise(
                fetch(`/api/orders?userId=${user.id}`).then(r => r.json()),
                (e) => new ApiError("Fetch orders failed", e as Error)
            );
        });
}

// combine: 複数の Result をまとめる
import { Result as NResult } from 'neverthrow';

function validateForm(data: FormData): Result<ValidatedForm, ValidationError[]> {
    const nameResult = validateName(data.name);
    const emailResult = validateEmail(data.email);
    const ageResult = validateAge(data.age);

    return NResult.combine([nameResult, emailResult, ageResult])
        .map(([name, email, age]) => ({ name, email, age }));
    // 全て Ok なら Ok、1つでも Err なら最初の Err
}

// safeTry: generator ベースの ? 演算子風構文
import { safeTry } from 'neverthrow';

function processOrder(orderId: string): Result<Receipt, AppError> {
    return safeTry(function* () {
        const order = yield* getOrder(orderId).safeUnwrap();
        const user = yield* getUser(order.userId).safeUnwrap();
        const payment = yield* chargePayment(user, order.total).safeUnwrap();
        return ok({ orderId, paymentId: payment.id, amount: order.total });
    });
}
```

### 4.4 ts-results ライブラリ

```typescript
// ts-results: もう一つの人気ライブラリ
import { Ok, Err, Result } from 'ts-results';

function parsePort(input: string): Result<number, string> {
    const port = parseInt(input, 10);
    if (isNaN(port)) return Err(`Invalid port number: ${input}`);
    if (port < 1 || port > 65535) return Err(`Port out of range: ${port}`);
    return Ok(port);
}

// val プロパティで Ok/Err の値にアクセス
const result = parsePort("8080");
if (result.ok) {
    console.log(`Port: ${result.val}`);  // 8080
} else {
    console.error(`Error: ${result.val}`);  // エラーメッセージ
}

// expect: Ok なら値を返す、Err ならメッセージ付きでスロー
const port = parsePort("8080").expect("Port must be valid");

// map と andThen
const configResult = parsePort("8080")
    .map(port => ({ port, host: "localhost" }))
    .andThen(config => {
        if (config.host === "") return Err("Host required");
        return Ok(config);
    });
```

---

## 5. Option/Maybe型との関係

### 5.1 Option型とは

```
Option<T> = Some(T) | None

Result<T, E> = Ok(T) | Err(E)

違い:
  Option: 値が「あるかないか」
  Result: 値が「あるか、なぜないのか」

  Option は「エラーの理由がない Result」と言える:
  Option<T> ≒ Result<T, ()>  // エラーの情報なし
```

```rust
// Rust: Option と Result の相互変換
fn find_user(id: &str) -> Option<User> {
    users.get(id).cloned()
}

fn get_user(id: &str) -> Result<User, AppError> {
    find_user(id).ok_or_else(|| AppError::UserNotFound {
        user_id: id.to_string(),
    })
}

// Option のメソッド
let opt: Option<i32> = Some(42);

opt.map(|v| v * 2);           // Some(84)
opt.and_then(|v| if v > 0 { Some(v) } else { None });
opt.unwrap_or(0);              // 42
opt.ok_or("value is none")?;  // Option → Result
```

```typescript
// TypeScript での Option 型
type Option<T> = T | null | undefined;

// Result と Option の使い分け
function findUser(id: string): Option<User> {
    // 「見つからない」は正常なケース → Option
    return users.get(id) ?? null;
}

function getUser(id: string): Result<User, NotFoundError> {
    // 「見つからない」がエラーのケース → Result
    const user = users.get(id);
    if (!user) return err(new NotFoundError("User", id));
    return ok(user);
}

// 使い分けの指針:
// Option を使う場面:
//   → 値が存在しないことが通常の状況
//   → 例: Map.get(), Array.find(), cache.get()
//
// Result を使う場面:
//   → 失敗の理由を伝える必要がある
//   → 例: API呼び出し、バリデーション、ファイル読み込み
```

### 5.2 Haskell の Either と Maybe

```haskell
-- Haskell: Either と Maybe は Result と Option の元祖

-- Maybe a = Nothing | Just a
findUser :: String -> Maybe User
findUser userId = lookup userId userMap

-- Either e a = Left e | Right a（Left がエラー、Right が成功）
getUser :: String -> Either AppError User
getUser userId = case findUser userId of
    Nothing -> Left (UserNotFound userId)
    Just user -> Right user

-- do 記法でチェーン（モナド）
processOrder :: String -> Either AppError Receipt
processOrder orderId = do
    order <- getOrder orderId           -- Err なら即座に返る
    user  <- getUser (orderUserId order) -- Err なら即座に返る
    payment <- chargePayment user order  -- Err なら即座に返る
    return Receipt { receiptOrder = order, receiptPayment = payment }

-- これは以下と等価:
processOrder' :: String -> Either AppError Receipt
processOrder' orderId =
    getOrder orderId >>= \order ->
    getUser (orderUserId order) >>= \user ->
    chargePayment user order >>= \payment ->
    Right Receipt { receiptOrder = order, receiptPayment = payment }
```

---

## 6. Result型 vs 例外の使い分け

### 6.1 場面別の選択基準

```
Result型を使うべき場面:
  ✓ 予期されるエラー（バリデーション、ファイル未存在）
  ✓ ライブラリ/APIのパブリックインターフェース
  ✓ 型安全性が重要な場面
  ✓ エラーの種類が限定的
  ✓ パフォーマンスクリティカルなコード
  ✓ 関数型スタイルのコード
  ✓ エラーの合成が必要な場面

例外を使うべき場面:
  ✓ 予期しないエラー（プログラミングミス）
  ✓ 回復不能なエラー（OutOfMemory）
  ✓ フレームワークが例外を期待する場合
  ✓ 深いコールスタックからのエラー伝播
  ✓ コンストラクタやプロパティアクセスでのエラー
  ✓ 外部ライブラリとの境界

組み合わせ（推奨）:
  → ドメインロジック: Result型（予期されるエラー）
  → インフラ層: 例外（ネットワーク、DB障害）
  → 境界（Controller）: 例外を Result に変換
```

### 6.2 レイヤー別の使い分け

```typescript
// レイヤー別の使い分け例

// ========== インフラ層: 例外を投げる ==========
class UserRepository {
    async findById(id: string): Promise<User | null> {
        // DB エラーは例外として伝播
        const row = await db.query("SELECT * FROM users WHERE id = $1", [id]);
        return row ? mapToUser(row) : null;
    }
}

// ========== ドメイン層: Result型を使う ==========
class UserService {
    constructor(private repo: UserRepository) {}

    async getUser(id: string): Promise<Result<User, UserError>> {
        try {
            const user = await this.repo.findById(id);
            if (!user) return err(new UserNotFoundError(id));
            return ok(user);
        } catch (error) {
            // インフラ例外を Result に変換
            return err(new UserServiceError("Database error", { cause: error }));
        }
    }

    async createUser(data: CreateUserDto): Promise<Result<User, UserError>> {
        // バリデーション
        const validation = validateCreateUser(data);
        if (!validation.ok) return validation;

        // 重複チェック
        const existing = await this.repo.findByEmail(data.email);
        if (existing) return err(new EmailAlreadyExistsError(data.email));

        try {
            const user = await this.repo.create(data);
            return ok(user);
        } catch (error) {
            return err(new UserServiceError("Failed to create user", { cause: error }));
        }
    }
}

// ========== プレゼンテーション層: Result を HTTP レスポンスに変換 ==========
class UserController {
    constructor(private service: UserService) {}

    async getUser(req: Request, res: Response): Promise<void> {
        const result = await this.service.getUser(req.params.id);

        result.match({
            ok: user => res.json(user),
            err: error => {
                if (error instanceof UserNotFoundError) {
                    res.status(404).json({ error: error.message });
                } else {
                    res.status(500).json({ error: "Internal server error" });
                }
            }
        });
    }
}
```

### 6.3 実務での移行戦略

```typescript
// 既存の例外ベースのコードから Result型へ段階的に移行する戦略

// Step 1: Result型のユーティリティを用意
function tryCatch<T>(fn: () => T): Result<T, Error> {
    try {
        return ok(fn());
    } catch (e) {
        return err(e instanceof Error ? e : new Error(String(e)));
    }
}

async function tryCatchAsync<T>(fn: () => Promise<T>): Promise<Result<T, Error>> {
    try {
        return ok(await fn());
    } catch (e) {
        return err(e instanceof Error ? e : new Error(String(e)));
    }
}

// Step 2: 新しいコードから Result型を使い始める
// 既存コードとの境界で変換

// 例外 → Result 変換
async function getUserSafe(id: string): Promise<Result<User, AppError>> {
    return tryCatchAsync(async () => {
        // 既存の例外ベースの関数を呼ぶ
        return await legacyGetUser(id);
    }).then(result =>
        result.mapErr(e => new AppError("USER_FETCH_FAILED", e.message))
    );
}

// Result → 例外 変換（フレームワークが例外を期待する場合）
function unwrapOrThrow<T, E extends Error>(result: Result<T, E>): T {
    if (result.ok) return result.value;
    throw result.error;
}

// Step 3: 重要なドメインロジックから順に移行
// バリデーション → ビジネスルール → サービス層 の順で
```

---

## 7. 高度なパターン

### 7.1 Railway Oriented Programming

```
Railway Oriented Programming（鉄道指向プログラミング）:

  正常系と異常系を「2本のレール」として表現する。
  各関数は Success レールから Error レールに切り替わる可能性がある。

  Success ──────→ validate ──→ transform ──→ save ──→ Success
                      │              │           │
  Error   ◁──────────┘    ◁─────────┘    ◁──────┘     Error

  Result型の andThen（flatMap）はまさにこのパターン:
  → Success の場合のみ次の関数を実行
  → Error の場合はそのまま Error レールを流れる
```

```typescript
// Railway Oriented Programming の実装例
type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

// パイプライン関数
function pipe<T, E>(
    initial: Result<T, E>,
    ...fns: Array<(value: any) => Result<any, E>>
): Result<any, E> {
    let result: Result<any, E> = initial;
    for (const fn of fns) {
        if (!result.ok) return result;  // Error レールをそのまま通過
        result = fn(result.value);
    }
    return result;
}

// 使用例: ユーザー登録パイプライン
function registerUser(input: unknown): Result<User, AppError> {
    return pipe(
        ok(input),
        validateInput,    // 入力検証
        normalizeEmail,   // メール正規化
        checkDuplicate,   // 重複チェック
        hashPassword,     // パスワードハッシュ化
        saveToDatabase,   // DB保存
        sendWelcomeEmail, // メール送信
    );
}

// 各関数は独立してテスト可能
function validateInput(input: unknown): Result<RegisterDto, AppError> {
    if (!input || typeof input !== "object") {
        return err({ code: "INVALID_INPUT", message: "Invalid input" });
    }
    // ... バリデーションロジック
    return ok(input as RegisterDto);
}

function normalizeEmail(dto: RegisterDto): Result<RegisterDto, AppError> {
    return ok({ ...dto, email: dto.email.toLowerCase().trim() });
}
```

### 7.2 Result の並列処理

```typescript
// 複数の Result を並列で処理するユーティリティ

// all: 全て成功なら Ok、1つでも失敗なら最初の Err
function all<T, E>(results: Result<T, E>[]): Result<T[], E> {
    const values: T[] = [];
    for (const result of results) {
        if (!result.ok) return result;
        values.push(result.value);
    }
    return ok(values);
}

// allSettled: 全ての結果を収集
function allSettled<T, E>(
    results: Result<T, E>[]
): { successes: T[]; failures: E[] } {
    const successes: T[] = [];
    const failures: E[] = [];
    for (const result of results) {
        if (result.ok) successes.push(result.value);
        else failures.push(result.error);
    }
    return { successes, failures };
}

// 非同期版
async function allAsync<T, E>(
    promises: Promise<Result<T, E>>[]
): Promise<Result<T[], E>> {
    const results = await Promise.all(promises);
    return all(results);
}

// 使用例
async function validateBulkUsers(
    users: CreateUserDto[]
): Promise<Result<ValidatedUser[], ValidationError[]>> {
    const results = users.map(user => validateUser(user));
    const { successes, failures } = allSettled(results);

    if (failures.length > 0) {
        return err(failures);
    }
    return ok(successes);
}
```

### 7.3 Result と Either の拡張

```typescript
// Either 型: Result の一般化（Left/Right に意味付けしない）
type Either<L, R> = { tag: "left"; value: L } | { tag: "right"; value: R };

function left<L>(value: L): Either<L, never> {
    return { tag: "left", value };
}

function right<R>(value: R): Either<never, R> {
    return { tag: "right", value };
}

// bimap: 両方のケースを変換
function bimap<L, R, L2, R2>(
    either: Either<L, R>,
    leftFn: (l: L) => L2,
    rightFn: (r: R) => R2,
): Either<L2, R2> {
    if (either.tag === "left") return left(leftFn(either.value));
    return right(rightFn(either.value));
}

// Validation 型: エラーを累積する Result
type Validation<T, E> = { ok: true; value: T } | { ok: false; errors: E[] };

function validateAll<T, E>(
    validations: Validation<T, E>[]
): Validation<T[], E> {
    const values: T[] = [];
    const errors: E[] = [];

    for (const v of validations) {
        if (v.ok) {
            values.push(v.value);
        } else {
            errors.push(...v.errors);
        }
    }

    if (errors.length > 0) {
        return { ok: false, errors };
    }
    return { ok: true, value: values };
}

// 使用例: フォームバリデーション（全エラーを一度に返す）
function validateRegistrationForm(data: FormData): Validation<ValidForm, FieldError> {
    return validateAll([
        validateName(data.name),
        validateEmail(data.email),
        validatePassword(data.password),
        validateAge(data.age),
    ]).map(([name, email, password, age]) => ({
        name, email, password, age,
    }));
}
// Result.andThen は最初のエラーで短絡するが、
// Validation.validateAll は全てのエラーを収集する
```

---

## 8. Scala と関数型言語での Result

### 8.1 Scala の Either と Try

```scala
// Scala: Either[L, R]
def divide(a: Double, b: Double): Either[String, Double] = {
  if (b == 0) Left("Division by zero")
  else Right(a / b)
}

// for 内包表記でチェーン（Haskell の do 記法に相当）
def calculate(a: Double, b: Double, c: Double): Either[String, Double] = {
  for {
    x <- divide(a, b)     // Err なら即座に返る
    y <- divide(x, c)     // Err なら即座に返る
    z <- if (y > 0) Right(y) else Left("Result must be positive")
  } yield z * 100
}

// Try[T]: 例外を自動キャッチ
import scala.util.{Try, Success, Failure}

val result: Try[Int] = Try("42".toInt)
// Success(42)

val result: Try[Int] = Try("abc".toInt)
// Failure(java.lang.NumberFormatException)

val processed = Try("42".toInt)
  .map(_ * 2)
  .flatMap(n => if (n > 0) Success(n) else Failure(new Exception("negative")))
  .recover { case _: NumberFormatException => 0 }
  .getOrElse(-1)
```

### 8.2 F# の Result

```fsharp
// F#: Result<'T, 'Error> は標準ライブラリの型
let divide a b : Result<float, string> =
    if b = 0.0 then Error "Division by zero"
    else Ok (a / b)

// パイプ演算子でチェーン
let processOrder orderId =
    getOrder orderId
    |> Result.bind validateOrder
    |> Result.bind calculateTotal
    |> Result.bind processPayment
    |> Result.map createReceipt

// Computation Expression（do 記法に相当）
type ResultBuilder() =
    member _.Bind(x, f) = Result.bind f x
    member _.Return(x) = Ok x

let result = ResultBuilder()

let processOrder orderId = result {
    let! order = getOrder orderId
    let! validated = validateOrder order
    let! total = calculateTotal validated
    let! payment = processPayment total
    return createReceipt payment
}
```

---

## 9. テスト戦略

### 9.1 Result型のテスト

```typescript
// Result型を使ったコードのテスト
describe("UserService.createUser", () => {
    it("有効なデータで Ok(User) を返す", async () => {
        const result = await userService.createUser({
            name: "Test User",
            email: "test@example.com",
            password: "SecurePass123!",
        });

        expect(result.ok).toBe(true);
        if (result.ok) {
            expect(result.value.name).toBe("Test User");
            expect(result.value.email).toBe("test@example.com");
            expect(result.value.id).toBeDefined();
        }
    });

    it("無効なメールで Err(ValidationError) を返す", async () => {
        const result = await userService.createUser({
            name: "Test User",
            email: "invalid-email",
            password: "SecurePass123!",
        });

        expect(result.ok).toBe(false);
        if (!result.ok) {
            expect(result.error).toBeInstanceOf(ValidationError);
            expect(result.error.code).toBe("VALIDATION_ERROR");
        }
    });

    it("重複メールで Err(ConflictError) を返す", async () => {
        // 既存ユーザーを作成
        await userService.createUser({
            name: "Existing",
            email: "existing@example.com",
            password: "Pass123!",
        });

        const result = await userService.createUser({
            name: "New User",
            email: "existing@example.com",
            password: "Pass123!",
        });

        expect(result.ok).toBe(false);
        if (!result.ok) {
            expect(result.error.code).toBe("EMAIL_ALREADY_EXISTS");
        }
    });
});

// ヘルパー関数でテストを簡潔に
function expectOk<T, E>(result: Result<T, E>): T {
    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error(`Expected Ok, got Err: ${result.error}`);
    return result.value;
}

function expectErr<T, E>(result: Result<T, E>): E {
    expect(result.ok).toBe(false);
    if (result.ok) throw new Error(`Expected Err, got Ok: ${result.value}`);
    return result.error;
}

// 使用例
it("ユーザーを正常に作成できる", async () => {
    const user = expectOk(await userService.createUser(validData));
    expect(user.name).toBe("Test User");
});

it("バリデーションエラーを返す", async () => {
    const error = expectErr(await userService.createUser(invalidData));
    expect(error.code).toBe("VALIDATION_ERROR");
});
```

### 9.2 プロパティベーステスト

```typescript
// fast-check でのプロパティベーステスト
import fc from 'fast-check';

describe("Result invariants", () => {
    it("map の恒等法則: result.map(x => x) === result", () => {
        fc.assert(
            fc.property(fc.integer(), (n) => {
                const result = ok(n);
                const mapped = result.map(x => x);
                expect(mapped).toEqual(result);
            })
        );
    });

    it("andThen の結合法則", () => {
        fc.assert(
            fc.property(fc.integer(), (n) => {
                const f = (x: number) => ok(x * 2);
                const g = (x: number) => ok(x + 1);

                const left = ok(n).andThen(f).andThen(g);
                const right = ok(n).andThen(x => f(x).andThen(g));

                expect(left).toEqual(right);
            })
        );
    });

    it("parsePort は常に 1-65535 の範囲か Err を返す", () => {
        fc.assert(
            fc.property(fc.string(), (input) => {
                const result = parsePort(input);
                if (result.ok) {
                    expect(result.value).toBeGreaterThanOrEqual(1);
                    expect(result.value).toBeLessThanOrEqual(65535);
                }
                // Err の場合はバリデーションが正しく機能している
            })
        );
    });
});
```

---

## 10. 実務での導入パターン

### 10.1 段階的導入のロードマップ

```
Result型の段階的導入:

  Phase 1: ユーティリティの準備
  → Result 型の定義（またはライブラリの選定）
  → ok(), err() ヘルパー関数
  → tryCatch, tryCatchAsync ユーティリティ

  Phase 2: バリデーション層から導入
  → フォームバリデーション
  → API リクエストのバリデーション
  → 設定値のバリデーション
  → ← 最も効果が高く、リスクが低い

  Phase 3: サービス層に拡大
  → ドメインロジックの戻り値を Result に
  → 例外との境界を明確化
  → レポジトリ層は例外のまま

  Phase 4: API レスポンスとの統合
  → Controller で Result をレスポンスに変換
  → エラーレスポンスの統一
  → OpenAPI スキーマとの整合

  Phase 5: チーム全体での採用
  → コーディング規約の更新
  → コードレビューガイドラインの整備
  → テストパターンの標準化
```

### 10.2 チームでの合意形成

```
Result型導入時の議論ポイント:

  1. ライブラリの選定
     → neverthrow: 最も人気、ResultAsync あり
     → ts-results: 軽量、Rust 風
     → 自前実装: 柔軟だが保守コスト
     → 組み込みの union type: ライブラリ不要だが機能少

  2. 例外との境界ルール
     → どのレイヤーから Result を使うか
     → 例外を Result に変換する場所
     → フレームワークとの接点

  3. エラー型の設計
     → string vs カスタムエラークラス
     → エラーコード体系
     → エラーの詳細度

  4. 既存コードとの共存
     → 段階的移行 vs 一括移行
     → アダプター層の設計
     → テストの移行戦略
```

---

## まとめ

| 言語 | エラー手法 | 特徴 |
|------|-----------|------|
| Rust | Result<T, E> + ? | 最も洗練されたResult型 |
| Go | (value, error) | シンプルだが冗長 |
| TypeScript | Union型 / neverthrow | 型で表現可能 |
| Java | 例外（+ Optional） | checked exception |
| Python | 例外 | 型ヒントで補完 |
| Haskell | Either a b | 元祖、モナドによるチェーン |
| Scala | Either / Try | for 内包表記で簡潔 |
| F# | Result<'T, 'E> | Computation Expression |
| Kotlin | runCatching / Result | Java 互換 |
| Swift | throws + Result | 両方使える |

---

## 次に読むべきガイド
→ [[02-error-boundaries.md]] — エラー境界

---

## 参考文献
1. The Rust Programming Language. "Error Handling."
2. Go Blog. "Error handling and Go." 2011.
3. Wlaschin, S. "Railway Oriented Programming." F# for Fun and Profit.
4. neverthrow. "Type-Safe Error Handling in TypeScript." GitHub.
5. Bloch, J. "Effective Java." Item 71: Avoid unnecessary use of checked exceptions.
6. Syme, D. et al. "The F# Component Design Guidelines."
7. Kotlin Documentation. "Exceptions."
8. Apple Developer Documentation. "Error Handling in Swift."
