# エラーハンドリング

> エラーを「どう表現し、どう伝播し、どう処理するか」は言語設計の重要な決断。例外、Result型、エラーコードの3大戦略を理解する。

## この章で学ぶこと

- [ ] 例外ベース・Result型・エラーコードの違いを理解する
- [ ] 各言語のエラーハンドリング哲学を把握する
- [ ] 適切なエラー処理パターンを選択できる
- [ ] カスタムエラー型の設計ができる
- [ ] エラーの伝播と変換のベストプラクティスを習得する
- [ ] 実務でのエラーハンドリング戦略を設計できる

---

## 1. エラーハンドリングの3大戦略

### 1.1 概要

```
3つの戦略:

  1. 例外（Exceptions）
     → 正常系と異常系のコードを分離
     → 暗黙的な伝播（呼び出し元にスタックを巻き戻し）
     → 代表: Python, Java, C#, JavaScript, Ruby

  2. Result型 / Either型
     → エラーを「値」として型で表現
     → 明示的な伝播（? 演算子、map/and_then）
     → 代表: Rust, Haskell, Elm, OCaml, F#

  3. エラーコード / 複数戻り値
     → 関数の戻り値でエラーを示す
     → 明示的な伝播（if err != nil）
     → 代表: C, Go

各戦略のトレードオフ:
  例外:    書きやすい ⟷ エラーが見えにくい
  Result:  型安全     ⟷ ボイラープレートが多い
  エラーコード: シンプル ⟷ チェック忘れの危険
```

---

## 2. 例外（Exceptions）

### 2.1 Python のエラーハンドリング

```python
# Python: try-except

# ========================================
# 基本的な例外処理
# ========================================
try:
    result = int("not a number")
    file = open("missing.txt")
except ValueError as e:
    print(f"Invalid value: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected: {e}")
else:
    print("Success")  # 例外が発生しなかった場合
finally:
    print("Always executed")  # 常に実行（クリーンアップ）

# ========================================
# 例外の階層構造
# ========================================
# BaseException
#   ├── SystemExit
#   ├── KeyboardInterrupt
#   ├── GeneratorExit
#   └── Exception
#       ├── ValueError
#       ├── TypeError
#       ├── KeyError
#       ├── IndexError
#       ├── AttributeError
#       ├── IOError
#       │   └── FileNotFoundError
#       ├── RuntimeError
#       └── ...

# ========================================
# カスタム例外の設計
# ========================================
class AppError(Exception):
    """アプリケーションの基底例外"""
    def __init__(self, message: str, code: str = "UNKNOWN"):
        super().__init__(message)
        self.code = code

class NotFoundError(AppError):
    """リソースが見つからない"""
    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            f"{resource} with id '{resource_id}' not found",
            code="NOT_FOUND"
        )
        self.resource = resource
        self.resource_id = resource_id

class ValidationError(AppError):
    """バリデーションエラー"""
    def __init__(self, errors: dict[str, list[str]]):
        messages = []
        for field, field_errors in errors.items():
            for error in field_errors:
                messages.append(f"{field}: {error}")
        super().__init__(
            f"Validation failed: {'; '.join(messages)}",
            code="VALIDATION_ERROR"
        )
        self.errors = errors

class AuthenticationError(AppError):
    """認証エラー"""
    def __init__(self, reason: str = "Invalid credentials"):
        super().__init__(reason, code="AUTHENTICATION_ERROR")

class AuthorizationError(AppError):
    """認可エラー"""
    def __init__(self, action: str, resource: str):
        super().__init__(
            f"Not authorized to {action} {resource}",
            code="AUTHORIZATION_ERROR"
        )
        self.action = action
        self.resource = resource

# ========================================
# 例外の使用例
# ========================================
class UserService:
    def __init__(self, db, auth_service):
        self.db = db
        self.auth_service = auth_service

    def get_user(self, user_id: str) -> User:
        user = self.db.find_user(user_id)
        if user is None:
            raise NotFoundError("User", user_id)
        return user

    def update_user(self, user_id: str, data: dict, requester: User) -> User:
        if requester.id != user_id and not requester.is_admin:
            raise AuthorizationError("update", f"user/{user_id}")

        errors = self._validate_user_data(data)
        if errors:
            raise ValidationError(errors)

        user = self.get_user(user_id)  # NotFoundError が伝播する
        user.update(data)
        return self.db.save_user(user)

    def _validate_user_data(self, data: dict) -> dict[str, list[str]]:
        errors = {}
        if "name" in data and len(data["name"]) < 2:
            errors.setdefault("name", []).append("Must be at least 2 characters")
        if "email" in data and "@" not in data["email"]:
            errors.setdefault("email", []).append("Invalid email format")
        if "age" in data and (data["age"] < 0 or data["age"] > 150):
            errors.setdefault("age", []).append("Must be between 0 and 150")
        return errors

# ========================================
# コンテキストマネージャ（自動クリーンアップ）
# ========================================
# with 文（リソースの自動解放）
with open("file.txt") as f:
    content = f.read()
# ファイルは自動的にクローズされる（例外発生時も）

# カスタムコンテキストマネージャ
from contextlib import contextmanager

@contextmanager
def database_transaction(db):
    """トランザクション管理"""
    tx = db.begin_transaction()
    try:
        yield tx
        tx.commit()
    except Exception:
        tx.rollback()
        raise  # 例外を再送出

# 使用例
with database_transaction(db) as tx:
    tx.execute("INSERT INTO users ...")
    tx.execute("INSERT INTO profiles ...")
    # 例外が発生すると自動的にロールバック

# ========================================
# EAFP vs LBYL
# ========================================
# LBYL (Look Before You Leap) — 事前チェック
if key in dictionary:
    value = dictionary[key]
else:
    value = default

# EAFP (Easier to Ask Forgiveness than Permission) — Pythonic
try:
    value = dictionary[key]
except KeyError:
    value = default

# ベスト: dict.get() を使う
value = dictionary.get(key, default)

# ========================================
# 例外チェーン（Python 3）
# ========================================
class DatabaseError(AppError):
    pass

def get_user_from_db(user_id):
    try:
        return db.query(f"SELECT * FROM users WHERE id = {user_id}")
    except sqlite3.OperationalError as e:
        # raise ... from e で元の例外を保持
        raise DatabaseError(f"Failed to query user {user_id}") from e
    # __cause__ で元の例外にアクセス可能

# 暗黙的な例外チェーン
try:
    try:
        1 / 0
    except ZeroDivisionError:
        raise ValueError("Invalid computation")
    # __context__ で暗黙的な例外チェーンにアクセス可能
except ValueError as e:
    print(f"Error: {e}")
    print(f"Caused by: {e.__context__}")

# ========================================
# 例外の抑制
# ========================================
from contextlib import suppress

# 例外を無視（安全に）
with suppress(FileNotFoundError):
    os.remove("tempfile.txt")
# ファイルがなくてもエラーにならない

# 等価なコード
try:
    os.remove("tempfile.txt")
except FileNotFoundError:
    pass
```

### 2.2 Java のエラーハンドリング

```java
// Java: チェック例外 vs 非チェック例外

// ========================================
// チェック例外（Checked Exceptions）
// ========================================
// コンパイラが処理を強制する例外
// → IOException, SQLException, ClassNotFoundException など
public String readFile(String path) throws IOException {
    return Files.readString(Path.of(path));
}

// 呼び出し側は処理するか、throws で伝播するか選ぶ必要がある
public void processFile(String path) {
    try {
        String content = readFile(path);
        System.out.println(content);
    } catch (IOException e) {
        logger.error("Failed to read file: " + path, e);
    }
}

// ========================================
// 非チェック例外（Unchecked Exceptions）
// ========================================
// RuntimeException のサブクラス
// → NullPointerException, IllegalArgumentException,
//    IndexOutOfBoundsException など
public int divide(int a, int b) {
    if (b == 0) throw new IllegalArgumentException("Division by zero");
    return a / b;
}

// ========================================
// try-with-resources（自動クリーンアップ）
// ========================================
// AutoCloseable インターフェースを実装したリソースを自動クローズ
try (var reader = new BufferedReader(new FileReader("file.txt"));
     var writer = new BufferedWriter(new FileWriter("output.txt"))) {
    String line;
    while ((line = reader.readLine()) != null) {
        writer.write(line);
        writer.newLine();
    }
} catch (IOException e) {
    logger.error("I/O error", e);
}

// ========================================
// カスタム例外（Java）
// ========================================
public class AppException extends Exception {
    private final String errorCode;
    private final int statusCode;

    public AppException(String message, String errorCode, int statusCode) {
        super(message);
        this.errorCode = errorCode;
        this.statusCode = statusCode;
    }

    public AppException(String message, String errorCode, int statusCode, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
        this.statusCode = statusCode;
    }

    public String getErrorCode() { return errorCode; }
    public int getStatusCode() { return statusCode; }
}

public class NotFoundException extends AppException {
    public NotFoundException(String resource, String id) {
        super(resource + " with id '" + id + "' not found", "NOT_FOUND", 404);
    }
}

public class ValidationException extends AppException {
    private final Map<String, List<String>> errors;

    public ValidationException(Map<String, List<String>> errors) {
        super("Validation failed", "VALIDATION_ERROR", 400);
        this.errors = Collections.unmodifiableMap(errors);
    }

    public Map<String, List<String>> getErrors() { return errors; }
}

// ========================================
// マルチキャッチ（Java 7+）
// ========================================
try {
    // ...
} catch (IOException | SQLException e) {
    logger.error("I/O or Database error", e);
}

// ========================================
// チェック例外の問題点と対策
// ========================================
// 問題: ラムダ式でチェック例外を扱えない
// ❌ コンパイルエラー
// list.stream().map(path -> Files.readString(path));

// ✅ ラッパーを使用
@FunctionalInterface
interface ThrowingFunction<T, R> {
    R apply(T t) throws Exception;
}

static <T, R> Function<T, R> unchecked(ThrowingFunction<T, R> f) {
    return t -> {
        try {
            return f.apply(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    };
}

// 使用例
list.stream()
    .map(unchecked(path -> Files.readString(Path.of(path))))
    .collect(Collectors.toList());
```

### 2.3 JavaScript / TypeScript のエラーハンドリング

```javascript
// JavaScript: try-catch-finally

// ========================================
// 基本的な例外処理
// ========================================
try {
    const data = JSON.parse(invalidJson);
    processData(data);
} catch (error) {
    if (error instanceof SyntaxError) {
        console.error("Invalid JSON:", error.message);
    } else if (error instanceof TypeError) {
        console.error("Type error:", error.message);
    } else {
        console.error("Unknown error:", error);
    }
} finally {
    cleanup();
}

// ========================================
// Promise のエラーハンドリング
// ========================================
// .catch() メソッド
fetch("/api/data")
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => processData(data))
    .catch(error => {
        console.error("Fetch failed:", error);
    });

// async/await
async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new HttpError(response.status, response.statusText);
        }
        return await response.json();
    } catch (error) {
        if (error instanceof HttpError) {
            console.error(`HTTP Error ${error.status}: ${error.message}`);
        } else if (error instanceof TypeError) {
            console.error("Network error:", error.message);
        } else {
            throw error; // 予期しないエラーは再送出
        }
    }
}

// ========================================
// Promise.allSettled（全ての結果を取得）
// ========================================
const results = await Promise.allSettled([
    fetch("/api/users"),
    fetch("/api/posts"),
    fetch("/api/comments"),
]);

for (const result of results) {
    if (result.status === "fulfilled") {
        console.log("Success:", result.value);
    } else {
        console.error("Failed:", result.reason);
    }
}

// ========================================
// AggregateError（Promise.any のエラー）
// ========================================
try {
    const first = await Promise.any([
        fetch("/api/primary"),
        fetch("/api/secondary"),
        fetch("/api/tertiary"),
    ]);
} catch (error) {
    if (error instanceof AggregateError) {
        console.error("All promises failed:");
        for (const e of error.errors) {
            console.error(" -", e.message);
        }
    }
}

// ========================================
// グローバルエラーハンドリング
// ========================================
// ブラウザ
window.addEventListener("error", (event) => {
    reportError({
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error,
    });
});

window.addEventListener("unhandledrejection", (event) => {
    reportError({
        message: "Unhandled promise rejection",
        reason: event.reason,
    });
    event.preventDefault();
});

// Node.js
process.on("uncaughtException", (error) => {
    logger.fatal("Uncaught exception:", error);
    process.exit(1);
});

process.on("unhandledRejection", (reason, promise) => {
    logger.error("Unhandled rejection:", reason);
});
```

```typescript
// TypeScript: カスタムエラークラス
class AppError extends Error {
    constructor(
        message: string,
        public readonly code: string,
        public readonly statusCode: number,
        public readonly details?: Record<string, unknown>,
    ) {
        super(message);
        this.name = "AppError";
        // プロトタイプチェーンの修復（TypeScript のクラス継承の注意点）
        Object.setPrototypeOf(this, new.target.prototype);
    }

    toJSON() {
        return {
            name: this.name,
            code: this.code,
            message: this.message,
            statusCode: this.statusCode,
            details: this.details,
        };
    }
}

class NotFoundError extends AppError {
    constructor(resource: string, id?: string) {
        const msg = id
            ? `${resource} with id '${id}' not found`
            : `${resource} not found`;
        super(msg, "NOT_FOUND", 404, { resource, id });
        this.name = "NotFoundError";
    }
}

class ValidationError extends AppError {
    constructor(public readonly fields: Record<string, string[]>) {
        const messages = Object.entries(fields)
            .flatMap(([field, errors]) => errors.map(e => `${field}: ${e}`));
        super(
            `Validation failed: ${messages.join("; ")}`,
            "VALIDATION_ERROR",
            400,
            { fields },
        );
        this.name = "ValidationError";
    }
}

class ConflictError extends AppError {
    constructor(resource: string, conflictField: string) {
        super(
            `${resource} already exists with this ${conflictField}`,
            "CONFLICT",
            409,
            { resource, conflictField },
        );
        this.name = "ConflictError";
    }
}

// ========================================
// Express.js のエラーハンドリングミドルウェア
// ========================================
function errorHandler(err: Error, req: Request, res: Response, next: NextFunction) {
    if (err instanceof AppError) {
        res.status(err.statusCode).json(err.toJSON());
    } else {
        logger.error("Unhandled error:", err);
        res.status(500).json({
            name: "InternalError",
            code: "INTERNAL_ERROR",
            message: "An internal error occurred",
            statusCode: 500,
        });
    }
}

// ========================================
// 型安全なエラーハンドリング（TypeScript）
// ========================================
// エラー型の判別関数
function isAppError(error: unknown): error is AppError {
    return error instanceof AppError;
}

function isNotFoundError(error: unknown): error is NotFoundError {
    return error instanceof NotFoundError;
}

// 安全なエラーハンドリング
async function handleRequest(req: Request): Promise<Response> {
    try {
        const result = await processRequest(req);
        return new Response(JSON.stringify(result), { status: 200 });
    } catch (error) {
        if (isNotFoundError(error)) {
            return new Response(JSON.stringify({ error: error.message }), { status: 404 });
        }
        if (isAppError(error)) {
            return new Response(JSON.stringify(error.toJSON()), { status: error.statusCode });
        }
        // 予期しないエラー
        console.error("Unexpected error:", error);
        return new Response(JSON.stringify({ error: "Internal server error" }), { status: 500 });
    }
}
```

### 2.4 例外の問題点

```
問題1: 見えない制御フロー
  → 関数のシグネチャを見ただけでは、どの例外が飛ぶか分からない
  → Java のチェック例外は解決策だが、冗長になりがち
  → JavaScript/Python は完全に暗黙的

問題2: パフォーマンスコスト
  → スタックトレースの生成は高コスト
  → 正常系で例外を使うとパフォーマンス劣化
  → 例外はあくまで「例外的」な状況のために

問題3: 例外の飲み込み
  → catch して何もしない → バグの隠蔽
  → 空の catch ブロックは最悪のアンチパターン

問題4: 例外安全性の保証が困難
  → リソースリークのリスク
  → 部分的に更新された状態
  → トランザクションの整合性
```

---

## 3. Result 型（値ベースのエラー処理）

### 3.1 Rust の Result<T, E>

```rust
// Rust: Result<T, E> — 成功か失敗かを型で表現

// ========================================
// 基本的な使用
// ========================================
fn parse_number(s: &str) -> Result<i32, ParseIntError> {
    s.parse::<i32>()
}

// パターンマッチで処理
match parse_number("42") {
    Ok(n) => println!("Parsed: {}", n),
    Err(e) => println!("Error: {}", e),
}

// ========================================
// ? 演算子（エラー伝播の省略記法）
// ========================================
fn read_config() -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string("config.toml")?;  // エラーなら即return
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}

// ? 演算子は以下の糖衣構文:
fn read_config_expanded() -> Result<Config, Box<dyn Error>> {
    let content = match fs::read_to_string("config.toml") {
        Ok(c) => c,
        Err(e) => return Err(e.into()),  // From トレイトで変換
    };
    let config = match toml::from_str(&content) {
        Ok(c) => c,
        Err(e) => return Err(e.into()),
    };
    Ok(config)
}

// ========================================
// Result のメソッドチェーン
// ========================================
// map — Ok の中身を変換
let doubled: Result<i32, _> = parse_number("42").map(|n| n * 2);

// map_err — Err の中身を変換
let result: Result<i32, AppError> = parse_number("abc")
    .map_err(|e| AppError::Validation(e.to_string()));

// and_then — Result を返す関数を連鎖
fn validate_positive(n: i32) -> Result<i32, String> {
    if n > 0 { Ok(n) } else { Err("Must be positive".to_string()) }
}

let result = parse_number("42")
    .map_err(|e| e.to_string())
    .and_then(validate_positive)
    .map(|n| n * 2);

// unwrap_or — デフォルト値
let n = parse_number("abc").unwrap_or(0);

// unwrap_or_else — 遅延評価のデフォルト値
let n = parse_number("abc").unwrap_or_else(|e| {
    eprintln!("Parse error: {}", e);
    0
});

// ok — Result を Option に変換（Err を捨てる）
let opt: Option<i32> = parse_number("42").ok();

// ========================================
// 複数の Result の組み合わせ
// ========================================
// 順次処理（? 演算子）
fn process() -> Result<Output, AppError> {
    let a = step1()?;
    let b = step2(a)?;
    let c = step3(b)?;
    Ok(c)
}

// collect で Vec<Result<T, E>> → Result<Vec<T>, E>
fn parse_all(inputs: &[&str]) -> Result<Vec<i32>, ParseIntError> {
    inputs.iter().map(|s| s.parse::<i32>()).collect()
}

// 全て成功 → Ok(vec![1, 2, 3])
// いずれかが失敗 → 最初の Err

// ========================================
// カスタムエラー型（thiserror クレート）
// ========================================
use thiserror::Error;

#[derive(Error, Debug)]
enum AppError {
    #[error("User not found: {0}")]
    NotFound(String),

    #[error("Database error")]
    Database(#[from] sqlx::Error),

    #[error("Validation failed: {0}")]
    Validation(String),

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("Rate limit exceeded: retry after {retry_after}s")]
    RateLimit { retry_after: u64 },

    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

// From トレイトの自動実装（#[from] アトリビュート）
// sqlx::Error → AppError::Database への自動変換
// → ? 演算子で自動的に変換される

// ========================================
// anyhow クレート（プロトタイプや CLIツール向け）
// ========================================
use anyhow::{Context, Result, bail, ensure};

fn read_config(path: &str) -> Result<Config> {
    let content = fs::read_to_string(path)
        .context(format!("Failed to read config file: {}", path))?;

    let config: Config = toml::from_str(&content)
        .context("Failed to parse config file")?;

    ensure!(config.port > 0, "Port must be positive, got {}", config.port);

    if config.host.is_empty() {
        bail!("Host must not be empty");
    }

    Ok(config)
}

// ========================================
// エラーの伝播とコンテキスト追加
// ========================================
fn get_user(id: u32) -> Result<User, AppError> {
    let user = db.find_user(id)
        .map_err(AppError::Database)?;

    match user {
        Some(u) => Ok(u),
        None => Err(AppError::NotFound(format!("user id={}", id))),
    }
}

fn get_user_profile(user_id: u32) -> Result<UserProfile, AppError> {
    let user = get_user(user_id)?;  // AppError が自動伝播
    let profile = db.find_profile(user.id)
        .map_err(AppError::Database)?;
    Ok(UserProfile { user, profile })
}

// ========================================
// Result と Option の相互変換
// ========================================
// Option → Result
let value: Result<i32, &str> = some_option.ok_or("Value is None");
let value: Result<i32, AppError> = some_option
    .ok_or_else(|| AppError::NotFound("value".to_string()));

// Result → Option
let value: Option<i32> = some_result.ok();   // Err を捨てる
let error: Option<E> = some_result.err();    // Ok を捨てる

// transpose（Option<Result<T, E>> ⟷ Result<Option<T>, E>）
let opt_result: Option<Result<i32, E>> = Some(Ok(42));
let result_opt: Result<Option<i32>, E> = opt_result.transpose();
// → Ok(Some(42))
```

### 3.2 Haskell の Either / Maybe

```haskell
-- Haskell: Either a b（Left = エラー、Right = 成功）

-- ========================================
-- Maybe: 値の有無
-- ========================================
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing
safeDivide a b = Just (a / b)

safeHead :: [a] -> Maybe a
safeHead []    = Nothing
safeHead (x:_) = Just x

-- Maybe の連鎖（do 記法）
lookupAddress :: Map String User -> String -> Maybe String
lookupAddress users name = do
    user <- Map.lookup name users      -- Maybe User
    address <- userAddress user        -- Maybe Address
    return (addressCity address)       -- Maybe String

-- ========================================
-- Either: 詳細なエラー情報
-- ========================================
data AppError
    = NotFound String
    | ValidationError [String]
    | DatabaseError String
    | AuthError String
    deriving (Show)

parseAge :: String -> Either AppError Int
parseAge s = case reads s of
    [(n, "")] | n >= 0 && n <= 150 -> Right n
              | otherwise -> Left (ValidationError ["Age must be 0-150"])
    _ -> Left (ValidationError ["Invalid number format"])

-- Either の連鎖
createUser :: String -> String -> Either AppError User
createUser name ageStr = do
    validatedName <- validateName name
    age <- parseAge ageStr
    Right (User validatedName age)

validateName :: String -> Either AppError String
validateName name
    | null name = Left (ValidationError ["Name is required"])
    | length name < 2 = Left (ValidationError ["Name too short"])
    | otherwise = Right name

-- ========================================
-- ExceptT（モナド変換子で IO と Either を組み合わせ）
-- ========================================
import Control.Monad.Except

type App = ExceptT AppError IO

getUser :: String -> App User
getUser userId = do
    result <- liftIO $ queryDB ("SELECT * FROM users WHERE id = " ++ userId)
    case result of
        Nothing -> throwError (NotFound $ "User " ++ userId)
        Just user -> return user

-- ========================================
-- MonadError 型クラス
-- ========================================
handleError :: MonadError AppError m => m User -> m User
handleError action = catchError action $ \err -> case err of
    NotFound msg -> do
        liftIO $ putStrLn $ "Warning: " ++ msg
        return defaultUser
    _ -> throwError err  -- 他のエラーは再送出
```

### 3.3 TypeScript の Result パターン

```typescript
// TypeScript: Result型をユニオン型で表現

// ========================================
// 自作 Result 型
// ========================================
type Result<T, E> =
    | { ok: true; value: T }
    | { ok: false; error: E };

function ok<T>(value: T): Result<T, never> {
    return { ok: true, value };
}

function err<E>(error: E): Result<never, E> {
    return { ok: false, error };
}

// ========================================
// Result の便利メソッド（ユーティリティ関数）
// ========================================
function mapResult<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => U
): Result<U, E> {
    if (result.ok) {
        return ok(fn(result.value));
    }
    return result;
}

function flatMapResult<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => Result<U, E>
): Result<U, E> {
    if (result.ok) {
        return fn(result.value);
    }
    return result;
}

function mapError<T, E, F>(
    result: Result<T, E>,
    fn: (error: E) => F
): Result<T, F> {
    if (result.ok) {
        return result;
    }
    return err(fn(result.error));
}

// ========================================
// 実務的な使用例
// ========================================
type AppError =
    | { code: "NOT_FOUND"; resource: string }
    | { code: "VALIDATION"; fields: Record<string, string[]> }
    | { code: "UNAUTHORIZED"; reason: string }
    | { code: "INTERNAL"; message: string };

function parseNumber(s: string): Result<number, string> {
    const n = Number(s);
    if (isNaN(n)) {
        return err(`Invalid number: ${s}`);
    }
    return ok(n);
}

function validateAge(age: number): Result<number, AppError> {
    if (age < 0 || age > 150) {
        return err({
            code: "VALIDATION",
            fields: { age: ["Must be between 0 and 150"] },
        });
    }
    return ok(age);
}

// Result のパイプライン
function processAgeInput(input: string): Result<number, AppError> {
    const parsed = parseNumber(input);
    if (!parsed.ok) {
        return err({
            code: "VALIDATION",
            fields: { age: [parsed.error] },
        });
    }
    return validateAge(parsed.value);
}

// ========================================
// neverthrow ライブラリ
// ========================================
import { ok, err, Result, ResultAsync } from 'neverthrow';

function divide(a: number, b: number): Result<number, string> {
    if (b === 0) return err("Division by zero");
    return ok(a / b);
}

// メソッドチェーン
const result = divide(10, 2)
    .map(n => n * 3)
    .mapErr(e => new Error(e))
    .andThen(n => n > 0 ? ok(n) : err(new Error("Must be positive")));

// ResultAsync（非同期版）
function fetchUser(id: string): ResultAsync<User, AppError> {
    return ResultAsync.fromPromise(
        fetch(`/api/users/${id}`).then(r => r.json()),
        (e) => ({ code: "INTERNAL" as const, message: String(e) })
    );
}

// combine（複数の Result を結合）
const combined = Result.combine([
    parseNumber("10"),
    parseNumber("20"),
    parseNumber("30"),
]);
// → ok([10, 20, 30]) or err("...")
```

### 3.4 Go のエラーハンドリング

```go
// Go: 複数戻り値でエラーを返す

// ========================================
// 基本パターン
// ========================================
func readFile(path string) (string, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return "", fmt.Errorf("read %s: %w", path, err)
    }
    return string(data), nil
}

// 呼び出し側
content, err := readFile("config.txt")
if err != nil {
    log.Fatal(err)
}
// err チェックを忘れると、ゼロ値（""）で処理が続行 → バグの温床

// ========================================
// エラーのラッピング（Go 1.13+）
// ========================================
func processFile(path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        // %w でラップ → errors.Is/As で検査可能
        return fmt.Errorf("process file %s: %w", path, err)
    }
    return processData(data)
}

// エラーの検査
if errors.Is(err, os.ErrNotExist) {
    fmt.Println("File does not exist")
}

var pathErr *os.PathError
if errors.As(err, &pathErr) {
    fmt.Println("Path:", pathErr.Path)
}

// ========================================
// カスタムエラー型
// ========================================
type AppError struct {
    Code    string `json:"code"`
    Message string `json:"message"`
    Err     error  `json:"-"`
}

func (e *AppError) Error() string {
    if e.Err != nil {
        return fmt.Sprintf("%s: %s (%v)", e.Code, e.Message, e.Err)
    }
    return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *AppError) Unwrap() error {
    return e.Err
}

// センチネルエラー
var (
    ErrNotFound     = &AppError{Code: "NOT_FOUND", Message: "Resource not found"}
    ErrUnauthorized = &AppError{Code: "UNAUTHORIZED", Message: "Unauthorized"}
    ErrForbidden    = &AppError{Code: "FORBIDDEN", Message: "Forbidden"}
)

func NewNotFoundError(resource, id string) *AppError {
    return &AppError{
        Code:    "NOT_FOUND",
        Message: fmt.Sprintf("%s with id '%s' not found", resource, id),
    }
}

// ========================================
// エラーハンドリングのパターン
// ========================================
// パターン1: 即座にリターン
func getUser(id string) (*User, error) {
    user, err := db.FindUser(id)
    if err != nil {
        return nil, fmt.Errorf("get user %s: %w", id, err)
    }
    if user == nil {
        return nil, NewNotFoundError("User", id)
    }
    return user, nil
}

// パターン2: defer でクリーンアップ
func processWithTransaction(db *sql.DB) error {
    tx, err := db.Begin()
    if err != nil {
        return fmt.Errorf("begin transaction: %w", err)
    }
    defer func() {
        if err != nil {
            tx.Rollback()
        }
    }()

    if err = step1(tx); err != nil {
        return fmt.Errorf("step1: %w", err)
    }
    if err = step2(tx); err != nil {
        return fmt.Errorf("step2: %w", err)
    }

    return tx.Commit()
}

// パターン3: エラーグループ（複数のエラーを収集）
type MultiError struct {
    Errors []error
}

func (me *MultiError) Error() string {
    msgs := make([]string, len(me.Errors))
    for i, err := range me.Errors {
        msgs[i] = err.Error()
    }
    return strings.Join(msgs, "; ")
}

func validateUser(user *User) error {
    var errs []error
    if user.Name == "" {
        errs = append(errs, fmt.Errorf("name is required"))
    }
    if user.Email == "" {
        errs = append(errs, fmt.Errorf("email is required"))
    }
    if user.Age < 0 || user.Age > 150 {
        errs = append(errs, fmt.Errorf("age must be 0-150"))
    }
    if len(errs) > 0 {
        return &MultiError{Errors: errs}
    }
    return nil
}
```

---

## 4. 各言語のエラーハンドリング比較

```
┌──────────────┬──────────────────────┬────────────────────────────┐
│ 方式          │ 代表言語              │ 特徴                       │
├──────────────┼──────────────────────┼────────────────────────────┤
│ 例外          │ Python, Java, C#,    │ 暗黙的な伝播               │
│              │ JavaScript, Ruby      │ 見えない制御フロー          │
│              │                      │ 正常系と異常系の分離        │
├──────────────┼──────────────────────┼────────────────────────────┤
│ Result型      │ Rust, Haskell,       │ 明示的な伝播               │
│              │ Elm, OCaml, F#       │ 型で安全に表現             │
│              │                      │ コンパイル時に処理を強制    │
├──────────────┼──────────────────────┼────────────────────────────┤
│ エラーコード   │ C, Go               │ シンプルだが               │
│              │                      │ チェック忘れの危険          │
│              │                      │ Go は慣例で強制            │
├──────────────┼──────────────────────┼────────────────────────────┤
│ ハイブリッド   │ Swift(throw+Result)  │ 場面で使い分け             │
│              │ Kotlin(throw+Result)  │ 最大の柔軟性               │
│              │ Scala(Try+Either)    │                            │
├──────────────┼──────────────────────┼────────────────────────────┤
│ パニック      │ Rust(panic!),        │ 回復不可能なエラー          │
│              │ Go(panic/recover)    │ プロセス終了前提            │
└──────────────┴──────────────────────┴────────────────────────────┘
```

### 4.1 Swift のエラーハンドリング

```swift
// Swift: ハイブリッド方式（throws + Result + Optional）

// ========================================
// throws / do-catch
// ========================================
enum AppError: Error {
    case notFound(String)
    case validation([String])
    case unauthorized
}

func getUser(id: String) throws -> User {
    guard let user = db.findUser(id: id) else {
        throw AppError.notFound("User \(id)")
    }
    return user
}

// 呼び出し側
do {
    let user = try getUser(id: "123")
    print(user.name)
} catch AppError.notFound(let resource) {
    print("Not found: \(resource)")
} catch {
    print("Unexpected error: \(error)")
}

// try? — エラーを Optional に変換
let user: User? = try? getUser(id: "123")

// try! — エラー時にクラッシュ（確信がある場合のみ）
let user: User = try! getUser(id: "known-id")

// ========================================
// Result 型（Swift 5+）
// ========================================
func fetchData(url: URL, completion: (Result<Data, Error>) -> Void) {
    URLSession.shared.dataTask(with: url) { data, response, error in
        if let error = error {
            completion(.failure(error))
            return
        }
        if let data = data {
            completion(.success(data))
        }
    }.resume()
}

// Result の利用
fetchData(url: apiURL) { result in
    switch result {
    case .success(let data):
        processData(data)
    case .failure(let error):
        handleError(error)
    }
}

// map / flatMap
let decoded: Result<User, Error> = result
    .map { data in try JSONDecoder().decode(User.self, from: data) }
```

### 4.2 Kotlin のエラーハンドリング

```kotlin
// Kotlin: ハイブリッド方式（例外 + Result + sealed class）

// ========================================
// sealed class でエラーを表現
// ========================================
sealed class AppResult<out T> {
    data class Success<T>(val value: T) : AppResult<T>()
    data class Failure(val error: AppError) : AppResult<Nothing>()
}

sealed class AppError {
    data class NotFound(val resource: String) : AppError()
    data class Validation(val errors: List<String>) : AppError()
    data object Unauthorized : AppError()
}

fun getUser(id: String): AppResult<User> {
    val user = db.findUser(id) ?: return AppResult.Failure(
        AppError.NotFound("User $id")
    )
    return AppResult.Success(user)
}

// when 式での処理
when (val result = getUser("123")) {
    is AppResult.Success -> println("User: ${result.value.name}")
    is AppResult.Failure -> when (result.error) {
        is AppError.NotFound -> println("Not found: ${result.error.resource}")
        is AppError.Validation -> println("Validation: ${result.error.errors}")
        is AppError.Unauthorized -> println("Unauthorized")
    }
}

// ========================================
// Kotlin stdlib の Result 型
// ========================================
val result: Result<Int> = runCatching {
    "42".toInt()
}

result
    .map { it * 2 }
    .onSuccess { println("Value: $it") }
    .onFailure { println("Error: ${it.message}") }

val value = result.getOrDefault(0)
val value2 = result.getOrElse { error -> handleError(error); 0 }

// ========================================
// require / check（前提条件チェック）
// ========================================
fun processOrder(order: Order) {
    require(order.items.isNotEmpty()) { "Order must have items" }
    require(order.total > 0) { "Order total must be positive" }
    check(order.status == OrderStatus.PENDING) { "Order must be pending" }

    // require → IllegalArgumentException
    // check → IllegalStateException
}
```

---

## 5. エラー処理のベストプラクティス

### 5.1 回復可能 vs 回復不可能

```
1. 回復可能（Recoverable）なエラー
   → ファイルが見つからない → 別のパスを試す
   → ネットワークタイムアウト → リトライ
   → バリデーションエラー → ユーザーにフィードバック
   → 認証エラー → 再ログイン

   Rust:  Result<T, E>
   Go:    error を返す
   Java:  チェック例外
   Python: 特定の例外をキャッチ

2. 回復不可能（Unrecoverable）なエラー
   → メモリ不足 → プロセスを終了
   → 不変条件の違反 → バグ、プログラムを修正すべき
   → 設定ファイルの致命的な欠落 → 起動時に失敗

   Rust:  panic!()
   Go:    panic()（基本的に使わない）
   Java:  RuntimeException
   Python: SystemExit
```

### 5.2 エラーの粒度設計

```rust
// Rust: エラーの粒度を適切に設計

// ❌ 粒度が粗すぎる
enum Error {
    SomethingWentWrong(String),
}

// ❌ 粒度が細かすぎる
enum Error {
    FileNotFoundAtPath(PathBuf),
    FilePermissionDenied(PathBuf),
    FileAlreadyExists(PathBuf),
    FileIsDirectory(PathBuf),
    // ... 何十もの種類
}

// ✅ 適切な粒度（ドメインに合わせて）
#[derive(Error, Debug)]
enum UserServiceError {
    #[error("User not found: {0}")]
    NotFound(String),

    #[error("Validation failed")]
    Validation(Vec<ValidationIssue>),

    #[error("Duplicate email: {0}")]
    DuplicateEmail(String),

    #[error("Database error")]
    Database(#[source] sqlx::Error),

    #[error("External service error")]
    ExternalService(#[source] reqwest::Error),
}

// レイヤーごとにエラー型を分ける
// リポジトリ層
#[derive(Error, Debug)]
enum RepositoryError {
    #[error("Record not found")]
    NotFound,
    #[error("Constraint violation: {0}")]
    Constraint(String),
    #[error("Connection error")]
    Connection(#[source] sqlx::Error),
}

// サービス層（リポジトリのエラーを変換）
impl From<RepositoryError> for UserServiceError {
    fn from(err: RepositoryError) -> Self {
        match err {
            RepositoryError::NotFound => UserServiceError::NotFound("unknown".to_string()),
            RepositoryError::Constraint(msg) if msg.contains("email") => {
                UserServiceError::DuplicateEmail(msg)
            }
            other => UserServiceError::Database(other.into()),
        }
    }
}
```

### 5.3 エラーメッセージの設計

```
エラーメッセージの3要素:
  1. 何が（What）    — 何が失敗したか
  2. どこで（Where）  — どのリソース/操作で
  3. なぜ（Why）     — 原因は何か

❌ 悪い例
  "Error occurred"
  "Failed"
  "Something went wrong"
  "Invalid input"

✅ 良い例
  "Failed to read config file '/etc/app.toml': Permission denied"
  "User 'alice@example.com' not found in database 'users'"
  "Validation failed for field 'email': must contain '@'"
  "Connection to Redis at localhost:6379 timed out after 5s"

エラーメッセージのガイドライン:
  - 技術的すぎず、かつ曖昧すぎない
  - 機密情報（パスワード、トークン）を含めない
  - ユーザー向けメッセージと開発者向けメッセージを分ける
  - コンテキスト情報（ID、パス、パラメータ）を含める
  - 解決策のヒントを含める（可能な場合）
```

### 5.4 エラーの伝播パターン

```rust
// Rust: エラーの伝播とコンテキスト追加

// パターン1: そのまま伝播（? 演算子）
fn load_config() -> Result<Config, AppError> {
    let path = find_config_path()?;
    let content = read_file(&path)?;
    let config = parse_config(&content)?;
    Ok(config)
}

// パターン2: コンテキストを追加して伝播
fn load_config() -> Result<Config, anyhow::Error> {
    let path = find_config_path()
        .context("Failed to find config file")?;
    let content = read_file(&path)
        .context(format!("Failed to read {}", path.display()))?;
    let config = parse_config(&content)
        .context("Failed to parse config")?;
    Ok(config)
}

// パターン3: エラーを変換して伝播
fn load_user_config(user_id: &str) -> Result<UserConfig, UserError> {
    let path = find_config_path()
        .map_err(|_| UserError::ConfigNotFound(user_id.to_string()))?;
    let content = read_file(&path)
        .map_err(|e| UserError::ConfigReadError {
            user_id: user_id.to_string(),
            source: e,
        })?;
    Ok(parse_user_config(&content)?)
}

// パターン4: エラーを回復
fn load_config_with_fallback() -> Config {
    match load_config() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Warning: Failed to load config: {}. Using defaults.", e);
            Config::default()
        }
    }
}
```

```python
# Python: エラーの伝播パターン

# パターン1: そのまま伝播（何もしない）
def load_config():
    path = find_config_path()    # 例外が伝播する
    content = read_file(path)    # 例外が伝播する
    return parse_config(content) # 例外が伝播する

# パターン2: コンテキストを追加
def load_config():
    try:
        path = find_config_path()
    except FileNotFoundError:
        raise ConfigError("Config file not found") from None

    try:
        content = read_file(path)
    except IOError as e:
        raise ConfigError(f"Failed to read {path}") from e

    try:
        return parse_config(content)
    except ValueError as e:
        raise ConfigError(f"Invalid config format") from e

# パターン3: 回復
def load_config_with_fallback():
    try:
        return load_config()
    except ConfigError as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")
        return Config.default()

# パターン4: リトライ
def load_with_retry(url, max_retries=3, delay=1.0):
    last_error = None
    for attempt in range(max_retries):
        try:
            return fetch(url)
        except (ConnectionError, TimeoutError) as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))  # 指数バックオフ
    raise last_error
```

---

## 6. 実践パターン

### 6.1 Web APIのエラーハンドリング

```python
# Python (FastAPI): エラーハンドリング

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# カスタム例外ハンドラ
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": str(exc),
                "details": exc.details,
            }
        },
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Validation failed",
                "details": {"fields": exc.errors},
            }
        },
    )

@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
            }
        },
    )

@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    user = await user_service.get_user(user_id)  # NotFoundError が伝播
    return user
```

```rust
// Rust (Axum): エラーハンドリング
use axum::{
    response::{IntoResponse, Response},
    http::StatusCode,
    Json,
};

#[derive(Debug, thiserror::Error)]
enum ApiError {
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Validation error")]
    Validation(Vec<String>),
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Internal error")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match &self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "NOT_FOUND", msg.clone()),
            ApiError::Validation(errors) => (
                StatusCode::BAD_REQUEST,
                "VALIDATION_ERROR",
                errors.join(", "),
            ),
            ApiError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "UNAUTHORIZED",
                "Unauthorized".to_string(),
            ),
            ApiError::Internal(e) => {
                tracing::error!("Internal error: {:?}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "INTERNAL_ERROR",
                    "An internal error occurred".to_string(),
                )
            }
        };

        let body = serde_json::json!({
            "error": {
                "code": code,
                "message": message,
            }
        });

        (status, Json(body)).into_response()
    }
}

async fn get_user(Path(id): Path<String>) -> Result<Json<User>, ApiError> {
    let user = user_service.get_user(&id).await?;  // ApiError に変換
    Ok(Json(user))
}
```

### 6.2 バッチ処理のエラーハンドリング

```python
# Python: バッチ処理でのエラーハンドリング

from dataclasses import dataclass, field

@dataclass
class BatchResult:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    errors: list[dict] = field(default_factory=list)

def process_batch(items: list[dict]) -> BatchResult:
    result = BatchResult(total=len(items))

    for i, item in enumerate(items):
        try:
            process_item(item)
            result.succeeded += 1
        except ValidationError as e:
            result.failed += 1
            result.errors.append({
                "index": i,
                "item": item,
                "error_type": "validation",
                "message": str(e),
            })
        except ExternalServiceError as e:
            result.failed += 1
            result.errors.append({
                "index": i,
                "item": item,
                "error_type": "external_service",
                "message": str(e),
            })
            # 外部サービスエラーが続く場合は中断
            if result.failed > result.total * 0.5:
                logger.error("Too many failures, aborting batch")
                break
        except Exception as e:
            result.failed += 1
            result.errors.append({
                "index": i,
                "item": item,
                "error_type": "unexpected",
                "message": str(e),
            })
            logger.exception(f"Unexpected error processing item {i}")

    return result
```

```rust
// Rust: バッチ処理のエラーハンドリング
#[derive(Debug)]
struct BatchResult<T> {
    succeeded: Vec<T>,
    failed: Vec<BatchError>,
}

#[derive(Debug)]
struct BatchError {
    index: usize,
    error: AppError,
}

fn process_batch<T, F>(items: &[Item], processor: F) -> BatchResult<T>
where
    F: Fn(&Item) -> Result<T, AppError>,
{
    let mut result = BatchResult {
        succeeded: Vec::new(),
        failed: Vec::new(),
    };

    for (index, item) in items.iter().enumerate() {
        match processor(item) {
            Ok(output) => result.succeeded.push(output),
            Err(error) => result.failed.push(BatchError { index, error }),
        }
    }

    result
}

// 並列バッチ処理（rayon）
fn process_batch_parallel(items: &[Item]) -> BatchResult<Output> {
    let results: Vec<(usize, Result<Output, AppError>)> = items
        .par_iter()
        .enumerate()
        .map(|(i, item)| (i, process_item(item)))
        .collect();

    let mut batch_result = BatchResult {
        succeeded: Vec::new(),
        failed: Vec::new(),
    };

    for (index, result) in results {
        match result {
            Ok(output) => batch_result.succeeded.push(output),
            Err(error) => batch_result.failed.push(BatchError { index, error }),
        }
    }

    batch_result
}
```

### 6.3 リトライパターン

```typescript
// TypeScript: リトライパターン

interface RetryOptions {
    maxRetries: number;
    baseDelay: number;     // ミリ秒
    maxDelay: number;      // ミリ秒
    backoffFactor: number; // 指数バックオフの倍率
    retryableErrors?: string[];
}

async function withRetry<T>(
    operation: () => Promise<T>,
    options: RetryOptions,
): Promise<T> {
    const {
        maxRetries,
        baseDelay,
        maxDelay,
        backoffFactor,
        retryableErrors,
    } = options;

    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await operation();
        } catch (error) {
            lastError = error as Error;

            // リトライ可能なエラーかチェック
            if (retryableErrors && !isRetryable(error, retryableErrors)) {
                throw error;
            }

            if (attempt < maxRetries) {
                const delay = Math.min(
                    baseDelay * Math.pow(backoffFactor, attempt),
                    maxDelay,
                );
                // ジッター（ランダムな揺らぎ）を追加
                const jitter = delay * 0.1 * Math.random();
                console.log(
                    `Attempt ${attempt + 1} failed, retrying in ${delay + jitter}ms...`
                );
                await sleep(delay + jitter);
            }
        }
    }

    throw lastError;
}

function isRetryable(error: unknown, retryableErrors: string[]): boolean {
    if (error instanceof Error) {
        return retryableErrors.some(re =>
            error.message.includes(re) || error.name === re
        );
    }
    return false;
}

// 使用例
const data = await withRetry(
    () => fetch("/api/data").then(r => r.json()),
    {
        maxRetries: 3,
        baseDelay: 1000,
        maxDelay: 10000,
        backoffFactor: 2,
        retryableErrors: ["ECONNREFUSED", "ETIMEDOUT", "503"],
    },
);
```

---

## 7. エラーハンドリングのアンチパターン

### 7.1 よくある間違い

```python
# ❌ アンチパターン1: 例外の飲み込み
try:
    process(data)
except Exception:
    pass  # エラーを完全に無視 → バグの隠蔽

# ✅ 改善: 少なくともログに記録
try:
    process(data)
except Exception as e:
    logger.error(f"Failed to process data: {e}")
    # 必要に応じてリレイズ
```

```python
# ❌ アンチパターン2: 広すぎる例外キャッチ
try:
    user = get_user(user_id)
    order = create_order(user, items)
    payment = process_payment(order)
except Exception as e:
    return {"error": str(e)}  # 何が失敗したか分からない

# ✅ 改善: 具体的な例外をキャッチ
try:
    user = get_user(user_id)
except UserNotFoundError:
    return {"error": f"User {user_id} not found"}

try:
    order = create_order(user, items)
except ValidationError as e:
    return {"error": f"Invalid order: {e}"}

try:
    payment = process_payment(order)
except PaymentError as e:
    return {"error": f"Payment failed: {e}"}
```

```javascript
// ❌ アンチパターン3: throw に文字列を使用
throw "Something went wrong";  // Error オブジェクトでない

// ✅ 改善: Error オブジェクトを使用
throw new Error("Something went wrong");
throw new AppError("Something went wrong", "UNKNOWN", 500);
```

```go
// ❌ アンチパターン4: エラーチェックの省略
data, _ := readFile(path)  // エラーを無視
processData(data)           // data がゼロ値でクラッシュの可能性

// ✅ 改善: 必ずエラーをチェック
data, err := readFile(path)
if err != nil {
    return fmt.Errorf("read file: %w", err)
}
processData(data)
```

```rust
// ❌ アンチパターン5: unwrap() の乱用
let config = load_config().unwrap();  // パニック
let user = get_user(id).unwrap();     // パニック

// ✅ 改善: 適切なエラーハンドリング
let config = load_config()
    .context("Failed to load config")?;
let user = get_user(id)
    .map_err(|e| AppError::NotFound(format!("user {}", id)))?;

// unwrap() が許される場面:
// - テストコード
// - 論理的に失敗しないことが証明できる場合
// - プロトタイプ（TODO コメント付き）
let regex = Regex::new(r"^\d+$").unwrap();  // コンパイル時に確定するリテラル
```

---

## まとめ

| 方式 | エラー伝播 | 型安全 | チェック強制 | 代表言語 |
|------|----------|--------|-----------|---------|
| 例外 | 暗黙（throw） | 低い | Java のみ | Python, Java, JS |
| Result | 明示（?/map） | 高い | コンパイル時 | Rust, Haskell |
| Either | 明示（bind） | 高い | コンパイル時 | Haskell, Scala |
| エラーコード | 明示（if err） | 低い | なし | Go, C |
| ハイブリッド | 両方 | 中 | 部分的 | Swift, Kotlin |

### 判断基準

```
例外を使うべき場面:
  - 呼び出し階層が深く、上位でまとめて処理したい
  - ライブラリが例外を使う言語（Python, Java, JS）

Result を使うべき場面:
  - エラーが「予期される」もの（バリデーション、検索ミス）
  - 型安全性を最大限に活用したい
  - 関数型スタイルのコードベース

エラーコードを使うべき場面:
  - シンプルさが最優先（C, Go）
  - エラーの種類が少ない
  - パフォーマンスが重要

ハイブリッドを使うべき場面:
  - チームの習熟度に合わせて選択
  - API 境界と内部ロジックで使い分け
```

---

## 次に読むべきガイド
→ [[03-iterators-and-generators.md]] — イテレータとジェネレータ

---

## 参考文献
1. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.9, 2023.
2. Bloch, J. "Effective Java." 3rd Ed, Item 69-77, Addison-Wesley, 2018.
3. Martin, R. "Clean Code." Ch.7 (Error Handling), Prentice Hall, 2008.
4. Donovan, A. & Kernighan, B. "The Go Programming Language." Ch.5.4, Addison-Wesley, 2015.
5. "Error Handling in Rust." doc.rust-lang.org/book/ch09-00-error-handling.html.
6. "PEP 3134: Exception Chaining and Embedded Tracebacks." python.org, 2005.
7. "thiserror crate documentation." docs.rs/thiserror.
8. "anyhow crate documentation." docs.rs/anyhow.
9. "neverthrow library documentation." github.com/supermacro/neverthrow.
10. Lipovaca, M. "Learn You a Haskell for Great Good!" Ch.8, No Starch Press, 2011.
