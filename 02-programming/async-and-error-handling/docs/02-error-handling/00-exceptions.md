# 例外処理

> 例外（Exception）は「通常の制御フローでは処理できない異常事態」を表す仕組み。try/catch/finally の正しい使い方、例外階層の設計、checked vs unchecked の議論を理解する。

## この章で学ぶこと

- [ ] 例外処理の仕組みとコールスタックの巻き戻しを理解する
- [ ] 例外の適切な使い方と濫用の違いを把握する
- [ ] 各言語の例外モデルの違いを学ぶ
- [ ] try/catch/finally の正しいパターンとアンチパターンを身につける
- [ ] 例外安全性（Exception Safety）の概念を理解する
- [ ] パフォーマンスへの影響と最適な使い分けを学ぶ

---

## 1. 例外処理の基礎概念

### 1.1 例外とは何か

例外（Exception）は、プログラムの正常な実行フローでは処理できない異常事態を表現する仕組みである。例外が発生すると、通常の制御フロー（順次実行、条件分岐、ループ）を中断し、コールスタックを巻き戻しながら適切なハンドラを探す。

```
例外処理の基本的な流れ:

  1. 例外の発生（throw / raise）
     → 関数が「この状況は自分では処理できない」と宣言

  2. コールスタックの巻き戻し（Stack Unwinding）
     → catch/except が見つかるまでコールスタックを遡る
     → 途中の関数は全てスキップされる

  3. 例外の捕捉（catch / except）
     → 適切なハンドラが例外を処理する

  4. クリーンアップ（finally / defer / with）
     → リソースの解放を保証する

例外処理がない世界:
  → 全ての関数がエラーコードを返す必要がある
  → 呼び出し元は毎回エラーチェックが必要
  → エラーハンドリングコードが本来のロジックを埋め尽くす
  → C言語のerrno方式がまさにこれ

例外処理がある世界:
  → 正常系のコードと異常系のコードを分離できる
  → エラーは自動的に伝播する（明示的な受け渡し不要）
  → 処理できる場所で一括して対応できる
```

### 1.2 例外の歴史

```
例外処理の進化:

  1960年代: PL/I の ON 条件ハンドリング
    → 最初の構造化例外処理

  1985年: C++ に例外が導入
    → try/catch/throw の原型

  1995年: Java の checked exception
    → コンパイラによるエラー処理の強制

  2000年代: C#, Python, Ruby の unchecked exception
    → checked exception への反省

  2010年代: Go の多値返却、Rust の Result<T, E>
    → 例外を使わないエラー処理の台頭

  2020年代: TypeScript の Effect, Rust の ? 演算子
    → 型安全なエラー処理の洗練
```

---

## 2. try/catch/finally

### 2.1 TypeScript での基本パターン

```typescript
// TypeScript: 基本的な例外処理
async function fetchUserData(userId: string): Promise<UserData> {
  try {
    const response = await fetch(`/api/users/${userId}`);

    if (!response.ok) {
      throw new HttpError(response.status, `HTTP ${response.status}`);
    }

    const data = await response.json();
    return data;

  } catch (error) {
    if (error instanceof HttpError) {
      if (error.status === 404) {
        throw new UserNotFoundError(userId);
      }
      throw new ApiError(`API error: ${error.message}`);
    }
    // ネットワークエラーなど
    throw new NetworkError("Network request failed");

  } finally {
    // 成功・失敗どちらでも実行される
    // リソースのクリーンアップに使う
    logger.log(`fetchUserData completed for ${userId}`);
  }
}
```

### 2.2 Python での例外処理

```python
# Python: 例外処理
def parse_config(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {path}: {e}")
    except PermissionError:
        raise ConfigError(f"Permission denied: {path}")
    finally:
        logger.info(f"Config parsing attempted for {path}")
```

### 2.3 Python の else 節

```python
# Python 固有: try/except/else/finally
def process_file(path: str) -> ProcessResult:
    """
    else節はtryブロックが例外なしに完了した場合にのみ実行される。
    finallyとは異なり、例外発生時には実行されない。
    """
    file_handle = None
    try:
        file_handle = open(path, 'r')
        data = file_handle.read()
    except FileNotFoundError:
        logger.warning(f"File not found: {path}")
        return ProcessResult(success=False, error="File not found")
    except PermissionError:
        logger.warning(f"Permission denied: {path}")
        return ProcessResult(success=False, error="Permission denied")
    else:
        # try が成功した場合のみ実行
        # ここで例外が発生しても except には捕捉されない
        result = parse_and_validate(data)
        logger.info(f"Successfully processed: {path}")
        return ProcessResult(success=True, data=result)
    finally:
        # 成功・失敗どちらでも実行
        if file_handle:
            file_handle.close()

# else 節を使う理由:
# 1. try ブロックを最小限に保てる（意図しない例外を捕捉しない）
# 2. 「成功時のみ」の処理を明確に分離できる
# 3. コードの意図が明確になる
```

### 2.4 Java の try-with-resources

```java
// Java: try-with-resources（AutoCloseable の自動クローズ）
public List<String> readLines(String path) throws IOException {
    // try() 内で宣言したリソースは自動的にクローズされる
    try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
        List<String> lines = new ArrayList<>();
        String line;
        while ((line = reader.readLine()) != null) {
            lines.add(line);
        }
        return lines;
    }
    // reader.close() が自動的に呼ばれる（例外の有無にかかわらず）
}

// 複数リソースの管理
public void copyFile(String src, String dst) throws IOException {
    try (
        InputStream in = new FileInputStream(src);
        OutputStream out = new FileOutputStream(dst)
    ) {
        byte[] buffer = new byte[8192];
        int bytesRead;
        while ((bytesRead = in.read(buffer)) != -1) {
            out.write(buffer, 0, bytesRead);
        }
    }
    // in と out の両方が自動クローズされる
    // close() 時の例外は Suppressed Exception として保持
}

// Suppressed Exception の取得
public void demonstrateSuppressedException() {
    try {
        try (AutoCloseableResource resource = new AutoCloseableResource()) {
            throw new RuntimeException("メインの例外");
        }
        // resource.close() でも例外が発生した場合
    } catch (Exception e) {
        System.out.println("メイン: " + e.getMessage());
        for (Throwable suppressed : e.getSuppressed()) {
            System.out.println("Suppressed: " + suppressed.getMessage());
        }
    }
}
```

### 2.5 C# の using ステートメント

```csharp
// C#: using ステートメント（IDisposable の自動破棄）
public async Task<string> ReadFileAsync(string path)
{
    // using 宣言（C# 8.0+）: スコープ終了時に自動Dispose
    using var stream = new FileStream(path, FileMode.Open);
    using var reader = new StreamReader(stream);
    return await reader.ReadToEndAsync();
}

// using ブロック（従来の書き方）
public void ProcessData(string connectionString)
{
    using (var connection = new SqlConnection(connectionString))
    {
        connection.Open();
        using (var command = new SqlCommand("SELECT * FROM Users", connection))
        using (var reader = command.ExecuteReader())
        {
            while (reader.Read())
            {
                ProcessRow(reader);
            }
        }
    }
    // connection, command, reader が全て自動Dispose
}

// await using（非同期Dispose、C# 8.0+）
public async Task ProcessStreamAsync()
{
    await using var stream = new AsyncStream();
    await stream.WriteAsync(data);
}
```

### 2.6 Go の defer

```go
// Go: defer によるクリーンアップ
func readConfig(path string) (*Config, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, fmt.Errorf("failed to open config: %w", err)
    }
    defer file.Close() // 関数終了時に必ず実行

    decoder := json.NewDecoder(file)
    var config Config
    if err := decoder.Decode(&config); err != nil {
        return nil, fmt.Errorf("failed to decode config: %w", err)
    }
    return &config, nil
}

// defer は LIFO（後入れ先出し）順で実行
func multipleDefers() {
    fmt.Println("start")
    defer fmt.Println("first defer")  // 3番目に実行
    defer fmt.Println("second defer") // 2番目に実行
    defer fmt.Println("third defer")  // 1番目に実行
    fmt.Println("end")
}
// 出力: start, end, third defer, second defer, first defer

// defer + recover でパニックからの回復
func safeOperation() (result string, err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic recovered: %v", r)
        }
    }()

    // パニックが発生しても recover で捕捉
    riskyOperation()
    return "success", nil
}
```

### 2.7 Rust の Drop トレイトと RAII

```rust
// Rust: RAII（Resource Acquisition Is Initialization）
// Drop トレイトで自動クリーンアップ
struct DatabaseConnection {
    connection_string: String,
    is_open: bool,
}

impl DatabaseConnection {
    fn new(conn_str: &str) -> Result<Self, DbError> {
        // 接続の確立
        Ok(DatabaseConnection {
            connection_string: conn_str.to_string(),
            is_open: true,
        })
    }

    fn query(&self, sql: &str) -> Result<Vec<Row>, DbError> {
        if !self.is_open {
            return Err(DbError::ConnectionClosed);
        }
        // クエリ実行
        Ok(vec![])
    }
}

impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        if self.is_open {
            // 接続のクローズ（自動的に呼ばれる）
            println!("Connection closed: {}", self.connection_string);
            self.is_open = false;
        }
    }
}

fn process_data() -> Result<(), DbError> {
    let conn = DatabaseConnection::new("postgres://localhost/mydb")?;
    let rows = conn.query("SELECT * FROM users")?;
    // conn はスコープ終了時に自動的に drop される
    // → Drop::drop() が呼ばれて接続がクローズされる
    Ok(())
}
```

---

## 3. コールスタックの巻き戻し

### 3.1 スタックアンワインドの仕組み

```
例外発生時のコールスタック:

  main()
    └── processOrder()
        └── validatePayment()
            └── chargeCard()
                └── apiCall()  ← 例外発生！

  巻き戻し（スタックアンワインド）:
  apiCall()    → catch なし → 伝播
  chargeCard() → catch なし → 伝播
  validatePayment() → catch あり！ → ここで処理
                     → または再スロー

  原則:
  → 例外は「処理できる場所」でキャッチする
  → 処理できないなら捕まえない（上位に伝播させる）
  → 握りつぶし（catch して無視）は厳禁
```

### 3.2 スタックトレースの読み方

```typescript
// スタックトレースの例
// Error: User not found: user-123
//     at UserService.getUser (/app/services/user.ts:45:11)
//     at OrderService.createOrder (/app/services/order.ts:23:28)
//     at OrderController.create (/app/controllers/order.ts:15:30)
//     at Layer.handle [as handle_request] (/app/node_modules/express/lib/router/layer.js:95:5)
//     at next (/app/node_modules/express/lib/router/route.js:144:13)

// スタックトレースの読み方:
// 1. 最上行がエラーメッセージ
// 2. 2行目が例外の発生元（最も重要）
// 3. 下に行くほど呼び出し元（ルートに近い）
// 4. node_modules 内のフレームは通常無視

// カスタムスタックトレース
class AppError extends Error {
    constructor(message: string) {
        super(message);
        this.name = this.constructor.name;
        // コンストラクタ自身をスタックから除外
        Error.captureStackTrace(this, this.constructor);
    }
}

// エラーが発生した箇所ではなく、
// AppError のコンストラクタの呼び出し元がスタックの先頭になる
```

### 3.3 非同期コードでのスタックトレース

```typescript
// 非同期コードでスタックトレースが途切れる問題
async function fetchUser(id: string): Promise<User> {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) throw new Error("Fetch failed");
    return response.json();
}

// Node.js の --async-stack-traces フラグで非同期スタックトレースを有効化
// または Error.cause で原因チェーンを構築

async function processUser(id: string): Promise<void> {
    try {
        const user = await fetchUser(id);
        await updateUser(user);
    } catch (error) {
        // ES2022: Error.cause で原因チェーンを構築
        throw new ProcessError("User processing failed", {
            cause: error,  // 元のエラーを原因として保持
        });
    }
}

// 原因チェーンの走査
function getRootCause(error: Error): Error {
    let current = error;
    while (current.cause instanceof Error) {
        current = current.cause;
    }
    return current;
}

// Python の例外チェーン
// try:
//     result = parse_data(raw)
// except ParseError as e:
//     raise ProcessingError("Data processing failed") from e
//     # __cause__ に元の例外が保存される
```

### 3.4 例外伝播のコスト

```
例外のパフォーマンスコスト:

  try ブロックに入る:
  → ほぼゼロコスト（多くのランタイムで最適化済み）
  → 「ゼロコスト例外」モデル（C++, Rust のパニック）

  例外をスロー:
  → 非常にコストが高い
  → スタックトレースの構築: O(n)（nはスタック深度）
  → スタックアンワインド: 各フレームのデストラクタ呼び出し
  → 一般的に通常の関数リターンの100-1000倍遅い

  ベンチマーク例（概算）:
  ┌──────────────────┬────────────────┐
  │ 操作             │ 相対コスト     │
  ├──────────────────┼────────────────┤
  │ 関数リターン     │ 1x             │
  │ try ブロック進入  │ ~1x            │
  │ 例外スロー       │ 100-1000x      │
  │ スタックトレース  │ 200-2000x      │
  └──────────────────┴────────────────┘

  結論:
  → try/catch を書くこと自体はコスト問題にならない
  → 例外は「例外的な」状況にのみ使う
  → ループ内で例外を使った制御フローは絶対にNG
```

---

## 4. 例外の使い方: 適切な場面と濫用

### 4.1 例外を使うべき場面

```
例外を使うべき場面:
  ✓ ファイルが見つからない
  ✓ ネットワーク接続エラー
  ✓ データベース接続失敗
  ✓ 不正なデータ（バリデーションエラー）
  ✓ リソース不足（メモリ、ディスク）
  ✓ 設定ファイルの不整合
  ✓ 外部サービスの障害
  ✓ 認証・認可の失敗
  ✓ データの整合性違反
  ✓ タイムアウト

例外を使うべきでない場面:
  ✗ 通常の制御フロー（if/else で判断できる）
  ✗ 予期される状況（ユーザーの入力ミス）
  ✗ パフォーマンスクリティカルなコード
  ✗ コレクションが空であること
  ✗ 検索結果が0件であること
  ✗ ファイル末尾への到達
```

### 4.2 アンチパターン: 例外で制御フロー

```typescript
// ❌ 例外で制御フロー（アンチパターン）
function findUserByEmail(email: string): User | null {
    try {
        const user = db.query("SELECT * FROM users WHERE email = ?", [email]);
        if (!user) throw new Error("Not found");
        return user;
    } catch (e) {
        return null;  // 「見つからない」は例外ではない
    }
}

// ✅ 戻り値で制御フロー
function findUserByEmail(email: string): User | null {
    const user = db.query("SELECT * FROM users WHERE email = ?", [email]);
    return user ?? null;  // null を返す（正常な制御フロー）
}

// ❌ ループ内で例外を使った制御（最悪のパターン）
function parseNumbers(inputs: string[]): number[] {
    const results: number[] = [];
    for (const input of inputs) {
        try {
            results.push(parseInt(input));
        } catch (e) {
            // パースできない = 通常のこと
            continue;
        }
    }
    return results;
}

// ✅ 事前チェック
function parseNumbers(inputs: string[]): number[] {
    return inputs
        .filter(input => /^\d+$/.test(input))
        .map(input => parseInt(input, 10));
}
```

### 4.3 アンチパターン: Pokemon Exception Handling

```typescript
// ❌ 全てをキャッチ（Pokemon: "Gotta Catch 'Em All"）
async function processOrder(orderId: string): Promise<void> {
    try {
        const order = await getOrder(orderId);
        await validateOrder(order);
        await chargePayment(order);
        await sendConfirmation(order);
    } catch (error) {
        // 全ての例外を握りつぶし！
        console.log("Something went wrong");
    }
}

// ✅ 適切な例外処理
async function processOrder(orderId: string): Promise<void> {
    try {
        const order = await getOrder(orderId);
        await validateOrder(order);
        await chargePayment(order);
        await sendConfirmation(order);
    } catch (error) {
        if (error instanceof ValidationError) {
            // バリデーションエラーは具体的に処理
            await notifyUser(orderId, error.message);
            return;
        }
        if (error instanceof PaymentError) {
            // 支払いエラーはリトライ可能かもしれない
            await queueForRetry(orderId);
            return;
        }
        // 予期しないエラーは上位に伝播
        throw error;
    }
}
```

### 4.4 アンチパターン: 例外の握りつぶし

```python
# ❌ 握りつぶし（Swallowing Exception）
def save_user(user):
    try:
        db.save(user)
    except Exception:
        pass  # 何もしない！データが保存されていないのに成功したように見える

# ❌ ログだけ出して握りつぶし
def save_user(user):
    try:
        db.save(user)
    except Exception as e:
        logger.error(f"Failed to save user: {e}")
        # 呼び出し元は成功したと思っている

# ✅ 適切な処理
def save_user(user) -> bool:
    try:
        db.save(user)
        return True
    except IntegrityError as e:
        logger.warning(f"Duplicate user: {e}")
        raise DuplicateUserError(user.email) from e
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise ServiceUnavailableError("Database unavailable") from e
```

### 4.5 アンチパターン: 例外の再スローミス

```typescript
// ❌ 情報を失う再スロー
try {
    await processPayment(order);
} catch (error) {
    throw new Error("Payment failed");  // 元のエラー情報が全て消える
}

// ❌ スタックトレースを壊す再スロー
try {
    await processPayment(order);
} catch (error) {
    throw error;  // OK だが、コンテキスト情報を追加する機会を逃している
}

// ✅ 原因チェーンを保持した再スロー
try {
    await processPayment(order);
} catch (error) {
    throw new PaymentError("Payment processing failed", {
        cause: error,
        orderId: order.id,
        amount: order.total,
    });
}

// ✅ Python の例外チェーン
# try:
#     process_payment(order)
# except StripeError as e:
#     raise PaymentError(f"Payment failed for order {order.id}") from e
```

---

## 5. Java: checked vs unchecked

### 5.1 Java の例外階層

```java
// Java の例外階層
// Throwable
// ├── Error（回復不能: OutOfMemoryError, StackOverflowError）
// └── Exception
//     ├── IOException（checked: コンパイラが強制）
//     ├── SQLException（checked）
//     └── RuntimeException（unchecked: コンパイラ強制なし）
//         ├── NullPointerException
//         ├── IllegalArgumentException
//         └── IndexOutOfBoundsException

// checked exception: catchまたはthrows宣言が必須
public String readFile(String path) throws IOException {
    return Files.readString(Path.of(path));
    // IOException を catch しないなら throws で宣言必須
}

// unchecked exception: 宣言不要
public void validateAge(int age) {
    if (age < 0) throw new IllegalArgumentException("Age must be >= 0");
    // throws 宣言不要
}
```

### 5.2 checked exception の議論

```
checked exception の議論:

  賛成派:
  → エラー処理を忘れない
  → APIの契約が明確
  → コンパイル時にエラー処理の漏れを検出

  反対派（多数派）:
  → ボイラープレートが多い
  → 握りつぶしの原因（catch して無視）
  → Kotlin, C#, Python, TS は全て unchecked のみ
  → 例外仕様の変更が連鎖的に波及（throws の伝播）
  → ラムダ式と相性が悪い

  現代の主流:
  → unchecked exception + 型でエラーを表現（Result型）
```

### 5.3 checked exception の問題点（具体例）

```java
// checked exception の問題1: ボイラープレート
// ラムダ式で checked exception を使えない
public List<String> readAllFiles(List<String> paths) throws IOException {
    // ❌ コンパイルエラー: ラムダ内の checked exception
    // return paths.stream()
    //     .map(path -> Files.readString(Path.of(path))) // IOException!
    //     .collect(Collectors.toList());

    // ✅ 回避策1: try/catch をラムダ内に書く（冗長）
    return paths.stream()
        .map(path -> {
            try {
                return Files.readString(Path.of(path));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        })
        .collect(Collectors.toList());
}

// checked exception の問題2: throws の連鎖
// 低レベルの実装詳細が上位のインターフェースに漏洩する
public interface UserRepository {
    // ❌ 実装詳細（SQL）がインターフェースに漏洩
    User findById(String id) throws SQLException;

    // ✅ 抽象化されたエラー
    User findById(String id) throws RepositoryException;
}

// checked exception の問題3: 握りつぶしの誘発
public void processData(String data) {
    try {
        riskyOperation(data);
    } catch (CheckedException e) {
        // 面倒なので握りつぶし（最悪のパターン）
        // コンパイラを黙らせるためだけのcatch
    }
}
```

### 5.4 Kotlin のアプローチ

```kotlin
// Kotlin: checked exception を廃止
// Java の checked exception を呼び出しても catch 不要
fun readFile(path: String): String {
    return File(path).readText()  // IOException は unchecked として扱われる
}

// try は式（expression）として使える
val result: Int = try {
    input.toInt()
} catch (e: NumberFormatException) {
    0  // デフォルト値
}

// runCatching（Result型ライクな使い方）
val result: Result<Int> = runCatching {
    input.toInt()
}

result
    .onSuccess { value -> println("Parsed: $value") }
    .onFailure { error -> println("Failed: ${error.message}") }

val value: Int = result.getOrDefault(0)
val valueOrNull: Int? = result.getOrNull()
```

---

## 6. 各言語の例外モデル比較

### 6.1 TypeScript/JavaScript

```typescript
// TypeScript: 全て unchecked
// 型システムでは例外の型を表現できない（throw する型の注釈なし）

// any 型で catch される問題（TypeScript 4.0 以前）
try {
    throw new Error("test");
} catch (error) {
    // error は unknown 型（TypeScript 4.4+ の useUnknownInCatchVariables）
    // 型を絞り込む必要がある
    if (error instanceof Error) {
        console.log(error.message);
    }
}

// Error の種類
// Error: 汎用エラー
// TypeError: 型エラー
// ReferenceError: 未定義変数の参照
// RangeError: 範囲外の値
// SyntaxError: 構文エラー
// URIError: URI のエンコード/デコードエラー

// 非 Error オブジェクトの throw（非推奨）
throw "エラーです";        // ❌ string
throw 42;                  // ❌ number
throw { code: "ERR" };     // ❌ object
throw new Error("エラー");  // ✅ Error オブジェクト
// 非 Error オブジェクトはスタックトレースが取得できない
```

### 6.2 Python

```python
# Python: 全て unchecked + 豊富な組み込み例外

# 例外階層
# BaseException
# ├── KeyboardInterrupt
# ├── SystemExit
# ├── GeneratorExit
# └── Exception
#     ├── StopIteration
#     ├── ArithmeticError
#     │   ├── ZeroDivisionError
#     │   └── OverflowError
#     ├── LookupError
#     │   ├── KeyError
#     │   └── IndexError
#     ├── OSError
#     │   ├── FileNotFoundError
#     │   └── PermissionError
#     ├── ValueError
#     ├── TypeError
#     └── RuntimeError

# 複数例外の同時キャッチ
try:
    result = process(data)
except (ValueError, TypeError) as e:
    logger.error(f"Data error: {e}")
except OSError as e:
    logger.error(f"System error: {e}")
except Exception as e:
    logger.error(f"Unexpected: {e}")
    raise  # 再送出

# コンテキストマネージャ（with文）
class DatabaseConnection:
    def __enter__(self):
        self.conn = create_connection()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.conn.rollback()
            logger.error(f"Transaction rolled back: {exc_val}")
        else:
            self.conn.commit()
        self.conn.close()
        return False  # True にすると例外を握りつぶす

# 使い方
with DatabaseConnection() as conn:
    conn.execute("INSERT INTO users ...")
    conn.execute("UPDATE accounts ...")
    # 例外が発生 → rollback + close
    # 正常完了 → commit + close

# Python 3.11+: ExceptionGroup（複数例外の同時発生）
async def fetch_all(urls: list[str]) -> list[str]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch(url)) for url in urls]
    # 複数のタスクが同時に失敗した場合
    # ExceptionGroup が発生する

try:
    await fetch_all(urls)
except* ValueError as eg:
    # ExceptionGroup 内の ValueError のみをハンドル
    for exc in eg.exceptions:
        logger.error(f"Value error: {exc}")
except* OSError as eg:
    # ExceptionGroup 内の OSError のみをハンドル
    for exc in eg.exceptions:
        logger.error(f"OS error: {exc}")
```

### 6.3 C++ の例外

```cpp
// C++: 例外は使えるが、パフォーマンスコストが議論される

#include <stdexcept>
#include <string>

// 標準例外階層
// std::exception
// ├── std::logic_error
// │   ├── std::invalid_argument
// │   ├── std::out_of_range
// │   └── std::domain_error
// └── std::runtime_error
//     ├── std::overflow_error
//     ├── std::underflow_error
//     └── std::range_error

// 基本的な例外処理
void processFile(const std::string& path) {
    try {
        auto data = readFile(path);
        auto result = parseData(data);
        saveResult(result);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
    } catch (...) {
        // 全てのキャッチ（C++ 固有）
        std::cerr << "Unknown error" << std::endl;
        throw;  // 再スロー
    }
}

// noexcept 指定（C++11）
// この関数は例外を投げないことを宣言
void swap(int& a, int& b) noexcept {
    int temp = a;
    a = b;
    b = temp;
}
// noexcept 関数で例外が投げられると std::terminate() が呼ばれる

// RAII パターン（Resource Acquisition Is Initialization）
class FileHandle {
    FILE* file_;
public:
    explicit FileHandle(const char* path) : file_(fopen(path, "r")) {
        if (!file_) throw std::runtime_error("Failed to open file");
    }
    ~FileHandle() {
        if (file_) fclose(file_);  // デストラクタで自動クリーンアップ
    }
    // コピー禁止
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
};
```

### 6.4 Go のエラー処理（例外なし）

```go
// Go: 例外の代わりに多値返却
// 例外機構がない（panic/recover はあるが、通常使わない）

func readConfig(path string) (*Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("failed to read %s: %w", path, err)
    }

    var config Config
    if err := json.Unmarshal(data, &config); err != nil {
        return nil, fmt.Errorf("failed to parse %s: %w", path, err)
    }

    return &config, nil
}

// エラーのラッピングとアンラッピング
func processOrder(orderID string) error {
    order, err := getOrder(orderID)
    if err != nil {
        // %w でラッピング（Go 1.13+）
        return fmt.Errorf("processOrder: get order: %w", err)
    }

    if err := validateOrder(order); err != nil {
        return fmt.Errorf("processOrder: validate: %w", err)
    }

    return nil
}

// errors.Is: エラーの同一性チェック
if errors.Is(err, os.ErrNotExist) {
    fmt.Println("File does not exist")
}

// errors.As: エラーの型チェック
var pathErr *os.PathError
if errors.As(err, &pathErr) {
    fmt.Printf("Path error: %s, Op: %s\n", pathErr.Path, pathErr.Op)
}

// Go のエラー処理の議論:
// 賛成: シンプル、明示的、見落としにくい
// 反対: if err != nil の繰り返し、ボイラープレート
```

### 6.5 Swift のエラー処理

```swift
// Swift: do/try/catch + 型安全なエラー

// Error プロトコルに準拠した列挙型
enum NetworkError: Error {
    case invalidURL(String)
    case timeout(seconds: Int)
    case serverError(statusCode: Int, message: String)
    case noConnection
}

// エラーを投げる関数（throws キーワード）
func fetchData(from urlString: String) throws -> Data {
    guard let url = URL(string: urlString) else {
        throw NetworkError.invalidURL(urlString)
    }

    let (data, response) = try await URLSession.shared.data(from: url)

    if let httpResponse = response as? HTTPURLResponse,
       httpResponse.statusCode >= 400 {
        throw NetworkError.serverError(
            statusCode: httpResponse.statusCode,
            message: "Server error"
        )
    }

    return data
}

// 呼び出し方法（3パターン）
// 1. do/try/catch
do {
    let data = try fetchData(from: "https://api.example.com")
    process(data)
} catch NetworkError.invalidURL(let url) {
    print("Invalid URL: \(url)")
} catch NetworkError.timeout(let seconds) {
    print("Timeout after \(seconds)s")
} catch {
    print("Unknown error: \(error)")
}

// 2. try?（失敗時は nil）
let data = try? fetchData(from: "https://api.example.com")

// 3. try!（失敗時はクラッシュ）
let data = try! fetchData(from: "https://api.example.com")  // 危険
```

---

## 7. エラー階層の設計

### 7.1 TypeScript でのエラー階層

```typescript
// カスタムエラー階層
class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number = 500,
    public readonly isOperational: boolean = true,
  ) {
    super(message);
    this.name = this.constructor.name;
  }
}

// 認証エラー
class AuthenticationError extends AppError {
  constructor(message: string = "認証が必要です") {
    super(message, "AUTH_REQUIRED", 401);
  }
}

// 認可エラー
class AuthorizationError extends AppError {
  constructor(message: string = "権限がありません") {
    super(message, "FORBIDDEN", 403);
  }
}

// リソース未発見
class NotFoundError extends AppError {
  constructor(resource: string, id: string) {
    super(`${resource} not found: ${id}`, "NOT_FOUND", 404);
  }
}

// バリデーションエラー
class ValidationError extends AppError {
  constructor(
    message: string,
    public readonly fields: Record<string, string[]>,
  ) {
    super(message, "VALIDATION_ERROR", 400);
  }
}

// 使い分け
function getUser(id: string): User {
  const user = db.findById(id);
  if (!user) throw new NotFoundError("User", id);
  return user;
}
```

### 7.2 エラー階層設計のベストプラクティス

```typescript
// 実務的なエラー階層の全体像
abstract class AppError extends Error {
    abstract readonly code: string;
    abstract readonly httpStatus: number;
    readonly timestamp: string;
    readonly requestId?: string;

    constructor(
        message: string,
        public readonly isOperational: boolean = true,
        options?: { cause?: Error; requestId?: string }
    ) {
        super(message, { cause: options?.cause });
        this.name = this.constructor.name;
        this.timestamp = new Date().toISOString();
        this.requestId = options?.requestId;
        Error.captureStackTrace(this, this.constructor);
    }

    toJSON() {
        return {
            error: {
                code: this.code,
                message: this.message,
                timestamp: this.timestamp,
                ...(this.requestId && { requestId: this.requestId }),
            }
        };
    }
}

// ======== 認証・認可 ========
class AuthenticationError extends AppError {
    readonly code = "AUTHENTICATION_REQUIRED";
    readonly httpStatus = 401;
    constructor(message = "認証が必要です", options?: { cause?: Error }) {
        super(message, true, options);
    }
}

class TokenExpiredError extends AuthenticationError {
    readonly code = "TOKEN_EXPIRED";
    constructor() {
        super("トークンが期限切れです");
    }
}

class InvalidCredentialsError extends AuthenticationError {
    readonly code = "INVALID_CREDENTIALS";
    constructor() {
        super("認証情報が無効です");
    }
}

class AuthorizationError extends AppError {
    readonly code = "FORBIDDEN";
    readonly httpStatus = 403;
    constructor(
        public readonly requiredRole: string,
        public readonly actualRole: string,
    ) {
        super(`権限が不足しています: ${requiredRole} が必要です（現在: ${actualRole}）`);
    }
}

// ======== リソース ========
class NotFoundError extends AppError {
    readonly code = "NOT_FOUND";
    readonly httpStatus = 404;
    constructor(
        public readonly resourceType: string,
        public readonly resourceId: string,
    ) {
        super(`${resourceType} が見つかりません: ${resourceId}`);
    }
}

class ConflictError extends AppError {
    readonly code = "CONFLICT";
    readonly httpStatus = 409;
    constructor(message: string) {
        super(message);
    }
}

// ======== バリデーション ========
interface FieldError {
    field: string;
    message: string;
    value?: unknown;
}

class ValidationError extends AppError {
    readonly code = "VALIDATION_ERROR";
    readonly httpStatus = 400;
    constructor(public readonly fieldErrors: FieldError[]) {
        super(`入力値が不正です: ${fieldErrors.map(e => e.field).join(", ")}`);
    }

    toJSON() {
        return {
            error: {
                code: this.code,
                message: this.message,
                details: this.fieldErrors,
                timestamp: this.timestamp,
            }
        };
    }
}

// ======== 外部サービス ========
class ExternalServiceError extends AppError {
    readonly code = "EXTERNAL_SERVICE_ERROR";
    readonly httpStatus = 502;
    constructor(
        public readonly serviceName: string,
        message: string,
        options?: { cause?: Error }
    ) {
        super(`${serviceName}: ${message}`, true, options);
    }
}

class RateLimitError extends AppError {
    readonly code = "RATE_LIMIT_EXCEEDED";
    readonly httpStatus = 429;
    constructor(
        public readonly retryAfterMs: number,
    ) {
        super(`レート制限に達しました。${retryAfterMs}ms 後に再試行してください`);
    }
}

// ======== 内部エラー ========
class InternalError extends AppError {
    readonly code = "INTERNAL_ERROR";
    readonly httpStatus = 500;
    constructor(message: string, options?: { cause?: Error }) {
        super(message, false, options);  // isOperational = false
    }
}
```

### 7.3 エラー階層を使ったミドルウェア

```typescript
// Express ミドルウェアでのエラーハンドリング
function errorHandler(
    error: Error,
    req: Request,
    res: Response,
    next: NextFunction
): void {
    // リクエストIDの付与
    const requestId = req.headers['x-request-id'] as string || generateId();

    if (error instanceof AppError) {
        // 操作エラー: 適切なレスポンスを返す
        logger.warn({
            code: error.code,
            message: error.message,
            requestId,
            path: req.path,
            method: req.method,
        });

        res.status(error.httpStatus).json({
            ...error.toJSON(),
            error: {
                ...error.toJSON().error,
                requestId,
            }
        });
    } else {
        // プログラマエラー: 内部情報を隠す
        logger.error({
            message: error.message,
            stack: error.stack,
            requestId,
            path: req.path,
            method: req.method,
        });

        // Sentry等に送信
        Sentry.captureException(error, {
            tags: { requestId },
            extra: { path: req.path },
        });

        res.status(500).json({
            error: {
                code: "INTERNAL_ERROR",
                message: "サーバーエラーが発生しました",
                requestId,
            }
        });
    }
}
```

---

## 8. 例外安全性（Exception Safety）

### 8.1 例外安全性のレベル

```
例外安全性の4レベル（C++ 由来だが概念は言語共通）:

  Level 0: 例外安全でない（No guarantee）
  → 例外が発生するとリソースリーク、データ破損の可能性
  → 避けるべき

  Level 1: 基本保証（Basic guarantee）
  → 例外が発生してもリソースリークしない
  → オブジェクトは有効だが、内容は不定
  → 最低限これを目指す

  Level 2: 強い保証（Strong guarantee）
  → 例外が発生すると操作前の状態に完全に戻る
  → トランザクションのロールバックに相当
  → コストが高いが、安全

  Level 3: 例外なし保証（No-throw guarantee）
  → 例外が絶対に発生しない
  → デストラクタ、スワップ操作に求められる
```

### 8.2 例外安全なコードの実例

```typescript
// ❌ 例外安全でないコード（Level 0）
class UserService {
    async transferBalance(fromId: string, toId: string, amount: number): Promise<void> {
        const from = await this.getUser(fromId);
        from.balance -= amount;
        await this.saveUser(from);  // ← ここで例外が起きたら?

        const to = await this.getUser(toId);
        to.balance += amount;
        await this.saveUser(to);  // ← from は減額済み、to は未加算
        // データ不整合が発生!
    }
}

// ✅ 強い保証のコード（Level 2）: トランザクション使用
class UserService {
    async transferBalance(fromId: string, toId: string, amount: number): Promise<void> {
        await this.db.transaction(async (tx) => {
            const from = await tx.getUser(fromId);
            const to = await tx.getUser(toId);

            if (from.balance < amount) {
                throw new InsufficientBalanceError(amount, from.balance);
            }

            from.balance -= amount;
            to.balance += amount;

            await tx.saveUser(from);
            await tx.saveUser(to);
            // tx.commit() はトランザクション終了時に自動実行
            // 例外発生時は tx.rollback() が自動実行
        });
    }
}

// ✅ 強い保証のコード（Level 2）: Copy-and-Swap イディオム
class Configuration {
    private data: Map<string, string>;

    updateMultiple(updates: Record<string, string>): void {
        // 1. コピーを作成
        const newData = new Map(this.data);

        // 2. コピーに対して変更（ここで例外が起きても元データは無傷）
        for (const [key, value] of Object.entries(updates)) {
            if (!this.isValidKey(key)) {
                throw new ValidationError(`Invalid key: ${key}`);
            }
            newData.set(key, value);
        }

        // 3. アトミックにスワップ（例外なし保証の操作）
        this.data = newData;
    }
}
```

### 8.3 finally の正しい使い方

```typescript
// finally のベストプラクティス

// ✅ リソース解放
async function processWithLock(key: string): Promise<void> {
    const lock = await acquireLock(key);
    try {
        await doWork();
    } finally {
        await lock.release();  // 成功でも失敗でも必ずロック解放
    }
}

// ✅ 一時ファイルの削除
async function processWithTempFile(): Promise<void> {
    const tempPath = await createTempFile();
    try {
        await writeToFile(tempPath, data);
        await processFile(tempPath);
    } finally {
        await fs.unlink(tempPath).catch(() => {});  // ベストエフォートで削除
    }
}

// ❌ finally 内で return してはいけない
function badFinally(): number {
    try {
        throw new Error("error");
    } finally {
        return 42;  // ❌ 例外が握りつぶされて 42 が返る！
    }
}

// ❌ finally 内で throw してはいけない（元の例外を上書き）
async function badFinallyThrow(): Promise<void> {
    try {
        throw new Error("original error");
    } finally {
        throw new Error("cleanup error");  // ❌ original error が消える
    }
}

// ✅ finally 内のエラーは安全に処理
async function safeFinally(): Promise<void> {
    let resource: Resource | null = null;
    try {
        resource = await acquireResource();
        await resource.process();
    } finally {
        if (resource) {
            try {
                await resource.release();
            } catch (cleanupError) {
                logger.warn("Cleanup failed:", cleanupError);
                // 元の例外を保持するため、ここでは throw しない
            }
        }
    }
}
```

---

## 9. 非同期例外処理

### 9.1 Promise と例外

```typescript
// Promise チェーンでの例外処理
fetchUser(userId)
    .then(user => fetchOrders(user.id))
    .then(orders => calculateTotal(orders))
    .then(total => updateDashboard(total))
    .catch(error => {
        // チェーン内のどこかで発生した例外をキャッチ
        if (error instanceof UserNotFoundError) {
            showEmptyState();
        } else if (error instanceof NetworkError) {
            showRetryButton();
        } else {
            showGenericError();
        }
    });

// async/await での例外処理（推奨）
async function loadDashboard(userId: string): Promise<void> {
    try {
        const user = await fetchUser(userId);
        const orders = await fetchOrders(user.id);
        const total = calculateTotal(orders);
        updateDashboard(total);
    } catch (error) {
        handleDashboardError(error);
    }
}

// Promise.all での例外
async function fetchMultiple(ids: string[]): Promise<User[]> {
    try {
        // 1つでも失敗すると全体が失敗
        return await Promise.all(ids.map(id => fetchUser(id)));
    } catch (error) {
        // 最初に失敗した Promise の例外
        throw new BatchFetchError("Some users could not be fetched", {
            cause: error,
        });
    }
}

// Promise.allSettled で個別のエラーを処理
async function fetchMultipleSafe(ids: string[]): Promise<{
    users: User[];
    errors: Array<{ id: string; error: Error }>;
}> {
    const results = await Promise.allSettled(
        ids.map(id => fetchUser(id).then(user => ({ id, user })))
    );

    const users: User[] = [];
    const errors: Array<{ id: string; error: Error }> = [];

    for (const result of results) {
        if (result.status === "fulfilled") {
            users.push(result.value.user);
        } else {
            errors.push({
                id: "unknown",  // allSettled では元のidが分からない場合
                error: result.reason,
            });
        }
    }

    return { users, errors };
}
```

### 9.2 未処理の Promise rejection

```typescript
// ❌ 未処理の rejection（UnhandledPromiseRejection）
async function dangerousCode(): Promise<void> {
    fetchUser("123");  // await を忘れている！
    // fetchUser が reject しても誰もキャッチしない
}

// ❌ catch を忘れた Promise チェーン
someAsyncFunction().then(data => {
    process(data);
    // .catch() がない → rejection が未処理になる
});

// ✅ 必ず await または .catch() を付ける
await someAsyncFunction().catch(error => {
    logger.error("Failed:", error);
});

// グローバルハンドラ（最後の砦）
// Node.js
process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection:', reason);
    Sentry.captureException(reason);
    // Node.js 15+ ではプロセスが終了する
});

// ブラウザ
window.addEventListener('unhandledrejection', (event) => {
    logger.error('Unhandled Rejection:', event.reason);
    event.preventDefault();  // ブラウザのデフォルトエラー表示を抑制
});
```

### 9.3 並行処理での例外処理パターン

```typescript
// パターン1: fail-fast（1つ失敗したら全て中止）
async function failFast(tasks: Array<() => Promise<void>>): Promise<void> {
    const controller = new AbortController();

    try {
        await Promise.all(
            tasks.map(async (task) => {
                if (controller.signal.aborted) return;
                try {
                    await task();
                } catch (error) {
                    controller.abort();  // 他のタスクにキャンセルを通知
                    throw error;
                }
            })
        );
    } catch (error) {
        throw new BatchError("One or more tasks failed", { cause: error });
    }
}

// パターン2: best-effort（できるだけ多くを完了させる）
async function bestEffort<T>(
    tasks: Array<() => Promise<T>>
): Promise<{ results: T[]; errors: Error[] }> {
    const settled = await Promise.allSettled(tasks.map(t => t()));

    const results: T[] = [];
    const errors: Error[] = [];

    for (const result of settled) {
        if (result.status === "fulfilled") {
            results.push(result.value);
        } else {
            errors.push(result.reason instanceof Error
                ? result.reason
                : new Error(String(result.reason)));
        }
    }

    return { results, errors };
}

// パターン3: retry-on-failure（失敗時にリトライ）
async function withRetry<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    delayMs: number = 1000,
): Promise<T> {
    let lastError: Error;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error));

            if (attempt < maxRetries) {
                const delay = delayMs * Math.pow(2, attempt - 1);  // 指数バックオフ
                logger.warn(`Attempt ${attempt} failed, retrying in ${delay}ms...`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    throw new RetryExhaustedError(
        `All ${maxRetries} attempts failed`,
        { cause: lastError! }
    );
}
```

---

## 10. テストにおける例外処理

### 10.1 例外のテスト

```typescript
// Jest での例外テスト
describe("UserService", () => {
    describe("getUser", () => {
        it("存在しないユーザーで NotFoundError をスローする", async () => {
            await expect(
                userService.getUser("nonexistent-id")
            ).rejects.toThrow(NotFoundError);
        });

        it("NotFoundError に正しいプロパティが設定される", async () => {
            try {
                await userService.getUser("user-999");
                fail("例外がスローされるべき");
            } catch (error) {
                expect(error).toBeInstanceOf(NotFoundError);
                expect((error as NotFoundError).code).toBe("NOT_FOUND");
                expect((error as NotFoundError).httpStatus).toBe(404);
                expect(error.message).toContain("user-999");
            }
        });

        it("データベースエラーは InternalError にラップされる", async () => {
            // DB障害をモック
            jest.spyOn(db, "findById").mockRejectedValue(
                new Error("Connection refused")
            );

            await expect(
                userService.getUser("user-1")
            ).rejects.toThrow(InternalError);
        });
    });
});

// Python: pytest での例外テスト
# def test_get_user_not_found():
#     with pytest.raises(NotFoundError) as exc_info:
#         user_service.get_user("nonexistent")
#
#     assert exc_info.value.code == "NOT_FOUND"
#     assert "nonexistent" in str(exc_info.value)

# def test_get_user_not_found_match():
#     with pytest.raises(NotFoundError, match="nonexistent"):
#         user_service.get_user("nonexistent")
```

### 10.2 例外パスのテスト戦略

```typescript
// テストピラミッドにおける例外テスト

// 1. ユニットテスト: 個別のエラーケースを網羅
describe("validateEmail", () => {
    it.each([
        ["", "メールアドレスは必須です"],
        ["invalid", "メールアドレスの形式が不正です"],
        ["a@b", "メールアドレスの形式が不正です"],
        ["@example.com", "メールアドレスの形式が不正です"],
    ])("'%s' で ValidationError をスロー: %s", (email, expectedMessage) => {
        expect(() => validateEmail(email)).toThrow(ValidationError);
        try {
            validateEmail(email);
        } catch (e) {
            expect((e as ValidationError).message).toBe(expectedMessage);
        }
    });
});

// 2. 統合テスト: エラーの伝播を確認
describe("POST /api/users", () => {
    it("重複メールで 409 Conflict を返す", async () => {
        // 既存ユーザーを作成
        await createUser({ email: "test@example.com" });

        const response = await request(app)
            .post("/api/users")
            .send({ email: "test@example.com", name: "Test" });

        expect(response.status).toBe(409);
        expect(response.body.error.code).toBe("EMAIL_ALREADY_EXISTS");
    });

    it("バリデーションエラーで 400 と詳細を返す", async () => {
        const response = await request(app)
            .post("/api/users")
            .send({});  // 空のボディ

        expect(response.status).toBe(400);
        expect(response.body.error.code).toBe("VALIDATION_ERROR");
        expect(response.body.error.details).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ field: "email" }),
                expect.objectContaining({ field: "name" }),
            ])
        );
    });
});
```

---

## 11. 実務でのベストプラクティス

### 11.1 ロギングとモニタリング

```typescript
// エラーログの構造化
interface ErrorLog {
    level: "warn" | "error" | "fatal";
    code: string;
    message: string;
    stack?: string;
    requestId?: string;
    userId?: string;
    path?: string;
    method?: string;
    duration?: number;
    metadata?: Record<string, unknown>;
}

function logError(error: Error, context: Partial<ErrorLog> = {}): void {
    const log: ErrorLog = {
        level: error instanceof AppError && error.isOperational ? "warn" : "error",
        code: error instanceof AppError ? error.code : "UNKNOWN_ERROR",
        message: error.message,
        stack: error.stack,
        ...context,
    };

    if (log.level === "error") {
        // 非操作エラーは即座にアラート
        logger.error(log);
        Sentry.captureException(error, { extra: context });
        metrics.increment("app.errors.unexpected");
    } else {
        logger.warn(log);
        metrics.increment(`app.errors.operational.${log.code}`);
    }
}

// エラー率のモニタリング
class ErrorRateMonitor {
    private errors: Map<string, number[]> = new Map();

    record(code: string): void {
        const now = Date.now();
        const timestamps = this.errors.get(code) ?? [];
        timestamps.push(now);
        // 5分以上前のエントリを削除
        const fiveMinAgo = now - 5 * 60 * 1000;
        this.errors.set(code, timestamps.filter(t => t > fiveMinAgo));
    }

    getRate(code: string, windowMs: number = 60_000): number {
        const now = Date.now();
        const timestamps = this.errors.get(code) ?? [];
        return timestamps.filter(t => t > now - windowMs).length;
    }

    isAlerting(code: string, threshold: number = 10): boolean {
        return this.getRate(code) > threshold;
    }
}
```

### 11.2 エラーメッセージのガイドライン

```
エラーメッセージのベストプラクティス:

  1. 何が起きたかを明確に述べる
     ❌ "Error"
     ❌ "Something went wrong"
     ✅ "ユーザー user-123 が見つかりません"
     ✅ "メールアドレスの形式が不正です: missing @"

  2. 何をすべきかを示す（ユーザー向け）
     ❌ "Internal Server Error"
     ✅ "サーバーが一時的に利用できません。しばらく待ってから再試行してください"
     ✅ "セッションが期限切れです。再度ログインしてください"

  3. コンテキスト情報を含める（開発者向け）
     ❌ "Database error"
     ✅ "Failed to insert user (email: test@example.com): unique constraint violation on 'users_email_key'"

  4. セキュリティに配慮する
     ❌ "SQL syntax error: SELECT * FROM users WHERE password = '...'"
     ❌ "Authentication failed for admin@company.com"
     ✅ "認証に失敗しました"（ユーザー向け）
     ✅ "Auth failed: invalid password for user_id=123"（ログ向け）

  5. 国際化を考慮する
     → エラーコード（機械可読）+ メッセージテンプレート
     → i18n キーとして利用可能な設計
```

### 11.3 エラー処理のチェックリスト

```
プロジェクト開始時のエラー処理チェックリスト:

  □ エラー階層の設計
    → AppError 基底クラス
    → ドメインエラーのサブクラス
    → エラーコード体系

  □ グローバルエラーハンドラ
    → Express/Fastify ミドルウェア
    → React Error Boundary
    → uncaughtException / unhandledRejection

  □ ログとモニタリング
    → 構造化ログ（JSON）
    → エラートラッキング（Sentry）
    → アラート設定（PagerDuty）

  □ APIエラーレスポンス
    → 統一フォーマット（RFC 7807 準拠推奨）
    → HTTPステータスコードの正しい使い分け
    → エラーコードの文書化

  □ テスト
    → 正常系と異常系の両方
    → エラーの伝播パスのテスト
    → エッジケースのテスト

  □ ドキュメント
    → エラーコード一覧
    → トラブルシューティングガイド
    → エラーレスポンスの例
```

---

## 12. パフォーマンスとトレードオフ

### 12.1 例外 vs Result型 のパフォーマンス

```typescript
// パフォーマンス比較: 例外 vs Result型

// ❌ 例外をパフォーマンスクリティカルなコードで使う
function parseIntWithException(s: string): number {
    const n = Number(s);
    if (isNaN(n)) throw new Error(`Invalid number: ${s}`);
    return n;
}

// 10万回呼び出し（50%失敗）: ~500ms
// → 例外スローのスタックトレース構築が重い

// ✅ Result型でパフォーマンスクリティカルなコードを処理
function parseIntWithResult(s: string): { ok: true; value: number } | { ok: false; error: string } {
    const n = Number(s);
    if (isNaN(n)) return { ok: false, error: `Invalid number: ${s}` };
    return { ok: true, value: n };
}

// 10万回呼び出し（50%失敗）: ~5ms
// → 通常の関数リターンと同じコスト

// 結論:
// → 頻繁に失敗する操作（バリデーション、パース）は Result型
// → 稀に失敗する操作（I/O、ネットワーク）は例外でOK
// → 例外のコストは「発生時」のみ。try ブロック自体は無料
```

### 12.2 スタックトレースの制御

```typescript
// スタックトレースの省略（パフォーマンス最適化）
class LightweightError extends Error {
    constructor(message: string, public readonly code: string) {
        super(message);
        this.name = this.constructor.name;
        // スタックトレースを省略（パフォーマンス向上）
        // ただし、デバッグが困難になるため慎重に判断
    }
}

// V8: Error.stackTraceLimit でスタックの深さを制限
Error.stackTraceLimit = 10;  // デフォルトは10（Node.js）

// 条件付きスタックトレース
class ConfigurableError extends Error {
    constructor(message: string, options?: { includeStack?: boolean }) {
        super(message);
        if (options?.includeStack === false) {
            this.stack = `${this.name}: ${this.message}`;
        }
    }
}
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| try/catch | 例外をキャッチして処理 |
| finally | 必ず実行されるクリーンアップ |
| 伝播 | 処理できる場所でキャッチ |
| checked vs unchecked | 現代は unchecked + Result型 |
| カスタムエラー | 階層を設計してコード化 |
| 例外安全性 | 基本保証を最低限、強い保証を推奨 |
| 非同期例外 | Promise rejection の処理を忘れない |
| パフォーマンス | try は無料、throw は高コスト |
| テスト | 正常系と異常系の両方を網羅 |
| ログ | 構造化ログ + エラートラッキング |

---

## 次に読むべきガイド
→ [[01-result-type.md]] — Result型

---

## 参考文献
1. Bloch, J. "Effective Java." Items 69-77, 2018.
2. Sutter, H. "When and How to Use Exceptions." 2004.
3. Abramov, D. "Error Handling in React 16." React Blog, 2017.
4. Goldberg, J. "Error Handling in Node.js." joyent.com, 2014.
5. The Rust Programming Language. "Error Handling."
6. Go Blog. "Error handling and Go." 2011.
7. Python Documentation. "Errors and Exceptions."
8. Stroustrup, B. "The C++ Programming Language." 4th Edition, 2013.
9. Apple Developer Documentation. "Error Handling." Swift Documentation.
10. Node.js Documentation. "Errors."
