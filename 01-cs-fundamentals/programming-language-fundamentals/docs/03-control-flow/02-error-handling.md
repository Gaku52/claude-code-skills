# エラーハンドリング

> エラーを「どう表現し、どう伝播し、どう処理するか」は言語設計の重要な決断。例外、Result型、エラーコードの3大戦略を理解する。

## この章で学ぶこと

- [ ] 例外ベース・Result型・エラーコードの違いを理解する
- [ ] 各言語のエラーハンドリング哲学を把握する
- [ ] 適切なエラー処理パターンを選択できる

---

## 1. 例外（Exceptions）

```python
# Python: try-except
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
```

```java
// Java: チェック例外 vs 非チェック例外
// チェック例外（コンパイラが処理を強制）
public String readFile(String path) throws IOException {
    return Files.readString(Path.of(path));
}

// 非チェック例外（RuntimeException、処理は任意）
public int divide(int a, int b) {
    if (b == 0) throw new IllegalArgumentException("Division by zero");
    return a / b;
}

// try-with-resources（自動クリーンアップ）
try (var reader = new BufferedReader(new FileReader("file.txt"))) {
    String line = reader.readLine();
} catch (IOException e) {
    logger.error("Read failed", e);
}
```

### 例外の問題点

```
問題1: 見えない制御フロー
  → 関数のシグネチャを見ただけでは、どの例外が飛ぶか分からない
  → Java のチェック例外は解決策だが、冗長になりがち

問題2: パフォーマンスコスト
  → スタックトレースの生成は高コスト
  → 正常系で例外を使うとパフォーマンス劣化

問題3: 例外の飲み込み
  → catch して何もしない → バグの隠蔽
  → 空の catch ブロックは最悪のアンチパターン
```

---

## 2. Result 型（値ベースのエラー処理）

```rust
// Rust: Result<T, E> — 成功か失敗かを型で表現
fn parse_number(s: &str) -> Result<i32, ParseIntError> {
    s.parse::<i32>()
}

// パターンマッチで処理
match parse_number("42") {
    Ok(n) => println!("Parsed: {}", n),
    Err(e) => println!("Error: {}", e),
}

// ? 演算子（エラー伝播の省略記法）
fn read_config() -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string("config.toml")?;  // エラーなら即return
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}

// メソッドチェーン
let result = parse_number("42")
    .map(|n| n * 2)
    .unwrap_or(0);

// 複数のResult を組み合わせ
let (a, b) = (parse_number("10")?, parse_number("20")?);
```

```typescript
// TypeScript: Result型をユニオン型で表現
type Result<T, E> =
    | { ok: true; value: T }
    | { ok: false; error: E };

function parseNumber(s: string): Result<number, string> {
    const n = Number(s);
    if (isNaN(n)) {
        return { ok: false, error: `Invalid number: ${s}` };
    }
    return { ok: true, value: n };
}

const result = parseNumber("42");
if (result.ok) {
    console.log(result.value);  // 型安全に value にアクセス
} else {
    console.error(result.error);
}

// neverthrow ライブラリ
import { ok, err, Result } from 'neverthrow';

function divide(a: number, b: number): Result<number, string> {
    if (b === 0) return err("Division by zero");
    return ok(a / b);
}
```

```go
// Go: 複数戻り値でエラーを返す
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
```

---

## 3. 各言語の比較

```
┌──────────────┬──────────────────────┬────────────────┐
│ 方式          │ 代表言語              │ 特徴           │
├──────────────┼──────────────────────┼────────────────┤
│ 例外          │ Python, Java, C#,    │ 暗黙的な伝播   │
│              │ JavaScript, Ruby      │ 見えない制御フロー│
├──────────────┼──────────────────────┼────────────────┤
│ Result型      │ Rust, Haskell,       │ 明示的な伝播   │
│              │ Elm, Kotlin(Result)   │ 型で安全に表現  │
├──────────────┼──────────────────────┼────────────────┤
│ エラーコード   │ C, Go               │ シンプルだが    │
│              │                      │ チェック忘れの危険│
├──────────────┼──────────────────────┼────────────────┤
│ ハイブリッド   │ Swift(throw+Result)  │ 場面で使い分け  │
│              │ Kotlin(throw+Result)  │                │
└──────────────┴──────────────────────┴────────────────┘
```

---

## 4. エラー処理のベストプラクティス

```
1. 回復可能 vs 回復不可能
   回復可能: ファイルが見つからない → 別のパスを試す
   回復不可能: メモリ不足 → プロセスを終了

   Rust の使い分け:
     回復可能: Result<T, E>
     回復不可能: panic!()

2. エラーの粒度
   ❌ 全て同じエラー型
   catch (Exception e)  // 全例外をキャッチ → バグを隠蔽

   ✅ エラーの種類を区別
   enum AppError {
       NotFound(String),
       Unauthorized,
       ValidationError(Vec<String>),
       InternalError(Box<dyn Error>),
   }

3. エラーメッセージ
   ❌ "Error occurred"（何が起きたか分からない）
   ✅ "Failed to read config file '/etc/app.toml': Permission denied"
      （何が / どこで / なぜ）

4. エラーの伝播
   ❌ エラーを握りつぶす
   ✅ 適切な層でキャッチし、コンテキストを追加して上位に伝播
```

---

## 5. 実践パターン

```rust
// Rust: カスタムエラー型（thiserror クレート）
use thiserror::Error;

#[derive(Error, Debug)]
enum AppError {
    #[error("User not found: {0}")]
    NotFound(String),

    #[error("Database error")]
    Database(#[from] sqlx::Error),

    #[error("Validation failed: {0}")]
    Validation(String),
}

fn get_user(id: u32) -> Result<User, AppError> {
    let user = db.find_user(id)
        .map_err(AppError::Database)?;

    match user {
        Some(u) => Ok(u),
        None => Err(AppError::NotFound(format!("id={}", id))),
    }
}
```

```typescript
// TypeScript: カスタムエラークラス
class AppError extends Error {
    constructor(
        message: string,
        public readonly code: string,
        public readonly statusCode: number,
    ) {
        super(message);
        this.name = "AppError";
    }
}

class NotFoundError extends AppError {
    constructor(resource: string) {
        super(`${resource} not found`, "NOT_FOUND", 404);
    }
}

class ValidationError extends AppError {
    constructor(public readonly fields: Record<string, string>) {
        super("Validation failed", "VALIDATION_ERROR", 400);
    }
}
```

---

## まとめ

| 方式 | エラー伝播 | 型安全 | チェック強制 | 代表言語 |
|------|----------|--------|-----------|---------|
| 例外 | 暗黙（throw） | 低い | Java のみ | Python, Java |
| Result | 明示（?/map） | 高い | コンパイル時 | Rust, Haskell |
| エラーコード | 明示（if err） | 低い | なし | Go, C |

---

## 次に読むべきガイド
→ [[03-iterators-and-generators.md]] — イテレータとジェネレータ

---

## 参考文献
1. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.9, 2023.
2. Bloch, J. "Effective Java." 3rd Ed, Item 69-77, 2018.
