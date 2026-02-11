# Result型

> Result型は「成功または失敗」を型で表現する手法。例外を使わずにエラーを明示的に扱い、コンパイラが「エラー処理忘れ」を検出する。Rust, Go, TypeScript での実装を比較する。

## この章で学ぶこと

- [ ] Result型の仕組みと例外との違いを理解する
- [ ] 各言語でのResult型の実装を把握する
- [ ] Result型のメリットとデメリットを学ぶ

---

## 1. 例外 vs Result型

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

---

## 2. Rust の Result

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

---

## 3. Go のエラー

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

---

## 4. TypeScript での Result型

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

// neverthrow ライブラリ（より高機能）
// import { ok, err, Result } from 'neverthrow';
// result.map(v => ...).mapErr(e => ...).andThen(v => ...)
```

---

## 5. Result型 vs 例外の使い分け

```
Result型を使うべき場面:
  ✓ 予期されるエラー（バリデーション、ファイル未存在）
  ✓ ライブラリ/APIのパブリックインターフェース
  ✓ 型安全性が重要な場面
  ✓ エラーの種類が限定的

例外を使うべき場面:
  ✓ 予期しないエラー（プログラミングミス）
  ✓ 回復不能なエラー（OutOfMemory）
  ✓ フレームワークが例外を期待する場合
  ✓ 深いコールスタックからのエラー伝播

組み合わせ（推奨）:
  → ドメインロジック: Result型（予期されるエラー）
  → インフラ層: 例外（ネットワーク、DB障害）
  → 境界（Controller）: 例外を Result に変換
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

---

## 次に読むべきガイド
→ [[02-error-boundaries.md]] — エラー境界

---

## 参考文献
1. The Rust Programming Language. "Error Handling."
2. Go Blog. "Error handling and Go." 2011.
