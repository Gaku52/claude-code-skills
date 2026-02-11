# 例外処理

> 例外（Exception）は「通常の制御フローでは処理できない異常事態」を表す仕組み。try/catch/finally の正しい使い方、例外階層の設計、checked vs unchecked の議論を理解する。

## この章で学ぶこと

- [ ] 例外処理の仕組みとコールスタックの巻き戻しを理解する
- [ ] 例外の適切な使い方と濫用の違いを把握する
- [ ] 各言語の例外モデルの違いを学ぶ

---

## 1. try/catch/finally

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

---

## 2. コールスタックの巻き戻し

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

---

## 3. 例外の使い方

```
例外を使うべき場面:
  ✓ ファイルが見つからない
  ✓ ネットワーク接続エラー
  ✓ データベース接続失敗
  ✓ 不正なデータ（バリデーションエラー）
  ✓ リソース不足（メモリ、ディスク）

例外を使うべきでない場面:
  ✗ 通常の制御フロー（if/else で判断できる）
  ✗ 予期される状況（ユーザーの入力ミス）
  ✗ パフォーマンスクリティカルなコード

❌ 例外で制御フロー
  try {
    const user = findUser(id);
  } catch (e) {
    // ユーザーが見つからない = 普通のこと
    return null;
  }

✅ 戻り値で制御フロー
  const user = findUser(id); // null を返す
  if (!user) return null;
```

---

## 4. Java: checked vs unchecked

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

```
checked exception の議論:

  賛成派:
  → エラー処理を忘れない
  → APIの契約が明確

  反対派（多数派）:
  → ボイラープレートが多い
  → 握りつぶしの原因（catch して無視）
  → Kotlin, C#, Python, TS は全て unchecked のみ

  現代の主流:
  → unchecked exception + 型でエラーを表現（Result型）
```

---

## 5. エラー階層の設計

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

---

## まとめ

| 概念 | ポイント |
|------|---------|
| try/catch | 例外をキャッチして処理 |
| finally | 必ず実行されるクリーンアップ |
| 伝播 | 処理できる場所でキャッチ |
| checked vs unchecked | 現代は unchecked + Result型 |
| カスタムエラー | 階層を設計してコード化 |

---

## 次に読むべきガイド
→ [[01-result-type.md]] — Result型

---

## 参考文献
1. Bloch, J. "Effective Java." Items 69-77, 2018.
2. Sutter, H. "When and How to Use Exceptions." 2004.
