# エラーハンドリング ── 例外・Result型・堅牢なエラー処理

> エラーハンドリングはプログラムの信頼性を決定づける。正常系だけを書くのは簡単だが、異常系を適切に処理するコードこそがプロフェッショナルの仕事である。例外・Result型・エラーコードの使い分けを理解し、堅牢なシステムを構築する。

---

## この章で学ぶこと

1. **エラーハンドリングの基本戦略** ── 例外、Result型、エラーコードの使い分けを理解する
2. **例外設計の原則** ── 検査例外 vs 非検査例外、カスタム例外の設計を身につける
3. **堅牢なエラー処理パターン** ── Fail Fast、グレースフルデグラデーションを習得する

---

## 1. エラーハンドリング戦略の全体像

```
+-----------------------------------------------------------+
|  エラー処理の3つのアプローチ                               |
+-----------------------------------------------------------+
|  1. 例外 (Exception)                                      |
|     → 予期しないエラーをスタックを遡って処理               |
|     → Java, Python, C#, TypeScript                        |
+-----------------------------------------------------------+
|  2. Result/Either型                                       |
|     → 成功/失敗を型で表現、コンパイル時にチェック          |
|     → Rust (Result), Go (error), Haskell (Either)         |
+-----------------------------------------------------------+
|  3. エラーコード                                          |
|     → 関数の戻り値でエラーを示す（レガシー手法）          |
|     → C言語の伝統。現代では非推奨                          |
+-----------------------------------------------------------+
```

```
  エラーの分類

  ┌─────────────────────────────────────────────────┐
  │              プログラミングエラー                 │
  │  ・null参照、配列範囲外、型エラー                │
  │  → 修正すべきバグ。例外で即座に停止              │
  ├─────────────────────────────────────────────────┤
  │              ビジネスエラー                       │
  │  ・在庫不足、残高不足、権限なし                   │
  │  → 想定内。Result型や専用例外で表現              │
  ├─────────────────────────────────────────────────┤
  │              インフラエラー                       │
  │  ・DB接続失敗、ネットワークタイムアウト           │
  │  → リトライ or グレースフルデグラデーション       │
  └─────────────────────────────────────────────────┘
```

---

## 2. 例外設計

**コード例1: カスタム例外の階層設計**

```python
# アプリケーション固有の例外階層
class AppError(Exception):
    """アプリケーションの基底例外"""
    def __init__(self, message: str, code: str = "UNKNOWN"):
        super().__init__(message)
        self.code = code

class ValidationError(AppError):
    """入力バリデーションエラー"""
    def __init__(self, field: str, message: str):
        super().__init__(message, code="VALIDATION_ERROR")
        self.field = field

class NotFoundError(AppError):
    """リソースが見つからないエラー"""
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            f"{resource} (ID: {identifier}) が見つかりません",
            code="NOT_FOUND"
        )
        self.resource = resource
        self.identifier = identifier

class AuthorizationError(AppError):
    """権限不足エラー"""
    def __init__(self, action: str):
        super().__init__(f"操作 '{action}' の権限がありません", code="FORBIDDEN")
        self.action = action

# 使用例
class UserService:
    def get_user(self, user_id: str) -> User:
        user = self.repository.find_by_id(user_id)
        if user is None:
            raise NotFoundError("User", user_id)
        return user
```

**コード例2: try-catchの適切な範囲**

```java
// 悪い: try の範囲が広すぎる
try {
    User user = userRepository.findById(userId);
    Order order = new Order(user);
    order.addItem(item);
    order.calculateTotal();
    paymentService.charge(order);
    emailService.sendConfirmation(user, order);
    analyticsService.track(order);
} catch (Exception e) {
    logger.error("エラーが発生しました", e);  // 何のエラーか不明
}

// 良い: try の範囲を最小限に、例外を具体的にキャッチ
User user = userRepository.findById(userId);
Order order = new Order(user);
order.addItem(item);
order.calculateTotal();

try {
    paymentService.charge(order);
} catch (PaymentDeclinedException e) {
    return OrderResult.paymentFailed(e.getReason());
} catch (PaymentGatewayException e) {
    logger.error("決済ゲートウェイ接続エラー", e);
    return OrderResult.systemError("決済システムに接続できません");
}

try {
    emailService.sendConfirmation(user, order);
} catch (EmailServiceException e) {
    logger.warn("確認メール送信失敗（注文自体は成功）", e);
    // メール失敗は注文をロールバックしない
}
```

**コード例3: 例外の変換（レイヤー間）**

```typescript
// リポジトリ層: インフラ例外をドメイン例外に変換
class UserRepository {
  async findById(id: string): Promise<User> {
    try {
      const row = await this.db.query('SELECT * FROM users WHERE id = $1', [id]);
      if (!row) {
        throw new UserNotFoundError(id);
      }
      return this.mapToUser(row);
    } catch (error) {
      if (error instanceof UserNotFoundError) throw error;
      // インフラ例外をドメイン例外に変換
      throw new DataAccessError(`ユーザー取得失敗 (ID: ${id})`, { cause: error });
    }
  }
}

// サービス層: ドメイン例外をそのまま伝播
class UserService {
  async getUser(id: string): Promise<User> {
    return this.userRepository.findById(id);  // 例外はそのまま伝播
  }
}

// コントローラ層: ドメイン例外をHTTPレスポンスに変換
class UserController {
  async getUser(req: Request, res: Response) {
    try {
      const user = await this.userService.getUser(req.params.id);
      res.json(user);
    } catch (error) {
      if (error instanceof UserNotFoundError) {
        res.status(404).json({ error: error.message });
      } else if (error instanceof DataAccessError) {
        res.status(500).json({ error: 'サーバーエラー' });
      }
    }
  }
}
```

---

## 3. Result型パターン

**コード例4: Result型の実装と使用**

```typescript
// Result型の定義
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

// ビジネスエラーの型定義
type TransferError =
  | { type: 'INSUFFICIENT_BALANCE'; available: number; requested: number }
  | { type: 'ACCOUNT_LOCKED'; reason: string }
  | { type: 'DAILY_LIMIT_EXCEEDED'; limit: number };

// Result型を使った関数
function transfer(
  from: Account,
  to: Account,
  amount: number
): Result<TransferReceipt, TransferError> {
  if (from.isLocked) {
    return err({ type: 'ACCOUNT_LOCKED', reason: from.lockReason });
  }
  if (from.balance < amount) {
    return err({
      type: 'INSUFFICIENT_BALANCE',
      available: from.balance,
      requested: amount
    });
  }
  if (from.dailyTransferred + amount > DAILY_LIMIT) {
    return err({ type: 'DAILY_LIMIT_EXCEEDED', limit: DAILY_LIMIT });
  }

  from.balance -= amount;
  to.balance += amount;
  return ok(new TransferReceipt(from, to, amount));
}

// 呼び出し側: エラーハンドリングが強制される
const result = transfer(accountA, accountB, 10000);
if (!result.ok) {
  switch (result.error.type) {
    case 'INSUFFICIENT_BALANCE':
      console.log(`残高不足: ${result.error.available}円`);
      break;
    case 'ACCOUNT_LOCKED':
      console.log(`口座ロック: ${result.error.reason}`);
      break;
    case 'DAILY_LIMIT_EXCEEDED':
      console.log(`日次上限超過: ${result.error.limit}円`);
      break;
  }
}
```

**コード例5: Rust風のResult型**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar, Union

T = TypeVar('T')
E = TypeVar('E')

@dataclass
class Ok(Generic[T]):
    value: T

@dataclass
class Err(Generic[E]):
    error: E

Result = Union[Ok[T], Err[E]]

# 使用例
def parse_age(input_str: str) -> Result[int, str]:
    try:
        age = int(input_str)
    except ValueError:
        return Err(f"'{input_str}' は数値ではありません")

    if age < 0 or age > 150:
        return Err(f"年齢 {age} は範囲外です (0-150)")

    return Ok(age)

# パターンマッチング（Python 3.10+）
match parse_age(user_input):
    case Ok(age):
        print(f"年齢: {age}")
    case Err(message):
        print(f"エラー: {message}")
```

---

## 4. 例外 vs Result型 の使い分け

| 基準 | 例外 | Result型 |
|------|------|---------|
| プログラミングエラー | 適切（即座にクラッシュ） | 不適切 |
| ビジネスエラー | 可能だが冗長 | 最適（型で表現） |
| 予期しないエラー | 適切 | 不適切 |
| エラー処理の強制 | 強制できない（catchし忘れ） | コンパイル時に強制 |
| パフォーマンス | スタックトレース生成コスト | ゼロコスト |

| 言語 | 推奨アプローチ |
|------|---------------|
| Java | 検査例外（ビジネス） + 非検査例外（プログラミング） |
| Python | 例外が標準的。カスタム例外階層を設計 |
| TypeScript | Result型 + 例外の組み合わせ |
| Rust | Result型が標準。panic!は致命的エラーのみ |
| Go | error インターフェースが標準 |

---

## 5. アンチパターン

### アンチパターン1: 例外の握りつぶし（Swallowing Exceptions）

```python
# アンチパターン: エラーを握りつぶす
try:
    process_payment(order)
except Exception:
    pass  # 何もしない → 決済失敗が闇に消える

# 改善: 最低限ログを残し、適切に処理
try:
    process_payment(order)
except PaymentDeclinedError as e:
    logger.warning(f"決済拒否: {e}")
    return PaymentResult.declined(e.reason)
except Exception as e:
    logger.error(f"予期しない決済エラー: {e}", exc_info=True)
    raise PaymentSystemError("決済処理に失敗しました") from e
```

### アンチパターン2: 制御フローとしての例外利用

```python
# アンチパターン: 正常な制御フローに例外を使う
def find_user(users, name):
    try:
        for user in users:
            if user.name == name:
                raise StopIteration(user)  # 見つかったら例外で脱出
    except StopIteration as e:
        return e.args[0]
    return None

# 改善: 通常の制御フローを使う
def find_user(users, name):
    for user in users:
        if user.name == name:
            return user
    return None
```

---

## 6. FAQ

### Q1: すべての関数にtry-catchを書くべきか？

いいえ。**例外は適切なレイヤーで1回だけキャッチする**のが原則。すべての関数でtry-catchすると、エラーの伝播が妨げられ、コードが冗長になる。キャッチすべき場所は: APIのエントリポイント、バッチ処理のループ、外部サービスとの境界。

### Q2: 例外メッセージには何を含めるべきか？

1. **何が起きたか**: 「ユーザーが見つかりません」
2. **コンテキスト**: 「ID: user-123」
3. **原因（可能なら）**: 「データベース接続タイムアウト」
4. **回復方法のヒント**: 「管理者に連絡してください」

**含めてはいけないもの**: パスワード、トークン、個人情報などの機密データ。

### Q3: エラーログとエラーレスポンスは同じ内容でよいか？

**異なるべき**。ログには技術的詳細（スタックトレース、内部ID）を含め、ユーザー向けレスポンスにはユーザーが理解・行動できる情報のみを含める。内部実装の詳細をユーザーに公開するとセキュリティリスクになる。

---

## まとめ

| 戦略 | 用途 | 長所 | 短所 |
|------|------|------|------|
| 例外 | 予期しないエラー | 自然な伝播 | 処理漏れのリスク |
| Result型 | ビジネスエラー | 型安全、処理強制 | 冗長になりがち |
| Fail Fast | 入力検証 | バグの早期発見 | ── |
| ログ+通知 | インフラエラー | 運用可視化 | ── |

---

## 次に読むべきガイド

- [関数設計](./01-functions.md) ── エラーハンドリングを組み込んだ関数設計
- [テスト原則](./04-testing-principles.md) ── エラーケースのテスト方法
- [関数型原則](../03-practices-advanced/02-functional-principles.md) ── Maybe/Either モナドによるエラー処理

---

## 参考文献

1. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 (Chapter 7: Error Handling)
2. **Michael T. Nygard** 『Release It!: Design and Deploy Production-Ready Software』 Pragmatic Bookshelf, 2018 (2nd Edition)
3. **Joe Duffy** "The Error Model" (blog post, 2016) ── Result型と例外の比較分析
